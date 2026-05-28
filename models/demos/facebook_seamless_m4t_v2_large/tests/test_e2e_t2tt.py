# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end BLEU validation for the SeamlessM4T-v2 T2TT TTNN demo.

Runs the TTNN ``TextToTextModel.translate`` over a small set of short
prompts (``demo/inputs/t2tt_samples.json``), computes BLEU against the
hand-written reference translations, and asserts the TTNN BLEU is at
most 1.0 point below the HuggingFace BLEU on the same prompts. The HF
score is computed in the same harness using
``SeamlessM4Tv2ForTextToText.generate(do_sample=False, num_beams=1)``.

Run with::

    pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_t2tt.py -v

The gate is intentionally loose: a single short-form translation may
diverge late in the AR loop on bf16, so we score the whole batch with
corpus BLEU and accept up to 1.0 point of drift versus HF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.demo.validate import bleu
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_text_model import TextToTextModel

# Path to the canonical 10-sample test set.
SAMPLES_PATH = Path(__file__).resolve().parent.parent / "demo" / "inputs" / "t2tt_samples.json"

# AR generation budget for the test. Includes the 2 prefix tokens.
MAX_NEW_TOKENS = 32

# BLEU drift tolerance: TTNN must be within 1.0 point of HF on the same set.
BLEU_TOLERANCE = 1.0


@pytest.fixture(scope="module")
def samples() -> List[dict]:
    if not SAMPLES_PATH.is_file():
        pytest.fail(f"samples file not found: {SAMPLES_PATH}")
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    assert len(data) > 0, "samples must be non-empty"
    return data


@pytest.fixture(scope="module")
def hf_sd():
    return wl.load_hf_state_dict()


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(wl.HF_PATH)


@pytest.fixture(scope="module")
def hf_translations(samples, processor) -> List[str]:
    """Run HF reference on all samples (do_sample=False, num_beams=1)."""
    from transformers import SeamlessM4Tv2ForTextToText

    model = SeamlessM4Tv2ForTextToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    outs: List[str] = []
    with torch.no_grad():
        for s in samples:
            toks = processor(text=s["src"], src_lang=s["src_lang"], return_tensors="pt")
            out = model.generate(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                tgt_lang=s["tgt_lang"],
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=1,
            )
            if hasattr(out, "sequences"):
                out = out.sequences
            text = processor.decode(out[0].tolist(), skip_special_tokens=True)
            outs.append(text)
    del model
    return outs


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_t2tt_bleu_matches_hf(samples, hf_sd, processor, hf_translations, device):
    """TTNN BLEU on a 10-sample short-form set must be within 1.0 of HF BLEU."""

    refs = [s["ref"] for s in samples]

    # 1. TTNN translations -- one TextToTextModel instance, reused across samples.
    model = TextToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
    ttnn_outs: List[str] = []
    for s in samples:
        out = model.translate(
            src_text=s["src"],
            src_lang=s["src_lang"],
            tgt_lang=s["tgt_lang"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        ttnn_outs.append(out)

    # 2. Score and print side-by-side for debugging.
    ttnn_bleu = bleu(ttnn_outs, refs)
    hf_bleu = bleu(hf_translations, refs)

    print("")
    print(f"{'idx':>3}  {'src':<30}  {'ref':<30}  {'HF':<35}  {'TTNN':<35}")
    for i, (s, hf_t, tt_t) in enumerate(zip(samples, hf_translations, ttnn_outs)):
        print(f"{i:>3}  {s['src']:<30.30}  {s['ref']:<30.30}  {hf_t:<35.35}  {tt_t:<35.35}")
    print("")
    print(f"TTNN_BLEU = {ttnn_bleu:.3f}")
    print(f"HF_BLEU   = {hf_bleu:.3f}")
    print(f"drift     = {hf_bleu - ttnn_bleu:.3f}  (tolerance = {BLEU_TOLERANCE})")

    assert ttnn_bleu >= hf_bleu - BLEU_TOLERANCE, (
        f"TTNN BLEU {ttnn_bleu:.3f} fell more than {BLEU_TOLERANCE} below HF BLEU {hf_bleu:.3f} "
        f"on {len(samples)} samples — investigate AR drift / cross-attn cache integrity."
    )
