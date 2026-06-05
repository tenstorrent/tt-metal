# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Task-level accuracy gate for SeamlessM4T v2 ASR — TT-vs-HF faithfulness, the layer above per-op PCC.

The per-op PCC suite verifies numerical correlation but cannot catch a regression where every op
stays in-PCC yet the decoded *text* diverges from the reference. This test gates that for ASR using
tokenizer-free chrF/CER of **TT output vs the HF reference on the same audio**.

Why faithfulness-to-HF rather than "must be Hindi": on long audio SeamlessM4T is utterance-level and
will *translate* (e.g. to English) instead of transcribing — and the HF reference does this too, so an
absolute Devanagari gate would fail on behavior the reference also produces. The TT port's job is to
**match the reference**; chrF/CER of TT-vs-HF catches TT-specific divergence (a wrong-language flip
where HF is right, garbage output, a precision cliff) without penalizing the model's own behavior. The
Devanagari fractions of both are logged for visibility. (Robust long-form ASR would segment the audio
into utterances — see the yito branch's demo VAD segmentation — which is out of scope for this gate.)

Caveat: if a TT change makes TT *more* robust than the bf16 HF reference on this clip (TT Hindi, HF
English), chrF would drop and this gate could false-fail — update the reference/threshold if so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.demo.demo import (  # noqa: E402
    _decode,
    _waveform_to_mono_fp32,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (  # noqa: E402
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import (  # noqa: E402
    ensure_seamless_m4t_v2_large_weights,
)
from models.experimental.seamless_m4t_v2_large.tests.accuracy.metrics import (  # noqa: E402
    corpus_cer,
    corpus_chrf,
    script_fraction,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (  # noqa: E402
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (  # noqa: E402
    TTSeamlessM4Tv2GenerationOutput,
)

# Long English prompt -> long Hindi T2ST audio (the demo's chained ASR input regime).
_PROMPT = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather's old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window."""

_TGT_HIN = "hin"
_MAX_NEW_TOKENS = 200
# TT must match the HF reference on the same audio. Observed chrF 90.3 / CER 0.14 with HF (both
# transcribing this clip the same way); thresholds leave margin for greedy TT/HF bf16 desync while
# still catching a gross divergence (wrong language vs HF, garbage) which lands near chrF 0 / CER 1.
_MIN_CHRF_VS_HF = 70.0
_MAX_CER_VS_HF = 0.30

# Allow skipping the (CPU, ~10 min) HF decode for a quick smoke run; the gate then degrades to a
# non-empty + matches-HF-is-skipped check. Default: run HF (it is the reference this test gates on).
_SKIP_HF = os.environ.get("SEAMLESS_EVAL_SKIP_HF", "0") != "0"


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
        raise
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")
        raise


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
def test_asr_long_audio_matches_hf_reference(mesh_device, device_params, reset_seeds):
    """TT ASR (hin) on a long chained clip must match the HF reference (chrF/CER), not diverge."""
    _ = reset_seeds
    _ = device_params

    weights_dir = _weights_dir_or_skip()
    path = os.fspath(weights_dir)
    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    sr = int(getattr(cfg, "sampling_rate", 16000))

    # Production-faithful text input: processor sets the src_lang token (the demo's path).
    text_enc = processor(text=_PROMPT, src_lang="eng", return_tensors="pt")
    common = dict(do_sample=False, num_beams=1, use_kv_cache=True, use_decode_trace=True, use_2cq=True)

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, model, cfg, t2u_cfg)

        # 1) T2ST: English text -> long Hindi speech (the ASR input).
        t2st_out = tt_model.generate(
            input_ids=torch_ids_to_ttnn(mesh_device, text_enc["input_ids"]),
            attention_mask=torch_ids_to_ttnn(mesh_device, text_enc["attention_mask"]),
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=_TGT_HIN,
            speaker_id=0,
            max_new_tokens=_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            **common,
        )
        assert isinstance(t2st_out, TTSeamlessM4Tv2GenerationOutput), type(t2st_out)
        hin_wav = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)

        # Reset the generation runtime between the speech-gen (T2ST) and the ASR text decode, so a
        # leftover speech-gen / decode-trace runtime can't skew the ASR decode (the demo's state-leak,
        # fixed in c7dc1d57d29 via the same calls).
        tt_model.clear_runtime_program_cache()
        tt_model.release_generation_runtime()
        ttnn.synchronize_device(mesh_device)

        # 2) TT ASR (tgt_lang = source language).
        audio_inputs = processor(audios=hin_wav, sampling_rate=sr, return_tensors="pt")
        mel_frames = int(audio_inputs["input_features"].shape[1])
        asr_out = tt_model.generate(
            input_features=torch_feats_to_ttnn(mesh_device, audio_inputs["input_features"]),
            attention_mask=torch_ids_to_ttnn(mesh_device, audio_inputs["attention_mask"]),
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            max_new_tokens=_MAX_NEW_TOKENS,
            repetition_penalty=1.0,
            **common,
        )
        hyp_tt = _decode(tokenizer, asr_out.sequences)

    logger.info(f"[ASR long-audio] audio={hin_wav.size} samples mel={mel_frames}")
    logger.info(f"[ASR long-audio] TT hyp ({script_fraction(hyp_tt, 'deva'):.2f} deva): {hyp_tt}")
    assert len(hyp_tt.strip()) > 0, "ASR produced empty output"

    if _SKIP_HF:
        pytest.skip("SEAMLESS_EVAL_SKIP_HF=1: skipped the HF reference comparison (smoke mode).")

    # HF reference on the SAME audio — the thing TT must match.
    with torch.no_grad():
        hf_out = model.generate(
            input_features=audio_inputs["input_features"].float(),
            attention_mask=audio_inputs["attention_mask"],
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            do_sample=False,
            num_beams=1,
            max_new_tokens=_MAX_NEW_TOKENS,
            repetition_penalty=1.0,
        )
    hf_ids = hf_out.sequences[0] if hasattr(hf_out, "sequences") else hf_out[0]
    hyp_hf = tokenizer.decode(hf_ids.tolist(), skip_special_tokens=True)
    chrf = corpus_chrf([hyp_tt], [hyp_hf])
    cer = corpus_cer([hyp_tt], [hyp_hf])
    logger.info(f"[ASR long-audio] HF hyp ({script_fraction(hyp_hf, 'deva'):.2f} deva): {hyp_hf}")
    logger.info(f"[ASR long-audio] TT-vs-HF chrF={chrf:.2f} CER={cer:.3f}")

    assert chrf >= _MIN_CHRF_VS_HF, (
        f"TT ASR diverged from HF: chrF {chrf:.2f} < {_MIN_CHRF_VS_HF}. "
        f"TT[:100]={hyp_tt[:100]!r} HF[:100]={hyp_hf[:100]!r}"
    )
    assert cer <= _MAX_CER_VS_HF, f"TT-vs-HF CER {cer:.3f} > {_MAX_CER_VS_HF}"
