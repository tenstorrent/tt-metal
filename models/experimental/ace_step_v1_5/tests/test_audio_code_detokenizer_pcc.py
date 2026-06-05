# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: ``TtAceStepAudioCodeDetokenizer`` vs HF ``model.detokenizer`` @ production dims.

5 Hz audio codes expand to 25 Hz hints: ``[1, N_codes × pool_window_size, 64]`` with
``pool_window_size=5`` (e.g. 75 codes → 375 latent frames for 15 s).
"""

from __future__ import annotations

import os
import re

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.tests._prod_test_helpers import (
    base_model_safetensors,
    ckpt_root,
    ensure_vendored_acestep_on_path,
)

_PCC = float(os.environ.get("ACE_STEP_DETOK_PCC", "0.97"))


def _load_hf_detokenizer():
    from transformers import AutoModel

    ensure_vendored_acestep_on_path()
    model_dir = ckpt_root() / "acestep-v15-base"
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=torch.float32).eval()
    return model.tokenizer.quantizer, model.detokenizer


def _code_str(n_codes: int, *, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(int(seed))
    ids = torch.randint(100, 60000, (int(n_codes),), generator=g).tolist()
    return "".join(f"<|audio_code_{int(i)}|>" for i in ids)


@pytest.mark.parametrize("n_codes,label", [(75, "15s_75codes"), (150, "30s_150codes")])
def test_audio_code_detokenizer_pcc_vs_hf(device, n_codes: int, label: str):
    if base_model_safetensors() is None:
        pytest.skip("ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.")

    import ttnn
    from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import TtAceStepAudioCodeDetokenizer

    quantizer, detok = _load_hf_detokenizer()

    code_str = _code_str(n_codes, seed=42)
    code_ids = [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str)]
    assert len(code_ids) == int(n_codes)

    indices = torch.tensor(code_ids, dtype=torch.long).reshape(1, n_codes, 1)
    with torch.inference_mode():
        quantized = quantizer.get_output_from_indices(indices)
        ref = detok(quantized).float()

    tt_detok = TtAceStepAudioCodeDetokenizer(
        device=device,
        checkpoint_safetensors_path=str(base_model_safetensors()),
        dtype=getattr(ttnn, "bfloat16", None),
    )
    out_tt = tt_detok.forward(code_str)
    assert out_tt is not None
    got = ttnn.to_torch(out_tt).float()

    t = min(int(ref.shape[1]), int(got.shape[1]))
    c = min(int(ref.shape[2]), int(got.shape[2]))
    ref_s = ref[:, :t, :c]
    got_s = got[:, :t, :c]

    print(
        f"\n[detok_pcc][{label}] n_codes={n_codes} ref={tuple(ref.shape)} got={tuple(got.shape)}",
        flush=True,
    )
    score = assert_pcc_print(f"audio_detokenizer_{label}", ref_s, got_s, pcc=_PCC)
    print(
        f"[ace_step_v1_5][PCC] audio_detokenizer_{label}_summary: pcc={score:.6f}",
        flush=True,
    )
