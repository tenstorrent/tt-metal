# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: full ``TtAceStepInstrumentalConditionEncoder.forward_payload`` vs HF ``prepare_condition``.

Uses real handler preprocess (5 Hz LM + lyric/timbre/text packing) at production shapes:
``enc`` [1, S, 2048], ``ctx`` [1, T_lat, 128] with ``T_lat = duration_sec × 25``.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.tests._prod_test_helpers import (
    base_model_safetensors,
    build_instrumental_filtered,
    init_dit_and_lm_handlers,
)

_PCC_ENC = float(os.environ.get("ACE_STEP_COND_PAYLOAD_PCC_ENC", "0.97"))
_PCC_CTX = float(os.environ.get("ACE_STEP_COND_PAYLOAD_PCC_CTX", "0.99"))


@pytest.mark.parametrize("duration_sec,label", [(15.0, "15s"), (30.0, "30s")])
def test_condition_encoder_forward_payload_pcc_vs_hf(device, duration_sec: float, label: str):
    if base_model_safetensors() is None:
        pytest.skip("ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.")

    import ttnn
    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import (
        handler_prepare_condition_payload,
        handler_prepare_condition_tensors,
    )
    from models.experimental.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

    dit, llm = init_dit_and_lm_handlers()
    filtered, expected_frames = build_instrumental_filtered(dit, llm, duration_sec=duration_sec)
    payload, frames = handler_prepare_condition_payload(dit, filtered)
    assert int(frames) == int(expected_frames)

    enc_ref, enc_mask_ref, ctx_ref, frames_ref, _null = handler_prepare_condition_tensors(dit, filtered)
    assert int(frames_ref) == int(frames)

    enc_tt = TtAceStepInstrumentalConditionEncoder(
        device=device,
        checkpoint_safetensors_path=str(base_model_safetensors()),
        dtype=getattr(ttnn, "bfloat16", None),
    )
    enc_got, enc_mask_got, ctx_got, _null_tt = enc_tt.forward_payload(payload)

    enc_got_t = ttnn.to_torch(enc_got).float()
    if enc_got_t.ndim == 4:
        enc_got_t = enc_got_t.squeeze(1)
    ctx_got_t = ttnn.to_torch(ctx_got).float()

    s_enc = min(int(enc_ref.shape[1]), int(enc_got_t.shape[1]))
    d_enc = min(int(enc_ref.shape[2]), int(enc_got_t.shape[2]))
    enc_ref_s = enc_ref[:, :s_enc, :d_enc]
    enc_got_s = enc_got_t[:, :s_enc, :d_enc]

    t_ctx = min(int(ctx_ref.shape[1]), int(ctx_got_t.shape[1]))
    c_ctx = min(int(ctx_ref.shape[2]), int(ctx_got_t.shape[2]))
    ctx_ref_s = ctx_ref[:, :t_ctx, :c_ctx]
    ctx_got_s = ctx_got_t[:, :t_ctx, :c_ctx]

    print(
        f"\n[condition_payload_pcc][{label}] frames={frames} "
        f"enc={tuple(enc_ref.shape)} ctx={tuple(ctx_ref.shape)} "
        f"enc_mask_sum={float(enc_mask_ref.sum()):.0f}",
        flush=True,
    )
    enc_pcc = assert_pcc_print(f"condition_enc_{label}", enc_ref_s, enc_got_s, pcc=_PCC_ENC)
    ctx_pcc = assert_pcc_print(f"condition_ctx_{label}", ctx_ref_s, ctx_got_s, pcc=_PCC_CTX)
    print(
        f"[ace_step_v1_5][PCC] condition_payload_{label}_summary: " f"enc_pcc={enc_pcc:.6f} ctx_pcc={ctx_pcc:.6f}",
        flush=True,
    )

    mask_got = torch.as_tensor(enc_mask_got, dtype=torch.float32)
    mask_ref = enc_mask_ref.float()
    m = min(int(mask_ref.shape[1]), int(mask_got.shape[1]))
    assert torch.allclose(
        mask_ref[:, :m], mask_got[:, :m], atol=1e-4, rtol=0.0
    ), f"encoder_attention_mask mismatch max_abs={(mask_ref[:, :m] - mask_got[:, :m]).abs().max():.4f}"
