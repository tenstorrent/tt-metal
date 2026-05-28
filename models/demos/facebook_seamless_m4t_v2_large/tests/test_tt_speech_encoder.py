# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN speech encoder.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/speech_encoder.pt``,
runs the TTNN ``SpeechEncoder`` block on the open p150 (blackhole) device
for both the unmasked and masked paths, and asserts PCC > 0.99 against the
saved reference outputs (the worst of the two is reported).

The golden uses ``speech_encoder_layers=2`` (the saved state_dict has 2
encoder layers); the full model has 24, but per-layer correctness against
HF is already covered by ``test_tt_conformer_encoder_layer.py`` and
``test_tt_conformer_adapter_layer.py``. The 2-layer config still exercises
the multi-layer composition path plus the intermediate-FFN residual, the
adapter, and the terminal inner LayerNorm.

Can also be run as a standalone script
(``python test_tt_speech_encoder.py``) which opens its own device, runs
the PCC comparison, prints the result and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Tuple

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_encoder import SpeechEncoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "speech_encoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_features: torch.Tensor = golden["input_features"]  # [B, T, 160] float32
    attention_mask_2d: torch.Tensor = golden["attention_mask_2d"]  # [B, T] int64
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    batch = int(cfg["batch"])
    seq_len = int(cfg["seq_len"])
    feature_size = int(cfg["feature_size"])
    hidden = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    left_max = int(cfg["left_max_position_embeddings"])
    right_max = int(cfg["right_max_position_embeddings"])
    pos_type = cfg["position_embeddings_type"]
    conv_kernel = int(cfg["conv_depthwise_kernel_size"])
    adaptor_kernel = int(cfg["adaptor_kernel_size"])
    adaptor_stride = int(cfg["adaptor_stride"])
    chunk_size = cfg["speech_encoder_chunk_size"]
    left_chunk_num = int(cfg["speech_encoder_left_chunk_num"])
    add_adapter = bool(cfg["add_adapter"])
    eps = float(cfg["eps"])
    act_fn = cfg["speech_encoder_hidden_act"]

    tt_block = SpeechEncoder(
        device=device,
        state_dict=state_dict,
        feature_size=feature_size,
        hidden=hidden,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch,
        eps=eps,
        speech_encoder_hidden_act=act_fn,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type=pos_type,
        conv_depthwise_kernel_size=conv_kernel,
        adaptor_kernel_size=adaptor_kernel,
        adaptor_stride=adaptor_stride,
        speech_encoder_chunk_size=chunk_size,
        speech_encoder_left_chunk_num=left_chunk_num,
        add_adapter=add_adapter,
        weight_dtype=ttnn.bfloat16,
    )

    def _to_tt(t: torch.Tensor, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _run_path(attn_mask_2d: torch.Tensor | None, ref: torch.Tensor) -> float:
        x_tt = _to_tt(input_features)
        out_tt = tt_block(x_tt, attention_mask_2d=attn_mask_2d)
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(
            f"comp_pcc(attn_mask={'yes' if attn_mask_2d is not None else 'no'}):"
            f" passing={passing}, message={pcc_message}"
        )
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, ref_unmasked)
    pcc_masked = _run_path(attention_mask_2d, ref_masked)
    return pcc_unmasked, pcc_masked


def test_tt_speech_encoder():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_masked)
    assert worst > 0.99, f"PCC unmasked={pcc_unmasked}, masked={pcc_masked}; worst={worst} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_unmasked = float("nan")
    pcc_masked = float("nan")
    try:
        pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_masked)
    result = {
        "pcc_unmasked": pcc_unmasked,
        "pcc_masked": pcc_masked,
        "pcc": worst,
        "passed": worst > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
