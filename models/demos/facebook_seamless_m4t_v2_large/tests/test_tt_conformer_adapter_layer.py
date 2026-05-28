# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 Conformer TTNN adapter-layer block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/conformer_adapter_layer.pt``,
runs the TTNN ``ConformerAdapterLayer`` block on the open p150 (blackhole)
device for both the unmasked and masked paths, and asserts PCC > 0.99 against
the saved reference outputs (the worst of the two is reported).

Can also be run as a standalone script (``python test_tt_conformer_adapter_layer.py``)
which opens its own device, runs the PCC comparison, prints the result and
exits 0 on pass / 1 on fail.
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
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_adapter_layer import ConformerAdapterLayer

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "conformer_adapter_layer.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]  # [B, T, C] float32
    sub_attention_mask_torch: torch.Tensor = golden["sub_attention_mask_4d"]  # [B, 1, T_sub, T_sub]
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    batch = int(cfg["batch"])
    seq_len = int(cfg["seq_len"])
    sub_seq_len = int(cfg["sub_seq_len"])
    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    kernel_size = int(cfg["kernel_size"])
    stride = int(cfg["stride"])
    eps = float(cfg["eps"])

    tt_block = ConformerAdapterLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        sub_seq_len=sub_seq_len,
        state_dict=state_dict,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        batch_size=batch,
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

    def _run_path(attn_mask: torch.Tensor | None, ref: torch.Tensor) -> float:
        x_tt = _to_tt(x_torch)
        attn_mask_tt = None
        if attn_mask is not None:
            attn_mask_tt = _to_tt(attn_mask)
        out_tt = tt_block(x_tt, attention_mask=attn_mask_tt)
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(
            f"comp_pcc(attn_mask={'yes' if attn_mask is not None else 'no'}):"
            f" passing={passing}, message={pcc_message}"
        )
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, ref_unmasked)
    pcc_masked = _run_path(sub_attention_mask_torch, ref_masked)
    return pcc_unmasked, pcc_masked


def test_tt_conformer_adapter_layer():
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
