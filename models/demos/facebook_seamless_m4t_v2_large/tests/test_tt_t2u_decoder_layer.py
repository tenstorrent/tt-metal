# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 Text-to-Unit decoder layer TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/t2u_decoder_layer.pt``,
runs the TTNN ``T2UDecoderLayer`` block on the open p150 (blackhole) device
for the unmasked, triangular-causal, and padded paths, and asserts PCC > 0.99
against the saved reference outputs (the worst of the three is reported).

The masked-causal path is the most realistic deployment-relevant path; the
unmasked/padded paths are included as extra coverage.

Can also be run as a standalone script
(``python test_tt_t2u_decoder_layer.py``) which opens its own device,
runs the PCC comparison, prints the result and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder_layer import T2UDecoderLayer

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "t2u_decoder_layer.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]  # [B, T, C] float32
    self_attention_mask_torch: torch.Tensor = golden["self_attention_mask"]  # [B,1,T,T] triangular causal
    pad_attention_mask_torch: torch.Tensor = golden["pad_attention_mask"]  # [B,1,T,T] padding-aware
    padding_mask_torch: torch.Tensor = golden["padding_mask"]  # [B, T] bool
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_causal: torch.Tensor = golden["output_causal"]
    ref_padded: torch.Tensor = golden["output_padded"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    conv_kernel_size = int(cfg["conv_kernel_size"])
    eps = float(cfg["eps"])

    tt_block = T2UDecoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=state_dict,
        conv_kernel_size=conv_kernel_size,
        eps=eps,
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

    def _padding_mask_to_float(pm_bool: torch.Tensor) -> torch.Tensor:
        """Convert (B, T) bool padding mask to (B, T, 1) float multiplier.

        True/1 -> 1.0 (keep), False/0 -> 0.0 (zero out).
        """
        return pm_bool.to(torch.float32).unsqueeze(-1).contiguous()

    def _run_path(
        attn_mask_torch: Optional[torch.Tensor],
        pad_mask_torch: Optional[torch.Tensor],
        ref: torch.Tensor,
        label: str,
    ) -> float:
        x_tt = _to_tt(x_torch)
        attn_mask_tt = _to_tt(attn_mask_torch) if attn_mask_torch is not None else None
        pad_mask_tt = _to_tt(_padding_mask_to_float(pad_mask_torch)) if pad_mask_torch is not None else None

        out_tt = tt_block(
            x_tt,
            attention_mask=attn_mask_tt,
            padding_mask=pad_mask_tt,
        )
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(f"comp_pcc({label}): passing={passing}, message={pcc_message}")
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, None, ref_unmasked, "unmasked")
    pcc_causal = _run_path(self_attention_mask_torch, None, ref_causal, "causal")
    pcc_padded = _run_path(pad_attention_mask_torch, padding_mask_torch, ref_padded, "padded")
    return pcc_unmasked, pcc_causal, pcc_padded


def test_tt_t2u_decoder_layer():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_unmasked, pcc_causal, pcc_padded = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_causal, pcc_padded)
    assert worst > 0.99, (
        f"PCC unmasked={pcc_unmasked}, causal={pcc_causal}, padded={pcc_padded}; " f"worst={worst} <= 0.99"
    )


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_unmasked = float("nan")
    pcc_causal = float("nan")
    pcc_padded = float("nan")
    try:
        pcc_unmasked, pcc_causal, pcc_padded = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_causal, pcc_padded)
    result = {
        "pcc_unmasked": pcc_unmasked,
        "pcc_causal": pcc_causal,
        "pcc_padded": pcc_padded,
        "pcc": worst,
        "passed": worst > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
