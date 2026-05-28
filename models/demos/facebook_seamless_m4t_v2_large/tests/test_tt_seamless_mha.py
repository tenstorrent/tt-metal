# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN base MHA block.

Loads the saved golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/seamless_mha.pt``,
runs the TTNN block on the open p150 (blackhole) device twice (self-attn and
cross-attn), and asserts PCC > 0.99 against the reference outputs.

Can be run as a standalone script (``python test_tt_seamless_mha.py``) which
is how the bring-up orchestrator invokes it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_mha import SeamlessMHA

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "seamless_mha.pt"


def _extract_pcc(message) -> float:
    msg_str = str(message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_mha_pcc(device):
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    enc_torch: torch.Tensor = golden["encoder_hidden_states"]
    self_mask: torch.Tensor = golden["self_attention_mask"]
    cross_mask: torch.Tensor = golden["cross_attention_mask"]
    ref_self: torch.Tensor = golden["output_self"]
    ref_cross: torch.Tensor = golden["output_cross"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])

    tt_mha = SeamlessMHA(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=golden["state_dict"],
        weight_dtype=ttnn.bfloat16,
    )

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def mask_to_tt(m):
        if m is None:
            return None
        # Mask is [B, 1, T, S] additive log-mask; tile and store on device.
        return ttnn.from_torch(
            m,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # --- Self-attention path ---
    x_tt = to_tt(x_torch)
    self_mask_tt = mask_to_tt(self_mask)
    out_self_tt = tt_mha(x_tt, encoder_hidden_states=None, attention_mask=self_mask_tt)
    out_self_torch = ttnn.to_torch(out_self_tt).to(torch.float32).reshape(ref_self.shape)
    passing_s, msg_s = comp_pcc(ref_self, out_self_torch, 0.99)
    pcc_self = _extract_pcc(msg_s)
    print(f"self-attn comp_pcc: passing={passing_s}, message={msg_s}")

    # --- Cross-attention path ---
    # Re-upload inputs in case the prior call deallocated them.
    x_tt2 = to_tt(x_torch)
    enc_tt = to_tt(enc_torch)
    cross_mask_tt = mask_to_tt(cross_mask)
    out_cross_tt = tt_mha(x_tt2, encoder_hidden_states=enc_tt, attention_mask=cross_mask_tt)
    out_cross_torch = ttnn.to_torch(out_cross_tt).to(torch.float32).reshape(ref_cross.shape)
    passing_c, msg_c = comp_pcc(ref_cross, out_cross_torch, 0.99)
    pcc_cross = _extract_pcc(msg_c)
    print(f"cross-attn comp_pcc: passing={passing_c}, message={msg_c}")

    return pcc_self, pcc_cross


def test_tt_seamless_mha():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_self, pcc_cross = _run_mha_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_self > 0.99, f"self-attn PCC {pcc_self} <= 0.99"
    assert pcc_cross > 0.99, f"cross-attn PCC {pcc_cross} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_self = float("nan")
    pcc_cross = float("nan")
    try:
        pcc_self, pcc_cross = _run_mha_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_self, pcc_cross)
    result = {
        "pcc_self": pcc_self,
        "pcc_cross": pcc_cross,
        "pcc": worst,
        "passed": worst > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
