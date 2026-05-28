# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN Conformer self-attention block.

Loads the saved golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/conformer_self_attention.pt``,
runs the TTNN block on the open p150 (blackhole) device twice -- once
unmasked, once with the additive log-mask -- and asserts PCC > 0.99 against
both reference outputs.

Can be run as a standalone script (``python test_tt_conformer_self_attention.py``)
which is how the bring-up orchestrator invokes it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_self_attention import ConformerSelfAttention

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "conformer_self_attention.pt"


def _extract_pcc(message) -> float:
    msg_str = str(message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device):
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    mask_torch: torch.Tensor = golden["attention_mask"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]
    state_dict = golden["state_dict"]
    distance_embedding_weight: torch.Tensor = golden["distance_embedding_weight"]

    batch = int(cfg["batch"])
    seq_len = int(cfg["seq_len"])
    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    left_max = int(cfg["left_max_position_embeddings"])
    right_max = int(cfg["right_max_position_embeddings"])
    pos_type = cfg["position_embeddings_type"]

    tt_block = ConformerSelfAttention(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        state_dict=state_dict,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type=pos_type,
        batch_size=batch,
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

    # --- Unmasked path ---
    x_tt = to_tt(x_torch)
    out_unmasked_tt = tt_block(x_tt, attention_mask=None)
    out_unmasked_torch = ttnn.to_torch(out_unmasked_tt).to(torch.float32).reshape(ref_unmasked.shape)
    passing_u, msg_u = comp_pcc(ref_unmasked, out_unmasked_torch, 0.99)
    pcc_unmasked = _extract_pcc(msg_u)
    print(f"unmasked comp_pcc: passing={passing_u}, message={msg_u}")

    # --- Masked path ---
    x_tt2 = to_tt(x_torch)
    mask_tt = to_tt(mask_torch)
    out_masked_tt = tt_block(x_tt2, attention_mask=mask_tt)
    out_masked_torch = ttnn.to_torch(out_masked_tt).to(torch.float32).reshape(ref_masked.shape)
    passing_m, msg_m = comp_pcc(ref_masked, out_masked_torch, 0.99)
    pcc_masked = _extract_pcc(msg_m)
    print(f"masked   comp_pcc: passing={passing_m}, message={msg_m}")

    return pcc_unmasked, pcc_masked


def test_tt_conformer_self_attention():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"


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
