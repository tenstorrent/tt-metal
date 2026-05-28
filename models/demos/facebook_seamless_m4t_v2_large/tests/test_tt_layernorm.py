# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN LayerNorm block.

Loads the saved golden tensors from
`models/demos/facebook_seamless_m4t_v2_large/reference/golden/layernorm.pt`,
runs the TTNN block on the open p150 (blackhole) device, and asserts
PCC > 0.99 against the reference output.

Can also be run as a standalone script (`python test_tt_layernorm.py`) which
is how the bring-up orchestrator invokes it. It opens its own device, runs
the comparison, prints PCC, and exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "layernorm.pt"


def _run_layernorm_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    weight: torch.Tensor = golden["weight"]
    bias: torch.Tensor = golden["bias"]
    eps: float = float(golden["eps"])
    ref_out: torch.Tensor = golden["output"]
    dim = x_torch.shape[-1]

    tt_layernorm = LayerNorm(
        device=device,
        dim=dim,
        weight=weight,
        bias=bias,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )

    tt_input = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_layernorm(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    # Match shapes for PCC compute (ttnn may add a leading dim depending on padding).
    tt_output_torch = tt_output_torch.reshape(ref_out.shape)

    passing, pcc_message = comp_pcc(ref_out, tt_output_torch, 0.99)
    # comp_pcc returns either a bare float or a string containing the float.
    msg_str = str(pcc_message).strip()
    try:
        pcc_value = float(msg_str)
    except ValueError:
        # Fallback: pull first floating-point-looking token from the message.
        import re

        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        pcc_value = float(match.group(0)) if match else float("nan")
    print(f"comp_pcc: passing={passing}, message={pcc_message}")
    return pcc_value


def test_tt_layernorm():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_value = _run_layernorm_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_value > 0.99, f"PCC {pcc_value} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_value = float("nan")
    try:
        pcc_value = _run_layernorm_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {"pcc": pcc_value, "passed": pcc_value > 0.99}
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
