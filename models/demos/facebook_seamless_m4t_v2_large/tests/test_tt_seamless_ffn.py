# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN FFN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/seamless_ffn.pt``,
runs the TTNN ``SeamlessFfn`` block on the open p150 (blackhole) device, and
asserts PCC > 0.99 against the saved reference output.

Can also be run as a standalone script (``python test_tt_seamless_ffn.py``)
which opens its own device, runs the PCC comparison, prints the result and
exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_ffn import SeamlessFfn

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "seamless_ffn.pt"


def _run_seamless_ffn_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    fc1_weight: torch.Tensor = golden["fc1_weight"]
    fc1_bias: torch.Tensor = golden["fc1_bias"]
    fc2_weight: torch.Tensor = golden["fc2_weight"]
    fc2_bias: torch.Tensor = golden["fc2_bias"]
    ref_out: torch.Tensor = golden["output"]

    tt_ffn = SeamlessFfn(
        device=device,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        weight_dtype=ttnn.bfloat16,
    )

    tt_input = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_ffn(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    # Re-shape to the reference shape (ttnn may add padding dims).
    tt_output_torch = tt_output_torch.reshape(ref_out.shape)

    passing, pcc_message = comp_pcc(ref_out, tt_output_torch, 0.99)
    msg_str = str(pcc_message).strip()
    try:
        pcc_value = float(msg_str)
    except ValueError:
        import re

        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        pcc_value = float(match.group(0)) if match else float("nan")
    print(f"comp_pcc: passing={passing}, message={pcc_message}")
    return pcc_value


def test_tt_seamless_ffn():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_value = _run_seamless_ffn_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_value > 0.99, f"PCC {pcc_value} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_value = float("nan")
    try:
        pcc_value = _run_seamless_ffn_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {"pcc": pcc_value, "passed": pcc_value > 0.99}
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
