# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN ``HifiGanResidualBlock``.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/hifigan_residual_block.pt``,
runs the TTNN ``HifiGanResidualBlock`` on the open p150 (blackhole) device,
and asserts PCC > 0.99 against the saved reference output.

Can also be run as a standalone script which opens its own device, runs the
PCC comparison, prints a single-line JSON, and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_residual_block import HifiGanResidualBlock

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "hifigan_residual_block.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]  # [B, C, T] float32
    sd = golden["state_dict"]
    ref_out: torch.Tensor = golden["output"]
    cfg = golden.get("config", {})
    kernel_size = int(cfg.get("kernel_size", 3))
    dilation = tuple(cfg.get("dilation", (1, 3, 5)))
    leaky_relu_slope = float(cfg.get("leaky_relu_slope", 0.1))

    convs1_weights = [layer["weight"] for layer in sd["convs1"]]
    convs1_biases = [layer["bias"] for layer in sd["convs1"]]
    convs2_weights = [layer["weight"] for layer in sd["convs2"]]
    convs2_biases = [layer["bias"] for layer in sd["convs2"]]

    tt_block = HifiGanResidualBlock(
        device=device,
        convs1_weights=convs1_weights,
        convs1_biases=convs1_biases,
        convs2_weights=convs2_weights,
        convs2_biases=convs2_biases,
        kernel_size=kernel_size,
        dilation=dilation,
        leaky_relu_slope=leaky_relu_slope,
        weight_dtype=ttnn.bfloat16,
    )

    tt_input = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = tt_block(tt_input)
    tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32).reshape(ref_out.shape)

    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, 0.99)
    print(f"comp_pcc(hifigan_residual_block): passing={passing}, message={pcc_message}")
    return _pcc_from_message(passing, pcc_message)


def test_tt_hifigan_residual_block():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc > 0.99, f"PCC {pcc} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc = float("nan")
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {
        "block": "hifigan_residual_block",
        "pcc": pcc,
        "passed": pcc > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
