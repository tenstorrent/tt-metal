# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN ``VariancePredictor`` block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/variance_predictor.pt``,
runs the TTNN ``VariancePredictor`` block on the open p150 (blackhole) device
for both the unmasked and masked paths, and asserts PCC > 0.99 against the
saved reference outputs (the worst of the two is reported).

Can also be run as a standalone script which opens its own device, runs the
PCC comparison, prints a single-line JSON, and exits 0 on pass / 1 on fail.
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
from models.demos.facebook_seamless_m4t_v2_large.tt.variance_predictor import VariancePredictor

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "variance_predictor.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]  # [B, T, embed] float32
    x_masked_torch: torch.Tensor = golden["input_masked"]
    sd = golden["state_dict"]
    padding_mask_torch: torch.Tensor = golden["padding_mask"]  # [B, T] bool
    ref_out_unmasked: torch.Tensor = golden["output"]
    ref_out_masked: torch.Tensor = golden["output_masked"]
    cfg = golden.get("config", {})
    kernel_size = int(cfg.get("kernel_size", 3))
    eps = float(cfg.get("eps", 1e-5))

    tt_block = VariancePredictor(
        device=device,
        conv1_weight=sd["conv1"]["weight"],
        conv1_bias=sd["conv1"]["bias"],
        ln1_weight=sd["ln1"]["weight"],
        ln1_bias=sd["ln1"]["bias"],
        conv2_weight=sd["conv2"]["weight"],
        conv2_bias=sd["conv2"]["bias"],
        ln2_weight=sd["ln2"]["weight"],
        ln2_bias=sd["ln2"]["bias"],
        proj_weight=sd["proj"]["weight"],
        proj_bias=sd["proj"]["bias"],
        kernel_size=kernel_size,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )

    def _run_path(x_t: torch.Tensor, mask_torch, ref: torch.Tensor) -> float:
        tt_input = ttnn.from_torch(
            x_t,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_mask = None
        if mask_torch is not None:
            # Convert HF bool mask [B, T] to float [B, T, 1] for broadcast.
            mask_f = mask_torch.to(torch.float32).reshape(mask_torch.shape[0], mask_torch.shape[1], 1)
            tt_mask = ttnn.from_torch(
                mask_f,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_out = tt_block(tt_input, padding_mask=tt_mask)
        tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, tt_out_torch, 0.99)
        print(
            f"comp_pcc(mask={'yes' if mask_torch is not None else 'no'}):" f" passing={passing}, message={pcc_message}"
        )
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(x_torch, None, ref_out_unmasked)
    pcc_masked = _run_path(x_masked_torch, padding_mask_torch, ref_out_masked)
    return pcc_unmasked, pcc_masked


def test_tt_variance_predictor():
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
