# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 Conformer TTNN feature-projection block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/conformer_feature_projection.pt``,
runs the TTNN ``ConformerFeatureProjection`` block on the open p150
(blackhole) device, and asserts PCC > 0.99 against the saved reference output.

Can also be run as a standalone script (``python
test_tt_conformer_feature_projection.py``) which opens its own device, runs
the PCC comparison, prints the result and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_feature_projection import ConformerFeatureProjection

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "conformer_feature_projection.pt"


def _run_conformer_feature_projection_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    sd = golden["state_dict"]
    layer_norm_weight: torch.Tensor = sd["layer_norm"]["weight"]
    layer_norm_bias: torch.Tensor = sd["layer_norm"]["bias"]
    projection_weight: torch.Tensor = sd["projection"]["weight"]
    projection_bias: torch.Tensor = sd["projection"]["bias"]
    eps: float = float(golden["config"].get("eps", 1e-5))
    ref_out: torch.Tensor = golden["output"]

    tt_block = ConformerFeatureProjection(
        device=device,
        layer_norm_weight=layer_norm_weight,
        layer_norm_bias=layer_norm_bias,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
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

    tt_output = tt_block(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    # Re-shape to the reference shape (ttnn may add padding dims).
    tt_output_torch = tt_output_torch.reshape(ref_out.shape)

    passing, pcc_message = comp_pcc(ref_out, tt_output_torch, 0.99)
    msg_str = str(pcc_message).strip()
    try:
        pcc_value = float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        pcc_value = float(match.group(0)) if match else float("nan")
    print(f"comp_pcc: passing={passing}, message={pcc_message}")
    return pcc_value


def test_tt_conformer_feature_projection():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_value = _run_conformer_feature_projection_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_value > 0.99, f"PCC {pcc_value} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_value = float("nan")
    try:
        pcc_value = _run_conformer_feature_projection_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {"pcc": pcc_value, "passed": pcc_value > 0.99}
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
