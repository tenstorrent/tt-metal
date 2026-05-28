# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 VariancePredictor block.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/variance_predictor.traced_ops.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.variance_predictor import VariancePredictor

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "variance_predictor.pt"
OUT_PATH = Path(__file__).resolve().parent / "variance_predictor.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    sd = golden["state_dict"]
    cfg = golden.get("config", {})
    kernel_size = int(cfg.get("kernel_size", 3))
    eps = float(cfg.get("eps", 1e-5))

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
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
        tt_input = ttnn.from_torch(
            x_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(tt_input)
        ttnn.synchronize_device(device)
        captured = ttnn.graph.end_graph_capture()

        _ = ttnn.to_torch(tt_output)

        op_names: list[str] = []
        for node in captured:
            if node.get("node_type") == "function_start":
                name = node.get("params", {}).get("name")
                if name:
                    op_names.append(name)
    finally:
        ttnn.close_device(device)

    OUT_PATH.write_text(json.dumps(op_names, indent=2))
    print(f"Wrote {OUT_PATH} with {len(op_names)} ops")
    print(json.dumps(op_names, indent=2))


if __name__ == "__main__":
    main()
