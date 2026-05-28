# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 HifiGanResidualBlock.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/hifigan_residual_block.traced_ops.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_residual_block import HifiGanResidualBlock

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "hifigan_residual_block.pt"
OUT_PATH = Path(__file__).resolve().parent / "hifigan_residual_block.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    sd = golden["state_dict"]
    cfg = golden.get("config", {})
    kernel_size = int(cfg.get("kernel_size", 3))
    dilation = tuple(cfg.get("dilation", (1, 3, 5)))
    leaky_relu_slope = float(cfg.get("leaky_relu_slope", 0.1))

    convs1_weights = [layer["weight"] for layer in sd["convs1"]]
    convs1_biases = [layer["bias"] for layer in sd["convs1"]]
    convs2_weights = [layer["weight"] for layer in sd["convs2"]]
    convs2_biases = [layer["bias"] for layer in sd["convs2"]]

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
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


if __name__ == "__main__":
    main()
