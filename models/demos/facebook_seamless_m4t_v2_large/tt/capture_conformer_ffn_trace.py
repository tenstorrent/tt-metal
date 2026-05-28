# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 Conformer FFN block.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/conformer_ffn.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for
``kind=mlp``, which requires at least one matmul-family call and at least
one supported activation (here ``ttnn.silu``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_ffn import ConformerFfn

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "conformer_ffn.pt"
OUT_PATH = Path(__file__).resolve().parent / "conformer_ffn.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    sd = golden["state_dict"]
    intermediate_weight: torch.Tensor = sd["intermediate_dense"]["weight"]
    intermediate_bias: torch.Tensor = sd["intermediate_dense"]["bias"]
    output_weight: torch.Tensor = sd["output_dense"]["weight"]
    output_bias: torch.Tensor = sd["output_dense"]["bias"]

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_ffn = ConformerFfn(
            device=device,
            intermediate_weight=intermediate_weight,
            intermediate_bias=intermediate_bias,
            output_weight=output_weight,
            output_bias=output_bias,
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
        tt_output = tt_ffn(tt_input)
        ttnn.synchronize_device(device)
        captured = ttnn.graph.end_graph_capture()

        # Read output back to ensure the kernel actually ran.
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
