# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 T2U encoder.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/t2u_encoder.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for the
full-T2U-encoder composite block: the trace must include the per-layer
matmul/SDPA/LayerNorm calls plus the final terminal LayerNorm. Unlike the
NLLB text encoder, the T2U encoder has no token/positional embedding ops.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_encoder import T2uEncoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "t2u_encoder.pt"
OUT_PATH = Path(__file__).resolve().parent / "t2u_encoder.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    inputs_embeds: torch.Tensor = golden["inputs_embeds"]
    attention_mask_4d: torch.Tensor = golden["attention_mask_4d"]
    state_dict = golden["state_dict"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = T2uEncoder(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            layers_state_dict=state_dict["layers"],
            final_layer_norm_state_dict=state_dict["final_layer_norm"],
            eps=eps,
            weight_dtype=ttnn.bfloat16,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(inputs_embeds, attention_mask_torch=attention_mask_4d)
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


if __name__ == "__main__":
    main()
