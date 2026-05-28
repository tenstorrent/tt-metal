# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 T2U decoder.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/t2u_decoder.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for the
full-T2U-decoder composite block: the trace must include the per-layer
matmul/SDPA/LayerNorm calls plus the variance predictor's conv/relu/linear
ops, the char-token and positional embeddings, and the terminal LayerNorm.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder import T2uDecoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "t2u_decoder.pt"
OUT_PATH = Path(__file__).resolve().parent / "t2u_decoder.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    char_input_ids: torch.Tensor = golden["char_input_ids"]
    char_count_per_id: torch.Tensor = golden["char_count_per_id"]
    encoder_hidden_states: torch.Tensor = golden["encoder_hidden_states"]
    state_dict = golden["state_dict"]
    char_positional_weights: torch.Tensor = golden["char_positional_weights"]
    positional_weights: torch.Tensor = golden["positional_weights"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    embed_scale = float(cfg["embed_scale"])
    padding_idx = int(cfg["padding_idx"])
    conv_kernel_size = int(cfg["conv_kernel_size"])
    variance_predictor_kernel_size = int(cfg["variance_predictor_kernel_size"])

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = T2uDecoder(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            char_positional_weights=char_positional_weights,
            positional_weights=positional_weights,
            embed_scale=embed_scale,
            padding_idx=padding_idx,
            eps=eps,
            variance_predictor_kernel_size=variance_predictor_kernel_size,
            conv_kernel_size=conv_kernel_size,
            weight_dtype=ttnn.bfloat16,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = tt_block(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_hidden_states,
        )
        ttnn.synchronize_device(device)
        captured = ttnn.graph.end_graph_capture()

        # Read output back to ensure the kernel actually ran.
        _ = ttnn.to_torch(out["last_hidden_state"])

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
