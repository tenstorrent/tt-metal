# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 text decoder.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/text_decoder.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for the
full-decoder composite block: the trace must include the embedding gather
(``ttnn.embedding``), the per-layer matmul/SDPA/LayerNorm calls (with
cross-attention), and the final terminal LayerNorm.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "text_decoder.pt"
OUT_PATH = Path(__file__).resolve().parent / "text_decoder.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]
    encoder_hidden_states: torch.Tensor = golden["encoder_hidden_states"]
    decoder_attention_mask: torch.Tensor = golden["decoder_attention_mask"]
    encoder_attention_mask: torch.Tensor = golden["encoder_attention_mask"]
    state_dict = golden["state_dict"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    padding_idx = int(cfg["padding_idx"])

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = TextDecoder(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            embed_tokens_weight=state_dict["embed_tokens"]["weight"],
            embed_positions_weights=state_dict["embed_positions_weights"],
            layers_state_dict=state_dict["layers"],
            final_layer_norm_state_dict=state_dict["layer_norm"],
            eps=eps,
            padding_idx=padding_idx,
            weight_dtype=ttnn.bfloat16,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(
            input_ids,
            encoder_hidden_states_torch=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
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
