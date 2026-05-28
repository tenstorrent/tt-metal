# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 text decoder layer.

Writes a JSON list of function names invoked during one full forward pass
(self-attn + cross-attn + FFN, the masked path) to
``models/demos/facebook_seamless_m4t_v2_large/tt/text_decoder_layer.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for
``kind=decoder_layer`` composite blocks: the trace must include both the
matmul-family calls (FFN / projections) and the supporting LayerNorm /
softmax / activation calls used by the inner sub-blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder_layer import TextDecoderLayer

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "text_decoder_layer.pt"
OUT_PATH = Path(__file__).resolve().parent / "text_decoder_layer.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    enc_torch: torch.Tensor = golden["encoder_hidden_states"]
    self_mask_torch: torch.Tensor = golden["self_attention_mask"]
    enc_mask_torch: torch.Tensor = golden["encoder_attention_mask"]
    state_dict = golden["state_dict"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = TextDecoderLayer(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            eps=eps,
            weight_dtype=ttnn.bfloat16,
        )

        def _to_tt(t: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                t,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tt_input = _to_tt(x_torch)
        tt_enc = _to_tt(enc_torch)
        tt_self_mask = _to_tt(self_mask_torch)
        tt_enc_mask = _to_tt(enc_mask_torch)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(
            tt_input,
            encoder_hidden_states=tt_enc,
            self_attention_mask=tt_self_mask,
            encoder_attention_mask=tt_enc_mask,
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
