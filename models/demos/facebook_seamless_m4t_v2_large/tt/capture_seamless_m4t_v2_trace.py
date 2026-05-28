# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 top-level T2TT block.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/seamless_m4t_v2.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for the
top-level composite block: the trace must include the encoder embedding
gather (``ttnn.embedding``), the decoder embedding gather, the per-layer
matmul/SDPA/LayerNorm calls (with cross-attention), the final terminal
LayerNorm, and the LM head ``ttnn.linear``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tests.test_tt_seamless_m4t_v2 import _build_hf_state_dict
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_m4t_v2 import SeamlessM4Tv2

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "seamless_m4t_v2.pt"
OUT_PATH = Path(__file__).resolve().parent / "seamless_m4t_v2.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]
    decoder_input_ids: torch.Tensor = golden["decoder_input_ids"]
    encoder_attention_mask: torch.Tensor = golden["encoder_attention_mask"]
    decoder_attention_mask: torch.Tensor = golden["decoder_attention_mask"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    encoder_padding_idx = int(cfg["encoder_padding_idx"])
    decoder_padding_idx = int(cfg["decoder_padding_idx"])

    state_dict = _build_hf_state_dict()

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = SeamlessM4Tv2(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            text_encoder_state_dict=state_dict["text_encoder"],
            text_decoder_state_dict=state_dict["text_decoder"],
            lm_head_state_dict=state_dict["lm_head"],
            eps=eps,
            encoder_padding_idx=encoder_padding_idx,
            decoder_padding_idx=decoder_padding_idx,
            weight_dtype=ttnn.bfloat16,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(
            input_ids,
            decoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
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
