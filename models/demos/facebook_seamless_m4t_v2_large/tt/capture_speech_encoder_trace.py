# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 speech encoder.

Writes a JSON list of function names invoked during one forward pass
(through the full feature_projection -> N x conformer_encoder_layer ->
intermediate_ffn -> adapter -> inner_layer_norm pipeline) to
``models/demos/facebook_seamless_m4t_v2_large/tt/speech_encoder.traced_ops.json``.
This satisfies the orchestrator's ``assert_traced_ops`` guard for the
composite-encoder block: the trace must include the matmul-family calls
(FFN / projections), the supporting LayerNorm / softmax / GLU / Conv1d
calls, and the residual / multiply / add calls used by the macaron and
intermediate-FFN paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_encoder import SpeechEncoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "speech_encoder.pt"
OUT_PATH = Path(__file__).resolve().parent / "speech_encoder.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_features: torch.Tensor = golden["input_features"]
    state_dict = golden["state_dict"]
    cfg = golden["config"]

    batch = int(cfg["batch"])
    seq_len = int(cfg["seq_len"])
    feature_size = int(cfg["feature_size"])
    hidden = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    left_max = int(cfg["left_max_position_embeddings"])
    right_max = int(cfg["right_max_position_embeddings"])
    pos_type = cfg["position_embeddings_type"]
    conv_kernel = int(cfg["conv_depthwise_kernel_size"])
    adaptor_kernel = int(cfg["adaptor_kernel_size"])
    adaptor_stride = int(cfg["adaptor_stride"])
    chunk_size = cfg["speech_encoder_chunk_size"]
    left_chunk_num = int(cfg["speech_encoder_left_chunk_num"])
    add_adapter = bool(cfg["add_adapter"])
    eps = float(cfg["eps"])
    act_fn = cfg["speech_encoder_hidden_act"]

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_block = SpeechEncoder(
            device=device,
            state_dict=state_dict,
            feature_size=feature_size,
            hidden=hidden,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            batch_size=batch,
            eps=eps,
            speech_encoder_hidden_act=act_fn,
            left_max_position_embeddings=left_max,
            right_max_position_embeddings=right_max,
            position_embeddings_type=pos_type,
            conv_depthwise_kernel_size=conv_kernel,
            adaptor_kernel_size=adaptor_kernel,
            adaptor_stride=adaptor_stride,
            speech_encoder_chunk_size=chunk_size,
            speech_encoder_left_chunk_num=left_chunk_num,
            add_adapter=add_adapter,
            weight_dtype=ttnn.bfloat16,
        )
        tt_input = ttnn.from_torch(
            input_features,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(tt_input, attention_mask_2d=None)
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
