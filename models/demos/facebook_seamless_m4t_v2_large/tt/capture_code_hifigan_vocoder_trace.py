# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the traced ttnn op list for the SeamlessM4T-v2 CodeHifiGanVocoder block.

Writes a JSON list of function names invoked during one forward pass to
``models/demos/facebook_seamless_m4t_v2_large/tt/code_hifigan_vocoder.traced_ops.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "code_hifigan_vocoder.pt"
OUT_PATH = Path(__file__).resolve().parent / "code_hifigan_vocoder.traced_ops.json"


def main() -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]
    speaker_id: torch.Tensor = golden["speaker_id"]
    lang_id: torch.Tensor = golden["lang_id"]
    sd = golden["state_dict"]
    cfg = golden.get("config", {})

    pad_token_id = int(cfg.get("t2u_pad_token_id", 1))
    kernel_size = int(cfg.get("variance_predictor_kernel_size", 3))
    upsample_rates = tuple(cfg.get("upsample_rates", (5, 4, 4, 2, 2)))
    upsample_kernel_sizes = tuple(cfg.get("upsample_kernel_sizes", (11, 8, 8, 4, 4)))
    resblock_kernel_sizes = tuple(cfg.get("resblock_kernel_sizes", (3, 7, 11)))
    resblock_dilation_sizes = tuple(tuple(d) for d in cfg.get("resblock_dilation_sizes", ((1, 3, 5),) * 3))
    leaky_relu_slope = float(cfg.get("leaky_relu_slope", 0.1))

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        tt_block = CodeHifiGanVocoder(
            device=device,
            state_dict=sd,
            pad_token_id=pad_token_id,
            variance_predictor_kernel_size=kernel_size,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            leaky_relu_slope=leaky_relu_slope,
            weight_dtype=ttnn.bfloat16,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output = tt_block(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)
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
