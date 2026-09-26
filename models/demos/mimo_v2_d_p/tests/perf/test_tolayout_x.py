# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cost of the model's x prep today: ttnn.to_layout(row-major bf16 dispatch buffer [E * M, H] -> TILE bfp8)."""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
H = int(os.environ.get("MIMO_TL_H", "7168"))
E = int(os.environ.get("MIMO_TL_EXPERTS", "4"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "m", [int(v) for v in os.environ.get("MIMO_TL_M", "32,128,256,512,1024").split(",")], ids=lambda m: f"M{m}"
)
def test_tolayout_x(device, m):
    x = ttnn.from_torch(
        torch.randn(E * m, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tag = f"tolayout_H{H}_M{m}_E{E}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps({"tag": tag, "M": m, "E": E, "weight_bytes": 0, "flops": 0}) + "\n")
    for it in range(4):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        t = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        t.deallocate(True)
