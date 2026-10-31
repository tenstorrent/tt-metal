# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can ttnn.embedding write straight into the resident residual layout? Compares with embedding + to_resident."""
import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    from_resident,
    residual_memory_config,
    to_resident,
)

B, S, W, V = 32, 512, 2048, 128256
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
table = ttnn.from_torch(torch.randn(V, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
ids = torch.randint(0, V, (B * S,), dtype=torch.int32).reshape(1, B * S)
tok = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
ref = to_resident(
    ttnn.reshape(ttnn.embedding(tok, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), [1, 1, B * S, W]), grid
)
try:
    direct = ttnn.embedding(
        tok, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=residual_memory_config(B * S, W, grid)
    )
    print("direct shape", direct.shape, direct.memory_config().memory_layout)
    a = ttnn.to_torch(from_resident(ref, B * S, W)).float().reshape(-1)
    b = ttnn.to_torch(ttnn.to_memory_config(direct, ttnn.DRAM_MEMORY_CONFIG)).float().reshape(-1)
    print("max diff", (a - b).abs().max().item())
except Exception as e:
    print("direct failed:", str(e).splitlines()[0][:300])
ttnn.close_device(dev)
