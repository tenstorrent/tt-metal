# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can the resident residual be made without padding (uneven block shards)? Compares the fused op's y."""
import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    make_matmul_add_norm_constants,
    matmul_add_norm,
    residual_memory_config,
    to_resident,
)

M, K, N = 16384, 2048, 2048
dev = ttnn.open_device(device_id=0, l1_small_size=0)
grid = dev.compute_with_storage_grid_size()
torch.manual_seed(0)
x = ttnn.from_torch(torch.randn(1, 1, M, K) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
w = ttnn.from_torch(torch.randn(K, N) / K**0.5, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
r = ttnn.from_torch(torch.randn(1, 1, M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
consts = make_matmul_add_norm_constants(torch.ones(N), dev)
outs = []
for name in ("padded", "uneven"):
    if name == "padded":
        res = to_resident(r, grid)
    else:
        res = ttnn.to_memory_config(r, residual_memory_config(M, N, grid))
        print("uneven shard spec", res.memory_config().shard_spec, "padded shape", res.padded_shape)
    y = matmul_add_norm(x, w, res, consts, eps=1e-5, block=(4, 8, 7, 4, 1))
    outs.append(ttnn.to_torch(y).float())
    ttnn.deallocate(res)
d = (outs[0] - outs[1]).abs().max().item()
print(f"max |y_padded - y_uneven| = {d}")
ttnn.close_device(dev)
