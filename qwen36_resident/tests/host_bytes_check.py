# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""device check: vllm_rows._host_bytes counts every chip's shard of a mesh host tensor (volume() is one shard's)."""
import ttnn
from qwen36_resident.tests.long_ctx import close_mesh, open_mesh
from qwen36_resident.vllm_rows import _host_bytes

mesh = open_mesh()
try:
    t = ttnn.empty([1024, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    h = {"n_tokens": 5, "state": ttnn.from_device(t)}
    print("host bytes", _host_bytes(h), "expect", 4 * 1024 * 32 * 2)
finally:
    close_mesh(mesh)
