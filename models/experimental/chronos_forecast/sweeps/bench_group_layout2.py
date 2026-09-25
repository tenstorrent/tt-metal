# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Row-major permute round trip with bf8 input and L1 placement at an L1 chunk shape."""

import time

import torch
import ttnn


def bench(fn, iters=5):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    return (time.perf_counter() - t0) / iters * 1e3


def rm_permute(x, mem):
    r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
    p = ttnn.permute(r, (1, 0, 2), memory_config=mem)
    ttnn.deallocate(r)
    t = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=mem)
    ttnn.deallocate(p)
    return t


dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
dev.enable_program_cache()
try:
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        for shape in ((64, 133, 768), (133, 64, 768)):
            for mem_name, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
                src = torch.randn(shape)
                x = ttnn.from_torch(src, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)
                try:
                    tiled = bench(lambda: ttnn.permute(x, (1, 0, 2), memory_config=mem))
                    rm = bench(lambda: rm_permute(x, mem))
                    y = rm_permute(x, mem)
                    ok = torch.equal(ttnn.to_torch(y), ttnn.to_torch(ttnn.permute(x, (1, 0, 2))))
                    print(
                        f"[LAYOUT2] {dtype} {shape} {mem_name}: tiled {tiled:.3f} ms, rm {rm:.3f} ms, "
                        f"exact={ok} out_dtype={y.dtype}"
                    )
                    ttnn.deallocate(y)
                except Exception as e:  # noqa: BLE001
                    print(f"[LAYOUT2] {dtype} {shape} {mem_name}: FAILED {str(e).splitlines()[0][:160]}")
                ttnn.deallocate(x)
finally:
    ttnn.close_mesh_device(dev)
