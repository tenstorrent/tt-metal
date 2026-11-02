# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 37 scoping: the row-parallel w2 output's all-reduce (reduce-scatter on dim 3 + all-gather) on the
token-major [1, 1, 1024, 5120] shape against the transposed [1, 1, 5120, 1024] one the transposed-orientation w2
forward would hand it, through ttml's wrappers on the 1x4 mesh (defaults)."""

import time

import numpy as np
import ttnn
import ttml

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


for name, rs_shape, ag_shape in (
    ("token-major", (1, 1, 1024, 5120), (1, 1, 1024, 1280)),
    ("transposed ", (1, 1, 5120, 1024), (1, 1, 5120, 256)),
):
    x_rs, x_ag = dev_t(rs_shape), dev_t(ag_shape)
    t_rs = bench(lambda: ttml.ops.distributed.reduce_scatter(x_rs, 3))
    t_ag = bench(lambda: ttml.ops.distributed.all_gather(x_ag, 3))
    print(f"{name} {rs_shape}: reduce_scatter {t_rs:.0f} us, all_gather {t_ag:.0f} us", flush=True)
ttml.close_device_mesh()
