# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The row-parallel sum over a mesh axis (attention o_proj, MLP down, experts down).

``ttnn.all_reduce`` is a reduce_scatter + all_gather whose per-tile summation order depends on where the tile sits in
the tensor (the workers and ring directions split the tiles), so a token's output bits depend on its row in the chunk.
tt-d-gen's pulled-back last chunk recomputes rows the chunk before it already shipped at another offset and must
rewrite them with the same bytes. ALL_REDUCE_IMPL=gather_sum gathers the partials (an exact copy) and adds them in mesh
order on every chip, so the result depends on the token only and every replica holds the same bytes. gather_sum keeps
ttnn.all_reduce's precision (bf16 partials, each running sum rounded to bf16, as at every ring hop);
gather_sum_fp32 sums fp32 partials in fp32 and rounds once (partial_dtype).
"""

import ttnn

from .settings import settings


def partial_dtype():
    """dtype of the row-parallel matmul output that all_reduce sums."""
    return ttnn.float32 if settings.get("ALL_REDUCE_IMPL") == "gather_sum_fp32" else ttnn.bfloat16


def all_reduce(t, cluster_axis: int):
    """bf16 sum of `t` [1, 1, S, H] (partial_dtype) over the mesh axis, replicated along it; `t` is kept (the caller
    frees it)."""
    impl = settings.get("ALL_REDUCE_IMPL")
    if impl == "ttnn":
        return ttnn.all_reduce(t, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    n = t.device().shape[cluster_axis]
    if n == 1:
        return ttnn.typecast(t, ttnn.bfloat16) if t.dtype != ttnn.bfloat16 else ttnn.clone(t)
    acc_dtype = ttnn.float32 if impl == "gather_sum_fp32" else ttnn.bfloat16
    g = ttnn.all_gather(t, dim=1, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, n, S, H]
    s, h = t.shape[-2], t.shape[-1]
    acc = None
    for i in range(n):
        p = ttnn.slice(g, [0, i, 0, 0], [1, i + 1, s, h])
        if p.dtype != acc_dtype:
            pc = ttnn.typecast(p, acc_dtype)
            ttnn.deallocate(p)
            p = pc
        if acc is None:
            acc = p
            continue
        ttnn.add_(acc, p)
        ttnn.deallocate(p)
    ttnn.deallocate(g)
    if acc.dtype == ttnn.bfloat16:
        return acc
    out = ttnn.typecast(acc, ttnn.bfloat16)
    ttnn.deallocate(acc)
    return out
