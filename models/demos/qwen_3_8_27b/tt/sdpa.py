# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SP-sharded causal GQA attention via ``ttnn.transformer.ring_joint_scaled_dot_product_attention``.

Structure (call shape, persistent gather buffers, CCL grid offset, Linear topology, is_balanced=False)
from ``minimax_m3/tt/attention/dense_sp.py``; shape-tuned values re-derived for head_dim 256:

* ``k_chunk_size`` — minimax uses 512 at head_dim 128; at 256 the K/V chunk CBs double, so 256 here.
* compute config — HiFi4. ``fp32_dest_acc_en``: **True** for the live-KV ring (measured: rel. error
  0.041 -> 0.012 at 24q/4kv/hd256, about CPU bf16's 0.010), **False** only for the cache-read ring,
  which refuses fp32 accumulation (TT_FATAL in ring_joint_sdpa_program_factory) — the recipe's
  "local constraint". minimax sets False for both (inherited, over-broad).

Two entry points:
  ``ring_sdpa_nocache`` — first (or only) chunk: ring over the chunk's own SP-sharded K/V.
  ``ring_sdpa_cache``   — later chunks: ring over the accumulated block-cyclic cache prefix.
"""

import ttnn

Q_CHUNK = 128
K_CHUNK = 256


def sdpa_configs(mesh_device, q_chunk=Q_CHUNK, k_chunk=K_CHUNK, fp32_acc=False):
    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False
    )
    return prog, kcfg


def _common(mesh_device, ccl, scale, sp_axis, program_config, compute_kernel_config):
    return dict(
        joint_strategy="rear",
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
        num_links=ccl.num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
    )


def ring_sdpa_nocache(tt_q, tt_k, tt_v, *, mesh_device, ccl, n_kv, head_dim, logical_n, scale, sp_axis=0, configs=None):
    """q ``[1, nq_local, s_local, D]``, k/v ``[1, nkv_local, s_local, D]`` bf16, SP-contiguous -> ``[1, nq_local, s_local, D]``."""
    prog, kcfg = configs or sdpa_configs(mesh_device, fp32_acc=True)
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl.get_ring_gather_buffer("nocache_k", n_kv, logical_n, head_dim, tt_k.dtype),
        persistent_output_buffer_v=ccl.get_ring_gather_buffer("nocache_v", n_kv, logical_n, head_dim, tt_v.dtype),
        logical_n=logical_n,
        **_common(mesh_device, ccl, scale, sp_axis, prog, kcfg),
    )
    return out


def cache_attn_mode() -> str:
    """How chunks after the first read the cached prefix: ``masked`` (default) or ``ring``."""
    import os

    mode = os.environ.get("QWEN38_CACHE_ATTN", "masked")
    assert mode in ("masked", "ring"), mode
    return mode


def cache_read_mask(sp: int, start: int, tokens: int, dtype=None):
    """Host additive mask ``[sp, 1, s_local, (start + tokens)]`` for ``masked_sdpa_cache``.

    Keys arrive in the SP all-gather order of the block-cyclic cache rows (chip-major: chip j's local
    row lr holds position ``(lr // cl) * tokens + j * cl + lr % cl``, cl = tokens / sp); row r's query i
    sits at ``start + r * cl + i``. Causality is by true position, so key order does not matter.
    """
    import torch

    cl = tokens // sp
    rows = (start + tokens) // sp
    lr = torch.arange(rows)
    kpos = torch.cat([(lr // cl) * tokens + j * cl + lr % cl for j in range(sp)])  # [sp*rows]
    qpos = start + torch.arange(sp)[:, None] * cl + torch.arange(cl)[None, :]  # [sp, cl]
    allowed = kpos[None, None, :] <= qpos[:, :, None]
    m = torch.zeros(sp, 1, cl, kpos.numel())
    m.masked_fill_(~allowed[:, None], float("-inf"))
    return m


def masked_sdpa_cache(
    tt_q, caches, *, mesh_config, rows_local, slot, mask, scale, program_config, compute_kernel_config
):
    """Cache read composed from primitives (the ring cache-read op refuses fp32 accumulation, costing
    ~4% relative error at hd 256): slice this layer's written cache rows, all-gather them over SP, and
    run plain SDPA with fp32 accumulation under the position mask from ``cache_read_mask``."""
    kv = []
    for cache in (caches.k, caches.v):
        D = cache.shape[-1]
        sl = ttnn.slice(cache, [slot, 0, 0, 0], [slot + 1, 1, rows_local, D], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.typecast(sl, ttnn.bfloat16)
        ttnn.deallocate(sl)
        kv.append(mesh_config.all_gather_sp(b, dim=2))
        ttnn.deallocate(b)
    out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        kv[0],
        kv[1],
        attn_mask=mask,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    for t in kv:
        ttnn.deallocate(t)
    return out


def plain_sdpa_configs(mesh_device, q_chunk=Q_CHUNK, k_chunk=K_CHUNK):
    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    return prog, kcfg


def ring_sdpa_cache(
    tt_q,
    caches,
    *,
    mesh_device,
    ccl,
    n_kv,
    kv_actual,
    logical_n,
    slot_idx,
    layer_idx,
    scale,
    sp_axis=0,
    configs=None,
):
    """q ``[1, nq_local, s_local, D]`` for the chunk starting at ``kv_actual`` (already written into the
    cache) attends causally over the cached prefix ``[0, logical_n)``."""
    prog, kcfg = configs or sdpa_configs(mesh_device)
    D = caches.head_dim
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        caches.k,
        caches.v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl.get_ring_gather_buffer("cache_k", n_kv, caches.max_seq_len, D, caches.k.dtype),
        persistent_output_buffer_v=ccl.get_ring_gather_buffer("cache_v", n_kv, caches.max_seq_len, D, caches.v.dtype),
        logical_n=logical_n,
        kv_cache_batch_idx=slot_idx,
        kv_actual_isl=kv_actual,
        kv_cache_num_layers=caches.num_layers,
        kv_cache_layer_idx=layer_idx,
        **_common(mesh_device, ccl, scale, sp_axis, prog, kcfg),
    )
    return out
