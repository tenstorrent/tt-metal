# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill attention core over the block-cyclic SP KV cache.

* SP > 1 (2x2, 4x1): cache-backed ``ring_joint_scaled_dot_product_attention`` (gpt_oss_d_p
  ``dense_sp_attention``); sliding layers use the one-hop compact halo. The halo exchange is cyclic,
  so the SP-axis CCL topology must be Ring where the fabric wraps (see ``per_axis_topology``).
* SP == 1 (1x4): the ring op's sliding path needs a neighbour halo, so it is composed from primitives:
    - full layers: the same ring op (ring of one) works as-is,
    - sliding layers: tail = cache[kv_actual - H : kv_actual] (H = window rounded to tiles),
      K/V = [tail | chunk], Q front-padded by H rows, local causal SDPA with the window, drop the pad.
  Marked as an op-generation target (``ring_sdpa.sliding_sp1``).

Requires (validated): per-device chunk slab >= sliding window when SP > 1 (one-hop halo).
"""

import math

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.dense_sp import dense_sp_attention

TILE = 32


def ring_program_config(mesh_device, q_chunk=128, k_chunk=128):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # last column = CCL workers
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def compute_config(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _local_sliding_sdpa(tt_q, cache_k, cache_v, tt_k, tt_v, *, kv_actual, window, slot, mesh_device, scale):
    """SP=1 sliding attention: [cache tail | chunk] keys, front-padded Q, local windowed SDPA."""
    S = tt_q.shape[2]
    halo = min(kv_actual, math.ceil((window - 1) / TILE) * TILE)
    k_bf8 = tt_k if tt_k.dtype == ttnn.bfloat8_b else ttnn.typecast(tt_k, ttnn.bfloat8_b)
    v_bf8 = tt_v if tt_v.dtype == ttnn.bfloat8_b else ttnn.typecast(tt_v, ttnn.bfloat8_b)
    if halo > 0:
        nkv, hd = cache_k.shape[1], cache_k.shape[3]
        sl = lambda c: ttnn.slice(
            c, [slot, 0, kv_actual - halo, 0], [slot + 1, nkv, kv_actual, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_full = ttnn.concat([sl(cache_k), k_bf8], dim=2)
        v_full = ttnn.concat([sl(cache_v), v_bf8], dim=2)
        # On-device tile pad has no front padding: prepend a zero block instead.
        zeros = ttnn.zeros(
            [tt_q.shape[0], tt_q.shape[1], halo, tt_q.shape[3]],
            dtype=tt_q.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_full = ttnn.concat([zeros, tt_q], dim=2)
        zeros.deallocate(True)
    else:
        k_full, v_full, q_full = k_bf8, v_bf8, tt_q
    out = ttnn.transformer.scaled_dot_product_attention(
        q_full,
        k_full,
        v_full,
        is_causal=True,
        scale=scale,
        sliding_window_size=window,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        ),
        compute_kernel_config=compute_config(mesh_device),
    )
    if halo > 0:
        sliced = ttnn.slice(out, [0, 0, halo, 0], [out.shape[0], out.shape[1], halo + S, out.shape[3]])
        for t in (out, k_full, v_full, q_full):
            t.deallocate(True)
        out = sliced
    return out


def chunk_attention(
    tt_q,
    kv_cache,
    tt_k,
    tt_v,
    *,
    kv_actual,
    logical_n,
    window,
    layer_slot,
    num_slots_layers,
    mesh_device,
    ccl_manager,
    sp_axis,
    scale=1.0,
):
    """Attention for one chunk whose K/V were ALREADY written into ``kv_cache`` at ``kv_actual``.

    tt_q [1, nq_local, S_local, D] block-cyclic over the chunk; tt_k/tt_v this chunk's K/V (needed only by the
    SP=1 sliding path). ``layer_slot`` = user*num_layers + layer (the cache batch index).
    """
    sp = mesh_device.shape[sp_axis]
    if sp == 1 and window is not None:
        return _local_sliding_sdpa(
            tt_q, kv_cache.k, kv_cache.v, tt_k, tt_v, kv_actual=kv_actual, window=window, slot=layer_slot, mesh_device=mesh_device, scale=scale
        )
    return dense_sp_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        kv_actual=kv_actual,
        logical_n=logical_n,
        n_kv=kv_cache.n_kv_local * mesh_device.shape[1 - sp_axis],
        cache_global=kv_cache.max_seq_len,
        head_dim=kv_cache.head_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        program_config=ring_program_config(mesh_device),
        compute_kernel_config=compute_config(mesh_device),
        scale=scale,
        cluster_axis=sp_axis,
        sliding_window_size=window,
        slot_idx=0,
        layer_idx=layer_slot,
        num_layers=num_slots_layers,
        write_chunk=False,
    )
