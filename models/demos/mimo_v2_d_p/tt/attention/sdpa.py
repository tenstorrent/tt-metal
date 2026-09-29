# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill attention core over the block-cyclic SP KV cache (ring-joint SDPA, SP > 1).

* GA layers: full causal ring over the cached prefix; K head_dim 192, V head_dim 128 (the causal
  chunked path accepts DK != DV).
* SWA layers: one-hop compact-halo sliding ring (window 128) with per-head attention sink, V at 128 (the
  sliding path's VDH == DH check was relaxed to VDH <= DH; its kernels are generic in vDHt).

Requires chunk_size / SP >= window (one-hop halo) and chunk-aligned ``kv_actual`` for SWA (the sliding
ring needs ``logical_n`` on a ring-group boundary).
"""

import math

import ttnn

TILE = 32


def _largest_dividing(n, candidates):
    for c in candidates:
        if n % c == 0:
            return c
    return candidates[-1]


def ring_program_config(
    mesh_device,
    q_chunk=None,
    k_chunk=None,
    sliding=False,
    q_local=None,
    kv_local=None,
    k_split=1,
    exp_approx=False,
    two_level=False,
    two_level_fold=0,
):
    """GA: q128 / k1024 measured best on BH 2x2 at 640 and 2048 tokens/chip (53.5% / 57.5% of HiFi2 peak at
    32k context vs 50.2% / 54.9% for q256/k512). q is shrunk to divide the per-device Q slab (Galaxy SP8:
    640-token slabs); k stays 1024 (the op masks a padded last K chunk). SWA: the sliding ring supports q 64/128, k 128.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if q_chunk is None:
        q_chunk = _largest_dividing(q_local, (128, 64, 32)) if q_local else 128
    if k_chunk is None:
        if sliding:
            k_chunk = 128
        else:
            # A padded (masked) last K chunk is much cheaper than a small k_chunk (k=256 on a 16640-token shard
            # cost 53.6% -> 46.9% FPU util), so keep 1024 unless the shard itself is shorter.
            k_chunk = 1024 if not kv_local or kv_local >= 1024 else max(128, 1 << (kv_local.bit_length() - 1))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # last column = CCL workers
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
        ring_k_split=k_split,
        ring_two_level=two_level,
        ring_two_level_fold=two_level_fold,
    )


def default_k_split(window, chunk_local, kv_actual, sp, k_chunk=1024, forced=None, n_kv_local=None):
    """K split for GA (full-attention) layers: 2 at >= 2048 tokens per chip, 3 below (2048 tok/chip: 512 units ->
    1024 on 100 cores; 640: 160 -> 480), only when every partition gets K chunks from the prefix in every ring
    iteration (the op requires a valid chunk per partition), and not with one local KV head (see below). ``forced`` = N (MiMoRuntimeOptions.sdpa_k_split) forces N
    (still subject to that condition), 0 / 1 turns it off."""
    if window is not None:
        return 1
    if forced is None and n_kv_local == 1:
        # one local KV head (Galaxy TP4): the unsplit op multicasts K / V over each core row, the split cannot (its
        # partitions read disjoint K chunks) and loses: 640 tok/chip 1.40 -> 1.60 + 0.045 ms, 2048 4.05 -> 4.07 +
        # 0.09 ms (32K context, test_sdpa_ksplit_perf glx-heads)
        return 1
    s = forced if forced is not None else (2 if chunk_local >= 2048 else 3)
    if s <= 1:
        return 1
    # every ring iteration's shard holds kv_actual / sp prefix tokens in whole chunk rounds: need s K chunks of them
    return s if kv_actual // sp >= s * k_chunk else 1


def schedulable_k_split(k_split, n_heads_local, q_local, q_chunk, n_cores):
    """The largest split <= ``k_split`` the op can schedule: every core needs >= 2 (head, Q chunk, partition) units
    (heads x ceil(q_local / q_chunk) x split >= 2 x cores); 1 when no split > 1 qualifies."""
    units = n_heads_local * -(-q_local // q_chunk)
    for s in range(k_split, 1, -1):
        if units * s // n_cores >= 2:
            return s
    return 1


def compute_config(mesh_device, fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # sinks require the streaming (non-fp32-dest) compute path
        packer_l1_acc=False,
    )


def gather_seq_len(window, k_chunk, full_seq):
    if window is None:
        return full_seq
    return max(math.ceil((window - 1) / k_chunk) * k_chunk, TILE)


def ring_attention(
    tt_q,
    kv_cache,
    *,
    kv_actual,
    logical_n,
    window,
    sink,
    layer_slot,
    mesh_device,
    ccl_manager,
    sp_axis,
    scale,
    program_config=None,
    compute_kernel_config=None,
    k_split=1,
    two_level=False,
    two_level_fold=0,
):
    """q [1, nq_local, S_local, 192] (block-cyclic chunk, K/V already written at ``kv_actual``) ->
    [1, nq_local, S_local, v_dim]. ``k_split`` > 1 (GA only): every (head, Q chunk) is split over k_split key
    partitions on separate cores (even work over the grid); the op merges them (ttnn.transformer.sdpa_k_split_merge)."""
    assert kv_cache.k.dtype == ttnn.bfloat8_b and kv_cache.v.dtype == ttnn.bfloat8_b
    sp = mesh_device.shape[sp_axis]
    cfg = lambda ks: ring_program_config(
        mesh_device,
        sliding=window is not None,
        q_local=tt_q.shape[2],
        kv_local=kv_cache.max_seq_len // sp,
        k_split=ks,
        two_level=two_level,
        two_level_fold=two_level_fold,
    )
    if program_config is None:
        base = cfg(1)
        grid = base.compute_with_storage_grid_size
        k_split = schedulable_k_split(k_split, tt_q.shape[1], tt_q.shape[2], base.q_chunk_size, grid.x * grid.y)
    pc = program_config or cfg(k_split)
    ckc = compute_kernel_config or compute_config(mesh_device)
    tp = mesh_device.shape[1 - sp_axis]
    n_kv = kv_cache.n_kv_local * tp
    bufseq = gather_seq_len(window, pc.k_chunk_size, kv_cache.max_seq_len)
    out, _, stats = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            f"mimo_k_{bufseq}", n_kv, bufseq, kv_cache.k_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            f"mimo_v_{bufseq}", n_kv, bufseq, kv_cache.v_dim, ttnn.bfloat8_b
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=pc,
        compute_kernel_config=ckc,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        kv_cache_batch_idx=layer_slot,
        kv_actual_isl=kv_actual,
        attention_sink=sink,
        sliding_window_size=window,
    )
    stats.deallocate(True)
    return out
