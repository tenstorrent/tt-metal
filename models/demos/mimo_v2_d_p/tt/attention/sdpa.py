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
import os

import ttnn

TILE = 32


def _fidelity():
    return getattr(ttnn.MathFidelity, os.environ.get("MIMO_SDPA_FIDELITY", "HiFi2"))


def _largest_dividing(n, candidates):
    for c in candidates:
        if n % c == 0:
            return c
    return candidates[-1]


def ring_program_config(mesh_device, q_chunk=None, k_chunk=None, sliding=False, q_local=None, kv_local=None):
    """GA: q128 / k1024 measured best on BH 2x2 at 640 and 2048 tokens/chip (53.5% / 57.5% of HiFi2 peak at
    32k context vs 50.2% / 54.9% for q256/k512). q is shrunk to divide the per-device Q slab (Galaxy SP8:
    640-token slabs); k stays 1024 (the op masks a padded last K chunk). SWA: the sliding ring supports q 64/128, k 128."""
    grid = mesh_device.compute_with_storage_grid_size()
    if q_chunk is None:
        q_chunk = int(os.environ["MIMO_SDPA_Q_CHUNK"]) if os.environ.get("MIMO_SDPA_Q_CHUNK") else (
            _largest_dividing(q_local, (128, 64, 32)) if q_local else 128)
    if k_chunk is None:
        if sliding:
            k_chunk = 128
        elif os.environ.get("MIMO_SDPA_K_CHUNK"):
            k_chunk = int(os.environ["MIMO_SDPA_K_CHUNK"])
        else:
            # A padded (masked) last K chunk is much cheaper than a small k_chunk (k=256 on a 16640-token shard
            # cost 53.6% -> 46.9% FPU util), so keep 1024 unless the shard itself is shorter.
            k_chunk = 1024 if not kv_local or kv_local >= 1024 else max(128, 1 << (kv_local.bit_length() - 1))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # last column = CCL workers
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=os.environ.get("MIMO_SDPA_EXP_APPROX", "0") == "1",
    )


def compute_config(mesh_device, fidelity=None):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity or _fidelity(),
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
):
    """q [1, nq_local, S_local, 192] (block-cyclic chunk, K/V already written at ``kv_actual``) ->
    [1, nq_local, S_local, v_dim]."""
    assert kv_cache.k.dtype == ttnn.bfloat8_b and kv_cache.v.dtype == ttnn.bfloat8_b
    sp = mesh_device.shape[sp_axis]
    pc = program_config or ring_program_config(
        mesh_device, sliding=window is not None, q_local=tt_q.shape[2], kv_local=kv_cache.max_seq_len // sp
    )
    ckc = compute_kernel_config or compute_config(mesh_device)
    tp = mesh_device.shape[1 - sp_axis]
    n_kv = kv_cache.n_kv_local * tp
    bufseq = gather_seq_len(window, pc.k_chunk_size, kv_cache.max_seq_len)
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(f"mimo_k_{bufseq}", n_kv, bufseq, kv_cache.k_dim, ttnn.bfloat8_b),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(f"mimo_v_{bufseq}", n_kv, bufseq, kv_cache.v_dim, ttnn.bfloat8_b),
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
    return out
