# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
THE LINK: ring_joint reading from the GQA chunked-KV cache (cache-read mode), SP=8 × TP=4 on (8,4).

Ties the two validated building blocks together — the GQA chunked-KV cache write
(test_kv_cache_gqa_sp_vs_ref) and the SP ring_joint op (test_ring_joint_sp_vs_ref) — into the actual
chunked-prefill attention: write K/V into the cache, then run
`ring_joint_scaled_dot_product_attention` with `kv_cache_batch_idx` + `kv_actual_isl` so it reads the
accumulated prefix FROM the cache (block-cyclic, SP-sharded) and runs causal online-softmax over it.
Grouped V (cache stays 4 heads → 1/chip; NO inflation). vs a full-causal-GQA torch golden.

Single chunk = whole sequence (kv_actual_isl=0, logical_n=S) — the minimal cache-read verification;
multi-chunk accumulation + SP-RoPE are follow-ups. This is the path that wires into prefill.py:124.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, HEAD_DIM = 64, 4, 128


def _torch_gqa_causal(q, k, v):
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v  # [1, NQ, S, HD]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "n_chunks,cap_chunks,chunk_local",
    # (chunks written = logical_n, cache/buffer capacity = max_seq_len, chunk len/chip).
    #   2x256, 2x5120: tight buffer (cap==written), logical_n=10240 — PASS.
    #   3x5120_tight (cap==written, logical_n=15360) — PASSES (tight buffer is fine!).
    #   3x5120_cap4 (write 3 -> logical_n=15360, buffer=4 chunks=20480 > logical_n): the MODEL's chunk-2
    #     state (cache_global=max_seq_len > logical_n, the "oversize persistent buffer" case #45840).
    #     Isolates the 50k chunked-prefill hang to ring_joint GQA cache-read with an OVERSIZE gather buffer.
    [(2, 2, 32), (2, 2, 640), (3, 3, 640), (3, 4, 640)],
    ids=["2x256", "2x5120", "3x5120_tight", "3x5120_cap4"],
)
def test_ring_joint_cache_read_sp(mesh_device, device_params, n_chunks, cap_chunks, chunk_local, reset_seeds):
    """ring_joint cache-read: last chunk's Q attends to the full cached prefix, SP=8 x TP=4, vs golden.

    Chunked mode requires Q.seq < cached K.seq, so we write n_chunks into the cache (= valid prefix =
    logical_n) and run attention for ONLY the LAST chunk's queries (kv_actual_isl = prefix-before-it).
    The cache/persistent-gather buffer is allocated at cap_chunks (== max_seq_len); when cap > n_chunks
    the gather buffer is OVERSIZE vs logical_n (the real M3 chunked-prefill layout).
    """
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis, tp_axis = rows, cols, 0, 1
    assert cap_chunks >= n_chunks, "capacity must hold the written prefix"
    C = chunk_local
    chunk_global = sp * C  # per-chunk global seq
    cap_global = cap_chunks * chunk_global  # cache + persistent-buffer capacity (== max_seq_len)
    logical_global = n_chunks * chunk_global  # valid written prefix (== logical_n)
    kv_actual_last = (n_chunks - 1) * chunk_global  # prefix length before the last chunk

    torch.manual_seed(0)
    q = torch.randn(1, NQ, logical_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, logical_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, logical_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ref_full = _torch_gqa_causal(q.float(), k.float(), v.float())  # [1, NQ, logical_global, HD]
    ref = ref_full[:, :, kv_actual_last:, :]  # golden for the LAST chunk's query positions

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # --- write all chunks into the GQA chunked-KV cache (block-cyclic) ---
    cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, cap_global, list(mesh_device.shape), sp_axis, 1)
    cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, cap_global, list(mesh_device.shape), sp_axis, 1)
    wr_dims = [None, None]
    wr_dims[sp_axis], wr_dims[tp_axis] = 2, 1

    def bc_index(kv_actual):
        pos = rotated_chip_positions(kv_actual, sp, C)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def make_chunk(src, kv_actual):
        chunk = src[:, bc_index(kv_actual), :].reshape(1, NKV, chunk_global, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=wr_dims),
        )

    def write(cache, src, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(src, kv_actual),
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )

    # Write the PRIOR chunks into the cache; the LAST chunk is written by dense_sp_attention below.
    for c in range(n_chunks - 1):
        kv_actual = c * chunk_global
        write(cache_k, k[0], kv_actual)
        write(cache_v, v[0], kv_actual)
    ttnn.synchronize_device(mesh_device)

    # --- Q = the LAST chunk's queries, block-cyclic within that chunk, sharded (seq rows, heads cols) ---
    last_idx = bc_index(kv_actual_last)  # global positions of the last chunk, block-cyclic
    q_bc = q[:, :, last_idx, :]  # [1, NQ, chunk_global, HD]
    q_dims = [None, None]
    q_dims[sp_axis], q_dims[tp_axis] = 2, 1
    tt_q = ttnn.from_torch(
        q_bc,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=q_dims),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,  # Pavle's minimax3_gqa_causal_perf tuning
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    # dense_sp_attention writes the LAST chunk into the cache, then ring_joint cache-read: K/V come
    # from the cache (kv_cache_batch_idx=slot 0); kv_actual_isl = prefix before the last chunk,
    # logical_n = full valid prefix (Q attends causally over [0:logical_n]).
    out = dense_sp_attention(
        tt_q,
        cache_k,
        cache_v,
        make_chunk(k[0], kv_actual_last),
        make_chunk(v[0], kv_actual_last),
        kv_actual=kv_actual_last,
        logical_n=logical_global,
        n_kv=NKV,
        cache_global=cap_global,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=HEAD_DIM**-0.5,
        cluster_axis=sp_axis,
    )

    # out per chip [1, NQ/tp, C, HD], block-cyclic over the LAST chunk. Gather heads (cols) + seq (rows).
    dts = ttnn.get_device_tensors(out)
    row_t = []
    for r in range(rows):
        row_t.append(torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for c in range(cols)], dim=1))  # [1,NQ,C,HD]
    full_bc = torch.cat(row_t, dim=2)  # [1, NQ, chunk_global, HD] block-cyclic over the last chunk
    # invert block-cyclic -> natural order WITHIN the last chunk (last_idx are global; subtract the offset)
    local_pos = last_idx - kv_actual_last  # positions in [0, chunk_global)
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[local_pos] = torch.arange(chunk_global)
    full = full_bc[:, :, inv, :]  # [1, NQ, chunk_global, HD] natural order (= positions kv_actual_last:cache_global)

    passing, pcc = comp_pcc(ref, full, 0.99)
    logger.info(f"ring_joint CACHE-READ SP=8 x TP=4 vs ref: pcc={pcc}")
    assert passing, f"cache-read PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "chunk_list",
    # Two successive cache-read calls sharing ONE ccl + buffer, processing these chunk indices:
    #   grow      : chunk 1 (logical_n=10240) then chunk 2 (logical_n=15360) — logical_n GROWS.
    #   same15360 : chunk 2 (logical_n=15360) twice — logical_n IDENTICAL across the two calls.
    # If grow HANGS but same15360 PASSES -> the bug is the logical_n CHANGE across calls (stale
    # ring/mask/all-gather state). If same15360 ALSO hangs -> it's reuse-invariant (any 2nd call
    # at 15360), i.e. op-internal state not reset per dispatch regardless of logical_n.
    [[1, 2], [2, 2]],
    ids=["grow", "same15360"],
)
@pytest.mark.parametrize("reset_between", [False, True], ids=["noreset", "reset"])
def test_ring_joint_cache_read_sp_accumulate(mesh_device, device_params, reset_between, chunk_list, reset_seeds):
    """Multi-call REUSE repro of the 50k chunked-prefill hang, op-level (no model/weights).

    The single-call test above PASSES even at logical_n=15360 with an oversize buffer. The model
    instead REUSES one CCLManager (shared ring-attention semaphores) + one persistent gather buffer
    (get_ring_gather_buffer, keyed by cache_global==max_seq_len) across successive chunk calls with a
    GROWING logical_n. This drives that exact pattern: cache capacity = 4 chunks (max_seq_len=20480),
    then run dense_sp_attention for chunk 1 (logical_n=10240) and chunk 2 (logical_n=15360) sharing the
    SAME ccl + buffer. Chunk 1 succeeds; if the op mis-handles cross-call semaphore/buffer reuse at the
    larger logical_n, chunk 2 DEADLOCKS — the op-level face of the galaxy hang.

    reset_between: when True, zero ALL of the CCLManager's global semaphores between the two calls
    (ring-attention + reduce-scatter/all-gather ping-pong + barrier). If that unblocks chunk 2, the
    dirty state is a caller-resettable global semaphore the GQA path fails to clear (→ M3 workaround +
    which semaphore to fix); if it still hangs, the dirty state is op-INTERNAL (program-local mcast
    semaphores / ring index) and must be fixed inside the op.
    """
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis, tp_axis = rows, cols, 0, 1
    C = 640
    chunk_global = sp * C  # 5120
    cap_global = 4 * chunk_global  # 20480 (== max_seq_len)
    n_process = 3  # chunks 0..2; the cache-read calls are chunks 1 and 2

    torch.manual_seed(0)
    total = n_process * chunk_global  # 15360 valid tokens
    k = torch.randn(1, NKV, total, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, total, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    q = torch.randn(1, NQ, total, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, cap_global, list(mesh_device.shape), sp_axis, 1)
    cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, cap_global, list(mesh_device.shape), sp_axis, 1)
    wr = [None, None]
    wr[sp_axis], wr[tp_axis] = 2, 1

    def bc_index(kv_actual):
        pos = rotated_chip_positions(kv_actual, sp, C)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def make_chunk(src, kv_actual, nheads):
        chunk = src[:, bc_index(kv_actual), :].reshape(1, nheads, chunk_global, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b if nheads == NKV else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=wr),
        )

    def write(cache, src, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(src, kv_actual, NKV),
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    # Pre-write chunks 0 and 1 so ANY cache-read call (incl. chunk 2 at logical_n=15360) has a valid prefix.
    for pc in (0, 1):
        write(cache_k, k[0], pc * chunk_global)
        write(cache_v, v[0], pc * chunk_global)
    ttnn.synchronize_device(mesh_device)

    all_sems = (
        list(ccl.ring_attention_ccl_semaphore_handles)
        + list(ccl.rs_ping_pong_semaphores)
        + list(ccl.ag_ping_pong_semaphores)
        + list(ccl.barrier_semaphore)
    )

    # Successive cache-read calls — SHARED ccl + persistent buffer (cache_global=cap_global fixed).
    for idx, c in enumerate(chunk_list):
        if reset_between and idx != 0:
            for sem in all_sems:
                ttnn.reset_global_semaphore_value(sem, 0)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[accumulate] reset {len(all_sems)} global semaphores before chunk {c}")
        kv_actual = c * chunk_global  # prefix before this chunk
        logical_n = (c + 1) * chunk_global  # valid prefix incl this chunk (10240 then 15360)
        q_bc = q[:, :, bc_index(kv_actual), :]
        tt_q = ttnn.from_torch(
            q_bc,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=wr),
        )
        logger.info(f"[accumulate] chunk {c}: kv_actual={kv_actual} logical_n={logical_n} (buffer cap={cap_global})")
        out = dense_sp_attention(
            tt_q,
            cache_k,
            cache_v,
            make_chunk(k[0], kv_actual, NKV),
            make_chunk(v[0], kv_actual, NKV),
            kv_actual=kv_actual,
            logical_n=logical_n,
            n_kv=NKV,
            cache_global=cap_global,
            head_dim=HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            program_config=prog,
            compute_kernel_config=kcfg,
            scale=HEAD_DIM**-0.5,
            cluster_axis=sp_axis,
            write_chunk=True,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
        logger.info(f"[accumulate] chunk {c} DONE (logical_n={logical_n})")

    logger.info("[accumulate] all cache-read chunks completed — NO HANG")
