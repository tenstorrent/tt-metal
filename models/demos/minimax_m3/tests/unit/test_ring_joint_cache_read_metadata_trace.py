# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""dense_sp cache-read through ring_joint's trace-safe metadata path, checked against the host-int path:
  - bit-exact at two chunk depths served by one cached program (logical_n is off the hash on this path);
  - a cached program follows freshly allocated metadata tensors;
  - dense_sp's chunk write lands in the tensor-selected slot and offset;
  - the op refuses the hazards of this path: a wrong tensor form on a cache hit, host scalars next to the tensors,
    a layer index past the (user, layer) fold;
  - on the host path successive layers share one program and the folded batch index is re-patched per dispatch;
  - one captured trace re-targets the user slot and the depth by rewriting tensors in place between replays.

Two users x two layers with DISTINCT K/V, so a read from the wrong slot or layer changes the output instead of
reproducing it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric
from .ring_joint_cache_read_helpers import (
    HEAD_DIM,
    NKV,
    NQ,
    PCC_BF8_CACHE,
    SP_AXIS,
    gather_chunk,
    host_scalar,
    make_kv_chunk,
    make_q_chunk,
    meta_scalar,
    sdpa_configs,
    torch_gqa_causal,
)

# Attention on layer 1 of 2: exercises the (user, layer) cache fold beyond slot 0 / layer 0.
NUM_USERS, NUM_LAYERS, LAYER_IDX = 2, 2, 1


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2x256"])
def test_ring_joint_cache_read_metadata_trace(
    mesh_device, device_params, expect_error, n_chunks, chunk_local, reset_seeds
):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, sp_axis = rows, SP_AXIS
    chunk_global = sp * chunk_local
    cache_global = n_chunks * chunk_global
    last = (n_chunks - 1) * chunk_global  # kv_actual before the last chunk

    torch.manual_seed(0)
    q = torch.randn(1, NQ, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    v = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    refs = [torch_gqa_causal(q.float(), k[u].float(), v[u].float()) for u in range(NUM_USERS)]

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    cache_k = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )
    cache_v = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )

    def make_chunk(src, kv_actual):
        return make_kv_chunk(src, kv_actual, mesh_device, chunk_local)

    def make_q(kv_actual, on_device=True):
        return make_q_chunk(q, kv_actual, mesh_device, chunk_local, on_device)

    def gather(out, kv_actual=last):
        return gather_chunk(out, kv_actual, mesh_device, chunk_local)

    def write(cache, src, user, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(src, kv_actual),
            slot_idx=user,
            layer_idx=LAYER_IDX,
            num_layers=NUM_LAYERS,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )

    # Every chunk of every user goes into (user, LAYER_IDX); the other layer's slots stay zero.
    for u in range(NUM_USERS):
        for c in range(n_chunks):
            write(cache_k, k[u][0], u, c * chunk_global)
            write(cache_v, v[u][0], u, c * chunk_global)
    ttnn.synchronize_device(mesh_device)

    tt_q = make_q(last)
    prog, kcfg = sdpa_configs(mesh_device)
    common = dict(
        n_kv=NKV,
        cache_global=cache_global,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=HEAD_DIM**-0.5,
        cluster_axis=sp_axis,
        layer_idx=LAYER_IDX,
        num_layers=NUM_LAYERS,
        write_chunk=False,
    )

    def run_host(u, q_t=tt_q, kv_actual=last):
        out = dense_sp_attention(
            q_t,
            cache_k,
            cache_v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=kv_actual + chunk_global,
            slot_idx=u,
            **common,
        )
        return gather(out, kv_actual)

    def run_meta(slot_t, kv_t, q_t=tt_q, kv_actual=last):
        return dense_sp_attention(
            q_t,
            cache_k,
            cache_v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=kv_actual + chunk_global,
            slot_id=slot_t,
            kv_actual_isl_tensor=kv_t,
            **common,
        )

    host = [run_host(u) for u in range(NUM_USERS)]
    for u in range(NUM_USERS):
        passing, pcc = comp_pcc(refs[u][:, :, last:, :], host[u], PCC_BF8_CACHE)
        logger.info(f"host-int path user {u}: pcc={pcc}")
        assert passing, f"host-int cache-read PCC fail for user {u}: {pcc}"
    assert not torch.equal(host[0], host[1]), "users must produce distinct outputs for the slot check to mean anything"

    # The host path hashes neither layer factor, so successive layers must share one program while the folded batch
    # index is re-patched per dispatch. Layer 0 holds zeros here, so a stale layer index would reproduce layer 1.
    entries_host = mesh_device.num_program_cache_entries()
    layer0 = gather(
        dense_sp_attention(
            tt_q,
            cache_k,
            cache_v,
            None,
            None,
            kv_actual=last,
            logical_n=cache_global,
            slot_idx=0,
            **{**common, "layer_idx": 0},
        )
    )
    assert mesh_device.num_program_cache_entries() == entries_host, "host-path layer change compiled a new program"
    assert torch.equal(layer0, torch.zeros_like(layer0)), "host-path layer 0 read another layer's cache"
    assert torch.equal(run_host(0), host[0]), "host-path layer 1 after layer 0 did not re-patch the folded slot"

    # Metadata tensors live outside the capture and are the only thing that changes between replays.
    t_slot = meta_scalar(0, mesh_device)
    t_kv = meta_scalar(last, mesh_device)

    # Depth 0 first with its real logical_n creates the metadata program; depth 1 must then be a cache HIT that still
    # attends over the full two-chunk prefix, i.e. nothing on the host bounded the program by the depth-0 logical_n.
    tt_q0 = make_q(0)
    host0 = run_host(0, tt_q0, 0)
    passing, pcc = comp_pcc(refs[0][:, :, :chunk_global, :], host0, PCC_BF8_CACHE)
    assert passing, f"host-int cache-read PCC fail for user 0 at depth 0: {pcc}"
    entries_before = mesh_device.num_program_cache_entries()
    meta0_d0 = gather(run_meta(t_slot, meta_scalar(0, mesh_device), tt_q0, 0), 0)
    assert torch.equal(
        meta0_d0, host0
    ), f"metadata path != host-int path for user 0 at depth 0: max_abs={(meta0_d0 - host0).abs().max()}"
    entries_d0 = mesh_device.num_program_cache_entries()
    assert entries_d0 > entries_before, "depth-0 metadata call did not create a program (test setup)"

    # Eager metadata call at depth 1: bit-exact with the host-int path, and it warms the ring-gather buffers.
    meta0 = gather(run_meta(t_slot, t_kv))
    assert torch.equal(
        meta0, host[0]
    ), f"metadata path != host-int path for user 0: max_abs={(meta0 - host[0]).abs().max()}"
    assert (
        mesh_device.num_program_cache_entries() == entries_d0
    ), "depth 1 compiled a new program: logical_n is still part of the hash on the metadata path"

    # Freshly allocated metadata tensors on a cache hit: the kernels hold the tensors' addresses, so the framework
    # must re-point them. The first tensors stay alive so the new ones land elsewhere; a stale address reads slot 0.
    t_slot_fresh, t_kv_fresh = meta_scalar(1, mesh_device), meta_scalar(last, mesh_device)
    meta1 = gather(run_meta(t_slot_fresh, t_kv_fresh))
    assert torch.equal(meta1, host[1]), (
        f"cached program did not follow fresh metadata tensors: max_abs vs user 1={(meta1 - host[1]).abs().max()}, "
        f"pcc_vs_user0={comp_pcc(host[0], meta1, 0.0)[1]}"
    )

    # write_chunk on this path must follow the tensors, not the host slot_idx / kv_actual (both left at 0 on
    # purpose): blank user 1's last chunk, rewrite it with the tensors at user 1 / depth 1. A host-slot write would
    # land in user 0 and a host-offset write in depth 0; either way the read of user 1 would see zeros.
    zeros = torch.zeros(NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16)
    write(cache_k, zeros, 1, last)
    write(cache_v, zeros, 1, last)
    meta1_written = gather(
        dense_sp_attention(
            tt_q,
            cache_k,
            cache_v,
            make_chunk(k[1][0], last),
            make_chunk(v[1][0], last),
            kv_actual=0,
            logical_n=cache_global,
            slot_id=t_slot_fresh,
            kv_actual_isl_tensor=t_kv_fresh,
            **{**common, "write_chunk": True},
        )
    )
    assert torch.equal(meta1_written, host[1]), (
        f"write_chunk on the metadata path did not write user 1's slot: max_abs vs user 1="
        f"{(meta1_written - host[1]).abs().max()}"
    )
    meta0_after = gather(run_meta(t_slot, t_kv))
    assert torch.equal(meta0_after, host[0]), "write_chunk on the metadata path clobbered user 0's slot"

    # The accessor is baked for a single-page tensor and the hash never sees the tensor, so a two-element scalar with
    # the same memory config is a genuine cache hit that must still be refused.
    two = ttnn.from_torch(
        torch.tensor([1, 1], dtype=torch.int64).reshape(1, 1, 1, 2),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error(RuntimeError, "slot_id must contain exactly one element"):
        run_meta(two, t_kv)

    # Direct op call with dense_sp's kwargs plus the metadata tensors; `extra` adds or overrides kwargs.
    def direct_call(**extra):
        kwargs = dict(
            persistent_output_buffer_k=ccl.get_ring_gather_buffer(
                "dense_k", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            persistent_output_buffer_v=ccl.get_ring_gather_buffer(
                "dense_v", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            joint_strategy="rear",
            logical_n=cache_global,
            program_config=prog,
            compute_kernel_config=kcfg,
            dim=2,
            multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=HEAD_DIM**-0.5,
            slot_id=t_slot,
            kv_actual_isl_tensor=t_kv,
            kv_cache_num_layers=NUM_LAYERS,
            kv_cache_layer_idx=LAYER_IDX,
        )
        kwargs.update(extra)
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q, cache_k, cache_v, None, None, None, **kwargs
        )

    # A host scalar next to the tensors would still steer the all-gather extent, so the mix is refused.
    with expect_error(RuntimeError, "metadata tensors replace the host"):
        direct_call(kv_actual_isl=last)
    # slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx is a DRAM offset used unchecked; bound the host factors.
    with expect_error(RuntimeError, "kv_cache_layer_idx=.* must be less than kv_cache_num_layers"):
        direct_call(kv_cache_layer_idx=NUM_LAYERS)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tr = run_meta(t_slot, t_kv)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    def replay_expecting(expected, kv_actual, what):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = gather(out_tr, kv_actual)
        assert torch.equal(got, expected), (
            f"traced replay != host-int path for {what}: max_abs={(got - expected).abs().max()}, "
            f"pcc_vs_user0={comp_pcc(host[0], got, 0.0)[1]}"
        )

    try:
        replay_expecting(host[0], last, "user 0")
        ttnn.copy_host_to_device_tensor(host_scalar(1), t_slot)  # re-target the slot outside the trace
        replay_expecting(host[1], last, "user 1")
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_slot)  # back, to rule out a one-way latch
        replay_expecting(host[0], last, "user 0 again")
        # Depth re-target: Q slab and kv_actual scalar are read by address, so refreshed in place the same captured
        # program attends at depth 0 -- length, Q mapping, ring masks and the gather extent all re-derived on device.
        ttnn.copy_host_to_device_tensor(make_q(0, on_device=False), tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_kv)
        replay_expecting(host0, 0, "user 0 at depth 0")
        ttnn.copy_host_to_device_tensor(make_q(last, on_device=False), tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(last), t_kv)
        replay_expecting(host[0], last, "user 0 back at depth 1")
    finally:
        ttnn.release_trace(mesh_device, tid)
    logger.info("ring_joint metadata path: bit-exact vs host-int for both users; trace re-targets slot and depth")
