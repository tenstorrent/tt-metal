# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ring_joint cache-read on (8,4), SP=8 x TP=4, through dense_sp_attention, vs a full-causal GQA torch golden.
Two users x two layers with DISTINCT K/V in the GQA chunked-KV cache (block-cyclic, SP-sharded), so a read from
the wrong slot or layer changes the output instead of reproducing it. Grouped V (cache stays NKV heads, 1/chip).
The chunk / gather helpers and the metadata scalars live in ring_joint_cache_read_helpers.

test_ring_joint_cache_read_sp -- n_chunks-1 chunks pre-written; dense_sp_attention writes the last chunk and
reads the accumulated prefix for each user, on either slot form: host kv_cache_batch_idx / kv_actual_isl, or the
trace-safe metadata tensors slot_id / kv_actual_isl_tensor.

test_ring_joint_cache_read_metadata_retarget -- the multi-call contract of the metadata path, bit-exact against the
host path: two depths served by one cached program (logical_n is off the hash), a cached program following freshly
allocated tensors, the write landing in the tensor-selected slot and offset, the refused hazards (wrong tensor form
on a cache hit, host scalars next to the tensors, a layer index past the (user, layer) fold), host-path layers
sharing one program with the folded batch index re-patched per dispatch, and one captured trace re-targeting the
user and the depth by rewriting the scalars in place between replays.
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


class _Setup:
    """Per-user goldens, the two-user x two-layer caches with the first `prewrite_chunks` chunks of every user written
    into (user, LAYER_IDX), the last chunk's Q slab, and the dense_sp kwargs both tests share."""

    def __init__(self, mesh_device, n_chunks, chunk_local, *, prewrite_chunks):
        rows, cols = tuple(mesh_device.shape)
        assert (rows, cols) == (8, 4)
        self.mesh_device, self.chunk_local = mesh_device, chunk_local
        self.chunk_global = rows * chunk_local
        self.cache_global = n_chunks * self.chunk_global
        self.last = (n_chunks - 1) * self.chunk_global  # kv_actual before the last chunk

        torch.manual_seed(0)
        self.q = torch.randn(1, NQ, self.cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        self.k = [
            torch.randn(1, NKV, self.cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)
        ]
        self.v = [
            torch.randn(1, NKV, self.cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)
        ]
        self.refs = [torch_gqa_causal(self.q.float(), self.k[u].float(), self.v[u].float()) for u in range(NUM_USERS)]

        self.ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
        shape = list(mesh_device.shape)
        self.cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, self.cache_global, shape, SP_AXIS, NUM_LAYERS, NUM_USERS)
        self.cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, self.cache_global, shape, SP_AXIS, NUM_LAYERS, NUM_USERS)
        # The other layer's slots stay zero, so a read of layer 0 tells a stale layer index from a correct one.
        for u in range(NUM_USERS):
            for c in range(prewrite_chunks):
                self.write(self.cache_k, self.k[u][0], u, c * self.chunk_global)
                self.write(self.cache_v, self.v[u][0], u, c * self.chunk_global)
        ttnn.synchronize_device(mesh_device)

        self.tt_q = self.make_q(self.last)
        self.prog, self.kcfg = sdpa_configs(mesh_device)
        self.common = dict(
            n_kv=NKV,
            cache_global=self.cache_global,
            head_dim=HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=self.ccl,
            program_config=self.prog,
            compute_kernel_config=self.kcfg,
            scale=HEAD_DIM**-0.5,
            cluster_axis=SP_AXIS,
            layer_idx=LAYER_IDX,
            num_layers=NUM_LAYERS,
        )

    def make_chunk(self, src, kv_actual):
        return make_kv_chunk(src, kv_actual, self.mesh_device, self.chunk_local)

    def make_q(self, kv_actual, on_device=True):
        return make_q_chunk(self.q, kv_actual, self.mesh_device, self.chunk_local, on_device)

    def gather(self, out, kv_actual=None):
        return gather_chunk(out, self.last if kv_actual is None else kv_actual, self.mesh_device, self.chunk_local)

    def write(self, cache, src, user, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            self.make_chunk(src, kv_actual),
            slot_idx=user,
            layer_idx=LAYER_IDX,
            num_layers=NUM_LAYERS,
            kv_actual_global=kv_actual,
            cluster_axis=SP_AXIS,
        )


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "n_chunks,chunk_local",
    [(2, 32), (2, 640)],  # 2x256 (quick) and 2x5120 -- the REAL M3 prefill chunk (640/chip at SP=8)
    ids=["2x256", "2x5120"],
)
@pytest.mark.parametrize("slot_form", ["host", "tensor"])
def test_ring_joint_cache_read_sp(mesh_device, device_params, slot_form, n_chunks, chunk_local, reset_seeds):
    """Chunked mode needs Q.seq < cached K.seq: the prior chunks are pre-written, dense_sp_attention writes the LAST
    chunk into (user, LAYER_IDX) and runs ONLY that chunk's queries over the whole prefix (kv_actual = prefix before
    it, logical_n = full), per user, vs the golden."""
    s = _Setup(mesh_device, n_chunks, chunk_local, prewrite_chunks=n_chunks - 1)
    outs = []
    for u in range(NUM_USERS):
        if slot_form == "host":
            slot = dict(slot_idx=u)
        else:
            slot = dict(slot_id=meta_scalar(u, mesh_device), kv_actual_isl_tensor=meta_scalar(s.last, mesh_device))
        out = dense_sp_attention(
            s.tt_q,
            s.cache_k,
            s.cache_v,
            s.make_chunk(s.k[u][0], s.last),
            s.make_chunk(s.v[u][0], s.last),
            kv_actual=s.last,
            logical_n=s.cache_global,
            **slot,
            **s.common,
        )
        outs.append(s.gather(out))
        passing, pcc = comp_pcc(s.refs[u][:, :, s.last :, :], outs[-1], PCC_BF8_CACHE)
        logger.info(f"ring_joint cache-read SP=8 x TP=4, {slot_form} slot, user {u}: pcc={pcc}")
        assert passing, f"cache-read PCC fail ({slot_form} slot, user {u}): {pcc}"
    assert not torch.equal(outs[0], outs[1]), "users must produce distinct outputs for the slot check to mean anything"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2x256"])
def test_ring_joint_cache_read_metadata_retarget(
    mesh_device, device_params, expect_error, n_chunks, chunk_local, reset_seeds
):
    s = _Setup(mesh_device, n_chunks, chunk_local, prewrite_chunks=n_chunks)
    last, chunk_global, cache_global = s.last, s.chunk_global, s.cache_global
    common = dict(s.common, write_chunk=False)

    def run_host(u, q_t=None, kv_actual=last):
        out = dense_sp_attention(
            s.tt_q if q_t is None else q_t,
            s.cache_k,
            s.cache_v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=kv_actual + chunk_global,
            slot_idx=u,
            **common,
        )
        return s.gather(out, kv_actual)

    def run_meta(slot_t, kv_t, q_t=None, kv_actual=last):
        return dense_sp_attention(
            s.tt_q if q_t is None else q_t,
            s.cache_k,
            s.cache_v,
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
        passing, pcc = comp_pcc(s.refs[u][:, :, last:, :], host[u], PCC_BF8_CACHE)
        logger.info(f"host-int path user {u}: pcc={pcc}")
        assert passing, f"host-int cache-read PCC fail for user {u}: {pcc}"
    assert not torch.equal(host[0], host[1]), "users must produce distinct outputs for the slot check to mean anything"

    # The host path hashes neither layer factor, so successive layers must share one program while the folded batch
    # index is re-patched per dispatch. Layer 0 holds zeros here, so a stale layer index would reproduce layer 1.
    entries_host = mesh_device.num_program_cache_entries()
    layer0 = s.gather(
        dense_sp_attention(
            s.tt_q,
            s.cache_k,
            s.cache_v,
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
    tt_q0 = s.make_q(0)
    host0 = run_host(0, tt_q0, 0)
    passing, pcc = comp_pcc(s.refs[0][:, :, :chunk_global, :], host0, PCC_BF8_CACHE)
    assert passing, f"host-int cache-read PCC fail for user 0 at depth 0: {pcc}"
    entries_before = mesh_device.num_program_cache_entries()
    meta0_d0 = s.gather(run_meta(t_slot, meta_scalar(0, mesh_device), tt_q0, 0), 0)
    assert torch.equal(
        meta0_d0, host0
    ), f"metadata path != host-int path for user 0 at depth 0: max_abs={(meta0_d0 - host0).abs().max()}"
    entries_d0 = mesh_device.num_program_cache_entries()
    assert entries_d0 > entries_before, "depth-0 metadata call did not create a program (test setup)"

    # Eager metadata call at depth 1: bit-exact with the host-int path, and it warms the ring-gather buffers.
    meta0 = s.gather(run_meta(t_slot, t_kv))
    assert torch.equal(
        meta0, host[0]
    ), f"metadata path != host-int path for user 0: max_abs={(meta0 - host[0]).abs().max()}"
    assert (
        mesh_device.num_program_cache_entries() == entries_d0
    ), "depth 1 compiled a new program: logical_n is still part of the hash on the metadata path"

    # Freshly allocated metadata tensors on a cache hit: the kernels hold the tensors' addresses, so the framework
    # must re-point them. The first tensors stay alive so the new ones land elsewhere; a stale address reads slot 0.
    t_slot_fresh, t_kv_fresh = meta_scalar(1, mesh_device), meta_scalar(last, mesh_device)
    meta1 = s.gather(run_meta(t_slot_fresh, t_kv_fresh))
    assert torch.equal(meta1, host[1]), (
        f"cached program did not follow fresh metadata tensors: max_abs vs user 1={(meta1 - host[1]).abs().max()}, "
        f"pcc_vs_user0={comp_pcc(host[0], meta1, 0.0)[1]}"
    )

    # write_chunk on this path must follow the tensors, not the host slot_idx / kv_actual (both left at 0 on
    # purpose): blank user 1's last chunk, rewrite it with the tensors at user 1 / depth 1. A host-slot write would
    # land in user 0 and a host-offset write in depth 0; either way the read of user 1 would see zeros.
    zeros = torch.zeros(NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16)
    s.write(s.cache_k, zeros, 1, last)
    s.write(s.cache_v, zeros, 1, last)
    meta1_written = s.gather(
        dense_sp_attention(
            s.tt_q,
            s.cache_k,
            s.cache_v,
            s.make_chunk(s.k[1][0], last),
            s.make_chunk(s.v[1][0], last),
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
    meta0_after = s.gather(run_meta(t_slot, t_kv))
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
            persistent_output_buffer_k=s.ccl.get_ring_gather_buffer(
                "dense_k", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            persistent_output_buffer_v=s.ccl.get_ring_gather_buffer(
                "dense_v", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            joint_strategy="rear",
            logical_n=cache_global,
            program_config=s.prog,
            compute_kernel_config=s.kcfg,
            dim=2,
            multi_device_global_semaphore=s.ccl.ring_attention_ccl_semaphore_handles,
            num_links=s.ccl.num_links,
            cluster_axis=SP_AXIS,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=s.ccl.ring_attention_ccl_core_grid_offset,
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
            s.tt_q, s.cache_k, s.cache_v, None, None, None, **kwargs
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
        got = s.gather(out_tr, kv_actual)
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
        ttnn.copy_host_to_device_tensor(s.make_q(0, on_device=False), s.tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_kv)
        replay_expecting(host0, 0, "user 0 at depth 0")
        ttnn.copy_host_to_device_tensor(s.make_q(last, on_device=False), s.tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(last), t_kv)
        replay_expecting(host[0], last, "user 0 back at depth 1")
    finally:
        ttnn.release_trace(mesh_device, tid)
    logger.info("ring_joint metadata path: bit-exact vs host-int for both users; trace re-targets slot and depth")
