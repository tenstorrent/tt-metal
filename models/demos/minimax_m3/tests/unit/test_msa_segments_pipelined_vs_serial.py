# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The packed forward's pipelined MSA schedule against the serial chain, on the real multi-user cache.

``msa_segments_pipelined`` gathers segment i+1's cache slot on the gather sub-device while segment i's block
selection and sparse attention run on the compute sub-device, alternating two gather buffers. Every segment
must get exactly what the serial ``msa_sp_attention_cache_read`` gives it: the right (user, layer) slot, a
gather that has landed, a buffer no later gather has overwritten, and ops confined to a sub-grid that
reproduce the full-grid arithmetic. The outputs are compared bit for bit, over two windows (the second on
warm programs) and after a window that aborted mid-schedule.
"""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches, write_index_k_chunk, write_kv_chunk
from models.demos.minimax_m3.tt.attention.msa import MsaSegment, msa_segments_pipelined, msa_sp_attention_cache_read
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128
CHUNK_LOCAL = 640  # per-chip rows of one segment (5120 / SP=8)
NUM_LAYERS = 2  # the tested layer is the second: a slot mix-up lands on another layer's (zeroed) slot
LAYER_IDX = 1
# (user, chunks already in the cache) per segment, in forward order: three users at two depths, user 0 twice
# (its second segment attends the first one too), so both gather buffers and a repeated slot are exercised.
SEGMENTS = [(0, 1), (1, 2), (2, 1), (0, 2)]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16cache", "bf8cache"])
def test_msa_segments_pipelined_matches_serial(mesh_device, device_params, cache_dtype, reset_seeds, expect_error):
    rows, cols = mesh_device.shape
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis, tp_axis = rows, cols, 0, 1
    chunk = sp * CHUNK_LOCAL
    num_users = 1 + max(u for u, _ in SEGMENTS)
    chunks_written = {u: max(n for uu, n in SEGMENTS if uu == u) + 1 for u in range(num_users)}
    capacity = max(chunks_written.values()) * chunk
    scale = HEAD_DIM**-0.5

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    def shard(t, split_heads):
        dims = [None, None]
        dims[sp_axis] = 2
        dims[tp_axis] = 1 if split_heads else None
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    def collect(out_t):
        dts = ttnn.get_device_tensors(out_t)
        groups = [
            torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for r in range(rows)], dim=2) for c in range(cols)
        ]
        return torch.cat(groups, dim=1)  # [1, NQ, chunk, HD]

    torch.manual_seed(0)
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=capacity,
        sp_axis=sp_axis,
        num_users=num_users,
        head_dim=HEAD_DIM,
        cache_dtype=cache_dtype,
    )
    segments = []
    try:
        # Every user's chunks, the tested layer only; the other layer's slots stay zeroed.
        for u, n in chunks_written.items():
            for c in range(n):
                k = torch.randn(1, NKV, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
                v = torch.randn(1, NKV, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
                ik = torch.randn(1, 1, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
                write_kv_chunk(
                    kv,
                    shard(k, True),
                    shard(v, True),
                    slot_idx=u,
                    layer_idx=LAYER_IDX,
                    kv_actual=c * chunk,
                    sp_axis=sp_axis,
                )
                write_index_k_chunk(
                    kv, shard(ik, False), slot_idx=u, layer_idx=LAYER_IDX, kv_actual=c * chunk, sp_axis=sp_axis
                )
        for u, n_cached in SEGMENTS:
            q = torch.randn(1, NQ, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
            iq = torch.randn(1, NIDX, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
            segments.append(
                MsaSegment(
                    q=shard(q, True), index_q=shard(iq, True), user=u, cached_len=n_cached * chunk, s_local=CHUNK_LOCAL
                )
            )
        common = dict(
            mesh_config=mesh_config, ccl_manager=ccl, scale=scale, block_size=128, topk_blocks=16, num_groups=1
        )

        serial = [
            collect(
                msa_sp_attention_cache_read(
                    s.q,
                    s.index_q,
                    kv,
                    slot=kv.slot(s.user, LAYER_IDX),
                    cached_len=s.cached_len,
                    chunk_local=s.s_local,
                    **common,
                )
            )
            for s in segments
        ]

        def pipelined():
            return [collect(o) for o in msa_segments_pipelined(segments, kv_cache=kv, layer_idx=LAYER_IDX, **common)]

        def check(outs, label):
            for i, (u, n_cached) in enumerate(SEGMENTS):
                assert torch.equal(
                    serial[i], outs[i]
                ), f"{label}: segment {i} (user {u}, {n_cached} cached chunks) differs"

        try:
            check(pipelined(), "first window")
            # A window that raises after its first gather and attention are in flight (segment 1's rows are not a
            # whole number of blocks) must leave the manager and the gather semaphores clean for the next one.
            broken = list(segments)
            broken[1] = replace(segments[1], s_local=CHUNK_LOCAL + 32)
            with expect_error(AssertionError, "whole number of"):
                msa_segments_pipelined(broken, kv_cache=kv, layer_idx=LAYER_IDX, **common)
            entries = mesh_device.num_program_cache_entries()
            check(pipelined(), "window after an aborted one")
            assert mesh_device.num_program_cache_entries() == entries, "the second window must hit its programs"
        finally:
            ccl.release_msa_overlap()
    finally:
        for s in segments:
            ttnn.deallocate(s.q)
            ttnn.deallocate(s.index_q)
        kv.deallocate()
