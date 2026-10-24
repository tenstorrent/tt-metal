# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa on a restricted core grid, and the sparse KV push it can fuse (GLM shared layers).

GLM-5.3 prefill shape on the 8x4 Galaxy (SP8 x TP4, 64 heads resharded head->seq, TOPK 2048, latent 512 +
rope 64). Timings come from back-to-back trace replays, measured on the host.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import golden, make_inputs, pcc

K_DIM, V_DIM, TOPK, H = 576, 512, 2048, 64
SP, TP = 8, 4

_TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 2048,
    "trace_region_size": 64 * 1024 * 1024,
}


def _cols(first, last, rows=10):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(first, 0), ttnn.CoreCoord(last, rows - 1))])


# KVPE cache formats: production is scaled FP8 (512 fp8 latents | 4 fp32 block scales | 64 bf16 rope = 656 B rows).
_KV_FORMATS = {
    "fp8": (ttnn.transformer.SparseKVFormat.SCALED_FP8, ttnn.fp8_e4m3, 656),
    "bf16": (ttnn.transformer.SparseKVFormat.BF16, ttnn.bfloat16, K_DIM),
}
_KV_FMT = pytest.mark.parametrize("kv_fmt", list(_KV_FORMATS))


def _kv_cache(mesh_device, batch, tokens, kv_fmt, mesh_mapper, seed=0):
    """A KVPE cache in kv_fmt. Scaled FP8 is packed on device by the same ops the model's cache writer uses."""
    gen = torch.Generator().manual_seed(seed)
    common = dict(
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mesh_mapper
    )
    if kv_fmt == "bf16":
        host = torch.randn(batch, 1, tokens, K_DIM, generator=gen).to(torch.bfloat16)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, **common)
    latent = torch.randn(batch, 1, tokens, V_DIM, generator=gen).to(torch.bfloat16)
    rope = torch.randn(batch, 1, tokens, K_DIM - V_DIM, generator=gen).to(torch.bfloat16)
    fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        ttnn.from_torch(latent, dtype=ttnn.bfloat16, **common), round_scale_to_power_of_two=True
    )
    return ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(
        fp8, scales, ttnn.from_torch(rope, dtype=ttnn.bfloat16, **common)
    )


def _zero_kv_buffer(mesh_device, tokens, kv_fmt):
    _, dtype, width = _KV_FORMATS[kv_fmt]
    zeros = ttnn.from_torch(
        torch.zeros(1, 1, tokens, width, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return zeros if dtype == ttnn.bfloat16 else ttnn.typecast(zeros, dtype)


def _sparse_sdpa(q, kv, idx, chunk_local, kv_fmt, k_chunk=128, **grid):
    return ttnn.transformer.sparse_sdpa(
        q,
        kv,
        idx,
        V_DIM,
        kv_format=_KV_FORMATS[kv_fmt][0],
        scale=K_DIM**-0.5,
        k_chunk_size=k_chunk,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=chunk_local,
        block_cyclic_cache_tp_sharded=True,
        **grid,
    )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("cols", [(0, 7), (2, 9), (0, 0)], ids=["80core", "offset80", "10core"])
def test_sparse_sdpa_sub_core_grids_matches_golden(mesh_device, cols):
    """A restricted grid only changes which core owns which tokens; results must not change."""
    Hs, S, T, topk, kc = 32, 96, 1024, 128, 64
    q, kv, indices = make_inputs(Hs, S, T, topk, K_DIM, lambda s: 1 + (s * 7) % topk, seed=3)
    ref = golden(q, kv, indices, K_DIM**-0.5, V_DIM)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, **common)
    tt_kv = ttnn.from_torch(kv.to(torch.bfloat16), dtype=ttnn.bfloat16, **common)
    tt_idx = ttnn.from_torch(indices.to(torch.int32), dtype=ttnn.uint32, **common)
    run = lambda **grid: ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=ttnn.transformer.SparseKVFormat.BF16, k_chunk_size=kc, **grid
        )
    )
    full = run()
    restricted = run(sub_core_grids=_cols(*cols))
    assert torch.equal(full, restricted), "restricted grid changed the output"
    assert pcc(restricted, ref) >= 0.99


def _glm_inputs(mesh_device, chunk_global, cache_tokens, kv_fmt):
    """Per-chip GLM shapes after the head->seq reshard: q [1,64,chunk/(sp*tp),576], natural-position
    indices into a populated prefix, and a replicated block-cyclic gather buffer of cache_tokens rows."""
    s_local = chunk_global // (SP * TP)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, H, s_local * SP * TP, K_DIM, generator=gen).to(torch.bfloat16)
    idx = torch.randint(0, cache_tokens, (1, 1, s_local * SP * TP, TOPK), generator=gen, dtype=torch.int32)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # 32-way, matches the post-reshard sharding
    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, mesh_mapper=shard_seq, **common)
    tt_idx = ttnn.from_torch(idx, dtype=ttnn.uint32, mesh_mapper=shard_seq, **common)
    tt_kv = _kv_cache(mesh_device, 1, cache_tokens, kv_fmt, ttnn.ReplicateTensorToMesh(mesh_device), seed=1)
    return tt_q, tt_kv, tt_idx, s_local * TP  # chunk_local = per-SP-row chunk


def _traced_us(mesh_device, fn, n=10, reps=10):
    """Mean time of one fn call over n back-to-back replays of a trace of `reps` calls (the per-replay host cost
    amortised, as inside the model's layer trace)."""
    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    # Held until the trace is released: replays write the outputs captured here, which must stay allocated.
    outputs = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / (n * reps) * 1e6
    ttnn.release_trace(mesh_device, tid)
    del outputs
    return us


_PERF = pytest.mark.skipif(os.environ.get("CI") == "true", reason="local production-shape perf qualification")
_CHUNKS = pytest.mark.parametrize("chunk_global", [5120, 2048], ids=["5k", "2k"])


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@_CHUNKS
@_KV_FMT
def test_sparse_sdpa_glm_core_count_perf(mesh_device, chunk_global, kv_fmt):
    """sparse_sdpa alone on all cores and on the overlap profiles' attention grids, at the GLM shape (traced)."""
    grid = mesh_device.compute_with_storage_grid_size()
    tt_q, tt_kv, tt_idx, chunk_local = _glm_inputs(mesh_device, chunk_global, chunk_global * 51, kv_fmt)
    rows_2_on = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    rows = []
    for name, sub in [("all", None), ("rows 2+", rows_2_on), ("cols 0-7", _cols(0, 7, grid.y))]:
        kw = {} if sub is None else {"sub_core_grids": sub}
        rows.append(
            (name, _traced_us(mesh_device, lambda: _sparse_sdpa(tt_q, tt_kv, tt_idx, chunk_local, kv_fmt, **kw)))
        )
    logger.info(f"sparse_sdpa GLM {chunk_global} {kv_fmt} traced us per call (grid, us): {rows}")


# ---------------------------------------------------------------------------------------------------------
# Sparse KV push with real GLM-5.3 top-k at a 1M prefix: the next layer's buffer, filled by the prefix push fused
# into sparse_sdpa plus the remainder push (or by one push of every row), read through the plan's rewritten indices,
# must give exactly the dense gather's attention.
# ---------------------------------------------------------------------------------------------------------

_TORUS_XY_GLM = {**_TORUS_XY}
_GLM_ROUTER = ttnn.FabricRouterConfig()
_GLM_ROUTER.max_packet_payload_size_bytes = 6144  # GLM53Config.FABRIC_PAYLOAD_SIZE
_TORUS_XY_GLM["fabric_router_config"] = _GLM_ROUTER
_TRACE_1M = Path(
    os.environ.get(
        "GLM53_1M_TRACE", "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120"
    )
)
_TRACE_1M_START = 1039360  # prompt position of the trace's first row (a whole number of 5120 slabs)
_LINKS = 2  # Blackhole trains 2 fabric routing planes (ttMLA.ccl_num_links)


class _PushRig:
    """One GLM-5.3 shared-layer push at a 1M prefix with layer `layer`'s real top-k: a 2-slot KVPE cache sharded
    over the mesh (slots: layer L, layer L + 1), the plan of the real indices, and the compact buffers of both
    parities. sparse_sdpa runs on `sdpa_grid` (None: every core); a fused push's senders and muxes on `push_grid`."""

    def __init__(self, mesh_device, layer, kv_fmt, front_bytes, sdpa_grid, push_grid, hot=True):
        from safetensors.numpy import load_file

        from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPlanner, _shard_over_mesh

        f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
        if not f.exists():
            pytest.skip(f"1M GLM-5.3 top-k not staged: {f}")
        grid = mesh_device.compute_with_storage_grid_size()
        if (grid.x, grid.y) != (12, 10):
            pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
        self.mesh_device, self.kv_fmt = mesh_device, kv_fmt
        self.sdpa_grid, self.push_grid = sdpa_grid, push_grid
        chunk_global = 5120
        n_dev = mesh_device.get_num_devices()
        self.stripe_chunk = self.s_local = chunk_global // n_dev
        self.cache_tokens = _TRACE_1M_START + chunk_global
        width = _KV_FORMATS[kv_fmt][2] * (1 if kv_fmt == "fp8" else 2)
        self.page_bytes = -(-width // 64) * 64  # DRAM-aligned row pages: 704 B scaled FP8, 1152 B BF16
        self.planner = SparseKvPushPlanner(mesh_device, _LINKS, l1_front_bytes_per_core=front_bytes, hot=hot)
        self.pages_per_packet = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // self.page_bytes)
        self.buffer_rows = self.planner.buffer_rows(self.cache_tokens, self.page_bytes)

        common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self.cache = _kv_cache(mesh_device, 2, self.cache_tokens, kv_fmt, shard_seq, seed=2)
        self.q = ttnn.from_torch(
            torch.randn(1, H, n_dev * self.s_local, K_DIM).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            mesh_mapper=shard_seq,
            **common,
        )
        self.start = ttnn.from_torch(
            torch.tensor([[[[_TRACE_1M_START]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
        )
        self.sems = [
            ttnn.create_global_semaphore(mesh_device, _cols(0, grid.x - 1), 0, ttnn.BufferType.L1_SMALL)
            for _ in range(2)
        ]
        natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, self.s_local, TOPK)
        self.idx_natural = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n_dev * self.s_local, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )
        self._shard_over_mesh = _shard_over_mesh
        table, self.compact_idx = self.plan()
        self.tables = ttnn.from_torch(
            torch.zeros(1, 1, table.shape[2] * n_dev, table.shape[3], dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
            **common,
        )
        self.table = table
        self.gather_table()
        self.hot_table = self.hot(self.compact_idx) if hot else None
        self.compact = [self.compact_buffer(0), self.compact_buffer(1)]
        self.sdpa_kw = dict(kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128)

    def plan(self):
        """(per-chip plan table, compact indices) of the real top-k."""
        table, compact_idx = ttnn.experimental.sparse_push_plan(
            self.idx_natural,
            num_devices=self.mesh_device.get_num_devices(),
            stripe_chunk=self.stripe_chunk,
            max_rows=self.cache_tokens,
            pages_per_packet=self.pages_per_packet,
            banks=self.planner.banks,
            chunk_start=self.start,
        )
        self._shard_over_mesh(table, self.mesh_device)
        return table, compact_idx

    def hot(self, compact_idx, table=None):
        """The hot table of the gathered tables; rewrites compact_idx (of the plan of `table`, default the one built
        at setup) in place."""
        from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import HOT_PREFIX_CAP, HOT_THRESHOLD

        return ttnn.experimental.sparse_push_hot(
            self.idx_natural,
            compact_idx,
            self.table if table is None else table,
            self.tables,
            num_devices=self.mesh_device.get_num_devices(),
            stripe_chunk=self.stripe_chunk,
            max_rows=self.cache_tokens,
            pages_per_packet=self.pages_per_packet,
            banks=self.planner.banks,
            threshold=HOT_THRESHOLD,
            prefix_cap=HOT_PREFIX_CAP,
            chunk_start=self.start,
        )

    def gather_table(self, table=None):
        """All-gather a plan's per-chip tables (default: the one built at setup) into the push table."""
        table = self.table if table is None else table
        return ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self.tables, cluster_axis=None, num_links=_LINKS
        )

    def compact_buffer(self, parity):
        """(kv, spill): an L1 front and its DRAM spill, or one DRAM buffer and None."""
        front = self.planner.front_buffer(parity, self.cache, self.page_bytes)
        if front is None:
            return _zero_kv_buffer(self.mesh_device, self.buffer_rows, self.kv_fmt), None
        return front, self.planner.spill_buffer(parity, self.buffer_rows, self.cache, self.page_bytes)

    def push(self, parity, slot, part):
        """Push slot `slot`'s rows into the compact buffer of `parity`: all rows (part 0) or the remainder (part 2)."""
        buf = self.compact[parity]
        return ttnn.experimental.high_bw_all_gather(
            self.cache,
            dim=2,
            output_tensor=buf[0],
            push_spill=buf[1],
            cluster_axis=None,
            num_links=_LINKS,
            input_batch_index=slot,
            push_part=part,
            ready_semaphore=self.sems[0],
            data_valid_semaphore=self.sems[1],
            push_table=self.tables,
            push_stripe_chunk=self.stripe_chunk,
            push_max_rows=self.cache_tokens,
            push_hot_table=self.hot_table,
        )

    def fused_push(self, parity, slot):
        """sparse_sdpa arguments that push slot `slot`'s prefix rows into the compact buffer of `parity`."""
        buf = self.compact[parity]
        return dict(
            push_input=self.cache,
            push_table=self.tables,
            push_output=buf[0],
            push_output_spill=buf[1],
            push_stripe_chunk=self.stripe_chunk,
            push_max_rows=self.cache_tokens,
            push_batch_index=slot,
            push_num_links=_LINKS,
            push_ready_semaphore=self.sems[0],
            push_data_valid_semaphore=self.sems[1],
            push_core_grid=self.push_grid,
            push_hot_table=self.hot_table,
        )

    def sdpa_compact(self, parity, **fused):
        """sparse_sdpa over the compact buffer of `parity` (a shared layer); `fused` from fused_push."""
        buf = self.compact[parity]
        grid = {} if self.sdpa_grid is None else {"sub_core_grids": self.sdpa_grid}
        return ttnn.transformer.sparse_sdpa(
            self.q, buf[0], self.compact_idx, V_DIM, **self.sdpa_kw, kv_spill=buf[1], **grid, **fused
        )

    def dense_buffer(self):
        return _zero_kv_buffer(self.mesh_device, self.cache_tokens, self.kv_fmt)

    def dense_gather(self, slot, out):
        """Every prefix row of slot `slot` into every chip: the all-gather the push replaces."""
        return ttnn.experimental.high_bw_all_gather(
            self.cache, dim=2, output_tensor=out, cluster_axis=None, num_links=_LINKS, input_batch_index=slot
        )

    def sdpa_dense(self, kv, **fused):
        """sparse_sdpa over a dense gather (a full layer: natural indices, block-cyclic rows)."""
        grid = {} if self.sdpa_grid is None else {"sub_core_grids": self.sdpa_grid}
        return _sparse_sdpa(self.q, kv, self.idx_natural, self.s_local * TP, self.kv_fmt, **grid, **fused)


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("layer", [14, 62], ids=["L14", "L62"])
# Compact buffer placement: all DRAM; the model's L1 front (most rows in L1); a tiny front (most rows in the spill).
@pytest.mark.parametrize("front_bytes", [0, None, 16 * 704], ids=["dram", "l1_front", "l1_front_spill"])
# Hot rows multicast into the hot region (sparse_push_hot), or every row pushed per requester.
@pytest.mark.parametrize("hot", [True, False], ids=["hot", "no_hot"])
def test_sparse_kv_push_real_topk_matches_gather(mesh_device, layer, front_bytes, hot):
    rig = _PushRig(mesh_device, layer, "fp8", front_bytes, sdpa_grid=_cols(0, 7), push_grid=_cols(8, 11), hot=hot)
    # Layer L's buffer by one push of every row; layer L + 1's by its prefix fused into layer L's attention, then its
    # remainder.
    rig.push(0, 0, 0)
    rig.sdpa_compact(0, **rig.fused_push(1, 1))
    rig.push(1, 1, 2)

    def dense(slot):
        out = rig.dense_buffer()
        rig.dense_gather(slot, out)
        return rig.sdpa_dense(out)

    host = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    for slot in (0, 1):
        got = host(rig.sdpa_compact(slot))
        assert torch.equal(
            host(dense(slot)), got
        ), f"layer {layer} slot {slot}: push path differs from the dense gather"


# ---------------------------------------------------------------------------------------------------------
# What the unfused sparse KV push costs at a 1M prefix (traced, real top-k): the plan, the all-gather of its tables,
# and one push of every row, as a shared layer runs them when no earlier layer prefetched its prefix.
# ---------------------------------------------------------------------------------------------------------


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("layer", [14, 62], ids=["L14", "L62"])
@pytest.mark.parametrize("front_bytes", [None, 0], ids=["l1_front", "dram"])
@_KV_FMT
@pytest.mark.parametrize("hot", [True, False], ids=["hot", "no_hot"])
def test_sparse_kv_push_unfused_perf(mesh_device, layer, front_bytes, kv_fmt, hot):
    rig = _PushRig(mesh_device, layer, kv_fmt, front_bytes, sdpa_grid=None, push_grid=None, hot=hot)

    def plan_gather_push():
        table, compact_idx = rig.plan()
        gathered = rig.gather_table(table)
        return table, compact_idx, gathered, rig.hot(compact_idx, table) if hot else None, rig.push(0, 0, 0)

    # A shared layer's exposed push is the remainder: this chunk's own rows. Against it, a plain all-gather of the
    # chunk's rows (160 per chip) into every chip.
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    chunk_rows = rig.s_local * mesh_device.get_num_devices()
    chunk = _kv_cache(mesh_device, 1, chunk_rows, kv_fmt, shard_seq, seed=3)
    chunk_out = _zero_kv_buffer(mesh_device, chunk_rows, kv_fmt)
    chunk_gather = lambda: ttnn.experimental.high_bw_all_gather(
        chunk, dim=2, output_tensor=chunk_out, cluster_axis=None, num_links=_LINKS
    )
    us = {
        "plan": _traced_us(mesh_device, rig.plan),
        "table gather": _traced_us(mesh_device, rig.gather_table),
        **({"hot": _traced_us(mesh_device, lambda: rig.hot(rig.compact_idx))} if hot else {}),
        "push all rows": _traced_us(mesh_device, lambda: rig.push(0, 0, 0)),
        "push remainder": _traced_us(mesh_device, lambda: rig.push(0, 0, 2)),
        "chunk all-gather": _traced_us(mesh_device, chunk_gather),
        "plan + gather + push": _traced_us(mesh_device, plan_gather_push),
    }
    table = "\n".join(f"  {k:<22} {v:9.1f} us" for k, v in us.items())
    logger.info(
        f"unfused sparse KV push, GLM-5.3 L{layer} 1M {kv_fmt} {'l1 front' if front_bytes is None else 'dram'}"
        f"{' hot' if hot else ''} "
        f"(traced, us per call):\n{table}"
    )
