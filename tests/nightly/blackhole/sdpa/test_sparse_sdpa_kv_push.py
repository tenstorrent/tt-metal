# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa on a restricted core grid, and the sparse KV push beside it on a sub-device (GLM shared layers).

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


def _traced_us(mesh_device, fn, n=10):
    """Mean wall time of one traced replay of fn, over n back-to-back replays."""
    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / n * 1e6
    ttnn.release_trace(mesh_device, tid)
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
    """sparse_sdpa alone on all cores and on 96- and 80-core subsets, at the GLM shape (traced)."""
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


def _trace_1m_topk(layer):
    """The staged 1M top-k of `layer`. Skips locally when it is not staged; in CI it must be."""
    f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
    if not f.exists():
        if os.environ.get("CI") == "true":
            pytest.fail(f"1M GLM-5.3 top-k not staged on this CI machine: {f}")
        pytest.skip(f"1M GLM-5.3 top-k not staged: {f}")
    return f


@run_for_blackhole()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("layer", [14, 62], ids=["L14", "L62"])
def test_sparse_kv_push_real_topk_matches_gather(mesh_device, layer):
    """With real 1M top-k, a buffer filled by one full push, and one filled by the prefix push (beside attention,
    on a sub-device) plus the remainder, must both give the dense gather's attention exactly."""
    from safetensors.numpy import load_file

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart as Part
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh, packet_shape

    f = _trace_1m_topk(layer)
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    kv_fmt, chunk_global, links = "fp8", 5120, 2
    sdpa_grid, push_grid = _cols(0, 7), _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    stripe_chunk = s_local = chunk_global // n_dev
    cache_tokens = _TRACE_1M_START + chunk_global
    page_bytes = 704  # 656 B scaled FP8 rows, DRAM aligned
    pages_per_packet, banks = packet_shape(mesh_device, page_bytes)
    # Capped at the chip's query rows x top-k, as the model sizes it, not at the cache.
    buffer_rows = min(cache_tokens, s_local * TOPK) + n_dev * pages_per_packet * banks

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)  # slots: layer L, layer L + 1
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )
    start = ttnn.from_torch(
        torch.tensor([[[[_TRACE_1M_START]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    sems = [
        ttnn.create_global_semaphore(mesh_device, _cols(0, grid.x - 1), 0, ttnn.BufferType.L1_SMALL) for _ in range(2)
    ]
    natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, s_local, TOPK)
    idx_natural = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )

    table, compact_idx = ttnn.experimental.sparse_push_plan(
        idx_natural,
        num_devices=n_dev,
        stripe_chunk=stripe_chunk,
        max_rows=cache_tokens,
        page_bytes=page_bytes,
        chunk_start=start,
    )
    _shard_over_mesh(table, mesh_device)
    tables = ttnn.from_torch(
        torch.zeros(1, 1, table.shape[2] * n_dev, table.shape[3], dtype=torch.int32),
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
        **common,
    )
    ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)

    def push(buf, slot, part, **grid):
        return ttnn.experimental.sparse_kv_push(
            cache,
            output_tensor=buf,
            table=tables,
            stripe_chunk=stripe_chunk,
            max_rows=cache_tokens,
            part=part,
            num_links=links,
            batch_index=slot,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
            **grid,
        )

    # Layer L's buffer: one full push. Layer L + 1's: its prefix beside L's attention, then the remainder.
    compact_l = push(_zero_kv_buffer(mesh_device, buffer_rows, kv_fmt), 0, Part.ALL)
    compact_next = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    sdpa_kw = dict(kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([push_grid])], 0)
    mesh_device.load_sub_device_manager(manager)
    push(compact_next, 1, Part.PREFIX, core_grid=push_grid)
    ttnn.transformer.sparse_sdpa(q, compact_l, compact_idx, V_DIM, **sdpa_kw)
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(manager)
    push(compact_next, 1, Part.REST)

    def dense(slot):
        out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
        ttnn.experimental.high_bw_all_gather(
            cache, dim=2, output_tensor=out, cluster_axis=None, num_links=links, input_batch_index=slot
        )
        return _sparse_sdpa(q, out, idx_natural, s_local * TP, kv_fmt)

    host = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    for slot, buf in ((0, compact_l), (1, compact_next)):
        got = host(ttnn.transformer.sparse_sdpa(q, buf, compact_idx, V_DIM, **sdpa_kw))
        assert torch.equal(
            host(dense(slot)), got
        ), f"layer {layer} slot {slot}: push path differs from the dense gather"


def _segmented_traced_us(mesh_device, fn, n=20):
    """Mean wall time of one replay of fn(controller), captured as a segmented trace so fn may load and clear a
    sub-device manager, over n back-to-back replays."""
    from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

    controller = SubDeviceTraceController(mesh_device)
    fn(None)
    ttnn.synchronize_device(mesh_device)
    controller.begin_capture()
    fn(controller)
    controller.end_capture()
    controller.replay()
    t0 = time.perf_counter()
    for _ in range(n):
        controller._replay_steps(controller._program, None)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / n * 1e6
    controller.release()
    return us


# Per layer: the most each production step may take at the 1M request on one Galaxy (traced, us), measured with a
# margin. None: report only.
_GLM_PUSH_PERF_CEILING_US = None


@run_for_blackhole()
@pytest.mark.skipif(os.getenv("MESH_DEVICE") != "TG", reason="Blackhole Galaxy perf gate requires MESH_DEVICE=TG")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("layer", [14, 62], ids=["L14", "L62"])
def test_sparse_kv_push_glm_perf(mesh_device, layer):
    """Each step of the push in a GLM-5.3 shared layer at the 1M request, with real top-k and the production
    configuration (2 links, scaled FP8 rows, 160-row blocks, the planner's grids and sub-device manager)."""
    from safetensors.numpy import load_file

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart as Part
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPlanner, _shard_over_mesh

    f = _trace_1m_topk(layer)
    kv_fmt, chunk_global, links = "fp8", 5120, 2
    num_devices = mesh_device.get_num_devices()
    stripe_chunk = query_rows = chunk_global // num_devices
    cache_tokens = _TRACE_1M_START + chunk_global
    page_bytes = 704  # 656 B scaled FP8 rows, DRAM aligned
    planner = SparseKvPushPlanner(mesh_device, links)
    attention_grid, push_grid = planner.grids(query_rows)
    manager = planner.prefetch_manager(query_rows)
    buffer_rows = planner.buffer_rows(query_rows * TOPK, cache_tokens, stripe_chunk, page_bytes)

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)
    q = ttnn.from_torch(
        torch.randn(1, H, num_devices * query_rows, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )
    start = ttnn.from_torch(
        torch.tensor([[[[_TRACE_1M_START]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(num_devices, query_rows, TOPK)
    indices = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, num_devices * query_rows, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )
    plan_kw = dict(
        num_devices=num_devices,
        stripe_chunk=stripe_chunk,
        max_rows=cache_tokens,
        page_bytes=page_bytes,
        chunk_start=start,
    )
    table, compact_idx = ttnn.experimental.sparse_push_plan(indices, **plan_kw)
    _shard_over_mesh(table, mesh_device)
    tables = ttnn.from_torch(
        torch.zeros(1, 1, table.shape[2] * num_devices, table.shape[3], dtype=torch.int32),
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
        **common,
    )
    gather_kw = dict(dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
    ttnn.experimental.high_bw_all_gather(table, **gather_kw)

    compact_this = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    compact_next = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    push_kw = dict(
        table=tables,
        stripe_chunk=stripe_chunk,
        max_rows=cache_tokens,
        num_links=links,
        ready_semaphore=planner.ready_semaphore,
        data_valid_semaphore=planner.data_valid_semaphore,
    )
    ttnn.experimental.sparse_kv_push(cache, output_tensor=compact_this, part=Part.ALL, batch_index=0, **push_kw)
    sdpa_kw = dict(kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128)

    def plan(controller):
        t, c = ttnn.experimental.sparse_push_plan(indices, **plan_kw)
        ttnn.deallocate(t)
        ttnn.deallocate(c)

    def push(part, **grid):
        return lambda controller: ttnn.experimental.sparse_kv_push(
            cache, output_tensor=compact_next, part=part, batch_index=1, **push_kw, **grid
        )

    def attention(controller):
        ttnn.deallocate(ttnn.transformer.sparse_sdpa(q, compact_this, compact_idx, V_DIM, **sdpa_kw))

    def attention_beside_prefix(controller):
        # The production overlap: the next layer's prefix on sub-device 1 beside this layer's attention.
        (controller.sub_device_load if controller else mesh_device.load_sub_device_manager)(manager)
        push(Part.PREFIX, core_grid=push_grid)(controller)
        attention(controller)
        (controller.sub_device_clear if controller else mesh_device.clear_loaded_sub_device_manager)()

    steps = {
        "plan": plan,
        "table gather": lambda controller: ttnn.experimental.high_bw_all_gather(table, **gather_kw),
        "push all rows": push(Part.ALL),
        "push prefix, push grid": push(Part.PREFIX, core_grid=push_grid),
        "push rest": push(Part.REST),
        "attention, full grid": attention,
        "attention + prefix push on sub-devices": attention_beside_prefix,
    }
    measured = {name: _segmented_traced_us(mesh_device, fn) for name, fn in steps.items()}
    for name, us in measured.items():
        logger.info(f"GLM push perf L{layer} {name}: {us:.1f} us")
    planner.release()

    ceilings = (_GLM_PUSH_PERF_CEILING_US or {}).get(layer)
    if ceilings is not None:
        over = {name: (us, ceilings[name]) for name, us in measured.items() if us > ceilings[name]}
        assert not over, f"L{layer} steps over their ceiling (measured, ceiling us): {over}"
