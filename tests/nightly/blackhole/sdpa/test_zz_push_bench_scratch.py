# SCRATCH (not for merge): per-step traced timing of the sparse KV push on real GLM-5.3 1M top-k, no gates.
# BENCH_LAYERS=14,62  BENCH_STARTS=1039360,51200 (chunk starts; indices are scaled into the shorter prefix)
# BENCH_INBOX=1: the push exchanges the plan tables itself (no table all-gather); also checks the pushed buffer
# against the all-gathered push bit for bit.
import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_sparse_sdpa_kv_push import (
    _KV_FORMATS,
    _TORUS_XY_GLM,
    _TRACE_1M_START,
    TOPK,
    _kv_cache,
    _push_steps_us,
    _segmented_traced_us,
    _trace_1m_topk,
    _zero_kv_buffer,
)

_PAYLOAD = int(os.environ.get("BENCH_PAYLOAD", "6144"))
_PARAMS = {**_TORUS_XY_GLM}
_ROUTER = ttnn.FabricRouterConfig()
_ROUTER.max_packet_payload_size_bytes = _PAYLOAD
_PARAMS["fabric_router_config"] = _ROUTER


def _inbox_steps_us(mesh_device, natural, chunk_start, dense_below=0):
    """Inbox mode: plan, push (all / prefix on the push grid / rest), and the v1 push for comparison; checks the
    inbox push's compact buffer equals the all-gathered push's."""
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart as Part
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPlanner

    kv_fmt, chunk_global, links = "fp8", 5120, 2
    n = mesh_device.get_num_devices()
    stripe_chunk = query_rows = chunk_global // n
    cache_tokens = chunk_start + chunk_global
    page_bytes = 704
    planner = SparseKvPushPlanner(mesh_device, links)
    _, push_grid = planner.grids(query_rows)
    _, kv_dtype, kv_width = _KV_FORMATS[kv_fmt]
    buffer_rows = ttnn.experimental.sparse_kv_push_output_spec(
        mesh_device,
        distinct_rows=query_rows * TOPK,
        max_rows=cache_tokens,
        stripe_chunk=stripe_chunk,
        row_width=kv_width,
        dtype=kv_dtype,
    ).shape[2]
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)
    start = ttnn.from_torch(
        torch.tensor([[[[chunk_start]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    indices = ttnn.from_torch(
        torch.from_numpy(natural.reshape(1, 1, n * query_rows, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )
    plan_kw = dict(
        stripe_chunk=stripe_chunk, max_rows=cache_tokens, page_bytes=page_bytes, chunk_start_tensor=start
    )
    if dense_below:
        plan_kw["dense_below"] = dense_below
    table, compact_idx = ttnn.experimental.sparse_push_plan(indices, **plan_kw)[:2]
    table_om = ttnn.experimental.sparse_push_plan(indices, **plan_kw, owner_major=True)[0]
    slot_words = ttnn._ttnn.operations.experimental.sparse_kv_push_inbox_slot_bytes(cache_tokens, stripe_chunk, n) // 4
    inbox = ttnn.from_torch(torch.zeros(1, 1, n, slot_words, dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common)
    tables = ttnn.from_torch(
        torch.zeros(1, 1, table.shape[2] * n, table.shape[3], dtype=torch.int32),
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
        **common,
    )
    ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
    out_v1 = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    out_v2 = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    push_kw = dict(
        stripe_chunk=stripe_chunk,
        max_rows=cache_tokens,
        num_links=links,
        ready_semaphore=planner.ready_semaphore,
        done_semaphore=planner.done_semaphore,
        batch_index=0,
    )
    ttnn.experimental.sparse_kv_push(cache, output_tensor=out_v1, table=tables, part=Part.ALL, **push_kw)
    ttnn.experimental.sparse_kv_push(cache, output_tensor=out_v2, table=table_om, inbox=inbox, part=Part.ALL, **push_kw)
    ttnn.synchronize_device(mesh_device)
    a, b = ttnn.to_torch(out_v1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)), ttnn.to_torch(
        out_v2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    a, b = a.view(torch.uint8), b.view(torch.uint8)  # FP8 rows: compare bits (random data holds NaNs)
    same = bool(torch.equal(a, b))
    if not same:
        for chip in range(n):
            x, y = a[chip].reshape(-1, a.shape[-1]), b[chip].reshape(-1, b.shape[-1])
            diff = (x != y).any(dim=-1)
            nz_x, nz_y = (x != 0).any(dim=-1), (y != 0).any(dim=-1)
            if diff.any():
                idx = diff.nonzero().flatten()
                logger.warning(
                    f"chip {chip}: {int(diff.sum())} rows differ of {int(nz_x.sum())} v1 rows / {int(nz_y.sum())} v2 rows; "
                    f"v2 zero where v1 set: {int((nz_x & ~nz_y).sum())}, v2 set where v1 zero: {int((~nz_x & nz_y).sum())}; "
                    f"first rows {idx[:8].tolist()}"
                )
    # PREFIX then REST must fill the same slots as ALL.
    out_v3 = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    ttnn.experimental.sparse_kv_push(
        cache, output_tensor=out_v3, table=table_om, inbox=inbox, part=Part.PREFIX, core_grid=push_grid, **push_kw
    )
    ttnn.experimental.sparse_kv_push(
        cache, output_tensor=out_v3, table=table_om, inbox=inbox, part=Part.REST, reuse_inbox=True, **push_kw
    )
    ttnn.synchronize_device(mesh_device)
    c = ttnn.to_torch(out_v3, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).view(torch.uint8)
    same_parts = bool(torch.equal(a, c))

    def plan(controller):
        outs = ttnn.experimental.sparse_push_plan(indices, **plan_kw, owner_major=True)
        for t in outs:
            ttnn.deallocate(t)

    def push(part, **kw):
        return lambda controller: ttnn.experimental.sparse_kv_push(
            cache, output_tensor=out_v2, part=part, **push_kw, **kw
        )

    steps = {
        "plan": plan,
        "v1 table gather": lambda controller: ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links
        ),
        "v1 push all rows": push(Part.ALL, table=tables),
        "v2 push all rows": push(Part.ALL, table=table_om, inbox=inbox),
        "v2 push prefix, push grid": push(Part.PREFIX, table=table_om, inbox=inbox, core_grid=push_grid),
        "v2 push rest": push(Part.REST, table=table_om, inbox=inbox),
        "v2 push rest, reused inbox": push(Part.REST, table=table_om, inbox=inbox, reuse_inbox=True),
    }
    measured = {name: _segmented_traced_us(mesh_device, fn) for name, fn in steps.items()}
    measured["v2 == v1 (bit-exact)"] = float(same)
    measured["v2 prefix+rest == v1 all"] = float(same_parts)
    planner.release()
    return measured


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_push_bench(mesh_device):
    from safetensors.numpy import load_file

    layers = [int(x) for x in os.environ.get("BENCH_LAYERS", "14,62").split(",")]
    starts = [int(x) for x in os.environ.get("BENCH_STARTS", str(_TRACE_1M_START)).split(",")]
    inbox = os.environ.get("BENCH_INBOX") == "1"
    n = mesh_device.get_num_devices()
    rows = []
    for layer in layers:
        natural = next(iter(load_file(str(_trace_1m_topk(layer))).values())).astype(np.int64).reshape(n, -1, TOPK)
        for start in starts:
            idx = natural if start == _TRACE_1M_START else (natural * (start + 5120)) // (_TRACE_1M_START + 5120)
            measured = (_inbox_steps_us if inbox else _push_steps_us)(mesh_device, idx, start)
            for name, us in measured.items():
                rows.append(f"BENCH L{layer} start={start} {name}: {us:.1f} us")
    for r in rows:
        logger.warning(r)
