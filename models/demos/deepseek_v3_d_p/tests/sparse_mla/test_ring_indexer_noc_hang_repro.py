# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro: Blackhole Galaxy chip NoC hang in GLM-5.3's fused ring indexer (ttnn.experimental.ring_indexer_score_dsa).

Builds GLM-5.3's sparse-MLA layer exactly as test_sparse_mla_perf.py::test_mla_chunked_perf does for
``glm_5_3 / long / sparse-kv_scaled_fp8`` (layer 74 = the last full indexer layer, compact index-cache slot 20 of 21,
8x4 Galaxy SP=8 x TP=4, torus-xy fabric, one 5120-token chunk at start = 512000 cached tokens, random weights), runs
one forward recording the ring_indexer_score_dsa call (args exactly as the model makes it), then replays ONLY that
call as a trace, with a device synchronize after every replay, for RING_IDX_REPRO_SECS seconds.

Observed on bh-glx-120-b08u02 (8x4 Galaxy, FW 19.16, 2026-11-11):
  * default kernels: one chip's NoC wedges within ~1-40 s (metal's dispatch timeout fires; host MMIO reads of the dead
    chip time out - "MMIO per-op timeout: 4B load took 55 ms" - while its ARC telemetry keeps updating; the other
    31 chips idle, waiting on it). Same signature as the full-model GLM-5.3 prefill soak hang (50-100 min there).
  * RING_AG_SWAP_NOC=1: 320,344 replays in 600 s, no hang.
The fused ring all-gather's sender reader (ring_attention_all_gather_reader.cpp) is created with WriterConfigDescriptor,
i.e. it streams DRAM reads on NOC1 - the NoC the fabric routers use to write the incoming K slices into DRAM. With
RING_AG_SWAP_NOC=1 (see ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp) the sender reader
runs on NOC0 and the sender writer on NOC1. NOTE: that switch is an experiment, not a fix: a NOC0 reader that
forwards slices a NOC1 fabric write just landed reads across DRAM ports (see the cross-port stale-read issue).
It also does not hang under the watcher (TT_METAL_WATCHER=1 slows the kernels).

Run (one 8x4 Blackhole Galaxy):
    pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_ring_indexer_noc_hang_repro.py -s
    RING_AG_SWAP_NOC=1 pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_ring_indexer_noc_hang_repro.py -s

Knobs (env): RING_IDX_REPRO_SECS (default 300), RING_IDX_REPRO_SYNC_EVERY (default 1; 0 = sync once a minute),
RING_IDX_REPRO_SCENARIO (default long = 0.5M cache; warm = 50k, slower to hang), RING_AG_SWAP_NOC (0/1).
"""

import contextlib
import copy
import os
import time

import pytest
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import (
    PERF_FABRIC,
    PERF_SKIP_REASON,
    PERF_WORKLOAD,
    SCENARIOS,
    _detect_perf_workload,
    _local_cache_tokens,
)
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE

pytestmark = pytest.mark.perf
OP_NAME = "ring_indexer_score_dsa"


@pytest.fixture(autouse=True, scope="module")
def _require_perf(request):
    if not is_marker_explicitly_selected(request.config, "perf"):
        pytest.skip("stress repro: run with pytest -m perf")


def _tensors(obj):
    if isinstance(obj, ttnn.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def _key(t):
    try:
        return (t.buffer_address(), str(t.memory_config().buffer_type))
    except Exception:
        return None


@contextlib.contextmanager
def _record(calls):
    """Records every ttnn.experimental.ring_indexer_score_dsa call of one forward and keeps its inputs alive."""
    orig_op, orig_dealloc = getattr(ttnn.experimental, OP_NAME), ttnn.deallocate
    keep, keys = [], set()

    def op(*args, **kwargs):
        for t in _tensors((args, kwargs)):
            keep.append(t)
            keys.add(_key(t))
        calls.append((args, kwargs))
        return orig_op(*args, **kwargs)

    def dealloc(t, *a, **k):
        if isinstance(t, ttnn.Tensor) and _key(t) in keys:
            return None
        return orig_dealloc(t, *a, **k)

    setattr(ttnn.experimental, OP_NAME, op)
    ttnn.deallocate = dealloc
    try:
        yield orig_op
    finally:
        setattr(ttnn.experimental, OP_NAME, orig_op)
        ttnn.deallocate = orig_dealloc


@pytest.mark.parametrize("mesh_device", [PERF_WORKLOAD.mesh_shape], ids=[PERF_WORKLOAD.id], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": PERF_FABRIC,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            "trace_region_size": 16 * 1024 * 1024,
        }
    ],
    ids=["torus-xy" if PERF_FABRIC == ttnn.FabricConfig.FABRIC_2D_TORUS_XY else "fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="hardware stress repro")
@pytest.mark.timeout(0)
def test_ring_indexer_noc_hang_repro(mesh_device, variant, config_only):
    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)
    workload, skip_reason = _detect_perf_workload(variant.name)
    if skip_reason:
        pytest.skip(skip_reason)
    scenario = os.environ.get("RING_IDX_REPRO_SCENARIO", "long")
    secs = int(os.environ.get("RING_IDX_REPRO_SECS", "300"))
    sync_every = int(os.environ.get("RING_IDX_REPRO_SYNC_EVERY", "1"))

    # ---- GLM-5.3 sparse-MLA layer, as test_mla_chunked_perf builds it for sparse-kv_scaled_fp8 (tp-sharded KV)
    kv_cache_format = MlaKvCacheFormat.SCALED_FP8
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    assert (sp, tp) == workload.mesh_shape, f"expected mesh {workload.mesh_shape}, got {(sp, tp)}"
    cache, chunk = _local_cache_tokens(SCENARIOS[scenario]["cache"], sp), workload.chunk_tokens
    total = cache + chunk
    config = copy.deepcopy(config_only)
    config.max_seq_len = total
    weights = random_mla_weights(config)
    full_indexer_layers = num_full_indexer_layers(config)
    mla_layer_idx = [i for i, mode in enumerate(config.indexer_types) if mode == "full"][-1]  # layer 74 (slot 20)
    mla = ttMLA(
        config,
        dict(weights),
        mesh_device,
        topology=per_axis_topology(PERF_FABRIC),
        layer_idx=mla_layer_idx,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        has_indexer=True,
        sparse_kv_cache_format=kv_cache_format,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    kvpe_cache = init_mla_kv_cache(
        cache_format=kv_cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=tp_axis,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=mla._indexer.index_args.index_head_dim,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=full_indexer_layers,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        tp_axis=tp_axis,
    )
    hidden = make_hidden(chunk, config.hidden_size, seed=42)
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_x = ttnn.from_torch(
        hidden[:, :chunk].unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    def forward():
        return mla.forward(tt_x, rope, kvpe_cache, actual_start=cache, cache_layer_idx=0, index_kv_cache=index_kv_cache)

    logger.info(
        f"RING_IDX_REPRO: glm_5_3 layer {mla_layer_idx}, {scenario}: {chunk}-token chunk at start={cache} "
        f"(total {total}), SP={sp} x TP={tp}, {PERF_FABRIC}, RING_AG_SWAP_NOC={os.environ.get('RING_AG_SWAP_NOC', '0')}"
    )
    ttnn.deallocate(forward())  # compile every program, create the CCL resources
    ttnn.synchronize_device(mesh_device)
    calls = []
    with _record(calls) as op:
        out = forward()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    assert len(calls) == 1, f"expected one {OP_NAME} call per forward, got {len(calls)}"
    args, kwargs = calls[0]

    # ---- replay only the indexer call
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    captured = op(*args, **kwargs)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"RING_IDX_REPRO: replaying the traced {OP_NAME} call for {secs} s (sync every {sync_every})")
    t0 = last = time.monotonic()
    n = 0
    try:
        while time.monotonic() - t0 < secs:
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            n += 1
            if sync_every and n % sync_every == 0:
                ttnn.synchronize_device(mesh_device)
            if time.monotonic() - last >= 30:
                ttnn.synchronize_device(mesh_device)
                last = time.monotonic()
                logger.info(f"RING_IDX_REPRO {last - t0:.0f} s: {n} replays")
        ttnn.synchronize_device(mesh_device)
        logger.info(f"RING_IDX_REPRO done: {n} replays in {time.monotonic() - t0:.0f} s, no hang")
    finally:
        ttnn.release_trace(mesh_device, tid)
        del captured
