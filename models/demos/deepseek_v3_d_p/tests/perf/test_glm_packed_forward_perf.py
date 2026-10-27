# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.3 traced prefill time per chunk vs chunks per forward (packed forwards), on synthetic weights.

Each row builds a traced TtPrefillRuntime with ``forward_chunks`` chunks per forward (1 = the regular
one-chunk traced path) and times replays of one forward whose chunks come from different users at the
same depth (one chunk of KV prefix each), so attention costs the same in every row. Layers 0-2 are dense
and own full indexers; layer 3 is MoE and reuses layer 2's top-k, like most GLM-5.3 layers. The 4-layer
time minus the 3-layer time is one such MoE layer.

Record only, no gate. ``GLM_PACKED_PERF_REPLAYS`` sets the timed replays (median reported).
"""

import copy
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

# TODO: import from models.demos.common.prefill.prefill_chunk once #62535 moves it there.
from models.demos.minimax_m3.tt.prefill_chunk import PrefillChunk

_REPLAYS = int(os.environ.get("GLM_PACKED_PERF_REPLAYS", "5"))
_TRACE_REGION_SIZE = 512 * 1024 * 1024
_DEPTH_CHUNKS = 1  # KV prefix each timed chunk attends to, in chunks
# Profile only: the prefix each profiled chunk attends to, in tokens (default one chunk). Timing does not depend
# on the prefix's values, so it is left unwritten: 40960 profiles the workload test's depth without 20 fills.
_PROFILE_DEPTH = int(os.environ.get("GLM_PROFILE_DEPTH", "0"))


def _metadata_msg(mesh_device, words):
    return ttnn.from_torch(
        torch.tensor(list(words), dtype=torch.int64).reshape(1, 1, 1, len(words)),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("forward_chunks", [1, 2, 4], ids=["f1", "f2", "f4"])
@pytest.mark.parametrize("num_layers", [3, 4], ids=["L3", "L4"])
# Two links is what the galaxy runs; one is the shared LoudBox block-test default.
@pytest.mark.parametrize("num_links", [1, 2], ids=["link1", "link2"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=_TRACE_REGION_SIZE,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="loudbox-fabric2d-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
def test_glm_packed_forward_perf(
    variant, config_only, mesh_device, device_params, num_links, num_layers, forward_chunks
):
    sp = mesh_device.shape[0]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    seq_cache = (_DEPTH_CHUNKS + 1) * chunk
    start = _DEPTH_CHUNKS * chunk
    config = copy.deepcopy(config_only)
    config.max_seq_len = seq_cache
    rt_config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=chunk,
        forward_tokens_budget=forward_chunks * chunk,
        num_users=forward_chunks,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        model_cfg=GLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(config, GLM53Config, num_layers)
    runtime = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
    del state_dict
    slots = rt_config.cache_num_users
    mesh_shape = list(mesh_device.shape)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=num_layers,
            num_users=slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=full_indexer_rank(config, num_layers),
            num_users=slots,
            dtype=ttnn.bfloat8_b,
        ),
    )
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        g = torch.Generator().manual_seed(7)

        def tokens():
            return torch.randint(0, config.vocab_size, (chunk,), generator=g).tolist()

        def run(chunk_start):
            if forward_chunks == 1:
                msg = _metadata_msg(mesh_device, (0, chunk_start, chunk_start + chunk))
                runtime.prefill_chunk(
                    runtime.make_chunk_input(tokens()),
                    caches,
                    slot_id=0,
                    actual_start=chunk_start,
                    actual_end=chunk_start + chunk,
                    metadata_msg=msg,
                )
                ttnn.deallocate(msg)
            else:
                chunks = [PrefillChunk(u, chunk_start, chunk_start + chunk) for u in range(forward_chunks)]
                runtime.prefill_forward(
                    runtime.make_forward_input([tokens() for _ in chunks], [c.actual_start for c in chunks]),
                    caches,
                    chunks,
                )
            ttnn.synchronize_device(mesh_device)

        # Fill each user's prefix, then time forwards at that depth (each replay rewrites the same chunk).
        for c in range(_DEPTH_CHUNKS):
            run(c * chunk)
        times_ms = []
        for _ in range(max(1, _REPLAYS)):
            t0 = time.perf_counter()
            run(start)
            times_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        runtime.release_trace()

    median_ms = statistics.median(times_ms)
    logger.warning(
        f"GLM_PACKED_PERF mesh={sp}x{mesh_device.shape[1]} links={num_links} layers={num_layers} forward_chunks={forward_chunks} "
        f"chunk={chunk} depth={start} forward_ms={median_ms:.2f} per_chunk_ms={median_ms / forward_chunks:.2f} "
        f"runs_ms={' '.join(f'{t:.2f}' for t in times_ms)}"
    )


# Device profile of one eager forward, for `python -m tracy -r -p -v -m pytest ... -k profile`. The measured
# forward sits between PROFILE_MEASURE_START / PROFILE_MEASURE_END signposts and the transformer's
# forward_layer_{i}_start / _end signposts, so per-op device time can be split by layer.
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("forward_chunks", [1, 2, 4], ids=["f1", "f2", "f4"])
# Rows per chip of one chunk: 640 is the production chunk (5K on 8x4); 256 is 8x4's 2,048-token chunk.
@pytest.mark.parametrize("rows_per_chip", [PREFILL_CHUNK_TOKENS_PER_CHIP, 256], ids=["r640", "r256"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config), l1_small_size=GLM53Adapter.l1_small_size
            ),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="loudbox-fabric2d-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config), l1_small_size=GLM53Adapter.l1_small_size
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="galaxy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
def test_glm_packed_forward_profile(
    variant, config_only, mesh_device, device_params, num_links, rows_per_chip, forward_chunks
):
    from tracy import signpost

    num_layers = 4
    sp = mesh_device.shape[0]
    chunk = rows_per_chip * sp
    start = _PROFILE_DEPTH or _DEPTH_CHUNKS * chunk
    assert start % chunk == 0, f"GLM_PROFILE_DEPTH={start} must be a whole number of {chunk}-token chunks"
    seq_cache = start + chunk
    config = copy.deepcopy(config_only)
    config.max_seq_len = seq_cache
    rt_config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=chunk,
        forward_tokens_budget=forward_chunks * chunk,
        num_users=forward_chunks,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        model_cfg=GLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(config, GLM53Config, num_layers)
    runtime = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
    del state_dict
    mesh_shape = list(mesh_device.shape)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=num_layers,
            num_users=forward_chunks,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=full_indexer_rank(config, num_layers),
            num_users=forward_chunks,
            dtype=ttnn.bfloat8_b,
        ),
    )
    g = torch.Generator().manual_seed(7)

    def run(chunk_start):
        # The model directly, with read_profiler: the device profiler is flushed after every layer, so a
        # packed forward (4x the ops of a chunk) cannot overflow its buffer and drop records.
        tokens = [torch.randint(0, config.vocab_size, (chunk,), generator=g).tolist() for _ in range(forward_chunks)]
        if forward_chunks == 1:
            inp = runtime.make_chunk_input(tokens[0])
            runtime.model.forward(
                inp,
                caches.kvpe,
                actual_isl=chunk,
                actual_start=chunk_start,
                actual_end=chunk_start + chunk,
                cache_user_id=0,
                index_kv_cache=caches.index,
                read_profiler=True,
            )
        else:
            chunks = [PrefillChunk(u, chunk_start, chunk_start + chunk) for u in range(forward_chunks)]
            inp = runtime.make_forward_input(tokens, [c.actual_start for c in chunks])
            runtime.model.forward(
                inp, caches.kvpe, actual_isl=chunk, chunks=chunks, index_kv_cache=caches.index, read_profiler=True
            )
        ttnn.deallocate(inp)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)

    try:
        if not _PROFILE_DEPTH:
            for c in range(_DEPTH_CHUNKS):
                run(c * chunk)
        run(start)  # one more at the measured depth, so the measured forward is a program-cache hit
        signpost("PROFILE_MEASURE_START")
        run(start)
        signpost("PROFILE_MEASURE_END")
    finally:
        runtime.model.release_sub_device_managers()
