# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Narrow per-RUN DISPATCH PROFILING harness for the CHUNKED-PREFILL MLA module (Kimi K2.6) with RANDOM
weights, parametrized WITH and WITHOUT the H2D stream service, and synced vs pipelined run modes.

Exercises the CHUNKED MLA path (update_padded_kv_cache + chunked ring_mla / SDPA). The MLA input is ONE
chunk of `PREFILL_MLA_CHUNK` tokens; the KV cache holds `kv_actual + chunk` and the chunked ring_mla
attends over logical_n = `PREFILL_MLA_KV` + chunk — so a deep, chunk-aligned `PREFILL_MLA_KV` profiles
the long-KV chunked SDPA (at kv=0 the SDPA only sees the current chunk). Cache content is irrelevant to
timing (no PCC, no prefix preload); only the cache SIZE and the `kv_actual` depth matter.

Runs ttMLA.forward `PREFILL_MLA_ITERS` + 1 times (iter 0 = cold/compile, iters 1.. = warm, reusing the
program cache). Each iter is wrapped in `mla_run_{it}_start/_end` signposts so the device-profiler CSV
can be segmented PER RUN, and the compile run (run 0) excluded. Captures per-run device-op KERNEL time
and OP-TO-OP (dispatch) gap.

The `service` variant BUILDS the (unused) H2D stream service before the loop — inputs are local random
tensors, so the service's only effect is its per-op dispatch contention. NOT a correctness test (no PCC).

Run under the device profiler + tracy via `kimi_perf_overnight/profile_mla_test.sh <noservice|service>
[synced|pipelined]`, or directly (see that script's header). Random weights => NO weight cache / HF
download needed, only the mesh. Env:
    PREFILL_MLA_ITERS  warm iters (default 10; total passes = ITERS + 1 compile)
    PREFILL_MLA_CHUNK  chunk size = MLA input length (default 5120)
    PREFILL_MLA_KV     KV depth fed to chunked ring_mla; chunk-aligned (default 51200 -> logical_n 56320)
"""

import gc
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import build_h2d_service
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

# --- H2D stream service wiring (copied from prefill_runner.py so we don't import the runner main). ---
_H2D_WORKER_COL = int(os.environ.get("PREFILL_H2D_WORKER_COL", "0"))
_H2D_WORKER_ROW = int(os.environ.get("PREFILL_H2D_WORKER_ROW", "0"))
H2D_SYNC_WORKER_CORES = ttnn.CoreRange(
    ttnn.CoreCoord(_H2D_WORKER_COL, _H2D_WORKER_ROW), ttnn.CoreCoord(_H2D_WORKER_COL, _H2D_WORKER_ROW)
)
H2D_METADATA_SIZE_BYTES = 12
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)
H2D_SERVICE_CHUNK = 5 * 1024  # service global tensor sized to one chunk (matches runner)


def run_mla_profile(config, weights, mesh_device, topology, build_service, run_mode):
    iters = int(os.environ.get("PREFILL_MLA_ITERS", "10"))
    # CHUNKED PREFILL: the MLA input is ONE chunk (chunk tokens); the KV cache holds kv_actual + chunk,
    # and kv_actual is the depth fed to the chunked ring_mla (SDPA reads logical_n = kv_actual + chunk
    # from the cache). At kv_actual=0 the SDPA only sees the current chunk (small); a deep, chunk-aligned
    # kv_actual exercises the long-KV chunked SDPA — the op we care about. Cache content is irrelevant
    # to timing (no PCC, no preload); only the SIZE and kv_actual depth matter.
    chunk = int(os.environ.get("PREFILL_MLA_CHUNK", str(5 * 1024)))
    kv_actual = int(os.environ.get("PREFILL_MLA_KV", str(50 * 1024)))
    pipelined = run_mode == "pipelined"  # no synchronize_device between forwards, input pushed once
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this harness targets mesh-8x4, got {mesh_shape}"
    assert chunk % sp == 0, f"chunk {chunk} must be divisible by sp {sp}"
    assert kv_actual % chunk == 0, f"PREFILL_MLA_KV={kv_actual} must be chunk-aligned (multiple of {chunk})"
    # KV-cache window: holds kv_actual + chunk, chunk-aligned, >= 2 chunks (ring_mla needs >= 2 slabs).
    seq_len_cache = max(kv_actual + chunk, 2 * chunk)
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = seq_len_cache

    logger.info(
        f"[mla-profile] CHUNKED RANDOM weights; chunk={chunk} kv_actual={kv_actual} (logical_n="
        f"{kv_actual + chunk}) cache={seq_len_cache} iters={iters} (+1 compile) "
        f"service={'YES' if build_service else 'no'} run_mode={run_mode} mesh={mesh_shape}"
    )

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,  # ttMLA seq_len = KV-cache window for the chunked path
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        slot_num=1,
        layer_num=1,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len_cache, chunk_size_global=chunk)
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=1,
    )

    # Random input hidden states for ONE chunk (sharded SP x TP). Content irrelevant to timing (no PCC).
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1, chunk, config.hidden_size).to(torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2

    # Optionally build the (UNUSED) H2D stream service to introduce its per-op dispatch contention.
    h2d_service = None
    if build_service:
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=tuple(mesh_shape),
            max_seq_len=H2D_SERVICE_CHUNK,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
        )
        logger.info("[mla-profile] built (unused) H2D stream service — measuring its dispatch contention")

    def push_input():
        return ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=shard_dims),
        )

    mesh_device.enable_program_cache()

    # PIPELINED mode: push the input ONCE and reuse it; do NOT synchronize between forwards, so the
    # forwards issue back-to-back and the dispatcher can run ahead (measures true steady-state op2op,
    # no per-run host barrier / input-upload boundary gap).
    # SYNCED mode: re-push the input and synchronize_device after every forward (each run is isolated;
    # its first op's op2op then carries the inter-run sync + host-upload boundary — reported separately).
    shared_input = push_input() if pipelined else None

    profiler.start("mla_forward")
    for it in range(iters + 1):  # iter 0 = cold/compile, 1.. = warm
        tt_hidden_states = shared_input if pipelined else push_input()
        signpost(f"mla_run_{it}_start")
        tt_output = mla_tt.forward(
            hidden_states=tt_hidden_states,
            rope_tensors=indexed_rope,
            kvpe_cache=tt_kvpe_cache,
            kv_actual_isl=kv_actual,
            cache_user_id=0,
        )
        signpost(f"mla_run_{it}_end")
        if not pipelined:
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(tt_hidden_states)
        ttnn.deallocate(tt_output)
        logger.info(f"[mla-profile] run {it}/{iters} ({'compile' if it == 0 else 'WARM'}) done")
    ttnn.synchronize_device(mesh_device)  # drain (the only barrier in pipelined mode)
    if pipelined:
        ttnn.deallocate(shared_input)
    profiler.end("mla_forward")

    if h2d_service is not None:
        del h2d_service
        gc.collect()
        ttnn.synchronize_device(mesh_device)

    logger.success(
        f"[mla-profile] done: chunk={chunk} kv_actual={kv_actual} iters={iters} "
        f"service={'YES' if build_service else 'no'} run_mode={run_mode}; "
        f"mla_forward={profiler.get('mla_forward') * 1000:.1f} ms total"
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            },
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("run_mode", ["synced", "pipelined"], ids=["synced", "pipelined"])
@pytest.mark.parametrize("build_service", [False, True], ids=["noservice", "service"])
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_mla_profile(
    variant,
    config_only,
    random_weights,
    mesh_device,
    device_params,
    topology,
    build_service,
    run_mode,
):
    config, weights = random_weights
    run_mla_profile(config, weights, mesh_device, topology, build_service, run_mode)
