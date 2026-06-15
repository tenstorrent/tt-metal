# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Narrow per-op DISPATCH PROFILING harness for TtPrefillTransformer (Kimi K2.6), scoped to a small
number of layers (default 3) and ONE warm chunk at a fixed deep KV position, parametrized
WITH and WITHOUT the H2D stream service.

Purpose: measure the per-op / per-layer kernel time and op-to-op (dispatch) gap for a warm chunk,
isolating the effect of the resident H2D stream service on dispatch (the service is BUILT but UNUSED
here — tokens still come from the local longbook trace — so its only effect is its per-op dispatch
contention, exactly the runner-vs-test methodology).

This is NOT a correctness test (no PCC). It runs `PREFILL_PROFILE_ITERS` passes of forward_chunk at
`kv_actual = PREFILL_PROFILE_KV` (so logical_n = kv + chunk). Pass 0 is the cold/compile pass; passes
1.. are warm and reuse the program cache. Each pass emits `forward_chunk_layer_{i}_start/_end`
signposts (from TtPrefillTransformer.forward_chunk) so the device-profiler CSV can be sliced to the
LAST (warm) pass only — i.e. the compile pass is excluded by the parser, never captured.

Run under the device profiler + tracy via the companion script `kimi_perf_overnight/profile_3L_test.sh`,
or directly (see the run command in that script's header). Requires an 8x4 Blackhole mesh, the Kimi
TTNN weight cache, and the longbook golden trace (for real token ids). Env:
    PREFILL_PROFILE_KV     deep KV position to profile (default 51200 -> chunk 10, logical_n 56320)
    PREFILL_PROFILE_ITERS  forward_chunk passes (default 2: pass0 compile, pass1 warm)
    PREFILL_PROFILE_NUM_LAYERS  layers to build/run (default 3)
    DEEPSEEK_PREFILL_TRACE_DIR  override the golden trace dir
"""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import build_h2d_service
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

CHUNK = 5 * 1024  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (1 user)

DEFAULT_TRACE_DIR = (
    "/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok"
)

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


def _resolve_trace_dir(variant) -> Path:
    env = getattr(variant, "prefill_trace_env", None)
    if env:
        return Path(os.environ.get(env, getattr(variant, "prefill_trace_default", DEFAULT_TRACE_DIR)))
    return Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_TRACE_DIR))


def _load_metadata_token_ids(trace_dir: Path, total_len: int) -> torch.Tensor:
    with open(trace_dir / "metadata.json") as f:
        token_ids = json.load(f)["token_ids"]
    return torch.tensor(token_ids[:total_len], dtype=torch.int64)


def run_3l_profile(variant, config, mesh_device, weight_cache_path, num_links, topology, build_service):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")

    num_layers = int(os.environ.get("PREFILL_PROFILE_NUM_LAYERS", "3"))
    profile_kv = int(os.environ.get("PREFILL_PROFILE_KV", "51200"))
    iters = int(os.environ.get("PREFILL_PROFILE_ITERS", "2"))
    assert profile_kv % CHUNK == 0, f"PREFILL_PROFILE_KV={profile_kv} must be a multiple of chunk {CHUNK}"
    logical_n = profile_kv + CHUNK
    assert logical_n <= SEQ_CACHE, f"logical_n={logical_n} exceeds cache {SEQ_CACHE}"

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this harness targets mesh-8x4, got {mesh_shape}"
    chunk_local = CHUNK // sp  # 640
    config.max_seq_len = SEQ_CACHE
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank

    logger.info(
        f"[3L-profile] layers={num_layers} mesh={mesh_shape} kv={profile_kv} logical_n={logical_n} "
        f"iters={iters} service={'YES' if build_service else 'no'}"
    )

    # Real token ids from the golden longbook trace (content is irrelevant to timing; no PCC).
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")
    token_ids_full = _load_metadata_token_ids(trace_dir, logical_n)
    assert token_ids_full.numel() >= logical_n, f"trace has {token_ids_full.numel()} tokens, need {logical_n}"

    # --- Weights from the prebuilt TTNN cache. ---
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,
        mla_seq_len=SEQ_CACHE,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=SEQ_CACHE, chunk_size_global=CHUNK)
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    # Block-cyclic SP-sharded token tile for the single profiled chunk (kv_actual = profile_kv).
    positions = rotated_chip_positions(profile_kv, sp, chunk_local)
    flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
    chunk_tok = token_ids_full[flat].reshape(sp, 1, chunk_local)

    # Optionally build the (UNUSED) H2D stream service to introduce its per-op dispatch contention.
    # Mirrors prefill_runner.py's PREFILL_FORCE_BUILD_SERVICE path: compile() leaves a custom sub-device
    # manager loaded; revert to the whole-chip default so the service's init program validates.
    h2d_service = None
    if build_service:
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=tuple(mesh_shape),
            max_seq_len=CHUNK,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
        )
        logger.info("[3L-profile] built (unused) H2D stream service — measuring its dispatch contention")

    mesh_device.enable_program_cache()

    profiler.start("tt_forward")
    for it in range(iters):
        warm = it > 0
        logger.info(f"[3L-profile] pass {it + 1}/{iters} ({'WARM' if warm else 'cold/compile'}) kv={profile_kv}")
        tt_tokens = ttnn.from_torch(
            chunk_tok,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )
        h, _ = transformer.forward_chunk(
            tt_tokens,
            tt_kvpe_cache,
            indexed_rope,
            kv_actual_isl=profile_kv,
            cache_user_id=0,
            return_layer_outputs=False,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(h)
        ttnn.deallocate(tt_tokens)
    profiler.end("tt_forward")

    # Release the H2D service while the mesh + command queues are still alive (its dtor frees a command
    # queue + service-core L1; running it after the mesh_device fixture closes would abort).
    if h2d_service is not None:
        del h2d_service
        gc.collect()
        ttnn.synchronize_device(mesh_device)

    logger.success(
        f"[3L-profile] done: layers={num_layers} kv={profile_kv} iters={iters} "
        f"service={'YES' if build_service else 'no'}; tt_forward={profiler.get('tt_forward') * 1000:.1f} ms"
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("build_service", [False, True], ids=["noservice", "service"])
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_prefill_3l_profile(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_links,
    topology,
    build_service,
):
    run_3l_profile(variant, config_only, mesh_device, weight_cache_path, num_links, topology, build_service)
