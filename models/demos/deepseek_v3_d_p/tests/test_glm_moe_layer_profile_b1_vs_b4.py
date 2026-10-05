# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One GLM-5.3 MoE layer, batch 1 (TP=4 over the 8x4 mesh) vs batch 4 (one user per column), for device
profiling at a given KV depth.

Layer 6 (full indexer + MoE) runs as a TtPrefillBlock over consecutive chunks of the 55k vLLM trace,
teacher-forced (input = golden decoder_output_layer_5), so the layer's own KVPE / index-K cache builds up
chunk by chunk exactly as in the full model. Chunk 0 is the empty-cache case, chunk 51200/chunk the ~50k-KV
one. Every chunk's block output is PCC'd against golden decoder_output_layer_6 (every user in batch 4), and
the KVPE and index-K caches against their goldens after the run, so the profiled run is also the accuracy
check.

  test_glm_moe_layer_profile         -- eager: every chunk is bracketed by `prof_chunk{c}` signposts and
                                        MLA / MoE by the model's own MLA_START/END and MoE_START/END.
  test_glm_moe_layer_profile_traced  -- the layer is captured once (metadata-driven, segmented around the
                                        MoE sub-device swaps) and every chunk is a trace replay, so device
                                        time carries no host-dispatch gaps. One eager warm pass between
                                        PERF_WARM_START/END compiles the programs and is the region-label
                                        template: replayed ops carry no host signposts, so a parser maps
                                        each replayed op onto the warm op at the same position. The device
                                        profiler is flushed after the warm pass and after the capture, then
                                        after every replay (`replay_chunk{c}` signpost before each).
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked import (
    GLM_L1_SMALL_SIZE,
    SEQ_CACHE,
    _load_layer_rows,
    _resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata, RotarySetup, write_chunk_metadata
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, gather_cache_natural, unrotate_cache_layer
from tests.ttnn.utils_for_testing import comp_pcc

LAYER = 6  # first full-indexer MoE layer
SP_AXIS, TP_AXIS = 0, 1
# Same capacity factors the full-model drivers use: 8 single-user, 4 for batch 4 (4x the rows).
DISPATCH_FACTOR = {"b1": 8, "b4": 4}
# Chunks covering the 55k golden: 11 x 5120, or 27 x 2048 (the 28th 2k chunk would be half padding).
CHUNK_CASES = {5120: 11, 2048: 27}


def _seq_cache(chunk):
    return -(-SEQ_CACHE // chunk) * chunk


def _setup(variant, config_only, mesh_device, device_params, num_links, weight_cache_path, mode, chunk, n_chunks):
    """Block, caches, rope, goldens and per-chunk host inputs for one (mode, chunk) case."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    batch = mode == "b4"
    topology = per_axis_topology(device_params["fabric_config"])
    sp, cols = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    num_users = cols if batch else 1
    chunk_local = chunk // sp
    total_len = n_chunks * chunk
    seq_cache = _seq_cache(chunk)
    config = config_only
    config.max_seq_len = seq_cache
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = seq_cache
    hidden, idx_dim = config.hidden_size, config.index_head_dim

    trace_dir = _resolve_trace_dir(variant)
    layout = variant.prefill_trace_layout
    x_in = _load_layer_rows(
        trace_dir, layout, "hidden_states", LAYER - 1, f"decoder_output_layer_{LAYER - 1}", 0, total_len
    )
    ref_out = _load_layer_rows(trace_dir, layout, "hidden_states", LAYER, f"decoder_output_layer_{LAYER}", 0, total_len)
    logger.info(
        f"[prof] layer {LAYER} {mode}: {num_users} user(s) x {n_chunks} chunks of {chunk} "
        f"(KV depth per chunk c = c*{chunk}), cache {seq_cache}"
    )

    cache_path = weight_cache_path / f"{sp}x{cols}"
    init_checker(cache_path)
    assert TtPrefillBlock.check_cache_complete(
        cache_path, LAYER, False, GLM53Config.NUM_ROUTED_EXPERTS // (sp * cols), batch_axis=batch, config=config
    ), f"layer {LAYER} cache incomplete for {mode} at {cache_path}"
    block = TtPrefillBlock(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM53Config,
        state_dict={},
        layer_idx=LAYER,
        seq_len=chunk,
        dispatch_buffer_capacity_factor=DISPATCH_FACTOR[mode],
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=cache_path,
        is_chunked=True,
        slot_num=1,
        layer_num=1,
        max_seq_len=seq_cache,
        routing_use_l1_small_for_semaphores=True,
        first_layer_idx=LAYER,
        batch_axis=TP_AXIS if batch else None,
    )
    cache_kwargs = {"batch_axis": TP_AXIS} if batch else {"tp_axis": TP_AXIS}
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        num_users=1,
        **cache_kwargs,
    )
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=idx_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        **cache_kwargs,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_cache, chunk_size_global=chunk
    )
    # batch 1: SP on the sequence, TP on hidden. batch 4: SP on the sequence, user on the column axis.
    dims = (2, 0) if batch else (2, 3)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)

    # Per chunk: the rows each chip holds in the rotated block-cyclic chunked layout.
    flats, x_hosts = [], []
    for c in range(n_chunks):
        positions = rotated_chip_positions(c * chunk, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        x_host = x_in[flat].reshape(1, 1, chunk, hidden)
        if batch:
            x_host = x_host.expand(num_users, 1, chunk, hidden).contiguous()
        flats.append(flat)
        x_hosts.append(x_host.to(torch.bfloat16))

    return SimpleNamespace(
        batch=batch,
        sp=sp,
        num_users=num_users,
        total_len=total_len,
        seq_cache=seq_cache,
        config=config,
        trace_dir=trace_dir,
        layout=layout,
        ref_out=ref_out,
        block=block,
        kvpe_cache=kvpe_cache,
        index_cache=index_cache,
        rope=rope,
        mapper=mapper,
        composer=composer,
        flats=flats,
        x_hosts=x_hosts,
    )


def _output_pcc(ctx, out, c, tag):
    got = ttnn.to_torch(out, mesh_composer=ctx.composer).to(torch.float32)  # [U, 1, chunk, hidden]
    ref = ctx.ref_out[ctx.flats[c]]
    pccs = [comp_pcc(ref, got[u, 0])[1] for u in range(ctx.num_users)]
    logger.info(f"[prof] {tag} chunk {c}: block output PCC per user {[round(p, 6) for p in pccs]}")
    return min(pccs)


def _check_caches(ctx, mesh_device, mode, chunk, n_chunks, out_pcc):
    """Caches vs golden, per user, then the teacher-forced single-layer accuracy floors."""
    config = ctx.config
    kv_lora, idx_dim = config.kv_lora_rank, config.index_head_dim
    g_kv = _load_layer_rows(
        ctx.trace_dir, ctx.layout, "kv_cache", LAYER, f"kv_post_transform_layer_{LAYER}", 0, ctx.total_len
    )
    g_idx = _load_layer_rows(ctx.trace_dir, ctx.layout, "dsa", LAYER, f"indexer_k_layer_{LAYER}", 0, ctx.total_len)
    hadamard = normalized_hadamard_matrix(idx_dim).float()
    if ctx.batch:
        cc = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape)
        kv_users = [t[0] for t in ttnn.to_torch(ctx.kvpe_cache.storage, mesh_composer=cc).float()]
        idx_users = [t[0] for t in ttnn.to_torch(ctx.index_cache, mesh_composer=cc).float()]
        stripes = ctx.sp
    else:
        kv_all, stripes = gather_cache_natural(ctx.kvpe_cache.storage, mesh_device, tp_shard_kv=True)
        idx_all, _ = gather_cache_natural(ctx.index_cache, mesh_device, tp_shard_kv=True)
        kv_users, idx_users = [kv_all[0]], [idx_all[0]]
    p = blockcyclic_positions(stripes, chunk, ctx.seq_cache)
    kv_pcc = [min(cache_half_pccs(g_kv, unrotate_cache_layer(k, p, ctx.total_len), kv_lora, True)) for k in kv_users]
    idx_pcc = [
        min(cache_half_pccs(g_idx, unrotate_cache_layer(k, p, ctx.total_len) @ hadamard, idx_dim // 2, False))
        for k in idx_users
    ]
    logger.info(
        f"[prof] SUMMARY layer {LAYER} {mode} chunk={chunk} x{n_chunks}: block output min PCC "
        f"{min(out_pcc):.6f} (chunk 0 {out_pcc[0]:.6f}, last {out_pcc[-1]:.6f}); KVPE per user "
        f"{[round(v, 6) for v in kv_pcc]}; index-K per user {[round(v, 6) for v in idx_pcc]}"
    )
    # Teacher-forced single layer: the single-user teacher-forced test's 0.98 floor.
    assert min(out_pcc) >= 0.98, f"block output PCC {min(out_pcc):.6f} < 0.98"
    assert min(kv_pcc) >= 0.98 and min(idx_pcc) >= 0.98, f"cache PCC KVPE {kv_pcc} index-K {idx_pcc} < 0.98"


_PROFILE_PARAMS = [
    pytest.mark.parametrize("mode", ["b1", "b4"]),
    pytest.mark.parametrize(
        "chunk, n_chunks", [pytest.param(c, n, id=f"c{c // 1024}k") for c, n in CHUNK_CASES.items()]
    ),
    pytest.mark.parametrize(
        "mesh_device, device_params, num_links",
        [
            pytest.param(
                (8, 4),
                torus_xy_device_params(
                    fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE, l1_small_size=GLM_L1_SMALL_SIZE
                ),
                2,
                marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
                id="torus-xy-8x4",
            )
        ],
        indirect=["mesh_device", "device_params"],
    ),
    pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"]),
    pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops are Blackhole-only"),
    pytest.mark.timeout(0),
]


def _profile_params(fn):
    for mark in reversed(_PROFILE_PARAMS):
        fn = mark(fn)
    return fn


@_profile_params
def test_glm_moe_layer_profile(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, mode, chunk, n_chunks
):
    ctx = _setup(variant, config_only, mesh_device, device_params, num_links, weight_cache_path, mode, chunk, n_chunks)
    mesh_device.enable_program_cache()
    out_pcc = []
    for c in range(n_chunks):
        kv = c * chunk
        x = ttnn.from_torch(
            ctx.x_hosts[c],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ctx.mapper,
        )
        ttnn.synchronize_device(mesh_device)
        signpost(f"prof_chunk{c}")
        out, _ = ctx.block.forward(
            x,
            ctx.rope,
            ctx.kvpe_cache,
            cache_layer_idx=0,
            actual_start=kv,
            actual_end=kv + chunk,
            cache_user_id=0,
            actual_isl=chunk,
            index_kv_cache=ctx.index_cache,
        )
        ttnn.synchronize_device(mesh_device)
        signpost(f"prof_chunk{c}_end")
        out_pcc.append(_output_pcc(ctx, out, c, f"{mode} (KV {kv})"))
        ttnn.deallocate(out)
        # Flush the device-side profiler buffers every chunk so long runs never drop markers.
        ttnn.ReadDeviceProfiler(mesh_device)

    _check_caches(ctx, mesh_device, mode, chunk, n_chunks, out_pcc)
    ctx.block.release_sub_device_managers()


@_profile_params
def test_glm_moe_layer_profile_traced(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, mode, chunk, n_chunks
):
    ctx = _setup(variant, config_only, mesh_device, device_params, num_links, weight_cache_path, mode, chunk, n_chunks)
    mesh_device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def _meta1(val):
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )

    # Persistent input + per-chunk scalars at fixed addresses, refreshed in place before every replay.
    x_dev = ttnn.from_torch(
        ctx.x_hosts[0],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ctx.mapper,
    )
    x_host_tt = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ctx.mapper) for t in ctx.x_hosts
    ]
    metadata = ChunkMetadata(_meta1(0), _meta1(0), _meta1(chunk), None)

    def _fwd():
        out, _ = ctx.block.forward(
            x_dev,
            ctx.rope,
            ctx.kvpe_cache,
            cache_layer_idx=0,
            actual_start=None,
            actual_end=None,  # metadata carries the clamp
            cache_user_id=0,
            actual_isl=chunk,
            index_kv_cache=ctx.index_cache,
            metadata=metadata,
        )
        return out

    controller = SubDeviceTraceController(mesh_device)
    ctx.block.set_trace_controller(controller)

    # Warm pass (chunk 0 input + metadata): compiles the metadata-variant programs and is the region-label
    # template for the replays. Its KV writes land on chunk 0's rows, which replay 0 rewrites.
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_WARM_START")
    warm_out = _fwd()
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_WARM_END")
    ttnn.deallocate(warm_out)
    ttnn.ReadDeviceProfiler(mesh_device)

    controller.begin_capture()
    trace_out = _fwd()
    controller.end_capture()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(
        f"[prof] {mode} captured layer {LAYER}: {controller.num_segments} segments, "
        f"{controller.trace_bytes() / 1024 / 1024:.2f} MB"
    )
    signpost("PERF_TRACE_REPLAYS")

    out_pcc = []
    for c in range(n_chunks):
        kv = c * chunk
        ttnn.copy_host_to_device_tensor(x_host_tt[c], x_dev)
        write_chunk_metadata(
            metadata,
            (0, kv, kv + chunk),
            hf_config=ctx.config,
            mesh_device=mesh_device,
            chunk_size_global=chunk,
            sp_axis=SP_AXIS,
        )
        ttnn.synchronize_device(mesh_device)
        signpost(f"replay_chunk{c}")
        controller.replay()
        ttnn.synchronize_device(mesh_device)
        signpost(f"replay_chunk{c}_end")
        out_pcc.append(_output_pcc(ctx, trace_out, c, f"{mode} traced (KV {kv})"))
        ttnn.ReadDeviceProfiler(mesh_device)

    controller.release()
    ctx.block.set_trace_controller(None)
    _check_caches(ctx, mesh_device, mode, chunk, n_chunks, out_pcc)
    ctx.block.release_sub_device_managers()
