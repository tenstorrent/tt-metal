# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 chunked prefill, batch 4: one user per mesh column.

The chunked transformer of test_glm_prefill_transformer_chunked, with the 8x4 mesh's column axis
carrying four users instead of tensor parallelism (TtPrefillTransformer(batch_axis=1)):

  * attention is SP=8 / TP=1 per column -- replicated norms + MLA weights, per-column KVPE and index-K
    caches, SP-axis rings for the KVPE gather and the fused ring indexer;
  * the embedding, the MoE and the dense FFN stay tensor-parallel with their existing weights, fed by
    an all-to-all that interleaves every user's rows (tt/batch_axis.py) -- the MoE keeps today's four
    dispatch groups, one per column.

Data: the GLM-5.3 55k vLLM trace, replicated onto all four columns, so the four users are identical
and every column is checked against the same golden: per-layer decoder output per chunk (eager), and
after the run the KVPE cache (every layer) and the index-K cache (every full-indexer layer). Floors are
the single-user chunked test's. Chunks are 5120 (11 cover the trace) or 2048 (27 cover 55296 of it).

  test_glm_build_batch_axis_ttnn_cache                -- writes the replicated (``*_repl``) norm + MLA
                                                         tensorbins next to the existing GLM-5.3 cache
  test_glm_prefill_transformer_chunked_batch4         -- accuracy: eager, per-layer + cache PCC
  test_glm_prefill_transformer_chunked_batch4_perf    -- per-chunk wall time, eager or traced (captured
                                                         once, replayed per chunk; one metadata set for all
                                                         users), cache PCC still asserted
  test_glm_prefill_transformer_chunked_batch4_profile -- 7 layers x 2 chunks with signposts, for tracy

A/B knobs (perf sweep only): B4_DISPATCH_FACTOR, B4_OVERLAP_SHARED; B4_PERF_ITERS sets the iteration count. Measured on one galaxy, L10 traced:
factor 2 vs 4 makes no difference; B4_OVERLAP_SHARED=0 is slower AND corrupts the KVPE cache under trace
(0.62 vs 0.988) -- a real bug in that configuration, which is not the default.
"""

import copy
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked import (
    GLM_L1_SMALL_SIZE,
    GLM_TRACE_REGION_SIZE,
    INDEXER_K_PCC_THRESHOLD,
    KV_CACHE_PCC_THRESHOLD,
    LAYER_PCC_THRESHOLD,
    SEQ_CACHE,
    _load_layer_rows,
    _load_metadata_token_ids,
    _ref_layer_slice,
    _resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata, write_chunk_metadata
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, unrotate_cache_layer
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import comp_pcc

SP_AXIS, BATCH_AXIS = 0, 1
# Dispatch capacity per chip = 8 chips x (4 users x chunk/8 rows) x factor. The single-user test runs 8;
# 4 keeps the same absolute headroom per expert as factor 8 at 4x the tokens would, at half the buffer.
# Traced L10 measured no time difference between 2 and 4.
DISPATCH_BUFFER_CAPACITY_FACTOR = 4

_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE,
            l1_small_size=GLM_L1_SMALL_SIZE,
            trace_region_size=GLM_TRACE_REGION_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


def _batch_cache_complete(cache_path: Path, num_layers: int, config, experts_per_chip: int) -> bool:
    return TtPrefillTransformer.check_cache_complete(
        cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=GLM53Config.NUM_DENSE_LAYERS,
        batch_axis=True,
        config=config,
    )


# ---------------------------------------------------------------------------
# Weight provisioning: the replicated norm + MLA tensorbins batch-axis attention loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_layers", [10, 78], ids=["L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM requires Blackhole")
@pytest.mark.timeout(0)
def test_glm_build_batch_axis_ttnn_cache(
    variant, config_only, mesh_device, device_params, num_links, num_layers, weight_cache_path, model_path
):
    """Write ``layer_{i}.{attn_norm,ffn_norm}_repl`` and ``layer_{i}.mla_repl.*`` (+ indexer on full layers)
    into the existing GLM-5.3 8x4 cache directory. The embedding, dense FFN and MoE tensorbins there are
    reused unchanged, so nothing else is written. Layers already complete are skipped (resumable)."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    config = config_only
    sp, cols = list(mesh_device.shape)
    cache_path = weight_cache_path / f"{sp}x{cols}"
    assert cache_path.exists(), f"base GLM-5.3 cache missing at {cache_path}; build it with test_glm_build_ttnn_cache"
    # The shared cache is group-readable for everyone (rw-rw-r--); a 0007 umask would write files only
    # cache-writers can read.
    old_umask = os.umask(0o002)
    # The fast checker snapshots the directory listing; the per-layer skip checks below read that
    # snapshot (taken before anything is written), and the final check re-snapshots.
    init_checker(cache_path)
    try:
        weight_map = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
        for layer_idx in range(num_layers):
            if TtPrefillBlock.check_cache_complete(
                cache_path,
                layer_idx,
                layer_idx < GLM53Config.NUM_DENSE_LAYERS,
                GLM53Config.NUM_ROUTED_EXPERTS // (sp * cols),
                batch_axis=True,
                config=config,
            ):
                logger.info(f"[batch-axis cache] layer {layer_idx}: complete, skipping")
                continue
            prefix = f"model.layers.{layer_idx}."
            shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
            mla_weights = pretrained_mla_weights(
                config, layer=layer_idx, checkpoint_path=[str(Path(model_path) / s) for s in shards]
            )
            norms = load_hf_state_dict_filtered(
                str(model_path), [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."]
            )
            TtPrefillBlock.build_batch_axis_ttnn_cache(
                {
                    "mla_weights": mla_weights,
                    "attn_norm_weight": norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16),
                    "ffn_norm_weight": norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16),
                },
                layer_idx,
                cache_path,
                mesh_device,
                config,
            )
    finally:
        os.umask(old_umask)
    experts_per_chip = GLM53Config.NUM_ROUTED_EXPERTS // (sp * cols)
    assert _batch_cache_complete(cache_path, num_layers, config, experts_per_chip), "batch-axis cache still incomplete"


# ---------------------------------------------------------------------------
# Chunked prefill, batch 4
# ---------------------------------------------------------------------------
# Chunk sizes and how many of them cover the 55k golden (56320 tokens): as many whole chunks as fit, e.g. 11 x
# 5120 exactly, or 27 x 2048 = 55296 (a 28th 2048-chunk would be half padding). The KV cache is a whole number
# of chunks (block-cyclic slabs are chunk-sized), so the 2k cache is 28 x 2048 = 57344. The 2.5k..4.5k sizes
# are the chunk-size sweep; every one keeps chunk / SP tile-aligned (320..576 rows per chip).
CHUNK_SIZES = (5120, 2048, 2560, 3072, 3584, 4096, 4608)
CHUNK_CASES = {c: SEQ_CACHE // c for c in CHUNK_SIZES}


def chunk_id(chunk: int) -> str:
    """Pytest id for a chunk size in units of 1024 tokens: c5k, c2k, c2p5k, ... (no '.', so -k can match it)."""
    whole, half = divmod(chunk, 1024)
    assert half in (0, 512), f"chunk {chunk} is not a multiple of 512"
    return f"c{whole}p5k" if half else f"c{whole}k"


def _seq_cache(chunk: int) -> int:
    return -(-SEQ_CACHE // chunk) * chunk


def run_batch4_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    weight_cache_path,
    *,
    num_layers,
    chunk,
    n_chunks,
    use_trace=False,
    num_iters=1,
    check_layer_pcc=False,
):
    """Batch-4 chunked prefill driver: build the batch-axis transformer once, run `n_chunks` chunks of
    `chunk` tokens for every user `num_iters` times (eager, or captured once and replayed), report the
    per-chunk wall time, then PCC the caches per user against the golden. `check_layer_pcc` adds the
    per-layer decoder-output PCC (eager only: it reads each layer back mid-forward). All four users run
    the same trace, so one metadata set serves them all."""
    assert not (check_layer_pcc and use_trace), "per-layer PCC reads back mid-forward; impossible under capture"
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    topology = per_axis_topology(device_params["fabric_config"])
    sp, num_users = mesh_device.shape[SP_AXIS], mesh_device.shape[BATCH_AXIS]
    assert (sp, num_users) == (8, 4), f"this test targets mesh-8x4, got {list(mesh_device.shape)}"
    assert chunk % (sp * ttnn.TILE_SIZE) == 0, f"chunk {chunk} must give whole tiles per chip"
    chunk_local = chunk // sp
    total_len = n_chunks * chunk
    seq_cache = _seq_cache(chunk)
    assert total_len <= SEQ_CACHE, f"{n_chunks} x {chunk} = {total_len} runs past the {SEQ_CACHE}-token golden"

    # Shared lru_cache'd config: copy before mutating (as the single-user driver does).
    config = copy.deepcopy(config_only)
    config.max_seq_len = seq_cache
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = seq_cache
    emb_dim, kv_lora = config.hidden_size, config.kv_lora_rank

    trace_dir = _resolve_trace_dir(variant)
    layout = variant.prefill_trace_layout
    token_ids = _load_metadata_token_ids(trace_dir, total_len, require_full=True) % config.vocab_size

    cache_path = weight_cache_path / f"{sp}x{num_users}"
    experts_per_chip = GLM53Config.NUM_ROUTED_EXPERTS // (sp * num_users)
    assert _batch_cache_complete(cache_path, num_layers, config, experts_per_chip), (
        f"batch-axis TTNN cache incomplete for {num_layers} layers at {cache_path}; "
        "run test_glm_build_batch_axis_ttnn_cache first"
    )
    logger.info(
        f"[batch4] GLM-5.3 chunked prefill: {num_users} users x {total_len} tokens ({n_chunks} x {chunk}), "
        f"{num_layers} layers, {'traced' if use_trace else 'eager'}, {num_iters} iteration(s); "
        f"dispatch factor {os.environ.get('B4_DISPATCH_FACTOR', DISPATCH_BUFFER_CAPACITY_FACTOR)}, "
        f"shared-expert overlap {os.environ.get('B4_OVERLAP_SHARED', '1')}"
    )

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=chunk,
        max_seq_len=seq_cache,
        # A/B knobs for the perf sweep (env, default = the configuration under test).
        dispatch_buffer_capacity_factor=int(os.environ.get("B4_DISPATCH_FACTOR", DISPATCH_BUFFER_CAPACITY_FACTOR)),
        overlap_shared_expert_with_dispatch=os.environ.get("B4_OVERLAP_SHARED", "1") == "1",
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=BATCH_AXIS,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=cache_path,
        is_chunked=True,
        slot_num=1,
        kv_only_last_layer=True,
        routing_use_l1_small_for_semaphores=True,
        batch_axis=BATCH_AXIS,
    )
    ttnn.synchronize_device(mesh_device)

    # Per-column caches: user c on column c, striped over that column's 8 chips.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
        batch_axis=BATCH_AXIS,
    )
    num_index_slots = full_indexer_rank(config, num_layers)
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_index_slots,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        batch_axis=BATCH_AXIS,
    )

    # Every chip of mesh row r gets row-block r of EVERY user, users in column order (the layout the TP
    # embedding consumes; the transformer de-interleaves right after it). The users are the same trace,
    # so the four blocks are identical.
    token_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None))
    chunk_tokens, chunk_local_pos = [], []
    for c in range(n_chunks):
        positions = rotated_chip_positions(c * chunk, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        rows = token_ids[flat].reshape(sp, 1, chunk_local)
        chunk_tokens.append(rows.repeat(1, 1, num_users))  # [sp, 1, U * chunk_local] = [u0 | u1 | u2 | u3]
        chunk_local_pos.append(flat - c * chunk)

    # Per-layer decoder-output PCC, taken at each block's end through on_layer_hidden (the
    # return_intermediates dict would hold every layer of every user on host at once).
    layer_min = {}  # layer -> min PCC over chunks and users
    state = {"cross_user_max_diff": 0.0}

    def on_layer_hidden(layer_idx, x):
        users = transformer._to_host(x).to(torch.float32)  # [U, chunk, emb], block-cyclic row order
        ref = _ref_layer_slice(trace_dir, layout, layer_idx, state["kv_actual"], state["kv_actual"] + chunk)
        for u in range(num_users):
            natural = torch.empty(chunk, emb_dim, dtype=torch.float32)
            natural[state["local_pos"]] = users[u]
            _, pcc = comp_pcc(ref, natural)
            layer_min[layer_idx] = min(layer_min.get(layer_idx, 1.0), pcc)
            if u == 0:
                logger.info(f"  chunk {state['chunk']} layer {layer_idx} user 0 PCC: {pcc:.6f}")
            else:
                diff = (users[u] - users[0]).abs().max().item()
                state["cross_user_max_diff"] = max(state["cross_user_max_diff"], diff)

    mesh_device.enable_program_cache()

    trace_controller = None
    if use_trace:
        # Captured once, replayed per chunk: the token input lives in a persistent buffer refreshed in
        # place, and the per-chunk scalars (slot, actual_start, actual_end) in 1-element device tensors
        # the metadata ops read on-device. One set serves every user (same lengths for all).
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

        trace_input = ttnn.from_torch(
            chunk_tokens[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=token_mapper,
        )
        trace_metadata = ChunkMetadata(_meta1(0), _meta1(0), _meta1(chunk), None)
        host_tok = [
            ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=token_mapper)
            for t in chunk_tokens
        ]

        def _fwd_meta():
            transformer.forward(
                trace_input,
                kvpe_cache,
                actual_isl=chunk,
                actual_start=None,
                actual_end=None,  # metadata carries the clamp
                cache_user_id=0,
                metadata=trace_metadata,
                index_kv_cache=index_cache,
            )

        trace_controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(trace_controller)
        _fwd_meta()  # compile the metadata-variant programs before recording
        ttnn.synchronize_device(mesh_device)
        trace_controller.begin_capture()
        _fwd_meta()
        trace_controller.end_capture()
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"[batch4] captured {num_layers}-layer forward: {trace_controller.num_segments} segments, "
            f"{trace_controller.trace_bytes() / 1024 / 1024:.2f} MB"
        )
        assert trace_controller.num_segments > 0, "capture recorded nothing to replay"

    iteration_times = []
    for it in range(num_iters):
        times = []
        for c in range(n_chunks):
            kv_actual = c * chunk
            state.update(chunk=c, kv_actual=kv_actual, local_pos=chunk_local_pos[c])
            signpost(f"batch4_iter{it}_chunk{c}")
            if use_trace:
                ttnn.copy_host_to_device_tensor(host_tok[c], trace_input)
                write_chunk_metadata(
                    trace_metadata,
                    (0, kv_actual, kv_actual + chunk),
                    hf_config=config,
                    mesh_device=mesh_device,
                    chunk_size_global=chunk,
                    sp_axis=SP_AXIS,
                )
                t0 = time.time()
                trace_controller.replay()
                ttnn.synchronize_device(mesh_device)
                times.append(time.time() - t0)
            else:
                tt_tokens = ttnn.from_torch(
                    chunk_tokens[c],
                    device=mesh_device,
                    dtype=ttnn.uint32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=token_mapper,
                )
                t0 = time.time()
                transformer.forward(
                    tt_tokens,
                    kvpe_cache,
                    actual_isl=chunk,
                    actual_start=kv_actual,
                    actual_end=kv_actual + chunk,
                    cache_user_id=0,
                    index_kv_cache=index_cache,
                    on_layer_hidden=on_layer_hidden if check_layer_pcc else None,
                )
                ttnn.synchronize_device(mesh_device)
                times.append(time.time() - t0)
                ttnn.deallocate(tt_tokens)
            logger.info(f"[batch4] [chunk timing] iter={it} chunk={c} {times[-1] * 1000:.2f} ms")
        iteration_times.append(times)
        logger.info(f"[batch4] iter {it} total {sum(times):.3f} s over {n_chunks} chunks")

    # Timing summary: iteration 0 carries compile / first-touch cost when num_iters > 1.
    measured = iteration_times[1:] if num_iters > 1 else iteration_times
    per_chunk = [statistics.median(t[c] for t in measured) for c in range(n_chunks)]
    total_s = sum(per_chunk)
    tokens = num_users * total_len
    logger.info(
        f"[batch4] PERF {'traced' if use_trace else 'eager'} L{num_layers} chunk={chunk} x{n_chunks}: "
        f"total {total_s:.3f} s, mean {total_s / n_chunks * 1000:.1f} ms/chunk, "
        f"first {per_chunk[0] * 1000:.1f} ms, last {per_chunk[-1] * 1000:.1f} ms, "
        f"{tokens / total_s:.0f} tok/s ({num_users} users x {total_len})"
    )
    logger.info(f"[batch4] PERF per-chunk ms: {[round(t * 1000, 1) for t in per_chunk]}")
    if len(measured) > 1:
        spread = [round((max(t[c] for t in measured) - min(t[c] for t in measured)) * 1000, 1) for c in range(n_chunks)]
        logger.info(f"[batch4] PERF per-chunk max-min ms over {len(measured)} iters: {spread}")

    failures = []
    if check_layer_pcc:
        # kv_only_last_layer leaves the last layer with no output; the KVPE PCC below covers it.
        assert len(layer_min) == num_layers - 1, f"compared {len(layer_min)} layers, expected {num_layers - 1}"
        for i in sorted(layer_min):
            logger.info(f"[batch4] layer {i} min PCC over chunks and users: {layer_min[i]:.6f}")
        worst = min(layer_min.values())
        logger.info(
            f"[batch4] per-layer min PCC {worst:.6f}; users vs user 0 max|diff| {state['cross_user_max_diff']:.3e}"
        )
        if worst < LAYER_PCC_THRESHOLD:
            failures.append(f"per-layer min PCC {worst:.6f} < {LAYER_PCC_THRESHOLD}")

    # KVPE cache, every layer, every user. Per chip [L, 1, T/sp, 576] -> host [U*L, 1, T, 576] (user u owns
    # slots [u*L, (u+1)*L)); bf16 on host, one slot converted at a time.
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape)
    p = blockcyclic_positions(sp, chunk, seq_cache)
    kv_host = ttnn.to_torch(kvpe_cache.storage, mesh_composer=composer)
    kv_min = 1.0
    for i in range(num_layers):
        g = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, total_len)
        per_user = []
        for u in range(num_users):
            dev = unrotate_cache_layer(kv_host[u * num_layers + i, 0].float(), p, total_len)
            per_user.append(min(cache_half_pccs(g, dev, kv_lora, pe_interleave=True)))
        kv_min = min(kv_min, *per_user)
        logger.info(f"[batch4] KVPE layer {i} PCC per user: {[round(v, 6) for v in per_user]}")
    del kv_host
    logger.info(f"[batch4] KVPE min PCC over layers and users: {kv_min:.6f}")
    if kv_min < KV_CACHE_PCC_THRESHOLD:
        failures.append(f"KVPE min PCC {kv_min:.6f} < {KV_CACHE_PCC_THRESHOLD}")

    # Index-K cache, every full-indexer layer with a golden. Stored in the Hadamard basis (H symmetric and
    # orthonormal, so one more multiply returns the golden's basis).
    idx_host = ttnn.to_torch(index_cache, mesh_composer=composer).to(torch.float32)  # [U*slots, 1, T, D]
    hadamard = normalized_hadamard_matrix(config.index_head_dim).float()
    idx_layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    idx_min = 1.0
    for i in idx_layers:
        g = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, total_len)
        slot = full_indexer_rank(config, i)
        per_user = []
        for u in range(num_users):
            dev = unrotate_cache_layer(idx_host[u * num_index_slots + slot, 0], p, total_len) @ hadamard
            per_user.append(min(cache_half_pccs(g, dev, config.index_head_dim // 2, pe_interleave=False)))
        idx_min = min(idx_min, *per_user)
        logger.info(f"[batch4] index-K layer {i} PCC per user: {[round(v, 6) for v in per_user]}")
    logger.info(f"[batch4] index-K min PCC over {len(idx_layers)} layers and users: {idx_min:.6f}")
    if idx_layers and idx_min < INDEXER_K_PCC_THRESHOLD:
        failures.append(f"index-K min PCC {idx_min:.6f} < {INDEXER_K_PCC_THRESHOLD}")

    # Release the captured trace and the sub-device managers that own its buffers before the mesh closes
    # (otherwise teardown segfaults freeing them after the allocator, as the single-user driver notes).
    if trace_controller is not None:
        trace_controller.release()
        transformer.set_trace_controller(None)
    transformer.release_sub_device_managers()
    assert not failures, "; ".join(failures)
    logger.success(f"[batch4] GLM-5.3 L{num_layers} {n_chunks} x {chunk}: all {num_users} users pass")


_CHUNK_PARAMS = [pytest.param(c, n, id=f"{chunk_id(c)}-chunks{n}") for c, n in CHUNK_CASES.items()]


@pytest.mark.parametrize("chunk, n_chunks", _CHUNK_PARAMS)
@pytest.mark.parametrize("num_layers", [10, 78], ids=["L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked_batch4(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, num_layers, chunk, n_chunks
):
    """Accuracy: eager, per-layer decoder-output PCC per chunk and user, then cache PCC."""
    run_batch4_chunked(
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        weight_cache_path,
        num_layers=num_layers,
        chunk=chunk,
        n_chunks=n_chunks,
        check_layer_pcc=True,
    )


# notrace/traced, not trace: "notrace" CONTAINS "trace", so `-k trace` would select both modes.
@pytest.mark.parametrize("use_trace", [False, True], ids=["notrace", "traced"])
@pytest.mark.parametrize("chunk, n_chunks", _CHUNK_PARAMS)
@pytest.mark.parametrize("num_layers", [10, 78], ids=["L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked_batch4_perf(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    weight_cache_path,
    num_layers,
    chunk,
    n_chunks,
    use_trace,
):
    """Per-chunk wall time over a full-context prefill (iteration 1 of 2; iteration 0 warms up), traced
    or eager, with the cache PCCs still asserted so a perf number is never reported for a wrong run."""
    run_batch4_chunked(
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        weight_cache_path,
        num_layers=num_layers,
        chunk=chunk,
        n_chunks=n_chunks,
        use_trace=use_trace,
        # Iteration 0 warms up; the reported per-chunk time is the median over the rest (B4_PERF_ITERS, default 2).
        num_iters=int(os.environ.get("B4_PERF_ITERS", "2")),
    )


@pytest.mark.parametrize("chunk", [5120, 2048], ids=["c5k", "c2k"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked_batch4_profile(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, chunk
):
    """Per-op device-time source for tracy: 7 layers (3 dense, a full-indexer MoE at 6, shared-indexer MoE
    at 3-5) over 2 eager chunks, each preceded by a `batch4_iter0_chunk{c}` signpost; chunk 1 is the warm
    one with a populated KV prefix."""
    run_batch4_chunked(
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        weight_cache_path,
        num_layers=7,
        chunk=chunk,
        n_chunks=2,
    )
    ttnn.ReadDeviceProfiler(mesh_device)
