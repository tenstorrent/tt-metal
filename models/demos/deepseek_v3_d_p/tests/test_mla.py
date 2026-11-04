# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MLA device tests: chunked prefill against the CPU torch reference / a recorded GPU trace, plus the
chunked perf gate. ``run_mla_inference`` is the shared single-shot forward helper, also used by
tests/test_kv_cache_table.py and tests/sparse_mla/test_sparse_mla.py.
"""

import functools
import math
import os
import re
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc, hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tests.reference_runners import run_reference_mla
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import (
    ChunkMetadata,
    RotarySetup,
    refresh_llama4_scale,
    write_chunk_metadata,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.chunked_prefill_utils import (
    cpu_mla_reference,
    load_trace,
    partition_iters,
    resolve_traces,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural, preload_cache, tp_stripe_major_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mla_inference(
    config,
    weights,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    tp_axis,
    is_balanced,
    topology,
    tt_kvpe_cache,
    return_indices=False,
    inject_indices=None,
    is_chunked=False,
    active_seq_len=None,
    actual_start=None,
):
    """
    Utility function to run MLA inference without host comparison.

    Args:
        config: Model configuration
        weights: Model weights dictionary
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        tp_axis: Tensor parallel axis
        is_balanced: Whether to use balanced chunk ordering
        topology: Topology (Linear or Ring)
        tt_kvpe_cache: Initialized KVPE cache on device
        is_chunked / active_seq_len / actual_start: prefill mode, passed straight to ttMLA and
            forward(). No chunk loop here -- one forward over the whole sequence -- so only two
            triples fit: single-shot (False/None/None) or one chunk (True/seq_len/0). Sparse must chunk.

    Returns:
        Tuple of (tt_output, hidden_states, chunk_order, shard_dims)
    """
    # Create TT MLA
    logger.info("Creating TT MLA...")

    # The mode is the caller's, not derived: dense chunked prefill is real (test_mla_chunked_prefill,
    # tt_prefill_runtime.py), so sparsity cannot decide it -- it only decides what a sparse caller may
    # declare, since ttMLA binds _apply_rope_padded + _sparse_chunked_attn from _has_indexer alone.
    has_indexer = resolve_has_indexer(config)
    assert not has_indexer or is_chunked, (
        "sparse (DSA) MLA has no single-shot path -- it is block-cyclic by construction. Pass "
        "is_chunked=True: this helper's single forward is then declared as one full-length chunk at "
        "offset 0, which is what actually executes."
    )

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
        is_chunked=is_chunked,
        active_seq_len=active_seq_len,
        layer_num=1,
        sparse_kv_cache_format=tt_kvpe_cache.format,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    index_kv_cache = None
    if has_indexer:
        rope_tensors = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=seq_len)
        # Layer-slot count mirrors the serving adapter: the indexer strides the folded user-major cache by
        # num_full_indexer_layers (GLM-5.3 cross-layer reuse), so the cache must carry that many slots for
        # update_padded_kv_cache's cache_batch % num_layers check. Falls back to 1 (no indexer_types).
        index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
            num_users=1,
            dtype=ttnn.bfloat8_b,
            tp_axis=tp_axis,
        )
    else:
        rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Verify TT MLA exists
    assert mla_tt is not None, "TT MLA should exist"

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor (generate in float32, then convert to bfloat16)
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    # Reorder hidden_states for balanced ring attention
    sp_factor = mesh_shape[sp_axis]
    chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]
    if is_balanced:
        tt_input = reorder_tensor_chunks(tt_input, chunk_order, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    # GLM-5.3 indexer reuse (return_indices / inject_indices): capture this layer's top-k selection, or
    # feed a prior layer's to skip the indexer. Defaults leave the forward unchanged.
    mla_out = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
        actual_start=actual_start,
        indexer_indices=inject_indices,
        return_indexer_indices=return_indices,
        index_kv_cache=index_kv_cache,
    )
    indices = None
    if return_indices:
        tt_output, indices = mla_out
    else:
        tt_output = mla_out

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    if return_indices:
        return tt_output, hidden_states, chunk_order, shard_dims, indices
    return tt_output, hidden_states, chunk_order, shard_dims


def run_model(
    variant,
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    skip_host_comparison,
    scale_down_sl,
    is_balanced,
    is_ci_env,
    is_ci_v2_env,
    device_params,
):
    if use_pretrained and not variant.supports_pretrained:
        pytest.skip(f"{variant.name!r}: pretrained weights not available")

    weight_type = "Pretrained" if use_pretrained else "Random"
    logger.info("=" * 80)
    logger.info(f"Test: Reference vs TT Comparison ({weight_type} Weights, variant={variant.name})")
    logger.info("=" * 80)

    # Conditionally load fixtures - only load what we need!
    if use_pretrained:
        config, weights = request.getfixturevalue("pretrained_mla_layer_weights")
    else:
        config, weights = request.getfixturevalue("random_weights")

    topology = per_axis_topology(device_params["fabric_config"])

    sp_axis = 0
    tp_axis = 1

    mesh_shape = list(mesh_device.shape)

    # 640 tokens on every chip; the global length follows the mesh. max_sl keeps the literal seq_len.
    if scale_down_sl:
        seq_len = PREFILL_CHUNK_TOKENS_PER_CHIP * mesh_shape[sp_axis]

    # temp hack
    config.max_seq_len = seq_len

    # Create reference MLA
    if use_pretrained:
        logger.info("Creating reference MLA with pretrained weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )
    else:
        logger.info("Creating reference MLA with random weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )

    # Verify reference MLA exists
    assert mla_ref is not None, "Reference MLA should exist"

    # Test forward pass comparison
    logger.info("=" * 80)
    logger.info(f"Testing forward pass comparison (seq_len={seq_len})")
    logger.info("=" * 80)

    # Initialize KVPE cache
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Run MLA inference using utility function
    tt_output, hidden_states, chunk_order, shard_dims = run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    batch_size = 1

    # Host comparison: Run reference forward pass if needed
    if skip_host_comparison == False:
        # Check for cached reference results to avoid expensive host attention computation
        env = variant.mla_ref_cache_env or "DEEPSEEK_V3_MLA_REF_CACHE"
        cache_dir = Path(os.environ.get(env, f"/tmp/{variant.name}_mla_ref_cache"))
        cache_path = cache_dir / f"{weight_type.lower()}_seq{seq_len}.pt"

        if cache_path.exists():
            logger.info(f"Loading cached reference results from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            ref_output = cached["ref_output"]
            ref_kvpe = cached["ref_kvpe"]
            logger.info(f"✓ Loaded cached reference results")
            logger.info(f"  Output shape: {ref_output.shape}")
        else:
            assert not (
                (is_ci_env or is_ci_v2_env) and not scale_down_sl
            ), "We should not execute CPU computation in the CI for max sl, output cache is missing"

            # Create position IDs
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

            # Run reference forward pass with cache to capture KVPE
            # Uses F.scaled_dot_product_attention with is_causal=True (no explicit mask needed)
            logger.info("Running reference CPU forward pass...")
            mla_ref = mla_ref.eval().to(torch.bfloat16)
            ref_cache = DynamicCache()
            with torch.no_grad():
                ref_output, _, ref_cache = mla_ref(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=ref_cache,
                    use_cache=True,
                )

            ref_kvpe = hf_cache_layer_kv(ref_cache, 0)[0]  # layer 0

            if not (is_ci_env or is_ci_v2_env):
                # Save to cache for future runs
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
                logger.info(f"✓ Saved reference results to {cache_path}")

            logger.info(f"✓ Reference forward pass complete")
            logger.info(f"  Input shape:  {hidden_states.shape}")
            logger.info(f"  Output shape: {ref_output.shape}")
            logger.info(f"  Output dtype: {ref_output.dtype}")
            logger.info(f"  Output mean:  {ref_output.mean().item():.4f}")
            logger.info(f"  Output std:   {ref_output.std().item():.4f}")

        # Compare TT output with reference output
        tt_output_cpu = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)

        if is_balanced:
            tt_output_cpu = reverse_reorder_tensor_chunks(tt_output_cpu, chunk_order, seq_dim=2)

        _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, 0.98)
        logger.info(f"Output PCC is {pcc_message}")

        # Validate KVPE cache contents
        # Reference KVPE: [batch, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        # ref_kvpe is already available (loaded from cache or computed above)

        # Read back KVPE cache from device
        # Cache is replicated across TP, so concat TP replicas on dim 1 (unused) and discard extras
        tt_kvpe_cache_torch = ttnn.to_torch(
            tt_kvpe_cache.storage,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        tt_kvpe_cache_torch = tt_kvpe_cache_torch[:1, :1, :, :]

        logger.info("Starting synchronize call")
        ttnn.synchronize_device(mesh_device)
        logger.info("Synchronize call ended")

        logger.debug("  Distributed synchronization started")
        ttnn.distributed_context_barrier()
        logger.debug("✓ Distributed synchronization completed")

        if is_balanced:
            tt_kvpe_cache_torch = reverse_reorder_tensor_chunks(tt_kvpe_cache_torch, chunk_order, seq_dim=2)

        # Check PCC separately for KV (latent) and PE (rope) parts
        kv_lora_rank = config.kv_lora_rank
        _, kv_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, :kv_lora_rank], tt_kvpe_cache_torch[:, :, :, :kv_lora_rank], 0.99
        )
        logger.info(f"KVPE cache KV part PCC is {kv_pcc_message}")
        _, pe_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, kv_lora_rank:], tt_kvpe_cache_torch[:, :, :, kv_lora_rank:], 0.99
        )
        logger.info(f"KVPE cache PE part PCC is {pe_pcc_message}")

        # MLA reference check. Returns None when the variant has no reference.
        # Only run reference for shorter sequence lengths so we don't go OOM on host.
        if seq_len <= 5 * 1024:
            position_ids_ref = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            logger.info(f"Running MLA reference (model={variant.name})")
            ref_out = run_reference_mla(
                variant,
                config=config,
                weights=weights,
                hidden_states=hidden_states,
                position_ids=position_ids_ref,
            )
            if ref_out is not None:
                _, ref_pcc_message = assert_with_pcc(ref_out.unsqueeze(0), tt_output_cpu, variant.mla_pcc_threshold)
                logger.info(f"[reference_output] PCC: {ref_pcc_message}")
                del ref_out
        else:
            logger.info(f"Skipping MLA reference comparison for seq_len={seq_len}")
    else:
        logger.info("Starting synchronize call")
        ttnn.synchronize_device(mesh_device)
        logger.info("Synchronize call ended")

        logger.debug("  Distributed synchronization started")
        ttnn.distributed_context_barrier()
        logger.debug("✓ Distributed synchronization completed")

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")


# sp x tp -- Mistral Small 4 bringup. Modelled on the DeepSeek single-shot MLA test (removed upstream in #55619) and deliberately narrowed;
# every axis that test sweeps is pinned here to ONE value unless it changes the number being measured.
#   mesh_device        (8, 4) only. Production SP-axis shape for this box; a (2, 4) PCC is not a
#                      production-shape result, and the TTNN weight cache is keyed on device count.
#   device_params      fabric2d only. FABRIC_1D is not in CI_ALLOWED_FABRICS for BH galaxy (8,4), so
#                      pinning it skips the row; TorusXY needs a certified cabling descriptor.
#   use_pretrained     random only. This row measures the MLA math, which the random-weight path
#                      exercises identically; the pretrained checkpoint is covered by the chunked
#                      rows below and by the transformer row.
#   seq_len            5k and 25k, as Kimi. 25k is the row that crosses position 8192, where the
#                      llama4 query temperature starts to apply (scale 1.139 at 25600, exactly 1.0
#                      at 5120), so it is the only case here that exercises it.
#   skip_host_comparison  check_pcc only. skip_check does the device work and computes NO PCC; the
#                      gate for this step IS the PCC, so that case would be a silent zero-signal run.
#   is_balanced        sequential only.
# scale_down_sl stays swept (2 cases per seq_len): it is the one axis that changes the seq_len
# actually run (max_sl 5120/25600 vs scaled_sl (sl//32)*8 = 1280/6400), which exercises different
# padding/rotation paths. Only the max_sl references are staged; the scaled ones recompute.
#
# NOT WIRED ON PURPOSE: reference_attention_cls. run_reference_mla returns None for a variant with
# no reference (reference_runners.py:63), so the [reference_output] PCC is skipped and the three
# gate PCCs below still come from create_mla_reference -- the vendored DeepSeek MLA, which is
# variant-independent. Whether that vendored reference is the right truth for Mistral is a separate
# question and is deliberately NOT settled by this test.
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        },
    ],
    ids=["fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"])
@pytest.mark.parametrize("scale_down_sl", [False, True], ids=["max_sl", "scaled_sl"])
@pytest.mark.parametrize("seq_len", [PREFILL_CHUNK_TOKENS, 25 * 1024], ids=["seq5k", "seq25k"])
@pytest.mark.parametrize("skip_host_comparison", [False], ids=["check_pcc"])
@pytest.mark.parametrize("is_balanced", [False], ids=["sequential"])
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bringup targets the Blackhole galaxy")
@pytest.mark.timeout(0)
def test_mistral_small4_mla(
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    skip_host_comparison,
    scale_down_sl,
    is_balanced,
    is_ci_env,
    is_ci_v2_env,
    device_params,
    variant,
):
    run_model(
        variant,
        use_pretrained,
        request,
        mesh_device,
        seq_len,
        skip_host_comparison,
        scale_down_sl,
        is_balanced,
        is_ci_env,
        is_ci_v2_env,
        device_params,
    )


# ---------------------------------------------------------------------------------------------------
# Unified chunked-prefill driver. One loop (preload -> N iters of write+rope+SDPA -> compare)
# parametrized by where the prefix/reference come from. See test_mla_chunked_prefill below.
# ---------------------------------------------------------------------------------------------------
# reference='trace' takes its trace dirs from variant.mla_trace_defaults. MLA_CHUNKED_TRACE_PATH
# overrides that with ONE specific trace dir (the leaf holding mla_io/ + kv_cache/), shared across all
# users -- for a trace that is not the variant's registered one.
MLA_CHUNKED_TRACE_PATH = os.environ.get("MLA_CHUNKED_TRACE_PATH")

# Per-iteration VALID token counts for the rotation/padding edge cases, tuned for the TARGET 8x4 mesh
# (sp=8, chunk_local=640, chunk=5120). Each cumulative kv_actual lands on a distinct rotation edge:
# which chip the boundary falls on (0..7), chip-aligned vs mid-chip straddle (offset != 0), single vs
# multi-slab, and how much of the chunk is pad. All values are tile-aligned (multiple of 32).
ROTATED_VALID_LISTS = [
    [640, 5120],  # aligned_min: iter0 = 1 chip valid (7 chips pad), then chip-1 rotated full
    [672, 5120],  # midchip_straddle: frontier 1 tile into chip 1, then rotated with offset=32 straddle
    [4480, 5120],  # lastchip: iter0 = 7 chips, rotation at the LAST chip (chip 7)
    [1280, 1920, 5120],  # rot_partial: iter1 is rotated AND partial (3-chip valid, 5-chip pad)
    [5120, 1280, 5120],  # multislab: rotation in slab 1 (multi-slab), partial then full
    [5120, 5120],  # allfull: sanity, two full chunks at slab boundaries (aligned, no rotation)
]
ROTATED_VALID_IDS = ["aligned_min", "midchip_straddle", "lastchip", "rot_partial", "multislab", "allfull"]

# Determinism gate, same contract as test_prefill_block / test_prefill_transformer: repeats must be
# bit-identical, so the threshold is exactly 1.0. Rep 0 is the baseline, hence >= 2.
DETERMINISM_PCC_THRESHOLD = 1.0
DETERMINISM_REPS = 3

# Realtime ("lightweight") profiler perf gate: in-process device program records, so no Tracy
# subprocess, no signposts and no ops-CSV re-parse -- it runs on the plain build. Not comparable to
# the Tracy path's number -- Tracy averages collectives across chips, this takes the max.
# Kimi-K3 50k + 5k, TorusXY Galaxy 8x4 (DDR 14000), +/-5%.
K3_CHUNKED_RT_PERF_NS = 5_830_000
K3_CHUNKED_RT_PERF_MARGIN = 0.05


def _op_label(kernel_sources):
    """Op directories of a program's kernels, e.g. 'experimental/minimal_matmul'; a fused program lists each
    (the ring SDPAs carry their all-gather kernels)."""
    ops = set()
    for source in kernel_sources:
        match = re.search(r"operations/(.+?)/(?:device/)?kernels/", source)
        if match:
            ops.add(match.group(1).removesuffix("/device"))
    return " + ".join(sorted(ops)) if ops else "?"


def _rt_profile_forward_ns(mesh_device, run_fn):
    """Profile one region; return (result, total_ns) where each program contributes its MAX duration
    across chips (slowest chip gates that program) -- the sparse-MLA/PR #49840 convention. Logs the
    per-op breakdown."""
    result, records = profile_realtime_program(mesh_device, run_fn, collect_all=True)
    per_program = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:  # 0 is the profiler's own sentinel
            continue
        duration_ns, label = per_program.get(runtime_id, (0.0, _op_label(record["kernel_sources"])))
        per_program[runtime_id] = (max(duration_ns, float(record["duration_ns"])), label)
    per_op = {}
    for duration_ns, label in per_program.values():
        count, total = per_op.get(label, (0, 0.0))
        per_op[label] = (count + 1, total + duration_ns)
    for label, (count, total) in sorted(per_op.items(), key=lambda item: -item[1][1]):
        logger.info(f"    {label}: {total / 1e6:.3f} ms over {count} program(s)")
    return result, sum(duration_ns for duration_ns, _ in per_program.values())


def _meta_scalar(mesh_device, value):
    """A replicated 1-element uint32 [1, 1, 1, 1] DRAM tensor: one field of the chunk metadata."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _new_chunk_metadata(mesh_device, chunk, scale_buf):
    """Persistent chunk metadata (slot 0, start 0, end chunk) that write_chunk_metadata rewrites in place per step."""
    return ChunkMetadata(
        _meta_scalar(mesh_device, 0), _meta_scalar(mesh_device, 0), _meta_scalar(mesh_device, chunk), scale_buf
    )


def _new_layered_kv_cache(config, mesh_device, seq_len, num_layers, num_users, sp_axis, tp_axis, tp_sharded=True):
    """A num_users x num_layers BFP8 KVPE cache over the whole mesh, TP-deduped when tp_sharded."""
    return init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
        tp_axis=tp_axis if tp_sharded else None,
    )


def _natural_cache(mesh_device, cache, tp_sharded, chunk, seq_len):
    """Every (user, layer) slot of the cache, un-striped to natural token order."""
    flat, stripes = gather_cache_natural(cache.storage, mesh_device, tp_sharded)
    nat = torch.empty_like(flat)
    nat[:, blockcyclic_positions(stripes, chunk, seq_len)] = flat
    return nat


def _run_chunked_prefill(
    request,
    mesh_device,
    *,
    iters_isl,
    reference="cpu",
    chunk_size_global=5120,
    prefill_len=0,
    num_users=1,
    use_pretrained=False,
    topology=None,
    use_metadata_tensor=False,
    determinism_check=False,
    profile=False,
    tight_cache=False,
    tp_shard_kv=False,
    return_outputs=False,
    return_pccs=False,
    mla_kwargs=None,
):
    """Unified chunked-prefill scenario, decoupled from the reference.

    `reference` selects how inputs + ground truth are produced -- independent of prefill_len / env:
      * "cpu"   -> synthetic inputs + torch MLA reference (k_pe in Meta basis). Partial-chunk iters
                   (rotation) allowed; any prefix is preloaded from the CPU reference KV.
      * "trace" -> GPU-trace inputs + reference (k_pe re-interleaved for a roped trace, compared
                    directly under NoPE). TRACE ONLY: dirs come from variant.mla_trace_defaults (or
                    MLA_CHUNKED_TRACE_PATH); supports partial iters.
      * None    -> no reference (functional/perf): random inputs + random prefix, finite-output check.
    Multi-user partitions iters_isl across users (last gets the remainder); each user is independent in
    its own cache slot, so cross-user contamination surfaces as a per-user output PCC drop.
    mla_kwargs: extra ttMLA arguments, e.g. expanded_k.
    """
    assert reference in ("cpu", "trace", None), f"reference must be 'cpu'|'trace'|None, got {reference!r}"
    # Mutually exclusive like determinism_check vs pcc_validation in test_prefill_block: a reference
    # comparison measures accuracy, the repeats measure device determinism. Pick one.
    if determinism_check and reference is not None:
        pytest.skip("determinism_check needs reference=None (func) -- accuracy is the cpu/trace path's job")
    # Same exclusion, same reason: a reference run pays a CPU torch pass this measurement does not want.
    if profile and reference is not None:
        pytest.skip("profile needs reference=None (func) -- accuracy is the cpu/trace path's job")
    if profile and not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler inactive (needs Blackhole, WORKER dispatch, fabric-tensix DM off)")
    sp_axis, tp_axis = 0, 1
    if topology is None:
        topology = per_axis_topology()
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert not tp_shard_kv or tp > 1, f"tp_shard_kv needs a TP axis to dedup across, got tp={tp}"
    tile = ttnn.TILE_SIZE
    chunk_local = chunk_size_global // sp

    assert chunk_size_global % (tile * sp) == 0, f"chunk_size_global {chunk_size_global} % (TILE*sp={tile * sp}) != 0"
    for v in iters_isl:
        assert 0 < v <= chunk_size_global and v % tile == 0, f"iter isl {v}: tile-aligned and <= {chunk_size_global}"
    assert prefill_len % tile == 0, f"prefill_len {prefill_len} must be tile-aligned"

    use_trace = reference == "trace"
    traces = None
    if use_trace:
        trace_variant = request.getfixturevalue("variant")
        trace_paths = [MLA_CHUNKED_TRACE_PATH] if MLA_CHUNKED_TRACE_PATH else trace_variant.mla_trace_defaults
        assert trace_paths, (
            f"reference='trace' is not supported for variant '{trace_variant.name}': no golden MLA "
            "trace was ever recorded for it (mla_trace_defaults is empty). Use reference='cpu' or "
            "reference='func', or point MLA_CHUNKED_TRACE_PATH at a trace."
        )
        traces = resolve_traces(trace_paths, num_users)
        # The trace is a DENSE token sequence; iters_isl just chunks it variably. Partial iters pad
        # the device's fixed-width chunk (masked by causality) -- they are not pad in the sequence --
        # so any iters_isl / prefill works exactly like the CPU ref. The only trace constraint is
        # total_len <= trace length, asserted per-user below.
        use_pretrained = True  # the GPU trace was generated with the real checkpoint

    groups = partition_iters(iters_isl, num_users)

    # Cache holds the max (kv_actual + chunk) window across all users/iters, slab-aligned, >= 2 slabs.
    max_window = chunk_size_global * 2
    max_real_end = 0
    for g in groups:
        ka = prefill_len
        for v in g:
            max_window = max(max_window, ka + chunk_size_global)
            ka += v
            max_real_end = max(max_real_end, ka)
    if tight_cache:
        # Cache sized to the REAL tokens, not the padded window, so a late iteration's window runs past
        # the end of it -- the server's tail chunk. update_padded_kv_cache clamps the write and the SDPA's
        # logical_n is capped at the cache.
        write_end = -(-max_real_end // tile) * tile
        max_window = max(chunk_size_global * 2, write_end)
    seq_len_cache = ((max_window + chunk_size_global - 1) // chunk_size_global) * chunk_size_global
    if tight_cache:
        overruns = [
            ka
            for g in groups
            for ka in [prefill_len + sum(g[:i]) for i in range(len(g))]
            if ka + chunk_size_global > seq_len_cache
        ]
        assert overruns, (
            f"tight_cache scenario does not actually overrun: cache {seq_len_cache}, chunk "
            f"{chunk_size_global}, real end {max_real_end} -- no iteration's padded window passes the cache"
        )
        logger.info(
            f"tight_cache: cache {seq_len_cache} for {max_real_end} real tokens; "
            f"{len(overruns)} iteration(s) pad past it, worst {max(overruns) + chunk_size_global}"
        )

    if use_pretrained:
        # MLA-only fixture: this driver uses nothing but the attention weights, and the full-layer one
        # cannot load Kimi-K3's MXFP4 MoE side.
        config, weights = request.getfixturevalue("pretrained_mla_layer_weights")
    else:
        config, weights = request.getfixturevalue("random_weights")
    config.max_seq_len = seq_len_cache
    kvpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
    hidden_size = config.hidden_size
    # A roped GPU trace stores k_pe HF half-split while the device cache is Meta interleaved; under NoPE
    # (Kimi-K3) neither side rotates, so there is no basis difference to correct.
    trace_pe_interleave = use_trace and not getattr(config, "mla_use_nope", False)

    logger.info(
        f"chunked prefill: mesh={tuple(mesh_device.shape)} chunk={chunk_size_global} prefill={prefill_len} "
        f"iters={iters_isl} users={num_users} reference={reference} "
        f"weights={'pretrained' if use_pretrained else 'random'} seq_len_cache={seq_len_cache}"
    )

    # ---- per-user inputs + references. Each source provides hidden + (ref_out, ref_kvpe); the prior
    #      prefix KV is carved from that same reference (random for the functional, ref-less mode). ----
    users = []  # each: {group, total_len, hidden, ref_out|None, kv_prior|None, kv_post|None}
    for u in range(num_users):
        g = groups[u]
        total_len = prefill_len + sum(g)
        if reference == "trace":
            mi, mo, kv = load_trace(traces[u])
            assert total_len <= mi.shape[0], f"user {u}: prefill+iters {total_len} > trace len {mi.shape[0]}"
            hidden, ref_out, ref_kvpe = mi[:total_len], mo[:total_len], kv[:total_len]
        elif reference == "cpu":
            torch.manual_seed(42 + u)
            hidden = torch.randn(total_len, hidden_size, dtype=torch.bfloat16)
            ref_out, ref_kvpe = cpu_mla_reference(config, weights, hidden)
        else:  # None -> functional / perf, no reference
            torch.manual_seed(100 + u)
            # Random rows only past the prefix: a 512k prefix would otherwise draw ~3.8G values on the host (about two
            # minutes). Gathered positions are all >= kv_actual >= prefill_len, so the zero prefix rows are never read.
            hidden = torch.zeros(total_len, hidden_size, dtype=torch.bfloat16)
            hidden[prefill_len:] = torch.randn(total_len - prefill_len, hidden_size, dtype=torch.bfloat16)
            ref_out, ref_kvpe = None, None

        if prefill_len == 0:
            kv_prior = None
        elif ref_kvpe is not None:
            kv_prior = ref_kvpe[:prefill_len]  # preload the reference's prior KV (cpu or trace)
        else:
            kv_prior = torch.randn(prefill_len, kvpe_dim, dtype=torch.bfloat16)  # functional: random prefix
        users.append(
            dict(group=g, total_len=total_len, hidden=hidden, ref_out=ref_out, kv_prior=kv_prior, kv_post=ref_kvpe)
        )

    # ---- device setup ----
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        active_seq_len=chunk_size_global,
        slot_num=num_users,
        layer_num=1,
        # None, not False: False would turn off a sparse variant's dedup.
        tp_shard_kv=True if tp_shard_kv else None,
        **(mla_kwargs or {}),
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(
        # tail_slack: an overrunning iteration still ropes its whole padded slab, so the table needs one
        # chunk of headroom (as the production transformer passes).
        cache_seq_len_global=seq_len_cache,
        chunk_size_global=chunk_size_global,
        tail_slack=tight_cache,
    )
    # Persistent, refreshed per chunk; None for every variant without a query temperature.
    llama4_scale_buf = rope_setup.make_llama4_scale_buffer(chunk_size_global)
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
        tp_axis=tp_axis if tp_shard_kv else None,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2
    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2
    if tp_shard_kv:
        # A mapper can't split dim 2 over both axes, so TP rides dim 1.
        cache_shard_dims[tp_axis] = 1

    # ---- preload the prior prefix (trace or random) into each slot, block-cyclic ----
    if prefill_len > 0:
        logger.info(f"Preloading {prefill_len}-token prefix into {num_users} slot(s) (block-cyclic host->device)...")
        cache_host = torch.zeros(
            num_users,
            tp if tp_shard_kv else 1,
            seq_len_cache // (tp if tp_shard_kv else 1),
            kvpe_dim,
            dtype=torch.bfloat16,
        )
        for u in range(num_users):
            kv_prior = users[u]["kv_prior"]
            if trace_pe_interleave:
                # Same transform the post-run cache comparison applies (k_nope is basis-agnostic);
                # without it the preloaded prefix attends in the wrong basis.
                kv_prior = kv_prior.clone()
                d = kvpe_dim - config.kv_lora_rank
                pe = kv_prior[:, config.kv_lora_rank :]
                kv_prior[:, config.kv_lora_rank :] = torch.stack([pe[:, : d // 2], pe[:, d // 2 :]], dim=-1).reshape(
                    pe.shape[0], d
                )
            stripes = sp * tp if tp_shard_kv else sp
            bc = blockcyclic_cache_host(kv_prior, stripes, chunk_size_global, seq_len_cache, kvpe_dim)[0, 0]
            cache_host[u] = tp_stripe_major_cache(bc, sp, tp) if tp_shard_kv else bc.unsqueeze(0)
        cache_host_tt = ttnn.from_torch(
            cache_host,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
        )
        preload_cache(cache_host_tt, tt_kvpe_cache.storage)
        ttnn.synchronize_device(mesh_device)

    mesh_device.enable_program_cache()
    # Accumulated natural-order output per user (only the measured region is filled).
    out_accum = [torch.zeros(1, 1, users[u]["total_len"], hidden_size, dtype=torch.bfloat16) for u in range(num_users)]

    # ---- iterate: interleave users by local iter index (exercises cross-user isolation) ----
    det_failures = []
    profiled_ns = 0.0
    n_iters = max(len(u["group"]) for u in users)
    logger.info(f"Starting DEVICE chunked prefill: up to {n_iters} iters x {num_users} user(s)")
    for i in range(n_iters):
        for u in range(num_users):
            g = users[u]["group"]
            if i >= len(g):
                continue
            isl = g[i]
            kv_actual = prefill_len + sum(g[:i])
            valid_end = kv_actual + isl
            total_len = users[u]["total_len"]

            positions = rotated_chip_positions(kv_actual, sp, chunk_local)
            flat = [positions[c][r] for c in range(sp) for r in range(chunk_local)]
            gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
            chunk_in = users[u]["hidden"][gather_idx].clone()
            chunk_in[torch.tensor([gp >= valid_end for gp in flat])] = 0.0

            tt_h = ttnn.from_torch(
                chunk_in.reshape(1, 1, chunk_size_global, hidden_size),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
                ),
            )
            # Trace-safe metadata variant: build the runner's canonical metadata and hand it to forward
            # verbatim -- ttMLA threads it to all chunked ops (update/rope/zero_pad/SDPA), which read
            # their per-chunk scalars on-device. The contract is a 3-tuple of separate 1-element uint32
            # tensors indexed as metadata[0]=slot_id, [1]=actual_start, [2]=actual_end -- NOT one packed
            # tensor. slot_id = cache_user_id (layer_num=1, so it is also the flat cache slot).
            kv_pad_metadata = None
            if use_metadata_tensor:
                scalars = tuple(
                    ttnn.from_torch(
                        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                    for val in (u, kv_actual, valid_end)
                )
                # 4th field is Mistral's query-scale buffer (None elsewhere), refreshed here by
                # write_chunk_metadata so the scalars and the scale advance together.
                kv_pad_metadata = ChunkMetadata(*scalars, llama4_scale_buf)
                write_chunk_metadata(
                    kv_pad_metadata,
                    (u, kv_actual, valid_end),
                    hf_config=config,
                    mesh_device=mesh_device,
                    chunk_size_global=chunk_size_global,
                    sp_axis=sp_axis,
                )
            # Metadata path: pass ONLY the per-element metadata operands (the runtime's _trace_metadata
            # equivalent) -- actual_start/actual_end are read on-device, so leave them None to prove
            # forward needs no host per-chunk scalars. cache_user_id is unused on this path (slot comes
            # from metadata[0]).
            # Determinism re-issues the SAME forward on the same device inputs. forward takes
            # actual_start/cache_user_id from the caller, so a repeat rewrites the same cache slots
            # with the same data -- idempotent, like the repeated block() in test_prefill_block.
            det_baseline = None
            for rep in range(DETERMINISM_REPS if determinism_check else 1):

                def _forward():
                    return mla_tt.forward(
                        hidden_states=tt_h,
                        rope_tensors=indexed_rope,
                        kvpe_cache=tt_kvpe_cache,
                        cache_layer_idx=0,
                        actual_start=None if use_metadata_tensor else kv_actual,
                        # Scalar path needs actual_end only when the window overruns; None otherwise keeps
                        # the existing scenarios writing the whole padded slab. (Metadata always carries it.)
                        actual_end=valid_end if (tight_cache and not use_metadata_tensor) else None,
                        cache_user_id=u,
                        metadata=kv_pad_metadata,
                    )

                if profile:
                    tt_out, fwd_ns = _rt_profile_forward_ns(mesh_device, _forward)
                    profiled_ns += fwd_ns
                    logger.info(f"  user {u} iter {i}: forward {fwd_ns / 1e6:.3f} ms (realtime profiler)")
                else:
                    tt_out = _forward()
                out_flat = ttnn.to_torch(
                    tt_out,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                    ),
                ).to(torch.bfloat16)[0, 0]
                if not determinism_check:
                    continue
                if rep == 0:
                    det_baseline = out_flat.clone()
                    continue
                # Collect rather than assert: one report listing every non-deterministic (user, iter,
                # rep) beats aborting on the first, same as the prefill-block/transformer checks.
                _, pcc = comp_pcc(det_baseline.float(), out_flat.float())
                status = "PASS" if pcc >= DETERMINISM_PCC_THRESHOLD else "FAIL"
                logger.info(f"  user {u} iter {i} rep {rep} vs rep0: PCC = {pcc:.6f}  {status}")
                if pcc < DETERMINISM_PCC_THRESHOLD:
                    det_failures.append((u, i, rep, pcc))
            if kv_pad_metadata is not None:
                # .scalars: field 3 is a persistent buffer allocated outside this loop, so a
                # blanket deallocate would free it before the next chunk reads it.
                for meta_tensor in kv_pad_metadata.scalars:
                    ttnn.deallocate(meta_tensor)

            assert torch.isfinite(out_flat).all(), f"user {u} iter {i}: non-finite output"
            valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < valid_end]
            src = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
            dst = torch.tensor([gp for _, gp in valid_pairs], dtype=torch.long)
            out_accum[u][0, 0, dst, :] = out_flat[src, :]

            if users[u]["ref_out"] is not None:
                _, msg = assert_with_pcc(
                    users[u]["ref_out"][kv_actual:valid_end].reshape(1, 1, isl, hidden_size),
                    out_accum[u][:, :, kv_actual:valid_end, :],
                    0.98,
                )
                rot = "rotated" if kv_actual % chunk_size_global != 0 else "aligned"
                logger.info(f"  user {u} iter {i} (kv_actual={kv_actual} isl={isl} {rot}): out PCC {msg}")
        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()

    if profile:
        return profiled_ns
    if return_outputs:
        if return_pccs:
            assert reference is not None, "return_pccs needs a reference"
            # Each user's full measured-region output PCC against the reference (the check below, as values; the
            # KV-cache comparison below does not run on this path).
            pccs = [
                comp_pcc(
                    users[u]["ref_out"][prefill_len:].reshape(1, 1, -1, hidden_size).float(),
                    out_accum[u][:, :, prefill_len:, :].float(),
                )[1]
                for u in range(num_users)
            ]
            return out_accum, pccs
        return out_accum

    if determinism_check:
        if det_failures:
            msg = "; ".join(f"user {u} iter {i} rep {rep}: {pcc:.6f}" for u, i, rep, pcc in det_failures)
            pytest.fail(f"Determinism PCC below {DETERMINISM_PCC_THRESHOLD}: {msg}")
        logger.success(
            f"✓ Chunked prefill determinism: {DETERMINISM_REPS} reps of every forward bit-identical "
            f"({n_iters} iter(s) x {num_users} user(s))"
        )
        return

    if reference is None:
        logger.success(f"✓ Functional chunked prefill ran ({num_users} user(s), finite output)")
        return

    # ---- per-user full-measured-region output PCC ----
    for u in range(num_users):
        if users[u]["ref_out"] is None:
            continue
        meas = out_accum[u][:, :, prefill_len:, :]
        ref_meas = users[u]["ref_out"][prefill_len:].reshape(1, 1, -1, hidden_size)
        _, msg = assert_with_pcc(ref_meas, meas, 0.98)
        logger.info(f"  user {u} full measured output PCC: {msg}")

    # ---- check the measured KV cache vs the reference. The rotation accumulates into the canonical
    #      block-cyclic layout, so blockcyclic_positions un-rotates the final cache (incl. partial
    #      chunks). k_nope is compared directly; k_pe is direct for the CPU ref (mla_reference is
    #      Meta-style) and for NoPE, re-interleaved for a roped GPU trace -- see trace_pe_interleave. ----
    if any(users[u]["kv_post"] is not None for u in range(num_users)):
        cache_flat, stripes = gather_cache_natural(
            tt_kvpe_cache.storage, mesh_device, tp_shard_kv
        )  # [num_users, seq_cache, kvpe]
        p = blockcyclic_positions(stripes, chunk_size_global, seq_len_cache)
        kv_lora = config.kv_lora_rank
        d = kvpe_dim - kv_lora
        for u in range(num_users):
            if users[u]["kv_post"] is None:
                continue
            nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
            nat[p] = cache_flat[u]
            dev = nat[prefill_len : users[u]["total_len"]]
            ref = users[u]["kv_post"][prefill_len:].to(torch.float32)
            ref_pe = ref[:, kv_lora:]
            if trace_pe_interleave:  # HF half-split -> device Meta basis
                ref_pe = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)
            _, nope_msg = assert_with_pcc(ref[:, :kv_lora], dev[:, :kv_lora], 0.98)
            _, pe_msg = assert_with_pcc(ref_pe, dev[:, kv_lora:], 0.98)
            basis = "Meta-aligned" if trace_pe_interleave else "direct"
            logger.info(f"  user {u} KV cache PCC -- k_nope: {nope_msg}  k_pe[{basis}]: {pe_msg}")

    logger.success(f"✓ Chunked prefill passed ({'trace' if use_trace else 'cpu'} ref, {num_users} user(s))")


# ---------------------------------------------------------------------------------------------------
# Direct checks on the llama4 query-scale tensor. These read the tensor back off-device and compare it
# to rotated_chip_positions ground truth, INSTEAD of inferring correctness from an output PCC.
#
# Why not an output-PCC test: the query temperature moves full-output PCC by ~0.002 against this file's
# 0.98 gate (measured by disabling the device side: rot-allfull 0.9918 -> 0.9910, deep-50k+5k
# 0.9994 -> 0.9972). So every chunked scenario passes with the term absent, AND passes with it applied
# to the WRONG ROWS -- which is the bug this was written after finding. The first implementation used
# arange(start, start + chunk), correct only when `start` is slab-aligned; both scenarios used to
# "confirm" it happened to be slab-aligned, so neither could see it.
#
# Offsets cover aligned (0, 5120, 51200) and rotated (640 chip-aligned, 672 mid-chip straddle, 4480
# last chip, 6400 multi-slab) starts, each either crossing 8192 or sitting past it, so the scale
# actually varies across chips rather than being uniform.
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("kv_actual", [0, 640, 672, 4480, 5120, 6400, 7680, 8192, 51200])
@pytest.mark.parametrize("route", ["host", "buffer", "runtime"])
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 is validated on Blackhole only")
@pytest.mark.timeout(0)
def test_llama4_query_scale_matches_rotated_positions(request, mesh_device, kv_actual, route, variant, device_params):
    """The per-chip scale values must match the positions those chips actually carry.

    All three delivery routes, because nothing else covers them:
      * "host"    -> ttMLA builds the tensor from a host actual_start (single-shot / scalar path).
      * "buffer"  -> refresh_llama4_scale writes the persistent buffer the traced path reads.
      * "runtime" -> TtPrefillRuntime's own writer, so the arguments it passes are covered too.

    The chunked scenarios cannot check any of this: an unrefreshed buffer holds ones, which is just
    "no scale", and that is inside the PCC gate's noise.
    """
    chunk_size_global = 5120
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    chunk_local = chunk_size_global // sp
    seq_len_cache = ((kv_actual + 2 * chunk_size_global) // chunk_size_global) * chunk_size_global

    config, weights = request.getfixturevalue("random_weights")
    config.max_seq_len = seq_len_cache

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=ttnn.Topology.Linear,
        is_chunked=True,
        active_seq_len=chunk_size_global,
        slot_num=1,
        layer_num=1,
    )
    assert mla_tt._llama4_beta is not None, "llama_4_scaling_beta is not reaching ttMLA"

    if route == "host":
        tt_scale = mla_tt._llama4_scale(kv_actual, chunk_local, None)
    else:
        setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
        buf = setup.make_llama4_scale_buffer(chunk_size_global)
        assert buf is not None, "RotarySetup did not allocate the persistent buffer"
        if route == "buffer":
            refresh_llama4_scale(buf, config, mesh_device, kv_actual, chunk_size_global, sp_axis=sp_axis)
        else:
            # Nothing in-tree constructs a TtPrefillRuntime (it needs the full model and a trace
            # region), so a stub carrying the four attributes the method touches stands in for one.
            meta = tuple(
                ttnn.from_torch(
                    torch.tensor([0], dtype=torch.int64).reshape(1, 1, 1, 1),
                    device=mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                for _ in range(3)
            )
            write_chunk_metadata(
                ChunkMetadata(*meta, buf),
                (0, kv_actual, kv_actual + chunk_size_global),
                hf_config=config,
                mesh_device=mesh_device,
                chunk_size_global=chunk_size_global,
                sp_axis=sp_axis,
            )
            start_back = ttnn.to_torch(meta[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            assert int(start_back.flatten()[0]) == kv_actual, "metadata[1] was not written"
        tt_scale = buf

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    shard_dims[tp_axis] = 1  # head dim is replicated; concat and read one slice
    got = ttnn.to_torch(
        tt_scale, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    )

    beta = mla_tt._llama4_beta
    orig_max = mla_tt._llama4_orig_max
    positions = rotated_chip_positions(kv_actual, sp, chunk_local)
    flat = torch.tensor([positions[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.float32)
    want = 1.0 + beta * torch.log(1.0 + torch.floor(flat / orig_max))

    actual = got[0, 0, :, 0].float()
    n_distinct = int(want.unique().numel())
    logger.info(
        f"kv_actual={kv_actual} route={route}: {n_distinct} distinct scale value(s), "
        f"range [{want.min():.6f}, {want.max():.6f}]"
    )
    torch.testing.assert_close(actual, want.to(actual.dtype), rtol=0, atol=4e-3)

    # A uniform chunk cannot distinguish a correct row map from a permuted one, so pin which cases are
    # supposed to vary. A chunk varies iff it straddles a multiple of orig_max; sitting entirely inside
    # one window (kv_actual=8192 -> positions 8192..13311) is uniform and correct.
    straddles = (kv_actual // orig_max) != ((kv_actual + chunk_size_global - 1) // orig_max)
    assert (n_distinct > 1) == straddles, (
        f"kv_actual={kv_actual}: expected {'varying' if straddles else 'uniform'} scale, "
        f"got {n_distinct} distinct value(s)"
    )


# The rows above check the scale tensor's VALUES. None of them checks that it reaches tt_q: the 27
# direct rows call _llama4_scale itself, and the trace-replay row multiplies a synthetic q of its own.
# Deleting the multiply in _q_stem leaves every one of them green. This row closes that gap by running
# _q_stem twice, scale active and scale suppressed, and asserting the ratio between the two outputs is
# the scale itself -- so the wiring is covered rather than inferred. Output PCC cannot do this job, for
# the same reason given above: the temperature moves full-output PCC by ~0.002 against a 0.98 gate.
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
# Past 8192, so the scale is not 1.0 and the assertion can actually fail. At 5120 the scale is exactly
# 1.0 and this test would pass with the multiply deleted.
@pytest.mark.parametrize("kv_actual", [25600], ids=["past_8192"])
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 is validated on Blackhole only")
@pytest.mark.timeout(0)
def test_llama4_query_scale_is_applied_in_q_stem(request, mesh_device, kv_actual, variant, device_params):
    """The query temperature must be applied to tt_q, not merely computed correctly."""
    chunk_size_global = 5120
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    chunk_local = chunk_size_global // sp
    seq_len_cache = ((kv_actual + 2 * chunk_size_global) // chunk_size_global) * chunk_size_global

    config, weights = request.getfixturevalue("random_weights")
    config.max_seq_len = seq_len_cache

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=ttnn.Topology.Linear,
        is_chunked=True,
        active_seq_len=chunk_size_global,
        slot_num=1,
        layer_num=1,
    )
    assert mla_tt._llama4_beta is not None, "llama_4_scaling_beta is not reaching ttMLA"

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )

    # _q_stem deallocates qr, so build one input per call from the same host tensor.
    torch.manual_seed(0)
    qr_host = torch.randn(1, 1, chunk_local, config.q_lora_rank, dtype=torch.float32)

    def _q_stem_output(apply_scale: bool):
        qr = ttnn.from_torch(
            qr_host,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        saved = mla_tt._llama4_beta
        if not apply_scale:
            mla_tt._llama4_beta = None  # skips the multiply in _q_stem
        try:
            out = mla_tt._q_stem(qr, rope_tensors, kv_actual, chunk_local, metadata=None)
        finally:
            mla_tt._llama4_beta = saved
        shard_dims = [None, None]
        shard_dims[sp_axis] = 2
        shard_dims[tp_axis] = 1
        return ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).float()

    scaled = _q_stem_output(apply_scale=True)
    plain = _q_stem_output(apply_scale=False)

    beta, orig_max = mla_tt._llama4_beta, mla_tt._llama4_orig_max
    positions = rotated_chip_positions(kv_actual, sp, chunk_local)
    flat = torch.tensor([positions[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.float32)
    want = 1.0 + beta * torch.log(1.0 + torch.floor(flat / orig_max))
    assert want.min() > 1.0, f"kv_actual={kv_actual} produced scale 1.0; the test cannot fail as written"

    # ttMLA applies the scale to the q_a latent, upstream of q_b_proj, wkv_b1 and RoPE, so each output
    # element is a long dot product of rounded scaled inputs. An element-wise scaled/plain ratio is
    # then noisy wherever that dot product nearly cancels. Compare per row (= per position, where the
    # scale is constant) with the least-squares ratio <scaled, plain> / <plain, plain> instead: it is
    # the scale when the multiply is wired, and 1.0 when it is not.
    ref = plain[0, :, :, :].permute(1, 0, 2).reshape(plain.shape[2], -1)  # [rows, heads * width]
    got = scaled[0, :, :, :].permute(1, 0, 2).reshape(scaled.shape[2], -1)
    ratio = (got * ref).sum(-1) / (ref * ref).sum(-1)
    logger.info(
        f"kv_actual={kv_actual}: expected scale range [{want.min():.6f}, {want.max():.6f}], "
        f"observed per-row ratio range [{ratio.min():.6f}, {ratio.max():.6f}]"
    )
    torch.testing.assert_close(ratio, want.to(ratio.dtype), rtol=1e-2, atol=0)


# ---------------------------------------------------------------------------------------------------
# Does a CAPTURED trace read the scale buffer's live contents, or a capture-time snapshot?
#
# That one ttnn semantic is why the traced path uses a persistent buffer refreshed per chunk instead of
# a host-built tensor. If a replay saw a snapshot, every chunk after the captured one would silently
# carry the captured chunk's temperature, and the chunked PCC gate could not see it.
#
# Deliberately a two-op trace, not a full MLA forward: attention at a 51200-token offset reads 51200
# tokens of prior KV, so a replay-vs-eager comparison there has to preload the cache identically before
# each measured run -- which tests cache bookkeeping on top of the thing actually in question.
#
# Offsets 0 (scale 1.0) and 51200 (1.194591) are 19% apart and each uniform across the chunk, so a
# snapshot and a live read are unmistakable and no rotation reasoning is needed to read the result.
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 16 * 1024 * 1024}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 is validated on Blackhole only")
@pytest.mark.timeout(0)
def test_llama4_scale_buffer_is_live_under_trace_replay(request, mesh_device, variant, device_params):
    """A replay must pick up a refreshed scale buffer, and must not change without one."""
    chunk_size_global = 5120
    sp_axis, tp_axis = 0, 1
    off_a, off_b = 0, 51200
    seq_len_cache = ((off_b + 2 * chunk_size_global) // chunk_size_global) * chunk_size_global

    config, _ = request.getfixturevalue("random_weights")
    config.max_seq_len = seq_len_cache
    scale_buf = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).make_llama4_scale_buffer(
        chunk_size_global
    )
    assert scale_buf is not None, "RotarySetup did not allocate the persistent buffer"

    rope_scaling = config.rope_scaling
    beta, orig_max = rope_scaling["llama_4_scaling_beta"], rope_scaling["original_max_position_embeddings"]
    expect = {off: 1.0 + beta * math.log(1.0 + off // orig_max) for off in (off_a, off_b)}
    assert expect[off_a] != expect[off_b], "offsets must differ in scale or this test proves nothing"

    # The operand stands in for what ttMLA scales: the normed q_a latent, [1, 1, S, q_lora_rank],
    # SP-sharded on the sequence and replicated across TP (see rope._llama4_scale_geometry). The read
    # back concatenates the replicated TP copies on dim 1 and keeps one.
    in_dims = [None, None]
    in_dims[sp_axis] = 2
    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    shard_dims[tp_axis] = 1
    q = ttnn.from_torch(
        torch.ones(1, 1, chunk_size_global, config.q_lora_rank, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=in_dims),
    )

    refresh_llama4_scale(scale_buf, config, mesh_device, off_a, chunk_size_global, sp_axis=sp_axis)
    out = ttnn.multiply(q, scale_buf)  # warm/compile before recording
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = ttnn.multiply(q, scale_buf)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def replayed():
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        )
        return got[0, 0, :, 0].float()

    try:
        at_a = replayed()
        torch.testing.assert_close(
            at_a, torch.full_like(at_a, expect[off_a]), rtol=0, atol=4e-3, msg="replay at the captured offset is wrong"
        )

        refresh_llama4_scale(scale_buf, config, mesh_device, off_b, chunk_size_global, sp_axis=sp_axis)
        at_b = replayed()
        logger.info(f"trace replay: offset {off_a} -> {at_a[0]:.6f}, offset {off_b} -> {at_b[0]:.6f}")
        torch.testing.assert_close(
            at_b,
            torch.full_like(at_b, expect[off_b]),
            rtol=0,
            atol=4e-3,
            msg=(
                f"after refreshing to offset {off_b} a replay still produced {at_b[0]:.6f} (expected "
                f"{expect[off_b]:.6f}); the captured graph reads a capture-time SNAPSHOT, so the "
                "persistent-buffer design does not hold and the traced path needs the position math "
                "evaluated on-device instead"
            ),
        )

        torch.testing.assert_close(replayed(), at_b, rtol=0, atol=0, msg="replay is not deterministic")
    finally:
        ttnn.release_trace(mesh_device, tid)


# Functionality scenarios (id, kwargs) -- PURE FUNCTIONALITY: no mesh, no reference. Mesh and
# reference are SEPARATE pytest axes below (chunk=5120 is valid for sp in {2,4,8}), so the same
# scenario runs on any mesh and is validated against either ground truth (or run functional) without
# duplicating the case.
_CHUNKED_SCENARIOS = (
    [(f"rot-{rid}", dict(iters_isl=lst)) for rid, lst in zip(ROTATED_VALID_IDS, ROTATED_VALID_LISTS)]
    # One representative case packing the most sp=8 edges: iter0 aligned partial, iter1 rotated
    # chip-aligned (offset=0) partial, iter2 rotated mid-chip straddle (offset=32) + multi-slab + full.
    # NOTE: ids must not nest as substrings, else `-k <id>` can't isolate one (pytest -k is substring).
    # Convention: "-Nu" = N users. "maxedge"/"deep" are intentional families (single- + multi-user).
    + [("maxedge-1u", dict(iters_isl=[2560, 2592, 5120]))]
    # One aligned full chunk, no padding and no rotation -- the determinism baseline, where a repeat
    # diff can only come from the device, not from the pad/rotate bookkeeping maxedge exercises.
    + [("plain-5k", dict(iters_isl=[5120]))]
    + [
        ("production-50k+5k", dict(iters_isl=[5120] * 11)),
        ("fullchunk-2u", dict(iters_isl=[5120] * 4, num_users=2)),
        # Multi-user WITH padding/rotation: each user runs the full maxedge pattern in its own slot
        # (partition splits [..]*2 into one maxedge per user), exercising rotation + cross-user isolation.
        ("maxedge-2u", dict(iters_isl=[2560, 2592, 5120] * 2, num_users=2)),
        # Tail chunk padding PAST the cache end: each iter leaves a tile of pad, so kv_actual drifts and
        # the final 32-token iter's window ends 5056 past the 10240 cache while its real tokens fit. Only
        # a drifted start can overrun. See tight_cache in the driver.
        ("padoverflow-1u", dict(iters_isl=[5120 - 32] * 2 + [32], tight_cache=True)),
        ("deep-50k+5k", dict(iters_isl=[5120], prefill_len=50 * 1024)),
        ("deep-2u", dict(iters_isl=[5120, 5120], prefill_len=50 * 1024, num_users=2)),
    ]
)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((2, 2), fabric2d_device_params(l1_small_size=1152), id="fabric2d-2x2"),
        pytest.param((2, 4), fabric2d_device_params(l1_small_size=1152), id="fabric2d-2x4"),
        # high_bw_all_gather parks readiness/completion semaphores in L1_SMALL. On 8x4 the
        # fallback fragments general L1 enough that a later op's static circular buffers collide.
        #
        pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("reference", ["cpu", "trace", None], ids=["cpu", "trace", "func"])
@pytest.mark.parametrize("kwargs", [kw for _, kw in _CHUNKED_SCENARIOS], ids=[sid for sid, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "variant",
    ["kimi_k2_7", "kimi_k3", "mistral_small_4"],
    indirect=True,
    # Name the Kimi generation explicitly. pytest -k is substring-based, so the ids must stay
    # disjoint: "k2_7" and "k3" cannot cross-match, whereas a bare "kimi" id would match both
    # generations and silently widen every `-k` selector (CI yaml, tests/perf/test_mla_perf.py).
    # "mistral4" matches the transformer/block rows' id, so one `-k mistral4` selects this model
    # across all three; `-k mistral` still finds it, and no other variant here shares the prefix.
    ids=["k2_7", "k3", "mistral4"],
)
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("tp_shard_kv", [False, True], ids=["sp_only", "tp_sharded"])
@pytest.mark.timeout(0)
def test_mla_chunked_prefill(
    request, mesh_device, kwargs, reference, device_params, variant, use_metadata_tensor, determinism_check, tp_shard_kv
):
    """Unified chunked-prefill driver crossed with independent mesh and reference axes. Each
    functionality scenario (rotation edges, production depth, multi-user, deep prefix) runs on any mesh
    and is validated against the CPU torch reference ('cpu'), the GPU trace ('trace', asserts if the
    variant has no registered trace), or run with no reference ('func'). Select with e.g.
    -k 'maxedge-1u and trace and 8x4'. See _run_chunked_prefill.

    Real weights on the CPU-reference path: point the variant's HF env var (KIMI_K2_7_HF_MODEL /
    KIMI_K3_HF_MODEL) at a checkpoint to validate the chunked path against the CPU torch reference
    with pretrained weights instead of random. create_mla_reference is config-driven and
    architecture-agnostic (Kimi's YaRN/theta flow through, absorbed-MLA math matches the variant's own
    reference), so this works for both variants. It complements the GPU-trace path, which only
    replays full-chunk iters and so never exercises real weights across the rotation/partial-chunk edge
    scenarios that the cpu path covers. Without the env var, fall back to random. kimi_k2_7 has no
    registered golden MLA trace (https://github.com/tenstorrent/tt-metal/issues/54973), so selecting
    'trace' for it fails rather than silently passing; it runs the cpu and func paths, otherwise the
    same config-driven driver on any arch/mesh.

    kimi_k3 (NoPE + output gate, 96 heads) runs 'scalar' only -- 'metadata' is skipped explicitly
    below. Unlike kimi_k2_7 it runs 'trace', taking real weights from layer 3 via
    variant.pretrained_mla_layer. Its rotation scenarios still matter: rotation comes from the
    block-cyclic cache write and the causal offset, not from RoPE."""
    # Per-variant, not module-level: two CI selectors for this test are variant-unqualified, so
    # without this a kimi_k3 case would run on Wormhole T3K where it has never been validated.
    if variant.name == "kimi_k3" and not is_blackhole():
        pytest.skip("kimi_k3 is validated on Blackhole only")
    # The metadata contract serves the trace-safe runtime (inbound_socket_service_sync feeds forward
    # tt_metadata directly). K3 has no runtime -- build_runtime/allocate_kv_cache deliberately raise
    # -- so the path is unreachable for it and passes only via the shared arch-agnostic ttMLA.forward.
    # Incidental, not a K3 guarantee; re-enable when K3 has a runtime that actually feeds metadata.
    if variant.name == "kimi_k3" and use_metadata_tensor:
        pytest.skip("kimi_k3 has no runtime, so the metadata (device-scalar) path is unreachable for it")
    # Same reason as K3's: the variant-unqualified CI selectors for this test would otherwise run
    # Mistral on Wormhole T3K, where it has never been brought up.
    if variant.name == "mistral_small_4" and not is_blackhole():
        pytest.skip("mistral_small_4 is validated on Blackhole only")
    # Expanded K/V has measured schedules for Kimi on 8x4 only (_EXPANDED_K_SDPA_CHUNKS).
    if tp_shard_kv and (variant.name == "mistral_small_4" or tuple(mesh_device.shape) != (8, 4)):
        pytest.skip("a TP-deduped cache runs expanded K/V only, which has no schedule for this model and mesh")
    # Variant-agnostic, not a name check: mla_trace_defaults is the adapter's own declaration that
    # no golden was ever recorded ("Empty = ..." in adapter.py), which is exactly the condition the
    # assert in _run_chunked_prefill reports. Skipping up front reads as out of scope instead of as
    # a hard failure, and any future variant without a trace is covered without editing this file.
    # Mistral is the case that needs it today; 'cpu' and 'func' cover all 13 scenarios between them.
    if reference == "trace" and not variant.mla_trace_defaults:
        pytest.skip(f"no golden MLA trace recorded for {variant.name}; use reference='cpu' or 'func'")
    # Opt into real weights on the cpu path when the variant's checkpoint env var is set. The "trace"
    # path already forces pretrained; "func" is ref-less so weights don't matter. The pretrained
    # fixture skips the test if the env var is set but the checkpoint is incomplete.
    if reference == "cpu" and os.environ.get(variant.env_var) and not kwargs.get("use_pretrained"):
        kwargs = {**kwargs, "use_pretrained": True}
    topology = per_axis_topology(device_params["fabric_config"])
    _run_chunked_prefill(
        request,
        mesh_device,
        reference=reference,
        topology=topology,
        use_metadata_tensor=use_metadata_tensor,
        determinism_check=determinism_check,
        tp_shard_kv=tp_shard_kv,
        **kwargs,
    )


@pytest.mark.parametrize("mesh_device", [(8, 1)], ids=["8x1"], indirect=True)
@pytest.mark.parametrize("device_params", [torus_y_device_params(l1_small_size=1152)], ids=["torus-y"], indirect=True)
@pytest.mark.parametrize("variant", ["mistral_small_4"], ids=["mistral4"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 LoudBox perf targets Blackhole")
@pytest.mark.timeout(0)
def test_mistral4_mla_chunked_prefill_loudbox(request, mesh_device, device_params, variant):
    """One functional 50k-prefix + 5k-chunk MLA forward at the PP4 stage shape.

    Prefix preparation is outside MLA_START/MLA_END. Scalar metadata and determinism checks are
    pinned so the perf wrapper measures exactly one forward. Use an eight-device LoudBox or a
    Galaxy column exposed with TT_VISIBLE_DEVICES; the fixture requires eight visible devices.
    """
    _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        use_metadata_tensor=False,
        determinism_check=False,
        iters_isl=[5120],
        prefill_len=50 * 1024,
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((2, 4), fabric2d_device_params(l1_small_size=1152), id="fabric2d-2x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.skipif(not is_blackhole(), reason="LoudBox (8x p150b) perf; the realtime profiler is Blackhole-only")
@pytest.mark.timeout(0)
def test_mla_chunked_perf_loudbox(request, mesh_device, device_params, variant):
    """Chunked-prefill MLA forward on the LoudBox as a 2x4 mesh, logged with a per-op breakdown. sp=2 / tp=4 with a
    640-row-per-chip chunk gives every chip the Galaxy's work -- 640 Q rows of its per-chip heads against the 50k
    cached prefix -- at the production SDPA chunks; only the ring is shorter (2 chips, not 8)."""
    chunk_size_global = PREFILL_CHUNK_TOKENS_PER_CHIP * mesh_device.shape[0]
    total_ns = _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        profile=True,
        chunk_size_global=chunk_size_global,
        iters_isl=[chunk_size_global],
        prefill_len=50 * 1024,
    )
    logger.info(
        f"{variant.name} chunked 50k + {chunk_size_global} MLA forward (LoudBox 2x4): "
        f"{total_ns / 1e6:.3f} ms (realtime profiler)"
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1152),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k3"], indirect=True, ids=["k3"])
@pytest.mark.skipif(not is_blackhole(), reason="kimi_k3 and the realtime profiler are Blackhole-only")
@pytest.mark.timeout(0)
def test_mla_chunked_perf_check(request, mesh_device, device_params, variant):
    """Kimi-K3 chunked-prefill MLA perf on the 8x4 Galaxy: 50k cached prefix + one fresh 5k chunk,
    timed with the realtime (lightweight) profiler instead of Tracy.

    NOT comparable to test_kimi_k3_mla_chunked_perf_galaxy's target: that number comes from the
    Tracy merge path, which averages collectives across chips, while this takes the max for every
    program. K3's forward is ~7% CCL, so the two disagree by construction."""
    require_high_power("kimi_k3 MLA chunked perf")
    topology = per_axis_topology(device_params["fabric_config"])
    total_ns = _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=topology,
        profile=True,
        iters_isl=[5120],
        prefill_len=50 * 1024,
    )
    lower = K3_CHUNKED_RT_PERF_NS * (1 - K3_CHUNKED_RT_PERF_MARGIN)
    upper = K3_CHUNKED_RT_PERF_NS * (1 + K3_CHUNKED_RT_PERF_MARGIN)
    logger.info(
        f"kimi_k3 chunked 50k+5k realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms), "
        f"expected {K3_CHUNKED_RT_PERF_NS:,} ns, band [{lower:,.0f}, {upper:,.0f}]"
    )
    emit_perf_metrics(
        "kimi_k3_mla_chunked_50k_5k_torus-xy-8x4",
        "Kimi-K3 MLA chunked 50k+5k perf (torus-xy 8x4)",
        {
            "prefill_len": 50 * 1024,
            "chunk": 5120,
            "measured_ns": total_ns,
            "measured_ms": total_ns / 1e6,
            "expected_ns": K3_CHUNKED_RT_PERF_NS,
            "perf_margin_pct": K3_CHUNKED_RT_PERF_MARGIN * 100.0,
            "within_band": lower <= total_ns <= upper,
        },
    )
    assert lower <= total_ns <= upper, (
        f"device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {K3_CHUNKED_RT_PERF_NS:,} ns, margin +/- {K3_CHUNKED_RT_PERF_MARGIN * 100:.1f}%)"
    )


# Output, TP-sharded vs TP-replicated cache: the full-mesh ring (32 sources, each a TP stripe of a slice) and the
# SP-axis ring (8 whole slices) merge the KV blocks' softmax states in different orders, so the two agree only to
# rounding, for ring_mla and expanded K/V alike (measured: K2.7 ring_mla 0.99987, expanded K/V 0.99983).
_TP_SHARDED_MIN_PCC = 0.999


def _assert_tp_sharded_matches_sp_only(sp_only, tp_sharded, what):
    """TP-sharded output agrees with the TP-replicated (sp_only) output to PCC >= _TP_SHARDED_MIN_PCC (see above)."""
    assert torch.isfinite(tp_sharded).all(), f"{what}: non-finite tp_sharded output"
    _, pcc = comp_pcc(sp_only.float(), tp_sharded.float(), _TP_SHARDED_MIN_PCC)
    logger.info(f"{what}: expanded K/V tp_sharded vs sp_only PCC {pcc:.6f}")
    assert pcc >= _TP_SHARDED_MIN_PCC, f"{what}: tp_sharded vs sp_only PCC {pcc}"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "reference,kwargs",
    [pytest.param(None, kw, id=f"func-{sid}") for sid, kw in _CHUNKED_SCENARIOS]
    # CPU-reference inputs (its prefix is the reference KV, not random) on every scenario short enough for the CPU MLA.
    + [
        pytest.param("cpu", kw, id=f"cpu-{sid}")
        for sid, kw in _CHUNKED_SCENARIOS
        if not sid.startswith(("deep", "prod"))
    ]
    + [pytest.param("trace", dict(iters_isl=[5120] * 11), id="trace-production-50k+5k")],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_tp_sharded_matches_sp_only(
    request, mesh_device, device_params, reference, kwargs, variant, use_metadata_tensor
):
    """The TP-sharded cache gives the TP-replicated (sp_only) run's output on the same inputs. With a TP-sharded cache
    the SDPA gathers the cache over the full mesh (32 sources) instead of the SP-axis ring (8), which merges the KV
    blocks in a different order, so the outputs match to _TP_SHARDED_MIN_PCC rather than bit for bit.
    """
    if reference == "trace" and not variant.mla_trace_defaults and not MLA_CHUNKED_TRACE_PATH:
        pytest.skip(f"no golden MLA trace recorded for {variant.name}")
    topology = per_axis_topology(device_params["fabric_config"])
    outs = {}
    for tp_shard_kv in (False, True):
        outs[tp_shard_kv] = _run_chunked_prefill(
            request,
            mesh_device,
            reference=reference,
            topology=topology,
            use_metadata_tensor=use_metadata_tensor,
            tp_shard_kv=tp_shard_kv,
            return_outputs=True,
            mla_kwargs=dict(expanded_k=True),
            **kwargs,
        )
    assert len(outs[False]) == len(outs[True]), "the two layouts returned different user counts"
    for u, (sp_only, tp_sharded) in enumerate(zip(outs[False], outs[True])):
        _assert_tp_sharded_matches_sp_only(sp_only, tp_sharded, f"user {u}")
    logger.success(f"✓ tp_sharded output matches sp_only ({len(outs[True])} user(s))")


# Per-chip DRAM saving of the TP-deduped cache, scaled to the model's MLA layers.
_TP_SHARDED_DRAM_POINTS = [
    pytest.param("kimi_k3", 512 * 1024, 24, 0.60e9, id="k3-512k"),
    pytest.param("kimi_k2_7", 256 * 1024, 61, 0.85e9, id="k2_7-256k"),
]


def _dram_allocated_bytes(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant,seq_len,mla_layers,min_saving", _TP_SHARDED_DRAM_POINTS, indirect=["variant"])
def test_mla_tp_sharded_dram_saving(mesh_device, device_params, config_only, variant, seq_len, mla_layers, min_saving):
    """Measured per-chip allocation of one layer's cache slot, TP-replicated vs TP-deduped; the model saves
    mla_layers * (replicated - deduped) per user slot."""
    seq_len = -(-seq_len // 5120) * 5120  # whole chunks, as the runner allocates
    sp_axis, tp_axis = 0, 1
    mesh_shape = tuple(mesh_device.shape)

    def cache_bytes(tp):
        before = _dram_allocated_bytes(mesh_device)
        cache = init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BFP8_TILE,
            hf_config=config_only,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
            num_users=1,
            tp_axis=tp,
        )
        ttnn.synchronize_device(mesh_device)
        used = _dram_allocated_bytes(mesh_device) - before
        ttnn.deallocate(cache.storage)
        return used

    replicated, deduped = cache_bytes(None), cache_bytes(tp_axis)
    saving = mla_layers * (replicated - deduped)
    logger.info(
        f"{variant.name} seq {seq_len}: per-chip cache slot replicated {replicated / 1e6:.1f} MB, deduped "
        f"{deduped / 1e6:.1f} MB -> {mla_layers}-layer saving {saving / 1e9:.3f} GB per user slot "
        f"(gate >= {min_saving / 1e9:.2f} GB)"
    )
    assert deduped * 3 < replicated, "the deduped cache is not ~1/4 of the replicated one"
    assert saving >= min_saving, f"saving {saving / 1e9:.3f} GB below {min_saving / 1e9:.2f} GB"


# A captured three-layer forward replayed with the user and prefix changed between replays, on the TP-sharded cache,
# must equal the eager forwards bit for bit, and those must match the TP-replicated (sp_only) eager forwards (see
# _assert_tp_sharded_matches_sp_only). Two users share each layer's cache, the user order makes the prefix go
# long -> short -> long, and user 1's starts are rotated (not chunk-aligned).
_REPLAY_STEPS = [(0, 0), (1, 0), (0, 5120), (0, 10240), (1, 2560), (0, 15360), (1, 7680)]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1152, trace_region_size=64 * 1024 * 1024),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("chunk", [5120, 2048], ids=["c5k", "c2k"])
@pytest.mark.timeout(0)
def test_mla_tp_sharded_trace_replay_multi_layer(request, mesh_device, device_params, variant, chunk):
    # Expanded K/V: every layer passes its own wkv_b1 and W_UV^T as the in-kernel K and V weights.
    config, weights = request.getfixturevalue("random_weights")
    sp_axis, tp_axis = 0, 1
    mesh_shape = tuple(mesh_device.shape)
    # _REPLAY_STEPS is written for 5k chunks; scale its starts so 2k chunks keep the same long/short/rotated pattern.
    replay_steps = [(user, start * chunk // 5120) for user, start in _REPLAY_STEPS]
    seq_len_cache = 4 * chunk
    config.max_seq_len = seq_len_cache
    num_layers, num_users = 3, 2
    hidden_size = config.hidden_size
    topology = per_axis_topology(device_params["fabric_config"])

    def layer_weights(layer):
        # Distinct per-layer kv_b (W_UK, W_UV): a captured program that kept another layer's weight address would
        # differ.
        return {**weights, "kv_b_proj.weight": (weights["kv_b_proj.weight"].float() * (1 + 0.25 * layer)).bfloat16()}

    layers = [
        ttMLA(
            config,
            layer_weights(layer),
            mesh_device,
            layer_idx=layer,
            seq_len=seq_len_cache,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=False,
            topology=topology,
            is_chunked=True,
            active_seq_len=chunk,
            slot_num=num_users,
            layer_num=num_layers,
            tp_shard_kv=True,
            expanded_k=True,
        )
        for layer in range(num_layers)
    ]
    assert all(mla_layer._expanded_k for mla_layer in layers), "expanded K mode not applied"
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len_cache, chunk_size_global=chunk)
    scale_buf = rope_setup.make_llama4_scale_buffer(chunk)

    hidden_dims = [None, None]
    hidden_dims[tp_axis] = -1
    hidden_dims[sp_axis] = -2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_dims)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=hidden_dims, mesh_shape=mesh_device.shape)

    def host_hidden(step, layer):
        gen = torch.Generator().manual_seed(10_000 * step + layer)
        return torch.randn(1, 1, chunk, hidden_size, generator=gen, dtype=torch.bfloat16)

    def to_device(x):
        return ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    metadata = _new_chunk_metadata(mesh_device, chunk, scale_buf)

    def set_step(user, start):
        write_chunk_metadata(
            metadata,
            (user, start, start + chunk),
            hf_config=config,
            mesh_device=mesh_device,
            chunk_size_global=chunk,
            sp_axis=sp_axis,
        )

    def forward_all(cache, hiddens):
        # One pass over the layers of a chunk.
        return [
            layers[layer].forward(
                hidden_states=hiddens[layer],
                rope_tensors=rope,
                kvpe_cache=cache,
                actual_start=None,
                actual_end=None,
                cache_layer_idx=layer,
                cache_user_id=0,
                metadata=metadata,
            )
            for layer in range(num_layers)
        ]

    # Eager reference on a fresh cache.
    eager = []
    cache_eager = _new_layered_kv_cache(config, mesh_device, seq_len_cache, num_layers, num_users, sp_axis, tp_axis)
    for step, (user, start) in enumerate(replay_steps):
        set_step(user, start)
        outs = forward_all(cache_eager, [to_device(host_hidden(step, layer)) for layer in range(num_layers)])
        eager.append([ttnn.to_torch(o, mesh_composer=composer) for o in outs])

    # Traced: one capture, replayed once per step with the inputs and metadata rewritten in place.
    cache_traced = _new_layered_kv_cache(config, mesh_device, seq_len_cache, num_layers, num_users, sp_axis, tp_axis)
    # Allocated before the capture: anything persistent allocated while the trace is alive could land on the trace's
    # intermediate buffers and be overwritten by a later replay.
    cache_sp = _new_layered_kv_cache(
        config, mesh_device, seq_len_cache, num_layers, num_users, sp_axis, tp_axis, tp_sharded=False
    )
    tt_hidden = [to_device(host_hidden(0, layer)) for layer in range(num_layers)]
    set_step(*replay_steps[0])
    # Compile the sp_only programs now too (writing step 0's rows of the replicated cache, as its pass will again), so
    # nothing is compiled -- kernel binaries are allocated -- while the trace is alive.
    for mla_layer in layers:
        mla_layer.set_tp_shard_kv(False)
    forward_all(cache_sp, tt_hidden)
    for mla_layer in layers:
        mla_layer.set_tp_shard_kv(True)
    forward_all(cache_traced, tt_hidden)  # compile; rewrites step 0's rows with step 0's data, as the replay will
    ttnn.synchronize_device(mesh_device)
    # Live-buffer allocation during the trace is checked only with TT_METAL_TRACE_ALLOC_TRACKING=1.
    program_entries = mesh_device.num_program_cache_entries()
    # The dense MLA forward loads no sub-device manager, so the capture is one trace segment.
    controller = SubDeviceTraceController(mesh_device)
    for mla_layer in layers:
        mla_layer.set_trace_controller(controller)
    controller.begin_capture()
    traced_outs = forward_all(cache_traced, tt_hidden)
    controller.end_capture()
    assert controller.num_segments == 1, f"{controller.num_segments} trace segments"

    def replay(steps):
        for step in steps:
            user, start = replay_steps[step]
            set_step(user, start)
            for layer in range(num_layers):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        host_hidden(step, layer), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
                    ),
                    tt_hidden[layer],
                )
            controller.replay()
            for layer in range(num_layers):
                got = ttnn.to_torch(traced_outs[layer], mesh_composer=composer)
                mismatched = (got != eager[step][layer]).sum().item()
                assert mismatched == 0, (
                    f"step {step} (user {user}, start {start}) layer {layer}: {mismatched} elements of the traced "
                    "replay differ from the eager forward"
                )

    try:
        replay(range(len(replay_steps)))

        # The same steps eagerly on the TP-replicated cache: the TP-sharded layer and cache indexing must give the same
        # output (see _assert_tp_sharded_matches_sp_only).
        for mla_layer in layers:
            mla_layer.set_tp_shard_kv(False)
        for step, (user, start) in enumerate(replay_steps):
            set_step(user, start)
            outs = forward_all(cache_sp, [to_device(host_hidden(step, layer)) for layer in range(num_layers)])
            for layer in range(num_layers):
                _assert_tp_sharded_matches_sp_only(
                    ttnn.to_torch(outs[layer], mesh_composer=composer),
                    eager[step][layer],
                    f"step {step} (user {user}, start {start}) layer {layer}",
                )
            # Allocated while the trace is alive: free before any replay (allocation tracking rejects live ones).
            for out in outs:
                ttnn.deallocate(out)

        # Back to TP-sharded after the replicated forwards: eager sharded forwards (program-cache hits with patched
        # runtime arguments) and then the same capture replayed again must still match.
        for mla_layer in layers:
            mla_layer.set_tp_shard_kv(True)
        for step, (user, start) in enumerate(replay_steps):
            set_step(user, start)
            outs = forward_all(cache_eager, [to_device(host_hidden(step, layer)) for layer in range(num_layers)])
            for layer in range(num_layers):
                got = ttnn.to_torch(outs[layer], mesh_composer=composer)
                assert torch.equal(
                    got, eager[step][layer]
                ), f"eager tp_sharded after sp_only: step {step} layer {layer}"
            for out in outs:
                ttnn.deallocate(out)
        replay(range(len(replay_steps)))
        assert (
            mesh_device.num_program_cache_entries() == program_entries
        ), "a program compiled while the trace was alive"
    finally:
        controller.release()
        for mla_layer in layers:
            mla_layer.set_trace_controller(None)

    # The caches themselves, un-striped to natural order: the traced, TP-sharded eager and TP-replicated eager runs must
    # have written identical bytes into every (user, layer) slot, and every slot must hold data. Equal outputs alone
    # would miss a cache write that ignores the layer, since every layout would read the same wrong slot.
    nat_eager, nat_traced, nat_sp = (
        _natural_cache(mesh_device, cache, tp_sharded, chunk, seq_len_cache)
        for cache, tp_sharded in ((cache_eager, True), (cache_traced, True), (cache_sp, False))
    )
    assert torch.equal(nat_traced, nat_eager), "traced and eager TP-sharded caches differ"
    assert torch.equal(nat_sp, nat_eager), "TP-replicated and TP-sharded caches differ"
    # Every slot is written, and rows past user 1's extent (12800 at 5k chunks) are written only in user 0's slots
    # (user 0 writes to 20480): a transposed (layer, user) slot mapping would put user 0's rows into a user-1 slot.
    user1_end = max(start for user, start in replay_steps if user == 1) + chunk
    for user in range(num_users):
        for layer in range(num_layers):
            slot = user * num_layers + layer
            assert nat_eager[slot].abs().sum() > 0, f"cache slot of user {user} layer {layer} was never written"
            tail = nat_eager[slot, user1_end:].abs().sum()
            assert (tail == 0) == (user == 1), f"user {user} layer {layer}: rows past {user1_end} in the wrong slot"
    logger.success(
        f"✓ traced replay == eager (tp_sharded) ~ eager (sp_only) outputs, caches bitwise, over "
        f"{len(replay_steps)} steps x {num_layers} layers, replayed again after the sp_only pass"
    )


class _SdpaRecorder:
    """Wraps ttnn.transformer.ring_joint_scaled_dot_product_attention (eager tests only) and keeps, per call, whether K
    was expanded, the Q chunk size, the matmul fidelity, the scale, and per-device host copies of the output (and, with
    keep_inputs, of Q, the latent K, V (None when expanded in-op), W_UK and W_UV^T). Device lists are in mesh order:
    index sp * tp_size + tp."""

    def __init__(self, monkeypatch, keep_inputs=False, keep_device_rows=None):
        self.calls = []
        # Always wrap the real op: a second recorder in the same test must not record through the first.
        sdpa = ttnn.transformer.ring_joint_scaled_dot_product_attention
        sdpa = getattr(sdpa, "_recorded_op", sdpa)

        def host(t):
            return None if t is None else [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]

        def recorded(q, k, v, *args, **kwargs):
            result = sdpa(q, k, v, *args, **kwargs)
            weight = kwargs.get("k_expand_weight")
            call = {
                "expanded": weight is not None,
                "q_chunk": kwargs["program_config"].q_chunk_size,
                "fidelity": kwargs["program_config"].matmul_math_fidelity,
                "scale": kwargs["scale"],
                "out": host(result[0]),
            }
            if keep_inputs:
                call.update(q=host(q), k=host(k), v=host(v), w=host(weight), w_uv=host(kwargs.get("v_expand_weight")))
            if keep_device_rows is not None:
                # Device copies of Q and of the first keep_device_rows rows of the latent K, and the call's arguments,
                # to replay the op directly (expanded K/V: no V tensor).
                assert v is None, "keep_device_rows replays expanded K/V calls only"
                rows = keep_device_rows
                call["device"] = {
                    "q": ttnn.clone(q),
                    "k": ttnn.slice(k, [0, 0, 0, 0], [1, 1, rows, k.shape[3]]),
                    "kwargs": dict(kwargs),
                }
            self.calls.append(call)
            return result

        recorded._recorded_op = sdpa
        monkeypatch.setattr(ttnn.transformer, "ring_joint_scaled_dot_product_attention", recorded)


# Layer output PCC floors vs the CPU reference at test_mla_expanded_kv_accuracy_floor's shape, per dense chunked
# attention. Expanded K/V sits below ring_mla because the in-op V expansion accumulates its kv_lora_rank-term product
# in bf16 (measured K3 0.99296 vs ring_mla 0.99805, K2.7 0.99632 vs 0.99738); a fix is pending.
_RING_MLA_MIN_PCC = {"kimi_k3": 0.9978, "kimi_k2_7": 0.9968}
_EXPANDED_KV_MIN_PCC = {"kimi_k3": 0.9925, "kimi_k2_7": 0.9958}
# Expanded K/V vs ring_mla layer outputs: the bf16 V accumulation above is the main difference between the two
# (bf8 rounding of the expanded K and V aside).
_EXPANDED_KV_MIN_PCC_VS_ABSORBED = 0.99


def _expanded_kv_max_pcc_loss(variant_name):
    """How far the expanded K/V output's PCC vs the CPU reference may fall below ring_mla's: the floors' gap, plus
    0.001 for scenario-to-scenario spread."""
    return _RING_MLA_MIN_PCC[variant_name] - _EXPANDED_KV_MIN_PCC[variant_name] + 0.001


def _record_ring_mla(monkeypatch):
    """Wraps ttnn.transformer.ring_mla; returns the list that gets each call's program config."""
    calls = []
    ring_mla = ttnn.transformer.ring_mla

    def recorded(*args, **kwargs):
        calls.append(kwargs["program_config"])
        return ring_mla(*args, **kwargs)

    monkeypatch.setattr(ttnn.transformer, "ring_mla", recorded)
    return calls


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param(kw, id=sid) for sid, kw in _CHUNKED_SCENARIOS if not sid.startswith(("deep", "prod"))],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.parametrize("tp_shard_kv", [False, True], ids=["sp_only", "tp_sharded"])
@pytest.mark.timeout(0)
def test_mla_expanded_k_matches_absorbed(
    request, mesh_device, device_params, kwargs, variant, use_metadata_tensor, tp_shard_kv, monkeypatch
):
    """Expanded K/V vs ring_mla (absorbed) on CPU-reference inputs: per user, the expanded output's PCC against the
    reference is at most _expanded_kv_max_pcc_loss below ring_mla's, and the two outputs agree to
    _EXPANDED_KV_MIN_PCC_VS_ABSORBED. tp_shard_kv picks expanded K/V's cache layout; ring_mla runs on the TP-replicated
    cache, the only one it supports."""
    _check_expanded_k_vs_absorbed(
        request, mesh_device, device_params, variant, kwargs, use_metadata_tensor, tp_shard_kv, monkeypatch
    )


def _check_expanded_k_vs_absorbed(
    request,
    mesh_device,
    device_params,
    variant,
    kwargs,
    use_metadata_tensor,
    tp_shard_kv,
    monkeypatch,
    q_chunk=None,
    fidelity=None,
):
    topology = per_axis_topology(device_params["fabric_config"])
    outs, ref_pccs = {}, {}
    for expanded in (False, True):
        sdpa_calls = _SdpaRecorder(monkeypatch).calls
        ring_mla_configs = _record_ring_mla(monkeypatch)
        outs[expanded], ref_pccs[expanded] = _run_chunked_prefill(
            request,
            mesh_device,
            reference="cpu",
            topology=topology,
            use_metadata_tensor=use_metadata_tensor,
            tp_shard_kv=tp_shard_kv and expanded,
            return_outputs=True,
            return_pccs=True,
            mla_kwargs=dict(expanded_k=expanded),
            **kwargs,
        )
        # The mode (and the expected schedule) really took effect: a declined gate would compare ring_mla with itself.
        if expanded:
            assert not ring_mla_configs, "expanded K/V ran ring_mla"
            assert sdpa_calls and all(c["expanded"] for c in sdpa_calls), "expanded K/V not applied"
            if q_chunk is not None:
                assert all(c["q_chunk"] == q_chunk for c in sdpa_calls), f"schedule not q{q_chunk}"
            fidelities = [c["fidelity"] for c in sdpa_calls]
        else:
            assert ring_mla_configs and not sdpa_calls, "expanded_k=False did not run ring_mla"
            fidelities = [c.matmul_math_fidelity for c in ring_mla_configs]
        if fidelity is not None:
            assert all(f == fidelity for f in fidelities), f"SDPA fidelity not {fidelity}"
    max_loss = _expanded_kv_max_pcc_loss(variant.name)
    for u, (absorbed, expanded) in enumerate(zip(outs[False], outs[True])):
        assert torch.isfinite(expanded).all(), f"user {u}: non-finite expanded K/V output"
        _, pcc = comp_pcc(absorbed.float(), expanded.float(), _EXPANDED_KV_MIN_PCC_VS_ABSORBED)
        ref_absorbed, ref_expanded = ref_pccs[False][u], ref_pccs[True][u]
        logger.info(
            f"user {u}: PCC vs CPU reference ring_mla {ref_absorbed:.6f} expanded {ref_expanded:.6f}; expanded vs "
            f"ring_mla {pcc:.6f}"
        )
        assert (
            ref_expanded >= ref_absorbed - max_loss
        ), f"user {u}: expanded K/V PCC {ref_expanded} < ring_mla {ref_absorbed} - {max_loss:.4f}"
        assert pcc >= _EXPANDED_KV_MIN_PCC_VS_ABSORBED, f"user {u}: expanded K/V vs ring_mla PCC {pcc}"


# 2k chunks (256 rows per chip; q128 head pairs): partial and rotated chunks, a preloaded
# prefix with a short chunk between full ones, and two users in their own slots.
_EXPANDED_K_2K_SCENARIOS = [
    ("c2k-maxedge", dict(iters_isl=[1024, 1056, 2048], chunk_size_global=2048)),
    ("c2k-prefix", dict(iters_isl=[2048, 1024, 2048], prefill_len=10240, chunk_size_global=2048)),
    ("c2k-2u", dict(iters_isl=[1024, 2048] * 2, num_users=2, chunk_size_global=2048)),
]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("kwargs", [pytest.param(kw, id=sid) for sid, kw in _EXPANDED_K_2K_SCENARIOS])
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.parametrize("tp_shard_kv", [False, True], ids=["sp_only", "tp_sharded"])
@pytest.mark.timeout(0)
def test_mla_expanded_k_matches_absorbed_2k(
    request, mesh_device, device_params, kwargs, variant, use_metadata_tensor, tp_shard_kv, monkeypatch
):
    """test_mla_expanded_k_matches_absorbed at 2k chunks, on the table's q128 schedule."""
    _check_expanded_k_vs_absorbed(
        request,
        mesh_device,
        device_params,
        variant,
        kwargs,
        use_metadata_tensor,
        tp_shard_kv,
        monkeypatch,
        q_chunk=128,
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "variant", ["kimi_k2_7", "kimi_k3", "mistral_small_4"], indirect=True, ids=["k2_7", "k3", "mistral4"]
)
@pytest.mark.parametrize("expanded_k", [None, False, True], ids=["default", "off", "on"])
@pytest.mark.timeout(0)
def test_mla_attention_selection(request, mesh_device, device_params, variant, expanded_k, monkeypatch, expect_error):
    """The dense chunked attention comes from mla_config.MLA_ATTENTION_CONFIG: Kimi runs expanded K/V by default, and
    expanded_k=False runs ring_mla; a model without an entry (Mistral) runs ring_mla, where expanded_k=True is
    refused."""
    ring_mla_calls = _record_ring_mla(monkeypatch)
    recorder = _SdpaRecorder(monkeypatch)
    run = functools.partial(
        _run_chunked_prefill,
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[5120],
        mla_kwargs=dict(expanded_k=expanded_k),
    )
    if variant.name == "mistral_small_4" and expanded_k:
        with expect_error(ValueError, "expanded K does not apply"):
            run()
        return
    run()
    if variant.name == "mistral_small_4" or expanded_k is False:
        assert ring_mla_calls and not recorder.calls, f"{variant.name} expanded_k={expanded_k} did not run ring_mla"
        return
    assert not ring_mla_calls, f"{variant.name} ran ring_mla"
    assert recorder.calls and all(c["expanded"] for c in recorder.calls), "expanded K/V not applied"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7"], indirect=True, ids=["k2_7"])
@pytest.mark.timeout(0)
def test_mla_attention_overrides(request, mesh_device, device_params, variant, monkeypatch, expect_error):
    """expanded_k=False puts Kimi on ring_mla, which a TP-deduped cache refuses (it runs expanded K/V only);
    expanded_k=True at a shape without a measured head-pair schedule (2.5k chunks: 320 rows per chip) is refused, where
    the model default would run ring_mla."""
    ring_mla_calls = _record_ring_mla(monkeypatch)
    recorder = _SdpaRecorder(monkeypatch)
    run = functools.partial(
        _run_chunked_prefill,
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
    )
    run(iters_isl=[5120], mla_kwargs=dict(expanded_k=False))
    assert ring_mla_calls and not recorder.calls, "expanded_k=False did not run ring_mla"
    with expect_error(AssertionError, "dense tp_shard_kv needs expanded K and V"):
        run(iters_isl=[5120], tp_shard_kv=True, mla_kwargs=dict(expanded_k=False))
    with expect_error(ValueError, "expanded K does not apply"):
        run(chunk_size_global=2560, iters_isl=[2560], mla_kwargs=dict(expanded_k=True))


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("expanded_k", [False, True], ids=["absorbed", "expanded_k"])
@pytest.mark.timeout(0)
def test_mla_expanded_kv_accuracy_floor(request, mesh_device, device_params, variant, expanded_k):
    """Layer output PCC vs the CPU reference at two 5k chunks after a 10k prefix (metadata path) is at least
    _EXPANDED_KV_MIN_PCC (expanded K/V) or _RING_MLA_MIN_PCC (ring_mla): tighter than the harness's own checks, so an
    accuracy loss in either attention shows here."""
    _, pccs = _run_chunked_prefill(
        request,
        mesh_device,
        reference="cpu",
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[5120, 5120],
        prefill_len=10240,
        use_metadata_tensor=True,
        return_outputs=True,
        return_pccs=True,
        mla_kwargs=dict(expanded_k=expanded_k),
    )
    floor = (_EXPANDED_KV_MIN_PCC if expanded_k else _RING_MLA_MIN_PCC)[variant.name]
    logger.info(f"{variant.name} expanded_k={expanded_k}: output PCC vs CPU reference {pccs[0]:.6f} (floor {floor})")
    assert pccs[0] >= floor, f"output PCC {pccs[0]} < {floor}"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param(kw, id=sid) for sid, kw in _CHUNKED_SCENARIOS if sid in ("rot-multislab", "rot-rot_partial")],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.timeout(0)
def test_mla_expanded_k_pairs_without_k_split(
    request, mesh_device, device_params, kwargs, variant, use_metadata_tensor, monkeypatch
):
    """test_mla_expanded_k_matches_absorbed's checks (sp_only) on a pair schedule the table does not pick: q320 / k256
    with no K split, so the pairs and their exchange run without bands on rotated chunked metadata."""
    chunks = (320, 256, 1)
    # Only the 640-row (5k chunk) entries: q320 gives pairs there, not at 256 rows.
    table = {key: chunks if key[3] == 640 else value for key, value in ttMLA._EXPANDED_K_SDPA_CHUNKS.items()}
    monkeypatch.setattr(ttMLA, "_EXPANDED_K_SDPA_CHUNKS", table)
    _check_expanded_k_vs_absorbed(
        request,
        mesh_device,
        device_params,
        variant,
        kwargs,
        use_metadata_tensor,
        False,
        monkeypatch,
        q_chunk=chunks[0],
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(_CHUNKED_SCENARIOS)["rot-multislab"], id="c5k-pairs"),
        pytest.param(dict(_EXPANDED_K_2K_SCENARIOS)["c2k-prefix"], id="c2k-pairs"),
    ],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.timeout(0)
def test_mla_expanded_k_lofi(request, mesh_device, device_params, kwargs, variant, monkeypatch):
    """The SDPA matmul fidelity override (sdpa_matmul_fidelity=LoFi) also sets the in-op expansion's fidelity: the
    expanded vs ring_mla checks of test_mla_expanded_k_matches_absorbed, with LoFi QK / PV (both modes) and LoFi
    expansion, on the table's head-pair schedules (4 x 2 expansion blocks) at 5k and 2k chunks."""
    init = ttMLA.__init__

    def init_lofi(self, *args, **init_kwargs):
        init(self, *args, **init_kwargs)
        self.sdpa_matmul_fidelity = ttnn.MathFidelity.LoFi

    monkeypatch.setattr(ttMLA, "__init__", init_lofi)
    _check_expanded_k_vs_absorbed(
        request,
        mesh_device,
        device_params,
        variant,
        kwargs,
        False,
        False,
        monkeypatch,
        fidelity=ttnn.MathFidelity.LoFi,
    )


# test_mla_expanded_k_distinct_layer_weights's steps, (user, start, valid tokens) over two users: chunk-aligned and
# chip-boundary (2560) rotated starts, and a partial chunk.
_DISTINCT_WEIGHTS_STEPS = [(0, 0, 5120), (1, 0, 2560), (0, 5120, 5120), (1, 2560, 5120)]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.timeout(0)
def test_mla_expanded_k_distinct_layer_weights(request, mesh_device, device_params, variant):
    """Expanded K/V's W_UK and W_UV^T are runtime-patched buffers. Each layer's kv_b is rotated by one head against the
    previous layer's, so a program reading another layer's weights would use a neighbouring head's: expanded vs
    ring_mla must agree per layer and step (expanded K/V on the TP-sharded cache, ring_mla on the replicated one,
    metadata path). On sp_only and one cache
    layer, layers that differ only in their weights must share every program (no new program-cache entry)."""
    config, weights = request.getfixturevalue("random_weights")
    sp_axis, tp_axis = 0, 1
    mesh_shape = tuple(mesh_device.shape)
    chunk = 5120
    seq_len_cache = 4 * chunk
    config.max_seq_len = seq_len_cache
    num_layers, num_users = 3, 2
    topology = per_axis_topology(device_params["fabric_config"])
    kv_b = weights["kv_b_proj.weight"]
    heads = config.num_attention_heads

    def layer_weights(layer):
        per_head = kv_b.reshape(heads, -1, kv_b.shape[-1])
        return {**weights, "kv_b_proj.weight": per_head.roll(layer, dims=0).reshape(kv_b.shape)}

    layers = [
        ttMLA(
            config,
            layer_weights(layer),
            mesh_device,
            layer_idx=layer,
            seq_len=seq_len_cache,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=False,
            topology=topology,
            is_chunked=True,
            active_seq_len=chunk,
            slot_num=num_users,
            layer_num=num_layers,
            tp_shard_kv=True,
            expanded_k=True,
        )
        for layer in range(num_layers)
    ]
    assert all(mla_layer._expanded_k for mla_layer in layers), "expanded K was not enabled for this shape"
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len_cache, chunk_size_global=chunk)
    scale_buf = rope_setup.make_llama4_scale_buffer(chunk)
    hidden_dims = [None, None]
    hidden_dims[tp_axis] = -1
    hidden_dims[sp_axis] = -2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_dims)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=hidden_dims, mesh_shape=mesh_device.shape)

    metadata = _new_chunk_metadata(mesh_device, chunk, scale_buf)

    def forward_layer(cache, step, layer, cache_layer_idx=None):
        user = _DISTINCT_WEIGHTS_STEPS[step][0]
        gen = torch.Generator().manual_seed(1000 * step + layer)
        hidden = ttnn.from_torch(
            torch.randn(1, 1, chunk, config.hidden_size, generator=gen, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        out = layers[layer].forward(
            hidden_states=hidden,
            rope_tensors=rope,
            kvpe_cache=cache,
            actual_start=None,
            actual_end=None,
            cache_layer_idx=layer if cache_layer_idx is None else cache_layer_idx,
            cache_user_id=user,
            metadata=metadata,
        )
        host = ttnn.to_torch(out, mesh_composer=composer).float()
        ttnn.deallocate(out)
        return host

    def set_step(step):
        user, start, valid = _DISTINCT_WEIGHTS_STEPS[step]
        write_chunk_metadata(
            metadata,
            (user, start, start + valid),
            hf_config=config,
            mesh_device=mesh_device,
            chunk_size_global=chunk,
            sp_axis=sp_axis,
        )

    outputs = {}
    for expanded in (False, True):
        # ring_mla runs on the TP-replicated cache, expanded K/V on the TP-deduped one.
        for mla_layer in layers:
            mla_layer._expanded_k = expanded
            mla_layer.set_tp_shard_kv(expanded)
        cache = _new_layered_kv_cache(
            config, mesh_device, seq_len_cache, num_layers, num_users, sp_axis, tp_axis, tp_sharded=expanded
        )
        outputs[expanded] = []
        for step in range(len(_DISTINCT_WEIGHTS_STEPS)):
            set_step(step)
            outputs[expanded].append([forward_layer(cache, step, layer) for layer in range(num_layers)])
        ttnn.deallocate(cache.storage)
    for step in range(len(outputs[True])):
        for layer in range(num_layers):
            _, pcc = comp_pcc(outputs[False][step][layer], outputs[True][step][layer], _EXPANDED_KV_MIN_PCC_VS_ABSORBED)
            logger.info(f"step {step} layer {layer}: expanded K/V vs ring_mla PCC {pcc:.6f}")
            assert pcc >= _EXPANDED_KV_MIN_PCC_VS_ABSORBED, f"step {step} layer {layer}: expanded vs ring_mla PCC {pcc}"

    # The layers really share programs: on sp_only and the same cache layer (the metadata path keys the
    # SDPA program by cache layer), once layer 0 has compiled its programs, layers 1 and 2 -- only their weights
    # differ -- add no program-cache entry, i.e. W_UK's and W_UV^T's addresses are patched, not hashed.
    for mla_layer in layers:
        mla_layer.set_tp_shard_kv(False)
        mla_layer._expanded_k = True
    cache = _new_layered_kv_cache(
        config, mesh_device, seq_len_cache, num_layers, num_users, sp_axis, tp_axis, tp_sharded=False
    )
    set_step(0)
    # Start cold, so layer 0 must compile (a reused mesh may already hold these programs).
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()
    before = mesh_device.num_program_cache_entries()
    forward_layer(cache, 0, 0)
    entries = mesh_device.num_program_cache_entries()
    assert entries > before, "the program cache did not record layer 0's sp_only programs"
    shared = {layer: forward_layer(cache, 0, layer, cache_layer_idx=0) for layer in (1, 2)}
    assert mesh_device.num_program_cache_entries() == entries, "a layer with different weights compiled new programs"
    # ... and the shared program used each layer's own W_UK and W_UV^T (step 0 writes only its own chunk, so each
    # run's inputs match).
    for layer in (1, 2):
        layers[layer]._expanded_k = False
        absorbed = forward_layer(cache, 0, layer, cache_layer_idx=0)
        _, pcc = comp_pcc(absorbed, shared[layer], _EXPANDED_KV_MIN_PCC_VS_ABSORBED)
        assert (
            pcc >= _EXPANDED_KV_MIN_PCC_VS_ABSORBED
        ), f"layer {layer} on a shared program: expanded vs ring_mla PCC {pcc}"
    ttnn.deallocate(cache.storage)


def _sp_gathered(per_device, sp, tp, t, take):
    """take() of TP group t's tensor on every SP chip, concatenated in SP order along dim -2 (natural token order at
    position 0)."""
    return torch.cat([take(per_device[s * tp + t]) for s in range(sp)], dim=-2)


# Expanded K/V SDPA output per (query chip, head) vs fp32 attention on the op's own inputs: the in-op V expansion
# accumulates in bf16 (see _EXPANDED_KV_MIN_PCC), which bounds this below fp32-accumulation accuracy.
_EXPANDED_KV_MIN_HEAD_PCC = 0.99


def _per_head_attention_reference(call, sp, tp, t, rows, keep):
    """Yield (head, fp32 attention output) for TP group t from a recorded expanded K/V SDPA call's per-device inputs,
    gathered over the SP chips' first `rows` rows: K_h = [c_kv @ W_UK_h^T | k_pe] and V_h = c_kv @ (W_UV^T_h)^T, c_kv
    the latent's first kv_lora_rank columns. `keep` is the [query, key] mask of scores kept, so the caller owns
    causality and pad rows."""
    k_latent = _sp_gathered(call["k"], sp, tp, t, lambda x: x[0, 0, :rows])  # [N, 576], this TP device's copy
    q = _sp_gathered(call["q"], sp, tp, t, lambda x: x[0, :, :rows])  # [heads, N, D]
    for h in range(q.shape[0]):
        w_uk = call["w"][t][0, h]  # [qk_nope_head_dim, kv_lora_rank]
        w_uv_t = call["w_uv"][t][0, h]  # [v_head_dim, kv_lora_rank]
        c_kv = k_latent[:, : w_uk.shape[1]]
        k = torch.cat([c_kv @ w_uk.T, k_latent[:, w_uk.shape[1] :]], dim=-1)
        scores = ((q[h] @ k.T) * call["scale"]).masked_fill(~keep, float("-inf"))
        yield h, torch.softmax(scores, dim=-1) @ (c_kv @ w_uv_t.T)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.parametrize("tp_shard_kv", [False, True], ids=["sp_only", "tp_sharded"])
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.parametrize("valid", [5120, 2560], ids=["full", "partial"])
@pytest.mark.timeout(0)
def test_ring_joint_expanded_k_per_head_vs_reference(
    request, mesh_device, device_params, variant, tp_shard_kv, use_metadata_tensor, valid, monkeypatch
):
    """The expanded K/V ring_joint SDPA output per head against fp32 attention on the device's own inputs (K and V
    expanded from the recorded latent, Q un-absorbed), for one causal chunk at position 0 (natural token order), full
    or partial (a partial chunk adds unwritten pad rows in the cache and buffers). PCC is checked per (query chip,
    head): output norms fall with the row index, so one PCC over all rows would be dominated by chip 0."""
    recorder = _SdpaRecorder(monkeypatch, keep_inputs=True)
    _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[valid],
        prefill_len=0,
        use_metadata_tensor=use_metadata_tensor,
        tp_shard_kv=tp_shard_kv,
        mla_kwargs=dict(expanded_k=True),
    )
    (call,) = recorder.calls
    assert call["expanded"] and call["v"] is None, "expanded K/V not applied"
    sp, tp = tuple(mesh_device.shape)
    rows = 5120 // sp

    keep = torch.ones(5120, 5120, dtype=torch.bool).tril()
    keep[:, valid:] = False
    worst, worst_err = 1.0, 0.0
    for t in range(tp):
        out = _sp_gathered(call["out"], sp, tp, t, lambda x: x[0, :, :rows])
        for h, ref in _per_head_attention_reference(call, sp, tp, t, rows, keep):
            assert torch.isfinite(ref[:valid]).all(), f"TP group {t} head {h}: non-finite reference"
            assert torch.isfinite(out[h, :valid]).all(), f"TP group {t} head {h}: non-finite SDPA output"
            for s in range(sp):
                lo, hi = s * rows, min((s + 1) * rows, valid)
                if lo >= hi:
                    continue
                _, pcc = comp_pcc(ref[lo:hi], out[h, lo:hi])
                worst = min(worst, pcc)
                worst_err = max(worst_err, (ref[lo:hi] - out[h, lo:hi]).abs().max().item())
                assert (
                    pcc >= _EXPANDED_KV_MIN_HEAD_PCC
                ), f"TP group {t} head {h} query chip {s}: SDPA PCC {pcc} vs fp32 reference"
    logger.success(
        f"✓ expanded K/V SDPA per (query chip, head) vs fp32 reference: min PCC {worst:.6f}, max abs error "
        f"{worst_err:.4f}"
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7", "kimi_k3"], indirect=True, ids=["k2_7", "k3"])
@pytest.mark.timeout(0)
def test_ring_joint_expanded_k_causal_vs_reference(
    request, mesh_device, device_params, variant, monkeypatch, expect_error
):
    """The non-chunked causal expanded K/V ring_joint SDPA (Q and K one chunk each), per (query chip, head) against fp32
    attention. It skips K chunks above the diagonal per Q chunk, so in head pairs one core of a pair skips K chunks its
    peer needs and must still expand and exchange them. Inputs are recorded from a chunk-0 MLA forward and replayed on
    the op directly. Also checks that the op refuses schedules without head pairs."""
    sp, tp = tuple(mesh_device.shape)
    rows = 5120 // sp
    # Pin the schedule this test exists for: q320 gives two Q chunks per head (a pair), one Q chunk per core.
    monkeypatch.setattr(ttMLA, "_EXPANDED_K_SDPA_CHUNKS", {key: (320, 256, 1) for key in ttMLA._EXPANDED_K_SDPA_CHUNKS})
    recorder = _SdpaRecorder(monkeypatch, keep_inputs=True, keep_device_rows=rows)
    _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[5120],
        prefill_len=0,
        mla_kwargs=dict(expanded_k=True),
    )
    (call,) = recorder.calls
    assert call["expanded"], "expanded K/V not applied"
    dev, kw = call["device"], call["device"]["kwargs"]
    # Pair prerequisites (the factory's head pairs): two Q chunks per head, at most one Q chunk per core.
    grid = mesh_device.compute_with_storage_grid_size()
    units = dev["q"].shape[1] * (rows // call["q_chunk"])
    assert call["q_chunk"] == 320 and rows // call["q_chunk"] == 2, f"q chunk {call['q_chunk']}: no head pairs"
    assert units <= (grid.x - 1) * grid.y, f"{units} (head, Q chunk) units: more than one per core, no pairs"
    tt_ccl = get_tt_ccl(mesh_device)
    sdpa = ttnn.transformer.ring_joint_scaled_dot_product_attention._recorded_op

    def run(program_config):
        return sdpa(
            dev["q"],
            dev["k"],
            None,
            persistent_output_buffer_k=tt_ccl.get_mla_chunked_kv_buffer(
                cache_batch=1, seq_len=5120, kvpe_dim=dev["k"].shape[3]
            ),
            logical_n=5120,
            program_config=program_config,
            compute_kernel_config=kw["compute_kernel_config"],
            dim=2,
            multi_device_global_semaphore=kw["multi_device_global_semaphore"],
            num_links=kw["num_links"],
            cluster_axis=kw["cluster_axis"],
            mesh_device=mesh_device,
            topology=kw["topology"],
            ccl_core_grid_offset=kw["ccl_core_grid_offset"],
            use_column_major_ccl=True,
            is_causal=True,
            scale=call["scale"],
            is_balanced=False,
            k_expand_weight=kw["k_expand_weight"],
            v_expand_weight=kw["v_expand_weight"],
        )

    out, _, _ = run(kw["program_config"])
    out = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)]

    keep = torch.ones(5120, 5120, dtype=torch.bool).tril()
    worst = 1.0
    for t in range(tp):
        got = _sp_gathered(out, sp, tp, t, lambda x: x[0, :, :rows])
        for h, ref in _per_head_attention_reference(call, sp, tp, t, rows, keep):
            assert torch.isfinite(got[h]).all(), f"TP group {t} head {h}: non-finite SDPA output"
            for s in range(sp):
                _, pcc = comp_pcc(ref[s * rows : (s + 1) * rows], got[h, s * rows : (s + 1) * rows])
                worst = min(worst, pcc)
                assert pcc >= _EXPANDED_KV_MIN_HEAD_PCC, f"TP group {t} head {h} query chip {s}: causal SDPA PCC {pcc}"
    logger.success(f"✓ non-chunked causal expanded K/V SDPA per (query chip, head) vs fp32: min PCC {worst:.6f}")
    # Expanded K runs on head pairs only: the op refuses, before launching anything, a schedule without them (q160: four
    # Q chunks per head; a 4x4 grid: more than one Q chunk per core) and segmented accumulation.
    base = kw["program_config"]
    small = ttnn.CoreCoord(4, 4)
    for grid_size, q_chunk_size, segmented, error in (
        (base.compute_with_storage_grid_size, 160, False, "needs head pairs"),
        (small, 320, False, "needs head pairs"),
        (base.compute_with_storage_grid_size, 320, True, "segmented_accumulation"),
    ):
        config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=base.k_chunk_size,
            exp_approx_mode=False,
            matmul_math_fidelity=base.matmul_math_fidelity,
            segmented_accumulation=segmented,
        )
        with expect_error(RuntimeError, error):
            run(config)
