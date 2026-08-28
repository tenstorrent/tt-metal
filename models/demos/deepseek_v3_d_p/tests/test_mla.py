# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc, hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
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
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.chunked_prefill_utils import (
    cpu_mla_reference,
    load_trace,
    partition_iters,
    resolve_traces,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.utils_for_testing import assert_with_pcc

_WORKER_L1_SIZE = ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE


def _local_fabric2d_params():
    return fabric2d_device_params(worker_l1_size=_WORKER_L1_SIZE)


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

    Returns:
        Tuple of (tt_output, hidden_states, chunk_order, shard_dims)
    """
    # Create TT MLA
    logger.info("Creating TT MLA...")

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
        # Match the single-layer test cache (num_kvpe_cache_layers=1): the sparse single-shot write now
        # goes through update_padded_kv_cache, which asserts cache_batch % layer_num == 0. Dense is
        # unaffected (its single-shot write uses fill_cache_for_user_, which ignores layer_num).
        layer_num=1,
        sparse_kv_cache_format=tt_kvpe_cache.format,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    # Sparse (DSA) single-shot is folded onto the block-cyclic path (one full-seq chunk at offset 0):
    # it uses the indexed rope tables and a caller-owned indexer key cache, exactly like the chunked
    # path. Dense keeps natural rope + no index cache.
    has_indexer = resolve_has_indexer(config)
    index_kv_cache = None
    if has_indexer:
        rope_tensors = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=seq_len)
        # Layer-slot count mirrors the serving adapter: the indexer strides the folded user-major cache by
        # num_full_indexer_layers (GLM-5.2 cross-layer reuse), so the cache must carry that many slots for
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
    # GLM-5.2 indexer reuse (return_indices / inject_indices): capture this layer's top-k selection, or
    # feed a prior layer's to skip the indexer. Defaults leave the single-shot forward unchanged.
    mla_out = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
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


def _ci_unsupported_param_combos(**params):
    on_ci = params["is_ci_env"] or params["is_ci_v2_env"]
    is_balanced = params["is_balanced"]

    if not on_ci:
        return False

    if not is_balanced:
        return True
    return False


# sp x tp
@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        # Multi-host 32x4 is a four-Galaxy scale-out diagnostic. There is no certified descriptor
        # that closes this entire logical mesh into one XY torus, so it remains unwrapped Fabric2D.
        pytest.param((32, 4), _local_fabric2d_params(), id="fabric2d-32x4"),
        pytest.param((8, 4), torus_xy_device_params(worker_l1_size=_WORKER_L1_SIZE), id="torus-xy-8x4"),
        pytest.param((2, 4), _local_fabric2d_params(), id="fabric2d-2x4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("scale_down_sl", [False, True], ids=["max_sl", "scaled_sl"])
@pytest.mark.parametrize(
    "seq_len",
    [PREFILL_CHUNK_TOKENS],
    ids=["seq5k"],
)
@pytest.mark.parametrize("skip_host_comparison", [False, True], ids=["check_pcc", "skip_check"])
@pytest.mark.parametrize("is_balanced", [False, True], ids=["sequential", "balanced"])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_ds_mla(
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


# sp x tp -- Mistral Small 4 bringup. Modelled on test_ds_mla above and deliberately narrowed;
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
            "worker_l1_size": _WORKER_L1_SIZE,
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
# Unified chunked-prefill driver. One loop (preload -> N iters of write+rope+ring_mla -> compare)
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
# subprocess, no signposts and no ops-CSV re-parse -- it runs on the plain build (PR #49840).
# Measured 2026-08-05 on bh_sc1_high_power (run 31010521345): 12.073 ms. Reads 4.4% above the Tracy
# path's 11_562_468 as expected -- Tracy averages collectives across chips, this takes the max.
K3_CHUNKED_RT_PERF_NS = 12_073_303
K3_CHUNKED_RT_PERF_MARGIN = 0.03


def _rt_profile_forward_ns(mesh_device, run_fn):
    """Profile one region; return (result, total_ns) where each program contributes its MAX duration
    across chips (slowest chip gates that program) -- the sparse-MLA/PR #49840 convention."""
    result, records = profile_realtime_program(mesh_device, run_fn, collect_all=True)
    per_program = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:  # 0 is the profiler's own sentinel
            continue
        duration_ns = float(record["duration_ns"])
        per_program[runtime_id] = max(per_program.get(runtime_id, 0.0), duration_ns)
    return result, sum(per_program.values())


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
    for g in groups:
        ka = prefill_len
        for v in g:
            max_window = max(max_window, ka + chunk_size_global)
            ka += v
    seq_len_cache = ((max_window + chunk_size_global - 1) // chunk_size_global) * chunk_size_global

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
            hidden = torch.randn(total_len, hidden_size, dtype=torch.bfloat16)
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
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
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
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2
    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2

    # ---- preload the prior prefix (trace or random) into each slot, block-cyclic ----
    if prefill_len > 0:
        logger.info(f"Preloading {prefill_len}-token prefix into {num_users} slot(s) (block-cyclic host->device)...")
        cache_host = torch.zeros(num_users, 1, seq_len_cache, kvpe_dim, dtype=torch.bfloat16)
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
            cache_host[u, 0] = blockcyclic_cache_host(kv_prior, sp, chunk_size_global, seq_len_cache, kvpe_dim)[0, 0]
        cache_host_tt = ttnn.from_torch(
            cache_host,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
        )
        ttnn.copy_host_to_device_tensor(cache_host_tt, tt_kvpe_cache.storage)
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
            # verbatim -- ttMLA threads it to all chunked ops (update/rope/zero_pad/ring_mla), which read
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
                        actual_start=None if use_metadata_tensor else kv_actual,
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
                # Scalars only: field 3 is a persistent buffer allocated outside this loop, so a
                # blanket deallocate would free it before the next chunk reads it.
                for meta_tensor in kv_pad_metadata[:3]:
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
        cache_sr = ttnn.to_torch(
            tt_kvpe_cache.storage,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).to(torch.float32)[
            :, :1
        ]  # TP replica 0 -> [num_users, 1, seq_cache, kvpe]
        p = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
        kv_lora = config.kv_lora_rank
        d = kvpe_dim - kv_lora
        for u in range(num_users):
            if users[u]["kv_post"] is None:
                continue
            nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
            nat[p] = cache_sr[u, 0]
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


# Functionality scenarios (id, kwargs) -- PURE FUNCTIONALITY: no mesh, no reference. Mesh and
# reference are SEPARATE pytest axes below (chunk=5120 is valid for sp in {2,4,8}), so the same
# scenario runs on any mesh and is validated against either ground truth (or run functional) without
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

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    shard_dims[tp_axis] = 1
    width = config.kv_lora_rank + config.qk_rope_head_dim
    q = ttnn.from_torch(
        torch.ones(1, config.num_attention_heads, chunk_size_global, width, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
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
        pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("reference", ["cpu", "trace", None], ids=["cpu", "trace", "func"])
@pytest.mark.parametrize("kwargs", [kw for _, kw in _CHUNKED_SCENARIOS], ids=[sid for sid, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "variant",
    ["kimi_k2_6", "kimi_k3", "mistral_small_4"],
    indirect=True,
    # Name the Kimi generation explicitly. pytest -k is substring-based, so the ids must stay
    # disjoint: "k2_6" and "k3" cannot cross-match, whereas a bare "kimi" id would match both
    # generations and silently widen every `-k` selector (CI yaml, tests/perf/test_mla_perf.py).
    # "mistral4" matches the transformer/block rows' id, so one `-k mistral4` selects this model
    # across all three; `-k mistral` still finds it, and no other variant here shares the prefix.
    ids=["k2_6", "k3", "mistral4"],
)
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.timeout(0)
def test_mla_chunked_prefill(
    request, mesh_device, kwargs, reference, device_params, variant, use_metadata_tensor, determinism_check
):
    """Unified chunked-prefill driver crossed with independent mesh and reference axes. Each
    functionality scenario (rotation edges, production depth, multi-user, deep prefix) runs on any mesh
    and is validated against the CPU torch reference ('cpu'), the GPU trace ('trace', asserts if the
    variant has no registered trace), or run with no reference ('func'). Select with e.g.
    -k 'maxedge-1u and trace and 8x4'. See _run_chunked_prefill.

    Real weights on the CPU-reference path: point the variant's HF env var (KIMI_K2_6_HF_MODEL /
    KIMI_K3_HF_MODEL) at a checkpoint to validate the chunked path against the CPU torch reference
    with pretrained weights instead of random. create_mla_reference is config-driven and
    architecture-agnostic (Kimi's YaRN/theta flow through, absorbed-MLA math matches the variant's own
    reference), so this works for both variants. It complements the GPU-trace path, which only
    replays full-chunk iters and so never exercises real weights across the rotation/partial-chunk edge
    scenarios that the cpu path covers. Without the env var, fall back to random. kimi_k2_6 runs the
    trace path (loader + k_pe re-interleave are arch-agnostic) against its own registered traces. It
    otherwise runs the same config-driven driver on any arch/mesh.

    kimi_k3 (NoPE + output gate, 96 heads) runs 'scalar' only -- 'metadata' is skipped explicitly
    below. It runs 'trace' like kimi_k2_6, taking real weights from layer 3 via
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
    # No K3 checkpoint is reachable, so no GPU trace was ever recorded for it. _run_chunked_prefill
    # already asserts on supports_pretrained, but only once a trace root is configured -- so on a box
    # with MLA_CHUNKED_TRACE_DIR set these cases would hard-fail instead of being cleanly out of
    # scope. Skip up front; the assert stays as the backstop for any future supports_pretrained=False
    # variant and for the silent K2.6-trace-substitution it was written to catch.
    if variant.name == "kimi_k3" and reference == "trace":
        pytest.skip("kimi_k3 has no reachable checkpoint, so no GPU trace exists for it")
    # Same reason as K3's: the variant-unqualified CI selectors for this test would otherwise run
    # Mistral on Wormhole T3K, where it has never been brought up.
    if variant.name == "mistral_small_4" and not is_blackhole():
        pytest.skip("mistral_small_4 is validated on Blackhole only")
    # No GPU trace has ever been recorded for Mistral and none is required for CI; 'cpu' and 'func'
    # cover all 13 scenarios between them. Skipped up front for K3's reason: the supports_pretrained
    # assert inside _run_chunked_prefill only fires once a trace root is configured, so on a box with
    # MLA_CHUNKED_TRACE_DIR set these would hard-fail instead of reading as out of scope.
    if variant.name == "mistral_small_4" and reference == "trace":
        pytest.skip("no GPU trace recorded for mistral_small_4; cpu and func cover all 13 scenarios")
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
        **kwargs,
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
@pytest.mark.skipif(
    not is_high_power(),
    reason="perf job requires a high-power (>=130W TDP) host; guards the exabox.tenstorrent.com/power=14kw label",
)
@pytest.mark.timeout(0)
def test_mla_chunked_perf_check(request, mesh_device, device_params, variant):
    """Kimi-K3 chunked-prefill MLA perf on the 8x4 Galaxy: 50k cached prefix + one fresh 5k chunk,
    timed with the realtime (lightweight) profiler instead of Tracy.

    NOT comparable to test_kimi_k3_mla_chunked_perf_galaxy's 11_562_468: that number comes from the
    Tracy merge path, which averages collectives across chips, while this takes the max for every
    program. K3's forward is ~7% CCL, so the two disagree by construction."""
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
    assert lower <= total_ns <= upper, (
        f"device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {K3_CHUNKED_RT_PERF_NS:,} ns, margin +/- {K3_CHUNKED_RT_PERF_MARGIN * 100:.1f}%)"
    )
