# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.reference_runners import run_reference_mla
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

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
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
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
    tt_output = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
    )

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

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
        config, sd = request.getfixturevalue("pretrained_transformer_weights")
        weights = sd["layers"][0]["mla_weights"]
    else:
        config, weights = request.getfixturevalue("random_weights")

    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear

    production_mesh = [32, 4]
    sp_axis = 0
    tp_axis = 1

    mesh_shape = list(mesh_device.shape)

    if scale_down_sl:
        seq_len = (seq_len // production_mesh[sp_axis]) * mesh_shape[sp_axis]

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
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank  # 576
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
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

            ref_kvpe = ref_cache.key_cache[0]  # layer 0

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
            tt_kvpe_cache,
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


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4), (2, 4)],
    ids=["32x4", "8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
    ],
    ids=["line", "ring", "fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("scale_down_sl", [False, True], ids=["max_sl", "scaled_sl"])
@pytest.mark.parametrize("seq_len", [128 * 1024, 100 * 1024], ids=["seq128k", "seq100k"])
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


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
    ],
    ids=["line", "ring"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"])
@pytest.mark.parametrize("scale_down_sl", [False], ids=["max_sl"])
@pytest.mark.parametrize(
    "seq_len",
    [5 * 1024, 25 * 1024],
    ids=["seq5k", "seq25k"],
)
@pytest.mark.parametrize("skip_host_comparison", [False, True], ids=["check_pcc", "skip_check"])
@pytest.mark.parametrize("is_balanced", [False], ids=["sequential"])
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_mla(
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


# Each entry is a list of per-iteration valid token counts (sp=2, chunk_size_global=5120,
# chunk_local=2560); the only requirement is tile-alignment and fitting the cache. The rope helper
# + SDPA rotation are slab-aware, so kv_actual_isl may span multiple slabs.
ROTATED_VALID_LISTS = [
    [2560, 5120],  # iter0partial — original 2-iter case (only iter 0 padded)
    [5120, 5120],  # allfull — both chunks full, rotation degenerates (sanity)
    [2560, 2560, 5120],  # iter1rotpartial — iter 1 is BOTH rotated AND partial
    [1280, 3840, 5120],  # iter1multichippad — iter 1 pad-fill spans both chips
    [2560, 2560],  # lastpartial — final iter partial, exactly fills the OLD slab
    [2592, 5120, 5120],  # padded_partial — whole-tile (1-tile) offset so the boundary chip's
    # write straddles a slab, and kv_actual spans multiple slabs
]
ROTATED_VALID_IDS = ["iter0partial", "allfull", "iter1rotpartial", "iter1multichippad", "lastpartial", "padded_partial"]


def _run_chunked_prefill_scenario(request, mesh_device, valid_per_iter):
    """Run an N-iteration chunked-prefill scenario end-to-end and PCC-check it.

    Drives ttMLA.forward's chunked path: update_padded_kv_cache (per-chip write),
    rotary_embedding_indexed (whole-cache cos/sin), and ring_mla (latent-V streaming) with
    kv_actual_isl rotation; wkv_b2 is applied post-attention inside _chunked_attn. Each iteration's
    valid token count drives which iterations are padded and whether the prefix lands mid-slab
    (rotation). Output is un-rotated to natural order and compared per-iter + full-prefix to a torch
    MLA reference (chunked flash attention, so it scales to production cache sizes).
    """
    config, weights = request.getfixturevalue("random_weights")

    sp_axis = 0
    tp_axis = 1
    is_balanced = False
    topology = ttnn.Topology.Linear

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]

    # Fixed scope for this rotation test.
    chunk_size_global = 5120
    chunk_local = chunk_size_global // sp  # 2560
    tile = ttnn.TILE_SIZE

    # Validate the parametrization against the rope helper's supported regime and
    # derive the cumulative kv_actual_isl entering each iteration.
    kv_actuals = []
    prefix = 0
    for v in valid_per_iter:
        assert (
            v % tile == 0 and 0 < v <= chunk_size_global
        ), f"valid count {v} must be tile-aligned and <= {chunk_size_global}"
        assert prefix % tile == 0, f"kv_actual {prefix} must be tile-aligned"
        kv_actuals.append(prefix)
        prefix += v
    valid_total = sum(valid_per_iter)
    # Cache must hold the highest (kv_actual + chunk_size_global) window; round up
    # to slab granularity. At least 2 slabs (the documented MLA rotation scope).
    needed = max(ka + chunk_size_global for ka in kv_actuals)
    seq_len_cache = max(
        chunk_size_global * 2, ((needed + chunk_size_global - 1) // chunk_size_global) * chunk_size_global
    )
    config.max_seq_len = seq_len_cache

    logger.info(
        f"Rotated chunked prefill: sp={sp}, chunk_size_global={chunk_size_global}, "
        f"valid_per_iter={valid_per_iter}, kv_actuals={kv_actuals}, valid_total={valid_total}, "
        f"seq_len_cache={seq_len_cache}"
    )

    # Reference: torch MLA forward over the VALID cumulative prefix (length valid_total).
    mla_ref = create_mla_reference(
        config=config,
        state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    )
    mla_ref = mla_ref.eval().to(torch.bfloat16)

    torch.manual_seed(42)
    hidden_states_valid = torch.randn(1, valid_total, config.hidden_size).to(torch.bfloat16)
    position_ids = torch.arange(valid_total, dtype=torch.long).unsqueeze(0)

    ref_cache = DynamicCache()
    logger.warning(
        f"===== HOST ATTENTION START: torch MLA reference over {valid_total} tokens "
        f"(CPU chunked-flash, {config.num_attention_heads} heads) -- this is the slow CPU phase ====="
    )
    _host_attn_t0 = time.perf_counter()
    with torch.no_grad():
        ref_output, _, _ = mla_ref(
            hidden_states=hidden_states_valid,
            position_ids=position_ids,
            past_key_value=ref_cache,
            use_cache=True,
        )
    logger.warning(
        f"===== HOST ATTENTION END: torch reference done in {time.perf_counter() - _host_attn_t0:.1f}s "
        f"(ref_output {tuple(ref_output.shape)}) ====="
    )
    # ref_output: [1, valid_total, hidden_size]

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
        is_chunked=True,
        slot_num=1,
        layer_num=1,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2

    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2

    def _to_tt_hidden(host_tensor):
        # host_tensor: [1, 1, chunk_size_global, hidden_size] in chip-concat order
        # (first chunk_local rows → chip 0, next chunk_local rows → chip 1).
        return ttnn.from_torch(
            host_tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
            ),
        )

    # Indexed rotated path: build the whole-cache cos/sin ONCE (block-cyclic-reordered keyed by the
    # per-chip chunk, then SP-sharded). MLA.forward's _apply_rope feeds these to
    # rotary_embedding_indexed, which derives each chunk's per-chip shard offset on-device from
    # kv_actual_global -- so the same tensors are reused for every iteration and only
    # kv_actual_isl changes.
    indexed_rope = rope_setup.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )
    mesh_device.enable_program_cache()

    # Accumulated natural-order output across all iterations, filled per iter by
    # un-rotating each chunk's valid rows back to their global positions.
    full_out_natural = torch.zeros(1, 1, valid_total, config.hidden_size, dtype=torch.bfloat16)

    logger.info(f"Starting DEVICE chunked prefill: {len(valid_per_iter)} iterations on {tuple(mesh_device.shape)}")
    for it, (kv_actual, valid) in enumerate(zip(kv_actuals, valid_per_iter)):
        valid_end = kv_actual + valid  # exclusive global position of last valid new token + 1
        logger.info(
            f"[device iter {it + 1}/{len(valid_per_iter)}] kv_actual={kv_actual} valid={valid} "
            f"(ring_mla logical_n={kv_actual + chunk_size_global})..."
        )

        # Per chip-local row (chip-concat order: chip 0 rows then chip 1 rows),
        # the global position it carries after the server rotation.
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat_positions = [positions[c][r] for c in range(sp) for r in range(chunk_local)]  # len chunk_size_global

        # Input: gather each row's hidden state from its global position; rows whose
        # position is >= valid_end are logical pad → zeros (don't-care, masked out).
        gather_idx = torch.tensor([min(gp, valid_total - 1) for gp in flat_positions], dtype=torch.long)
        chunk_in = hidden_states_valid[0, gather_idx, :].clone()  # [chunk_size_global, hidden]
        pad_mask = torch.tensor([gp >= valid_end for gp in flat_positions])
        chunk_in[pad_mask] = 0.0
        iter_h = chunk_in.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size_global, hidden]

        tt_iter_h = _to_tt_hidden(iter_h)

        tt_iter_out = mla_tt.forward(
            hidden_states=tt_iter_h,
            rope_tensors=indexed_rope,
            kvpe_cache=tt_kvpe_cache,
            kv_actual_isl=kv_actual,
        )

        out_host = ttnn.to_torch(
            tt_iter_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)

        # Un-rotate: output row (c*chunk_local + r) holds global position flat_positions[row].
        # Scatter the valid rows into natural order.
        out_flat = out_host[0, 0]  # [chunk_size_global, hidden]
        valid_pairs = [(row, gp) for row, gp in enumerate(flat_positions) if gp < valid_end]
        src_rows = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
        dst_pos = torch.tensor([gp for _, gp in valid_pairs], dtype=torch.long)
        full_out_natural[0, 0, dst_pos, :] = out_flat[src_rows, :]

        iter_natural = full_out_natural[:, :, kv_actual:valid_end, :]
        _, msg = assert_with_pcc(
            ref_output[:, kv_actual:valid_end, :].unsqueeze(0),
            iter_natural,
            0.98,
        )
        rot = "rotated" if kv_actual % chunk_size_global != 0 else "aligned"  # mid-slab (any slab) => rotated
        logger.info(f"  iter {it} (valid {kv_actual}..{valid_end - 1}, kv_actual={kv_actual}, {rot}): PCC {msg}")

        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()

    # Full assembled-prefix correctness across all iterations.
    _, msg = assert_with_pcc(ref_output.unsqueeze(0), full_out_natural, 0.98)
    logger.info(f"  full prefix (0..{valid_total - 1}) output PCC {msg}")
    logger.success(f"✓ Chunked prefill scenario passed (valid_per_iter={valid_per_iter})")


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4), (8, 4)], ids=["2x2", "2x4", "8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("valid_per_iter", ROTATED_VALID_LISTS, ids=ROTATED_VALID_IDS)
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_rotated_partial(request, mesh_device, valid_per_iter, device_params):
    """N-iter KV-pad-aware rotation: per-iteration valid token counts drive which iteration(s) are
    padded; rotation is exercised whenever a prefix lands mid-slab. Small-scale correctness sweep.
    """
    _run_chunked_prefill_scenario(request, mesh_device, valid_per_iter)


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_production(request, mesh_device, device_params):
    """Production-scale chunked prefill on 8x4: a 50k-deep KV cache plus one 5k new chunk.

    Ten full 5k chunks fill the cache to 50k (real KV written by ttMLA.forward, so the device cache
    matches the torch reference), then the 11th chunk is the production step at kv_actual=51200 (50k).
    All chunks are chunk-aligned (steady state). Exercises ring_mla's full ~56k-wide streaming
    attention at production depth and PCC-checks every step (including the 50k+5k step) against the
    chunked-flash torch reference. The torch reference is the slow CPU phase -- bracketed by
    HOST ATTENTION START/END logs.
    """
    _run_chunked_prefill_scenario(request, mesh_device, [5120] * 11)


# Layer-0 MLA traces from a 56320-token single-shot GPU prefill (deepseek_math). Set this env var to
# the model dir, e.g. .../armla_sdpa_traces/deepseek_math_56320_sdpa_mla. We only need 3 tensors:
#   mla_io/mla_input_layer_0.safetensors[mla_input_layer_0]    [56320, 7168]  -> new-chunk hidden in
#   kv_cache/layer_0.safetensors[kv_post_transform_layer_0]    [56320, 576]   -> prior KV to preload
#   mla_io/mla_output_layer_0.safetensors[mla_output_layer_0]  [56320, 7168]  -> reference output
DEEPSEEK_MLA_TRACE_DIR = os.environ.get("DEEPSEEK_MLA_TRACE_DIR")


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.skipif(DEEPSEEK_MLA_TRACE_DIR is None, reason="set DEEPSEEK_MLA_TRACE_DIR to the layer-0 trace dir")
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_trace_layer0(request, mesh_device, device_params, variant):
    """Validate the 11th chunked-prefill step against real GPU traces on 8x4.

    Loads layer-0 MLA traces from a 56320-token single-shot GPU prefill: preloads the first 50k of
    the GPU KV cache onto the device (block-cyclic slab layout, host->device copy), then runs ONE
    chunked-prefill forward over the last 5k of the GPU MLA input at kv_actual=51200 (chunk-aligned)
    with the SAME pretrained weights test_ds_mla uses, and PCC-checks the output against the GPU MLA
    output for those 5k tokens. No torch attention -- the reference is the GPU trace itself.
    """
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]

    chunk_size_global = 5120  # 5k new ISL
    prior_len = 50 * 1024  # 50k already in the GPU cache
    kv_actual_isl = prior_len  # chunk-aligned 11th step
    seq_len_cache = prior_len + chunk_size_global  # 56320
    assert kv_actual_isl % chunk_size_global == 0, "11th step must be chunk-aligned"

    trace = Path(DEEPSEEK_MLA_TRACE_DIR)
    logger.info(f"[1/6] Loading layer-0 traces from {trace} ...")
    mla_input = load_file(trace / "mla_io" / "mla_input_layer_0.safetensors")["mla_input_layer_0"]
    mla_output = load_file(trace / "mla_io" / "mla_output_layer_0.safetensors")["mla_output_layer_0"]
    kv_post = load_file(trace / "kv_cache" / "layer_0.safetensors")["kv_post_transform_layer_0"]
    logger.info(f"[1/6] mla_input {list(mla_input.shape)} mla_output {list(mla_output.shape)} kv {list(kv_post.shape)}")
    assert mla_input.shape[0] >= seq_len_cache, f"trace seq {mla_input.shape[0]} < {seq_len_cache}"

    logger.info("[2/6] Loading pretrained layer-0 weights (same path as test_ds_mla)...")
    config, sd = request.getfixturevalue("pretrained_transformer_weights")
    weights = sd["layers"][0]["mla_weights"]
    config.max_seq_len = seq_len_cache
    kvpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
    hidden_size = config.hidden_size
    assert kv_post.shape[1] == kvpe_dim, f"kv_post kvpe {kv_post.shape[1]} != {kvpe_dim}"

    logger.info("[3/6] Building ttMLA (is_chunked=True) + indexed RoPE...")
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
        slot_num=1,
        layer_num=1,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )

    logger.info(f"[4/6] Preloading first {prior_len} GPU KV rows into the device cache (block-cyclic)...")
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )
    cache_host = blockcyclic_cache_host(
        kv_post[:prior_len].to(torch.bfloat16), sp, chunk_size_global, seq_len_cache, kvpe_dim
    )
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2  # SP shards the global seq; TP replicated
    cache_host_tt = ttnn.from_torch(
        cache_host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
    )
    # Init allocates the cache with the right (nd-sharded) layout; copy the trace data straight in.
    # If copy rejects the spec, fall back to building the cache via from_torch(..., memory_config=kv_mem).
    ttnn.copy_host_to_device_tensor(cache_host_tt, tt_kvpe_cache)
    ttnn.synchronize_device(mesh_device)
    logger.info("[4/6] KV cache preloaded")

    logger.info(
        f"[5/6] Running chunked forward over the last {chunk_size_global} tokens at kv_actual={kv_actual_isl}..."
    )
    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2
    # Chunk-aligned step: chip-concat order == natural order, so the last 5k tokens map straight in.
    new_hidden = mla_input[prior_len:seq_len_cache].reshape(1, 1, chunk_size_global, hidden_size).to(torch.bfloat16)
    tt_new = ttnn.from_torch(
        new_hidden,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims),
    )
    mesh_device.enable_program_cache()
    tt_out = mla_tt.forward(
        hidden_states=tt_new,
        rope_tensors=indexed_rope,
        kvpe_cache=tt_kvpe_cache,
        kv_actual_isl=kv_actual_isl,
    )
    ttnn.synchronize_device(mesh_device)

    # Check the KV the forward just wrote for the new chunk (slab 10) against the GPU cache, split
    # into the latent (k_nope) and roped (k_pe) halves -- validates projection + rope + cache write
    # independently of the attention output. Read the cache back (SP concat on seq, TP replica 0),
    # undo the block-cyclic shuffle to natural order, then slice the last chunk.
    logger.info("[6/6] Checking new-chunk KV cache (k_nope / k_pe) vs GPU, then comparing output...")
    cache_sr = (
        ttnn.to_torch(
            tt_kvpe_cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        )
        .to(torch.float32)[:, :1]  # pick TP replica 0
        .reshape(seq_len_cache, kvpe_dim)
    )  # block-cyclic shard-row order
    p = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    cache_natural = torch.empty_like(cache_sr)
    cache_natural[p] = cache_sr  # shard row r -> global position p[r]

    new_kv_dev = cache_natural[prior_len:seq_len_cache]
    new_kv_ref = kv_post[prior_len:seq_len_cache].to(torch.float32)
    kv_lora = config.kv_lora_rank

    # k_nope is the latent half (rms_norm of compressed_kv) -- no rotation, so it must match the GPU
    # exactly. Hard assert.
    _, nope_msg = assert_with_pcc(new_kv_ref[:, :kv_lora], new_kv_dev[:, :kv_lora], 0.98)

    # k_pe is rotated, and the device stores it in a different rope basis than the GPU trace: the TT
    # model uses Meta-style cos/sin (get_cos_sin_matrix converts HF->Meta) and rotates adjacent dim
    # pairs, while the GPU's kv_kpe_roped is HF half-split. So a raw comparison is NOT apples-to-apples
    # (empirically ~0.44). Re-interleave the GPU k_pe to the Meta basis (meta[2i]=hf[i],
    # meta[2i+1]=hf[i+d/2]) and assert that -- this lands at ~0.9999, confirming the device k_pe is
    # correct, just stored in the interleaved basis. raw PCC is logged for context.
    dev_pe = new_kv_dev[:, kv_lora:]
    ref_pe_hf = new_kv_ref[:, kv_lora:]
    d = ref_pe_hf.shape[-1]
    ref_pe_meta = torch.stack([ref_pe_hf[:, : d // 2], ref_pe_hf[:, d // 2 :]], dim=-1).reshape(-1, d)
    _, raw_msg = comp_pcc(ref_pe_hf, dev_pe, 0.98)
    _, pe_msg = assert_with_pcc(ref_pe_meta, dev_pe, 0.98)
    logger.info(
        f"  new-chunk KV cache PCC -- k_nope: {nope_msg}  k_pe[Meta-aligned]: {pe_msg}  "
        f"(k_pe raw HF basis: {raw_msg})"
    )

    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2
    out_host = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
    ).to(torch.float32)
    ref = mla_output[prior_len:seq_len_cache].reshape(1, 1, chunk_size_global, hidden_size).to(torch.float32)
    _, msg = assert_with_pcc(ref, out_host, 0.98)
    logger.success(f"✓ Trace layer-0 chunked step PCC vs GPU mla_output: {msg}")


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len", [8192, 10 * 1024, 25 * 1024, 50 * 1024], ids=["seq8k", "seq10k", "seq25k", "seq50k"]
)
@pytest.mark.parametrize("num_chunks", [2, 4, 5, 10], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("num_users", [2], ids=lambda u: f"U{u}")
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_multi_user(
    request,
    mesh_device,
    seq_len,
    num_chunks,
    num_users,
    device_params,
):
    """
    Multi-user chunked-prefill correctness for MLA.

    Allocates one shared KVPE cache sized for N users (user-major layout:
    slot = user_id * num_layers + layer_idx, num_layers=1 here). For each
    chunk, interleaves a prefill call per user into its own slot via
    cache_user_id + num_cache_layers, then asserts:
      - per-user, per-chunk output PCC against that user's own reference
      - each user's cache slot PCC against that user's reference KVPE

    Cross-user contamination would show up as a PCC drop for one of the users.
    Chunks are chunk-aligned, so the unified chunked path runs with
    kv_actual_isl = chunk_start (the degenerate, non-rotated case of the rotation path).
    """
    config, weights = request.getfixturevalue("random_weights")

    sp_axis = 0
    tp_axis = 1
    is_balanced = False
    topology = ttnn.Topology.Linear
    num_cache_layers = 1  # one layer in this test; cache batch dim = num_users * 1

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]

    chunk_size = seq_len // num_chunks
    if seq_len % num_chunks != 0 or chunk_size % (ttnn.TILE_SIZE * sp) != 0:
        pytest.skip(
            f"chunked-prefill alignment: seq_len={seq_len}, num_chunks={num_chunks}, sp={sp} "
            f"requires chunk_size % (TILE_SIZE*sp) == 0 (got chunk_size={chunk_size})"
        )

    config.max_seq_len = seq_len

    logger.info(
        f"Multi-user chunked prefill: num_users={num_users} seq_len={seq_len} "
        f"num_chunks={num_chunks} chunk_size={chunk_size} sp={sp} tp={mesh_shape[tp_axis]}"
    )

    # Reference: one MLA module (weights are shared), run independently per user.
    mla_ref = create_mla_reference(
        config=config,
        state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    )
    mla_ref = mla_ref.eval().to(torch.bfloat16)

    torch.manual_seed(42)
    hidden_states_per_user = [torch.randn(1, seq_len, config.hidden_size).to(torch.bfloat16) for _ in range(num_users)]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    ref_outputs = []
    ref_kvpes = []
    for u in range(num_users):
        ref_cache = DynamicCache()
        with torch.no_grad():
            ref_output, _, ref_cache = mla_ref(
                hidden_states=hidden_states_per_user[u],
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
        ref_outputs.append(ref_output)
        ref_kvpes.append(ref_cache.key_cache[0])

    # TT MLA + rope + shared multi-user cache.
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
        is_chunked=True,
        slot_num=num_users,
        layer_num=num_cache_layers,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_cache_layers,
        num_users=num_users,
    )
    assert (
        tt_kvpe_cache.shape[0] == num_users * num_cache_layers
    ), f"expected cache batch dim {num_users * num_cache_layers}, got {tt_kvpe_cache.shape[0]}"

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2

    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2

    # Unified chunked path: build whole-cache indexed cos/sin ONCE; chunk-aligned chunks pass
    # kv_actual_isl = chunk_start (rotation degenerates to natural order).
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=chunk_size)
    mesh_device.enable_program_cache()

    # per_user_chunk_outputs[u] is the list of chunk outputs for user u.
    per_user_chunk_outputs = [[] for _ in range(num_users)]

    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = chunk_start + chunk_size

        # Interleave users at the same chunk position. Each user reads/writes only
        # its own slot via cache_user_id.
        for u in range(num_users):
            chunk_h = hidden_states_per_user[u][:, chunk_start:chunk_end, :].unsqueeze(0)
            tt_chunk_h = ttnn.from_torch(
                chunk_h,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
                ),
            )

            tt_chunk_out = mla_tt.forward(
                hidden_states=tt_chunk_h,
                rope_tensors=indexed_rope,
                kvpe_cache=tt_kvpe_cache,
                kv_actual_isl=chunk_start,
                cache_user_id=u,
            )

            out_host = ttnn.to_torch(
                tt_chunk_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                ),
            ).to(torch.bfloat16)
            per_user_chunk_outputs[u].append(out_host)

            _, msg = assert_with_pcc(
                ref_outputs[u][:, chunk_start:chunk_end, :].unsqueeze(0),
                out_host,
                0.98,
            )
            logger.info(f"  user {u} chunk {c}: per-chunk output PCC {msg}")

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    # Concatenated per-user output PCC.
    for u in range(num_users):
        full = torch.cat(per_user_chunk_outputs[u], dim=2)
        _, pcc = assert_with_pcc(ref_outputs[u].unsqueeze(0), full, 0.98)
        logger.info(f"User {u} full output PCC: {pcc}")

    # Gather the full cache once and slice per user.
    cache_stacked = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    # cache_stacked: [num_users * num_layers, tp, sp * seq_len_local, kvpe_dim]; pick one TP replica.
    cache_stacked = cache_stacked[:, :1, :, :]

    chunk_local = chunk_size // sp
    kv_lora_rank = config.kv_lora_rank
    for u in range(num_users):
        slot = cache_stacked[u : u + 1, :, :, :]  # [1, 1, sp*seq_len_local, kvpe_dim]
        # Merge chunked-interleaved SP layout back to global token order.
        merged = slot.reshape(1, 1, sp, num_chunks, chunk_local, kvpe_dim)
        merged = merged.permute(0, 1, 3, 2, 4, 5).reshape(1, 1, seq_len, kvpe_dim)
        _, kv_pcc = assert_with_pcc(ref_kvpes[u][:, :, :, :kv_lora_rank], merged[:, :, :, :kv_lora_rank], 0.99)
        logger.info(f"User {u} cache KV part PCC: {kv_pcc}")
        _, pe_pcc = assert_with_pcc(ref_kvpes[u][:, :, :, kv_lora_rank:], merged[:, :, :, kv_lora_rank:], 0.99)
        logger.info(f"User {u} cache PE part PCC: {pe_pcc}")

    logger.success(f"✓ Multi-user chunked prefill (U={num_users}, N={num_chunks}) test passed")


@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
        },
    ],
    ids=["line"],
    indirect=True,
)
def test_kvpe_cache_multi_user_slot_indexing(mesh_device, device_params):
    """
    Unit test for the user-major slot indexing of init_kvpe_cache +
    ttnn.kv_cache.fill_cache_for_user_.

    Writes a constant V = (layer_idx + 1) * (user_id + 1) into every slot
    (batch_index = user_id * num_layers + layer_idx, the user-major layout),
    plus a +0.5 offset on user 2's slots so each slot has a unique value (no
    aliasing between (1,2) and (2,1) of the bare product). Then reads the
    cache back to host and checks each slot contains its V.
    Confirms:
      - the flat slot index plumbs through fill_cache_for_user_ → cache buffer
      - no slot bleeds into a neighbor (each (user, layer) lands on its own row)

    Values (1-indexed in description, 0-indexed in code):
      user 1, layer 1 → 1.0   user 2, layer 1 → 2.5
      user 1, layer 2 → 2.0   user 2, layer 2 → 4.5
      user 1, layer 3 → 3.0   user 2, layer 3 → 6.5
    """
    sp_axis = 0
    tp_axis = 1
    num_users = 2
    num_layers = 3
    seq_len = 512
    kvpe_dim = 64 + 512  # qk_rope_head_dim + kv_lora_rank — both multiples of TILE_SIZE

    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    seq_len_local = seq_len // sp
    assert seq_len_local % ttnn.TILE_SIZE == 0, f"seq_len_local {seq_len_local} must be tile-aligned"

    tt_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )
    assert (
        tt_cache.shape[0] == num_users * num_layers
    ), f"cache batch dim {tt_cache.shape[0]} != num_users*num_layers ({num_users * num_layers})"

    # Write a constant per slot. User 2 (user_id == 1) gets a +0.5 offset so
    # the otherwise-commutative product doesn't alias (u=0,l=1) with (u=1,l=0).
    expected = {}
    for user_id in range(num_users):
        for layer_idx in range(num_layers):
            slot_idx = user_id * num_layers + layer_idx
            V = (layer_idx + 1) * (user_id + 1) + (0.5 if user_id == 1 else 0.0)
            expected[slot_idx] = (user_id, layer_idx, V)

            input_torch = torch.full((1, 1, seq_len_local, kvpe_dim), float(V), dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(
                input_torch,
                device=mesh_device,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            ttnn.kv_cache.fill_cache_for_user_(tt_cache, tt_input, slot_idx)
            logger.info(f"  wrote slot {slot_idx} (user={user_id}, layer={layer_idx}) = {V}")

    ttnn.synchronize_device(mesh_device)

    # Read back. ConcatMesh2dToTensor concats SP along seq (dim=2) and TP along head (dim=1);
    # cache is TP-replicated so all TP shards carry the same data — pick TP shard 0.
    cache_host = ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)
    cache_host = cache_host[:, :1, :, :]
    assert cache_host.shape == (
        num_users * num_layers,
        1,
        seq_len,
        kvpe_dim,
    ), f"unexpected gathered cache shape {cache_host.shape}"

    failures = []
    for slot_idx, (user_id, layer_idx, V) in expected.items():
        slot = cache_host[slot_idx]
        slot_min = slot.min().item()
        slot_max = slot.max().item()
        slot_mean = slot.mean().item()
        ok = abs(slot_min - V) < 1e-3 and abs(slot_max - V) < 1e-3
        marker = "✓" if ok else "✗"
        logger.info(
            f"  {marker} slot {slot_idx} (user={user_id}, layer={layer_idx}): "
            f"expected {V}, got min={slot_min} max={slot_max} mean={slot_mean:.4f}"
        )
        if not ok:
            failures.append(
                f"slot {slot_idx} (user={user_id}, layer={layer_idx}): expected {V}, "
                f"got min={slot_min} max={slot_max}"
            )

    assert not failures, "Slot indexing mismatch:\n  " + "\n  ".join(failures)
    logger.success(f"✓ Multi-user KVPE cache slot indexing verified (U={num_users}, L={num_layers})")
