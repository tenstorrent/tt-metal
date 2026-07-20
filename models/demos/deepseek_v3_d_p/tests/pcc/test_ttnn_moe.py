# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtMoe module with integrated gate.

Tests that TTNN TtMoe produces matching outputs to TorchMoe reference.
This validates the full MoE pipeline:
Gate → Dispatch → Routed Experts → Combine → Split → Add Shared.
"""

import gc
import os
import random
from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tests.reference_runners import run_reference_moe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_sp_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    compare_recall,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
    validate_composed,
    validate_dispatch_buffer,
    validate_dispatch_buffer_pcc,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import (
    log_validation_results,
    visualize_expert_dispatch_table,
)
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import GOLDEN_LONGBOOK_TRACE, load_trace_gate_input
from tests.ttnn.utils_for_testing import comp_pcc

# First MoE layer in DeepSeek-V3 (metadata moe_layer_offset == 3); the golden
# trace stores its post-attention RMSNorm output, i.e. the MoE block input.
_MOE_LAYER_IDX = 3


# dispatch_buffer_capacity_factor below is ceil(N/2) of the most conservative
# integer N such that dgs*seq*N >= theoretical worst-case dispatch buffer.
# Real traffic never approaches the worst case, so half-capacity is sufficient.
def run_model(
    variant,
    config,
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    run_pcc_check,
    num_links,
    topology,
    gate_fallback_mode,
    request,
    is_balanced=False,
    padded_percent=0,
    use_fp8_compression=False,
):
    """TtMoe PCC body — shared between `test_ds_moe` / `test_kimi_moe`.

    The gate's grouping (n_group, topk_group) and route_scale are read from
    the variant's HF config. DSv3 values are a no-op; Kimi values switch the
    gate routing rule.

    ``is_balanced`` selects zigzag placement so padding-aware dispatch shrinks every
    SP device's token loop. ``padded_percent`` requests right-padding: it is only
    engaged on the perf (non-PCC) path — a full-tensor PCC check would (correctly)
    mismatch on the skipped padded rows, and padded-row correctness is covered by the
    dedicated grouped_topk / routing_setup tests. HOST_ALL gates ignore padding entirely
    (TtMoe falls back to padding_config=None for non-DEVICE_FP32 gates).
    """

    # Scoped: only the linear-8 / 64-expert / HOST_ALL / pcc-check case OOMs without this.
    # Cached all-gather semaphores get placed at the wrong offset for that specific config.
    # Test ID matched: test_ttnn_moe[blackhole-linear-8-1600-7168-2048-64-8-2-GateComputeMode.HOST_ALL-True]
    n_sp_devices_pre, n_tp_devices_pre = mesh_device.shape
    if (
        n_sp_devices_pre == 8
        and n_tp_devices_pre == 1
        and num_routed_experts == 64
        and gate_fallback_mode == GateComputeMode.HOST_ALL
        and run_pcc_check
    ):
        mesh_device.disable_and_clear_program_cache()

    profiler.clear()
    profiler.start("test_ttnn_moe")

    random.seed(42)
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    n_sp_devices, n_tp_devices = mesh_device.shape
    layer_idx = 0

    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test")
    logger.debug(f"{'='*60}")
    logger.debug(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.debug(f"dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")

    signpost(
        f"TtMoe PCC test - mesh {mesh_device.shape}, seq_len={seq_len_per_chip}, "
        f"emb_dim={emb_dim}, experts={num_routed_experts}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
        emb_dim=emb_dim,
        fp8_scaled_input=use_fp8_compression,
    )
    logger.debug(f"experts_per_chip={experts_per_chip}, metadata_len={metadata_len}")
    logger.debug(
        f"max_dispatch_buffer_token_size={max_dispatch_buffer_token_size}, max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}"
    )

    # ========================================
    # Step 1: Create weights (cache-aware)
    # ========================================
    # Perf runs (run_pcc_check=False) build the routed/shared experts from
    # placeholder torch.empty weights, while PCC runs build them from realistic
    # seeded weights. The cache dir is keyed only on shapes, so without a
    # weights-type suffix a perf run would persist placeholder (≈zero) expert
    # weights that a later PCC run loads as "complete" — producing all-zero
    # expert outputs (PCC=0). Keep the two cohorts in separate directories.
    weights_type = "realistic" if run_pcc_check else "dummy"
    # Base dir is env-overridable so concurrent users don't collide on a single shared /tmp path
    # (the default /tmp/{variant}_moe_cache is world-visible but owner-writable → cross-user EACCES).
    _moe_cache_base = os.environ.get("DS_MOE_CACHE_DIR", f"/tmp/{variant.name}_moe_cache")
    moe_cache_dir = Path(
        f"{_moe_cache_base}/{num_routed_experts}experts_{n_sp_devices}x{n_tp_devices}mesh_{emb_dim}emb_{hidden_dim}hid_{weights_type}"
    )
    moe_cache_dir.mkdir(parents=True, exist_ok=True)

    init_checker(moe_cache_dir)
    ttnn_cache_complete = TtMoe.check_cache_complete(
        moe_cache_dir, layer_idx=layer_idx, experts_per_chip=experts_per_chip
    )
    need_torch_weights = not ttnn_cache_complete or run_pcc_check
    logger.info(f"Cache status: TTNN={ttnn_cache_complete}, need_torch_weights={need_torch_weights}")

    if need_torch_weights:
        logger.info("Creating torch weights...")
        profiler.start("weights_creation")
        # Fixed per-creator seeds make each weight tensor a pure function of its
        # shape + seed, independent of how much global RNG was consumed before it.
        # This is required because these weights are persisted to a shape-keyed
        # on-disk cache (moe_cache_dir): without it, the gate weight depends on
        # whether routed/shared weights were drawn first (run_pcc_check branch),
        # so a perf-built cache (gate drawn first) silently mismatches the PCC
        # reference (gate drawn third) and collapses gate recall to ~random.
        if run_pcc_check:
            all_routed_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim, seed=1234)
            shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim, seed=5678)
        else:
            all_routed_weights = None
            shared_expert_weights = None
        gate_weights = create_gate_weights(num_routed_experts, emb_dim, seed=9012)
        profiler.end("weights_creation")

        # Build TTNN cache if not already complete
        if not ttnn_cache_complete:
            logger.info("Building TTNN cache...")
            profiler.start("ttnn_cache_build")
            TtMoe.build_ttnn_cache(
                gate_weights=gate_weights,
                routed_expert_weights=all_routed_weights,
                shared_expert_weights=shared_expert_weights,
                experts_per_chip=experts_per_chip,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                mesh_device=mesh_device,
                routed_expert_weights_dtype=ttnn.bfloat4_b,
                shared_expert_weights_dtype=ttnn.bfloat8_b,
                cache_path=moe_cache_dir,
                layer_idx=layer_idx,
            )
            profiler.end("ttnn_cache_build")

        # For non-PCC runs, free the heavy weights now that TTNN cache is built
        if not run_pcc_check:
            all_routed_weights = None
            shared_expert_weights = None
    else:
        logger.info("TTNN cache complete, skipping torch weight creation")
        all_routed_weights = None
        shared_expert_weights = None
        gate_weights = None

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    visualize_expert_dispatch_table(
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        num_routed_experts,
    )

    # ========================================
    # Step 2: Create input tensor
    # ========================================
    profiler.start("input_creation")

    # Prefer a realistic MoE-block input (post-attention RMSNorm of the first MoE
    # layer) from the golden trace; fall back to synthetic noise when unavailable.
    # Restricted to PCC runs on the DeepSeek hidden dim so perf baselines and the
    # Kimi variant keep their established synthetic input.
    # currently cannot use ttnn.empty on x; because indices become ND beyond max dispatch token limit.
    x = None
    if run_pcc_check and emb_dim == DeepSeekV3Config.EMB_SIZE:
        total_tokens = dispatch_group_size * seq_len_per_chip
        trace_input = load_trace_gate_input(
            GOLDEN_LONGBOOK_TRACE, layer_idx=_MOE_LAYER_IDX, max_seq_len=total_tokens, dim=emb_dim
        )
        if trace_input is not None:
            x = trace_input.reshape(dispatch_group_size, seq_len_per_chip, emb_dim).to(torch.bfloat16)
    if x is None:
        x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    profiler.end("input_creation")

    # TtMoe.forward deallocates its input (tt_moe.py:522), so tt_x must be re-uploaded each iter.
    def upload_tt_x():
        return ttnn.from_torch(
            x,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )

    # Engage right-padding only on the perf (non-PCC) path; see the run_model docstring.
    if padded_percent > 0 and not run_pcc_check:
        actual_isl = int(dispatch_group_size * seq_len_per_chip * (1 - padded_percent / 100))
    else:
        actual_isl = None

    # ========================================
    # Step 3: Run TorchMoe reference with intermediates
    # ========================================
    if run_pcc_check:
        profiler.start("torch_moe_creation")
        torch_moe = TorchMoe(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            expert_dispatch_table=expert_dispatch_table,
            num_dispatch_groups=num_dispatch_groups,
            routed_expert_weights=all_routed_weights,
            shared_expert_weights=shared_expert_weights,
            gate_weights=gate_weights,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            route_scale=config.routed_scaling_factor,
        )
        profiler.end("torch_moe_creation")

        profiler.start("torch_forward")
        torch_output, torch_intermediates = torch_moe(x, return_intermediates=True)
        profiler.end("torch_forward")

    # ========================================
    # Step 4: TtMoe forward
    # ========================================
    profiler.start("tt_moe_creation")
    logger.debug("Creating TtMoe...")
    tt_moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=num_links,
        topology=topology,
        routed_expert_weights=all_routed_weights,
        shared_expert_weights=shared_expert_weights,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights=gate_weights,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=moe_cache_dir,
        layer_idx=layer_idx,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        route_scale=config.routed_scaling_factor,
        is_balanced=is_balanced,
        use_fp8_compression=use_fp8_compression,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_moe_creation")

    profiler.start("tt_forward")
    logger.debug("Running TtMoe forward pass...")

    tt_x = upload_tt_x()
    signpost(header="tt_forward_START")
    tt_output, tt_intermediates = tt_moe(
        tt_x, return_intermediates=run_pcc_check, actual_isl=actual_isl, padding_side="right"
    )
    ttnn.synchronize_device(mesh_device)
    signpost(header="tt_forward_END")

    profiler.end("tt_forward")
    logger.debug(f"  tt_forward: {profiler.get('tt_forward') * 1000:.2f} ms")

    # Early return when run_pcc_check=False (profiling mode)
    if not run_pcc_check:
        profiler.end("test_ttnn_moe")
        logger.debug("run_pcc_check=False, skipping PCC validation")
        for key in profiler.times:
            logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
        return

    # ========================================
    # Step 5: Validate
    # ========================================
    profiler.start("pcc_validation")
    logger.debug("Comparing intermediate outputs...")

    all_passed = True
    validation_results = []

    # Gate recall: compare TtMoe gate indices vs TorchMoe gate indices
    tt_indices = ttnn.to_torch(
        tt_intermediates.gate_indices,
        mesh_composer=get_sp_mesh_composer(mesh_device),
        dtype=torch.int32,
    )

    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        target_recall = 0.99
    else:
        target_recall = 0.977

    recall_result = validate_composed(
        tt_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        torch_intermediates.gate_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        1,
        n_sp_devices,
        compare_recall(target_recall),
        name="gate_indices_recall",
        broadcast_groups=n_tp_devices,
    )
    log_validation_results(
        results=[recall_result],
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Recall Validation",
    )
    if recall_result.passed:
        logger.info(f"[gate_indices_recall] PASSED")
    else:
        logger.error(
            f"[gate_indices_recall] FAILED {len(recall_result.mismatches)}/{recall_result.total} below threshold {target_recall}"
        )
        recall_result.log_mismatches(limit=5)
        all_passed = False

    # Dense tensor checks with PCC
    # fmt: off
    dense_checks = [
        ("shared_output", tt_intermediates.shared_output, torch_intermediates.shared_output, get_tp_mesh_composer(mesh_device), 0.997),
        ("routed_output", tt_intermediates.routed_output, torch_intermediates.routed_output, get_tp_mesh_composer(mesh_device), 0.96),
        ("final_output", tt_output, torch_output, get_tp_mesh_composer(mesh_device), 0.982),
    ]
    # fmt: on

    for name, tt_tensor, torch_tensor, composer, threshold in dense_checks:
        if tt_tensor is None:
            logger.warning(f"[{name}] validation SKIPPED")
            continue
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        _, pcc = comp_pcc(torch_tensor.float(), tt_host.float())
        if pcc >= threshold:
            logger.info(f"[{name}] PASSED - PCC: {pcc:.6f} (threshold: {threshold})")
        else:
            logger.error(f"[{name}] FAILED - PCC: {pcc:.6f} below threshold {threshold}")
            all_passed = False

    del torch_moe
    gc.collect()

    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        # Sparse tensor validation using slot-aware comparisons
        # fmt: off
        sparse_checks = [
            ("dispatched_buffer", "dispatched_buffer", tt_intermediates.dispatched_buffer, torch_intermediates.dispatched_buffer,
            get_ep_mesh_composer(mesh_device), torch.bfloat16, validate_dispatch_buffer, {}),
            ("dispatch_metadata", "metadata", tt_intermediates.metadata, torch_intermediates.metadata,
            get_ep_mesh_composer(mesh_device), None, validate_dispatch_metadata, {}),
            ("expert_outputs", "expert_outputs", tt_intermediates.expert_outputs, torch_intermediates.expert_outputs,
            get_ep_mesh_composer(mesh_device), torch.bfloat16, validate_dispatch_buffer_pcc, {"pcc_threshold": 0.95}),
        ]
        # fmt: on

        expert_token_counts = torch_intermediates.expert_token_counts
        expert_region_offsets = torch_intermediates.expert_region_offsets

        for i, (name, torch_field, tt_tensor, torch_tensor, composer, dtype, validate_fn, extra_kwargs) in enumerate(
            sparse_checks
        ):
            if tt_tensor is None or torch_tensor is None:
                logger.warning(f"[{name}] validation SKIPPED")
                continue
            tt_host = (
                ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=dtype)
                if dtype
                else ttnn.to_torch(tt_tensor, mesh_composer=composer)
            )
            torch_ref = torch_tensor.to(dtype) if dtype else torch_tensor
            result = validate_fn(
                torch_ref,
                tt_host,
                expert_region_offsets,
                expert_token_counts,
                expert_dispatch_table,
                num_dispatch_groups,
                dispatch_group_size,
                experts_per_chip,
                verbose=True,
                **extra_kwargs,
            )
            result.name = name
            validation_results.append(result)
            if result.passed:
                logger.info(f"[{name}] PASSED - {result.matches}/{result.total} slots matched")
            else:
                logger.error(f"[{name}] FAILED - {result.matches}/{result.total} slots matched")
                result.log_mismatches(limit=5)
                all_passed = False

            del tt_host, torch_ref, tt_tensor, torch_tensor
            sparse_checks[i] = None
            setattr(torch_intermediates, torch_field, None)
            gc.collect()

        # Validate combined_output (before reduce step)
        if tt_intermediates.combined_output is not None and torch_intermediates.combined_output is not None:
            name = "combined_output"
            logger.debug(f"  {name} tt shape: {tt_intermediates.combined_output.shape}")
            logger.debug(f"  {name} torch shape: {torch_intermediates.combined_output.shape}")

            tt_combined_torch = ttnn.to_torch(
                tt_intermediates.combined_output,
                mesh_composer=get_ep_mesh_composer(mesh_device),
                dtype=torch.bfloat16,
            )

            combine_pcc = 0.95
            combine_result = validate_combine_output(
                torch_intermediates.combined_output,
                tt_combined_torch,
                tt_indices,
                num_dispatch_groups,
                num_routed_experts,
                use_pcc=True,
                pcc_threshold=combine_pcc,
                verbose=True,
                expert_dispatch_table=expert_dispatch_table,
                expert_token_counts=expert_token_counts,
                experts_per_chip=experts_per_chip,
            )

            log_validation_results(
                results=[combine_result],
                num_dispatch_groups=num_dispatch_groups,
                dispatch_group_size=dispatch_group_size,
                title="Combined Output Validation",
            )

        if combine_result.passed:
            logger.info(
                f"[{name}] PASSED - {combine_result.matches}/{combine_result.total} slots matched (PCC >= {combine_pcc})"
            )
        else:
            logger.error(f"[{name}] FAILED - {combine_result.matches}/{combine_result.total} slots matched")
            log_combine_mismatch_details(
                combine_result.mismatches, torch_intermediates.combined_output, tt_combined_torch, use_pcc=True
            )
            log_per_chip_statistics(
                combine_result.mismatches, dispatch_group_size, seq_len_per_chip, num_experts_per_tok
            )
            all_passed = False
    else:
        logger.warning("[combined_output] SKIPPED - TTNN tensor is None")

    # Log validation summary
    if validation_results:
        log_validation_results(
            results=validation_results,
            num_dispatch_groups=num_dispatch_groups,
            dispatch_group_size=dispatch_group_size,
            title="Sparse Tensor Validation Results",
        )

    logger.debug("Note: Final PCC expected to be low until full pipeline is enabled")
    profiler.end("pcc_validation")

    # Upstream MoE reference cross-check. Returns None when the variant has no reference bundled.
    profiler.start("reference")
    ref_out = run_reference_moe(
        variant,
        config=config,
        gate_weights=gate_weights,
        routed_expert_weights=all_routed_weights,
        shared_expert_weights=shared_expert_weights,
        x=x,
    )
    if ref_out is not None and tt_output is not None:
        logger.info("Running upstream MoE reference")
        tt_final_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
        _, ref_pcc = comp_pcc(ref_out.float(), tt_final_host.float())
        threshold = variant.moe_pcc_threshold
        if ref_pcc >= threshold:
            logger.info(f"[reference_output] PASSED - PCC: {ref_pcc:.6f} (threshold: {threshold})")
        else:
            logger.error(f"[reference_output] FAILED - PCC: {ref_pcc:.6f} below threshold {threshold}")
            all_passed = False
        del ref_out
    profiler.end("reference")

    assert all_passed, "One or more comparisons failed. See logs for details."

    profiler.end("test_ttnn_moe")
    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test PASSED!")
    logger.debug(f"{'='*60}")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")


@pytest.mark.parametrize(
    (
        "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, "
        "dispatch_buffer_capacity_factor, gate_fallback_mode, run_pcc_check, is_balanced"
    ),
    [
        # fmt: off
        # is_balanced=True (zigzag placement) spreads real tokens evenly across SP devices so
        # padding-aware dispatch shrinks every device's token loop. Only enabled for the
        # perf-device-256 (DEVICE_FP32, non-PCC) row — the only one that builds a padding_config;
        # the rest keep sequential placement (their reference / PCC path isn't zigzag).
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 8, GateComputeMode.DEVICE_FP32,   False, True,  marks=pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), id="perf-device-256"),
        # PCC gate on the production 256-expert / 32-per-chip path. The unified
        # routed-expert MoE op switches into the unfused extract -> FFN -> insert
        # chain whenever num_routed_experts > 64; without this variant that
        # branch ships PCC-untested on Blackhole. Lighter dispatch capacity (5
        # vs 8) keeps the soak time bounded.
        pytest.param(1600, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 5, GateComputeMode.DEVICE_FP32,   True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(900)], id="pcc-device-256"),
        pytest.param(1600, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,  64, 8, 5, GateComputeMode.HOST_ALL, True,  False, marks=pytest.mark.timeout(900)),
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 5, GateComputeMode.HOST_ALL, True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.skipif(not is_galaxy(), reason="Requires Galaxy")], id="pcc-host-256"),
        # Perf: LB 8x1 dispatch/combine proxy. 64 experts + 2 picks/tok match one glx column's per-chip traffic (balanced_load=800).
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,  64, 2, 8, GateComputeMode.HOST_ALL, False, False, marks=pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), id="perf-host-64"),
        # GLM-5.1 MoE (256 experts / top-8, emb 6144, moe_int 2048). Exercises the >64-expert unfused
        # extract->FFN->insert routed-expert path on GLM dims. Gate is generic here (op-level test);
        # GLM's noaux_tc knife-edge gate is validated at the transformer level. 25k = 3200 per-chip x 8.
        pytest.param(1600, GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE, GLM51Config.NUM_ROUTED_EXPERTS, GLM51Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(900)], id="pcc-device-glm-256"),
        pytest.param(3200, GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE, GLM51Config.NUM_ROUTED_EXPERTS, GLM51Config.NUM_EXPERTS_PER_TOKEN, 8, GateComputeMode.DEVICE_FP32, False, True,  marks=pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), id="perf-device-glm-256"),
        pytest.param(3200, GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE, GLM51Config.NUM_ROUTED_EXPERTS, GLM51Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.HOST_ALL,    True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.skipif(not is_galaxy(), reason="Requires Galaxy")], id="pcc-host-glm-256"),
        # fmt: on
    ],
)
@pytest.mark.parametrize("padded_percent", [0, 50], ids=lambda p: f"pad{p}")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
def test_ds_moe(
    variant,
    config_only,
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    run_pcc_check,
    is_balanced,
    num_links,
    topology,
    gate_fallback_mode,
    request,
    padded_percent,
):
    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        seq_len_per_chip,
        emb_dim,
        hidden_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        run_pcc_check,
        num_links,
        topology,
        gate_fallback_mode,
        request,
        is_balanced=is_balanced,
        padded_percent=padded_percent,
    )


@pytest.mark.parametrize(
    (
        "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, "
        "dispatch_buffer_capacity_factor, gate_fallback_mode, run_pcc_check"
    ),
    [
        # fmt: off
        pytest.param( 640, KimiK26Config.EMB_SIZE, KimiK26Config.MOE_INTERMEDIATE_SIZE, KimiK26Config.NUM_ROUTED_EXPERTS, KimiK26Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi-5k-perf"),
        pytest.param( 640, KimiK26Config.EMB_SIZE, KimiK26Config.MOE_INTERMEDIATE_SIZE, KimiK26Config.NUM_ROUTED_EXPERTS, KimiK26Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, True, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi-5k-pcc"),
        pytest.param(3200, KimiK26Config.EMB_SIZE, KimiK26Config.MOE_INTERMEDIATE_SIZE, KimiK26Config.NUM_ROUTED_EXPERTS, KimiK26Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi-25k-perf"),
        pytest.param(3200, KimiK26Config.EMB_SIZE, KimiK26Config.MOE_INTERMEDIATE_SIZE, KimiK26Config.NUM_ROUTED_EXPERTS, KimiK26Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, True, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi-25k-pcc"),
        # fmt: on
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            },
            2 if is_blackhole() else 1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
def test_kimi_moe(
    variant,
    config_only,
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    run_pcc_check,
    num_links,
    topology,
    gate_fallback_mode,
    request,
):
    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        seq_len_per_chip,
        emb_dim,
        hidden_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        run_pcc_check,
        num_links,
        topology,
        gate_fallback_mode,
        request,
    )
