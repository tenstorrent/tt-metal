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
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, ACTIVATION_SITU
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tests.reference_runners import run_reference_moe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_gate_weights,
    create_latent_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_sp_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, assert_gate_mode_matches_adapter
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
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
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import GOLDEN_LONGBOOK_TRACE, load_trace_gate_input
from tests.ttnn.utils_for_testing import comp_pcc

# First MoE layer in DeepSeek-V3 (metadata moe_layer_offset == 3); the golden
# trace stores its post-attention RMSNorm output, i.e. the MoE block input.
_MOE_LAYER_IDX = 3


# dispatch_buffer_capacity_factor below is ceil(N/2) of the most conservative
# integer N such that dgs*seq*N >= theoretical worst-case dispatch buffer.
# Real traffic never approaches the worst case, so half-capacity is sufficient.
# Fused routed-expert activation -> the TorchExpert reference that must match it.
_TORCH_ROUTED_ACTIVATION = {
    ttnn.RoutedExpertActivation.Silu: ACTIVATION_SILU,
    ttnn.RoutedExpertActivation.SituGlu: ACTIVATION_SITU,
}

# ...and -> the upstream vendored model's hidden_act spelling. None leaves the variant config's own
# hidden_act alone, which is what every SiLU model wants.
_UPSTREAM_ACT = {
    ttnn.RoutedExpertActivation.Silu: None,
    ttnn.RoutedExpertActivation.SituGlu: "situ",
}


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
    routed_emb_dim=None,
    shared_hidden_dim=None,
    latent_use_norm=True,
    rms_norm_eps=1e-5,
    final_output_pcc=0.982,
    routed_activation=ttnn.RoutedExpertActivation.Silu,
    shared_activation=ACTIVATION_SILU,
    measure=None,
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

    ``routed_activation`` selects the fused routed-expert kernel's activation and ``shared_activation``
    the shared expert's; each is mirrored onto the matching torch reference. They are separate knobs
    because the two sites run different implementations -- a fused kernel vs the Python-composed
    ttnn ops in TtSharedExpert -- even where Kimi-K3 sets both to SiTU (#53625).

    ``measure`` wraps the forward for a perf caller: it is called as ``measure(forward)``,
    must invoke the thunk and return its result, and owns the device sync. The perf gates use
    it to run the forward inside a real-time-profiler window (see
    ``tests/perf/test_kimi_k3_moe_perf.py``) so the measured region is the forward alone --
    the constructor's one-time weight tilize/typecast stays outside it.

    """
    if routed_activation not in _TORCH_ROUTED_ACTIVATION or routed_activation not in _UPSTREAM_ACT:
        raise ValueError(f"no torch reference for {routed_activation}; supported: {list(_TORCH_ROUTED_ACTIVATION)}")
    torch_routed_activation = _TORCH_ROUTED_ACTIVATION[routed_activation]
    upstream_activation = _UPSTREAM_ACT[routed_activation]
    if shared_activation not in (ACTIVATION_SILU, ACTIVATION_SITU):
        raise ValueError(f"unknown shared_activation {shared_activation!r}")
    assert_gate_mode_matches_adapter(variant, gate_fallback_mode)

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
    # Mirrors TtMoe's own defaulting, so the two cannot disagree on what "no latent space" means.
    routed_emb = emb_dim if routed_emb_dim is None else routed_emb_dim
    shared_hidden = hidden_dim if shared_hidden_dim is None else shared_hidden_dim
    use_latent = routed_emb != emb_dim
    if use_latent:
        logger.info(f"LatentMoE: routed side at {routed_emb} (emb_dim={emb_dim}), shared inter={shared_hidden}")

    bias_free_router = not getattr(variant.model_config, "ROUTER_HAS_CORRECTION_BIAS", True)
    weights_type = ("realistic" if run_pcc_check else "dummy") + ("_bias0" if bias_free_router else "")
    # Base dir is env-overridable so concurrent users don't collide on a single shared /tmp path
    # (the default /tmp/{variant}_moe_cache is world-visible but owner-writable → cross-user EACCES).
    _moe_cache_base = os.environ.get("DS_MOE_CACHE_DIR", f"/tmp/{variant.name}_moe_cache")
    # Every dim that shapes a cached tensor goes in the key unconditionally: filenames carry dtype
    # and layout but not shape, so a colliding key is a silent wrong-weights load.
    moe_cache_dir = Path(
        f"{_moe_cache_base}/{num_routed_experts}experts_{n_sp_devices}x{n_tp_devices}mesh_"
        f"{emb_dim}emb_{hidden_dim}hid_{routed_emb}rout_{shared_hidden}sh_{weights_type}"
    )
    moe_cache_dir.mkdir(parents=True, exist_ok=True)

    init_checker(moe_cache_dir)
    ttnn_cache_complete = TtMoe.check_cache_complete(
        moe_cache_dir,
        layer_idx=layer_idx,
        experts_per_chip=experts_per_chip,
        use_latent_moe=use_latent,
        latent_use_norm=latent_use_norm,
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
            # Routed experts at the latent width, shared expert at emb_dim with its own intermediate.
            all_routed_weights = create_torch_expert_weights(num_routed_experts, routed_emb, hidden_dim, seed=1234)
            shared_expert_weights = create_shared_expert_weights(emb_dim, shared_hidden, seed=5678)
        else:
            all_routed_weights = None
            shared_expert_weights = None
        gate_weights = create_gate_weights(num_routed_experts, emb_dim, seed=9012)
        if bias_free_router:
            # Zero is this bias's exact identity in every consumer -- top-k on (logits + bias) and
            # the sigmoid affinity alike -- so the golden matches a router that has no bias at all.
            gate_weights["e_score_correction_bias"] = torch.zeros_like(gate_weights["e_score_correction_bias"])
        # Fixed seed for the same reason as above: a perf-built cache must match the PCC reference.
        latent_weights = create_latent_weights(emb_dim, routed_emb, seed=3456) if use_latent else None
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
                shared_hidden_dim=shared_hidden,
                routed_emb_dim=routed_emb,
                latent_weights=latent_weights,
                latent_use_norm=latent_use_norm,
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
        latent_weights = None

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
            topk_method=(
                "gpt_softmax"
                if gate_fallback_mode in (GateComputeMode.GPT_HOST, GateComputeMode.GPT_DEVICE)
                else "noaux_tc"
            ),
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
            routed_emb_dim=routed_emb_dim,
            shared_hidden_dim=shared_hidden_dim,
            latent_weights=latent_weights,
            latent_use_norm=latent_use_norm,
            rms_norm_eps=rms_norm_eps,
            # Each side matches whatever the device runs there.
            activation=torch_routed_activation,
            situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
            situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
            shared_activation=shared_activation,
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
        routed_expert_activation=routed_activation,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        shared_expert_activation=shared_activation,
        shared_expert_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        shared_expert_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
        gate_weights=gate_weights,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=moe_cache_dir,
        layer_idx=layer_idx,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        route_scale=config.routed_scaling_factor,
        is_balanced=is_balanced,
        routed_emb_dim=routed_emb_dim,
        shared_hidden_dim=shared_hidden_dim,
        latent_weights=latent_weights,
        latent_use_norm=latent_use_norm,
        rms_norm_eps=rms_norm_eps,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_moe_creation")

    profiler.start("tt_forward")
    logger.debug("Running TtMoe forward pass...")

    tt_x = upload_tt_x()

    def forward():
        return tt_moe(tt_x, return_intermediates=run_pcc_check, actual_isl=actual_isl, padding_side="right")

    signpost(header="tt_forward_START")
    if measure is None:
        tt_output, tt_intermediates = forward()
        ttnn.synchronize_device(mesh_device)
    else:
        # measure() syncs: the real-time profiler stops collecting once the window closes, so
        # the sync has to happen inside it or the last programs' records are still in flight.
        tt_output, tt_intermediates = measure(forward)
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
        ("final_output", tt_output, torch_output, get_tp_mesh_composer(mesh_device), final_output_pcc),
    ]
    if use_latent:
        # Checked before routed_output, which bundles the reduce, the latent norm and the
        # up-projection; this isolates everything up to and including the reduce. 0.965 sits ~0.005
        # under K3's measured 0.969778.
        dense_checks.insert(1, (
            "latent_routed_output", tt_intermediates.latent_routed_output,
            torch_intermediates.latent_routed_output, get_tp_mesh_composer(mesh_device), 0.965,
        ))
        # Post down-projection, pre-dispatch. Composed with the SP composer: to_latent() all-gathers
        # on the TP axis, so this tensor is replicated across columns rather than sharded.
        dense_checks.insert(1, (
            "latent_input", tt_intermediates.latent_input,
            torch_intermediates.latent_input, get_sp_mesh_composer(mesh_device), 0.998,
        ))
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
        latent_weights=latent_weights,
        x=x,
        # Same per-site activations the device runs; see run_reference_moe.
        hidden_act=upstream_activation,
        shared_hidden_act=shared_activation if upstream_activation is not None else None,
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
        # GLM-5.2 MoE (256 experts / top-8, emb 6144, moe_int 2048). Exercises the >64-expert unfused
        # extract->FFN->insert routed-expert path on GLM dims. Gate is generic here (op-level test);
        # GLM's noaux_tc knife-edge gate is validated at the transformer level. 25k = 3200 per-chip x 8.
        pytest.param(1600, GLM52Config.EMB_SIZE, GLM52Config.MOE_INTERMEDIATE_SIZE, GLM52Config.NUM_ROUTED_EXPERTS, GLM52Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(900)], id="pcc-device-glm-256"),
        pytest.param(3200, GLM52Config.EMB_SIZE, GLM52Config.MOE_INTERMEDIATE_SIZE, GLM52Config.NUM_ROUTED_EXPERTS, GLM52Config.NUM_EXPERTS_PER_TOKEN, 8, GateComputeMode.DEVICE_FP32, False, True,  marks=pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), id="perf-device-glm-256"),
        pytest.param(3200, GLM52Config.EMB_SIZE, GLM52Config.MOE_INTERMEDIATE_SIZE, GLM52Config.NUM_ROUTED_EXPERTS, GLM52Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.HOST_ALL,    True,  False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.skipif(not is_galaxy(), reason="Requires Galaxy")], id="pcc-host-glm-256"),
        # fmt: on
    ],
)
@pytest.mark.parametrize("padded_percent", [0, 50], ids=lambda p: f"pad{p}")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 1),
            torus_y_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="torus-y-8x1",
        ),
        pytest.param(
            (4, 2),
            fabric2d_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            fabric2d_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
    gate_fallback_mode,
    request,
    padded_percent,
):
    topology = per_axis_topology(device_params["fabric_config"])
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 1),
            torus_y_device_params(fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="torus-y-8x1",
        ),
        pytest.param(
            (4, 2),
            fabric2d_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
    gate_fallback_mode,
    request,
):
    topology = per_axis_topology(device_params["fabric_config"])
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


# ---------------------------------------------------------------------------
# Kimi-K3 LatentMoE
# ---------------------------------------------------------------------------
#
# Capacity factor 5 carries over from Kimi-K2.6: K3 halves the row width (7168 -> 3584 latent) and
# doubles the token slots (top-8 -> top-16), so per-chip dispatch bytes are roughly unchanged.
@pytest.mark.parametrize(
    (
        "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, "
        "dispatch_buffer_capacity_factor, gate_fallback_mode, run_pcc_check"
    ),
    [
        # fmt: off
        pytest.param( 640, KimiK3Config.EMB_SIZE, KimiK3Config.MOE_INTERMEDIATE_SIZE, KimiK3Config.NUM_ROUTED_EXPERTS, KimiK3Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, False, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi_k3-5k-perf"),
        pytest.param( 640, KimiK3Config.EMB_SIZE, KimiK3Config.MOE_INTERMEDIATE_SIZE, KimiK3Config.NUM_ROUTED_EXPERTS, KimiK3Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.DEVICE_FP32, True, marks=[pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"), pytest.mark.timeout(0)], id="kimi_k3-5k-pcc"),
        # fmt: on
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        # Loudbox proxy for local bring-up; no pipeline selects it. TP stays 4 as on the 8x4 anchor:
        # at TP=1 the shared expert is unsharded and its gate matmul's CBs exceed L1.
        pytest.param(
            (2, 4),
            fabric2d_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k3"], indirect=True, ids=["kimi_k3"])
def test_kimi_k3_moe(
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
    gate_fallback_mode,
    request,
):
    """Kimi-K3 MoE: 896 experts / top-16 with the LatentMoE projections around the routed side.

    Both expert kinds run the checkpoint's SiTU-GLU on device, each matched by a SiTU torch
    reference: the routed side through the fused kernel (``RoutedExpertActivation.SituGlu``), the
    shared side through TtSharedExpert's composed softcap/sigmoid/multiply. This is also the only
    test that reaches that composed path's sub_core_grids branch -- the shared expert runs on a
    sub-device here, overlapped with the dispatch, which test_shared_expert does not set up.

    One deliberate limit remains from the bring-up scope:

      * **Seeded random weights, not the checkpoint.** Everything routed is MXFP4 and no dequantizer
        exists yet, so device-vs-torch parity is checked on identical seeded weights. That is the same
        thing ``test_kimi_moe`` and ``test_ds_moe`` do -- the ``"realistic"`` cache cohort means
        *seeded* rather than *placeholder*, not *from a checkpoint*. The router is the one MoE tensor
        group K3 leaves unquantized, and real-weight gate coverage lives in
        ``tests/pcc/test_moe_gate_prefill2d.py``.

    Expect to relax ``moe_pcc_threshold`` below K2.6's 0.971: top-16 doubles the combine accumulation
    depth and it accumulates in the 3584 latent space at bf8, with the latent RMSNorm immediately
    after the sum. The failure mode to avoid is hunting a kernel defect that is really accumulation
    error.
    """
    topology = per_axis_topology(device_params["fabric_config"])
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
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        final_output_pcc=0.965,
        routed_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[KimiK3Config.ROUTED_EXPERT_ACTIVATION],
        shared_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
    )


# Mistral-Small-4-119B MoE. Own test function rather than a row on test_ds_moe because the upstream
# reference is a different class; the shared run_model body is unchanged.
#
# 640 x dgs 8 = 5120 tokens, matching the rest of the mistral4 suite. 128 experts at top-4 exercises
# the unfused extract -> FFN -> insert path that DSv3/Kimi/GLM only cover at top-8. 3200 x dgs 8 =
# 25600 tokens is the same shape at 5x the load. Random weights only: the checkpoint stacks the
# routed experts, so the pretrained fixture loads attention alone.
#
# GPT_DEVICE, not DEVICE_FP32. Mistral's router is softmax -> top-4 -> renormalize, which at zero
# bias equals top-4 on the raw logits followed by softmax over the selection -- what the GPT gate
# computes. Exact in real arithmetic; in bf16 the softmax quantizes before the top-k, so ~0.3% of
# tokens tie differently and land on another expert. That residual is the gap to a perfect PCC. The sigmoid modes would apply a different affinity silently. The synthesized
# correction bias is zeroed to match a router that has none, and the weight cache is keyed on that so
# a cache built with a random bias cannot be loaded over it.
@pytest.mark.parametrize(
    (
        "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, "
        "dispatch_buffer_capacity_factor, gate_fallback_mode, run_pcc_check"
    ),
    [
        # fmt: off
        pytest.param( 640, MistralSmall4Config.EMB_SIZE, MistralSmall4Config.MOE_INTERMEDIATE_SIZE, MistralSmall4Config.NUM_ROUTED_EXPERTS, MistralSmall4Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.GPT_DEVICE, True, marks=[pytest.mark.skipif(not is_blackhole(), reason="Mistral-Small-4 requires Blackhole"), pytest.mark.timeout(0)], id="mistral4-5k-pcc"),
        pytest.param(3200, MistralSmall4Config.EMB_SIZE, MistralSmall4Config.MOE_INTERMEDIATE_SIZE, MistralSmall4Config.NUM_ROUTED_EXPERTS, MistralSmall4Config.NUM_EXPERTS_PER_TOKEN, 5, GateComputeMode.GPT_DEVICE, True, marks=[pytest.mark.skipif(not is_blackhole(), reason="Mistral-Small-4 requires Blackhole"), pytest.mark.timeout(0)], id="mistral4-25k-pcc"),
        # fmt: on
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            # fabric2d, not torus_xy, and deliberately unlike the sibling 8x4 rows: measured on CI run
            # 32567382271, every torus_xy mistral4 case SKIPPED ("Galaxy TorusXY ... requires an
            # explicit ring/ring descriptor and a cabling-certified allocation"), and a skipped leg
            # reports green. FABRIC_2D is what this test ran under on ssalice/mistral4-119b-prefill,
            # where it genuinely passed on CI. Revert once bh_sc1 is ring-cabled.
            fabric2d_device_params(
                fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE,
            ),
            2 if is_blackhole() else 1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
def test_mistral4_moe(
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
    gate_fallback_mode,
    request,
):
    topology = per_axis_topology(device_params["fabric_config"])
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
