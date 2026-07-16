# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill combine operation in isolation.

This test verifies that the TTNN combine operation produces the same output as the
PyTorch reference implementation when combining expert outputs back to token positions.
Uses torch-generated dispatch inputs to isolate the combine operation.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


def run_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    run_pcc_check,
    dispatched_buffer_layout,
    use_fp8_output,
    is_ci_env,
    is_ci_v2_env,
):
    """Run the TTNN combine op in isolation against the torch reference. Shared body for the
    per-model test entrypoints below — they differ only on the (emb_dim, num_routed_experts,
    num_experts_per_tok) shape axis."""
    num_devices = mesh_device.get_num_devices()
    if num_devices >= 8 and not run_pcc_check and use_predictable_data:
        pytest.skip("8-chip perf only runs with random data")

    # Predictable inputs are torch.arange(...), which produces values up to ~1.8M and
    # overflows fp8_e4m3fn's ±448 range — overflow encodes as NaN, breaking PCC.
    # Only exercise the fp8 path with random (N(0,1)) data that fits in range.
    if use_fp8_output and use_predictable_data:
        pytest.skip("predictable inputs overflow fp8_e4m3fn range; run fp8 with random data")

    if use_fp8_output and not run_pcc_check:
        pytest.skip("fp8 perf test doesn't run PCC")

    # The fp8 output path is only wired up in combine_program_factory.cpp inside the
    # is_tile_layout branch (the c_18 untilized_output CB swap to Fp8_e4m3). The ROW_MAJOR
    # path has no untilize stage to retarget, so fp8 + row_major isn't a supported combo.
    if use_fp8_output and dispatched_buffer_layout != ttnn.TILE_LAYOUT:
        pytest.skip("fp8 combine output is only supported with TILE layout")

    # FP8_E4M3 hardware support (Fp8_e4m3 DataFormat in CBs, packer FP8 path) only exists on
    # Blackhole. TtCombineModule already raises ValueError if fp8_output is requested on
    # non-BH; skip cleanly here so this surfaces as "skipped" instead of an error.
    if use_fp8_output and mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("fp8 combine output requires Blackhole hardware")

    # ROW_MAJOR perf coverage is redundant in CI; TILE (all paths) and ROW_MAJOR PCC still run.
    if (is_ci_env or is_ci_v2_env) and not run_pcc_check and dispatched_buffer_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("ROW_MAJOR perf coverage does not run in CI")

    # 1-link linear/ring coverage is redundant on BH in CI. `1 in shape` selects the 1D
    # linear/ring meshes; 2D mesh / fabric2d (both dims > 1) and 2-link variants still run.
    if (is_ci_env or is_ci_v2_env) and is_blackhole() and num_links == 1 and 1 in tuple(mesh_device.shape):
        pytest.skip("1-link linear/ring coverage does not run on BH in CI")

    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()

    # Log fabric config
    logger.debug(f"Fabric max payload size: {ttnn.get_tt_fabric_max_payload_size_bytes()}")

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Combine {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} {seq_len_per_chip=} {emb_dim=} "
        f"{num_routed_experts=} {num_experts_per_tok=} {use_predictable_data=} {num_links=} {topology=}"
    )

    # Compute configuration
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
    logger.debug(
        f"{experts_per_chip=}, {metadata_len=}, {max_dispatch_buffer_token_size=}, {max_dispatched_tokens_per_expert=}"
    )

    # Step 1: Generate initial inputs using torch
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            dispatch_group_size,
            seq_len_per_chip,
            emb_dim,
            num_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size,
            seq_len_per_chip,
            emb_dim,
            num_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using RANDOM test data")

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    log_expert_dispatch_table(
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
    )

    # Compute gate outputs before dispatch (same for all EP ranks since indices are shared)
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Initialize torch dispatch module with num_dispatch_groups support
    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Run dispatch for each EP rank with rank-specific weights
    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    # Use different sharding: shard both dimensions
    mesh_mapper = get_ep_mesh_mapper(mesh_device)

    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper,
        layout=dispatched_buffer_layout,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_region_offsets = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    torch_combine = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )

    torch_output = torch_combine(dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets)

    # Quantize the torch combine output to fp8_e4m3fn so the reference matches the dtype
    # the TT combine produces in fp8 mode. Round-trip back to bfloat16 because downstream
    # validation expects a real float dtype; values keep fp8 precision.
    if use_fp8_output:
        torch_output = torch_output.to(torch.float8_e4m3fn).to(torch.bfloat16)

    # Run ttnn combine
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=False,
        fp8_output=use_fp8_output,
    )

    tt_output = tt_combine(
        tt_dispatched_buffer,
        tt_dispatched_metadata,
        tt_expert_token_counts,
        tt_expert_region_offsets,
    )

    # [debug] Drain the combine flow-control traces ([rxlog]/[txlog]) that the fabric routers flushed to DRAM
    # during the combine window into per-(device, eth core) files, sitting alongside the DPRINT output. Sync
    # first so every router's DRAM write (including the STOP-marker tail flush) has landed before we read back.
    ttnn.synchronize_device(mesh_device)
    ttnn.dump_detailed_fabric_logs(mesh_device, "generated/dprint/combine-dump/")

    # [debug] Dump the per-device combine connectivity (eth-to-eth links + sender/untilizer placement) the
    # program factory captured, alongside the flow-control traces so the topology and traces sit together.
    ttnn.experimental.deepseek_prefill.dump_combine_connectivity(mesh_device, "generated/dprint/combine-dump/")

    if not run_pcc_check:
        ttnn.synchronize_device(mesh_device)
        logger.debug("Skipping PCC validation (run_pcc_check=False)")
        return

    # Step 6: Convert ttnn output to torch for comparison
    mesh_composer = get_ep_mesh_composer(mesh_device)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)
    if use_fp8_output:
        # ttnn.to_torch returns a torch.float8_e4m3fn tensor for FP8_E4M3 device tensors
        # (see ttnn/ttnn/operations/core.py). Widen to bfloat16 for validation, since
        # validate_combine_output expects a regular float dtype.
        assert (
            tt_output_torch.dtype == torch.float8_e4m3fn
        ), f"expected torch.float8_e4m3fn fp8 combine output, got {tt_output_torch.dtype}"
        tt_output_torch = tt_output_torch.to(torch.bfloat16)

    # Step 7: Verify correctness
    assert_output_shape(tt_output_torch, num_dispatch_groups, dispatch_group_size, "combine output")

    # Validate combine output (EP-rank aware)
    # NOTE: Current combine kernel does NOT all-reduce across EP ranks.
    # Each EP rank's output only contains data for tokens that EP rank processed.
    # Output positions not written by local combine contain uninitialized garbage.
    # This comparison only checks the EP rank that actually processed each token.
    #
    # FP8 path: ~3-bit mantissa quantization makes allclose too tight (single-ULP rounding
    # near magnitude 2 already produces 0.25 differences). Switch to PCC, matching what
    # the dispatch fp8 PR does for the same reason.
    result = validate_combine_output(
        torch_output,
        tt_output_torch,
        indices,
        num_dispatch_groups,
        num_routed_experts,
        use_pcc=use_fp8_output,
        verbose=True,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
        experts_per_chip=experts_per_chip,
    )

    log_validation_results(
        results=[result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Combine Validation Results",
    )

    if not result.passed:
        log_combine_mismatch_details(result.mismatches, torch_output, tt_output_torch)
        log_per_chip_statistics(result.mismatches, dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

    result.assert_passed("Combine data mismatch")

    logger.debug("✅ TTNN combine operation matches torch reference!")


# Per-model combine shapes as (id_prefix, config, extended_model). Each model contributes a pcc
# param (seq 128, // 16 experts, top-4) and a perf param (seq 640, // 4 experts, top-2). DeepSeek
# V3 is the baseline and runs by default; every other model is gated behind
# @pytest.mark.extended_model. dispatch_buffer_capacity_factor is ceil(N/2) of the most
# conservative integer N such that dgs*seq*N >= worst-case dispatch buffer.
COMBINE_MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("glm_51", GLM51Config, True),
    ("kimi_k26", KimiK26Config, True),
    ("minimax_m27", MiniMaxM27Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
]


def combine_shape_params():
    """Build the per-model (shape, run_pcc_check) parametrization. Non-baseline models carry the
    extended_model marker on their params so they stay gated exactly as the separate tests were."""
    params = []
    for name, config, extended in COMBINE_MODELS:
        marks = (pytest.mark.extended_model,) if extended else ()
        shapes = [
            ("pcc", 128, config.NUM_ROUTED_EXPERTS // 16, 4, 4, True),
            ("perf_no_pcc", 640, config.NUM_ROUTED_EXPERTS // 4, 2, 8, False),
        ]
        for shape_id, seq, num_experts, topk, capacity, run_pcc in shapes:
            params.append(
                pytest.param(
                    seq,
                    config.EMB_SIZE,
                    num_experts,
                    topk,
                    capacity,
                    run_pcc,
                    marks=marks,
                    id=f"{name}-{shape_id}",
                )
            )
    return params


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor, run_pcc_check",
    combine_shape_params(),
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    ALL_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
@pytest.mark.parametrize(
    "dispatched_buffer_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "row_major"],
)
@pytest.mark.parametrize("use_fp8_output", [False, True], ids=["bf16_out", "fp8_out"])
def test_ttnn_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    run_pcc_check,
    dispatched_buffer_layout,
    use_fp8_output,
    is_ci_env,
    is_ci_v2_env,
):
    run_combine(
        mesh_device,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        num_links,
        topology,
        use_predictable_data,
        run_pcc_check,
        dispatched_buffer_layout,
        use_fp8_output,
        is_ci_env,
        is_ci_v2_env,
    )
