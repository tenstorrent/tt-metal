# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill dispatch operation in isolation.

This test verifies that the TTNN dispatch operation produces the same output as the
PyTorch reference implementation when dispatching tokens to experts.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    validate_dispatch_buffer,
    validate_dispatch_buffer_pcc,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results

# =====
# mesh 4x2
#
# ---------
# | X0  | X0 |
# | W0  | W0 |
# | I0  | I0 |
# ------------
# | X1  | X1 |
# | W1  | W1 |
# | I1  | I1 |
# ------------
# | X2  | X2 |
# | W2  | W2 |
# | I2  | I2 |
# ------------
# | X3  | X3 |
# | W3  | W3 |
# | I3  | I3 |
# ------------
#                   MeshDevice(rows=4, cols=2)
# ┌──────────────────────────────┬──────────────────────────────┐
# │          Dev. ID: 4          │          Dev. ID: 6          │
# │            (0, 0)            │            (0, 1)            │
# │       LinMeshCoord=0         │       LinMeshCoord=1         │
# │       LogicalCoord=0         |       LogicalCoord=0         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 2          │          Dev. ID: 3          │
# │            (1, 0)            │            (1, 1)            │
# │       LinMeshCoord=2         │       LinMeshCoord=3         │
# │       LogicalCoord=1         |       LogicalCoord=1         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 1          │          Dev. ID: 0          │
# │            (2, 0)            │            (2, 1)            │
# │       LinMeshCoord=4         │       LinMeshCoord=5         │
# │       LogicalCoord=2         |       LogicalCoord=2         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 5          │          Dev. ID: 7          │
# │            (3, 0)            │            (3, 1)            │
# │       LinMeshCoord=6         │       LinMeshCoord=7         │
# │       LogicalCoord=3         |       LogicalCoord=3         │
# │                              │                              │
# └──────────────────────────────┴──────────────────────────────┘
# Dev. ID is physical mapping
# LinMeshCoord is used for fabric transfers
# LogicalCoord is coordinate in withing a2a dispatch group


def run_dispatch(
    mesh_device,
    model_name,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    input_layout,
    input_dtype,
    output_dtype,
    fp8_scaled_input,
    verbose,
    run_pcc_check,
    is_ci_env,
    is_ci_v2_env,
):
    """Run the TTNN dispatch op in isolation against the torch reference. Shared body for the
    per-model test entrypoints below — they differ only on the (emb_dim, num_routed_experts,
    num_experts_per_tok) shape axis."""
    num_devices = mesh_device.get_num_devices()
    if num_devices >= 8 and not run_pcc_check and use_predictable_data:
        pytest.skip("8-chip perf only runs with random data")

    fp8_input = input_dtype == ttnn.fp8_e4m3
    fp8_output = output_dtype == ttnn.fp8_e4m3

    # Predictable inputs are torch.arange(...), which produces values up to ~1.8M and
    # overflows fp8_e4m3fn's ±448 range — overflow encodes as NaN, breaking PCC.
    # Only exercise the fp8 path (input or output) with random (N(0,1)) data that fits in range.
    if (fp8_output or fp8_input) and use_predictable_data:
        pytest.skip("predictable inputs overflow fp8_e4m3fn range; run fp8 with random data")

    if (fp8_output or fp8_input) and is_wormhole_b0():
        pytest.skip("fp8 (input or output) not supported on Wormhole hardware")

    # FP8_E4M3 is a ROW_MAJOR-only tensor spec (no tiled fp8 layout exists), so an fp8 input
    # tensor can only be ROW_MAJOR. The tile path's input is therefore always bf16.
    if fp8_input and input_layout == ttnn.TILE_LAYOUT:
        pytest.skip("FP8_E4M3 input is ROW_MAJOR-only; no tiled fp8 input tensor exists")

    # Row-major dispatch is a pure byte copy (no compute), so it cannot convert dtypes: the input
    # dtype must equal the output dtype. The tile path has a compute packer and converts freely.
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype != output_dtype:
        pytest.skip("row_major dispatch requires input dtype == output dtype")

    # 1-link linear/ring coverage is redundant on BH in CI. `1 in shape` selects the 1D
    # linear/ring meshes; 2D mesh / fabric2d (both dims > 1) and 2-link variants still run.
    if (is_ci_env or is_ci_v2_env) and is_blackhole() and num_links == 1 and 1 in tuple(mesh_device.shape):
        pytest.skip("1-link linear/ring coverage does not run on BH in CI")

    torch.manual_seed(42)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Dispatch {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} {seq_len_per_chip=} {emb_dim=} {num_routed_experts=} {num_experts_per_tok=} {use_predictable_data=} {num_links=} {topology=}"
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
        fp8_scaled_input=fp8_scaled_input,
    )
    logger.debug(
        f"{experts_per_chip=}, {metadata_len=}, {max_dispatch_buffer_token_size=}, {max_dispatched_tokens_per_expert=}"
    )

    # Initialize inputs using helper function
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using RANDOM test data")

    # Clamp inputs to fp8_e4m3fn's finite range so any future scale/seed change can't push
    # values into the overflow→NaN region. randn(0,1) is already well inside ±448, so this
    # is a no-op for the current data and a guardrail for later.
    if fp8_output or fp8_input:
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        x = x.clamp(min=fp8_info.min, max=fp8_info.max)

    # For fp8 input the device quantizes the input tensor to fp8, so quantize the reference input
    # too — otherwise bf16-output combos would compare fp8-rounded device values against
    # full-precision reference values.
    if fp8_input:
        x = x.to(torch.float8_e4m3fn).to(torch.float32)

    logger.debug(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # x and indices: sharded across SP axis, replicated across EP ranks
    mesh_mapper_replicated = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

    scales = None
    tt_scales = None
    if fp8_input:
        # FP8_E4M3 is ROW_MAJOR-only, and the on-device float32->fp8 conversion path would build an
        # illegal FP8_E4M3 TILE intermediate (typecast requires TILE). Construct on host instead
        # (host does float32->fp8 directly with no tile round-trip), then move to device.
        tt_x = ttnn.from_torch(x, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.fp8_e4m3)
        tt_x = ttnn.to_device(tt_x, mesh_device)
    else:
        tt_x = ttnn.from_torch(
            x, mesh_mapper=mesh_mapper_replicated, layout=input_layout, device=mesh_device, dtype=input_dtype
        )

    # Per-token fp8 scales: arbitrary fp32 values (dispatch only byte-copies them into the metadata
    # tail), one row of emb_dim/128 per token, sharded/replicated exactly like x.
    if fp8_scaled_input:
        num_scale_blocks = emb_dim // 128
        scales = torch.randn(dispatch_group_size, seq_len_per_chip, num_scale_blocks, dtype=torch.float32)
        tt_scales = ttnn.from_torch(
            scales,
            mesh_mapper=mesh_mapper_replicated,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.float32,
        )

    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_replicated,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_replicated,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )

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

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        fp8_output=fp8_output,
    )

    # Compute gate outputs (offsets and token counts) before dispatch
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Forward pass through TTNN dispatch
    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    tt_dispatched, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table, scales=tt_scales
    )

    if not run_pcc_check:
        ttnn.synchronize_device(mesh_device)
        logger.debug("Skipping PCC validation (run_pcc_check=False)")
        return

    # Run torch reference for all EP ranks at once
    torch_dispatched, torch_metadata = torch_dispatch_module(x, weights, indices, expert_offsets, scales=scales)

    # Convert TTNN outputs to torch for comparison
    mesh_composer = get_ep_mesh_composer(mesh_device)
    if fp8_output:
        # Quantize the torch reference to fp8_e4m3fn so it carries the same precision as the TT
        # dispatch output (which packs BF16->FP8 at the untilize stage), isolating routing
        # correctness from fp8 quantization noise. Round-trip to float32 since
        # validate_dispatch_buffer_pcc expects a real float dtype — matching the TT side below.
        torch_dispatched = torch_dispatched.to(torch.float8_e4m3fn).to(torch.float32)

        # ttnn.to_torch returns a torch.float8_e4m3fn tensor for FP8_E4M3 device tensors; widen to fp32 for PCC.
        tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer)
        assert (
            tt_out_dispatched.dtype == torch.float8_e4m3fn
        ), f"expected float8_e4m3fn fp8 output, got {tt_out_dispatched.dtype}"
        tt_out_dispatched = tt_out_dispatched.to(torch.float32)
    else:
        tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)

    assert_output_shape(tt_out_dispatched, num_dispatch_groups, dispatch_group_size, "dispatched buffer")

    # Quick sanity check of first elements (verbose mode only)
    if verbose:
        logger.debug(f"{tt_out_dispatched[0][0][0][0][0]=} | {tt_out_dispatched[0][1][0][0][0]=}")
        logger.debug(f"{torch_dispatched[0][0][0][0][0]=} | {torch_dispatched[0][1][0][0][0]=}")
        logger.debug(f"{tt_out_metadata[0][0][0][0][0:4]=} | {tt_out_metadata[0][1][0][0][0:4]=}")
        logger.debug(f"{torch_metadata[0][0][0][0][0:4]=} | {torch_metadata[0][1][0][0][0:4]=}")
        logger.debug(f"{expert_token_counts.shape=}, {expert_token_counts=}")
        logger.debug(f"{expert_offsets.shape=}, {expert_offsets=}")

    # Verify dispatched data matches reference (each EP rank against its torch reference).
    # FP8 path quantizes the buffer (~3-bit mantissa), so allclose is too tight — use PCC.
    if fp8_output:
        buffer_result = validate_dispatch_buffer_pcc(
            torch_dispatched,
            tt_out_dispatched,
            expert_region_offsets,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=verbose,
        )
    else:
        buffer_result = validate_dispatch_buffer(
            torch_dispatched,
            tt_out_dispatched,
            expert_region_offsets,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=verbose,
        )

    metadata_result = validate_dispatch_metadata(
        torch_metadata,
        tt_out_metadata,
        expert_region_offsets,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        verbose=verbose,
    )

    # Log summaries and visualization
    log_validation_results(
        results=[buffer_result, metadata_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Dispatch Validation Results",
    )

    assert (
        buffer_result.passed and metadata_result.passed
    ), f"Some slots did not match! buffer={buffer_result.passed} metadata={metadata_result.passed} Check logs for details."

    # validate_dispatch_metadata only checks the 3 routing fields; the fp8 scale tail (fields 3..)
    # is dispatched as a pure int32 bit-copy, so it must match the reference exactly. Compare only
    # the slots the reference actually filled (torch initializes metadata to -1; field 1 = token_idx
    # is >= 0 only for real dispatched tokens), since unfilled device slots are uninitialized.
    if fp8_scaled_input:
        filled = torch_metadata[..., 1] >= 0
        mask = filled.unsqueeze(-1).expand_as(torch_metadata[..., 3:])
        ref_tail = torch_metadata[..., 3:][mask]
        out_tail = tt_out_metadata[..., 3:].to(torch.int32)[mask]
        assert torch.equal(ref_tail, out_tail), (
            "fp8 per-token scales in the metadata tail (fields 3..) do not match the reference "
            "(dispatch must byte-copy each token's scales unchanged)."
        )
        logger.debug("✅ fp8 per-token scales match in the metadata tail!")

    logger.debug("✅ TTNN dispatch operation matches torch reference!")


# Per-model dispatch shapes as (id_prefix, config, extended_model). Each model contributes two
# param sets sharing the same scaling rationale: these models deploy their routed experts across a
# 32-chip Galaxy (experts/chip = NUM_ROUTED_EXPERTS // num_devices), but this op test runs on at
# most 8 chips. The perf param scales experts down by 32/8 = 4 to preserve per-chip load; the PCC
# param shrinks further (// 16 experts, half experts/token) to keep the full comparison cheap.
# dispatch_buffer_capacity_factor is ceil(N/2) of the most conservative integer N such that
# dgs*seq*N >= worst-case dispatch buffer; real traffic stays well under.
#
# DeepSeek V3 is the baseline shape and runs by default; every other model is gated behind
# @pytest.mark.extended_model.
DISPATCH_MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("glm_51", GLM51Config, True),
    ("kimi_k26", KimiK26Config, True),
    ("minimax_m27", MiniMaxM27Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
]

# Models whose dispatch supports the fp8-compression path (fp8 input + per-token scale tail). Their
# params carry the fp8_disp_compression marker so a workflow -k/-m can select the fp8-scaled dispatch
# tests without enumerating model names. Must match the fp8-scaled support gate in run_dispatch.
FP8_DISP_COMPRESSION_MODELS = ("dsv3", "dsv4_pro", "dsv4_flash", "kimi_k26")


def dispatch_shape_params():
    """Build the per-model (shape, run_pcc_check) parametrization. Non-baseline models carry the
    extended_model marker; fp8-compression-capable models additionally carry fp8_disp_compression, so
    both stay selectable exactly as the separate tests were."""
    params = []
    for name, config, extended in DISPATCH_MODELS:
        marks = []
        if extended:
            marks.append(pytest.mark.extended_model)
        if name in FP8_DISP_COMPRESSION_MODELS:
            marks.append(pytest.mark.fp8_disp_compression)
        marks = tuple(marks)
        # (model_name, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok,
        #  dispatch_buffer_capacity_factor, run_pcc_check)
        params.append(
            pytest.param(
                name, 32, config.EMB_SIZE, config.NUM_ROUTED_EXPERTS // 16, 4, 4, True, marks=marks, id=f"{name}-pcc"
            )
        )
        params.append(
            pytest.param(
                name,
                640,
                config.EMB_SIZE,
                config.NUM_ROUTED_EXPERTS // 4,
                2,
                8,
                False,
                marks=marks,
                id=f"{name}-perf_no_pcc",
            )
        )
    return params


@pytest.mark.parametrize(
    "model_name, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor, run_pcc_check",
    dispatch_shape_params(),
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    ALL_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
@pytest.mark.parametrize(
    "input_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "row_major"],
)
# input_dtype folds the fp8-scaled flavor into the input axis. fp8 only ever reaches dispatch as
# compressed input carrying its per-token fp32 scale tail (fp8_scaled_in); a bf16 input that dispatch
# casts to fp8 internally is covered by bf16_in + fp8 output. Unscaled fp8 input is not a real path,
# so it is not parametrized. There is no bf16+scaled combo — scales only apply to fp8 input.
@pytest.mark.parametrize(
    "input_dtype, fp8_scaled_input",
    [
        (ttnn.bfloat16, False),
        (ttnn.fp8_e4m3, True),
    ],
    ids=["bf16_in", "fp8_scaled_in"],
)
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["bf16_out", "fp8_out"])
@pytest.mark.parametrize("verbose", [False])
def test_ttnn_dispatch(
    mesh_device,
    model_name,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    input_layout,
    input_dtype,
    output_dtype,
    fp8_scaled_input,
    verbose,
    run_pcc_check,
    is_ci_env,
    is_ci_v2_env,
):
    run_dispatch(
        mesh_device,
        model_name,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        num_links,
        topology,
        use_predictable_data,
        input_layout,
        input_dtype,
        output_dtype,
        fp8_scaled_input,
        verbose,
        run_pcc_check,
        is_ci_env,
        is_ci_v2_env,
    )
