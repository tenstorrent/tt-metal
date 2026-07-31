# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill combine operation in isolation.

This test verifies that the TTNN combine operation produces the same output as the
PyTorch reference implementation when combining expert outputs back to token positions.
Uses torch-generated dispatch inputs to isolate the combine operation.
"""

from dataclasses import dataclass
import itertools
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_20b_config import GptOss20BConfig
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
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
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology


def run_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    topology,
    use_predictable_data,
    run_pcc_check,
    dispatched_buffer_layout,
    use_fp8_output,
    num_links=2,
):
    """Run the TTNN combine op in isolation against the torch reference. Shared body for the
    per-model test entrypoints below — they differ only on the (emb_dim, num_routed_experts,
    num_experts_per_tok) shape axis."""
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


@dataclass
class _Test_Mesh:
    full_model_mesh: tuple[int, int]  # Intended for full production-scale testing
    target_meshes: list[
        # Any mesh smaller than the full_model_mesh will be assumed to be intended for a proxy test and
        # will result in model hyperparams dowscaling according to the op-specific downscaling function
        dict[tuple[int, int], ttnn.FabricConfig]
    ]


SINGLE_GLX_AND_PROXY_MESHES = _Test_Mesh(
    (8, 4),
    {
        # Preserve the existing SP=8 LoudBox proxy with its wrapped Fabric2D equivalent.
        (8, 4): ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        (8, 1): ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        (4, 1): ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        (4, 2): ttnn.FabricConfig.FABRIC_2D,
        (2, 2): ttnn.FabricConfig.FABRIC_2D,
    },
)


ONLY_PROXY_QB_MESH = _Test_Mesh(
    (8, 4),
    {
        (2, 2): ttnn.FabricConfig.FABRIC_2D,
    },
)


# Per-model combine shapes as (id_prefix, config, extended_model). Each model contributes a pcc
# param (seq 128, // 16 experts, top-4) and a perf param (seq 640, // 4 experts, top-2). DeepSeek
# V3 is the baseline and runs by default; every other model is gated behind
# @pytest.mark.extended_model. dispatch_buffer_capacity_factor is ceil(N/2) of the most
# conservative integer N such that dgs*seq*N >= worst-case dispatch buffer.
COMBINE_MODELS = [
    ("dsv3", DeepSeekV3Config, SINGLE_GLX_AND_PROXY_MESHES),
    ("glm_51", GLM51Config, ONLY_PROXY_QB_MESH),
    ("kimi_k26", KimiK26Config, ONLY_PROXY_QB_MESH),
    ("minimax_m27", MiniMaxM27Config, ONLY_PROXY_QB_MESH),
    ("dsv4_pro", DeepSeekV4ProConfig, ONLY_PROXY_QB_MESH),
    ("dsv4_flash", DeepSeekV4FlashConfig, ONLY_PROXY_QB_MESH),
    ("gptoss_120b", GptOss120BConfig, ONLY_PROXY_QB_MESH),
]


# Scales down model hyper-params for a given hardware to obtain good/meaningful proxy test
# How exactly to scale it down is op-specific (more precisely - even op-implementation specific)
# Thus it makes sense for this to be combine-specific function
def _model_scaledown(model, ref_mesh, target_mesh, pcc_only):
    # number of experts has to be reduced to preserve the experts per chip
    ref_num_chips = ref_mesh[0] * ref_mesh[1]
    target_num_chips = target_mesh[0] * target_mesh[1]
    if ref_num_chips != target_num_chips:
        # oreder of the operation keeps the number of routed experts divisible by the number of chips
        model.NUM_ROUTED_EXPERTS = (model.NUM_ROUTED_EXPERTS // ref_num_chips) * target_num_chips

    # number of experts selected to proces every token (top-K) has to be scaled to preserve average expert activation per dispatch group
    # Clamped at 1: the ideal scale is topK * target_groups / ref_groups, which lands below 1 whenever the
    # model routes to fewer experts than the reference mesh has dispatch groups (e.g. GPT-OSS top-4 on a
    # 4x8 reference collapsed to a single-group proxy). Routing every token to zero experts is not a test.
    if ref_mesh[1] != target_mesh[1]:
        model.NUM_EXPERTS_PER_TOKEN = max(1, (model.NUM_EXPERTS_PER_TOKEN // ref_mesh[1]) * target_mesh[1])

    # further reduce these two hyperparams in case of pcc check test to get faster, although not perf-representative test
    if pcc_only:
        model.NUM_ROUTED_EXPERTS = max(target_num_chips, model.NUM_ROUTED_EXPERTS // 16)
        model.NUM_EXPERTS_PER_TOKEN = max(2, model.NUM_EXPERTS_PER_TOKEN // 4)

    return model


def _topo_marker(mesh, fabric_cfg):
    if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_Y:
        return "ring"
    return f"mesh-{mesh[0]}x{mesh[1]}"


def _mesh_id(mesh, fabric_cfg):
    profile = {
        ttnn.FabricConfig.FABRIC_2D: "fabric2d",
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y: "torus-y",
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY: "torus-xy",
    }[fabric_cfg]
    return f"{profile}-{mesh[0]}x{mesh[1]}"


def _cross_product_conflated_cmb_test_dimensions():
    params = []
    for model_name, model_config_class, test_meshes in COMBINE_MODELS:
        for target_mesh, fabric_cfg in test_meshes.target_meshes.items():
            device_params = fabric_to_device_params(fabric_cfg)
            topo_marker = _topo_marker(target_mesh, fabric_cfg)
            marks = pytest.mark.requires_mesh_topology(mesh_shape=target_mesh, topology=topo_marker)
            test_scenarios = [
                ("pcc", 128, 4, True),
                ("perf_no_pcc", 640, 8, False),
            ]
            for test_scenario_id, seq_len_per_chip, dispatch_buffer_capacity_factor, run_pcc in test_scenarios:
                model_config = _model_scaledown(model_config_class(), test_meshes.full_model_mesh, target_mesh, run_pcc)

                num_experts = model_config.NUM_ROUTED_EXPERTS
                topk = model_config.NUM_EXPERTS_PER_TOKEN
                shape = target_mesh

                params.append(
                    pytest.param(
                        shape,
                        device_params,
                        seq_len_per_chip,
                        model_config.EMB_SIZE,
                        num_experts,
                        topk,
                        dispatch_buffer_capacity_factor,
                        run_pcc,
                        marks=marks,
                        id=f"{model_name}-{_mesh_id(target_mesh, fabric_cfg)}-{test_scenario_id}",
                    )
                )

    return params


def _unsupported_param_combos(**params):
    mesh_device = params["mesh_device"]
    run_pcc_check = params["run_pcc_check"]
    use_predictable_data = params["use_predictable_data"]
    use_fp8_output = params["use_fp8_output"]
    dispatched_buffer_layout = params["dispatched_buffer_layout"]
    is_ci_env = params["is_ci_env"]
    is_ci_v2_env = params["is_ci_v2_env"]
    is_bh = params["is_bh"]

    # This function is called before test cases are fully formed, so 'mesh_device' here, unlike in the test_ttnn_combine
    # function is not fully formed device object. Rather it is the first parametrization axis argument that parametrization
    # logic itterates over (which is also named 'mesh_device') and which is a simple shape tuple.
    num_devices = mesh_device[0] * mesh_device[1]
    if num_devices >= 8 and not run_pcc_check and use_predictable_data:
        return True

    # Predictable inputs are torch.arange(...), which produces values up to ~1.8M and
    # overflows fp8_e4m3fn's ±448 range — overflow encodes as NaN, breaking PCC.
    # Only exercise the fp8 path with random (N(0,1)) data that fits in range.
    if use_fp8_output and use_predictable_data:
        return True

    # fp8 perf test doesn't run PCC
    if use_fp8_output and not run_pcc_check:
        return True

    # The fp8 output path is only wired up in combine_program_factory.cpp inside the
    # is_tile_layout branch (the c_18 untilized_output CB swap to Fp8_e4m3). The ROW_MAJOR
    # path has no untilize stage to retarget, so fp8 + row_major isn't a supported combo.
    if use_fp8_output and dispatched_buffer_layout != ttnn.TILE_LAYOUT:
        return True

    # FP8_E4M3 hardware support (Fp8_e4m3 DataFormat in CBs, packer FP8 path) only exists on
    # Blackhole. TtCombineModule already raises ValueError if fp8_output is requested on
    # non-BH; skip cleanly here so this surfaces as "skipped" instead of an error.
    if use_fp8_output and not is_bh:
        return True

    # ROW_MAJOR perf coverage is redundant in CI; TILE (all paths) and ROW_MAJOR PCC still run.
    if (is_ci_env or is_ci_v2_env) and not run_pcc_check and dispatched_buffer_layout == ttnn.ROW_MAJOR_LAYOUT:
        return True

    # Otherwise don't uncollect the test case. Keep it.
    return False


# Test parametrization axes are semantically
# 1. Chip count and layout
#   1.1. mesh column size, which is consequently the size of a dispatch group
#   1.2. mesh row size, which is consequently the number of dispatch groups
# 2. Model-related (embed-dim, topK, num-experts, ...)
# 3. Input related (ISL, tile/RM, datum format, ...)
# 4. Scenario/type of test
#   4.1. accuracy or perf test
#   4.2. production test / proxy test on smaller hardware (like single DG simulation) / op generality test
#   4.3. random or predictable (fixed) data
#
# Each level-2 item in this list is a valid parametrization axis (semantically).
# However pytest requires some marks to be populated. And it allows marks to be calculated only based on a single axis values.
# Some of our marks are besed on multiple semantic axes (e.g. topology marker is calculated based on 1.1, 1.2, 2.1, 2.2 and 2.3)
# Industry standard work-around is conflating these axes into a single @pytest.mark.parametrize axis + a function which generates
# its values as a cross product of the semantical axis which are conflated. Then calculate marks based on the value of the resulting
# conflated axis, which from the perspective of the pytest is a single parametrization axis.
#
@pytest.mark.uncollect_if(pred=_unsupported_param_combos)
@pytest.mark.parametrize(
    "mesh_device, device_params, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor, run_pcc_check",
    _cross_product_conflated_cmb_test_dimensions(),
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
    device_params,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    use_predictable_data,
    run_pcc_check,
    dispatched_buffer_layout,
    use_fp8_output,
    is_ci_env,
    is_ci_v2_env,
):
    topology = per_axis_topology(device_params["fabric_config"])[0]
    run_combine(
        mesh_device,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        topology,
        use_predictable_data,
        run_pcc_check,
        dispatched_buffer_layout,
        use_fp8_output,
    )


def _all_externally_owned_test_cases():
    def _tc(mesh, fabric_cfg, seq_len_per_chip, num_links, model):
        model_name = model.__class__.__name__.removesuffix("Config")
        return pytest.param(
            mesh,
            fabric_to_device_params(fabric_cfg),
            seq_len_per_chip,
            model.EMB_SIZE,
            model.NUM_ROUTED_EXPERTS,
            model.NUM_EXPERTS_PER_TOKEN,
            num_links,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=_topo_marker(mesh, fabric_cfg)),
            id=f"{model_name}-{mesh[0]}x{mesh[1]}-{fabric_cfg.name.lower()}-{seq_len_per_chip}-{num_links}link-rand-pcc-tile-bf16",
        )

    return [
        # Full scale mesh tests as invoked in production scenarios
        _tc((4, 8), ttnn.FabricConfig.FABRIC_1D, 1024, 4, GptOss20BConfig()),
        _tc((4, 8), ttnn.FabricConfig.FABRIC_1D, 128, 2, GptOss120BConfig()),
        _tc((4, 8), ttnn.FabricConfig.FABRIC_1D, 1280, 2, GptOss120BConfig()),
        _tc((8, 4), ttnn.FabricConfig.FABRIC_1D, 128, 2, MiniMaxM3Config()),
        _tc((8, 4), ttnn.FabricConfig.FABRIC_1D, 640, 2, MiniMaxM3Config()),
        # Proxy tests executable on CIs which run op unit tests
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 1024, 4, _model_scaledown(GptOss20BConfig(), (4, 8), (4, 1), False)),
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 128, 2, _model_scaledown(GptOss120BConfig(), (4, 8), (4, 1), False)),
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 1280, 2, _model_scaledown(GptOss120BConfig(), (4, 8), (4, 1), False)),
        _tc((8, 1), ttnn.FabricConfig.FABRIC_1D, 128, 2, _model_scaledown(MiniMaxM3Config(), (8, 4), (8, 1), False)),
        _tc((8, 1), ttnn.FabricConfig.FABRIC_1D, 640, 2, _model_scaledown(MiniMaxM3Config(), (8, 4), (8, 1), False)),
    ]


def _unsupported_externally_owned_param_combos(**params):
    # Blackhole exposes 2 fabric links per device, Wormhole 4.
    if params["num_links"] > 2 and params["is_bh"]:
        return True

    return False


@pytest.mark.uncollect_if(pred=_unsupported_externally_owned_param_combos)
@pytest.mark.parametrize(
    "mesh_device, device_params, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, num_links",
    _all_externally_owned_test_cases(),
    indirect=["mesh_device", "device_params"],
)
def test_externally_owned_cases(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    num_links,
):
    # Same derivation test_ttnn_combine uses: combine drives cluster_axis=0, so it wants the
    # sp-axis topology of whatever fabric the case opened (Linear for the unwrapped FABRIC_1D).
    topology = per_axis_topology(device_params["fabric_config"])[0]
    run_combine(
        mesh_device,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        2,  # buffer capacity factor
        topology,
        use_predictable_data=False,
        run_pcc_check=True,
        dispatched_buffer_layout=ttnn.TILE_LAYOUT,
        use_fp8_output=False,
        num_links=num_links,
    )


# ---------------------------------------------------------------------------
# CombineFabric2D — isolated fabric-transfer experiment (see cmb-f2d plan).
#
# The test states what it wants moved as a list of explicit movement descriptors — (src, in_base_token,
# dst, out_base_token) — hands them to the op along with an input region and a zeroed output region, and
# afterwards walks the same list to check every movement landed. Nothing about HOW the op moves the data
# appears here: no producer identity, no core count, no placement, no telemetry. The coupling that used
# to be implicit (an assumed mapping from a neighbour's input region to this chip's output region) is now
# exactly the descriptor list, written by the test and read by the op.
#
# The one hardware assumption is that each device has two distinct neighbours along CMBF2D_AXIS, which is
# how production boards are wired; cables per neighbour comes from the mesh config's num_links. On
# differently wired hardware the op rejects the descriptor list rather than silently doing something else.
#
# Select the config with: -k 'fabric2d-torus-xy-8x4-2link'
# ---------------------------------------------------------------------------
CMBF2D_BWINFO_PATH = "generated/cmbf2d/bwinfo.txt"

# Sweep knobs. The op's own defaults live in C++; these let a sweep driver vary one axis per pytest
# invocation (one device open per point) without touching the test body. CMBF2D_TAG, when set, moves
# the report to bwinfo_<tag>.txt so a sweep's points don't overwrite each other.
#
# tokens  tokens one movement copies — the knob that sets the total traffic. Nothing about how the op
#         buffers them (its L1 ring depth) is visible here.
CMBF2D_TOKENS = int(os.environ.get("CMBF2D_TOKENS", "100"))
CMBF2D_TOKEN_BYTES = int(os.environ.get("CMBF2D_TOKEN_BYTES", "14336"))
CMBF2D_TAG = os.environ.get("CMBF2D_TAG", "")
# 1 => producer records the fine-grained stall buckets (costs a few percent of the bandwidth
# it is measuring). Off for headline numbers, on to explain them.
CMBF2D_STALL = int(os.environ.get("CMBF2D_STALL", "0"))
# Forwarded tokens between semaphore bumps to the downstream reader. Purely a tuning knob — a bump always
# follows a chunk's sentinel, so accuracy holds for any value >= 1; this only sets how finely the downstream
# reader can pipeline WITHIN a chunk. Swept in P9.3.
CMBF2D_FWD_BUMP = int(os.environ.get("CMBF2D_FWD_BUMP", "32"))
# 0 = nearest destination first then all forwarding; 1 = furthest first with forwarding interleaved, so
# downstream cores are handed work earlier. Scheduling only — accuracy is identical either way.
CMBF2D_ORDER = int(os.environ.get("CMBF2D_ORDER", "1"))
# Mesh axis whose neighbours the op talks to. Matches the op's `axis` argument.
CMBF2D_AXIS = 0
# Blackhole AICLK the bandwidth numbers assume. Telemetry gives cycle counts; turning those into GB/s
# needs a clock, and a device running at a different frequency would silently rescale every result. So the
# clock is ASSERTED rather than trusted: a mismatch fails the test, which is the loudest signal available
# during development. (Candidate for demotion to a warning before check-in.)
CMBF2D_EXPECTED_CLOCK_MHZ = 1350
CMBF2D_CLOCK_TOLERANCE = 0.05
# Bytes the op asks the fabric to carry per packet: the token PLUS the 64-byte routing tail phase 9 needs
# to forward alongside it. 14336 + 64 = 14400 = 64 * 225, so a forwarding-buffer page stays DRAM-aligned,
# and it is comfortably under Blackhole's 15232 B ceiling ((16384 NoC max - 96 header) floored to Bfp8_b
# tiles, erisc_datamover_builder.hpp:465-476) and 16-byte aligned as the fabric validator requires.
CMBF2D_PACKET_BYTES = CMBF2D_TOKEN_BYTES + 64
# Raising the fabric payload is a DEVICE-WIDE setting, so CombineFabric2D carries its own mesh config
# rather than taking one from ALL_MESH_CONFIGS. The op only ever runs on this single config, and this way
# a bigger packet cannot perturb any other test that shares that list. The id is unchanged, so
# `-k fabric2d-torus-xy-8x4-2link` still selects it.
CMBF2D_MESH_CONFIGS = [
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": create_fabric_router_config(max_payload_size=CMBF2D_PACKET_BYTES),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        ttnn.Topology.Ring,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-torus-xy-8x4-2link",
    ),
]


def _cmbf2d_bwinfo_path():
    return f"generated/cmbf2d/bwinfo_{CMBF2D_TAG}.txt" if CMBF2D_TAG else CMBF2D_BWINFO_PATH


def _cmbf2d_slots_per_device(mesh_device, axis, num_links):
    """How many `tokens`-sized slots each device's input and output region is cut into.

    Every device sends to every OTHER device on `axis`, once per link: (axis_extent - 1) * num_links. The
    two regions are the same size because the traffic pattern is ring-symmetric — a device sends
    num_links * tokens to each of the other axis_extent-1 devices, and receives exactly that much from
    each of them.
    """
    return (tuple(mesh_device.shape)[axis] - 1) * num_links


def _cmbf2d_plan_movements(mesh_device, axis, num_links, tokens):
    """Every data movement in the run: every device sends `tokens` tokens to EVERY other device on
    `axis`, once per link.

    Slot scheme. Both regions are cut into `(axis_extent - 1) * num_links` slots of `tokens` tokens each,
    and a slot is named by the pair (ring offset `delta`, `link`):

        slot(delta, link) = (delta - 1) * num_links + link,   delta in 1 .. axis_extent-1

    Chip C's INPUT slot (delta, link) goes to chip C+delta's OUTPUT slot (delta, link). That one rule
    makes both sides collision-free with no ordering convention to remember:
      * input  — each (delta, link) is used exactly once per source chip;
      * output — chip D's slot (delta, link) can only be written by chip D-delta, so exactly one
                 movement claims it.

    So every slot on every chip is both read once and written once. The op's validation re-derives that
    property from the list alone (no destination claimed twice, input coverage gap-free from 0) rather
    than trusting this docstring.

    NOTE this says nothing about HOW the traffic gets there. Destinations more than one hop away are the
    op's problem, not the test's: the descriptor names a chip, and the op decides the route.
    """
    mesh_shape = tuple(mesh_device.shape)
    extent = mesh_shape[axis]
    assert extent >= 3, f"axis {axis} extent {extent}: need 3+ for two distinct neighbours"

    def _at_offset(coord, delta):
        nbr = list(coord)
        nbr[axis] = (nbr[axis] + delta) % extent
        return nbr

    movements = []
    for coord in itertools.product(*(range(n) for n in mesh_shape)):
        for delta in range(1, extent):
            for link in range(num_links):
                slot = (delta - 1) * num_links + link
                movements.append(
                    ttnn._ttnn.operations.experimental.CombineFabric2dMovement(
                        src=list(coord),
                        in_base_token=slot * tokens,
                        dst=_at_offset(coord, delta),
                        out_base_token=slot * tokens,
                    )
                )
    return movements


def _cmbf2d_make_regions(mesh_device, axis, tokens, token_size_bytes, num_links):
    """The op's two DRAM regions, sized for one slot per (destination, link) per device: random input,
    zeroed output (so an unwritten token fails the check instead of passing on leftover garbage).

    Returns `(host_input, dev_input, dev_output)`. The host tensor is retained deliberately: it, and not
    a readback of `dev_input`, is the reference the accuracy check compares against — see
    `_cmbf2d_check_accuracy`.

    Sharded with the explicit 2D form (tensor dim0 -> mesh axis 0, dim1 -> mesh axis 1) so "which device
    holds which block" needs no ordering convention.
    """
    rows, cols = tuple(mesh_device.shape)
    slots = _cmbf2d_slots_per_device(mesh_device, axis, num_links)
    shape = (rows * slots * tokens, cols * token_size_bytes // 4)
    gen = torch.Generator().manual_seed(0xF2D5)
    # Capped at 2**31 - 1 so the int32 -> uint32 (device) -> int32 (readback) round-trip is value-exact
    # and the host reference can be compared against the readback without a bitcast.
    host_input = torch.randint(1, 2**31 - 1, shape, dtype=torch.int32, generator=gen)

    def _to_device(host):
        return ttnn.from_torch(
            host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (rows, cols), dims=(0, 1)),
        )

    return (
        host_input,
        _to_device(host_input),
        _to_device(torch.zeros(shape, dtype=torch.int32)),
    )


def _cmbf2d_check_accuracy(mesh_device, host_input, dev_output, movements, tokens):
    """True if every movement's destination region on the device equals its source region on the HOST,
    token for token.

    The expected values come from `host_input` — the tensor the input region was uploaded from — and
    never from a readback of the device input. That matters: the op is handed the input tensor and could
    in principle write to it, and a reference the op can influence is not a reference. Comparing device
    input against device output would, for instance, be passed by an op that zeroed the input region and
    moved nothing at all, because the output region starts zeroed; likewise by anything that aliases the
    two regions in its address arithmetic.

    Only the output region is read back, which also halves this check's PCIe traffic.
    """
    rows, cols = tuple(mesh_device.shape)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, (rows, cols), dims=(0, 1))
    host_out = ttnn.to_torch(dev_output, mesh_composer=composer).to(torch.int32)
    # A sharded mesh tensor's shape is its per-device shard, i.e. one device's region; `host_input` and
    # `host_out` are both the reassembled whole, so a device's block starts at coord * shard_extent.
    dev_rows, elems = tuple(dev_output.shape)

    extent = rows
    failures = []
    for m in movements:
        sr, sc = m.src
        dr, dc = m.dst
        src_row0 = sr * dev_rows + m.in_base_token
        dst_row0 = dr * dev_rows + m.out_base_token
        src = host_input[src_row0 : src_row0 + tokens, sc * elems : (sc + 1) * elems]
        got = host_out[dst_row0 : dst_row0 + tokens, dc * elems : (dc + 1) * elems]
        if not torch.equal(src, got):
            bad_rows = (src != got).any(dim=1)
            failures.append(
                {
                    "offset": (dr - sr) % extent,
                    "src": tuple(m.src),
                    "dst": tuple(m.dst),
                    "in_base": m.in_base_token,
                    "bad_tokens": int(bad_rows.sum()),
                    "first_bad": int(bad_rows.nonzero()[0][0]),
                    "all_zero": bool((got[bad_rows] == 0).all()),
                }
            )

    if not failures:
        return True

    # Group by ring offset: the single most diagnostic cut, because it separates "routing is wrong" from
    # "far destinations lose a race" from "one specific hop distance is broken".
    by_offset = {}
    for f in failures:
        e = by_offset.setdefault(f["offset"], {"movements": 0, "bad_tokens": 0, "all_zero": 0})
        e["movements"] += 1
        e["bad_tokens"] += f["bad_tokens"]
        e["all_zero"] += 1 if f["all_zero"] else 0
    logger.error(f"accuracy FAILED: {len(failures)} of {len(movements)} movements wrong")
    logger.error(f"{'ring offset':>12} {'bad movements':>14} {'bad tokens':>11} {'wholly unwritten':>17}")
    for off in sorted(by_offset):
        e = by_offset[off]
        logger.error(f"{off:>12} {e['movements']:>14} {e['bad_tokens']:>11} {e['all_zero']:>17}")
    for f in failures[:5]:
        logger.error(
            f"  e.g. off{f['offset']} src{f['src']} -> dst{f['dst']} in_base {f['in_base']}: "
            f"{f['bad_tokens']}/{tokens} tokens wrong, first at {f['first_bad']}, "
            f"{'all zero (never written)' if f['all_zero'] else 'wrong content'}"
        )
    return False


def _cmbf2d_producer_token_budget(mesh_device, axis, num_links, tokens):
    """`(allowed_per_producer, mesh_total)` PACKET counts, derived from the op's forwarding scheme.

    A producer's packets are its own tokens PLUS everything it forwards for other chips, because the op does
    the forwarding itself. So `tokens_sent` equals the traffic crossing its cable — the number comparable to
    the 25 GB/s line rate — and no derived estimate is needed.

    Since P9.3 the split is balanced: each producer serves H = R/2 destinations (ring distances 1..H-1 in its
    own direction, plus half of the diametrically-opposite chip, which is equally far either way). So every
    producer is identical, and puts on its link
        H*H/2 * tokens   payload tokens   (= H(H-1)/2 full distances + H/2 for the halved one)
      + H(H-1)/2         sentinels, one per forwarding chunk it writes
    """
    extent = tuple(mesh_device.shape)[axis]
    half = extent // 2
    per_producer = half * half // 2 * tokens + half * (half - 1) // 2
    mesh_total = mesh_device.get_num_devices() * num_links * 2 * per_producer
    return {per_producer}, mesh_total


def _dump_combine_fabric2d_bwinfo(
    mesh_device,
    num_links,
    axis,
    expected_workers,
    allowed_tokens,
    mesh_total_tokens,
    path=CMBF2D_BWINFO_PATH,
):
    """Read each producer's L1 telemetry record, write the bandwidth report, then assert it is sound.

    Telemetry is trusted for CYCLE COUNTS ONLY. Everything else a rate depends on — the clock, the
    payload size, how many producers reported at all — is asserted against what the test asked for,
    because a bandwidth figure with an unverified denominator is worse than none: it looks authoritative
    and is silently wrong. Concretely, payload is CMBF2D_TOKENS * CMBF2D_TOKEN_BYTES, never the record's
    own tokens_sent * token_size_bytes (those are cross-checked instead), and a clock more than
    CMBF2D_CLOCK_TOLERANCE off CMBF2D_EXPECTED_CLOCK_MHZ fails the test.

    The report is written BEFORE the assertions, so a failing run still leaves the artifact to look at.

    Four windows land in the file, widest to narrowest:
      * kus   producer kernel entry -> exit, including the fabric connect/teardown. TOTAL kernel time —
              the number total-time optimisation has to reduce.
      * us    reader's first DRAM read -> EDM drain complete. The transfer proper; the drain forces every
              payload credit back from the far chip, so it is an UPPER bound on the transfer and GB/s is
              therefore a LOWER bound on bandwidth.
      * sus   first send -> last send: the send loop alone, giving sGB/s, the fabric push rate. Comparing
              it against `us` prices the DRAM read and the ring handshake.
      * rdy   first DRAM read -> first send, i.e. how long the ring took to prime.

    Nothing here needs to know where the worker cores are — the binding recomputes placement from
    (num_links, axis). This is the ONE place in the test that knows the op runs producer kernels at all.
    """
    # Plain diagnostic function rather than a registered ttnn operation, so it lives on the raw
    # nanobind module instead of under ttnn.experimental.deepseek_prefill.* like the op itself.
    telem = ttnn._ttnn.operations.experimental.combine_fabric2d_read_telemetry(
        mesh_device, num_links=num_links, axis=axis
    )
    clock_mhz = telem["clock_mhz"]
    workers = telem["workers"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Per-producer payload comes from each producer's own token count, but only AFTER that count has been
    # checked against the test's plan (allowed_tokens / mesh_total_tokens) in the assertions below.
    def payload_of(w):
        return w["tokens_sent"] * CMBF2D_TOKEN_BYTES

    # Cycles -> us needs a clock. Guarded only so a zero clock cannot produce inf and crash the report
    # before the assertion below gets to explain what went wrong.
    usable_clock = clock_mhz if clock_mhz > 0 else 1.0

    rows, bad, mismatched = [], [], []
    for w in workers:
        if not w["valid"]:
            bad.append(w)
            continue
        if w["tokens_sent"] not in allowed_tokens or w["token_size_bytes"] != CMBF2D_TOKEN_BYTES:
            mismatched.append(w)
        us = (w["t_drained"] - w["t_start"]) / usable_clock
        send_cycles = w["t_last_send"] - w["t_first_send"]
        send_us = send_cycles / usable_clock
        # Stall shares of the send window: wait_slot = eth side cannot drain us, issue = our own packet
        # issue cost, ring_wait = blocked on the reader.
        denom = send_cycles if send_cycles > 0 else 1
        rows.append(
            {
                "w": w,
                "us": us,
                "gbps": (payload_of(w) / (us * 1e-6)) / 1e9 if us > 0 else 0.0,
                "send_us": send_us,
                "send_gbps": (payload_of(w) / (send_us * 1e-6)) / 1e9 if send_us > 0 else 0.0,
                "kernel_us": (w["t_kernel_end"] - w["t_kernel_start"]) / usable_clock,
                "rdy_us": (w["t_first_send"] - w["t_start"]) / usable_clock,
                "shares": {k: 100.0 * w[f"{k}_cycles"] / denom for k in ("wait_slot", "issue", "ring_wait")},
            }
        )

    def _stats(vals):
        v = sorted(vals)
        return v[0], v[len(v) // 2], v[-1], sum(v) / len(v)

    with open(path, "w") as f:
        first = rows[0]["w"] if rows else {}
        f.write(
            f"# CombineFabric2D bandwidth telemetry\n"
            f"# mesh={tuple(mesh_device.shape)} num_links={num_links} axis={axis} "
            f"tokens={CMBF2D_TOKENS} per movement token={CMBF2D_TOKEN_BYTES}B clock={clock_mhz}MHz "
            f"edm_sender_slots={first.get('edm_slots', 0)} l1_slots={first.get('num_l1_slots', 0)} "
            f"batch={first.get('batch', 0)} fwd_bump_every={CMBF2D_FWD_BUMP} order={CMBF2D_ORDER} stall_telemetry={CMBF2D_STALL}\n"
            f"# payload per producer = tokens_sent x {CMBF2D_TOKEN_BYTES}B, where tokens_sent is asserted to be\n"
            f"#   one of {sorted(allowed_tokens)} and to sum to {mesh_total_tokens} across the mesh. Producers are\n"
            f"#   deliberately unequal: the fabric picks one direction for the opposite chip. Not taken\n"
            f"#   from telemetry. Telemetry supplies cycle counts only; its tokens_sent / token_size_bytes\n"
            f"#   are asserted to match, and the clock is asserted within "
            f"{CMBF2D_CLOCK_TOLERANCE:.0%} of {CMBF2D_EXPECTED_CLOCK_MHZ} MHz.\n"
            f"# kus    = producer kernel entry -> exit, incl. fabric connect/teardown. TOTAL kernel time.\n"
            f"# us     = first DRAM read -> transfer provably complete  [EFFECTIVE end-to-end, includes the\n"
            f"#          DRAM read; LOWER bound on bandwidth. The completion proof is the EDM sender-channel\n"
            f"#          drain: dpk header-only fillers plus one more free slot force every payload credit\n"
            f"#          back from the far chip.\n"
            f"# sus    = first send -> last send, the SEND LOOP alone.\n"
            f"# GB/s   = own payload / us   |   sGB/s = own payload / sus  [push rate of OWN tokens]\n"
            f"# Own payload is NOT link utilisation once destinations pass the immediate neighbour: each cable\n"
            f"#   also carries traffic forwarded on other chips' behalf, which is real eth traffic but appears in\n"
            f"#   no producer's tokens_sent. The derived link figure is in the summary at the bottom.\n"
            f"# rdy    = first DRAM read -> first send, i.e. how long the ring took to prime.\n"
            f"# wait%/iss%/ring% = share of the send window spent waiting for an EDM slot (eth side is the\n"
            f"#   limiter) / issuing payload packets / waiting on the reader (DRAM side is the limiter).\n"
            f"# workers: {len(rows)} valid, {len(bad)} missing/invalid records\n"
            f"#\n"
            f"{'dev':>4} {'coord':>8} {'worker_l':>9} {'worker_p':>9} {'eth':>7} {'ex':>3} {'lnk':>4}"
            f" {'reloc':>6} {'peer':>5} {'tok':>6} {'obase':>6} {'dpk':>4}"
            f" {'kus':>9} {'us':>9} {'GB/s':>7} {'sus':>9} {'sGB/s':>7} {'rdy':>7}"
            f" {'wait%':>6} {'iss%':>6} {'ring%':>6}\n"
        )
        for r in sorted(rows, key=lambda r: (r["w"]["device_id"], r["w"]["eth_phys_x"])):
            w, sh = r["w"], r["shares"]
            f.write(
                f"{w['device_id']:>4} {str(tuple(w['mesh_coord'])):>8}"
                f" {str(tuple(w['worker_logical'])):>9} {str(tuple(w['worker_physical'])):>9}"
                f" {str(tuple(w['eth_logical'])):>7} {w['eth_phys_x']:>3} {w['link_idx']:>4}"
                f" {'Y' if w['relocated'] else '':>6} {w['peer_chip_id']:>5}"
                f" {w['tokens_sent']:>6} {w['out_base_page']:>6} {w['drain_packets']:>4}"
                f" {r['kernel_us']:>9.2f} {r['us']:>9.2f} {r['gbps']:>7.2f}"
                f" {r['send_us']:>9.2f} {r['send_gbps']:>7.2f} {r['rdy_us']:>7.2f}"
                f" {sh['wait_slot']:>6.1f} {sh['issue']:>6.1f} {sh['ring_wait']:>6.1f}\n"
            )
        for w in bad:
            f.write(
                f"# NO RECORD: dev {w['device_id']} coord {tuple(w['mesh_coord'])} "
                f"worker_logical {tuple(w['worker_logical'])} eth {tuple(w['eth_logical'])}\n"
            )
        for w in mismatched:
            f.write(
                f"# PAYLOAD MISMATCH: dev {w['device_id']} coord {tuple(w['mesh_coord'])} reported "
                f"{w['tokens_sent']} tokens x {w['token_size_bytes']}B, test allows "
                f"{sorted(allowed_tokens)} x {CMBF2D_TOKEN_BYTES}B\n"
            )
        if rows:
            g_min, g_p50, g_max, g_mean = _stats([r["gbps"] for r in rows])
            s_min, s_p50, s_max, s_mean = _stats([r["send_gbps"] for r in rows])
            l_min, l_p50, l_max, l_mean = _stats([r["send_us"] for r in rows])
            k_min, k_p50, k_max, k_mean = _stats([r["kernel_us"] for r in rows])
            xfer_mean = sum(r["us"] for r in rows) / len(rows)
            mean_sh = {k: sum(r["shares"][k] for r in rows) / len(rows) for k in ("wait_slot", "issue", "ring_wait")}
            # LINK utilisation, now DIRECTLY MEASURED: a producer's tokens_sent counts its own tokens plus
            # everything it forwards for other chips, so it IS the traffic crossing its cable. The derived
            # estimate P8.3 needed is gone, and so is the non-uniformity caveat — the P9.3 split is balanced,
            # every producer carrying H*H/2 tokens.
            # The number that ends the transfer is the SLOWEST producer, so that is quoted first: a mean over
            # per-producer rates flatters the result (it is a mean of reciprocals).
            slowest_gbps = min(r["send_gbps"] for r in rows)
            f.write(
                f"#\n# LINK UTILISATION (measured, headline)\n"
                f"#   SLOWEST producer: {slowest_gbps:.2f} GB/s of 25 over {l_max:.2f} us"
                f"   <-- the transfer is not done until this one is\n"
                f"#   p50 {s_p50:.2f}   mean {s_mean:.2f} GB/s   (own + forwarded packets, "
                f"{sorted(allowed_tokens)} per producer, {mesh_total_tokens} mesh-wide)\n"
            )
            f.write(
                f"# per-producer GB/s (incl. drain tail): min {g_min:.2f} p50 {g_p50:.2f} max {g_max:.2f} "
                f"mean {g_mean:.2f}\n"
                f"# per-producer sGB/s: min {s_min:.2f} p50 {s_p50:.2f} max {s_max:.2f} mean {s_mean:.2f}\n"
                f"# mean send-window shares: wait_slot {mean_sh['wait_slot']:.1f}% issue {mean_sh['issue']:.1f}% "
                f"ring_wait {mean_sh['ring_wait']:.1f}%\n"
                f"# send-loop us: min {l_min:.2f} p50 {l_p50:.2f} max {l_max:.2f} mean {l_mean:.2f}\n"
                f"# producer-kernel us: min {k_min:.2f} p50 {k_p50:.2f} max {k_max:.2f} mean {k_mean:.2f}\n"
                f"# kernel time outside the transfer window: mean {k_mean - xfer_mean:.2f} us "
                f"({100.0 * (k_mean - xfer_mean) / k_mean if k_mean > 0 else 0.0:.1f}% of kernel)\n"
                f"# aggregate payload across the mesh: {sum(payload_of(r['w']) for r in rows) / 1e6:.1f} MB over "
                f"{len(rows)} producers\n"
            )
            # The two windows bracket the transfer, so their ratio says how much of the reported time is
            # drain tail rather than payload push.
            f.write(f"# end-to-end vs push-rate spread: mean {g_mean:.2f} vs {s_mean:.2f} GB/s\n")

    # ---- The artifact exists; now refuse to report numbers we cannot stand behind.
    assert (
        not bad
    ), f"{len(bad)} of {expected_workers} producer(s) produced no telemetry record; see 'NO RECORD' in {path}"
    assert (
        len(rows) == expected_workers
    ), f"expected {expected_workers} producer records (one per movement), got {len(rows)}; see {path}"
    total_sent = sum(r["w"]["tokens_sent"] for r in rows)
    assert total_sent == mesh_total_tokens, (
        f"producers sent {total_sent} tokens in total but the test's plan moves {mesh_total_tokens}; "
        f"the op is not covering the movement list. See {path}"
    )
    assert not mismatched, (
        f"{len(mismatched)} producer(s) reported a payload outside the allowed {sorted(allowed_tokens)} x "
        f"{CMBF2D_TOKEN_BYTES}B; see 'PAYLOAD MISMATCH' in {path}"
    )
    assert abs(clock_mhz - CMBF2D_EXPECTED_CLOCK_MHZ) <= CMBF2D_CLOCK_TOLERANCE * CMBF2D_EXPECTED_CLOCK_MHZ, (
        f"device clock {clock_mhz} MHz is more than {CMBF2D_CLOCK_TOLERANCE:.0%} off the assumed "
        f"{CMBF2D_EXPECTED_CLOCK_MHZ} MHz, so every rate in {path} is scaled by "
        f"{clock_mhz / CMBF2D_EXPECTED_CLOCK_MHZ if CMBF2D_EXPECTED_CLOCK_MHZ else 0.0:.3f}"
    )
    return rows, clock_mhz


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    CMBF2D_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_combine_fabric2d(mesh_device, device_params, num_links, topology):
    num_devices = mesh_device.get_num_devices()
    logger.debug(
        f"combine_fabric2d: shape={tuple(mesh_device.shape)} num_devices={num_devices} "
        f"num_links={num_links} topology={topology}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    # The op relies on the fabric carrying token + 64B routing tail in ONE packet. Assert the device came
    # up that way rather than trusting the config went through.
    fabric_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    assert fabric_payload >= CMBF2D_PACKET_BYTES, (
        f"fabric max payload is {fabric_payload} B but the op needs {CMBF2D_PACKET_BYTES} "
        f"({CMBF2D_TOKEN_BYTES} token + 64 routing tail)"
    )
    logger.info(f"combine_fabric2d fabric max payload {fabric_payload} B (needs {CMBF2D_PACKET_BYTES})")

    tokens = CMBF2D_TOKENS
    token_size_bytes = CMBF2D_TOKEN_BYTES
    # The op halves the movement to the diametrically-opposite chip between its two producers (that chip
    # is reachable equally far in either direction), so an odd count would not split evenly.
    assert tokens % 2 == 0, f"CMBF2D_TOKENS must be even (the +N/2 movement is split in half), got {tokens}"

    # What we want moved, stated up front and independently of how the op will do it.
    movements = _cmbf2d_plan_movements(mesh_device, CMBF2D_AXIS, num_links, tokens)
    host_input, dev_input, dev_output = _cmbf2d_make_regions(
        mesh_device, CMBF2D_AXIS, tokens, token_size_bytes, num_links
    )
    slots = _cmbf2d_slots_per_device(mesh_device, CMBF2D_AXIS, num_links)
    logger.info(
        f"combine_fabric2d plan: {len(movements)} movements, {slots} slots/device of {tokens} tokens "
        f"({slots * tokens} tokens per region per device)"
    )

    signpost(f"combine_fabric2d start num_links={num_links} tokens={tokens}")
    output = ttnn.experimental.deepseek_prefill.combine_fabric2d(
        mesh_device,
        dev_input,
        dev_output,
        movements,
        num_links=num_links,
        tokens_per_movement=tokens,
        token_size_bytes=token_size_bytes,
        axis=CMBF2D_AXIS,
        fwd_bump_every=CMBF2D_FWD_BUMP,
        assignment_order=CMBF2D_ORDER,
        stall_telemetry=CMBF2D_STALL,
    )
    ttnn.synchronize_device(mesh_device)
    signpost("combine_fabric2d end")

    # The op has to return a tensor — ttnn's device-op framework requires a tensor_return_value_t, there
    # is no void device op — and this one opts into caller-owned output by returning the very tensor it
    # was handed (create_output_tensors -> tensor_args.output). Assert that contract rather than assume
    # it: buffer_address() is the MeshBuffer address, so this is exact and costs no readback. If someone
    # later switches the op to the usual "allocate a fresh output" pattern, this fires instead of the
    # test silently validating a tensor we never asked to be written.
    assert (
        output.buffer_address() == dev_output.buffer_address()
    ), "op did not write into the caller-owned output region"

    # Read back dev_output — the region WE allocated, zeroed and named — not the handle the op returned.
    # Same reasoning as using host_input for the source side: neither side of the comparison should be a
    # thing the op chose for us.
    assert _cmbf2d_check_accuracy(mesh_device, host_input, dev_output, movements, tokens)

    # ---- Bandwidth telemetry. Writes the report and asserts it is internally sound (clock, payload,
    # every producer accounted for) — see _dump_combine_fabric2d_bwinfo. The numbers themselves live in
    # the file, not in the console: there are 128 producer rows here, and a console summary is only ever
    # a lossy copy of what the file already says.
    #
    # One telemetry record per PRODUCER CORE, not per movement: a device has 2 * num_links producers (one
    # per link per direction) and each now serves several movements, so these two counts diverged the
    # moment destinations went beyond the immediate neighbours.
    _cmbf2d_allowed, _cmbf2d_total = _cmbf2d_producer_token_budget(mesh_device, CMBF2D_AXIS, num_links, tokens)
    bwinfo_path = _cmbf2d_bwinfo_path()
    _dump_combine_fabric2d_bwinfo(
        mesh_device,
        num_links,
        axis=CMBF2D_AXIS,
        expected_workers=num_devices * 2 * num_links,
        allowed_tokens=_cmbf2d_allowed,
        mesh_total_tokens=_cmbf2d_total,
        path=bwinfo_path,
    )
    logger.info(f"combine_fabric2d telemetry -> {bwinfo_path}")
