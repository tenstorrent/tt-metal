# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill combine operation in isolation.

This test verifies that the TTNN combine operation produces the same output as the
PyTorch reference implementation when combining expert outputs back to token positions.
Uses torch-generated dispatch inputs to isolate the combine operation.
"""

import os
import sys
from dataclasses import dataclass

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
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_dram_alignment,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import (
    COMBINE_SF_TAIL_BYTES,
    TtCombineModule,
    combine_sf_levels,
    make_combine_staging_buffer,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results

# Must match sf::MAGIC and the tail offsets in combine/device/combine_sf.hpp.
COMBINE_SF_MAGIC = 0x5AF2C0DE
COMBINE_SF_TAIL_MAGIC_WORD = 2


def count_staged_slots(staging_buffer, emb_dim, use_fp8_output):
    """Count staging slots whose tail carries the routing magic.

    This is a witness that the relay path ran, not a token count: a slot keeps its magic once
    written, and slots are reused round-robin, so the number tracks distinct slots touched. It
    therefore scales with the stream count (two sender cores means twice as many streams) and only
    loosely with traffic. Treat it as a lower bound on relayed tokens and nothing more.
    """
    element_bytes = 1 if use_fp8_output else 2
    alignment = get_dram_alignment()
    token_bytes = ((emb_dim * element_bytes + alignment - 1) // alignment) * alignment
    magic_word_index = token_bytes // 4 + COMBINE_SF_TAIL_MAGIC_WORD

    staged = 0
    for shard in ttnn.get_device_tensors(staging_buffer):
        pages = ttnn.to_torch(shard).reshape(-1, staging_buffer.shape[-1])
        staged += int((pages[:, magic_word_index] == COMBINE_SF_MAGIC).sum())
    return staged


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
    use_store_and_forward=False,
    invocations=1,
    staging_slots_per_stream=16,
    init_zeros=False,
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

    # EXPERIMENT, not for commit: lets the Phase 4 SF_SLOTS sweep vary ring depth without a
    # parametrize axis, so every point keeps the same test id and the device-time samples stay
    # comparable. Unset reproduces the caller's value exactly.
    _slots_override = os.getenv("TT_DS_SF_SLOTS")
    if _slots_override:
        staging_slots_per_stream = int(_slots_override)

    # Run ttnn combine
    staging_buffer = (
        make_combine_staging_buffer(
            mesh_device,
            emb_dim,
            output_dtype=ttnn.fp8_e4m3 if use_fp8_output else ttnn.bfloat16,
            num_links=num_links,
            topology=topology,
            cluster_axis=sp_axis,
            slots_per_stream=staging_slots_per_stream,
        )
        if use_store_and_forward
        else None
    )

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
        init_zeros=init_zeros,
        fp8_output=use_fp8_output,
        use_store_and_forward=use_store_and_forward,
        staging_buffer=staging_buffer,
    )

    # Repeated invocations share one module and one staging buffer, which is how the model uses it.
    # The cross-chip counters live on GlobalSemaphores that are zeroed only when they are created, so
    # a second call is what proves they get cleared; skipping it would leave that reset unexercised.
    for _ in range(invocations):
        tt_output = tt_combine(
            tt_dispatched_buffer,
            tt_dispatched_metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
        )

    # Correct output alone does not prove a token was ever relayed -- if every token happened to be
    # one hop from home, or if the level arithmetic collapsed everything into a direct write, the
    # result would look identical. Counting the routing tails left in the staging buffer is direct
    # evidence that the relay path carried traffic.
    if staging_buffer is not None:
        ttnn.synchronize_device(mesh_device)
        staged_slots = count_staged_slots(staging_buffer, emb_dim, use_fp8_output)
        logger.debug(f"store-and-forward touched {staged_slots} staging slot(s) across the mesh")
        assert staged_slots > 0, (
            "use_store_and_forward was set on a mesh with relay levels, but no staging page carries a "
            "routing tail -- the relay path did not run, so a passing output proves nothing about it"
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
        dict[tuple[int, int], ttnn.FabricConfig]
    ]  # Intended for [0..N] proxy tests, typically on smaller HW


SINGLE_GLX_AND_PROXY_MESHES = _Test_Mesh(
    (8, 4),
    {
        # Ideally all would run torus XY, but some HW configurations like LB/QB cannot support
        # rings in all configurations. Pick fabric option as representative as possible.
        # A quad-member galaxy is one such configuration: its 8-axis wrap leaves the host for the
        # neighbouring galaxy, so the axis is a line locally and a torus MGD fails to map onto it.
        # Linear at extent 8 is also the deeper relay test -- 6 levels against the torus's 3.
        (8, 4): ttnn.FabricConfig.FABRIC_2D,
        (8, 1): ttnn.FabricConfig.FABRIC_1D_RING,  # unexpectedly FABRIC_2D variants hang
        (4, 2): ttnn.FabricConfig.FABRIC_2D,
        (4, 1): ttnn.FabricConfig.FABRIC_1D_RING,  # unexpectedly FABRIC_2D variants hang
        (2, 2): ttnn.FabricConfig.FABRIC_2D,
    },
)


# Per-model combine shapes as (id_prefix, config, extended_model). Each model contributes a pcc
# param (seq 128, // 16 experts, top-4) and a perf param (seq 640, // 4 experts, top-2). DeepSeek
# V3 is the baseline and runs by default; every other model is gated behind
# @pytest.mark.extended_model. dispatch_buffer_capacity_factor is ceil(N/2) of the most
# conservative integer N such that dgs*seq*N >= worst-case dispatch buffer.
COMBINE_MODELS = [
    ("dsv3", DeepSeekV3Config, False, SINGLE_GLX_AND_PROXY_MESHES),
    ("glm_51", GLM51Config, True, SINGLE_GLX_AND_PROXY_MESHES),
    ("kimi_k26", KimiK26Config, True, SINGLE_GLX_AND_PROXY_MESHES),
    ("minimax_m27", MiniMaxM27Config, True, SINGLE_GLX_AND_PROXY_MESHES),
    ("dsv4_pro", DeepSeekV4ProConfig, True, SINGLE_GLX_AND_PROXY_MESHES),
    ("dsv4_flash", DeepSeekV4FlashConfig, True, SINGLE_GLX_AND_PROXY_MESHES),
    ("gptoss_120b", GptOss120BConfig, True, SINGLE_GLX_AND_PROXY_MESHES),
]


# Scales down model hyper-params for a given hardware to obtain good/meaningful proxy test
# How exactly to scale it down is op-specific (more precisely - even op-implementation specific)
# Thus it makes sense for this to be combine-specific function
def _model_scaledown_for_combine(model, ref_mesh, target_mesh, pcc_only):
    # number of experts has to be reduced to preserve the experts per chip
    ref_num_chips = ref_mesh[0] * ref_mesh[1]
    target_num_chips = target_mesh[0] * target_mesh[1]
    if ref_num_chips != target_num_chips:
        # oreder of the operation keeps the number of routed experts divisible by the number of chips
        model.NUM_ROUTED_EXPERTS = (model.NUM_ROUTED_EXPERTS // ref_num_chips) * target_num_chips

    # number of experts selected to proces every token (top-K) has to be scaled to preserve average expert activation per dispatch group
    if ref_mesh[1] != target_mesh[1]:
        model.NUM_EXPERTS_PER_TOKEN = (model.NUM_EXPERTS_PER_TOKEN // ref_mesh[1]) * target_mesh[1]

    # further reduce these two hyperparams in case of pcc check test to get faster, although not perf-representative test
    if pcc_only:
        model.NUM_ROUTED_EXPERTS = max(target_num_chips, model.NUM_ROUTED_EXPERTS // 16)
        model.NUM_EXPERTS_PER_TOKEN = max(2, model.NUM_EXPERTS_PER_TOKEN // 4)

    return model


def _topo_marker(mesh, fabric_cfg):
    if mesh[1] == 1 and fabric_cfg == ttnn.FabricConfig.FABRIC_1D:
        return "linear"
    if mesh[1] == 1 and fabric_cfg == ttnn.FabricConfig.FABRIC_1D_RING:
        return "ring"
    return f"mesh-{mesh[0]}x{mesh[1]}"


def _mesh_id(mesh, fabric_cfg):
    if mesh[1] == 1 and fabric_cfg == ttnn.FabricConfig.FABRIC_1D:
        return f"linear-{mesh[0]}"
    if mesh[1] == 1 and fabric_cfg == ttnn.FabricConfig.FABRIC_1D_RING:
        return f"ring-{mesh[0]}"
    return f"mesh-{mesh[0]}x{mesh[1]}"


def _fabric_cfg_to_fabric_topo_for_cmb_op(fabric_cfg):
    # This mapping is specific to an op because fabric topology is the CCL algorithm's data-flow shape (Linear / Ring / Mesh / Torus)
    # ttnn.FabricConfig is the device-level fabric wiring (FABRIC_1D / FABRIC_1D_RING / FABRIC_2D / FABRIC_2D_TORUS_Y) —
    # set via the `mesh_device` fixture's device_params.
    # ttnn.Topology on the other hand is what data-movement pattern the algorithm will use in op kernel runtime.
    # It depends on the fabric config because config might prevent some topology (e.g. FABRIC_1D does not support Ring topology), but topology
    # doesn't have to use all of the supported movements. For example it is perfectly valid to ask for Topology.Linear (and not Topology.Mesh)
    # on FABRIC_2D. Thus it is natural for every CCL op to derive which topology to use from the given fabric config.
    if fabric_cfg in (ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_2D, ttnn.FabricConfig.FABRIC_2D_TORUS_X):
        return ttnn.Topology.Linear
    else:
        return ttnn.Topology.Ring


def _cross_product_conflated_cmb_test_dimensions():
    params = []
    for model_name, model_config_class, is_extended_model, test_meshes in COMBINE_MODELS:
        for target_mesh, fabric_cfg in test_meshes.target_meshes.items():
            device_params = fabric_to_device_params(fabric_cfg, payload_tail_bytes=COMBINE_SF_TAIL_BYTES)
            fabric_topo = _fabric_cfg_to_fabric_topo_for_cmb_op(fabric_cfg)
            topo_marker = _topo_marker(target_mesh, fabric_cfg)
            mesh_requirements_marker = pytest.mark.requires_mesh_topology(mesh_shape=target_mesh, topology=topo_marker)
            marks = (
                (pytest.mark.extended_model, mesh_requirements_marker)
                if is_extended_model
                else (mesh_requirements_marker)
            )
            test_scenarios = [
                ("pcc", 128, 4, True),
                ("perf_no_pcc", 640, 8, False),
            ]
            for test_scenario_id, seq_len_per_chip, dispatch_buffer_capacity_factor, run_pcc in test_scenarios:
                model_config = _model_scaledown_for_combine(
                    model_config_class(), test_meshes.full_model_mesh, target_mesh, run_pcc
                )

                num_experts = model_config.NUM_ROUTED_EXPERTS
                topk = model_config.NUM_EXPERTS_PER_TOKEN
                shape = target_mesh

                params.append(
                    pytest.param(
                        shape,
                        device_params,
                        fabric_topo,
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


# Test parametrization axes are semantically
# 1. Chip count and layout
#   1.1. mesh column size, which is consequently the size of a dispatch group
#   1.2. mesh row size, which is consequently the number of dispatch groups
# 2. Number of fabric links in a single chip-to-chip connection. (Other fabric props are either cherry-picked
#    for the test case, such as fabric config or packet size, or are derivable from them, like fabric topology)
# 3. Model-related (embed-dim, topK, num-experts, ...)
# 4. Input related (ISL, tile/RM, datum format, ...)
# 5. Scenario/type of test
#   5.1. accuracy or perf test
#   5.2. production test / proxy test on smaller hardware (like single DG simulation) / op generality test
#   5.3. random or predictable (fixed) data
#
# Each level-2 item in this list is a valid parametrization axis (semantically).
# However there are two reasons some of them are conflated into single parametrization axis.
# 1. pytest requires some marks to be populated. And it allows marks to be calculated only based on a single axis values.
#    Some of our marks are besed on multiple semantic axes (e.g. topology marker is calculated based on 1.1, 1.2, 2.1, 2.2 and 2.3)
#    Industry standard work-around is conflating these axes into a single @pytest.mark.parametrize axis + a function which generates
#    its values as a cross product of the semantical axis which are conflated. Then calculate marks based on the value of the resulting
#    conflated axis, which from the perspective of the pytest is a single parametrization axis.
# 2. Even on a semantical level, we don't want full cross product of some axes. E.g. production test makes sense only on 8x4 mesh.
#    Or fp8 test doesn't run PCC. Or fabric 1d doesn't support both x and y rings. Such combination are either prevented during necesarry
#    test-code cross product calculation, or are skipped in the body of the test, depending on where it was less cumbersome to implement it.
#
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor, run_pcc_check",
    _cross_product_conflated_cmb_test_dimensions(),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
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
        # EXPERIMENT, not for commit: matches the store-and-forward entrypoint's invocation count so
        # the two can be compared warm-to-warm. With the default of 1 this test yields only a cold
        # sample, and the relay path's first invocation is far slower than its steady state, so a
        # one-versus-three comparison reads as a regression that is really a cold/warm mismatch.
        invocations=3,
    )


# Store-and-forward is a separate entrypoint rather than another axis on test_ttnn_combine so that
# every existing test ID stays byte-identical -- an extra parametrize axis appends its id to all of
# them and silently breaks the CI -k filters that select on those names. It also keeps the matrix
# down to the axes that actually interact with the relay path: the two dispatched-buffer layouts
# (the path is meant to be layout-agnostic) and link count (which sets the stream count).
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor, run_pcc_check",
    _cross_product_conflated_cmb_test_dimensions(),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.parametrize(
    "dispatched_buffer_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "row_major"],
)
def test_ttnn_combine_store_and_forward(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    run_pcc_check,
    dispatched_buffer_layout,
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
        # Predictable inputs make a routing bug read as a specific wrong token rather than noise, which
        # is worth having wherever the output is actually checked.  The perf scenarios do not check it,
        # and run_combine skips predictable data on 8 or more chips, so tying this to run_pcc_check is
        # what lets the perf geometry -- production top-k, full expert count -- run at all.
        run_pcc_check,
        run_pcc_check,
        dispatched_buffer_layout,
        False,  # bf16 output
        is_ci_env,
        is_ci_v2_env,
        use_store_and_forward=True,
        # Several calls through one module and one staging buffer, as the model does. A single call
        # would leave the cross-invocation counter reset untested, and that failure mode is a
        # timing-dependent hang rather than a wrong answer.
        invocations=3,
    )


# A validator that never fires is worthless, so each store-and-forward rejection gets exercised.
# One cheap mesh is enough: these are host-side shape and consistency checks, not data paths.
@pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 2), fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D))],
    indirect=True,
)
def test_ttnn_combine_store_and_forward_rejects(
    mesh_device, device_params, monkeypatch, expect_error, is_ci_env, is_ci_v2_env
):
    assert combine_sf_levels(mesh_device, ttnn.Topology.Linear, 0) == 2, "4-device linear axis needs 2 relay levels"

    # Captured before any patching: the module global is rebound below, so the bare name would
    # otherwise resolve to whichever stub was installed last.
    real_make_staging_buffer = make_combine_staging_buffer

    def run():
        run_combine(
            mesh_device,
            128,
            7168,
            8,
            4,
            4,
            1,
            ttnn.Topology.Linear,
            True,
            True,
            ttnn.TILE_LAYOUT,
            False,
            is_ci_env,
            is_ci_v2_env,
            use_store_and_forward=True,
        )

    monkeypatch.setattr(sys.modules[__name__], "make_combine_staging_buffer", lambda *a, **k: None)
    with expect_error(ValueError, "needs a staging_buffer"):
        run()

    def wrong_page_size(mesh, emb_dim, **kwargs):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 64, 1024]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    monkeypatch.setattr(sys.modules[__name__], "make_combine_staging_buffer", wrong_page_size)
    with expect_error(RuntimeError, "aligned page size"):
        run()

    def indivisible_page_count(mesh, emb_dim, **kwargs):
        reference = real_make_staging_buffer(mesh, emb_dim, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=0)
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 65, reference.shape[-1]]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    monkeypatch.setattr(sys.modules[__name__], "make_combine_staging_buffer", indivisible_page_count)
    with expect_error(RuntimeError, "does not divide evenly"):
        run()

    # A buffer handed over with the flag clear would let the program-cache key and the workload's
    # buffer list disagree about whether the path is live.
    monkeypatch.setattr(sys.modules[__name__], "make_combine_staging_buffer", real_make_staging_buffer)
    buffered_module = TtCombineModule

    class FlagOffButBuffered(buffered_module):
        def __init__(self, *args, **kwargs):
            kwargs["use_store_and_forward"] = False
            super().__init__(*args, **kwargs)
            self.use_store_and_forward = False

    monkeypatch.setattr(sys.modules[__name__], "TtCombineModule", FlagOffButBuffered)
    with expect_error(RuntimeError, "must not be supplied"):
        run()


# The credit protocol is only exercised when a ring actually fills. At the default depth that may
# never happen on a small test, so this pins the depth to the minimum the op accepts: every stream
# is then permanently full and every token pays the credit round trip. It is also the shape a
# deadlock would take, which is why it runs with several invocations.
#
# Both meshes matter and neither substitutes for the other, because credit pressure is per level
# and the two meshes have very different level counts: extent 4 has 2 relay levels, extent 8 has 6.
# Pinning every ring to its minimum on the deeper mesh is what puts a credit cycle across levels
# within reach. Neither is a ring, so both still owe the cyclic case to a 32x4 quad run — on a line
# the staging FIFOs are acyclic by chip position, which is exactly what the level index has to
# supply on a ring.
# The expert counts differ because experts_per_chip is num_routed_experts // num_devices and must
# stay >= 1: 8 experts over 32 chips is zero, so the 8x4 case carries the same numbers the 8x4 pcc
# scenario scales down to (32 experts, top-2). Note Blackhole skips any mesh that is not the whole
# machine, so exactly one of these two runs on a given host.
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_routed_experts, num_experts_per_tok",
    [
        pytest.param(
            (4, 2),
            fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D),
            ttnn.Topology.Linear,
            8,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (8, 4),
            fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D),
            ttnn.Topology.Linear,
            32,
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
def test_ttnn_combine_store_and_forward_tight_rings(
    mesh_device,
    device_params,
    topology,
    num_routed_experts,
    num_experts_per_tok,
    num_links,
    is_ci_env,
    is_ci_v2_env,
):
    run_combine(
        mesh_device,
        128,
        7168,
        num_routed_experts,
        num_experts_per_tok,
        4,
        num_links,
        topology,
        True,
        True,
        ttnn.TILE_LAYOUT,
        False,
        is_ci_env,
        is_ci_v2_env,
        use_store_and_forward=True,
        invocations=10,
        staging_slots_per_stream=2,
    )


# Two things change the kernels' shape rather than just their inputs, and both interact with the
# relay path in ways the main matrix does not reach:
#   init_zeros adds runtime args ahead of the store-and-forward ones, so a mis-sized parse shows up
#     here and nowhere else;
#   fp8 halves the output page, which moves the routing tail's offset and the staging page stride.
# fp8 needs random inputs because predictable ones overflow e4m3's range, and TILE layout because the
# cast happens in the packer during untilize.
@pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 2), fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D))],
    indirect=True,
)
@pytest.mark.parametrize("init_zeros", [False, True], ids=["no_init", "init_zeros"])
@pytest.mark.parametrize("use_fp8_output", [False, True], ids=["bf16_out", "fp8_out"])
def test_ttnn_combine_store_and_forward_variants(
    mesh_device, device_params, init_zeros, use_fp8_output, is_ci_env, is_ci_v2_env
):
    run_combine(
        mesh_device,
        128,
        7168,
        8,
        4,
        4,
        1,
        ttnn.Topology.Linear,
        not use_fp8_output,
        True,
        ttnn.TILE_LAYOUT,
        use_fp8_output,
        is_ci_env,
        is_ci_v2_env,
        use_store_and_forward=True,
        invocations=2,
        init_zeros=init_zeros,
    )


# An axis only two devices deep puts every token one hop from home, so no relay level exists and the
# flag is inert. Worth covering on real hardware because it has its own host-side branch -- the op
# must accept the flag with no staging buffer rather than demanding one. (2,4) rather than (2,2)
# because Blackhole only admits mesh configs that use every device.
@pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 4), fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D))],
    indirect=True,
)
def test_ttnn_combine_store_and_forward_no_relay_levels(mesh_device, device_params, is_ci_env, is_ci_v2_env):
    assert combine_sf_levels(mesh_device, ttnn.Topology.Linear, 0) == 0, "a 2-device axis has no relay level"
    run_combine(
        mesh_device,
        128,
        7168,
        8,
        4,
        4,
        1,
        ttnn.Topology.Linear,
        True,
        True,
        ttnn.TILE_LAYOUT,
        False,
        is_ci_env,
        is_ci_v2_env,
        use_store_and_forward=True,
        invocations=2,
    )


# A staged token is payload plus a routing tail, which overruns the default fabric payload and so
# chunks into a second packet -- an extra EDM slot and fabric header per relayed token. Widening the
# device payload by the tail collapses it back to one. This is the A/B counterpart of the default
# store-and-forward tests: identical work, one packet per staged token instead of two. It only pays
# for itself once there are per-link numbers, so it is kept as a separate config rather than made
# the default, and it is device-wide, so nothing else inherits the wider payload.
@pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 2), fabric_to_device_params(ttnn.FabricConfig.FABRIC_2D, payload_tail_bytes=COMBINE_SF_TAIL_BYTES))],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
def test_ttnn_combine_store_and_forward_single_packet_tail(
    mesh_device, device_params, num_links, is_ci_env, is_ci_v2_env
):
    emb_dim = 7168
    token_bytes = emb_dim * 2
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    # Without this the test would silently exercise the two-packet path and prove nothing new.
    assert payload >= token_bytes + COMBINE_SF_TAIL_BYTES, (
        f"fabric payload {payload} cannot hold a {token_bytes}-byte token plus a "
        f"{COMBINE_SF_TAIL_BYTES}-byte tail; the single-packet path is not being tested"
    )
    run_combine(
        mesh_device,
        128,
        emb_dim,
        8,
        4,
        4,
        num_links,
        ttnn.Topology.Linear,
        True,
        True,
        ttnn.TILE_LAYOUT,
        False,
        is_ci_env,
        is_ci_v2_env,
        use_store_and_forward=True,
        invocations=2,
    )
