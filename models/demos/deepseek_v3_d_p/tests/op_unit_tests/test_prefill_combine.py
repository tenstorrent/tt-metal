# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill combine operation in isolation.

This test verifies that the TTNN combine operation produces the same output as the
PyTorch reference implementation when combining expert outputs back to token positions.
Uses torch-generated dispatch inputs to isolate the combine operation.
"""

from dataclasses import dataclass

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
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    dispatch_buffer_used_tokens,
    extract_mesh_config,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule, TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

_PERF_ITERATIONS = 10

# The operating point the per-model cases run: the production sequence length, timed over
# _PERF_ITERATIONS trace replays with the last replay's output checked. The capacity factor is
# ceil(N/2) of the most conservative integer N such that dgs*seq*N >= worst-case dispatch buffer.
_SEQ_LEN_PER_CHIP = 640
_DISPATCH_BUFFER_CAPACITY_FACTOR = 8


def _upload_dispatch_buffer(
    host_buffer,
    capacity_tokens,
    used_tokens,
    pad_value,
    *,
    mesh_mapper,
    layout,
    dtype,
    mesh_device,
):
    """Put a full-capacity dispatch buffer on device out of the `used_tokens` a router fills.

    The flat dispatch buffer is sized for the worst-case router -- one expert receiving the whole
    dispatch group -- and a real router fills a few percent of it: under 4% for DeepSeek V3 at seq
    640 with capacity factor 8, where the worst case is 35 GiB. Everything past `used_tokens` is the
    constant dispatch initialised the buffer to, so `ttnn.pad` writes it back on device in
    milliseconds rather than the host converting and shipping it (29.5s measured, per upload).

    The op gets a bit-identical input either way -- the tail dispatch leaves is exactly `pad_value`
    -- and it still sees a full-capacity buffer, so nothing about what is measured changes.

    `used_tokens` is one number for the whole mesh, not one per chip, which keeps this a single
    mesh-wide upload and a single mesh-wide op: a prefix that covers the busiest chip covers them all.
    A `host_buffer` already cut to that prefix is uploaded as it is; one at full capacity is sliced.
    """
    assert used_tokens <= capacity_tokens, f"{used_tokens=} is past the end of a {capacity_tokens}-token buffer"

    prefix = ttnn.from_torch(
        host_buffer[:, :, :used_tokens, :].contiguous(),
        mesh_mapper=mesh_mapper,
        layout=layout,
        device=mesh_device,
        dtype=dtype,
    )
    if used_tokens == capacity_tokens:
        return prefix

    full = ttnn.pad(prefix, [(0, 0), (0, 0), (0, capacity_tokens - used_tokens), (0, 0)], pad_value)
    ttnn.deallocate(prefix)
    return full


# Which index of `ttnn.get_device_tensors` holds which (dispatch group, chip), per mesh shape. Read
# off the mesh once per process rather than assumed -- see `_ep_shard_order`.
_EP_SHARD_ORDER = {}


def _ep_shard_order(mesh_device):
    """The (dispatch group, chip) each shard of `ttnn.get_device_tensors` belongs to, in shard order.

    `get_device_tensors` hands back shards in the tensor's own coordinate order, and that order is
    not part of its contract, so this reads it off the mesh instead of assuming row-major: hand the
    EP mapper a tensor whose every element is its own (group, chip), pull the shards back, and see
    where each one landed. A few hundred bytes and one round trip, cached per mesh shape because the
    mapper depends on nothing else.

    The permutation assert is the real check: if this ever stopped being a bijection onto the mesh,
    the alternative is an output tensor assembled from the right data in the wrong places.
    """
    dispatch_group_size, num_dispatch_groups = mesh_device.shape[0], mesh_device.shape[1]
    key = (dispatch_group_size, num_dispatch_groups)
    if key in _EP_SHARD_ORDER:
        return _EP_SHARD_ORDER[key]

    # Laid out like the combine output's leading two dims -- (group, chip) -- so the EP mapper sends
    # each element to the device that will hold that (group, chip) of the output. The trailing 32 is
    # only to keep a row-major page a sensible size.
    marker = torch.arange(num_dispatch_groups * dispatch_group_size, dtype=torch.int32)
    marker = marker.reshape(num_dispatch_groups, dispatch_group_size, 1, 1).repeat(1, 1, 1, ttnn.TILE_SIZE)
    tt_marker = ttnn.from_torch(
        marker,
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    order = []
    for shard in ttnn.get_device_tensors(tt_marker):
        value = int(ttnn.to_torch(shard).flatten()[0].item())
        order.append((value // dispatch_group_size, value % dispatch_group_size))
    ttnn.deallocate(tt_marker)

    expected = sorted((g, c) for g in range(num_dispatch_groups) for c in range(dispatch_group_size))
    assert sorted(order) == expected, f"EP shard order is not a permutation of the mesh: {order}"

    logger.debug(f"[ep shard order] {key} -> {order}")
    _EP_SHARD_ORDER[key] = order
    return order


def _download_ep_output(tt_output, mesh_device):
    """Bring an EP-sharded combine output to host, without the mesh composer.

    `to_torch(mesh_composer=...)` on this tensor is almost all host work: measured at 8x4, the whole
    call is 14.85s, of which the device-to-host transfer is 0.31s (2.24 GiB at 7.2 GB/s) and the
    other 14.51s is stitching 32 shards into one rank-5 tensor. Converting those shards one at a
    time costs 2.91s, so copying them into a preallocated destination -- one pass over 2.24 GiB at
    memory speed -- gets the same tensor for about a quarter of the time.

    Same result, not an approximation: every shard is copied in full, into the place the composer
    would have put it. The shape assert downstream and the PCC check itself are what would catch a
    misplacement.
    """
    shards = ttnn.get_device_tensors(tt_output)
    order = _ep_shard_order(mesh_device)
    assert len(shards) == len(order), f"{len(shards)} shards from a {len(order)}-device mesh"

    dispatch_group_size, num_dispatch_groups = mesh_device.shape[0], mesh_device.shape[1]
    out = None
    for shard, (group, chip) in zip(shards, order):
        local = ttnn.to_torch(shard)
        assert (
            local.shape[0] == 1 and local.shape[1] == 1
        ), f"expected a (1, 1, ...) per-device shard, got {local.shape}"
        if out is None:
            out = torch.empty((num_dispatch_groups, dispatch_group_size, *local.shape[2:]), dtype=local.dtype)
        out[group, chip] = local[0, 0]
    return out


def run_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    topology,
    use_predictable_data,
    dispatched_buffer_layout,
    use_fp8_output,
    num_links=2,
    cmb_version=1,
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

    # How much of the worst-case buffer this router will fill. Known from the expert regions the gate
    # outputs above already lay out, so it is known before dispatch runs rather than after. Both the
    # buffer and its metadata share the dispatch buffer's token layout, so one prefix bounds both.
    used_tokens = dispatch_buffer_used_tokens(expert_token_counts, expert_region_offsets)
    logger.debug(f"{used_tokens=} of {max_dispatch_buffer_token_size=} dispatch buffer tokens")

    # Initialize torch dispatch module with num_dispatch_groups support.
    #
    # Sized to the prefix the router fills, not to the worst-case capacity the device buffer has: the
    # rest of that capacity is a constant, and `_upload_dispatch_buffer` puts it back on device. At
    # 8x4 with DeepSeek V3 that is the difference between generating a 35 GiB host tensor per case
    # and a 1.3 GiB one -- 3.7% of it held tokens. Every index dispatch writes comes from
    # expert_offsets, which is bounded by exactly this prefix, so a wrong prefix would raise here
    # rather than quietly hand the op a truncated buffer.
    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=used_tokens,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Run dispatch for each EP rank with rank-specific weights
    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    # Use different sharding: shard both dimensions
    mesh_mapper = get_ep_mesh_mapper(mesh_device)

    tt_dispatched_buffer = _upload_dispatch_buffer(
        dispatched_buffer,
        max_dispatch_buffer_token_size,
        used_tokens,
        # Dispatch zero-fills the buffer it writes into.
        0.0,
        mesh_mapper=mesh_mapper,
        layout=dispatched_buffer_layout,
        dtype=ttnn.bfloat16,
        mesh_device=mesh_device,
    )

    tt_dispatched_metadata = _upload_dispatch_buffer(
        dispatched_metadata,
        max_dispatch_buffer_token_size,
        used_tokens,
        # Metadata starts at -1, not 0: an unwritten slot has no origin chip, token or top-k slot.
        -1,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_device=mesh_device,
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
    if cmb_version == 1:
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

        combine_inputs = (
            tt_dispatched_buffer,
            tt_dispatched_metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
        )
    else:
        tt_expert_offsets = ttnn.from_torch(
            expert_offsets,
            # expert_offsets has to be replicated along the ring axis: every chip needs every origin
            # chip's run boundaries for the experts it hosts, which is what lets each kernel walk the
            # token sequence locally instead of exchanging addresses.
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

        tt_combine = TtCombine2dModule(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=topology,
        )

        combine_inputs = (
            tt_dispatched_buffer,
            tt_dispatched_metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
            tt_expert_offsets,
        )

    # combine cannot be launched for the FIRST time inside a trace capture.
    #
    # A capture records commands instead of running them, and rejects host-to-device writes:
    # fd_mesh_command_queue.cpp, "Writes are not supported during trace capture". combine writes on a
    # program cache MISS and only then -- create_workload_descriptor allocates two cross-device
    # GlobalSemaphores and zeroes them on device. So a cold op inside a capture is a guaranteed
    # TT_FATAL, whatever the shapes.
    #
    # This launch exists to take that miss outside the capture. The launch inside the capture is then
    # a cache hit, which builds nothing and writes nothing. It is not a warmup: the numbers still
    # come only from the replays. Its output is thrown away -- nothing reads it, and at 8x4 it is
    # 2.3 GB the capture below needs back.
    cache_miss_output = tt_combine(*combine_inputs)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(cache_miss_output)

    # Capture does NOT execute -- the op runs again on the replays below, and `tt_output` ends up
    # holding the LAST replay's result, which is what the check further down reads. Checking the last
    # launch rather than the first is deliberate: it proves a launch leaves the op's counters fit for
    # the next one, which a check placed between capture and replays would silently stop testing.
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_output = tt_combine(*combine_inputs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(_PERF_ITERATIONS):
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)

    # Step 6: Convert ttnn output to torch for comparison
    tt_output_torch = _download_ep_output(tt_output, mesh_device)
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


# Per-model combine shapes as (id_prefix, config, extended_model). DeepSeek V3 is the baseline and
# runs by default; every other model is gated behind @pytest.mark.extended_model.
COMBINE_MODELS = [
    ("dsv3", DeepSeekV3Config, SINGLE_GLX_AND_PROXY_MESHES),
    ("glm_51", GLM51Config, ONLY_PROXY_QB_MESH),
    ("kimi_k2_7", KimiK27Config, ONLY_PROXY_QB_MESH),
    ("minimax_m27", MiniMaxM27Config, ONLY_PROXY_QB_MESH),
    ("dsv4_pro", DeepSeekV4ProConfig, ONLY_PROXY_QB_MESH),
    ("dsv4_flash", DeepSeekV4FlashConfig, ONLY_PROXY_QB_MESH),
    ("gptoss_120b", GptOss120BConfig, ONLY_PROXY_QB_MESH),
]


# Scales down model hyper-params for a given hardware to obtain good/meaningful proxy test
# How exactly to scale it down is op-specific (more precisely - even op-implementation specific)
# Thus it makes sense for this to be combine-specific function
def _model_scaledown(model, ref_mesh, target_mesh):
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
            model_config = _model_scaledown(model_config_class(), test_meshes.full_model_mesh, target_mesh)
            params.append(
                pytest.param(
                    target_mesh,
                    fabric_to_device_params(fabric_cfg),
                    _SEQ_LEN_PER_CHIP,
                    model_config.EMB_SIZE,
                    model_config.NUM_ROUTED_EXPERTS,
                    model_config.NUM_EXPERTS_PER_TOKEN,
                    _DISPATCH_BUFFER_CAPACITY_FACTOR,
                    marks=pytest.mark.requires_mesh_topology(
                        mesh_shape=target_mesh, topology=_topo_marker(target_mesh, fabric_cfg)
                    ),
                    id=f"{model_name}-{_mesh_id(target_mesh, fabric_cfg)}-perf_no_pcc",
                )
            )

    return params


def _unsupported_param_combos(**params):
    use_predictable_data = params["use_predictable_data"]
    use_fp8_output = params["use_fp8_output"]
    dispatched_buffer_layout = params["dispatched_buffer_layout"]
    is_bh = params["is_bh"]

    # Predictable inputs are torch.arange(...), which produces values up to ~1.8M and
    # overflows fp8_e4m3fn's ±448 range — overflow encodes as NaN, breaking PCC.
    # Only exercise the fp8 path with random (N(0,1)) data that fits in range.
    if use_fp8_output and use_predictable_data:
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
    "mesh_device, device_params, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor",
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
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 1024, 4, _model_scaledown(GptOss20BConfig(), (4, 8), (4, 1))),
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 128, 2, _model_scaledown(GptOss120BConfig(), (4, 8), (4, 1))),
        _tc((4, 1), ttnn.FabricConfig.FABRIC_1D, 1280, 2, _model_scaledown(GptOss120BConfig(), (4, 8), (4, 1))),
        _tc((8, 1), ttnn.FabricConfig.FABRIC_1D, 128, 2, _model_scaledown(MiniMaxM3Config(), (8, 4), (8, 1))),
        _tc((8, 1), ttnn.FabricConfig.FABRIC_1D, 640, 2, _model_scaledown(MiniMaxM3Config(), (8, 4), (8, 1))),
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
        dispatched_buffer_layout=ttnn.TILE_LAYOUT,
        use_fp8_output=False,
        num_links=num_links,
    )


def _cmb_fabric2d_dimensions():
    """The (mesh, device_params, ...) tuple the fabric2d case runs on: the production mesh at the
    operating point this op's development has been measured against."""
    mesh = SINGLE_GLX_AND_PROXY_MESHES.full_model_mesh
    fabric_cfg = SINGLE_GLX_AND_PROXY_MESHES.target_meshes[mesh]
    model_config = _model_scaledown(DeepSeekV3Config(), SINGLE_GLX_AND_PROXY_MESHES.full_model_mesh, mesh)

    return [
        pytest.param(
            mesh,
            fabric_to_device_params(fabric_cfg),
            per_axis_topology(fabric_cfg)[0],  # sp axis; the op rings along cluster_axis=sp_axis
            _SEQ_LEN_PER_CHIP,
            model_config.EMB_SIZE,
            model_config.NUM_ROUTED_EXPERTS,
            model_config.NUM_EXPERTS_PER_TOKEN,
            _DISPATCH_BUFFER_CAPACITY_FACTOR,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=_topo_marker(mesh, fabric_cfg)),
            id=f"dsv3-fabric2d-{_mesh_id(mesh, fabric_cfg)}-row_major-2link-perf_no_pcc",
        )
    ]


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor",
    _cmb_fabric2d_dimensions(),
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_combine_fabric2d(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    topology,
):
    run_combine(
        mesh_device,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        num_links=2,
        topology=topology,
        use_predictable_data=False,
        dispatched_buffer_layout=ttnn.ROW_MAJOR_LAYOUT,
        use_fp8_output=False,
        cmb_version=2,
    )
