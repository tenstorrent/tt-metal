# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for end-to-end TTNN MoE dispatch→combine round-trip over the fabric2d ops.

The fabric2d version of test_ttnn_dispatch_combine.py: TtDispatch2dModule then TtCombine2dModule, with
no expert in between. The dispatch buffer, its metadata and the combine output are each compared
byte for byte against the torch reference.

combine_fabric2d cannot take a buffer that dispatch dropped tokens from, so the buffer is sized to
hold every routed token.

A right-padded case checks padded routing end to end: padded tokens carry the expert id
num_routed_experts, as the gate marks them, which the dispatch table's last column maps to no chip, so
neither the ops nor the reference route them. Dispatch also gets a padding_config built as
TtMoEGatePrefill.build_padding_config builds it, but the outputs cannot show whether the op used it.
"""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
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
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatch2dModule, TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import MoERoutingOutputs, TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    compare_exact,
    validate_composed,
    validate_dispatch_data,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results

# Same meshes as the fabric2d op tests: the 8x1 TorusY LoudBox proxy and the production 8x4 TorusXY
# Galaxy.
_MESH_IDS = (
    "fabric2d-torus-y-8x1-2link",
    "fabric2d-torus-xy-8x4-2link",
)
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id in _MESH_IDS]
assert len(_MESH_CONFIGS) == len(_MESH_IDS), "dispatch/combine fabric2d mesh configs missing from ALL_MESH_CONFIGS"


@dataclass
class RoundTripInputs:
    """Routing and tokens for one round trip, on host and on device.

    Host tensors feed the torch reference; device tensors feed the TTNN modules. Routing outputs come
    from TtMoERoutingSetup, the path the model takes, and are checked against the host ones.
    """

    sp_axis: int
    dispatch_group_size: int
    num_dispatch_groups: int
    experts_per_chip: int
    metadata_len: int
    max_dispatch_buffer_token_size: int
    max_dispatched_tokens_per_expert: int
    x: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    expert_dispatch_table: torch.Tensor
    expert_offsets: torch.Tensor
    expert_token_counts: torch.Tensor
    expert_region_offsets: torch.Tensor
    tt_x: ttnn.Tensor
    tt_indices: ttnn.Tensor
    tt_expert_dispatch_table: ttnn.Tensor
    tt_routing: MoERoutingOutputs
    tt_padding_config: ttnn.Tensor | None


def prepare_round_trip_inputs(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    routing,
    seed,
    padded_percent=0,
) -> RoundTripInputs:
    """Build the tokens and routing, run TtMoERoutingSetup, and check its outputs against the host.

    `routing` is "random" (initialize_test_inputs) or "round_robin" (initialize_predictable_test_inputs).
    Either way every token gets its own random embedding, so a token in the wrong slot cannot match.

    `padded_percent` right-pads the whole sequence, laid out chip after chip, by that share.
    """
    torch.manual_seed(seed)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        mesh_device.get_num_devices(),
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )

    init_inputs = {"random": initialize_test_inputs, "round_robin": initialize_predictable_test_inputs}[routing]
    _, weights, indices = init_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )
    x = torch.randn((dispatch_group_size, seq_len_per_chip, emb_dim), dtype=torch.bfloat16)

    tt_padding_config = None
    if padded_percent:
        actual_isl = int(dispatch_group_size * seq_len_per_chip * (1 - padded_percent / 100))
        real_tokens = [
            min(seq_len_per_chip, max(0, actual_isl - chip * seq_len_per_chip)) for chip in range(dispatch_group_size)
        ]
        logger.info(f"right padding: real tokens per chip {real_tokens}")
        for chip, real in enumerate(real_tokens):
            indices[chip, real:, :] = num_routed_experts
        # [real_token_count, pad_side], pad_side 0 = right. Sharded over SP, replicated over groups.
        tt_padding_config = ttnn.from_torch(
            torch.tensor([[real, 0] for real in real_tokens], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        )

    # x and indices: sharded across SP axis, replicated across dispatch groups
    mesh_mapper_dispatch_inputs = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    tt_routing = TtMoERoutingSetup(
        mesh_device=mesh_device,
        expert_dispatch_table=expert_dispatch_table,
        num_links=num_links,
        experts_per_chip=experts_per_chip,
    )(
        ttnn_top_k_experts_indices=indices,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # The fabric2d ops do not check this table: a chip that forwards tokens uses it to know how many
    # to wait for and pass on. So check that every chip holds every source chip's offsets.
    ep_composer = get_ep_mesh_composer(mesh_device)
    host_all_offsets = ttnn.to_torch(
        ttnn.unsqueeze_to_4D(tt_routing.all_global_dispatch_offsets), mesh_composer=ep_composer
    )
    host_token_counts = ttnn.to_torch(
        ttnn.unsqueeze_to_4D(tt_routing.total_counts_per_expert), mesh_composer=ep_composer
    ).squeeze(2)
    all_offsets_result = validate_composed(
        host_all_offsets.int(),
        expert_offsets.int().unsqueeze(1).expand(-1, dispatch_group_size, -1, -1),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="all_expert_offsets",
    )
    counts_result = validate_composed(
        host_token_counts.int(),
        expert_token_counts.int(),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="expert_token_counts",
    )
    log_validation_results(
        results=[all_offsets_result, counts_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Routing Setup Validation",
    )
    all_offsets_result.assert_passed("Offsets of every source chip differ from the host before dispatch")
    counts_result.assert_passed("Expert token counts mismatch before dispatch")

    buffer_end = int((expert_region_offsets + expert_token_counts).max())
    assert buffer_end <= max_dispatch_buffer_token_size, (
        f"The routing needs {buffer_end} dispatch buffer pages, but the buffer has only "
        f"{max_dispatch_buffer_token_size}. Dispatch would drop tokens, which combine_fabric2d cannot handle."
    )

    return RoundTripInputs(
        sp_axis=sp_axis,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        x=x,
        weights=weights,
        indices=indices,
        expert_dispatch_table=expert_dispatch_table,
        expert_offsets=expert_offsets,
        expert_token_counts=expert_token_counts,
        expert_region_offsets=expert_region_offsets,
        tt_x=tt_x,
        tt_indices=tt_indices,
        tt_expert_dispatch_table=tt_expert_dispatch_table,
        tt_routing=tt_routing,
        tt_padding_config=tt_padding_config,
    )


def dispatch_and_combine_2d(
    mesh_device,
    inp: RoundTripInputs,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    num_links,
    dispatched_buffer_layout,
):
    """TtDispatch2dModule then TtCombine2dModule, with no expert in between.

    Returns (dispatched_buffer, metadata, combine_output). dispatched_buffer is dispatch's ROW_MAJOR
    output; combine gets it in `dispatched_buffer_layout`.
    """
    tt_dispatch_module = TtDispatch2dModule(
        mesh_device=mesh_device,
        experts_per_chip=inp.experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatch_buffer_token_size=inp.max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=inp.sp_axis,
        num_links=num_links,
    )
    tt_combine_module = TtCombine2dModule(
        mesh_device=mesh_device,
        experts_per_chip=inp.experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=inp.sp_axis,
        num_links=num_links,
    )

    routing = inp.tt_routing
    tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
        inp.tt_x,
        inp.tt_indices,
        expert_dispatch_table=inp.tt_expert_dispatch_table,
        expert_token_counts=routing.total_counts_per_expert,
        expert_region_offsets=routing.expert_region_offsets,
        all_expert_offsets=routing.all_global_dispatch_offsets,
        padding_config=inp.tt_padding_config,
    )
    tt_output = tt_combine_module(
        ttnn.to_layout(tt_dispatched_buffer, dispatched_buffer_layout),
        tt_metadata,
        expert_token_counts=routing.total_counts_per_expert,
        expert_region_offsets=routing.expert_region_offsets,
        all_expert_offsets=routing.all_global_dispatch_offsets,
    )
    ttnn.synchronize_device(mesh_device)
    return tt_dispatched_buffer, tt_metadata, tt_output


def _same_bits(expected, actual, *_):
    """Byte-exact comparison for validate_dispatch_data."""
    if torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
        return True, None
    return False, f"{int((expected != actual).any(-1).sum())}/{expected.shape[0]} tokens differ"


def check_round_trip(
    mesh_device,
    inp: RoundTripInputs,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    num_links,
    dispatched_buffer_layout,
):
    """Run one round trip and compare dispatch buffer, metadata and combine output with torch, byte for byte."""
    tt_dispatched_buffer, tt_metadata, tt_output = dispatch_and_combine_2d(
        mesh_device,
        inp,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        num_links,
        dispatched_buffer_layout,
    )

    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=inp.dispatch_group_size,
        experts_per_chip=inp.experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=inp.metadata_len,
        max_dispatched_tokens_per_expert=inp.max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=inp.max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=inp.num_dispatch_groups,
        expert_dispatch_table=inp.expert_dispatch_table,
    )
    torch_dispatched_buffer, torch_dispatched_metadata = torch_dispatch_module(
        inp.x, inp.weights, inp.indices, inp.expert_offsets
    )
    # The reference buffer is float32 holding bf16 values, so this cast is exact.
    torch_dispatched_buffer = torch_dispatched_buffer.to(torch.bfloat16)
    torch_combine_module = TorchCombineModule(
        dispatch_group_size=inp.dispatch_group_size,
        experts_per_chip=inp.experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=inp.num_dispatch_groups,
    )
    torch_output = torch_combine_module(
        torch_dispatched_buffer, torch_dispatched_metadata, inp.expert_token_counts, inp.expert_region_offsets
    )

    # Dispatch: the pages each expert's tokens fill. The rest of the buffer is never written.
    ep_composer = get_ep_mesh_composer(mesh_device)
    table = inp.expert_dispatch_table[:, :num_routed_experts]
    slot_args = (
        inp.expert_region_offsets,
        inp.expert_token_counts,
        table,
        inp.num_dispatch_groups,
        inp.dispatch_group_size,
        inp.experts_per_chip,
    )
    buffer_result = validate_dispatch_data(
        torch_dispatched_buffer,
        ttnn.to_torch(tt_dispatched_buffer, mesh_composer=ep_composer, dtype=torch.bfloat16),
        *slot_args,
        compare_fn=_same_bits,
        name="dispatched_buffer",
    )
    metadata_result = validate_dispatch_metadata(
        torch_dispatched_metadata,
        ttnn.to_torch(tt_metadata, mesh_composer=ep_composer),
        *slot_args,
    )
    log_validation_results(
        results=[buffer_result, metadata_result],
        num_dispatch_groups=inp.num_dispatch_groups,
        dispatch_group_size=inp.dispatch_group_size,
        title="Dispatch Validation",
    )
    buffer_result.assert_passed("Dispatch buffer differs from the torch reference")
    metadata_result.assert_passed("Dispatch metadata differs from the torch reference")

    # Combine: (num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim).
    # A dispatch group writes only the slots of the experts it hosts and leaves the rest untouched.
    y = ttnn.to_torch(tt_output, mesh_composer=ep_composer, dtype=torch.bfloat16)
    slots_checked = 0
    bad = []
    for group in range(inp.num_dispatch_groups):
        # The full table, whose last column maps padded tokens to -1.
        kept = inp.expert_dispatch_table[group][inp.indices.long()] != -1
        got, want = y[group][kept], torch_output[kept]
        slots_checked += int(kept.sum())
        if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
            bad.append(f"group {group}: {int((got != want).any(-1).sum())}/{int(kept.sum())} slots differ")
    logger.info(f"combine output: {slots_checked} slots compared byte for byte")
    assert slots_checked > 0, "No slot routed to any dispatch group; the routing is wrong"
    assert not bad, f"Combine output differs from the torch reference: {bad}"


# Same shape as the DeepSeek V3 case of test_ttnn_dispatch_combine: production emb_dim, a quarter of
# the routed experts, top-2, capacity factor 2. top-k equal to the capacity factor means the buffer
# holds every routed token even if all of them land on one chip, which combine_fabric2d needs.
@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor",
    [
        pytest.param(
            640,
            DeepSeekV3Config.EMB_SIZE,
            DeepSeekV3Config.NUM_ROUTED_EXPERTS // 4,
            2,
            2,
            id="dsv3-640-avg",
        )
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# Combine gets the buffer as bf16 TILE or ROW_MAJOR. In the model it gets the routed expert's bfloat8_b
# TILE output, which only the PCC tests in test_ttnn_moe.py cover.
# Exclusions: ROW_MAJOR and padding run on the 8x4 mesh only, to keep them out of the LoudBox CI job, and
# never together. 30% padding on SP=8 leaves one chip partly padded and two fully padded.
@pytest.mark.parametrize(
    "dispatched_buffer_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "row_major"],
)
@pytest.mark.parametrize("padded_percent", [0, 30], ids=lambda p: f"pad{p}")
@pytest.mark.uncollect_if(
    pred=lambda **params: (
        (params["dispatched_buffer_layout"] == ttnn.ROW_MAJOR_LAYOUT or params["padded_percent"])
        and params["mesh_device"] != (8, 4)
    )
    or (params["dispatched_buffer_layout"] == ttnn.ROW_MAJOR_LAYOUT and params["padded_percent"])
)
@pytest.mark.timeout(900)
def test_ttnn_dispatch_combine_fabric2d(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    dispatched_buffer_layout,
    padded_percent,
):
    # Two round trips with different tokens. combine_fabric2d does not clear its output, so a slot it
    # failed to write could still hold the right value from an earlier run with the same tokens; the
    # second trip's tokens differ from the first's. The second trip also uses a different routing.
    cache_entries = []
    for routing, seed in (("random", 1), ("round_robin", 2)):
        logger.info(f"Round trip with {routing} routing, seed {seed}")
        inp = prepare_round_trip_inputs(
            mesh_device,
            seq_len_per_chip,
            emb_dim,
            num_routed_experts,
            num_experts_per_tok,
            dispatch_buffer_capacity_factor,
            num_links,
            routing,
            seed,
            padded_percent=padded_percent,
        )
        check_round_trip(
            mesh_device,
            inp,
            seq_len_per_chip,
            emb_dim,
            num_routed_experts,
            num_experts_per_tok,
            num_links,
            dispatched_buffer_layout,
        )
        cache_entries.append(mesh_device.num_program_cache_entries())
    # Both trips have the same shapes, so the second must reuse every program the first built.
    assert cache_entries[1] == cache_entries[0], f"the second round trip built new programs: {cache_entries}"
