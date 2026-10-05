# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Mesh test for hybrid_routed_expert_moe overlapped with combine_fabric2d in one program.

Graded against a host PyTorch reference run sequentially: TorchExpert (the SwiGLU FFN, fp32) over every
expert's dispatched rows, then TorchCombineModule. The device runs bfloat4_b weights, bfloat8_b activations
and LoFi, so the check is per-slot PCC over every (chip, token, topk) slot combine writes, the same
validate_combine_output the combine unit test uses. A slot combine read before the routed expert finished
writing it holds unrelated data, so a handoff bug fails that slot however close the rest is.

seq 640, because shorter sequences finish every expert before combine reaches it and never exercise
the wait. Both cases run the threshold the model ships: on (8, 1) balanced leaves every expert under it
so the fused pass takes them all, and hot-expert is the only case that lifts one into the unified half.

TtMoe turns this overlap on by default wherever the op exists, so the model suites cover the production
path; this module is pruned from CI and run by hand.
"""

from pathlib import Path
from types import SimpleNamespace

import statistics
import time

import pytest
import heapq

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import validate_combine_output
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_FULL_MESH = (8, 4)
_MESHES = {
    (8, 4): ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    (8, 1): ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
}
_SEQ_LEN_PER_CHIP = 640
_CAPACITY_FACTOR = 8
# The models this op is deployed for. Each contributes its own emb, MoE hidden, expert count and top-k.
_MODELS = {"kimi-k27": KimiK27Config, "glm-53": GLM53Config}
# Routing captured off-device from total_counts_per_expert, one dispatch group of a 32-chip chunked prefill
# of each model's code_debug golden, scaled to this test's 10240 in-group routings. Scaling leaves the mean
# intact -- 106.7 for Kimi's 96 experts, 160 for GLM's 64 -- and keeps the shape the model actually routes.
# Real routing is far lumpier than a uniform draw: the flattest captured layer still runs max/mean 1.8-2.8,
# and the most skewed puts a quarter of the chunk on one expert.
# fmt: off
_CAPTURED_COUNTS = {
    "kimi-k27": {
        "balanced": (
            # MoE layer 34, chunk 1
            282, 85, 112, 78, 94, 35, 60, 73, 10, 80, 87, 122, 107, 215, 41, 85, 105, 76, 114, 37, 284, 146, 59,
            44, 120, 78, 47, 0, 36, 162, 120, 247, 95, 115, 131, 228, 122, 103, 183, 52, 157, 18, 2, 294, 238,
            105, 107, 27, 249, 23, 216, 99, 141, 162, 181, 62, 144, 125, 17, 33, 52, 73, 35, 200, 80, 138, 217,
            69, 38, 114, 160, 38, 155, 42, 165, 114, 28, 87, 88, 2, 63, 93, 2, 99, 67, 45, 141, 287, 105, 27,
            244, 159, 53, 194, 87, 9
        ),
        "hot-expert": (
            # MoE layer 5, chunk 0
            8, 109, 86, 102, 20, 221, 107, 54, 188, 5, 53, 67, 36, 50, 73, 91, 25, 108, 36, 43, 36, 172, 61, 46,
            32, 115, 40, 4, 0, 58, 1, 40, 2878, 4, 108, 53, 286, 58, 92, 97, 48, 145, 68, 70, 67, 132, 35, 340,
            41, 17, 53, 288, 130, 66, 49, 230, 52, 6, 47, 163, 6, 42, 17, 65, 56, 6, 2, 98, 138, 16, 43, 31,
            112, 49, 110, 94, 85, 44, 121, 140, 76, 124, 65, 155, 181, 109, 60, 29, 38, 16, 53, 16, 139, 54, 82,
            58
        ),
    },
    "glm-53": {
        "balanced": (
            # MoE layer 3, chunk 5
            196, 166, 180, 180, 212, 259, 136, 103, 250, 195, 176, 66, 203, 171, 179, 111, 61, 211, 78, 130,
            129, 152, 231, 92, 178, 145, 149, 198, 92, 181, 126, 180, 216, 85, 192, 135, 190, 129, 205, 159,
            114, 44, 157, 102, 123, 151, 146, 288, 167, 101, 185, 172, 204, 120, 249, 108, 179, 224, 208, 190,
            66, 188, 102, 225
        ),
        "hot-expert": (
            # MoE layer 21, chunk 9
            146, 56, 90, 99, 149, 120, 215, 350, 25, 55, 129, 58, 153, 80, 155, 91, 52, 3, 143, 115, 32, 183,
            142, 62, 148, 23, 0, 33, 69, 85, 178, 145, 4, 148, 23, 226, 129, 2877, 264, 54, 163, 117, 91, 49,
            14, 182, 223, 419, 114, 78, 185, 124, 150, 209, 33, 120, 4, 177, 55, 29, 68, 67, 308, 152
        ),
    },
}
# fmt: on
# One real cell per model for (8, 4): MoE layer 45, chunk 3 of the same code_debug prefill, the median-skew
# cell of both models (busiest expert 7.6x the mean for GLM, 11.7x for Kimi). Unlike balanced and hot-expert, it
# keeps the model's own top-8 spread across the dispatch groups, so a token lands a variable 0-8 of its experts
# in each group and the groups carry unequal totals. GLM 5.3 replays the device's top-8 ids token by token, in
# origin-chip order; Kimi K2.7 only has per-expert counts captured, so its tokens are rebuilt from those.
_GLM53_REAL_ROUTING = Path(__file__).parent / "routing_captures" / "glm53_code_debug_L45_c3.pt"
# fmt: off
_KIMI_K27_REAL_COUNTS = (
    8, 94, 151, 131, 63, 110, 79, 2, 49, 130, 104, 168, 48, 71, 222, 62, 151, 38, 22, 32, 123, 77, 151, 148, 51, 49,
    26, 168, 76, 50, 12, 49, 12, 113, 64, 35, 125, 89, 44, 197, 41, 127, 250, 47, 42, 45, 307, 65, 128, 12, 56, 11,
    68, 88, 109, 80, 21, 126, 294, 80, 10, 28, 24, 83, 154, 259, 13, 432, 365, 93, 105, 39, 56, 103, 139, 6, 38, 62,
    55, 53, 5, 111, 31, 182, 138, 47, 27, 35, 30, 38, 99, 17, 92, 125, 109, 93, 24, 79, 22, 77, 759, 36, 8, 72, 76,
    143, 78, 138, 423, 71, 43, 38, 80, 73, 163, 65, 39, 98, 131, 148, 64, 39, 243, 23, 68, 149, 24, 54, 46, 104, 73,
    45, 110, 4, 51, 499, 231, 86, 80, 90, 85, 108, 212, 109, 6, 69, 68, 162, 33, 87, 65, 153, 39, 249, 47, 83, 134,
    110, 43, 344, 81, 58, 408, 14, 112, 164, 79, 190, 123, 127, 18, 50, 208, 37, 329, 81, 53, 246, 74, 336, 43,
    1252, 2, 138, 46, 319, 101, 95, 32, 35, 127, 190, 133, 125, 80, 70, 1129, 43, 113, 161, 36, 63, 266, 94, 24, 8,
    130, 3, 53, 3, 321, 38, 81, 307, 111, 72, 112, 45, 45, 82, 32, 7, 111, 101, 214, 30, 19, 3, 99, 19, 48, 54, 74,
    300, 165, 108, 2, 67, 82, 564, 194, 24, 100, 440, 140, 57, 1156, 52, 44, 52, 39, 114, 29, 177, 89, 102, 148, 77,
    25, 115, 56, 16, 260, 39, 34, 117, 96, 50, 118, 13, 70, 156, 37, 14, 68, 80, 131, 57, 80, 117, 102, 150, 67, 58,
    95, 278, 393, 157, 44, 33, 70, 114, 63, 123, 106, 44, 125, 58, 174, 312, 115, 13, 130, 2, 136, 39, 49, 53, 226,
    61, 81, 199, 34, 162, 69, 42, 38, 6, 8, 35, 28, 79, 102, 47, 72, 53, 97, 67, 71, 34, 2, 366, 58, 45, 233, 106,
    49, 42, 11, 15, 30, 71, 105, 86, 124, 61, 17, 83, 94, 40, 53, 60, 26, 78, 66, 197, 149, 128, 81, 72, 90, 118,
    49, 79, 6, 60, 38, 42, 136, 197, 8, 50, 112, 129, 27, 130, 90, 100, 86, 185, 134, 116, 31, 27
)
# fmt: on
# Measured programs per configuration; the median is reported.
_PERF_ITERS = 5
# What tells the three programs apart in the real-time profiler's records. The overlap builds the routed
# expert's kernels and combine_fabric2d's, so neither directory names it on its own; the collector, which
# only the overlap has, does.
_OVERLAP_KERNELS = "/collector_combine_fabric2d.cpp"
_SOLO_RE_KERNELS = "/hybrid_routed_expert_ffn/device/kernels/"
_COMBINE_KERNELS = "/deepseek_prefill/combine_fabric2d/"


def _scaled_model(mesh, model_id):
    """`model_id` at the full mesh's experts-per-chip and per-group expert activation."""
    model = _MODELS[model_id]()
    chips = mesh[0] * mesh[1]
    model.NUM_ROUTED_EXPERTS = (model.NUM_ROUTED_EXPERTS // (_FULL_MESH[0] * _FULL_MESH[1])) * chips
    if mesh[1] != _FULL_MESH[1]:
        model.NUM_EXPERTS_PER_TOKEN = max(1, (model.NUM_EXPERTS_PER_TOKEN // _FULL_MESH[1]) * mesh[1])
    return model


# The perf test replays its programs from traces, which need a reserved region; the overlap is one program.
_TRACE_REGION_SIZE = 64 * 1024 * 1024


def _device_params(fabric_cfg):
    params = dict(fabric_to_device_params(fabric_cfg))
    params["trace_region_size"] = _TRACE_REGION_SIZE
    return params


def _mesh_params():
    params = []
    for mesh, fabric_cfg in _MESHES.items():
        topo = "ring" if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_Y else f"mesh-{mesh[0]}x{mesh[1]}"
        for model_id in _MODELS:
            for threshold_id in ("balanced", "hot-expert") + (("real",) if mesh == _FULL_MESH else ()):
                params.append(
                    pytest.param(
                        mesh,
                        _device_params(fabric_cfg),
                        threshold_id,
                        model_id,
                        None,  # variant
                        marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=topo),
                        id=f"{model_id}-{mesh[0]}x{mesh[1]}-{threshold_id}",
                    )
                )
    # The (8, 1) ring on a Galaxy: opened alone its fabric cannot come up, since every router must handshake with a
    # live partner, so the full mesh opens on its own torus fabric and the op runs on one column -- 8 chips, one
    # dispatch group, the same model scaling and routing as a standalone (8, 1).
    for model_id in _MODELS:
        for threshold_id in ("balanced", "hot-expert"):
            params.append(
                pytest.param(
                    _FULL_MESH,
                    _device_params(_MESHES[_FULL_MESH]),
                    threshold_id,
                    model_id,
                    "8x1-submesh",
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=_FULL_MESH, topology="mesh-8x4"),
                    id=f"{model_id}-8x1-galaxy-{threshold_id}",
                )
            )
    # The whole (8, 4) mesh up -- fabric, program, all 32 chips -- but only dispatch group 0 routed: it carries
    # exactly the (8, 1) work and the other three columns get no tokens. Set against the (8, 1) submesh, it
    # separates the four rings' concurrent traffic from the cost of a 32-chip program.
    for model_id in _MODELS:
        for threshold_id in ("balanced", "hot-expert"):
            params.append(
                pytest.param(
                    _FULL_MESH,
                    _device_params(_MESHES[_FULL_MESH]),
                    threshold_id,
                    model_id,
                    "dg0-only",
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=_FULL_MESH, topology="mesh-8x4"),
                    id=f"{model_id}-8x4-dg0only-{threshold_id}",
                )
            )
    return params


def _int_tensor(torch_tensor, mesh_device, mesh_mapper, dtype=ttnn.int32):
    return ttnn.from_torch(
        torch_tensor, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=dtype
    )


def _indices_from_counts(counts, chips, seq, topk):
    """Routing indices whose per-expert totals are exactly `counts`.

    Each token takes the topk experts with the most tokens still to place, which keeps its picks distinct
    and lands every count exactly -- feasible because no expert is owed more tokens than there are tokens.
    The token order is then permuted so a heavily loaded expert is not confined to the first rows.
    """
    tokens = chips * seq
    assert sum(counts) == tokens * topk, f"counts sum to {sum(counts)}, need {tokens * topk}"
    assert max(counts) <= tokens, "an expert cannot take the same token twice"
    remaining = [(-c, e) for e, c in enumerate(counts) if c]
    heapq.heapify(remaining)
    rows = []
    for _ in range(tokens):
        taken = [heapq.heappop(remaining) for _ in range(topk)]
        rows.append([e for _, e in taken])
        for c, e in taken:
            if c + 1 < 0:
                heapq.heappush(remaining, (c + 1, e))
    order = torch.randperm(tokens, generator=torch.Generator().manual_seed(42)).tolist()
    return torch.tensor([rows[i] for i in order], dtype=torch.int32).reshape(chips, seq, topk)


def _spread_indices_from_counts(counts, chips, seq, topk):
    """Routing indices whose per-expert totals are exactly `counts`, with each token's picks spread the way
    independent routing spreads them.

    `_indices_from_counts` hands each token the experts with the most tokens left to place, which clumps a
    token's picks: most tokens then land all or none of theirs in one dispatch group. Here each expert, heaviest
    first, takes its tokens from those with the most free slots, ties broken at random -- which always leaves a
    completion (Gale-Ryser) and scatters every token's picks across the groups.
    """
    tokens = chips * seq
    assert sum(counts) == tokens * topk, f"counts sum to {sum(counts)}, need {tokens * topk}"
    assert max(counts) <= tokens, "an expert cannot take the same token twice"
    generator = torch.Generator().manual_seed(42)
    free = torch.full((tokens,), topk, dtype=torch.int64)
    rows = torch.empty((tokens, topk), dtype=torch.int32)
    for expert in sorted(range(len(counts)), key=lambda e: -counts[e]):
        if not counts[expert]:
            continue
        key = free.double() + torch.rand(tokens, generator=generator, dtype=torch.float64)
        chosen = key.topk(counts[expert]).indices
        rows[chosen, topk - free[chosen]] = expert
        free[chosen] -= 1
    assert int(free.sum()) == 0
    return rows.reshape(chips, seq, topk)


def _build_case(mesh_device, device_params, threshold_id, model_id, dg0_only=False):
    """One seq-640 layer of `model_id` on `mesh_device`: its inputs, and the three ways to run it.

    `dg0_only` routes every token to dispatch group 0 alone, at that group's top-k share, so the other groups
    get no tokens."""
    torch.manual_seed(42)
    model = _scaled_model(tuple(mesh_device.shape), model_id)
    emb_dim = model.EMB_SIZE
    hidden_dim = model.MOE_INTERMEDIATE_SIZE
    num_routed_experts = model.NUM_ROUTED_EXPERTS
    num_experts_per_tok = model.NUM_EXPERTS_PER_TOKEN
    num_links = 2

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    num_devices = mesh_device.get_num_devices()
    if dg0_only:
        assert num_experts_per_tok % num_dispatch_groups == 0, f"top-{num_experts_per_tok} over {num_dispatch_groups}"
        num_experts_per_tok //= num_dispatch_groups

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        _SEQ_LEN_PER_CHIP,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        _CAPACITY_FACTOR,
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size,
        _SEQ_LEN_PER_CHIP,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )
    idx_table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    # Replay the measured routing rather than the draw initialize_test_inputs made; x and the gate weights
    # it produced are kept.
    if threshold_id == "real":
        assert tuple(mesh_device.shape) == _FULL_MESH, "the real cell is whole-mesh routing"
        if model_id == "glm-53":
            indices = torch.load(_GLM53_REAL_ROUTING)["expert_ids"].to(torch.int32)
        else:
            indices = _spread_indices_from_counts(
                _KIMI_K27_REAL_COUNTS, dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok
            )
        assert tuple(indices.shape) == (dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok)
        assert int(indices.max()) < num_routed_experts
    else:
        # Every dispatch group replays the same one-group routing, so each ring carries exactly the (8, 1) work --
        # same counts, same tokens, same order -- and the meshes differ only in their links. Group g owns experts
        # g * group_experts onward, laid out chip for chip like (8, 1)'s, and each token takes its share of the
        # top-k from every group.
        group_counts = _CAPTURED_COUNTS[model_id][threshold_id]
        group_experts = len(group_counts)
        assert (
            group_experts * num_dispatch_groups == num_routed_experts
        ), f"{group_experts} captured counts x {num_dispatch_groups} groups for {num_routed_experts} experts"
        if dg0_only:
            # Group 0's experts are 0 .. group_experts-1, so no offset: the other groups see no token at all.
            indices = _indices_from_counts(group_counts, dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok)
        else:
            assert (
                num_experts_per_tok % num_dispatch_groups == 0
            ), f"top-{num_experts_per_tok} over {num_dispatch_groups} groups"
            group_indices = _indices_from_counts(
                group_counts, dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok // num_dispatch_groups
            )
            indices = torch.cat([group_indices + group * group_experts for group in range(num_dispatch_groups)], dim=-1)
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        _SEQ_LEN_PER_CHIP,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    dispatched_buffer, dispatched_metadata = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=_SEQ_LEN_PER_CHIP,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )(x, weights, indices, expert_offsets)

    counts = expert_token_counts.flatten()
    # Both cases run the crossover the model ships. Balanced leaves every expert under it, so the fused
    # pass takes them all -- which is what production does at this chunk size; the hot expert is the one
    # the threshold lifts into the unified pass.
    threshold = model.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD
    assert threshold < max_dispatched_tokens_per_expert
    hot = int(counts.argmax())
    logger.info(
        f"{num_routed_experts=} {num_experts_per_tok=} {experts_per_chip=} {max_dispatched_tokens_per_expert=} "
        f"{threshold=} fused experts={(counts <= threshold).sum().item()}/{counts.numel()} "
        f"busiest expert={hot} at {int(counts[hot])} tokens"
    )

    ep_mapper = get_ep_mesh_mapper(mesh_device)
    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    # ROW_MAJOR bf16: the tilized path is the one that writes the bf8 tiles combine reads.
    tt_x = ttnn.from_torch(
        dispatched_buffer, mesh_mapper=ep_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_metadata = _int_tensor(dispatched_metadata, mesh_device, ep_mapper)
    # UINT32, which the routed expert requires and combine also takes.
    # (1, experts) per device: the routed expert takes 1D or 2D, combine reads the last two dims.
    tt_counts = ttnn.squeeze(_int_tensor(expert_token_counts, mesh_device, counts_mapper, dtype=ttnn.uint32), 0)
    tt_region_offsets = ttnn.squeeze(
        _int_tensor(expert_region_offsets, mesh_device, counts_mapper, dtype=ttnn.uint32), 0
    )
    # Replicated along the ring axis: every chip needs every origin chip's run boundaries.
    tt_expert_offsets = _int_tensor(
        expert_offsets,
        mesh_device,
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
        dtype=ttnn.uint32,
    )
    tt_idx_slice = _int_tensor(idx_table, mesh_device, ep_mapper, dtype=ttnn.uint32)
    tt_idx_slice = ttnn.squeeze(ttnn.squeeze(tt_idx_slice, 0), 0)
    # Each chip holds its own dispatch group's rows -- one per ring chip -- replicated down the ring: combine only
    # relays tokens between the chips of one ring. Groups are mesh columns, so they shard across columns.
    tt_idx_full = _int_tensor(
        idx_table,
        mesh_device,
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
        dtype=ttnn.uint32,
    )

    # One weight set shared by every expert keeps host conversion bounded; the handoff does not care
    # which weights an expert has, only when its rows are written.
    expert_weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim) * 0.02,
    }
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=tt_idx_slice,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=[expert_weights] * (num_devices * experts_per_chip),
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    def routed_expert(**overlap):
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_x,
            tt_region_offsets,
            tt_counts,
            tt_idx_slice,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            hybrid_token_threshold=threshold,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
            **overlap,
        )

    # Solo, the routed expert writes the same bfloat8_b tiles it hands combine inside the overlap.
    def solo_routed_expert():
        return routed_expert()

    def combine(re_output, widen=True):
        # The standalone op takes bfloat16 only; bfloat8_b widens exactly, so this is the bytes the
        # overlap's untilizers produce. `widen=False` takes an already-widened input, so a trace of
        # this call is combine alone, as the eager measurement is.
        if widen:
            re_output = ttnn.typecast(re_output, ttnn.bfloat16)
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            re_output,
            tt_metadata,
            tt_counts,
            tt_region_offsets,
            tt_expert_offsets,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=per_axis_topology(device_params["fabric_config"])[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def overlapped():
        return routed_expert(
            dispatched_metadata=tt_metadata,
            expert_offsets=tt_expert_offsets,
            replicated_global_expert_idx_table=tt_idx_full,
            combine_axis=sp_axis,
            combine_num_links=num_links,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
        )

    reference = []

    def torch_reference():
        """TorchExpert over each expert's dispatched rows, then TorchCombineModule.

        Rewrites the host dispatched buffer in place -- it is already on the device, and a copy would be
        another ~19 GB at (8, 4) -- so the result is computed once and cached."""
        if reference:
            return reference[0]
        expert = TorchExpert(emb_dim, hidden_dim, torch_weights=expert_weights)
        with torch.no_grad():
            for group in range(num_dispatch_groups):
                for chip in range(dispatch_group_size):
                    for local_expert in range(experts_per_chip):
                        global_expert = ExpertMapping.get_global_expert_idx(
                            group=group,
                            chip=chip,
                            local_expert=local_expert,
                            experts_per_chip=experts_per_chip,
                            dispatch_group_size=dispatch_group_size,
                            num_dispatch_groups=num_dispatch_groups,
                            is_col_major=True,
                        )
                        start = int(expert_region_offsets[group, chip, global_expert])
                        rows = int(expert_token_counts[group, 0, global_expert])
                        if rows:
                            region = dispatched_buffer[group, chip, start : start + rows]
                            region.copy_(expert(region.float()).to(region.dtype))
        combine_ref = TorchCombineModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
            num_dispatch_groups=num_dispatch_groups,
        )
        reference.append(
            combine_ref(dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets)
        )
        return reference[0]

    return SimpleNamespace(
        emb_dim=emb_dim,
        composer=get_ep_mesh_composer(mesh_device),
        solo_routed_expert=solo_routed_expert,
        combine=combine,
        overlapped=overlapped,
        torch_reference=torch_reference,
        indices=indices,
        num_dispatch_groups=num_dispatch_groups,
        num_routed_experts=num_routed_experts,
        experts_per_chip=experts_per_chip,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id, model_id, variant",
    _mesh_params(),
    indirect=["mesh_device", "device_params"],
)
def test_hybrid_routed_expert_combine_overlap(mesh_device, device_params, threshold_id, model_id, variant):
    if variant == "8x1-submesh":
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(8, 1))
    case = _build_case(mesh_device, device_params, threshold_id, model_id, dg0_only=variant == "dg0-only")
    # Run the device first: the reference rewrites the host dispatched buffer in place.
    actuals = [ttnn.to_torch(case.overlapped(), mesh_composer=case.composer)]
    # Twice: the second run is a program-cache hit, which reuses the cached arena and fwd_arrived.
    actuals.append(ttnn.to_torch(case.overlapped(), mesh_composer=case.composer))
    expected = case.torch_reference()

    for run, actual in enumerate(actuals):
        result = validate_combine_output(
            expected,
            actual,
            case.indices,
            case.num_dispatch_groups,
            case.num_routed_experts,
            use_pcc=True,
            verbose=True,
            expert_dispatch_table=case.expert_dispatch_table,
            expert_token_counts=case.expert_token_counts,
            experts_per_chip=case.experts_per_chip,
        )
        worst = min((m[-1] for m in result.mismatches), default=None)
        logger.info(
            f"run {run}: {result.matches}/{result.total} combine slots match the PyTorch RE + combine reference"
            + (f", worst slot PCC {worst:.6f}" if worst is not None else "")
        )
        result.assert_passed(f"run {run}: overlapped RE + combine vs PyTorch RE + combine")


def _median_program_ns(mesh_device, run_fn, iters, is_target, label):
    """Median device time, slowest chip, of the one program per run that `is_target` picks out.

    Also logs each chip's median as a grid -- rows the chips of a ring (mesh rows), columns the dispatch groups
    (mesh columns) -- so a slow run can be pinned to the chips that set it."""

    def run_all():
        return [run_fn() for _ in range(iters)]

    outputs, records = profile_realtime_program(mesh_device, run_all, collect_all=True)
    per_program: dict = {}  # runtime_id -> {chip_id: duration_ns}, in arrival (= dispatch) order
    for record in records:
        if record["runtime_id"] and is_target(record["kernel_sources"]):
            per_program.setdefault(record["runtime_id"], {})[record["chip_id"]] = record["duration_ns"]
    # The warm-up run's record can be delivered after the window opens; records arrive in dispatch order,
    # so the measured runs are the last `iters`.
    runs = list(per_program.values())[-iters:]
    assert len(runs) == iters, f"{label}: expected {iters} programs, the profiler matched {len(per_program)}"

    rows, cols = tuple(mesh_device.shape)
    device_ids = list(mesh_device.get_device_ids())
    per_chip = {chip: statistics.median(run[chip] for run in runs if chip in run) for chip in runs[-1]}
    grid = [
        "  ".join(
            f"{per_chip[device_ids[r * cols + c]] / 1e3:7.1f}" if device_ids[r * cols + c] in per_chip else "      -"
            for c in range(cols)
        )
        for r in range(rows)
    ]
    slowest = max(per_chip, key=per_chip.get)
    at = device_ids.index(slowest)
    logger.info(
        f"{label} per chip, median us (rows: chip in ring, columns: dispatch group); slowest chip {slowest} at "
        f"row {at // cols}, group {at % cols}:\n" + "\n".join(grid)
    )
    return outputs, statistics.median(max(run.values()) for run in runs)


def _capture(mesh_device, run_fn):
    """Trace one call of `run_fn`. Returns the trace and what the call returned, which must stay alive until the
    trace is released: a replay writes into the buffers the capture allocated."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        result = run_fn()
    except BaseException:
        # A device left mid-capture hangs on close, so end and drop the capture before the error propagates.
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.release_trace(mesh_device, trace_id)
        raise
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    return trace_id, result


def _replay_us(mesh_device, trace_id, iters):
    """Wall time per replay over `iters` back-to-back replays, after one untimed replay. The real-time profiler
    does not report replays one by one, and back to back the host is off the critical path, so this is the
    device's throughput for the program."""
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) / iters * 1e6


def _has(sources, path):
    return any(path in source.replace("\\", "/") for source in sources)


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id, model_id, variant",
    _mesh_params(),
    indirect=["mesh_device", "device_params"],
)
def test_hybrid_routed_expert_combine_overlap_perf(mesh_device, device_params, threshold_id, model_id, variant):
    """The overlapped program against the same two ops back to back: hybrid routed expert, then
    combine_fabric2d. Sequential is the sum of the two programs' times, so it assumes no gap between
    them and is the best a sequential dispatch can do; the overlap must beat it.

    Measured twice. Eager, per chip with the real-time profiler, logged for information. Traced, as the
    model runs it, as wall time per replay: that is the verdict.

    Every solo routed-expert run comes first: its per-call arena takes all free L1, which combine's
    fwd_arrived holds a piece of from its first call on. One warm-up run of each program fills the
    program cache outside the measured window.
    """
    require_realtime_profiler("the RE + combine overlap perf test")
    if variant == "8x1-submesh":
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(8, 1))
    case = _build_case(mesh_device, device_params, threshold_id, model_id, dg0_only=variant == "dg0-only")

    warm_re = case.solo_routed_expert()
    re_outputs, re_ns = _median_program_ns(
        mesh_device,
        case.solo_routed_expert,
        _PERF_ITERS,
        lambda k: _has(k, _SOLO_RE_KERNELS) and not _has(k, _OVERLAP_KERNELS),
        "solo routed expert",
    )

    # Traced, as the model runs it. Eager, the host sends a 32-chip mesh this large a program slowly enough
    # that a ring's chips start it apart, and an overlapped chip then waits on its late neighbours' relays; a
    # replay starts every chip together. Each trace is captured, replayed and released in the same place its
    # eager run sits, for the same reason: the solo routed expert sized its arena to the whole of L1 on its
    # first call, so it must replay before combine's fwd_arrived takes any; the overlap keeps an arena of its
    # own, so it comes last. The widening sits outside the combine trace, as it does outside the eager number.
    re_trace, traced_re_out = _capture(mesh_device, case.solo_routed_expert)
    try:
        traced_re_us = _replay_us(mesh_device, re_trace, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, re_trace)

    case.combine(warm_re)
    re_iter = iter(re_outputs)
    _, combine_ns = _median_program_ns(
        mesh_device,
        lambda: case.combine(next(re_iter)),
        _PERF_ITERS,
        lambda k: _has(k, _COMBINE_KERNELS) and not _has(k, _OVERLAP_KERNELS),
        "combine_fabric2d",
    )

    widened = ttnn.typecast(traced_re_out, ttnn.bfloat16)
    combine_trace, combine_out = _capture(mesh_device, lambda: case.combine(widened, widen=False))
    try:
        traced_combine_us = _replay_us(mesh_device, combine_trace, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, combine_trace)
        del traced_re_out, widened, combine_out

    case.overlapped()
    _, overlap_ns = _median_program_ns(
        mesh_device, case.overlapped, _PERF_ITERS, lambda k: _has(k, _OVERLAP_KERNELS), "overlap"
    )

    sequential_ns = re_ns + combine_ns
    saved_ns = sequential_ns - overlap_ns
    logger.info(
        f"RE+combine {model_id} {threshold_id} eager: sequential {sequential_ns / 1e3:.1f} us (RE {re_ns / 1e3:.1f} + "
        f"combine {combine_ns / 1e3:.1f}), overlap {overlap_ns / 1e3:.1f} us -> {sequential_ns / overlap_ns:.3f}x, "
        f"saved {saved_ns / 1e3:.1f} us ({saved_ns / sequential_ns:.1%}); combine hidden "
        f"{min(saved_ns, combine_ns) / combine_ns:.0%}"
    )

    overlap_trace, overlap_out = _capture(mesh_device, case.overlapped)
    try:
        traced_overlap_us = _replay_us(mesh_device, overlap_trace, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, overlap_trace)
        del overlap_out

    traced_sequential_us = traced_re_us + traced_combine_us
    traced_saved_us = traced_sequential_us - traced_overlap_us
    logger.info(
        f"RE+combine {model_id} {threshold_id} traced: sequential {traced_sequential_us:.1f} us (RE {traced_re_us:.1f} "
        f"+ combine {traced_combine_us:.1f}), overlap {traced_overlap_us:.1f} us -> "
        f"{traced_sequential_us / traced_overlap_us:.3f}x, saved {traced_saved_us:.1f} us "
        f"({traced_saved_us / traced_sequential_us:.1%}); combine hidden "
        f"{min(traced_saved_us, traced_combine_us) / traced_combine_us:.0%}"
    )
    assert traced_overlap_us < traced_sequential_us, (
        f"traced overlap {traced_overlap_us:.1f} us is not faster than traced RE then combine "
        f"({traced_re_us:.1f} + {traced_combine_us:.1f} = {traced_sequential_us:.1f} us)"
    )
