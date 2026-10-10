# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Mesh test for hybrid_routed_expert_moe overlapped with combine_fabric2d in one program.

Graded against a host PyTorch reference run sequentially: TorchExpert (the SwiGLU FFN, fp32) over every
expert's dispatched rows, then TorchCombineModule. The device runs bfloat4_b weights, bfloat8_b activations
and LoFi, so the check is per-slot PCC over every (chip, token, topk) slot combine writes, the same
validate_combine_output the combine unit test uses. A slot combine read before the routed expert finished
writing it holds unrelated data, so a handoff bug fails that slot however close the rest is.

seq 640, because shorter sequences finish every expert before combine reaches it and never exercise
the wait. Every case runs the threshold the model ships: on (8, 1) balanced leaves every expert under it
so the fused pass takes them all, hot-expert lifts one into the unified half, and real-L45c3 and real-L11c9
(Kimi K2.7, 8x4 only) replay measured routing from that layer and chunk.

The perf test times every program replayed from a trace, as the model runs it. Eager, the host writes each
chip's program in turn and, on 8x4, issues the overlap slower than the device runs it, so a ring's chips
start hundreds of microseconds apart and the overlap waits on its late neighbours.

TtMoe runs this overlap in every traced Blackhole prefill, so CI runs the 8x4 accuracy cases on a BH Galaxy
(Disaggregated prefill op unit tests); the perf cases, the 8x1 cases and dg0-only are run by hand.
"""

from types import SimpleNamespace

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
# Two real Kimi K2.7 cells for (8, 4), from a code_debug prefill, named by layer and chunk. L45c3 is the
# median-skew cell (busiest expert 11.7x the mean). L11c9 is the most skewed of its 60 x 11 cells (expert 205
# takes 3458 tokens, 32.4x the mean). Unlike balanced and hot-expert, a real cell keeps the model's own top-8
# spread across the dispatch groups, so a token lands a variable 0-8 of its experts in each group and the groups
# carry unequal totals. Only per-expert counts are captured, so the tokens are rebuilt from those.
# fmt: off
_KIMI_K27_REAL_COUNTS = {
    "real-L45c3": (
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
        ),
    "real-L11c9": (
        127, 310, 144, 63, 61, 39, 133, 14, 73, 6, 11, 99, 60, 59, 55, 71, 32, 120, 36, 63, 82, 68, 37, 58, 2, 56, 83,
        20, 0, 207, 29, 108, 93, 74, 65, 29, 85, 50, 189, 29, 91, 22, 20, 72, 83, 73, 187, 78, 74, 126, 32, 4, 118, 36,
        74, 137, 71, 110, 67, 124, 189, 88, 72, 21, 111, 31, 98, 47, 113, 104, 51, 159, 66, 309, 167, 29, 24, 110, 23,
        107, 18, 188, 70, 117, 54, 49, 85, 60, 73, 76, 272, 92, 81, 41, 53, 2, 104, 192, 75, 127, 131, 121, 56, 68, 177,
        138, 30, 70, 122, 49, 87, 192, 66, 302, 46, 1, 10, 295, 180, 29, 44, 26, 3, 86, 270, 50, 32, 110, 186, 5, 7,
        131, 66, 95, 82, 140, 155, 45, 91, 69, 340, 367, 74, 162, 62, 35, 157, 5, 53, 19, 20, 31, 92, 96, 54, 19, 322,
        31, 44, 105, 680, 72, 76, 43, 50, 48, 90, 18, 21, 244, 95, 113, 178, 72, 7, 102, 94, 148, 53, 151, 47, 109, 61,
        335, 106, 9, 150, 69, 10, 1, 63, 77, 116, 56, 76, 19, 213, 136, 77, 108, 78, 113, 59, 251, 85, 3458, 136, 50,
        80, 67, 93, 56, 87, 121, 116, 151, 50, 196, 79, 498, 1, 49, 243, 79, 320, 151, 373, 146, 130, 31, 135, 50, 117,
        90, 57, 95, 85, 234, 144, 27, 39, 75, 108, 101, 3, 6, 63, 36, 49, 32, 69, 114, 93, 44, 78, 50, 25, 51, 90, 144,
        134, 161, 3, 59, 81, 75, 353, 182, 71, 69, 81, 20, 36, 38, 196, 25, 43, 469, 47, 150, 73, 112, 163, 106, 30, 10,
        21, 132, 64, 45, 109, 90, 475, 80, 815, 175, 119, 34, 25, 4, 37, 54, 46, 119, 288, 28, 71, 66, 207, 563, 119,
        31, 84, 70, 17, 58, 109, 51, 116, 45, 70, 57, 89, 5, 10, 120, 95, 136, 58, 86, 39, 147, 27, 384, 104, 46, 34,
        60, 64, 136, 69, 235, 165, 82, 69, 169, 251, 89, 50, 66, 9, 17, 73, 43, 91, 93, 132, 94, 168, 64, 38, 54, 0,
        138, 86, 88, 57, 10, 8, 17, 122, 70, 86, 118, 114, 87, 59, 58, 243, 114, 43, 22, 112, 184
    ),
}
# fmt: on
# The real cells Kimi K2.7 runs on (8, 4), named by MoE layer and chunk.
_REAL_CELLS = tuple(_KIMI_K27_REAL_COUNTS)
# Timed replays per program; the mean wall time per replay is reported.
_PERF_ITERS = 5


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
        # The package conftest skips a mesh that is not the whole Blackhole system: a smaller mesh's fabric
        # cannot come up alone, since every router must handshake with a live partner.
        marks = pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=topo)
        for model_id in _MODELS:
            real_cells = _REAL_CELLS if mesh == _FULL_MESH and model_id == "kimi-k27" else ()
            for threshold_id in ("balanced", "hot-expert") + real_cells:
                params.append(
                    pytest.param(
                        mesh,
                        _device_params(fabric_cfg),
                        threshold_id,
                        model_id,
                        None,  # variant
                        marks=marks,
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
    if threshold_id in _REAL_CELLS:
        assert tuple(mesh_device.shape) == _FULL_MESH, "a real cell is whole-mesh routing"
        assert model_id == "kimi-k27", "only Kimi K2.7 has real cells"
        indices = _spread_indices_from_counts(
            _KIMI_K27_REAL_COUNTS[threshold_id], dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok
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
        # this call is combine alone.
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

    # The overlap's fwd_arrived, final_arrived and expert_go outlive every launch -- neighbouring chips bump them
    # across launches -- so the case keeps one set for all of its calls, as a model keeps one per mesh. Created on
    # the overlap's first call, not here: the solo routed expert's arena is the whole L1 bank, so it only fits
    # while nothing else is allocated, and the perf test runs it first. That first call is eager, never a capture.
    overlap_semaphores = []

    def overlapped():
        if not overlap_semaphores:
            grid = mesh_device.compute_with_storage_grid_size()
            all_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            )
            overlap_semaphores.extend(ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3))
            ttnn.synchronize_device(mesh_device)
        fwd_arrived, final_arrived, expert_go = overlap_semaphores
        return routed_expert(
            dispatched_metadata=tt_metadata,
            expert_offsets=tt_expert_offsets,
            replicated_global_expert_idx_table=tt_idx_full,
            combine_axis=sp_axis,
            combine_num_links=num_links,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
            fwd_arrived_semaphore=fwd_arrived,
            final_arrived_semaphore=final_arrived,
            expert_go_semaphore=expert_go,
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
    # Twice: the second run is a program-cache hit over a fresh arena and the case's semaphores.
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

    Every program is replayed from a trace, as the model runs it, and timed as wall time per replay. One eager
    call of each first compiles it: compiling writes to the device, which a capture does not allow.

    The routed expert comes first: its first call sizes its arena to all free L1, and combine's ring semaphores
    take a piece of L1 from its own first call on. The overlap's arena is per call too, so it comes last only
    to match how the model orders them.
    """
    if variant == "8x1-submesh":
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(8, 1))
    case = _build_case(mesh_device, device_params, threshold_id, model_id, dg0_only=variant == "dg0-only")

    case.solo_routed_expert()
    re_trace, traced_re_out = _capture(mesh_device, case.solo_routed_expert)
    try:
        traced_re_us = _replay_us(mesh_device, re_trace, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, re_trace)

    # The widening sits outside the combine trace, so the trace is combine alone.
    widened = ttnn.typecast(traced_re_out, ttnn.bfloat16)
    case.combine(widened, widen=False)
    combine_trace, combine_out = _capture(mesh_device, lambda: case.combine(widened, widen=False))
    try:
        traced_combine_us = _replay_us(mesh_device, combine_trace, _PERF_ITERS)
    finally:
        ttnn.release_trace(mesh_device, combine_trace)
        del traced_re_out, widened, combine_out

    case.overlapped()
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
