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

from types import SimpleNamespace

import statistics

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
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

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
# The same cells over the full mesh, for (8, 4): every expert, unscaled, since a whole chunk is exactly this
# test's 5120 tokens x top-8 = 40960 routings. Across the full mesh no captured cell leaves every expert under
# the threshold, so balanced here still lifts a few experts into the unified pass (Kimi 9, GLM 5).
_CAPTURED_COUNTS_FULL_MESH = {
    "kimi-k27": {
        "balanced": (
            # MoE layer 34, chunk 1
            249, 75, 99, 69, 83, 31, 53, 65, 9, 71, 77, 108, 95, 190, 36, 75, 93, 67, 101, 33, 250, 129, 52, 39, 106,
            69, 42, 0, 32, 143, 106, 218, 84, 102, 116, 202, 108, 91, 162, 46, 139, 16, 2, 259, 211, 93, 95, 24, 219,
            20, 191, 88, 125, 143, 160, 55, 127, 111, 15, 29, 46, 65, 31, 177, 71, 122, 192, 61, 34, 101, 142, 34, 137,
            37, 146, 101, 25, 77, 78, 2, 56, 82, 2, 88, 59, 40, 125, 253, 93, 24, 216, 141, 47, 172, 77, 8, 225, 50, 37,
            34, 6, 33, 128, 140, 87, 120, 89, 175, 174, 178, 116, 55, 76, 70, 41, 84, 38, 174, 115, 39, 1, 0, 33, 107,
            136, 89, 72, 309, 93, 61, 72, 83, 97, 135, 58, 50, 94, 83, 67, 69, 285, 118, 91, 91, 42, 86, 123, 89, 98,
            247, 99, 85, 62, 108, 191, 577, 136, 80, 122, 251, 97, 62, 69, 149, 50, 348, 53, 131, 32, 52, 41, 85, 28,
            122, 70, 12, 132, 18, 82, 195, 73, 30, 82, 162, 184, 78, 37, 97, 26, 123, 357, 102, 117, 153, 54, 94, 90,
            58, 78, 183, 108, 37, 108, 199, 15, 90, 111, 41, 85, 225, 28, 19, 57, 101, 64, 151, 30, 108, 61, 128, 55,
            64, 17, 56, 145, 914, 103, 195, 78, 70, 76, 209, 183, 65, 43, 104, 20, 75, 130, 99, 1, 155, 30, 68, 183, 73,
            151, 106, 253, 45, 21, 75, 11, 13, 32, 62, 228, 92, 99, 148, 140, 19, 25, 66, 63, 358, 59, 134, 203, 159,
            43, 17, 36, 133, 101, 100, 29, 612, 118, 76, 24, 94, 63, 173, 155, 150, 42, 112, 85, 158, 79, 94, 46, 53,
            46, 40, 32, 91, 118, 186, 34, 192, 193, 81, 95, 79, 94, 89, 157, 250, 93, 101, 17, 89, 35, 146, 123, 83,
            170, 83, 20, 221, 855, 83, 160, 131, 56, 28, 73, 141, 81, 180, 126, 21, 47, 99, 230, 89, 109, 95, 167, 105,
            87, 76, 284, 51, 151, 235, 3, 106, 25, 49, 130, 31, 103, 31, 43, 128, 51, 281, 14, 340, 8, 227, 381, 142,
            109, 82, 164, 114, 164, 158, 128, 12, 234, 84, 77, 112, 50, 55, 117, 216, 92, 9
        ),
        "hot-expert": (
            # MoE layer 5, chunk 0
            7, 91, 72, 85, 17, 184, 89, 45, 157, 4, 44, 56, 30, 42, 61, 76, 21, 90, 30, 36, 30, 143, 51, 38, 27, 96, 33,
            3, 0, 48, 1, 33, 2399, 3, 90, 44, 238, 48, 77, 81, 40, 121, 57, 58, 56, 110, 29, 284, 34, 14, 44, 240, 108,
            55, 41, 192, 43, 5, 39, 136, 5, 35, 14, 54, 47, 5, 2, 82, 115, 13, 36, 26, 93, 41, 92, 78, 71, 37, 101, 117,
            63, 103, 54, 129, 151, 91, 50, 24, 32, 13, 44, 13, 116, 45, 68, 48, 77, 569, 71, 52, 22, 36, 39, 9, 130, 43,
            9, 78, 70, 43, 2532, 44, 77, 154, 1, 186, 41, 38, 14, 173, 243, 232, 63, 40, 1224, 45, 40, 16, 189, 57, 19,
            61, 52, 48, 243, 209, 57, 43, 58, 109, 150, 247, 3, 8, 99, 23, 18, 0, 26, 148, 73, 209, 83, 92, 60, 3, 77,
            5, 29, 73, 260, 81, 26, 23, 120, 70, 294, 93, 116, 28, 168, 149, 72, 22, 90, 46, 241, 112, 127, 72, 46, 408,
            55, 53, 242, 116, 42, 7, 12, 111, 122, 65, 1436, 110, 35, 90, 41, 57, 56, 15, 133, 65, 54, 179, 13, 45, 56,
            74, 12, 120, 1, 175, 103, 419, 153, 56, 261, 89, 83, 49, 16, 144, 139, 84, 71, 154, 142, 122, 20, 14, 103,
            413, 47, 82, 47, 175, 15, 133, 84, 171, 30, 70, 129, 95, 50, 91, 15, 111, 37, 4, 8, 2, 77, 50, 312, 46, 82,
            196, 0, 62, 15, 25, 187, 85, 98, 12, 44, 24, 53, 130, 81, 351, 289, 90, 111, 90, 124, 299, 193, 3, 424, 35,
            592, 65, 45, 65, 35, 300, 167, 68, 53, 37, 1, 51, 29, 39, 75, 16, 1, 292, 83, 67, 0, 116, 134, 62, 97, 32,
            45, 94, 172, 47, 63, 34, 0, 54, 14, 44, 64, 288, 30, 14, 94, 31, 74, 31, 24, 21, 45, 58, 103, 104, 120, 30,
            130, 50, 3, 21, 143, 116, 234, 74, 28, 39, 47, 19, 66, 32, 280, 86, 84, 277, 3, 10, 35, 44, 38, 105, 103,
            73, 64, 145, 16, 51, 123, 75, 56, 1, 122, 77, 88, 54, 1596, 112, 70, 84, 10, 81, 25, 28, 60, 23, 166, 82
        ),
    },
    "glm-53": {
        "balanced": (
            # MoE layer 3, chunk 5
            204, 173, 187, 187, 221, 270, 142, 107, 261, 203, 183, 69, 211, 178, 186, 115, 63, 219, 81, 135, 134, 158,
            240, 96, 185, 151, 155, 206, 96, 188, 131, 187, 225, 88, 200, 140, 198, 134, 213, 165, 119, 46, 163, 106,
            128, 157, 152, 301, 174, 105, 192, 179, 212, 125, 259, 112, 186, 233, 216, 198, 69, 196, 106, 234, 172, 146,
            159, 183, 160, 129, 222, 205, 368, 148, 141, 220, 118, 34, 54, 104, 129, 245, 168, 202, 101, 145, 85, 76,
            180, 70, 142, 242, 180, 333, 190, 131, 193, 130, 233, 221, 189, 157, 182, 121, 148, 245, 185, 59, 196, 100,
            80, 273, 154, 171, 69, 134, 200, 167, 167, 119, 149, 234, 304, 186, 84, 96, 166, 197, 138, 176, 137, 88,
            131, 94, 187, 133, 155, 109, 158, 164, 157, 107, 300, 235, 153, 91, 102, 19, 121, 93, 115, 146, 171, 83,
            199, 164, 224, 131, 252, 117, 153, 90, 98, 93, 218, 127, 123, 386, 268, 118, 182, 177, 269, 155, 148, 167,
            137, 184, 111, 93, 63, 419, 163, 173, 74, 144, 132, 203, 222, 186, 161, 192, 43, 132, 170, 212, 148, 142,
            122, 144, 183, 167, 228, 168, 84, 166, 171, 131, 195, 131, 85, 235, 108, 123, 182, 148, 175, 249, 183, 113,
            80, 126, 137, 83, 212, 82, 83, 126, 153, 113, 201, 202, 200, 75, 61, 155, 269, 83, 84, 176, 97, 159, 184,
            354, 133, 44, 49, 154, 190, 75, 267, 283, 138, 255, 237, 79
        ),
        "hot-expert": (
            # MoE layer 21, chunk 9
            166, 64, 102, 113, 170, 136, 245, 398, 29, 63, 147, 66, 174, 91, 176, 103, 59, 3, 163, 131, 36, 208, 161,
            70, 168, 26, 0, 38, 79, 97, 202, 165, 4, 168, 26, 257, 147, 3275, 300, 61, 185, 133, 103, 56, 16, 207, 254,
            478, 130, 89, 211, 141, 171, 238, 38, 136, 5, 201, 63, 33, 77, 76, 350, 173, 26, 240, 131, 71, 49, 9, 42,
            147, 86, 85, 235, 70, 144, 87, 11, 55, 66, 63, 202, 10, 67, 222, 16, 207, 85, 258, 31, 66, 105, 104, 282, 2,
            104, 80, 51, 256, 30, 136, 358, 124, 291, 161, 106, 94, 69, 143, 155, 83, 116, 83, 426, 53, 176, 49, 49, 47,
            181, 192, 74, 98, 165, 168, 271, 126, 250, 122, 272, 35, 209, 79, 187, 8, 324, 105, 274, 89, 59, 131, 59,
            611, 714, 77, 39, 120, 121, 257, 154, 223, 4, 71, 91, 116, 325, 148, 57, 479, 138, 153, 52, 157, 85, 276,
            35, 151, 15, 10, 71, 30, 23, 51, 1, 620, 65, 122, 41, 122, 77, 16, 327, 70, 181, 82, 23, 147, 107, 171, 258,
            137, 154, 75, 15, 16, 64, 29, 1133, 12, 940, 0, 4, 264, 50, 86, 163, 529, 147, 228, 148, 78, 2, 40, 207,
            150, 65, 14, 304, 289, 94, 211, 169, 16, 724, 320, 29, 132, 118, 435, 22, 180, 540, 609, 106, 231, 216, 394,
            217, 114, 45, 42, 177, 84, 606, 72, 130, 75, 78, 105, 102, 83, 75, 15, 88, 36
        ),
    },
}
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


def _mesh_params():
    params = []
    for mesh, fabric_cfg in _MESHES.items():
        topo = "ring" if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_Y else f"mesh-{mesh[0]}x{mesh[1]}"
        for model_id in _MODELS:
            for threshold_id in ("balanced", "hot-expert"):
                params.append(
                    pytest.param(
                        mesh,
                        fabric_to_device_params(fabric_cfg),
                        threshold_id,
                        model_id,
                        marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=topo),
                        id=f"{model_id}-{mesh[0]}x{mesh[1]}-{threshold_id}",
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


def _build_case(mesh_device, device_params, threshold_id, model_id):
    """One seq-640 layer of `model_id` on `mesh_device`: its inputs, and the three ways to run it."""
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
    captured = _CAPTURED_COUNTS_FULL_MESH if tuple(mesh_device.shape) == _FULL_MESH else _CAPTURED_COUNTS
    counts_in = captured[model_id][threshold_id]
    assert len(counts_in) == num_routed_experts, f"{len(counts_in)} captured counts for {num_routed_experts} experts"
    indices = _indices_from_counts(counts_in, dispatch_group_size, _SEQ_LEN_PER_CHIP, num_experts_per_tok)
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
    tt_idx_full = _int_tensor(idx_table, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device), dtype=ttnn.uint32)

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

    def combine(re_output):
        # The standalone op takes bfloat16 only; bfloat8_b widens exactly, so this is the bytes the
        # overlap's untilizers produce.
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
    "mesh_device, device_params, threshold_id, model_id", _mesh_params(), indirect=["mesh_device", "device_params"]
)
def test_hybrid_routed_expert_combine_overlap(mesh_device, device_params, threshold_id, model_id):
    case = _build_case(mesh_device, device_params, threshold_id, model_id)
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
    """Median device time, slowest chip, of the one program per run that `is_target` picks out."""

    def run_all():
        return [run_fn() for _ in range(iters)]

    outputs, per_program = profile_realtime_program_merged(mesh_device, run_all)
    matched = [e["duration_ns"] for e in per_program.values() if is_target(e["kernel_sources"])]
    # The warm-up run's record can be delivered after the window opens; records arrive in dispatch order,
    # so the measured runs are the last `iters`.
    assert len(matched) >= iters, f"{label}: expected {iters} programs, the profiler matched {len(matched)}"
    return outputs, statistics.median(matched[-iters:])


def _has(sources, path):
    return any(path in source.replace("\\", "/") for source in sources)


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id, model_id", _mesh_params(), indirect=["mesh_device", "device_params"]
)
def test_hybrid_routed_expert_combine_overlap_perf(mesh_device, device_params, threshold_id, model_id):
    """The overlapped program against the same two ops back to back: hybrid routed expert, then
    combine_fabric2d. Sequential is the sum of the two programs' device times, so it assumes no gap
    between them and is the best a sequential dispatch can do; the overlap must beat it.

    Every solo routed-expert run comes first: its per-call arena takes all free L1, which combine's
    fwd_arrived holds a piece of from its first call on. One warm-up run of each program fills the
    program cache outside the measured window.
    """
    require_realtime_profiler("the RE + combine overlap perf test")
    case = _build_case(mesh_device, device_params, threshold_id, model_id)

    warm_re = case.solo_routed_expert()
    re_outputs, re_ns = _median_program_ns(
        mesh_device,
        case.solo_routed_expert,
        _PERF_ITERS,
        lambda k: _has(k, _SOLO_RE_KERNELS) and not _has(k, _OVERLAP_KERNELS),
        "solo routed expert",
    )

    case.combine(warm_re)
    re_iter = iter(re_outputs)
    _, combine_ns = _median_program_ns(
        mesh_device,
        lambda: case.combine(next(re_iter)),
        _PERF_ITERS,
        lambda k: _has(k, _COMBINE_KERNELS) and not _has(k, _OVERLAP_KERNELS),
        "combine_fabric2d",
    )

    case.overlapped()
    _, overlap_ns = _median_program_ns(
        mesh_device, case.overlapped, _PERF_ITERS, lambda k: _has(k, _OVERLAP_KERNELS), "overlap"
    )

    sequential_ns = re_ns + combine_ns
    saved_ns = sequential_ns - overlap_ns
    logger.info(
        f"RE+combine {model_id} {threshold_id}: sequential {sequential_ns / 1e3:.1f} us (RE {re_ns / 1e3:.1f} + combine "
        f"{combine_ns / 1e3:.1f}), overlap {overlap_ns / 1e3:.1f} us -> {sequential_ns / overlap_ns:.3f}x, "
        f"saved {saved_ns / 1e3:.1f} us ({saved_ns / sequential_ns:.1%}); combine hidden "
        f"{min(saved_ns, combine_ns) / combine_ns:.0%}"
    )
    assert overlap_ns < sequential_ns, (
        f"overlap {overlap_ns / 1e3:.1f} us is not faster than RE then combine "
        f"({re_ns / 1e3:.1f} + {combine_ns / 1e3:.1f} = {sequential_ns / 1e3:.1f} us)"
    )
