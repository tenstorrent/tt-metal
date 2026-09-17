# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Local B1 full-128-expert SP8/TP1 controlled dispatch/combine replay.
B1_ROUTING_CASE points to JSON from prepare_cases.py; B1_ITERATIONS defaults to10.
Isolated dispatch/combine experiment; excludes FFN/shared expert/reduce.
Router weights are unused by the current dispatch API; numerical PCC is not tested.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_y_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

DISPATCH_GROUP_SIZE = 8
CHUNK = 5 * 1024  # tokens per chunk in the chunked-prefill run the captures come from
DISPATCH_BUFFER_CAPACITY_FACTOR = 8


_CHUNK_MODELS = [("mistral4", MistralSmall4Config)]
_TORUS_Y_MESH_CONFIGS = [
    pytest.param(
        (8, 1),
        torus_y_device_params(fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
        id="fabric2d-torus-y-8x1-2link",
    )
]


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, "
    "dispatch_buffer_capacity_factor, experts_per_chip_override, model",
    [
        pytest.param(
            CHUNK // DISPATCH_GROUP_SIZE,
            cfg.EMB_SIZE,
            cfg.NUM_ROUTED_EXPERTS,
            cfg.NUM_EXPERTS_PER_TOKEN,
            DISPATCH_BUFFER_CAPACITY_FACTOR,
            cfg.NUM_ROUTED_EXPERTS // DISPATCH_GROUP_SIZE,
            model,
            id=f"perf_captured_{model}_chunk",
        )
        for model, cfg in _CHUNK_MODELS
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _TORUS_Y_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_ttnn_dispatch_combine(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    experts_per_chip_override,
    model,
):
    case = json.loads(Path(os.environ["B1_ROUTING_CASE"]).read_text())
    layer_str = str(case["layer"])
    case_label = "full128-" + case["mode"]
    assert tuple(mesh_device.shape) == (8, 1)

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    topology = per_axis_topology(device_params["fabric_config"])[sp_axis]
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    signpost(
        f"DispatchCombine layer={layer_str} case={case_label} mesh={tuple(mesh_device.shape)} "
        f"num_links={num_links} topology={topology}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        _,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
        experts_per_chip_override=experts_per_chip_override,
    )
    logger.info(
        f"[dispatch_combine] config: layer={layer_str} case={case_label} "
        f"experts_per_chip={experts_per_chip} metadata_len={metadata_len} "
        f"max_dispatch_buffer_token_size={max_dispatch_buffer_token_size} "
        f"num_dispatch_groups(mesh)={num_dispatch_groups}"
    )

    indices = torch.tensor(case["indices"], dtype=torch.int32).view(8, 640, 4)
    assert indices.min() >= 0 and indices.max() < 128
    assert experts_per_chip == 16
    expert_dispatch_table = ExpertMapping.create_dispatch_table(128, 8, 1)
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    assert expert_offsets.shape == (1, 8, 128)
    assert expert_token_counts.shape == (1, 8, 128)

    x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    weights = torch.zeros(dispatch_group_size, seq_len_per_chip, num_experts_per_tok, dtype=torch.bfloat16)

    input_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x, mesh_mapper=input_mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=input_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices.to(torch.int16),
        mesh_mapper=input_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )

    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    tt_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)

    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    tt_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=counts_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_region_offsets = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=counts_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Diagnostic: use default manager; dispatch still selects the same first-row cores.
    dispatch_module = TtDispatchModule(
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
    )
    combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=False,
    )

    for iteration in range(int(os.getenv("B1_ITERATIONS", "10"))):
        signpost(f"B1_START layer={layer_str} mode={case['mode']} iteration={iteration}")
        dispatched_buffer, metadata = dispatch_module(tt_x, tt_weights, tt_indices, tt_offsets, tt_table)
        buf_2d = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
        buf_tiled = ttnn.to_layout(buf_2d, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        buf_for_combine = ttnn.unsqueeze(ttnn.unsqueeze(buf_tiled, dim=0), dim=0)
        result = combine_module(buf_for_combine, metadata, tt_counts, tt_region_offsets)
        ttnn.synchronize_device(mesh_device)
        signpost(f"B1_END layer={layer_str} mode={case['mode']} iteration={iteration}")
        del result, buf_for_combine, buf_tiled, buf_2d, dispatched_buffer, metadata
    logger.info(f"[B1 dispatch_combine] layer={layer_str} mode={case['mode']} done")
