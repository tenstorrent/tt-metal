# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end dispatch+combine perf worker for one Galaxy column replayed on LB 8x1 TorusY.

Runs TtDispatchModule → production layout transform (squeeze → TILE+bfp8 →
unsqueeze) → TtCombineModule(init_zeros=True) in a single forward pass on
device. Tracy captures DispatchDeviceOperation and CombineDeviceOperation in
one CSV; the perf wrapper asserts each independently.

Required env vars (set by the parent perf test via extra_env):
    TT_DS_CAPTURED_LAYER          int, MoE layer index
    TT_DS_CAPTURED_COL            int, Galaxy column [0, 4)
    TT_DS_USE_CAPTURED_INDICES    path to expert_routing_dispatch_combine_perf.safetensors, the
                                  chunked-prefill capture holding every model's cases
                                  (required; there is no default capture)
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    load_captured_routing,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

# Geometry of the replay: LB 8x1 stands in for one 8-chip Galaxy column.
GALAXY_NUM_DISPATCH_GROUPS = 4
DISPATCH_GROUP_SIZE = 8
CHUNK = 5 * 1024  # tokens per chunk in the chunked-prefill run the captures come from
DISPATCH_BUFFER_CAPACITY_FACTOR = 8


# One entry per model whose chunked-prefill capture we replay; add a model by extending this list.
_CHUNK_MODELS = [("dsv3", DeepSeekV3Config), ("kimi26", KimiK26Config), ("glm52", GLM52Config)]
_TORUS_Y_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-y-8x1-2link"]
assert len(_TORUS_Y_MESH_CONFIGS) == 1, "LoudBox TorusY proxy config missing from ALL_MESH_CONFIGS"


# One chunk (5120 tokens) spread over the 8-chip dispatch group => seq_len_per_chip 640. Expert
# count / embedding size come off each reference config, so they live in one place;
# experts_per_chip = experts_per_col / 8 (dsv3 256/4/8 = 8, kimi26 384/4/8 = 12), and
# model selects the per-model capture file (expert_routing_dispatch_combine_perf_<model>.safetensors).
# The parametrize id is what test_dispatch_combine_perf selects with `-k "<id> and ..."`; since -k
# matches substrings, no id may be a prefix of another or it would silently pull in the wrong entry too.
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
            cfg.NUM_ROUTED_EXPERTS // GALAXY_NUM_DISPATCH_GROUPS // DISPATCH_GROUP_SIZE,
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
    layer_str = os.getenv("TT_DS_CAPTURED_LAYER")
    col_str = os.getenv("TT_DS_CAPTURED_COL")
    if layer_str is None or col_str is None:
        pytest.skip("Requires TT_DS_CAPTURED_LAYER and TT_DS_CAPTURED_COL env vars")

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    topology = per_axis_topology(device_params["fabric_config"])[sp_axis]
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    signpost(
        f"DispatchCombine layer={layer_str} col={col_str} mesh={tuple(mesh_device.shape)} "
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
        f"[dispatch_combine] config: layer={layer_str} col={col_str} "
        f"experts_per_chip={experts_per_chip} metadata_len={metadata_len} "
        f"max_dispatch_buffer_token_size={max_dispatch_buffer_token_size} "
        f"num_dispatch_groups(mesh)={num_dispatch_groups}"
    )

    # Load captured routing (in-col indices shifted to [0, experts_per_col), rest -> sentinel 255,
    # col-0 dispatch table).
    indices, expert_dispatch_table = load_captured_routing(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        layer=int(layer_str),
        col=int(col_str),
        model=model,
        captured_indices_path=os.getenv("TT_DS_USE_CAPTURED_INDICES"),
    )
    # get_gate_outputs produces Galaxy-global rows; keep the selected proxy column only.
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    expert_offsets = expert_offsets[0:1].contiguous()
    expert_token_counts = expert_token_counts[0:1].contiguous()
    expert_region_offsets = expert_region_offsets[0:1].contiguous()

    # Synthesize x (random bf16) and weights (zeros bf16). Values don't drive kernel
    # cycle count — only indices and offsets do.
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
        init_zeros=True,
    )

    # Dispatch → production layout transform → combine (mirrors tt_moe.py).
    dispatched_buffer, metadata = dispatch_module(tt_x, tt_weights, tt_indices, tt_offsets, tt_table)
    buf_2d = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
    buf_tiled = ttnn.to_layout(buf_2d, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    buf_for_combine = ttnn.unsqueeze(ttnn.unsqueeze(buf_tiled, dim=0), dim=0)
    _ = combine_module(buf_for_combine, metadata, tt_counts, tt_region_offsets)

    ttnn.synchronize_device(mesh_device)
    logger.info(f"[dispatch_combine] layer={layer_str} col={col_str} done")
