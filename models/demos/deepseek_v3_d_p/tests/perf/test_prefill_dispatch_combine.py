# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end dispatch+combine perf worker for one Galaxy column replayed on LB 8x1.

Runs TtDispatchModule → production layout transform (squeeze → TILE+bfp8 →
unsqueeze) → TtCombineModule in a single forward pass on device. The Tracy
profiler captures DispatchDeviceOperation, the layout op(s), and
CombineDeviceOperation in one CSV — the perf wrapper sums them via the
existing merge_device_rows logic.

This test exists separately from `test_prefill_dispatch.py` /
`test_prefill_combine.py` because those workers each run only one op against
host-computed reference inputs and don't apply the production layout
transform between them. With the Galaxy real-indices kernel config
(num_routed_experts=256, top-k=8, experts_per_chip=8 explicit), the
host-source dispatched_buffer + missing layout transform combination hangs
combine — running both ops on device with the production layout in between
is the validated replay flow from the nmilicevic/ds-glx-lb-measure branch.

Required env vars (set by the parent perf test via `_perf_param`):
    TT_DS_CAPTURED_LAYER          int, MoE layer index
    TT_DS_CAPTURED_COL            int, Galaxy column [0, 4)
    TT_DS_USE_CAPTURED_INDICES    optional path override (defaults to LONGBOOK_QA_ENG_25600)
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
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


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, "
    "dispatch_buffer_capacity_factor, experts_per_chip_override",
    [pytest.param(3200, 7168, 256, 8, 8, 8, id="perf_real_indices")],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    ALL_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_ttnn_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
    experts_per_chip_override,
):
    layer_str = os.getenv("TT_DS_CAPTURED_LAYER")
    col_str = os.getenv("TT_DS_CAPTURED_COL")
    if layer_str is None or col_str is None:
        pytest.skip("Requires TT_DS_CAPTURED_LAYER and TT_DS_CAPTURED_COL env vars")

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
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
        max_dispatched_tokens_per_expert,
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

    # Load captured routing (indices remapped to [0, 64) ∪ {255}, col-0 dispatch table).
    indices, expert_dispatch_table = load_captured_routing(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Compute offsets/counts/region_offsets from indices + table. Slice to [0:1] because
    # get_gate_outputs derives num_dispatch_groups=4 from the formula (Galaxy-global);
    # LB 8x1 only has 1 dispatch group, and the col-0 table makes all 4 rows identical.
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
        indices, mesh_mapper=input_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
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
        num_dispatch_groups=1,  # LB has 1 physical dispatch group; we simulate one Galaxy col on it.
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,  # production setting; matches measure-branch replay
    )

    # Production layout (mirrors tt_moe.py:468-489): squeeze → TILE+bfp8 → unsqueeze.
    # The kernel-level dispatch buffer is row-major bf16 (~32MB on device); production
    # tilizes to bfp8 (~8MB) before feeding combine. Without this step combine sees a
    # row-major bf16 buffer and hangs at num_routed_experts=256 / experts_per_chip=8.
    dispatched_buffer, metadata = dispatch_module(tt_x, tt_weights, tt_indices, tt_offsets, tt_table)
    buf_2d = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
    buf_tiled = ttnn.to_layout(buf_2d, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    buf_for_combine = ttnn.unsqueeze(ttnn.unsqueeze(buf_tiled, dim=0), dim=0)
    _ = combine_module(buf_for_combine, metadata, tt_counts, tt_region_offsets)

    ttnn.synchronize_device(mesh_device)
    logger.info(f"[dispatch_combine] layer={layer_str} col={col_str} done")
