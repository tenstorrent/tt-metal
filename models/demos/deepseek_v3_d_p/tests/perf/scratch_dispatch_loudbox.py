# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: dispatch v1 (deepseek_prefill.dispatch, fabric multi-hop) vs v2 (dispatch_fabric2d, store-and-forward)
on the 8x1 TorusY LoudBox proxy, identical inputs, both checked byte-exact against dispatch_fabric2d's reference.

Same geometry and routing as scratch_combine_fabric2d_loudbox.py: DeepSeek 640 tok/chip, emb 7168, 256 experts,
top-8, one dispatch group (every pick in-group), uniform or hot (2x weight on chips 4-7).
Run with TT_MESH_GRAPH_DESC_PATH=.../flat_combine_tools/p150_x8_ring_8x1.textproto.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import (
    PRODUCTION_ROUTING,
    _draw_indices,
    _Fixture,
)
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config

SEQ_LEN_PER_CHIP = 5 * 1024 // 8
EMB_DIM = 7 * 1024
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
ITERATIONS = 10

_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-y-8x1-2link"]
assert len(_MESH_CONFIGS) == 1


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("routing", ["uniform", "hot"])
@pytest.mark.parametrize("layout", ["tile", "rm"])
@pytest.mark.parametrize("op", ["v2", "v1"])
@pytest.mark.timeout(900)
def test_dispatch_loudbox_perf(mesh_device, device_params, num_links, routing, layout, op):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert (H, G) == (8, 1), (H, G)

    fx = _Fixture(
        mesh_device, H, G, emb_dim=EMB_DIM, num_routed_experts=NUM_ROUTED_EXPERTS, topk=NUM_EXPERTS_PER_TOK, seed=42
    )
    if routing == "hot":
        torch.manual_seed(42)
        fx.indices = _draw_indices(
            G, H, SEQ_LEN_PER_CHIP, NUM_EXPERTS_PER_TOK, NUM_ROUTED_EXPERTS, 1.0, PRODUCTION_ROUTING[1]
        )
        fx.rebuild()
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    # v1 takes each source chip's own offsets row; v2 takes every source chip's, replicated.
    tt_offs_own = fx._shard(fx.offs.permute(1, 0, 2).contiguous(), (0, 1), ttnn.int32)

    def run():
        if op == "v2":
            return fx.run(sp_axis, num_links, layout=tt_layout)
        return ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=fx.tt_x[tt_layout],
            indices_tensor=fx.tt_idx,
            expert_offsets_tensor=tt_offs_own,
            expert_dispatch_table_tensor=fx.tt_table,
            dispatch_group_size=H,
            experts_per_chip=fx.experts_per_chip,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            metadata_len=3,
            max_dispatch_buffer_token_size=fx.capacity,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
        )

    per_chip = torch.bincount(fx.table[0, fx.indices.reshape(-1)].clamp(min=0), minlength=H).tolist()
    logger.info(
        f"dispatch perf: op={op} routing={routing} layout={layout} capacity={fx.capacity} picks_per_chip={per_chip}"
    )

    payload, metadata = run()[:2]  # warm-up
    ttnn.synchronize_device(mesh_device)
    fx.check(payload, metadata, f"{op}/{routing}/{layout}")

    signpost(f"dispatch-{op}-{routing}-{layout}")
    for _ in range(ITERATIONS):
        run()
        ttnn.synchronize_device(mesh_device)
    signpost("done")
