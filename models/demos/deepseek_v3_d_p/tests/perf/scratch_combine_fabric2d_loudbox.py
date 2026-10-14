# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: combine_fabric2d device-perf worker on the 8x1 TorusY LoudBox proxy (one dispatch group).

Same geometry as test_prefill_combine_fabric2d.py (DeepSeek: 5120-token chunk, emb 7168, 256 experts,
top-8). With one group every pick comes back, so routing is uniform in-group or hot-skewed in-group
(the production hot weight on half the chips). Run with
TT_MESH_GRAPH_DESC_PATH=.../flat_combine_tools/p150_x8_ring_8x1.textproto so TorusY maps as a ring.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_combine_fabric2d import (
    PRODUCTION_ROUTING,
    _draw_indices,
    _Fixture,
)
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config

CHUNK = 5 * 1024
SEQ_LEN_PER_CHIP = CHUNK // 8
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
@pytest.mark.parametrize("emb_dim", [7168], ids=lambda e: f"emb{e}")
# v2 = combine_fabric2d (store-and-forward); v1 = deepseek_prefill.combine (fabric multi-hop), with and
# without zero-initialising the output (the model passes init_zeros=False, the v1 perf gate True).
@pytest.mark.parametrize("op", ["v2", "v1", "v1z"])
@pytest.mark.timeout(900)
def test_combine_fabric2d_loudbox_perf(mesh_device, device_params, num_links, routing, layout, emb_dim, op):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert (H, G) == (8, 1), (H, G)

    torch.manual_seed(42)
    if routing == "uniform":
        draw = None
    else:
        draw = _draw_indices(
            G, H, SEQ_LEN_PER_CHIP, NUM_EXPERTS_PER_TOK, NUM_ROUTED_EXPERTS, 1.0, PRODUCTION_ROUTING[1]
        )
    fx = _Fixture(
        mesh_device,
        H,
        G,
        emb_dim=emb_dim,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        topk=NUM_EXPERTS_PER_TOK,
        seed=42,
        routing=draw,
        capacity="exact",
    )
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT

    def run():
        if op == "v2":
            return fx.run(sp_axis, num_links, layout=tt_layout)
        return ttnn.experimental.deepseek_prefill.combine(
            fx.tt_payload(tt_layout),
            fx.tt_meta,
            fx.tt_counts,
            fx.tt_region,
            dispatch_group_size=H,
            experts_per_chip=fx.experts_per_chip,
            num_experts_per_tok=fx.topk,
            seq_len_per_chip=fx.seq_len_per_chip,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=op == "v1z",
        )

    by_d = fx.picks_by_distance()
    logger.info(
        f"loudbox combine perf: op={op} routing={routing} layout={layout} emb={emb_dim} capacity={fx.capacity} "
        f"links={num_links} picks_by_distance={by_d}"
    )

    out = run()  # warm-up
    ttnn.synchronize_device(mesh_device)
    fx.check(out, f"{op}/{routing}/{layout}")

    signpost(f"combine-{op}-{routing}-{layout}")
    for _ in range(ITERATIONS):
        run()
        ttnn.synchronize_device(mesh_device)
    signpost("done")

    # Relaunch on a different payload, so output a previous launch left behind cannot pass the check.
    fx._tt_payload = {}
    fx.payload = torch.roll(fx.payload, shifts=1, dims=2)
    out = run()
    ttnn.synchronize_device(mesh_device)
    fx.check(out, f"{op}/{routing}/{layout} relaunch")
