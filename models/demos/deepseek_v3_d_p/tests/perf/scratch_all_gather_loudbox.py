# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: ttnn.bringup.fabric_all_gather on the 8x1 TorusY LoudBox proxy, measured like scratch_dispatch_loudbox.py
(profiler device kernel time, warm-up then ITERATIONS synced launches) so the two compare directly: the dense
all-gather of every chip's 640 tokens is the upper bound a sparse all-gather (dispatch with dedup) trims.
Run with TT_MESH_GRAPH_DESC_PATH=.../flat_combine_tools/p150_x8_ring_8x1.textproto.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config

SEQ_LEN_PER_CHIP = 5 * 1024 // 8
ITERATIONS = 10

_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-y-8x1-2link"]
assert len(_MESH_CONFIGS) == 1


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("emb", [7168, 4096], ids=lambda e: f"emb{e}")
@pytest.mark.parametrize("layout", ["rm", "tile"])
@pytest.mark.timeout(900)
def test_all_gather_loudbox_perf(mesh_device, device_params, num_links, emb, layout):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H = cfg.sp_axis, cfg.dispatch_group_size
    assert tuple(mesh_device.shape) == (8, 1)
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT

    torch.manual_seed(7)
    x = torch.randn(H, 1, SEQ_LEN_PER_CHIP, emb).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=tt_layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(H, 1), dims=(0, None)),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, H * SEQ_LEN_PER_CHIP, emb]), ttnn.bfloat16, tt_layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )

    def run():
        return ttnn.bringup.fabric_all_gather(
            tt_x,
            dim=2,
            output_tensor=out,
            cluster_axis=sp_axis,
            num_links=num_links,
        )  # no caller semaphores: the op makes and caches its own (this mesh reserves no L1_SMALL for them)

    run()  # warm-up
    ttnn.synchronize_device(mesh_device)
    want = x.reshape(H * SEQ_LEN_PER_CHIP, emb)
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).reshape(H * SEQ_LEN_PER_CHIP, emb)
        assert torch.equal(got, want), f"device {d}: gathered rows differ"
    logger.info(f"all_gather emb{emb}/{layout}: {H} chips x {SEQ_LEN_PER_CHIP} rows bit-exact")

    signpost(f"dispatch-ag-emb{emb}-{layout}")
    for _ in range(ITERATIONS):
        run()
        ttnn.synchronize_device(mesh_device)
    signpost("done")
