# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: MoEBlock on the mock Galaxy (8 x 4, both axes rings: FABRIC_2D_TORUS_XY) with head / tail "auto" at the
Galaxy's per-chip load (chunk 5120: 640 rows a mesh row) - the fused head (and K3's fused tail) chosen, and every
chip's programs built and enqueued through a forward (no execution, no numerics). Run with
TT_METAL_MOCK_CLUSTER_DESC_PATH=$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml
TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1 (the mock has no firmware to detect: the Galaxy's DRISC cores, as
the routed experts' layout uses them) (skipped otherwise)."""

import os

import pytest
import torch
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.moe_block.moe_block import MoEBlock, MoEOptions, WeightCache
from models.demos.moe_block.tests.test_moe_block import MODELS, random_weights

pytestmark = pytest.mark.skipif(
    "blackhole_galaxy" not in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")
    or os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES") != "1",
    reason="mock Galaxy with its DRAM cores programmable",
)
GALAXY_EXPERTS = {"glm53_flash": 288, "k3_448": 896}  # (the real models' experts: 9 / 28 a chip)


CASES = [  # (model, chunk: 5120 / 8192 = 640 / 1024 rows a mesh row, the router's placement)
    ("glm53_flash", 5120, "auto"),
    ("glm53_flash", 8192, "auto"),
    ("k3_448", 5120, "auto"),
    ("k3_448", 5120, "engine"),
    ("k3_448", 5120, "router_cores"),
    ("k3_448", 8192, "auto"),
]


@pytest.mark.parametrize("model, chunk, router", CASES, ids=[f"{m}-{c}-{r}" for m, c, r in CASES])
def test_block_mock_galaxy(model, chunk, router):
    import dataclasses

    from conftest import set_fabric
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

    m = dataclasses.replace(MODELS[model], num_experts=GALAXY_EXPERTS[model])
    params = fabric2d_device_params(fabric_payload_size=fabric_payload(m.routed_width))
    set_fabric(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        params["reliability_mode"],
        fabric_router_config=params["fabric_router_config"],
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=0)
    R, C = 8, 4
    try:
        block = MoEBlock(
            mesh, m, chunk, random_weights(m), WeightCache(path=None, layer=0), options=MoEOptions(router=router)
        )
        g = block.geometry
        rows = chunk // R
        assert g.tp_wraps and g.sp_wraps and g.tokens_per_row == rows
        assert block.head_mode == "fused", "auto: the fused head on the Galaxy torus"
        assert (block.tail is not None) == (m.latent_dim is not None), "auto: the fused tail for the latent model"
        x = ttnn.from_torch(
            torch.zeros(R, C, rows, m.emb_dim // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(R, C)),
        )
        ttnn.copy(x, block.input_buffer())
        out = block(block.input_buffer())  # (every program of the forward built and enqueued on every chip)
        assert tuple(out.shape)[-2:] == (rows, m.emb_dim // C)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
