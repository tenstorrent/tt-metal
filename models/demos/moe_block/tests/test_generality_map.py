# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only generality map of MoEBlock (tier: quick): every model x rows a mesh row x mesh, on a mock cluster - does
the block build and enqueue a forward, and what it picked (head fused / ops with the engine layout, chunks and the
router's placement, tail fused / ops, and the reason when a fused stage does not fit). A case that does not build
xfails with its first error line; every case appends one JSON line to generated/moe_block/gmap.jsonl.

Run one mock cluster a process (TT_METAL_MOCK_CLUSTER_DESC_PATH; the cases of the other meshes skip):
  LoudBox meshes: .../cluster_descriptor_examples/blackhole_8xP150.yaml (2x4 line); the 1x4 ring also needs
    TT_MESH_GRAPH_DESC_PATH=.../flat_combine_tools/p150_x4_ring_1x4.textproto and blackhole_4xP150.yaml
  Galaxy meshes: .../blackhole_galaxy.yaml with TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1 (8x4, 4x4)."""

import dataclasses
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.moe_block.moe_block import (
    DSV41_FLASH,
    GLM_53,
    GLM_53_FLASH,
    KIMI_K3,
    KIMI_K27,
    MINIMAX_M3,
    MoEBlock,
    WeightCache,
)
from models.demos.moe_block.tests.test_moe_block import random_weights

MAP_MODELS = {
    "k3": KIMI_K3,
    "glm53_flash": GLM_53_FLASH,
    "glm53": GLM_53,
    "m3": MINIMAX_M3,
    "dsv41_flash": DSV41_FLASH,
    "k27": KIMI_K27,
}
ROWS = (128, 256, 320, 512, 640, 768, 1024, 1280, 2048)  # (tokens a mesh row)


@dataclasses.dataclass(frozen=True)
class MockMesh:
    shape: tuple
    fabric: ttnn.FabricConfig
    cluster: str  # (the mock cluster descriptor it needs)
    mgd: str = ""  # (a mesh graph descriptor it needs, if any)


MESHES = {
    "2x4": MockMesh((2, 4), ttnn.FabricConfig.FABRIC_2D, "blackhole_8xP150"),
    "1x4ring": MockMesh((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_X, "blackhole_4xP150", "p150_x4_ring_1x4"),
    "4x4": MockMesh((4, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_X, "blackhole_galaxy"),
    "8x4": MockMesh((8, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "blackhole_galaxy"),
}


def _cases():
    out = []
    for mn in MAP_MODELS:
        for rows in (640,) if mn == "k27" else ROWS:  # (K2.7: chunk 5120 on the 8x4 only - being deprecated)
            for mesh, mm in MESHES.items():
                if mn == "k27" and mesh != "8x4":
                    continue
                out.append(pytest.param(mn, rows, mesh, id=f"{mn}-r{rows}-{mesh}"))
    return out


def _active(mm: MockMesh):
    cl = os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")
    mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")
    return cl.endswith(mm.cluster + ".yaml") and (mgd.endswith(mm.mgd + ".textproto") if mm.mgd else not mgd)


@pytest.mark.parametrize("model, rows, mesh", _cases())
@pytest.mark.timeout(900)
def test_generality_map(model, rows, mesh):
    from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

    from conftest import set_fabric
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

    mm = MESHES[mesh]
    if not _active(mm):
        pytest.skip(f"needs the mock cluster {mm.cluster}" + (f" and the descriptor {mm.mgd}" if mm.mgd else ""))
    R, C = mm.shape
    m = MAP_MODELS[model]
    F = 1  # (more experts a chip than the flat op's 64: the router folds onto fewer physical experts, as on device)
    while m.num_experts // F // (R * C) > 64:
        F *= 2
    if F > 1:
        m = dataclasses.replace(m, num_experts=m.num_experts // F, router=dataclasses.replace(m.router, fold=F))
    chunk = rows * R
    params = fabric2d_device_params(fabric_payload_size=fabric_payload(m.routed_width))
    set_fabric(mm.fabric, params["reliability_mode"], fabric_router_config=params["fabric_router_config"])
    rec = dict(model=model, rows=rows, mesh=mesh, chunk=chunk, experts_per_chip=m.num_experts // (R * C), fold=F)
    t0 = time.perf_counter()
    mesh_dev = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=0)
    err = None
    try:
        block = MoEBlock(mesh_dev, m, chunk, random_weights(m), WeightCache(path=None, layer=0))
        rec["plan"] = block.summary
        rec["flat_layout"] = str(block.experts.layout_fields)
        if block.fused is not None:  # (the arenas' L1 a core, KB: how close a stage is to the bank)
            rec["head_arena_kb"] = block.fused.head.eng.arena_bytes >> 10
        if block.tail is not None:
            rec["tail_arena_kb"] = block.tail.eng.arena_bytes >> 10
            rec["tail_slots"] = f"{block.tail.eng.NSLOT}/{block.tail.eng.NSTMAX}"
        x = ttnn.from_torch(
            torch.zeros(R, C, rows, m.emb_dim // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_dev,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_dev, dims=(0, 1), mesh_shape=(R, C)),
        )
        ttnn.copy(x, block.input_buffer())
        out = block(block.input_buffer())  # (every program of the forward built and enqueued)
        assert tuple(out.shape)[-2:] == (rows, m.emb_dim // C)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - the map records what does not build
        import traceback

        tb = [f for f in traceback.extract_tb(e.__traceback__) if "/models/" in f.filename or "/bringup/" in f.filename]
        fr = (tb or traceback.extract_tb(e.__traceback__))[-1]
        where = f"{os.path.basename(fr.filename)}:{fr.lineno}"
        rec["where"] = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in tb[-6:][::-1])
        err = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300] if str(e).strip() else where}"
        rec["status"] = "fail"
        rec["error"] = err
    finally:
        ttnn.close_mesh_device(mesh_dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    os.makedirs("generated/moe_block", exist_ok=True)
    with open("generated/moe_block/gmap.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if err is not None:
        pytest.xfail(err)
