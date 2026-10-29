# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE head's coverage: which (model, mesh, chunk, links, grid) it takes, and why not where it refuses.

test_fused_head_coverage (host only, no device, ~a minute): FusedHead.plan - the head's whole host plan, head_plan.fit
- for every model x mesh (8x4 torus, 4x4 with the TP or the SP axis a ring, 2x4 ring / line, 4x2, 1x4 ring / line, 1x8
ring / line) x chunk (a range, odd sizes too) x links (1, 2) x grid (12x10; the LoudBox meshes also 11x10, a stock
P150). Every case plans or refuses with a reason (the requirement, its numbers, what would fit); the summary table and
the reasons are logged, one JSON line a case in generated/moe_block/head_coverage.jsonl.

test_fused_head_mock_build (host only, a mock cluster; skipped without one): the plan's verdict against a real build
of the block on the mock mesh - a case the plan accepts must build the fused head, one it refuses runs the ops head.
One mock cluster a process (TT_METAL_MOCK_CLUSTER_DESC_PATH; the cases of other meshes skip):
  blackhole_galaxy.yaml (+ TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1): 8x4 torus, 4x4 TP ring, 4x4 SP ring,
    2x4 TP ring (a Galaxy submesh)
  blackhole_8xP150.yaml: 2x4 line
  blackhole_4xP150.yaml + TT_MESH_GRAPH_DESC_PATH .../flat_combine_tools/p150_x4_ring_1x4.textproto (or _line_): 1x4
    ring / line (an 11-column board)
HEAD_COV_MODELS / HEAD_COV_CHUNKS / HEAD_COV_LINKS (comma lists) narrow the mock cases."""

import collections
import dataclasses
import json
import os
import time

import pytest
from loguru import logger

from models.demos.moe_block.moe_block import (
    DSV41_FLASH,
    GLM_53,
    GLM_53_FLASH,
    KIMI_K3,
    KIMI_K27,
    MINIMAX_M3,
    FusedHead,
    Geometry,
    MoEDtypes,
)

COV_MODELS = {
    "k3": KIMI_K3,
    "k27": KIMI_K27,
    "glm53": GLM_53,
    "glm53_flash": GLM_53_FLASH,
    "dsv41_flash": DSV41_FLASH,
    "m3": MINIMAX_M3,
}


@dataclasses.dataclass(frozen=True)
class Mesh:
    R: int
    C: int
    tp_ring: bool
    sp_ring: bool
    grids: tuple = ((12, 10),)
    mock: str = ""  # (the mock cluster that builds it)
    mgd: str = ""  # (its mesh graph descriptor, if any)
    fabric: str = "FABRIC_2D"


LB = ((12, 10), (11, 10))  # (the LoudBox: 12x10 reported, a stock P150 11x10)
MESHES = {
    "8x4 torus": Mesh(8, 4, True, True, mock="blackhole_galaxy", fabric="FABRIC_2D_TORUS_XY"),
    "4x4 tp-ring": Mesh(4, 4, True, False, mock="blackhole_galaxy", fabric="FABRIC_2D_TORUS_X"),
    "4x4 sp-ring": Mesh(4, 4, False, True),  # (no mock: the Galaxy's Y ring is 8 chips - a 4x4's rows do not wrap)
    "2x4 ring": Mesh(2, 4, True, False, mock="blackhole_galaxy", fabric="FABRIC_2D_TORUS_X"),
    "2x4 line": Mesh(2, 4, False, False, LB, mock="blackhole_8xP150"),
    "4x2 line": Mesh(4, 2, False, False, LB),
    "1x4 ring": Mesh(1, 4, True, False, LB, "blackhole_4xP150", "p150_x4_ring_1x4", "FABRIC_2D_TORUS_X"),
    "1x4 line": Mesh(1, 4, False, False, LB, "blackhole_4xP150", "p150_x4_line_1x4"),
    "1x8 ring": Mesh(1, 8, True, False, LB),
    "1x8 line": Mesh(1, 8, False, False, LB),
}
CHUNKS = (256, 512, 1000, 1024, 1280, 2000, 2048, 2560, 4096, 5120, 6000, 8192, 10240)
LINKS = (2, 1)


def folded(m, chips):
    """The model as a block on `chips` builds it: the router folded onto fewer physical experts past the flat op's 64 a
    chip (RouterConfig.fold)."""
    F = 1
    while m.num_experts // F // chips > 64:
        F *= 2
    if F == 1:
        return m
    return dataclasses.replace(m, num_experts=m.num_experts // F, router=dataclasses.replace(m.router, fold=F))


def geometry(mesh: Mesh, m, chunk, links):
    """(Geometry, None) or (None, why the block refuses the chunk). A chunk not whole tiles a chip runs padded to the
    next that is (MoEBlock's chunk_pad) - whole rows a mesh row needed."""
    unit = mesh.R * mesh.C * 32
    if chunk % mesh.R:
        return None, f"block: chunk {chunk} is not whole rows a mesh row ({mesh.R} mesh rows)"
    chunk = -(-chunk // unit) * unit
    try:
        return (
            Geometry(
                sp=mesh.R,
                tp=mesh.C,
                sp_wraps=mesh.sp_ring,
                tp_wraps=mesh.tp_ring,
                links=links,
                zigzag=False,
                chunk_tokens=chunk,
                num_experts=m.num_experts,
            ),
            None,
        )
    except AssertionError as e:
        return None, f"block: {e}"


def plan_case(name, mesh_name, chunk, links, grid, explain=False):
    mesh = MESHES[mesh_name]
    m = folded(COV_MODELS[name], mesh.R * mesh.C)
    g, why = geometry(mesh, m, chunk, links)
    rec = dict(model=name, mesh=mesh_name, chunk=chunk, links=links, grid=f"{grid[0]}x{grid[1]}")
    if g is None:
        return dict(rec, ok=False, why=why)
    from models.demos.moe_block.tt import head_plan as hp

    env = dict(grid=grid, payload=None, l1_free=hp.L1_FREE - hp.L1_MARGIN)
    f = FusedHead.plan(g, m, MoEDtypes(), env=env, explain=explain)
    rec.update(ok=f.ok, rows=g.tokens_per_row)
    if f.ok:
        p = f.planned[0]
        rec.update(layout=hp.describe_lay(p.lay), predicted_us=round(p.t), arena_kb=p.arena >> 10, note=f.note)
    else:
        rec["why"] = f.why
    return rec


def _reason_class(why):
    """A refusal's requirement (the summary's grouping)."""
    for key, cls in (
        ("block:", "chunk: whole tiles a chip"),
        ("> 4 with a latent", "TP > 4 with a latent (all-to-all headers)"),
        ("router cores differ", "TP > 4: logits on the router cores"),
        ("counts word", "RS counts word"),
        ("L1:", "L1 (arena)"),
        ("tokens a chip > 2048", "routing: > 2048 tokens a chip"),
        ("experts a chip", "routing: experts a chip"),
        ("no matmul-core layout", "no matmul-core layout"),
        ("router cores", "router cores"),
    ):
        if key in why:
            return cls
    return why.split(":")[0]


def test_fused_head_coverage():
    t0 = time.perf_counter()
    import multiprocessing

    cases = [
        (name, mesh_name, chunk, links, grid)
        for name in COV_MODELS
        for mesh_name, mesh in MESHES.items()
        for chunk in CHUNKS
        for links in LINKS
        for grid in mesh.grids
    ]
    with multiprocessing.get_context("fork").Pool(min(32, os.cpu_count() or 1)) as pool:
        recs = pool.starmap(plan_case, cases, chunksize=4)
    dt = time.perf_counter() - t0
    os.makedirs("generated/moe_block", exist_ok=True)
    with open("generated/moe_block/head_coverage.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # the table: model x mesh, fused / cases (2 links, 12x10 grid: the main column; all: every links / grid)
    lines = [
        "| model | " + " | ".join(MESHES) + " |",
        "|---|" + "---|" * len(MESHES),
    ]
    for name in COV_MODELS:
        cells = []
        for mesh_name in MESHES:
            main = [r for r in recs if r["model"] == name and r["mesh"] == mesh_name and r["links"] == 2]
            main = [r for r in main if r["grid"] == "12x10"]
            cells.append(f"{sum(r['ok'] for r in main)}/{len(main)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    logger.info(f"FUSED HEAD COVERAGE (2 links, 12x10; cases = chunks {CHUNKS}):\n" + "\n".join(lines))
    by = collections.Counter((r["links"], r["grid"], r["ok"]) for r in recs)
    logger.info(
        "COVERAGE by links / grid: "
        + ", ".join(
            f"links {lk} {gr}: {by[(lk, gr, True)]}/{by[(lk, gr, True)] + by[(lk, gr, False)]}"
            for lk in LINKS
            for gr in ("12x10", "11x10")
        )
    )
    reasons = collections.Counter(_reason_class(r["why"]) for r in recs if not r["ok"])
    logger.info("COVERAGE refusals by requirement: " + "; ".join(f"{k}: {v}" for k, v in reasons.most_common()))
    for cls in reasons:
        ex = next(r for r in recs if not r["ok"] and _reason_class(r["why"]) == cls)
        logger.info(f"COVERAGE e.g. {ex['model']} {ex['mesh']} chunk {ex['chunk']} links {ex['links']}: {ex['why']}")
    deg = [r for r in recs if r["ok"] and r.get("note")]
    logger.info(f"COVERAGE L1-limited plans: {len(deg)} of {sum(r['ok'] for r in recs)}")
    logger.info(f"COVERAGE: {len(recs)} cases, {sum(r['ok'] for r in recs)} fused, in {dt:.0f} s")
    # every refusal names its requirement
    for r in recs:
        assert r["ok"] or (isinstance(r.get("why"), str) and len(r["why"]) > 10), r
    # the measured configs plan (the LoudBox's 1x4 ring / line and 2x4, the Galaxy's per-chip load)
    for name, mesh_name, chunk in (
        ("k3", "1x4 ring", 1280),
        ("k3", "1x4 line", 1280),
        ("glm53_flash", "2x4 line", 5120),
        ("glm53_flash", "2x4 line", 2560),
        ("glm53_flash", "2x4 line", 1280),
        ("k3", "8x4 torus", 5120),
        ("glm53_flash", "8x4 torus", 5120),
    ):
        r = plan_case(name, mesh_name, chunk, 2, (12, 10))
        assert r["ok"], r


# ---------------------------------------------------------------------------------------------------------------------
# mock builds: the plan's verdict against the block's build
# ---------------------------------------------------------------------------------------------------------------------


def _active(mesh: Mesh):
    cl = os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")
    mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")
    return (
        bool(mesh.mock)
        and cl.endswith(mesh.mock + ".yaml")
        and (mgd.endswith(mesh.mgd + ".textproto") if mesh.mgd else not mgd)
    )


def _env_list(k, default):
    v = os.environ.get(k)
    return default if not v else type(default)(type(default[0])(x) for x in v.split(","))


def _mock_cases():
    out = []
    for name in _env_list("HEAD_COV_MODELS", tuple(COV_MODELS)):
        for mesh_name, mesh in MESHES.items():
            if not mesh.mock:
                continue
            for chunk in _env_list("HEAD_COV_CHUNKS", (1024, 1280, 2048, 2560, 5120, 10240)):
                for links in _env_list("HEAD_COV_LINKS", LINKS):
                    out.append(
                        pytest.param(
                            name, mesh_name, chunk, links, id=f"{name}-{mesh_name.replace(' ', '_')}-c{chunk}-l{links}"
                        )
                    )
    return out


@pytest.mark.parametrize("model, mesh_name, chunk, links", _mock_cases())
@pytest.mark.timeout(3600)
def test_fused_head_mock_build(model, mesh_name, chunk, links):
    import torch

    import ttnn
    from conftest import set_fabric
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
    from models.demos.moe_block.moe_block import HeadBuildError, MoEBlock, WeightCache, _payload_of
    from models.demos.moe_block.tests.test_moe_block import random_weights

    mesh = MESHES[mesh_name]
    if not _active(mesh):
        pytest.skip(f"needs the mock cluster {mesh.mock}" + (f" and {mesh.mgd}" if mesh.mgd else ""))
    m = folded(COV_MODELS[model], mesh.R * mesh.C)
    g, why = geometry(mesh, m, chunk, links)
    if g is None:
        pytest.skip(why)
    params = fabric2d_device_params(fabric_payload_size=_payload_of(m))
    set_fabric(
        getattr(ttnn.FabricConfig, mesh.fabric),
        params["reliability_mode"],
        fabric_router_config=params["fabric_router_config"],
    )
    rec = dict(model=model, mesh=mesh_name, chunk=chunk, links=links)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(mesh.R, mesh.C), l1_small_size=0)
    err = None
    try:
        gs = dev.compute_with_storage_grid_size()
        rec["grid"] = f"{gs.x}x{gs.y}"
        try:
            block = MoEBlock(dev, m, chunk, random_weights(m), WeightCache(path=None, layer=0), num_links=links)
        except HeadBuildError as e:
            rec.update(status="HEAD BUILD FAILED", error=str(e)[:600])
            err = e
        except Exception as e:  # noqa: BLE001 - outside the head (the exchange, the tail): recorded, not the head's
            rec.update(status="block fails elsewhere", error=f"{type(e).__name__}: {str(e).strip()[:400]}")
        else:
            rec.update(status="ok", head=block.head_mode, plan_ok=block.head_fit.ok if block.head_fit else None)
            rec["summary"] = block.summary[:400]
            x = ttnn.from_torch(
                torch.zeros(mesh.R, mesh.C, chunk // mesh.R, m.emb_dim // mesh.C),  # (the caller's rows: unpadded)
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(0, 1), mesh_shape=(mesh.R, mesh.C)),
            )
            try:
                ttnn.copy(x, block.input_buffer())
                ttnn.deallocate(block(block.input_buffer()))  # (every program built and enqueued)
            except Exception as e:  # noqa: BLE001
                rec.update(status="forward fails", error=f"{type(e).__name__}: {str(e).strip()[:400]}")
                import traceback

                if block.head_mode == "fused" and any(
                    "fused_head" in fr.filename for fr in traceback.extract_tb(e.__traceback__)
                ):
                    err = e  # (the fused head's first call: a head failure)
            ttnn.deallocate(x)
            # the plan's verdict is the build's: fused where it accepts, the ops head where it refuses
            assert (block.head_mode == "fused") == bool(block.head_fit and block.head_fit.ok), rec
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        os.makedirs("generated/moe_block", exist_ok=True)
        with open("generated/moe_block/head_mock_build.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    if err is not None:
        raise err
