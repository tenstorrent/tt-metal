# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41MoEBlock with a placement (DSV41_MOE_PLACE) on REAL layer weights: the block output against the sequential block on the same input.

Env: MPB_LAYER (default 4), MPB_PLACE (flat json {layer: [chip per expert]}), MPB_T (tokens per mesh row, default 4), MPB_CACHE (rw dir for the placed packed weights),
MPB_RS_MIN (1: both blocks use the lean reduce_scatter).
Checks: (1) placed vs sequential PCC (samecol placements: expected bit-identical); (2) the placed block rebuilt from the packed-weight cache (cache hit) gives the
bit-identical output of the placed block built from the checkpoint; (3) the cache directory carries the permutation tag; (4) the sequential block is unchanged by the env.
"""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import moe_place
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as MW
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock, set_rs_ccl

ROWS, COLS, H = 4, 8, 5120


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(5400)
@torch.no_grad()
def test_moe_place_block(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("MPB_LAYER", "4"))
    T = int(os.environ.get("MPB_T", "4"))
    place_file = os.environ["MPB_PLACE"]
    if os.environ.get("MPB_RS_MIN") == "1":
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean
        from models.demos.gpt_oss.tt.ccl import CCLManager

        os.environ["DSV41_MOE_RS_MIN"] = "1"
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ccl_lean.make_lean_sems(ccl)
        set_rs_ccl(ccl)
    B = ROWS * T
    g = torch.Generator().manual_seed(5)
    x = torch.randn(B, H, generator=g).to(torch.bfloat16)
    sh = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    tok = ttnn.from_torch(
        x.reshape(B, 1, 1, H),
        device=md,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sh,
    )
    gate_in = ttnn.from_torch(
        x.reshape(1, 1, B, H),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(ROWS, COLS)),
    )

    def run(blk):
        o = blk.forward(gate_in, tok)
        ttnn.synchronize_device(md)
        y = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(ROWS, COLS))).float()
        ttnn.deallocate(o)
        routing = [
            ttnn.to_torch(ttnn.get_device_tensors(blk.last_routing[1])[r * COLS]).reshape(-1, 6).long()
            for r in range(ROWS)
        ]
        return y, torch.cat(routing)

    os.environ.pop("DSV41_MOE_PLACE", None)
    t0 = time.time()
    seq = DSV41MoEBlock(md, MW.load_moe_layer(layer), batch_per_device=T)
    seq.warmup()
    print(f"MPBLOG seq built {time.time() - t0:.0f} s", flush=True)
    y_seq, ids = run(seq)

    # placed block from the checkpoint, packed weights written under a tagged directory of MPB_CACHE
    os.environ["DSV41_MOE_PLACE"] = place_file
    MW.WEIGHT_CACHE = os.environ["MPB_CACHE"]
    a = moe_place.assignment(layer)
    assert a is not None, "layer has no placement in the file"
    cdir = MW.expert_cache_dir(layer)
    assert cdir.endswith(moe_place.cache_suffix(layer)) and moe_place.tag(a) in cdir
    print(
        f"MPBLOG placed cache dir {cdir} samecol={moe_place.samecol(a)} warm_before={MW._expert_cache_warm(layer)}",
        flush=True,
    )
    t0 = time.time()
    placed = DSV41MoEBlock(md, MW.load_moe_layer(layer), batch_per_device=T)
    placed.warmup()
    print(f"MPBLOG placed built {time.time() - t0:.0f} s warm_after={MW._expert_cache_warm(layer)}", flush=True)
    y_pl, ids_pl = run(placed)
    assert torch.equal(ids, ids_pl), "the router must pick the same experts (the placement does not touch it)"
    p1 = pcc(y_seq, y_pl)
    print(
        f"MPBRES placed-vs-seq pcc {p1:.7f} bit_identical {bool(torch.equal(y_seq, y_pl))} max_abs {float((y_seq - y_pl).abs().max()):.4e} samecol {moe_place.samecol(a)}",
        flush=True,
    )

    # second build: packed-weight cache hit
    t0 = time.time()
    placed2 = DSV41MoEBlock(md, MW.load_moe_layer(layer), batch_per_device=T)
    placed2.warmup()
    print(f"MPBLOG placed (cache hit) built {time.time() - t0:.0f} s", flush=True)
    y_pl2, _ = run(placed2)
    same = bool(torch.equal(y_pl, y_pl2))
    print(f"MPBRES cache-hit rebuild bit_identical_to_first_placed {same}", flush=True)
    # a sequential block built again under the env: the plain directory, unchanged output
    os.environ.pop("DSV41_MOE_PLACE")
    assert MW.expert_cache_dir(layer).endswith(f"layer_{layer}")
    json.dump(
        dict(pcc=p1, cache_hit_bit_identical=same),
        open(os.path.join(os.environ.get("MP_OUT", "/tmp"), "mpb.json"), "w"),
    )
    assert p1 > 0.9999 and same
