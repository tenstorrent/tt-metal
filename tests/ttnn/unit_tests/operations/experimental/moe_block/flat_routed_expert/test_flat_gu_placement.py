# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat routed expert's per-chip gate/up placement (FlatGuSource auto_tensix / auto_hybrid) on the whole system
mesh: the host-only link-load report (also on a mock cluster: distinct chip geometries, per-chip / reference-chip /
compromise busiest link), and, with programmable DRAM cores, every chip's y bit-identical to the DRISC senders'."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases = _load("flat_cases", HERE / "flat_cases.py")
fw = _load("fw", REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py")


def _mesh_shape():
    n = ttnn.get_num_devices()
    return {32: (8, 4), 8: (2, 4), 4: (1, 4)}.get(n, (1, n))


@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(*_mesh_shape()))
    yield m
    ttnn.close_mesh_device(m)


@pytest.mark.parametrize("model", ["k3", "glm53_flash"])
@pytest.mark.parametrize("source", ["auto_tensix", "auto_hybrid"])
def test_gu_placement_report(mesh, model, source):
    mb = ttnn._ttnn.operations.experimental.moe_block
    md, pc = cases.MODELS[model], cases.per_chip(model, "8x4", 5120)
    lo = mb.FlatLayoutOptions(
        rows_hint=round(pc["M"]),
        gu_source=getattr(mb.FlatGuSource, source),
        drisc_tiles=mb.FlatDriscTiles.free_only if source == "auto_hybrid" else mb.FlatDriscTiles.both,
    )
    r = mb.flat_gu_placement_report(mesh, md["H"], md["I"], pc["epc"], md["E"], 5120, layout=lo)
    chips = r["chips"]
    print(
        f"{model} {source}: {len(chips)} chips, {r['topologies']} distinct geometries; busiest link us (worst chip) "
        f"own {max(c['own_us'] for c in chips):.1f} reference {max(c['reference_us'] for c in chips):.1f} "
        f"compromise {max(c['compromise_us'] for c in chips):.1f} chosen {max(c['chosen_us'] for c in chips):.1f}; "
        f"bank floor {r['bank_floor_us']:.1f}; search {r['search_ms_max']:.0f} ms a geometry, compromise "
        f"{r['compromise_search_ms']:.0f} ms"
    )
    for c in chips:
        assert c["chosen_us"] <= min(c["own_us"], c["compromise_us"]) + 1e-6
        assert len(c["own_words"]) == len(chips[0]["own_words"])


@pytest.mark.parametrize("model", ["k3"])
def test_gu_placement_bit_identical(mesh, model):
    mb = ttnn._ttnn.operations.experimental.moe_block
    if not mb.FlatDriscSenders.supported(mesh):
        pytest.skip("needs programmable DRAM cores (the DRISC reference)")
    md, pc = cases.MODELS[model], cases.per_chip(model, "8x4", 5120)
    E, H, I, NG, M = pc["epc"], md["H"], md["I"], md["E"], round(pc["M"])
    N = mesh.get_num_devices()
    counts = [M] * E
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32 + 64
    x = torch.randn(rows, H, generator=torch.Generator().manual_seed(5))
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e in range(E):
        c_[0, e], r_[0, e] = counts[e], offs[e]

    def rep(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )  # fmt: skip

    xd, cd, rd = rep(x, ttnn.bfloat16), rep(c_, ttnn.uint32), rep(r_, ttnn.uint32)
    wsets = cases.weight_sets(H, I)
    local = [wsets[g % cases.N_W] for g in range(E)]
    real = torch.zeros(rows, dtype=torch.bool)
    for o, c in zip(offs, counts):
        real[o : o + c] = True
    ref = None
    for source, tiles in (("drisc", "both"), ("auto_tensix", "both"), ("auto_hybrid", "free_only")):
        lay = dict(
            rows_hint=M, drisc_tiles=getattr(mb.FlatDriscTiles, tiles), gu_source=getattr(mb.FlatGuSource, source),
            gu_read_depth=1 if source != "drisc" else -1,
        )  # fmt: skip
        op = fw.FlatRoutedExpert(
            mesh, [local] * N, m=5120, H=H, I=I, gids=[list(range(E))] * N, n_global=NG, wdtype="bf4",
            act=md["act"], pin=1, layout=lay,
        )  # fmt: skip
        for _ in range(3):
            op(xd, cd, rd)
        ys = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(op(xd, cd, rd))]
        op.close()
        ref = ref or ys
        for chip, (a, b) in enumerate(zip(ys, ref)):
            assert not (a != b).reshape(-1, rows, H).any(0)[real].any(), f"{source}: chip {chip} y differs from drisc"
