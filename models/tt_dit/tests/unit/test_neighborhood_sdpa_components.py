# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Attribute wall time inside ``neighborhood_scaled_dot_product_attention``.

Skipping TRISC (drain / qk) cannot split the 400 ms stage-5 call: those probes still walk every
gather slot. This test changes THAT walk.

1. Gather sweep -- same query bricks, windows 3 / 7 / 11. If ms tracks gather_brick_count, the
   bound is slots walked, not QK/softmax/PV.
2. Ablations at the real window-11 call -- skip_kv (no K/V DRAM), skip_slots (no 147-slot loop,
   still the CB handshake), skip_slots_drain (handshake only). skip_slots is the probe that lets
   compute show up.

WRONG output except the window-11 full row. Probes 5/6 deadlock; do not add them.

path_mode 0 (the default) auto-splits stride-1 + relative table into interior then edge
programs. Skip must fire for the quality path to be ~210 ms rather than ~530 ms (both programs
walking every brick). Pass path_mode=1 to time the tight gather alone (~130 ms). Why two
kernels, and the measurements: ``models/tt_dit/layers/NEIGHBORHOOD_STRIDE1_FINDINGS.md`` §9.

    ./run_na.sh -b -f models/tt_dit/tests/unit/test_neighborhood_sdpa_components.py -t 180 -s 60

``-b`` is required after host C++ (plan / nanobind / kernel_args). Kernel-only edits JIT on the
first launch.
"""

import time

import pytest
import torch

import ttnn
from models.tt_dit.layers.neighborhood_attention import (
    _build_relative_masks,
    _query_chunk_bricks,
    _tiles_per_kv_chunk,
    halo_sites,
)
from models.tt_dit.layers.neighborhood_permute import SITES_PER_BRICK, brick_count
from models.tt_dit.models.vae.diffvae_ltx_stage5 import _bands

STAGE5_GRID = (145, 272, 480)
SLAB_FRAMES = 78
STRIDE = (1, 1, 1)
BRICK = (2, 8, 2)
HEAD_COUNT = 1
HEAD_DIM = 64
W_SHARDS = 8
SCALE = 1.0

# Windows shrink the gather box. Query bricks stay the stage-5 first-band owned grid.
GATHER_WINDOWS = ((3, 3, 3), (7, 7, 7), (11, 11, 11))
# skip_slots = 7, skip_slots_drain = 8 -- hashed through NeighborhoodSDPAParams::probe.
STAGE5_ABLATIONS = (
    ("skip_kv", 1),
    ("skip_slots", 7),
    ("skip_slots_drain", 8),
)


def _stage5_first_band_t() -> int:
    band = _bands(STAGE5_GRID[0], frames=SLAB_FRAMES, kernel=11, align=BRICK[0])[0]
    return min(band.pad_hi, STAGE5_GRID[0]) - band.pad_lo


def _sharded_geometry(window: tuple[int, int, int]):
    volume = (_stage5_first_band_t(), STAGE5_GRID[1], STAGE5_GRID[2])
    width_local = volume[2] // W_SHARDS
    halo = halo_sites(window[2], BRICK[2])
    resident = (volume[0], volume[1], width_local + 2 * halo)
    owned = (volume[0], volume[1], width_local)
    shard_origin = (0, 0, -halo)
    query_extent = (resident[0], resident[1], width_local)
    query_origin = (0, 0, halo)
    query_chunk_bricks = _query_chunk_bricks(STRIDE, BRICK)
    plan = ttnn.transformer.neighborhood_plan(
        volume,
        window,
        STRIDE,
        BRICK,
        query_chunk_bricks=query_chunk_bricks,
        shard_extent=resident,
        shard_origin=shard_origin,
        query_extent=query_extent,
        query_origin=query_origin,
    )
    return {
        "volume": volume,
        "window": window,
        "resident": resident,
        "owned": owned,
        "shard_origin": shard_origin,
        "query_extent": query_extent,
        "query_origin": query_origin,
        "query_chunk_bricks": query_chunk_bricks,
        "plan": plan,
        "tiles_per_kv_chunk": _tiles_per_kv_chunk(plan["gather_brick_count"]),
    }


def _upload(mesh_device, geo):
    plan = geo["plan"]
    channels = HEAD_COUNT * HEAD_DIM
    kv_sites = plan["brick_count"] * SITES_PER_BRICK
    query_sites = plan["query_brick_count"] * SITES_PER_BRICK
    assert kv_sites == brick_count(geo["resident"], BRICK) * SITES_PER_BRICK
    assert query_sites == brick_count(geo["owned"], BRICK) * SITES_PER_BRICK
    seed = torch.randn(1, 1, SITES_PER_BRICK, channels)
    query = ttnn.from_torch(
        seed.repeat(1, 1, query_sites // SITES_PER_BRICK, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    kv = seed.repeat(1, 1, kv_sites // SITES_PER_BRICK, 1)
    key = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    value = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    origin = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
        1, 1, plan["chunk_count"], plan["gather_origin_columns"]
    )
    packed_w = origin[0, 0, :, 2].to(torch.int64)
    interior_hi = int((packed_w >> 31).sum().item())
    interior_c6 = int(origin[0, 0, :, 6].to(torch.int64).sum().item())
    edge_c7 = int((origin[0, 0, :, 7].to(torch.int64) == 0xFFFFFFFF).sum().item())
    print(
        f"  origin stamp: chunks={plan['chunk_count']} bit31={interior_hi} " f"col6={interior_c6} col7_edge={edge_c7}",
        flush=True,
    )
    origin_on_device = ttnn.from_torch(origin, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    print(
        f"  origin page={origin_on_device.buffer_aligned_page_size()} "
        f"padded={tuple(origin_on_device.padded_shape)}",
        flush=True,
    )
    origin_rt = ttnn.to_torch(origin_on_device)
    packed_w_d = origin_rt[0, 0, :, 2].to(torch.int64)
    col7_d = origin_rt[0, 0, :, 7].to(torch.int64)
    print(
        f"  origin device: bit31={int((packed_w_d >> 31).sum().item())} "
        f"col6={int(origin_rt[0, 0, :, 6].to(torch.int64).sum().item())} "
        f"col7_edge={int((col7_d == 0xFFFFFFFF).sum().item())} "
        f"col7_minmax=({int(col7_d.min().item())},{int(col7_d.max().item())})",
        flush=True,
    )
    interior_mask = ttnn.from_torch(
        _build_relative_masks(geo["window"], BRICK),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    return query, key, value, origin_on_device, interior_mask


def _time_probe(mesh_device, geo, tensors, probe: int, label: str) -> float:
    query, key, value, origin_on_device, interior_mask = tensors
    plan = geo["plan"]

    def run_once():
        return ttnn.transformer.neighborhood_scaled_dot_product_attention(
            query,
            key,
            value,
            origin_on_device,
            interior_mask=interior_mask,
            volume=geo["volume"],
            context_window=geo["window"],
            stride=STRIDE,
            brick=BRICK,
            query_chunk_bricks=geo["query_chunk_bricks"],
            shard_extent=geo["resident"],
            shard_origin=geo["shard_origin"],
            query_extent=geo["query_extent"],
            query_origin=geo["query_origin"],
            head_count=HEAD_COUNT,
            scale=SCALE,
            tiles_per_kv_chunk=geo["tiles_per_kv_chunk"],
            probe=probe,
        )

    print(f"  {label}: launching (may JIT) ...", flush=True)
    run_once()
    ttnn.synchronize_device(mesh_device)
    print(f"  {label}: warmup done", flush=True)
    iterations = 3
    start = time.perf_counter()
    for _ in range(iterations):
        run_once()
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - start) * 1000 / iterations
    gather = plan["gather_brick_count"]
    qbricks = plan["query_brick_count"]
    us_q = ms * 1000 / qbricks
    ns_slot = ms * 1e6 / (qbricks * gather)
    print(
        f"  {label}: {ms:.1f} ms  ({us_q:.2f} us/qbrick, {ns_slot:.1f} ns/slot, "
        f"gather {gather}, qbricks {qbricks})",
        flush=True,
    )
    return ms


@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_report_component_timing(mesh_device):
    rows = []

    print("", flush=True)
    print("===== gather sweep (probe 0; query bricks fixed, window shrinks the box) =====", flush=True)
    for window in GATHER_WINDOWS:
        geo = _sharded_geometry(window)
        tensors = _upload(mesh_device, geo)
        label = f"window{window[0]}"
        ms = _time_probe(mesh_device, geo, tensors, probe=0, label=label)
        rows.append((label, geo, ms, 0))

    stage5 = _sharded_geometry((11, 11, 11))
    stage5_tensors = _upload(mesh_device, stage5)
    print("", flush=True)
    print("===== ablations at stage-5 window 11 =====", flush=True)
    ablation_ms = {}
    for label, probe in STAGE5_ABLATIONS:
        ablation_ms[label] = _time_probe(mesh_device, stage5, stage5_tensors, probe=probe, label=label)

    print("")
    print("===== neighborhood_sdpa, stage-5 first band =====")
    print(
        f"  volume {stage5['volume']}  resident {stage5['resident']}  owned {stage5['owned']}  "
        f"brick {BRICK}  heads {HEAD_COUNT}x{HEAD_DIM}"
    )
    print("")
    print(f"  {'label':<18} {'gather':>6} {'qbricks':>8} {'ms':>8} {'us/qbrick':>10} {'ns/slot':>8}")
    print(f"  {'-' * 18} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")
    for label, geo, ms, _ in rows:
        gather = geo["plan"]["gather_brick_count"]
        qbricks = geo["plan"]["query_brick_count"]
        print(
            f"  {label:<18} {gather:6d} {qbricks:8d} {ms:8.1f} "
            f"{ms * 1000 / qbricks:10.2f} {ms * 1e6 / (qbricks * gather):8.1f}"
        )
    gather11 = stage5["plan"]["gather_brick_count"]
    qbricks11 = stage5["plan"]["query_brick_count"]
    for label, _ in STAGE5_ABLATIONS:
        ms = ablation_ms[label]
        print(
            f"  {label:<18} {gather11:6d} {qbricks11:8d} {ms:8.1f} "
            f"{ms * 1000 / qbricks11:10.2f} {ms * 1e6 / (qbricks11 * gather11):8.1f}"
        )

    full = next(ms for label, _, ms, _ in rows if label == "window11")
    skip_slots = ablation_ms["skip_slots"]
    skip_slots_drain = ablation_ms["skip_slots_drain"]
    skip_kv = ablation_ms["skip_kv"]
    print("")
    print("  how to read this (host wall-clock; overlapping stages do not sum):")
    print("    window11 is interior+edge (path_mode 0). Skip must fire: ~210 ms, not ~530 ms")
    print("    (~530 ms means both programs walked every brick).")
    print(f"    walk      (window11 - skip_slots)          {full - skip_slots:8.1f} ms")
    print(f"    compute   (skip_slots - skip_slots_drain)  {skip_slots - skip_slots_drain:8.1f} ms")
    print(f"    handshake (skip_slots_drain)               {skip_slots_drain:8.1f} ms")
    print(f"    K/V DRAM  (window11 - skip_kv)             {full - skip_kv:8.1f} ms")
    print("")
    gathers = [geo["plan"]["gather_brick_count"] for _, geo, _, _ in rows]
    times = [ms for _, _, ms, _ in rows]
    if gathers[-1] != gathers[0]:
        slope = (times[-1] - times[0]) / (gathers[-1] - gathers[0])
        print(
            f"    gather slope (window {GATHER_WINDOWS[0][0]}->{GATHER_WINDOWS[-1][0]}): "
            f"{slope:.2f} ms per gather brick"
        )
        if abs(slope) < 0.2:
            print("    ns/slot collapsing: time is NOT the slot walk (Q/origin/dispatch?).")
        else:
            print("    ns/slot holding: cost is walking gather slots. Next: per-brick sub-box, not softmax.")
    if skip_slots < full * 0.5:
        print("    skip_slots dropped the wall: the 147-loop was the bound. Compute is the skip_slots row.")
    print("=" * 70)
