# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``kv_copy`` on the device: frames copied between a frame-major pool and a slot-major workbench slot and inside
the pool, bank-local, no fabric.

    test_copy_bytes         one chip, raw twins filled from the host: after every copy BOTH tensors are read back
                            whole and compared with a host mirror (what was to be copied is there, nothing else
                            changed); copy-in, copy-out, inside the pool, the empty plan; several lane counts
    test_copy_bytes_64      one chip, 64 frames x 99 slots (947 MB) in one launch, byte-equal
    test_copy_rate          one chip, traced: GB/s per chip for 1, 12 and 64 frames x 99 slots; lanes, placement,
                            depth and batch sweeps; the read side and the write side alone
    test_copy_galaxy        the same program on all 32 chips: bytes on every chip, then the rates

Mock Galaxy (HY4_MOCK=1): the programs are compiled and their buffers checked, nothing runs."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.kv_store import copy as kc
from models.experimental.hy4_preview.fused.kv_store import device as dv
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
TRACE_REGION = 8 << 20
one_chip = pytest.mark.parametrize(
    "device_params, mesh_device", [({"trace_region_size": TRACE_REGION}, (1, 1))], indirect=True, ids=["1x1"]
)
# the 8x4 mesh as every other piece opens it on the board (the prefill stack's torus fabric); this piece's
# programs use no fabric
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), (8, 4))],
    indirect=True,
    ids=["8x4"],
)
RESULTS = Path("generated/hy4_preview/logs/mk10_kv_copy.json")
KINDS = (fr.LATENT, fr.INDEX)


def record(key, value):
    if MOCK:
        return
    data = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1))


class Mirror:
    """A raw twin on the device and its bytes on the host."""

    def __init__(self, mesh_device, geometry, n, seed):
        self.geometry, self.n = geometry, n
        gen = torch.Generator().manual_seed(seed)
        self.host = torch.randint(0, 256, dv.raw_shape(geometry, n), dtype=torch.uint8, generator=gen)
        self.tensor = dv.raw_to_device(mesh_device, geometry, self.host)
        dv.check_layout(self.tensor, geometry, n)

    def frame(self, frame, layer):
        return dv.frame_view(self.geometry, self.host, frame, layer)

    def device_bytes(self, mesh_device):
        """[devices, *host shape]"""
        devices = mesh_device.get_num_devices()
        t = ttnn.to_torch(self.tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        return t.reshape(devices, *self.host.shape)


def host_copy(src: Mirror, dst: Mirror, pairs):
    before = src.host.clone() if src is dst else src.host
    for s, d in pairs:
        for layer in range(src.geometry.kind.layers):
            dst.frame(d, layer).copy_(dv.frame_view(src.geometry, before, s, layer))


def compare(mesh_device, mirrors, what):
    for m in mirrors:
        got = m.device_bytes(mesh_device)
        for dev in range(got.shape[0]):
            if not torch.equal(got[dev], m.host):
                bad = (got[dev] != m.host).reshape(m.host.shape[0], -1).any(-1).nonzero().flatten().tolist()
                raise AssertionError(
                    f"{what}: device {dev}, {m.geometry.kind.name} tensor differs in slots {bad[:12]} ({len(bad)} slots)"
                )


def bytes_scenario(mesh_device, configs, pool_frames=8, bench_frames=6, steps=None):
    pool = {k: Mirror(mesh_device, fr.Geometry.pool(k), pool_frames, seed=20 + i) for i, k in enumerate(KINDS)}
    bench = {
        k: Mirror(mesh_device, fr.Geometry.contiguous(k, bench_frames), 1, seed=30 + i) for i, k in enumerate(KINDS)
    }
    mirrors = list(pool.values()) + list(bench.values())
    sides = {
        "in": [(pool[k].tensor, pool[k].geometry, bench[k].tensor, bench[k].geometry) for k in KINDS],
        "out": [(bench[k].tensor, bench[k].geometry, pool[k].tensor, pool[k].geometry) for k in KINDS],
        "pool": [(pool[k].tensor, pool[k].geometry, pool[k].tensor, pool[k].geometry) for k in KINDS],
    }
    # a conversation with frames [5, 2, 7]: its prefix goes to the workbench, 3 workbench frames come back to
    # other pool frames, 2 pool frames are forked; frame numbers out of order everywhere
    steps = steps or [
        ("in", [(5, 0), (2, 1), (7, 2)]),
        ("out", [(1, 0), (2, 3), (0, 6)]),
        ("pool", [(5, 1), (2, 4)]),
        ("in", []),
        ("in", [(6, 5)]),
    ]
    if not MOCK:
        compare(mesh_device, mirrors, "before any copy")
    for config in configs:
        copies = {name: kc.KvCopy(mesh_device, kinds, **config) for name, kinds in sides.items()}
        if MOCK:
            for c in copies.values():
                c.compile()
            continue
        copied = 0
        for name, pairs in steps:
            c = copies[name]
            c.set_pairs(pairs)
            c()
            src, dst = (pool, bench) if name == "in" else (bench, pool) if name == "out" else (pool, pool)
            for k in KINDS:
                host_copy(src[k], dst[k], pairs)
            compare(mesh_device, mirrors, f"{config} {name} {pairs}")
            copied += c.bytes_per_chip()
        logger.info(
            f"HY4_MK10 kv_copy bytes {config} on {mesh_device.get_num_devices()} chip(s): {len(steps)} plans "
            f"({', '.join(name for name, _ in steps)}: pool -> workbench, workbench -> pool, inside the pool), "
            f"{copied / 1e6:.1f} MB per chip copied, all four tensors of every chip byte-equal to the host mirror after every plan"
        )


@one_chip
@pytest.mark.parametrize(
    "configs",
    [
        [dict()],
        [
            dict(lanes=8, placement="grid", depth=4, write_batch=2),
            dict(lanes=1, depth=8, write_batch=3),
            dict(lanes=3, depth=2, write_batch=1),
            dict(lanes=12, placement="rows"),
            dict(lanes=8, placement="rows", depth=1),
        ],
    ],
    ids=["default", "variants"],
)
def test_copy_bytes(mesh_device, configs):
    bytes_scenario(mesh_device, configs)


@one_chip
def test_copy_bytes_64(mesh_device):
    """64 frames x 99 slots in one launch: 946.5 MB per chip, a scrambled plan, byte-equal."""
    frames = 64
    pool = {k: Mirror(mesh_device, fr.Geometry.pool(k), frames, seed=40 + i) for i, k in enumerate(KINDS)}
    bench_geometry = {k: fr.Geometry.contiguous(k, frames) for k in KINDS}
    bench = {k: dv.allocate(mesh_device, bench_geometry[k], 1, raw=True) for k in KINDS}
    copy = kc.KvCopy(mesh_device, [(pool[k].tensor, pool[k].geometry, bench[k], bench_geometry[k]) for k in KINDS])
    order = torch.randperm(frames, generator=torch.Generator().manual_seed(1)).tolist()
    pairs = [(f, k) for k, f in enumerate(order)]
    copy.set_pairs(pairs)
    if MOCK:
        copy.compile()
        return
    copy()
    for k in KINDS:
        got = ttnn.to_torch(bench[k])
        for f, pos in pairs:
            for layer in range(0, k.layers, 7):
                want = pool[k].frame(f, layer)
                have = dv.frame_view(bench_geometry[k], got, pos, layer)
                assert torch.equal(have, want), f"{k.name}: pool frame {f} -> workbench frame {pos}, layer {layer}"
        # and every byte, in the pool's order
        g = bench_geometry[k]
        per = got.shape[2] // frames
        whole = got.reshape(k.layers, frames, per, got.shape[3])[:, [pos for _, pos in sorted(pairs)]]
        assert torch.equal(whole.transpose(0, 1).reshape(pool[k].host.shape), pool[k].host), k.name
    logger.info(
        f"HY4_MK10 kv_copy bytes64: {frames} frames x 99 slots, {copy.bytes_per_chip() / 1e6:.1f} MB per chip in one launch, byte-equal"
    )


def rate_tensors(mesh_device, frames):
    """Real dtypes, not initialised (the rate does not depend on the bytes): a pool and a workbench of `frames`."""
    out = []
    for k in KINDS:
        pg, bg = fr.Geometry.pool(k), fr.Geometry.contiguous(k, frames)
        out.append((dv.allocate(mesh_device, pg, frames), pg, dv.allocate(mesh_device, bg, 1), bg))
    return out


def timed(mesh_device, copy, pairs, repeats=20):
    copy.set_pairs(pairs)
    ms = tracing.traced_ms(mesh_device, copy, repeats=repeats)["replay_ms"]
    gbs = copy.bytes_per_chip() / 1e9 / (ms / 1e3) if pairs else 0.0
    return ms, gbs


def wall_ms(mesh_device, copy, calls=5):
    """One launch at a time, not traced, each awaited: host dispatch included (a cross-check of the traced time)."""
    best = None
    for _ in range(calls):
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        copy()
        ttnn.synchronize_device(mesh_device)
        ms = 1e3 * (time.perf_counter() - start)
        best = ms if best is None else min(best, ms)
    return best


def plan(frames, n, seed=3):
    """n pool frames (scattered) -> the first n workbench frames."""
    order = torch.randperm(frames, generator=torch.Generator().manual_seed(seed)).tolist()[:n]
    return [(f, k) for k, f in enumerate(order)]


@one_chip
def test_copy_rate(mesh_device):
    frames = 64
    kinds = rate_tensors(mesh_device, frames)
    if MOCK:
        for config in (
            dict(),
            dict(lanes=12, placement="grid"),
            dict(mode=kc.MODE_NO_WRITE),
            dict(mode=kc.MODE_NO_READ),
            dict(mode=kc.MODE_RETURN),
        ):
            kc.KvCopy(mesh_device, kinds, **config).compile()
        return
    results = {}
    # --- lanes and placement at 12 frames
    best = None
    for placement in ("grid", "rows"):
        for lanes in (1, 2, 3, 4, 6, 8, 12):
            copy = kc.KvCopy(mesh_device, kinds, lanes=lanes, placement=placement)
            ms, gbs = timed(mesh_device, copy, plan(frames, 12))
            results[f"12f {placement} lanes={lanes}"] = [ms, gbs]
            logger.info(
                f"HY4_MK10 kv_copy rate 12 frames ({copy.bytes_per_chip() / 1e6:.1f} MB) {placement} lanes={lanes}: {ms:.3f} ms, {gbs:.1f} GB/s per chip"
            )
            if best is None or gbs > best[0]:
                best = (gbs, dict(lanes=lanes, placement=placement))
    config = best[1]
    # --- depth and write batch at the best lanes
    for depth, batch in ((1, 1), (2, 1), (2, 2), (4, 1), (4, 4), (8, 2), (8, 4)):
        copy = kc.KvCopy(mesh_device, kinds, depth=depth, write_batch=batch, **config)
        ms, gbs = timed(mesh_device, copy, plan(frames, 12))
        results[f"12f {config} depth={depth} batch={batch}"] = [ms, gbs]
        logger.info(
            f"HY4_MK10 kv_copy rate 12 frames {config} depth={depth} write_batch={batch}: {ms:.3f} ms, {gbs:.1f} GB/s per chip"
        )
        if gbs > best[0]:
            best = (gbs, dict(config, depth=depth, write_batch=batch))
    config = best[1]
    # --- the sizes asked for, and what bounds them
    for n in (1, 12, 64):
        copy = kc.KvCopy(mesh_device, kinds, **config)
        ms, gbs = timed(mesh_device, copy, plan(frames, n))
        wall = wall_ms(mesh_device, copy)
        results[f"{n}f best"] = [ms, gbs, wall]
        logger.info(
            f"HY4_MK10 kv_copy rate {n} frame(s) x 99 slots = {copy.bytes_per_chip() / 1e6:.1f} MB per chip, {config}: {ms:.3f} ms, "
            f"{gbs:.1f} GB/s per chip (traced, best of 3 x 20 replays); one awaited launch, not traced: {wall:.3f} ms"
        )
    for name, mode in (
        ("read side alone", kc.MODE_NO_WRITE),
        ("write side alone", kc.MODE_NO_READ),
        ("launch floor", kc.MODE_RETURN),
    ):
        copy = kc.KvCopy(mesh_device, kinds, mode=mode, **config)
        ms, gbs = timed(mesh_device, copy, plan(frames, 12))
        results[f"12f {name}"] = [ms, gbs]
        logger.info(
            f"HY4_MK10 kv_copy rate 12 frames, {name}: {ms:.3f} ms"
            + (f", {gbs:.1f} GB/s per chip" if mode != kc.MODE_RETURN else "")
        )
    copy = kc.KvCopy(mesh_device, kinds, **config)
    ms, _ = timed(mesh_device, copy, [])
    results["empty plan"] = [ms, 0.0]
    logger.info(f"HY4_MK10 kv_copy rate empty plan: {ms:.3f} ms")
    record("one_chip", {"best": config, "results": results})


@galaxy
def test_copy_galaxy(mesh_device):
    """The same program on all 32 chips at once (no fabric: every chip copies inside its own DRAM banks)."""
    devices = mesh_device.get_num_devices()
    one = json.loads(RESULTS.read_text()).get("one_chip", {}).get("best") if RESULTS.is_file() else None
    config = one or dict()
    # 89 MB per chip of raw twins, read back from all 32 chips after every plan
    steps = [("in", [(3, 0), (1, 1)]), ("out", [(0, 2), (1, 0)]), ("pool", [(3, 1)])]
    bytes_scenario(mesh_device, [config, dict(config, placement="own")], pool_frames=4, bench_frames=2, steps=steps)
    frames = 64
    kinds = rate_tensors(mesh_device, frames)
    rows, cols = tuple(mesh_device.shape)
    coords = [(r, c) for r in range(rows) for c in range(cols)]
    own = dict(config, placement="own")
    if MOCK:
        kc.KvCopy(mesh_device, kinds, **config).compile()
        kc.KvCopy(mesh_device, kinds, **own).compile()
        kc.KvCopy(mesh_device, kinds, coords=[(3, 2)], **config).compile()
        return
    results = {}
    # --- all chips at once
    for name, cfg in (("one placement for all chips", config), ("each chip its own placement", own)):
        for n in (1, 12, 64):
            copy = kc.KvCopy(mesh_device, kinds, **cfg)
            ms, gbs = timed(mesh_device, copy, plan(frames, n), repeats=10)
            wall = wall_ms(mesh_device, copy)
            results[f"{n}f {name}"] = [ms, gbs, wall]
            logger.info(
                f"HY4_MK10 kv_copy galaxy {devices} chips at once, {name}, {n} frame(s) x 99 slots = {copy.bytes_per_chip() / 1e6:.1f} MB per chip "
                f"({devices * copy.bytes_per_chip() / 1e9:.2f} GB in all): {ms:.3f} ms until the last chip is done = {gbs:.1f} GB/s per chip, "
                f"{devices * gbs:.0f} GB/s over the mesh (traced); one awaited launch, not traced: {wall:.3f} ms"
            )
    # --- chip by chip (the program on one chip of the mesh at a time), 12 frames; the chips differ in where the
    # right-hand DRAM banks' nearest worker column is
    column = {coord: max(x for x, _ in kc.bank_workers(mesh_device, coord)) for coord in coords}
    per_chip = {}
    for name, cfg in (("rows", config), ("own", own)):
        for coord in coords:
            copy = kc.KvCopy(mesh_device, kinds, coords=[coord], **cfg)
            ms, gbs = timed(mesh_device, copy, plan(frames, 12), repeats=10)
            per_chip[(name, coord)] = gbs
    for name in ("rows", "own"):
        for col in sorted(set(column.values())):
            chips = [c for c in coords if column[c] == col]
            rates = sorted(per_chip[(name, c)] for c in chips)
            logger.info(
                f"HY4_MK10 kv_copy galaxy per chip, placement {name}, right-hand bank column {col}: {len(chips)} chips, "
                f"12 frames: {rates[0]:.1f} / {rates[len(rates) // 2]:.1f} / {rates[-1]:.1f} GB/s (min / median / max)"
            )
        logger.info(
            f"HY4_MK10 kv_copy galaxy per chip, placement {name}: "
            + " ".join(f"{c[0]}{c[1]}:c{column[c]}:{per_chip[(name, c)]:.0f}" for c in coords)
        )
    record(
        f"galaxy_{devices}",
        {
            "config": config,
            "results": results,
            "column": {f"{c[0]},{c[1]}": column[c] for c in coords},
            "per_chip_gbs": {name: {f"{c[0]},{c[1]}": per_chip[(name, c)] for c in coords} for name in ("rows", "own")},
        },
    )


@one_chip
def test_compare(mesh_device):
    """KvCompare against what the host knows: equal after a copy, the rows counted; a frame that was not copied is
    found; rows of positions beyond the given length are not looked at."""
    pool = {k: Mirror(mesh_device, fr.Geometry.pool(k), 8, seed=50 + i) for i, k in enumerate(KINDS)}
    bench = {k: Mirror(mesh_device, fr.Geometry.contiguous(k, 6), 1, seed=60 + i) for i, k in enumerate(KINDS)}
    kinds = [(pool[k].tensor, pool[k].geometry, bench[k].tensor, bench[k].geometry) for k in KINDS]
    copy, cmp = kc.KvCopy(mesh_device, kinds), kc.KvCompare(mesh_device, kinds, chip_index=lambda coord: 5)
    copy.set_pairs([(5, 0), (2, 1)])
    cmp.set_pairs([(5, 0), (2, 1), (7, 2)], 3 * fr.FRAME_TOKENS)
    if MOCK:
        copy.compile()
        cmp.compile()
        return
    copy()
    per_frame = fr.LATENT.layers * fr.FRAME_ROWS + fr.INDEX.layers * fr.BANKS  # latent rows + index-key blocks
    cmp()
    r = cmp.read()
    assert r["rows"] == 3 * per_frame and r["bad"] == per_frame, r  # frame 7 was not copied: all of it differs
    assert r["first"][3] == 2, r
    cmp.set_pairs([(5, 0), (2, 1), (7, 2)], 2 * fr.FRAME_TOKENS)  # the conversation ends with its second frame
    cmp()
    r2 = cmp.read()
    assert r2["rows"] == 2 * per_frame and r2["bad"] == 0, r2
    # a length inside the third frame: chip 5 holds positions [640, 768) of every 4096: 128 + 64 of its rows there
    cmp.set_pairs([(5, 0), (2, 1), (7, 2)], 2 * fr.FRAME_TOKENS + 4096 + 640 + 64)
    cmp()
    r3 = cmp.read()
    shards = (128 + 64) // 32
    assert r3["visible"][0] == 512 + 128 + 64 and r3["rows"] == 2 * per_frame + shards * (
        fr.LATENT.layers * 32 + fr.INDEX.layers
    ), r3
    assert r3["bad"] == r3["rows"] - 2 * per_frame
    # one byte changed in a copied frame is one row
    host_copy(pool[fr.LATENT], bench[fr.LATENT], [(5, 0), (2, 1)])
    host = bench[fr.LATENT].host.clone()
    dv.frame_view(bench[fr.LATENT].geometry, host, 1, 40)[77, 655] ^= 1
    changed = dv.raw_to_device(mesh_device, bench[fr.LATENT].geometry, host)
    cmp1 = kc.KvCompare(
        mesh_device,
        [(pool[fr.LATENT].tensor, pool[fr.LATENT].geometry, changed, bench[fr.LATENT].geometry)],
        chip_index=lambda coord: 5,
    )
    cmp1.set_pairs([(5, 0), (2, 1)], 2 * fr.FRAME_TOKENS)
    cmp1()
    r4 = cmp1.read()
    assert r4["bad"] == 1 and r4["first"][3] == 1, r4
    logger.info(
        f"HY4_MK10 kv_compare: {r2['rows']} rows / blocks of 2 copied frames equal on the device; an uncopied frame: {r['bad']} "
        f"differ; a length inside a frame: {r3['rows']} compared; one flipped bit: {r4['bad']} row, pair {r4['first'][3]}"
    )
