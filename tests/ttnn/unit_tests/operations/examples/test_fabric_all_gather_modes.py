# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_all_gather on a 4-chip ring: relay vs direct (fabric-routed) vs mcast send modes, device time.

Variants (env MODES_VARIANTS, comma list):
  base  relay, the far shard one way            b   relay, far shard split between the directions (balance)
  d     direct: unicast to both neighbours + one unicast of the whole shard to the opposite chip (one way)
  db    direct, the opposite chip's shard split between the directions
  m     mcast: one multicast over 2 chips one way + a unicast to the neighbour the other way
  mb    mcast, the 2-chip leg alternating direction by chip parity (busiest hop still 2 shards; the write-and-forward
        first hop of every mcast rides a 1-shard link)
  ms    mcast_split: each half of the shard multicast both ways (toward p+1 half 1 over G/2 chips, half 2 over G/2 - 1;
        toward p-1 the reverse), so every hop carries G/2 - 1/2 shards, as with b, and nothing is relayed
  mh    mcast_halves: half 1 multicast toward p+1 over all G - 1 other chips, half 2 toward p-1 over all of them (the
        same hop load as ms, every packet going the whole way round)
A suffix "@auto" keeps the probe-based per-chip placement (the mcast variants then keep their arrival counters on sink
cores); otherwise every variant uses placement="uniform" (the same port cores on every chip).

Per case: bit-exact on every chip, then DEVICE TIME as a traced model sees it: [filler (~113 us ttnn.add), all-gather
into a reused output] x ROUNDS captured in one trace and replayed REPLAYS times; per call, the slowest chip's program
duration (real-time profiler, picked out by runtime ID); median over calls. The filler before every call hides
dispatch and lines the chips up.

    MODES_FABRICS=1d_ring,2d_torus_xy MODES_LINKS=1,2 MODES_VARIANTS=base,b,d,db,m,mb \
    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s

The ring: FAG_MESH (default 1,4) opens the mesh, FAG_AXIS (default 1) picks the axis whose rows / columns are the rings
(e.g. FAG_MESH=1,8 on a T3K, FAG_MESH=8,4 FAG_AXIS=0 on a Galaxy). G = the mesh size along FAG_AXIS; for G = 8, mcast
sends a 4-chip and a 3-chip multicast (m: the 4-chip leg always toward p+1; mb: alternating by chip parity).
Tables: fag_modes_analyze.py <log>.
"""

import os
import statistics
import threading
import time

import pytest
import torch
import ttnn

from ttnn.operations.examples.fabric_all_gather import fabric_all_gather, link_load, plan

_FABRICS = {"1d_ring": "FABRIC_1D_RING", "2d_torus_xy": "FABRIC_2D_TORUS_XY", "1d": "FABRIC_1D"}
FABRICS = tuple(os.environ.get("MODES_FABRICS", "1d_ring,2d_torus_xy").split(","))
LINKS = tuple(int(x) for x in os.environ.get("MODES_LINKS", "1,2").split(","))
VARIANTS = tuple(os.environ.get("MODES_VARIANTS", "base,b,d,db,m,mb,base@auto,b@auto").split(","))
_LADDER = "64,64;128,128;256,256;512,512;1024,1024;2048,2048;2048,4096;2048,32;32,2048;32768,32;32,32768"
SHAPES = [tuple(int(v) for v in s.split(",")) for s in os.environ.get("MODES_SHAPES", _LADDER).split(";") if s]
ROUNDS = int(os.environ.get("MODES_ROUNDS", "8"))
REPLAYS = int(os.environ.get("MODES_REPLAYS", "5"))
PAYLOAD = int(os.environ.get("FAG_PAYLOAD", "14336"))  # fabric packet payload bytes (Blackhole; lower it on Wormhole)
MESH = tuple(int(v) for v in os.environ.get("FAG_MESH", "1,4").split(","))
AXIS = int(os.environ.get("FAG_AXIS", "1"))
_MASK = 0xFFFF
_VARIANT_KW = {
    "base": dict(mode="relay", balance=False),
    "b": dict(mode="relay", balance=True),
    "d": dict(mode="direct", balance=False),
    "db": dict(mode="direct", balance=True),
    "m": dict(mode="mcast", balance=False),
    "mb": dict(mode="mcast", balance=True),
    "ms": dict(mode="mcast_split", balance=False),
    "mh": dict(mode="mcast_halves", balance=False),
}


class _Records:
    def __init__(self):
        self.lock = threading.Lock()
        self.recs = []

    def __call__(self, batch):
        with self.lock:
            for r in batch.records:
                try:
                    self.recs.append(
                        (int(r.runtime_id) & _MASK, int(r.chip_id), (r.end_timestamp - r.start_timestamp) / r.frequency)
                    )
                except Exception:  # noqa: BLE001
                    pass

    def drain(self, settle=0.3, timeout=5.0):
        t_end, last = time.time() + timeout, -1
        while time.time() < t_end:
            with self.lock:
                n = len(self.recs)
            if n == last:
                break
            last = n
            time.sleep(settle)
        with self.lock:
            out, self.recs = self.recs, []
        return out


def _expected(host, c):
    """Chip c's (row-major) gathered output: its ring's shards in ring-axis order."""
    r, col = divmod(c, MESH[1])
    shards = [host[k, col] for k in range(MESH[0])] if AXIS == 0 else [host[r, k] for k in range(MESH[1])]
    H, W = host.shape[-2:]
    return torch.cat([x.reshape(1, 1, H, W) for x in shards], dim=0)


def _router():
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = PAYLOAD
    return cfg


def _op_id():
    return int(ttnn._ttnn.get_device_operation_id())


def _traced_device_ns(mesh_device, rec, run, filler):
    """Median over calls of the slowest chip's program time, traced, with the filler before every call."""
    run()
    filler()
    ttnn.synchronize_device(mesh_device)
    windows, keep = [], []
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(ROUNDS):
        keep.append(filler())
        s = _op_id()
        run()
        windows.append((s, _op_id()))
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    rec.drain()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    for t in keep:
        ttnn.deallocate(t)
    by = {}
    for rid, chip, ns in rec.drain():
        by.setdefault((rid, chip), []).append(ns)
    calls = []
    for s, e in windows:
        ids = {(s + k) & _MASK for k in range(e - s)}
        chips = {c for rid, c in by if rid in ids}
        if chips:
            calls.append(max(sum(statistics.median(by[(rid, c)]) for rid in ids if (rid, c) in by) for c in chips))
    assert calls, "no real-time profiler records inside the traced op windows"
    return statistics.median(calls), min(calls), max(calls)


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "fabric_config": getattr(ttnn.FabricConfig, _FABRICS[f]),
                "fabric_router_config": _router(),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 64 * 1024 * 1024,
            },
            id=f"fabric_{f}",
        )
        for f in FABRICS
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], ids=[f"mesh{MESH[0]}x{MESH[1]}"], indirect=True)
def test_fabric_all_gather_modes(mesh_device):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("real-time profiler inactive")
    fabric = str(ttnn.get_fabric_config()).split(".")[-1]
    rec = _Records()
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(rec)
    big = ttnn.from_torch(
        torch.randn(1, 1, 2048, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    filler = lambda: ttnn.add(big, big)  # noqa: E731
    try:
        for H, W in SHAPES:
            torch.manual_seed(0)
            host = torch.randn((*MESH, H, W), dtype=torch.bfloat16)
            inp = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=MESH),
            )
            tiles = (H // 32) * (W // 32)
            for links in LINKS:
                for v in VARIANTS:
                    name, _, place = v.partition("@")
                    kw = dict(
                        _VARIANT_KW[name],
                        placement=place or "uniform",
                        cluster_axis=AXIS,
                        topology=ttnn.Topology.Ring,
                        num_links=links,
                    )
                    tag = f"{fabric} links={links} {v:<9} {H}x{W} tiles={tiles}"
                    try:
                        out = fabric_all_gather(inp, **kw)
                        ttnn.synchronize_device(mesh_device)
                    except (ValueError, RuntimeError) as e:
                        print(f"FAG_MODES {tag} unsupported: {str(e).splitlines()[0][:140]}", flush=True)
                        continue
                    for c, t in enumerate(ttnn.get_device_tensors(out)):
                        expected = _expected(host, c)
                        got = ttnn.to_torch(t).reshape(expected.shape)
                        assert torch.equal(got, expected), f"{tag}: chip {c} output != gathered shards"
                    chips, _ = plan(
                        mesh_device,
                        cluster_axis=AXIS,
                        topology=ttnn.Topology.Ring,
                        num_links=links,
                        placement=kw["placement"],
                        balance=kw["balance"],
                        mode=kw["mode"],
                    )
                    busiest = max(link_load(chips).values())
                    run = lambda: fabric_all_gather(inp, output=out, **kw)  # noqa: E731
                    med, lo, hi = _traced_device_ns(mesh_device, rec, run, filler)
                    print(
                        f"FAG_MODES {tag} device_us={med / 1e3:.2f} min={lo / 1e3:.2f} max={hi / 1e3:.2f} "
                        f"busiest_hop_shards={busiest:.2f} bitexact=1",
                        flush=True,
                    )
                    ttnn.deallocate(out)
            ttnn.deallocate(inp)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
