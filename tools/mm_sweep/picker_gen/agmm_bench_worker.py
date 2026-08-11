#!/usr/bin/env python3
"""One AGMM baseline, measured as DEVICE time. Emits one `SWEEP_JSON {...}` line.

The design spec (tools/mm_sweep/REGIME_A_AGMM_DESIGN_SPEC.md) requires four baselines at identical
shapes/configs, plus overlap_efficiency. All four are reachable from this one worker:

    mm1     single-chip full-K regime_a_matmul on a UNIT mesh, fabric disabled  -> T_mm
    ag      standalone ttnn.experimental.all_gather_async on the TP submesh     -> T_ag
    phase0  the AGMM op with TT_AGMM_FUSED_GATHER unset: all_gather + matmul    -> unfused baseline
    fused   the AGMM op with TT_AGMM_FUSED_GATHER=1 (+ TT_AGMM_DIRECT_L1=1 for Phase 2)

`phase0` and `fused` are the SAME op; which path it takes is an env decision made in the op at program
build time, which is why the variant is selected by the driver's environment and not by an argument here.

MEASUREMENT. Device time comes from the device-profiler CSV, which is only written when the device
CLOSES -- so each measurement runs in its own subprocess that opens a mesh, runs the op and closes it.
Host wall is NOT used: it folds dispatch overhead in, and the spec explicitly says host dispatch time is
not evidence of overlap. This mirrors tools/mm_sweep/picker_gen/prod_sweep_worker.py, with one difference
that matters on a mesh: durations are keyed by (device, core, risc), so cores from different devices
cannot overwrite each other. Per invocation we report the MAKESPAN -- max over devices of that device's
own max-core duration -- because a fused multi-device op is only finished when its slowest device is.

argv: variant M K N tp topology [num_links] [nblocks]      topology = ring | line, num_links defaults to 2
"""
import csv
import json
import os
import statistics
import sys

import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV_PATH = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 12

VARIANT = sys.argv[1]
M, K, N = (int(x) for x in sys.argv[2:5])
TP = int(sys.argv[5])
TOPO = sys.argv[6]
# 2 = the production config for these shapes on Galaxy, and what the correctness suite uses (NUM_LINKS in
# test_all_gather_regime_a_matmul_async.py). Keep the two in step: measuring one link count while the tests
# cover another is how ">64 mux channels" got reported as a scope limit when it was a 1-link artefact.
LINKS = int(sys.argv[7]) if len(sys.argv) > 7 else 2
# Optional pinned config "Pk,Ns,Sm,kb,nsb" as argv[9]; "auto" (default) lets the picker choose.
CFG = sys.argv[9] if len(sys.argv) > 9 else "auto"
NBLOCKS = int(sys.argv[8]) if len(sys.argv) > 8 else 2


def parse_runids():
    """runid -> {"wall": makespan us, "start": first cycle}, from the device profiler CSV.

    Keyed by (PCIe slot, core_x, core_y, RISC) so a mesh run does not collapse different devices' cores
    onto one key. Only *-KERNEL zones are considered, matching the single-chip worker.
    """
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH) as fh:
        rows = list(csv.reader(fh))

    # The file opens with an ARCH/CHIP_FREQ preamble line before the real header, so find the header by its
    # columns. Names are matched case-INSENSITIVELY: the column is spelled "run host ID", and matching it
    # as "run host id" silently yields zero runids, which reads as "the profiler was off".
    def norm(r):
        return [c.strip().lower() for c in r]

    hdr_i = next((i for i, r in enumerate(rows) if "run host id" in norm(r) and "zone name" in norm(r)), None)
    if hdr_i is None:
        return {}
    idx = {c: i for i, c in enumerate(norm(rows[hdr_i]))}
    need = ["pcie slot", "core_x", "core_y", "risc processor type", "run host id", "zone name", "type"]
    tcol = next((c for c in idx if c.startswith("time[cycles")), None)
    if any(c not in idx for c in need) or tcol is None:
        return {}
    raw = {}
    for r in rows[hdr_i + 1 :]:
        if len(r) <= max(idx.values()):
            continue
        if not r[idx["zone name"]].strip().endswith("-KERNEL"):
            continue
        dev = r[idx["pcie slot"]].strip()
        key = (dev, r[idx["core_x"]].strip(), r[idx["core_y"]].strip(), r[idx["risc processor type"]].strip())
        raw.setdefault(r[idx["run host id"]].strip(), {}).setdefault(key, []).append(
            (r[idx["type"]].strip(), int(r[idx[tcol]]))
        )
    # Run host IDs are PER DEVICE (verified: 4 devices x 30 ops = 120 runids, each on exactly one device), so
    # a mesh op appears as tp separate runids. Return dev -> [(start, wall_us)] ordered by that device's OWN
    # clock; cross-device timestamps are not comparable (independent cycles-since-reset), but durations are.
    out = {}
    for runid, per_key in raw.items():
        for (dev, _cx, _cy, _ri), events in per_key.items():
            st = None
            for typ, cyc in events:
                if typ == "ZONE_START":
                    st = cyc
                elif typ == "ZONE_END" and st is not None:
                    e = out.setdefault(dev, {}).setdefault(runid, [None, 0])
                    e[0] = st if e[0] is None else min(e[0], st)
                    e[1] = max(e[1], cyc - st)  # slowest core on this device == this device's op duration
                    st = None
    return {
        dev: [(v[0], v[1] / FREQ * 1e6) for _r, v in sorted(per_run.items(), key=lambda kv: kv[1][0])]
        for dev, per_run in out.items()
    }


def mesh_geometry(tp):
    n = ttnn.get_num_devices()
    if n >= 32:
        return ((4, 8), 0 if tp == 4 else 1)
    if n >= tp:
        return ((1, tp), 1)
    return None


def main():
    try:
        os.remove(CSV_PATH)
    except OSError:
        pass
    cfg = None
    if CFG != "auto":
        pk, ns, sm, kbt, nsb = (int(x) for x in CFG.split(","))
        cfg = ttnn.RegimeAMatmulConfig(k_slices=pk, n_slices=ns, m_slices=sm, k_block_tiles=kbt, n_subblock_tiles=nsb)
    res = {
        "cfg": CFG,
        "variant": VARIANT,
        "M": M,
        "K": K,
        "N": N,
        "tp": TP,
        "topology": TOPO,
        "num_links": LINKS,
        "outcome": "runtime",
        "err": "",
        "fused_gather": os.environ.get("TT_AGMM_FUSED_GATHER", ""),
        "direct_l1": os.environ.get("TT_AGMM_DIRECT_L1", ""),
    }
    labels = []
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)

    # ---- mm1: single chip, full K, no fabric. The T_mm baseline. ----
    if VARIANT == "mm1":
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        ran = False
        try:
            in0 = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh)
            wc = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, mesh)
            in1 = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh, memory_config=wc)
            for blk in range(NBLOCKS):
                for _ in range(WARMUP):
                    ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
                    labels.append(f"b{blk}_w")
                for _ in range(TIMED):
                    ttnn.experimental.regime_a_matmul(in0, in1, config=None)
                    labels.append(f"b{blk}_t")
                ttnn.synchronize_device(mesh)
            ran = True
        except Exception as e:  # noqa: BLE001
            res["err"] = str(e)[:400]
        finally:
            ttnn.close_mesh_device(mesh)
        return finish(res, labels, ran)

    # ---- ag / phase0 / fused: the TP submesh. ----
    geom = mesh_geometry(TP)
    if geom is None:
        res["err"] = f"need >= {TP} devices, have {ttnn.get_num_devices()}"
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    (rows, cols), cluster_axis = geom
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if TOPO == "ring" else ttnn.FabricConfig.FABRIC_1D
    topology = ttnn.Topology.Ring if TOPO == "ring" else ttnn.Topology.Linear
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    ran = False
    try:
        sub_shape = [1, 1]
        sub_shape[cluster_axis] = TP
        mesh = parent.create_submesh(ttnn.MeshShape(tuple(sub_shape)))

        dims = [None, None]
        dims[cluster_axis] = 1
        in0 = ttnn.from_torch(
            a,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        wc = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, mesh)
        in1 = ttnn.from_torch(
            b,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh,
            memory_config=wc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # CCL resources: double-buffered and barriered before first use, per the op's caller contract.
        # Getting this wrong yields intermittent per-device partial corruption, not a clean failure.
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        ttnn.synchronize_device(mesh)
        sem_sets = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
        bufs = [
            ttnn.from_torch(
                torch.zeros((M, K), dtype=torch.float32),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh,
            )
            for _ in range(2)
        ]
        ttnn.synchronize_device(mesh)

        slot = [0]

        def call():
            i = slot[0]
            slot[0] = (i + 1) % 2
            if VARIANT == "ag":
                # dim=-1: in0 is [M, K/TP] sharded along K, which is the LAST dim. This mirrors the
                # dim the AGMM op's own Phase-0 composition passes, so T_ag measures the same gather.
                return ttnn.experimental.all_gather_async(
                    in0,
                    bufs[i],
                    -1,
                    sem_sets[i],
                    num_links=LINKS,
                    topology=topology,
                    cluster_axis=cluster_axis,
                )
            return ttnn.experimental.all_gather_regime_a_matmul_async(
                in0,
                in1,
                persistent_output_buffer=bufs[i],
                multi_device_global_semaphore=sem_sets[i],
                barrier_semaphore=None,
                num_links=LINKS,
                topology=topology,
                cluster_axis=cluster_axis,
                config=cfg,
            )

        for blk in range(NBLOCKS):
            for _ in range(WARMUP):
                call()
                labels.append(f"b{blk}_w")
            for _ in range(TIMED):
                call()
                labels.append(f"b{blk}_t")
            ttnn.synchronize_device(mesh)
        ran = True
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:400]
    finally:
        for s in parent.get_submeshes():
            ttnn.close_mesh_device(s)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return finish(res, labels, ran)


def finish(res, labels, ran):
    if not ran:
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    per_dev = parse_runids()
    if not per_dev:
        res["err"] = "no profiler data (is TT_METAL_DEVICE_PROFILER=1 set and the build tracy-enabled?)"
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    # Building the inputs also runs device work (from_torch writes, the zeroed persistent buffers), and all
    # of it precedes the measurement loop, so the ops we want are each device's TAIL. Taking the tail rather
    # than requiring an exact count is what makes this robust to setup ops -- and `ops_discarded` is reported
    # so a mis-slice shows up as a number instead of quietly shifting every measurement by one op.
    #
    # phase0 issues TWO device ops per call (all_gather, then the full-K matmul); its per-call cost is their
    # sum. That is known from the variant, not inferred from the count -- inferring it would silently absorb
    # a stray setup op into a bogus ops-per-call.
    per_call = 2 if VARIANT == "phase0" else 1
    want = per_call * len(labels)
    short = {d: len(v) for d, v in per_dev.items() if len(v) < want}
    if short:
        res["err"] = f"demux: devices short of {want} ops: {short}"
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    res["ops_per_call"] = per_call
    res["ndev"] = len(per_dev)
    res["ops_discarded"] = {d: len(v) - want for d, v in per_dev.items()}
    tails = {d: [w for _s, w in v[-want:]] for d, v in per_dev.items()}
    # MAKESPAN per call: a fused multi-device op is only finished when its slowest device is finished, so
    # take the max over devices rather than any single device's number.
    walls = [max(sum(t[c * per_call + j] for j in range(per_call)) for t in tails.values()) for c in range(len(labels))]
    # phase0 is literally all_gather_async THEN the full-K matmul, so its two constituent ops ARE two of the
    # four baselines the spec wants -- at identical shapes and config, in the same process, which is stronger
    # than measuring them separately. Report each sub-op (also as a makespan) as well as the total.
    if per_call > 1:
        timed = [c for c, lab in enumerate(labels) if lab.endswith("_t")]
        res["sub_op_median_us"] = [
            round(statistics.median([max(t[c * per_call + j] for t in tails.values()) for c in timed]), 3)
            for j in range(per_call)
        ]
    blocks = []
    for blk in range(NBLOCKS):
        w = [x for x, lab in zip(walls, labels) if lab == f"b{blk}_t"]
        if w:
            blocks.append([round(v, 3) for v in w])
    allw = [x for bl in blocks for x in bl]
    if not allw:
        res["err"] = "no timed iterations recovered"
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    res["block_medians"] = [round(statistics.median(bl), 3) for bl in blocks]
    res["median_us"] = round(statistics.median(allw), 3)
    res["min_us"] = round(min(allw), 3)
    res["max_us"] = round(max(allw), 3)
    res["n_iters"] = len(allw)
    res["outcome"] = "ok"
    print("SWEEP_JSON " + json.dumps(res), flush=True)


main()
