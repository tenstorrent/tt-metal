#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""chunk_load.py: the chunked sustained load. 4x8 mesh, FABRIC_1D (Linear), opened once.
load chunks of --chunk-min minutes (default chunk_sched.RAMP_MIN: 1, 2, 3, 4, 15), each followed by --gap-s s of forced
quiet (chunk_sched.py: synchronize, then no enqueue until the window ends), so every restart is a whole-galaxy di/dt step. Launches
run back to back with no host sync: the host stays at most RUN_AHEAD_OPS
heavy ops ahead of the device (ub0_repro.py's RunAhead, ttnn events); without events it synchronizes every
RUN_AHEAD_OPS launches and the LOAD line names that fallback. One launch, its parts in this order:
  heavy --heavy-n ops of --heavy (default ub0_repro.HEAVY_DEFAULT, the C4 burst's load, so a heavier default there is
        the chunks' too; lofi_all), in bursts of --burst-steps with --burst-gap-ms of
        pause after each (ub0_repro.BurstGap; 0 = back to back): the full-load step after each quiet; its last output is
        checked
  h3    --blocks H3-shaped blocks (h3shape_repro.py's): all_gather_async on the SP (8-way) and TP (4-way) axes,
        reduce_scatter_minimal_async on TP, --mm matmuls + a 5376x5376 matmul (FPU), gelu, rms_norm, untilize/tilize
  mm    a replicated matmul (FPU), the same on every chip: its output is compared across chips
  sfpu  exp -> sqrt -> reciprocal -> gelu on a replicated tensor (SFPU), --sfpu-reps times. With --sfpu-par 1 (the
        sustained part's default) the heavy matmul also runs a fused --sfpu-act SFPU epilogue on every
        output tile, so the SFPU works through the whole heavy part, not only in this part's slot
  dram  add of two replicated --dram-mb MB DRAM-interleaved tensors (every DRAM bank), --dram-reps times
  pcie  a host -> device write of a replicated --pcie-mb MB tensor into a DRAM tensor (copy_host_to_device_tensor), on
        command queue 1 when the mesh opens with 2: PCIe and DRAM writes run beside cq 0's
        matmul instead of after it; with one queue it runs in line and the LOAD line names that fallback
Pre-ramp: --pre-ramp-s s of the same full
load right before chunk 1, not counted in its minutes (chunk_sched.run's pre_s). Its C8 telemetry picks the chips whose mean
power and current sat below 97% of their own limits (read, never written); TopUp gives each of them more
16k bf8 LoFi matmuls per burst on its own 1x1 submesh (no sync or readback inside a burst), then --pre-warm-s more s of
load run before chunk 1 starts. The top-up is off unless FLEET_TOPUP=1 (the device hung at its plan() on 2/2
default runs; with it off the same load ran chunks 1-5). Its launches count as chunk 1's (one numeric check); its top-up ops do not.
Numeric check on the device, no host readback in the hot loop: every launch adds ne(out, ref) of each part into a
per-part counter tensor on each chip (ref = launch 0's output, kept on the device). At the chunk end, before the
quiet, the counters and the replicated parts' last outputs are read back: a chip fails a part on any counted
mismatch, a non-finite element, or disagreeing with the other chips (replicated parts). A part whose op the installed
ttnn lacks is turned off at launch 0 (row "part_off"), the rest go on.
Utilization rows: "calib" (each part alone, synchronized: its measured device seconds per launch) and per chunk
launches/s, TFLOP/s per chip and DRAM/ETH/PCIe GB/s per chip derived from the measured launch rate and the parts'
tensor sizes (not hardware counters). Beat protocol as h3shape_repro.py ("launch N", "done N", "verify K", "close").
Each chunk row carries "load" (heavy op, run-ahead, checks queued, fallbacks); chunk_sched.report() makes one LOAD
line per chunk from it and the tray power poll. Exit 0 clean, 4 numeric failure, 1 error (a chunk-end readback that raised too)."""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunk_sched as cs  # noqa: E402
import link_pattern as lp  # noqa: E402
import load_variant as lv  # noqa: E402
import numeric_capture as nc  # noqa: E402
import numeric_check as nck  # noqa: E402
import pinpoint_verdict as pv  # noqa: E402
import tray_lanes as tl  # noqa: E402
from ub0_repro import (  # noqa: E402
    BURST_GAP_MS,
    BURST_STEPS,
    HEAVY,
    HEAVY_DEFAULT,
    HEAVY_ELTWISE,
    HEAVY_PCIE_MB,
    RUN_AHEAD_OPS,
    SFPU_ACT,
    BurstGap,
    RunAhead,
    heavy_dram_bytes,
    heavy_op,
)

ap = argparse.ArgumentParser()
ap.add_argument("--rows-n", type=int, default=4)
ap.add_argument("--cols-n", type=int, default=8)
SCHED, GAP = cs.schedule()  # the ramp, or FLEET_CHUNK_MIN / FLEET_GAP_S
ap.add_argument(
    "--chunk-min", default=",".join(cs.mins(c) for c in SCHED), help="chunk lengths, minutes, comma-separated"
)
ap.add_argument("--gap-s", type=float, default=GAP, help="forced quiet after each chunk")
ap.add_argument("--lead-s", type=float, default=0.0, help="quiet before chunk 0")
ap.add_argument("--mm", type=int, default=3)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--sp-h", type=int, default=896)
ap.add_argument(
    "--sp-w",
    type=int,
    default=None,
    help="SP-axis AG input columns per chip (default 5376 / mesh rows: 1344 on the 4x8, 2688 on a 2x4 tray submesh)",
)
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--parts", default="heavy,h3,mm,sfpu,dram,pcie")


def heavy_list(v):
    """--heavy: one HEAVY op, or several comma-separated, run back to back in each launch, each output checked on its own
    (the stage 5 DIDT numeric chunk runs 16k_hifi2,16k_bf8_lofi)."""
    ok = sorted(k for k, x in HEAVY.items() if x is not None)
    hs = [h for h in v.split(",") if h]
    if not hs or any(h not in ok for h in hs) or len(set(hs)) != len(hs):
        raise argparse.ArgumentTypeError(
            f"--heavy takes one or more of {', '.join(ok)}, comma-separated, each once (got '{v}')"
        )
    return ",".join(hs)


ap.add_argument(
    "--heavy",
    type=heavy_list,
    default=os.environ.get("TRAY_HEAVY") or HEAVY_DEFAULT,
    help="the heavy part's op (ub0_repro.py's --heavy); several comma-separated run back to back, each checked",
)


def heavy_x_list(v):
    """--heavy-x: --heavy's check, but empty is allowed (no extra ops)."""
    return heavy_list(v) if v.strip(",") else ""


ap.add_argument(
    "--heavy-x",
    type=heavy_x_list,
    default="",
    help="extra number formats after --heavy's ops (16k_hifi4, bfloat4_b, float32, int32/"
    "int8): each is built and tried on its own (a refused one is turned off alone, part_off heavy:<op>), runs "
    "--heavy-n times per launch on launch 0 and in each chunk's last --heavy-x-s seconds, and is checked and "
    "voted across chips like --heavy's ops",
)
ap.add_argument(
    "--heavy-x-s",
    type=float,
    default=0.0,
    help="seconds at the end of each chunk that also run the --heavy-x ops (run.sh: 10 plain, 90 full, 2 smoke)",
)
ap.add_argument(
    "--heavy-n",
    type=int,
    default=16,
    help="heavy ops per launch, back to back (16, so most of each " "launch is the heaviest matmul; was 4)",
)
ap.add_argument(
    "--burst-gap-ms",
    type=float,
    default=float(os.environ.get("TRAY_BURST_GAP_MS") or BURST_GAP_MS),
    help="pause (not a quiet: no clock check) after every --burst-steps heavy ops (ub0_repro.py's; 0 = back to back)",
)
ap.add_argument(
    "--heavy-pcie-mb",
    type=int,
    default=HEAVY_PCIE_MB,
    help="lofi_all's PCIe write per step, MB per chip "
    "(run.sh's sustained part passes ub0_repro.SUSTAINED_PCIE_MB; quick keeps the C4 burst's)",
)
ap.add_argument(
    "--heavy-check-each",
    type=int,
    choices=(0, 1),
    default=0,
    help="1: every one of a launch's --heavy-n heavy outputs is compared with its reference on the device, not only "
    "the last (the stage 5 DIDT numeric chunk; 16x the checked matmuls per launch)",
)
ap.add_argument(
    "--sfpu-vote",
    type=int,
    choices=(0, 1),
    default=0,
    help="1: each checked heavy output's first rows, height-sharded one 32-row shard per core, go through one SFPU op "
    "per launch (exp, reciprocal, sqrt, gelu in turn), each compared on the device with its launch-0 reference "
    "and voted across chips at launch 0; a mismatch names the chip and the core x,y",
)
ap.add_argument("--burst-steps", type=int, default=BURST_STEPS, help="heavy ops per burst")
ap.add_argument(
    "--capture-ring",
    type=int,
    default=int(os.environ.get("FLEET_CAPTURE_RING") or 4),
    help="hold the last N checked heavy outputs and poll the counter every N launches; a chip "
    "whose counter rose is captured at its first wrong word (numeric_capture.py); 0 = off",
)
ap.add_argument(
    "--cqs", type=int, choices=(1, 2), default=2, help="command queues: 2 runs the pcie part on cq 1, beside cq 0"
)
ap.add_argument("--sfpu-reps", type=int, default=1)
ap.add_argument(
    "--sfpu-par",
    type=int,
    choices=(0, 1),
    default=0,
    help="1: the heavy matmul carries a fused --sfpu-act "
    "SFPU epilogue on every output tile, so the SFPU works inside the FPU's matmul with no op of its own ("
    "run.sh's sustained part: FLEET_SUSTAIN_SFPU_PAR, default 1); 0: the old load",
)
ap.add_argument("--sfpu-act", default=SFPU_ACT, help="--sfpu-par 1's activation, <ttnn.UnaryOpType>[:<param>]")
ap.add_argument("--dram-mb", type=int, default=16)
ap.add_argument("--dram-reps", type=int, default=2)
ap.add_argument("--pcie-mb", type=int, default=4)
ap.add_argument(
    "--pcie-every",
    type=int,
    default=1,
    help="the pcie part (and its check) on every Nth launch only "
    "(in line on cq 0, its 4 MiB x chips write paced the bus-bar mix to half rate on 16 GT/s links)",
)
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--label", default="")
ap.add_argument(
    "--pre-ramp-s",
    type=float,
    default=0.0,
    help="full load before chunk 0, measured for the top-up, not " "counted in the chunks; 0 = none",
)
ap.add_argument("--pre-warm-s", type=float, default=0.0, help="more full load after the top-up is set, before chunk 0")
ap.add_argument(
    "--topup-telem", default=None, help="C8 telemetry file (lib/fleet.sh's c8_telemetry.txt) for the top-up"
)
ap.add_argument("--topup-limits", default=None, help="C8 limits file (c8_limits.txt), read only")
ap.add_argument(
    "--topup-gain",
    type=float,
    default=float(os.environ.get("FLEET_TOPUP_GAIN") or 1.0),
    help="extra heavy ops per burst = ceil(burst-steps * gain * the chip's shortfall to 97%% of its limit)",
)
ap.add_argument(
    "--fast-poll-s",
    type=float,
    default=float(os.environ.get("FLEET_FASTPOLL_S") or 4.0),
    help="read each chip's power and current from sysfs every --fast-poll-ms for this long after each chunk "
    "starts (the PEAK line's peaks); read-only, 0 = off",
)
ap.add_argument("--fast-poll-ms", type=float, default=float(os.environ.get("FLEET_FASTPOLL_MS") or 100.0))
ap.add_argument(
    "--rw-after",
    type=int,
    default=0,
    help="G3: after the verify of this chunk (1-based) and before "
    "its quiet, numeric_dram.rw_check: the x/w/y-in-DRAM C4n-size matmul with readbacks; 0 = off",
)
ap.add_argument("--rw-s", type=float, default=15.0, help="G3's seconds of load (readbacks not counted)")
ap.add_argument("--rw-check-s", type=float, default=3.0, help="G3: a readback at least every this many s of load")
ap.add_argument("--rw-wall-s", type=float, default=35.0, help="G3's wall-clock cap, readbacks included")
ap.add_argument("--rw-per-check", type=int, default=8, help="G3: matmuls per batch (one device sync each)")
ap.add_argument(
    "--dbi-wall-s",
    type=float,
    default=0.0,
    help="G3b: right after G3, dram_dbi.dbi_check: "
    "DRAM byte/DBI patterns written, copied on the device and read back on every chip, passes until this wall cap; 0 = off",
)
ap.add_argument("--dbi-hops", type=int, default=8, help="G3b: DRAM-to-DRAM copies on the device per pass")
ap.add_argument(
    "--l1-wall-s",
    type=float,
    default=0.0,
    help="N1: right after G3b, l1_march.l1_check: "
    "L1 patterns written one shard per Tensix core, copied L1 to L1 to the next core and read back on every chip, "
    "passes until this wall cap; 0 = off",
)
ap.add_argument("--l1-full", action="store_true", help="N1: add the inverse patterns (--mode=full)")
ap.add_argument("--l1-kb", type=int, default=0, help="N1: per-core shard KB (0 = sized to the allocatable L1)")
ap.add_argument(
    "--trays",
    default=None,
    help="N[,N..]: load only these trays (1-4): the full mesh "
    "still opens, every part runs on each chosen tray's own submesh (CCL inside it), other trays run no "
    "kernels; default every chip (run.sh: FLEET_LOAD_TRAYS)",
)
ap.add_argument(
    "--shape",
    default=None,
    help="tray-drop shape (drop_shapes.py: h3-replay, h3-cadence, "
    "imbalance, weights, cold-full): per launch, each part runs only on the trays whose hi/lo state lists it",
)
ap.add_argument("--shape-tray", type=int, default=2, help="the tray the shape aims at (the repro BDF's tray, 1-4)")
ap.add_argument("--shape-seed", type=int, default=3290, help="the shape's random phase seed (logged in its shape row)")
ap.add_argument(
    "--variant",
    default=os.environ.get("FLEET_LOAD_VARIANT", ""),
    help=f"dev load variant (fleet/load_variant.py): comma list of {', '.join(lv.VARIANTS + tuple(lv.ALIASES))}; a WARNING on every chunk",
)
a = ap.parse_args()
try:
    VAR = lv.parse(a.variant)
except ValueError as e:
    ap.error(str(e))
lv.apply(a, VAR)
if VAR:
    print(lv.warning(VAR), flush=True)
HS = lv.HostStress("host-stress" in VAR)
a.heavy_act = a.sfpu_act if a.sfpu_par else None  # ub0_repro.heavy_op's fused epilogue
try:
    TRAYS = tl.parse_trays(a.trays)  # None: every chip, the one 4x8 mesh as before
except ValueError as e:
    ap.error(str(e))
SH = None  # the tray-drop shape, its part overrides applied before any tensor is made
if a.shape:
    import drop_shapes as ds

    try:
        SH = ds.Shape(a.shape, TRAYS or tl.TRAYS, a.shape_tray, a.shape_seed)
    except ValueError as e:
        ap.error(str(e))
    if SH.per_tray() and not TRAYS:
        ap.error(f"--shape {a.shape} runs parts on some trays only: it needs --trays (one submesh per tray)")
    for k, v in ds.SHAPES[a.shape]["args"].items():
        setattr(a, k, v)
lead = a.lead_s
chunks_s, _ = cs.schedule({"FLEET_CHUNK_MIN": a.chunk_min, "FLEET_GAP_S": str(a.gap_s)})
parts = [p for p in a.parts.split(",") if p]
HEAVY_OPS = a.heavy.split(",")  # --heavy's ops, run back to back in each launch
HEAVY_CHECKS = a.heavy_n if a.heavy_check_each else 1  # checked outputs per heavy op per launch
rows = open(a.rows, "a")


TU = []  # the TopUp, once made: its op count restarts when the pre-ramp ends (its ops are not chunk 1's TFLOP/s)


def h3_dims(sp_w, rows_n):
    """(sp_w, w2's width) for the h3 block on a mesh of rows_n rows. Its residual add r + a1 needs the
    reduce-scatter of y over the rows to come out sp_w wide, so y = hg @ w2 is rows_n * sp_w wide. The w2 of 5376 x 5376
    fit only the 4x8 (5376 / 4 = 1344): on a 2x4 tray submesh r was 2688 wide against a1's 1344 and binary_ng threw
    'Invalid subtile broadcast type', which turned h3 off. sp_w None: 5376 / rows_n (tile-rounded), so w2 stays 5376 x 5376.
    """
    if sp_w is None:
        sp_w = max(32, 5376 // max(rows_n, 1) // 32 * 32)
    return sp_w, rows_n * sp_w


def check_count(ref, parts, pcie_every, n_launch=0, heavy_checks=1):
    """(checks per launch text, checks in n_launch launches): pcie's checks run on every pcie_every'th launch only;
    heavy's heavy_checks per op (--heavy-check-each)."""
    pe = max(pcie_every, 1)
    base = sum(len(ref[p]) * (heavy_checks if p == "heavy" else 1) for p in parts if p != "pcie")
    pc = len(ref["pcie"]) if "pcie" in parts else 0
    per = str(base + pc) if pe == 1 or not pc else f"{base} + {pc} (pcie every {pe})"
    return per, base * n_launch + pc * (n_launch // pe)


def row(**kw):
    if kw.get("event") == "preramp_end" and TU:
        TU[0].n.clear()
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw}
    rows.write(json.dumps(kw) + "\n")
    rows.flush()
    os.fsync(rows.fileno())
    print(json.dumps(kw), flush=True)


def beat(s):
    with open(a.beat + ".tmp", "w") as f:
        f.write(f"{s} {time.time():.0f}\n")
    os.replace(a.beat + ".tmp", a.beat)


def kmd_pci():
    """/dev/tenstorrent id -> PCI address (sysfs, read-only)."""
    out = {}
    for d in glob.glob("/sys/class/tenstorrent/tenstorrent!*"):
        out[int(d.rsplit("!", 1)[1])] = os.path.basename(os.path.realpath(os.path.join(d, "device")))
    return out


class TopUp:
    """The per-chip top-up after the pre-ramp. plan(p0, p1) reads the C8 telemetry of the pre-ramp [p0, p1] and the chips' own limits (read only, never
    written) and picks, by load_line.topup_plan, each chip whose mean power and current sat below 97% of
    its limits; each such chip gets `extra` more 16k bf8 LoFi matmuls per burst on its own 1x1 submesh. after(pos)
    enqueues them while the burst runs (heavy op `pos` of it): no sync and no readback inside a burst; their
    outputs are freed unread; sync() waits for them at the burst end, beside BurstGap's synchronize. Chips at their
    limit get nothing. Any refusal turns it off and says why (the LOAD line's fallback, the PRERAMP line)."""

    S = 16384

    def __init__(self, ttnn, torch, mesh, chips, shape, steps, telem=None, limits=None, gain=1.0, on=True):
        self.ttnn, self.mesh, self.chips, self.shape, self.steps = ttnn, mesh, chips, shape, steps
        self.telem, self.limits, self.gain = telem, limits, gain
        self.plan_, self.subs, self.k, self.n, self.res = [], [], 0, {}, None
        self.why = None if on else "off by default (FLEET_TOPUP=1 turns it on)"
        if self.why is None and not (telem and limits):
            self.why = "no C8 telemetry or limits file given"
        self.host = None
        if self.why is None:
            try:  # the host tensors once, before the load starts: plan() then only uploads them

                def h(seed, scale):
                    torch.manual_seed(seed)
                    return ttnn.from_torch(
                        torch.randn(1, 1, self.S, self.S, dtype=torch.bfloat16) * scale,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                    )

                self.host = (h(2, 1.0), h(3, 0.02))
            except Exception as e:  # noqa: BLE001
                self.why = f"host tensors refused ({type(e).__name__}: {str(e)[:120]})"

    def coord(self, metal):
        """The mesh coordinate (r, c) of a chip's metal id; row-major order when the mesh cannot say."""
        R, C = self.shape
        try:
            for r in range(R):
                for c in range(C):
                    if self.mesh.get_device_id(self.ttnn.MeshCoordinate(r, c)) == metal:
                        return r, c
        except Exception:  # noqa: BLE001
            pass
        i = [ch["metal"] for ch in self.chips].index(metal)
        return i // C, i % C

    def plan(self, p0, p1):
        """Called once, between the pre-ramp's measured part and its warm-up lap: pick the chips, upload their inputs.
        Returns info() for the preramp_end row."""
        if self.why is not None:
            return self.info()
        try:
            import load_line as ll

            self.res, picked = ll.topup_plan(
                ll.c8_rows(self.telem), ll.c8_limits(self.limits), p0, p1, self.steps, self.gain
            )
            if self.res is None:
                self.why = "no C8 samples or limits over the pre-ramp"
                return self.info()
            t, by = self.ttnn, {ll.short(ch["pci"]): ch for ch in self.chips if ch.get("pci")}
            # the parent mesh still has pre-lap ops in flight here; submeshes over the same chips need a
            # barrier between the phases (tt-metal mesh_device.hpp quiesce_devices), once, outside the counted chunks
            self.quiesce()
            ck = t.init_device_compute_kernel_config(
                self.mesh.arch(),
                math_fidelity=t.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            mm = getattr(getattr(t, "experimental", None), "minimal_matmul", None)
            for p in picked:
                ch = by.get(p["bdf"])
                if ch is None:
                    continue
                r, c = self.coord(ch["metal"])
                sub = self.mesh.create_submesh(t.MeshShape(1, 1), t.MeshCoordinate(r, c))
                try:
                    A, B = (t.to_device(x, sub, memory_config=t.DRAM_MEMORY_CONFIG) for x in self.host)
                except Exception:  # noqa: BLE001
                    A, B = (
                        t.from_torch(
                            t.to_torch(x),
                            dtype=t.bfloat8_b,
                            layout=t.TILE_LAYOUT,
                            device=sub,
                            memory_config=t.DRAM_MEMORY_CONFIG,
                            mesh_mapper=t.ReplicateTensorToMesh(sub),
                        )
                        for x in self.host
                    )
                g = sub.compute_with_storage_grid_size()
                cfg = (
                    t.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=16,
                        N_block_size=8,
                        subblock_h=2,
                        subblock_w=4,
                        compute_with_storage_grid_size=g,
                    )
                    if mm is not None
                    else None
                )
                self.subs.append(dict(p, sub=sub, A=A, B=B, ck=ck, cfg=cfg, mm=mm))
            self.sync()  # the uploads done before the warm lap puts parent-mesh work back on the same chips
            self.host = None
        except Exception as e:  # noqa: BLE001
            self.why, self.subs = f"{type(e).__name__}: {str(e)[:160]}", []
        return self.info()

    def op(self, s):
        t = self.ttnn
        if s["mm"] is not None:
            try:
                return s["mm"](s["A"], s["B"], config=s["cfg"], compute_kernel_config=s["ck"])
            except Exception:  # noqa: BLE001
                s["mm"] = None  # refused on the host before anything is queued: plain matmul, same fidelity
        return t.matmul(s["A"], s["B"], compute_kernel_config=s["ck"])

    def after(self, pos):
        """Heavy op `pos` (0-based) of the burst was just enqueued: one more matmul on each chip whose extra > pos."""
        if self.why is not None:
            return
        for s in self.subs:
            if pos < s["extra"]:
                try:
                    self.ttnn.deallocate(self.op(s))
                except Exception as e:  # noqa: BLE001
                    self.why = f"top-up op refused on {s['bdf']} ({type(e).__name__}: {str(e)[:120]})"
                    return
                self.n[self.k] = self.n.get(self.k, 0) + 1

    def quiesce(self):
        """Every queued op of the parent mesh done (quiesce_devices, else synchronize_device)."""
        q = getattr(self.mesh, "quiesce_devices", None)
        q() if q is not None else self.ttnn.synchronize_device(self.mesh)

    def sync(self):
        """The burst end (or a chunk end): wait for the submeshes' queued top-up ops."""
        for s in self.subs:
            self.ttnn.synchronize_device(s["sub"])

    def flops(self, k):
        """Top-up FLOPs of chunk k averaged over every chip (the LOAD line's TFLOP/s per chip)."""
        return 2 * self.S**3 * self.n.get(k, 0) / max(1, len(self.chips))

    def info(self):
        r = self.res or {}
        return dict(
            why=self.why,
            steps=self.steps,
            op="16k bf8 LoFi matmul",
            measured=r.get("chips"),
            at_limit=r.get("at_limit"),
            chips=[
                {k: v for k, v in s.items() if k in ("bdf", "tray", "extra", "power_w", "tdp_w", "current_a", "tdc_a")}
                for s in self.subs
            ],
        )

    def desc(self, k):
        if self.why is not None or not self.subs:
            return None
        return (
            f"top-up {sum(s['extra'] for s in self.subs)} matmuls per burst on {len(self.subs)} chip(s) below their "
            f"limit, {self.n.get(k, 0)} this chunk"
        )


import stop_signal as ss  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

ss.install("LOAD")  # TERM/INT: the finally below closes the mesh
beat("open")
R, C = a.rows_n, a.cols_n
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
cq_why = None
try:
    m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C), num_command_queues=a.cqs) if a.cqs > 1 else None
except Exception as e:  # noqa: BLE001
    m, cq_why = None, f"2 command queues refused ({type(e).__name__}: {str(e)[:120]})"
if m is None:
    m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C))
    cq_why = cq_why or ("one command queue (--cqs 1)" if a.cqs == 1 else None)
CQ1 = (ttnn.QueueId(1) if hasattr(ttnn, "QueueId") else 1) if cq_why is None else None
M, SUBS, tt0 = (
    m,
    [],
    ttnn,
)  # the parent mesh, the tray submeshes (--trays), the real ttnn: closed in this order's reverse
LIN = ttnn.Topology.Linear
DRAM = ttnn.DRAM_MEMORY_CONFIG
gate = cs.Gate()
bad = nerr = 0
try:
    ss.arm()
    ids = list(m.get_device_ids())
    try:
        from numeric_check import umd_map_from_stdout

        umd = umd_map_from_stdout()
    except Exception:  # noqa: BLE001
        umd = {}
    pcis = kmd_pci()
    chipinfo = []
    for mid in ids:
        dev = umd.get(mid, mid) if umd else mid
        pci = pcis.get(dev)
        chipinfo.append(dict(metal=mid, dev=dev, pci=pci, tray=cs.tray_of(pci) if pci else None))
    if TRAYS:  # one submesh per chosen tray (its block in the mesh map); the load runs on them only
        by = {c["metal"]: c for c in chipinfo}
        cmap = {}
        for r in range(R):
            for c in range(C):
                try:
                    mid = m.get_device_id(ttnn.MeshCoordinate(r, c))
                except Exception:  # noqa: BLE001
                    mid = ids[r * C + c]  # row-major, as TopUp.coord
                cmap[(r, c)] = by.get(mid, {}).get("tray")
        try:
            boxes = tl.tray_boxes(cmap, TRAYS)
        except ValueError as e:
            row(event="trays_refused", trays=TRAYS, error=str(e))
            sys.exit(1)
        for t in TRAYS:
            r0, c0, h, w = boxes[t]
            SUBS.append(m.create_submesh(ttnn.MeshShape(h, w), ttnn.MeshCoordinate(r0, c0)))
        R, C = boxes[TRAYS[0]][2:]
        m, ttnn = tl.Lane(SUBS), tl.Lanes(tt0)  # every op below runs once per tray submesh
        ids = list(m.get_device_ids())
        chipinfo = [by[i] for i in ids]
        print(tl.describe(TRAYS, chipinfo), flush=True)
    g = m.compute_with_storage_grid_size()
    GRID = [int(g.x), int(g.y)]  # loc_probe's grid in its own name; never rebound below
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    sem = lambda: ttnn.create_global_semaphore(m, cores, 0)  # noqa: E731
    ag_s = {ax: [sem() for _ in range(4)] for ax in (0, 1)}
    rs_s = {ax: [sem() for _ in range(6)] for ax in (0, 1)}
    bar_s = {ax: [sem() for _ in range(2)] for ax in (0, 1)}
    ttnn.synchronize_device(m)
    idx = {"ag": [0, 0], "rs": [0, 0], "bar": [0, 0]}

    def nxt(kind, ax, pool, n):
        i = idx[kind][ax]
        idx[kind][ax] = (i + 1) % 2
        return pool[ax][i * n : (i + 1) * n] if n > 1 else pool[ax][i]

    def ag(t, dim, ax):
        hp = (
            {"chunks_per_sync": 16, "num_workers_per_link": 3, "num_buffers_per_channel": 2}
            if t.shape[2] > 512
            else {"chunks_per_sync": 10, "num_workers_per_link": 2, "num_buffers_per_channel": 2}
        )
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=nxt("ag", ax, ag_s, 2),
            barrier_semaphore=nxt("bar", ax, bar_s, 1),
            num_links=a.links,
            memory_config=DRAM,
            topology=LIN,
            cluster_axis=ax,
            **hp,
        )

    def rs(t, dim, ax):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=nxt("rs", ax, rs_s, 3),
            barrier_semaphore=nxt("bar", ax, bar_s, 1),
            num_links=a.links,
            memory_config=DRAM,
            topology=LIN,
            cluster_axis=ax,
            chunks_per_sync=2,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    torch.manual_seed(0)

    def put(shape, scale=0.1, rep=False, lo=None):
        if rep:
            full = torch.rand(*shape) + lo if lo is not None else torch.randn(*shape) * scale
            mapper = ttnn.ReplicateTensorToMesh(m)
        else:
            full = torch.randn(R, C, *shape[2:]) * scale
            mapper = ttnn.ShardTensor2dMesh(m, mesh_shape=(R, C), dims=(0, 1))
        return ttnn.from_torch(
            full.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=m,
            memory_config=DRAM,
            mesh_mapper=mapper,
        )

    W = 2048
    dram_h = max(32, (a.dram_mb << 20) // (2 * W) // 32 * 32)
    pcie_h = max(32, (a.pcie_mb << 20) // (2 * W) // 32 * 32)

    def put_pat(shape, salt):
        """link data: chip (r, c) gets its own host-built bit pattern; returns the tensor and the patterns"""
        pats = [[lp.pattern(4802 + salt, r * C + c, shape[2], shape[3]) for c in range(C)] for r in range(R)]
        full = torch.stack([torch.stack([lp.to_torch(q) for q in pr]) for pr in pats])
        return (
            ttnn.from_torch(
                full,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=m,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensor2dMesh(m, mesh_shape=(R, C), dims=(0, 1)),
            ),
            pats,
        )

    T, LP = {}, {}
    a.sp_w, H3_N = h3_dims(a.sp_w, R)
    if "h3" in parts:
        (T["x"], LP["x"]), (T["s"], LP["s"]) = put_pat([1, 1, a.sp_h, a.sp_w], 0), put_pat([1, 1, 32, 1152], 1000)
        T.update(w1=put([1, 1, a.sp_w, 5376], 0.03), w2=put([1, 1, 5376, H3_N], 0.015))
    if "mm" in parts:
        T.update(xr=put([1, 1, 1024, W], 1.0, rep=True), wr=put([1, 1, W, W], 0.02, rep=True))
    if "sfpu" in parts:
        T.update(xs=put([1, 1, 1024, 1024], rep=True, lo=0.5))  # in [0.5, 1.5): exp/sqrt/recip stay finite
    if "dram" in parts:
        T.update(da=put([1, 1, dram_h, W], 1.0, rep=True), db=put([1, 1, dram_h, W], 1.0, rep=True))
    if "pcie" in parts:
        hp_t = (torch.randn(1, 1, pcie_h, W)).to(torch.bfloat16)
        T["hp"] = ttnn.from_torch(
            hp_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(m)
        )
        T["pd"] = put([1, 1, pcie_h, W], 1.0, rep=True)
        T["pw"] = put([1, 1, pcie_h, W], 1.0, rep=True)  # later launches write here, so ne() has a real reference
    REPL = {"heavy", "mm", "sfpu", "dram", "pcie"}  # parts whose output is the same on every chip

    def ne16(o, r):
        """ne(o, r) as a bfloat16 0/1 tensor, so the bf16 counters add it: float32 and int32 outputs give
        their own dtype's 0/1, typecast here; bf16 and bf8 outputs are as before."""
        d = ttnn.ne(o, r, memory_config=DRAM)
        if d.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            d2 = ttnn.typecast(d, ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(d)
            d = d2
        return d

    ra = RunAhead(ttnn, m, RUN_AHEAD_OPS, 0)
    hv, HVS = None, []
    if "heavy" in parts:
        try:
            # (op, FLOPs per op, description) per --heavy op; op.state["mm"] None = fell back. hv: the first, HVS: all
            HVS = [
                heavy_op(
                    argparse.Namespace(**{**vars(a), "heavy": h}), ttnn, torch, m, None, None, cq1=CQ1, cq_why=cq_why
                )
                for h in HEAVY_OPS
            ]
            hv = (HVS[0][0], sum(x[1] for x in HVS), " then ".join(x[2] for x in HVS), [x[0] for x in HVS])
        except Exception as e:  # noqa: BLE001
            parts.remove("heavy")
            row(event="part_off", part="heavy", error=f"{type(e).__name__}: {str(e)[:300]}")
    NMAIN = len(HEAVY_OPS)  # --heavy's ops; the --heavy-x ops that pass their trial follow them in HEAVY_OPS and hv[3]
    if "heavy" in parts and a.heavy_x:
        for h in a.heavy_x.split(","):
            if h in HEAVY_OPS:
                continue
            x = o = d = z = s0 = None
            try:  # one extra format this ttnn or chip refuses is turned off alone, never the heavy part
                x = heavy_op(
                    argparse.Namespace(**{**vars(a), "heavy": h}), ttnn, torch, m, None, None, cq1=CQ1, cq_why=cq_why
                )
                o = x[0]()
                d = ne16(o, o)
                z = ttnn.zeros_like(d)  # one trial launch plus the check the chunks run on it
                s0 = ttnn.add(z, d, memory_config=DRAM)
                ttnn.synchronize_device(m)
                for t in (o, d, z, s0):
                    ttnn.deallocate(t)
            except Exception as e:  # noqa: BLE001
                ttnn.synchronize_device(m)  # a device fault here is not a refused op: it ends the run
                row(event="part_off", part=f"heavy:{h}", error=f"{type(e).__name__}: {str(e)[:300]}")
                # free the refused op's uploaded inputs and any trial output (up to ~768 MiB per chip)
                for t in (*(getattr(x[0], "tensors", ()) if x else ()), o, d, z, s0):
                    if t is not None:
                        try:
                            ttnn.deallocate(t)
                        except Exception:  # noqa: BLE001
                            pass
                continue
            HVS.append(x)
            HEAVY_OPS.append(h)
            try:  # each extra's own rate (the chunks log only the main ops' TFLOP/s): median of 3 warm runs
                ts, o = [], None  # o: the trial's output is already freed above
                for _ in range(3):
                    ttnn.synchronize_device(m)
                    t0 = time.monotonic()
                    o = x[0]()
                    ttnn.synchronize_device(m)
                    ts.append(time.monotonic() - t0)
                    ttnn.deallocate(o)
                    o = None
                dt = sorted(ts)[1]
                row(
                    event="heavy_x_rate",
                    op=h,
                    s=round(dt, 4),
                    tflops_chip=round(x[1] / dt / 1e12, 2),
                    dram_gbs_chip=round(
                        heavy_dram_bytes(h, int8=(getattr(x[0], "state", None) or {}).get("int8", True)) / dt / 1e9, 1
                    ),
                )
            except Exception as e:  # noqa: BLE001
                ttnn.synchronize_device(m)
                if o is not None:  # free a timed run's output the way the success path does
                    try:
                        ttnn.deallocate(o)
                    except Exception:  # noqa: BLE001
                        pass
                row(event="heavy_x_rate", op=h, error=f"{type(e).__name__}: {str(e)[:300]}")
        hv = (
            hv[0],
            hv[1],
            " then ".join(x[2] for x in HVS[:NMAIN])
            + (
                f"; in each chunk's last {a.heavy_x_s:g} s also " + " then ".join(x[2] for x in HVS[NMAIN:])
                if len(HVS) > NMAIN
                else ""
            ),
            [x[0] for x in HVS],
        )
    XT = {
        "end": None,
        "dur": 0.0,
        "on": True,
    }  # --heavy-x window: launch 0 (references) runs the extras, calib does not

    def free(*ts):
        for t in ts:
            if t is not None:
                ttnn.deallocate(t)

    def p_h3():
        outs = []
        for b in range(a.blocks):
            gate.enqueue("h3")
            a1 = ag(T["x"], 2, 1)
            h = None
            for _ in range(a.mm):
                free(h)
                h = ttnn.matmul(a1, T["w1"], memory_config=DRAM)
            hg = ttnn.gelu(h, memory_config=DRAM)
            free(h)
            y = ttnn.matmul(hg, T["w2"], memory_config=DRAM)
            free(hg)
            r = rs(y, 3, 0)
            free(y)
            z = ag(T["s"], 2, 0)
            r2 = ttnn.add(r, a1, memory_config=DRAM)
            free(r)
            if b < a.blocks - 1:
                free(a1)
            n = ttnn.rms_norm(r2, epsilon=1e-6, memory_config=DRAM)
            free(r2)
            u = ttnn.untilize(n, memory_config=DRAM)
            free(n)
            o = ttnn.tilize(u, memory_config=DRAM)
            free(u)
            if b < a.blocks - 1:
                free(o, z)
            else:
                outs = [o, z, a1]  # z and a1: the all-gathered link data, checked bit for bit at launch 0
        return outs

    def p_mm():
        gate.enqueue("mm")
        return [ttnn.matmul(T["xr"], T["wr"], memory_config=DRAM)]

    def p_sfpu():
        o = None
        for _ in range(a.sfpu_reps):
            gate.enqueue("sfpu")
            free(o)
            e = ttnn.exp(T["xs"], memory_config=DRAM)
            q = ttnn.sqrt(e, memory_config=DRAM)
            free(e)
            r = ttnn.reciprocal(q, memory_config=DRAM)
            free(q)
            o = ttnn.gelu(r, memory_config=DRAM)
            free(r)
        return [o]

    def p_dram():
        o = None
        for _ in range(a.dram_reps):
            gate.enqueue("dram")
            free(o)
            o = ttnn.add(T["da"], T["db"], memory_config=DRAM)
        return [o]

    def p_pcie():
        gate.enqueue("pcie")
        if "pcie" not in ref:  # launch 0, on cq 0: fills the reference T["pd"] and T["pw"] alike, so a cq 1 write still
            ttnn.copy_host_to_device_tensor(
                T["hp"], T["pw"]
            )  # in flight while cq 0 checks T["pw"] rewrites the same bytes
            ttnn.copy_host_to_device_tensor(T["hp"], T["pd"])
            return [T["pd"]]
        if CQ1 is not None:
            try:
                ttnn.copy_host_to_device_tensor(
                    T["hp"], T["pw"], cq_id=CQ1
                )  # beside cq 0, never waited on before the chunk end
                return [T["pw"]]
            except Exception as e:  # noqa: BLE001
                fell["pcie"] = f"cq 1 write refused, pcie on cq 0 ({type(e).__name__}: {str(e)[:120]})"
                globals()["CQ1"] = None
        ttnn.copy_host_to_device_tensor(T["hp"], T["pw"])
        return [T["pw"]]

    gap, nhv = BurstGap(ttnn, m, a.burst_steps, a.burst_gap_ms, ra), [0]
    # the per-chip top-up: only with a pre-ramp to measure, and only with bursts (their end is its sync)
    tu = (
        TopUp(
            ttnn,
            torch,
            m,
            chipinfo,
            (R, C),
            a.burst_steps,
            a.topup_telem,
            a.topup_limits,
            a.topup_gain,
            on=os.environ.get("FLEET_TOPUP", "0") == "1",
        )
        if a.pre_ramp_s > 0 and "heavy" in parts
        else None
    )
    if tu is not None and tu.why is None and not gap.on():
        tu.why = "no burst gap (--burst-gap-ms 0): no burst end to wait for its ops at"
    if tu is not None and TRAYS:
        tu.why = "off under --trays (its 1x1 submeshes hang off the parent mesh)"
    TU[:] = [tu] if tu is not None else []

    def p_heavy():
        outs = []
        for j, op in enumerate(
            hv[3] if len(hv) > 3 else [hv[0]]
        ):  # each --heavy op heavy_n times, back to back; one output per op, each checked against its own reference
            outs.append(
                p_heavy1(op, j) if j < NMAIN or XT["on"] else None
            )  # --heavy-x ops: only in their window (None: not run)
        return outs

    def p_heavy1(op, j=0):
        o = None
        for i in range(a.heavy_n):
            gate.enqueue("heavy")
            n = op()
            free(o)
            o = n
            if HEAVY_CHECKS > 1 and i < a.heavy_n - 1 and "heavy" in acc:  # every output on the device, no
                check1(
                    "heavy", j, o
                )  # readback or sync; the last goes through check() as before; launch 0 makes the reference
            ra.queued(0)  # an event after each heavy op; the host waits on the one RUN_AHEAD_OPS back
            if tu is not None:
                tu.after(nhv[0] % a.burst_steps if a.burst_steps > 0 else 0)  # chips below their limit: more, no sync
            nhv[0] += 1
            if tu is not None and gap.on() and nhv[0] % a.burst_steps == 0:
                tu.sync()  # the burst end: the top-up's ops too, beside gap.after's synchronize
            if gap.after(nhv[0]):  # the end of a burst: synchronize, then --burst-gap-ms of quiet
                first_bad_probe()  # the device is synchronized and idle here: the heavy counters, at most every 30 s
        return o

    fbp = cs.FirstBad(cs.FIRST_BAD_PROBE_S)  # when each chip's first wrong heavy output happened

    def first_bad_probe():
        """In a burst gap only (the device already synchronized), at most every FIRST_BAD_PROBE_S s: each heavy counter's
        max per chip (verify's two on-device reductions), read back. A chip whose counter is above 0 for the first time
        gets its first wrong output bounded by this read and the one before it, logged at once by BDF and tray. The chunk
        is time-bounded, so the read adds no seconds; it never zeroes a counter, never fails the chunk, and a read that
        raises turns the reads off (the chunk-end verify still bounds every chip)."""
        now = time.time()
        if HEAVY_CHECKS <= 1 or "heavy" not in acc or not fbp.due(now):
            return
        bad = {}
        try:
            for j, t in enumerate(acc["heavy"]):
                rt, _ = reduce(t, False)
                for i, v in enumerate(to_host(rt)):
                    if v.numel() and v.max().item() != 0 and str(i) not in bad:
                        bad[str(i)] = HEAVY_OPS[j] if len(acc["heavy"]) > 1 else None
                free(rt)
        except Exception as e:  # noqa: BLE001
            fbp.fail(f"{type(e).__name__}: {str(e)[:120]}")
            return
        t1 = time.time()
        for i in fbp.saw(t1, nl.get(fbp.k, 0), bad, t1 - now):
            c, f = chipinfo[int(i)], fbp.found[i]
            print(
                f"{time.strftime('%F %T')} heavy first wrong output: chip {i} {c.get('pci') or '?'} (tray {c.get('tray') or '?'})"
                f" between {f['s0']:.0f} s and {f['s1']:.0f} s into chunk {fbp.k + 1} (launches {f['l0']}-{f['l1']}"
                + (f", {f['op']}" if f.get("op") else "")
                + ")",
                flush=True,
            )

    fell = {}
    FN = {"heavy": p_heavy, "h3": p_h3, "mm": p_mm, "sfpu": p_sfpu, "dram": p_dram, "pcie": p_pcie}
    ref, acc = {}, {}
    # the in-run capture (numeric_capture.py); off under --trays/--shape (lanes, not one mesh) and without heavy
    CAP = [nc.Ring(a.capture_ring)] if a.capture_ring > 0 and "heavy" in parts and not TRAYS and SH is None else []
    cap_prev, cap_n = {}, [0]  # (op, chip) -> counter at the last poll; events captured by this load (at most 4)

    def zeros_like(t):
        """A bf16 counter shaped like t, made on the device (heavy's 16k x 16k from the host would be 16 GB a chunk end)."""
        try:
            return ttnn.zeros(list(t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=m, memory_config=DRAM)
        except Exception:  # noqa: BLE001
            pass
        return ttnn.from_torch(
            torch.zeros(list(t.shape), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=m,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(m),
        )

    def to_host(t):
        return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]

    SV = dict(on=False, rot=0, n={}, h=None, w=None, dropped=False)  # the heavy SFPU vote, set up after launch 0

    def sv_label(j, s):
        return f"{HEAVY_OPS[j]}/{nck.SFPU_OPS[s]}" if len(HEAVY_OPS) > 1 else nck.SFPU_OPS[s]

    def sv_op(o, s):
        """SFPU op s on o's first h x w block, height-sharded one 32-row shard per core; the result back in DRAM. Each device
        op goes through gate.enqueue like the load parts', so none of them runs inside a quiet window."""
        op = nck.SFPU_OPS[s]
        gate.enqueue("check sfpu-vote:slice")
        x = ttnn.slice(o, [0, 0, 0, 0], [1, 1, SV["h"], SV["w"]])
        if x.dtype != ttnn.bfloat16:
            gate.enqueue("check sfpu-vote:typecast")
            y = ttnn.typecast(x, ttnn.bfloat16)
            free(x)
            x = y
        xs = ttnn.to_memory_config(x, SV["mc"])
        free(x)
        gate.enqueue(f"check sfpu-vote:{op}")
        if op == "sqrt":
            t = ttnn.abs(xs, memory_config=SV["mc"])
            y = ttnn.sqrt(t, memory_config=SV["mc"])
            free(t)
        else:
            y = getattr(ttnn, op)(xs, memory_config=SV["mc"])
        free(xs)
        yd = ttnn.to_memory_config(y, DRAM)
        free(y)
        return yd

    def sv_check(j, o):
        """On the device: this launch's SFPU op on heavy output j, acc += ne(y, its reference). No readback. An error
        turns the vote off (a row says so); the load goes on, and the chunk-end verify still reads the counters gathered
        so far (SV["dropped"] says it must). A QuietViolation is the gate's, not the vote's: it propagates like the
        parts' own. NaN: ne(NaN, NaN) counts, so a NaN in a heavy output mismatches here on every chip; it is not masked,
        because the heavy check of the same output (check1, just before) already counts it as non-finite and fails it.
        """
        s = SV["rot"]
        try:
            y = sv_op(o, s)
            gate.enqueue("check sfpu-vote:ne")
            d = ttnn.ne(y, SV["ref"][j][s], memory_config=DRAM)
            t = ttnn.add(SV["acc"][j][s], d, memory_config=DRAM)
            free(y, d, SV["acc"][j][s])
            SV["acc"][j][s] = t
            SV["n"][(j, s)] = SV["n"].get((j, s), 0) + 1
        except cs.QuietViolation:
            raise
        except Exception as e:  # noqa: BLE001
            SV["on"] = False
            SV["dropped"] = bool(SV["n"])
            row(event="part_off", part="sfpu-vote", error=f"{type(e).__name__}: {str(e)[:300]}")

    # launch 0: compiles every kernel; each part's outputs become its on-device reference. A part that raises here
    # (an op this ttnn lacks) is turned off; its partial work is drained first.
    beat("launch 0")
    t0 = time.time()
    for p in list(parts):
        try:
            ref[p] = FN[p]()
            ttnn.synchronize_device(m)
            acc[p] = [zeros_like(t) for t in ref[p]]
            d = ne16(ref[p][0], ref[p][0])
            s = ttnn.add(acc[p][0], d, memory_config=DRAM)
            ttnn.synchronize_device(m)
            free(d, s)
        except Exception as e:  # noqa: BLE001
            parts.remove(p)
            ref.pop(p, None)
            acc.pop(p, None)
            ttnn.synchronize_device(m)  # a device fault here is not a missing op: it ends the run
            row(event="part_off", part=p, error=f"{type(e).__name__}: {str(e)[:300]}")
    beat("done 0")
    if not parts:
        raise RuntimeError("no load part could run")

    # the references, once: non-finite elements per chip, and for the replicated parts a cross-chip vote (a chip
    # wrong from launch 0 on would otherwise be its own reference). Written as verify row k = -1.
    v0 = time.time()
    res0 = {str(i): dict(c, launches=1, mismatched=0, nonfinite=0, bad_parts=[]) for i, c in enumerate(chipinfo)}
    for p in parts:
        for j, t in enumerate(ref[p]):
            # heavy: a 16k x 16k output per chip is too big for the host; its first 64 rows (every column) are read
            sl = ttnn.slice(t, [0, 0, 0, 0], [1, 1, 64, t.shape[3]]) if p == "heavy" else None
            hs = to_host(sl if sl is not None else t)
            free(sl)
            keys = [hash(v.numpy().tobytes()) for v in hs]
            top = max(set(keys), key=keys.count)
            for i, v in enumerate(hs):
                r = res0[str(i)]
                nf = int((~torch.isfinite(v)).sum().item())
                r["nonfinite"] += nf
                odd = p in REPL and keys[i] != top and keys.count(top) > len(keys) // 2
                if (nf or odd) and f"{p}-ref" not in r["bad_parts"]:
                    r["bad_parts"].append(f"{p}-ref")
                if (nf or odd) and p == "heavy" and len(ref[p]) > 1:
                    r["bad_parts"].append(f"heavy-ref:{HEAVY_OPS[j]}")  # which --heavy op's reference
    # SFPU ops in the heavy vote: launch 0's heavy outputs through each SFPU op make that op's
    # on-device reference, voted across chips bit for bit here; later launches compare on the device (sv_check, verify)
    SV.update(gx=g.x, gy=g.y)
    if a.sfpu_vote and "heavy" in parts:
        try:
            SV["h"], SV["w"] = nck.sfpu_layout(g.x, g.y)
            if SV["h"] > ref["heavy"][0].shape[2] or SV["w"] > ref["heavy"][0].shape[3]:
                raise ValueError(
                    f"heavy output {list(ref['heavy'][0].shape)} smaller than the SFPU block {SV['h']}x{SV['w']}"
                )
            SV["mc"] = ttnn.create_sharded_memory_config(
                shape=(1, 1, SV["h"], SV["w"]),
                core_grid=ttnn.CoreGrid(y=g.y, x=g.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            # the vote covers --heavy's main ops only; N5's --heavy-x extra formats (int32 among them) stay out of it
            SV["ref"] = [[sv_op(t, s) for s in range(len(nck.SFPU_OPS))] for t in ref["heavy"][:NMAIN]]
            SV["acc"] = [[zeros_like(t) for t in svr] for svr in SV["ref"]]
            d = ttnn.ne(SV["ref"][0][0], SV["ref"][0][0], memory_config=DRAM)
            s_ = ttnn.add(SV["acc"][0][0], d, memory_config=DRAM)
            ttnn.synchronize_device(m)
            free(d, s_)
            for j, svr in enumerate(SV["ref"]):
                for s, t in enumerate(svr):
                    bad = nck.sfpu_vote(torch, [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], g.x)
                    for i, b in bad.items():
                        r = res0[str(i)]
                        for bp in ("sfpu-vote-ref", f"sfpu-vote-ref:{sv_label(j, s)}"):
                            if bp not in r["bad_parts"]:
                                r["bad_parts"].append(bp)
                        r.setdefault("sfpu", {})[sv_label(j, s)] = b["cores"]
            SV["on"] = True
        except Exception as e:  # noqa: BLE001
            SV["on"] = False
            ttnn.synchronize_device(m)
            row(event="part_off", part="sfpu-vote", error=f"{type(e).__name__}: {str(e)[:300]}")
    row(
        event="setup",
        mesh=[R, C],
        sfpu_vote=(dict(ops=list(nck.SFPU_OPS), grid=[g.x, g.y], block=[SV["h"], SV["w"]]) if SV["on"] else None),
        fabric="FABRIC_1D",
        topology="Linear",
        links=a.links,
        mm=a.mm,
        blocks=a.blocks,
        parts=parts,
        cqs=(2 if CQ1 is not None else 1),
        chunks=chunks_s,
        gap_s=a.gap_s,
        lead_s=lead,
        dram_shape=[dram_h, W],
        pcie_shape=[pcie_h, W],
        heavy=(hv[2] if hv and "heavy" in parts else None),
        heavy_n=a.heavy_n,
        heavy_ops=(HEAVY_OPS if hv and "heavy" in parts else None),
        heavy_checks=HEAVY_CHECKS,
        heavy_x=(HEAVY_OPS[NMAIN:] if hv and "heavy" in parts else None),
        heavy_x_s=a.heavy_x_s,
        sfpu_reps=a.sfpu_reps,
        dram_reps=a.dram_reps,
        mesh_ids=ids,
        chips=chipinfo,
        map_source="umd_log" if umd else "assumed_identity",
        visible=os.environ.get("TT_VISIBLE_DEVICES", ""),
        warm_s=round(time.time() - t0, 1),
        **({"trays": TRAYS} if TRAYS else {}),
        **({"variant": VAR, "variant_warning": lv.warning(VAR), "host_stress_n": HS.n if HS.on else 0} if VAR else {}),
    )
    row(event="verify", k=-1, start=round(v0, 3), end=round(time.time(), 3), chips=res0)
    # link data: launch 0's all-gathered z (axis 0) and a1 (axis 1) on every chip, bit for bit
    # against the host-built patterns of the chips that sent them; later launches are compared on the device with these
    LD = 0
    if "h3" in parts:
        for name, t, src, ax in (("z", ref["h3"][1], LP["s"], 0), ("a1", ref["h3"][2], LP["x"], 1)):
            hh = src[0][0].shape[0]
            for i, d in enumerate(ttnn.get_device_tensors(t)):
                rr, cc = divmod(i, C)
                got = lp.from_torch(ttnn.to_torch(d))
                for k, (sr, sc) in enumerate([(q, cc) for q in range(R)] if ax == 0 else [(rr, q) for q in range(C)]):
                    df = lp.diff(got[k * hh : (k + 1) * hh], src[sr][sc])
                    if df:
                        LD += 1
                        si = sr * C + sc
                        row(
                            event="linkdata",
                            k=-1,
                            op=f"h3 {name} all-gather axis {ax}",
                            send=si,
                            recv=i,
                            send_bdf=chipinfo[si].get("pci"),
                            recv_bdf=chipinfo[i].get("pci"),
                            link=si != i,
                            detail=lp.fmt(df),
                            **df,
                        )
                del got

    def check1(p, j, o):
        """On the device: acc += ne(o, ref), output j of part p. No readback; o is not freed."""
        gate.enqueue("check " + p)
        d = ne16(o, ref[p][j])
        s = ttnn.add(acc[p][j], d, memory_config=DRAM)
        free(d, acc[p][j])
        acc[p][j] = s
        if p == "heavy" and SV["on"] and j < NMAIN:  # not N5's --heavy-x extra formats
            sv_check(j, o)

    def check(p, outs):
        """On the device: acc += ne(out, ref), per output of part p. No readback. None: a --heavy-x op outside its window."""
        for j, o in enumerate(outs):
            if o is None:
                continue
            check1(p, j, o)
            if o is not T.get("pd") and o is not T.get("pw"):
                free(*(CAP[0].hold(j, nlaunch[0], o) if CAP and p == "heavy" else [o]))

    nlaunch = [0]

    sh_t0, sh_n = [None], {}  # --shape: the load's start (first launch), per tray launches at hi

    def launch():
        nb = 0
        nlaunch[0] += 1
        if SV["on"]:
            SV["rot"] = nlaunch[0] % len(nck.SFPU_OPS)  # one SFPU op per launch, in turn
        on = None
        if SH is not None:  # this launch's parts per tray, from the shape's state at this moment
            sh_t0[0] = sh_t0[0] or time.time()
            ts = time.time() - sh_t0[0]
            for tr, x in SH.state(ts).items():
                if x == "hi":
                    sh_n[tr] = sh_n.get(tr, 0) + 1
            on = SH.parts(ts, parts)
        for p in parts:
            if p == "pcie" and nlaunch[0] % max(a.pcie_every, 1):
                continue  # --pcie-every: this launch has no PCIe write
            if on is None:
                check(p, FN[p]())
            elif p in on:  # None: every loaded tray; else these trays' lanes only (lane order = TRAYS order)
                with tl.only(None if on[p] is None else [TRAYS.index(t) for t in on[p]]):
                    check(p, FN[p]())
            else:
                continue
            nb += a.blocks if p == "h3" else 0
        return nb

    # per-unit time share: each part alone (with its check), synchronized, 3 times; the measured seconds per launch
    calib = {}
    XT["on"] = False  # calib and the chunks run the main ops; the --heavy-x window turns them back on (step())
    for p in parts:
        ts = []
        for _ in range(3):
            c0 = time.time()
            check(p, FN[p]())
            ttnn.synchronize_device(m)
            ts.append(time.time() - c0)
        calib[p] = round(min(ts), 4)
    row(event="calib", part_s=calib, launch_s_sum=round(sum(calib.values()), 4))

    # per-launch work per chip, from the tensor shapes (used with the measured launch rate)
    MB = 1e-9
    sp_rows = a.sp_h * C
    fl = {
        "heavy": a.heavy_n * hv[1] if hv else 0,
        "h3": a.blocks * (a.mm * 2 * sp_rows * a.sp_w * 5376 + 2 * sp_rows * 5376 * H3_N),
        "mm": 2 * 1024 * W * W,
        "sfpu": 0,
        "dram": 0,
        "pcie": 0,
    }
    by = 2  # bf16
    S = 16384

    def hv_b(h):  # one op's DRAM bytes (ub0_repro.heavy_dram_bytes; the eltwise op's int8 round trip as it ran)
        i = HEAVY_OPS.index(h)
        st = (getattr(HVS[i][0], "state", None) if i < len(HVS) else None) or {}
        return heavy_dram_bytes(h, int8=st.get("int8", True))

    # --heavy's ops only: the --heavy-x ops run in each chunk's last --heavy-x-s seconds, not in every launch
    dram_b = {
        "heavy": sum(a.heavy_n * hv_b(h) for h in HEAVY_OPS[:NMAIN]),
        "h3": a.blocks
        * by
        * (
            a.mm * (sp_rows * a.sp_w + a.sp_w * 5376 + sp_rows * 5376)
            + 2 * sp_rows * 5376
            + 5376 * H3_N
            + sp_rows * H3_N
            + 6 * sp_rows * a.sp_w // R
        ),
        "mm": by * (1024 * W + W * W + 1024 * W),
        "sfpu": a.sfpu_reps * 8 * by * 1024 * 1024,
        "dram": a.dram_reps * 3 * by * dram_h * W,
        "pcie": by * pcie_h * W / max(a.pcie_every, 1),
    }
    eth_b = {
        "h3": a.blocks * by * (sp_rows - a.sp_h) * a.sp_w
        + a.blocks * by * (sp_rows * H3_N * (R - 1) // R)
        + a.blocks * by * 32 * 1152 * (R - 1)
    }
    pcie_b = by * pcie_h * W / max(a.pcie_every, 1) if "pcie" in parts else 0  # per launch, averaged over --pcie-every

    def util(k, n, blocks, secs):
        if secs <= 0:
            return {}
        f = (sum(fl[p] for p in parts) * n + (tu.flops(k) if tu is not None else 0)) / secs
        return dict(
            launch_per_s=round(n / secs, 3),
            tflops_chip=round(f * 1e-12, 2),
            dram_gbs_chip=round(sum(dram_b[p] for p in parts) * n / secs * MB, 2),
            eth_gbs_chip=round(eth_b.get("h3", 0) * n / secs * MB if "h3" in parts else 0, 2),
            pcie_gbs_chip=round(pcie_b * n / secs * MB, 3),
            pcie_gbs_host=round(pcie_b * len(ids) * n / secs * MB, 2),
            eth_links_used=(f"{a.links} per neighbour pair, both axes" if "h3" in parts else "none"),
        )

    def reduce(t, full_ok=True):
        """Chunk end only: the counter's max over each chip (two max reductions on the device); full tensor if not
        (never for heavy's 16k x 16k counter: that raises, so the part is reported not checked)."""
        try:
            m1 = ttnn.max(t, dim=3, keepdim=True)
            m2 = ttnn.max(m1, dim=2, keepdim=True)
            free(m1)
            return m2, True
        except Exception:  # noqa: BLE001
            if not full_ok:
                raise
            return t, False

    def probe_err(p, res):
        """Only when part p mismatched on some chip (a healthy verify never gets here, no extra time): one more launch of
        p, read back and compared with the reference on the host, gives each mismatching chip's max abs err, wrong
        elements and non-finite flag. The counters hold only counts, so it is a re-launch after the first
        mismatch, not that launch itself; the first part that measures a chip is kept. Never fails the verify."""
        bad = [i for i, r in res.items() if p in r["bad_parts"] and r.get("wrong_elems") is None]
        if not bad or p == "heavy":  # heavy's 16k x 16k output is too big for the host
            return
        try:
            outs = FN[p]()
            ttnn.synchronize_device(m)
            for j, o in enumerate(outs):
                got, want = to_host(o), to_host(ref[p][j])
                for i in bad:
                    w, n, nf = cs.err_stats_torch(got[int(i)], want[int(i)])
                    r = res[i]
                    if n and (r.get("wrong_elems") is None):
                        r.update(max_abs_err=w, wrong_elems=n, nonfinite_seen=nf, err_part=p)
            free(*[o for o in outs if o is not T.get("pd") and o is not T.get("pw")])
        except Exception as e:  # noqa: BLE001
            for i in bad:
                res[i].setdefault("err_probe_error", f"{p}: {type(e).__name__}: {str(e)[:120]}")

    def phys(x, y):
        """the translated worker core of logical core (x, y) (worker_core_from_logical_core; one tray lane's).
        A translated coordinate, the same on every chip, not the chip's die location."""
        tt = sys.modules["ttnn"]  # the module itself: ttnn here may be the tray lanes' stand-in
        w = m.worker_core_from_logical_core(tt.CoreCoord(x, y))
        w = getattr(w, "v", w)
        w = w[0] if isinstance(w, list) else w
        return w.x, w.y

    def loc_probe(p, t, res, j=0):
        """at the chunk-end verify only, for each chip whose part p counter was non-zero, read that
        chip's 2-D mismatch counter (heavy: the 16k x 16k one, before it is zeroed) or re-read the probe's got/want, and
        record the first wrong element's row/col, tile, in-tile element and owning compute core on the heavy matmul's
        GRID grid (and its translated worker core) plus a per-core and per-element histogram (res[i]["loc"]; no DRAM bank, an output tile
        says nothing about DRAM). Read-only on the device, no launch, never inside a burst, never a reset/remove/rescan
        (also none in hold mode), never fails the verify. At most 6 chips."""
        for i in [i for i, r in res.items() if p in r["bad_parts"] and not r.get("loc")][:6]:
            try:
                c = ttnn.to_torch(ttnn.get_device_tensors(t)[int(i)]).reshape(-1, t.shape[-1])
                nz = torch.nonzero(c)
                h = HEAVY_OPS[j] if p == "heavy" and j < len(HEAVY_OPS) else None
                res[i]["loc"] = cs.add_phys(
                    cs.loc_summary(
                        nz[:, 0].tolist(),
                        nz[:, 1].tolist(),
                        int(c.shape[0]),
                        int(c.shape[1]),
                        GRID,
                        eltwise=h in HEAVY_ELTWISE,
                    ),
                    phys,
                )
                if h is not None and len(HEAVY_OPS) > 1:
                    res[i]["loc"]["op"] = h
            except Exception as e:  # noqa: BLE001
                res[i].setdefault("loc_error", f"{p}: {type(e).__name__}: {str(e)[:120]}")

    def verify(k):
        """Chunk end, before the quiet: each part's counters, reduced on the device, read back per chip."""
        res = {
            str(i): dict(c, launches=nl.get(k, 0), mismatched=0, nonfinite=0, bad_parts=[])
            for i, c in enumerate(chipinfo)
        }
        if "heavy" in parts and HEAVY_CHECKS > 1:
            for r in res.values():
                r["heavy_checks"] = r["launches"] * HEAVY_CHECKS  # checked outputs per heavy op
        for p in parts:
            try:
                for j, t in enumerate(acc[p]):
                    rt, red = reduce(t, p != "heavy")
                    for i, v in enumerate(to_host(rt)):
                        r = res[str(i)]
                        try:  # one chip's bad counter stays on that chip (an inf here failed all 32)
                            # heavy under --heavy-check-each: up to HEAVY_CHECKS mismatching outputs per launch
                            n, sat = (
                                cs.mismatch_count(v.max().item(), r["launches"] * (HEAVY_CHECKS if p == "heavy" else 1))
                                if v.numel()
                                else (0, False)
                            )
                        except Exception as e:  # noqa: BLE001
                            r.setdefault("errors", []).append(f"{p}: {type(e).__name__}: {str(e)[:200]}")
                            r["error"] = "; ".join(r["errors"])
                            continue
                        if n:
                            r["mismatched"] = max(r["mismatched"], n)
                            r["saturated"] = r.get("saturated") or sat
                            if p not in r["bad_parts"]:
                                r["bad_parts"].append(p)
                            if (
                                p == "heavy" and len(ref[p]) > 1
                            ):  # which --heavy op mismatched, and in how many launches
                                r["bad_parts"].append(f"heavy:{HEAVY_OPS[j]}={n}")
                    if p == "heavy" and any(p in r["bad_parts"] for r in res.values()):
                        loc_probe(p, t, res, j)
                    if p == "heavy" and j == len(acc[p]) - 1 and HEAVY_CHECKS > 1:
                        for i, f in fbp.at_verify(
                            k, time.time(), nl.get(k, 0), [i for i, r in res.items() if "heavy" in r["bad_parts"]]
                        ).items():
                            res[i]["first_bad"] = f
                    if red:
                        free(rt)
                    free(t)
                    acc[p][j] = zeros_like(ref[p][j])
                probe_err(p, res)
                if p == "heavy":
                    cap_prev.clear()  # the counters start again from zero
            except Exception as e:  # noqa: BLE001
                for r in res.values():  # not checked: chunk_sched.report() fails it, the exit code is 1
                    r.setdefault("errors", []).append(f"{p}: {type(e).__name__}: {str(e)[:200]}")
                    r["error"] = "; ".join(r["errors"])
        if SV["on"] or SV.get("dropped"):
            sv_verify(res, last=not SV["on"])
        return res

    def sv_verify(res, last=False):
        """Chunk end: each SFPU op's counter per heavy output, reduced on the device; a chip that mismatched gets
        part sfpu-vote and sfpu_mis {<op>: n}, its count against that op's SFPU checks (sfpu_mis, sfpu_n: not the heavy count)
        and, read from its counter, the cores x,y that computed the wrong rows. last: the vote was turned off
        mid-chunk; the counters gathered before it are read once more, and a counter that cannot be read is a WARN row
        saying it was dropped, never a NUMERIC NOT CHECKED of the heavy part."""
        for r in res.values():
            r["sfpu_checks"] = sum(SV["n"].values())
            r["sfpu_n"] = {sv_label(j, s): n for (j, s), n in sorted(SV["n"].items())}
        for j, ts in enumerate(SV["acc"]):
            for s, t in enumerate(ts):
                lab = sv_label(j, s)
                if last and not SV["n"].get((j, s)):
                    continue
                try:
                    rt, red = reduce(t)
                    bad = []
                    for i, v in enumerate(to_host(rt)):
                        r = res[str(i)]
                        n, sat = cs.mismatch_count(v.max().item(), SV["n"].get((j, s), 0)) if v.numel() else (0, False)
                        if n:
                            r.setdefault("sfpu_mis", {})[lab] = n
                            r["saturated"] = r.get("saturated") or sat
                            if "sfpu-vote" not in r["bad_parts"]:  # per-op counts ride sfpu_mis: no ':' in the FAIL
                                r["bad_parts"].append("sfpu-vote")  # line's part list (fault ledger regex)
                            bad.append(i)
                    if red:
                        free(rt)
                    for i in bad[:6]:
                        res[str(i)].setdefault("sfpu", {})[lab] = nck.cores_from_counter(
                            torch, ttnn.to_torch(ttnn.get_device_tensors(t)[i]), SV["gx"]
                        )
                except Exception as e:  # noqa: BLE001
                    if last:  # after the turn-off: say these counts were dropped, the heavy check stands
                        row(
                            event="part_off",
                            part="sfpu-vote",
                            error=f"{lab}: {SV['n'][(j, s)]} SFPU checks gathered "
                            f"before the turn-off dropped, counter unreadable: {type(e).__name__}: {str(e)[:200]}",
                        )
                    else:
                        for r in res.values():
                            r.setdefault("errors", []).append(f"sfpu-vote:{lab}: {type(e).__name__}: {str(e)[:200]}")
                            r["error"] = "; ".join(r["errors"])
                if last:  # the vote stays off: its counter is not re-armed (it may already be gone)
                    try:
                        free(t)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                free(t)
                SV["acc"][j][s] = zeros_like(SV["ref"][j][s])
        SV["n"] = {}
        SV["dropped"] = False

    nl = {}

    class CapAd:
        """numeric_capture.capture()'s view of this mesh: chip i's copy of a tensor, the heavy op's reference, counter,
        inputs and one more run."""

        grid, phys, free = (int(g.x), int(g.y)), staticmethod(phys), staticmethod(free)

        def read(self, t, i):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[int(i)]).float()

        def ref(self, j):
            return ref["heavy"][j]

        def counter(self, j):
            return acc["heavy"][j]

        def inputs(self, j):
            return tuple(self.op(j).tensors[:2])

        def op(self, j):
            return (hv[3] if len(hv) > 3 else [hv[0]])[j]

        def rerun(self, j):
            o = self.op(j)()
            ttnn.synchronize_device(m)
            return o

    def cap_poll(k):
        """every --capture-ring launches, each chip's heavy counter (two max reductions on the device, one
        small readback); a chip whose counter rose is captured while the held outputs are still in DRAM. A capture
        error turns the capture off for the load (the verify still counts). The poll's launch goes as poll_launch: a
        capture event's own launch is the held output that read back wrong."""
        for j, t in enumerate(acc.get("heavy") or []):
            try:
                rt, _ = reduce(t, False)
                now = {i: v.max().item() for i, v in enumerate(to_host(rt))}
                free(rt)
                hit = nc.risen({i: cap_prev.get((j, i), 0.0) for i in now}, now)
                bad = {h[0] for h in hit}
                # the control chip: one whose counter never rose, else one not hit in this poll
                ctl = next((c for c in sorted(now) if cap_prev.get((j, c), 0.0) == 0 and now[c] == 0), None)
                ctl = next((c for c in sorted(now) if c not in bad), None) if ctl is None else ctl
                for i, v0, v1 in hit[:2] if ctl is not None else []:
                    if cap_n[0] >= 4:
                        break
                    t0 = time.time()
                    ev = nc.capture(CapAd(), torch, cs, j, i, ctl, CAP[0].held.get(j, []), pv)
                    cap_n[0] += 1
                    c = chipinfo[i]
                    row(
                        event="capture",
                        k=k,
                        poll_launch=nlaunch[0],
                        pci=c.get("pci"),
                        tray=c.get("tray"),
                        dev=c.get("dev"),
                        op_name=HEAVY_OPS[j] if j < len(HEAVY_OPS) else "heavy",
                        counter=[v0, v1],
                        capture_s=round(time.time() - t0, 1),
                        **ev,
                    )
                cap_prev.update({(j, i): v for i, v in now.items()})
            except Exception as e:  # noqa: BLE001
                row(event="capture", k=k, poll_launch=nlaunch[0], error=f"poll: {type(e).__name__}: {str(e)[:160]}")
                free(*CAP.pop().drop())
                return

    def step(k, i):
        """One launch, no synchronize: the heavy part's events bound the run-ahead; with no heavy part an event
        per launch does; with no ttnn events, a synchronize every RUN_AHEAD_OPS launches."""
        nl[k] = nl.get(k, 0) + 1
        if fbp.k != k and nl[k] == 1:
            fbp.start(k, time.time())  # chunk k's first launch: its first-wrong-output windows count from here
        if tu is not None:
            tu.k = k
        t0 = (
            time.time()
        )  # --heavy-x: on when this launch ends inside the chunk's last --heavy-x-s s (last launch's length ahead)
        XT["on"] = bool(len(HEAVY_OPS) > NMAIN and XT["end"] is not None and t0 + XT["dur"] >= XT["end"] - a.heavy_x_s)
        nb = launch()
        if CAP and CAP[0].due(nlaunch[0]):
            cap_poll(k)
        XT["dur"] = time.time() - t0
        if "heavy" not in parts:
            ra.queued(0)
        if ra.why is not None and i % RUN_AHEAD_OPS == 0:
            ttnn.synchronize_device(m)
        return nb

    def sync():
        if tu is not None:
            tu.sync()
        ttnn.synchronize_device(m)
        ra.evs = []

    def load(k):
        """The chunk row's "load": what ran and every way it fell short of the full back-to-back load."""
        fb = [lv.warning(VAR)] if VAR else []
        if "heavy" not in parts:
            fb.append("no heavy part (off: see part_off)")
        elif any(x[0].state.get("mm") is None for x in HVS):
            fb.append("minimal_matmul refused, ttnn.matmul")
        if hv and "heavy" in parts and hv[0].ran.get("sfpu"):
            fb.append(f"--sfpu-par 1: {hv[0].ran['sfpu']}")
        if hv and "heavy" in parts and hv[0].ran.get("extras_fallback"):
            fb.append(f"{a.heavy}: {hv[0].ran['extras_fallback']}")
        if ra.why is not None:
            fb.append(f"synchronize every {RUN_AHEAD_OPS} launches ({ra.why})")
        if "pcie" in parts and (cq_why or fell.get("pcie")):
            fb.append(f"pcie in line on cq 0: {fell.get('pcie') or cq_why}")
        if tu is not None and tu.why is not None:
            fb.append(f"top-up off: {tu.why}")
        fb += [f"part {p} off" for p in FN if p not in parts and p in a.parts.split(",") and p != "heavy"]
        return dict(
            heavy=(f"{hv[2]} x{a.heavy_n} per launch" if "heavy" in parts else None),
            gap=(gap.desc() if "heavy" in parts else None),
            first_bad=(fbp.desc() if "heavy" in parts and HEAVY_CHECKS > 1 else None),
            run_ahead=(
                f"host at most {RUN_AHEAD_OPS} heavy ops ahead (ttnn events), no per-launch sync"
                if ra.why is None
                else ra.desc()
            ),
            concurrent=("pcie on cq 1 beside cq 0" if "pcie" in parts and CQ1 is not None else None),
            pcie_every=(a.pcie_every if "pcie" in parts else None),
            topup=(tu.desc(k) if tu is not None else None),
            checks_per_launch=check_count(ref, parts, a.pcie_every, 0, HEAVY_CHECKS)[0],
            checks=check_count(ref, parts, a.pcie_every, nl.get(k, 0), HEAVY_CHECKS)[1],
            fallback=fb,
        )

    # the clock check at each quiet: the quiet counts from every chip at <= 800 MHz
    # (at most 5 min, then a QUIET FAIL finding by BDF and tray 1-4); each chunk after a quiet starts at the busy clock
    import quiet_clock as qcm

    qc = qcm.QuietClock(pcis, say=lambda s: print(s, flush=True), tick=beat)

    def rw_after(k):
        """G3: after chunk k's verify, before its quiet (never inside a burst), --rw-s s of the
        x/w/y-in-DRAM C4n-size matmul on every chip, its output read back and compared with the majority; one "rw" row.
        A mismatch is the chip's NUMERIC FAIL (chunk_sched.rw_lines); an error is a NUMERIC WARN, the stage goes on."""
        if k + 1 != a.rw_after:
            return
        import numeric_dram as nd

        t0 = time.time()
        beat("rw check 0")
        say = lambda s: print(f"{time.strftime('%F %T')} G3 {s}", flush=True)  # noqa: E731
        nb = 8
        try:
            nb = int(m.get_devices()[0].num_dram_channels()) or 8
        except Exception:  # noqa: BLE001
            pass
        say(
            f"x/w/y-in-DRAM matmul {nd.M}x{nd.K}x{nd.N} on {len(chipinfo)} chips after chunk {k + 1}: {a.rw_s:g} s of "
            f"load, a readback every {a.rw_check_s:g} s of load, wall cap {a.rw_wall_s:g} s, banks {nb}"
        )
        chips = {i: dict(pci=c["pci"], tray=c["tray"], dev=c["dev"]) for i, c in enumerate(chipinfo)}
        try:
            got = nd.rw_check(
                ttnn, torch, m, chips, a.rw_s, a.rw_wall_s, a.rw_per_check, a.rw_check_s, nb, beat=beat, say=say
            )
        except Exception as e:  # noqa: BLE001
            got = dict(error=f"{type(e).__name__}: {str(e)[:300]}", chips={})
            say(f"error: {got['error']}")
        rw_rows.append(got)
        row(event="rw", k=k, start=round(t0, 3), end=round(time.time(), 3), **got)
        if a.dbi_wall_s > 0:
            import dram_dbi as dd

            t1 = time.time()
            beat("dbi pass 0")
            sayb = lambda s: print(f"{time.strftime('%F %T')} G3b {s}", flush=True)  # noqa: E731
            sayb(
                f"DRAM byte/DBI patterns on {len(chipinfo)} chips after chunk {k + 1}: {dd.ROWS * dd.ROW_B >> 20} MB per "
                f"chip, {a.dbi_hops} copies on the device per pass, wall cap {a.dbi_wall_s:g} s, banks {nb}"
            )
            try:
                gb = dd.dbi_check(ttnn, torch, m, chips, a.dbi_wall_s, nb, a.dbi_hops, beat=beat, say=sayb)
            except Exception as e:  # noqa: BLE001
                gb = dict(error=f"{type(e).__name__}: {str(e)[:300]}", chips={})
                sayb(f"error: {gb['error']}")
            rw_rows.append(gb)
            row(event="dbi", k=k, start=round(t1, 3), end=round(time.time(), 3), **gb)
        if a.l1_wall_s > 0:
            import l1_march as lm

            t2 = time.time()
            beat("l1 pass 0")
            sayl = lambda s: print(f"{time.strftime('%F %T')} N1 {s}", flush=True)  # noqa: E731
            sayl(
                f"L1 march on {len(chipinfo)} chips after chunk {k + 1}: one shard per Tensix core, copied L1 to L1 to "
                f"the next core, {len(lm.patterns(a.l1_full))} patterns, wall cap {a.l1_wall_s:g} s"
            )
            try:
                gl = lm.l1_check(ttnn, torch, m, chips, a.l1_wall_s, a.l1_full, a.l1_kb, beat=beat, say=sayl)
            except Exception as e:  # noqa: BLE001
                gl = dict(error=f"{type(e).__name__}: {str(e)[:300]}", chips={})
                sayl(f"error: {gl['error']}")
            rw_rows.append(gl)
            row(
                event="l1",
                k=k,
                start=round(t2, 3),
                end=round(time.time(), 3),
                smoke=os.environ.get("FL_SMOKE") == "1",
                **gl,
            )

    rw_rows = []
    if SH is not None:
        row(
            event="shape",
            shape=a.shape,
            what=ds.SHAPES[a.shape]["what"],
            why=ds.SHAPES[a.shape]["why"],
            trays=list(SH.trays),
            target_tray=a.shape_tray,
            seed=a.shape_seed,
            parts=parts,
        )
    try:
        hs_row0 = HS.row(row)  # host-stress rows (a no-op wrapper when off)

        def hs_row(**kw):  # each chunk's start row sets the --heavy-x window's end; preramp and quiets have none
            ev = kw.get("event")
            if ev == "start":
                XT["end"] = kw["start"] + kw["chunk_s"]
            elif ev in ("preramp", "chunk"):
                XT["end"] = None
            return hs_row0(**kw)

        total = cs.run(
            chunks_s,
            a.gap_s,
            step,
            sync,
            verify,
            hs_row,
            beat=beat,
            lead_s=lead,
            gate=gate,
            util=util,
            extra=load,
            qc=qc,
            pre_s=a.pre_ramp_s,
            warm_s=a.pre_warm_s,
            pre_done=(tu.plan if tu is not None else None),
            after=(rw_after if a.rw_after > 0 else None),
            fast=(
                cs.FastPoll(
                    os.path.join(os.path.dirname(a.rows) or ".", "fastpoll.txt"),
                    window_s=a.fast_poll_s,
                    every=a.fast_poll_ms / 1000,
                )
                if a.fast_poll_s > 0
                else None
            ),
            hang_stop=os.environ.get("HOLD_ACTIVE", "0") != "1",
        )
    # no close: the device is hung; --hold never gets here (hang_stop off), the chip stays for triage
    except cs.HangStop as h:
        row(
            event="hang_exit",
            k=h.k,
            chips=[{"bdf": c.get("pci"), "tray": c.get("tray"), "error": c.get("error")} for c in h.chips],
            msg=str(h),
        )
        print("HANG EXIT " + str(h), flush=True)
        HS.stop()
        os._exit(3)
    nbad = nerr = 0
    for r in [json.loads(x) for x in open(a.rows) if '"event": "verify"' in x]:
        nbad += sum(1 for c in r["chips"].values() if c.get("mismatched") or c.get("bad_parts"))
        nerr += sum(1 for c in r["chips"].values() if c.get("error"))
    nrw = sum(
        1 for r in rw_rows for c in r["chips"].values() if c.get("bad_checks") or c.get("bad_passes")
    )  # G3, G3b: exit 4 too
    bad = nbad + nrw + LD
    if SH is not None:
        row(
            event="shape_end",
            shape=a.shape,
            hi_launches={str(t): sh_n.get(t, 0) for t in SH.trays},
            load_s=round(time.time() - (sh_t0[0] or time.time()), 1),
        )
    row(
        event="end",
        launches=total,
        verify_failures=bad,
        verify_errors=nerr,
        **({"rw_failures": nrw} if rw_rows else {}),
    )
finally:
    with ss.closing():  # a TERM during the close, HS.stop() included, waits for it
        HS.stop()
        beat("close")
        for sm in SUBS:
            tt0.close_mesh_device(sm)
        tt0.close_mesh_device(M)
        beat("closed")
sys.exit(4 if bad else 1 if nerr else 0)
