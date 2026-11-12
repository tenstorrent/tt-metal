#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pinpoint_load.py: the numeric pinpoint under run.sh's max-load stage load.

a06u02 47:00.0 (tray 2) and a07u08 84:00.0 give wrong DRAM bank 3 results only inside run.sh's heavy stage: the solo
pinpoint got 0/86..0/93 and an uncapped 7200 s pinpoint 0 in 14509 heavy matmuls (notes/a06u02-47-numeric.md). So each
pinpoint check runs under the max-load stage's own load: lib/pinpoint.sh passes `fl_load_args heavy` (the one load
definition, lib/fleet.sh) as --load-args, and numeric_pinpoint.py builds that op with ub0_repro.heavy_op (lofi_all with the
fused SFPU epilogue, the PCIe write on cq 1, the DRAM add; no all_gather: the pinpoint opens no fabric) on the whole
mesh. Every chip carries it, the failing one too: a load on a 1x1 submesh beside the others hung the device 2/2.
Each check: a burst of --burst-steps heavy ops, the check op, one more heavy op, the readback, the --burst-gap-ms gap.

Per launch (one check's readback, or one repeat-split burst), every chip's hwmon power and temperature are read (sysfs, no
device open) into <result>.load.jsonl. Box kW is the sum of the 32 chips' W (chip power, not PSU input). A chip's
temperature counts only from launches before its first fault (a bad check, or an implausible reading: at or
below 0, over TEMP_MAX C, power over 1000 W), never at or after it (charter goal 29).
"""
import argparse
import json
import time

TEMP_MAX = 150.0   # fleet_py.TEMP_MAX: hotter is a garbage read (all ones scales to 4.29e6)
W_MAX = 1000.0     # a 4293 W reading once seen was a garbage read


def parse(s, heavy, ub):
    """--load-args: the max-load stage's chunk_load flags (fl_load_args heavy; the rest are ignored, '' the ub0_repro
    defaults). 'off': None, no load."""
    if (s or "").strip() == "off":
        return None
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--heavy", default=heavy or ub.HEAVY_DEFAULT)
    p.add_argument("--burst-steps", type=int, default=ub.BURST_STEPS)
    p.add_argument("--burst-gap-ms", type=float, default=ub.BURST_GAP_MS)
    p.add_argument("--heavy-pcie-mb", type=int, default=ub.HEAVY_PCIE_MB)
    p.add_argument("--sfpu-par", type=int, default=1)
    p.add_argument("--sfpu-act", default=ub.SFPU_ACT)
    p.add_argument("--cqs", type=int, default=2)
    p.add_argument("--variant", default="")   # only no-gap runs here (its 0 ms gap); named in the load line
    a, _ = p.parse_known_args(s.split())
    a.burst_steps = max(1, a.burst_steps)
    return a


def plaus(v, hi):
    return v if isinstance(v, (int, float)) and 0 < v <= hi else None


def read_chip(dev):
    """(W, C) from the chip's hwmon (sysfs, opens no device); None where unreadable."""
    import ub0_repro as ub
    return ub.hwmon(dev, "power1_input", 1e6), ub.hwmon(dev, "temp1_input", 1e3)


class Load:
    """The max-load stage's op on the whole mesh, built lazily on the mesh it runs on (the clock split reopens it)."""

    def __init__(self, la, ttnn, torch, log, sleep, cq1=None, cq_why=None):
        self.la, self.ttnn, self.torch, self.log, self.sleep = la, ttnn, torch, log, sleep
        self.cq1, self.cq_why = cq1, cq_why
        self.mesh = self.op = None
        self.held, self.desc, self.off, self.pending, self.bursts = [], "", None, False, 0

    def build(self, mesh, k=1024):
        if self.off or (self.op is not None and self.mesh is mesh):
            return self.op
        self.free()
        import ub0_repro as ub
        la, ttnn = self.la, self.ttnn
        try:
            x = w = None
            if ub.HEAVY.get(la.heavy, 0) is None:   # the plain matmul load (tests): its own small x and w
                rep, D = ttnn.ReplicateTensorToMesh(mesh), ttnn.DRAM_MEMORY_CONFIG
                x, w = (ttnn.from_torch(self.torch.randn(*s, dtype=self.torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=D, mesh_mapper=rep)
                        for s in ((1, 1, 32, k), (1, 1, k, 32)))
                self.held += [x, w]
            ns = argparse.Namespace(heavy=la.heavy, heavy_act=la.sfpu_act if la.sfpu_par else None,
                                    heavy_pcie_mb=la.heavy_pcie_mb, m=32, k=k, n=32, topo="none")
            op, _, d = ub.heavy_op(ns, ttnn, self.torch, mesh, x, w, cq1=self.cq1, fabric=False, cq_why=self.cq_why)
            self.held += list(getattr(op, "tensors", ()))
            fb = [v for v in ((getattr(op, "ran", None) or {}).get(f) for f in ("fallback", "sfpu", "extras_fallback")) if v]
            self.op, self.mesh = op, mesh
            self.desc = (f"{la.heavy}: {d}; bursts of {la.burst_steps} with {la.burst_gap_ms:g} ms gaps, whole mesh"
                         + (f" ({'; '.join(fb)})" if fb else ""))
            self.log(f"load: {self.desc}" + (f"; WARNING: dev load variant {self.la.variant}: only no-gap runs in pinpoint "
                                             f"(dram-mm and host-stress do not)" if getattr(self.la, "variant", "") else ""))
        except Exception as e:   # the load cannot run here: the checks go on solo, and the report says so
            self.off = f"{type(e).__name__}: {str(e)[:160]}"
            self.log(f"load off: {self.off}; the checks run without it")
            self.free()
        return self.op

    def _run(self, n):
        try:
            for _ in range(n):
                y = self.op()
                if y is not None:
                    self.ttnn.deallocate(y)
        except Exception as e:
            self.off = f"{type(e).__name__}: {str(e)[:160]}"
            self.log(f"load off: {self.off}; the checks run without it")
            self.op = None

    def burst(self, mesh):
        """A burst ahead of the next check op (queued, no sync)."""
        if self.build(mesh) is not None:
            self._run(self.la.burst_steps)
            self.bursts += 1

    def before_read(self, mesh):
        """Before a readback: a burst if none was queued ahead of the check, then one heavy op behind it."""
        if self.build(mesh) is None:
            return False
        if not self.pending:
            self.burst(mesh)
        self.pending = False
        if self.op is not None:
            self._run(1)
        return self.op is not None

    def after_read(self, mesh):
        """After a readback (the device is idle): the gap, then the next check's burst."""
        if self.op is None:
            return
        self.sleep(self.la.burst_gap_ms / 1000.0)
        self.burst(mesh)
        self.pending = self.op is not None

    def free(self):
        if self.held and self.mesh is not None:
            try:
                self.ttnn.synchronize_device(self.mesh)
            except Exception:
                pass
        for t in self.held:
            try:
                self.ttnn.deallocate(t)
            except Exception:
                pass
        self.held, self.op, self.mesh, self.pending = [], None, None, False


class Tele:
    """Per-launch chip W, chip temperature (pre-fault only) and box kW, to <path> (JSONL) and the log."""

    def __init__(self, chips, targets, path, log, reader=None):
        self.chips, self.targets, self.path, self.log = chips, set(targets), path, log
        self.reader = reader or read_chip
        self.n, self.pend, self.fault, self.kw, self.dropped = 0, None, {}, [], 0
        self.peak = {}
        open(path, "w").close()

    def sample(self, split):
        self.commit()
        self.n += 1
        row = dict(launch=self.n, split=split, t=time.strftime("%F %T"), w={}, c={})
        for bdf, ch in self.chips.items():
            try:
                w, c = self.reader(ch.get("dev"))
            except Exception:
                w = c = None
            pw, pc = plaus(w, W_MAX), plaus(c, TEMP_MAX)
            if (w is not None and pw is None) or (c is not None and pc is None):   # read but implausible (all ones,
                self.dropped += 1                                                    # out of range): a fault of its own,
                self.fault.setdefault(bdf, self.n)                                   # no later temperature counts
            w, c = pw, pc
            if w is not None:
                row["w"][bdf] = round(w, 1)
            if c is not None:
                row["c"][bdf] = round(c, 1)
        self.pend = row

    def faulted(self, bdf):
        """bdf gave a bad result in the current launch: this launch's and every later temperature are dropped."""
        self.fault.setdefault(bdf, max(1, self.n))

    def commit(self):
        row, self.pend = self.pend, None
        if row is None:
            return
        for bdf in list(row["c"]):
            if bdf in self.fault and self.fault[bdf] <= row["launch"]:
                del row["c"][bdf]
        row["box_kw"] = round(sum(row["w"].values()) / 1000, 3) if row["w"] else None
        row["chips_w"] = len(row["w"])
        if row["box_kw"] is not None:
            self.kw.append(row["box_kw"])
        for bdf in self.chips:
            p = self.peak.setdefault(bdf, dict(w=None, c=None, c_launch=None, c_split=None))
            w, c = row["w"].get(bdf), row["c"].get(bdf)
            if w is not None and (p["w"] is None or w > p["w"]):
                p["w"] = w
            if c is not None and (p["c"] is None or c > p["c"]):
                p.update(c=c, c_launch=row["launch"], c_split=row["split"])
        with open(self.path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        tg = ", ".join(f"{b} tray {self.chips[b].get('tray')} {row['w'].get(b, '-')} W {row['c'].get(b, '-')} C"
                       for b in sorted(self.targets) if b in self.chips)
        self.log(f"LOAD launch {row['launch']} {row['split']}: box {row['box_kw']} kW ({row['chips_w']} chips' W); {tg}")

    def summary(self):
        return dict(launches=self.n, box_kw_mean=round(sum(self.kw) / len(self.kw), 3) if self.kw else None,
                    box_kw_max=max(self.kw) if self.kw else None, dropped_reads=self.dropped,
                    faults={b: n for b, n in sorted(self.fault.items())},
                    chips={b: p for b, p in sorted(self.peak.items())})
