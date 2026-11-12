# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dev load variants of the bus-bar stage load: FLEET_LOAD_VARIANT (or
chunk_load.py / ub0_repro.py --variant, which lib/fleet.sh passes since sudo -u drops the environment), a comma list of:

  dram-mm      the heavy op is 16k_bf8_lofi: the same LoFi bf8 16k matmul with A and B in DRAM, without lofi_all's
               serialized all-gather and DRAM add after each op, so the FPU idles less between matmuls. Only in chunks that
               run the h3, dram and pcie parts beside it (they keep the link traffic, DRAM and PCIe writes); the heavy-only
               chunk, the qb/step bursts and pinpoint keep lofi_all, whose extras are their only link and PCIe traffic.
  host-stress  host CPU and memory workers (nice 10) for each chunk's load and each qb/step burst only: started at the
               chunk's start row (the burst's first op), stopped (our own pids) at its end row (the burst's synchronize),
               and at exit. Not in pinpoint (a short diagnosis after a NUMERIC FAIL).
  no-gap       record only: no 50 ms pause after every 8 heavy ops (--burst-gap-ms 0) in chunks, bursts and pinpoint.
               Never the landed load.
  combined     variants 1+3+4 as far as they build: dram-mm,host-stress (dram-eth, its 1, is refused below).

dram-eth (variant 1: DRAM streaming on the Ethernet cores the all-gather and fabric leave idle) is refused with its
reason (NOT_BUILT): no kernel can be put on an Ethernet core from Python.

A variant is a development override: a loud WARNING on the run's first line, on every chunk's LOAD line, in each burst's
log and in the summary, naming the parts each variant ran in. Numeric checks, link traffic and PCIe writes stay in every
variant."""
import os
import signal
import subprocess
import sys
import time

VARIANTS = ("dram-mm", "host-stress", "no-gap")
ALIASES = {"combined": ("dram-mm", "host-stress")}   # variants 1+3+4 without 1 (dram-eth, NOT_BUILT)
NOT_BUILT = {"dram-eth": "not buildable: ttnn's ProgramDescriptor takes Tensix reader, writer, data-movement and compute "
                         "kernels only (KernelDescriptor has no Ethernet config in tt-metal a869f55198 or main), so no DRAM "
                         "reader can run on the idle Ethernet cores from Python"}
# where each variant runs (the WARNING and the measurement logs name them)
RAN_IN = {"dram-mm": "chunks with the h3, dram and pcie parts", "host-stress": "chunks and qb/step bursts",
          "no-gap": "chunks, qb/step bursts and pinpoint"}
NOT_IN = {"dram-mm": "heavy-only chunk, qb/step bursts and pinpoint keep lofi_all (their only link and PCIe traffic)",
          "host-stress": "pinpoint"}
DRAM_MM_HEAVY = "16k_bf8_lofi"
DRAM_MM_NEEDS = ("h3", "dram", "pcie")   # the parts that carry link traffic, DRAM and PCIe writes once lofi_all's extras go


def parse(v):
    """'a,b' -> ['a', 'b'] in VARIANTS order (combined expands); ValueError on an unknown, repeated or unbuilt name."""
    raw = [x for x in (v or "").split(",") if x]
    nb = [x for x in raw if x in NOT_BUILT]
    if nb:
        raise ValueError(f"FLEET_LOAD_VARIANT={v}: {nb[0]} is {NOT_BUILT[nb[0]]}")
    vs = [y for x in raw for y in ALIASES.get(x, (x,))]
    bad = [x for x in vs if x not in VARIANTS]
    if bad or len(set(vs)) != len(vs):
        raise ValueError(f"FLEET_LOAD_VARIANT takes one or more of {', '.join(VARIANTS + tuple(ALIASES))}, "
                         f"comma-separated, each once (got '{v}')")
    return [x for x in VARIANTS if x in vs]


def parts_line(vs):
    """'dram-mm in chunks with ... (not: ...); host-stress in ...': which parts ran each variant."""
    return "; ".join(f"{x} in {RAN_IN[x]}" + (f" (not: {NOT_IN[x]})" if x in NOT_IN else "") for x in vs)


def warning(vs):
    return (f"WARNING: dev override FLEET_LOAD_VARIANT={','.join(vs)}: not the standard bus-bar load "
            f"(no-gap is record only); {parts_line(vs)}") if vs else ""


def apply(a, vs):
    """chunk_load.py's args under the variants (lib/fleet.sh sets no-gap's 0 ms for every stage too). dram-mm only where
    the h3, dram and pcie parts run beside the heavy op, and only on the standard op (stage 5's --heavy DIDT ops stay)."""
    parts = (getattr(a, "parts", "") or "").split(",")
    if "dram-mm" in vs and a.heavy == "lofi_all" and all(p in parts for p in DRAM_MM_NEEDS):
        a.heavy = DRAM_MM_HEAVY
    if "no-gap" in vs:
        a.burst_gap_ms = 0.0


# one worker: a 64 MB buffer copied over and over (memory bandwidth and CPU), until SIGTERM or its parent is gone
WORKER = ("import os,signal,sys\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\npp = os.getppid()\n"
          "a = bytearray(64 << 20); b = bytearray(64 << 20); n = 0\n"
          "while os.getppid() == pp:\n    b[:] = a; a[n % len(a)] = n & 255; n += 1\n")


class HostStress:
    """host-stress: n workers (FLEET_HOST_STRESS_N, default nproc - 2, at least 1), nice 10, in chunk_load.py's own
    process group (run.sh's group kill reaches them; each also exits once its parent is gone). start() and stop() are
    idempotent; stop() TERMs our own pids, waits up to 5 s, then KILLs them."""

    def __init__(self, on, n=None, py=sys.executable, popen=subprocess.Popen):
        self.on, self.procs, self.py, self.popen, self.start_s = on, [], py, popen, None
        self.n = n if n is not None else int(os.environ.get("FLEET_HOST_STRESS_N") or max(1, (os.cpu_count() or 4) - 2))

    def start(self):
        """n workers under nice -n 10 (no preexec_fn: a fork of a threaded process may deadlock before exec); start_s is how long their start took (logged in the chunk's start row and the burst's log)."""
        if not self.on or self.procs:
            return 0
        t0 = time.monotonic()
        for _ in range(self.n):
            self.procs.append(self.popen(["nice", "-n", "10", self.py, "-c", WORKER],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.start_s = round(time.monotonic() - t0, 3)
        return len(self.procs)

    def stop(self):
        ps, self.procs = self.procs, []
        for p in ps:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in ps:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        return len(ps)

    def row(self, write):
        """row() for chunk_sched.run: workers on at each chunk's start row, off at its end row."""
        def w(**kw):
            ev = kw.get("event")
            if ev == "start":
                kw["host_stress"] = self.start()
                kw["host_stress_start_s"] = self.start_s
            elif ev == "chunk":
                kw["host_stress_stopped"] = self.stop()
            write(**kw)
        return w if self.on else write


if __name__ == "__main__":   # lib/fleet.sh's fl_init: '<variants>' -> line 1 the parsed list, line 2 the WARNING; rc 1 refused
    try:
        vs = parse(sys.argv[1] if len(sys.argv) > 1 else "")
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(",".join(vs)); print(warning(vs))
