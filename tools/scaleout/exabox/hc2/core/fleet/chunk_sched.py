#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""chunk_sched.py: the chunk scheduler, quiet gate and rows parser of the
chunked sustained stage (fleet/chunk_load.py). Standard library only, so it is unit-tested without a device
(tests/test_chunk_sched.py).

Schedule (the one place it is kept): RAMP_MIN, load chunks of 1, 2, 3, 4 and 15 min (25 min), each followed by its
chunk-end verify (readback) and QUIET_S s of forced quiet: the device is synchronized, nothing is enqueued, no data
moves. Override for tests and short runs: FLEET_CHUNK_MIN="1,2,3" (minutes, decimals allowed), FLEET_GAP_S (s).
Every chip restarts at once after a quiet (one process drives the whole mesh), so each restart is a whole-galaxy di/dt
step. Every window's epochs go to rows.jsonl (k counts chunks from 0; lines for people say chunk k+1 of N):
  {"event": "start", "k": K, "start": E, "chunk_s": S}                    written as chunk K starts
  {"event": "chunk", "k": K, "start": E, "end": E, "chunk_s": S, "launches": N, "blocks": B, "util": {...}}
  {"event": "verify", "k": K, "start": E, "end": E, "chips": {...}}       chunk-end per-chip numeric readback
  {"event": "quiet", "k": K, "start": E, "end": E}                       K = the chunk it follows (-1: a lead quiet)
  {"event": "preramp", "k": -1, "start": E, "pre_s": S, "warm_s": W}      the pre-ramp starts
  {"event": "preramp_end", "k": -1, "start": E, "end": E, "measured": [E, E], "launches": N, "topup": {...}}
Pre-ramp: with pre_s, full load runs right before chunk 1 and is not counted in the
chunks: pre_s s of load, then pre_done(start, end) (chunk_load.py: the per-chip top-up from that window's C8
telemetry), then warm_s s more load with the top-up on; chunk 1 follows with no sync and no quiet, so it starts at full
power. Its launches are chunk 1's (step(0, i)): chunk 1's verify reads their counters too.

  chunk_sched.py report <rows.jsonl> [power.txt [c8_telemetry.txt [c8_limits.txt]]]   the stage's lines for fleet.sh (CHUNK, LOAD, CHIPS, QUIET, NUMERIC FAIL/PASS);
                                                   with fleet.sh's power.txt, each window's BMC tray power; with c8_limits.txt,
                                                   why a chunk under 12 kW fell short (power-limit ceiling or load stall)
  chunk_sched.py windows <rows.jsonl>              "<kind> <k> <start> <end>" per window (power analysis)
  chunk_sched.py sched                             "<chunks> <load s> <quiet s> <chunk minutes, comma-separated>"
  chunk_sched.py where <rows.jsonl> <epoch> [bdf[@k]...]   where in the schedule <epoch> fell, naming each BDF's tray (and tt-smi k)
  chunk_sched.py stalled <rows.jsonl>               a load hang's PINPOINT line (the chips that stopped, or the whole box), then
                                                   the stopped chips' BDFs on line 2 (empty: the whole box)
"""
import json
import math
import os
import re
import sys
import time

RAMP_MIN = (1, 2, 3, 4, 15)      # quiets after 1 min of load, then 2, 3, 4, 15 (25 min of load)
QUIET_S = 10.0                   # everything forced quiet for 10 s after each chunk

BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}   # PCI bus & 0xF0 -> tray 1-4 (BMC name UBB<tray-1>)


def tray_of(bdf):
    try:
        return BUS_TO_TRAY.get(int(str(bdf).split(":")[-2], 16) & 0xF0)
    except (IndexError, ValueError):
        return None


def short(bdf):
    return bdf[5:] if bdf and bdf.count(":") == 2 else (bdf or "?")


def schedule(env=None):
    """([chunk seconds], quiet seconds): the ramp, or FLEET_CHUNK_MIN / FLEET_GAP_S from env."""
    env = os.environ if env is None else env
    raw = (env.get("FLEET_CHUNK_MIN") or "").replace(" ", ",")
    mins = [float(x) for x in raw.split(",") if x] if raw.strip(",") else list(RAMP_MIN)
    gap = float(env.get("FLEET_GAP_S") or QUIET_S)
    chunks = [m * 60.0 for m in mins]
    plan(chunks, gap)
    return chunks, gap


def mins(seconds):
    return f"{seconds / 60:g}"


def plan(chunks_s, gap_s, lead_s=0.0):
    """The nominal timeline from t=0: [(kind, k, start, end)]: an optional lead quiet (k -1), then each chunk k and the
    quiet k after it."""
    if not chunks_s or any(c <= 0 for c in chunks_s) or gap_s < 0 or lead_s < 0:
        raise ValueError(f"bad schedule: chunks {list(chunks_s)} s, gap {gap_s} s, lead {lead_s} s")
    out, t = [], 0.0
    if lead_s:
        out.append(("quiet", -1, t, t + lead_s)); t += lead_s
    for k, c in enumerate(chunks_s):
        out.append(("chunk", k, t, t + c)); t += c
        if gap_s:
            out.append(("quiet", k, t, t + gap_s)); t += gap_s
    return out


class QuietViolation(RuntimeError):
    pass


class Gate:
    """Load parts call enqueue() before they put work on the device; inside a quiet window that raises."""

    def __init__(self):
        self.quiet = False

    def enqueue(self, what=""):
        if self.quiet:
            raise QuietViolation(f"enqueue during a quiet window: {what}")


def hold_quiet(gate, sync, seconds, clock=time.time, sleep=time.sleep, qc=None, label="quiet"):
    """Drain the device (sync), then keep the gate shut for `seconds` measured from the drain's end, or with qc (a
    quiet_clock.QuietClock) from the moment every chip reads <= 800 MHz (its settle(), capped at 5 min;
    qc.log[-1] is that record). Returns (start, end) epochs of the idle window."""
    sync()
    gate.quiet = True
    try:
        if qc is not None:
            qc.settle(label)
        start = clock(); end = start + seconds
        while True:
            left = end - clock()
            if left <= 0:
                break
            sleep(min(left, 0.5))
        return start, clock()
    finally:
        gate.quiet = False


def quiet_row(qc):
    """The clock check's fields for a quiet row: seconds to idle clock and the chips that never got there."""
    if qc is None or not qc.log:
        return {}
    r = qc.log[-1]
    return dict(settle_s=r["settle_s"], never=r["never"], cap_s=r["cap_s"])


class FastPoll:
    """A short fast window at each chunk start, so the PEAK line sees the first bursts' spikes that the 5 s
    C8 sample misses. A thread reads every chip's hwmon power1_input and curr1_input from sysfs every `every` s for `window_s` s
    and appends "<epoch> <bdf> - <uW> <MHz> <mA> - <mA max>" rows (c8_telemetry.txt's format). Read-only: no device open, no
    write, no sync or readback in the burst; never a power limit."""

    def __init__(self, out, sysroot="/sys", window_s=4.0, every=0.1, clock=time.time, sleep=time.sleep):
        self.out, self.root, self.window_s, self.every, self.clock, self.sleep = out, sysroot, window_s, every, clock, sleep
        self.threads = []
        self.samples = 0

    def chips(self):
        import glob
        r = []
        for d in sorted(glob.glob(os.path.join(self.root, "class/tenstorrent/tenstorrent!*"))):
            h = sorted(glob.glob(os.path.join(d, "device/hwmon/hwmon*")))
            if h:
                r.append((os.path.basename(os.path.realpath(os.path.join(d, "device"))), h[0], d))
        return r

    @staticmethod
    def _rd(path):
        try:
            with open(path) as f:
                return f.read().strip() or "-"
        except OSError:
            return "-"

    def poll(self, start):
        cs = self.chips()
        end = start + self.window_s
        with open(self.out, "a") as f:
            while self.clock() < end:
                t0 = self.clock()
                for b, h, d in cs:
                    f.write(f"{t0:.3f} {b} - {self._rd(h + '/power1_input')} {self._rd(d + '/tt_aiclk')} "
                            f"{self._rd(h + '/curr1_input')} - {self._rd(h + '/curr1_max')}\n")
                    self.samples += 1
                f.flush()
                self.sleep(max(0.0, self.every - (self.clock() - t0)))

    def start(self, k, start):
        import threading
        t = threading.Thread(target=self.poll, args=(start,), daemon=True)
        t.start()
        self.threads.append(t)

    def join(self):
        for t in self.threads:
            t.join(self.window_s + 2)


class HangStop(Exception):
    def __init__(self, k, chips):
        self.k, self.chips = k, chips
        super().__init__("device timeout in chunk %d verify: %s" % (k, ", ".join(
            "%s (tray %s)" % (c.get("pci"), c.get("tray")) for c in chips)))


def timeout_chips(chips):
    """Chips of a verify row whose errors carry a device timeout."""
    return [c for c in chips.values() if re.search(r"timeout|timed out", c.get("error") or "", re.I)]


def run(chunks_s, gap_s, step, sync, verify, row, beat=lambda s: None, lead_s=0.0, clock=time.time,
        sleep=time.sleep, gate=None, util=lambda k, launches, blocks, secs: {}, stop=lambda: False, extra=None, qc=None,
        pre_s=0.0, warm_s=0.0, pre_done=None, after=None, fast=None, hang_stop=False):
    """Drive the schedule: each chunk k of chunks_s[k] s of load, its chunk-end verify, then gap_s s of quiet. step(k, i)
    runs one launch (chunk_load.py: enqueued, bounded run-ahead, no wait) and returns its block count; verify(k) does the chunk-end readback and returns its
    per-chip dict; row(**kw) writes a row; extra(k), if given, is the chunk row's "load" (chunk_load.py: heavy op,
    run-ahead, checks, fallbacks). qc, if given (quiet_clock.QuietClock): each quiet counts from every chip
    at <= 800 MHz, and each chunk after a quiet starts with every chip asked for its busy clock, its clocks in the start
    row. pre_s: the pre-ramp before chunk 1 (module doc); pre_done(start, end)'s dict goes in its end row as
    "topup". after(k), if given, runs after chunk k's verify row and before its quiet, never inside a burst (for
    chunk_load.py's G3 readback). fast, if given (FastPoll), starts its window with each chunk's start row. Returns launches done."""
    gate = gate or Gate()
    i = 1
    if lead_s:
        beat(f"done {i - 1} quiet -1")
        s, e = hold_quiet(gate, sync, lead_s, clock, sleep, qc, "sustained lead quiet")
        row(event="quiet", k=-1, start=round(s, 3), end=round(e, 3), lead=True, **quiet_row(qc))
    quieted = bool(lead_s)
    if pre_s > 0:   # full load before chunk 1, not counted in its chunk_s; no sync and no quiet after it
        p0 = clock(); p1 = None; n = 0; top = None
        row(event="preramp", k=-1, start=round(p0, 3), pre_s=pre_s, warm_s=warm_s)
        for lap, secs in enumerate((pre_s, warm_s)):
            if lap:
                p1 = clock(); top = pre_done(p0, p1) if pre_done else None
            end = clock() + secs
            while clock() < end and not stop():
                beat(f"launch {i}")
                step(0, i)
                beat(f"done {i}")
                i += 1; n += 1
        row(event="preramp_end", k=-1, start=round(p0, 3), end=round(clock(), 3), measured=[round(p0, 3), round(p1 if p1 is not None else clock(), 3)],
            launches=n, **({"topup": top} if top is not None else {}))
        quieted = False
    for k, chunk_s in enumerate(chunks_s):
        aic = {"aiclk": qc.busy(f"chunk {k + 1}")} if qc is not None and quieted else {}
        start = clock(); end = start + chunk_s; n = blocks = 0
        row(event="start", k=k, start=round(start, 3), chunk_s=chunk_s, n=len(chunks_s), **aic)
        if fast is not None:
            fast.start(k, start)
        while clock() < end and not stop():
            beat(f"launch {i}")
            blocks += step(k, i)
            beat(f"done {i}")
            i += 1; n += 1
        sync()
        t_end = clock()
        row(event="chunk", k=k, start=round(start, 3), end=round(t_end, 3), chunk_s=chunk_s, launches=n, blocks=blocks,
            util=util(k, n, blocks, t_end - start), **({"load": extra(k)} if extra else {}))
        beat(f"verify {k}")
        v0 = clock(); chips = verify(k)
        row(event="verify", k=k, start=round(v0, 3), end=round(clock(), 3), chips=chips)
        if hang_stop and timeout_chips(chips):   # a device timeout ends the run at once (never in --hold)
            raise HangStop(k, timeout_chips(chips))
        if after is not None:
            after(k)
        beat(f"done {i - 1}")
        if stop():
            break
        if gap_s:
            beat(f"done {i - 1} quiet {k}")
            s, e = hold_quiet(gate, sync, gap_s, clock, sleep, qc, f"sustained quiet after chunk {k + 1}")
            row(event="quiet", k=k, start=round(s, 3), end=round(e, 3), **quiet_row(qc))
            quieted = True
    return i - 1


def load(path):
    rows = []
    if not os.path.isfile(path):   # a hang before chunk 1 wrote no rows
        return rows
    with open(path, errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def windows(rows):
    return [(r["event"], r["k"], r["start"], r["end"]) for r in rows if r.get("event") in ("chunk", "quiet", "verify")]


CCL_PARTS = {"h3", "h3-ref"}   # chunk_load parts whose output mixes every chip's data (h3-ref: its launch-0 reference)


BF16_COUNTER_MAX = 256


def mismatch_count(x, launches):
    """A chip's chunk-end counter max (launches whose output differed from the reference) as (count, saturated).
    A non-finite or out-of-range counter (inf, NaN, 2^127, below 0: seen on one chip in testing) is that chip
    mismatched with its count saturated at its launches; int() of inf raised and failed all 32 chips."""
    x = float(x)
    # a bf16 counter stops at 256 (+1 no longer changes it): 256 with more launches after it is saturated, not exact and holds integers only: above 256 or a fraction is corrupt
    if not math.isfinite(x) or x < 0 or x > launches or x > BF16_COUNTER_MAX or x != int(x) or (x == BF16_COUNTER_MAX and launches > x):
        return max(int(launches), 1), True
    return int(x), False


FIRST_BAD_PROBE_S = 30.0   # the heavy counters are read at most this often, in a burst gap


class FirstBad:
    """When each chip's first wrong heavy output happened, not only that its chunk-end verify mismatched.
    chunk_load.py reads the on-device heavy counters at a burst gap, when the device is already
    synchronized and idle, at most every every_s s. The chunk has a fixed length in seconds, so no seconds are added.
    The first read that shows a chip's counter above 0 bounds that chip's first wrong output between the read
    before it and this one. A chip first seen at the chunk-end verify is bounded by the last read and the verify.
    Each record: s0/s1 s after the chunk start, launches l0/l1, t the epoch of the read that saw it, seen 'probe' or
    'verify', op (the --heavy op, when there are several)."""

    def __init__(self, every_s=FIRST_BAD_PROBE_S, clock=time.time):
        self.every_s, self.clock = every_s, clock
        self.k = None; self.t0 = self.last_t = 0.0; self.last_l = 0
        self.found, self.n, self.spent, self.error = {}, 0, 0.0, None

    def start(self, k, t):
        """Chunk k's first launch at epoch t: its own reads and records."""
        self.k, self.t0, self.last_t, self.last_l, self.found = k, t, t, 0, {}

    def due(self, now):
        return self.k is not None and self.error is None and self.every_s > 0 and now - self.last_t >= self.every_s

    def _rec(self, now, launches, seen, op=None):
        r = dict(s0=round(self.last_t - self.t0, 1), s1=round(now - self.t0, 1), l0=self.last_l, l1=int(launches),
                 t=round(now, 3), seen=seen)
        if op:
            r["op"] = op
        return r

    def saw(self, now, launches, bad, spent=0.0):
        """One read at epoch now after launches launches: bad = {chip id: op or None} with a counter above 0.
        Returns the chip ids first seen by this read."""
        new = [i for i in bad if i not in self.found]
        for i in new:
            self.found[i] = self._rec(now, launches, "probe", bad[i])
        self.last_t, self.last_l = now, int(launches)
        self.n += 1; self.spent += spent
        return new

    def fail(self, err):
        """A read that raised: no more reads (the chunk-end verify still bounds every chip)."""
        self.error = err

    def at_verify(self, k, now, launches, bad_ids):
        """Chunk k's verify at epoch now: {chip id: record} for each mismatching chip; a chip no read saw gets the
        window from the last read to the verify. {} when chunk k never started here."""
        if self.k != k:
            return {}
        out = {i: self.found.get(i) or self._rec(now, launches, "verify") for i in bad_ids}
        self.k = None
        return out

    def desc(self):
        return (f"heavy counters read every {self.every_s:g} s in burst gaps for the first wrong output: {self.n} reads, "
                f"{self.spent:.2f} s" + (f", reads off: {self.error}" if self.error else ""))


def first_bad_text(c):
    """', first wrong output 30-60 s (0.5-1.0 min) into chunk 1 (launches 41-82, read 12:34:56)' for a chip's numeric()
    entry with a first_bad record, '' without one."""
    f = c.get("first_bad")
    if not f:
        return ""
    return (f", first wrong output {f['s0']:.0f}-{f['s1']:.0f} s ({f['s0'] / 60:.1f}-{f['s1'] / 60:.1f} min) into chunk "
            f"{f.get('k', 0) + 1} (launches {f['l0']}-{f['l1']}" + (f", {f['op']}" if f.get("op") else "")
            + (", chunk-end read" if f.get("seen") == "verify" else ", read") + f" {hms(f['t'])})")


def err_stats(out, ref):
    """How wrong one launch's output is against the reference, both flat float lists: (max abs err, wrong elements,
    any non-finite). A non-finite output element is wrong and has an infinite error."""
    worst, wrong, nonfin = 0.0, 0, False
    for o, r in zip(out, ref):
        if o == r:
            continue
        wrong += 1
        if not math.isfinite(o):
            nonfin = True
            worst = math.inf
        elif not math.isfinite(r):
            worst = math.inf
        else:
            worst = max(worst, abs(o - r))
    return worst, wrong, nonfin


def err_stats_torch(out, ref):
    """err_stats on two torch tensors of one shape, with torch ops (no per-element Python loop): same result."""
    import torch
    out, ref = out.float().flatten(), ref.float().flatten()
    bad = out != ref
    wrong = int(bad.sum())
    if not wrong:
        return 0.0, 0, False
    o, r = out[bad], ref[bad]
    fo, fr = torch.isfinite(o), torch.isfinite(r)
    if not bool(fo.all()) or not bool(fr.all()):
        return math.inf, wrong, not bool(fo.all())
    return float((o - r).abs().max()), wrong, False


POS_MAX = 8   # bound on the logged wrong-element positions per chip (rest only in the per-core counts)


def core_of(r, c, H, W, grid, eltwise=False):
    """The logical worker core that computes output element (r, c) of an H x W matmul output on a gx x gy core grid
    The output is 32x32 tiles, Mt = ceil(H/32) rows by Nt = ceil(W/32)
    columns; each core owns a block of pm = ceil(Mt/gy) tile rows by pn = ceil(Nt/gx) tile columns, x along columns and
    y along rows (minimal_matmul's and ttnn.matmul's per-core work split). Returns core [x, y], tile [row, col], the
    tile's place in its core's block, the element in its tile [row, col] (0-31), its 16x16 face [row, col] and the
    element in that face. Pure arithmetic; no DRAM bank: an output tile index says nothing about DRAM."""
    gx, gy = int(grid[0]), int(grid[1])
    mt, nt = -(-H // 32), -(-W // 32)
    pm, pn = -(-mt // gy), -(-nt // gx)
    tr, tc, er, ec = r // 32, c // 32, r % 32, c % 32
    x, y = tc // pn, tr // pm
    if eltwise:   # 8k_int32_eltwise: tiles split in order over the cores, row-major (x fastest), as
        # tt-metal's split_work_to_cores does for binary_ng (work_split.cpp): the first tiles % cores cores get
        # tiles // cores + 1 tiles, the rest tiles // cores. The op chain's typecast may split its own way.
        q, nbig = divmod(mt * nt, gx * gy)
        t, edge = tr * nt + tc, nbig * (q + 1)
        i = t // (q + 1) if t < edge else nbig + (t - edge) // q
        per = q + 1 if i < nbig else q
        first = i * (q + 1) if i < nbig else edge + (i - nbig) * q
        return dict(core=[i % gx, i // gx], tile=[tr, tc], block_tile=[t - first, 0], element=[er, ec],
                    face=[er // 16, ec // 16], in_face=[er % 16, ec % 16], per_core=[per, 1], grid=[gx, gy])
    return dict(core=[x, y], tile=[tr, tc], block_tile=[tr - y * pm, tc - x * pn], element=[er, ec],
                face=[er // 16, ec // 16], in_face=[er % 16, ec % 16], per_core=[pm, pn], grid=[gx, gy])


def loc_summary(rows, cols, H, W, grid, eltwise=False):
    """Where wrong elements sit in an H x W matmul output computed on a gx x gy core grid (owning
    core as in core_of()). rows/cols: element indices of the mismatches. Returns {first, tiles, cores, positions, wrong,
    grid}: first = core_of() of the first wrong element plus its row and col, cores = [[x, y, n]] most hit first,
    elements = [[row, col, n]] in-tile positions most hit first, positions = the first POS_MAX wrong elements as
    [row, col, tile row, tile col, core x, core y, element row, element col]. Pure arithmetic on results already read."""
    rows, cols = list(rows), list(cols)
    if not rows:
        return None
    cores, elems, tiles, pos = {}, {}, set(), []
    for r, c in zip(rows, cols):
        o = core_of(r, c, H, W, grid, eltwise)
        tiles.add(tuple(o["tile"]))
        k, e = tuple(o["core"]), tuple(o["element"])
        cores[k] = cores.get(k, 0) + 1
        elems[e] = elems.get(e, 0) + 1
        if len(pos) < POS_MAX:
            pos.append([r, c] + o["tile"] + o["core"] + o["element"])
    first = dict(core_of(rows[0], cols[0], H, W, grid, eltwise), row=rows[0], col=cols[0])
    top = lambda d: sorted(([a, b, n] for (a, b), n in d.items()), key=lambda t: -t[2])[:8]
    return dict(first=first, tiles=len(tiles), positions=pos, wrong=len(rows), cores=top(cores), elements=top(elems),
                grid=first["grid"], shape=[H, W], **({"layout": "eltwise tile split"} if eltwise else {}))


def add_phys(l, fn):
    """The translated worker core next to each logical core of a loc_summary() dict,
    fn(x, y) -> (px, py) (chunk_load: the mesh device's worker_core_from_logical_core: a translated coordinate, the same on every chip, not the die location).
    Sets l["first"]["phys"] and
    l["phys"] = {"x,y": [px, py]} for the most-hit cores. Never fails: a core fn cannot map gets no entry."""
    if not l:
        return l
    ph = {}
    for c in [l["first"]["core"]] + [[a, b] for a, b, _ in l.get("cores") or []]:
        k = "%d,%d" % (c[0], c[1])
        if k in ph:
            continue
        try:
            px, py = fn(int(c[0]), int(c[1]))
            ph[k] = [int(px), int(py)]
        except Exception:  # noqa: BLE001
            continue
    if ph:
        l["phys"] = ph
        k = "%d,%d" % tuple(l["first"]["core"])
        if k in ph:
            l["first"]["phys"] = ph[k]
    return l


def loc_where(f):
    """'compute core (X,Y) of the GXxGY grid, element (ER,EC) of tile TR,TC (face F, element FR,FC of that face)' for a
    core_of() dict: the class a numeric fault gets unless a DRAM readback check failed."""
    ph = (f" (translated worker core {f['phys'][0]},{f['phys'][1]})"   # not the die location
          if f.get("phys") else "")
    return (f"compute core ({f['core'][0]},{f['core'][1]}){ph} of the {f['grid'][0]}x{f['grid'][1]} grid, "
            f"element ({f['element'][0]},{f['element'][1]}) of tile {f['tile'][0]},{f['tile'][1]} "
            f"(16x16 face {f['face'][0]},{f['face'][1]}, element {f['in_face'][0]},{f['in_face'][1]} of that face; "
            f"tile {f['block_tile'][0]},{f['block_tile'][1]} of the core's {f['per_core'][0]}x{f['per_core'][1]} tile block)")


def loc_text(c):
    """', first wrong element row R col C in compute core (X,Y) ..., N bad tiles, compute cores x,y,n [...], ...'"""
    l = c.get("loc")
    if not l:
        return ""
    f = l["first"]
    return (f", first wrong element row {f['row']} col {f['col']} in {loc_where(f)}, {l['tiles']} bad tiles, "
            f"compute cores x,y,n {l['cores']}, in-tile elements row,col,n {l['elements']}"
            + (f", first {len(l['positions'])} of {l['wrong']} positions (row,col,tile row,tile col,core x,core y,"
               f"element row,element col) {l['positions']}" if l.get("positions") else ""))


def power_policy():
    """The KMD power_policy parameter run.sh read (TRAY_POWER_POLICY, exported from
    /sys/module/tenstorrent/parameters/power_policy); 'unknown' when it is missing."""
    return os.environ.get("TRAY_POWER_POLICY") or "unknown"


def bank_spread(banks):
    """Tag for a numeric FAIL whose bad DRAM banks are all bank 3 (GDDR3 CA failure after
    self-refresh exit with KMD power_policy enabled). banks: the per-bank histogram."""
    if banks and sum(banks) and all(not b or i == 3 for i, b in enumerate(banks)):
        return f" matches the bank-3 pattern (GDDR3 CA after self-refresh exit; power_policy={power_policy()})"
    return ""


def sfpu_text(c):
    """', SFPU vote core x,y X,Y (<op>, N elems); ...' for a chip whose heavy SFPU vote mismatched: the cores
    that computed the wrong rows of the height-sharded SFPU output, most wrong elements first; '' when none."""
    items = sorted(((n, op, xy) for op, cs_ in (c.get("sfpu") or {}).items() for xy, n in cs_.items()),
                   key=lambda t: (-t[0], t[1], t[2]))
    return (", SFPU vote " + "; ".join(f"core x,y {xy} ({op}, {n} elems)" for n, op, xy in items[:8])) if items else ""


def sfpu_count(c):
    """'k/N SFPU checks mismatched (<op> k/N, ...)' for a chip whose heavy SFPU vote mismatched, '' when it did not: counted
    against the SFPU checks per op (the verify rows' sfpu_n), never the heavy part's launches or checks."""
    mis, n = c.get("sfpu_mis") or {}, c.get("sfpu_n") or {}
    if not mis:
        return ""
    ops = sorted(set(mis) | set(n))
    return (f"{sum(mis.values())}/{sum(n.get(o, 0) for o in ops) or '?'} SFPU checks mismatched ("
            + ", ".join(f"{o} {mis.get(o, 0)}/{n.get(o, '?')}" for o in ops) + ")")


def err_text(c):
    """', max abs err E, wrong elements N[, non-finite]' for a chip's numeric() entry, '' when no error stats were taken."""
    return _err_text(c) + sfpu_text(c)


def _err_text(c):
    if c.get("wrong_elems") is None:
        return loc_text(c)
    e = c["max_abs_err"]
    return (f", max abs err {'inf' if e == math.inf else format(e, '.3g')}, wrong elements {c['wrong_elems']}, "
            f"{'non-finite values present' if c['nonfinite_seen'] else 'all finite'} (a re-launch after the first mismatch)"
            + loc_text(c))


def numeric(rows):
    """Per chip over all chunk-end verifies: {bdf: {dev, tray, launches, mismatched, parts, first_chunk, nonfinite}}."""
    out = {}
    for r in rows:
        if r.get("event") != "verify":
            continue
        end = r.get("k", -1) >= 0   # k=-1: the pre-load reference readback: its faults count, it never counts as checked
        for cid, c in (r.get("chips") or {}).items():
            key = c.get("pci") or f"metal{c.get('metal', cid)}"
            a = out.setdefault(key, dict(dev=c.get("dev", cid), tray=c.get("tray"), launches=0, mismatched=0,
                                         parts=set(), first_chunk=None, nonfinite=0, errors=set(), saturated=False,
                                         by_chunk={}, chunks=set(), bad_chunks=set(), sat_chunks=set()))
            if end:
                a["chunks"].add(r["k"])
                if c.get("mismatched") or c.get("bad_parts"):
                    a["bad_chunks"].add(r["k"])
                if c.get("saturated"):
                    a["sat_chunks"].add(r["k"])
            if end:
                a["launches"] += c.get("launches", 0)
                for op, k in (c.get("sfpu_n") or {}).items():   # heavy SFPU vote: checks per op
                    a.setdefault("sfpu_n", {})[op] = a.get("sfpu_n", {}).get(op, 0) + k
                if c.get("heavy_checks") is not None:   # --heavy-check-each: checked outputs per heavy op
                    a["heavy_checks"] = a.get("heavy_checks", 0) + c["heavy_checks"]
            a["mismatched"] += c.get("mismatched", 0)
            a["by_chunk"][r.get("k", -1)] = a["by_chunk"].get(r.get("k", -1), 0) + c.get("mismatched", 0)
            a["nonfinite"] += c.get("nonfinite", 0)
            if c.get("wrong_elems") is not None and a.get("wrong_elems") is None:   # the first verify that measured it
                a.update(max_abs_err=c.get("max_abs_err", 0.0), wrong_elems=c["wrong_elems"], nonfinite_seen=bool(c.get("nonfinite_seen")))
            if c.get("loc") and not a.get("loc"):
                a["loc"] = c["loc"]
            for op, k in (c.get("sfpu_mis") or {}).items():   # heavy SFPU vote: mismatching checks per op
                a.setdefault("sfpu_mis", {})[op] = a.get("sfpu_mis", {}).get(op, 0) + k
            for op, cs_ in (c.get("sfpu") or {}).items():   # heavy SFPU vote: wrong elements per core x,y
                d = a.setdefault("sfpu", {}).setdefault(op, {})
                for xy, n in cs_.items():
                    d[xy] = max(d.get(xy, 0), n)
            a["saturated"] = a["saturated"] or bool(c.get("saturated"))
            for p in c.get("bad_parts") or []:
                a["parts"].add(p)
            for e in c.get("errors") or ([c["error"]] if c.get("error") else []):   # "<part>: <exception>"
                a["errors"].add(e)
            if (c.get("mismatched") or c.get("bad_parts")) and a["first_chunk"] is None:
                a["first_chunk"] = r["k"]
            if c.get("first_bad") and not a.get("first_bad"):   # the first chunk's
                a["first_bad"] = dict(c["first_bad"], k=r.get("k", 0))
    return out


def ccl_groups(rows, chips, skip=()):
    """CCL groups of the h3 all-gather whose every chip has the same nonzero CCL-only mismatch count in every chunk: one
    bad result spread by the group's all-gather, not a fault on each chip (seen in testing: mesh
    columns 03/07/47/43:00.0 at 0/2/2/2 and c4/c8/88/84:00.0 at 0/3/0/0). Groups from the setup row's mesh [rows, cols]
    and mesh_ids (metal id per mesh position, row-major): the whole mesh, then each column, then each row; a chip goes
    in one group only. [(label, [bdf in mesh order])]; [] when the rows carry no mesh."""
    st = next((r for r in rows if r.get("event") == "setup" and r.get("mesh") and r.get("mesh_ids")), None)
    try:
        nr, nc = (int(x) for x in st["mesh"])
        pci = {c["metal"]: c["pci"] for c in st.get("chips") or []}
        pos = [pci.get(m) for m in st["mesh_ids"]]
    except (TypeError, ValueError, KeyError):
        return []
    if len(pos) != nr * nc:
        return []
    out, used = [], set()
    for label, ks in ([("the whole mesh", pos)] + [(f"mesh column {j} (of 0-{nc - 1})", pos[j::nc]) for j in range(nc)]
                      + [(f"mesh row {i} (of 0-{nr - 1})", pos[i * nc:(i + 1) * nc]) for i in range(nr)]):
        if len(ks) < 2 or used & set(ks) or any(k not in chips or k in skip for k in ks):
            continue
        cs_ = [chips[k] for k in ks]
        if all(c["mismatched"] and c["parts"] and c["parts"] <= CCL_PARTS for c in cs_) \
                and len({tuple(sorted(c["by_chunk"].items())) for c in cs_}) == 1:
            out.append((label, ks))
            used |= set(ks)
    return out


def hms(epoch):
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def load_power(path):
    """fleet.sh's power.txt rows "<epoch> <UBB0..3 W> <HSC W> <vr_pg UBB0..3>" -> [(epoch, {tray: W})]; NA/ERR skipped."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                v = line.split()
                if len(v) < 5:
                    continue
                try:
                    t = float(v[0])
                except ValueError:
                    continue
                out.append((t, {u + 1: int(v[1 + u]) for u in range(4) if v[1 + u].isdigit()}))
    except OSError:
        pass
    return out


def load_psu(path):
    """power.txt's PSU columns 11-14 (Power_PSU0-3): [(epoch, [W x4])] for rows where all 4 read. The box kW
    is their sum: the tray sensors cap at 2550 W each, so the trays' sum cannot show 12-14 kW."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                v = line.split()
                if len(v) >= 14 and all(x.isdigit() for x in v[10:14]):
                    try:
                        out.append((float(v[0]), [int(x) for x in v[10:14]]))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


AICLK_MAX = float(os.environ.get("FLEET_AICLK_MAX", "1350"))   # Blackhole AICLK at full speed; below it under load = limited


def load_telem(path):
    """C8's c8_telemetry.txt rows "<epoch> <bdf> <temp mC> <power uW> <aiclk MHz> <curr mA> <vcore mV> [curr max mA]"
    -> [(epoch, bdf, W, MHz, A, A max or None)]; a row with no power or clock is skipped."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                v = line.split()
                try:
                    w, mhz = float(v[3]) * 1e-6, float(v[4])
                    a = float(v[5]) * 1e-3 if len(v) > 5 and v[5] != "-" else None
                    am = float(v[7]) * 1e-3 if len(v) > 7 and v[7] != "-" else None
                except (ValueError, IndexError):
                    continue
                if 1 <= w <= 1000 and 100 <= mhz <= 3000:
                    out.append((float(v[0]), v[1], w, mhz, a, am))
    except OSError:
        pass
    return out


def chips_in(telem, start, end):
    """Per chip over [start, end] (the first 10 s skipped: the clock ramps): median W, MHz, A and whether the firmware
    held it at a limit (median clock under AICLK_MAX - 20 MHz, or current within 2% of its max)."""
    import statistics as st
    by = {}
    for t, b, w, mhz, a, am in telem:
        if start + 10 <= t <= end:
            by.setdefault(b, []).append((w, mhz, a, am))
    out = {}
    for b, xs in by.items():
        w, mhz = st.median(x[0] for x in xs), st.median(x[1] for x in xs)
        aa = [x[2] for x in xs if x[2] is not None]
        a = st.median(aa) if aa else None
        am = next((x[3] for x in xs if x[3]), None)
        lim = mhz < AICLK_MAX - 20 or (a is not None and am is not None and a >= 0.98 * am)
        out[b] = dict(w=w, mhz=mhz, a=a, am=am, limit=lim, tray=tray_of(b))
    return out


def chips_line(r, n, telem):
    """CHIPS line per chunk: per-chip median power and clock, and the chips the firmware held at a limit."""
    c = chips_in(telem or [], r["start"], r["end"])
    if not c:
        return f"CHIPS chunk {r['k'] + 1}/{n}: no chip telemetry in the chunk"
    ws, ms = sorted(x["w"] for x in c.values()), sorted(x["mhz"] for x in c.values())
    lim = sorted((b for b, x in c.items() if x["limit"]), key=lambda b: (c[b]["tray"] or 9, b))
    order = sorted(c, key=lambda b: (c[b]["tray"] or 9, b))
    return (f"CHIPS chunk {r['k'] + 1}/{n}: {len(c)} chips, power min/median/max {ws[0]:.0f}/{ws[len(ws) // 2]:.0f}/{ws[-1]:.0f} W, "
            f"aiclk min/median/max {ms[0]:.0f}/{ms[len(ms) // 2]:.0f}/{ms[-1]:.0f} MHz, at firmware limit {len(lim)}/{len(c)}"
            + (" (" + ", ".join(f"{short(b)} tray {c[b]['tray'] or '?'}" for b in lim[:8]) + (" ..." if len(lim) > 8 else "") + ")"
               if lim else "")
            + "; per chip: " + ", ".join(f"{short(b)} t{c[b]['tray'] or '?'} {c[b]['w']:.0f} W {c[b]['mhz']:.0f} MHz"
                                         + (f" {c[b]['a']:.0f} A" if c[b]["a"] is not None else "") + ("*" if c[b]["limit"] else "")
                                         for b in order))


def peaks_line(r, n, power, psu, telem, fast=None):
    """PEAK line per chunk (do the pauses between bursts shoot the chips over their limit?):
    each chip's peak W and peak A over the whole chunk, by BDF and tray 1-4, next to the chunk's box kW (PSU0-3 sum, else
    the trays' sum) and TFLOP/s per chip. Peaks are the highest C8 sample (sysfs hwmon, every FLEET_TELEM_S s), so a spike
    shorter than one sample can be missed: a floor on the true peak, never above it. fast: rows of the fast window
    FastPoll takes at each chunk start (every ~0.1 s for a few s); they join the peaks, never the CHIPS medians."""
    u = r.get("util") or {}
    s = f"PEAK chunk {r['k'] + 1}/{n}: "
    full = [(t, sum(p)) for t, p in psu] if psu else [(t, sum(p.values())) for t, p in (power or []) if len(p) == 4]
    kw = [w for t, w in full if r["start"] <= t <= r["end"]]
    src = "box (PSU0-3 sum)" if psu else "trays' sum (capped 2550 W a tray)"
    s += (f"{src} peak {max(kw) / 1000:.2f} kW mean {sum(kw) / len(kw) / 1000:.2f} kW ({len(kw)} polls), " if kw
          else "box kW not read, ")
    s += f"{u['tflops_chip']:g} TFLOP/s per chip" if u.get("tflops_chip") is not None else "TFLOP/s not reported"
    by = {}
    nf = 0
    for t, b, w, mhz, a, am in list(telem or []) + list(fast or []):
        if r["start"] <= t <= r["end"]:
            nf += (t, b, w, mhz, a, am) in (fast or [])
            x = by.setdefault(b, [[], [], []])
            x[0].append(w)
            x[2].append(t)
            if a is not None:
                x[1].append(a)
    if not by:
        return s + "; chip peaks: no chip telemetry in the chunk"
    order = sorted(by, key=lambda b: (tray_of(b) or 9, b))
    ts = {b: sorted(set(by[b][2])) for b in by}
    gaps = sorted(y - x for b in by for x, y in zip(ts[b], ts[b][1:]) if y - x >= 1)
    every = f"every ~{gaps[len(gaps) // 2]:.0f} s" if gaps else "1 sample"
    if nf:
        fg = sorted(y - x for b in by for x, y in zip(ts[b], ts[b][1:]) if 0 < y - x < 1)
        every += f" + {nf} fast samples" + (f" every ~{fg[len(fg) // 2] * 1000:.0f} ms" if fg else "")
    nm = lambda b: f"{short(b)} (tray {tray_of(b) or '?'})"  # noqa: E731
    wb = max(order, key=lambda b: max(by[b][0]))
    ab = [b for b in order if by[b][1]]
    ab = max(ab, key=lambda b: max(by[b][1])) if ab else None
    return (s + f"; chip peaks ({len(by)} chips, C8 samples {every}): top {max(by[wb][0]):.0f} W {nm(wb)}"
            + (f", top {max(by[ab][1]):.0f} A {nm(ab)}" if ab else ", A not read")
            + "; per chip peak W/A: " + ", ".join(f"{nm(b)} {max(by[b][0]):.0f} W "
                                                  + (f"{max(by[b][1]):.0f} A" if by[b][1] else "? A") for b in order))


def psu_drops(psu, rows, n):
    """A PSU reading 0 W inside a chunk while the others read power: that PSU dropped out under load (a finding)."""
    out = []
    for r in rows:
        if r.get("event") != "chunk":
            continue
        for u in range(4):
            z = [t for t, p in psu if r["start"] <= t <= r["end"] and p[u] == 0 and sum(p) > 0]
            if z:
                out.append(f"POWER WARN PSU{u} read 0 W at {hms(z[0])} in chunk {r['k'] + 1}/{n} ({len(z)} polls) while the "
                           f"box drew load: a PSU dropped out (SEL lines in c1_post)")
    return out


def power_in(power, start, end):
    """BMC tray power of the polls inside [start, end]: "power tray 1-4 avg a/b/c/d W, total avg T min M W (n polls)"."""
    rows = [p for t, p in power if start <= t <= end]
    if not rows:
        return "power: no poll in the window"
    avg = [sum(p[t] for p in rows if t in p) / max(1, sum(1 for p in rows if t in p)) for t in (1, 2, 3, 4)]
    tot = [sum(p.values()) for p in rows if len(p) == 4]
    tr = "/".join(f"{a:.0f}" if any(t in p for p in rows) else "?" for t, a in zip((1, 2, 3, 4), avg))
    return (f"power tray 1-4 avg {tr} W" + (f", total avg {sum(tot) / len(tot):.0f} min {min(tot)} W" if tot else "")
            + f" ({len(rows)} poll{'s' if len(rows) != 1 else ''})")


def topup_text(t):
    """The pre-ramp's top-up (chunk_load.py's TopUp.info()): which chips got more heavy ops per burst, and why."""
    if not isinstance(t, dict):
        return "top-up not run"
    if t.get("why"):
        return f"top-up off: {t['why']}"
    if not t.get("chips"):
        return f"top-up: none needed ({t.get('at_limit', '?')}/{t.get('measured', '?')} chips at their power or current limit)"
    f = lambda v: "?" if v is None else f"{v:.0f}"  # noqa: E731
    return (f"top-up on {len(t['chips'])} chip(s) below their limit ({t.get('at_limit', '?')}/{t.get('measured', '?')} at it), "
            f"{t.get('op', 'heavy matmul')} per {t.get('steps', '?')}-op burst: "
            + ", ".join(f"{c['bdf']} (tray {c.get('tray') or '?'}) +{c['extra']} at {f(c.get('power_w'))}/{f(c.get('tdp_w'))} W"
                        for c in t["chips"]))


def load_line(r, n, power, quiet=None, psu=None, telem=None):
    """One LOAD line per chunk row: the trays' summed kW at the step
    (the first 10 s after the quiet), the chunk mean and the idle just before, from the BMC tray power poll; TFLOP/s per
    chip from the launch rate (idle: the quiet row just before, if any); the heavy op, run-ahead and checks from the row's "load"; and every fallback."""
    u, ld = r.get("util") or {}, r.get("load")
    ln = f" ({mins(r['chunk_s'])} min)" if r.get("chunk_s") else ""
    s = f"LOAD chunk {r['k'] + 1}/{n}{ln}: "
    kw = lambda v: f"{v / 1000:.2f}"  # noqa: E731
    full = [(t, sum(p.values())) for t, p in (power or []) if len(p) == 4]
    src = "trays' sum (capped 2550 W a tray)"
    if psu:   # the box's draw is the PSU sum; the trays' sum is only the fallback
        full, src = [(t, sum(p)) for t, p in psu], "box (PSU0-3 sum)"
    step = [w for t, w in full if r["start"] <= t <= r["start"] + 10]
    mean = [w for t, w in full if r["start"] <= t <= r["end"]]
    idle = [w for t, w in full if quiet and quiet["start"] <= t <= quiet["end"]]
    if power is None or not mean:
        s += f"box power not read ({len(mean)} polls in the chunk), "
    else:
        s += (f"{src} step peak {kw(max(step)) if step else '-'} kW in the first 10 s, chunk mean "
              f"{kw(sum(mean) / len(mean))} kW peak {kw(max(mean))} kW ({len(mean)} polls), idle before "
              + (f"{kw(sum(idle) / len(idle))} kW, " if idle else "- (no quiet polled before it), "))
    s += f"{u['tflops_chip']:g} TFLOP/s per chip" if u.get("tflops_chip") is not None else "TFLOP/s not reported"
    if r.get("chunk_s"):
        s += f", ends {r['end'] - r['start'] - r['chunk_s']:+.1f} s after plan"
    if mean and psu:   # 12-14 kW through each whole chunk
        mk = sum(mean) / len(mean) / 1000
        c = chips_in(telem or [], r["start"], r["end"])
        nl = sum(1 for x in c.values() if x["limit"])
        if os.environ.get("CHUNK_KW_TARGET", "1") == "0":   # the bus-bar stage: kW is not its target
            s += f", no kW target (bus-bar stage, goal 12; {mk:.2f} kW): " + (
                f"chips at their firmware limit {nl}/{len(c)}" if c else "no chip telemetry")
        elif mk < 12:
            s += f", BELOW the 12-14 kW target ({mk:.2f} kW)"   # why: the kW-why line (95% of TDP rule), sustained chunks only
        elif mk > 14:
            s += f", above the 12-14 kW target ({mk:.2f} kW)"
        else:
            s += f", in the 12-14 kW target ({mk:.2f} kW)"
    if not isinstance(ld, dict):
        return s + "; fallback: load not reported (no per-launch-sync-free chunk_load)"
    if ld.get("heavy"):
        s += f", {ld['heavy']}"
    if ld.get("gap"):   # the heavy part's bursts and the gap after each
        s += f", {ld['gap']}"
    if ld.get("run_ahead"):
        s += f", {ld['run_ahead']}"
    if ld.get("concurrent"):
        s += f", {ld['concurrent']}"
    if ld.get("topup"):   # more heavy ops on the chips the pre-ramp found below their limit
        s += f", {ld['topup']}"
    if ld.get("checks") is not None:
        s += f", {ld['checks']} numeric checks ({ld.get('checks_per_launch', '?')} per launch)"
    return s + "; fallback: " + ("; ".join(ld.get("fallback") or []) or "none")


def nchunks(rows):
    for r in rows:
        if r.get("event") == "setup" and isinstance(r.get("chunks"), (list, int)):
            return len(r["chunks"]) if isinstance(r["chunks"], list) else r["chunks"]
    return next((r["n"] for r in rows if r.get("event") == "start" and isinstance(r.get("n"), int)), None)  # run()'s own rows


def kw_why_line(r, psu, c8, limits):
    """Under a chunk's LOAD line, when its box kW is under 12: power-limit ceiling or load stall,
    from C8's telemetry and its read-only limits (load_line.at_limit/kw_why). None at or above 12 kW, with no PSU poll in
    the chunk, or in the bus-bar stage (CHUNK_KW_TARGET=0: kW is not its target)."""
    mean = [sum(p) for t, p in (psu or []) if r["start"] <= t <= r["end"]]
    if not mean or os.environ.get("CHUNK_KW_TARGET", "1") == "0" or sum(mean) / len(mean) >= 12000:
        return None
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import load_line as ll
    rec = ll.at_limit(ll.c8_rows(c8), ll.c8_limits(limits), r["start"], r["end"])
    return ll.kw_why(r["k"] + 1, sum(mean) / len(mean) / 1000, rec, (r.get("util") or {}).get("launch_per_s"), min(mean) / 1000)


def report(rows, power=None, psu=None, telem=None, c8=None, limits=None, fast=None):
    """fleet.sh's lines. NUMERIC FAIL keeps numeric_report.py's shape (bin/fault-ledger parses it), naming the tray.
    power: load_power()'s list; each chunk and quiet line then ends with its BMC tray power. c8, limits: C8's telemetry
    and limits files; with limits, a chunk under 12 kW gets kw_why_line() under its LOAD line."""
    out = []
    n = nchunks(rows) or "?"
    legacy = not any(r.get("event") == "start" for r in rows)   # older rows: quiet k came before chunk k
    last_quiet = None
    pw = (lambda r: "; " + power_in(power, r["start"], r["end"])) if power is not None else (lambda r: "")
    st = next((r for r in rows if r.get("event") == "setup" and r.get("chips")), None)   # part_off rows come before it
    by_tray = {}
    for c in (st or {}).get("chips") or []:
        by_tray.setdefault(c.get("tray") or tray_of(c.get("pci")), []).append(short(c.get("pci")))
    trays = [(f"tray {t or '?'}", " ".join(sorted(bs)))
             for t, bs in sorted(by_tray.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))]   # BDFs under their tray
    for r in rows:
        if r.get("event") == "chunk":
            u = r.get("util") or {}
            ln = f" ({mins(r['chunk_s'])} min)" if r.get("chunk_s") else ""
            out.append(f"CHUNK {r['k'] + 1}/{n}{ln}: {hms(r['start'])}-{hms(r['end'])}, {r['end'] - r['start']:.0f} s, "
                       f"{r['launches']} launches, {r['blocks']} blocks"
                       + "".join(f", {k} {v}" for k, v in sorted(u.items()) if not isinstance(v, (dict, list))) + pw(r))
            out.append(load_line(r, n, power, last_quiet, psu, telem))
            try:
                why = kw_why_line(r, psu, c8, limits) if limits is not None else None
            except Exception as e:   # never lose the stage's NUMERIC/CHUNK/QUIET FAIL lines to the extra line (fl_chunk_report: || return 0)
                why = f"LOAD chunk {r['k'] + 1}: kW-why not computed ({type(e).__name__}: {e})"
            if why:
                out.append(why)
            if telem is not None:
                out.append(chips_line(r, n, telem))
                out.append(peaks_line(r, n, power, psu, telem, fast))
        elif r.get("event") == "quiet":
            at = (f"before chunk {r['k'] + 1}/{n}" if legacy else
                  "before chunk 1" if r["k"] < 0 else f"after chunk {r['k'] + 1}/{n}")
            st = f", {r['settle_s']:.1f} s to idle clock before it" if r.get("settle_s") is not None else ""
            out.append(f"QUIET {at}: {hms(r['start'])}-{hms(r['end'])}, {r['end'] - r['start']:.1f} s idle{st}" + pw(r))
            if r.get("never"):    # a chip above 800 MHz after the 5 min cap is a finding
                out.append(f"QUIET FAIL {at}: chip(s) not at idle clock (<= 800 MHz) after {r['settle_s']:.0f} s (cap "
                           f"{r.get('cap_s', 300):.0f} s): " + ", ".join(
                               f"{x.get('pci')} (tray {x.get('tray') or '?'}) chip {x.get('chip')} at {x.get('mhz')} MHz"
                               for x in r["never"]) + "; the quiet went on without them")
            last_quiet = r
        elif r.get("event") == "start" and r.get("aiclk"):
            sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from quiet_clock import grouped
            out.append(f"AICLK chunk {r['k'] + 1}/{n} start: "
                       + grouped({int(c): v for c, v in r["aiclk"].items()}))
        elif r.get("event") == "part_off":   # name the chips that ran without it; h3 off = no CCL mix
            p = r.get("part")
            what = " (the CCL mix: no all-gather or reduce-scatter on the ETH links, the load went on without it)" if p in CCL_PARTS else ""
            head = f"CHUNK WARN part {p} off{what}: {str(r.get('error', ''))[:160]}"
            line = f"{head}; chips: " + ("; ".join(f"{t}: {b}" for t, b in trays) or "not named (no setup row with chips)")
            if len(line) <= 600 or not trays:
                out.append(line)
            else:   # too long for one line (4x8): the head, then one line per tray
                out.append(head + "; chips by tray:")
                out += [f"CHUNK WARN part {p} off, {t}: {b}" for t, b in trays]
        elif r.get("event") == "preramp_end":   # the uncounted full-load pre-ramp and the top-up it chose
            out.append(f"PRERAMP before chunk 1/{n} (not counted): {hms(r['start'])}-{hms(r['end'])}, "
                       f"{r['end'] - r['start']:.0f} s, {r.get('launches', 0)} launches" + pw(r) + "; " + topup_text(r.get("topup")))
    out += psu_drops(psu or [], rows, n)
    chips = numeric(rows)
    checked = {t: 0 for t in (1, 2, 3, 4)}
    # a CCL part mixes every chip's data (all-gather, reduce-scatter): a chip that fails a chip-local part spreads its
    # bad data to the others, so their CCL-only mismatch is not their fault (seen in testing: [h3] on all 32 chips)
    local_bad = sorted((k for k, c in chips.items() if c["parts"] - CCL_PARTS),
                       key=lambda k: (chips[k]["tray"] is None, chips[k]["tray"] or 0, k))
    spread = {k for k, c in chips.items() if local_bad and c["parts"] and c["parts"] <= CCL_PARTS}
    groups = ccl_groups(rows, chips, spread)
    grouped = {k for _, ks in groups for k in ks}
    for key, c in sorted(chips.items(), key=lambda kv: (kv[1]["tray"] is None, kv[1]["tray"] or 0, kv[0])):
        if c["tray"] in checked and c["launches"]:
            checked[c["tray"]] += 1
        if key in grouped:   # its counts stay in the log; the CCL group's one finding follows
            out.append(f"NUMERIC CCL member {short(key)} (tray {c['tray'] or '?'}) chip {c['dev']} "
                       f"[{','.join(sorted(c['parts']))}]: {c['mismatched']}/{c['launches']} launches mismatched, per chunk "
                       f"{per_chunk(c)}, first in chunk {c['first_chunk'] + 1}, non-finite elements {c['nonfinite']}{err_text(c)}{first_bad_text(c)}; "
                       f"counted in the CCL group fault of {next(g for g, ks in groups if key in ks)}")
        elif (c["mismatched"] or c["parts"]) and key not in spread:
            out.append(f"NUMERIC FAIL {short(key)} (tray {c['tray'] if c['tray'] else '?'}) chip {c['dev']} "
                       f"[{','.join(sorted(c['parts'])) or 'acc'}]: "
                       + (f"{len(c['bad_chunks'])} mismatching chunks of {len(c['chunks'])}, counter saturated in chunks "
                          f"{','.join(str(k + 1) for k in sorted(c['sat_chunks'])) or '?'} (launch counts not summed)"
                          if c['saturated'] else f"{sfpu_count(c)} ({c['launches']} launches; heavy outputs all matched)"
                          if c.get('sfpu_mis') and not c['mismatched']
                          else (f"{c['mismatched']}/{c['heavy_checks']} checked outputs mismatched "
                                f"({c['launches']} launches, every heavy output checked)" if c.get('heavy_checks')
                                else f"{c['mismatched']}/{c['launches']} launches mismatched")
                          + (f", SFPU vote {sfpu_count(c)}" if c.get('sfpu_mis') else ""))
                       + f", "
                       f"first in chunk {c['first_chunk'] + 1}, non-finite elements {c['nonfinite']}{err_text(c)}{first_bad_text(c)}")
    # a chunk-end readback that raised: that chip's part was not checked, never a PASS; one line per part if all chips
    order = lambda k: (chips[k]["tray"] is None, chips[k]["tray"] or 0, k)  # noqa: E731
    errs = {}
    for k, c in chips.items():
        for e in c["errors"]:
            errs.setdefault(e, []).append(k)
    # the part is the error's '<part>: ' prefix only when it names a part that ran (the setup row's parts); an error with
    # no prefix ('RuntimeError: ...') names the one part that ran, else says so, never the exception type
    ran = next((r.get("parts") for r in rows if r.get("event") == "setup" and r.get("parts")), None)
    ran = ran.split(",") if isinstance(ran, str) else list(ran or [])
    for e, ks in sorted(errs.items()):
        p, _, msg = e.partition(": ")
        if msg and p.startswith("sfpu-vote:") and ("heavy" in ran if ran else True):   # 'sfpu-vote:<op>: '
            pass
        elif not (msg and (p in ran or p.removesuffix("-ref") in ran if ran else re.fullmatch(r"[a-z][a-z0-9]*(-ref)?", p))):
            p, msg = (ran[0] if len(ran) == 1 else "part not named"), e
        why = f"[{p}]: NUMERIC NOT CHECKED, the chunk-end readback failed: {msg[:200]}"
        if len(ks) > 1 and len(ks) == len(chips) and re.search(r"timeout|timed out", msg, re.I):
            # a device timeout on every chip together is a box-wide hang: no chip or tray named, no hardware
            # verdict (INCOMPLETE); a timeout on some chips, or any other error on all, names its chips and trays below
            out.append(f"HANG INCOMPLETE whole box: all {len(ks)} chips [{p}]: NUMERIC NOT CHECKED, the chunk-end readback "
                       f"hit a device timeout on every chip together; no chip named, no hardware verdict: {msg[:200]}")
        elif len(ks) > 1 and len(ks) == len(chips):
            out.append(f"CHUNK FAIL all {len(ks)} chips ("
                       + ", ".join(f"tray {t}" for t in sorted({chips[k]["tray"] or "?" for k in ks}, key=str)) + f") {why}")
        else:
            out += [f"CHUNK FAIL {short(k)} (tray {chips[k]['tray'] or '?'}) chip {chips[k]['dev']} {why}"
                    for k in sorted(ks, key=order)]
    if spread:
        out.append(f"NUMERIC WARN [{','.join(sorted(set().union(*(chips[k]['parts'] for k in spread))))}] mismatch on "
                   f"{len(spread)} other chip(s) follows from "
                   + ", ".join(f"{short(k)} (tray {chips[k]['tray'] or '?'})" for k in local_bad)
                   + " through the CCL (all-gather/reduce-scatter); not counted as their fault")
    out += group_lines(rows, chips, groups, n)
    rw_bad = set()
    for r in rows:
        if r.get("event") in ("rw", "dbi", "l1"):
            lines, bad = {"rw": rw_lines, "dbi": dbi_lines, "l1": l1_lines}[r["event"]](r, n)
            out += lines; rw_bad |= bad
    caps = [r for r in rows if r.get("event") == "capture"]
    if caps:   # the in-run capture's events (numeric_capture.py), one NUMERIC-CAPTURE line each
        sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import numeric_capture as nc
        for r in caps:
            out.append(nc.line(r, r.get("pci") or "?", r.get("tray") or "?", r.get("op_name") or "heavy")
                       if r.get("pci") else f"NUMERIC-CAPTURE off: {r.get('error', '?')}")
    lost = lost_chunks(rows)
    if lost:
        ref = {}   # chips the pre-load verify saw: what the lost chunk should have read back
        for r in rows:
            if r.get("event") == "verify" and r.get("k", -1) < 0:
                for cid, c in (r.get("chips") or {}).items():
                    ref[c.get("pci") or f"metal{c.get('metal', cid)}"] = c.get("tray")
        for k, st in lost:
            out.append(f"CHUNK FAIL chunk {k + 1}/{n} (started {hms(st)}): no chunk-end readback, the load died or hung; "
                       "numeric not checked on " + ("; ".join(
                           f"tray {t}: " + ",".join(short(b) for b in sorted(ref) if ref[b] == t)
                           for t in sorted({v for v in ref.values() if v}))
                                                  or "any chip"))
    ld, ld_bad = linkdata_lines(rows)
    out += ld
    bad_trays = {c["tray"] for k, c in chips.items() if (c["mismatched"] or c["parts"]) and k not in spread or c["errors"]} | rw_bad | ld_bad
    for t in (1, 2, 3, 4):
        if t in bad_trays:
            continue
        if lost:
            out.append(f"NUMERIC NOT CHECKED tray {t} (chunk {lost[0][0] + 1}/{n} never reached its chunk-end readback)")
            continue
        out.append(f"NUMERIC PASS tray {t} ({checked[t]} chips)" if checked[t] else
                   f"NUMERIC NOT CHECKED tray {t} (no chip of it in a chunk-end verify)")
    out += launch_lines(rows, chips)
    if not chips and not lost:
        out.append("NUMERIC: not checked (no chunk-end verify ran)")
    return out


def launch_lines(rows, chips):
    """The DIDT numeric step's launch counts, so the odds of catching a
    rare fault (81:00.0, tray 4: bf8 ~1 in 2700) can be computed. One "NUMERIC LAUNCHES <op>" line per --heavy op (dtype):
    its runs and checked outputs per chip over the checked chunks (each launch runs every op heavy_n times, back to back;
    under --heavy-check-each every run's output is checked, else the last of each launch) and its checked outputs/s over
    those chunks' wall time. Then one "NUMERIC RATE" line per chip: BDF, tray 1-4, launches, seconds of its
    checked chunks, launches/s and checked outputs/s per op. Only when the setup row names the ops (chunk_load.py's
    heavy_ops); else nothing."""
    st = next((r for r in rows if r.get("event") == "setup" and r.get("heavy_ops")), None)
    if st is None:
        return []
    n, hc = int(st.get("heavy_n") or 1), int(st.get("heavy_checks") or 1)
    ops = st["heavy_ops"]
    dur = {r["k"]: r["end"] - r["start"] for r in rows if r.get("event") == "chunk" and r.get("k", -1) >= 0}
    secs = sum(dur.values())
    ls = [c["launches"] for c in chips.values() if c.get("launches")]
    if not ls:
        return [f"NUMERIC LAUNCHES {op}: 0 per chip (no chip reached a chunk-end verify)" for op in ops]
    lo, hi = min(ls), max(ls)
    span = (lambda x, y: f"{x}" if x == y else f"{x}-{y}")
    what = "every run's output checked" if hc > 1 else "the last run of each launch checked"
    rate = (lambda x, s: f"{x / s:.2f}/s over {s:.0f} s") if secs > 0 else (lambda x, s: "rate unknown (no chunk time)")
    out = [f"NUMERIC LAUNCHES {op}: {span(lo * n, hi * n)} runs, {span(lo * hc, hi * hc)} checked per chip ({len(ls)} chips; "
           f"{span(lo, hi)} launches x heavy_n {n}, {what}), {rate(hi * hc, secs)} checked" for op in ops]
    order = lambda k: (chips[k]["tray"] is None, chips[k]["tray"] or 0, k)  # noqa: E731
    for k in sorted(chips, key=order):
        c = chips[k]
        s = sum(dur.get(x, 0) for x in c.get("chunks") or ())
        lr = f"{c['launches'] / s:.3f} launches/s" if s > 0 else "launches/s unknown (no chunk time)"
        per = ", ".join(f"{op} {c['launches'] * hc}" + (f" ({c['launches'] * hc / s:.2f}/s)" if s > 0 else "") for op in ops)
        out.append(f"NUMERIC RATE {short(k)} (tray {c['tray'] or '?'}): {c['launches']} launches in {s:.0f} s, {lr}; "
                   f"checked outputs {per}")
    return out


def linkdata_lines(rows):
    """chunk_load's linkdata rows: one LINK-DATA FAIL per receiving chip, sender and all-gather, naming both
    chips by BDF and tray, the words wrong and the flipped bits with their place in a 32 B chunk. A link-data check, never
    numeric or bus-bar. Returns (lines, bad trays)."""
    import link_pattern as lp
    acc, bad = {}, set()
    for r in rows:
        if r.get("event") != "linkdata":
            continue
        key = (r.get("recv_bdf") or f"chip {r.get('recv')}", r.get("send_bdf") or f"chip {r.get('send')}", r.get("op") or "?",
               bool(r.get("link", True)))
        a = acc.setdefault(key, {"words": 0, "f10": {}, "f01": {}, "first": r.get("first")})
        a["words"] += int(r.get("words") or 0)
        for f in ("f10", "f01"):
            for q, n in (r.get(f) or {}).items():
                a[f][q] = a[f].get(q, 0) + int(n)
    out = []
    for (rv, sd, op, link), a in sorted(acc.items()):
        tr, ts = tray_of(rv), tray_of(sd)
        bad.add(tr)
        src = f"<- {short(sd)} (tray {ts or '?'})" if link else "local copy"
        out.append(f"LINK-DATA FAIL {short(rv)} (tray {tr or '?'}) {src} [{op}, link-data check]: {lp.fmt(a)}")
    return out, bad - {None}


def per_chunk(c):
    """A chip's mismatch count per chunk-end verify, "0/2/2/2" (the pre-load reference readback first as "ref N", if any)."""
    b = c.get("by_chunk") or {}
    return (f"ref {b[-1]}, " if b.get(-1) else "") + "/".join(str(v) for k, v in sorted(b.items()) if k >= 0)


def group_lines(rows, chips, groups, n):
    """One NUMERIC FAIL per CCL group from ccl_groups(), in the ledger shape: its lead chip is the
    likely source, a member that also failed outside the CCL (a G3 [rw] mismatch, a chip-local part), else the group's
    first chip with 'source not isolated'. The line lists every member by BDF and tray 1-4."""
    nonccl = {}
    for r in rows:
        if r.get("event") == "rw":
            for c in (r.get("chips") or {}).values():
                if c.get("bad_checks") and c.get("pci"):
                    nonccl.setdefault(c["pci"], []).append(
                        f"G3 [rw] {c['bad_checks']}/{c.get('checks', 0)} checks mismatched after chunk {r['k'] + 1}/{n}")
        if r.get("event") == "dbi":
            for c in (r.get("chips") or {}).values():
                if c.get("bad_passes") and c.get("pci"):
                    nonccl.setdefault(c["pci"], []).append(
                        f"G3b [dbi] {c['bad_passes']}/{c.get('passes', 0)} DRAM pattern passes mismatched after chunk {r['k'] + 1}/{n}")
        if r.get("event") == "l1":
            for c in (r.get("chips") or {}).values():
                if c.get("bad_passes") and c.get("pci"):
                    nonccl.setdefault(c["pci"], []).append(
                        f"N1 [l1] {c['bad_passes']}/{c.get('passes', 0)} L1 pattern passes mismatched after chunk {r['k'] + 1}/{n}")
    for k, c in chips.items():
        if c["parts"] - CCL_PARTS:
            nonccl.setdefault(k, []).append(f"[{','.join(sorted(c['parts'] - CCL_PARTS))}] mismatch")
    name = lambda k: f"{short(k)} (tray {chips[k]['tray'] or '?'}) chip {chips[k]['dev']}"  # noqa: E731
    out = []
    for label, ks in groups:
        src = [k for k in ks if k in nonccl]
        lead = chips[src[0] if src else ks[0]]
        c0 = chips[ks[0]]
        out.append(f"NUMERIC FAIL {name(src[0] if src else ks[0])} [{','.join(sorted(set().union(*(chips[k]['parts'] for k in ks))))}]: "
                   f"{lead['mismatched']}/{lead['launches']} launches mismatched, CCL group fault: the same count in every "
                   f"chunk ({per_chunk(c0)}, first in chunk {c0['first_chunk'] + 1}) on all {len(ks)} chips of {label}: "
                   + ", ".join(name(k) for k in ks) + "; one bad result spread by the all-gather, "
                   + (f"likely source{'s' if len(src) > 1 else ''} " + ", ".join(f"{name(k)} ({'; '.join(nonccl[k])})" for k in src)
                      if src else "source not isolated (no chip of the group failed a check outside the CCL)"))
    return out


def rw_lines(r, n):
    """An "rw" row (chunk_load.py --rw-after): G3, the C4n-size x/w/y-in-DRAM matmul readback (the same
    shape) after chunk k. One NUMERIC FAIL per mismatched chip in the ledger shape, with "DRAM" in it so tray_why.py
    names the chip's DRAM, then one summary line; a burst that raised is a NUMERIC WARN, never a pass. Returns (lines,
    trays that failed)."""
    at = f"after chunk {r['k'] + 1}/{n}"
    cs_ = r.get("chips") or {}
    order = lambda kv: (kv[1].get("tray") is None, kv[1].get("tray") or 0, str(kv[1].get("pci")))  # noqa: E731
    out, bad = [], set()
    for _, c in sorted(cs_.items(), key=order):
        if not c.get("bad_checks"):
            continue
        bad.add(c.get("tray"))
        ft = f" ({hms(c['first_t'])})" if c.get("first_t") else ""
        lanes = c.get("per_lane") or [0, 0]
        out.append(f"NUMERIC FAIL {short(c.get('pci') or '?')} (tray {c.get('tray') or '?'}) chip {c.get('dev')} [rw]: G3 "
                   f"x/w/y-in-DRAM C4n-size matmul {at}: {c['bad_checks']}/{c.get('checks', 0)} checks mismatched, "
                   f"first at {c.get('first'):g} s of load{ft}, {c.get('bad', 0)} words, DRAM banks {c.get('per_bank')}, "
                   f"lanes even/odd {lanes[0]}/{lanes[1]}, Inf {c.get('inf', 0)} NaN {c.get('nan', 0)}"
                   + bank_spread(c.get("per_bank")))
    nbad = len(out)
    if r.get("error"):
        out.append(f"NUMERIC WARN G3 x/w/y-in-DRAM readback {at} not checked"
                   + (f" past check {r.get('checks', 0)}" if r.get("checks") else "") + f": {str(r['error'])[:200]}")
    if r.get("checks") or not r.get("error"):
        out.append(f"NUMERIC G3 {at}: {hms(r['start'])}-{hms(r['end'])}, {r.get('checks', 0)} checks on {len(cs_)} "
                   f"chips, {r.get('load_s', 0):g} s of load in {r.get('wall_s', 0):g} s, {r.get('matmuls', 0)} matmuls, "
                   f"{r.get('tflops', 0):g} TFLOP/s per chip; "
                   + (f"{nbad} chip(s) mismatched" if nbad else f"no chip mismatched in {r.get('checks')} checks before the error"
                      if r.get("error") else "every chip matched" if r.get("checks") else "no check ran"))
    return out, bad


GDDR = "ABCDEFGH"


def dbi_lines(r, n):
    """A "dbi" row (chunk_load.py --dbi-wall-s): G3b, the DRAM byte/DBI pattern check right after G3.
    One NUMERIC FAIL per chip with a bad byte, with "DRAM" and its worst bank (GDDR controller A-H) in it, the byte offset
    in the 32-byte burst, the XOR of the flipped bits and the patterns; then one summary line. A check that raised is a
    NUMERIC WARN, never a pass. Returns (lines, trays that failed)."""
    at = f"after chunk {r['k'] + 1}/{n}"
    cs_ = r.get("chips") or {}
    order = lambda kv: (kv[1].get("tray") is None, kv[1].get("tray") or 0, str(kv[1].get("pci")))  # noqa: E731
    out, bad = [], set()
    for _, c in sorted(cs_.items(), key=order):
        if not c.get("bad_passes"):
            continue
        bad.add(c.get("tray"))
        pb = c.get("per_bank") or []
        wb = max(range(len(pb)), key=lambda b: pb[b]) if pb else None
        by = c.get("per_byte") or []
        offs = ",".join(str(o) for o in sorted(range(len(by)), key=lambda o: -by[o]) if by[o])[:60]
        pats = ", ".join(f"{p} {v}" for p, v in sorted((c.get("pats") or {}).items(), key=lambda kv: -kv[1])[:4])
        ft = f" ({hms(c['first_t'])})" if c.get("first_t") else ""
        out.append(f"NUMERIC FAIL {short(c.get('pci') or '?')} (tray {c.get('tray') or '?'}) chip {c.get('dev')} [dbi]: G3b "
                   f"DRAM byte/DBI pattern readback {at}: {c['bad_passes']}/{c.get('passes', 0)} passes mismatched, first in "
                   f"pass {c.get('first_pass')}{ft}, {c.get('bad', 0)} bytes, "
                   + (f"worst DRAM bank {wb} (GDDR {GDDR[wb] if wb < len(GDDR) else wb}) " if wb is not None else "")
                   + f"banks {pb}, burst byte offsets {offs or '-'}, flipped bits 0x{int(c.get('xor', 0)):02x}, "
                   f"patterns {pats or '-'}")
    nbad = len(out)
    if r.get("error"):
        out.append(f"NUMERIC WARN G3b DRAM byte/DBI pattern readback {at} not checked: {str(r['error'])[:200]}")
    else:
        cp = (f"{r.get('hops', 0)} device copies per pass" if r.get("copy") == "clone"
              else f"host write and readback only (ttnn.clone: {str(r.get('copy_error'))[:120]})")
        out.append(f"NUMERIC G3b {at}: {hms(r['start'])}-{hms(r['end'])}, {r.get('passes', 0)} passes of {r.get('patterns', 0)} "
                   f"byte/DBI patterns, {r.get('mb', 0):g} MB per chip on {len(cs_)} chips, {cp}, {r.get('wall_s', 0):g} s; "
                   + (f"{nbad} chip(s) mismatched" if nbad else "every chip matched" if r.get("passes") else "no pass ran"))
    return out, bad


L1_CAP = 8   # L1 FAIL lines per chip; the rest fold into one line
L1_BASE = 5  # N1's base patterns (l1_march.BASE); full adds their inverses


def calm(e, n=120):
    """An error text (its first n characters) inside a WARN or NUMERIC line, with the words that make a stage or run FAIL
    (FAIL, HANG, DROP, MISSING, LOST, lost: fl_vd_fin, fl_find) lowercased or reworded, so the line stays a WARN."""
    e = re.sub(r"FAIL|HANG|DROP|MISSING|LOST", lambda m: m.group(0).lower(), str(e)[:n])
    return re.sub(r"lost", "gone", e)


def l1_lines(r, n):
    """An "l1" row (chunk_load.py --l1-wall-s): the L1 march right after G3/G3b. One
    'L1 FAIL <BDF> tray N core x,y pattern P words k' per chip, core and pattern (worst first, L1_CAP per chip, the rest
    in one more line); then one NUMERIC summary line. A check that raised, or a mismatch most chips shared (the vote), is
    an L1 WARN that names no chip, never a pass. Returns (lines, trays that failed)."""
    at = f"after chunk {r['k'] + 1}/{n}"
    cs_ = r.get("chips") or {}
    order = lambda kv: (kv[1].get("tray") is None, kv[1].get("tray") or 0, str(kv[1].get("pci")))  # noqa: E731
    out, bad = [], set()
    for _, c in sorted(cs_.items(), key=order):
        if not c.get("bad_passes"):
            continue
        bad.add(c.get("tray"))
        who = f"{short(c.get('pci') or '?')} tray {c.get('tray') or '?'}"
        hits = sorted(((xy, p, w) for xy, ps in (c.get("cores") or {}).items() for p, w in ps.items()),
                      key=lambda h: (-h[2], h[0], h[1]))
        for xy, p, w in hits[:L1_CAP]:
            out.append(f"L1 FAIL {who} core {xy} pattern {p} words {w}")
        if len(hits) > L1_CAP:
            more = hits[L1_CAP:]
            out.append(f"L1 FAIL {who} core {','.join(sorted({h[0] for h in more}))[:80]} and {len(more)} more "
                       f"core/pattern entries, words {sum(h[2] for h in more)}")
        if not hits:
            out.append(f"L1 FAIL {who} core ? pattern ? words {c.get('bad', 0)}")
    nbad = len({c.get("pci") for c in cs_.values() if c.get("bad_passes")})
    for t in r.get("test_side") or []:
        out.append(f"L1 WARN N1 L1 march {at}: pattern {t.get('pattern')} pass {t.get('pass_')}: {t.get('chips')} of "
                   f"{len(cs_)} chips mismatched the same way ({t.get('words')} words): test-side, not judged")
    if r.get("error"):
        out.append(f"L1 WARN N1 L1 march {at} not checked: {calm(r['error'], 200)}")
    else:
        # a weakened N1 shows in the findings, never as a quiet pass; WARN/INFO only
        if r.get("copy") == "host":
            out.append(f"L1 WARN N1 L1 march {at}: L1-to-L1 copy unavailable: {calm(r.get('copy_error'))}; "
                       "host write and readback only")
        k5 = len({p.lstrip("~") for p in r.get("done") or []})
        if k5 < L1_BASE:
            ps = r.get("wall_s", 0) / max(1, r.get("passes", 0))
            out.append(f"L1 {'INFO' if r.get('smoke') else 'WARN'} N1 L1 march {at}: covered {k5} of {L1_BASE} patterns "
                       f"in {r.get('wall_s', 0):g} s (pass {ps:.1f} s)")
        cp = ("copied L1 to L1 to the next core" if r.get("copy") == "reshard"
              else f"host write and readback only (ttnn.to_memory_config: {calm(r.get('copy_error'))})")
        out.append(f"NUMERIC N1 L1 march {at}: {hms(r['start'])}-{hms(r['end'])}, {r.get('passes', 0)} passes, patterns "
                   f"{len(r.get('done') or [])} of {r.get('patterns', 0)} ({','.join(r.get('done') or []) or '-'}), "
                   f"{r.get('kb_per_core', 0)} KB per core on {r.get('cores', 0)} cores, {len(cs_)} chips, {cp}, "
                   f"{r.get('wall_s', 0):g} s; "
                   + (f"{nbad} chip(s) mismatched" if nbad else "every chip matched" if r.get("passes") else "no pass ran"))
    return out, bad


def lost_chunks(rows):
    """Started chunks whose chunk-end verify never came (chunk_load died, a tray dropped): [(k, started)]."""
    done = {r["k"] for r in rows if r.get("event") == "verify" and r.get("k", -1) >= 0}
    return [(r["k"], r["start"]) for r in rows if r.get("event") == "start" and r["k"] not in done]


def where(rows, epoch, bdfs=()):
    """Where <epoch> fell in the schedule (a drop or a hang): the chunk and minute, then each BDF with its tray (a
    "<bdf>@<k>" argument adds its tt-smi index k: "02:00.0 (tray 1, tt-smi 1)")."""
    n = nchunks(rows) or "?"
    last = None
    for r in rows:
        if r.get("event") in ("start", "verify", "rw", "dbi", "l1", "quiet", "preramp") and r.get("start", epoch + 1) <= epoch:
            last = r
    if last is None:
        at = "before chunk 1 (mesh open, compile, references)"
    elif last["event"] == "preramp":
        at = f"in the pre-ramp before chunk 1/{n} (not counted, started {hms(last['start'])}) at s {epoch - last['start']:.0f} of it"
    elif last["event"] == "start":
        m = (epoch - last["start"]) / 60
        at = (f"in chunk {last['k'] + 1}/{n} ({mins(last['chunk_s'])} min, started {hms(last['start'])}) "
              f"at minute {m:.1f} of it")
    elif last["event"] == "rw":
        at = f"in the G3 numeric readback (x/w/y-in-DRAM matmul) after chunk {last['k'] + 1}/{n}"
    elif last["event"] == "dbi":
        at = f"in the G3b DRAM byte/DBI pattern readback after chunk {last['k'] + 1}/{n}"
    elif last["event"] == "l1":
        at = f"in the N1 L1 march after chunk {last['k'] + 1}/{n}"
    elif last["event"] == "verify":
        at = "before chunk 1 (reference readback)" if last["k"] < 0 else f"in the readback after chunk {last['k'] + 1}/{n}"
    else:
        at = "in the lead quiet before chunk 1" if last["k"] < 0 else f"in the quiet after chunk {last['k'] + 1}/{n}"
    chips = ", ".join((f"{short(b)} (tray {tray_of(b) or '?'}{f', tt-smi {k}' if k else ''})" if b != "?" or not k
                       else f"tt-smi {k} (BDF not in map)")   # never 'tray ?' for an index missing from RK_MAP
                      for b, _, k in (str(x).partition("@") for x in bdfs))   # <bdf>@<tt-smi index>
    return at + (f": {chips}" if chips else "")


def stalled(rows):
    """A load hang's pinpoint from the per-chip readback rows: (line, [stopped BDFs]). The last readback names the chips that errored, or that launched fewer times than the rest; when
    every chip errored or none lagged, the whole box stalled together (one mesh: launch counts move in lockstep)."""
    n = nchunks(rows) or "?"
    vs = [r for r in rows if r.get("event") == "verify" and r.get("k", -1) >= 0 and isinstance(r.get("chips"), dict) and r["chips"]]
    rates = [r["util"]["launch_per_s"] for r in rows if r.get("event") == "chunk" and isinstance(r.get("util"), dict)
             and isinstance(r["util"].get("launch_per_s"), (int, float))]
    rate = f"launch rate {rates[0]:g} -> {rates[-1]:g}/s per chunk" if len(rates) > 1 else ""
    if not vs:
        return "PINPOINT: not pinpointed: no per-chip readback before the hang (it came in chunk 1 or before)", []
    r = vs[-1]   # the last readback: an earlier chip error is context, the box-wide hang after it is the stop
    chips = r["chips"]
    err = {i for i, c in chips.items() if c.get("errors") or c.get("error")}
    lau = {i: c.get("launches") for i, c in chips.items() if isinstance(c.get("launches"), int)}
    top = max(lau.values()) if lau else None
    lag = {i for i, v in lau.items() if v < top}
    txt = " ".join(str((chips[i].get("errors") or [chips[i].get("error")])[0]) for i in sorted(err))
    why = "Metal device timeout" if "device timeout" in txt else (txt.split("\n")[0][:80] if txt else "")

    def name(i):
        b = chips[i].get("pci") or "?"
        return f"{short(b)} (tray {chips[i].get('tray') or tray_of(b) or '?'})"
    at = f"chunk {r['k'] + 1}/{n}"
    stop = err if err and len(err) < len(chips) else (lag if lag and len(lag) < len(chips) else set())
    if stop:
        o = sorted(stop, key=lambda i: chips[i].get("pci") or "")
        how = f"{why} at its readback" if stop == err else f"{min(lau[i] for i in o)} launches vs {top} on the rest"
        return (f"PINPOINT: chips stopped: {', '.join(name(i) for i in o)} in {at} ({how}); the other "
                f"{len(chips) - len(o)} chips kept going", [chips[i].get("pci") for i in o if chips[i].get("pci")])
    trays = sorted({chips[i].get("tray") or tray_of(chips[i].get("pci")) for i in chips} - {None})
    tr = ", ".join(f"tray {t}" for t in trays)   # each tray by name: tray_why.py files the line under every one
    if err:
        what = f"stalled together in {at} ({why} on every chip at its readback, {top} launches each{'; ' + rate if rate else ''})"
    else:
        what = (f"stalled together after {at} (its readback clean on every chip, {top} launches each{'; ' + rate if rate else ''}); "
                "no chip stopped before the others")
    return f"PINPOINT: whole box: all {len(chips)} chips ({tr}) {what}", []


def main(argv):
    if not argv or argv[0] not in ("report", "windows", "sched", "where", "stalled") or (argv[0] != "sched" and len(argv) < 2) \
            or (argv[0] == "where" and len(argv) < 3):
        print(__doc__, file=sys.stderr); return 2
    if argv[0] == "sched":
        chunks, gap = schedule()
        print(len(chunks), f"{sum(chunks):g}", f"{gap:g}", ",".join(mins(c) for c in chunks))
        return 0
    rows = load(argv[1])
    if argv[0] == "windows":
        for w in windows(rows):
            print(*w)
    elif argv[0] == "where":
        print(where(rows, float(argv[2]), argv[3:]))
    elif argv[0] == "stalled":
        line, bdfs = stalled(rows)
        print(line)
        print(" ".join(bdfs))
    else:
        for line in report(rows, load_power(argv[2]) if len(argv) > 2 else None, load_psu(argv[2]) if len(argv) > 2 else None,
                           load_telem(argv[3]) if len(argv) > 3 else None, argv[3] if len(argv) > 4 else None,
                           argv[4] if len(argv) > 4 else None,
                           load_telem(os.path.join(os.path.dirname(argv[1]), "fastpoll.txt"))):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
