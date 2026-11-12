#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""numeric_capture.py: in-run capture of a rare numeric fault at the first wrong word of the heavy part's
checked outputs, before anything else runs, so the post-run pinpoint need not wait for the fault to come back.

fleet/chunk_load.py holds the last --capture-ring checked heavy outputs per op instead of freeing them, and every
--capture-ring launches reads each chip's on-device mismatch counter (two max reductions, one small readback). When a
chip's counter rose, capture() runs on the held outputs while they are still in DRAM:
  1. the first held output whose host read differs from the reference; two more reads of it:
       written-wrong  both re-reads show the same wrong words: they are stored wrong (the compute or its write)
       read-path      both re-reads are right (or no held output reads back wrong twice, though the device check
                      counted wrong words): the stored words are right, a read of them was not
       transient      the reads disagree otherwise: the words change in DRAM or on the way out
       not-held       no held output was wrong and the counter rose before the ring (--heavy-check-each's middle outputs)
  2. DRAM readback: the chip's x and w (and its reference) read back and compared with a control chip's copy (the
     tensors are replicated, so equal bit for bit; a host copy would need the device's block-float rounding)
  3. recompute: the op once more on the whole mesh; the wrong chip and the control chip compared with their references
  4. the wrong-word map: tile rows and columns and owning cores (chunk_sched.core_of); a DRAM bank only where the
     output's tile columns are decoupled from the banks (pinpoint_verdict.bank_comp)
The event goes to the chunk's rows (event "capture"), chunk_sched.report() prints it as a NUMERIC-CAPTURE line and
pinpoint_verdict.py merges it into the run's pinpoint.json ("captures"). Pure logic over an adapter (see capture());
the unit tests drive it with fake chips that inject faults.
"""
import re

BANKS = 8   # Blackhole DRAM channels: interleaved pages go round robin over them
CAP_RE = re.compile(r"^NUMERIC-CAPTURE (?:[0-9a-fA-F]{4}:)?(\S+) tray (\S+): (written-wrong|read-path|transient|not-held)\b.*?"
                    r"recompute (ok|fail)")
MAP_RE = re.compile(r", map (held|counter)\b")


class Ring:
    """The last n checked outputs of each heavy op j, oldest first; hold() returns what falls out (to free)."""

    def __init__(self, n):
        self.n, self.held = max(1, int(n)), {}

    def hold(self, j, launch, t):
        lst = self.held.setdefault(j, [])
        lst.append((launch, t))
        out = [x for _, x in lst[:-self.n]]
        del lst[:-self.n]
        return out

    def due(self, launch):
        return launch % self.n == 0

    def drop(self):
        out = [t for lst in self.held.values() for _, t in lst]
        self.held = {}
        return out


def risen(prev, now):
    """Chips whose counter (per chip max, any float) went up since the last poll: [(chip, before, after)]."""
    return [(i, prev.get(i, 0.0), v) for i, v in sorted(now.items()) if v != v or v > prev.get(i, 0.0)]


def wrong(torch, got, want):
    """Element positions where got differs from want (NaN != NaN counts, as ttnn.ne does)."""
    return torch.nonzero(got != want)


def same(torch, a, b):
    return a.shape == b.shape and bool(torch.equal(torch.nan_to_num(a, 1e30), torch.nan_to_num(b, 1e30)))


def classify(reads, torch):
    """reads: the wrong positions ([r, c] tensors) of the first wrong held output, read 1 then the two re-reads. Both
    re-reads show read 1's words: written-wrong; both right: read-path (read 1 came back wrong); else transient."""
    keys = [frozenset(map(tuple, r.tolist())) for r in reads]
    if all(k == keys[0] for k in keys[1:]):
        return "written-wrong"
    return "read-path" if not any(keys[1:]) else "transient"


def decoupled(W, banks=BANKS):
    """Interleaved output pages go round robin over the banks: with a tile-column count that is a multiple of the
    bank count, every tile column sits in one bank (columns and banks coupled; a bank count says nothing)."""
    return (-(-int(W) // 32)) % banks != 0


def word_map(cs, pv, rows, cols, H, W, grid, phys=None):
    """loc_summary of the wrong words (+ translated cores) and, only when decoupled, bank_comp over their pages."""
    loc = cs.loc_summary(rows, cols, H, W, grid)
    if loc and phys is not None:
        loc = cs.add_phys(loc, phys)
    if loc and pv is not None and decoupled(W):
        nt, per = -(-W // 32), {}
        for r, c in zip(rows, cols):
            b = ((r // 32) * nt + c // 32) % BANKS
            per[b] = per.get(b, 0) + 1
        comp, detail = pv.bank_comp(per, decoupled=True)
        loc["bank"] = dict(per_bank=per, comp=comp, detail=detail)
    return loc


def capture(ad, torch, cs, j, i, ctl, held, pv=None):
    """One event: op j's counter rose on chip i. ad (adapter): read(t, i) -> chip i's copy as a float torch tensor,
    ref(j), inputs(j) -> (x, w), rerun(j) -> a new output (freed with free(t)), counter(j) -> the 2-D counter, grid,
    phys(x, y) or None. held: [(launch, tensor)] oldest first. ctl: a chip whose counter did not rise. Returns the
    event dict (no line); never raises."""
    ev = dict(op=j, chip=i, control=ctl)
    try:
        want = ad.ref(j)
        wi, wc = ad.read(want, i), ad.read(want, ctl)
        H, W = int(wi.shape[-2]), int(wi.shape[-1])
        wi2, wc2 = wi.reshape(-1, W), wc.reshape(-1, W)
        first = None
        for launch, t in held:
            g = ad.read(t, i).reshape(-1, W)
            pos = wrong(torch, g, wi2)
            if len(pos):
                first = (launch, t, pos)
                break
        if first is None:
            again = [wrong(torch, ad.read(t, i).reshape(-1, W), wi2) for _, t in held]
            ev["class"] = "transient" if any(len(p) for p in again) else ("read-path" if held else "not-held")
            c = ad.read(ad.counter(j), i).reshape(-1, W)
            pos = torch.nonzero(c)
            ev["map_from"] = "device counter"
        else:
            launch, t, pos = first
            reads = [pos] + [wrong(torch, ad.read(t, i).reshape(-1, W), wi2) for _ in range(2)]
            ev.update({"class": classify(reads, torch), "launch": launch, "reads_wrong": [len(r) for r in reads],
                       "map_from": "held output"})
        ev["wrong"] = len(pos)
        ev["loc"] = word_map(cs, pv, pos[:, 0].tolist(), pos[:, 1].tolist(), H, W, ad.grid, ad.phys) if len(pos) else None
        x, w = ad.inputs(j)
        ev["dram"] = dict(x=same(torch, ad.read(x, i), ad.read(x, ctl)), w=same(torch, ad.read(w, i), ad.read(w, ctl)),
                          ref=same(torch, wi, wc))
        o = ad.rerun(j)
        try:
            ri, rc = ad.read(o, i).reshape(-1, W), ad.read(o, ctl).reshape(-1, W)
            ev["recompute"] = dict(chip=len(wrong(torch, ri, wi2)), control=len(wrong(torch, rc, wc2)))
        finally:
            ad.free(o)
    except Exception as e:  # noqa: BLE001
        ev["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return ev


def line(ev, bdf, tray, op_name="heavy"):
    """The NUMERIC-CAPTURE line of one capture event."""
    loc = ev.get("loc") or {}
    cores = " ".join("(%d,%d)x%d" % (x, y, n) for x, y, n in (loc.get("cores") or [])[:4]) or "none mapped"
    d, rc = ev.get("dram") or {}, ev.get("recompute") or {}
    bad_d = [k for k in ("x", "w", "ref") if d.get(k) is False]
    dram = ("fail (" + ",".join(bad_d) + " differ from the control chip's)") if bad_d else ("ok" if d else "not read")
    rec = ("fail" if rc.get("chip") else "ok") if rc else "not run"
    s = (f"NUMERIC-CAPTURE {bdf} tray {tray}: {ev.get('class', 'error')}, {op_name} op {ev.get('op', 0)}, "
         f"{ev.get('wrong', '?')} wrong words")
    if loc.get("first"):
        f = loc["first"]
        s += f", first ({f['row']},{f['col']}) tile ({f['tile'][0]},{f['tile'][1]})"
    s += f", cores {cores}"
    if (loc.get("bank") or {}).get("comp"):
        s += f", {loc['bank']['comp']}"
    s += f", dram readback {dram}, recompute {rec}"
    if rc:
        s += f" ({rc.get('chip')} wrong; control chip {ev.get('control')} {'ok' if not rc.get('control') else 'fail'})"
    if ev.get("map_from"):
        s += ", map " + ("held" if ev["map_from"] == "held output" else "counter")
    if ev.get("launch") is not None:
        s += f", launch {ev['launch']}"
    if ev.get("error"):
        s += f", capture error: {ev['error']}"
    return s


def parse(lines):
    """NUMERIC-CAPTURE lines -> {bdf: [dict(cls, recompute, map, line)]}, bdf without its domain (pinpoint_verdict's);
    map: held (a held output read back wrong), counter (only the device counter mapped it) or None (not in the line)."""
    out = {}
    for ln in lines:
        m = CAP_RE.match(ln.strip())
        if m:
            mp = MAP_RE.search(ln)
            out.setdefault(m.group(1).lower(), []).append(dict(cls=m.group(3), recompute=m.group(4),
                                                               map=mp.group(1) if mp else None, line=ln.strip()))
    return out


def attributed(c):
    """A capture that attributed the fault at the first wrong word: a held output read back wrong (written-wrong;
    transient and read-path only when their map came from the held output). not-held and a counter-mapped transient or
    read-path did not."""
    return c["cls"] == "written-wrong" or (c["cls"] in ("transient", "read-path") and c.get("map") == "held")


def counter_transient(c):
    """A transient whose map came from the device counter: the held reads disagreed, but no first wrong word was held."""
    return c["cls"] == "transient" and c.get("map") == "counter"


def retrigger(fail_bdfs, caps):
    """Step 3: the post-run pinpoint after a capture. 'skip' when every failing chip has an attributed capture (see
    attributed()) and none recomputed wrong (the capture is the attribution; a fault that does not come back would only
    spend the pinpoint's budget), 'norepeat' when every failing chip has one (or only counter-mapped transients) and
    some reproduce (the splits still run, the repeat split is the capture's recompute), 'transient' when the failing
    chips not attributed have only counter-mapped transients and none recomputed wrong (the dram and rw splits: where
    words change after the write), '' otherwise (the pinpoint as before): a chip with other
    unattributed captures keeps the full pinpoint, its only attribution path."""
    fail_bdfs = {b.lower() for b in fail_bdfs}
    caps = {b: caps.get(b) or [] for b in fail_bdfs}
    att = {b: [c for c in caps[b] if attributed(c)] for b in fail_bdfs}
    rest = [b for b in fail_bdfs if not att[b]]
    if not fail_bdfs or not all(caps[b] and all(counter_transient(c) for c in caps[b]) for b in rest):
        return ""
    if any(c["recompute"] == "fail" for b in fail_bdfs for c in att[b] or caps[b]):
        return "norepeat"
    return "transient" if rest else "skip"
