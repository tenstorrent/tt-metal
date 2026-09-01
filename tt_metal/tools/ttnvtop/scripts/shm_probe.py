#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Samples every /dev/shm/tt_device_*_util at a fixed rate and answers two questions the
# TUI can only gesture at:
#
#   1. CONSISTENCY -- per chip: worst staleness, and how many times the publish stalled
#      past a threshold. "It looks hung" becomes a number, per chip, so a chip that really
#      does stall (chip 7 was reported stalling) is separated from one that never did.
#   2. ACCURACY -- mean fpu/sfpu/dispatch/DRAM within each calibration phase, so the
#      reading can be regressed against a known duty cycle. The keys are named for the
#      PerCoreView field they read, not F/S/D, because the letters were what let a
#      mislabelling survive.
#
# ---------------------------------------------------------------------------------------
# Why the layout guard below is as heavy as it is
#
# A size assert is NOT enough. The original version of this script used
# V = struct.Struct("<IIHHHHIIIIII"), which is ALSO exactly 40 bytes, so `assert V.size
# == 40` passed while every field from byte 4 on was read at the wrong offset -- the two
# leading "I"s swallowed the six uint8s *and* sfpu_busy_p1000, so the field labelled "F"
# (FPU) was really dispatch_busy_p1000, "S" was compute_busy_p1000, "D" was
# unpack_busy_p1000 (which the collector never writes, hence a constant 0), and
# sfpu_busy_p1000 was never sampled at all. Every calibration slope recorded with that
# version describes a different field than its label. See PLAN_ETH_AGGREGATOR.md 5w.
#
# There are TWO independent things that can be wrong, and they need two different checks:
#
#   (a) STRIDE.  The format string maps tuple slots to the wrong byte offsets.
#       Caught by round-tripping a synthesised record, one marker per field.
#   (b) LABEL.   The offsets are right but the human-readable name attached to a slot is
#       wrong -- which is the bug that actually happened, restated. A hand-maintained
#       V_FIELDS table checked only against itself cannot catch this: swap the names at
#       offsets 6 and 8 and a self-consistent guard still passes, handing out
#       VI['sfpu_busy_p1000'] == 7.
#       Caught by PARSING common/shm_schema.hpp at import time and requiring V_FIELDS to
#       equal the C declaration name-for-name, offset-for-offset. The C header is the
#       thing the collector actually compiles against, so it is the only authority that
#       is not a copy of the thing under test.
#
# The guard is raise-based, not assert-based, on purpose: `python3 -O` / PYTHONOPTIMIZE=1
# deletes assert statements, and a guard that vanishes under an environment variable is
# not a guard.
# ---------------------------------------------------------------------------------------

import argparse
import glob
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict, namedtuple

# UtilShmHeader, common/shm_schema.hpp:42 -- 72 bytes, no interior padding.
H = struct.Struct("<4sHHQIIQQ8I")
# PerCoreView, common/shm_schema.hpp:72 -- 6x u8, 10x u16, 2 bytes tail padding
# (u32 alignment), 3x u32. Identical to tests/test_ttnvtop_accuracy.py:27.
V = struct.Struct("<6B10H2x3I")

# Header tuple indices (named so a future field insertion breaks loudly, not silently).
# _check_layout() proves each of these against the C declaration -- ttnvtop_py.py:74 read
# hdr[6] for last_update_us when 6 is epoch_us, which is this same bug in a sibling file.
H_MAGIC, H_VERSION, H_STRUCT_SIZE, H_ASIC_ID = 0, 1, 2, 3
H_ARCH_ID, H_SIGNAL_SRC, H_EPOCH_US, H_LAST_UPDATE_US = 4, 5, 6, 7
H_NUM_CORES, H_PROGRAM_ID, H_PID, H_AICLK_MHZ = 8, 9, 10, 11
H_DRAM_RD_MBPS, H_DRAM_WR_MBPS, H_DRAM_PEAK_MBPS = 12, 13, 14

H_EXPECTED = {
    "magic": H_MAGIC,
    "version": H_VERSION,
    "struct_size": H_STRUCT_SIZE,
    "asic_id": H_ASIC_ID,
    "arch_id": H_ARCH_ID,
    "signal_sources": H_SIGNAL_SRC,
    "epoch_us": H_EPOCH_US,
    "last_update_us": H_LAST_UPDATE_US,
    "num_cores": H_NUM_CORES,
    "host_assigned_id": H_PROGRAM_ID,
    "collector_pid": H_PID,
    "aiclk_mhz": H_AICLK_MHZ,
    "dram_rd_mbps": H_DRAM_RD_MBPS,
    "dram_wr_mbps": H_DRAM_WR_MBPS,
    "dram_peak_mbps": H_DRAM_PEAK_MBPS,
}

# (tuple index, C field name, byte offset within PerCoreView, width in bytes).
# This table is NOT the authority -- common/shm_schema.hpp is. It is an assertion about
# the header, checked field-for-field at import time.
V_FIELDS = (
    (0, "noc_x", 0, 1),
    (1, "noc_y", 1, 1),
    (2, "logical_x", 2, 1),
    (3, "logical_y", 3, 1),
    (4, "is_remote", 4, 1),
    (5, "dispatched", 5, 1),
    (6, "sfpu_busy_p1000", 6, 2),
    (7, "dispatch_busy_p1000", 8, 2),
    (8, "compute_busy_p1000", 10, 2),  # FPU / MATH pipe
    (9, "unpack_busy_p1000", 12, 2),  # declared but never written by the collector
    (10, "pack_busy_p1000", 14, 2),  # declared but never written by the collector
    (11, "stall_p1000", 16, 2),  # declared but never written by the collector
    (12, "noc0_in_mbps", 18, 2),
    (13, "noc0_out_mbps", 20, 2),
    (14, "noc1_in_mbps", 22, 2),
    (15, "noc1_out_mbps", 24, 2),
    (16, "samples_seen", 28, 4),
    (17, "last_kernel_id", 32, 4),
    (18, "reserved_1", 36, 4),
)


class LayoutError(RuntimeError):
    """The Python view of the SHM schema does not match common/shm_schema.hpp."""


def _require(cond, msg):
    # Deliberately not `assert`: python3 -O strips asserts and would delete the guard.
    if not cond:
        raise LayoutError(msg)


# ---------------------------------------------------------------------------------------
# Minimal C struct parser for common/shm_schema.hpp.
#
# Scope is deliberately tiny: fixed-width scalar members and fixed-size arrays of them,
# natural alignment, no bitfields, no nesting. Anything it cannot parse is a hard error,
# never a skip -- a silently skipped member would shift every offset after it.
# ---------------------------------------------------------------------------------------

CField = namedtuple("CField", "slot name off elem_size count width code")

_CTYPES = {
    "uint8_t": (1, "B"),
    "int8_t": (1, "b"),
    "uint16_t": (2, "H"),
    "int16_t": (2, "h"),
    "uint32_t": (4, "I"),
    "int32_t": (4, "i"),
    "uint64_t": (8, "Q"),
    "int64_t": (8, "q"),
    "char": (1, "s"),
}
_MEMBER_RE = re.compile(r"^\s*(u?int(?:8|16|32|64)_t|char)\s+([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?\s*;\s*$")


def _find_schema_header():
    """Locate common/shm_schema.hpp. Absence is fatal, by design.

    Without the C declaration this script can check strides but not names, and the bug
    that produced the retracted 5w numbers was a name bug. Set TTNVTOP_SHM_SCHEMA if the
    script has been copied out of the tree.
    """
    cands = []
    env = os.environ.get("TTNVTOP_SHM_SCHEMA")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, os.pardir, "common", "shm_schema.hpp"))
    cands.append(os.path.join(here, "shm_schema.hpp"))
    for c in cands:
        if os.path.exists(c):
            return os.path.normpath(c)
    raise LayoutError(
        "cannot find common/shm_schema.hpp (tried: "
        + ", ".join(os.path.normpath(c) for c in cands)
        + "). The field-name guard needs the C declaration; set TTNVTOP_SHM_SCHEMA."
    )


def _parse_c_struct(text, struct_name, path="<schema>"):
    """Return (fields, sizeof) for `struct <struct_name>` under the SysV x86-64 rules."""
    m = re.search(r"\bstruct\s+" + re.escape(struct_name) + r"\s*\{(.*?)\n\};", text, re.S)
    _require(m is not None, f"{path}: no `struct {struct_name} {{ ... }};` found")
    fields = []
    off = 0
    slot = 0
    maxalign = 1
    for lineno, raw in enumerate(m.group(1).splitlines(), 1):
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        mm = _MEMBER_RE.match(line)
        _require(
            mm is not None,
            f"{path}: struct {struct_name} line {lineno} is not a plain scalar/array member "
            f"and this parser refuses to guess its size: {raw.strip()!r}",
        )
        ctype, name, count = mm.group(1), mm.group(2), mm.group(3)
        esz, code = _CTYPES[ctype]
        n = int(count) if count else 1
        maxalign = max(maxalign, esz)
        off += (-off) % esz  # natural alignment
        nslots = 1 if code == "s" else n  # char[N] unpacks as one bytes object
        fields.append(CField(slot, name, off, esz, n, esz * n, code))
        off += esz * n
        slot += nslots
    size = off + (-off) % maxalign
    return tuple(fields), size


def _slot_count(fields):
    total = 0
    for f in fields:
        total += 1 if f.code == "s" else f.count  # char[N] unpacks as ONE bytes object
    return total


def _check_struct_strides(s, fields, csize, label):
    """Prove `s`'s format string maps every C field to its declared offset and width.

    Writes a unique marker at one field's byte range in an otherwise zeroed record and
    requires the unpack to surface it at the expected tuple slot and nowhere else. A
    same-size but wrongly strided format string cannot survive this.
    """
    _require(s.size == csize, f"{label}: struct.Struct size {s.size} != sizeof from header {csize}")
    zero = s.unpack(bytes(csize))
    nslots = _slot_count(fields)
    _require(
        len(zero) == nslots,
        f"{label}: format string yields {len(zero)} tuple slots, header declares {nslots} fields",
    )

    covered = set()
    for f in fields:
        if f.code == "s":  # char[N] -> single bytes slot
            marker = bytes(range(0xA0, 0xA0 + f.count))
            buf = bytearray(csize)
            buf[f.off : f.off + f.width] = marker
            got = s.unpack(bytes(buf))
            _require(got[f.slot] == marker, f"{label}.{f.name}: slot {f.slot} = {got[f.slot]!r}, want {marker!r}")
            _require(
                all(v == zero[j] for j, v in enumerate(got) if j != f.slot),
                f"{label}.{f.name} at byte {f.off} bled into another slot: {got!r}",
            )
            covered.update(range(f.off, f.off + f.width))
            continue
        for e in range(f.count):
            slot = f.slot + e
            eoff = f.off + e * f.elem_size
            marker = {1: 0xA5, 2: 0xBEEF, 4: 0xDEADBEEF, 8: 0xDEADBEEFCAFEF00D}[f.elem_size]
            buf = bytearray(csize)
            buf[eoff : eoff + f.elem_size] = marker.to_bytes(f.elem_size, "little")
            got = s.unpack(bytes(buf))
            _require(
                got[slot] == marker,
                f"{label}.{f.name}[{e}]: expected slot {slot} == {marker:#x} from byte {eoff}, got {got[slot]:#x}",
            )
            for j, val in enumerate(got):
                if j != slot:
                    _require(
                        val == zero[j],
                        f"{label}.{f.name} at byte {eoff} bled into slot {j} = {val!r}",
                    )
            covered.update(range(eoff, eoff + f.elem_size))

    # Every byte NOT claimed by a declared field must be inert padding. Derived from the
    # header, so no hardcoded pad offsets to go stale.
    for pad_off in sorted(set(range(csize)) - covered):
        buf = bytearray(csize)
        buf[pad_off] = 0xFF
        _require(
            s.unpack(bytes(buf)) == zero,
            f"{label}: byte {pad_off} should be padding but reaches a tuple slot",
        )


def _check_layout():
    """Bind the Python structs to common/shm_schema.hpp, by stride AND by name.

    What this actually guarantees, in order:

      1. `H`/`V` are the size the C compiler gives UtilShmHeader/PerCoreView, as computed
         from the parsed declaration (not a hardcoded 72/40).
      2. Every C field is reachable at its declared byte offset through the declared
         tuple slot, and cannot be reached through any other slot; every remaining byte
         is inert padding.  (STRIDE)
      3. V_FIELDS -- the (slot, name, offset, width) table the rest of this script indexes
         through -- is equal, entry for entry, to the parsed C declaration. A V_FIELDS
         with two names transposed is rejected here even though it is perfectly
         self-consistent and yields the right total size.  (LABEL)
      4. The positional H_* constants name the header field they claim to name.

    What it does NOT guarantee: that the running collector was compiled from THIS copy of
    shm_schema.hpp. That is what the magic / version / struct_size check in sample()
    is for -- see SHM_VERSION.
    """
    path = _find_schema_header()
    with open(path, "r") as f:
        text = f.read()

    vfields, vsize = _parse_c_struct(text, "PerCoreView", path)
    hfields, hsize = _parse_c_struct(text, "UtilShmHeader", path)

    # The header's own static_asserts are a cross-check on our parser.
    for name, size in (("PerCoreView", vsize), ("UtilShmHeader", hsize)):
        m = re.search(r"static_assert\(sizeof\(" + name + r"\) == (\d+)", text)
        _require(m is not None, f"{path}: no static_assert on sizeof({name})")
        _require(
            int(m.group(1)) == size,
            f"{path}: static_assert says sizeof({name})=={m.group(1)} but this parser computed {size}",
        )

    _check_struct_strides(V, vfields, vsize, "PerCoreView")
    _check_struct_strides(H, hfields, hsize, "UtilShmHeader")

    # ---- LABEL check: V_FIELDS must BE the C declaration, name included.
    derived = tuple((f.slot, f.name, f.off, f.width) for f in vfields)
    if derived != V_FIELDS:
        want = {n: (i, o, w) for i, n, o, w in derived}
        have = {n: (i, o, w) for i, n, o, w in V_FIELDS}
        detail = []
        for n in sorted(set(want) | set(have)):
            if want.get(n) != have.get(n):
                detail.append(f"    {n}: shm_schema.hpp says {want.get(n)}, V_FIELDS says {have.get(n)}")
        raise LayoutError(
            "V_FIELDS disagrees with " + path + " ((slot, offset, width) per name):\n" + "\n".join(detail)
        )

    # ---- LABEL check for the header's positional constants (the ttnvtop_py.py:74 bug).
    hslot = {f.name: f.slot for f in hfields}
    for name, idx in H_EXPECTED.items():
        _require(name in hslot, f"{path}: UtilShmHeader has no field {name!r}")
        _require(
            hslot[name] == idx,
            f"header constant for {name!r} is tuple index {idx}, but {path} puts it at {hslot[name]}",
        )

    m = re.search(r"constexpr\s+uint16_t\s+kShmVersion\s*=\s*(\d+)\s*;", text)
    _require(m is not None, f"{path}: cannot find kShmVersion")
    return int(m.group(1))


SHM_VERSION = _check_layout()

# Safe to build the name->slot map only now: _check_layout() has proven the names.
VI = {name: idx for idx, name, _off, _w in V_FIELDS}
V_SFPU = VI["sfpu_busy_p1000"]
V_DISPATCH = VI["dispatch_busy_p1000"]
V_FPU = VI["compute_busy_p1000"]

_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg, file=sys.stderr)


def sample():
    out = {}
    now = time.monotonic()
    for f in glob.glob("/dev/shm/tt_device_*_util"):
        try:
            b = open(f, "rb").read()
        except OSError:
            continue
        if len(b) < H.size:
            continue
        h = H.unpack_from(b, 0)
        if h[H_MAGIC] != b"TTUT":
            continue
        # A struct_size mismatch means the writer's PerCoreView is not the one this
        # script knows. Skip rather than mis-stride -- silence is what caused the bug.
        if h[H_STRUCT_SIZE] != V.size:
            _warn_once(
                ("size", f),
                f"SKIP {f}: struct_size={h[H_STRUCT_SIZE]} != {V.size} (schema v{h[H_VERSION]}); rebuild the probe",
            )
            continue
        # struct_size alone is not sufficient. kShmVersion exists precisely to flag a
        # SAME-SIZE semantic change: shm_schema.hpp:24 records that v2 repurposed
        # PerCoreView::reserved_0 as sfpu_busy_p1000 -- a rename, 40 bytes before and
        # after. A future v4 that transposes two u16s would also keep struct_size == 40
        # and would otherwise be read here with the wrong field mapping, which is the
        # exact class of defect that invalidated PLAN_ETH_AGGREGATOR.md 5w.
        if h[H_VERSION] != SHM_VERSION:
            _warn_once(
                ("ver", f),
                f"SKIP {f}: schema v{h[H_VERSION]} but this probe was checked against "
                f"v{SHM_VERSION} of common/shm_schema.hpp. Same struct_size does NOT mean "
                f"same field mapping. Rebuild the collector, or re-check the probe against "
                f"the writer's schema.",
            )
            continue
        nc = h[H_NUM_CORES]
        if nc == 0:
            continue
        if len(b) < H.size + nc * V.size:
            continue
        fpu_sum = sfpu_sum = disp_sum = 0
        fpu_nz = 0
        for i in range(nc):
            v = V.unpack_from(b, H.size + i * V.size)
            fpu_sum += v[V_FPU]
            sfpu_sum += v[V_SFPU]
            disp_sum += v[V_DISPATCH]
            if v[V_FPU]:
                fpu_nz += 1
        out[os.path.basename(f)] = {
            "t": now,
            "age": now - h[H_LAST_UPDATE_US] / 1e6,
            "aiclk": h[H_AICLK_MHZ],
            # p1000 (per-mille) -> percent. Keys are named for the C field they read.
            "fpu": fpu_sum / nc / 10.0,  # compute_busy_p1000
            "sfpu": sfpu_sum / nc / 10.0,  # sfpu_busy_p1000
            "dispatch": disp_sum / nc / 10.0,  # dispatch_busy_p1000
            "fpu_nz": fpu_nz,
            "dram_rd_mbps": h[H_DRAM_RD_MBPS],
            "dram_wr_mbps": h[H_DRAM_WR_MBPS],
            "dram_peak_mbps": h[H_DRAM_PEAK_MBPS],
            "cores": nc,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--stall-ms", type=float, default=1000.0, help="age above this counts as a stall")
    ap.add_argument("--phases", type=str, default="", help="calib_duty.py JSON to align against")
    ap.add_argument("--raw", type=str, default="", help="write every sample here as JSONL")
    args = ap.parse_args()

    period = 1.0 / args.hz
    t0 = time.monotonic()
    end = t0 + args.seconds
    rows = []
    raw = open(args.raw, "w") if args.raw else None
    # Deadline scheduling, so the achieved rate is 1/period rather than 1/(period+work).
    # `late` counts sweeps where the work alone overran the period; the loop resyncs to
    # the clock instead of trying to catch up, which would burst the samples.
    sweeps = 0
    late = 0
    next_t = t0
    while time.monotonic() < end:
        s = sample()
        sweeps += 1
        for name, r in s.items():
            r["chip"] = name
            rows.append(r)
            if raw:
                raw.write(json.dumps(r) + "\n")
        next_t += period
        slack = next_t - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        else:
            late += 1
            next_t = time.monotonic()
    elapsed = time.monotonic() - t0
    if raw:
        raw.close()

    if not rows:
        if _warned:
            print("NO USABLE SHM FILES -- every candidate was rejected; see the SKIP lines above.")
        else:
            print("NO SHM FILES -- is the collector running?")
        return 1

    # ---- consistency, per chip
    per = defaultdict(list)
    for r in rows:
        per[r["chip"]].append(r)
    # Report the MEASURED sweep rate, not the requested one. They differ whenever a sweep
    # costs a non-negligible fraction of the period, and quoting the request as if it were
    # a measurement is the same kind of unearned claim as a mislabelled column.
    measured = sweeps / elapsed if elapsed > 0 else 0.0
    print(
        f"\n=== CONSISTENCY  ({len(rows)} samples in {sweeps} sweeps over {elapsed:.1f}s, "
        f"{measured:.2f} Hz measured / {args.hz:g} Hz requested"
        + (f", {late} sweeps overran the period" if late else "")
        + f", stall > {args.stall_ms:.0f} ms) ==="
    )
    print(f"{'chip':22s} {'n':>5s} {'max age':>9s} {'mean age':>9s} {'stalls':>7s} {'worst gap':>10s} {'aiclk=0':>8s}")
    for name in sorted(per):
        rs = per[name]
        ages = [r["age"] for r in rs]
        stalls = sum(1 for a in ages if a > args.stall_ms / 1000.0)
        # worst gap between successive observations of a CHANGED last_update
        gap = 0.0
        prev_t = None
        for r in rs:
            if prev_t is not None:
                gap = max(gap, r["t"] - prev_t)
            prev_t = r["t"]
        zc = sum(1 for r in rs if r["aiclk"] == 0)
        print(
            f"{name:22s} {len(rs):5d} {max(ages):8.2f}s {sum(ages)/len(ages):8.2f}s " f"{stalls:7d} {gap:9.2f}s {zc:7d}"
        )

    # ---- accuracy, aligned to the calibration phases
    if args.phases and os.path.exists(args.phases):
        ph = json.load(open(args.phases))
        print(f"\n=== ACCURACY vs known duty cycle (matmul size {ph.get('size')}) ===")
        print(f"{'target':>7s} {'host busy':>10s} | per-chip mean FPU% (compute_busy_p1000)")
        fits = defaultdict(list)
        for p in ph["phases"]:
            # skip the first 25% of each phase: the EWMA and the aggregator sweep both
            # need time to reach the new level, and including the ramp biases the fit.
            lo = p["t_start"] + 0.25 * (p["t_end"] - p["t_start"])
            line = f"{p['duty_target']:6.0f}% {p['duty_actual_host']:9.1f}% |"
            for name in sorted(per):
                sel = [r["fpu"] for r in per[name] if lo <= r["t"] <= p["t_end"]]
                if sel:
                    m = sum(sel) / len(sel)
                    line += f" {name.split('_')[2]}:{m:5.2f}"
                    fits[name].append((p["duty_actual_host"], m))
            print(line)

        print(f"\n=== LINEARITY  (monitor% = slope x host_busy% + intercept) ===")
        print(f"{'chip':22s} {'slope':>8s} {'intercept':>10s} {'R^2':>7s}   verdict")
        for name in sorted(fits):
            pts = fits[name]
            if len(pts) < 3:
                continue
            n = len(pts)
            sx = sum(x for x, _ in pts)
            sy = sum(y for _, y in pts)
            sxx = sum(x * x for x, _ in pts)
            sxy = sum(x * y for x, y in pts)
            syy = sum(y * y for _, y in pts)
            den = n * sxx - sx * sx
            if den == 0:
                continue
            slope = (n * sxy - sx * sy) / den
            icept = (sy - slope * sx) / n
            dr = n * syy - sy * sy
            r2 = ((n * sxy - sx * sy) ** 2 / (den * dr)) if dr > 0 else 0.0
            if r2 > 0.9 and slope > 0.02:
                verdict = "TRACKS duty"
            elif slope <= 0.02:
                verdict = "FLAT -- reading does not respond to compute"
            else:
                verdict = "responds but NOT linear"
            print(f"{name:22s} {slope:8.3f} {icept:10.2f} {r2:7.3f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
