# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Time verdict of a candidate bench against a back-to-back baseline re-run (CONTRACT §4, D3).

    python models/experimental/kairos_forecast/tests/perf/compare_bench.py <cand bench.json> <baseline bench.json>
        [--target paper:accuracy] [--pcc-key pcc_normalized] [--pcc-floor 0.995] [--min-gain 2] [--guard-max 3]
        [--allow-cross-chip]

Records match on (name, precision, layers). Each record's device time is ``device_ms`` (the median of its
``split_runs``), and its noise is the spread (max - min) of the split runs' ``device_s``. Verdicts:
- target: gain % = (base - cand) / base. ACCEPT needs gain >= --min-gain **and** a saving larger than the base's own
  split-run spread (a gain inside the spread is not a gain); otherwise REJECT.
- guards: every other record present in both; a guard slower by more than --guard-max % rejects.
- pcc: the candidate's --pcc-key on the target must be >= --pcc-floor where the record has it (reduced-layer screens
  have no accuracy keys: printed n/a). Accuracy itself is check_gates.py's job, against the fixed record.
- chip: candidate and baseline records must come from the same chip unless --allow-cross-chip (never on chip 1).
Exit 0 on ACCEPT, 1 on REJECT, 2 on a usage error."""

from __future__ import annotations

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def key(r):
    return r["name"], r["precision"], tuple(r.get("layers") or ())


def spread_ms(r):
    runs = [s["device_s"] for s in r.get("split_runs") or [] if "device_s" in s]
    return 1000 * (max(runs) - min(runs)) if len(runs) > 1 else 0.0


def compare(cand, base, target, pcc_key, pcc_floor, min_gain, guard_max, allow_cross_chip):
    """-> (accepted, lines)."""
    lines, ok = [], True
    cb, bb = {key(r): r for r in cand}, {key(r): r for r in base}
    name, prec = target.split(":")
    tk = [k for k in cb if k[:2] == (name, prec) and k in bb]
    if not tk:
        return False, [f"REJECT no {target} record in both files"]
    for k in sorted(set(cb) & set(bb)):
        c, b = cb[k], bb[k]
        if c.get("chip") != b.get("chip") and not allow_cross_chip:
            lines.append(f"REJECT {k[0]}/{k[1]} chip {c.get('chip')} vs baseline {b.get('chip')}")
            ok = False
            continue
        if "device_ms" not in c or "device_ms" not in b:
            lines.append(f"info {k[0]}/{k[1]}: no device_ms (not timed)")
            continue
        cm, bm = c["device_ms"], b["device_ms"]
        gain = 100 * (bm - cm) / bm
        sp = spread_ms(b)
        layers = "+".join(map(str, k[2]))
        if k in tk:
            good = gain >= min_gain and (bm - cm) > sp
            why = "" if good else (f" (< {min_gain} %)" if gain < min_gain else f" (inside base spread {sp:.2f} ms)")
            lines.append(
                f"{'ok  ' if good else 'FAIL'} target {k[0]}/{k[1]} L{layers} {cm:.2f} vs {bm:.2f} ms: "
                f"{gain:+.2f} % (base spread {sp:.2f} ms){why}"
            )
            ok &= good
            if pcc_key in c:
                pg = c[pcc_key] >= pcc_floor
                lines.append(f"{'ok  ' if pg else 'FAIL'} target {pcc_key} {c[pcc_key]:.6f} >= {pcc_floor}")
                ok &= pg
            else:
                lines.append(f"n/a  target {pcc_key}: not in the record (reduced-layer screen)")
        else:
            good = -gain <= guard_max
            lines.append(
                f"{'ok  ' if good else 'FAIL'} guard {k[0]}/{k[1]} L{layers} {cm:.2f} vs {bm:.2f} ms: {gain:+.2f} % "
                f"(slower by at most {guard_max} %)"
            )
            ok &= good
    for k in sorted(set(bb) - set(cb)):
        lines.append(f"info baseline record {k[0]}/{k[1]} not in the candidate")
    lines.append("ACCEPT" if ok else "REJECT")
    return ok, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand")
    ap.add_argument("baseline")
    ap.add_argument("--target", default="paper:accuracy")
    ap.add_argument("--pcc-key", default="pcc_normalized")
    ap.add_argument("--pcc-floor", type=float, default=0.995)
    ap.add_argument("--min-gain", type=float, default=2.0)
    ap.add_argument("--guard-max", type=float, default=3.0)
    ap.add_argument("--allow-cross-chip", action="store_true")
    a = ap.parse_args(argv)
    ok, lines = compare(
        load(a.cand), load(a.baseline), a.target, a.pcc_key, a.pcc_floor, a.min_gain, a.guard_max, a.allow_cross_chip
    )
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
