# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Backfill new metrics into a fixed record from its saved heads (PLAN v1.32).

    python -m models.experimental.kairos_forecast.tests.perf.backfill_heads <heads dir> <record> [--write]

<heads dir> holds `<name>_<precision>.pt` ({"head", "q"}) from a job run with $KAIROS_SAVE_HEADS. Each record is
re-scored from its heads; every metric the record already has must match within 1e-12 (else exit 1), and the
metrics it lacks are added. Also writes `sha256.txt` in <heads dir>.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from models.experimental.kairos_forecast.tests.perf.check_gates import load_records
from models.experimental.kairos_forecast.tests.perf.suite import load, score

TOL = 1e-12


def golden(name):
    """(golden raw head, golden final, scale) matching what run_case / teacher_forced scored."""
    base = name.removesuffix("_tf")
    rec = load(base)
    if name.endswith("_tf"):
        from models.experimental.kairos_forecast.reference.dataflow import head_to_quantiles

        n = len(rec["head"])
        gq = torch.cat([head_to_quantiles(rec["head"][k], *rec["loc_scale"][k]) for k in range(n)])
        sc = torch.cat([rec["loc_scale"][k][1].reshape(-1) for k in range(n)])
        return torch.cat(rec["head"]), gq, sc, 2 if rec["gift"] else 1
    steps = -(-rec["horizon"] // 128)
    gold = torch.cat(rec["head"]) if steps == 1 else rec["head"][0]
    halves = 2 if rec["gift"] and steps == 1 else 1
    return gold, rec["prediction_outputs"], rec["loc_scale"][0][1].reshape(-1), halves


def diffs(new, old, path=""):
    if isinstance(old, dict):
        for k, v in old.items():
            if k in new:
                yield from diffs(new[k], v, f"{path}/{k}")
    elif isinstance(old, list):
        if len(new) != len(old):
            yield f"{path}: length {len(new)} vs {len(old)}"
            return
        for i, (a, b) in enumerate(zip(new, old)):
            yield from diffs(a, b, f"{path}[{i}]")
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        if new is None or abs(new - old) > TOL:
            yield f"{path}: {new} vs {old}"
    elif new != old:
        yield f"{path}: {new!r} vs {old!r}"


def main(heads_dir, record, write=False):
    heads_dir, record = Path(heads_dir), Path(record)
    recs = load_records(record)
    bad, added = [], set()
    for r in recs:
        h = torch.load(heads_dir / f"{r['name']}_{r['precision']}.pt")
        gold, g_q, scale, halves = golden(r["name"])
        m = score(h["head"], gold, h["q"], g_q, scale, halves)
        bad += [f"{r['name']}/{r['precision']}{d}" for d in diffs(m, {k: r[k] for k in m if k in r})]
        for k in m.keys() - r.keys():
            r[k] = m[k]
            added.add(k)
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(recs)} records, {len(bad)} mismatches; added {sorted(added)}")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(heads_dir.glob("*.pt"))]
    (heads_dir / "sha256.txt").write_text("\n".join(sums) + "\n")
    if bad:
        return 1
    if write:
        text = (
            "".join(json.dumps(r) + "\n" for r in recs)
            if str(record).endswith(".jsonl")
            else json.dumps(recs, indent=1) + "\n"
        )
        record.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--write" in sys.argv))
