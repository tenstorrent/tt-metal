# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F186: a golden made on another host, imported: the G task of a seeded run (plan/ledger_gen.py generate_seeded).

    python -m models.demos.common.bringup.reference.golden_import --spec S --rung R      # the G task's gate
    python -m models.demos.common.bringup.reference.golden_import --spec S --rung R --index   # where it was made

Goldens at real size cost CPU hours to days (DeepSeek-V4.1, 40 layers: 2 x 5120 tokens ~50 min, 11 chunks 13.5 h)
and need the whole checkpoint in RAM, so they are made once on a big host and copied to the box that runs the model.
The copy may leave out files no test of the run reads (files.json lists them all). The gate checks, without the
checkpoint: the manifest is this rung's (seq, chunk, layers, model, full-precision numerics); every copied file matches
the index and the index gives the manifest's content_hash; the drift file of every rung reading this golden is there
(made with the golden, reference/drift.py: the end-to-end margins). A golden not copied yet fails with where it goes.
"""

from __future__ import annotations

import argparse
import json

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference import drift
from models.demos.common.bringup.reference.golden import TARGET_NUMERICS, load_spec, rung_dir, verify_copy, write_index


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--rung", required=True)
    ap.add_argument("--index", action="store_true", help="write files.json beside a complete golden (before copying)")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    rung = spec.rung(a.rung)
    d = rung_dir(spec, rung)
    if not (d / "manifest.json").exists():
        raise SystemExit(
            f"golden {a.rung} not on this host: copy it to {d} (made elsewhere; files.json lists its files)"
        )
    if a.index:
        print(f"{d}: files.json, {len(write_index(d))} files")
        return
    m = json.loads((d / "manifest.json").read_text())
    want = {
        "seq": rung["seq"],
        "chunk": rung["chunk"],
        "layers": spec.layers(),
        "model": spec.data["hf_id"],
        "numerics": TARGET_NUMERICS,
    }
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if bad:
        raise SystemExit(f"{d}: the golden is not this rung's (got, want): {bad}")
    ok, missing = verify_copy(d)
    readers = drift.readers(spec, a.rung)
    absent = [r for r in readers if not drift.drift_path(d, r).exists()]
    if absent:
        raise SystemExit(
            f"{d}: no drift for rung(s) {absent} (the end-to-end margin of every rung reading this golden); make them "
            f"where the golden was made: python -m models.demos.common.bringup.reference.drift --rung <rung>"
        )
    metrics.record("golden_layers", len(m["layers"]))
    metrics.record("golden_chunks", m["n_chunks"])
    metrics.record("golden_hash_ok", int(ok))
    metrics.record("golden_files_missing", len(missing))
    metrics.record("golden_imported", 1)
    metrics.record("drift_metrics", sum(len(json.loads(drift.drift_path(d, r).read_text())) for r in readers))
    for k, v in drift.margin(json.loads(drift.drift_path(d, a.rung).read_text())).items():
        metrics.record(k, v)
    print(f"imported {d} (content_hash {m.get('content_hash', '')[:12]}, {len(missing)} files not copied)")


if __name__ == "__main__":
    main()
