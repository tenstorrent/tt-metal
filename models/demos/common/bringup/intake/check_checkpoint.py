# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Intake gate: the checkpoint holds every tensor the spec expects, with the listed shapes, and its config agrees
with the spec.

    python -m models.demos.common.bringup.intake.check_checkpoint --spec S

spec.yaml:
    checkpoint:
      expect:                                   # glob -> shape; "*" in a shape matches any size
        "model.language_model.layers.*.self_attn.q_proj.weight": [4096, 2816]
        "model.language_model.embed_tokens.weight": ["*", 2816]
      count: {"model.language_model.layers.*.self_attn.q_proj.weight": 30}   # optional exact match counts
      config: {num_hidden_layers: 30, hidden_size: 2816}   # optional: text config fields that must match

Records checkpoint_tensors, missing_tensors (expected globs matching nothing), shape_mismatches, count_mismatches,
config_mismatches.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.plan.memory import checkpoint_tensors
from models.demos.common.bringup.reference.golden import hf_path, load_spec


def check(spec, tensors: dict, config: dict) -> dict:
    exp = spec.get("checkpoint.expect") or {}
    missing, bad_shape, bad_count, bad_cfg = [], [], [], []
    for glob, shape in exp.items():
        hits = {n: s for n, s in tensors.items() if fnmatch.fnmatchcase(n, glob)}
        if not hits:
            missing.append(glob)
            continue
        for n, s in hits.items():
            if len(s) != len(shape) or any(w != "*" and int(w) != g for w, g in zip(shape, s)):
                bad_shape.append(f"{n}: {s} != {shape}")
    for glob, n in (spec.get("checkpoint.count") or {}).items():
        got = sum(fnmatch.fnmatchcase(t, glob) for t in tensors)
        if got != n:
            bad_count.append(f"{glob}: {got} tensors != {n}")
    want_cfg = dict(spec.get("checkpoint.config") or {})
    want_cfg.setdefault("num_hidden_layers", spec.num_layers)
    for k, v in want_cfg.items():
        if config.get(k) != v:
            bad_cfg.append(f"config {k}={config.get(k)!r} != spec {v!r}")
    return {"missing": missing, "shape": bad_shape, "count": bad_count, "config": bad_cfg}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    hf = Path(hf_path(spec))
    tensors = checkpoint_tensors(hf)
    cfg = json.loads((hf / "config.json").read_text())
    res = check(spec, tensors, cfg.get("text_config", cfg))
    for kind, items in res.items():
        for x in items:
            print(f"{kind.upper():8} {x}")
    print(f"{len(tensors)} tensors in {hf}")
    metrics.record("checkpoint_tensors", len(tensors))
    metrics.record("missing_tensors", len(res["missing"]))
    metrics.record("shape_mismatches", len(res["shape"]))
    metrics.record("count_mismatches", len(res["count"]))
    metrics.record("config_mismatches", len(res["config"]))


if __name__ == "__main__":
    main()
