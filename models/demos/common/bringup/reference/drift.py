# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): the end-to-end margin. The rung's golden at the reference's own shipped precision (``-official``) is
compared with its full-precision golden (the target) by the same metrics the ladder and the contract use; the device
may then miss the full-precision golden by calib x that drift (testing/e2e_limits.py), never by a fixed bar.

    python -m models.demos.common.bringup.reference.drift --rung s55296

Writes ``drift.json`` into the full-precision golden: {metric: {"drift": value, "res": resolution floor}}; the floor
of a float metric is the same metric for the target rounded once to bf16 (no device output can do better), of an
index overlap one miss per hundred (defaults e2e.overlap_res). Records drift_<metric> for the gate.
"""

from __future__ import annotations

import argparse
import json

import torch

from models.demos.common.bringup.core import defaults, metrics
from models.demos.common.bringup.reference.golden import Golden, load_spec, official_dir, rung_dir

OVERLAP_RES = float(defaults.get("e2e.overlap_res"))


def _pcc_entry(official: torch.Tensor, target: torch.Tensor) -> dict:
    t = target.float()
    return {"drift": metrics.pcc(official.float(), t), "res": metrics.pcc(t.bfloat16().float(), t)}


def drift(target: Golden, official: Golden, layers: list[int], full_stack: bool) -> dict:
    from models.demos.common.bringup.testing.ladder import index_overlap

    last = target.n_chunks - 1
    out = {}
    for i in layers:
        out[f"pcc_layer_L{i:02d}"] = _pcc_entry(official.layer(last, i)["out"], target.layer(last, i)["out"])
        want, got = target.state(i), official.state(i)
        for name, t in want.items():
            if t.is_floating_point():
                out[f"pcc_state_{name}_L{i:02d}"] = _pcc_entry(got[name], t)
            else:
                out[f"overlap_state_{name}_L{i:02d}"] = {"drift": index_overlap(got[name], t), "res": 1 - OVERLAP_RES}
    if full_stack:
        gt, go = target.model(last), official.model(last)
        out["pcc_final_hidden"] = _pcc_entry(go["final_norm"], gt["final_norm"])
        top1 = gt["top32_ids"][:, 0].long()
        out["top5_overlap"] = {
            "drift": float((go["top32_ids"][:, :5].long() == top1[:, None]).any(-1).float().mean()),
            "res": 1 - OVERLAP_RES,
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--rung", required=True)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    rung = spec.rung(a.rung)
    target, official = Golden(rung_dir(spec, rung)), Golden(official_dir(spec, rung))
    for g, want in ((target, "native"), (official, "official")):
        if g.manifest.get("numerics", "official") != want:
            raise SystemExit(f"{g.dir}: numerics {g.manifest.get('numerics', 'official')!r}, want {want!r} (F128)")
    layers = [i for i in target.layers if i in official.layers]
    d = drift(target, official, layers, max(layers) == spec.num_layers - 1)
    (target.dir / "drift.json").write_text(json.dumps(d, indent=1, sort_keys=True))
    for k, v in d.items():
        metrics.record(f"drift_{k}", v["drift"])
    metrics.record("drift_metrics", len(d))
    worst = min(d, key=lambda k: d[k]["drift"])
    print(
        f"drift of the official reference from full precision: {len(d)} metrics, worst {worst} {d[worst]['drift']:.6f}"
    )


if __name__ == "__main__":
    main()
