# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Plan gate: the plan fits per-chip DRAM (computed from the checkpoint), components.yaml maps every step,
the task ledger is valid, and a person approved exactly these plan files.

    python -m models.demos.common.bringup.plan.check_plan --spec S

Writes <bringup_dir>/results/plan_memory.json (the dashboard's sharding section reads it) and records
plan_fits, per_chip_total_gb, per_chip_budget_gb, unplaced_tensors, plan_errors, component_errors, ledger_errors,
plan_approved. With a deployment target in the spec (F64) it also writes results/plan_boxes.json (plan/boxes.py) and
records deploy_fits and box_path_differences.
"""

from __future__ import annotations

import argparse
import json
import re

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.plan import approvals, components
from models.demos.common.bringup.plan.boxes import check_boxes, table
from models.demos.common.bringup.plan.memory import check_plan, checkpoint_tensors, load_plan
from models.demos.common.bringup.reference.golden import hf_path, load_spec


def text_config(hf_dir) -> dict:
    from pathlib import Path

    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    return cfg.get("text_config", cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    hf = hf_path(spec)
    plan, tensors, cfg = load_plan(spec), checkpoint_tensors(hf), text_config(hf)
    # F64: a deployment plan places every layer; the run's own check keeps the run's layers
    res = check_plan(spec, plan, tensors, cfg, layers=spec.layers() if spec.has_deploy else None)
    out = spec.bringup_dir / "results" / "plan_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in res.items() if k != "unplaced"}, indent=1) + "\n")
    for r in res.get("rows", []):
        print(f"  {r['group']:60s} {r['gb']:8.2f} GB")
    if "total_gb" in res:
        print(f"  {'TOTAL per chip':60s} {res['total_gb']:8.2f} GB of {res['budget_gb']:.2f} GB budget")
    # F58: a plan made after the serving contract must say how its layout meets it
    md = spec.bringup_dir / "serving_contract.md"
    plan_md = spec.bringup_dir / "plan.md"
    if md.exists() and not (plan_md.exists() and re.search(r"^##\s+Serving contract\b", plan_md.read_text(), re.M)):
        res["errors"].append("plan.md has no '## Serving contract' section answering serving_contract.md")
    if spec.has_deploy:  # F64: the plan is for the deployment box; every box gets its fit, scale and path table
        bx = check_boxes(spec, plan, tensors, cfg)
        (out.parent / "plan_boxes.json").write_text(json.dumps(bx, indent=1) + "\n")
        print(table(bx["rows"]))
        res["errors"] += bx["errors"]
        if not (plan_md.exists() and re.search(r"^##\s+Boxes\b", plan_md.read_text(), re.M)):
            res["errors"].append("plan.md has no '## Boxes' section (the per-box table and every path that differs)")
        dep = next((r for r in bx["rows"] if r["deploy"]), None)
        metrics.record("deploy_fits", int(bool(dep) and dep["full"]["fits_deploy_seq"]))
        metrics.record(
            "box_path_differences", sum(not c["same_as_deploy"] for r in bx["rows"] for c in r["conditions"])
        )
    for e in res["errors"]:
        print(f"PLAN ERROR {e}")

    ref = spec.hooks().reference(
        spec, layers=sorted({spec.representative_layer(b) for b in spec.data["block_types"]}), dtype=torch.float32
    )
    comp_errs = components.validate(spec, ref)
    for e in comp_errs:
        print(f"COMPONENT ERROR {e}")
    led_errs = Ledger(spec.bringup_dir).validate()
    for e in led_errs:
        print(f"LEDGER ERROR {e}")
    approved = approvals.is_approved(spec, "plan")
    print(f"plan approved: {approved}")

    metrics.record("plan_fits", int(res["fits"]))
    metrics.record("per_chip_total_gb", round(res.get("total_gb", -1), 2))
    metrics.record("per_chip_budget_gb", round(res.get("budget_gb", -1), 2))
    metrics.record("unplaced_tensors", len(res.get("unplaced", [])))
    metrics.record("plan_errors", len(res["errors"]))
    metrics.record("component_errors", len(comp_errs))
    metrics.record("ledger_errors", len(led_errs))
    metrics.record("plan_approved", int(approved))


if __name__ == "__main__":
    main()
