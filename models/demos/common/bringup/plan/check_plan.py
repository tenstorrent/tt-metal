# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Plan gate: the plan fits per-chip DRAM (computed from the checkpoint), components.yaml maps every step,
the task ledger is valid, and a person approved exactly these plan files.

    python -m models.demos.common.bringup.plan.check_plan --spec S

Writes <bringup_dir>/results/plan_memory.json (the dashboard's sharding section reads it) and records
plan_fits, per_chip_total_gb, per_chip_budget_gb, unplaced_tensors, plan_errors, component_errors, ledger_errors,
plan_approved.
"""

from __future__ import annotations

import argparse
import json

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.plan import approvals, components
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
    res = check_plan(spec, load_plan(spec), checkpoint_tensors(hf), text_config(hf))
    out = spec.bringup_dir / "results" / "plan_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in res.items() if k != "unplaced"}, indent=1) + "\n")
    for r in res.get("rows", []):
        print(f"  {r['group']:60s} {r['gb']:8.2f} GB")
    if "total_gb" in res:
        print(f"  {'TOTAL per chip':60s} {res['total_gb']:8.2f} GB of {res['budget_gb']:.2f} GB budget")
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
