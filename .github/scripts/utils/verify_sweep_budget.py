"""Verify that a sweep workflow's total runner-minutes per SKU fits the time budget.

Usage:
    python3 verify_sweep_budget.py <time_budget.yaml> <workflow_key> \
        --sku <sku_name> --job-count <N> --timeout-minutes <M>

Exits non-zero when the total (job_count * timeout_minutes) exceeds the budget
for the given team + workflow + SKU.
"""

import argparse
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description="Verify sweep job budget")
    parser.add_argument("budget_file", help="Path to .github/time_budget.yaml")
    parser.add_argument("workflow_key", help="Budget category key (e.g. sweep, sweep_validation)")
    parser.add_argument("--team", default="ttnn", help="Team name in budget file")
    parser.add_argument("--sku", required=True, help="SKU name (e.g. wh_galaxy)")
    parser.add_argument("--job-count", type=int, required=True, help="Number of parallel jobs")
    parser.add_argument("--timeout-minutes", type=int, required=True, help="Per-job timeout in minutes")
    args = parser.parse_args()

    with open(args.budget_file) as f:
        budgets = yaml.safe_load(f)

    team_budgets = budgets.get(args.team, {})
    workflow_budgets = team_budgets.get(args.workflow_key, {})
    allowed = workflow_budgets.get(args.sku)

    total_requested = args.job_count * args.timeout_minutes

    if allowed is None:
        print(f"⚠️  No budget defined for {args.team}.{args.workflow_key}.{args.sku} — skipping check")
        sys.exit(0)

    print(f"Budget check: {args.team}.{args.workflow_key}.{args.sku}")
    print(f"  Jobs: {args.job_count} × {args.timeout_minutes} min = {total_requested} min")
    print(f"  Budget: {allowed} min")

    if total_requested > allowed:
        print(f"❌ OVER BUDGET by {total_requested - allowed} min")
        sys.exit(1)
    else:
        print(f"✅ Within budget ({allowed - total_requested} min headroom)")
        sys.exit(0)


if __name__ == "__main__":
    main()
