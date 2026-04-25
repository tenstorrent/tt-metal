#!/usr/bin/env python3
"""
budget_check.py — CI time-budget enforcer for tenstorrent/tt-metal

Calculates machine-hour consumption per team × SKU × pipeline,
compares against budgets in .github/time_budgets/, and enforces limits.

Usage:
  python3 scripts/budget_check.py --report           # full status report
  python3 scripts/budget_check.py --diff HEAD~1      # delta check for PR
  python3 scripts/budget_check.py --update-status    # regenerate BUDGET_STATUS.md

Duration source:
  - pipeline_reorg tests: `timeout:` field per test entry (minutes)
  - non-pipeline_reorg jobs: `timeout-minutes:` field in the GHA job
  Cost = timeout_hours × shards × runs_per_week

Team ownership:
  - pipeline_reorg: `team:` field in the test YAML (file-level or per-entry)
  - non-pipeline_reorg: resolved from CODEOWNERS; unknown → 'unattributed' (warning only)
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_DIR = REPO_ROOT / ".github" / "time_budgets"
PIPELINE_REORG_DIR = REPO_ROOT / "tests" / "pipeline_reorg"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CODEOWNERS_PATH = REPO_ROOT / ".github" / "CODEOWNERS"
STATUS_FILE = BUDGET_DIR / "BUDGET_STATUS.md"

SKU_CONFIG_PATH = REPO_ROOT / ".github" / "sku_config.yaml"

_SKU_CONFIG_CACHE: dict | None = None


def _sku_config() -> dict[str, list[str]]:
    """Return {sku_name: [required_labels_lowercase]} from sku_config.yaml."""
    global _SKU_CONFIG_CACHE
    if _SKU_CONFIG_CACHE is None:
        if SKU_CONFIG_PATH.exists():
            raw = load_yaml(SKU_CONFIG_PATH)
            _SKU_CONFIG_CACHE = {
                sku: [str(l).lower() for l in (cfg.get("runs_on") or [])]
                for sku, cfg in (raw.get("skus") or {}).items()
            }
        else:
            _SKU_CONFIG_CACHE = {}
    return _SKU_CONFIG_CACHE

WARN_THRESHOLD = 0.85  # warn at 85% of budget


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def git_show(ref: str, path: Path) -> str | None:
    """Return file content at a git ref, or None if absent."""
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout if result.returncode == 0 else None


def changed_files(base_ref: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return [REPO_ROOT / f.strip() for f in result.stdout.splitlines() if f.strip()]


def extract_sku(runs_on) -> str | None:
    """Match a GHA runs-on value against sku_config.yaml entries.

    Returns the sku_config key whose runs_on labels are all present in the
    job's runner labels.  Prefers the most-specific match (most labels).
    """
    labels = [runs_on] if isinstance(runs_on, str) else (runs_on if isinstance(runs_on, list) else [])
    label_set = {str(l).lower() for l in labels}
    best_sku, best_score = None, 0
    for sku, required in _sku_config().items():
        if not required:
            continue
        if all(r in label_set for r in required) and len(required) > best_score:
            best_score = len(required)
            best_sku = sku
    return best_sku


def matrix_shards(job: dict) -> int:
    """Count the maximum matrix dimension in a GHA job (proxy for shard count)."""
    matrix = job.get("strategy", {})
    if not isinstance(matrix, dict):
        return 1
    matrix = matrix.get("matrix", {})
    if not isinstance(matrix, dict):
        return 1
    # Ignore 'include'/'exclude' keys
    sizes = [len(v) for k, v in matrix.items() if isinstance(v, list) and k not in ("include", "exclude")]
    return max(sizes, default=1)


# ─── CODEOWNERS parsing ───────────────────────────────────────────────────────

def parse_codeowners() -> list[tuple[str, str]]:
    """Return [(pattern, team_slug), ...] in file order (last match wins)."""
    if not CODEOWNERS_PATH.exists():
        return []
    rules = []
    for line in CODEOWNERS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern = parts[0]
        # Use first owner; strip org prefix and normalise to snake_case
        owner = parts[1].lstrip("@")
        if "/" in owner:
            owner = owner.split("/", 1)[1]
        owner = re.sub(r"[-.]", "_", owner).lower()
        rules.append((pattern, owner))
    return rules


def _codeowners_match(pattern: str, rel_path: str) -> bool:
    """Simplified CODEOWNERS glob matching (covers the common cases)."""
    import fnmatch
    p = pattern.lstrip("/")
    # Directory rule: matches everything under it
    if p.endswith("/"):
        return rel_path.startswith(p)
    # No slash: match against filename anywhere in tree
    if "/" not in p:
        return fnmatch.fnmatch(Path(rel_path).name, p)
    # Anchored pattern
    return fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, f"**/{p}")


def codeowner_for(rules: list, path: Path) -> str | None:
    """Return the last matching CODEOWNERS team for a repo-relative path."""
    try:
        rel = str(path.relative_to(REPO_ROOT))
    except ValueError:
        return None
    matched = None
    for pattern, team in rules:
        if _codeowners_match(pattern, rel):
            matched = team
    return matched


# ─── Pipeline-reorg parsing ───────────────────────────────────────────────────

def parse_pipeline_reorg_yaml(path: Path, content: str | None = None) -> list[dict]:
    """
    Parse a tests/pipeline_reorg/*.yaml file into a list of cost entries.

    Actual schema — a YAML list where each item is a test job:

      - name: Galaxy CCL tests
        cmd:  pytest tests/nightly/tg/ccl ...
        skus:
          wh_galaxy:
            timeout: 60          # minutes on this SKU
          bh_galaxy:
            timeout: 30
        team: ttnn
        owner_id: U05ACKAJTHS

    Each (test, sku) pair becomes one entry with shards=1 (one parallel job).
    Files that are not a YAML list (e.g. ttsim-skip-list.yaml) are silently skipped.
    """
    if content is None:
        if not path.exists():
            return []
        content = path.read_text()

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return []

    if not isinstance(data, list):
        return []  # skip dict-shaped files (skip lists, etc.)

    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        team = item.get("team")
        skus_field = item.get("skus")
        if not isinstance(skus_field, dict):
            continue
        for sku_name, sku_cfg in skus_field.items():
            if isinstance(sku_cfg, dict):
                timeout = sku_cfg.get("timeout")
            elif isinstance(sku_cfg, (int, float)):
                timeout = sku_cfg
            else:
                continue
            if not (team and sku_name and timeout):
                continue
            entries.append({
                "name": item.get("name", "unknown"),
                "team": str(team),
                "sku": str(sku_name),
                "timeout_minutes": float(timeout),
                "shards": 1,  # each SKU entry is one parallel job
                "source": path.name,
            })
    return entries


def is_pipeline_reorg_workflow(path: Path, content: str | None = None) -> bool:
    if content is None:
        content = path.read_text(errors="replace") if path.exists() else ""
    return "prepare_test_matrix" in content or "prepare-test-matrix" in content


# ─── Non-pipeline-reorg parsing ──────────────────────────────────────────────

def parse_workflow_yaml(
    path: Path,
    codeowners_rules: list,
    content: str | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Parse a regular .github/workflows/*.yaml into cost entries.
    Pipeline-reorg workflows are skipped (they delegate to test YAMLs).
    Returns (entries, warnings).
    """
    if content is None:
        content = path.read_text(errors="replace") if path.exists() else ""

    if is_pipeline_reorg_workflow(path, content):
        return [], []

    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        return [], [f"YAML parse error in {path.name}"]

    jobs = data.get("jobs", {})
    if not isinstance(jobs, dict):
        return [], []

    team = codeowner_for(codeowners_rules, path)
    warnings = []
    if team is None:
        warnings.append(
            f"No CODEOWNERS match for {path.relative_to(REPO_ROOT)} — cost attributed to 'unattributed'"
        )
        team = "unattributed"

    entries = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        timeout = job.get("timeout-minutes")
        if not timeout:
            continue
        sku = extract_sku(job.get("runs-on"))
        if not sku:
            continue  # CPU/Linux runners not tracked
        entries.append({
            "name": job_name,
            "team": team,
            "sku": sku,
            "timeout_minutes": int(timeout),
            "shards": matrix_shards(job),
            "source": path.name,
        })
    return entries, warnings


# ─── Cost aggregation ────────────────────────────────────────────────────────

def compute_costs(entries: list[dict], runs_per_week: dict) -> dict[tuple, float]:
    """
    Returns {(team, sku): machine_hours_per_week}.
    runs_per_week keys: workflow filename (with or without .yaml extension).
    """
    costs: dict[tuple, float] = defaultdict(float)
    for e in entries:
        # Try with and without .yaml suffix
        src = e["source"]
        rpw = runs_per_week.get(src) or runs_per_week.get(Path(src).stem) or 0.0
        machine_hours = (e["timeout_minutes"] / 60.0) * e["shards"] * float(rpw)
        costs[(e["team"], e["sku"])] += machine_hours
    return dict(costs)


def get_all_entries(
    repo_root: Path,
    codeowners_rules: list,
) -> tuple[list[dict], list[str]]:
    """Scan the full repo and return (entries, warnings)."""
    entries: list[dict] = []
    warnings: list[str] = []

    pr_dir = repo_root / "tests" / "pipeline_reorg"
    if pr_dir.exists():
        for f in sorted(pr_dir.glob("*.yaml")):
            entries.extend(parse_pipeline_reorg_yaml(f))

    wf_dir = repo_root / ".github" / "workflows"
    if wf_dir.exists():
        for f in sorted(wf_dir.glob("*.yaml")):
            e, w = parse_workflow_yaml(f, codeowners_rules)
            entries.extend(e)
            warnings.extend(w)

    return entries, warnings


def budget_limit(pools: dict, allocations: dict, team: str, sku: str) -> float:
    return pools.get(sku, 0.0) * allocations.get(sku, {}).get(team, 0.0)


# ─── Report formatting ───────────────────────────────────────────────────────

def _status_symbol(used: float, limit: float) -> str:
    if limit <= 0:
        return "  ?"
    pct = used / limit
    if pct > 1.0:
        return "OVER"
    if pct > WARN_THRESHOLD:
        return "WARN"
    return "  OK"


def format_full_report(
    costs: dict,
    pools: dict,
    allocations: dict,
    warnings: list[str],
) -> tuple[str, bool]:
    lines: list[str] = []
    violations = False

    all_skus = sorted(set(s for _, s in costs) | set(pools))
    all_teams = sorted(set(t for t, _ in costs) - {"unattributed"})

    for sku in all_skus:
        pool = pools.get(sku, 0.0)
        lines.append(f"\n{'='*60}")
        lines.append(f"  {sku}  (pool: {pool:.0f} h/week)")
        lines.append(f"{'='*60}")
        lines.append(f"  {'Team':<24} {'Used h/wk':>10} {'Budget h/wk':>12} {'%':>6}  Status")
        lines.append(f"  {'-'*56}")
        for team in all_teams:
            used = costs.get((team, sku), 0.0)
            limit = budget_limit(pools, allocations, team, sku)
            if limit == 0 and used == 0:
                continue
            pct = (used / limit * 100) if limit > 0 else 0.0
            sym = _status_symbol(used, limit)
            if sym == "OVER":
                violations = True
            lines.append(f"  {team:<24} {used:>10.1f} {limit:>12.1f} {pct:>5.0f}%  {sym}")
        unattr = costs.get(("unattributed", sku), 0.0)
        if unattr > 0:
            lines.append(f"  {'(unattributed)':<24} {unattr:>10.1f} {'—':>12} {'—':>6}  WARN")

    if warnings:
        lines.append(f"\n--- Ownership warnings ({len(warnings)}) ---")
        for w in warnings[:20]:
            lines.append(f"  ! {w}")
        if len(warnings) > 20:
            lines.append(f"  ... and {len(warnings) - 20} more")

    return "\n".join(lines), violations


# ─── Diff / PR mode ──────────────────────────────────────────────────────────

def _entries_at_ref(
    files: list[Path],
    ref: str | None,
    codeowners_rules: list,
) -> list[dict]:
    """Parse a set of files, optionally at a git ref (None = working tree)."""
    entries: list[dict] = []
    for f in files:
        content = git_show(ref, f) if ref else (f.read_text() if f.exists() else "")
        if not content:
            continue
        if "pipeline_reorg" in str(f):
            entries.extend(parse_pipeline_reorg_yaml(f, content))
        elif ".github/workflows" in str(f):
            e, _ = parse_workflow_yaml(f, codeowners_rules, content)
            entries.extend(e)
    return entries


def compute_diff(
    base_ref: str,
    codeowners_rules: list,
    runs_per_week: dict,
) -> tuple[dict, dict, list[str]]:
    """Returns (old_costs, new_costs, warnings) for the PR delta."""
    changed = changed_files(base_ref)
    relevant = [
        f for f in changed
        if ("pipeline_reorg" in str(f) or ".github/workflows" in str(f) or ".github/time_budgets" in str(f))
        and f.suffix == ".yaml"
    ]
    if not relevant:
        return {}, {}, []

    old_entries = _entries_at_ref(relevant, base_ref, codeowners_rules)
    new_entries = _entries_at_ref(relevant, None, codeowners_rules)

    # Collect ownership warnings from new state only
    warnings: list[str] = []
    for f in relevant:
        if ".github/workflows" in str(f) and f.exists():
            _, w = parse_workflow_yaml(f, codeowners_rules)
            warnings.extend(w)

    old_costs = compute_costs(old_entries, runs_per_week)
    new_costs = compute_costs(new_entries, runs_per_week)
    return old_costs, new_costs, warnings


def format_diff_report(
    old_costs: dict,
    new_costs: dict,
    pools: dict,
    allocations: dict,
    baseline_costs: dict,
    warnings: list[str],
) -> tuple[str, bool]:
    """
    Show per-team/SKU delta and whether the post-merge total exceeds budget.
    baseline_costs = full current costs from the base ref (not just changed files).
    """
    lines = ["=== Budget Impact of This PR ===", ""]
    violations = False

    all_keys = sorted(set(old_costs) | set(new_costs))
    if not all_keys:
        lines.append("No relevant pipeline changes detected — budget unaffected.")
        return "\n".join(lines), False

    for (team, sku) in all_keys:
        old = old_costs.get((team, sku), 0.0)
        new = new_costs.get((team, sku), 0.0)
        delta = new - old
        if abs(delta) < 0.01:
            continue

        # Post-merge total = (current full total) - old contribution + new contribution
        post_merge = baseline_costs.get((team, sku), 0.0) - old + new
        limit = budget_limit(pools, allocations, team, sku)
        pct = (post_merge / limit * 100) if limit > 0 else 0.0
        sym = _status_symbol(post_merge, limit)
        if sym == "OVER":
            violations = True

        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {team} / {sku}: {sign}{delta:.1f} h/wk delta"
            f"  →  post-merge: {post_merge:.1f} / {limit:.1f} h/wk ({pct:.0f}%)  [{sym}]"
        )

    if warnings:
        lines.append("")
        lines.append("Ownership warnings:")
        for w in warnings[:5]:
            lines.append(f"  ! {w}")

    lines.append("")
    if violations:
        lines.append("RESULT: BUDGET EXCEEDED — PR is blocked.")
        lines.append("Reduce test timeouts/shards, remove tests, or request a budget increase.")
    else:
        lines.append("RESULT: All teams within budget after this change.")

    return "\n".join(lines), violations


# ─── BUDGET_STATUS.md generation ─────────────────────────────────────────────

def generate_status_md(
    costs: dict,
    pools: dict,
    allocations: dict,
    warnings: list[str],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CI Time Budget Status",
        "",
        f"> Auto-generated by `scripts/budget_check.py`. Last updated: {now}",
        "> Do not edit manually — re-run `budget_check.py --update-status` to refresh.",
        "",
    ]

    all_skus = sorted(set(s for _, s in costs) | set(pools))
    all_teams = sorted(set(t for t, _ in costs) - {"unattributed"})

    for sku in all_skus:
        pool = pools.get(sku, 0.0)
        lines.append(f"## {sku}  (pool: {pool:.0f} h/week)")
        lines.append("```")
        lines.append(f"{'Team':<25} {'Used':>8} {'Budget':>8} {'%':>6}  Status")
        lines.append("-" * 57)
        for team in all_teams:
            used = costs.get((team, sku), 0.0)
            limit = budget_limit(pools, allocations, team, sku)
            if limit == 0 and used == 0:
                continue
            pct = (used / limit * 100) if limit > 0 else 0.0
            sym = _status_symbol(used, limit)
            lines.append(f"  {team:<23} {used:>7.1f}h {limit:>7.1f}h {pct:>5.0f}%  {sym}")
        unattr = costs.get(("unattributed", sku), 0.0)
        if unattr > 0:
            lines.append(f"  {'(unattributed)':<23} {unattr:>7.1f}h {'—':>8} {'—':>6}  WARN")
        lines.append("```")
        lines.append("")

    if warnings:
        lines.append("## Ownership Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI time-budget checker for tenstorrent/tt-metal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--report", action="store_true", help="Print full budget report")
    parser.add_argument("--diff", metavar="BASE_REF", help="Check budget delta vs BASE_REF")
    parser.add_argument("--update-status", action="store_true", help="Regenerate BUDGET_STATUS.md")
    args = parser.parse_args()

    if not any([args.report, args.diff, args.update_status]):
        parser.print_help()
        sys.exit(0)

    pools = load_yaml(BUDGET_DIR / "pools.yaml")
    allocations = load_yaml(BUDGET_DIR / "allocations.yaml")
    runs_per_week = load_yaml(BUDGET_DIR / "runs_per_week.yaml")
    codeowners_rules = parse_codeowners()

    entries, warnings = get_all_entries(REPO_ROOT, codeowners_rules)
    all_costs = compute_costs(entries, runs_per_week)

    violations = False

    if args.report or args.update_status:
        report, v = format_full_report(all_costs, pools, allocations, warnings)
        print(report)
        violations = violations or v

    if args.update_status:
        md = generate_status_md(all_costs, pools, allocations, warnings)
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(md)
        print(f"\n→ Written: {STATUS_FILE}")

    if args.diff:
        old_costs, new_costs, diff_warnings = compute_diff(
            args.diff, codeowners_rules, runs_per_week
        )
        report, v = format_diff_report(
            old_costs, new_costs, pools, allocations, all_costs, diff_warnings + warnings
        )
        print(report)
        violations = violations or v

    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
