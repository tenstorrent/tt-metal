# Brief: $tid, role $role (attempt $attempt)

Model: $model (`$spec_path`). Run: $run. Written by the orchestrator at $now.

## Task
$title

$role_text

$rules
$prior
$deferred
## Read first
- `models/demos/common/bringup/knowledge/repo_map.md`
- `models/demos/common/bringup/knowledge/known_issues.md`
- `$spec_path` (the model spec)
$read_list

## You may change only
$allowed

Everything else is read-only for this step. The orchestrator diffs the tree when you finish.

## Gate you must pass
```
$gate_cmd
```
Thresholds (the runner compares the recorded metrics; you cannot change them):
$thresholds

Run it from the repo root with `PYTHONPATH=$repo`. Device code only through `scripts/run_safe_pytest.sh` or
`scripts/tt-probe.sh`, in the foreground.

$previous
## When you finish
Add a known-issues or repo-map proposal if you learned something new, append to `$breadcrumbs`, and end with a short summary.
Do not commit.
