# Brief: $tid, role $role (attempt $attempt)

Model: $model (`$spec_path`). Run: $run. Written by the orchestrator at $now.

## Task
$title

$role_text

$rules
$prior
$deferred
$serving
## Read first
- `models/demos/common/bringup/knowledge/repo_map.md`
- `models/demos/common/bringup/knowledge/known_issues.md`
- `$bringup_dir/repo_map.md` and `$bringup_dir/known_issues.md` (this model's own; absent until a step adds to them)
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
If you learned something new, add it to `$bringup_dir/known_issues.md` with the file's next id (or a pointer to
`$bringup_dir/repo_map.md`); append to `$breadcrumbs`. Write the knowledge-use report `$knowledge_report` (agent rule
11): the entries that changed what you did, the ones you found wrong, and what was missing. End with a short summary.
Do not commit.
