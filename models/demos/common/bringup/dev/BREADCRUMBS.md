# Bring-up framework build: breadcrumbs

Append-only log, one section per framework task. The design is
`models/demos/ernie45_d_p/bringup/docs/pipeline_design.html` (draft 4); the plan is section 7 of
`models/demos/ernie45_d_p/bringup/HANDOFF.md`. The framework is validated with synthetic ledgers and fixtures
and by reading ERNIE's existing records; nothing of ERNIE is regenerated. The first real use is the next model.

Drive this ledger with
`python -m models.demos.common.bringup gate <id> --spec models/demos/common/bringup/dev/spec.yaml --commit`.

## F1 (2026-09-25): core
- `core/spec.py`: model spec (YAML). Only `model` is needed to run gates; `validate()` checks the full model schema
  (layers, block types covering every layer once, ladder rungs, state tensors, mesh). Paths: `repo`, `bringup_dir`
  (ledger in git), `art` (`/localdev/$USER/bringup/<model>`, outside git) with `hf/ golden/ tt_cache/<mesh>/<version>/
  profiles/ runs/<run>/`.
- `core/ledger.py`: tasks.yaml + state.json + flock. Topological order, downstream closure (for rerun), runnable set.
- `core/gate.py`: the verdict order is deps -> frozen hashes -> device policy -> rc -> metrics -> artifacts.
  HANG is its own verdict (safe runner exit 2) and the triage report is copied next to the log. Logs go to
  `$ART/runs/<run>/logs/`, not git. The commit stages only the ledger files and the task's `paths`, and nothing
  is written to state after the commit, so the tree stays clean.
- Device policy parses the command quote-aware (shlex): a device gate must use `scripts/run_safe_pytest.sh` or
  `scripts/tt-probe.sh`; any direct `pytest` / `python -m pytest` fails; `VAR=x scripts/run_safe_pytest.sh` is allowed.
- `core/metrics.py`: `record(name, value)`; the task and the results dir come from `BRINGUP_TASK` / `BRINGUP_RESULTS_DIR`,
  which the gate sets. Own `pcc()` (the precompile pass stubs `comp_pcc`).
- Gotcha: the ledger `.lock` showed up as untracked after a commit; the ledger writes a `.gitignore` for it.
- Selftest gate: `selftest/test_core.py` (31 tests, temp git repos, no device). The selftest conftest records
  `selftest_passed` / `selftest_failed`, so an empty collection cannot pass.
