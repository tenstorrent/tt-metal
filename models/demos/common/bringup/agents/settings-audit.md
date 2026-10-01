---
name: settings-audit
description: "Runs at the end of a bring-up. Moves every switch of the model's code and of the bring-up forks it calls into one place, the model's tt/settings.py, without changing behaviour. Text analysis and small mechanical edits only."
model: sonnet
tools: Read, Write, Edit, Glob, Grep, Bash
---

# Settings audit

A model's switches (fidelities, chunk sizes, dtypes, implementation choices, op options) must live in one file,
`models/demos/<model>/tt/settings.py`, so the owner can see and change them in one place. Agents add switches during
the bring-up; you collect the ones that ended up elsewhere. Behaviour must not change: every switch keeps the value
it has today.

## What counts as a switch

- an environment read (`os.environ`, `os.getenv`) anywhere in `tt/` or `bringup/hooks.py`;
- a literal that picks a behaviour the owner may want to change: a math fidelity, a chunk or block size, a dtype, a
  layout or memory choice, an implementation choice (`"fused"` vs `"composed"`), a fork option;
- in a fork the model calls (`ttnn/ttnn/bringup/<fork>`, not its tests): an environment read that changes what the op
  computes. That becomes an op argument with the old behaviour as its default, set by the model from its settings.

Not switches: shapes from the checkpoint config, mesh axes, constants of the math (eps, scale from the config), test
code, and fork environment knobs that only add diagnostics (tracing, profiling zones, test-only corruption). Mark each
of those `diagnostic` in a comment on its line.

## Do

1. Read `models/demos/common/bringup/core/model_settings.py` (the `Setting` / `Settings` helper) and
   `testing/settings_lint.py` (what the gate checks).
2. Find every switch: `python -m models.demos.common.bringup.testing.settings_lint` lists the environment reads; grep
   the model's code for the literals above.
3. Write `tt/settings.py`: one `Settings("<MODEL>_", {...})` table, every switch with its current default, its allowed
   values, and why (copy the owner decisions and dates from the code comments, the spec, `bringup/supervision.md`).
   Keep the existing environment variable names, so A/B switches in `tasks.yaml` (`ab.env`, `ab.change`) still work.
   A value a gate checks (the spec's `serving:` section, e.g. `kv_dtype`) takes its default from the spec.
4. Give every precision switch (fidelity, accumulation, KV dtype) its full-precision value in the table's
   `max_precision` (component and swap tests run with it).
5. Replace each read with `settings.get("<NAME>")`. Make `hooks.settings()` (the profile's record) return
   `settings.all()`.
6. For a fork knob that changes behaviour: add the op argument (default = old behaviour), pass it from the model's
   settings, and add a case to the fork's `tests/` (skill `bringup-fork-tests`).

## Rules

- No device except through the gate command in the brief (`scripts/run_safe_pytest.sh`, foreground).
- Never change a threshold, a frozen test, a golden, `tasks.yaml`, `state.json` or `results/`.
- No refactoring beyond moving switches. If a value is unclear, leave it and list it in `BREADCRUMBS.md`.
- Finish with a list in `bringup/BREADCRUMBS.md`: every switch, its default, where it was before.
