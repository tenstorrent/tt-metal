# CI backup — fork automation that keeps this branch synced

These are **copies**, not the live files. The live ones sit on the fork's default branch
(`main`), because GitHub only runs `schedule` triggers from the default branch. They cannot run
from here.

They are copied here because `main` on this fork is otherwise just a stale upstream snapshot
carrying these few commits. GitHub's "Sync fork" offers *"Discard N commits and match upstream"*,
which would erase the automation in one click with nothing to alert you but the next morning's
routine reporting `FAIL-NO-RUN`. This directory means the wipe costs a restore, not a rewrite.

| File | Runs where | Does what |
|---|---|---|
| `sync-upstream.yml` | GitHub-hosted | Daily, merges latest `tenstorrent/tt-metal` into `feature/tt-hw-planner` and pushes. Merge only, no build. Files/comments an issue on unexpected conflict. |
| `berlin_gate.sh` | GitHub-hosted | Picks which of the two crons means 06:00 Europe/Berlin. GitHub cron is UTC-only and DST-blind, so both `0 4` and `0 5` are scheduled and this drops the wrong twin. Fails open. |
| `berlin-gate-selftest.yml` | GitHub-hosted | Dispatch-only proof of that gate: every DST x cron combination plus the fail-open cases. |
| `verify-tt-device.yml` | self-hosted `tt-verify` | Daily on a TT box: build, `import ttnn`, on-device matmul PCC, `test_rms_norm` PCC. Yields to humans — a busy or unhealthy board postpones instead of failing. |

## Restoring after a wipe

    cp scripts/tt_hw_planner/ci_backup/sync-upstream.yml        .github/workflows/
    cp scripts/tt_hw_planner/ci_backup/verify-tt-device.yml     .github/workflows/
    cp scripts/tt_hw_planner/ci_backup/berlin-gate-selftest.yml .github/workflows/
    cp scripts/tt_hw_planner/ci_backup/berlin_gate.sh           .github/scripts/

...committed to the fork's **default branch**, then check Settings -> Actions that the workflows
are `active`. Prevention is better: branch protection on `main` blocking force pushes and
deletions makes "discard commits" impossible in the first place.
