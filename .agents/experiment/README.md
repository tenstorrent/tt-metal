# Per-cell run isolation for multigoal experiments

**Status: working prototype, validated by dry run and negative tests. Not wired into any run plan yet.**

A controlled experiment (N arms × M models, one factor toggled per arm) is only controlled if cell N
cannot observe cell N−1. This directory holds tooling that makes that true structurally, rather than
by asking the operator to be careful.

Full write-up, incident history and rationale:
`agentic-research` → branch `mvasiljevic/pipeline-isolation` →
`skill-contribution-experiments/skillexp-fusing-advisor/PIPELINE-HARDENING.md`

---

## Why

The skillexp 2×2 experiment (graph-fusing × shard-advise, 4 models, 16 cells) ran every stage in one
reused clone, switching arms with `git checkout -B skillexp-work <arm>`. Five separate incidents
traced back to that one decision, because a reused clone shares four things between cells:

| shared thing | what leaked | incident |
|---|---|---|
| git **object database** | another arm's committed artifacts, readable via `git show <sha>:<path>` | advisor output appeared in a `noadvise` cell |
| **untracked files** | a previous run's outputs (these survive `git reset --hard`) | a `b32` run's output leaked into a `b1` rerun |
| **branch namespace** | a reused work-branch name, rebased onto the wrong base | both suppressed factors silently restored |
| **working tree** | publisher and stage runner colliding | ~7 h idle; a tag created at the pre-merge commit, twice |

Each was fixed with its own guard. All the guards passed on the incident that mattered, and honestly
so: `git show <sha>:<path>` reads the object store without moving `HEAD`, touching the index, or
changing a file, so every guard that inspected the *checkout* was describing the wrong surface.

The lesson is not "add a sixth guard". It is that **an experiment's controls should be enforced by
what a run can reach, not by what it is asked to avoid.**

## How

```
$SKILLEXP_ROOT/mirrors/<arm>.git      bare, ONE per arm. Refspec admits ONLY:
                                        base + this arm + refs/tags/.../fd-ready/*
                                      Deliberately NOT other arms' branches or done tags.

$SKILLEXP_ROOT/cells/<arm>__<model>/  git clone --shared --single-branch from that mirror.
                                      Created per cell, destroyed after publish.
```

Because the mirror is arm-scoped, a cell's `git log --all` returns only its own history and another
arm's commit does not resolve at all:

```console
$ git show 9c405211e7f:.../shard_advise/report.json
fatal: invalid object name '9c405211e7f'.
```

An operator audit that fetches another arm's refs now lands in the admin clone, and no cell borrows
from it — which is what actually caused the original incident.

### Two traps, both found while building this

**`--shared`/`--reference` against an all-arms clone silently restores the bug.** Borrowed objects
stay resolvable, `git log --all` finds them again, and you are back where you started while believing
you are isolated. This is why there are four arm-scoped mirrors rather than one shared one.

**The canonical path must stay fixed.** `python_env/lib/*/site-packages/*.pth` hardcodes the original
checkout path. A cell root at a *new* path therefore imports `ttnn` **and the model code** from the
**old** checkout while looking completely fresh — the wrong source measured with no symptom. So the
canonical path stays put and becomes a symlink to the active cell; cells are swapped by repointing it.
Build artifacts keep resolving for the same reason.

## Files

| file | purpose |
|---|---|
| `newcell.sh` | create the arm-scoped mirror and a fresh cell root; assert isolation; strip artifacts that contradict the arm's condition |
| `run_cell.sh` | one entry point per cell: fresh root → preflight lock → stage → postflight lock → writable-paths diff audit → publish → teardown |
| `arms/<arm>.lock` | declarative per-arm condition: which factors are on, which paths must be absent, which must exist afterwards, which paths the stage may write |

```bash
export SKILLEXP_NS=refs/heads/<your>/skillexp     # ref namespace
bash run_cell.sh <arm> <model_dir> --dry-run      # whole harness, no model call, ~2 s
bash run_cell.sh <arm> <model_dir>                # real run
```

`--dry-run` exercises every step except the model call (via `multigoal --dry-run`). This is worth
using before every campaign: the three most expensive failures in the source experiment were harness
bugs, not model bugs, and each would have been caught in seconds.

## Verified

| check | result |
|---|---|
| cell creation | 1.2 s, 395 MB per cell, 517 MB per arm mirror (source clone was 24 GB) |
| another arm's commit inside a cell | `fatal: invalid object name` |
| consecutive cells | cell 2 reports cell 1's tip unreachable |
| negative test: planted forbidden artifact | guard fires |
| negative test: stage writes outside its model dir | audit fires |
| `multigoal --dry-run`, both prompts | `rc=0`, substitutions resolved |

## Known gaps

- **Stage gates.** `.agents/prompts/model_bringup_multigoal/*.check.sh` exists only for stages 05+.
  Stages 01, 01b and 02 have none, so "complete" for them is a model-authored status field. This is
  independent of isolation and probably the larger correctness gap.
- **`multigoal` has no git awareness** — it makes no git calls at all, so run-root hygiene is left to
  whatever shell each operator writes. Folding `--fresh-root` / `--lock` / `--manifest` into the runner
  would give every future experiment these properties by default.
- **Machine-specific bits remain.** Paths are env-overridable (`SKILLEXP_ROOT`, `SKILLEXP_CANON`,
  `SKILLEXP_NS`, `SKILLEXP_LOGROOT`) but arm names and the MoE model list are still hardcoded.
- **Activation requires the canonical checkout to be moved aside** (see the path trap above).
  `run_cell.sh` hard-stops on a real run until that is done rather than silently running in place.
