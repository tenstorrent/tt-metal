# didt_runner: rules for extending it

Read before changing `tests/didt/didt_runner.py`, `tests/didt/didt_runner_config.json` or anything in `tests/didt/runner/`.
Design and reasons: `docs/superpowers/specs/2026-11-26-didt-runner-design.md`.

## What it is

- One galaxy, one command, every configured DIDT ND/hang screen.
- `didt_runner_config.json` says what runs, how, and which fallbacks.
- `status.json` is the only result. People and the sweep page read it.
- Its job: name the failing chip, then the core, by hardware identity.

## Invariants (never break)

| rule | why |
|---|---|
| A FAIL/HANG never stops another step | one bad step must not hide others; only a failed reset inside `pin_core` aborts the run (user's rule) |
| Reset before every attempt, fallbacks included | each result starts from a clean chip state |
| Step status = first attempt; fallbacks never change it | the first result is the screen result |
| Every fallback runs, in config order, own folder `<dir>_<name>` | each factor gets its own result; nothing overwritten |
| Exact node ids; no `-k`, no `-m` | substring ids pick wrong params (`iter2` matches `iter25`) |
| Every id passes the collect gate before device work | a stale id fails in seconds, not hours |
| HANG only from a non-empty triage CSV | FAIL and HANG must stay apart |
| A chip is never reported by number alone | `dev` is run-local; `bdf tray asic asic_id kmd` name hardware |
| Missing data is omitted, never guessed | a wrong chip is worse than none |
| `runner/` imports without ttnn | unit tests run anywhere, with `--noconftest` |
| Nothing from `models/demos/...` | tests must not depend on a model demo |
| `status.json` written atomically after every change | a killed run still leaves valid JSON |

## Where things go

| change | where |
|---|---|
| new screen of an existing kind | config entry only |
| new pass/fail or device parser | `runner/checks.py` or `runner/devices.py`, + a `check` name |
| new way to launch | `runner/steps.py`, + a `runner` name |
| new fallback | config `fallbacks{}`; code only for a new fallback kind |
| new status field | `runner/status.py` + spec's status.json section |

## Adding a screen

1. Confirm the exact node id collects: `pytest --collect-only -q '<id>'`.
2. Determinism check on (interval > 0) if the test uses `OpTestBase`: no ND lines otherwise.
3. Set `nd_core_block` / `nd_core_grid_size` in the test for per-core data.
4. Key: `nd:<short-name>`; folder follows the key.
5. Config entry with `note` saying why it is there.
6. Unit test: shipped config builds the new step list in order.

## Adding a check or parser

- Test first, from REAL log text saved in `runner/testdata/` (cite the source run dir).
- Return the spec's `devices[]` shape; identity joined from the chip snapshot on `dev`.
- A per-chip child process sees itself as `device 0`: join on the parent's chip id.
- Log format comes from the test (`op_test_base.py`, test file). Change both together, never one.

## Fallbacks and step two

- Today: `throttle` (`TT_MM_THROTTLE_PERF=5`), then `pin_core` on every step.
- `pin_core`: each failing device of the first attempt (ND or hang) reruns alone via the step's `pin_id` with
  `DIDT_DEVICE_ID=<dev>`, at twice the iterations. A per-chip test must pick its chip with
  `tests/didt/didt_chip.py` `pick_submesh`, never `submeshes[logical_chip_id]` alone: the submesh index is not the
  device id. A new per-chip test = a `pin_id` in the config, collected on a galaxy first.
- Not reproduced on the chip alone is WARN, not PASS. A reset failure inside `pin_core` aborts the run.
- Step two direction, in this order: galaxy -> chip -> core -> voltage/frequency.
  - `per_chip`: done as `pin_core`.
  - `per_core`: core sweep on that chip.
  - `vf_sweep`: no knob found yet; find and document it before building.
- A new fallback kind consumes `devices[]` of the attempt before it. Never re-parse logs for it.
- `TT_VISIBLE_DEVICES`: always full BDF form `0000:c1:00.0`; never `TT_METAL_VISIBLE_DEVICES`.

## Don'ts

- No gating between steps.
- No CORE_CAP-style fallback without a stated question it answers.
- No new dependency in `runner/` (stdlib only).
- No `pkill`, no `kill_aliens`, no `recover.sh` from the runner.
- No writes outside the run dir; script tools get a dedicated subfolder (they may rmtree it).
- No device runs on slurm-login.

## Before you call it done

- Unit tests: `python_env/bin/python -m pytest -q --noconftest -p no:cacheprovider tests/didt/runner`
- `python tests/didt/didt_runner.py --co` on a galaxy: exit 0, all steps collect.
- Spec updated if behavior or `status.json` changed.
