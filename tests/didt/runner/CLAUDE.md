# didt_runner: rules for extending it

Read before changing `tests/didt/didt_runner.py`, `tests/didt/didt_runner_config.json` or anything in `tests/didt/runner/`.

## What it is

- One galaxy, one command, every configured DIDT ND/hang screen.
- `didt_runner_config.json` says what runs, how, and which fallbacks.
- `status.json` is the only result. People and the sweep page read it.
- Its job: name the failing chip, then the core, by hardware identity.

## Component map

```
didt_runner.py               CLI: args, chip snapshot, run dir, exit code
 -> runner/config.py         strict loader of didt_runner_config.json; all errors at once, before device work
 -> runner/steps.py          run loop: collect gate, reset, attempt, tt-smi, fallbacks (throttle, pin_core)
     -> runner/checks.py     log parsers and verdicts (Status), GDDR errors from tt-smi, loop time
     -> runner/devices.py    failing devices[] of one attempt, identity joined from chips on `dev`
     -> runner/chips.py      sysfs snapshot `dev bdf tray asic asic_id kmd`; opens no device
     -> runner/status.py     status.json writer, atomic
```

Read by the tests the runner launches, never imported by `runner/`:

- `tests/didt/didt_chip.py`: `pick_submesh`, `DIDT_DEVICE_ID` picks a per-chip test's submesh by device id.
- `tt_py_test_utils_common/op_test_base.py`: writes the `Chip map`, `ND summary` and iteration lines the parsers read.
- `tt_py_test_utils_common/nd_core_map.py`: per-core ND attribution (cores, pattern) inside those lines.

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
| a test run once per chip, all chips at once | config entry with `"runner": "each_chip"` |
| new fallback | config `fallbacks{}`; code only for a new fallback kind |
| new status field | `runner/status.py` + the `status.json` section below |

## status.json

- Top level: gwdt's keys (`tool host run mode commit run_dir`) + `config` (`path`, `sha256`), `chips`, `current`,
  `steps`. Never rename or drop a gwdt key: `build-gwdt-sweep` renders the file unchanged.
- `chips[]`: host snapshot before the run, `dev bdf tray asic asic_id kmd`; `dev` = BDF rank on the host.
- Step: `key status reason details log start end devices fallbacks[]`. A fallback adds `name env opts` in front,
  plus its own kind fields (`pin_core`: `target trigger cores iterations loop`).
- `status`: `PENDING RUN PASS WARN FAIL HANG SKIP`. `details`: one line per failing device, then GDDR lines.
- `devices[]`: failing devices of that attempt only, empty on PASS. Fields the source does not give are left out.

| `devices[]` field | from |
|---|---|
| `dev bdf tray asic asic_id kmd` | `chips[]` joined on `dev`; if the log's `Chip map` BDF differs, keep the log's + a `details` warning |
| `hung` | always |
| `failed_checks checks pattern` | `nd_report` (`ND summary device N` line) |
| `op core_count` | HANG: triage CSV `dump_running_operations.py` (`each_chip`: the chip's own CSV) |
| `elements` | `core_sweep` |
| `cores[]` | up to 8, worst first: `x y`, `translated` if logged, then `failed_checks signature` or `elements` |

## Adding a screen

1. Confirm the exact node id collects: `pytest --collect-only -q '<id>'`.
2. Determinism check on (interval > 0) if the test uses `OpTestBase`: no ND lines otherwise.
3. Set `nd_core_block` / `nd_core_grid_size` in the test for per-core data.
4. Key: `nd:<short-name>`; folder follows the key.
5. Config entry with `note` saying why it is there.
6. Unit test: shipped config builds the new step list in order.

## Adding a check or parser

- Test first, from REAL log text saved in `runner/testdata/` (cite the source run dir).
- Return the `devices[]` shape above; identity joined from the chip snapshot on `dev`.
- A per-chip child process sees itself as `device 0`: join on the parent's chip id.
- Log format comes from the test (`op_test_base.py`, test file). Change both together, never one.

## each_chip runner

- The step's ids once per chip, all chips at once, in `chips/chip<dev>/`: `pytest.log`, `crash_triage.csv` on a hang.
- Each process: `TT_VISIBLE_DEVICES=<dev>`, own kernel cache, own inspector port (`50100 + dev`) and log path.
  Without its own inspector, triage reads another chip's process or none.
- Hang handling as gwdt: dispatch timeout -> tt-triage into the chip's CSV -> `kill -9` of that pytest.
- A chip with a non-empty CSV is hung; step HANG if any chip hung, else FAIL if any failed. The attempt `log` has one
  line per chip.
- Check is always `nd_report`, joined on the chip (the child calls itself device 0).

## Fallbacks and step two

- Today: `throttle` (`TT_MM_THROTTLE_PERF=5`), then `pin_core` on every step.
- `pin_core`: each failing device of the first attempt (hung, or with ND data) reruns alone via the step's `pin_id`
  with `DIDT_DEVICE_ID=<dev>` (an `each_chip` step: its own ids on that chip only, found by BDF after the reset), at
  twice the iterations and at least the original attempt's loop time (`DIDT_MIN_LOOP_SECONDS`, honoured by
  `op_test_base.run_op_test`; read from the original log's iteration debug lines, for an `each_chip` step from the
  target's `chips/chip<dev>/pytest.log`). Hang or ND is per device, never from the step status: an `each_chip` step
  can hang on one chip and show ND on another. The mcast wkv_b1 workload logs no iteration lines and ignores
  `DIDT_MIN_LOOP_SECONDS`. A per-chip test must pick its chip with
  `tests/didt/didt_chip.py` `pick_submesh`, never `submeshes[logical_chip_id]` alone: the submesh index is not the
  device id. A new per-chip test = a `pin_id` in the config, collected on a galaxy first.
- Not reproduced on the chip alone is WARN, not PASS. A reset failure inside `pin_core` aborts the run.
- Step two direction, in this order: galaxy -> chip -> core -> voltage/frequency.
  - `per_chip`: done as `pin_core`.
  - `per_core`: core sweep on that chip.
  - `vf_sweep`: no knob found yet; find and document it before building.
- A new fallback kind consumes `devices[]` of the attempt before it. Never re-parse logs for it.
- `TT_VISIBLE_DEVICES`: dropped from the caller's env (`DROPPED_ENV`), so `dev` = BDF rank. An `each_chip` child
  gets the rank: UMD maps an integer through its BDF-sorted device list, and the mcast test parses it as an integer.
  Never `TT_METAL_VISIBLE_DEVICES`.

## Don'ts

- No gating between steps.
- No CORE_CAP-style fallback without a stated question it answers.
- No new dependency in `runner/` (stdlib only).
- No `pkill`, no `kill_aliens`, no `recover.sh` from the runner.
- No writes outside the run dir, except `each_chip` kernel caches: local `/tmp/didt_runner_<run>/` (32 concurrent JIT
  builds fail on NFS), removed after each attempt.
- No device runs on slurm-login.

## Before you call it done

- Unit tests: `python_env/bin/python -m pytest -q --noconftest -p no:cacheprovider tests/didt/runner`
- `python tests/didt/didt_runner.py --co` on a galaxy: exit 0, all steps collect.
- This file updated if a rule or the `status.json` shape changed.
