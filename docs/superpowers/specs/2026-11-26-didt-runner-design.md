# didt_runner.py design

Branch: `mbezulj/didt-gwdt-tests` (PR #67274), on top of d9006726d27.

## Intent

- Run the DIDT ND/hang screens on one Blackhole galaxy, one command.
- Same settings and practices as gwdt's `matmul:*` steps in the 2026-11-21 sweeps (734b76a2caa).
- `config.json` says what runs, how, and which fallbacks.
- `status.json` is the result; the gwdt sweep page reads it.
- Every failing chip is named by hardware identity (galaxy-chip-id standard).
- Step two (later, not in this design): galaxy -> chip -> core -> voltage/frequency sweep.

## Goals

- One run = every configured step, each after its own reset.
- FAIL and HANG kept apart; a HANG has a tt-triage CSV.
- Per failing device: chip identity, failed checks, pattern, top cores.
- Nothing overwritten: one file set per attempt.
- A stale node id fails before any device work.

## Non-goals

- No model, MLA, fabric or prefill steps (gwdt keeps those).
- No chip/core drill-down or V/F sweep (step two).
- No CI wiring; no change to `deployment/didt/test_runner.sh` or the exabox health check.
- No live screen/TUI; plain log lines are enough.

## Anti-goals

- A failing step must never stop other steps.
- No `-k`: exact node ids only.
- No import from `models/demos/...`; forked code lives in `tests/didt/`.
- No silent config typo: unknown fields are errors.
- No device id ("Device 19") reported without bdf/tray/asic/asic_id.

## Constraints

- Python only, run from the repo root with `python_env`.
- Login node: never run it there.
- sbatch: needs `--propagate=NONE` and `sg slurmusers` (else ulimit -u 512 and tt-smi reset fail). Document in `--help`.
- `nd:minimal` per-chip test (`test_specific_chip_minimal_matmul`) is assumed done by a separate change.

## Approaches

| | config.json holds | Python holds | verdict |
|---|---|---|---|
| **A (picked)** | steps, ids, opts, env, timeouts, fallbacks by name, check name | runners (pytest, script), fixed set of checks | small Python; new test = config entry if a check fits |
| B | A + pass/fail regexes | one generic runner | regexes drift, hard to unit-test |
| C | command list | exit code + triage | loses ND chip/core data, the point of PR #67274 |

User picked A.

## Files

| file | role |
|---|---|
| `tests/didt/didt_runner.py` | CLI entry; `python tests/didt/didt_runner.py [...]` |
| `tests/didt/didt_runner_config.json` | shipped config = the step table below |
| `tests/didt/runner/config.py` | load + strict validate |
| `tests/didt/runner/steps.py` | runners, reset, triage env, fallbacks, run loop |
| `tests/didt/runner/checks.py` | log parsers (forked from `gwdt_checks.py`, trimmed) |
| `tests/didt/runner/chips.py` | host chip snapshot from sysfs (forked from galaxy-chip-id `chipid.py`) |
| `tests/didt/runner/status.py` | `status.json` writer |
| `tests/didt/runner/test_*.py` | unit tests, no device |

- `runner/` not `didt_runner/`: a package and `didt_runner.py` with one name clash on import. *(my call)*

## Steps (shipped config)

Settings from the gwdt sweeps unless marked. THROTTLE = env `TT_MM_THROTTLE_PERF=5`.

- Fallbacks: THROTTLE only (user). gwdt's CORE_CAP (`--grid-size 11x10` on minimal) is dropped on purpose.

| key | runner | target | iters / compare | env / opts | timeout | check | fallbacks |
|---|---|---|---|---|---|---|---|
| minimal | pytest | `tests/didt/test_minimal_matmul.py::test_minimal_matmul[blackhole-bf16_HiFi2-galaxy]` | `--didt-workload-iterations=1000` / `--determinism-check-interval=50` | `--timeout=0` | none | nd_report | throttle |
| ff1 | pytest | `tests/didt/test_ff1_matmul.py::test_ff1_matmul[blackhole-galaxy-without_gelu]` | 1000 / 50 | `--timeout=0` | none | nd_report | throttle |
| lm-head | pytest | `tests/didt/test_lm_head_matmul.py::test_lm_head_matmul[blackhole-galaxy]` | 1000 / 50 *(my call: same as ff1; not in gwdt)* | `--timeout=0` | none | nd_report | throttle |
| duty-cycle-bursty-5 | pytest | `tests/didt/test_duty_cycle.py::test_duty_cycle[blackhole-duty-5-rep-1000x-all]` | 10 / 1 | `--timeout 1800`; `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto` | none | nd_report | throttle |
| run-to-run | pytest | `tests/didt/test_matmul_run_to_run_determinism.py::test_matmul_run_to_run_determinism[blackhole-galaxy]` | `TT_DET_SWEEP_ITERS=10` | `-p no:randomly --timeout=0` | none | run_to_run | throttle |
| core-sweep | pytest | `tests/didt/scratch_core_sweep.py::test_core_sweep[blackhole-galaxy]` | `TT_SWEEP_ITERS=10` | `TT_SWEEP_MODE=grid TT_SWEEP_GRID=11x10`; `-p no:randomly --timeout=0` | none | core_sweep | throttle |
| mcast-ack | script | `python tests/didt/test_mm_mcast_write_ack_failure.py`; collect id `...::test_mm_mcast_write_ack_failure_scan[blackhole-1x1]` | `MATMUL_ITERS=10` | `MATMUL_WORKLOAD=wkv_b1` | 1800 s | write_ack | throttle |
| mcast-ack-duty | script | same script; collect id `...::test_mm_mcast_write_ack_failure_duty[blackhole-1x1]` | `MATMUL_ITERS=10`, compare every 1 (test default) | `MATMUL_WORKLOAD=duty` | 1800 s | write_ack + nd_report per chip | throttle |

- Exact node ids are checked by the collect gate; the builder confirms each collects on this branch.
- Keys in status.json use the prefix `nd:` (user): `nd:minimal`, `nd:ff1`, ... (table shows the short form). Ops other than matmul go under `nd:` too.
- The sweep page (`build-gwdt-sweep` `sweep.py`) knows only `matmul:` today: add `nd` to `GROUP_NAMES` and `nd:*` to `LABELS`. gwdt pages keep `matmul:`; the two do not share columns.

## config.json schema (version 1)

```json
{
  "version": 1,
  "hang_secs": 120,
  "reset": {"cmd": "tt-smi -glx_reset_auto", "tries": 3, "timeout": 600},
  "smi": {"cmd": "tt-smi -s --snapshot_no_tty", "timeout": 120},
  "collect": {"cmd": "pytest --collect-only -q -p no:benchmark", "timeout": 600},
  "fallbacks": {
    "throttle": {"env": {"TT_MM_THROTTLE_PERF": "5"}, "opts": []}
  },
  "steps": [
    {
      "key": "nd:minimal",
      "runner": "pytest",
      "ids": ["tests/didt/test_minimal_matmul.py::test_minimal_matmul[blackhole-bf16_HiFi2-galaxy]"],
      "opts": ["--didt-workload-iterations=1000", "--determinism-check-interval=50", "--timeout=0"],
      "env": {},
      "timeout": null,
      "check": "nd_report",
      "fallbacks": ["throttle"],
      "note": "free text, shown in --list"
    }
  ]
}
```

| field | rule |
|---|---|
| `runner` | `pytest` or `script` |
| `ids` | pytest: run + collect; script: collect only |
| `cmd` | script only; required there |
| `check` | one of `nd_report`, `run_to_run`, `core_sweep`, `write_ack` |
| `fallbacks` | names from top-level `fallbacks`; run in list order |
| `timeout` | seconds or null; wall limit on the whole attempt |
| `note` | free text; replaces code comments |

Validation, before any device work; all errors printed, then exit 2:
- unknown or missing fields; duplicate keys; unknown `check`, `runner`, fallback name.
- `-k` or `-m` in `opts`; `ids` without `::`.
- `env` values must be strings.

## Run flow

1. Load + validate config. `--co`: collect only, then write status and exit.
2. Chip snapshot (sysfs, opens no chip): `chips[]` in status.json. Must run before any device step.
3. Collect gate: one `pytest --co` over every step's `ids`. Missing id -> step FAIL "does not collect"; that step is skipped, the rest run.
4. Per step, in config order, unless `--only`/`--skip` excludes it:
   1. reset, up to `reset.tries`; all fail -> step FAIL, next step.
   2. run with triage env (below) and step `timeout`.
   3. tt-smi snapshot.
   4. classify (HANG if triage CSV non-empty, else the step's `check`).
   5. GDDR check on the step's tt-smi snapshots (uncorrected EDC -> FAIL, corrected -> WARN).
   6. FAIL or HANG -> every fallback in order, each its own reset/run/snapshot/classify.
   7. write status.json.
5. Exit code (below).

- Triage env, every pytest attempt: `TT_METAL_OPERATION_TIMEOUT_SECONDS=<hang_secs>`, `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` = tt-triage with gwdt's `HANG_TRIAGE` scripts, `--llm-output-path=<crash_triage_NN.csv>`, then `kill -9 $PPID`.
- Script runner (mcast-ack): no triage env (the script arms its own watchdog); `MCAST_ACK_WORK_DIR=<attempt dir>/chips` (script change: `__main__` passes `MCAST_ACK_WORK_DIR` to `scan_all_chips(work_dir=...)`; unset = `/tmp/mm_mcast_write_ack_failure`). The script deletes that dir at start, so it must be the dedicated `chips/` subfolder, never the attempt folder itself. Kernel caches: `MCAST_ACK_CACHE_DIR=/tmp/didt_runner_<run>/<attempt>` on local disk (32 concurrent JIT builds into one NFS tree fail with link/assembler write errors, which the scan reports as hangs); the script keys them `kmd<minor>/` per chip; the runner deletes the dir after the attempt.
- Step status = first attempt; fallbacks never change it.
- `status.json` is rewritten after every state change (atomic: temp file + rename).

## Run dir

`<out>/LOG_<YYYY_MM_DD_HH_MM local time>_<host>_<commit11>/` (gwdt layout; `--out` default `generated/didt_runner`).

| file | when |
|---|---|
| `status.json` | always, run root |
| `run.log` | runner's own log |
| `collect/collect.log` | collect gate |
| `<dir>/reset.log`, `reset.retryK.log` | reset before the attempt, retry K |
| `<dir>/log` | attempt output |
| `<dir>/crash_triage.csv` | attempt hung |
| `<dir>/tt-smi.json` | after the attempt |
| `<dir>/chips/chip<k>/pytest.log` | script runner, per chip |

- `<dir>` = step key with `:` -> `/` for the first attempt (`nd/minimal/`).
- Each fallback gets its own sibling folder `<dir>_<fallback name>` (`nd/minimal_throttle/`), same file names inside (user).

## status.json

Top level: same keys as gwdt so `build-gwdt-sweep` renders it unchanged, plus `config`, `chips`.

```json
{
  "tool": "didt_runner",
  "host": "bh-glx-110-a06u02",
  "run": "LOG_2026_11_27_09_10_bh-glx-110-a06u02_d9006726d27",
  "mode": "real",
  "commit": "d9006726d27",
  "run_dir": "/data/.../LOG_...",
  "config": {"path": "tests/didt/didt_runner_config.json", "sha256": "..."},
  "chips": [
    {"dev": 19, "bdf": "0000:84:00.0", "tray": 4, "asic": 4, "asic_id": "0x81caba39768f9912", "kmd": 27}
  ],
  "current": null,
  "steps": [
    {
      "key": "nd:minimal",
      "status": "HANG",
      "reason": "triage .../nd/minimal/crash_triage.csv",
      "details": ["device 19 bdf=0000:84:00.0 tray=4 asic=4: 3/20 checks, pattern single core"],
      "log": ".../nd/minimal/log",
      "start": 1795266232.5,
      "end": 1795269000.1,
      "devices": [],
      "fallbacks": [
        {
          "name": "throttle", "env": {"TT_MM_THROTTLE_PERF": "5"}, "opts": [],
          "status": "PASS", "reason": "1 passed", "details": [], "devices": [],
          "log": ".../nd/minimal_throttle/log", "start": 1795267000.0, "end": 1795268000.0
        }
      ]
    }
  ]
}
```

| field | meaning |
|---|---|
| `status` | `PENDING`, `RUN`, `PASS`, `WARN`, `FAIL`, `HANG`, `SKIP` |
| `reason` | one line |
| `details` | human lines; one per failing device, then GDDR lines |
| `devices[]` | failing devices of that attempt only (empty on PASS) |
| `fallbacks[].name` | config fallback name (new vs gwdt) |
| `chips[]` | host snapshot before the run; `dev` = `bdf_rank` |

`devices[]` entry:

```json
{
  "dev": 19, "bdf": "0000:84:00.0", "tray": 4, "asic": 4, "asic_id": "0x81caba39768f9912", "kmd": 27,
  "failed_checks": 3, "checks": 20,
  "pattern": "single core",
  "hung": false,
  "op": "MinimalMatmulDeviceOperation",
  "core_count": 120,
  "cores": [
    {"x": 6, "y": 8, "translated": [7, 9], "failed_checks": 3, "signature": "..."}
  ]
}
```

- Identity fields: from `chips[]` joined on `dev`. Cross-check `bdf` with the log's `Chip map device N` line; mismatch -> keep the log's, add a `details` warning.
- `cores`: up to 8, as logged by `ND summary device N core ...`.
- Absent data: field omitted, never guessed.

### Per check

HANG, any check: devices from the triage CSV, written by tt-triage while the workload is still up (dispatch-timeout hook), before the kill:
- `dump_running_operations.py` section: `Devices` (stalled device ids), `Op Name`, `Core Cnt`, `Cores`.
- `dump_op_mesh.py` section: `T<tray>:N<asic> (Device N)` per cell; cross-check tray/asic with `chips[]`.
- Each stalled device -> `hung: true`, `op`, `core_count`, `cores` (as listed; triage truncates with `...`).
- Checked on 110-a06u02 2026-11-21 `matmul/minimal/crash_triage_01.csv`: Device 27 = T3:N4, MinimalMatmulDeviceOperation, 120 cores.
- The end-of-run ND lines never print on a HANG; triage is the only source.

| check | PASS | FAIL / devices source |
|---|---|---|
| `nd_report` | pytest summary all passed | pytest failed; devices from `ND summary device N [...]` and its core lines |
| `run_to_run` | pytest passed | devices + cores from the test's `logical core (x,y) -> physical (a,b): run-to-run NOT deterministic, n elements` lines |
| `core_sweep` | pytest passed and 0 skipped windows | skipped window -> FAIL; devices from per-device differing-element lines |
| `write_ack` | `all N chips passed` | `chip N FAIL: HUNG ...` -> device `hung: true`; `chip N FAIL: ND ...` (duty) -> device from that chip's `pytest.log` via `nd_report`, `hung: false` *(my call)* |

- `nd_report` with interval 0 logs no ND lines: then `devices` stays empty, not an error.

## Device identity

- `dev` in logs = `bdf_rank` of the run (`TT_VISIBLE_DEVICES` unset: whole host).
- Snapshot reads `/sys/class/tenstorrent/tenstorrent!N/device` and `tt_asic_id`; tray/asic from the bus digits (galaxy-chip-id table).
- mcast-ack `chip N` = `bdf_rank` of the whole host (BDF-sorted, test_mm_mcast_write_ack_failure.py:113-118, :336).
- Each mcast-ack child runs with `TT_VISIBLE_DEVICES=<chip>`, so its own log says `device 0`. Join that chip's lines on the parent's `chip N`, not on the log's device number.
- Never report `dev` alone.

## CLI

| flag | meaning |
|---|---|
| `--config PATH` | default `tests/didt/didt_runner_config.json` |
| `--only k1,k2` / `--skip k1,k2` | step selection; unknown key -> exit 2 |
| `--co` | collect gate only, no device work |
| `--list` | print steps (key, ids, fallbacks, note) and exit |
| `--out DIR` | run dir parent |

## Exit codes

| code | meaning |
|---|---|
| 0 | every run step PASS or WARN |
| 1 | any step FAIL or HANG (first attempt) |
| 2 | config/usage error, or chip snapshot failed |

## Tests (no device)

- Config: shipped config loads; each validation rule rejects its bad case.
- Shipped config builds exactly the 8 steps above, in order.
- Each check against saved log snippets (real sweep logs where they exist).
- `devices[]` join: snapshot + log line -> identity fields; bdf mismatch -> warning.
- Fallback loop: FAIL runs all fallbacks; PASS runs none; step status stays first attempt's.
- status.json: renders with `build-gwdt-sweep` `sweep.py render`.

## Step two (placeholder only)

- New fallback kinds: `per_chip` (failing devices -> per-chip ids or `TT_VISIBLE_DEVICES`), `per_core` (core sweep on that chip), `vf_sweep` (voltage/frequency; no knob found yet).
- `devices[]` is the input to those: it already names chip and cores.

## Left to the builder

- Function names and module split inside `runner/`.
- Log line format of `run.log`.
- Exact regexes, verified against real logs.
