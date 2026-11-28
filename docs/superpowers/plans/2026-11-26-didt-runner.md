# didt_runner.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python tests/didt/didt_runner.py` runs the DIDT ND/hang screens on one galaxy from `didt_runner_config.json` and writes `status.json` naming every failing chip by hardware identity.

**Architecture:** Thin CLI over a `tests/didt/runner/` package: config load/validate, chip snapshot, log/CSV parsers, run loop, status writer. Parsers are forked from gwdt `gwdt_checks.py` @ 734b76a2caa, trimmed. All parsing is pure and unit-tested without a device; one device task at the end.

**Tech Stack:** Python 3.10 (repo `python_env`), stdlib only in `runner/` (no ttnn import), pytest for unit tests.

**Spec:** `docs/superpowers/specs/2026-11-26-didt-runner-design.md` (read it first; this plan does not repeat it).

## Global Constraints

- Branch `mbezulj/didt-gwdt-tests`; work in `/data/mbezulj/tt-metal`.
- Base: bd09236e8d4 (has `MCAST_ACK_WORK_DIR` and per-chip minimal matmul be6a115fb12). Stage files by name only; the tree has many unrelated untracked files.
- No import from `models/demos/...`. `runner/` must import without ttnn (unit tests run with `--noconftest`).
- Exact node ids only; reject `-k`/`-m` in config `opts`.
- Step keys: `nd:<name>`; folder = key with `:` -> `/`; fallback folder `<dir>_<fallback name>`.
- Device ids are reported only with `bdf`, `tray`, `asic`, `asic_id`, `kmd`.
- `MCAST_ACK_WORK_DIR=<attempt dir>/chips` (dedicated; the script rmtree's it).
- Never run on slurm-login. Device work only on `bh-glx-110-c10u08` (Task 7).
- `tests/didt/runner/CLAUDE.md` (shipped with this plan) holds the extension rules; follow it, commit it in Task 1.
- Commits: no push. Message style: `tests/didt: <what>`.
- Unit test command (from repo root): `python_env/bin/python -m pytest -q --noconftest -p no:cacheprovider tests/didt/runner`

## Review Focus

1. Host with 31 chips (one dropped): `dev` join must use the snapshot's own BDF order, not tray/asic arithmetic. -> Task 2 test `test_join_with_missing_chip`.
2. Triage CSV cells are quoted, multi-line, and core lists end in `...`. -> Task 4 test `test_triage_hang_devices_real_shape`.
3. Runner killed mid-run: `status.json` must stay valid JSON. -> Task 5 test `test_write_is_atomic`.
4. Wall timeout with no triage CSV (host-side hang): FAIL `timeout after N s`, not PASS, not HANG. -> Task 6 test `test_wall_timeout_is_fail`.
5. One stale node id: only that step FAILs; others still run. -> Task 6 test `test_missing_id_fails_only_its_step`.

---

### Task 0: Check the base

- [ ] **Step 1:** `git log --oneline -3` shows bd09236e8d4 (MCAST_ACK_WORK_DIR) on top of be6a115fb12. `grep -n MCAST_ACK_WORK_DIR tests/didt/test_mm_mcast_write_ack_failure.py` -> the `__main__` line.
- [ ] **Step 2:** `git grep -n "def test_specific_chip_minimal_matmul" tests/didt/test_minimal_matmul.py` -> found (step two uses it; nothing to do now).

### Task 1: Forked parsers (`runner/checks.py`)

**Files:**
- Create: `tests/didt/runner/__init__.py` (empty), `tests/didt/runner/checks.py`
- Test: `tests/didt/runner/test_checks.py`, fixtures in `tests/didt/runner/testdata/`

**Interfaces:**
- Produces:
  - `class Status(str, Enum)`: `PENDING RUN PASS WARN FAIL HANG SKIP`
  - `@dataclass Result: status: Status; reason: str = ""; details: list[str] = []`
  - `classify_pytest(text: str, expected_passed: int) -> Result` (fork, unchanged behavior)
  - `collected_node_ids(text: str) -> list[str]`; `missing_node_ids(wanted, collected) -> list[str]` (compare the FULL node id, not only the part after `::`)
  - `sweep_skipped_windows(text) -> int | None`; `write_ack_chips(text) -> tuple[Result, dict[int, str]]` (chip N -> "HUNG ..." / "ND ..." text)
  - `gddr_edc(snapshot: dict) -> list[GddrEdc]`; `gddr_result(r, found, metal_by_bus) -> Result`; `umd_metal_ids(text) -> dict[int,int]` (fork)
  - `read_tail(path: Path, nbytes: int = 8 << 20) -> str`

- [ ] **Step 1: Tests first.** Copy gwdt's cases for these functions from `git show 734b76a2caa:models/demos/deepseek_v3_d_p/scripts/test_gwdt_checks.py` into `test_checks.py`, keeping their asserts. Add:

```python
def test_missing_node_ids_compares_full_id():
    assert missing_node_ids(["a/x.py::t[p]"], ["b/y.py::t[p]"]) == ["a/x.py::t[p]"]

def test_write_ack_chips_splits_hung_and_nd():
    text = "chip 3 FAIL: HUNG chip 3 (runtime id 0): ...\nchip 5 FAIL: ND chip 5 ...\nchip 1 PASS\n"
    r, chips = write_ack_chips(text)
    assert r.status == Status.FAIL and r.reason == "hang or ND on 2/3 chips"
    assert chips[3].startswith("HUNG") and chips[5].startswith("ND")
```

- [ ] **Step 2:** Run unit test command. Expected: FAIL, `ModuleNotFoundError: runner.checks` (or import error).
- [ ] **Step 3:** Implement by copying the gwdt functions (734b76a2caa `gwdt_checks.py` lines 19-35, 95-118, 130-137, 168-177, 240-258, 289-357), then apply the two changes the tests pin. Drop everything else.
- [ ] **Step 4:** Run unit test command. Expected: PASS.
- [ ] **Step 5:** `git add tests/didt/runner/CLAUDE.md tests/didt/runner/__init__.py tests/didt/runner/checks.py tests/didt/runner/test_checks.py tests/didt/runner/testdata && git commit -m "tests/didt: fork gwdt log parsers for didt_runner"`

### Task 2: Chip snapshot and identity (`runner/chips.py`)

**Files:** Create `tests/didt/runner/chips.py`; Test `tests/didt/runner/test_chips.py`

**Interfaces:**
- Produces:
  - `@dataclass Chip: dev: int; bdf: str; tray: int; asic: int; asic_id: str; kmd: int` and `Chip.as_dict() -> dict`
  - `snapshot(sysfs: Path = Path("/sys/class/tenstorrent")) -> list[Chip]` (sorted by bdf; `dev` = index)
  - `by_dev(chips: list[Chip]) -> dict[int, Chip]`

- [ ] **Step 1: Tests** with a fake sysfs tree in `tmp_path` (`tenstorrent!N/device` symlink to `.../0000:84:00.0`, file `tt_asic_id`):

```python
def test_tray_asic_from_bus():          # 0000:84:00.0 -> tray 4, asic 4; 0000:c1:00.0 -> tray 3, asic 1; 0000:01 -> tray 1
def test_dev_is_bdf_rank():             # 3 fake chips out of order -> dev 0,1,2 by sorted bdf
def test_join_with_missing_chip():      # 31 chips, one tray-4 chip absent -> every later dev shifts by 1, bdf/asic_id still right
```

- [ ] **Step 2:** Run unit test command. Expected: FAIL (no module).
- [ ] **Step 3:** Implement from galaxy-chip-id `chipid.py` `snapshot()` / `_tray_asic()` (copy at `/Users/mbezulj/.claude/plugins/cache/galaxy-skills/galaxy-skills/0.3.0/skills/galaxy-chip-id/chipid.py` on the laptop; on the cluster `/data/$USER/.claude/skills/galaxy-chip-id/chipid.py`). Bus high digit `0x0`->1, `0x4`->2, `0xC`->3, `0x8`->4; low digit = asic.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `tests/didt/runner/chips.py tests/didt/runner/test_chips.py`: `tests/didt: chip snapshot from sysfs for didt_runner`

### Task 3: Config (`runner/config.py`, shipped JSON)

**Files:** Create `tests/didt/runner/config.py`, `tests/didt/didt_runner_config.json`; Test `tests/didt/runner/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass Fallback: name: str; env: dict[str,str]; opts: list[str]`
  - `@dataclass StepCfg: key: str; runner: str; ids: list[str]; cmd: str | None; opts: list[str]; env: dict[str,str]; timeout: float | None; check: str; fallbacks: list[Fallback]; note: str`
  - `@dataclass Config: path: Path; sha256: str; hang_secs: int; reset: dict; smi: dict; collect: dict; steps: list[StepCfg]`
  - `load(path: Path) -> Config`; raises `ConfigError(errors: list[str])` listing ALL errors

- [ ] **Step 1: Tests:**

```python
SHIPPED = Path(__file__).parents[1] / "didt_runner_config.json"
def test_shipped_config_steps():
    c = load(SHIPPED)
    assert [s.key for s in c.steps] == ["nd:minimal", "nd:ff1", "nd:lm-head", "nd:duty-cycle-bursty-5",
        "nd:run-to-run", "nd:core-sweep", "nd:mcast-ack", "nd:mcast-ack-duty"]
    assert all([f.name for f in s.fallbacks] == ["throttle"] for s in c.steps)
    assert c.steps[0].fallbacks[0].env == {"TT_MM_THROTTLE_PERF": "5"}
# one test per rule, each asserting the message is in ConfigError.errors:
# unknown field, missing field, duplicate key, unknown check, unknown runner, unknown fallback,
# "-k" in opts, "-m" in opts, id without "::", non-string env value, script without cmd
def test_all_errors_reported_together():  # two bad fields -> len(errors) == 2
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write the JSON from the spec's step table and schema verbatim (ids, opts, env, timeouts, checks; mcast-ack `cmd: "python tests/didt/test_mm_mcast_write_ack_failure.py"`, env `MATMUL_ITERS=10` + `MATMUL_WORKLOAD=wkv_b1|duty`, collect ids `...::test_mm_mcast_write_ack_failure_scan[blackhole-1x1]` / `..._duty[blackhole-1x1]`). Implement `load` with hand-written validation (no jsonschema dependency).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit both files + test: `tests/didt: didt_runner config and strict loader`

### Task 4: Failing-device extraction (`runner/devices.py`)

**Files:** Create `tests/didt/runner/devices.py`; Test `tests/didt/runner/test_devices.py`; fixtures in `testdata/`

**Interfaces:**
- Consumes: `Chip`, `by_dev` (Task 2)
- Produces (each returns `list[dict]` in the spec's `devices[]` shape, identity filled from `chips`, missing data omitted):
  - `nd_report_devices(text: str, chips: dict[int, Chip], dev_override: int | None = None) -> tuple[list[dict], list[str]]` (second item = warning lines, e.g. bdf mismatch)
  - `run_to_run_devices(text, chips) -> list[dict]`
  - `core_sweep_devices(text, chips) -> list[dict]`
  - `triage_hang_devices(csv_text: str, chips) -> list[dict]` (`hung: True`, `op`, `core_count`, `cores`)
  - `write_ack_devices(chip_status: dict[int, str], chip_logs: dict[int, str], chips) -> list[dict]` (HUNG -> `hung: True`; ND -> `nd_report_devices(log, chips, dev_override=N)`)

- [ ] **Step 1: Fixtures.** Copy real text into `testdata/`:
  - `crash_triage_hang.csv` = `/data/mbezulj/galaxy-workload-deployment-test/LOG_2026_11_21_13_27_bh-glx-110-a06u02_734b76a2caa/matmul/minimal/crash_triage_01.csv`, cut to the `dump_op_mesh.py` and `dump_running_operations.py` sections.
  - ND lines: build from the f-strings in `tt_py_test_utils_common/op_test_base.py:144-171` (`Chip map device N: ...`, `ND report: ...`, `ND summary device N [...]: k/c checks; pattern ...; failing checks per core: (x,y)xN`, per-core lines).
  - run-to-run lines from `test_matmul_run_to_run_determinism.py:169-171`; core-sweep lines from `scratch_core_sweep.py:108-110`.
- [ ] **Step 2: Tests:**

```python
def test_triage_hang_devices_real_shape():
    d = triage_hang_devices(read("crash_triage_hang.csv"), chips32)
    assert [x["dev"] for x in d] == [27] and d[0]["tray"] == 3 and d[0]["asic"] == 4
    assert d[0]["hung"] is True and d[0]["op"] == "MinimalMatmulDeviceOperation" and d[0]["core_count"] == 120
def test_nd_report_top_cores_and_pattern(): ...   # failed_checks, checks, pattern, cores[:8] with translated
def test_nd_report_bdf_mismatch_warns(): ...      # Chip map bdf != snapshot bdf -> warning line, log's bdf kept
def test_nd_report_interval_zero_is_empty(): ...  # no ND lines -> ([], [])
def test_write_ack_nd_chip_uses_parent_chip_id(): # chip 5 log says "device 0" -> dev 5, hung False
def test_run_to_run_and_core_sweep_devices(): ...
```

- [ ] **Step 3:** Run. Expected: FAIL.
- [ ] **Step 4:** Implement; parse CSV sections with the `csv` module (cells are quoted and multi-line); split sections on the `<script>.py:` header lines.
- [ ] **Step 5:** Run. Expected: PASS.
- [ ] **Step 6:** Commit: `tests/didt: name failing chips from ND, run-to-run, sweep and triage output`

### Task 5: status.json (`runner/status.py`)

**Files:** Create `tests/didt/runner/status.py`; Test `tests/didt/runner/test_status.py`

**Interfaces:**
- Produces:
  - `@dataclass Attempt: name: str | None; env: dict; opts: list; status: Status; reason: str; details: list[str]; devices: list[dict]; log: str; start: float | None; end: float | None`
  - `@dataclass StepState: key: str; first: Attempt; fallbacks: list[Attempt]`
  - `write(path: Path, header: dict, steps: list[StepState], current: str | None) -> None` (temp file + `os.replace`)
- Top-level keys exactly as the spec: `tool host run mode commit run_dir config chips current steps`; step keys `key status reason details log start end devices fallbacks`; fallback adds `name env opts`.

- [ ] **Step 1: Tests:**

```python
def test_write_is_atomic(tmp_path, monkeypatch): # make json.dump raise midway -> old file still parses
def test_shape_matches_spec(): ...                # keys above, fallbacks[0]["name"] == "throttle"
def test_renders_with_sweep_page(): ...           # skip if build-gwdt-sweep sweep.py not found;
    # else run `python3 sweep.py render <scan.json with this status> labels.json out.html` -> rc 0
```

- [ ] **Step 2-4:** Run (FAIL), implement, run (PASS).
- [ ] **Step 5:** Commit: `tests/didt: didt_runner status.json writer`

### Task 6: Run loop and CLI (`runner/steps.py`, `didt_runner.py`)

**Files:** Create `tests/didt/runner/steps.py`, `tests/didt/didt_runner.py`; Test `tests/didt/runner/test_steps.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces:
  - `sh(cmd: str, log: Path, env: dict, cwd: Path, timeout: float | None) -> int` (124 on wall timeout, writes `TIMEOUT after N s` to the log)
  - `run(cfg: Config, run_dir: Path, chips: list[Chip], only: set[str] | None, skip: set[str], co: bool, shell=sh) -> int` (returns exit code)
  - `main(argv: list[str]) -> int` in `didt_runner.py`: `--config --only --skip --co --list --out`; `--help` text includes the sbatch note (`--propagate=NONE`, `sg slurmusers`)
- Per attempt (dir = `<key>` path or `<key>_<fallback>`): reset (`reset.log`, `reset.retryK.log`, up to `tries`) -> run (`log`) with triage env for `runner == "pytest"` -> `tt-smi.json` -> classify -> GDDR check. Triage env exactly as gwdt `triage_env()` (734b76a2caa `gwdt_steps.py:124-140`), CSV path `<dir>/crash_triage.csv`. Script runner: env `MCAST_ACK_WORK_DIR=<dir>/chips`, no triage env, delete `<dir>/chips/chip*/cache` after.
- `status.json` rewritten (Task 5 `write`) after every state change; `current` = running step key, null at the end. `run.log` gets one line per attempt start/end.
- Run dir: `<--out, default generated/didt_runner>/LOG_<local %Y_%m_%d_%H_%M>_<host>_<commit11>/`; commit from `git rev-parse --short=11 HEAD`.
- Classify: non-empty `crash_triage.csv` -> HANG + `triage_hang_devices`; rc 124 -> FAIL `timeout after N s`; else the step's check -> devices from Task 4.
- Exit: 0 all run steps PASS/WARN; 1 any first attempt FAIL/HANG; 2 `ConfigError`, unknown `--only/--skip` key, or snapshot error.

- [ ] **Step 1: Tests** with a fake `shell` that writes canned logs/CSVs per command (no device):

```python
def test_fail_runs_every_fallback_status_stays_first(): # first FAIL, throttle PASS -> step FAIL, fallbacks[0] PASS, exit 1
def test_pass_runs_no_fallback(): ...
def test_hang_reads_triage_csv(): ...            # devices[0]["dev"] == 27, hung True
def test_wall_timeout_is_fail(): ...             # shell returns 124, no CSV -> FAIL "timeout after"
def test_missing_id_fails_only_its_step(): ...   # collect log lacks nd:ff1's id -> ff1 FAIL "does not collect", others ran
def test_reset_fails_three_times(): ...          # step FAIL, reset.retry3.log exists, next step still runs
def test_fallback_folder_names(): ...            # nd/minimal/log and nd/minimal_throttle/log both exist
def test_script_runner_work_dir(): ...           # env MCAST_ACK_WORK_DIR == <dir>/chips
def test_cli_unknown_only_key_exit_2(): ...
def test_cli_list_prints_keys(): ...
```

- [ ] **Step 2-4:** Run (FAIL), implement, run (PASS). Then the full unit test command. Expected: all pass.
- [ ] **Step 5:** Commit: `tests/didt: didt_runner run loop and CLI`

### Task 7: Device check on bh-glx-110-c10u08

No new files. Host work must be watchable: named tmux session + log on `/data`.

- [ ] **Step 1:** Before touching the host: on slurm-login, `squeue -w bh-glx-110-c10u08` and `sinfo -N -n bh-glx-110-c10u08`. Expected: no other user's job. If another user holds it: stop and tell mbezulj.
- [ ] **Step 2:** `ssh bh-glx-110-c10u08.exabox.tenstorrent.com`; `cd /data/mbezulj/tt-metal`; `source python_env/bin/activate`; `python -c "import ttnn;print(ttnn.__file__)"`. Expected: path under `/data/mbezulj/tt-metal`. Else stop and tell mbezulj.
- [ ] **Step 3:** `python tests/didt/didt_runner.py --co --out /data/mbezulj/didt_runner_runs`. Expected: exit 0; `status.json` has 8 steps, 32 `chips` with tray/asic/asic_id.
- [ ] **Step 4:** Short real run with a copied config (`/data/mbezulj/didt_runner_runs/short.json`: `nd:ff1` iterations 50, `nd:mcast-ack` MATMUL_ITERS 2) in tmux `didt-c10u08`: `python tests/didt/didt_runner.py --config .../short.json --only nd:ff1,nd:mcast-ack --out /data/mbezulj/didt_runner_runs 2>&1 | tee /data/mbezulj/didt_runner_runs/short.log`. Expected: both steps finish; folders `nd/ff1/{reset.log,log,tt-smi.json}` and `nd/mcast-ack/chips/chip*/pytest.log`; no `/tmp/mm_mcast_write_ack_failure` created.
  - `Operation not permitted` on reset: rerun under `sg slurmusers -c '...'`.
- [ ] **Step 5:** Full shipped config in the same tmux, `tee` to `full.log`. Report per step: status, fallback status, `devices[]` (bdf/tray/asic). A HANG or FAIL here is a valid hardware result, not a runner bug, unless `status.json` lacks the chip identity.
- [ ] **Step 6:** Tell mbezulj: run dir path, the step table, tmux attach command. No commit in this task.
