# bring-up kit

Tools for taking a new model (target: MiMo-V2.6-Flash on a Blackhole Galaxy 4x8) from nothing to "done" with several autonomous agents in parallel
worktrees. Distilled from the DeepSeek-V4.1-Flash bring-up (`models/demos/blackhole/deepseek_v41_flash/tools`). Python 3 stdlib only; every tool has `--help`.
Tests (no hardware): `python_env/bin/python -m pytest models/common/bringup_kit/tests -q --noconftest -p no:cacheprovider`.

| tool | one line |
|---|---|
| `log_tool.py` | append-only `bringup/<model>/LOG.md`; `add` (clean tree, commit == branch tip, evidence exists), `check` (STALE/SUPERSEDED), `show --fresh-only` |
| `host_health.py` | pre-flight a host list: ssh, devices, stray python, device lock, load, disk, `bad_hosts.txt`; `--canary` computes on chips |
| `launch.py` | pin-the-commit detached launcher: clean tree + HEAD==hash, env scrub, GRID_ENV header, build-slot cap + stagger, flock, hangwatch |
| `hangwatch.py` | silence watchdog (>=45 min default, Engram 60): stacks, tt-triage, kill the whole tree, `tt-smi` reset rc logged |
| `results.py` | `extract` a results row from a demo log; `compare` two logs (cross-host = INVALID) |
| `scorecard.py` | data-driven gates -> PASS/FAIL/MISSING/STALE per gate from LOG entries; exit 1 unless all required gates PASS |
| `status_report.py` | one-screen coordinator view: jobs per host (age, CPU, silent minutes), build slots, recent log entries, branch tips |

## Workflow for a new model's agent
1. Own worktree + branch. `mkdir bringup/<model>`; `python scorecard.py template --model <m> --layer-types dense,moe,... > bringup/<model>/scorecard.json`
   and edit thresholds (gate list is data, never code).
2. Pick hosts: `host_health.py --all` (never use a host in `bad_hosts.txt`; add new bad hosts with the symptom).
3. Commit, then run: `launch.py --worktree $PWD --commit $(git rev-parse --short=11 HEAD) --host 45 --log <log> --env <MODEL>_SESSION=... -- pytest -x -s <demo>`.
   Use `--slot-prefix <MODEL>_` and copy DeepSeek `tt/build_slots.py` so the slot cap and stagger apply to full builds. Every log then carries a
   GRID_ENV header (host, commit, env) which `results.py` needs.
4. Read numbers with `results.py extract <log>`; compare runs only with `results.py compare` (same host, same commit, else flagged).
5. Record every fact: `log_tool.py add --log bringup/<m>/LOG.md --track prefill --claim "..." --evidence <log> --paths <dirs the claim depends on> --data '{"ttft_ms": 9358}'`.
   Use `--status inferred` for anything not measured at that commit.
6. **Stale-data rule: do not quote a number from a STALE, SUPERSEDED or inferred entry as a fact.** Run `log_tool.py check` / `show --fresh-only`
   before writing a report or deciding; re-measure and `add --supersedes <old id>`.
7. Done = `scorecard.py run --card ... --log ...` exits 0. The coordinator watches `status_report.py --all --logs bringup/*/LOG.md --branches ...`.

## Scorecard stages (decision: the CPU reference does not run the whole model)
Per-layer teacher-forced PCC (CPU oracle fed the captured real hidden state; one gate per layer type) -> short chained-prefix PCC curve (first 4-8
layers or one of each layer type) -> prefill-phase check (layer outputs + KV contents) -> decode-phase check (one step from a reference-prefilled
state) -> end-to-end acceptance with no CPU full-model run: **GSM8K (n, min correct) and determinism (two identical runs bit-identical) are REQUIRED**
(the only catch for compounding error beyond the CPU prefix), first-token sanity, memory fit table, prefill+decode perf recorded at the commit, and
optional hosted-oracle comparison (evidence = logged prompts/outputs file) and spec exactness. Each gate's `evidence` lists LOG ids whose `--data` JSON holds
the numbers (keys per gate kind: see `scorecard.py --help`). JSON cards work everywhere; `.yaml` works with PyYAML (or JSON content).
Example filled from the DeepSeek notes: `tests/fixtures/dsv41_scorecard.json` + `dsv41_LOG.md` (unknown evidence is MISSING; against the current repo the
old entries show STALE, which is the point).

## Formats
LOG.md entry: `## E<yymmdd>-<hhmm>-<3hex>` then `- key: value` lines: date, commit, branch, host, track, status (verified|inferred), claim, evidence, paths,
supersedes, data (JSON). Unique time-based ids avoid merge collisions between parallel branches; entries are never edited, only superseded.
`bad_hosts.txt`: `<host> <reason>` per line.

## Hardware status
Smoke-tested on real hosts: `launch.py` (ssh, flock, GRID_ENV, hangwatch kill + record, .47), `host_health.py`, `status_report.py` (all hosts, read-only).
NOT validated on a device: `canary.py` (`--canary`), `hangwatch.py` triage and `tt-smi` reset steps end to end (triage ran, reset only exercised with stubs).
Smoke recipe: `host_health.py <idle host> --canary --worktree <tree with working ttnn>`; hang: launch `-- python3 -c "import time; time.sleep(900)"` with `--stall-min 0.5`.
