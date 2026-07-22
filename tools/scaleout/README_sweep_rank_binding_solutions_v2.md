# `sweep_rank_binding_solutions.py` — generate-and-sweep every topology solution

> **One doc, two maturities.** Each capability is tagged **✅ implemented** (in tree today) or
> **🔜 planned** (the async-interleaved redesign). Read top-to-bottom as the target tool; the tags say
> what already works.

- **Tracking:** epic [#49514](https://github.com/tenstorrent/tt-metal/issues/49514) ·
  ticket [#49515](https://github.com/tenstorrent/tt-metal/issues/49515) · feeds
  [#49517](https://github.com/tenstorrent/tt-metal/issues/49517) (per-solution validation) &
  [#49519](https://github.com/tenstorrent/tt-metal/issues/49519) (DC bring-up wrapper).
- **Related:** [`README_generate_rank_bindings.md`](README_generate_rank_bindings.md) ·
  [`../../ttnn/ttnn/distributed/README_ttrun.md`](../../ttnn/ttnn/distributed/README_ttrun.md).

---

## Motivation — it's `tt-run`, but across every placement

*`tt-run` = one workload on one placement.** You hand it a single rank binding (one assignment of
   MPI ranks → hosts/devices); it runs your command under MPI on exactly those hosts:

   ```bash
   tt-run --rank-binding solution/rank_bindings.yaml -- ./my_test --gtest_filter=...   # one binding in, one run out
   ```

**But one MGD + host set has *many* valid placements.** `generate_rank_bindings --all-solutions`
   finds all of them (different host sets / rank assignments that satisfy the topology). `tt-run` has
   no "try them all" — you'd run it by hand, once per binding.

### Goal
**This tool = `tt-run` applied across every placement.** Give it an MGD + hosts; it enumerates all
   solutions and runs your workload once per solution — the *same* `tt-run` each time, different
   binding — then reports which pass:

  ```
                          one workload command  (e.g. ./my_test --gtest_filter=…)
                                       │
     generate_rank_bindings            │   the sweep runs, per solution:
     --all-solutions  ─────────▶       ├── #1  tt-run --rank-binding sol_1/rank_bindings.yaml -- <workload>  → PASS
     (enumerates every valid           ├── #2  tt-run --rank-binding sol_2/rank_bindings.yaml -- <workload>  → PASS
      placement: sol_1 … sol_N)        ├── #3  tt-run --rank-binding sol_3/rank_bindings.yaml -- <workload>  → FAIL
                                       └── #N  tt-run --rank-binding sol_N/rank_bindings.yaml -- <workload>  → …
                                       │
                              sweep_report.yaml  ← which placements passed / failed
  ```

- **Net:** run the sweep once → it does the enumeration + the N launches + result collection. Exactly
  what DC bring-up needs (run a validation command on *every* valid placement, report which work).
- **Stays a separate tool** (not folded into `tt-run`/`generate_rank_bindings`): it's an orchestrator
  over N launches with its own generation + reporting — and each launch still shells out to the **real
  `tt-run`**, so launch semantics are identical to running it yourself.
- **Key enabler (✅):** `generate_rank_bindings --all-solutions` now **streams** (writes each solution's
  dir + rewrites `solutions_index.yaml` the instant it's found; dir `fsync`'d before listed) and is
  **unbounded** (stops only at genuine exhaustion). → the sweep can test solution 1 *while generation
  is still producing* 2, 3, ….

---

## How it works — overview & example  🔜

What happens when you run it:

1. Run the sweep **once** (MGD + hosts + a workload command).
2. It launches generation in the **background** and starts streaming out solutions.
3. The moment solution #1 is ready, it runs your workload on it — **one workload at a time** (never
   two — avoids two runs fighting over the same devices).
4. While that workload runs, **generation races ahead** producing the next solutions for free.
5. When a workload finishes, the next queued solution starts; **results stream in** as they complete.
6. Each solution ends up in a **self-contained folder** you can re-run standalone.

**Example** — SC36 mock, up to 4 solutions, a fabric unit test as the workload:

```bash
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  -m .../supercluster_20.textproto \
  --mock-cluster-rank-binding .../SC36_32x4_revC_subtorus_aisleD_mapping.yaml \
  --max-solutions 4 \
  -- ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*PipelineBuilder*"
```

Timeline (workload ≈ 40 s each; first solution pays the one-time ~90 s minimal-host "prime"):

| time  | producer (generation, background)                     | consumer (workload, one at a time)          |
|-------|-------------------------------------------------------|---------------------------------------------|
| 0 s   | starts; minimizing host count (~90 s prime)           | idle — waiting for solution #1              |
| 94 s  | **#1** written (20 hosts)                             | sees #1 → **starts workload #1**            |
| 103 s | **#2** written (ahead, while #1 tests)                | #1 running                                  |
| 112 s | **#3** written                                        | #1 running                                  |
| 121 s | **#4** written; hit `--max-solutions 4` → **exits**   | #1 running                                  |
| 134 s | — (done)                                              | #1 → **PASS** → starts #2                   |
| 174 s | —                                                     | #2 → PASS → starts #3                       |
| 214 s | —                                                     | #3 → PASS → starts #4                       |
| 254 s | —                                                     | #4 → **FAIL** (rc=1) → nothing left → done  |

- **End state:** 4 solution folders (each: `rank_bindings.yaml`, `rankfile`, `run.sh`, `workload.log`,
  `result.yaml`) + `sweep/sweep_report.yaml` = `3/4 passed`; exit code ≠ 0 (so CI gates on it).
- **Re-run the failure:** `bash <out>/sweep/03765a…/run.sh`.
- **Why interleave matters:** generating #2–#4 (94→121 s) happened *under* workload #1 → **zero** extra
  wall-time. Serial would generate all 4 first, *then* test; here testing starts at 94 s.

> v1 today (✅) = the synchronous special case: generate all, then test each in sequence. `--no-interleave` keeps it.

---

## Example usage

```bash
# A) Generate + interleaved-sweep across every solution (mock cluster)
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  -m models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_supercluster_20.textproto \
  --mock-cluster-rank-binding .../SC36_..._mapping.yaml \
  --max-solutions 8 --solutions-output-dir generated/ttrun/sc36_sweep \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  -- ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*PipelineBuilder*"

# B) Real cluster: first 3 solutions, dry-run first (writes run.sh only)
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  -m <mgd> --hosts nodeA,nodeB,nodeC,nodeD --limit 3 --dry-run -- ./my_app

# C) Reproduce a single solution standalone — just run its script
bash generated/ttrun/sc36_sweep/f44380f2b9ec93b8/run.sh
```

---

## Directory & log layout — one self-contained folder per solution

- One solution ⇒ one folder holding **everything**: bindings, log, result, and a runnable `run.sh`.
- Hand someone a single `<hash>/` dir → they can re-run it standalone.

```
<solutions-output-dir>/                    # default generated/ttrun/sweep/ ; override with --solutions-output-dir
├── solutions_index.yaml                   # ✅ PRODUCER, rewritten per solution — source of truth / handoff
├── <hash>/                                # ONE self-contained folder per solution (id = content hash)
│   ├── rank_bindings.yaml                  #  ✅ PRODUCER: this solution's bindings
│   ├── rankfile                            #  ✅ PRODUCER: rankfile (real-cluster placement)
│   ├── solution_meta.yaml  .solution_key   #  ✅ PRODUCER: metadata + full canonical signature
│   ├── [phase2_mock_mapping.yaml]          #  ✅ PRODUCER: per-rank mock descriptors (mock mode only)
│   ├── run.sh                              #  🔜 CONSUMER: self-contained reproducer — env + exact tt-run command
│   ├── workload.log                        #  🔜 CONSUMER: full stdout+stderr of THIS solution's run
│   └── result.yaml                         #  🔜 CONSUMER: outcome (status/rc/duration/timestamps/cmd)
└── sweep/
    ├── generate.log                        # 🔜 PRODUCER full stdout+stderr (the enumeration)
    ├── sweep_report.yaml                   # ✅ schema / 🔜 co-located + incremental: aggregate of all result.yaml
    └── sweep_status.yaml                   # 🔜 live counters (generated / dispatched / running / passed / failed)
```

- `generate.log` — producer's entire output; a slow/failed enumeration is inspectable live.
- `sweep_report.yaml` — aggregate of every `result.yaml`, rewritten after each workload → crash/timeout
  leaves a valid partial report. Results in **index order** (stable) + per-result `started_at`/`finished_at`.
- `sweep_status.yaml` — tiny heartbeat (`{generated, dispatched, running, passed, failed, producer_alive}`)
  for a watcher/CI. (Proposed — drop if report + console suffice.)

---

## Per-solution `run.sh` — the exact reproducer *and* the launcher  🔜

- The consumer **writes `<hash>/run.sh`, then executes it** (`bash run.sh > workload.log 2>&1`).
- ⇒ the reproducer is **byte-identical** to what was tested — no "documented command" that can drift.

```bash
#!/usr/bin/env bash
# Reproduce the sweep run for solution <hash> exactly. Generated by sweep_rank_binding_solutions.py.
set -euo pipefail

# --- Environment captured at sweep time ---
export TT_METAL_HOME="/data/.../tt-metal2"
export ARCH_NAME="blackhole"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/tt_metal:$TT_METAL_HOME/build_Release/lib:${LD_LIBRARY_PATH:-}"
# ... every TT_* var, verbatim (TT_TOPOLOGY_SOLVER_ENGINE, TT_METAL_*, ...) ...

cd "$TT_METAL_HOME"

# --- Exact tt-run launch, ABSOLUTE paths to THIS solution's artifacts ---
exec tt-run \
  --rank-binding "$TT_METAL_HOME/generated/ttrun/sweep/<hash>/rank_bindings.yaml" \
  --mock-cluster-rank-binding "$TT_METAL_HOME/generated/ttrun/sweep/<hash>/phase2_mock_mapping.yaml" \  # mock mode
  --mpi-args "--map-by rankfile:file=$TT_METAL_HOME/generated/ttrun/sweep/<hash>/rankfile" \             # real cluster
  <forwarded tt-run passthrough args...> \
  -- <program> <args...>
```

Generation rules:

- **Env** — snapshot the sweep's environment (`TT_METAL_HOME`, `ARCH_NAME`, `LD_LIBRARY_PATH`,
  `PYTHONPATH`, **and every `TT_*`**) as `export VAR=<shell-quoted>`. (Allowlist rather than a full
  `env` dump — avoids leaking unrelated/secret vars into a committed artifact.)
- **Bindings** — always **absolute** paths into this solution's own dir; works from any CWD, always uses
  *this* placement.
- **Launch** — the exact `tt-run` invocation (mock vs real branch, all `--mpi-args` + passthrough),
  ending in `-- <program>`; `exec` so the script's exit code is the workload's.
- **`--dry-run`** — write every `run.sh` but execute none (inspect/hand-edit first).

---

## CLI

```
sweep_rank_binding_solutions.py [OPTIONS] -- <program> [args...]     # everything after -- runs once per solution
```

| Option | Status | Description |
|--------|--------|-------------|
| `--mesh-graph-descriptor / -m PATH` | ✅ | MGD to solve (required — the sweep always generates). |
| `--hosts a,b,c` | ✅ | Real-cluster host list, forwarded to `generate_rank_bindings`. |
| `--mock-cluster-rank-binding PATH` | ✅ | Mock per-rank mapping YAML (mock cluster). |
| `--solutions-output-dir PATH` | ✅ | Where solutions + sweep artifacts go (default `generated/ttrun/sweep`). |
| `--max-solutions N` | ✅ | Forwarded (`0` = all). Bounds how many the producer makes. |
| `--distinct-host-sets` | ✅ | Forwarded: one solution per unique host set. |
| `--allow-shape-permutations` | ✅ (hidden) | Forwarded: turn OFF the solver's always-on `unique_shapes` dedup. |
| `--limit N` | ✅ | Sweep at most the first N solutions. |
| `--per-solution-timeout SECONDS` | ✅ | Kill a launch exceeding this; record `timeout`. |
| `--continue-on-failure / --stop-on-failure` | ✅ | Keep sweeping after a failure (default continue). |
| `--mpi-args "…"` / `--rankfile-syntax` / `--tcp-interface` / `--bare` / `--tracy` / `-v` / … | ✅ | Forwarded verbatim to each `tt-run`. |
| `--dry-run` | ✅ | Write each `run.sh` / print each command; launch nothing. |
| `--interleave / --no-interleave` | 🔜 | Async single-producer/single-consumer (default) vs blocking two-phase. |

---

## `tt-run` parts reused  ✅

- The script **imports** from `ttnn.distributed.ttrun` (doesn't reimplement MPI/mock launch):

  | ttrun symbol | Use |
  |--------------|-----|
  | `find_generate_rank_bindings_executable()` | Locate the binary for the producer. |
  | `build_generate_rank_bindings_mpi_cmd(...)` | Build the generate MPI command (`--all-solutions`, `--max-solutions`, …). |
  | `load_mock_rank_to_descriptors(...)` | Mock per-rank descriptor mapping for the producer. |
  | `get_generate_rank_bindings_output_paths(dir)` | Resolve `(rank_bindings.yaml, rankfile)` in each solution dir. |

- **Per-solution launch shells out to the real `tt-run` CLI** (legacy `--rank-binding` mode) → identical
  launch semantics, no coupling to tt-run internals. In v2 that command is materialized as the
  solution's `run.sh` and executed.

---

## `sweep_report.yaml`  ✅ (schema) / 🔜 (co-located + incremental)

```yaml
mesh_graph_desc_path: .../supercluster_20.textproto
workload_command: "./build/test/.../fabric_unit_tests --gtest_filter=..."
enumeration: { mode: distinct-host-sets, max_solutions: 8, found: 8, truncated: true }
summary: { total: 8, passed: 7, failed: 1, timed_out: 0 }
results:
  - solution_id: f44380f2b9ec93b8
    status: pass            # pass | fail | timeout | dry-run
    returncode: 0
    duration_seconds: 42.1
    num_hosts: 20
    host_set: [bh-glx-120-d01u02, bh-glx-120-d01u08, ...]
    run_script: .../sweep/f44380f2b9ec93b8/run.sh        # 🔜 the exact reproducer
    log_path:   .../sweep/f44380f2b9ec93b8/workload.log
    rank_binding_path: .../sweep/f44380f2b9ec93b8/rank_bindings.yaml
    started_at: 2026-07-22T21:27:01Z                     # 🔜
    finished_at: 2026-07-22T21:27:43Z                    # 🔜
  - solution_id: 03765a03004d0370
    status: fail
    returncode: 1
    ...
```

- Each result carries the exact reproducer (`run_script` v2 / `tt_run_command` v1) → re-run a failure standalone.
- **No parsed failure-reason field** — the full per-solution log is the source of truth (log-scraping too brittle).
- Exit code ≠ 0 if any solution failed → CI can gate.

---

## Logging — verbose & self-describing  🔜

Three surfaces, each readable on its own (no cross-referencing needed):

### 1. Console — interleaved, every line tagged by stream + relative time

```
[tt-sweep] ┌ sweep start 2026-07-22 21:25:31
[tt-sweep] │ MGD          : supercluster_20.textproto
[tt-sweep] │ cluster      : mock (SC36_32x4_revC_subtorus_aisleD, 36 hosts)   [or: real, hosts=nodeA,nodeB,…]
[tt-sweep] │ enumerate    : --all-solutions --distinct-host-sets --max-solutions 12
[tt-sweep] │ workload     : ./build/test/.../fabric_unit_tests --gtest_filter=ControlPlaneFixture.*PipelineBuilder*
[tt-sweep] │ output dir   : generated/ttrun/sweep
[tt-sweep] │ concurrency  : 1 producer, 1 consumer (one workload at a time)
[tt-sweep] └ producer pid 40021  →  log generated/ttrun/sweep/sweep/generate.log

[tt-sweep][gen +094s] found #1  f44380f2b9ec93b8   hosts=20/36
[tt-sweep][gen +094s]            host_set: d01u02,d01u08,d01u14,d01u20,d02u02,d02u08,d03u02,d03u08,d04u02,d04u08 …(+10)
[tt-sweep][run +094s] ▶ START #1  f44380f2b9ec93b8  (20 hosts)
[tt-sweep][run +094s]            script : generated/ttrun/sweep/f44380f2b9ec93b8/run.sh
[tt-sweep][run +094s]            log    : generated/ttrun/sweep/f44380f2b9ec93b8/workload.log
[tt-sweep][run +094s]            cmd    : tt-run --rank-binding …/f44380…/rank_bindings.yaml --mock-cluster-rank-binding …/phase2_mock_mapping.yaml -- ./…/fabric_unit_tests --gtest_filter=…
[tt-sweep][gen +103s] found #2  2e1dcd535f46c42   hosts=20/36     (producer ahead — found 2, tested 0, queued 1)
[tt-sweep][run +136s] ✔ PASS  #1  f44380f2b9ec93b8   rc=0   42.1s
[tt-sweep][run +136s] ▶ START #2  2e1dcd535f46c42  (20 hosts)     (queue: 0 waiting; producer still running)
…
[tt-sweep][run +298s] ✘ FAIL  #7  03765a03004d0370   rc=1   39.7s   → see workload.log ; reproduce: bash …/03765a…/run.sh
[tt-sweep][gen +384s] producer done — 12 distinct solutions in 384s (truncated=false)
[tt-sweep] ┌ SUMMARY  11/12 passed · 1 failed · 0 timed out
[tt-sweep] │ FAILED   #7  03765a03004d0370  rc=1   log=…/03765a…/workload.log   repro=bash …/03765a…/run.sh
[tt-sweep] └ report   generated/ttrun/sweep/sweep/sweep_report.yaml
```

- `gen` = producer indexed a new solution → prints id, host count, host-set preview.
- `run` = the single consumer's workload → `START` (script/log/cmd), then `PASS`/`FAIL`/`TIMEOUT` (rc + duration).
- **Backlog indicator** (`found X, tested Y, queued Z`) makes producer-ahead-of-consumer obvious.
- Only ever **one `run`** in flight.

### 2. `sweep/generate.log` — the producer's own output, made verbose

```
[gen] enumerate: SC20 (80 meshes) into SC36 aisle-D fabric — 36 host groups, k_min = ceil(80/4) = 20
[gen] prime: minimizing host count …  warm=27 occupied → descent 26 → 24 → floor 20
[gen] prime: LOCKED at 20 hosts in 92.3s (full-packing hardlock); cap asserted ≤20 occupied
[gen] solution #1   id=f44380f2b9ec93b8   hosts=20/36   +1.8s   (total 94.1s)
[gen]    host_set: d01u02 d01u08 d01u14 d01u20 d02u02 d02u08 d03u02 d03u08 d04u02 d04u08
[gen]              d05u02 d05u08 d06u14 d07u02 d07u08 d07u14 d08u02 d08u08 d09u02 d09u08
[gen] solution #2   id=2e1dcd535f46c42   hosts=20/36   +9.1s   (total 103.2s)
[gen]    host_set: …
…
[gen] EXHAUSTED: 12 distinct solutions, 384s total, truncated=false (proved no more exist)
```

- Each solution line: **content-hash id** (= folder + index entry), **host count** (`used/available`),
  **incremental** + **cumulative** time, then the **full host set** (wrapped).
- The **prime** is called out explicitly → a long first-solution wait is never mistaken for a hang.
- Implementation: extend the existing `Wrote solution <id> (<n> hosts) [<k> so far]` line.

### 3. `<hash>/workload.log` — self-describing per-solution log

- Begins with a **banner** (written before exec'ing `run.sh`) → the file stands alone.

```
# ===== tt-sweep solution f44380f2b9ec93b8 =====
# hosts (20): d01u02 d01u08 d01u14 d01u20 d02u02 … d09u08
# rank_bindings: …/f44380f2b9ec93b8/rank_bindings.yaml
# rankfile:      …/f44380f2b9ec93b8/rankfile
# reproduce:     bash …/f44380f2b9ec93b8/run.sh
# command:       tt-run --rank-binding … -- ./…/fabric_unit_tests --gtest_filter=…
# started:       2026-07-22 21:27:01
# ================================================
<the workload's own stdout/stderr follows>
```

> v1 (✅): a simpler sequential `[i/N] solution … -> status` + a bare log (no banner). The above is the v2 target.

---

## Implementation notes  🔜

- **Two subprocesses**, coordinated through the on-disk `solutions_index.yaml`. No thread pool, no queue.

```
                          background process (never blocks the sweep)
      MGD + hosts ─▶ generate_rank_bindings --all-solutions [-n N]        [ PRODUCER ]
                          │  streams, unbounded, runs AHEAD up to N
                          ▼
                 solutions_index.yaml   ◀── rewritten per solution (the handoff / source of truth)
                          │  main thread polls (defensive read) + diffs against dispatched ids
        ┌─────────────────┴──────────────────── single CONSUMER ─────────────────────────────┐
        │  ONE workload at a time: write <hash>/run.sh → bash run.sh > workload.log (Popen);   │
        │  while it runs, keep polling the producer/index; when it exits, record result and    │
        │  launch the next queued solution.                                                    │
        └────────────────────────────────────────────────────────────────────────────────────┘
                          │  each result.yaml lands → rewrite sweep/sweep_report.yaml
                          ▼
              aggregate report + console + non-zero exit if any failed
```

- **Producer:** background `Popen`, stdout/stderr → `generate.log`. Runs ahead up to `N`; disk buffers backlog.
- **Consumer loop (main thread):**
  1. read the index **defensively** (tolerate a mid-rewrite read; retry next tick),
  2. **diff** its ids against already-dispatched,
  3. if no workload running → take the next new id, write `run.sh`, launch `bash run.sh > workload.log` (Popen),
  4. on workload exit → write `result.yaml` + rewrite `sweep_report.yaml`.
- **Race-freedom:** producer `fsync`s a dir **before** listing its id → "new id in index ⇒ dir complete ⇒ safe."
- **Termination:** producer exited **and** all eligible ids dispatched **and** current workload done
  (index re-read once after producer exits to catch the last few).

---

## Testing plan

- **Unit (no MPI):** index polling + new-id diffing, `--limit`, `run.sh` generation (env snapshot +
  absolute paths + mock/real branch), report aggregation, exit-code aggregation, `--dry-run` (writes
  `run.sh`, launches nothing) — against a fabricated index + fake solution dirs, `true` workload.
  (✅ v1 pieces; 🔜 producer/consumer + run.sh.)
- **Mock end-to-end (CPU):** generate a small set on the SC20 mock, sweep `true` interleaved, assert
  every produced solution got `run.sh` + `workload.log` + `result.yaml` and `summary.passed == found`.
  Reuses `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`.

---

## Open questions (meta)

1. **Entry-point name** — `tt-sweep-solutions`, or script-only under `tools/scaleout/`?
2. **Report format** — YAML (above). JSON too?
3. **Generate coupling** — once `tt-run` forwards `--all-solutions`, shell out to `tt-run` instead of
   `generate_rank_bindings` directly?
