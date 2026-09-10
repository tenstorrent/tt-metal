# Plan: `sweep_rank_binding_solutions.py` — run a command across every generated solution

Status: **implemented**. `tools/scaleout/sweep_rank_binding_solutions.py` is in the
tree and was validated end-to-end on the SC20 mock (5 solutions generated, swept a
fabric unit test across them, `sweep_report.yaml` produced, `2/2 passed`).

> **New-mode only.** The sweep drives off `--mesh-graph-descriptor` + `--hosts`
> (real cluster) or `--mock-cluster-rank-binding` (mock), or an existing
> `--solutions-dir`. tt-run's legacy `--rank-binding` (a single explicit binding)
> is intentionally **not** exposed — a sweep needs an MGD to enumerate solutions.

Tracking: epic [#49514](https://github.com/tenstorrent/tt-metal/issues/49514) ·
ticket [#49515](https://github.com/tenstorrent/tt-metal/issues/49515) (topology
solver sweep infrastructure); feeds the per-solution validation in
[#49517](https://github.com/tenstorrent/tt-metal/issues/49517) and the DC bring-up
wrapper in [#49519](https://github.com/tenstorrent/tt-metal/issues/49519).

Related:
[`README_generate_rank_bindings_all_solutions.md`](README_generate_rank_bindings_all_solutions.md) ·
[`../../ttnn/ttnn/distributed/README_ttrun.md`](../../ttnn/ttnn/distributed/README_ttrun.md).

The tt-blaze `tools/pipeline_sweep` wrapper drives this script for full-cluster passthrough sweeps
(e.g. SC36); see its `README.md` for the end-to-end build + run recipe.

---

## Recovery + retry (`--recover-command`)

`--recover-command` is a **required** string: an arbitrary recovery command run (via `bash -c`)
after a solution's tt-run fails or times out, after which the *same* tt-run is retried. Use it to
reset the machine/cluster between attempts. On a scaleout cluster this is typically the exabox
recovery script:

```bash
--recover-command "bash ${TT_METAL_HOME}/tools/scaleout/exabox/recover.sh --hosts $HOSTS --mpi-if ens5f0np0 --config 4x32"
```

Semantics:

- Both tt-run and the recover command are each retried `DEFAULT_RETRIES` times (hidden `--retries`),
  with `RETRY_DELAY_S` between attempts.
- A later tt-run pass is recorded as **pass**; a solution is fail/timeout only if every tt-run
  attempt fails. The sweep then **continues** to the next solution (unless `--stop-on-failure`).
- Recovery is run **before** the first tt-run (proactive reset) and again after any fail/timeout.
  It is **not** run on pass, but is still run after the final failure so the machine is left clean.
- Exhausting the recover retries is treated as an **unrecoverable hardware error**: the sweep aborts
  regardless of `--stop-on-failure` / `--continue-on-failure`.

The tt-blaze `tools/pipeline_sweep` wrapper exposes this as its own `--recover-command` and passes
the string **through verbatim**; `--no-reset` there sends `true` (a no-op) so this required flag is
still satisfied without touching the cluster.

---

## Motivation

`generate_rank_bindings --all-solutions` produces **many** valid placements under
`generated/ttrun/<cache_id>/`:

```
<solutions_dir>/
  solutions_index.yaml
  <solution_hash>/  rank_bindings.yaml  rankfile  solution_meta.yaml  .solution_key  [phase2_mock_mapping.yaml]
  <solution_hash>/  ...
```

Today there is no way to **exercise** them. The DC bring-up flow wants to: take a
host list (or an existing solutions dir), and for **every** solution, launch a
validation command (a PipelineBuilder passthrough test, a blitz-decode smoke test,
`true`, or any app) with that solution's rank bindings, then report which
placements pass. That is this script.

It is a **separate** tool (not folded into `tt-run` or `generate_rank_bindings`)
because it is an *orchestrator over N tt-run launches*, with its own
result-collection and reporting concerns.

---

## Where it lives

`tools/scaleout/sweep_rank_binding_solutions.py` (next to
`generate_rank_bindings` and the exabox scripts). Installed/entry-point name TBD
(e.g. `tt-sweep-solutions`); initially run as `python3 tools/scaleout/sweep_rank_binding_solutions.py`.

---

## What it does (overview)

```
                 ┌─ (optional) Phase 0: generate solutions ─┐
host list / MGD ─┤  run generate_rank_bindings --all-solutions │
                 └──────────────────────────────────────────┘
                                   │
                       solutions_index.yaml
                                   │
        ┌──────────────── for each solution ────────────────┐
        │  launch <command> via tt-run legacy mode using     │
        │  <solution>/rank_bindings.yaml + rankfile          │  (sequential)
        │  (+ phase2_mock_mapping.yaml in mock mode)          │
        │  capture exit code, stdout/stderr, duration         │
        └────────────────────────────────────────────────────┘
                                   │
                    sweep_report.yaml + console table
```

One user command is run **once per solution**. Each run reuses that solution's
already-generated bindings — no re-solve.

---

## CLI

```
sweep_rank_binding_solutions.py [OPTIONS] -- <command> [args...]
```

| Option | Description |
|--------|-------------|
| `--solutions-dir PATH` | Directory containing `solutions_index.yaml` (output of `generate_rank_bindings --all-solutions`). Mutually exclusive with the generate options below. |
| `--mesh-graph-descriptor / -m PATH` | (generate mode) MGD to solve. Triggers Phase 0. |
| `--hosts a,b,c` | (generate mode, real cluster) host list passed to `generate_rank_bindings`. |
| `--mock-cluster-rank-binding PATH` | (generate mode, mock) per-rank mock mapping YAML. |
| `--max-solutions N` | (generate mode) forwarded to `generate_rank_bindings`. |
| `--distinct-host-sets` | (generate mode) forwarded to `generate_rank_bindings`. |
| `--solutions-output-dir PATH` | (generate mode) where solutions are written (default `generated/ttrun/sweep`). |
| `--select id[,id...]` | Only sweep these solution ids (default: all in the index). |
| `--limit N` | Sweep at most the first N solutions (by index order / solver preference). |
| `--mpi-args "..."` | Extra MPI args forwarded to each launch (e.g. `--allow-run-as-root --oversubscribe`). |
| `--rankfile-syntax auto\|ompi\|prrte` | Forwarded to tt-run rankfile handling. |
| `--tcp-interface IFACE` | Forwarded (multi-host TCP exclusions). |
| `--per-solution-timeout SECONDS` | Kill a launch that exceeds this; record as `timeout`. |
| `--continue-on-failure / --stop-on-failure` | Whether to keep sweeping after a failing solution (default: continue). |
| `--sweep-report PATH` | Where to write `sweep_report.yaml` (default `<solutions-dir>/sweep_report.yaml`). |
| `--dry-run` | Print the tt-run command per solution; do not launch. |
| `-- <command>` | Everything after `--` is the command to run per solution (e.g. `./build/test/... --gtest_filter=...`, or `true`). |

The command may use placeholders expanded per solution:
`{solution_id}`, `{solution_dir}`, `{rank_bindings}`, `{rankfile}`.
If none are present, the command is launched as-is under that solution's bindings.

---

## `ttrun` parts reused

The script **imports** from `ttnn.ttnn.distributed.ttrun` rather than
reimplementing MPI launch (this is the "uses parts of tt-run" requirement). Key
symbols (all present today):

| ttrun symbol | Use in the sweep |
|--------------|------------------|
| `get_generate_rank_bindings_output_paths(dir)` | Resolve `(rank_bindings.yaml, rankfile)` inside each solution dir. |
| `find_generate_rank_bindings_executable()` | Locate the binary for Phase 0. |
| `build_generate_rank_bindings_mpi_cmd(...)` / `run_phase1_generate_rank_bindings(...)` | Phase 0 generation (add `--all-solutions` to the arg list). |
| `TTRunConfig` + its loader | Parse a solution's `rank_bindings.yaml`. |
| `build_mpi_command(config, program, mpi_args, rankfile_syntax, ...)` | Build the per-solution `mpirun` command — the core reuse. |
| `get_mpi_launcher()`, `detect_rankfile_syntax()`, `build_rankfile_args()`, `normalize_rankfile_mpi_args()`, `inject_rankfile_mpi_args()`, `rankfile_needs_oversubscribe()` | Correct rankfile/oversubscribe handling per solution. |
| `build_phase2_mock_mapping(...)`, `load_mock_rank_to_descriptors(...)`, `parse_rank_bindings_mapping(...)` | Mock-mode Phase-2 routing from `phase2_mock_mapping.yaml`. |
| `default_multihost_mpi_args(tcp_interface)` | Sensible multi-host defaults. |

**Execution strategy (as implemented):** Phase 1 reuses the generate helpers
directly (imported). Per-solution Phase 2 **shells out to the `tt-run` CLI in
legacy mode** — `tt-run --rank-binding <sol>/rank_bindings.yaml
[--mock-cluster-rank-binding <sol>/phase2_mock_mapping.yaml |
--mpi-args "... --map-by rankfile:file=<sol>/rankfile"] -- <command>` — forwarding
the user's tt-run passthrough args. Shelling out to the real CLI is the most robust
reuse (identical launch semantics, no coupling to tt-run internals); `--dry-run`
prints each fully-built `tt-run` command.

---

## Algorithm

```python
# 1. Obtain solutions
if args.solutions_dir:
    index = load_yaml(args.solutions_dir / "solutions_index.yaml")
else:
    # Phase 0: generate
    cmd = build_generate_rank_bindings_mpi_cmd(..., extra=["--all-solutions",
              *(["--max-solutions", str(args.max_solutions)] if args.max_solutions else []),
              *(["--distinct-host-sets"] if args.distinct_host_sets else [])])
    run_phase1_generate_rank_bindings(cmd, cwd=repo_root)   # reused
    index = load_yaml(args.output_dir / "solutions_index.yaml")

solutions = filter(index["solutions"], args.select, args.limit)

# 2. Sweep
results = []
for sol in solutions:
    sol_dir = solutions_dir / sol["dir"]
    rank_bindings, rankfile = get_generate_rank_bindings_output_paths(sol_dir)   # reused
    config = load_ttrun_config(rank_bindings)                                    # reused
    program = expand_placeholders(user_command, sol, sol_dir, rank_bindings, rankfile)
    mpi_args = merge(args.mpi_args, rankfile_for(rankfile, sol_dir))
    argv = build_mpi_command(config, program, mpi_args, rankfile_syntax=...)     # reused
    if args.dry_run:
        print(argv); continue
    rc, dur, log_path = run_capture(argv, timeout=args.per_solution_timeout,
                                    log=sol_dir / "sweep_run.log")
    results.append({"id": sol["id"], "num_hosts": sol["num_hosts"],
                    "host_set": sol["host_set"], "returncode": rc,
                    "status": classify(rc), "duration_s": dur, "log": str(log_path)})
    if rc != 0 and args.stop_on_failure:
        break

# 3. Report
write_sweep_report(args.report, index_meta=index["enumeration"], results=results)
print_table(results)
sys.exit(0 if all(r["status"] == "pass" for r in results) else 1)
```

Mock mode: when a solution dir contains `phase2_mock_mapping.yaml`, the per-rank
mock descriptors are wired via `build_phase2_mock_mapping` / the MPMD segments
exactly as `tt-run` does in mock mode (per-rank `-x TT_METAL_MOCK_CLUSTER_DESC_PATH`).

---

## Output: `sweep_report.yaml`

```yaml
mesh_graph_desc_path: .../9_quad_bh_galaxy_9x32x4_torus_graph_descriptor.textproto
command: ["./build/test/tt_metal/tt_fabric/fabric_unit_tests", "--gtest_filter=..."]
enumeration: { mode: distinct-host-sets, max_solutions: 5, found: 5, truncated: true }
summary: { total: 5, passed: 4, failed: 1, timed_out: 0, skipped: 0 }
results:
  - id: 44f80c400ca963a4
    status: pass          # pass | fail | timeout | dry-run
    returncode: 0
    duration_s: 42.1
    num_hosts: 4
    host_set: [bh-glx-...c01u02, bh-glx-...c01u08, bh-glx-...c02u02, bh-glx-...c02u08]
    tt_run_command: "tt-run --rank-binding .../rank_bindings.yaml --mock-cluster-rank-binding ... -- <program>"
    log: <log-dir>/44f80c400ca963a4.log
  - id: 03765a03004d0370
    status: fail
    returncode: 1
    duration_s: 39.7
    ...
```

Each result carries `tt_run_command` — the exact, copy-paste-reproducible `tt-run` invocation for that
solution — so a failing solution can be re-run standalone. (Deliberately no parsed failure-reason field:
the full per-solution log is the source of truth; log-scraping was too brittle to be trustworthy.)

Console prints per-solution progress + result lines and a final
`PASSED n/N`. Exit code is non-zero if any solution failed (so CI can gate on it).

---

## Concurrency

Default **sequential**: on real hardware the solutions overlap on physical
devices, so running them concurrently would contend for the same ASICs. A
`--jobs N` flag is deliberately **out of scope for v1**; a future version could
allow parallelism only when solutions have disjoint `host_set`s (derivable from
the index) — the report already carries `host_set` per solution to support this.

---


## Example usage

```bash
# A) Generate then sweep a passthrough test across every solution (mock cluster)
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  -m tt_metal/fabric/mesh_graph_descriptors/9_quad_bh_galaxy_9x32x4_torus_graph_descriptor.textproto \
  --mock-cluster-rank-binding .../SC36_..._mapping.yaml \
  --max-solutions 8 --output-dir generated/ttrun/sc36_sweep \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  -- ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*PipelineBuilder*"

# B) Sweep an existing solutions dir with a trivial command
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  --solutions-dir generated/ttrun/sc4pod_on_sc20_allsol \
  -- true

# C) Real cluster: generate for a host list, sweep just 3 solutions, dry-run first
python3 tools/scaleout/sweep_rank_binding_solutions.py \
  -m <mgd> --hosts nodeA,nodeB,nodeC,nodeD --limit 3 --dry-run -- ./my_app
```
