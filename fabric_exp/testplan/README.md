# Regression tooling for landing the new fabric router

Implements TEST_PLAN.md §4 (fabric_exp/consolidation/TEST_PLAN.md): a manifest of Slurm-sized groups and one runner.
Started 2026-11-03.

| file | what |
| --- | --- |
| `manifest.yaml` | groups (system, kind, routers, A/B passes, status), their steps (`from_ci` = a `tests/pipeline_reorg` entry read from the tree under test, or an explicit `cmd`), systems, trees, CI env, weights and the router-mode env |
| `run_group.py` | `plan`, `resolve`, `run`, `job`, `stage`, `submit`, `collect`, `status`, `compare` (see `--help`) |
| `regress.sbatch` | thin Slurm wrapper; `submit` stages it with the runner under `/data/aagarwal/regress/stage/` |
| `regress_junit_plugin.py` | pytest plugin: one junit file per pytest invocation of a step |
| `results/<label>/` | results in the collector format below; `results/main/` = the main baselines |

## Trees

A tree is built once, Release, tests and python bindings, then staged to /data for the Slurm nodes.

- **main**: `/home/aagarwal/tt-metal-main`, a detached worktree of the study repo at origin/main
  `d2bd12fcf1f1aac1203b735bf12540a34a8c2fc1` (the base of the port). Built 2026-11-03 like the study tree (clang-20
  libstdc++ toolchain, Release, ccache in /data/ccache, Tracy on, distributed on, unity builds):
  `./build_metal.sh --build-tests --enable-ccache` then `./create_venv.sh` (uv, Python 3.10.19, relocatable venv,
  ttnn editable). build_Release 3.1 GB, python_env 3.4 GB, .cpmcache 3.5 GB.
- Its Slurm copy: `/data/aagarwal/regress/trees/main_d2bd12f` (`run_group.py stage --tree main`): rsync without .git,
  .cpmcache, generated/ and build intermediates (6.2 GB, 152k files); the venv's paths are rewritten to the copy and
  its Python (`/data/aagarwal/regress/uv/cpython-3.10.19-linux-x86_64-gnu`). `stage` verifies from a09u32 that no
  binary, library or Python module resolves back into /home (`--verify-only` repeats the check).
- **One local patch on main (user OK, 2026-11-03):** `patches/main_no_bw_telemetry.patch`, the study's
  `TT_FABRIC_NO_BW_TELEMETRY` switch in test_tt_fabric (test harness only; libraries, router and kernels stay pure
  d2bd12f; without the env the binary behaves as main). Only `test_tt_fabric` was rebuilt (21:29) and copied to the
  /data tree; both trees' `.regress_tree.json` and every later `status.json` (`tree_patches`) record it. The L0 results
  of LB-L0-a and B0-health / infra / stab ran on the unpatched binary. The port carries the same switch.
- A new-router tree is added to `trees:` the same way (build, `stage`), and its router mode to `router_env:`. `new`
  stays `TBD` until the port names its switch; the runner refuses to run a TBD router.

## Where groups run

- **a09u32 (P150x8, the user's broker):** `run` detects the broker host. Every job: workspace `/home/aagarwal`,
  `timeout_sec` 1500, command
  `cd /home/aagarwal/tt-metal-main && python3 /home/aagarwal/tt-metal/fabric_exp/testplan/run_group.py run --group G --tree main --router main --label main --resume`.
  A job runs steps until its 1380 s budget would be exceeded and stops; the same command again resumes (PASS / FAIL /
  HANG / SKIP steps are not rerun). Repeat until the output ends with `REGRESS: ALL STEPS DONE`. A step that does not
  fit a fresh job runs capped at the budget (status CAPPED if it hits it; it reruns on the next resume). The runner
  never resets the device. Exit code 3 = a step was killed (HANG or CAPPED): reset the device before the next job.
- **Slurm (BH Galaxy 110-a; extra LoudBoxes a07u28 / a08u28 / a04u24):** `submit` stages the runner and sbatches
  `regress.sbatch`; without `--go` it only runs `sbatch --test-only`. The job (`run_group.py job`) checks the node
  (110-a regex, LB node list), stops if other users' device processes are on it, writes an environment report
  (mounts, disk, tools) to `/data/aagarwal/regress/results/_jobs/<job>/env_report.txt`, builds a per-job
  `TT_METAL_HOME` farm under `/data/aagarwal/regress/results/_scratch/<job>/` (symlinks into the /data tree, its own
  `generated/`, the JIT cache; removed at the end; nothing is written on the node outside /data/aagarwal),
  resets (`tt-smi -glx_reset` / `tt-smi -r`) and runs `test_system_health` like CI's ensure-bh-links-online, then
  every router / pass. After a FAIL / HANG it resets and checks health again before the next step; a busy device or
  a failed reset stops the job. Galaxy submits refuse below 150k free inodes on /data (`min_data_inodes_free`; the count goes to the job record) and check the 4-job cap (`squeue -u aagarwal`, Galaxy partitions, running and
  pending; interactive jobs and the multi-Galaxy quad on the pod partitions excepted). Ring/torus steps add `--exclude=bh-glx-110-a02u14`.
- **Not run here:** groups with status `planned` (cpu_only, QB2, the study yamls), `held` (fabric BW / latency perf:
  telemetry decision), `ci_only` (Wormhole, vLLM, multi-user isolation), `multi_galaxy` (glxq harness).

## Environment of a step

The CI container's (tm-fabric / blackhole-e2e impl workflows and setup-job): `TT_METAL_HOME`, `PYTHONPATH=$TT_METAL_HOME`,
`LD_LIBRARY_PATH=<tree>/build/lib`, `ARCH_NAME=blackhole`, `LOGURU_LEVEL=INFO`, `GTEST_OUTPUT=xml:<step>/gtest/`,
setup-job's auto-triage `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` (not for entries with `auto_triage: false`; no triage
command); for fabric_tests.yaml (tm-fabric) entries also `TTNN_CONFIG_OVERRIDES` validate_program_args and
`TT_LOGGER_LEVEL=warn` (so test_tt_fabric's info summary is hidden there and those steps are judged by exit code, as CI
does; the first L0 jobs had warn on every step, which changes only log verbosity).
Every inherited `TT_*`, `PYTHONPATH`, `VIRTUAL_ENV`, `PYTHON_ENV_DIR` (the broker injects study-tree values) is dropped
first; the tree's venv leads PATH. `TRACY_NO_WEB_SERVER=1` keeps tools/tracy from leaving a server behind. The router
mode's env (`router_env`) is added last, before the step's own `env`.

**Telemetry (user decision 2026-11-03):** fabric perf groups (LB-P, G-P) run with router bandwidth telemetry OFF for
both routers (`TT_FABRIC_NO_BW_TELEMETRY=1` on top of CI's env), labelled "telemetry off (prod default)" (`telemetry` in
status.json); two passes on one host per group. CI's fabric goldens were recorded telemetry-on and will not match;
re-goldening is a team proposal (TEST_PLAN §3.3). Those steps `need: no_bw_telemetry_switch`: they SKIP on a tree whose
test_tt_fabric binary lacks the switch, so nothing measures telemetry-on by accident. Coverage, CCL and model groups
keep CI's defaults.

Weights: steps that `need: weights_mlperf` get CI's `HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub`, `HF_HUB_OFFLINE=1`.
Where `/mnt/MLPerf/huggingface` is missing and `/mnt/models/huggingface` exists (a09u32), `path_map` rewrites the
prefix in the command and env, as CI's `yyz4-mnt-models` mode mounts it for QB2. A step that names a `/mnt` path that
exists nowhere is SKIP with the missing paths.

Differences from CI, on purpose: a multi-line command keeps going after a failing line (every failing line is logged
as `[regress] command failed` and fails the step; `--stop-on-fail` gives CI's `bash -e`); the per-step device reset of
CI happens on Slurm only after a non-PASS step (and at job start).

## Results (collector format)

```
results/<label>/<group>/<router>/pass<k>/<step>/
    cmd.txt  cmd.sh  env.txt  log.txt
    status.json     status PASS | FAIL | HANG | CAPPED | BUSY | SKIP, rc, start, end, duration_s, node, mode, tree, tree_sha,
                    ci {list, name, sku, ci_timeout_min}, timeout_s, tests {PASS, FAIL, SKIP}, tt_fabric {executed, failed},
                    marker_found, metrics, failed_lines, reason (SKIP), path_map, slurm_job
    tests.json      per test case {status, time, file, message} from gtest XML (gtest/) and pytest junit (junit/)
    metrics.csv     result: metrics (regex rows: key columns + value)
    fabric/*.csv    result: fabric_csv (generated/fabric files the step wrote)
    benchmark_data/ result: benchmark_json (generated/benchmark_data files the step wrote)
results/<label>/<group>/summary.json
```

Step statuses:

| status | meaning | scored in compare |
| --- | --- | --- |
| PASS / FAIL | exit code, junit / gtest cases, test_tt_fabric's failed count, log marker | yes |
| HANG | killed at its timeout, log silent for > 2 min | yes (as not passing) |
| TIMEOUT | killed at its timeout while the log was still advancing (slow, not stuck) | rerun, not a verdict |
| CAPPED | killed by the broker job budget, not its CI timeout; reruns on the next `--resume` | no |
| BUSY | another process held the device (KMD lock) | no |
| SKIP | a need is unmet here (weights, ring mapping, missing /mnt paths) | no |
| NOT_RUN | manifest `status: not_run` (needs CI, e.g. missing CI caches) | no |
| KNOWN_FAIL | manifest `known_fail`: fails on main for a reason outside the router (TEST_PLAN §4.5) | no |
| PERF_WINDOW | every failing case only missed its in-test perf window; the value is in metrics / tests.json | metrics only |
| ENV | manifest `env_fail`: the node's environment (e.g. quad cabling vs a single-Galaxy descriptor) | no |

An earlier attempt of a rerun step (CAPPED / BUSY) is kept as `<step>.attemptN/` and ignored by status / compare;
`_invalid_staging/` holds the Galaxy steps that ran before the /data tree had its runtime objects (jobs 171351/171352).

More options: `run --rerun-failed STEP` reruns each FAIL case of a step in its own pytest process (`STEP.rerun`);
`submit --node any` (LoudBox) takes whichever approved LB frees first (partitions + excludes, never a09u32);
`submit --node <galaxy>` pins a Galaxy; `submit --local-cache` puts the job's JIT cache in its own node /tmp dir.
Galaxy jobs always exclude a01u14 (full /tmp), a02u02, a02u08, a03u14 and a08u20 (multi-process stall), and a02u14 for
ring/torus; every Slurm job sets
TMPDIR to its own /tmp dir and stops if /tmp has < 2 GB free. Slurm jobs write the same tree
under `/data/aagarwal/regress/results/<label>/`; `collect --label L` copies it here. Slurm logs:
`/data/aagarwal/regress/logs/regress-<label>-<group>-<job>.log`.

## Compare

`run_group.py compare --base main --test new_<sha> [--group G]` (each side `LABEL[:ROUTER]`, so a same-job A/B is
`--base L:main --test L:new`). Writes `results/compare_<base>_vs_<test>.{md,csv}`:
- per step: statuses per pass, test cases PASS on base and FAIL on test (and the reverse), cases present on one side;
- perf rows: metrics.csv rows, fabric bandwidth (`avg_bandwidth_gigabytes_per_s`) and latency (`net_avg_ns`) summary
  rows joined on TEST_PLAN §4.2's keys, and benchmark JSON measurements; diff = test/base − 1, noise = base
  pass-to-pass spread, flagged beyond max(noise, `--threshold`, default 1 %) (TEST_PLAN §3.3).
- Exit 1 if any test case went PASS → FAIL.

## Quick reference

```
python3 run_group.py plan --levels L0,L1                       # groups, steps, CI timeouts, broker / submit lines
python3 run_group.py resolve --group G-L0-b                    # the exact commands, from the tree's CI lists
python3 run_group.py stage --tree main                         # /data copy for Slurm (+ verify)
python3 run_group.py submit --group G-L0-a --tree main --label main --routers main          # dry run
python3 run_group.py submit --group G-L0-a --tree main --label main --routers main --go     # after the user's OK
python3 run_group.py collect --label main
python3 run_group.py status --label main
python3 run_group.py compare --base main --test new_<sha>
python3 run_group.py noise --label main --group LB-P             # one router's pass-to-pass spread per perf row
python3 run_group.py rescore --label main                        # recompute PASS/FAIL after a collector-rule fix
```

A broker job on a09u32 exits 1 whenever the group has a FAIL anywhere (also a resume that finds nothing left to run).

## Baseline status, label `main` (d2bd12f; 2026-11-04 07:00)

`python3 run_group.py status --label main` has the per-step detail.

| group | host | result |
| --- | --- | --- |
| LB-L0-a | a09u32 broker | 7/7 PASS |
| LB-L0-b | a09u32 broker | 5/5 PASS (B0-eth: 2 slow-dispatch cases KNOWN_FAIL, two passes) |
| LB-L1 | a09u32 broker | CI=true: 10/10 PASS (night-a 2 + night-b 8 + watch 1 cases KNOWN_FAIL); pre-CI records `.attempt1` |
| LB-L1P | a09u32 broker | CI=true, 3 passes: ag / ar / bc PASS (25 metric rows, median noise 0.64 %, max 4.8 %); sdpa KNOWN_FAIL |
| LB-P | a09u32 broker | 2 passes PASS, telemetry off (134 BW + 12 latency rows; median noise 0.02 %, max 3.1 % on ring multicast) |
| LB-L2 | a09u32 broker | CI=true: k3 PASS (7 metrics); kda + m4 PERF_WINDOW (values in metrics); ds NOT_RUN (weights) |
| LB-X-a09fails | a07u28 Slurm | the 13 a09u32 failures reproduce on a07u28 (cross-host evidence) |
| G-L0-a | Galaxy a03u08 / a08u02 | 6 PASS + G0-night PASS (pass 2, local cache); health-cv ENV; night pass 1 TIMEOUT (/data cache) |
| G-L0-b | Galaxy a10u08 / a03u14 | 11 PASS; G0-mm-2x8-pt KNOWN_FAIL (ETH program 96 B over); a03u14 stall records ENV (.attempt1) |
| G-P | Galaxy a03u08 | 2 passes PASS, telemetry off (torus ubench; max noise 0.81 %) |
| G-L1 | Galaxy a03u08 / a09u14 | night PASS 100; dp-combine PASS 21; dp-hybrid PASS (10 TypeError cases KNOWN_FAIL); mh NOT_RUN |
| G-L1P | Galaxy a03u08 | CI=true, 2 passes PASS 28 (32 metric rows, median noise 0.25 %, max 5.0 % on the 1720-row case) |
| G-L2-a | Galaxy a08u08 / a03u08 | CI=true: qwen PASS (30-min timeout; TTFT 110 ms, 3.02 tok/s/user); oss PASS (TTFT 179 ms, 27.45 tok/s/user); mm3 KNOWN_FAIL (2400 B over) |
| G-L2-b | Galaxy a07u08 / a08u02 / a10u08 | more PASS; hyb PASS; attn KNOWN_FAIL (TT_FATAL + hang); fabric_2d per file: 10 PASS (pw with 1 known-fail case), lmh (fabric-relevant) / moe / pre / sdpa KNOWN_FAIL, brms + mhp PASS (all cases skipped); sweep NOT_RUN (TEST_PLAN 4.5) |

## Baseline findings log (main d2bd12f)

Main-failure classifications live in TEST_PLAN.md §4.5. Harness and environment findings:

- **Staging bug (fixed 2026-11-03 22:27):** `stage` excluded `*.o` everywhere and dropped runtime/hw/lib/*/*.o and the sfpi
  crt objects, so every kernel JIT link failed on Galaxy (jobs 171351/171352, first steps). The exclude is now anchored to
  build_Release. Those steps are in `/data/aagarwal/regress/results/main/_invalid_staging/`.
- **Galaxy multi-process tt-run stall (ENV, TEST_PLAN §4.5):** on bh-glx-110-a03u14 (job 171352) and a08u20 (job
  171987, main and the port alike, same-node two-arm job), every non-bare tt-run step stalls after "Starting devices in
  cluster completed". The same steps pass on a10u08 and a09u14.
  - WRONG (job 171869): "Open MPI TCP over the host's k8s/CNI interfaces".
  - Both nodes are in the standing Galaxy excludes. The a03u14 HANG records are kept as `<step>.attempt1` with status ENV.
- **Galaxy G0-mm-2x8 mesh pass-through:** KNOWN_FAIL on main (ACTIVE_ETH program 96 B over its kernel config buffer).
  It stays in main-vs-new: a pass there is a gain.
- **G-L1:**
  - G1-mh is NOT_RUN: SINGLE_BH selects 0 of 78 tests on main.
  - G1-dp ran past CI's 40 min here, so it runs as its two test files (G1-dp-combine, G1-dp-hybrid).
  - Killed pytest steps get their case outcomes from the -v log (`tests_from_log`).
- **G0-night timeout:** on a03u08, with the JIT cache on /data, it was killed at 1806 s (CI timeout 30 min) while still
  progressing (TIMEOUT, 125 OK so far). On a08u02, with a node-local cache, it passed in 1343 s (165 pass). Suggestive
  only: node and cache location both changed. Later Galaxy jobs use `--local-cache`.
- **G0-health:** run_cluster_validation fails on every 110-a node (all quad-cabled), so it is ENV; the torus stability
  part is scored.
- **a09u32 device rule:** twice a no-device harness check reached the device outside the broker (pytest
  --collect-only; `tt-smi -ls` in a job self-test). `job` now refuses on a broker host without `--no-reset`, and its
  env report skips tt-smi there.

## CI dispatch list (CI-only groups)

For the user to dispatch (agents don't use gh). **Base:** a branch whose head is exactly d2bd12f (e.g. push
`aagarwal/regress-base-d2bd12f` at d2bd12fcf1f1, no extra commit), so CI's main numbers share the local trees' base.
The alternative is the scheduled main runs, but none runs exactly at d2bd12f. **New router later:** the same dispatches
on the port branch plus one throwaway commit that flips the default router (TEST_PLAN §0), never merged.

| # | workflow | inputs (others default) | covers (TEST_PLAN IDs) | what to look at |
| --- | --- | --- | --- | --- |
| 1 | `tm-fabric-tests.yaml` | run-bh_loudbox=false, run-bh_p300=false (all others true: N300, T3K, WH Galaxy, QB2, CPU-only, perf) | N300-fab, W0-t3k, W0-t3k-udm, W0-6u, W0-6u-mp, X0-cpu; QB2 B0-*; perf W0-t3k-bench, B0-muxbw | per-leg pass/fail; artifacts `test_reports_*` (gtest XML, `generated/fabric/*.csv`: T3K BW/latency vs `golden_*_wormhole_b0_t3k.csv`, mux v2 throughput log) |
| 2 | `sanity-tests.yaml` | run-ttnn-sanity-tests=false, run-umd-sanity-tests=false, run-llk-sanity-tests=false, run-models-sanity-tests=false | X0-build incl. Quasar sim (run-sim), `t3k fabric tests`, W1-t3k-s, BH multi-card sanity (B1-apc on QB2, B1-qb2 perf) | sim legs compile; QB2 high-BW all-gather perf log |
| 3 | `t3000-tests.yaml` | run-e2e-tests=true | W0-t3k-night (fabric nightly, addrgen, stability), W1-t3k (ccl, 2d_ccl) | e2e legs `t3k_fabric_*`, `t3k_ccl_tests`, `t3k_ccl_2d_tests` |
| 4 | `galaxy-tests.yaml` | wormhole=true, blackhole=false, run-e2e-tests=true | W0-tg, W0-tg-e2e | fabric unit / multiprocess / e2e legs |
| 5 | `galaxy-tests.yaml` | wormhole=false, blackhole=true, run-unit-tests=false, run-integration-tests=false, run-multi-user-isolation-tests=true | G1-iso (BH multi-user isolation; containers, not reproducible from Slurm) | the three `Multi-user isolation ... blackhole` legs |
| 6 | `galaxy-sanity.yaml` | (defaults: wormhole) | W1-tg-s, WH Galaxy health | `Fabric CCL tests` leg |
| 7 | `tt-metal-l2-nightly.yaml` | run_ccl_tests=true, additional_test_categories=`data_movement,dp_op_unit_tests` | S1-ttnn-dm, B1-p2p, W1-n300, W1-tg, B1-night (QB2), G1-night (BH Galaxy, also run here on Slurm), B1-dp (P150/QB2/LB/Galaxy) | ops legs `ccl nightly tests`, `Galaxy CCL tests`, `Galaxy MoE tests`, `ttnn n300 ccl tests`, P2P, DP op unit tests |
| 8 | `runtime-unit-tests.yaml` | (defaults) | S1-dm (data movement, slow dispatch, NoC debugging) | `runtime_data_movement*`, `runtime_noc_debugging` |
| 9 | `runtime-integration-tests.yaml` | (defaults) | B0-eth / G0-eth cross-check, `bh_multicard_sockets`, WH eth | `*_eth`, sockets legs |
| 10 | `blackhole-e2e-tests.yaml` | system-type=`QuietBox 2 (2xP300)` | QB2: fabric infra / bench / UDM / fast / stability, CCL apc, DP prefill (QB-L0 / QB-L1 / B2-dp QB2 while bh_qb2 Slurm is unapproved) | per-leg results; `fabric sanity benchmark` CSVs vs `golden_*_blackhole_p150_x4.csv` (perf: telemetry decision) |
| 11 | `models-t1-e2e-tests.yaml` | model=`llama3.3-70b-galaxy`, then `llama3.1-8b-dp-galaxy`, `deepseek-v3-tg`, `llama3.1-8b`, `qwen3.6-27b`, `qwen3.6-35b-a3b`, `gpt-oss-120b`, `gemma-4-31b`, `flux.1-dev` (sku=all; one dispatch per model) | W2-l70, W2-l8dp, W2-ds, S2-l8b, B2-q36, B2-q36moe, B2-oss (QB2 / WH Galaxy), B2-gemma, B2-flux | `generated/benchmark_data/*.json` artifacts and `validate_perf_targets.py` step (model_targets.yaml, 15 % default) |
| 12 | `models-t1-device-perf-tests.yaml` | model=`deepseek-v3-tg`, then `llama3.1-8b` | W2-ds device perf, B2-l70perf (QB2) | in-test device-perf asserts; tracy CSVs |
| 13 | `models-t1-unit-tests.yaml` | model=`gpt-oss-120b`, then `deepseek-v3` | W2-t3k unit, DeepSeek unit | per-leg results |
| 14 | `vllm-model-tests.yaml` | sku=`bh_loudbox (BH LoudBox)`, then `bh_galaxy (BH Galaxy)` | B2-vllm-lb, G2-vllm | throughput / accuracy logs |
| 15 | `multi-host-physical.yaml` | glx_4x_unit_tests=true | W0-mg (dual / quad WH Galaxy, dual T3K) | multihost legs |
| 16 | `multi-host-deepseekv3.yaml` | setup=both, deepseekv3_unit_tests=true, deepseekv3_module_tests=true | W2-mg | per-leg results |
| later | `galaxy-stress-tests.yaml`; `fabric-multihost-exabox.yaml`; `blaze-models-prefill-tests.yaml` (test-type `gate` / `ccl`) | — | W0-tg-stress (before default-on); MG0 (multi-Galaxy agent); SC1-blaze gates | |

Record each run id and head sha with the results (`results/main/CI_RUNS.md` once dispatched).
