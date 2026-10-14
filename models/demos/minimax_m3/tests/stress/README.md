# MiniMax-M3 prefill stress test (one Blackhole galaxy)

Reproduces the SC24 P/D prefill deployment's pipeline stages on one galaxy, to catch device hangs that only show up
under long, production-shaped traffic. On a hang it captures tt-triage state of every rank before anything is killed.

SC24 runs M3 prefill as a chain of 32 pipeline stages, each a [2,4] mesh (SP=2 x TP=4, 8 chips). One galaxy holds 4
such stages. By default this test builds SC24's ranks 20-23 (model layers 37-44, 2 per rank) with the production
config and drives them with an AgentX-shaped producer.

## What is and is not modelled

Modelled:
- The per-stage config: `PREFILL_SP=2 PREFILL_TP=4 PREFILL_CHUNK_SIZE=2048 PREFILL_FABRIC_MODE=2d`, 90 users x 1M
  tokens of KV (`PREFILL_NUM_USERS=90 PREFILL_MAX_SEQ_LEN=1048576`), `M3_INDEX_K_TP_SHARD=1`.
- The real layers of the chosen SC24 ranks (weights and layer types), 2 per stage, through
  `TtPrefillRuntimeConfig.layer_indices`. KV slots, layer acks and the layer split stay logical (0..7).
- Z-chained D2D sockets between the stages, the H2D token stream into rank 0, the production chunk metadata.
- Agentic traffic: multi-turn sessions at deep contexts (median ~300K tokens), turns of ~1.6K new tokens, so most
  chunks start mid-slab and end ragged; concurrency from 2 to 90 sessions; the production duty cycle (stages idle
  ~95% of the time, long idles followed by a tiny request); cold starts right after a reset.

Not modelled:
- KV migration to decode. No migration worker runs; the runner leaves the KV chunk table in the run directory.
- The real request source. `stress_producer.py` replaces d-gen: tokens come from a trace file, and a deep session
  starts over an unwritten (zero) KV prefix instead of prefilling its history.
- 4 stages instead of 32, all on one galaxy, so there are no inter-galaxy links. Here stage 0 also builds the token
  embedding and the last stage the final norm and LM head, which in SC24 only the first and last ranks do.

## Quick start

The branch is a normal checkout: build it first (`./build_metal.sh`) and use its `python_env`. The weight cache
(`TT_CACHE_PATH`, default `/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill`) must hold the
`tensor_cache_bfp8_MeshShape([2, 4])` tensors of the chosen layers.

```bash
cd $TT_METAL_HOME
S=models/demos/minimax_m3/tests/stress

$S/run_stress.sh --dry-run                          # binding + env + tt-run dry run, no device access
$S/run_stress.sh                                    # production duty cycle, 2 h, layers 37-44
$S/run_stress.sh --first-layer 45                   # SC24 ranks 24-27 (layers 45-52)
$S/run_stress.sh --profile agentx --duration 90m    # saturated pipeline (~16x production's chunk rate)
$S/run_stress.sh --profile cold --duration 90m      # reset -> start -> few requests -> shutdown, repeated
$S/run_stress.sh --watcher --profile agentx --duration 40m
$S/run_stress.sh --env M3_MOE_HYBRID_THRESHOLD=0    # bisect knob (repeat --env for more)
$S/run_stress.sh --hard-stop 22:50                  # everything stopped and reset by 22:50 UTC

python3 $S/traffic.py --profile paced --sim-s 7200 --quiet   # preview a schedule on a simulated clock
```

Exit code: 0 = no hang, 3 = hang captured, 1 = refused or failed. Ctrl-C stops every process of the run and resets.

Timings with a warm weight cache: the runner is ready about 3-4 min after the reset; a cold-start cycle takes about
4 min. The first run on a tree also JIT-compiles all kernels. With 90 users x 1M, each chip has ~29 GiB of DRAM
allocated and ~2.5 GiB free.

## What a run does

1. Preflight: refuses if any process holds `/dev/tenstorrent/*` or another of your MPI / pytest jobs runs, or if the
   hard stop is less than 15 min away. Then `env -u TT_VISIBLE_DEVICES tt-smi -glx_reset`.
2. `make_binding.py` writes the tt-run rank binding (one rank per stage, `TT_VISIBLE_DEVICES` 0-7, 24-31, 16-23, 8-15,
   `STRESS_LAYERS`, per-rank `TT_METAL_LOGS_PATH`), the producer env and `run.json`. 4 stages use
   `single_bh_galaxy_4x4x2_z_chain_graph_descriptor.textproto`; 1-3 stages get a generated chain descriptor.
3. tt-run starts `stress_runner.py` on every rank: the stock `prefill_runner` with absolute layer placement, KV host
   zeros in bf16 (fp32 doubles the host RAM peak of 4 ranks on one host) and a per-chip DRAM report.
4. When every rank logs `request (unbounded) loop start`, the watchdog and `stress_producer.py` start. The producer
   runs for `--duration`, then sends the SHUTDOWN sentinel and the runner exits.
5. Stop: every process of the run is killed (they all carry `STRESS_RUN_ID`), the galaxy is reset, and
   `summarize_run.py` writes `summary.txt`: chunks started per rank vs producer pushes, sessions / turns, mid-slab
   and ragged shares, idle time, long idles, hang captures.

Verdicts (`RESULT` in `events.log`): `ok` (clean shutdown, every rank started every pushed chunk), `hang`, `no-exit`
(runner still up `EXIT_GRACE_S`=600 s after the producer finished), `startup-stall`, `startup-died`,
`runner-exit`, `producer-failed`, `hard-stop`, `refused`.

## Options

| flag | default | meaning |
|---|---|---|
| `--stages N` | 4 | stages (1-4), consecutive layers from `--first-layer` |
| `--first-layer L` / `--layers-per-stage N` | 37 / 2 | absolute layers; 37 = SC24 rank 20, 45 = rank 24 |
| `--layers "A,B C,D ..."` | | explicit layers per stage, e.g. `"51,52 37,38"` (overrides the three above) |
| `--users N` / `--max-seq-len N` / `--chunk-size N` | 90 / 1048576 / 2048 | KV slots, per-slot capacity, chunk |
| `--fabric MODE` | 2d | `PREFILL_FABRIC_MODE` |
| `--profile P` | paced | `paced`, `agentx`, `cold`, `custom` (below) |
| `--duration T` | 2h | load time (`5400`, `90m`, `2h`); startup and shutdown come on top. `cold`: total time |
| `--seed N` | 59542 | producer seed |
| `--env K=V` | | extra env for the runner (binding `global_env`) and the producer; repeatable |
| `--watcher` | off | the watcher env below, with its own kernel cache |
| `--watchdog-stall-s S` | 240 | stall threshold; 0 disables the watchdog |
| `--hard-stop T` | none | epoch or `HH:MM` UTC today; the run is cut short, killed and reset by then |
| `--out DIR` / `--label NAME` | `generated/m3_stress/<label>_<utc>` / profile | run directory |
| `--kernel-cache DIR` | `generated/m3_stress/kernel_cache/<HEAD>-<diff md5>-<prod\|watcher>` | `TT_METAL_CACHE` |
| `--cache-path DIR` | `$TT_CACHE_PATH` or the weka path | weight cache |
| `--trace-dir DIR` | `$PREFILL_TRACE_DIR` or `.../golden/longbook_56320` | producer token source |
| `--tcp-interface IF` | default-route interface | MPI TCP interface |
| `--ready-timeout-s S` | 3600 | runner startup limit (a capture is taken if it is exceeded) |
| `--no-reset` | | skip the resets (only when someone else just reset the box) |
| `--dry-run` | | write the files, print the commands, run `tt-run --dry-run`; no device access |

`STRESS_RESET_CMD` overrides the reset command.

### Traffic knobs

Flags set the `STRESS_*` variables below; `--env STRESS_X=...` or the caller's environment set any of them.

| variable | flag | default | profile |
|---|---|---|---|
| `STRESS_DEPTH` | `--depth` | `300000,0.55,32768,746000` | agentx, paced: session start depth, lognormal median, sigma, min, max (tokens) |
| `STRESS_FRESH_PROB` / `STRESS_FRESH_MED` | `--fresh-prob` | 0.1 / 12000 | share of sessions starting at 0, and their first-turn median |
| `STRESS_TURN` | `--turn` | `1600,1.0,30000` | new tokens per turn: lognormal median, sigma, max |
| `STRESS_CONT_PROB` / `STRESS_CTX_CAP` | | 0.95 / 760000 | chance a session takes another turn; context where it stops |
| `STRESS_CONC` / `STRESS_CONC_PERIOD_S` | `--conc` | `56,90` / 600 | agentx: active sessions, redrawn uniformly every period |
| `STRESS_THINK` | `--think` | `2,60` | agentx: lognormal median and max seconds between a session's turns |
| `STRESS_CONC_PHASES` / `STRESS_PHASE_S` | `--conc-phases` / `--phase-s` | `2,8,30,90` / 900 | paced: concurrency phases, shuffled per round |
| `STRESS_GAP` | `--gap` | `3,1.0,120` | paced: idle gap between bursts, lognormal median, sigma, max (s) |
| `STRESS_LONG_IDLE_PROB` / `STRESS_LONG_IDLE_S` | `--long-idle-prob` / `--long-idle-s` | 0.006 / `120,600` | paced: long idles, then one tiny (1-64 token) request |
| `STRESS_BURST_DIV` / `STRESS_INFLIGHT_MAX` | | 4 / 4 | paced: burst of 1 + Exp(conc / div) turns, 1..max turns interleaved |
| `STRESS_COLD` / `STRESS_COLD_GAP_MS` | `--cold` | `6,3` / `0,1500` | cold: tiny and normal requests per cycle, gap between them |
| `STRESS_CUSTOM_DEPTH` | `--custom-depth` | | custom: fresh requests start at a random depth in `[lo, hi)` |
| `STRESS_STATS_EVERY_S` | | 60 | producer progress line period |

`custom` runs the stock `prefill_producer` schedule; configure it with `--env PREFILL_PRODUCER_CHUNKS=1,8`,
`PREFILL_PRODUCER_MID_END_PROB`, `PREFILL_PRODUCER_MULTI_TURN_PROB`, `PREFILL_PRODUCER_INTERLEAVE` and so on.

### Runner env set by the binding

`PREFILL_MODEL=minimax_m3`, `PREFILL_MANIFEST` (M3 manifest), `PREFILL_SP/TP/NUM_LAYERS/PP_LAYER_COUNTS/CHUNK_SIZE/
MAX_SEQ_LEN/NUM_USERS/FABRIC_MODE`, `PREFILL_H2D_SERVICE_ID=m3_stress_<uid>`, `PREFILL_MIGRATION_TABLE_PATH` and
`PREFILL_MIGRATION_DEVICE_MAP_PATH` (in the run directory), `M3_INDEX_K_TP_SHARD=1`, `M3_INDEX_CACHE_BF16=0`,
`M3_WEIGHTS_FROM_CACHE=1` (required: the cache completeness check would look at the logical layers),
`TT_CACHE_PATH`, `TT_METAL_CACHE`, `TT_METAL_INSPECTOR=1`, `TT_METAL_INSPECTOR_RPC=1`, `STRESS_RUN_ID`, plus
`--env` and `--watcher`. Per rank: `TT_VISIBLE_DEVICES`, `STRESS_LAYERS`, `TT_METAL_LOGS_PATH`.

## Run directory

```
config.txt  binding.yaml  producer.env  run.json  [mesh_graph.textproto]
events.log           harness events and the RESULT line
runner.log           all ranks (tt-run --tag-output); grep "[pp rank N] CHUNK_START"
producer.log         pushes, [stress] progress lines, PHASE / IDLE / TINY events, the [stress] END stats
watchdog.log  reset.log  summary.txt
logs_r<N>/           per-rank TT_METAL_LOGS_PATH (watcher.log, inspector logs)
hang_<utc>/          capture (below)
c001/ c002/ ...      cold profile: one directory per cycle, cycles.txt one line per cycle
```

## Hang capture and classification

`stall_watchdog.py watch` reads `producer.log` and `runner.log`. A stall is: the producer pushed chunks that the last
rank has not started, and no rank logged a `CHUNK_START` for `--watchdog-stall-s`. Idle gaps of the paced producer
are not stalls, and an orderly SHUTDOWN ends the watch, so a healthy run is never triaged. The harness also captures
when the runner does not get ready in time, does not exit after the SHUTDOWN, or (cold profile) when the chunk
counts stop moving for 120 s.

A capture runs `tools/tt-triage.py` once per rank and step against the live run (inspector RPC port 50051 + rank,
the rank's `TT_VISIBLE_DEVICES`), writing `r<rank>_<step>.txt` and `.sqlite`:

| step | scripts |
|---|---|
| `00_moe_swiglu` | `dump_moe_swiglu` (in `triage/`): per core of a running moe_fused_swiglu program, its 15 semaphores, mailbox words, idx / counts scratch, protocol CB pushed / popped |
| `01_callstacks` | `dump_running_operations --full-cores`, `dump_callstacks --full-callstack` |
| `02_semaphores` | `dump_semaphores --include-global-unchanged` |
| `03_cbs` | `dump_circular_buffers --dump-cb-content` |
| `04_noc_eth` | `check_noc_status`, `check_eth_status`, `dump_risc_debug_signals`, `dump_lightweight_asserts` |
| `05_callstacks_again` | `dump_callstacks` (stuck vs moving) |
| `06_mesh_sockets` | `dump_mesh_sockets` (D2D / H2D socket credits between the stages) |
| `07_moe_swiglu_again` | `dump_moe_swiglu` (static vs moving) |

A full capture of 4 ranks takes about 7-8 min. Then `classify_moe.py <hang_dir>` (also written to `classify.txt`)
summarizes the moe_fused_swiglu state per device: reader block (`XSTAGED`), writer block (`WDSPLIT // 11`), `GO`,
`PHASE_FREE`, the expert in the mailbox, cores mid x-chain handshake, and whether the idx / counts scratch agrees
across cores (`--grid` prints it per core). `why.txt` has the last `CHUNK_START` of every rank.

Where to look first: which ranks still advance (`05` vs `01`), the waypoint of the stuck RISCs (`NABW` =
`noc_async_atomic_barrier`), and whether the stuck program is moe_fused_swiglu, combine or a socket op.

## Pitfalls

- Never run `dump_callstacks` (or the capture) against a healthy live run: it halts the RISCs and hangs the run.
  `dump_moe_swiglu` alone is safe on a live run.
- `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` does not work with fabric: the ETH kernels overflow their kernel config
  buffer and every rank dies at startup (`Program size ... too large for kernel config buffer ... on ACTIVE_ETH`).
- Watcher: this set is known to work with fabric (what `--watcher` sets): `TT_METAL_WATCHER=30 TT_METAL_WATCHER_DISABLE_ETH=1
  TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
  TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1`, with a per-rank `TT_METAL_LOGS_PATH`
  (ranks share `$TT_METAL_HOME/generated` otherwise). Without PR #60555 the watcher kills the runner during warm-up:
  deepseek_prefill `reader_worker_dispatch.cpp` exits with a non-posted atomic in flight (waypoint `NKFW`).
- Use a fresh kernel cache per tree and config. The default cache path includes HEAD, a hash of the tracked diff
  and the watcher mode; pass `--kernel-cache` for anything else that changes kernels (defines, patches).
- Reset is galaxy-wide (`tt-smi -glx_reset` with `TT_VISIBLE_DEVICES` unset). Preflight refuses to run, and the
  harness refuses to reset, while someone else holds a device. Size `--duration` / `--hard-stop` to your reservation.
- Under Slurm the harness unsets `SLURM_*` and sets `PRTE_MCA_ras=^slurm PRTE_MCA_plm=^slurm` so prterun runs the
  ranks locally.
- Long runs: start them inside tmux. If `/tmp` gets cleaned, use `tmux -S <path>` for the socket.

## Late-atomic microbenchmark

`late_atomic_bench/` is a single-chip microbenchmark of the mechanism behind one of the hangs: an NCRISC kernel that
exits with a non-posted atomic in flight leaks the late response into the next NCRISC kernel's NoC counters, and a
later atomic barrier there never completes. See [late_atomic_bench/README.md](late_atomic_bench/README.md):

```bash
TT_VISIBLE_DEVICES=0 python models/demos/minimax_m3/tests/stress/late_atomic_bench/late_atomic_bench.py --b-mode probe
TT_VISIBLE_DEVICES=0 models/demos/minimax_m3/tests/stress/late_atomic_bench/run_matrix.sh
```

## Files

| file | purpose |
|---|---|
| `run_stress.sh` | entry point: preflight, binding, runner, producer, watchdog, stop, reset, summary |
| `make_binding.py` | tt-run binding, mesh graph descriptor, producer env, `run.json` |
| `stress_runner.py` | stock prefill runner + absolute layers, bf16 KV host zeros, DRAM report |
| `stress_producer.py` | stock prefill producer + the traffic profile |
| `traffic.py` | agentx / paced / cold schedules (pure Python; `--sim-s` previews them) |
| `stall_watchdog.py` | stall detection and the tt-triage capture |
| `triage/dump_moe_swiglu.py` | tt-triage script for moe_fused_swiglu state |
| `classify_moe.py` | per-device summary of the moe_fused_swiglu captures |
| `summarize_run.py` | `summary.txt` of a run directory |
| `late_atomic_bench/` | single-chip late-atomic microbenchmark |
