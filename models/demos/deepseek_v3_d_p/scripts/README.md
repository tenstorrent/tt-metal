# DeepSeek-V3 prefill stress + monitor scripts

| File | Purpose |
|---|---|
| `common.sh` | Shared config + helpers (`TT_METAL_HOME`, `LOG_DIR`, the `MODEL` switch → `PYTEST_TARGET` / `ENV_VARS`, derived `INNER_ITERS`, `log_for`, `scan_log_dir`). Sourced by the others — not run directly. |
| `stress.sh` | Outer loop (`$LOOP`×): `tt-smi -glx_reset` then pytest. Each run's stdout is `tee`'d to `<log dir>/log_NN`. No timeout — pytest stays alive on hang for manual debug. Preflights the node id once before the loop. |
| `watch.sh` | Refreshing status table (PASS / HANG? / FAIL / RUN / STALE / PENDING) over the run logs. Refresh 15s (override via `REFRESH`). |
| `watch_multiple_dirs.sh` | Same status table across several log dirs at once — one block per `<log_name>` arg. Args are all log names (no positional `loop_count`); set scan depth via `LOOP=` env. |
| `tail.sh` | `tail -10` of the newest `log_NN`, refresh 30s (override via `REFRESH`). |
| `parse_iteration_times.py` | Per-iteration timing extractor (min / avg / max). Reads any pytest log with `Starting iteration:` markers. |

Most scripts take the same args: `<log_name> [loop_count]`. Logs go to `/data/$USER/<log_name>/log_NN`.
(Exception: `watch_multiple_dirs.sh` takes one or more `<log_name>` args and reads the scan depth from the `LOOP` env var — see below.)

---

## Picking the model — `MODEL`

`MODEL` is required; `common.sh` errors out if it is unset or unrecognized. It selects the test
function, the parametrize ids, and the model's env vars (each adapter defines its own env var names,
so these cannot be exported once and shared):

| `MODEL` | test function | node id (variant / layers) | env vars set for you |
|---|---|---|---|
| `KIMI_K2_6` | `test_kimi_prefill_transformer_chunked_perf` | `kimi` / `L61` | `KIMI_K2_6_HF_MODEL`, `TT_KIMI_PREFILL_TTNN_CACHE`, `PREFILL_TRACE_DIR` |
| `KIMI_K2_7` | `test_kimi_prefill_transformer_chunked_perf` | `k27` / `L61` | `KIMI_K2_7_HF_MODEL`, `TT_KIMI_PREFILL_TTNN_CACHE`, `PREFILL_TRACE_DIR` |
| `GLM5_2` | `test_glm_prefill_transformer_chunked_no_pcc` | `glm52` / `L78` | `GLM52_HF_MODEL`, `TT_GLM52_PREFILL_TTNN_CACHE`, `PREFILL_TRACE_DIR` |

Both Kimi variants read the *same* `TT_KIMI_PREFILL_TTNN_CACHE` but from different cache roots, which
is why `MODEL` sets it per run rather than you exporting it in your shell.

The run point is pinned by an exact pytest node id (`PYTEST_TARGET`), assembled from these shared ids
— override any of them to move the run point:

| Env | Default | Valid values |
|---|---|---|
| `MESH_ID` | `mesh-8x4` | `mesh-8x4` |
| `PRELOAD_ID` | `preload0` | `preload0`, `preload25k`, `preload50k`, `preload95k` |
| `CHUNKS_ID` | `chunks20` | `chunks1`, `chunks2`, `chunks5`, `chunks10`, `chunks_eleven`, `chunks20` |
| `ITERS_ID` | `iters20` | `iters1`, `two_iters`, `ten_iters`, `iters20`, `iters25` |
| `TRACE_ID` | `notrace` | `notrace`, `traced` — Kimi only; the GLM test has no `use_trace` axis |

A bad combination fails at `stress.sh`'s preflight (`pytest --collect-only`) in seconds, before any
device reset.

---

## Launch a run (3 tmux sessions: stress + watch + tail)

```bash
export MODEL=KIMI_K2_7          # or KIMI_K2_6 / GLM5_2
export TRACE_ID=notrace         # or traced — Kimi only, ignored for GLM5_2
export TT_METAL_HOME=/data/$USER/tt-metal
cd "$TT_METAL_HOME"

LOOP_CNT=20
COMMIT_HASH=$(git rev-parse --short HEAD)
DATE=$(date +%Y_%m_%d_%H_%M)
LOG_NAME="LOG_${DATE}_${HOSTNAME}_${MODEL}_${COMMIT_HASH}_loop_${LOOP_CNT}"
SCRIPTS="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/scripts"

# 1) stress loop
tmux new-session -d -s "stress_${HOSTNAME}"       -E "bash -l -c 'MODEL=$MODEL TRACE_ID=$TRACE_ID $SCRIPTS/stress.sh $LOG_NAME $LOOP_CNT |& tee $TT_METAL_HOME/$LOG_NAME.log'"
# 2) status table
tmux new-session -d -s "stress_watch_${HOSTNAME}" -E "bash -l -c 'MODEL=$MODEL $SCRIPTS/watch.sh  $LOG_NAME $LOOP_CNT'"
# 3) tail latest log
tmux new-session -d -s "stress_tail_${HOSTNAME}"  -E "bash -l -c 'MODEL=$MODEL $SCRIPTS/tail.sh   $LOG_NAME $LOOP_CNT'"
```

`MODEL` is passed explicitly into each tmux session because `-E` starts a login shell that does not
inherit your exported vars — the same reason `TRACE_ID` (and any other node-id override, e.g.
`CHUNKS_ID` / `ITERS_ID`) must be passed on the command line rather than exported. Exporting alone
looks like it works and silently runs the default node id instead. `TRACE_ID` goes to `stress.sh`
only: it feeds `PYTEST_TARGET`, which is the one thing `watch.sh` / `tail.sh` never use. `LOG_NAME`
carries the model so parallel per-model runs land in distinct log dirs (and read correctly in
`watch_multiple_dirs.sh`) — add `$TRACE_ID` to it too if you stress `notrace` and `traced` at once.

Attach (read-only):

```bash
tmux attach -t "stress_${HOSTNAME}" -r        # stress loop
tmux attach -t "stress_watch_${HOSTNAME}" -r  # status table
tmux attach -t "stress_tail_${HOSTNAME}" -r   # tail
```

## Monitoring several runs at once — `watch_multiple_dirs.sh`

When you have multiple stress runs going in parallel (e.g. different commits or
configs), pass each run's `<log_name>` as an argument to get one stacked status
block per run, all on one screen:

```bash
SCRIPTS=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/scripts
$SCRIPTS/watch_multiple_dirs.sh LOG_runA LOG_runB LOG_runC
```

With no args it falls back to the single default log dir (`deepseek_v3_d_p_log`).

Env overrides (all optional):
- `LOOP` — outer iterations to scan per dir (default 20; raise it if your runs use
  a larger `loop_count`, e.g. `LOOP=50 ... watch_multiple_dirs.sh ...`).
- `REFRESH` — refresh interval in seconds (default 15).
- `STALE_SECS` — idle seconds before a still-running iteration is flagged STALE
  (default 240).
