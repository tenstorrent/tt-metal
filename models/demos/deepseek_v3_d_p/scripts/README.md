# DeepSeek-V3 prefill stress + monitor scripts

| File | Purpose |
|---|---|
| `common.sh` | Shared config + helpers (`TT_METAL_HOME`, `LOG_DIR`, `KFILTER`, derived `INNER_ITERS`, `ENV_VARS`, `log_for`, `scan_log_dir`). Sourced by the others — not run directly. |
| `stress.sh` | Outer loop (`$LOOP`×): `tt-smi -glx_reset` then pytest. Each run's stdout is `tee`'d to `<log dir>/log_NN`. No timeout — pytest stays alive on hang for manual debug. |
| `watch.sh` | Refreshing status table (PASS / HANG? / FAIL / RUN / STALE / PENDING) over the run logs. Refresh 15s (override via `REFRESH`). |
| `watch_multiple_dirs.sh` | Same status table across several log dirs at once — one block per `<log_name>` arg. Args are all log names (no positional `loop_count`); set scan depth via `LOOP=` env. |
| `tail.sh` | `tail -10` of the newest `log_NN`, refresh 30s (override via `REFRESH`). |
| `parse_iteration_times.py` | Per-iteration timing extractor (min / avg / max). Reads any pytest log with `Starting iteration:` markers. |

Most scripts take the same args: `<log_name> [loop_count]`. Logs go to `/data/$USER/<log_name>/log_NN`.
(Exception: `watch_multiple_dirs.sh` takes one or more `<log_name>` args and reads the scan depth from the `LOOP` env var — see below.)

---

## Launch a run (3 tmux sessions: stress + watch + tail)

```bash
export TT_METAL_HOME=/data/$USER/tt-metal
cd "$TT_METAL_HOME"

LOOP_CNT=20
COMMIT_HASH=$(git rev-parse --short HEAD)
DATE=$(date +%Y_%m_%d_%H_%M)
LOG_NAME="LOG_${DATE}_${HOSTNAME}_${COMMIT_HASH}_loop_${LOOP_CNT}"
SCRIPTS="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/scripts"

# 1) stress loop
tmux new-session -d -s "stress_${HOSTNAME}"       -E "bash -l -c '$SCRIPTS/stress.sh $LOG_NAME $LOOP_CNT |& tee $TT_METAL_HOME/$LOG_NAME.log'"
# 2) status table
tmux new-session -d -s "stress_watch_${HOSTNAME}" -E "bash -l -c '$SCRIPTS/watch.sh  $LOG_NAME $LOOP_CNT'"
# 3) tail latest log
tmux new-session -d -s "stress_tail_${HOSTNAME}"  -E "bash -l -c '$SCRIPTS/tail.sh   $LOG_NAME $LOOP_CNT'"
```

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
