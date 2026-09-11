# DeepSeek-V3 prefill stress + monitor scripts

| File | Purpose |
|---|---|
| `common.sh` | Shared config + helpers (`TT_METAL_HOME`, `LOG_DIR`, `KFILTER`, derived `INNER_ITERS`, `ENV_VARS`, `log_for`, `scan_log_dir`). Sourced by the others — not run directly. |
| `stress.sh` | Outer loop (`$LOOP`×): `tt-smi -glx_reset` then pytest. Each run's stdout is `tee`'d to `<log dir>/log_NN`. No timeout — pytest stays alive on hang for manual debug. |
| `watch.sh` | Refreshing status table (PASS / HANG? / FAIL / RUN / STALE / PENDING) over the run logs. Refresh 15s (override via `REFRESH`). |
| `watch_multiple_dirs.sh` | Same status table across several log dirs at once — one block per `<log_name>` arg. Args are all log names (no positional `loop_count`); set scan depth via `LOOP=` env. |
| `tail.sh` | `tail -10` of the newest `log_NN`, refresh 30s (override via `REFRESH`). |
| `parse_iteration_times.py` | Per-iteration timing extractor (min / avg / max). Reads any pytest log with `Starting iteration:` markers. |
| `analyze_l1_dumps.py` | Reads the L1 dumps `attn_res_l1_by_depth` writes and reports per-core tenancy. |
| `plot_pipeline_trace.py` | Renders a pipeline Gantt from a run's `PREFILL_TIMING_DIR` CSVs. Also called by `run_multirank_pcc.sh`'s EXIT trap. |
| `slice_pipeline_run.py` | Cuts one request out of a pipeline run's logs for per-chunk inspection. |
| `build_full_cache.py` | Kimi-K3 TTNN weight-cache generator, one layer at a time (92 MoE layers at 47.9 GB/chip do not fit in 34.2 GB/chip), with per-layer completion markers so an interrupted build resumes. Asserts nothing and runs no forward, so it is a generator rather than a gate. |
| `attn_res_l1_by_depth.py` | Measures AttnRes L1 tenancy against sealed-set depth (the instrument behind #54876). Asserts nothing. |

The last two are pytest-shaped because they need the `mesh_device` / `device_params` fixtures, but
they are not gates -- pytest collects a file passed explicitly regardless of its name, so run them as:

```bash
pytest models/demos/deepseek_v3_d_p/scripts/build_full_cache.py -s
pytest models/demos/deepseek_v3_d_p/scripts/attn_res_l1_by_depth.py -s
```

Keeping them out of `tests/` is what stops them being collected by the CI legs that glob
`tests/kimi_k3/` and `tests/attn_res/` -- where the L1 instrument was previously collected and then
silently deselected on `bh_loudbox` for carrying `requires_mesh_topology((8,4))`.

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
