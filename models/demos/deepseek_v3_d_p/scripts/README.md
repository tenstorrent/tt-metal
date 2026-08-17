# DeepSeek-V3 prefill stress + monitor scripts

## Run it

Kimi K2.7, no trace, 20 iterations per run, 20 runs — copy-paste from anywhere:

```bash
MODEL=KIMI_K2_7 TRACE_ID=notrace ITERS_ID=iters20 TT_METAL_HOME=/data/$USER/tt-metal /data/$USER/tt-metal/models/demos/deepseek_v3_d_p/scripts/launch.sh 20
```

`./launch.sh [loop_count] [log_name]` — both optional (20, and
`LOG_<date>_<host>_<model>_<commit>_loop_<n>`). Logs go to `/data/$USER/<log_name>/log_NN`.

One detached session `stress_$HOSTNAME` (`SESSION=` to rename), one window, 2×2:

```
┌──────────────────────┬──────────────────────┐
│ 0  stress.sh         │ 2  watch.sh          │
│    the pytest loop   │    status table      │
├──────────────────────┼──────────────────────┤
│ 1  tail.sh           │ 3  host_stats.sh     │
│    newest log_NN     │    host CPU / DRAM   │
└──────────────────────┴──────────────────────┘
```

```bash
tmux attach -t "stress_${HOSTNAME}" -r       # ctrl-b z zooms a pane, ctrl-b <arrow> moves
tmux kill-session -t "stress_${HOSTNAME}"    # stop the run
```

## Knobs

`MODEL` is required. It picks the test function, the variant/layers ids, and the model's env vars
(`*_HF_MODEL`, the TTNN cache, `PREFILL_TRACE_DIR`) — both Kimi variants share
`TT_KIMI_PREFILL_TTNN_CACHE` but point at different roots, so it can't be exported once.

| `MODEL` | test function | variant / layers |
|---|---|---|
| `KIMI_K2_6` | `test_kimi_prefill_transformer_chunked_perf` | `kimi` / `L61` |
| `KIMI_K2_7` | `test_kimi_prefill_transformer_chunked_perf` | `k27` / `L61` |
| `GLM5_2` | `test_glm_prefill_transformer_chunked_no_pcc` | `glm52` / `L78` |

The rest of the node id, overridable per run. These are parametrize **ids**, not values —
`ITERS_ID=iters600`, not `600`:

| Env | What it sets | Default | Other values |
|---|---|---|---|
| `CHUNKS_ID` | chunks prefilled per iteration, 5120 tokens each | `chunks20` | `chunks1`, `chunks2`, `chunks5`, `chunks10`, `chunks_eleven` |
| `ITERS_ID` | iterations per pytest run (the inner loop) | `iters20` | `iters1`, `two_iters`, `ten_iters`, `iters25`, `iters600` |
| `PRELOAD_ID` | prior KV tokens faked into the cache, so the measured chunks run at that KV depth without prefilling up to it | `preload0` (empty cache) | `preload25k`, `preload50k`, `preload95k` — need a golden trace |
| `TRACE_ID` | whether the chunk forward is captured once and replayed | `notrace` | `traced` — Kimi only |

`MESH_ID` exists too but has one value (`mesh-8x4`, an 8×4 mesh over FABRIC_2D) — leave it alone.

A bad combination fails at `stress.sh`'s preflight (`pytest --collect-only`) in seconds, before any
device reset.

Also: `TT_METAL_HOME` (default `/data/$USER/tt-metal`) selects the repo under test — venv, test file,
`PYTHONPATH` — independently of where these scripts live; `launch.sh` prints a `NOTE:` when the two
differ. `STALE_SECS` (240) is the idle time before a running iteration is flagged STALE, `LOGURU_LEVEL`
(INFO) the log level, `PREFLIGHT=0` skips the collect check.

Set these on the `launch.sh` command line, not via `export`. Panes inherit the *tmux server's*
environment, so with a server already running an exported var never reaches them and the run silently
uses defaults. `launch.sh` re-emits them onto all three run panes.

## Files

| File | Purpose |
|---|---|
| `launch.sh` | Entry point — builds the 4-pane window above. |
| `common.sh` | Shared config + helpers (`LOG_DIR`, `MODEL` → `PYTEST_TARGET` / `ENV_VARS`, `INNER_ITERS`, `scan_log_dir`). Sourced, not run. |
| `stress.sh` | Outer loop: `tt-smi -glx_reset` then pytest, `tee` to `log_NN`. No timeout — stays alive on hang for debug. |
| `watch.sh` | Status table (PASS / HANG? / FAIL / RUN / STALE / PENDING), 15s. |
| `watch_multiple_dirs.sh` | Same table for several runs at once: one `<log_name>` arg each, scan depth from `LOOP=`. |
| `tail.sh` | `tail -10` of the newest `log_NN`, 30s. |
| `host_stats.sh` | Host CPU / DRAM / swap, **1 GB hugepage pool, and the live pytest process's memlock/pin/fd limits**, 5s. Snapshots a TSV row to `<log dir>/host_stats.tsv` every 60s (`SNAP_SECS=`). Reads `/proc` + sysfs. |
| `parse_iteration_times.py` | Per-iteration timing (min / avg / max) from any log with `Starting iteration:`. |

Args are `<log_name> [loop_count]` throughout, so any pane can be run standalone against a live run
(`MODEL=KIMI_K2_7 ./watch.sh LOG_name 20`). `REFRESH=` overrides a watcher's interval.

## Why host_stats.sh watches hugepages

On 2026-08-12 an `iters600` soak died on four hosts within 160 ms of each other, twice, always
`SIGBUS` (`TEST_DONE_EXIT=135`) in a **native** thread — faulthandler marked no `Current thread`, and
the Python main thread was mid-forward-pass. `dmesg` on the affected hosts had the answer:

```
tenstorrent 0000:42:00.0: pin_user_pages_longterm failed: -14      (EFAULT, x8 devices)
```

tt-kmd pins a hugepage-backed host buffer per device for DMA; when that pin fails, the process takes
SIGBUS at whatever instruction touched the mapping, with no tt-metal error of any kind. So the pane
tracks `free_hugepages` in the **1 GB** pool (`/sys/kernel/mm/hugepages/hugepages-1048576kB`, one page
per device — 32 on an 8×4 box) and the pytest process's real `RLIMIT_MEMLOCK` / `VmLck` / `VmPin` from
`/proc/PID`. Note `meminfo`'s `HugePages_*` rows describe only the default 2 MB pool and read `0` on a
healthy box — use `Hugetlb:` for the total, which is carved out of DRAM and **not** counted in
`MemAvailable`, so pinning can fail while the host looks 90% free.

The 60s TSV snapshot exists because none of this survives the crash otherwise. Add `dmesg -T | tail`
to your own post-mortem — that message is root-only, so the pane cannot capture it.
