# DeepSeek-V3 prefill stress + monitor scripts

## Run it

Kimi K2.7, no trace, 20 iterations per run, 20 runs — copy-paste from anywhere:

```bash
MODEL=KIMI_K2_7 TRACE_ID=notrace ITERS_ID=iters20 TT_METAL_HOME=/data/$USER/tt-metal /data/$USER/tt-metal/models/demos/deepseek_v3_d_p/scripts/launch.sh 20
```

`./launch.sh [loop_count] [log_name]` — both optional (20, and
`LOG_<date>_<host>_<model>_<commit>_loop_<n>`). Logs go to `${LOG_ROOT:-/data/$USER}/<log_name>/log_NN`;
`stress.log` in the same directory captures the complete outer-loop output.

Gemma4 has a model-specific wrapper with local HuggingFace paths, traced 8192-token chunks,
six resident slots, and no hardware resets. Its defaults are 20 chunks × 600 iterations × 5 runs:

```bash
models/demos/gemma4_d_p/scripts/launch.sh 5
```

See [Gemma4 prefill service](../../gemma4_d_p/docs/PREFILL_SERVICE.md) for the full command and checks.
`RESET_BETWEEN_RUNS=0` disables all `tt-smi` resets and stops the outer loop on the first failed run.
It still loads a new model and opens/closes the mesh for each outer run. The original models retain
`RESET_BETWEEN_RUNS=1` by default. `LOG_ROOT` chooses the output root for all three panes.

One detached session `stress_$HOSTNAME` (`SESSION=` to rename), one window, three panes:

```
┌──────────────────────┬──────────────────────┐
│ 0  watch.sh          │ 1  stress.sh         │
│    status table      │    the pytest loop   │
│                      ├──────────────────────┤
│                      │ 2  host_stats.sh     │
│                      │    host CPU / DRAM   │
└──────────────────────┴──────────────────────┘
```

```bash
tmux attach -t "stress_${HOSTNAME}" -r       # ctrl-b z zooms a pane, ctrl-b <arrow> moves
tmux kill-session -t "stress_${HOSTNAME}"    # stop the run
```

## Knobs

`MODEL` is required. It picks the test function, the variant/layers ids, and the model's env vars
(`*_HF_MODEL`, the TTNN cache, `PREFILL_TRACE_DIR`). The TTNN cache var has to be set per `MODEL`:
unset, the `weight_cache_path` fixture silently builds a fresh cache under the HF model dir.

| `MODEL` | test function | variant / layers |
|---|---|---|
| `KIMI_K2_7` | `test_kimi_prefill_transformer_chunked_perf` | `kimi_k2_7` / `L61` |
| `GLM5_3` | `test_glm_prefill_transformer_chunked_no_pcc` | `glm53` / `L78` |
| `GEMMA4` | `test_prefill_stress` | 60 layers, CP8/TP4, traced, six slots |

The rest of the node id, overridable per run. These are parametrize **ids**, not values —
`ITERS_ID=iters25`, not `25`:

| Env | What it sets | Default | Other values |
|---|---|---|---|
| `CHUNKS_ID` | chunks prefilled per iteration, 5120 tokens each | `chunks20` | `chunks1`, `chunks2`, `chunks5`, `chunks10`, `chunks_eleven` |
| `ITERS_ID` | iterations per pytest run (the inner loop) | `iters20` | `iters1`, `two_iters`, `ten_iters`, `iters25`, `iters600` (soak — amortizes the ~50 min weight load over hours of forward instead of minutes) |
| `PRELOAD_ID` | prior KV tokens faked into the cache, so the measured chunks run at that KV depth without prefilling up to it | `preload0` (empty cache) | `preload25k`, `preload50k`, `preload95k` — need a golden trace |
| `TRACE_ID` | whether the chunk forward is captured once and replayed | `notrace` | `traced` |

`MESH_ID` is `torus-xy-8x4` (an 8×4 torus over FABRIC_2D); GLM also collects `fabric2d-8x4`, the only
leg where the KV-dedup fallback gather runs at production shape.

A bad combination fails at `stress.sh`'s preflight (`pytest --collect-only`) in seconds, before any
device reset.

Gemma4 supports `CHUNKS_ID=chunks1|chunks20|chunks32` (8192 tokens per chunk),
`ITERS_ID=iters1|iters12|iters20|iters600`, `TRACE_ID=traced`, `PRELOAD_ID=preload0`, and `MESH_ID=8x4`.
The Gemma4 wrapper supplies these defaults and honors `HF_MODEL`, `HF_HOME`, `HF_HUB_OFFLINE`,
`TT_CACHE_PATH`, and `PREFILL_TRACE_DIR` from the calling shell.

Also: `TT_METAL_HOME` (default `/data/$USER/tt-metal`) selects the repo under test — venv, test file,
`PYTHONPATH` — independently of where these scripts live; `launch.sh` prints a `NOTE:` when the two
differ. `STALE_SECS` (240) is the idle time before a running iteration is flagged STALE, `LOGURU_LEVEL`
(INFO) the log level, `PREFLIGHT=0` skips the collect check.

`TRIAGE=1` arms hang detection: after `HANG_SECS` (120) with no dispatch progress, tt-triage writes
`<log dir>/crash_triage_NN.csv`, the hung pytest is killed, the iteration shows as `HANG`, and the loop
continues with the next iteration's galaxy reset.

Set these on the `launch.sh` command line, not via `export`. Panes inherit the *tmux server's*
environment, so with a server already running an exported var never reaches them and the run silently
uses defaults. `launch.sh` re-emits them onto both run panes.

## Files

| File | Purpose |
|---|---|
| `launch.sh` | Entry point — builds the 4-pane window above. |
| `common.sh` | Shared config + helpers (`LOG_DIR`, `MODEL` → `PYTEST_TARGET` / `ENV_VARS`, `INNER_ITERS`, `scan_log_dir`). Sourced, not run. |
| `stress.sh` | Outer loop: `tt-smi -glx_reset` then pytest, `tee` to `log_NN`. No timeout — stays alive on hang for debug. |
| `watch.sh` | Status table (PASS / HANG? / FAIL / RUN / STALE / PENDING), 15s. Each row splits its wall clock into `load` / `fwd`, and a row still in the weight load shows `loading layer N/M` where a forward row shows `forward_layer_N_end`. |
| `watch_multiple_dirs.sh` | Same table for several runs at once: one `<log_name>` arg each, scan depth from `LOOP=`. |
| `tail.sh` | `tail -10` of the newest `log_NN`, 30s. |
| `host_stats.sh` | Host CPU / DRAM / swap, NIC and weka traffic, **1 GB hugepage pool, and the live pytest process's memlock/pin/fd limits**, 5s. Snapshots a TSV row to `<log dir>/host_stats.tsv` every 60s (`SNAP_SECS=`). Reads `/proc` + sysfs. |
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
