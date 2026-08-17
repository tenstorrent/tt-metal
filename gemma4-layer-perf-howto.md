# gemma4 per-chunk layer perf — how to run it

Per-chunk cost of one global and one sliding decoder layer during a 256k chunked prefill.

- Test: `models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n`
- Driver: `models/demos/gemma4/tests/sweep_layer_perf.py`
- Output: `generated/gemma4_layer_perf/<run_id>/`

## Env

```bash
source python_env/bin/activate
export PYTHONPATH=$(pwd) TT_METAL_HOME=$(pwd) HF_HUB_OFFLINE=1 \
       HF_HOME=/localdev/svuckovic/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/localdev/svuckovic/huggingface/tt_cache/google--gemma-4-31B-it
```

## 1. Read the existing reports

`chunkNNN/global.perf.txt` already *is* tt-perf-report output. To re-slice by hand:

```bash
R=generated/gemma4_layer_perf/full_256k
tt-perf-report --start-signpost gemma4-layer-global-chunk3-start \
               --end-signpost   gemma4-layer-global-chunk3-stop \
               $R/chunk003/ops_perf.csv
```

Gotchas:

- Signposts use the **bare** index (`chunk3`); directories are **zero-padded** (`chunk003/`).
- Each chunk CSV holds **both** layer types — slice twice, swapping `global`/`sliding`.
  `--print-signposts <csv>` lists all four.
- `--csv out.csv` **replaces** the printed table instead of adding to it. Omit it to read a table.

Start from `timings.csv` (all 128 cells), `model_estimate.csv` (10x global + 50x sliding),
and `validation/VALIDATION.md` (why the numbers are trustworthy, and the 1.08 correction).

## 2. Generate new tracy logs

**The device is exclusive** — stop any running sweep first, by process *group*, or the driver's
pytest child is orphaned and keeps holding the mesh:

```bash
kill -- -$(ps -o pgid= -p $(pgrep -f "sweep_layer_[p]erf" | tail -1) | tr -d ' ')
```

The `[p]` bracket stops the pattern matching its own command line.

Via the driver:

```bash
# profiled per-op tables for specific chunks, into the existing run dir
python -m models.demos.gemma4.tests.sweep_layer_perf --phase profile --chunks 3,17,42 --run-id full_256k

# depth curve only, no profiler — all 64 chunks in ~3 min
python -m models.demos.gemma4.tests.sweep_layer_perf --phase timings --run-id full_256k

# preview without executing
python -m models.demos.gemma4.tests.sweep_layer_perf --chunks 0,8 --dry-run
```

`--chunks` takes `all`, a list `0,8,42`, or a range `0:16`. Omit `--run-id` for a fresh
timestamped dir; reuse it to add in (completed chunks are skipped, `--force` redoes them).

One cell directly, no driver:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -m pytest \
  'models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk7-global-4x8]' -sv
```

Swap `-global-` for `-sliding-` or `-both-`; use `chunkall` for the whole sweep in one process.
Raw CSV lands in `generated/profiler/reports/<ts>/`. No `--timeout=0` needed — the test carries
`@pytest.mark.timeout(7200)`.

## Env knobs

| var | effect |
|---|---|
| `GEMMA4_PERF_WARMUP_ITERS` | warm replays per cell (default 5; driver uses 2 when profiling) |
| `GEMMA4_PERF_KV_FILL` | how the prefix gets into the KV cache: `random` (default, flat cost), `replay` (exact, costs N replays), `none` (zeroed — reads ~15% low at depth) |
| `GEMMA4_PERF_CHUNKS=0,4,8` | narrow the `chunkall` param to a list |
| `GEMMA4_PERF_CONTEXT_LEN` | context length, default 262144 (sets the chunk count) |

## Watch out

- Profiled runs cost `~294s + 24s x chunk_index` — chunk 63 alone is ~30 min. The ring gather
  emits more ops with depth, so the profiler CSV grows (25MB at chunk 1, 50MB at chunk 7).
- Each profiled run leaves ~2.3GB in `generated/profiler/reports/<ts>/`, of which only the
  22MB+ `ops_perf_results_*.csv` matters; the driver copies that into the chunk dir. Delete
  `profile_log_device.csv` and `tracy_profile_log_host.tracy` from finished report dirs.
- A cold `TT_CACHE_PATH` makes this (and the canonical 256k test) skip. Populate with
  `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` plus `--timeout=0`; it needs ~60GB of disk and minutes.
