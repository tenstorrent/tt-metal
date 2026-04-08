# LLM utilization sweeps (`tools/sweep`)

This directory hosts **LLM demo** profiling sweeps that drive
[`tools/hw_debug/gen_util_report.py`](../hw_debug/gen_util_report.py) once per grid point
(`model_util_report.csv` per subdirectory).

It is **not** the same as [`tools/tracy/profile_sweeps.py`](../tracy/profile_sweeps.py),
which runs YAML-based eager op tests under `tests/tt_eager/.../sweep_tests/`.

See [DEBUGGING_NOTES.md](DEBUGGING_NOTES.md) for detailed root-cause analysis of issues
encountered during development (NOC trace hangs, `max_seq_len` clipping, etc.).

## Scripts

| Script | Role |
|--------|------|
| `run_llm_util_sweep.py` | Cartesian product over `--seqlens` and `--batch-sizes`; runs `pytest models/tt_transformers/demo/simple_text_demo.py` with `-k "performance and batch-1"`; calls `gen_util_report` per point; optionally aggregates CSVs. |
| `collect_model_util_reports.py` | Copies each `*/model_util_report.csv` under an experiment root to `<experiment>/perf_csvs/<parent_dir_name>.csv`. |
| `sweep_common.py` | Shared helpers: seqlen presets, env setup, pytest argv builder, environment guard. |

## Environment

**Before running**, activate the tt-metal venv and tt-npe environment:

```bash
cd "$TT_METAL_HOME"
source python_env/bin/activate
source ../tt-npe/ENV_SETUP          # sibling of the tt-metal repo
```

The script verifies these are active at startup and exits with an actionable error message if not.

The sweep script also sets (inherited by all children):

- `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000`
- `TT_METAL_DEVICE_PROFILER=1`
- `TT_METAL_PROFILER_SYNC=1`

Optional flags set `HF_MODEL` and `MAX_PREFILL_CHUNK_SIZE`.

## Seqlen presets and `max_seq_len`

Each preset maps to a prompt file and a `max_seq_len` large enough to process
the full prompt without clipping:

| Preset | `max_seq_len` | Notes |
|--------|---------------|-------|
| 128, 256 | 2048 | Short prompts, well within default |
| 1k | 2048 | |
| 2k | 4096 | |
| 4k | 8192 | |
| 8k | 16384 | NOC pass may timeout — use `--noc-timeout` |
| 16k | 32768 | NOC pass will likely timeout |
| 32k+ | 65536+ | NOC pass will likely timeout |

**Important:** Without an adequate `max_seq_len`, the model clips prompts to its
default (1024 for Llama 3.1 8B), meaning all presets above 1k would profile the
same ~824-token workload with identical hardware utilization. The presets ensure
you profile at the intended sequence length.

Use `--max-seq-len <value>` on the CLI to override the preset default for all
grid points.

## Examples

Prefill sweep (defaults: `--num-layers 1`, `--batch-sizes 1`):

```bash
cd "$TT_METAL_HOME"
source python_env/bin/activate
source ../tt-npe/ENV_SETUP
python tools/sweep/run_llm_util_sweep.py \
  --experiment-dir ./experiments/my_prefill \
  --mode prefill \
  --seqlens 256,1k,2k \
  --steady-state
```

Prefill sweep with large seqlens (use `--noc-timeout` for resilience):

```bash
python tools/sweep/run_llm_util_sweep.py \
  --experiment-dir ./experiments/my_prefill_large \
  --mode prefill \
  --seqlens 256,1k,2k,4k,8k,16k \
  --steady-state \
  --noc-timeout 600
```

Decode sweep:

```bash
python tools/sweep/run_llm_util_sweep.py \
  --experiment-dir ./experiments/my_decode \
  --mode decode \
  --seqlens 1k \
  --batch-sizes 1,8,32 \
  --max-generated-tokens 128
```

Validate pytest selection without profiling:

```bash
python tools/sweep/run_llm_util_sweep.py ... --dry-run
```

Collect only (after manual runs):

```bash
python tools/sweep/collect_model_util_reports.py ./experiments/my_prefill
```

## Relation to `profiling_postprocessing.md`

External helpers such as `collect_perf_csvs.sh` / `organize_perf_csv.py` target raw
`ops_perf_results_*.csv` from Tracy. This tree produces **merged**
`model_util_report.csv` files (NOC + perf columns, optional steady-state filter, scaled
`GLOBAL CALL COUNT`). Use `collect_model_util_reports.py` for a flat `perf_csvs/` layout
analogous to the raw-CSV workflow.

## `gen_util_report` flags

- `--steady-state` — last steady-state iteration (shared heuristic with `filter_iter.py`).
- `--single-model-iteration` — keep highest `METAL TRACE REPLAY SESSION ID` only.
- `--noc-timeout <seconds>` — cap the NOC trace pass; if exceeded, report is produced with perf counter data only (NOC columns absent).
- Final CSV scales **`GLOBAL CALL COUNT` ÷ 1024** (merge still uses raw integers).
