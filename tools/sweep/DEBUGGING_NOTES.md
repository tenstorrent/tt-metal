# LLM Profiling Sweep — Debugging Notes & Key Learnings

This document summarizes the issues encountered while building and running the
`tools/sweep/` parameter sweep infrastructure for profiling LLM workloads
(Llama 3.1 8B) with Tracy, and the changes made to resolve them.

---

## 1. Files created or modified

| File                                          | Status   | Purpose                                                                                           |
| --------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| `tools/hw_debug/gen_util_report.py`         | Modified | Two-pass Tracy profiler (NOC traces + perf counters), merge, metrics extraction                   |
| `tools/hw_debug/util_report_iter.py`        | Created  | Shared steady-state iteration filter (used by both `gen_util_report.py` and `filter_iter.py`) |
| `tools/hw_debug/filter_iter.py`             | Modified | Now imports from `util_report_iter.py` instead of duplicating logic                             |
| `tools/sweep/sweep_common.py`               | Created  | Shared helpers: presets, env setup, pytest argv builder, environment guard                        |
| `tools/sweep/run_llm_util_sweep.py`         | Created  | Main orchestrator: Cartesian sweep over (seqlen, batch_size), calls `gen_util_report` per point |
| `tools/sweep/collect_model_util_reports.py` | Created  | Post-processor: gathers `model_util_report.csv` files into flat `perf_csvs/` directory        |
| `tools/sweep/README.md`                     | Created  | Usage documentation                                                                               |

---

## 2. Changes to `gen_util_report.py`

### 2.1 Background: two-pass profiling (pre-existing)

`gen_util_report.py` runs Tracy twice before our changes:

1. **NOC trace pass** — `python -m tracy ... --collect-noc-traces`
   Produces NOC UTIL (%), DRAM BW UTIL (%), NPE CONG IMPACT (%).
2. **Perf counter pass** — `python -m tracy ... --profiler-capture-perf-counters=all`
   Produces FPU/SFPU/MATH utilization, kernel duration, packet sizes, etc.

The two resulting CSVs are merged on `(GLOBAL CALL COUNT, METAL TRACE REPLAY SESSION ID)`.

### 2.2 Resilient NOC trace pass (`--noc-timeout`)

**Problem:** For large sequence lengths (8k+), the NOC trace pass generates
millions of Tracy zones (14.9M–33.2M observed) and 125MB+ trace files. The
`tt-npe` analysis of this data either hangs indefinitely or gets OOM-killed.

**Fix:** The NOC trace pass is wrapped in `try/except` for both
`subprocess.TimeoutExpired` and `subprocess.CalledProcessError`. If either
occurs, the script prints a warning and continues with perf-counter data only.
The NOC columns (`NOC UTIL (%)`, `DRAM BW UTIL (%)`, `NPE CONG IMPACT (%)`)
will be absent from the report, but all other metrics remain valid.

```
--noc-timeout <seconds>   # optional; caps the NOC pass wall-clock time
```

### 2.3 Steady-state iteration filter (`--steady-state`)

Integrated the `filter_last_steady_state_iteration` heuristic (from
`util_report_iter.py`) to detect repeating OP CODE patterns in the trace and
keep only the last complete iteration. This eliminates warmup/compilation noise.

### 2.4 Additional columns in `model_util_report.csv`

Added to `extract_performance_metrics`:

- `OP TO OP LATENCY [ns]`
- `COMPUTE KERNEL SOURCE`, `COMPUTE KERNEL HASH`
- `DATA MOVEMENT KERNEL SOURCE`, `DATA MOVEMENT KERNEL HASH`
- `% of Total Cycles` (computed)
- `INPUT_0/1/2_LOGICAL_SIZE`, `OUTPUT_0_LOGICAL_SIZE` (computed from PAD columns)
- `INPUT_0/1/2_MEM_CONFIG`, `OUTPUT_0_MEM_CONFIG` (computed from LAYOUT/DATATYPE/MEMORY)

### 2.5 `GLOBAL CALL COUNT` scaling

The raw `GLOBAL CALL COUNT` values in Tracy CSVs are multiplied by 1024 internally.
The final report divides by 1024 so values match logical op indices. The merge
step still uses the raw (unscaled) integers as join keys.

### 2.6 Defensive error handling

- `find_csv_files` raises `FileNotFoundError` if no `ops_perf_results_*.csv` exists.
- `process_cleanup_data` returns `None` when all rows are filtered out; callers
  check for this and either fall back gracefully (NOC data) or raise
  `RuntimeError` (perf data).

---

## 3. The `max_seq_len` root cause

### 3.1 The symptom

The sweep script hung or was OOM-killed during the NOC trace pass for seqlen
presets of 8k and above. Direct runs of `gen_util_report.py` with the same
prompts completed successfully.

### 3.2 The root cause: prompt clipping

`max_seq_len` controls the KV cache allocation and the maximum number of input
tokens the model will process.

**Where the default comes from:** The pytest parametrization in
`models/tt_transformers/demo/simple_text_demo.py` defines the `batch-1` test
case (line 283–301) with `max_seq_len=1024`. The sweep's `-k "performance and batch-1"` filter selects this case, so when no `--max_seq_len` CLI override is
provided, `1024` is used.

**Where clipping happens:** `simple_text_demo.py` line 974 passes `max_seq_len`
to `preprocess_inputs_prefill()` (in `models/tt_transformers/tt/common.py`
line 160) as the `max_prefill_len` parameter. Inside that function:

1. **Line 179:** `max_prefill_len -= max_generated_tokens` — reserves room for
   decode tokens. With `max_seq_len=1024` and `max_generated_tokens=2` (prefill
   default), this gives `max_prefill_len=1022`.
2. **Line 196–229:** If any prompt exceeds `max_prefill_len`, the prompt is
   **left-clipped** (last N tokens kept). The log message reads:
   `"Left-clipping prompts to {max_prefill_len}"`.

So an 8k prompt (7,587 tokens) is clipped to **1,022 tokens** when
`max_seq_len=1024`.

**Impact on the sweep:**

- **Direct runs** (no `--max_seq_len` flag): the model used its 1024 default.
  An 8k prompt (7,587 tokens) was **clipped to ~1,022 tokens**. This produced a
  small trace (~hundreds of thousands of Tracy zones) that `tt-npe` handled
  easily.
- **Sweep runs** (`max_seq_len=16384` for 8k preset): the model processed the
  **full 7,587-token prompt**. Every attention, matmul, and norm op ran on much
  larger tensors. This produced 14.9M–33.2M Tracy zones and 125MB+ trace files,
  overwhelming `tt-npe`'s NOC analysis.

**Confirmation:** Running a direct `gen_util_report.py` command *with*
`--max_seq_len 16384` reproduced the same hang, proving `max_seq_len` was the
sole differentiator.

### 3.3 Why `max_seq_len` must stay

Removing `max_seq_len` from the presets would cause all presets above 1k to
profile the same ~1,022 tokens — identical tensor shapes, identical hardware
utilization. The sweep would be meaningless because:

- FPU/MATH utilization scales with matmul dimensions (O(n²) for attention)
- DRAM BW utilization depends on KV cache size
- NOC traffic patterns change with tensor volume

The presets keep `max_seq_len` at 2x the prompt length (e.g., 8k prompt →
`max_seq_len=16384`) to ensure the full prompt is processed. The `--noc-timeout`
resilience handles the trace volume problem for large presets — you get accurate
perf counter data, just without NOC metrics.

### 3.4 Key takeaway

When comparing profiling results across sequence lengths, always verify that
`max_seq_len` is large enough for the full prompt. If the model clips the input,
you are not profiling the intended workload.

---

## 4. Recommended usage for large sequence lengths

For presets 8k and above, the NOC trace pass will likely time out or fail.
Use `--noc-timeout` to cap it:

```bash
python tools/sweep/run_llm_util_sweep.py \
  --experiment-dir ./exp/prefill \
  --mode prefill \
  --seqlens 256,1k,2k,4k,8k,16k \
  --steady-state \
  --noc-timeout 600   # 10-minute cap per NOC pass
```

This produces complete reports for all points. Smaller presets (256–4k) will
include NOC metrics; larger presets (8k+) will have perf counter data only
(FPU, SFPU, MATH utilization, kernel durations, etc. are all present).
