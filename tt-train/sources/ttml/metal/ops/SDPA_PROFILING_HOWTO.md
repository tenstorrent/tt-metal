# SDPA Kernel Profiling How-To

Quick reference for profiling SDPA forward kernels on Tenstorrent hardware.

> **Status**: forward zones live; backward zones not re-added yet (TODO).

## Prerequisites

- Build with: `./build_metal.sh -b Release --build-tt-train`
- Python env active: `source python_env/bin/activate`

## Step 1: Run a Profile Test

Three opt-in tests are wired up in `tests/ops/sdpa_fw_op_test.cpp`. All three
deliberately use `Q_NH = 1` so that the WH grid has **≤ 1 Q row per core** —
this is what keeps the zone count under budget at long seq lengths (see
[Zone Budget](#zone-budget) below).

| Test                                          | Shape (B,Q_NH,KV_NH,S,D) | Ht  | Max chunks/row | Max zones/row |
|-----------------------------------------------|--------------------------|-----|----------------|---------------|
| `NIGHTLY_ProfileSDPA_FW_TinyLlama_Row`        | 1,1,1,2048,64            | 64  | 16             | 130*          |
| `NIGHTLY_ProfileSDPA_FW_Medium`               | 1,1,1,512,64             | 16  | 4              | 34            |
| `NIGHTLY_ProfileSDPA_FW_Small`                | 1,1,1,128,64             | 4   | 1              | 10            |

`*` = only the **single worst-case row** (the bottom-right one, 16 chunks) trips
slightly over the 125-zone budget; all other rows stay under. See
[Zone Budget](#zone-budget) below.

Each test runs SDPA fw twice (warmup + timed) with no reference comparison so
the profile output is clean.

```bash
./tt-train/run_profiler.sh ./build/tt-train/tests/ttml_tests \
    --gtest_filter="SDPAForwardTest.NIGHTLY_ProfileSDPA_FW_TinyLlama_Row"
```

The `run_profiler.sh` script wraps the binary with `python -m tracy -r -v -p`
and sets the required env vars (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000`).

## Step 2: Collect Output

Results go to `generated/profiler/reports/<timestamp>/`:
- `ops_perf_results_<timestamp>.csv` — per-kernel-launch timing
- `profile_log_device.csv` — raw device profiler data with per-zone timestamps
- `tracy_profile_log_host.tracy` — Tracy GUI file

## Step 3: Analyze

### Option A: Per-zone aggregation script

`tt-train/tools/profiling/analyze_sdpa_profile.py` parses
`profile_log_device.csv` and prints, per zone and per RISC, count / min /
mean / max / sum in ns:

```bash
python3 tt-train/tools/profiling/analyze_sdpa_profile.py \
    generated/profiler/reports/<timestamp>/
```

Pass multiple report dirs to compare runs side-by-side. The script filters
to `sdpa-fw-*` and `fw-*` zones only.

### Option B: Jupyter Notebook
```bash
jupyter lab tt-train/tools/profiling/profiling_analysis_single_exp.ipynb
```
Point it to the `ops_perf_results_*.csv`.

### Option C: Raw CSV
Open `profile_log_device.csv`. Each row has:
```
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles], stat value, Run ID, zone name, zone phase, source line, source file
```

Zone duration = `end` timestamp − `begin` timestamp (in cycles).
Convert cycles → µs: `duration_us = cycles / chip_freq_mhz` (freq is in CSV header).

## Current Zone Layout (forward)

### Outer zones (one per kernel, top of `kernel_main`)

| Kernel                          | Zone Name           | RISC       |
|---------------------------------|---------------------|------------|
| `sdpa_fw_compute_kernel.cpp`    | `sdpa-fw-compute`   | TRISC_MATH |
| `sdpa_fw_reader_kernel.cpp`     | `sdpa-fw-reader`    | BRISC      |
| `sdpa_fw_writer_kernel.cpp`     | `sdpa-fw-writer`    | NCRISC     |

### Inner zones (inside `process_single_row` in compute kernel)

For each iteration of the K-chunk loop (CAUSAL / BALANCED branch only):

| Zone                | Wraps                                                                                                 | Notes                            |
|---------------------|-------------------------------------------------------------------------------------------------------|----------------------------------|
| `fw-qk-mm`          | `mm_block_init_short` + `matmul_block` loop + pack unmasked scores                                    | every chunk                      |
| `fw-mask`           | l1-acc mask stamp (pop+reserve + `pack_tile<true>`)                                                   | **diagonal chunk only**          |
| `fw-sm-max`         | `update_cur_row_max_value` (per-tile row max + reduce + eltwise-max against running max)              | every chunk                      |
| `fw-sm-sub`         | `sub_tiles_bcast_cols(score, cur_max)` loop                                                           | every chunk (inside `apply_exp_inplace_and_find_exp_sum`) |
| `fw-sm-exp`         | `exp_tile` loop on the subtracted scores                                                              | every chunk (inside `apply_exp_inplace_and_find_exp_sum`) |
| `fw-sm-pack-scores` | pack the `Sk_chunk_t` exp tiles back into `cb_attention_weights`                                      | every chunk                      |
| `fw-sm-pack-sum`    | pack exp tiles into `cb_cur_exp_sum[0]` with L1-acc (first overwrite, rest accumulate)                | every chunk                      |
| `fw-pv-mm`          | `matmul_qk_by_v` + pop attention/value CBs                                                            | every chunk                      |
| `fw-online`         | `update_exp_max_diff` + `update_cur_exp_sum_inplace` + `update_cur_mm_out` (correction vs prev chunk) | every chunk except the first     |

Plus once per row, after the K-chunk loop:

| Zone       | Wraps                                                  |
|------------|--------------------------------------------------------|
| `fw-final` | `row_reduce` + (optional `compute_and_pack_lse`) + `recip` + per-tile normalize loop |

The USE_ATTN_MASK (arbitrary mask) branch is **not** instrumented — the per-`n`
matmul_tiles + `apply_mask_on_reg` path is the legacy F9 code and the per-chunk
zones above don't fit it cleanly. Use Causal mode for profiling.

## Zone Budget

**125 zones max per RISC per device read.** Each `DeviceZoneScopedN` inside a
loop consumes one zone slot **per iteration**, so the budget multiplies fast.

The three profile tests above are sized to stay under budget on a 64-core WH.

With the current per-chunk zone set (7 base + optional `fw-mask` on the
diagonal + optional `fw-online` for non-first chunks), the worst-case
zone-count formula is:

```
zones_per_core =   num_chunks_max × 7        # base: qk-mm + 5 sm-* + pv-mm
                 + 1                          # +fw-mask on the diagonal chunk
                 + (num_chunks_max - 1)       # +fw-online on chunks 1..N-1
                 + 1                          # +fw-final after the chunk loop
                 + 1                          # +outer sdpa-fw-compute
                 (all the above × rows_per_core)
```

For `TinyLlama_Row` (Q_NH=1, Ht=64 → 64 rows total → 1 row/core on 64 cores,
**worst row**: 16 chunks):
```
= 16 × 7 + 1 + 15 + 1 + 1
= 130 zones on the worst row's core   ✗ (over 125)
```

This worst row (last row, chunk-count == Ht/Sk_chunk_t = 16) **does**
overflow the 125 budget by 5 zones. In practice the profiler still produces
valid data for every other row (rows 0..62 stay under budget); the analyze
script aggregates across all rows so the means stay accurate, but the
zone-by-zone count from the worst core will be truncated and a couple of its
`fw-online` / `fw-final` zones will be missing from the CSV. If you need
exact per-zone numbers on the worst row, drop one of the optional zones
(`fw-mask` or `fw-online`) or shorten `S` to 1024 (8 chunks max, ~66 zones).

For `Medium` (S=512, 4 chunks): `4×7 + 1 + 3 + 1 + 1 = 34` ✓.
For `Small`  (S=128, 1 chunk):  `1×7 + 1 + 0 + 1 + 1 = 10` ✓.

### What if I want to profile real TinyLlama (Q_NH=32)?

`Q_NH=32, Ht=64 → 2048 rows / 64 cores = 32 rows per core`. Even **per-row**
zones (3 total) would land at `32 × 3 + 1 = 97` — workable but tight, and
per-chunk granularity is out of reach.

Two viable strategies:
1. **`Q_NH=1` trick (what we do now)**: per-row workload is identical to real
   TinyLlama since SDPA heads are independent, so per-chunk costs extrapolate.
2. **Coarse zones on real shape**: gate the per-chunk zones behind a
   compile-time arg, fall back to ≤ 3 per-row zones on production-shape runs.
   Not currently implemented.

## Adding More Zones

```cpp
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("my-outer-zone");
    // ...
    for (...) {
        DeviceZoneScopedN("my-inner-zone");
        // multiplies by loop trip count!
    }
}
```

Before adding a new zone, compute its multiplied cost:
```
zone_slots_added = num_iterations_of_enclosing_loops × runs_per_core
```
and check the new total stays under 125 on **every shape you profile**.

## Interpreting Results: Compute vs Memory Bound

Compare zone durations on the same core:
- **Compute-bound**: `sdpa-fw-compute` >> `sdpa-fw-reader` (reader finishes first, data waits in CB).
- **Memory-bound**: `sdpa-fw-reader` >> `sdpa-fw-compute` (compute stalls on `cb_wait_front`).

Previous (pre-F9/F10) profiling on TinyLlama showed FW is **compute-bound**
(reader ≈ 237µs vs compute ≈ 538µs per KV pair, 2.3× ratio). With F9
(multi-tile chunking) + F10 step 1 (QK^T `matmul_block` + L1-acc mask) the
kernel is **still compute-bound** — see Findings below.

## Findings (forward, F9 + F10 step 1, FP32 DST=4)

Profile run on `NIGHTLY_ProfileSDPA_FW_TinyLlama_Row` (Q_NH=1, S=2048, D=64,
Sk_chunk_t=4, causal mask). Numbers are per-zone wall-clock means in ns on
the **MATH** TRISC, aggregated across all chunks/rows by
`analyze_sdpa_profile.py`. Wall-clock is dominated by MATH; UNPACK / PACK
have noticeable idle gaps (consistent with a MATH-bound kernel).

| Zone                | mean MATH [ns] | share of MATH wall-clock |
|---------------------|----------------|--------------------------|
| `fw-sm-exp`         | ~5,500         | **~55 %**                |
| `fw-qk-mm`          | ~1,300         | ~13 %                    |
| `fw-pv-mm`          | ~1,250         | ~12 %                    |
| `fw-sm-max`         | ~700           | ~7 %                     |
| `fw-sm-sub`         | ~250           | ~2 %                     |
| `fw-sm-pack-scores` | ~400           | ~4 %                     |
| `fw-sm-pack-sum`    | ~250           | ~2 %                     |
| `fw-online`         | ~600           | ~6 %                     |
| `fw-mask`           | ~200           | <1 % (diag chunk only)   |

(Exact numbers depend on chip frequency; rerun the script for current values.)

**Headline**: `exp_tile` (inside `fw-sm-exp`) is by far the largest single
contributor on the MATH TRISC — ~5–6× larger than either matmul block.
Splitting `fw-sm-sub` from `fw-sm-exp` (which is why the doc above lists
them as two separate zones) confirms ~99 % of the previously-bundled
`fw-sm-sub-exp` cost lives in `exp_tile`, not in `sub_tiles_bcast_cols`.

The two FPU matmuls (`fw-qk-mm`, `fw-pv-mm`) are remarkably **balanced
against each other** (~13 % each) and small individually compared to exp.
That means a `matmul_block_no_mop` pipelining trick (which trades MATH
cycles for PACK overlap) would buy single-digit percent in MATH and lose
just as much to setup unless we can also overlap MATH and PACK across
chunks — which requires either BF16 (DST=8) or a substantial kernel
restructure. See `SDPA_OPTIMIZATION_PROPOSALS.md` for the trade-off
discussion.

**Single biggest optimization lever (forward)**: replace `exp_tile` with a
faster, equally accurate exponential (the team's WIP "diff exp") and aim
for ~18 % wall-clock reduction on FW. PACK-thread exp (`exp_packthread_tile`)
is *not* a fit because PACK doesn't have enough headroom in our FP32 path
to absorb the work without becoming the new bottleneck.

## TinyLlama Reference Shape

| Param  | Value | Notes                          |
|--------|-------|--------------------------------|
| B      | 1     | batch                          |
| Q_NH   | 32    | query heads                    |
| KV_NH  | 4     | key/value heads (GQA groups)   |
| S      | 2048  | sequence length                |
| D      | 64    | head dimension                 |
| Ht     | 64    | S / TILE_HEIGHT                |
| qWt    | 2     | D / TILE_WIDTH                 |
| Mask   | Causal| on-the-fly triangular mask     |
