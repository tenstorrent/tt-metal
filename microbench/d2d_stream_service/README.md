# D2D stream-service microbenchmarks

Host wall-clock microbenchmarks for the device-to-device (D2D) stream service
(`ttnn/core/tensor/d2d_stream_service.cpp`):

- **`BM_D2DStreamThroughput`** — single D2D hop, large DRAM-backed tensor, all worker cores.
  Reports sustained GB/s (`throughput_gbps`).
- **`BM_D2DStreamLatency`** — N-stage streaming pipeline (Host → H2D → [D2D]* → output).
  Reports per-iteration end-to-end latency and a per-hop figure.
- **`BM_D2DStreamScenario`** — env-driven sequence of create/transfer/teardown cycles, for
  debugging cross-cycle issues (off unless `D2D_BENCH_SCENARIO` is set).

## Hardware requirements

- **Blackhole** or **UBB Galaxy** under Fast Dispatch (service cores require one of these).
- A 2D mesh with **≥ 2 devices** (throughput) or **≥ num_stages devices** (latency).
- **Latency only:** the H2D front-end pins a host DMA buffer, which needs the IOMMU in
  DMA-translation mode. Without it, latency rows `SkipWithMessage` cleanly.

Rows that don't apply are skipped, not failed.

## Building

The benchmark is a standalone target wired in **without editing the root `CMakeLists.txt`**,
via `CMAKE_PROJECT_TOP_LEVEL_INCLUDES=microbench/inject.cmake` (see `inject.cmake` /
`bench_target.cmake`). It links `TTNN::CPP` + `test_common_libs`, so **tests must be enabled**.

Configure once (from the repo root):

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTT_METAL_BUILD_TESTS=ON -DTTNN_BUILD_TESTS=ON \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=microbench/inject.cmake
```

(Add `-G Ninja` if you prefer Ninja; this tree is configured with Unix Makefiles.)

(If you already configure tt-metal another way, just add the `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=microbench/inject.cmake`
and the two `*_BUILD_TESTS=ON` flags.)

Build the target — **always multi-threaded**; never run two `cmake --build` invocations on
the same build dir at once (they corrupt object/archive files):

```bash
cmake --build build -j "$(nproc)" --target d2d_stream_benchmarks
```

Binary: `build/test/microbench/d2d_stream_service/d2d_stream_benchmarks`

## Running

```bash
BIN=./build/test/microbench/d2d_stream_service/d2d_stream_benchmarks

# List all registered rows
$BIN --benchmark_list_tests

# Throughput sweep: size_index {0..4} x metadata_bytes {0,12} x lease {0,1}
$BIN --benchmark_filter='BM_D2DStreamThroughput'

# Latency sweep: num_stages {2,4,8} x metadata_bytes {0,12} x lease {0,1}
$BIN --benchmark_filter='BM_D2DStreamLatency'

# A single config
$BIN --benchmark_filter='BM_D2DStreamThroughput/size_index:3/metadata_bytes:12/lease:0'

# Export results (file format defaults to json; csv/console also available)
$BIN --benchmark_filter='BM_D2DStreamThroughput' \
     --benchmark_out=results.json --benchmark_out_format=json
```

`size_index` selects the throughput payload shape (0 ≈ 0.5 MB … 4 ≈ 512 MB);
`metadata_bytes` is the inline-metadata blob size in **bytes** (0 = disabled, 12 = default);
`lease` is the fabric-link mode (0 = OWN, 1 = LEASE).

### Sweep runner (per-config, robust)

`run_sweep.sh` runs each config in its own process, retries flaky init, resets the board
after any hang, and writes CSV summaries.

```bash
# both sweeps (default)
bash microbench/d2d_stream_service/run_sweep.sh
# just one
MODE=throughput bash microbench/d2d_stream_service/run_sweep.sh
MODE=latency    bash microbench/d2d_stream_service/run_sweep.sh
```

Outputs `d2d_sweep_results/summary_throughput.csv` + `summary_latency.csv` (+ per-config
JSON/logs). Env overrides: `MODE BIN OUTDIR TIMEOUT WARMUP ITERS CHECK_DATA SIZES STAGES MDS LEASES RESET_CMD`.

## Reading the output

Ignore Google Benchmark's `Time`/`CPU` columns — they include service construction. Use the
custom counters:

- **Throughput:** `throughput_gbps`, `payload_bytes`, `data_ok` (1 = receiver data verified
  correct, 0 = FAIL, -1 = unchecked).
- **Latency:** `total_avg_us`, `total_p50_us`, `total_p99_us`, and
  `per_hop_simple_us` = `total_avg / (num_stages - 1)`. Sweep `num_stages` and take the
  slope across N to cancel fixed H2D/dispatch overhead and isolate marginal hop latency.

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `D2D_BENCH_REUSE_SUBMESH` | 1 | Reuse one persistent submesh set across cycles. **Keep on** — per-cycle submesh recreation triggers a fabric hang/corruption (see `notes/d2d_throughput_sweep_hang.md`). Set 0 only for A/B. |
| `D2D_BENCH_CHECK_DATA` | 0 | Throughput: fill sender backing, verify receiver matches; reports `data_ok`. |
| `D2D_BENCH_WARMUP` | 5 | Warmup iterations. |
| `D2D_BENCH_TPUT_ITERS` | 20 | Throughput measured iterations. |
| `D2D_BENCH_LAT_ITERS` | 50 | Latency measured iterations. |
| `D2D_BENCH_VERBOSE` | off | Flushed `[d2d-bench]` stderr phase tracing (localizes a hang). |
| `D2D_BENCH_STEP` | 0 | Drive OWN-mode transfers one-at-a-time (debug; GB/s not representative). |
| `D2D_BENCH_SINGLE_CORE` | 0 | Use a 1×1 worker grid instead of the full compute grid. |
| `D2D_BENCH_MD_OVERRIDE` | 0 | Override the metadata blob size (bytes) on metadata rows. |
| `D2D_SVC_TRACE` | off | Service-side trace of `create_pair`/teardown (in `d2d_stream_service.cpp`). |

Scenario runner (debug; inert unless `D2D_BENCH_SCENARIO` set):
`D2D_BENCH_SCENARIO="0,0,0,0,12"` (per-op metadata_bytes), `D2D_BENCH_SCENARIO_SIZE` (size_index
for all ops), `D2D_BENCH_SCENARIO_SIZES` (per-op size list), `D2D_BENCH_SCENARIO_LEASE`.

## Troubleshooting

- **Hang.** A hard kill mid-transfer wedges an ethernet core; recover with `tt-smi -r`. Use
  `D2D_BENCH_VERBOSE=1` to see the last phase before the hang. (With `D2D_BENCH_REUSE_SUBMESH=1`,
  the known cross-cycle hang is avoided.)
- **`EXIT 134` at startup, "ethernet core … timed out".** Flaky eth init on some boards —
  `tt-smi -r` and retry (the sweep runner does this automatically).
- **"Sysmem mapped at unexpected NOC address … stale process".** A crashed/leftover process
  held sysmem. `pkill -9 -f d2d_stream_benchmarks`, then `tt-smi -r`.

See `notes/d2d_microbenchmarks.md` and `notes/d2d_throughput_sweep_hang.md` for deeper detail.
