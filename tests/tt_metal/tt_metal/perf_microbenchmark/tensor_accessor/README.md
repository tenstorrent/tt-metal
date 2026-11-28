# `tensor_accessor` — CI microbenchmark

Tracks the **on-device `TensorAccessor` address-calculation cost across all supported
tensor topologies** over time in CI, as part of extending the runtime microbenchmark
suite ([issue #46305](https://github.com/tenstorrent/tt-metal/issues/46305)).

Goal: capture the perf of address calculation (`get_noc_addr`) for every supported
`TensorAccessor` distribution shape — **interleaved L1, plus height / width / block
sharding, each in `ROW_MAJOR` and `COL_MAJOR`** — as a portable **cycles-per-call**
metric.

This is the CI/regression packaging of the accessor benchmark gtests
(built as `unit_tests_ttnn_accessor`). It runs the benchmarks with the device profiler,
post-processes the profiler output in Python, and gates one metric per topology
against a per-arch golden.

## What is measured

The `AccessorBenchmarks.GetNocAddr` gtest runs the `get_noc_addr(page_id)` kernel
(`kernels/accessor_get_noc_addr_page_id_benchmark.cpp`) on a **single core** for the
all-static arg config of each topology. Rather than one profiler zone per call, the
kernel wraps **N = 4096** calls in a single `DeviceZoneScopedN` zone, and also emits an
**`EMPTY`** baseline zone (zone enter/exit overhead only). A per-call zone would add
~18 cycles of Tracy overhead — the same order of magnitude as the address calc itself —
which swamps the signal, so the reported metric is:

```
cycles/call = (SHARDED_ACCESSOR_<static> - EMPTY) / N
```

where `<static>` is the all-static config emitted per layout (`0000000` interleaved,
`0000001` sharded). The standard device profiler records the zone cycle costs; the
post-processor averages across cores and subtracts the baseline.

`get_noc_addr` is the address-calculation hot path (page → bank/NOC address). The metric
is keyed by `GetNocAddr.<topology>` and comes only from the standard device profiler, so
it behaves identically on Wormhole and Blackhole.

The **realtime profiler is intentionally not used** here (unlike parts of the op-to-op
latency flow): the metric is a device-cycle count from the standard
`DeviceZoneScopedN` zone, which is portable across all single-card platforms, whereas the
RT profiler is unsupported on some WH setups. The CI command sets only
`TT_METAL_DEVICE_PROFILER=1` (no RT flag, no `TT_METAL_PROFILER_ACCUMULATE`).

### Topologies covered

| Topology (`<topology>`)        | Layout      | Sharding | Orientation | Buffer |
|--------------------------------|-------------|----------|-------------|--------|
| `interleaved_l1`               | interleaved | —        | —           | L1 |
| `sharded_height_row_major`     | sharded     | height   | ROW_MAJOR   | L1 |
| `sharded_height_col_major`     | sharded     | height   | COL_MAJOR   | L1 |
| `sharded_width_row_major`      | sharded     | width    | ROW_MAJOR   | L1 |
| `sharded_width_col_major`      | sharded     | width    | COL_MAJOR   | L1 |
| `sharded_block_row_major`      | sharded     | block    | ROW_MAJOR   | L1 |
| `sharded_block_col_major`      | sharded     | block    | COL_MAJOR   | L1 |

Interleaved accessors have no `dspec()`, so the kernel reads the page count from a
trailing compile-time arg (`INTERLEAVED_LAYOUT` path); sharded layouts read it from the
distribution spec. The same gtest also sweeps per-arg-config micro-suites (`Constructor`,
iterator suites) and a rank sweep for local characterization; only the `GetNocAddr`
topologies above are armed in this CI gate.

## Files

```
accessor_postprocess.py          runs the GetNocAddr suite + parses profiler CSVs into
                                 per-topology cycles/call; with --golden it gates
accessor_golden.json             Wormhole golden (record mode until populated)
accessor_blackhole_golden.json   Blackhole golden (record mode until populated)
```

The benchmark sources are the accessor gtests (built as `unit_tests_ttnn_accessor`). The
parse reuses the same official tracy parser (`tracy.process_device_log` /
`tracy.device_post_proc_config`) as the pre-existing accessor benchmark Python wrapper, so
it stays robust to profiler CSV schema changes.

## Gating

Every measured topology is printed for trending, and the gate fails the job if a
populated metric drifts outside its golden band (`tolerance_pct`). A metric whose golden
value is `null` is in **record mode** (printed, not gated), so new topologies can be
added and armed one at a time.

> **Goldens are armed** from the (Runtime) Performance Tests scheduled runs, per arch
> (`get_noc_addr` cycles are hardware-specific). `tolerance_pct` is 15% — ample headroom
> for board-to-board / firmware drift while still catching real regressions. See
> "Populating / updating the golden" below.

## Where it runs

On the **(Runtime) Performance Tests** pipeline
(`.github/workflows/runtime-perf-tests.yaml`, job `runtime-perf-profiler-tests`), via
`tests/pipeline_reorg/runtime_perf_profiler_tests.yaml` (entry
`runtime_perf_tensor_accessor`). That job uses a dedicated profiler build
(`ENABLE_TRACY=ON`), the same as the op-to-op latency microbenchmark. SKUs:
`wh_n300_civ2`, `bh_p150_perf`. Budget subheading: `perf_profiler`.

## Run locally

Needs a Tracy-enabled build (default; i.e. do **not** pass
`build_metal.sh --disable-profiler`) with tests:

```bash
export TT_METAL_HOME=$(pwd)
cmake --build build --target unit_tests_ttnn_accessor -j

# tools/ must be on PYTHONPATH so the profiler parser resolves
export PYTHONPATH="${TT_METAL_HOME}/tools:${TT_METAL_HOME}:${PYTHONPATH}"

# run the GetNocAddr suite + print per-topology cycles/call (no gate)
TT_METAL_DEVICE_PROFILER=1 \
  python3 tests/tt_metal/tt_metal/perf_microbenchmark/tensor_accessor/accessor_postprocess.py

# run + gate against the golden (compares each populated metric within tolerance_pct)
TT_METAL_DEVICE_PROFILER=1 \
  python3 tests/tt_metal/tt_metal/perf_microbenchmark/tensor_accessor/accessor_postprocess.py \
  --golden tests/tt_metal/tt_metal/perf_microbenchmark/tensor_accessor/accessor_golden.json

# only parse existing result dirs (skip the run)
python3 tests/tt_metal/tt_metal/perf_microbenchmark/tensor_accessor/accessor_postprocess.py --no-run
```

## Populating / updating the golden

The golden files ship with `null` values ("record mode"): until a key is populated the
post-proc only prints the measured value and skips its gate (passes).

To enable a gate:
1. Run the post-proc on the target SKU and read the printed `GetNocAddr.<topology>`
   cycles/call.
2. Copy the measured value into the matching golden file's `golden` block
   (WH → `accessor_golden.json`, BH → `accessor_blackhole_golden.json`).
3. Tune `golden.tolerance_pct` (start loose, tighten once run-to-run spread is known).
