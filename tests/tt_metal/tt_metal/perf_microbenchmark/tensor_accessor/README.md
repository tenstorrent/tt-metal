# `tensor_accessor` — CI microbenchmark

Tracks the **on-device `TensorAccessor` address-calculation cost across all supported
tensor topologies** over time in CI, as part of extending the runtime microbenchmark
suite ([issue #46305](https://github.com/tenstorrent/tt-metal/issues/46305)).

Goal: capture the perf of address calculation (`get_noc_addr`) for every supported
`TensorAccessor` topology — **interleaved L1, interleaved DRAM, 1D/2D/ND sharded L1,
sharded DRAM** — as **full-chip** tests (the kernel runs on every Tensix core).

This is the CI/regression packaging of the accessor benchmark gtests
(built as `unit_tests_ttnn_accessor`). It runs the benchmarks with the device profiler,
post-processes the profiler output in Python, and gates one metric per topology
against a per-arch golden.

## What is measured

The `AccessorFullChipBenchmarks.GetNocAddr` gtest runs the `get_noc_addr(page_id)`
kernel (`kernels/accessor_get_noc_addr_page_id_benchmark.cpp`) on the **full Tensix
compute grid** for one pinned all-static arg config per topology, wrapping each call in
`DeviceZoneScopedN`. The standard device profiler records the per-core zone cycle cost;
the post-processor averages across cores.

`get_noc_addr` is the address-calculation hot path (page → bank/NOC address). The
metric is `average device cycles`, keyed by `GetNocAddr.<topology>`, and comes only
from the standard device profiler, so it behaves identically on Wormhole and Blackhole.

The **realtime profiler is intentionally not used** here (unlike parts of the op-to-op
latency flow): the metric is a device-cycle count from the standard `{BRISC}-KERNEL`-style
`DeviceZoneScopedN` zone, which is portable across all single-card platforms, whereas the
RT profiler is unsupported on some WH setups. The CI command sets only
`TT_METAL_DEVICE_PROFILER=1` (no RT flag, no `TT_METAL_PROFILER_ACCUMULATE`).

### Topologies covered (full chip)

| Topology (`<topology>`) | Layout | Buffer | Rank |
|-------------------------|--------|--------|------|
| `interleaved_l1`   | interleaved | L1   | 2D |
| `interleaved_dram` | interleaved | DRAM | 2D |
| `sharded_l1_1d`    | sharded     | L1   | 1D |
| `sharded_l1_2d`    | sharded     | L1   | 2D |
| `sharded_l1_nd`    | sharded     | L1   | ND (5) |
| `sharded_dram_2d`  | sharded     | DRAM | 2D |
| `sharded_dram_nd`  | sharded     | DRAM | ND (5) |

Sharded DRAM buffers shard across DRAM banks (grid derived from
`mesh_device->dram_grid_size()` at runtime). Interleaved accessors have no `dspec()`,
so the kernel reads the page count from a trailing compile-time arg (`INTERLEAVED_LAYOUT`
path). The single-core, per-arg-config micro-suites (`Constructor`, `GetNocAddr` over the
15 static/dynamic configs, iterator suites) still exist in the same gtest for local
characterization; only the full-chip `GetNocAddr` suite is wired into this CI gate.

## Files

```
accessor_postprocess.py          runs the full-chip suite + parses profiler CSVs into
                                 per-topology average cycles; with --golden it gates
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
> (`get_noc_addr` cycles are hardware-specific). Two consecutive scheduled runs agreed to
> <0.05% on every metric, so `tolerance_pct` is 15% — ample headroom for board-to-board /
> firmware drift while still catching real regressions. See "Populating / updating the
> golden" below.

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

# run the full-chip suite + print per-topology metrics (no gate)
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
   average cycles.
2. Copy the measured value into the matching golden file's `golden` block
   (WH → `accessor_golden.json`, BH → `accessor_blackhole_golden.json`).
3. Tune `golden.tolerance_pct` (start loose, tighten once run-to-run spread is known).
