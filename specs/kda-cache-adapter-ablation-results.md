# KDA cache-adapter ablation results

Date: 2026-08-25
Bead: `tt-metal_tracker-858.3.9.1`
Branch: `kda-investigation/cache-adapter-ablation`
PR7 base: `479d04f6e38f45b0290eff6bdf5c73abe5e1b4e9`

## Verdict

The contract layout is viable with preallocated `ttnn.to_memory_config` adapters.
All three required real Kimi-K3 `B=1,T=5120` layouts passed CPU-oracle accuracy,
exact real-state round trip, exact patterned ordering round trip, and physical
segment checks on eight Blackhole devices.

For the one-way prefill-to-decode handoff, steady export costs 35-115 µs per
KDA layer, or 0.364-1.136% of the measured layer wall time. Import costs
45-155 µs, or 0.462-1.532%. Convolution resharing dominates and grows with
the SP replication factor. Use the adapters at the disaggregation boundary;
do not change the contract. A fully direct layout is not available in PR7 because
the recurrent scan and convolution consumer explicitly require interleaved input,
and the recurrent producer requires interleaved output.

## Steady trace-wall ablation

Adapter cells are median microseconds from 20 synchronized samples of 100 trace
replays. Parentheses give p95. The layer column remains in milliseconds.
`combined %` is combined median divided by the same layout's real KDA-layer median.

| Layout | Layer (ms) | Export S (µs) | Export conv (µs) | Export combined (µs) | Export % | Import S (µs) | Import conv (µs) | Import combined (µs) | Import % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.7171 | 12.53 (12.61) | 27.37 (27.40) | 35.40 (36.27) | 0.364% | 11.64 (11.65) | 36.24 (36.27) | 44.92 (45.19) | 0.462% |
| SP2xTP4 | 9.6486 | 16.34 (16.39) | 48.95 (48.98) | 61.72 (62.00) | 0.640% | 16.23 (16.28) | 69.75 (69.87) | 82.58 (82.64) | 0.856% |
| SP4xTP2 | 10.1103 | 23.65 (30.00) | 94.33 (94.39) | 114.82 (115.10) | 1.136% | 23.79 (23.95) | 133.95 (134.05) | 154.89 (155.02) | 1.532% |

A bidirectional export-plus-import transition is 80.3, 144.3, and 269.7 µs
per layer respectively: 0.827%, 1.496%, and 2.668% of one layer. If all 69 KDA
layers are converted serially, the measured medians imply one-way export totals
of 2.44, 4.26, and 7.92 ms; import totals are 3.10, 5.70, and 10.69 ms.
These totals exclude transport and assume no overlap.

## Accuracy and identity

| Layout | Output PCC | Recurrent PCC per SP rank | Convolution PCC per SP rank | Adapter identity |
| --- | ---: | ---: | ---: | --- |
| SP1xTP8 | 0.999902 | 0.999799 | 0.999999 | real and patterned bit-identical |
| SP2xTP4 | 0.999903 | 0.999925 | 0.999999 | real and patterned bit-identical on both SP ranks |
| SP4xTP2 | 0.999905 | 0.999926 | 0.999999 | real and patterned bit-identical on all four SP ranks |

All values exceed the existing 0.98 Kimi-K3 acceptance threshold. The patterned
state distinguishes S coordinates and convolution branch/head/half ordering; no
permutation is needed because ND sharding enumerates the existing logical width
in the contract order.

## Physical layout and traffic

The S memory config uses FP32 tile ND shards `[1,1,128,32]`. Hardware reports a
4096-byte aligned page and four pages per shard, exactly 16,384 bytes. The
convolution config uses BF16 row-major ND shards `[1,3,64]`; hardware reports a
128-byte page and three pages per shard, exactly 384 bytes. Both use DRAM and
`ROUND_ROBIN_1D`, analogous to the existing KV-cache allocation at
`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:1005`.

There are always 384 unique S segments (6,291,456 bytes) and 576 unique
convolution segments (221,184 bytes). SP replication makes the physical state
6,512,640, 13,025,280, and 26,050,560 bytes for SP1, SP2, and SP4. One conversion
reads and writes that state, so physical DRAM traffic is twice those values.

## Cold and allocation costs

Allocation of both contract and native destinations was 0.573 ms (SP1), 1.466 ms
(SP2), and 0.565 ms (SP4). First synchronized calls for shapes absent from the JIT
cache took 408-523 ms per primitive on SP2/SP4; SP1 reused kernels from the smoke
run and took 2.5-6.5 ms. These are compile/cache effects, not steady adapter
latency. Production must preallocate buffers and warm the four direction/shape
programs before serving a handoff.

## Direct-layout ablation

A complete direct-layout variant cannot be instantiated without changing PR7's
operation contracts:

- `recurrent_chunk_scan` rejects sharded recurrent output and initial state
  (`recurrent_chunk_scan_device_operation.cpp:65`, `:90`).
- the shared validators define those failures as “must use interleaved memory”
  (`factory/kda_factory_utils.cpp:90`, `:94`).
- `qkv_causal_conv1d_silu` rejects a sharded convolution history
  (`qkv_causal_conv1d_silu_device_operation.cpp:30`).
- the final convolution-state slice/halo can accept a caller-selected output
  memory config (`models/demos/deepseek_v3_d_p/tt/kda/ops.py:100`, `:114`), but
  PR7 requests interleaved DRAM (`kda.py:319`, `:324`) and converts any cache back
  to interleaved before consumption (`kda.py:314`). This partial producer option
  cannot eliminate the reverse adapter and was not treated as an end-to-end
  direct result.

Therefore the measured adapter path is the smallest supported mechanism. A
future producer-layout experiment should target convolution only and preserve the
same contract; S requires kernel contract work before it can be measured directly.

## Tracy attribution

The targeted SP4xTP2 profile identifies `CopyDeviceOperation` as the adapter
kernel. Across device rows with counters:

| Direction | State | Rows | Device-kernel min / median / max (µs) |
| --- | --- | ---: | ---: |
| interleaved → ND | S | 72 | 15.464 / 16.410 / 17.484 |
| ND → interleaved | S | 64 | 15.239 / 16.277 / 17.656 |
| interleaved → ND | convolution | 72 | 90.439 / 90.473 / 91.193 |
| ND → interleaved | convolution | 64 | 126.054 / 128.816 / 132.357 |

The convolution rows explain both the combined-path dominance and the larger
import penalty. The profiler run itself is instrumentation-heavy; synchronized
unprofiled trace wall is the latency verdict.

## Reproduction and artifacts

Build:

```bash
./build_metal.sh --build-type Release --enable-ccache
```

Full matrix:

```bash
env TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
  KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  KDA_ADAPTER_TIMING_SAMPLES=20 KDA_ADAPTER_TIMING_REPS=100 PERF_REPS=10 \
  scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_cache_adapter_perf.py -q -s
```

Result: 3 passed, 0 skipped, 59.07 s, `SAFE_PYTEST_RESULT: PASS`.

Targeted Tracy:

```bash
env TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
  KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  KDA_ADAPTER_TIMING_SAMPLES=1 KDA_ADAPTER_TIMING_REPS=1 PERF_REPS=1 \
  scripts/run_safe_pytest.sh --profile \
  'models/demos/deepseek_v3_d_p/tests/kda/perf/test_cache_adapter_perf.py::test_kimi_k3_cache_adapter_ablation[blackhole-SP4xTP2-fabric_1d]' -q -s
```

Result: 1 passed, profiler generated, `SAFE_PYTEST_RESULT: PASS`.

Artifacts:

- `specs/kda-cache-adapter-ablation-safe.log`: complete full-matrix log (local evidence, not committed).
- `specs/kda-cache-adapter-ablation-results.jsonl`: three raw result records.
- `specs/kda-cache-adapter-ablation-tracy.log`: complete targeted profiler log (local evidence, not committed).
- `generated/profiler/reports/2026_08_25_22_32_56/ops_perf_results_2026_08_25_22_32_56.csv`: per-op Tracy data (local evidence, not committed).

Observed non-fatal warnings were the existing Python/SWIG/Pydantic deprecations,
suboptimal 4352-byte fabric packet warnings during SP runs, Tracy's mixed-column
`DtypeWarning`, and normal profiler JIT compile pragma messages. No hang, reset,
accuracy failure, or adapter fallback occurred.
