# Reduced terminal operation attribution

`terminal_ops_v3.json` passes, devices closed cleanly, and the unmodified Tracy
postprocessor joined every host operation to device data. This is an eager
profile of real layers 0/1/3, batch 1, context capacity 128, and the physical
32-row common sampler. It is not full-model throughput or trace latency.

`coverage.json`, produced by `../validate_terminal_profile.py`, proves exact set
equality of all 23,048 host/device/final-report operations, six signposts, and no
skipped rows. Each measured host zone lies inside its signpost pair, each device
duration is positive and preserved, and all four ranks have equal operation
counts. The 19,553,165 recorded Tracy zones explain the large raw capture; this
is one current capture, not accumulated prior device records. No profiler buffer
overflow or source-location exhaustion was reported.

| Region | Operations/rank | Kernel sum, merged report | Per-rank kernel-sum range | Signpost wall window |
| --- | ---: | ---: | ---: | ---: |
| Model | 1,284 | 9.4403 ms | 9.0447–9.2591 ms | 35.8289 ms |
| Split greedy | 65 | 0.46754 ms | 0.46467–0.46623 ms | 3.1836 ms |
| Force argmax | 49 | 2.93959 ms | 2.8590–2.9868 ms | 4.3699 ms |

`tt-perf-report` 1.5.0 merges four-device operations according to its normal
policy; its sum is not a single rank's elapsed duration. Filtered `Device Time`
is in **microseconds**; raw `DEVICE KERNEL DURATION` is in **nanoseconds**.
Kernel sums are consistent with measured wall windows and the separate clean
trace experiment; they do not imply additive end-to-end latency.

Split uses 6.29× less merged kernel time than force argmax. Split's largest op is
`TopkLargeIndicesDeviceOperation` (265.643 µs). Force argmax spends 1,442.905 µs in
all-gather and 1,153.669 µs in argmax. Both return the same expected real-logit
token, 243975. Split is 4.72% of reduced model-plus-split kernel time, so this
screen finds no sampler-dominated token-out path. The existing split choice is
also favored by `../terminal_timing_v1.json`: clean traced token-out is
10.3061 ms versus 12.5737 ms, with identical 14-token feedback streams and no
token/position/page-table host refresh. Those clean reduced timings remain
separate from final all-layer performance evidence.

Advice stays enabled in all primary CSV and human-readable reports. The text
files include the complete advice section; CSV status output has separate
`.console.log` files. Advice disposition:

| Recommendation | Disposition and evidence |
| --- | --- |
| Trace high host-dispatch gaps | Already implemented and independently measured by `terminal_timing_v1.json`: model trace 9.6773 ms and split trace 0.5409 ms. The eager profiler intentionally exposes the same bodies without trace replay so individual operations are attributable. No dispatch-gap total is advertised as achievable additional speedup. |
| Replace accepted HiFi4 with HiFi2 | Rejected for this stage: changes the explicitly preserved decoder fidelity policy. No lower-fidelity result is substituted. |
| L1 input placement, larger reduction block/subblock for existing decoder matmuls | Source triage maps these to inherited decoder MoE/shared projections. No change was made to the accepted decoder; untried feasible layout/config advice remains work. The report identifies exact global operation IDs and shapes. `K=32` reductions have only one K tile, so an unconditional block width of two cannot apply without changing padding/layout. |
| DRAM-sharded LM-head matmul configuration | Source-confirmed candidate, experiment pending: the LM head is vocab-sharded across ranks but DRAM-interleaved within each rank, so this advice is not already satisfied. Largest model operation is `32 × 2560 × 62080`, 809.006 µs, global call count 5514242. No speedup is claimed without a compatible layout, correctness, trace, and clean timing test. |

Source review also identifies a full-model wrapper cost outside generic advice: 32 post-head untilizes (473.651 µs), concat (78.191 µs), tilize (28.452 µs), and copy (28.039 µs) total 608.333 µs. A tiled row-scatter path may preserve the common physical-32-row sampling contract while removing this conversion sequence; this is a proposed experiment, not measured improvement.

Commands (from the repository root; collection was serialized by the parent):

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_TRACE_ALLOC_TRACKING=0 \
TT_METAL_TRACE_ALLOC_TRACEBACKS=0 timeout 3600 python_env/bin/python -m tracy \
  -r -p -v --no-web-server --op-support-count 8192 \
  -o models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy_v3 \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
  --profile-ops-only --iterations 1 \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_ops_v3.json

python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/doc/full_model/validate_terminal_profile.py \
  models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy_v3 \
  models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_perf_v3/ops_perf_results.csv \
  models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_perf_v3/coverage.json
```

For each mapping `model:PERF_MODEL`, `split_greedy:PERF_SPLIT_GREEDY`, and
`force_argmax:PERF_FORCE_ARGMAX`, these exact report commands were run with the
corresponding name and marker (advice enabled by default):

```bash
python_env/bin/tt-perf-report <artifact>/ops_perf_results.csv \
  --start-signpost <MARKER> --end-signpost <MARKER>_END --no-color \
  --csv <artifact>/<name>_perf_report.csv > <artifact>/<name>_perf_report.console.log
python_env/bin/tt-perf-report <artifact>/ops_perf_results.csv \
  --start-signpost <MARKER> --end-signpost <MARKER>_END --no-color \
  --no-summary > <artifact>/<name>_perf_report.txt
```

The copied final CSV is byte-identical to
`../terminal_tracy_v3/reports/2026_11_14_04_03_11/ops_perf_results_2026_11_14_04_03_11.csv`.
Complete raw files remain under the repository symlink `../terminal_tracy_v3`
to `/home/ttuser/qwen38-flash/reference/full_model_raw/terminal_tracy_v3`.
`../raw_artifacts.json` records individual byte counts and SHA-256 digests.
The earlier failed v1/v2 captures remain intact and are not acceptance evidence.
