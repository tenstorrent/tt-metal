# Realtime profiler clean-room Milestone 3 evidence

**Target:** local Blackhole QuietBox, device 0, worker dispatch, one hardware
command queue

**Status:** implementation and local qualification complete

## Delivered end-to-end path

Milestone 3 completes the bounded device-to-host transport for the correlated
device intervals introduced in Milestone 2:

- dispatch_s consumes at most one completed record at each service point and
  never waits for transport space or acknowledgement;
- the resident observer publishes completed records through a fixed 64-entry
  SPSC ring and drops with a named counter when that ring is full;
- dispatch_s forwards a record through the existing reserved-core and D2H
  transport only when its one-entry mailbox is available;
- the host exposes the correlated device endpoints and a loss snapshot to
  Python and C++ consumers;
- shutdown stops the observer and performs a cycle-bounded, record-bounded
  drain before publishing terminal loss counters;
- the final inactive capability and loss snapshot remains queryable after
  device close.

Both interval endpoints are raw Blackhole device wall-clock ticks. Missing or
dropped records remain missing. There is no tensor/D2H timing readback, host
timestamp substitution, or other timing fallback.

No model, Sparse MLA, TopK, all-gather, scheduler, core-allocation, CI,
LoudBox, Galaxy, or Wormhole feature change is included.

## Local qualification

The following focused suites pass on the local QuietBox:

| Qualification | Result |
| --- | --- |
| Protocol, capability, and sanity | Shared layout, inactive reasons, callback/API behavior, and final capability snapshot pass. |
| Concurrent device path | Nine active tests pass; two compile-time adversarial variants are intentionally skipped in the normal image. |
| Reset-window variant | Passes in a fresh process with `TT_METAL_REALTIME_PROFILER_TEST_RESET_PAUSE=1`. |
| Observer-cycle variant | Passes in a fresh process with `TT_METAL_REALTIME_PROFILER_TEST_OBSERVER_CYCLES=1`. |
| Repeated final-record delivery | Ten consecutive fresh executions pass, including inactive retained state and zero terminal timeout/loss. |
| Non-trace burst pressure | 4,097 ordinary launches preserve every record. |
| Full 60-second peak pressure | 23,314,436 records across four devices, zero invalid intervals, zero ring-full waits, and a 64-page peak in the 32,768-page host FIFO. |
| Full 60-second callback drop accounting | 24,002,560 records are accounted for exactly by both a borderline and deliberately slow consumer. |
| Device-loss callback visibility | Descriptor pressure and actual reserved-core ring-full pressure each followed by a sentinel expose the same nonzero cumulative aggregate in the record, `device_loss` batch span, and detailed capability snapshot. |
| Python device-time consumer | Production TopK perf check passes using the profiler's device interval: 1.686 ms measured versus 1.684 ms target. |

The instrumented observer measured 24 cycles per idle loop, 90 cycles with one
active stream, and 464 cycles with all eight streams active. The worker-boundary
samples measured a median 568-cycle GO-delivery lead and 370-cycle observer
polling lag. These are device-cycle measurements and are not used to alter the
reported interval.

The approved direct GO-tail benchmark was rerun on matched M3 source with the
realtime profiler disabled and enabled. Each arm contains the warmup plus 32
measured launches:

| Arm | n | Min | Median | p95 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 disabled | 33 | 801 | 816 | 1,228 | 845.42 | 1,261 |
| M3 enabled | 33 | 953 | 977 | 1,372 | 1,014.70 | 1,530 |

The enabled increment is 161 median device cycles over the matched disabled
arm, passing the accepted 300-cycle gate. The endpoint is the outer dispatch_s
GO-command zone end, so it includes the accepted-launch descriptor work and
the post-command bounded transport service. Raw samples are in
[realtime_profiler_clean_room_milestone3_device_cycles.tsv](realtime_profiler_clean_room_milestone3_device_cycles.tsv).

## Bounded resource use

The shared profiler message is 2,956 bytes. The final normal firmware images
remain within their locked limits:

| Component | Baseline text | M3 text | Delta | Result |
| --- | ---: | ---: | ---: | --- |
| dispatch_s NCRISC | 3,372 B | 4,268 B | +896 B | PASS; limit +1,536 B |
| Resident observer TRISC0 | — | 1,848 B | — | PASS; limit 2,048 B |
| Reserved profiler BRISC | 828 B | 1,040 B | +212 B | PASS; limit +1,024 B |

The completed-ring capacity is 64 and each shutdown drain is limited to 64
records and 1,000,000 device cycles. Ring or descriptor pressure increments a
specific loss counter instead of delaying program dispatch.

## Sparse MLA Long test status

The exact GLM 5.2 Long Sparse MLA perf case was started with the realtime
profiler enabled. Profiler activation, Python binding, callback delivery, and
device interval decoding succeeded. The model then failed before producing a
performance result in `_gather_kvpe_prefix`: the SP=1 `ttnn.slice` inherited a
DRAM ND-sharded output configuration with shard coordinate `(0, 1)`, which is
invalid because a DRAM shard y-coordinate must be zero.

That failure reproduces in model code introduced by the main-branch Sparse MLA
high-bandwidth all-gather change and is outside this profiler-only diff. A dense
Long control is independently blocked by a ring-MLA static circular-buffer L1
allocation of 1,595,904 bytes exceeding the available 1,572,864 bytes. No
Sparse MLA speedup is claimed from this QuietBox run, and neither model blocker
is hidden by a host or D2H fallback.

## Reproduction commands

```bash
cmake --build build --target ttnn unit_tests_dispatch -j8

build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerProtocol.*:RealtimeProfilerCapability.*:RealtimeProfilerSanity.*:RealtimeProfilerConcurrentDevicePath.*'

TT_METAL_REALTIME_PROFILER_TEST_RESET_PAUSE=1 \
  build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerConcurrentDevicePath.NextLaunchCannotCrossClearToGenerationWindow'

TT_METAL_REALTIME_PROFILER_TEST_OBSERVER_CYCLES=1 \
  build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerConcurrentDevicePath.MeasuresIdleOneAndAllStreamObserverCycles'

TT_METAL_REALTIME_PROFILER=1 python -m pytest -m perf \
  'models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf[blackhole-glm_5_2-long-sparse-kv_bf16-fabric2d-quietbox_sp1xtp4]' -s
```

The first four commands pass locally. The final command reaches the unrelated
Sparse MLA model failure described above. `git diff --check` passes.
