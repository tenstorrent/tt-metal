# Blackhole Concurrent Profiler Milestone 3 Qualification

## Scope and result

Milestone 3 qualifies only the lightweight Blackhole real-time profiler. It
does not integrate sparse MLA, change any model operation, allocate operation
cores, add a tensor D2H fallback, or add Wormhole support.

The result passes the Milestone 0 gates:

- overlap and serialization are proven from raw device ticks;
- ten repeated overlap/serialization runs and five repeated peak-load runs
  have no unexplained loss or protocol error;
- five receiver-pause burst runs retain all 16,384 records per run;
- the measured always-on host-throughput signal is +1.91% by medians and
  +0.76% by means, within the 2.0% enabled-path gate;
- the disabled distribution is faster than the Milestone 1 disabled baseline;
- the production L1 layout and firmware image sizes are unchanged from
  Milestone 2 and remain inside their gates.

The final milestone diff and this evidence must receive exact `APPROVE` from
Claude Opus before commit and publication.

## Hardware and software

Measurements were taken locally, without CI, on the four-chip Blackhole P150b
QuietBox:

- firmware bundle: 19.10.0;
- KMD: 2.10.0;
- IOMMU: enabled;
- NPU clock reported by the benchmark: 1,350 MHz;
- build: `build_Release`;
- branch: `pjosipovic/sparse-mla-topk-kv-gather-overlap`.

This local system is the requested QuietBox. LoudBox and Galaxy were not used.

## Qualification coverage

### Raw device-time overlap and serialization

`RealtimeProfilerSanity.DeviceIntervalsProveOverlapAndSerialization` creates
two one-core sub-devices. After JIT warmup it runs:

1. a long program on stream 0 and a short program on stream 1 concurrently;
2. the same short kernel binary first on stream 0 and then on stream 1, with a
   distinct runtime ID for each invocation and an exact collection between
   them.

The assertions use only `start_timestamp` and `end_timestamp` device ticks.
They require the concurrent intervals to intersect, the short interval to end
before the long interval, and the serialized first interval to end before the
second starts. They also check command queue, stream, invocation, sequence,
schema, source loss, callback loss, and transport loss.

Ten repeated runs produced these raw intervals:

| Run | Overlap stream 0 `[start,end]` | Overlap stream 1 `[start,end]` | Serial stream 0 `[start,end]` | Serial stream 1 `[start,end]` |
| ---: | --- | --- | --- | --- |
| 1 | `[1065293385378,1065335186376]` | `[1065293771981,1065293777446]` | `[1065335955644,1065335960990]` | `[1065336492751,1065336498295]` |
| 2 | `[1065756558719,1065798360076]` | `[1065756939302,1065756944737]` | `[1065799022978,1065799028350]` | `[1065799535105,1065799540672]` |
| 3 | `[1066209373079,1066251174130]` | `[1066209799886,1066209805254]` | `[1066251849221,1066251854728]` | `[1066252413228,1066252418456]` |
| 4 | `[1066662672202,1066704473273]` | `[1066663055784,1066663060952]` | `[1066705156061,1066705161215]` | `[1066705621700,1066705627056]` |
| 5 | `[1067115386386,1067157187557]` | `[1067115782544,1067115788053]` | `[1067158076382,1067158081616]` | `[1067158473775,1067158479026]` |
| 6 | `[1067567685720,1067609486882]` | `[1067568047659,1067568053112]` | `[1067610168852,1067610174387]` | `[1067610758430,1067610763759]` |
| 7 | `[1068021584798,1068063385874]` | `[1068022036556,1068022041933]` | `[1068064106473,1068064112020]` | `[1068064650073,1068064655215]` |
| 8 | `[1068475138041,1068516939203]` | `[1068475599336,1068475604936]` | `[1068517700040,1068517705425]` | `[1068518221409,1068518226727]` |
| 9 | `[1068927357899,1068969158895]` | `[1068927764619,1068927769757]` | `[1068969849101,1068969854321]` | `[1068970318733,1068970324036]` |
| 10 | `[1069379987243,1069421788669]` | `[1069380401046,1069380406631]` | `[1069422460164,1069422465346]` | `[1069422913319,1069422918585]` |

Device-duration distributions, in ticks:

| Interval | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| overlap, long stream 0 | 41,800,996 | 41,801,119.0 | 41,801,147.0 | 41,801,426 |
| overlap, short stream 1 | 5,138 | 5,444.0 | 5,409.8 | 5,600 |
| serialized stream 0 | 5,154 | 5,359.0 | 5,348.2 | 5,547 |
| serialized stream 1 | 5,142 | 5,310.5 | 5,330.4 | 5,567 |

Every run reported zero source and transport loss. No host time was recorded or
used to decide overlap, serialization, or program duration.

### Observer device-cycle characterization

`RealtimeProfilerSanity.QualificationHookMeasuresObserverDeviceCycles` builds
an opt-in Blackhole TRISC0 image when
`TT_RT_PROFILER_QUALIFICATION_HOOK=1` is set before process startup. The test
skips when that process-start precondition is absent. The hook:

- measures full observer-loop cycles and record-handler cycles with device
  timestamps;
- writes four dedicated scratch words immediately after the production
  protocol message;
- reserves those 16 bytes only in the opt-in Blackhole qualification layout;
- is exercised with all eight stream slots live while streams 0 and 1 produce
  the measured workload;
- leaves the production protocol type and production allocation unchanged;
- is compiled completely out of the production image.

Before emitting the firmware define, kernel creation checks the actual cached
dispatch map has all 5,552 bytes and passes the exact reserved scratch address
to TRISC0. The manager independently checks the same mapped boundary and reads
from the resulting stored address. A late environment change therefore fails
before firmware launch instead of writing into the following telemetry region.

The host accessor performs diagnostic L1 reads after an exact device watermark.
It does not read a tensor, fabricate an operation duration, or participate in
normal collection.

Five focused repetitions all handled exactly 128 records. Handler cost ranged
from 13,700 to 13,756 instrumented TRISC0 cycles, or 107.03 to 107.47 cycles per
record. The largest instrumented full eight-stream scan was 950 cycles. Its
same-run minimum synchronized device frequency was 1.350 cycles/ns, yielding a
703.70 ns observed bound. The hook's own timestamp reads are included, so these
are conservative measurements for the tested workload. Every run had zero
source and transport loss.

| Run | Handler cycles | Cycles/record | Max scan ticks | Same-run min cycles/ns | Max scan ns |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13,756 | 107.47 | 940 | 1.349969 | 696.31 |
| 2 | 13,728 | 107.25 | 944 | 1.350000 | 699.26 |
| 3 | 13,711 | 107.12 | 782 | 1.350000 | 579.26 |
| 4 | 13,700 | 107.03 | 950 | 1.350000 | 703.70 |
| 5 | 13,716 | 107.16 | 946 | 1.350000 | 700.74 |

### Paused receiver and bursty completion

`RealtimeProfilerStress.DelayedHostDrainPreservesOneFullBurst` pauses the host
receiver, replays a 4,096-program trace on each of four devices, and starts an
exact collection. It verifies that collection is still pending while the host
receiver is paused, resumes the receiver, and then checks the device-produced
watermark and every record.

Five repeated runs were identical:

- 16,384/16,384 burst records delivered;
- exact collection record count 16,384;
- peak host FIFO 4,097/32,768 pages;
- zero callback, source, and transport loss;
- zero protocol errors and invalid intervals.

The existing peak-load test also passed five repeated 4,096-program replays
across four devices. Every run delivered 16,388 records including warmup,
reported zero transport drops and invalid intervals, and observed peak host
FIFO occupancy from 15 to 85 of 32,768 pages.

Overflow behavior remains covered by the existing stage-specific injection
tests: source and transport overflow are counted, application dispatch does not
stall, and a full-ring watermark remains pending until the control slot can
advance.

### Dispatch runtime coverage

A clean production-layout `unit_tests_dispatch` run selected 412 tests from 46
suites with the qualification environment unset. It completed every suite
through `Variants/Test_BufferCorePageMapping_Iterator`, including all profiler,
stress, randomized-program, and randomized-trace tests. The surrounding agent
session was externally interrupted after the first of the two final
large-buffer suites began, so no successful whole-process exit is claimed.
The two final suites were then run directly on the unchanged binaries: 24 tests
ran, 21 passed, and three sharded variants skipped themselves because they
require a 12x1 grid while this test exposes 8x1. This split run covers every
test selected by the unfiltered binary.

One unrelated order-dependent test remains:
`MeshDispatchFixture.TensixActiveEthTestCBsAcrossDifferentCoreTypes`, which
inspects circular-buffer configuration and is outside every file modified by
this milestone. On the final rebuilt binary, `MeshDispatchFixture.*` reported
10 passed, one platform skip, and this one failure; the failing test then
passed 1/1 when run alone. The qualification environment was unset in both
reproductions. This evidence is recorded explicitly rather than claiming that
all runtime tests passed.

## Lightweight overhead

The 1,024-program fast-dispatch benchmark assigns nonzero runtime IDs on every
timed launch whether or not `--use-realtime-profiler` is passed. The production
overhead comparison intentionally omits that option so it measures the
always-on profiler without also registering an optional test callback or
writing a CSV after the timed section. The timed value is host end-to-end
dispatch throughput through `Finish`; it is not used as an operation duration.

Ten pairs alternated enabled/disabled order:

| Pair | Enabled (us) | Disabled (us) | Enabled - disabled (us) |
| ---: | ---: | ---: | ---: |
| 1 | 56,178 | 54,171 | 2,007 |
| 2 | 52,869 | 52,802 | 67 |
| 3 | 50,654 | 52,846 | -2,192 |
| 4 | 52,915 | 51,761 | 1,154 |
| 5 | 50,189 | 50,616 | -427 |
| 6 | 53,032 | 51,347 | 1,685 |
| 7 | 53,195 | 51,887 | 1,308 |
| 8 | 51,771 | 51,848 | -77 |
| 9 | 50,812 | 49,843 | 969 |
| 10 | 52,845 | 53,409 | -564 |

| Statistic | Enabled (us) | Disabled (us) | Delta |
| --- | ---: | ---: | ---: |
| median | 52,857.0 | 51,867.5 | +1.91% |
| mean | 52,446.0 | 52,053.0 | +0.76% |

Pair differences ranged from -2,192 to +2,007 us, so the mean shift remains
well inside paired-run noise and the median stays within the 2.0% enabled gate.
Against the documented Milestone 1 disabled baseline, the current disabled
median improved by 0.36% and the mean improved by 0.65%; there is no measured
disabled-path regression.

## Production resource gates

The normal image has no qualification define. Current cached images show the
same production sizes as Milestone 2; the opt-in TRISC0 image is shown only to
make its isolation explicit:

| Resource | Production | Gate or comparison |
| --- | ---: | ---: |
| dispatch_s profiler message | 5,536 B | at most 8 KiB |
| qualification-only profiler allocation | 5,552 B | not shipped; dedicated 16 B scratch |
| reserved profiler-core L1 | 262,336 B | no increase over 262,336 B |
| dispatch_s NCRISC | 7,036 B text, 56 B data, 364 B BSS | +1,408 B text from M0; at most +1,536 B |
| dispatch_s TRISC0 | 1,664 B text | +764 B from M0; at most +2,048 B |
| qualification-only TRISC0 | 1,984 B text | not shipped; +320 B over production |
| profiler BRISC | 1,100 B text | +136 B from M0; at most +1,536 B |
| profiler NCRISC | 1,592 B text, 48 B data, 24 B BSS | unchanged from M0 |

The qualification source adds environment checks while constructing the
Blackhole dispatch memory map and host-side kernel. It adds no production
steady-state NOC transaction, polling, wait, L1 allocation, or device
instruction because the extra allocation and firmware define are absent from
the normal path.

## Reproduction commands

```bash
cmake --build build_Release --target tt-metalium-validation-basic -j 8
cmake --build build_Release --target unit_tests_dispatch -j 8

TT_METAL_HOME=$PWD \
  ./build_Release/tt_metal/test/tt-metalium-validation-basic \
  --gtest_filter='RealtimeProfilerProtocol.*:RealtimeProfilerSanity.*'

TT_METAL_HOME=$PWD \
  ./build_Release/tt_metal/test/tt-metalium-validation-basic \
  --gtest_filter='RealtimeProfilerSanity.DeviceIntervalsProveOverlapAndSerialization' \
  --gtest_repeat=10

TT_METAL_HOME=$PWD TT_RT_PROFILER_QUALIFICATION_HOOK=1 \
  ./build_Release/tt_metal/test/tt-metalium-validation-basic \
  --gtest_filter='RealtimeProfilerSanity.QualificationHookMeasuresObserverDeviceCycles' \
  --gtest_repeat=5

TT_METAL_HOME=$PWD \
  ./build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.DelayedHostDrainPreservesOneFullBurst' \
  --gtest_repeat=5

TT_METAL_HOME=$PWD TT_RT_PROFILER_SATURATION_SECONDS=0 \
  ./build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.PeakLoadPreservesRecords' \
  --gtest_repeat=5

TT_METAL_HOME=$PWD TT_RT_PROFILER_DROP_ACCOUNTING=1 \
  ./build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.ConsumerDropAccountingUnderLoad'

TT_METAL_HOME=$PWD TT_RT_PROFILER_SATURATION_SECONDS=0 \
  TT_RT_PROFILER_DROP_ACCOUNTING=1 \
  ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_color=no

TT_METAL_HOME=$PWD TT_RT_PROFILER_SATURATION_SECONDS=0 \
  TT_RT_PROFILER_DROP_ACCOUNTING=1 \
  ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_color=no \
  --gtest_filter='LargeInterleavedReadback/*:LargeShardedReadback/*'

TT_METAL_HOME=$PWD TT_RT_PROFILER_SATURATION_SECONDS=0 \
  TT_RT_PROFILER_DROP_ACCOUNTING=1 \
  ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_color=no \
  --gtest_filter='MeshDispatchFixture.*'

TT_METAL_HOME=$PWD TT_RT_PROFILER_SATURATION_SECONDS=0 \
  TT_RT_PROFILER_DROP_ACCOUNTING=1 \
  ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_color=no \
  --gtest_filter='MeshDispatchFixture.TensixActiveEthTestCBsAcrossDifferentCoreTypes'

# Run ten alternating pairs; disabled runs add the environment variable below.
TT_METAL_HOME=$PWD \
  ./build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 1024 --skip-output-validation

TT_METAL_HOME=$PWD TT_METAL_DISABLE_REALTIME_PROFILER=1 \
  ./build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 1024 --skip-output-validation
```

Final local results:

- Release builds passed;
- production protocol and sanity: 17 passed, qualification hook skipped;
- standalone process-start qualification hook: 5/5 passed;
- end-to-end device intervals: 10/10 passed;
- delayed-host burst: 5/5 passed;
- peak-load stress: 5/5 passed;
- consumer drop accounting: passed with a one-second production window;
- clean split dispatch coverage: every selected test body ran; every profiler
  test passed; final large-buffer suites passed 21 with three platform skips;
  one unrelated, order-dependent CB test failed and passed in isolation;
- no CI job was started.

## Review gate

Run the required review immediately before commit:

```bash
claude --dangerously-skip-permissions --model opus --effort high --print \
  "Review Milestone 3 of the Blackhole concurrent sub-device profiler. Check \
  correctness, bounded/nonblocking behavior, device-time semantics, test-only \
  qualification isolation, lightweight-profiler overhead, tests, evidence, \
  and scope. Respond with exactly APPROVE or CHANGES REQUIRED followed by \
  concrete findings."
```

Only an exact `APPROVE` permits the milestone commit and push.
