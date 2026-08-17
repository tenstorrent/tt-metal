# Clean-Room Realtime Profiler Milestone 0 Evidence

## Environment

| Item | Value |
| --- | --- |
| Branch baseline | `5fa57f6d18bdc46e0d623a6f86a1b7d4bfcc547b` |
| Host | `bh-qb-12-special-pjosipovic-for-reservation-68264` |
| System | four-chip Blackhole P150b QuietBox |
| KMD | 2.10.0 |
| Firmware bundle | 19.10.0 |
| IOMMU | enabled |
| AI clock during profiler test | 1350 MHz reported by runtime |
| Build | `build_Release` |

No CI or LoudBox job was launched. All hardware evidence is local QuietBox
evidence.

## Static protocol map

The clean-baseline source audit fixes these implementation anchors:

| Concern | Authoritative baseline path | Decision |
| --- | --- | --- |
| Go issuance | `cq_dispatch_subordinate.cpp::process_go_signal_mcast_cmd` | Copy every profiler field before aligned go storage overwrites the 16-byte command; publish locally immediately before the first go write. |
| Protected NOC state | The non-device-print multicast path calls `wait_for_workers` between `cq_noc_async_write_init_state` and `cq_noc_async_write_with_state` | Never perform profiler NOC service inside `wait_for_workers` or that stateful-write pair. |
| Completion source | `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX`, streams 48–55 on Blackhole | TRISC0 observes the same 17-bit counters used by dispatch worker waits. |
| Local reset | `cq_dispatch.cpp::process_wait`, `WAIT_STREAM|CLEAR_STREAM` | Supported reset owner; it waits, clears, publishes generation/reset epoch, then command order reaches the next subordinate notification. |
| Excluded reset | `cq_dispatch_subordinate.cpp::process_dispatch_s_wait_cmd` | This path asserts distributed dispatch and is excluded by capability gating. |
| Next-go carrier | `process_notify_dispatch_s_go_signal_cmd` plus dispatch_s's per-stream sync-semaphore wait | Prevents the first new-generation descriptor from entering the clear/publication window. |
| Baseline identity path | `program_id_fifo_append/pop` and the 32-entry fields in `realtime_profiler_msg_t` | Delete completely; runtime ID moves into the existing 16-byte go command. |
| Trace replay | Captured go `wait_count` plus trace counter rebasing/reset contributions | Captured commands carry zero profiler fields; replay is unprofiled without becoming an init-time capability reason. |

The ABI arithmetic is fixed: `CQDispatchGoSignalMcastCmd` occupies 15 payload
bytes inside the 16-byte `CQDispatchCmd`. Replacing the 32-bit stream with an
8-bit stream, 8-bit worker contribution, and 16-bit runtime ID preserves that
size. Host validation bounds Blackhole dispatch streams to 48–55 and worker
contribution to 255.

## Build and baseline functional test

```bash
cmake --build build_Release --target unit_tests_dispatch -j 32
build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerSanity.*'
```

Result: build passed and all five baseline tests passed:

```text
RealtimeProfilerSanity.FiveProgramsBackToBack
RealtimeProfilerSanity.CloseDrainsRegisteredCallback
RealtimeProfilerSanity.ThrowingCallbackIsIsolated
RealtimeProfilerSanity.LastProgramRecordDeliveredOnFinish
RealtimeProfilerSanity.TraceReplayResolvesKernelSources
```

These tests establish the inherited D2H socket, reserved-core transport,
receiver, callback isolation, teardown drain, and kernel-source lookup. They do
not establish concurrent sub-device correctness.

The Milestone 0 layout/register-ownership and deterministic-callback tests also
pass:

```bash
build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerProtocol.CleanBaselineL1BudgetAndScratchRegisterOwnership:RealtimeProfilerSanity.LastProgramRecordDeliveredOnFinish'
```

The layout test reports `dispatch_end=0xa4000`, Blackhole L1 size `0x180000`,
901,120 bytes of current headroom, and 8,192 bytes of worst-case aligned growth
from the 224-byte baseline message to the 8 KiB protocol budget.

The same on-device test proves the selected Blackhole register-space carrier.
An NCRISC kernel saves stream 8 scratch registers 3, 4, and 5, publishes three
distinct 24-bit values, and waits with a finite test bound. TRISC0 observes all
three values without an L1 cache operation and acknowledges through scratch 4,
the direction assigned to completed-record publication.
NCRISC observes the acknowledgement and restores all original values. The test
passes on the QuietBox. This proves cross-RISC visibility and preservation for
the reset, completed-publication, and descriptor-publication registers selected
by the protocol; it does not substitute for Milestone 2's end-to-end queue and
reset tests.

## Static baseline

The JIT images below came from the exact branch after the sanity run:

| Component | Text | Data | BSS |
| --- | ---: | ---: | ---: |
| dispatch_s NCRISC | 3,372 B | 56 B | 356 B |
| existing dispatch_s TRISC0 monitor | 840 B | 0 B | 0 B |
| reserved profiler BRISC | 828 B | 0 B | 0 B |
| reserved profiler NCRISC | 1,588 B | 48 B | 24 B |

The baseline `realtime_profiler_msg_t` is 224 bytes: fixed control words, four
16-byte A/B timestamps, sync words, and the 32-entry FIFO. The reserved-core
layout is 262,336 bytes: a 262,208-byte ring plus 128-byte socket config.

Matched image paths and `size` commands must be recorded again for each coding
milestone because JIT configurations can produce different images.

Exact baseline command:

```bash
/usr/bin/size \
  built/tt-metal-cache1801534666104373800/kernels/cq_dispatch_subordinate/14222124831421671137/ncrisc/ncrisc.elf \
  built/tt-metal-cache1801534666104373800/kernels/cq_dispatch_subordinate_compute/14288965371588785128/trisc0/trisc0.elf \
  built/tt-metal-cache1801534666104373800/kernels/cq_realtime_profiler/5470359895078090684/brisc/brisc.elf \
  built/tt-metal-cache1801534666104373800/kernels/cq_realtime_profiler_push/9328369681231821927/ncrisc/ncrisc.elf
```

## Baseline stress

One full 4,096-program trace replay across four devices passes on the clean
baseline:

```bash
TT_RT_PROFILER_SATURATION_SECONDS=0 timeout 120s \
  build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.PeakLoadPreservesRecords'
```

Result: 16,388 records, four active devices, peak FIFO 14/32,768 pages,
zero ring-full waits, zero invalid timestamps, and no missing record.

The callback-loss accounting stress passes with a one-second production window:

```bash
TT_RT_PROFILER_DROP_ACCOUNTING=1 timeout 120s \
  build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.ConsumerDropAccountingUnderLoad'
```

Result: 3,145,728 records produced; the keeping-up callback received all with
zero drops, and the deliberately slow callback reported 904,072 received plus
2,241,656 dropped, exactly covering production.

These two numbers are historical transport anchors only. The clean protocol
intentionally marks trace-captured go commands unprofiled because capture-time
completion targets are invalid after replay rebasing. They are not used as M1+
stress gates.

Milestone 0 adds a reusable non-trace burst driver and establishes its baseline:

```bash
build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerStress.NonTraceBurstDriverPreservesRecords'
```

Result: one registered-callback warm-up plus 4,096 ordinary asynchronous
enqueues of one compiled blank workload on four active devices delivered
exactly 16,388 records with zero host callback loss. Registering before the
warm-up makes the expected count race-free even though D2H delivery is
asynchronous. The test completes device work independently, then uses a finite
host-only callback wait. Milestone 3 combines this driver with explicit bounded
pressure hooks; it does not depend on trace replay to saturate the device or
callback stages.

## Active host-throughput variance

The exact baseline does not have a supported runtime flag that disables only
the realtime profiler. In these 30 baseline samples the existing observer TRISC
is resident, the reserved BRISC/NCRISC and D2H socket are initialized, and the
old realtime records flow even though the benchmark does not register a
callback. Milestone 1 adds and tests the latched disable gate before performing
matched enabled/disabled ABBA comparison. Milestone 0 therefore records
active-path variance without claiming an enabled/disabled delta.

Command, repeated in 30 fresh processes:

```bash
build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 1024 --num-pages-per-core 4 --compute-nops 0
```

Raw `FD back-to-back` totals in microseconds:

```text
52638 53157 52686 52435 53160 52126 51436 53413 52240 52028
52524 53214 53389 52364 53342 53951 52714 52644 52924 53624
53481 52259 52155 52937 64769 52782 52278 52647 52310 52589
```

| Statistic | Total for 1024 programs |
| --- | ---: |
| Median | 52,666.5 us |
| Mean | 53,140.5 us |
| p95 | 53,951 us |
| Minimum | 51,436 us |
| Maximum | 64,769 us |
| Mean excluding the single maximum outlier | 52,739.6 us |
| Median absolute deviation | 417 us (0.792%) |
| p05–p95 band relative to median | -1.212% to +2.439% |

The outlier demonstrates why later comparisons use paired ABBA order and
distributions rather than one sample.

## Baseline realtime device records

Command:

```bash
build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 64 --num-pages-per-core 4 --compute-nops 0 \
  --use-realtime-profiler
```

The inherited profiler delivered 64 raw device intervals. Summary:

| Metric | Value |
| --- | ---: |
| Duration median | 10,102.5 cycles |
| Duration range | 9,561–10,584 cycles |
| Gap-to-next-go median | 58,483 cycles |
| Gap range, excluding terminal zero | 54,309–76,133 cycles |

This is a transport/baseline anchor only. The old A/B protocol timestamps
dispatch command-loop events and is not evidence of exact concurrent completion.

## Locked Milestone 1 measurements

Milestone 1 must add the isolated disable capability and collect at least 30
paired samples per mode in alternating ABBA order using the 1024-program command
above. Mode A is a separate build/worktree at exact clean commit `5fa57f6d18b`,
with the baseline observer/transport active as described above. Mode B is the M1
build with its new latched disable environment setting. A second comparison uses
M1 enabled as Mode B. Each comparison reports median, p95, range, paired
difference, and a paired bootstrap 95% confidence interval. Gates are
adjudicated by the upper bound of that paired confidence interval, rather than
the unpaired p05–p95 spread:

- M1-disabled regression against the clean baseline's active shipped profiler:
  upper 95% confidence bound no more than 0.5%;
- enabled median regression: upper 95% confidence bound no more than 2.0%;
- unsupported/disabled dispatch adds no cache invalidation or NOC operation;
- Wormhole compile inspection creates no observer kernel. Wormhole runtime
  compatibility is intentionally outside scope.

Milestone 2 additionally measures device cycles for descriptor publication,
enabled-idle observation, one active stream, all active streams, completion
detection latency, and depth-two versus depth-four loss.

## Milestone 0 test scaffolding

`realtime_profiler_test_utils.hpp` provides a thread-safe callback collector
that waits for expected runtime IDs or a repeated-runtime-ID record count with a
finite host-only timeout and returns host callback loss with the wait result. The
`LastProgramRecordDeliveredOnFinish` baseline test now uses this collector
instead of sleeping for a fixed 500 ms.

The helper intentionally does not infer duration from its wait. Milestone 3
extends it with successful sequence and device-loss snapshots while preserving
that rule.

## Open implementation proofs owned by later milestones

- Milestone 1: exact support/inactive reason plumbing, FIFO deadlock regression,
  Wormhole observer non-creation, trace-command suppression, and matched disable
  baselines.
- Milestone 2: reset ordering under an adversarial pause, protected NOC
  stateful-write window, observer polling error, loss counter behavior, and
  buffer-depth selection.
- Milestone 3: final-record teardown delivery and transport pressure without
  dispatch backpressure. Callback loss snapshot encoding is fixed in the
  protocol record.

No runtime implementation claim is made by Milestone 0.

## Final Milestone 0 validation

After resolving the first Opus review findings, the complete focused set passes:

```bash
cmake --build build_Release --target unit_tests_dispatch -j 32
build_Release/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerProtocol.*:RealtimeProfilerSanity.*:RealtimeProfilerStress.NonTraceBurstDriverPreservesRecords'
```

Result: 7/7 tests passed on the local Blackhole QuietBox. No CI or LoudBox run
was started.

## Milestone review

Claude Opus reviewed the milestone with
`claude --dangerously-skip-permissions --model opus --effort high --print`.
After two concrete correction rounds covering trace suppression, staged loss,
cold initialization, wrap/layout details, reset semantics, test ordering, and
completed-ring capacity, the final review returned exact `APPROVE`. The reviewer
independently rechecked the ABI arithmetic, register definitions/ownership,
cross-RISC test ordering, 64-entry SPSC bounds, and message-size budget.
