# Realtime profiler clean-room Milestone 2 evidence

**Target:** local Blackhole QuietBox, device 0, worker dispatch, one hardware
command queue

**Status at capture:** technical blocker remediation and local qualification
complete; product-owner performance adjudication and Claude Opus re-review
were pending

## Delivered device protocol

Milestone 2 adds a bounded, device-local concurrent path while deliberately
leaving the public M1 start-only transport unchanged until Milestone 3:

- dispatch_s reserves a five-word descriptor in a fixed depth-four SPSC ring,
  stages its timestamp payload before GO, and publishes its index/epoch after
  the first GO write; a full ring never prevents GO;
- dispatch_d publishes a 32-bit generation only after the real completion
  counter clear, with a register-space reset epoch as the visibility carrier;
- resident TRISC0 parks until activation, checks one register-space epoch while
  idle, scans only locally active streams, and timestamps the first observation
  of a satisfied device completion target;
- TRISC0 publishes eight-word records into a fixed 64-entry SPSC completed
  ring, or increments a stage-specific loss counter without waiting;
- initialization requires version-matched ready flags from dispatch_d,
  dispatch_s, and TRISC0 before the reserved profiler core is launched;
- the explicit public capability query refreshes the six live M2 device loss
  counters; four transport/shutdown fields are reserved and remain zero until
  Milestone 3. This diagnostic L1 read is not a timing endpoint or steady-state
  collection path.

No tensor readback, host-derived duration, model code, Sparse MLA code, TopK,
all-gather, scheduler policy, CI job, LoudBox, or Wormhole qualification is part
of this milestone.

## Correctness and pressure qualification

The production-depth suite passed eleven active tests in one process; the two
compile-time adversarial tests passed in fresh processes with their respective
test defines.

| Qualification | Result |
| --- | --- |
| Same-stream correlation | Five records preserve invocation ID, generation, stream, schema, sequence, and two raw device endpoints. |
| Reverse completion | Two stream-1 invocations complete in order before an earlier long stream-0 invocation. |
| Actual reset lifecycle | `clear_loaded_sub_device_manager()` changes the next record's generation through the real `WAIT_STREAM|CLEAR_STREAM` path. |
| Clear/publication race | dispatch_d is paused after the real counter clear; a real queued GO cannot publish a descriptor or interval until generation publication. |
| Completed-ring saturation | The ring stops at 64 entries, `completed_record` rises, and ordinary `Finish` completes. |
| Descriptor saturation | An impossible full head counts exactly one `descriptor_full` and one `stuck_head`; the application completes. |
| Coalescing | Four already-satisfied descriptors emit only the newest and count exactly three `observer_coalesced`. |
| Counter wrap | The 17-bit half-range comparison accepts forward progress through natural wrap and rejects the reverse direction. |
| Protected GO window | A source-contract test proves the in-window profiler helper contains no NOC/stream-register/cache/index publication and commit occurs only after `with_state`. |
| All-stream publication | Forty records across all eight streams preserve payload-before-index visibility and increasing sequence with zero source loss. |
| Shortest two-stream pressure | 512 alternating nonblocking launches complete with zero depth-four descriptor loss. |

The adversarial reset and observer-cycle code is present only in separately
hashed test firmware when its environment variable is set. Both use a dedicated
six-word test-only block; the mutually exclusive reset hook uses word 0 and the
cycle hook uses totals 0–2 plus samples 3–5. Normal firmware does not contain
the branch or measurement stores. Test fault injection uses direct diagnostic
L1 access only; production timing never uses D2H data.

## Descriptor-depth decision

The shortest two-stream test was run twice from matched source, once with the
physical descriptor constant temporarily set to two and once at production
depth four. Both completed 512 alternating launches with zero descriptor loss.
The source was restored to depth four. Depth two saves only 320 bytes against
roughly 897 KiB of current dispatch-region headroom and does not materially
improve a locked resource gate, so it does not satisfy the plan's condition for
replacing depth four. Depth four retains additional observer-latency slack.

## Device-cycle evidence

The direct GO-tail command is the same one approved for Milestone 1:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=1 \
  build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 32 --num-pages-per-core 4 --compute-nops 0 \
  --num-active-cores 1 --use-device-profiler
```

`go_tail_cycles` is measured from the device timestamp carrying dispatch_s
command type 14 to that command's device-zone end. There are 33 samples,
including the warmup launch.

| Arm | Min | Median | p95 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| M2 disabled | 796 | 815 | 1,209 | 842.55 | 1,265 |
| M2 enabled | 979 | 1,068 | 1,465 | 1,091.88 | 1,548 |

The complete enabled descriptor path adds 253 median cycles over the matched
disabled binary, about 187 ns at 1.35 GHz. Relative to the approved M1 enabled
median of 945 cycles, Milestone 2 adds 123 cycles. The device-profiler teardown
prints its existing unfinished-resident-dispatch-core warnings; the benchmark
passes and emits all 33 dispatch_s endpoint pairs.

Raw samples:
[realtime_profiler_clean_room_milestone2_device_cycles.tsv](realtime_profiler_clean_room_milestone2_device_cycles.tsv)

The worker-boundary test independently measures both documented sources of
interval skew over 32 device samples. The existing realtime/device-profiler
clock-sharing path in `RealtimeProfilerManager::publish_device_profiler_sync_anchor`
states and relies on all cores on the chip sharing one wall clock; that is the
basis for subtracting dispatch-core and worker-core samples here:

| Device-cycle delta | Min | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Descriptor start to worker start | 576 | 583 | 591 | 591 |
| Worker end to observer end | 336 | 363 | 389 | 397 |

No host timestamp contributes to either endpoint. Device-cycle p95 uses the
nearest-rank order statistic; the host-throughput table below uses NumPy's
default interpolated percentile.

The test-only instrumented observer build measured upper-bound loop costs:

| Observer state | Mean cycles/loop | Samples |
| --- | ---: | ---: |
| Enabled idle | 24 | 9,244,329 |
| One active stream | 91 | 24,070,520 |
| All eight streams active | 466 | 835,035 |

Disabled cost is zero because the eligible-Blackhole decision omits the TRISC0
observer kernel entirely. The empty active mask reads only the descriptor epoch
and performs no L1 invalidation or stream scan unless it changes. Reset is read
only while a stream is active and before that stream can complete. Active work
iterates the observer-local active mask. The reported instrumented loop costs
include the timing hooks and are therefore conservative upper bounds.

## Host-throughput corroboration

Thirty same-binary paired blocks were run on the QuietBox, alternating
enabled-disabled and disabled-enabled order. Each arm launches 1,024 programs.
Host timing remains descriptive because the Milestone 1 no-op control proved
that this machinery cannot resolve the former host gates.

| Arm | Median | p95 | Range | Mean |
| --- | ---: | ---: | ---: | ---: |
| M2 disabled | 51,558.5 us | 53,251.3 us | 50,145–60,479 us | 52,012.73 us |
| M2 enabled | 52,676.0 us | 55,395.7 us | 51,147–56,575 us | 52,885.70 us |

The paired absolute median is +1,199.5 us and the paired relative median is
+2.2826%. A deterministic 100,000-resample paired bootstrap
(`numpy.random.default_rng(0)`) gives [0.8341%, 3.5368%]. The
upper confidence bound exceeds the protocol's original 2.0% locked host gate,
so that gate does not pass. The Milestone 1 no-op rebuild showed that this host
machinery cannot resolve either former host bound, but extending that
adjudication to M2 requires explicit product-owner approval; it is not inferred
here. Every enabled pressure run completed even after the device-local
completed ring became full and counted drops.

Raw pairs:
[realtime_profiler_clean_room_milestone2_interleaved_perf.tsv](realtime_profiler_clean_room_milestone2_interleaved_perf.tsv)

## Resource gates

| Component | M0 baseline text | M2 text | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| dispatch_s NCRISC | 3,372 B | 3,948 B | +576 B | PASS; limit +1,536 B |
| resident observer TRISC0 | — | 2,036 B | — | PASS; limit 2,048 B |
| reserved profiler BRISC | 828 B | 844 B | +16 B | PASS; limit +1,024 B |
| reserved profiler NCRISC | 1,588 B | 1,592 B | +4 B | No intentional source change |

The 2,956-byte dispatch-core message passes the 8 KiB gate. The reserved-core
source and 262,336-byte layout are unchanged by Milestone 2. Matched normal
firmware image paths are:

```text
built/tt-metal-cache1801534666104373800/kernels/cq_dispatch_subordinate/5439497712055163212/ncrisc/ncrisc.elf
built/tt-metal-cache1801534666104373800/kernels/cq_dispatch_subordinate_compute/17146445212160634297/trisc0/trisc0.elf
built/tt-metal-cache1801534666104373800/kernels/cq_realtime_profiler/8090457159864465995/brisc/brisc.elf
built/tt-metal-cache1801534666104373800/kernels/cq_realtime_profiler_push/9516854709705244049/ncrisc/ncrisc.elf
```

The generic device-profiler-instrumented observer image is 2,116 bytes because
the device profiler adds its own kernel-zone machinery. The locked 2,048-byte
gate applies to the resident normal firmware image above, which is the
always-on production cost.

The normal observer has 12 bytes of text headroom. Milestone 3 must remeasure
this hard gate rather than assume additional observer code will fit.

## Validation commands

```bash
cmake --build build --target ttnn unit_tests_dispatch -j8

build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerProtocol.*:RealtimeProfilerConcurrentDevicePath.*'

TT_METAL_REALTIME_PROFILER_TEST_RESET_PAUSE=1 \
  build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerConcurrentDevicePath.NextLaunchCannotCrossClearToGenerationWindow'

TT_METAL_REALTIME_PROFILER_TEST_OBSERVER_CYCLES=1 \
  build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerConcurrentDevicePath.MeasuresIdleOneAndAllStreamObserverCycles'
```

All commands above passed locally. `git diff --check` also passed. At the time
this historical Milestone 2 evidence was captured, commit and push remained
gated on exact Claude Opus `APPROVE`.
