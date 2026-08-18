# Realtime Profiler Clean-Room Milestone 1 Evidence

**Branch:** `pjosipovic/realtime-profiler-clean-room`
**Clean comparison:** `5fa57f6d18bdc46e0d623a6f86a1b7d4bfcc547b`
**Target:** local Blackhole QuietBox, 110 application tensix cores, one hardware command queue
**Scope:** profiler capability gating and protocol cleanup only
**Review:** Claude Opus high-effort milestone review returned exact `APPROVE` on 2026-08-18

## Result

Milestone 1 now provides a nonblocking, start-only profiler record with correct
invocation identity. The 16-byte Go command carries the 16-bit runtime ID and
an explicitly validated 8-bit worker contribution. The host patches
`wait_count`, `wait_stream`, worker contribution, and runtime ID with one
aligned 64-bit store. The old 32-entry program-ID FIFO and every dispatch wait
on it are gone.

The inherited observer endpoint systematically belongs to launch N-1 while the
new Go-carried runtime ID belongs to launch N. Publishing that pair would claim
a false interval. Milestone 1 therefore deletes the legacy completion observer,
publishes the device start tick with `end_timestamp == 0`, and makes Tracy skip
start-only records. Milestone 2 introduces the replacement completion-gated
observer and restores a correlated device endpoint.

No tensor D2H fallback exists and no host timestamp is substituted. The
existing D2H socket transports records already produced by the device.

## Correctness and ownership

- The host manager is the sole writer of activation/capability state.
  dispatch_s and TRISC0 only read it; competing firmware clears were removed.
- The host writes the remote state address before publishing the nonzero
  profiler-core NOC coordinate as the activation release. If D2H socket
  construction fails after the dispatch image is built, it publishes a
  permanent disabled sentinel; dispatch_s resolves that state once and performs
  no per-command polling.
- Profiler activation is decided before dispatch kernel construction.
  Unsupported or disabled configurations create no reserved profiler core or
  reserved-core kernel. M1 creates no completion observer in any configuration.
- Device telemetry is independent of the realtime-profiler gate.
  `cq_dispatch_subordinate_compute` remains present for TRISC1 telemetry,
  while TRISC0 is a stub.
- `num_workers` is accepted only in the inclusive range 1..255. An
  unencodable value produces zero profiler metadata and a host warning instead
  of silent `uint8_t` truncation.
- Trace-captured Go commands contain zero profiler metadata. Ordinary launches
  remain profiled before and after replay.
- Runtime ID is copied to a local before command-buffer reuse. The Go command
  remains 16 bytes and the four profiler-related fields are one aligned final
  64-bit host store.
- Only profiled GO commands capture a timestamp or notify the reserved core.
  Non-GO commands and runtime ID zero perform no realtime-profiler timestamp or
  NOC work.
- The M1 start tick is sampled after the prior same-stream worker wait and
  immediately before the first multicast or unicast GO write. It therefore
  matches the frozen Milestone 2 start boundary without including prior work.
- The obsolete workload-hash marker, profiler FIFO, profiler commands in
  ordinary `Finish`, and profiler FIFO flushes are absent.

## Focused qualification

Build:

```bash
cmake --build build --target \
  tt-metalium-validation-basic unit_tests_dispatch test_prefetcher \
  test_op_to_op_latency ttnn -- -j8
```

Passed on the QuietBox:

- 13 focused capability/protocol/API/deadlock tests: 12 passed and the
  fresh-process environment test skipped as designed;
- the skipped environment test passed separately with
  `TT_METAL_DISABLE_REALTIME_PROFILER=1`;
- `RealtimeProfilerStress.PeakLoadPreservesRecords` and
  `RealtimeProfilerStress.ConsumerDropAccountingUnderLoad` passed with
  one-second qualification windows after conversion to ordinary launches;
- both fast-dispatch packed-read smoke variants passed;
- Python binding smoke for `NONE`, the pre-initialization inactive state, and
  the empty pre-initialization capability snapshot;
- `git diff --check`.

The deadlock regression feeds 64 program IDs while dispatch_s is stalled behind
its synchronization semaphore, with the release command queued behind them.
dispatch_d reaches the release command and the test finishes. The deleted
32-entry FIFO could not make that progress.

## Device-cycle evidence

The direct GO-tail measurement is primary. All three binaries were launched
from their own source worktree so runtime JIT lookup could not select another
checkout's firmware:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=1 \
  test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 32 --num-pages-per-core 4 --compute-nops 0 \
  --num-active-cores 1 --use-device-profiler
```

For every launch, `go_tail_cycles` is the device-cycle distance from
`process_cmd_d_dispatch_subordinate` command type 14 to
`CQ_DISPATCH_CMD_SEND_GO_SIGNAL` zone end.

| Arm | n | Min | Median | p95 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact clean baseline | 33 | 466 | 927 | 1,149.6 | 944.03 | 1,377 |
| M1 disabled | 33 | 796 | 817 | 1,046.8 | 844.27 | 1,293 |
| M1 enabled | 33 | 923 | 945 | 1,155.4 | 972.52 | 1,409 |

The disabled GO tail is 110 cycles faster than the clean baseline. Enabling M1
adds 128 cycles over the same M1 binary's disabled path, approximately 95 ns at
1.35 GHz, and is 18 cycles above the clean baseline median. The endpoint is the
outer dispatch_s command-zone end, after ID publication and the reserved-core
notification, so this includes the complete device-side M1 GO service tail.

GO-arrival deltas are secondary because they include upstream and worker
timing. Excluding the warmup-to-measured transition, their medians were 9,849
cycles for baseline, 10,052 for M1 disabled, and 10,511 for M1 enabled.

Raw device samples:
[realtime_profiler_clean_room_milestone1_device_cycles.tsv](realtime_profiler_clean_room_milestone1_device_cycles.tsv)

## Unified host-throughput control

Host timing is corroborating evidence. Sixty balanced four-arm blocks were
run in one session. Odd blocks used disabled/enabled/baseline/no-op order and
even blocks reversed it. Each build had its own source working directory and
JIT cache.

```bash
test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-programs 1024 --num-pages-per-core 4 --compute-nops 0
```

| Arm | Median | p95 | Range | Mean |
| --- | ---: | ---: | ---: | ---: |
| M1 disabled | 51,988.0 us | 54,812.2 us | 49,560–62,056 us | 52,251.72 us |
| M1 enabled | 52,585.0 us | 60,099.5 us | 50,334–82,493 us | 53,758.75 us |
| Exact baseline | 51,341.5 us | 52,372.7 us | 50,151–52,715 us | 51,341.73 us |
| Baseline comment-only rebuild | 51,410.0 us | 52,486.1 us | 50,455–54,972 us | 51,523.95 us |

Paired medians use `(candidate - reference) / reference`; intervals are a
deterministic 100,000-resample paired bootstrap with
`numpy.random.default_rng(0)`.

| Comparison | Absolute median | Relative median | Bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| M1 enabled vs same-binary disabled | +515.5 us | +1.0168% | [0.5214%, 2.5584%] |
| Baseline no-op rebuild vs original | +126.5 us | +0.2461% | [0.1132%, 0.6083%] |
| M1 disabled vs baseline original | +630.0 us | +1.2306% | [0.4277%, 2.1268%] |
| M1 enabled vs baseline original | +1,474.0 us | +2.8673% | [1.4218%, 3.3421%] |

The same-binary point median is 1.02%, but isolated 60–82 ms process outliers
leave its bootstrap upper bound at 2.56%; it does not pass the former 2% host
gate. The comment-only baseline control also exceeds the old 0.5% cross-build
host ceiling. This host benchmark cannot resolve either device-path gate on the
QuietBox. Neither result is waived or called a pass.

Raw unified session:
[realtime_profiler_clean_room_milestone1_interleaved_perf.tsv](realtime_profiler_clean_room_milestone1_interleaved_perf.tsv)

## Invalidated cross-build measurements

Two earlier series invoked the detached baseline host binary from the current
checkout. Kernel lookup prefers the process working directory before
`TT_METAL_HOME`, so those processes JIT-compiled current firmware against the
baseline host library. The data are retained for audit but excluded:

- [earlier disabled series](realtime_profiler_clean_room_milestone1_disabled_perf.tsv)
- [earlier enabled series](realtime_profiler_clean_room_milestone1_enabled_perf.tsv)
- [later rejected three-arm series](realtime_profiler_clean_room_milestone1_invalid_cross_build_perf.tsv)

The working-directory mismatch was confirmed when a no-op baseline host binary
deadlocked with current JIT firmware and completed immediately after launch
from the baseline worktree.

## Static resource gates

| Component | Baseline text | M1 text | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| dispatch_s NCRISC | 4,676 B | 4,504 B | -172 B | PASS, at most +1,536 B |
| compute TRISC0 | 880 B | 316 B | -564 B | PASS; M1 observer removed |
| reserved profiler BRISC | 908 B | 908 B | 0 B | PASS |
| reserved profiler NCRISC | 1,660 B | 1,660 B | 0 B | PASS |

The dispatch-core profiler message is 88 bytes, down from the 224-byte
FIFO-bearing baseline and below the locked 8 KiB budget. The reserved
profiler-core L1 layout remains 262,336 bytes. These image sizes come from the
same matched device-dispatch-profiler runs used for the cycle table.

## Milestone decision

Correctness, deadlock, device-cycle, static-resource, and focused test gates
pass. Neither host-throughput ceiling is statistically resolved by the
process-level benchmark. The product owner accepted device-cycle adjudication,
with host throughput retained as descriptive evidence, on 2026-08-18. The
corrected milestone then received exact Claude Opus `APPROVE`.
