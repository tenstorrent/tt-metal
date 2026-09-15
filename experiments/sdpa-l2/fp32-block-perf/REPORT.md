# HiFi4 approximate-exp FP32-subtraction SDPA: block tuning and headroom

Status (2026-09-11): block sweep, final timing repeats, additional accuracy
checks, activity profiles, source restoration, and regression checks complete.
The best-tested blocks remain 128/1024; the selected algorithm sustains about
55 useful TFLOP/s. No precision-preserving kernel speedup is claimed by this study.

## Scope and measurement contract

This is mode 4 from `../accuracy-investigation/`: HiFi4 QK/PV and matched
denominator, original BF16 Q/K/V and output, FP32 destination/online state,
full-FP32 SFPU score subtraction, and the unbiased approximate exponential.
There is no Q preprocessing and no accurate-exp substitution. Arithmetic in
the compute/SFPU source is unchanged from that mode-4 investigation.

Blackhole P100A, yyzo-bh-26, reservation 216406; 110 compute cores (11x10),
not Galaxy. Noncausal B=1, H=10, D=128; Q=K length 32768/65536/131072/262144.
Main base is `2ba6fc2339d53300ae87c5202f335ef56492cfb3`. No clock, power,
reader/writer algorithm, or matmul subblock changes. QK/PV subblocks remain 1x4.

Useful attention FLOPs per invocation are `4*H*N*N*D`, counting QK and PV,
with multiply-add counted as two FLOPs. Throughput divides this by median
blocking trace-replay time. It does not count the softmax, denominator matmul,
or housekeeping as useful FLOPs. Timing excludes compilation, input transfers,
CPU reference, and output comparisons; it is not a raw device-cycle timer.
Screening uses 10 warmups/5 measured replays; finalists use 40/10.

References use the same already-rounded BF16 inputs in FP64, seed 1236, with
128 spread query rows per head in screening and 512 in final/stress runs.
Final/stress runs additionally check finiteness of the entire output and exact
ordinary-vs-trace equality of the entire output. L2/PCC are sampled-reference
metrics, not exhaustive full-output FP64 comparisons. A 0.25% normal aggregate
L2 filter excludes broken block geometries from the performance ranking; this
is a tuning screen, not a replacement for the project's qualification gates.

## Block search

Six numerically working candidates from the 32K screen are tested at every
sequence length: Q/K=128/256, 128/512, 128/1024, 256/128, 256/256, 256/512.
The two fastest at each length are repeated with the final timing protocol.
These are best-tested block sizes within this search, not a proof of global
optimality over arbitrary layouts, schedules, or input-buffering policies.

The temporary `TT_SDPA_BLOCK_SWEEP` host switch widens geometry eligibility
but does not implement general support for every admitted shape. Mode 4 still
rejects fallback. Existing double-buffered K/V is retained for Q<256/K<=512;
Q=256 or K>=1024 uses the established single-buffered layout. Q remains buffered
according to the existing scheduler. Thus block comparisons can also change
which existing buffering layout applies; this is disclosed, not a new pipeline.

Rejected geometries:

- Q=32/64 cases produce incorrect outputs and/or fail trace equality. Inspection
  identifies an unhandled constraint: the FP32 denominator loop always processes
  four Q tiles at a time and has no tail handling. Those shapes are not supported
  just by widening the host guard. Their times are excluded from useful rankings.
- Q=256/K=1024 exceeds L1: the allocation endpoint is 2,233,344 B versus
  the 1,572,864 B limit. Q=128/K=2048 reaches 2,495,488 B, and Q=64/K=2048
  reaches 1,840,128 B. No allocation limits are bypassed.
- All raw results, including failures, remain in the `screen-*` files. An initial
  sweep stopped at a trace-equality failure; after inspecting the clean device
  shutdown it resumed, retaining both logs and the failure record.

Screening medians in milliseconds (10 warmups/5 replays, not final timings):

| N | Q128/K256 | Q128/K512 | Q128/K1024 | Q256/K128 | Q256/K256 | Q256/K512 |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 110.479 | 99.901 | **94.736** | 140.081 | 110.449 | 99.407 |
| 65536 | 443.799 | 405.156 | **378.821** | 556.877 | 443.394 | 401.341 |
| 131072 | 1778.167 | 1643.810 | **1580.727** | 2171.703 | 1767.875 | 1630.447 |
| 262144 | 7087.522 | 6570.003 | **6308.708** | 8661.291 | 7064.376 | 6519.701 |

Doubling Q at fixed K brings little benefit here. Larger K helps, with rapidly
diminishing returns. Fitting the three Q128 timings at N=131072 to `A+B/K`
gives A=1513.55 ms, only 4.25% below the K1024 screen. This is a descriptive
fit, not a hardware lower bound: buffering changes at K1024, and scheduling,
power, and memory behavior need not follow this model at untested blocks.
It supports prioritizing the per-score loop over simply increasing K again.

## Final results

The winning block size is **Q=128, K=1024** at all four tested lengths.
It is the same geometry used in the previous mode-4 timing: this sweep did not
find a faster block configuration for the selected algorithm.

| N | Q/K blocks | Median ms | Useful TFLOP/s | Aggregate L2 % | Aggregate PCC | Worst head L2 % |
|---:|---|---:|---:|---:|---:|---:|
| 32768 | 128/1024 | 100.299 | 54.812 | 0.179594 | 0.999998386 | 0.181168 |
| 65536 | 128/1024 | 396.177 | 55.506 | 0.179023 | 0.999998397 | 0.179946 |
| 131072 | 128/1024 | 1585.642 | 55.473 | 0.179231 | 0.999998394 | 0.180342 |
| 262144 | 128/1024 | 6321.801 | 55.656 | 0.179230 | 0.999998393 | 0.180353 |

Measured replay ranges are 99.802–100.320 ms, 396.121–396.539 ms,
1582.829–1587.587 ms, and 6311.482–6328.997 ms respectively. Throughput
is approximately constant at 55 TFLOP/s once sustained timing is used; do not
substitute the faster short screening measurements for the final table.

The 256K sampled-output hash is exactly the same as
`../accuracy-investigation/hifi4-unbiased-fp32sub.jsonl`. Its L2 and PCC also
match exactly. This is the precision level the user selected, not a lower-
precision configuration that happened to run faster. All four winning runs
pass full-output finiteness and exact full-output trace-equality checks.

The Q256/K512 runner-up takes 102.128, 415.122, 1636.899, and 6524.171 ms
respectively: 1.82%, 4.78%, 3.23%, and 3.20% slower. All eight final timing
runs pass the full-output checks. Raw final results are the `final-*.jsonl` files.

Additional mode-4 checks at the winning blocks, seed 1236, H=10 and 512 spread
reference queries/head:

| N | Distribution | Aggregate L2 % | Aggregate PCC | Worst head L2 % |
|---:|---|---:|---:|---:|
| 32768 | Q/K scaled 2x | 0.197226 | 0.999998057 | 0.200919 |
| 32768 | Sparse outliers | 0.259614 | 0.999996632 | 0.383900 |
| 262144 | Q/K scaled 2x | 0.200323 | 0.999997996 | 0.204807 |
| 262144 | Sparse outliers | 0.302992 | 0.999995410 | 0.375300 |

All four also pass full-output finiteness and full trace equality. These are
additional performance-candidate checks, not a rerun of the frozen qualification
suite or an exhaustive worst-row guarantee. Since the winner is the original
128/1024 geometry, no new block configuration is being promoted on their basis.
The previous full qualification of mode 3 (accurate exp) must not be attributed
to this approximate-exp mode. Raw records: `stress-*.jsonl`.

## Activity counters

Three isolated profiles completed after all non-profiled timing and stress runs.
The shorter profiles avoid 32-bit counter wrap. `counters.py` checks the counter
reference against 64-bit TRISC1 timestamps and rejects wrapped measurements.
Percentages are over the 110 active cores, not the profiler header's 120-core
chip denominator. Profiled timing is not substituted into the throughput table.

| Variant | N | FPU active % | SFPU active % | Either active % | Overlap, percentage points | Neither active % |
|---|---:|---:|---:|---:|---:|---:|
| Selected mode 4, full-FP32 subtraction | 65536 | 46.72 | 36.24 | 74.98 | 7.98 | 25.02 |
| Selected mode 4, full-FP32 subtraction | 131072 | 46.74 | 35.94 | 74.72 | 7.95 | 25.28 |
| Mode 1, TF32 subtraction control | 131072 | 70.71 | 36.72 | 77.23 | 30.21 | 22.77 |

These are active-core means; medians and ranges are also in `counters.jsonl`.
All three runs contain 110 active cores and pass the no-wrap/timestamp checks.
Overlap is `FPU + SFPU - MATH-union`; it is not an extra throughput percentage.
Activity includes supporting math instructions, not just useful QK/PV FLOPs.

The selected algorithm is **not saturating either arithmetic unit**. The related
TF32-subtraction control has much more overlap, which supports investigating
the conservative one-tile schedule. This is not a clean measurement of batching
alone: that control also moves subtraction from SFPU to FPU and changes the
amount/type of work. It is not eligible as the precision-preserving winner.
Likewise, the 25% neither-active interval is not necessarily all removable;
dependencies, memory, synchronization, and startup/drain can contribute.

Raw profiles live under `profile-mode*/reports/`; their normal output metrics
are in the accompanying JSONL files. These are separate single-invocation
profiles, not the sustained 40-warmup timing measurements.

## Precision-preserving optimization opportunities

1. **Batch FP32 subtraction and exponential work in destination registers.**
   The current diagnostic acquires DST, unpacks one score tile, broadcasts its
   maximum, commits/waits, subtracts, runs grid exp and polynomial refinement,
   packs one tile, and releases DST. This happens separately for each score
   tile. A two-score-plus-one-max batch fits within the four-tile FP32 DST budget.
   It can reuse a maximum and amortize format setup, synchronization, constants,
   and replay setup. Keep the same FP32 subtraction, scale, grid, coefficients,
   and output rounding. Test bitwise equivalence before accepting any numerical
   tolerance. This is a proposed optimization, not a measured speedup.

2. **Recover overlap without changing Q/K traffic.** Once batching is correct,
   pipeline independent QK/PV work with SFPU processing and packing. The current
   one-tile loop exposes repeated waits and format switches. Hoist invariant
   setup where register/replay ownership allows it; do not remove the existing
   PACK-to-UNPACK visibility barriers without proving the replacement protocol.
   Keep original reduction order where practical. Better overlap requires an
   explicit DST ownership schedule, not merely fewer synchronization calls.
   A fused SFPU subtraction/grid/refinement pipeline could also remove repeated
   DST stores and reloads. Preserve the existing FP32 intermediate rounding;
   moving the scale across `score - max` is not a numerically equivalent shortcut.

3. **Reduce local storage or lifetime only if it unlocks a useful larger block.**
   The next large blocks are constrained by L1. Rescheduling score/P lifetimes
   may help, but the FP32 score alias already shares storage and does not allocate
   a second score matrix. Aliasing that existing alias again saves nothing.
   Do not shrink scores/state to BF16 or TF32 to buy capacity: that abandons the
   requested precision. A different storage schedule is a larger engineering
   project and can affect movement/overlap, so it is lower priority than batching.

4. **Denominator specialization is secondary and needs arithmetic validation.**
   P-times-ones computes just a sum but currently uses a separate HiFi4 matmul
   pass. A specialized matched-precision reduction could reduce work or passes.
   Do not restore the old LoFi/six-bit denominator: it would no longer match PV's
   effective probabilities. Any alternative reduction must be checked against
   the outlier/uniform/cancellation probes as well as normal L2.
   One concrete product-level observation from the ISA fidelity model: the ones
   operand is in SrcA and has no low mantissa part. Phases 1 and 3 therefore
   contribute zero products; phases 0 and 2 contain the probability bits.
   A HiFi3 denominator, or a custom phase-0/phase-2 sequence, is worth testing
   while keeping QK and PV HiFi4. This is not yet a claim of bitwise-equivalent
   hardware accumulation or a measured gain. The local ISA reference is
   `tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md`
   (`SrcAFidelityBits`/`SrcBFidelityBits`), shared by the Blackhole documentation.

Keep both large matmuls at HiFi4. Do not claim a gain from reverting to TF32
subtraction, changing probability precision, or using approximate online-state
updates: those change the accuracy contract being optimized.

### How much improvement to pursue

Block tuning alone found no gain over the existing 128/1024 baseline. For the
next kernel effort, a reasonable **initial engineering target is 65–70 useful
TFLOP/s**, equivalent to **5.41–5.03 seconds at 256K**: about **14–20% less
time** than the measured 6.322 seconds (17–26% greater throughput). This is a
target, not a measured result or performance guarantee. The 4.821-second
TF32-subtraction control from the preceding investigation is a useful scheduling
reference, but reverting to it is not the proposed optimization.

I would start with a two-score/max batch and require unchanged outputs on fixed
blocks before considering a more substantial pipeline redesign. The counter
gap leaves room beyond the initial target in theory, but a twofold speedup
would require a much more complete removal of serialization and idle time;
there is no evidence yet to promise that outcome. Do not add the individual
opportunity estimates together: their savings overlap.

## Reproduction and handoff

`full-candidate-from-main.patch` contains the tested four-file operator delta
from the main base. `SOURCE-SHA256.txt` records the exact sweep sources.
The host was built with `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build
build_Release --target install`; raw build output is in `build.log`.
Each candidate runs in a fresh Python process with mode 4 and the block-sweep
switch. The harness verifies source hashes first; this does not replace a host
rebuild. Do not apply the full patch on top of the already-modified retained
implementation. It includes those earlier improvements as well as the diagnostic.

From the matching source/build and existing Python environment:

```bash
python_env/bin/python experiments/sdpa-l2/fp32-block-perf/sweep.py --stage screen --lengths 32768
python_env/bin/python experiments/sdpa-l2/fp32-block-perf/sweep.py --stage long --lengths 65536 131072 262144
python_env/bin/python experiments/sdpa-l2/fp32-block-perf/sweep.py --stage final --lengths 32768 65536 131072 262144
python_env/bin/python experiments/sdpa-l2/fp32-block-perf/sweep.py --stage stress --lengths 32768 262144
bash experiments/sdpa-l2/fp32-block-perf/profile.sh
```

The sweep retains existing status records when resumed; use a separate copy of
the experiment directory for independent repeat runs. `profile.sh` refuses to
overwrite prior outputs. Run profiles separately from all timing workloads.

All four operator sources are restored locally and remotely to the retained
snapshot in `../qualification-v1/RESTORED-SHA256.txt`. The restored host was
forcibly rebuilt, and four frozen full-output/hash/trace regressions pass:
BF16 fast and retained HiFi2 FP32 at N=32768 and 262144, H=10, seed 1234.
See `restored-build.log` and `restored-regression.jsonl`. The tested HiFi4
block-sweep implementation is saved as a patch; it is not left active in the
retained build. No device workloads remain running from this investigation.

Python syntax compilation and Black checks pass for the two new Python scripts
and updated repro; shell syntax and `git diff --check` pass. The full candidate
patch passed reverse-apply validation before restoration. No dependencies were
installed, and no commits or PRs were made.
