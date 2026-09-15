# Accurate FP32 streaming: pack-issue and exp-refiner optimization

Investigation September 11–12 Eastern (final profiles September 13 UTC), 2026.
Continues [pipeline-v2](../fp32-pipeline-v2/REPORT.md), not pristine main.

**Result: another repeatable 1.6–1.7% reduction at 256K, with bit-identical
outputs. No large scheduling breakthrough. The 80% useful-FLOP target remains
unmet.** The new exp schedule is retained behind an explicit investigation
option, not enabled by default. Keep the previous schedule at 32K: the new
one is slower there. Do not infer an automatic dispatch threshold from these
three measured lengths.

## Configuration and retained change

One Blackhole P100A on yyzo-bh-26, 110 active compute cores, nominal 1350 MHz.
Initial screens used reservation 216406; after expiration, final build,
qualification, reverse-order timings and profiles used reservation 217840
on the same host. These are not Galaxy results. The renewed container uses
the `metal` IRD image and the existing local checkout/build environment.

Noncausal B1/H10/D128, BF16 Q/K/V and output; Q128/K1024. HiFi4 QK and PV,
FP32 DST and online numerator/sum, full-FP32 score subtraction through the
existing L1-adder pass, and the existing BF16 maximum CB. No Q preprocessing.
Denominator phases 0+2 are exact for its zero/one SrcA, not ordinary HiFi2.
The test's historical `fp32_hifi2` label is overridden by mode4.

Q/K chunks, reader code, input buffering and internal CB formats are unchanged
relative to pipeline-v2. In particular, the prior FP32 configuration already
reduces K/V buffering at K1024; this work does not introduce double buffering.

`TT_SDPA_FP32_REFINE_MACRO=1` changes only the cubic exp-refiner schedule:

- Same unbiased grid, cubic coefficients, Horner MAD order and rounding.
- Use load macros for exponent extraction and final multiply/store, reducing
  the repeating refiner from 14 to 10 issued instructions per two vectors.
- Reload the original linear-grid value from DST for the final multiply.
  This adds DST reads, not L1 score traffic. Restore the grid instruction
  templates before each subsequent grid pass and drain scheduled stores.

The standalone four-tile register-exp probe takes 45.969 ms versus the prior
58.055 ms, about 20.8% less time, at 110 cores and 65,536 repetitions.
It checks all cores' output against a known fixed point and final trace
equality. This is **not operator performance**, and its 1.070-second projected
256K arithmetic work is not a universal lower bound. It excludes subtraction,
recurring L1 reload/pack, matmuls and online state updates. Full operator
equality, rather than that fixed-point probe alone, qualifies the new schedule.

## Paired performance

Fresh processes, seed1236, 40 unmeasured warmup trace replays and 10 measured
blocking trace replays. Times are median host-wall durations of blocking
device trace execution, excluding compilation and tensor transfers; they are
not isolated device-profiler kernel durations. Profiling was run separately.
Clock/power settings were not changed or locked by this experiment. Warmup
count does not itself prove thermal equilibrium, particularly at 32K.

| N | Previous v2 ms | New refiner ms | Time reduction | New useful TFLOP/s | L2 %, both | PCC, both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 74.936 | 78.115 | **-4.24%** | 70.38 | 0.179594 | 0.999998386 |
| 131,072 | 1195.254 | 1186.092 | 0.77% | 74.16 | 0.179231 | 0.999998394 |
| 262,144 | 4849.936 | 4768.025 | 1.69% | 73.79 | 0.179230 | 0.999998393 |

Reverse-order confirmation (candidate first, then control, fresh processes):

| N | Previous v2 ms | New refiner ms | Time reduction |
| --- | ---: | ---: | ---: |
| 32,768 | 76.580 | 77.407 | **-1.08%** |
| 262,144 | 4842.839 | 4763.655 | 1.64% |

The initial shorter 128K screen suggested 2.4%; the longer comparison is only
0.77%. Do not advertise the screen's larger gain. The 32K regression occurs
in both measured orders; its exact magnitude varies. 128K has only one final
pair, so its sub-1% benefit is less established than the repeated 256K result.

Useful FLOPs are `4*B*H*N*N*D`, excluding softmax and denominator work.
73.79 TFLOP/s at 256K is **45.1%** using the provisional peak denominator
inferred from the user's 55.65 TFLOP/s ~=34% estimate. It is not an independently
calibrated peak. The corresponding 80% target remains about 130.94 TFLOP/s /
2.687 seconds: roughly **1.77x** more speedup is still required.

## Precision qualification

`final-v3.log`: all **16 comparisons passed** (15 unique inputs). Three
performance cases are compared with fresh v2 runs and frozen accepted-mode4
outputs. Ten N32768/H10 stress inputs and three N65536/H5/seed1237 holdouts
are compared with frozen accepted outputs. Both reverse-order pairs also
match exactly. `audit.py` verifies these comparisons and that current source
hashes match every final qualification provenance record.

Full BF16 output SHA256, finiteness and final-trace/nontrace equality are
checked. L2/PCC use 512 spread query rows per head against a FP64 reference
with complete K/V, not a full FP64 output for every row. Bit-identical full
outputs preserve precision, including existing failures; they do not imply
every distribution meets a universal 0.5% L2 requirement.

| Stress input, N32768 H10 | L2 % | PCC |
| --- | ---: | ---: |
| Normal | 0.179594 | 0.999998386 |
| Q/K scaled by 2 | 0.197226 | 0.999998057 |
| Sparse outliers | 0.259614 | 0.999996632 |
| Uniform attention, Q=0 | 0.173736 | 0.999998488 |
| Constant V=1 | ~0, FP64 residual | Undefined |
| Q=0, V=1 | 0 | Undefined |
| V bias +1 | 0.176753 | 0.981648634 |
| Common Q +32 | 0.232221 | 0.999997304 |
| Common K +32 | **0.731996** | 0.999973698 |
| Common V +32 | 0.028662 | Undefined |

Held-out normal/scaled-QK/outlier L2: 0.179143%, 0.198372%, 0.279635%, all
unchanged. PCC is undefined for zero-variance compared outputs. The known
common-K failure is unchanged, not fixed by this optimization.

## Rejected scheduling experiments

| Experiment | Control | Candidate | Decision |
| --- | ---: | ---: | --- |
| QK pack issue on MATH, 4-tile block-pack microprobe | 75.115 ms | 80.214 ms | Slower; do not integrate |
| Four-score fixed-half L1 pipeline, 128K screen | 1163.705 ms | 1196.736 ms | Exact output, slower |
| Same pipeline without PACK-side QK zero clear, 128K | 1163.705 ms | 1184.029 ms | Slower and output hash changed |
| Lighter direct-DST reload initialization, 128K | 1163.705 ms | 1166.979 ms | Exact output, no gain |

Microprobe times are not SDPA times. It uses 110 cores, 65,536 repetitions,
10 warmups and 5 measured replays, with complete output checking. The MATH
packer uses thread-local ADC/MOP state and a mutex around shared pack state;
the tested ownership protocol costs more than it saves. This does not prove
all possible independent-pack schedules lose.

The first MATH-pack probe hung because mailbox reads identify the sender,
whereas writes identify the recipient. Its own process was stopped and only
allocated card0 was reset with `tt-smi -r 0`. Corrected probes and all later
controls passed. The zero-clear-removal candidate had tiny numerical changes
(normal128K L2 0.17923184% instead of 0.17923054%); their cause was not isolated.
It was not retained merely because aggregate L2 looked similar.

Rejected operator branches were removed from active sources. Their final
snapshot is in `rejected-snapshot/`; the initial exact fixed-half version was
also preserved separately there. Raw failed and successful logs remain.
The snapshots are investigation artifacts, not a supported alternate build.

## Counters and remaining opportunity

Fresh control and candidate stage1 profiles, N65536/H10, all 110 active cores;
no reference-counter wrap. Stage2 gives consistent activity. Instrumented
times are not used in the performance table.

| Activity | Previous v2 | New refiner |
| --- | ---: | ---: |
| FPU active | 61.81% | 63.71% |
| SFPU active | 35.23% | 29.13% |
| Both active | 27.27% | 21.16% |
| Neither active | 30.24% | 28.31% |

SFPU-only activity stays approximately 8%. Much of the removed SFPU work was
already overlapped, explaining why its arithmetic improvement translates to
a small operator improvement. Counters alone do not separate configuration,
unpack/pack, DST contention, L1 traffic and synchronization causes.

Inclusive per-thread stage shares likewise show PACK subtraction/exp nearly
unchanged (45.76% ->45.68%), while MATH's subtraction/exp share falls
14.92% ->12.70%. MATH QK/PV shares become 29.08%/30.39%. Do not sum these
overlapping thread intervals or interpret them as exclusive critical-path
fractions. The parsers reject UNPACK slot0 and PACK slot1 because LLK CB
instrumentation contaminates them.

The next substantial optimization should separate the subtraction/reload/pack
and synchronization costs inside the PACK stage, then test a schedule that
avoids repeated shared-packer reconfiguration. The independent-MATH-pack
prototype is not that solution. Further polynomial issue reductions alone
look unlikely to close the target gap: even eliminating the observed 8%
SFPU-only share is far short of the required ~44% total-time reduction.
Those percentages are planning evidence, not a rigorous hardware bound.

## Reproduction and handoff

From the existing remote checkout with `TT_METAL_HOME`, `ARCH_NAME=blackhole`
and `PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages`:

```bash
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/fp32-pack-issue-v3/validate.py --label fresh-label
bash experiments/sdpa-l2/fp32-pack-issue-v3/profile.sh fresh-profile
python_env/bin/python experiments/sdpa-l2/fp32-pack-issue-v3/audit.py
```

The final build succeeded (`final-build.log`), and both kernel variants were
JIT-compiled and exercised. C++ was clang-formatted; the test and run/validation
drivers passed Black; `git diff --check` passes. Source SHA256 and raw results
are copied locally. `candidate.patch` is relative to the previous retained v2
state, including incidental formatting, not pristine main. HEAD remains
`2ba6fc2339d53300ae87c5202f335ef56492cfb3`; no commit or PR was made.

`run.py --mode l1macro` selects previous v2; `--mode refine` selects the new
schedule. Both clear inherited `TT_SDPA_*` options. The optional
`candidate-env.sh` must be sourced before a fresh process. Presence flags
must be unset to disable them, not set to zero. Factory guards reject
unsupported configurations; no fallback was enabled. No ordinary SDPA default
was changed, and no automatic length-based selection was introduced.

Reservation 217840 expired after all device tests and profiles completed.
The final report/audit-script upload then failed because that job no longer
existed. Raw results, profiles, formatted operator sources, provenance and
the completed build log had already been downloaded; the offline audit and
manifest check passed locally. The report and audit script are available in
this local worktree, but their final remote upload is not claimed. There is
no active hardware reservation left by this turn.
