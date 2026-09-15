# HiFi2/approx-exp status — September 9, 2026

See [REPORT.md](REPORT.md) for the math, measurements, limits and repro command.

Latest implementation/results: [perf-v4/REPORT.md](../perf-v4/REPORT.md).
FP32 now uses streaming compute, still HiFi2 for QK/PV. The final full run
is 3601.68 ms at 0.487625% L2, versus perf-v3's 4597.94 ms (21.67% less time).
Fresh main is 3329.98 ms at 8.798303% L2; improved overhead is 8.16%.
The 3320 ms target was not reached. Three normal seeds pass 0.5%;
outlier/common-mode limitations remain. Current patches are under perf-v4.
BF16 is unchanged. See the latest report for final timing and validation.

The following describes the preceding [perf-v2](../perf-v2/REPORT.md) iteration.
Latest diagnostic: [common-mode Q/K/V sweep](../common-mode/REPORT.md).
At +32, Q/K offsets raise improved FP32 L2 to 2.7523%/1.1204% and BF16 to
15.6773%/16.5910%. V+32 FP32 is at the BF16-output rounding floor; BF16 is
still 0.9643% L2, far above that floor. No kernel changes in this follow-up.
New: [distribution/max-relative-error sweep](../distributions/REPORT.md).
At 256K with spread sampling, improved FP32 L2 is 0.4883% on normal inputs,
but 0.8657% on scaled Q/K and 0.6297% on sparse outliers. BF16 L2 across six
distributions ranges from 0.5416% to 5.4911%. Raw maximum elementwise relative
errors can be huge near reference zeros; the linked report gives all values.
The paired/unrolled cubic preserves FP32 L2 at 0.4884% while reducing its
time from about 5396 to 5095 ms. Final matched 20-warmup medians are main
FP32 3315.52 ms and candidate 5095.30 ms (+53.68%); overhead remains large.
BF16 streaming now compensates BOTH
running states with BF16 hi/lo pairs on PACK: L2 18.9641% -> 3.1897%, with
matched 20-warmup trace medians 2777.65 -> 2930.22 ms (+5.49%). Activation
is restricted to the tested long-context geometry/features; see the report.
Without Q preprocessing, sampled-device BF16 L2 is 3.1496%; its compensation
does not require a new input-preprocessing step. The final candidate was
restored, rebuilt, and verified on device after the exact-main comparison.
Previous tracked sources are represented by perf-v2/final-candidate.patch;
final-approx-hifi2.patch below is now a historical pre-optimization snapshot.

Earlier: [NONCAUSAL_H10.md](NONCAUSAL_H10.md) shows the low-overhead claim does
not generalize. Full non-causal H=10/S=256K: candidate 5396 ms vs main FP32
3287 ms (+64%) and BF16 streaming 2703 ms (~2x). Candidate L2 is 0.4884%.
Full-chip FPU utilization is 48.0% streaming, 41.2% main FP32, 22.1% candidate.
The long candidate run wraps 32-bit profiler references; see the timestamp
correction and 128K validation in that report. Candidate sources were restored
and rebuilt after the main comparison.

Current candidate: refined 10-bit fast-exp grid + cubic + P rounding to six
fraction bits, shifted-grid BF16 Q preprocessing (c=1.0027), one final denominator
reduction at HiFi4, SFPU final normalization, atop the previous FP32 L1 hybrid.
QK and PV remain HiFi2. Small correction operations use higher precision.

Full causal H=4/S=256K/D=128, Q chunk=128/K chunk=512: tail-row aggregate L2
0.4930%, 0.4950%, 0.4899% for seeds 1234/1235/1236. Full operation ~1638 ms,
versus main ~1584 ms. BF16 bitwise Q preprocessing adds ~43 ms on the host.
Not universal: short-Q is ~0.50% with ~1.9x runtime; outliers fail the goal;
normal per-row p95 ~0.56%. These are P100A, not Galaxy measurements.

The earlier final-approx-hifi2.patch contains the pre-optimization candidate.
The previous optimization/final-hybrid.patch remains a historical snapshot.
Host build, JIT, FP64 reference checks, exhaustive normal-BF16 preprocessing
equivalence, trace equality and post-build full-prefill L2 acceptance passed.

Reservation 214149: yyzo-bh-26, container
yyzo-bh-26-special-cglagovich-for-reservation-214149.
Remote repo: /localdev/cglagovich/tt-metal-blackhole-20260908.
SSH control socket: /tmp/sdpa-ird-20260909.sock (yyz-ird).
Use python_env/bin/python with PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages.

Next work: Galaxy/model-derived inputs; wider-range exp implementation; isolate
outlier Q-quantization sensitivity versus exp clipping/state-update error;
fuse Q preprocessing into its producer; broaden shape and feature regression
coverage. No production-readiness claim is made.
