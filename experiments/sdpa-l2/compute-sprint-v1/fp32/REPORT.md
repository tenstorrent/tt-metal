# FP32 compute-only sprint: C and D

Status: final resident qualification, paired measurement, distinct-Q
fullchip checks and profiling complete. No canonical
kernel or production source has been edited; every candidate is isolated here.

## Contract and numerical invariants

One Blackhole core, Q256/K512/D128, original single K/V input slot for C/D.
The canonical `flux2-frontier-v1/device_attention.py:recipe()` determines all
numeric flags and fidelity. Reader/writer source, CB capacities/formats,
chunk sizes, matmul fidelity, exponent coefficients and operations,
normalization, max decisions, and recurrence are unchanged.

D remains FP32 streaming QK4/PV4, full FP32 score-minus-max subtraction and
unbiased cubic correction. Its two denominator phases are phases 0 and 2
for the zero/one operand, **not** ordinary HiFi2. C remains FP32 streaming
QK4/PV2 with the existing cheaper subtraction and biased cubic matched to
the effective HiFi2 weights. Neither takes input preprocessing.

The old `bfp4-lofi-v2/resident.py` unconditionally removes denominator-phase
defines, even for D. This benchmark does not use its numeric recipe.

## Candidate changes

Shared winners under final qualification:

1. Copy four independent numerator tiles per DST acquire in the existing
   exact-identity rescale branch, instead of two acquisitions of two tiles.
2. Batch adjacent FP32 recurrent-state unpacks into the existing multi-face
   MOP, retaining every tile's Blackhole ZEROACC workaround.
3. Defer subtraction operation setup until after the existing exact max
   equality decision. No subtraction is used when the correction is identity.
4. Compare the same 32 BF16 first-column max values with four rows per loop
   control iteration. Only the low 16 bits count, as in the original.

C additionally uses a load-macro implementation of the **same C cubic**,
with identical Horner MAD ordering and coefficients, and hoists its macro
configuration once per K iteration. Copying D's unbiased coefficients would
be incorrect; the isolated C implementation explicitly retains C's values.

A four-tile blocked denominator pack was measured but did not help and is
not selected. Earlier fixed-half/independent-MATH-pack and maximum repeat-pack
experiments were reviewed; their previously rejected slower schedules were
not repeated here.

## Evidence protocol

`bench.py` measures resident repeated-KV compute throughput. `paired.py`
interleaves baseline/candidate in ABBA order; its qualification mode instead
uses genuinely distinct KV, including normal at 256K and seven stress
distributions at 32K. `fullchip.py` uses canonical dataflow and several
distinct Q blocks/core. Every candidate is checked with BF16 uint16 equality,
full output SHA256, and a trace-replay output comparison. This is stronger
than preserving aggregate L2/PCC. Quantitative accuracy of each baseline is
reported for context; this scheduling work does not improve known baseline
stress failures.

Raw per-case JSON, output tensors, source hashes, CB descriptions and logs
are retained here. Instrumented profiles are separate from decisive
unprofiled timings. FPU activity is not useful FLOP utilization.

## Sustained unprofiled resident results

Q repeats=8, K chunks=512 (256K total repeated keys), eight warmup and eight
timed trace replays per leg, same-process ABBA order. All four legs per
variant produce identical BF16 bits, including the final trace output.

| Variant | Baseline medians, ms | Winner medians, ms | Time reduction | Baseline TF/core | Winner TF/core |
| --- | --- | --- | --- | --- | --- |
| D | 361.921 / 361.899 | 335.547 / 335.575 | 7.28% | 0.75952 | 0.81916 |
| C | 293.970 / 294.065 | 244.236 / 244.234 | 16.93% | 0.93490 | 1.12546 |

Artifact pairs: `D-final-abba-v1.json` and `C-final-abba-v1.json`.
These are one-core compute measurements, **not** measured chip throughput.

Final candidate names: D `state_lazy_scan`, C
`c_refine_hoist_state_lazy_scan`.

## Frozen-winner correctness

Both winners pass 8/8 distinct-KV baseline comparisons and 8/8 trace
comparisons, with zero differing uint16 output words in every case.
Artifacts: `D-final-qualify-v1.json`, `C-final-qualify-v1.json`.

| Distribution | Keys | D L2% unchanged | C L2% unchanged |
| --- | --- | --- | --- |
| Normal | 262144 | 0.17989 | 0.39054 |
| Scaled Q/K | 32768 | 0.19746 | 0.33993 |
| Outliers | 32768 | 0.27187 | 0.37763 |
| Common Q | 32768 | 0.39536 | 1.16810 |
| Common K | 32768 | 0.75457 | 1.11299 |
| Common V | 32768 | 0.02805 | 0.02805 |
| Constant V | 32768 | ~7e-14 | ~7e-14 |
| Uniform | 32768 | 0.15582 | 0.15594 |

These tests establish schedule preservation on this suite, not universal
bitwise equivalence. The numerical quality of the underlying recipes,
including their existing common-mode failure cases, remains unchanged.

## Distinct-Q validation with unchanged real dataflow

Q2048/K8192/H1/D128 on two cores gives four distinct Q chunks per core.
Canonical reader/writer code and descriptors are unchanged. Normal,
scaled-QK, outliers and common-K cases all pass uint16 equality and final
trace equality for both variants (4/4 each). Source hashes/descriptors and
outputs are recorded in `D-final-fullchip-v1.json` and
`C-final-fullchip-v1.json`.

Normal five-replay medians are D 5.851→5.763 ms and C 4.777→4.313 ms.
This is a small two-core, real-DM check with baseline then candidate order,
not the decisive interleaved no-DM experiment. Its smaller improvement is
expected where data movement and setup reduce compute savings' visibility.

## Device-cycle profile and utilization

Separate instrumentation-only runs used Q repeats=8/K chunks=512. The CSV
header reports 1350 MHz. Useful utilization below comes directly from FLOPs
per device cycle: D ceiling1024 useful FLOPs/cycle; C ceiling4096/3 with
equal QK4/PV2 work. Thus it does not depend on a guessed active frequency.
FPU/SFPU counters count activity, not useful matmul FLOPs.

| Variant | Useful own-roof utilization | FPU active | SFPU active | Both active | Neither active |
| --- | --- | --- | --- | --- | --- |
| D baseline | 54.99% | 59.04% | 25.64% | 19.91% | 35.23% |
| D winner | 59.22% | 63.58% | 27.62% | 21.37% | 30.18% |
| C baseline | 50.73% | 54.58% | 38.96% | 30.99% | 37.45% |
| C winner | 61.07% | 65.71% | 37.02% | 33.40% | 30.67% |

Measured resident cycles: D488122260→453265835; C396871383→329639084.
The corresponding header-clock throughputs are D0.76023→0.81869 and
C0.93503→1.12573 TF/core, closely matching unprofiled sustained results.
C reduces absolute SFPU-active cycles as well as idle overhead; D primarily
removes operation setup and state-handling gaps. Neither reaches 80%.

`profile-results.json` contains the parsed counters; raw CSVs and Tracy
captures are in each `*-final-profile-v1/` directory. Run IDs1024 and2048
are independent baseline invocations;3072 is the selected candidate.

## Integration handoff and reproduction

`winners.json` freezes the exact additional defines. `retained.patch` is an
**unapplied** narrow diff against the canonical header, with only retained
changes: no helper-include relocation and no rejected denominator-pack
experiment. It references the separately provided `c_refine.hpp` helper.
`git apply --check` passes locally. This is an integration aid, not a final
production API design; the benchmark still uses the frozen measured files.

All C++ changes were JIT-compiled on Blackhole in both plain and profiled
runs. Python syntax checks and Black formatting passed; the kernel header
was clang-formatted before final source freeze. No full host library build
was needed because these isolated generic-op sources are runtime compiled.

From the remote checkout, through the global lock wrapper (fresh labels):

```sh
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v1/fp32/paired.py \
  --label NEW-D-ABBA --variant D --candidate state_lazy_scan \
  --q-repeats 8 --k-chunks 512 --warmup 8 --iters 8 --rounds 2
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v1/fp32/paired.py \
  --label NEW-C-QUAL --variant C --candidate c_refine_hoist_state_lazy_scan --qualify
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v1/fp32/fullchip.py \
  --label NEW-D-DISTINCTQ --variant D --candidate state_lazy_scan
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  -m tracy -r --profiler-capture-perf-counters fpu \
  -o experiments/sdpa-l2/compute-sprint-v1/fp32/NEW-C-PROFILE \
  experiments/sdpa-l2/compute-sprint-v1/fp32/bench.py \
  --label NEW-C-PROFILE --variants C --candidates c_refine_hoist_state_lazy_scan \
  --q-repeats 8 --k-chunks 512 --iters 0
```

Exchange variant/candidate together to run the complementary cases.
`profile_analyze.py --variant C PATH/profile_log_device.csv` reproduces
cycle-derived metrics without a device.

## Environment and incidents

Reservation 223862, `bh-lb-08`, container
`bh-lb-08-special-cglagovich-for-reservation-223862`; logical device0,
12x10 grid, firmware 19.13.1, KMD 2.9.0. All jobs use the parent-owned global
exclusive device lock. The initial smoke had one missing relocated include;
it failed compilation before execution and was corrected. A later queued
qualification was refused by the shared dirty guard after another agent's
failure; it did not launch. Parent subsequently reset the reserved devices
following a device-initialization timeout in another job, and passed a new
matmul smoke. Final measurements use fresh post-reset controls, not a
pre-reset/post-reset comparison.

## Final frozen source pins

Private compute header:
`fef42cc8a8bfb272e4bdd401902c51ca7c880fc2b4f84f25fc930da69e76462c`

C refiner:
`cf38a29ddb6df1c8e349301012b721ab25c8be47e01ecf1bc10e88b9770b3ee3`

Resident wrapper:
`b963eed8f978cba1714042d4800f42ece535bd64c6fb69b12d45a8fb039c8e5b`

Canonical header:
`f795e49f09b34e388fdb3149b40656905fcee0ae496abdf8f7d04e221afcaac9`

Canonical recipe adapter:
`ec477bb574bd1503320802a400ccfde938eaf8e0cf054c6da39541fd2d779afe`
