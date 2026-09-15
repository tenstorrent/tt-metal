# Accurate FP32 SDPA: L1 subtraction and load-macro exp

Investigation date: 2026-09-11. This continues [util-v1](../fp32-util-v1/REPORT.md),
not a comparison against pristine main. Final sustained qualification is in
`final-v2.log`; raw per-case JSONL and provenance records are authoritative.

## Scope

One Blackhole P100A, 110 compute cores (11x10), reservation 216406 on
`yyzo-bh-26`. These are not Galaxy measurements. Noncausal B1, H10, D128,
Q/K chunks 128/1024, BF16 Q/K/V and output. Q/K chunk sizes, input buffering,
and reader data movement are unchanged. The new subtraction pass does add
internal L1 traffic.

The accepted mode4 algorithm remains: HiFi4 QK/PV, FP32 DST and numerator/sum
state, full-FP32 score subtraction, the unbiased negative-logit exp grid and
cubic correction, and a BF16 maximum CB. No Q preprocessing. Denominator
phases 0+2 remain exact for the zero/one SrcA operand; this is not ordinary
HiFi2. The historical public test label `fp32_hifi2` is overridden by mode4.

## Retained changes

1. Broadcast the BF16-derived row maximum once, negate it exactly, and subtract
   it from the FP32 score row using packer L1 accumulation. Wait for those L1
   writes before score reload. Keep the existing maximum CB unchanged.
2. Without a resident maximum occupying DST, batch four score tiles into each
   FP32 DST half. Coalesce their unpack and pack operations, retaining the
   Blackhole zero-flag clear for every unpacked tile.
3. Use the existing fast-exp load-macro grid with the same unbiased cubic
   refinement. Preserve the separate FP32 subtraction and the grid/refinement
   rounding sequence. The macro is not a general-purpose exp: its common
   multiplicative factor cancels in softmax normalization.

All are explicit investigation guards. Defaults are unchanged; unsupported
configurations are rejected rather than silently falling back. Source
[candidate-env.sh](candidate-env.sh) from the repository root before a fresh
process, or use `run.py --mode l1macro`, which clears all inherited `TT_SDPA_*`
options before setting the candidate options. Presence-based options must be
unset to disable them: setting their value to `0` still enables them.

## Validation

`validate.py --label final-v2` uses fresh processes for each operator run.
Performance cases at N32768/131072/262144 use 40 warmups and 10 blocking trace
replays. Both versions use the same seed1236 and 512 spread FP64-reference
query rows per head against full K/V. L2 and PCC are sampled-reference
metrics; full-output comparisons use BF16 SHA256 and finiteness checks.
The complete result after the final trace replay is compared exactly with
the nontrace output; intermediate replay outputs are not copied to the host.

The control is the prior util-v1 candidate, with the new options disabled.
Both are also checked against frozen original-mode4 full-output hashes.
Ten stress cases at N32768/H10/seed1236 and three N65536/H5/seed1237 holdouts
are checked against frozen baseline outputs. Stress/holdout one-replay,
zero-warmup times are not performance measurements. Exact output equality
preserves existing precision, including existing failures; it does not
establish a universal L2 <0.5% guarantee.

### Sustained paired results

| N | Prior util-v1 ms | New candidate ms | Less time | Useful TFLOP/s | L2 % (both) | PCC (both) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 79.040 | 76.757 | 2.89% | 71.62 | 0.179594 | 0.999998386 |
| 131,072 | 1291.377 | 1209.269 | 6.36% | 72.74 | 0.179231 | 0.999998394 |
| 262,144 | 5155.743 | 4845.210 | 6.02% | 72.62 | 0.179230 | 0.999998393 |

All **16 paired checks passed**: 3 performance cases, 10 stress cases, and
3 holdouts (15 unique inputs; normal32K occurs twice). Complete BF16 outputs
are bit-identical to the previous accepted algorithm, with unchanged sampled
L2/PCC and exact final-trace/nontrace equality. Both current control and
candidate use the same source tree and differ only in the two new options.

| Stress input, N32768 H10 | L2 % | PCC |
| --- | ---: | ---: |
| Normal | 0.179594 | 0.999998386 |
| Q/K scaled by 2 | 0.197226 | 0.999998057 |
| Sparse outliers | 0.259614 | 0.999996632 |
| Uniform attention (Q=0) | 0.173736 | 0.999998488 |
| Constant V=1 | ~0 (FP64 reference residual) | Undefined |
| Q=0, V=1 | 0 | Undefined |
| V bias +1 | 0.176753 | 0.981648634 |
| Common Q +32 | 0.232221 | 0.999997304 |
| Common K +32 | **0.731996** | 0.999973698 |
| Common V +32 | 0.028662 | Undefined |

The N65536/H5/seed1237 holdouts have L2 0.179143%, 0.198372%, and 0.279635%
for normal, scaled-QK, and outliers respectively, all unchanged. PCC is
undefined when a compared output has zero variance. Common K remains a
known mode4 limitation above 0.5%; this optimization does not fix it.

Useful FLOPs are `4*B*H*N*N*D`, excluding softmax/denominator work. At 256K,
72.62 TFLOP/s corresponds to **44.4%** on the provisional denominator inferred
from the user's 55.65 TFLOP/s ~=34% estimate. That denominator has not been
independently calibrated. The corresponding 80% target is 130.94 TFLOP/s,
or 2.687 seconds: another **1.80x speedup / 44.5% time reduction** is needed.
**The 80% goal remains unmet.** FPU-active counters are a different metric.

Relative to the older original-mode4 measurement of 6.208 seconds, current
4.845 seconds is about 22.0% less time. This is a historical comparison, not
a new paired original-mode4 measurement. The table above is the fresh paired
comparison. The current util-v1 control agrees with its earlier 5.158-second
measurement, but the shorter screen's 1.169-second result should not replace
the sustained 1.209-second result at 128K.

## Screened alternatives

Screens use 10 warmups/5 replays, H10 normal inputs, seed1236. They are useful
for selecting candidates, not substitutes for the sustained comparison.

| Configuration | 32K ms | 128K ms | Complete BF16 output |
| --- | ---: | ---: | --- |
| Prior util-v1 control | 74.545 | 1285.063 | Matches |
| Fixed-half pipeline | 77.226 | 1294.659 | Matches |
| Fixed-half, earlier QK release | 75.162 | 1287.269 | Matches |
| L1 subtraction, fused raw exp | 72.153 | 1217.845 | Matches |
| L1 subtraction, load-macro exp | 69.970 | 1169.221 | Matches |
| Above, four-copy maximum block pack | 71.520 | 1172.367 | Matches |

The fixed-half prototype dedicates lower DST to QK and upper DST to two
scores plus their resident maximum. Explicit ownership allows direct score
unpack without the usual MATH score handshake. It is correct in the tested
cases, but did not provide a performance benefit, so it is disabled.
Earlier QK release did not change this conclusion. A correct resident-input
microkernel likewise measured 111.372 ms versus 111.301 ms for the comparison
schedule. That comparison still overlaps QK with the first exp batch and
must not be described as a fully serialized baseline.

The first maximum repeat-pack experiment was incorrect: setting W stride
to zero does not repeat the same tile when the blocked pack MOP advances Z
across faces. It produced roughly 75.6% L2 at 32K and was rejected. The
corrected four-DST-copy version passed equality but was not faster and is
also disabled. Failed logs are retained.

The standalone fixed-half probe also exposed two low-level hazards before
passing: raw unpack CB addresses use `fifo_rd_ptr - 1`, and Blackhole's
ZEROACC unpack workaround needs the selected DST bank plus bank-local face
indices. Using full-DST face indices cleared the wrong bank's zero flags and
caused stale QK accumulation. No card reset was needed in this investigation.

## Bottleneck evidence and next step

The new N65536 stage1 profile averages over all 110 active cores, with no
32-bit reference-counter wrap. Stage2 independently agrees within 0.3
percentage points. These instrumented runs are separate from sustained timing.

| Activity metric | Prior util-v1 | New candidate |
| --- | ---: | ---: |
| FPU active | 55.61% | 61.86% |
| SFPU active | 42.03% | 35.25% |
| Both active | 19.84% | 27.27% |
| Neither active | 22.20% | 30.16% |

SFPU-only time falls from about 22.2% to **8.0%** of the profiled interval.
The fraction with neither unit active increases even though total runtime
decreases: shortening and overlapping arithmetic exposes the remaining
unpack/pack/configuration/synchronization work. Counters alone do not identify
which of those causes dominates, and these percentages must not be read as
an exclusive per-operation critical-path budget.

Uncontaminated inclusive thread-stage means:

| Thread | Stage | Share of its kernel interval |
| --- | --- | ---: |
| MATH | QK | 28.09% |
| MATH | PV | 29.45% |
| MATH | score subtraction/exp | 14.93% |
| MATH | denominator | 4.44% |
| PACK | score subtraction/exp | 45.76% |
| PACK | QK | 10.94% |
| UNPACK | PV | 21.25% |
| UNPACK | denominator | 4.59% |

Do not sum overlapping thread stages. UNPACK sum slot0 and PACK sum slot1
are contaminated by LLK CB wait/reserve instrumentation and are discarded.

The new register-resident four-tile macro-exp probe measures **508.62 billion
logits/s**, or **1.351 seconds** of arithmetic work for H10/N262144's logit
count. It performs the same grid/refinement sequence over four resident DST
tiles, repeatedly converging to a checked fixed point that includes the
grid's common scale factor. All cores' complete FP32 outputs and final trace
equality are checked. This probe excludes L1 subtraction, recurring score
reload/pack, matmuls, and online state work. It is not operator performance
or a universal lower bound. The earlier 1.846-second probe included FP32
subtraction and used two-tile raw replay, so the comparison is not an
apples-to-apples isolated-exp speedup.

The next large optimization should therefore target **scheduling and data
transfer stalls**, not a lower-precision exp or matmul:

1. Prototype independent QK pack issue while the PACK thread issues SFPU
   instructions. The fixed-half experiment changed DST ownership but left
   both QK packs and long SFPU replays on that thread. Test moving the QK
   pack issue to another compute thread in a small verified probe first.
   This needs explicit ownership of packer address/configuration state;
   merely removing waits is unsafe. Feasibility and benefit are unproven.
2. If the probe wins, carry that ownership through QK and the PV drain,
   retaining the exact partial-PV accumulation grouping and online update
   order. Use the four-tile L1-subtraction candidate as the new control,
   not the superseded two-score fixed-half path. Keep input chunks/buffering.
3. Profile the newly exposed no-FPU/no-SFPU intervals and reduce repeated
   unpack/packer configuration and drains. Do not assume all 30% is removable
   or attribute it all to L1 bandwidth without a separating measurement.

The target still needs more than hiding SFPU work. Under a deliberately
simplified model that holds FPU-active time fixed and eliminates everything
else, 61.9% of 4.845 seconds is about **3.00 seconds**, still above 2.687.
This combines a small-shape activity profile with sustained large-shape
timing, so it is a planning estimate, not a hardware bound. Reaching 80%
would also require roughly another 10% reduction in FPU-active work/time
under that model, through better issue efficiency or less auxiliary work.
The prior isolated 140 TFLOP/s matmul+pack calibration suggests headroom but
does not prove that the full operator can sustain the requested rate.

## Reproduction

The established remote environment supplies `TT_METAL_HOME`, `ARCH_NAME=blackhole`,
and `PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages`.

```bash
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/fp32-pipeline-v2/validate.py --label fresh-label
bash experiments/sdpa-l2/fp32-pipeline-v2/profile.sh fresh-profile
```

Do not run profiling concurrently with performance tests. Profiles use N65536
to stay below 32-bit counter wrap and must not be used as production timing.
The parsers in `../fp32-util-v1/stages.py` and `../fp32-block-perf/counters.py`
reject contaminated sum slots and reference-counter wrap respectively.

Base HEAD is `2ba6fc2339d53300ae87c5202f335ef56492cfb3` with preexisting changes.
No commit, PR, or change to ordinary SDPA defaults has been made.
`candidate.patch` is a delta against the retained util-v1 source state, not
against pristine main. It includes the disabled fixed-half/repeat-pack
investigation controls and is not a minimal PR patch. `SOURCE-SHA256.txt`
records the six operator/test source files used for final qualification;
each run also records its command, options, and source hashes independently.
The previous directory's source manifest describes the previous candidate,
not the now-updated working tree.
