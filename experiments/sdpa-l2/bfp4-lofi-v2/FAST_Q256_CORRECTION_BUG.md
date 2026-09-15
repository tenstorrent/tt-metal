# FAST Q256 recurrence regression found during LoFi research

Confirmed on Blackhole,2026-09-15. Frozen selected sources remain unchanged.

The new common-reader harness exposed incorrect output from the frozen
compensated BF16 FAST kernel for Q256 with distinct K/V chunks. This is not
a BFP8 quantization issue: unmodified BF16 inputs reproduce it. Main BF16
and ACCURATE pass in the same harness; input-preprocessing checks are exact.

## Reproduction and isolation

`fullchip.py --variant fast --length 4096 --heads 2 --cores 4 --q-chunk 256`
produces1206.87% relative L2 on128 explicitly sampled Q rows/head.
One head gives1516.43%, identically with1core or16cores(one Q job/core).
Q128 instead passes2.449%; a single512-token K chunk passes2.466%.

## Cause

The frozen `calculate_sdpa_exp_correction()` uses explicit SFPI destination
increments but does not reset `ADDR_MOD_7`. The paired compensation replay
leaves automatic destination increment2 active. On Q256, compensation of
an earlier PV row group precedes later correction exponentials; automatic
and explicit increments then combine incorrectly. Q128 computes both
corrections before its hoisted compensation updates. Repeated resident K/V
also hides the bug because its recurrent max corrections are one.

The standard SFPU call wrapper does not restore this address modifier.
Adjacent diagnostic exp helpers already explicitly reset it.

## Private fix and measured result

`fast_correction.hpp` wraps the frozen correction helper, restoring
`ADDR_MOD_7.dest.incr=0` (and source increments0) before calling it.
`fullchip.py --fix-correction` enables this override without editing the
four frozen source snapshots. Initial JIT attempt had an incorrect MATH-only
guard; the tested revision includes PACK, where the streaming call occurs.

|4K,heads2,cores4,Q256/K512/D128|Before L2%|After L2%|After PCC|
|---|---:|---:|---:|
|FAST HiFi2, original BF16 inputs|1206.867|2.482314|0.999697080|
|LoFi compensated BF16, RNE7 Q/RNE5 BFP8 K/V|698.434|3.066572|0.999532397|

Both fixed outputs are finite and bit-identical after trace replay.
Evidence:`fullchip-fix-fast-v2.json`,`fullchip-fix-lofi_fast_b8-v2.json`.
These initial accuracy checks sample Q rows. Later private-harness results
and current-source smoke coverage are summarized in [the final qualification](FINAL_QUALIFICATION.md);
the dedicated integration regression matrix remains a [next step](ENGINEERING_NEXT_STEPS.md).
Do not cite the failed finite-output benchmark as performance
evidence. Prior Q128 qualification and resident throughput measurements
remain measurements of their stated cases, not qualification of distinct
Q256 recurrence. Any future FAST promotion must include this regression.
