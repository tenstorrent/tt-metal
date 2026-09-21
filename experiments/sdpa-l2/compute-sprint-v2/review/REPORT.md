# Independent B scheduling review — v2

The exact early-identity guard improves the favorable resident B case by
1.561%, but improves distinct normal 256K by only 0.569% and regresses normal
32K by 0.776%. Retain v1 as the general baseline; this is a conditional
long-context research candidate, not an unconditional default improvement.
All final qualification and timing outputs preserve raw BF16 bits. The
separate compiler-attribute investigation rejects O2 for both A and B.

All measurements retain canonical B numerics, frozen v1 scheduling baseline,
Q256/K512/D128, BF16 DST, HiFi2, BF16 Q/K/V, and two unchanged input slots.
Only private experimental sources are changed; v1 is frozen.

## Screens completed

| Candidate | Paired baseline ms | Candidate ms | Change in time | Candidate TFLOP/core | Outcome |
|---|---:|---:|---:|---:|---|
| Correction-cache reordered copies | 337.115903 | 336.758356 | −0.1061% | 1.632493 | Tiny screen difference; not a material gain |
| Paired-half overlap, split8 | 337.245961 | 337.615122 | +0.1095% | 1.628351 | Reject as a B performance improvement |
| Exact identity correction, scalar guard | 337.277791 | 353.769128 | +4.8895% | 1.553996 | Reject; guard overhead exceeds savings |
| Exact identity correction, unrolled guard | 337.020077 | 335.929037 | −0.3237% | 1.636524 | Small resident-only gain |
| Exact identity correction, earlier unrolled scan (final) | 337.106430 | 331.844109 | −1.5610% | 1.656669 | Favorable resident case; data-dependent |

Each screen uses resident repeated K/V, sixteen Q repeats, 512 K chunks,
twelve warmups and ten alternating measured repetitions with original-v1,
disabled-private-wrapper and candidate controls. This removes recurring input
data movement but does not qualify varying online-softmax corrections.

Independent full-dataflow transfer tests for both candidates used Q2048/K1536,
two cores (four distinct Q blocks/core), normal/changing-max/outlier/common-V
inputs, and two trace replays. Every raw BF16 output bit matched the original
v1 winner and disabled wrapper: twelve records per candidate. Seeds were
20260919 for correction cache and 20260920 for overlap. These are narrow tests,
not a claim of full production shape/mask/distribution coverage.

Artifacts: `B-cache-paired-01.json`, `B-cache-transfer-01.json`,
`B-overlap-paired-01.json`, `B-overlap-transfer-01.json`; logs and source-hash
closures accompany each. See `ORDERING_REVIEW.md` for explicit source proofs
and `PHASE_FINDINGS.md` for sampled issue-timeline evidence and its limitations.

The exact identity specialization passed independent Q2048/K4096, two-core
qualification across normal, changing maxima, outliers, common V, and deliberate
identity/fallback transitions. The transition case repeats a common K block
with scales1,1,2,2,4,4,8,8, while retaining independently sampled V and multiple
distinct Q blocks. All fifteen baseline/disabled/candidate records match raw
BF16 bits and two trace replays (seed 20260921). Artifacts are
`B-identity-transfer-01.json` and `B-identity-paired-01.json`.

Despite exactness, the scalar identity scan regresses even on favorable
resident repeated-KV. The unrolled guard has a small B gain; moving its exact
comparisons into an existing preceding QK/PV work interval increases the
resident gain to 1.561%. No numerical approximation or guard weakening is used.

The earlier scan independently passed 27/27 initial fullchip records: normal,
growing-max, scaled-QK, outliers, common-Q/K/V, constant-V and deliberate
identity/fallback transitions; Q2048/K4096 on two cores, seed 20260923. Each
candidate output matches original-v1 and disabled wrappers in raw BF16 bits,
with two separately checked raw-bit trace replays. Held-out odd-K testing adds
12 records at Q1024/K1536 on one core, seed 20260924; short-loop testing adds nine
at Q1024/K1024 on two cores, seed 20260925. All 48 records pass. Unrolled-guard
qualification also passed nine records covering transitions/outliers/common-V,
seed 20260922.

## Distinct-KV timing is data-dependent

These timings repeat the same Q job to obtain an 80–100ms trace, but K/V tokens
within that job are distinct except for intentionally paired K in the transition
case. They are not the distinct-multi-Q fullchip tests described above. Same
geometry, two KV slots, reader/writer and CB layout as baseline.

| Input / KV context | Repeated Q jobs | v1 ms | Earlier scan ms | Time change |
|---|---:|---:|---:|---:|
| Normal /32,768 | 32 | 84.273710 | 84.928052 | +0.776% |
| Growing maxima /32,768 | 32 | 84.257330 | 84.694282 | +0.519% |
| Alternating identity/fallback /8,192 | 128 | 83.808487 | 83.462580 | −0.413% |
| Normal /262,144 | 4 | 84.428583 | 83.947889 | −0.569% |

All final rows above have raw eager equality and explicit raw-bit `trace_equal`.
Earlier screen scripts compared BF16 values on their final replay, not signed-zero
bits; retain that limitation for historical screen artifacts only. Fullchip
qualification always used raw bits. Do not interpret the repeated-KV gain as a
universal improvement over v1. Timings are synchronized host trace wall time,
not profiler device-cycle estimates; useful FLOPs count QK+PV only.

## Evidence and source closure

`FROZEN_MANIFEST.json` pins the selected implementation and experiment wrappers.
The source review requires finite bit-identical maxima, preserves every MAD,
compensation rounding/store, and retains the correction CB publication fence.
Earlier scan flags are fresh per K step and observe immutable published maxima.
See `ORDERING_REVIEW.md` for the lifetime/transition argument. No proof for
unqualified geometries, masks, or ring schedules is claimed.

Run `python3 experiments/sdpa-l2/compute-sprint-v2/review/audit_evidence.py`:
63 selected records pass (48 fullchip, 15 final timing controls), with frozen
candidate/helper hashes, raw output equality, replay gates, numerical flags and
CB contracts. These are selected source pins, not a complete compiler/firmware
closure. Historical screens can pin earlier harness versions. All new kernels
were device-JIT compiled; Python syntax checks passed. No production/v1 edits.

Independent review also read the compensated E/G final report and ran its
read-only audit: 33 frozen early-guard records passed at that review point.
Its conditional/non-default recommendation matches the B evidence. Do not
merge its shorter fullchip timing scope with this repeated-Q timing table.

## Separate compiler-attribute screen

Private `codegen/` wrappers compare original frozen v1 A/B, an O3 wrapper
control, and an O2 function-attribute wrapper. Global build/fast-math flags,
numerical recipes, input/CB geometry and reader/writer remain unchanged.
This experiment is separate from the identity specialization. Both smokes and
resident screens preserve raw eager/final-trace bits, but O2 is slower:

| Variant | Frozen v1 ms | O3 wrapper ms | O2 ms | O2 time increase |
|---|---:|---:|---:|---:|
| A | 275.159238 | 275.167489 | 283.708605 | 3.107% |
| B | 337.255675 | 337.254510 | 344.651028 | 2.193% |

Reject both O2 candidates without expanded qualification. See
`codegen/REPORT.md` and its four smoke/timing JSON artifacts. No identity/O2
combination was tested or retained. All device jobs finished and none remain
queued. The frozen v1 A/B winners remain the recommended general baselines.
