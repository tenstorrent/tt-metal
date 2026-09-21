# Compensated B/E/G scheduling, v2

All v1 sources/results remain frozen. This subtree's E/G harness uses the v1
`lowp/combined_fence/compute_streaming.hpp` winner as its timed baseline; it
retains canonical recipe/preparation and an additional direct canonical-output
check in distinct mode. Variant B transfer/independent review is owned by the
review agent. Q256/K512/D128, CB formats/depth, input readers/writers and all
numerical choices are unchanged.

## First hypothesis: reuse a row correction across numerator column groups

`correction_cache` processes the four numerator groups in column-pair-major
order: `(row0,col0), (row1,col0), (row0,col2), (row1,col2)`. Each row therefore
uses the same alternating DST half. The first two groups populate correction
slot6 and use a private release which retains the canonical pack-completion
stall, semaphore release and half flip, but omits clearing that half. The last
two groups overwrite all state slots0..5, reuse slot6, and use canonical release
including clearing. Both halves are clear before denominator and next matmul.

The default release really does clear DST: Blackhole
`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:63`.
The BF16 A2D state copies overwrite values using MOVA2D, not accumulate into
old DST: `llk_math_eltwise_unary_datacopy.h:437–461`. The unchanged numerator
SFPU replay writes only slots0..3; slots4/5 hold current chunks and slot6 remains
intact. The candidate's static assertion restricts it to two query-tile rows,
four D tiles, and eight BF16 slots per DST half. No arbitrary-shape support is
claimed.

This removes two correction broadcasts plus init/uninit pairs and two clears
per SALAD row group. It does not remove SFPU arithmetic, high/low stores, or
pack-completion barriers. Independent row/column updates change scheduling,
not per-element arithmetic or rounding order. A broad cross-group macro cache
is not reused; that separate v1 attempt failed correctness.

The initial E Q512/K1536 changing-max smoke passed raw-byte equality against
both v1 and canonical, exact preparation, immutable inputs/sources and two
trace replays. See `e-correction-cache-smoke-v1.json`. The seven-pair E resident
screen is effectively neutral: v1 287.017422 ms versus cache 286.998902 ms
(0.00645%, overlapping ranges); see `e-correction-cache-resident-v1.json`.
Do not promote this as a meaningful standalone gain.

## Second hypothesis: overlap next-half SFPU with previous-half packing

`pack_overlap` retains A until B is explicitly known committed, starts the
first eight compensation vectors of B while A is packing, then releases A
normally and finishes B while MATH refills A. It retains all original releases,
clears and producer ordering but reschedules one release. See the detailed
[ordering argument](pack_overlap/ORDERING.md). Independent source review and
device equality checks passed, including E Q256/K1024 and G Q2048/K1536 with
changing maxima. Scheduling was nevertheless not faster in the paired E
resident screen: split8 changed 287.124545 to 287.502062 ms (0.1315% slower).
Split2 was effectively neutral (286.976372 to 286.940051 ms); split4 was also
neutral (287.154665 to 287.165335 ms). Each corresponding
`e-pack-overlap*-resident-v1.json` contains alternating fresh baseline controls.
Do not retain these as performance wins. Split16 is prepared but unmeasured.

## Third hypothesis: specialize exactly-one online correction

`identity` checks all 64 first-column BF16 maxima in each two-row group for
bit-identical old/new values, explicitly rejecting exponent255 (NaN/Inf).
Opposite signed zeros take the ordinary path. When true, the canonical
accurate correction is exactly1: the private helper loads BF16 one into the
same correction registers and retains the existing MAD, addition, rounding,
high/low stores and releases. It skips correction generation, correction
broadcasts and replay correction loads, not state arithmetic.

The correction CB reserve/push/pop remains intact even though its data is not
read on the identity path. In particular its push retains the publication
fence for prior PV stores. Initializing the numerator helper reinstalls its
macro/address state whether or not correction exponentiation executed.

Both `identity-proof-lofi-v1.json` and `identity-proof-hifi2-v1.json` verify the
frozen correction path on every one of the 65,280 finite BF16 bit patterns,
including both zeros and all subnormals: every first-column output is raw
BF16 `0x3f80`, with two exact trace replays and immutable inputs. This is a
fixed-scale1/sqrt128 proof, not a claim for arbitrary scale or nonfinite input.
The initial attention transition smoke `e-identity-transition-smoke-v1.json`
uses Q512/K2048, distinct V and K chunk scales1,1,2,2; it matches canonical
and v1 raw output bytes through two trace replays. G also passes the wider
Q2048/K4096 paired-maxima case (`g-identity-transition-v1.json`). Independent
B transfer covers normal, growing maxima, outliers, common V and transitions.

The first E resident screen nevertheless regresses: 287.068294 to305.447280ms
(6.4023% slower). Original guard plus forced canonical arithmetic costs10.3263%
(287.085764 to316.731136ms; `e-identity-guardonly-resident-v1.json`). A strictly
equivalent branchless/unrolled guard reduces the regression to0.5306%
(286.990463 to288.513281ms; `e-identity-fastguard-resident-v1.json`). These are
fresh alternating v1 controls, not cross-run timing comparisons. Neither
identity implementation is a winner. The fastguard's initial JIT compile-only
failure was an in-macro `#pragma` placement; `_Pragma` fixed it and the retry
passed strict output equality. Failed and successful logs are retained.

`identity_early` precomputes exactly the same guard after issuing the next row's
first QK matmul, or the first partial PV for the final row. It retains the
original correction mailboxes and CB publication. It measures about 2.8% less
time for E/G resident and 1.36–1.39% less for distinct normal256K, but about
0.23–0.29% more time for short/fallback-heavy distinct inputs. See the final
[report](REPORT.md) for exact scopes, qualification and all paired evidence.
This is a conditional long-context candidate, not an unconditional default.

Run only through the coordinator's v1 `run_locked.sh`; one queued job at a
time and no nested lock/reset. Every record pins selected source dependencies,
baseline/candidate defines, CB specifications and input formats. Phase-profile
results, if added, are diagnostic; uninstrumented interleaved timing decides
whether to retain a candidate.
