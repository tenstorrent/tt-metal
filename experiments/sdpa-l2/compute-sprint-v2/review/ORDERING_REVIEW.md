# Independent B compensation scheduling review

Baseline: frozen v1 B winner, exactly `SDPA_BF16_BLOCK_STATE`,
`SDPA_BF16_CORRECTION_REUSE`, and `SDPA_BF16_CORRECTION_FENCE`, with canonical
B numerical defines. Q256/K512/D128, BF16 DST, HiFi2, two input slots unchanged.

## Remaining exposed work

V1 B counters: FPU64.239%, SFPU38.044%, both20.555%, neither18.273%.
SFPU-only occupancy is therefore17.489%. Four numerator state batches and one
denominator batch occur per two-row SALAD. Each numerator batch has six state
tiles and a broadcast correction tile in half-DST. Repeated copy↔COL-broadcast
configuration and SFPU→PACK→DST-release dependencies are deeper targets than
the setup-cache changes rejected in v1. Occupancy alone does not identify the
critical path; phase markers and isolated timing are required to quantify it.

## Correction-cache source argument

The initial idea of merely keeping correction slots across `tile_regs_release`
is invalid: `tt_llk_blackhole/llk_lib/llk_pack_common.h:63` first waits for PACK,
then `ZEROACC(CLR_HALF)` clears the complete active half, releases the semaphore
and flips the destination-half selector. This independent review identified
that hazard before the prototype was tested.

The revised fixed-geometry candidate changes numerator iteration order:

| Batch | Row / output-column pair | DST half relative to entry | Correction | Release |
|---|---|---|---|---|
| 0 | row0 / j0 | A | load slot6 | preserve, no ZEROACC |
| 1 | row1 / j0 | B | load slot6 | preserve, no ZEROACC |
| 2 | row0 / j2 | A | reuse slot6 | canonical full-half clear |
| 3 | row1 / j2 | B | reuse slot6 | canonical full-half clear |

A/B denote alternating halves, not an assumption that entry selects physical
half0. All six state slots are overwritten before either reuse. The BF16 A2D
datacopy MOP is **MOVA2D(DEST_NORM)**, not an accumulation instruction:
`llk_math_eltwise_unary_datacopy.h:437–461`. `copy_block` calls that same datacopy
MOP for each tile. Thus stale high/low/chunk tiles are not accumulated into the
new copies. Each two-tile copy covers its entire destination slots; three calls
cover0–5. Numerator SFPU stores affect0–3 only, leaving correction slot6 intact.

The private preserving release copies the original PACK-completion stall,
MATH_PACK semaphore operation, half flip and destination register selection;
only the first two ZEROACCs are omitted. Original releases in batches2 and3
clear **both** halves before denominator work and all subsequent matmuls.
There is no early exit inside this fixed four-batch sequence. Static assertions
must retain rows2/columns4/DST8 and BF16 half-DST restrictions.

Reordering different output tiles does not change any per-element arithmetic.
Old and current recurrent CBs remain distinct. Writes to j0 high/low planes do
not alias j2 chunk reads. Every SFPU batch runs the same replay and completes its
stores before packing/release. The replay/macro setup remains outside the four
batches and no broadcast/copy helper writes the PACK-thread SFPU replay state.
Each replay starts with its own correction load, including the reuse-vector
optimization already present in v1, so no LREG value is assumed to survive.

This is a source argument, not qualification. Independent transfer must compare
the frozen v1 B source, a disabled private wrapper, and the new candidate using
raw BF16 bits, changing maxima, multiple distinct Q jobs, odd/even K counts,
held-out seeds and trace replay. No generalization to different dimensions,
mask/ring/causal schedules, or FP32 DST is justified by this narrow argument.

## Paired-half SFPU/PACK overlap source review

The second candidate keeps the existing row-outer numerator iteration. For each
row it packs column-pair j0 in half A but defers release, waits for j2 in half B,
executes the first eight unchanged compensation vectors on B while A packs,
canonically releases A, executes the remaining 24 vectors on B, then packs and
releases B normally. Denominator scheduling is unchanged.

The readiness argument is specific to SyncHalf: `_llk_math_pack_sync_init_`
initializes MATH_PACK maximum=2. Math commit posts only after MATH/SFPU complete
(`cmath_common.h::set_math_semaphores`). With A held and no intervening release,
observing count=2 therefore proves B is also committed. Math cannot acquire a
third half until A is released. This does not rely on `tile_regs_wait`, whose
normal nonzero test alone is insufficient when A remains held.

`SETC16(DEST_TARGET_REG_CFG_MATH_Offset)` writes thread-local SFPU addressing.
Outstanding pack instructions use separate PACK_SEC0 addressing. Canonical
release flips only PACK_SEC0 (`cpack_common.h::select_packer_dest_registers`).
ZEROACC HALF chooses its half from the immediate and does not apply AddrMod;
the ISA functional model therefore preserves the SFPU RWC=16 reached after
eight vectors. Even-vector splitting preserves the existing correction reuse
across adjacent columns. Both halves still receive canonical clears/releases.

This review requested two explicit conservative fences before device launch:
SFPU completion must fence CFG/SYNC/SFPU before switching B's MATH offset;
the split SFPU drain must fence MATH/SYNC/SFPU before canonical A release.
The SYNC bit prevents the next STALLWAIT from replacing an unmet wait gate.
Neither added fence waits for PACK before beginning B's SFPU work. Lowp's owner
applied both requirements. Source review permits a bounded smoke; hardware
equality, replay, multi-Q/odd-K and stress tests remain necessary. No generic
API replacement or wider geometry safety is implied.

## Exact identity-correction specialization (source proposal)

For finite, bit-identical BF16 previous/current maxima, subtraction produces
zero (either signed zero is harmless), and the fixed finite attention scale
preserves zero. The frozen `calculate_sdpa_exp_correction` calls
`_sfpu_exp_fp32_accurate_`: input zero produces j=f=i=0; the last two polynomial
operations produce y=r=1; exponent reconstruction has e=127 and returns exact
FP32 1.0. Packing to BF16 preserves 0x3f80. This is an exact special case of the
selected implementation, not replacing its approximate behavior by ideal exp.

Guard each two-query-tile correction group separately by all 64 first-column
BF16 values. Compare bits and reject exponent bits 0x7f80 (Inf and NaN) before
using identity. Unequal values, including opposite signed-zero encodings, take
the unchanged fallback. The identity result must reach all three RISCs through
the established mailbox protocol.

The exact numerator specialization loads L6 with
`SFPLOADI(6, SFPLOADI_MOD0_FLOATB, 0x3f80)` then executes the same 14 arithmetic/
store instructions from replay offset1 for every vector. Denominator similarly
loads L6/L7=1 then executes replay offset17 length14. It must **not** substitute
ADD for MAD or remove compensated rounding/residual updates. Constant loads
belong after the applicable SFPU wait, not before a preceding live macro stream.

Correction broadcast setup/copies can be skipped for the guarded path; all
state copies and per-element operations remain. The correction CB still needs
its original reserve/push/wait/pop protocol even if its payload is unused:
publication drains prior PACK work and is the synchronization proof that
permitted v1 to remove the separate PACK_DONE triplet. Removing that token along
with the arithmetic would invalidate the ordering proof.

### Guard scheduling refinements

The unrolled guard OR-reduces all 32 first-column differences per tile and an
old-value nonfinite predicate. This is logically identical: when all bits
match, the new value is finite iff the old value is finite; any mismatch already
forces fallback. It scans the complete first tile before early exit, so fallback
scan cost can differ. `_Pragma` rather than a raw `#pragma` is needed inside the
UNPACK macro argument.

The earlier-scan prototype moves only the UNPACK RISC comparisons. For QK row
group j>0, the preceding `sub_exp_block_bcast_cols` has waited for current max
group j−1, and the earlier reduction has waited for the corresponding previous
max. After issuing j's first QK matmul, the prototype scans j−1 while that issued
work proceeds. For the last group, the phase2 drain has already waited for its
maximum before the first partial PV issue; the last scan follows that issue.
Neither previous nor current maximum read pointer/payload is changed before the
original correction site. Per-inner-K-call flags prevent stale cross-K or
cross-Q reuse. MATH/PACK receive the decision at the unchanged original mailbox
rendezvous; they must not use their own unfilled local flag arrays. Original CB
reserve/push/wait/pop and all correction/state arithmetic are retained. This
source review covers only the asserted eight-Q-tile, two-row QK/PV,
materialized-V geometry and is not hardware qualification.
