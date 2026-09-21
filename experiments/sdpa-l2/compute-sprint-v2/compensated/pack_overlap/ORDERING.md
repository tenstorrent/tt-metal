# Two-half SFPU / pack overlap hypothesis

Unmeasured prototype; do not infer a gain. Numerator groups only, fixed BF16
half-DST / two query-tile rows / four D tiles. Original independent-update order,
all arithmetic replays and BF16 stores remain unchanged. No CB or dataflow edits.

For each pair of numerator groups:

1. MATH copies and commits A normally. PACK waits, performs the original SFPU
   update and issues its two width-two packs, but defers release of A.
2. MATH can still acquire/fill/commit B because half-sync has two slots. PACK
   explicitly waits for `MATH_PACK == 2`: A is still owned, so this proves B is
   also committed. It is stronger than the ordinary nonzero test. MATH cannot
   acquire another half while both are held.
3. An explicit previous-SFPU drain (not a pack drain) gates CFG/SYNC/SFPU before
   PACK changes its thread-local `DEST_TARGET_REG_CFG_MATH_Offset` to B.
   The packer's separate `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` still addresses
   A. Unchanged SFPU replays process the first eight vectors of B while the
   previous A packs may complete.
4. Three original-style macro-drain NOPs plus an explicit SFPU drain precede
   **canonical** A release: wait for pack, clear A, decrement semaphore, flip
   packer bank to B. The SFPU drain blocks SYNC as well as MATH, so the release's
   following STALLWAIT cannot overwrite an unmet SFPU wait gate.
5. Continue the remaining 24 vectors of B. MATH can now fill the freed A while
   B SFPU executes. Pack B normally, then canonical release clears/releases B.

Thus every deferred release is paired with one extra release in the following
update, and every two-group boundary has the original semaphore count, bank
parity, and cleared-state contract. The denominator path is unchanged. There
is no exceptional early return between those groups.

Source anchors:

- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:238`:
  commit posts MATH_PACK only after the current thread's MATH/SFPU work.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:63`:
  canonical release waits PACK, issues ZEROACC for the current half, releases
  MATH_PACK and selects the next packer bank.
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:676`:
  selecting the next packer bank writes PACK_SEC0, not the SFPU MATH offset.
- The ISA ZEROACC half-clear mode selects the bank directly from its immediate
  and does not apply its address modifier; B's SFPU RWC remains at 16 across
  releasing A. ISA source: `tt-isa-documentation` commit
  `5287a62727350bcef35f7b411d1b8a706172ec4c`,
  `WormholeB0/TensixTile/TensixCoprocessor/ZEROACC.md` (shared instruction),
  plus Blackhole `STALLWAIT.md`, `SEMWAIT.md` and `SFPLOAD.md`.

Required test sequence after independent ordering review: Q256/K1024 first,
bounded timeout; then odd K loops and multiple Q jobs with changing maxima;
then stress distributions/trace replay and alternating uninstrumented timing
against the frozen v1 winner. Any assertion/hang follows global dirty-guard
recovery; no local reset.
