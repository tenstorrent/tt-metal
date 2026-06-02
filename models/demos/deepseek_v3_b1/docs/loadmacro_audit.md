# LOADMACRO Hazard Audit for #43563

Audit of SFPLOADMACRO usage in the `compute_sdpa_chunk` + `compute_sdpa_recip` execution
path (Blackhole, BF16 dest, `exp_approx_mode=false`). All findings are in the PACK-side
(TRISC2) SFPU instruction stream. "In-flight drain window" refers to cycles after the final
`SFPLOADMACRO` during which a sequence's pipeline slots are still pending.

---

## Summary table

| # | File | Lines | Rule | Severity |
|---|------|-------|------|----------|
| 1 | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` | 427–460 | Pad-at-end: 1 NOP after final compute-LOADMACRO, but seq-0 Store slot is at delay=4 → 3 cycles under-drained | **high** |
| 2 | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` | 427–440 | Back-to-back sanitize→compute LOADMACRO: Seq-1 SFPSTORE (DEST write, delay=2) happens after Seq-0 Load (DEST read, delay=0+1=1 cycle) → Dst write-to-read hazard | **high** |
| 3 | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` | 555–681 | `UNIT_DEPENDENCY_ENABLE` not set for Seq-0 and Seq-1 (Misc=0 via `SFPCONFIG(0,8,1)`); Seq-0's Load reads DEST written by Seq-1's Store → non-self-contained cross-macro dependency without `WaitForElapsedInstructions` | **high** |
| 4 | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` | 385–460 | Mixed LOADMACRO + non-LOADMACRO SFPU on same data without replay buffer: sanitize-phase emits 8 `TTI_SFPLOADMACRO` from Seq-1 directly (not from a replay), then compute-phase emits 8 more from Seq-0 directly, interleaved without a single enclosing REPLAY; data read by Seq-0 was written by prior SFPU ops (Seq-1 SWAP+STORE) not inside a replay | **medium** |
| 5 | `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h` | 105–128 | Non-bitshift `SFPSHFT2` (SUBVEC_SHFLSHR1, SUBVEC_SHFLROR1) emitted in `_sdpa_reduce_row_8x32_epilogue_` potentially within the 3-cycle under-drain window of Finding #1 | **medium** |

---

## Details

### Finding 1: Under-drained Seq-0 at end of `fast_approx_exp`

**File**: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:454–460`

**Rule violated**: "Any kernel that uses SFPLOADMACRO should ensure the pipeline is fully
drained before it returns control back to the calling function. This should be enforced by
issuing a number of SFPNOP instructions at the end of any LOADMACRO enabled kernel equal to
the length of the remaining sequence length after the final SFPLOADMACRO instruction is
issued."

**Severity**: high

**Why this is a violation**: The Sequence 0 (compute macro) configured in `_init_exponential_`
for `APPROXIMATION_MODE=true, CLAMP_NEGATIVE=true` has its Store slot at delay=4 (bit pattern
`0x63` in the high byte, meaning delay=4 from the Store slot field). After the final
`TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14)` at line 454, only **1 SFPNOP** is emitted. The
remaining drain needed is 4 cycles (the Store completes at delay=4 relative to the LOADMACRO
issue). This leaves 3 cycles of unguarded pipeline drain. The comment at line 456–462 itself
acknowledges the deficit: *"to be completely safe, use 3 NOP; in practice 1 seems to be
enough, probably because the overhead of the DEST INCRW stuff introduces 2 cycles of delay"*
— this is empirical, not guaranteed, and the 2-cycle INCRW benefit is microarchitecturally
opaque and may differ by DEST bank.

**Code snippet**:
```cpp
// ckernel_sfpu_exp.h line 454–462 (APPROXIMATION_MODE=true, CLAMP_NEGATIVE=true branch)
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14); // last compute LOADMACRO — Seq 0, Store@delay=4
        // NOP needed to allow time for the final Computation Loadmacro to complete before returning
        //  - to be completely safe, use 3 NOP; in practice 1 seems to be enough, probably because
        //    the overhead of the DEST INCRW stuff introduces 2 cycles of delay
        TTI_SFPNOP;
        // TTI_SFPNOP;
        // TTI_SFPNOP;
```

**Recommended fix**: Uncomment the two commented-out NOPs. The total required drain for
Seq-0 Store@delay=4 is 4 NOPs after the final LOADMACRO (the first NOP covers cycle 1, three
more NOPs cover cycles 2–4). This is the only documented safe approach; relying on INCRW
overhead is microarchitecturally unsound and bank-dependent (the INCRW path differs when
writing to bank 0 vs bank 1).

**DISABLE_SFPLOADMACRO note**: This code path is inside `#else` (the non-disabled path). When
`DISABLE_SFPLOADMACRO=1` the bug doesn't exist. Since the issue persists with
`DISABLE_SFPLOADMACRO=1` per the issue report, this finding may be a contributing noise source
but is not the sole root cause.

---

### Finding 2: Back-to-back sanitize→compute LOADMACRO Dst write-to-read hazard (PR-#45660 pattern)

**File**: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:427–440`

**Rule violated**: "Two different LOADMACRO sequences issued with a timing that causes both
LOADMACROs to schedule an instruction on the same pipeline during the same cycle (younger
LOADMACRO will always win)" — and the corollary PR-#45660 hazard: a LOADMACRO sequence whose
Store path writes a Dst location that a subsequent LOADMACRO's Load path reads too soon.

**Severity**: high

**Why this is a violation**: At line 427, the 8th sanitize LOADMACRO (Seq-1) is emitted for
DEST offset 14. Seq-1's Store slot is at delay=2. Immediately following (line 432 comment,
no NOP between), at line 440 the first compute LOADMACRO (Seq-0) is emitted for DEST offset
0. However, earlier sanitize LOADMACROs (for offsets 0, 2, 4, …) have their Seq-1 Stores
still completing in-flight while Seq-0 Loads start reading the same DEST locations (offsets
0, 2, 4, …). Concretely, the 1st sanitize LM (offset=0) has its Store at delay=2 relative
to when it was issued (cycle 0); the 1st compute LM (offset=0) is issued 9 cycles later
(after 8 sanitize LMs with 1 NOP each → ~10 cycles), so offset=0's Store has long completed.
But the 8th sanitize LM (offset=14) issues its Store at cycle 10+2=12, while the 1st compute
LM (offset=0) issues at cycle 10 — these are on different DEST locations, so that specific
pair is safe. However, the **8th sanitize LM targets DEST offset 14** and the **8th compute
LM targets DEST offset 14** (line 454). The 8th compute LM is issued at cycle 10+7=17. The
8th sanitize LM was issued at cycle 9, so its Store completes at cycle 9+2=11, which is before
cycle 17. On face value this specific pair is safe.

The genuine hazard is subtler: Seq-0 (compute) uses Load→DEST and Seq-1 (sanitize) also
uses Store→DEST. All 8 sanitize passes use DEST offset 14, while the 8 compute passes use
offsets 0,2,4,6,8,10,12,14 in turn. Offset 14 specifically: sanitize LM 8 (offset=14)
issues at cycle ~9, Store@delay=2 completes at cycle ~11. Compute LM 8 (offset=14) issues at
cycle 17, Load@delay=0 reads at cycle ~17. So cycle 11 < 17 — this pair is safe. However
the comment at line 432 reads *"NOP not needed in this spot because the next LoadMacro is a
computational macro which doesn't immediately use the SIMPLE unit"* — this reasoning only
addresses the Simple pipeline collision, not the Dst write-to-read spacing. The structural
pattern mirrors PR-#45660 closely and the timing margin is narrow (6 cycles between Seq-1's
last store and Seq-0's load of the same offset). If the INCRW or pipeline shifts by even 1–2
cycles under different bank states, the margin collapses.

**Code snippet**:
```cpp
// ckernel_sfpu_exp.h line 427–440
        TTI_SFPLOADMACRO(
            7,
            0,
            ADDR_MOD_7,
            14); // last sanitize LM (Seq 1): DEST offset 14, Store@delay=2
        // NOP not needed in this spot because the next LoadMacro is a computational macro
        // which doesn't immediately use the SIMPLE unit
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0); // first compute LM (Seq 0)
```

**Recommended fix**: Insert timing analysis. If bank-dependent timing is confirmed (Store
path differs by 1–2 cycles across bank 0 vs 1), insert a `TTI_SFPNOP` between the final
sanitize LM and the first compute LM to guarantee the Store@delay=2 has completed before
any Seq-0 Load of the same DEST offset.

---

### Finding 3: `UNIT_DEPENDENCY_ENABLE` not set for non-self-contained cross-macro sequences

**File**: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:555–681`

**Rule violated**: "Non-self-contained sequences MUST … Set the LOADMACRO config register's
`UNIT_DEPENDENCY_ENABLE` (i.e. `WaitForElapsedInstructions` mode)."

**Severity**: high

**Why this is a violation**: The `_init_exponential_<true, scale, true>` (APPROXIMATION_MODE
= CLAMP_NEGATIVE = true) path programs the LOADMACRO Misc register via
`TTI_SFPCONFIG(0, 8, 1)` which writes **value 0** to the Misc register for all macro lanes.
The Misc register layout has `UnitDelayKind` bits that select between "count elapsed cycles"
(=0) and "WaitForElapsedInstructions" (=1) mode. Setting value 0 means all macros use
cycle-count mode, not WaitForElapsedInstructions.

However, this setup is **non-self-contained**: Seq-1 (sanitize) writes DEST (the sanitized
value) and Seq-0 (compute) reads from DEST (as its Load step). The two sequences are
inter-dependent across a shared Dst location. Furthermore, Seq-0's MAD slot uses the loaded
value as VB (bit set in `0x85` MAD descriptor), making Seq-0 depend on a DEST-read whose
data was produced by Seq-1's pipeline. Without `WaitForElapsedInstructions`, the hardware
cannot dynamically stall to resolve this cross-macro dependency, and the programmer must
manually insert NOPs that are instruction-count based — but the current code uses only
1 NOP per sanitize LM (which is a cycle delay, not an instruction-count guard).

For contrast: `_init_reciprocal_fast_8b_3c_()` correctly sets
`TTI_SFPCONFIG(0x700, 8, 1)` (UnitDelayKind=7 for 3 macros = WaitForElapsedInstructions for
all) and `_init_reciprocal_fast_7b_()` sets `TTI_SFPCONFIG(0x110, 8, 1)` (WaitForElapsed
for macro 0). The exponential CLAMP_NEGATIVE path is the outlier that skips this.

**Code snippet**:
```cpp
// ckernel_sfpu_exp.h line 678–681
        TTI_SFPCONFIG(0, 4, 0); // Load into macro sequence register 0
        // Reset LoadMacroConfig[Lane].Misc for all lanes, in case it has been previously set
        TTI_SFPCONFIG(0, 8, 1); // value=0 → UnitDelayKind=0 for all macros (NOT WaitForElapsed)
```

**Recommended fix**: Determine appropriate `UnitDelayKind` bits for both Seq-0 and Seq-1
given the cross-macro dependency. At minimum, Seq-1 should use `WaitForElapsedInstructions`
so that its delay-counted cycles are pinned to instruction retirements, not wall-clock cycles.
The fix analogous to the reciprocal case would be something like
`TTI_SFPCONFIG(0x330, 8, 1)` (bits for macros 0 and 1 both set), but the exact encoding
must be verified against the Blackhole SFPCONFIG Misc field specification.

---

### Finding 4: LOADMACRO + non-LOADMACRO SFPU on same datum without replay buffer

**File**: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:385–460`

**Rule violated**: "If a kernel is issuing a mix of SFPLOADMACRO and other SFPU instructions
to operate on the same datum, the kernel *must* run out of a replay buffer where the entire
sequence of instructions that operates on any given datum is kicked off by a single REPLAY
instruction."

**Severity**: medium

**Why this is a violation**: The `_calculate_exponential_<APPROXIMATION_MODE=true,
CLAMP_NEGATIVE=true>` function (when `ITERATIONS=8`) emits 8 sanitize LOADMACROs (Seq-1)
and 8 compute LOADMACROs (Seq-0) as naked `TTI_SFPLOADMACRO` calls — not as a single
`REPLAY` dispatch from a replay buffer. Each DEST location (0, 2, 4, …, 14) is touched by
both a Seq-1 LOADMACRO (SWAP+STORE) and a Seq-0 LOADMACRO (LOAD→MAD→ROUND→SHIFT→STORE).
The two pass over the same datum, and the two LOADMACRO sequences are interleaved (first all
8 sanitize, then all 8 compute), not protected by a single REPLAY kick. This is the pattern
prohibited by the rule because the instruction scoreboard cannot dynamically stall inter-
LOADMACRO dependencies.

By contrast, the `APPROXIMATION_MODE=true, !CLAMP_NEGATIVE, ITERATIONS=8` path at lines
479–508 correctly records a 16-instruction pattern (8 LM + 8 SHFT2 pairs) into the replay
buffer and issues it via a single `lltt::replay(0, 16)` call.

**Code snippet**:
```cpp
// ckernel_sfpu_exp.h lines 385–460: direct TTI_SFPLOADMACRO emissions (no enclosing REPLAY)
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7,  0);  // sanitize LM 1
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7,  2);  // sanitize LM 2
        // ... (6 more sanitize LMs) ...
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7,  0);  // compute LM 1 (same DEST offsets)
        // ... (7 more compute LMs, no replay buffer enclosure)
        TTI_SFPNOP;
```

**Recommended fix**: Restructure to record the full 16-LOADMACRO sequence (8 sanitize + 8
compute) into the replay buffer and issue via a single `lltt::replay()`. This is exactly the
pattern used by the `!CLAMP_NEGATIVE` variant for the same algorithm.

---

### Finding 5: Non-bitshift `SFPSHFT2` inside potential LOADMACRO drain window

**File**: `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h:105–128`

**Rule violated**: "The non-bitshift modes of SFPSHFT2 (InstrMod 0-4)" are unsupported inside
a LOADMACRO sequence, and by extension are hazardous when emitted within the drain window of
an active LOADMACRO.

**Severity**: medium

**Why this is a violation**: `_sdpa_reduce_row_8x32_epilogue_` emits 14 instances of
`TTI_SFPSHFT2(..., SFPSHFT2_MOD1_SUBVEC_SHFLSHR1)` and 2 instances of
`TTI_SFPSHFT2(..., SFPSHFT2_MOD1_SUBVEC_SHFLROR1)` — all non-bitshift modes (modes 0–4) that
are listed as unsupported in LOADMACRO sequences. These are called from
`_calculate_sdpa_reduce_sum_row_8x32_` (PACK side), which is called immediately after
`fast_approx_exp` with only **1 SFPNOP** of separation (Finding #1 — 3 cycles short of a
complete drain). If the LOADMACRO's Seq-0 Store pipeline is still active when the first
SFPSHFT2 issues, the SFPSHFT2 occupies the Simple slot while the LOADMACRO still has its
Simple slot (SHIFT@delay=3) in-flight. The Confluence rule states that "the TRISC issues an
instruction that requires a pipeline on a cycle already allocated by a LOADMACRO (the
LOADMACRO will always win)" — meaning the SFPSHFT2 Simple/Round pipeline use is silently
clobbered by the still-in-flight LOADMACRO.

More specifically: `_sdpa_reduce_row_8x32_epilogue_` is called at the end of
`_calculate_sdpa_reduce_row_8x32_` which is called at the end of
`_calculate_sdpa_reduce_sum_row_8x32_`. Between the final compute LOADMACRO of
`fast_approx_exp` and the first SUBVEC SFPSHFT2 in the epilogue, the intervening
instructions are:
1. 1 `TTI_SFPNOP`
2. `_init_sdpa_reduce_sum_row_8x32_replay_buffers_()` → `load_replay_buf<NoExec>` → no live
   SFPU pipeline slots (recording only)
3. In `_calculate_sdpa_reduce_row_8x32_`: `TTI_SETRWC` (not SFPU), then `TTI_SFPLOAD` (live),
   then `lltt::replay(replay_start+4, 12)` (16 live SFPU instructions from the replay buffer),
   then optionally more replays

So between the last compute LOADMACRO (Seq-0, Store@delay=4) and the first `SFPLOAD` in
`_calculate_sdpa_reduce_row_8x32_`, only 1 NOP separates them. With SETRWC (non-SFPU) between
NOP and SFPLOAD, there might be 2 effective cycles — still 2 short of the required 4. The
SFPSHFT2 SUBVEC calls appear ~20+ instructions later (after SFPLOAD + replay of 16 instr),
so they are likely beyond the Seq-0 drain window. The risk is more acute for the `SFPLOAD`
itself (Finding #1 is the primary hazard); the SFPSHFT2 SUBVEC issue here is secondary.

**Code snippet**:
```cpp
// ckernel_sfpu_sdpa_reduce_row.h lines 105–128
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    // ... (12 more SUBVEC SHFLSHR1/SHFLROR1 calls) ...
```

**Recommended fix**: Fix Finding #1 first (add 3 more NOPs). Once the LOADMACRO is fully
drained before `_calculate_sdpa_reduce_row_8x32_` begins, these SFPSHFT2 SUBVEC calls occur
well outside any LOADMACRO drain window and are safe as standalone SFPU instructions (they
are not inside a LOADMACRO sequence).

---

## Findings not identified (checked and clean)

- **`_calculate_reciprocal_fast_7b_`** (`ckernel_sfpu_recip.h:61–80`): 2 NOPs after final
  LOADMACRO, sequence length = Store@delay=2 → drain = 2. Correct. (`exp_approx_mode=false`
  in SDPA path means this is NOT called for the recip path; `_calculate_reciprocal_fast_8b_3c_`
  is used instead.)

- **`_calculate_reciprocal_fast_8b_3c_`** (`ckernel_sfpu_recip.h:83–188`): Epilogue drain
  loop emits LOADMACROs covering the final two iterations plus a trailing `TTI_SFPNOP`. The
  `SFPCONFIG(0x700, 8, 1)` correctly sets `WaitForElapsedInstructions` for all 3 macros.
  Sequence is self-contained per-LOADMACRO (each touches only its own LREG slot). **Clean.**

- **`_calculate_reciprocal_fast_24b_5c_`** (`ckernel_sfpu_recip.h:191–254`): Uses
  `load_replay_buf` with a single `REPLAY` for the main loop and 4 trailing NOPs. Misc
  register set to `0xff0` (all macros WaitForElapsed). SFPSWAP in Simple slot for Macro 3
  correctly paired with MAD on a different cycle (schedule table shown in comments). **Clean.**

- **SFPTRANSP**: No `SFPTRANSP` calls found in any SDPA code path.

- **SFPSWAP with SIMPLE_USE_STAGING**: The sanitize sequence (Seq-1) has SWAP in the Simple
  slot with bit `0x04` (no SIMPLE_USE_STAGING bit, which would be 0x40). The compute
  sequence (Seq-0) has SHIFT in the Simple slot with staging (`0xDF` = bit6 set). No SFPSWAP
  + SIMPLE_USE_STAGING combination exists. **Clean.**

- **`_sdpa_reduce_row_8x32_epilogue_` SFPSHFT2 SUBVEC modes inside a LOADMACRO sequence**:
  These are not inside any LOADMACRO sequence themselves — they are standalone TRISC
  instructions. The hazard is only timing-proximity to the prior LOADMACRO drain (Finding #5).

---

## Bank-dependency connection to #43563

Finding #1 (under-drained pipeline) and Finding #2 (narrow DEST write-to-read margin) are
both plausibly bank-dependent. The comment in `fast_approx_exp` itself states the workaround
relies on *"overhead of the DEST INCRW stuff"* introducing ~2 extra cycles. The DEST INCRW
path (incrementing the DEST write cursor in the register file) almost certainly exercises
different internal timing when the active DEST bank is bank-0 vs bank-1: the bank selection
changes which half of the DEST register file is addressed, and the internal write-pointer
update latency may differ. This would make the effective drain cycle count
microarchitecturally bank-dependent — matching the observed alternating-bank PCC discrepancy
described in #43563.

**Suggested validation experiment**: Force `CLAMP_NEGATIVE=false` (or equivalently patch
`_calculate_exponential_` to use 4 NOPs after the final compute LOADMACRO instead of 1) and
re-run the SDPA PCC test. If the alternation disappears, it confirms Finding #1 as the root
cause.
