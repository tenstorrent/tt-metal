// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TTI REPLAY LOWERING FRAMEWORK — shared scaffolding for every kernel body
// that opts into the Tensix replay-buffer lowering (piecewise_generic.cpp
// poly/exp/log/trig shapes AND piecewise_rational.cpp rational shapes).
//
// Any eval kind can opt in by providing (a) a constexpr body-slot count, (b) a
// hand-scheduled TTI body obeying the hazard rules below, (c) a per-tile
// sequence: [SFPLOADI constant pins] -> TT_REPLAY_RECORD(slots) -> body ->
// TT_REPLAY_REST(slots, n-1) -> TT_REPLAY_TILE_EPILOGUE(). The opt-in gate
// MUST be a constexpr `k<Kind>TtiReplay` that folds capacity + shape checks;
// when it is false the kind's existing sfpi loop compiles unchanged (clean
// fallback — the two paths coexist behind `if constexpr`).
//
// HAZARD RULES (silicon-derived; NO offline check catches violations):
//   1. A MAD-unit result (SFPMAD/SFPADD/SFPMUL/SFPADDI/SFPMULI/SFPLUT*) is NOT
//      readable by the immediately-following instruction at replay stream
//      rate. Every consumer must sit >= 1 instruction after its producer.
//      Bit/int-unit results (SFPEXEXP/SFPEXMAN/SFPSHFT/SFPSETEXP/SFPSETSGN/
//      SFPSETMAN/SFPIADD/SFPABS/SFPCAST/SFP_STOCH_RND/SFPLOADI) are readable
//      gap-0. SFPLOAD is treated as needing a gap (no counter-evidence).
//      2-element interleave provides gaps for free; 1-element bodies use the
//      independent side-computation ops as fillers, then SFPNOP.
//   2. The body's dst walk (ADDR_MOD_6 dest+= on the final SFPSTORE) displaces
//      the D-RWC counter, which the next tile's A2D datacopy MOP shares. The
//      per-tile sequence MUST end with TT_REPLAY_TILE_EPILOGUE() (SETRWC SET_D)
//      or subsequent tiles copy 64 rows off.
//   3. Replay buffer capacity is 32 slots (start_idx 0 + len <= 32).
//      Instructions past slot 31 execute once at record time and silently
//      vanish from replays. Gate on tti_replay_fits(); prefer <= 16 slots
//      (slots 16-31 are conventionally FPU-side).
//
// Predication (CC) in replay bodies is functionally sanctioned IF (a) the
// body is CC-balanced (SETCC ... ENCC fully inside the recorded body) and
// (b) any SETCC reading a MAD-unit result gets the rule-1 hazard gap.
//
// Register conventions (BH): L0-L7 general (bodies own them inside the tile
// loop), L8 = 0.8373, L9 = 0.0, L10 = 1.0, L11 = -1.0, L12/13/14 =
// vConstFloatPrgm0/1/2 (programmed once in the pre-tile-loop preamble).
// ============================================================================

#pragma once

#include <cstdint>

// ============================================================================
// SFPU write-then-read hazard model (per arch, from tt-isa-documentation)
// ----------------------------------------------------------------------------
// A replay stream issues back-to-back, so a consumer that reads an SFPU
// register on the cycle after its producer wrote it reads stale data. The two
// architectures differ ONLY in who is responsible for the gap and in WHICH ops
// produce the hazard -- the gap itself is one cycle on both.
//
//   Blackhole  (BlackholeA0/TensixCoprocessor/SFPMAD.md)
//     "hardware will ensure that on the next cycle ... hardware will
//      automatically stall the thread for one cycle"; the page lists
//      "automatic instruction scheduling" as an upgrade over Wormhole.
//      That auto-stall does NOT cover replay/SFPLOADMACRO streams, so bodies
//      still space the MAD family by hand.
//
//   Wormhole   (WormholeB0/TensixCoprocessor/SFPMAD.md and siblings)
//     "software must ensure that on the next cycle, the Vector Unit (SFPU)
//      does not execute an instruction which reads from any location written
//      to by the SFPMAD. An SFPNOP instruction can be inserted."
//
// The hazard SET is what actually differs. Wormhole carries the rule on 13
// pages; Blackhole's replay-relevant set is the MAD family alone:
//
//   both arches : SFPMAD SFPMUL SFPMULI SFPADD SFPADDI
//   Wormhole ALSO: SFPSWAP SFPSHFT2 MOVB2A MOVD2A SFPLUT SFPLUTFP32 SHIFTXB
//
// Bodies transcribed from Blackhole objdump therefore classify SWAP/SHFT2/MOV
// as safe to consume back-to-back -- true there, false here.
//
// SCOPE -- read this before blaming a Wormhole miscompare on the hazard model.
// The asymmetry above is documented fact, but it does NOT explain the currently
// broken bodies, and an early version of this comment wrongly claimed it did.
// A static walk of log_hw found every MAD-family gap already present and ZERO
// WH-only hazard ops in the body, and the measured failure is not value
// corruption at all: with replay forced on, log_hw writes only every other
// datum (even lanes carry the correct log, odd lanes are the untouched input
// passed through). A missing hazard gap corrupts values; it does not skip
// stores. So log_hw's defect is in dst/lane coverage and is still open.
//
// Treat this macro as available and correct, but unproven: no body opts in
// yet, and no measurement has yet attributed a Wormhole failure to it.
//
// USAGE. After an op in the WH-only set whose result is read by the NEXT
// instruction, write TT_SFP_GAP_HAZARD(); it expands to one SFPNOP on
// Wormhole and to nothing on Blackhole, keeping BH byte-frozen. Add
// `n_sites * kSfpuWhHazardGapSlots` to the body's slot formula so
// tti_replay_fits() still tells the truth -- a body that no longer fits gates
// itself off and falls back to sfpi (C-LOW-2), which is the desired failure.
// ============================================================================
// ============================================================================
// AddrMod SECTION BASE -- the actual cause of the Wormhole replay corruption
// ============================================================================
// Wormhole's AddrMod instruction field is two bits, so it selects sections 0..3
// directly and sections 4..7 only when ADDR_MOD_SET_Base is set
// (WormholeB0/.../RWCs.md:25-26 "Index += 4"; cmath_common.h:256 "use addr mods
// 4..7"). The SFPU LLK sets that base on entry
// (llk_math_eltwise_sfpu_common.h:20) -- but a raw TTI replay body does not go
// through _llk_math_eltwise_sfpu_start_, so the base is CLEAR when our body
// runs.
//
// Consequence: the body's dst-advance operand resolves to section 2, not the
// section 6 the kernel programs. Section 2 on Wormhole belongs to the A2D
// datacopy and carries dest incr 8 (llk_math_eltwise_unary_datacopy.h:283).
// Measured signature with the base clear: per 128 logical bf16 datums the body
// wrote 32 at column stride 2 and skipped 64 -- i.e. even columns of rows 0-3,
// then rows 4-7 missed entirely, exactly what an increment of 8 instead of 2
// produces under
//    Row = (Addr & ~3) + Lane/8;  Column = (Lane&7)*2 + ((Addr & 2) ? 1 : 0)
// (identical text on both arches: WormholeB0/.../SFPSTORE.md:87-91). An LReg
// covers FOUR Dst rows x EIGHT columns of ONE parity, not two rows of sixteen.
//
// This is also why the AddrMod operand-width fix could not move it: ADDR_MOD_7
// truncates to field 3 and ADDR_MOD_6 to field 2, which is precisely what the
// corrected operands emit. Same fields, same wrong section, byte-identical output.
//
// ENTER/EXIT MUST BE PAIRED. Leaving the base set makes the NEXT tile's A2D
// datacopy MOP resolve sections 4..7 and corrupts the following tile -- and the
// config PERSISTS ACROSS PROCESS INVOCATIONS, so an unpaired set poisons later
// runs on the same device (observed: a clean sfpi arm reading 1974143 pure ULP
// until a paired run scrubbed it).
//
// Measured on wh-lb-47, log_p6_s1, 256 tiles, interleaved repeats:
//   sfpi   0.4996284246444702  5.68 / 5.67 us
//   replay 0.4996284246444702  4.42 / 4.41 / 4.43 us   (1.28x, ULP identical)
//
// Blackhole indexes all eight sections directly from its three-bit field and
// has no base indirection, so both macros are no-ops there and BH stays frozen.
#define TT_REPLAY_ADDRMOD_ENTER() ((void)0)
#define TT_REPLAY_ADDRMOD_EXIT() ((void)0)

// ============================================================================
// AddrMod OPERAND WIDTH -- the reason BH-transcribed bodies corrupt on Wormhole
// ============================================================================
// SFPLOAD / SFPSTORE / SFPLOADMACRO encode AddrMod in a field that is TWO bits
// wide on Wormhole and THREE on Blackhole:
//
//   WormholeB0/TensixCoprocessor/SFPLOAD.md:10
//     TT_SFPLOAD(/* u4 */ VD, /* u4 */ Mod0, /* u2 */ AddrMod, /* u10 */ Imm10)
//   BlackholeA0/TensixCoprocessor/SFPLOAD.md:19
//     TT_SFPLOAD(/* u4 */ VD, /* u4 */ Mod0, /* u3 */ AddrMod, /* u10 */ Imm10)
//   BlackholeA0/.../SFPLOAD.md:14
//     "There is also a minor change to the encoding of AddrMod, though this is
//      invisible to the programmer if using the TT_SFPLOAD macro"
//
// Invisible only if the operand is in range for the arch. It is not here, and
// nothing masks it: TTI_SFPLOAD -> TT_OP_SFPLOAD shifts by 14 on Wormhole and
// 13 on Blackhole (tt_llk_{wormhole_b0,blackhole}/common/inc/ckernel_ops.h),
// and TT_OP is a bare `(opcode << 24) + params` with no masking. Only the
// runtime TT_SFPLOAD form validates (is_valid(sfpu_addr_mode, 2) on WH); the
// TTI_ immediate form we use in replay bodies does not.
//
// So a literal 7 on Wormhole becomes 7 << 14 = bits 14,15,16 -- and bit 16 is
// the LSB of Mod0, corrupting the format field. A 6 does the same. The dst walk
// survives (the low two bits land correctly), so this is a silent data defect
// rather than a crash.
//
// WHAT THIS FIX DOES AND DOES NOT BUY. It is a real encoding defect and worth
// carrying on correctness grounds alone. It is NOT, however, the cause of the
// known Wormhole replay failures: converting all 36 operands and re-measuring
// log_hw with replay forced on returned output BIT-IDENTICAL to the overflowing
// version (pure ULP 32768.996 either way). Do not cite this block as the
// explanation for a Wormhole miscompare -- that hypothesis was tested and
// falsified. On Blackhole both macros are pure renames, so BH is byte-frozen.
//
// The section numbers are identical on both arches (7 = dest hold, 6 = dest
// advance; see llk_math_eltwise_unary_sfpu.h, which calls .set(ADDR_MOD_7) on
// both). Only the encoding differs: Wormhole's two bits index sections 4..7
// because the SFPU entry sequence sets ADDR_MOD_SET_Base
// (cmath_common.h:256 "use addr mods 4..7"; WormholeB0/.../RWCs.md:25-26 adds
// 4). Hence section 7 <-> field 3 and section 6 <-> field 2 on Wormhole.
// Corroboration: Wormhole's own SFPU LLK encodes ADDR_MOD_3 at 106 SFPLOAD
// sites and never exceeds 3, while the Blackhole copy writes ADDR_MOD_7 96
// times -- the same two sections, different field values.
//
// Use these in replay INSTRUCTION operands. Keep passing the raw ADDR_MOD_n
// section number to addr_mod_t{...}.set(), which is arch-independent.
// Sections 7 (dst hold) and 6 (dst advance) are the same on every arch; only
// the operand encoding differs, so derive it from the field width.
#define TT_ADDRMOD_DST_HOLD ckernel::ADDR_MOD_7
#define TT_ADDRMOD_DST_ADV ckernel::ADDR_MOD_6

namespace sfpi {
constexpr uint32_t kTtiReplaySlots = 32;
constexpr bool tti_replay_fits(uint32_t body_slots) { return body_slots >= 1 && body_slots <= kTtiReplaySlots; }
}  // namespace sfpi

// Record the next BODY_SLOTS instructions into replay slot 0 AND execute them.
// ENTER is folded in (and EXIT into TT_REPLAY_TILE_EPILOGUE) so the pairing
// cannot drift as bodies are added: every recorded body gets the Wormhole
// section base, and every per-tile sequence must already end with the epilogue.
#define TT_REPLAY_RECORD(BODY_SLOTS)       \
    do {                                   \
        TT_REPLAY_ADDRMOD_ENTER();         \
        TTI_REPLAY(0, (BODY_SLOTS), 1, 1); \
    } while (0)
// Re-issue the recorded body N_REST more times (one vector / pair each).
#define TT_REPLAY_REST(BODY_SLOTS, N_REST)                                                                         \
    do {                                                                                                           \
        _Pragma("GCC unroll 32") for (int _r = 0; _r < (int)(N_REST); _r++) { TTI_REPLAY(0, (BODY_SLOTS), 0, 0); } \
    } while (0)

// Re-issue a recorded core, then execute a non-recorded terminal on the SAME
// destination row before the next replay. This is the capacity escape hatch
// for a full core: the suffix remains in the one ordinary element loop and its
// final STORE owns the dst advance. The core must leave D-RWC unchanged and
// produce every suffix input on every replay; the suffix must retain no mutable
// state across rows. These rules prevent the silent "slot 32 executes only
// while recording" failure.
// Mandatory per-tile epilogue: re-zero the D-RWC counter (hazard rule 2).
#define TT_REPLAY_TILE_EPILOGUE()                                                      \
    do {                                                                               \
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D); \
        TT_REPLAY_ADDRMOD_EXIT();                                                      \
    } while (0)
// Pin a full fp32 constant into a general LREG (2 issue slots, per tile).
