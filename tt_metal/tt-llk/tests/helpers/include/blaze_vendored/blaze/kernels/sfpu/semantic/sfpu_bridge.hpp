// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// BUILTIN-BRIDGE KIT — lane EX, 2026-08-21.
//
// Typed wrappers for the SFPU mechanisms the sfpi TYPE SYSTEM does not (yet)
// spell, but the pin-18 COMPILER already models as builtins with audited
// multi-result/config effects (rvtt.md: rvtt_sfpswap_indexed_int,
// rvtt_sfptransp8_int, rvtt_sfpwriteconfig_v).  Each wrapper is
// instruction-for-instruction against the hand kernels' TTI words; semantics
// cross-checked against tt-isa-documentation BlackholeA0 (SFPSWAP.md,
// SFPTRANSP.md, SFPSHFT2.md, SFPCONFIG.md, SFPLOAD.md, SFPSTORE.md).
//
// This header is ALSO the concrete compiler-vocabulary invention queue: every
// wrapper here is a candidate typed-sfpi surface (the way subvec_shflror1 /
// subvec_transp already graduated).
//
// NAMED RESIDUAL RISK (TEN-2932): while LaneConfig.ENABLE_DEST_INDEX is set,
// only SFPLOAD/SFPLOADI/SFPSWAP/SFPTRANSP may write LReg[4..7].  The compiler
// has NO model of this window; lifts using dest_index_window must keep the
// window's SFPU content to exactly the exempt opcodes (loads, loadi,
// indexed_swap, transp8) and the compile gate must inspect the emitted window
// for allocator-inserted SFPMOVs into L4..L7.  A compiler-side window model
// (config-ownership lane) is the honest fix; until then this is a
// per-kernel-gated bridge, not a general recipe.

#include "sfpi.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

// ---------------------------------------------------------------------------
// LaneConfig.ENABLE_DEST_INDEX (bit 2) window control.
// Original spelling: TTI_SFPCONFIG(0x4 / 0x0, 0xF, 1)  (imm16-form, VD=15).
//
// VOCABULARY GAP (invention-queue item): the compiler exposes only the
// VALUE-form SFPCONFIG (__builtin_rvtt_sfpwriteconfig_v), which stages the
// value in LReg[0] — a 9th live vector.  The toggles sit at points where all
// 8 LRegs carry sort state, so the value-form is PRESSURE-INFEASIBLE there
// (lreg-pressure-exceeded, compile-proven); the imm16-form needs no register,
// which is exactly why the hand kernels use it.  Until an imm-form builtin
// (rvtt_sfpconfig_i class) exists, this is the ONE raw TTI word in the
// bridge lifts — emitted verbatim, identical to the original's word.
// Mixing a volatile instrn_buffer store with the volatile SFPU builtins keeps
// program order (both are volatile); the production kernels themselves mix
// TTI_SFPCONFIG with typed sfpi bodies the same way.
// ---------------------------------------------------------------------------
template <bool Enable>
sfpi_inline void set_dest_index_tracking()
{
    TTI_SFPCONFIG(Enable ? 0x4 : 0x0, 0xF, 1);
}

// Value-form spelling for reference/low-pressure use (state-equivalent modulo
// LaneConfig bits 17:16, which are reserved-zero).
template <bool Enable>
sfpi_inline void set_dest_index_tracking_valueform()
{
    sfpi::vInt cfg = Enable ? 4 : 0;
    __builtin_rvtt_sfpwriteconfig_v(cfg.get(), 15);
}

// ---------------------------------------------------------------------------
// Indexed compare-and-swap: ONE SFPSWAP executed under ENABLE_DEST_INDEX.
// (va, vb) are the sort keys; (ca, cb) the companion payloads that swap on
// the same per-lane decision (SFPSWAP.md ENABLE_DEST_INDEX leg).  The
// compiler's register alternatives pin companion == value + 4, so the emitted
// word is exactly the hand kernels' TTI_SFPSWAP(0, va, vb, Mod).
// Mod values (SFPSWAP.md == ckernel p_sfpswap):
//   0 UNCONDITIONALLY, 1 ALL_ROWS_MAX (VC=max, VD=min),
//   2 ROWS_01_MAX (SUBVEC_MIN01_MAX23), 3 ROWS_02_MAX (SUBVEC_MIN02_MAX13).
// ---------------------------------------------------------------------------
// LANE FD EXECUTION FIX (2026-08-21, first CRAQ execution of these lifts):
// the builtin's operand contract is arg0 = VD (gets MIN under mod1=1) and
// arg1 = VC (gets MAX), select(N) = argN's register result — proven by
// sfpi_lib.h's silicon-proven min_max (max(a,b) calls
// __builtin_rvtt_sfpswap(b, a, mod) and takes select2(r, 1)).  Lane EX's
// original bridge assumed arg0 = VC from an assembly-print probe and came
// out direction-inverted (the moe lift selected the BOTTOM-8 on the pinned
// sim; the byte-exact original selected the top-8).  These wrappers keep the
// hand kernels' reading — va plays VC (max under mod1=1), vb plays VD — by
// passing va as arg1.
template <unsigned Mod>
sfpi_inline void indexed_swap(sfpi::vFloat &va, sfpi::vFloat &vb,
                              sfpi::vUInt &ca, sfpi::vUInt &cb)
{
    static_assert(Mod <= 8, "SFPSWAP mod1 range");
    auto r = __builtin_rvtt_sfpswap_indexed(vb.get(), va.get(),
                                            cb.get(), ca.get(), Mod);
    vb = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 0));
    va = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 1));
    cb = sfpi::vUInt(__builtin_rvtt_sfpselect4(r, 2));
    ca = sfpi::vUInt(__builtin_rvtt_sfpselect4(r, 3));
}

// Plain (single-bank) swap with an arbitrary row-group mod — for use OUTSIDE
// the ENABLE_DEST_INDEX window.  sfpi::min_max only spells mod 1..8 through
// its mask calculus; this is the direct form.
// Same lane-FD operand-order fix as indexed_swap above (arg0 = VD).
template <unsigned Mod, typename V>
sfpi_inline void swap_mod(V &va, V &vb)
{
    auto r = __builtin_rvtt_sfpswap(vb.get(), va.get(), Mod);
    vb = V(__builtin_rvtt_sfpselect2(r, 0));
    va = V(__builtin_rvtt_sfpselect2(r, 1));
}

// ---------------------------------------------------------------------------
// Dual-bank SFPTRANSP: transposes the 4x4 (LReg index x subvector-row) matrix
// of BOTH banks in one instruction (SFPTRANSP.md).  Values ride the public
// 4-tuple; the companion-bank results are read back through the fixed-LReg
// window right after (the pattern rvtt.md documents for rvtt_sfptransp8_int:
// the L4..L7 SETs stay explicit in the same RTL insn).
// ---------------------------------------------------------------------------
sfpi_inline void transp8(sfpi::vFloat &v0, sfpi::vFloat &v1,
                         sfpi::vFloat &v2, sfpi::vFloat &v3,
                         sfpi::vUInt &c0, sfpi::vUInt &c1,
                         sfpi::vUInt &c2, sfpi::vUInt &c3)
{
    auto r = __builtin_rvtt_sfptransp8(v0.get(), v1.get(), v2.get(), v3.get(),
                                       c0.get(), c1.get(), c2.get(), c3.get());
    v0 = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 0));
    v1 = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 1));
    v2 = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 2));
    v3 = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 3));
    c0 = sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg4]);
    c1 = sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg5]);
    c2 = sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg6]);
    c3 = sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg7]);
}

// ---------------------------------------------------------------------------
// Packed-companion Dst access.  The hand kernels pack (index LO16 | score
// HI16) into one companion register with MERGING partial loads
// (MOD0_FMT_LO16_ONLY=14 then HI16_ONLY=15, SFPLOAD.md).  sfpi has no
// merging-load vocabulary — and the merge is PRESSURE-MANDATORY, not style:
// a zero-fill + OR spelling needs a 9th live vector and the compiler
// correctly refuses it (lreg-pressure-exceeded).  Bridge: the raw load plus
// the live-value load builtin (__builtin_rvtt_sfpload_lv), which is the
// compiler's own model of a partial-register write.  The first (LO16_ONLY)
// load leaves the high half undefined exactly as the original's first load
// into a stale LREG does; the second (HI16_ONLY) load defines it.
// Partial STORES have no typed vocabulary at all -> raw-mod0 builtin store
// (LO16_ONLY=14: Dst16b = Datum & 0xffff; HI16_ONLY=15: Dst16b = Datum >> 16;
// UINT16=6: Dst16b = Datum & 0xffff — SFPSTORE.md).
// Row units: dst_reg[i] == SFPLOAD/SFPSTORE address 2*i (SFP_DESTREG_STRIDE).
// The hand kernels' Imm10 offsets are used directly as `addr` here.
// ---------------------------------------------------------------------------
sfpi_inline sfpi::vUInt load_companion(unsigned idx_addr, unsigned score_addr)
{
    sfpi::vUInt c{__builtin_rvtt_sfpload(idx_addr, sfpi::SFPSTORE_MOD0_FMT_LO16_ONLY,
                                         sfpi::SFPLOAD_ADDR_MODE_NOINC)};
    c = sfpi::vUInt(__builtin_rvtt_sfpload_lv(ckernel::instrn_buffer, c.get(),
                                              score_addr, 0, 0,
                                              sfpi::SFPSTORE_MOD0_FMT_HI16_ONLY,
                                              sfpi::SFPLOAD_ADDR_MODE_NOINC));
    return c;
}

sfpi_inline void store_companion(const sfpi::vUInt &c,
                                 unsigned idx_addr, unsigned score_addr)
{
    __builtin_rvtt_sfpstore(c.get(), idx_addr, sfpi::SFPSTORE_MOD0_FMT_LO16_ONLY,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
    __builtin_rvtt_sfpstore(c.get(), score_addr, sfpi::SFPSTORE_MOD0_FMT_HI16_ONLY,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}

// 16-bit index store (hand: SFPSTORE mod0 InstrModLoadStore::LO16 == 6 ==
// MOD0_FMT_UINT16, "Dst16b = Datum & 0xffff").
sfpi_inline void store_uint16(const sfpi::vUInt &v, unsigned addr)
{
    __builtin_rvtt_sfpstore(v.get(), addr, sfpi::SFPSTORE_MOD0_FMT_UINT16,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}

// Value (sort-key) rows: plain 32-bit Dst access at hand-kernel addresses
// (hand mod0 0 == MOD0_FMT_SRCB default — identical to typed dst_reg access).
sfpi_inline sfpi::vFloat load_value(unsigned addr)
{
    return sfpi::vFloat(__builtin_rvtt_sfpload(addr, sfpi::SFPLOAD_MOD0_FMT_SRCB,
                                               sfpi::SFPLOAD_ADDR_MODE_NOINC));
}
sfpi_inline void store_value(const sfpi::vFloat &v, unsigned addr)
{
    __builtin_rvtt_sfpstore(v.get(), addr, sfpi::SFPSTORE_MOD0_FMT_SRCB,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}

// Subvector rotate-right-by-1 for companions (typed sfpi::subvec_shflror1
// returns the raw vector; these keep the types tidy).
sfpi_inline sfpi::vFloat ror1(const sfpi::vFloat &v)
{
    return sfpi::vFloat(sfpi::subvec_shflror1(v));
}
sfpi_inline sfpi::vUInt ror1(const sfpi::vUInt &v)
{
    return sfpi::vUInt(sfpi::subvec_shflror1(v));
}

// IN-PLACE rotate (original: SFPSHFT2(0, Ln, Ln, 3)).  The live-value builtin
// ties destination == source, which is both the original's register shape and
// what keeps an 8-live-vector kernel inside the LREG file (the plain form
// needs a 9th transient register and the pressure check rightly refuses).
template <typename V>
sfpi_inline void ror1_ip(V &v)
{
    v = V(__builtin_rvtt_sfpshft2_subvec_shfl1_lv(v.get(), v.get(),
                                                  sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1));
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
