// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// SFPI spelling of the vector instructions the generated BF16 kernels schedule by hand.
// Each call names the instruction and its operands as the raw form does, on the named
// LRegs (readlreg/writelreg). The compiler still owns ordering, copies and hazard NOPs,
// so the export accepts a kernel only when its issued stream matches the raw one.
// Replays use SFPI's lltt; predication is a pushc/popc region (Region).
#include <cstdint>
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16_sfpi {
using vec = __xtt_vector;

// Values live in the named LRegs; reads and writes are the registers themselves.
struct Pinned {
    [[gnu::always_inline]] vec rd(std::uint32_t r) const { return __builtin_rvtt_sfpreadlreg(r); }
    [[gnu::always_inline]] void wr(std::uint32_t r, vec x) const { __builtin_rvtt_sfpwritelreg(x, r); }
    // Only LReg 0..7 take a result; the hardware discards writes to the constants above them.
    [[gnu::always_inline]] void wr_if_writable(std::uint32_t r, vec x) const {
        if (r < 8) {
            wr(r, x);
        }
    }
};

// A predicated stretch. SFPI lowers a register write as a whole-register copy, so inside
// the region LReg 0..7 are values the compiler predicates; they return to their
// registers once the region closes.
struct Region {
    vec v[8];
    [[gnu::always_inline]] Region() {
        for (std::uint32_t r = 0; r < 8; ++r) {
            v[r] = __builtin_rvtt_sfpreadlreg(r);
        }
        __builtin_rvtt_sfppushc(sfpi::SFPPUSHC_MOD1_PUSH);
    }
    [[gnu::always_inline]] void close() {
        __builtin_rvtt_sfppopc(sfpi::SFPPOPC_MOD1_POP);
        for (std::uint32_t r = 0; r < 8; ++r) {
            __builtin_rvtt_sfpwritelreg(v[r], r);
        }
    }
    [[gnu::always_inline]] vec rd(std::uint32_t r) const { return r < 8 ? v[r] : __builtin_rvtt_sfpreadlreg(r); }
    // An assignment merges into the prior value: disabled lanes keep it (sfpi.h, "Liveness").
    [[gnu::always_inline]] void wr(std::uint32_t r, vec x) { v[r] = __builtin_rvtt_sfpassign_lv(v[r], x); }
    [[gnu::always_inline]] void wr_if_writable(std::uint32_t r, vec x) {
        if (r < 8) {
            wr(r, x);
        }
    }
};

[[gnu::always_inline]] inline void replay(
    std::uint32_t start, std::uint32_t length, std::uint32_t execute, std::uint32_t record) {
    __builtin_rvtt_ttreplay(start, length, execute, record);
}
[[gnu::always_inline]] inline void incrwc(std::uint32_t cr, std::uint32_t d, std::uint32_t b, std::uint32_t a) {
    __builtin_rvtt_ttincrwc(cr, d, b, a);
}
[[gnu::always_inline]] inline void sfpnop() { __builtin_rvtt_sfpnop(); }

template <class R>
[[gnu::always_inline]] inline void sfpload(
    std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr, R& f) {
    f.wr(l, __builtin_rvtt_sfpload(addr, mod0, mode));
}
[[gnu::always_inline]] inline void sfpload(
    std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr) {
    Pinned f;
    sfpload(l, mod0, mode, addr, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpstore(
    std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr, R& f) {
    __builtin_rvtt_sfpstore(f.rd(l), addr, mod0, mode);
}
[[gnu::always_inline]] inline void sfpstore(
    std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr) {
    Pinned f;
    sfpstore(l, mod0, mode, addr, f);
}

// The half-word modes keep the other half, so every load merges into the register.
template <class R>
[[gnu::always_inline]] inline void sfploadi(std::uint32_t l, std::uint32_t mod0, std::uint32_t imm, R& f) {
    f.wr(l, __builtin_rvtt_sfploadi_lv(ckernel::instrn_buffer, f.rd(l), imm, 0, 0, mod0));
}
[[gnu::always_inline]] inline void sfploadi(std::uint32_t l, std::uint32_t mod0, std::uint32_t imm) {
    Pinned f;
    sfploadi(l, mod0, imm, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpmad(
    std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpmad(f.rd(a), f.rd(b), f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpmad(
    std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpmad(a, b, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpmul(
    std::uint32_t a, std::uint32_t b, std::uint32_t, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpmul(f.rd(a), f.rd(b), m));
}
[[gnu::always_inline]] inline void sfpmul(
    std::uint32_t a, std::uint32_t b, std::uint32_t a2, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpmul(a, b, a2, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpadd(
    std::uint32_t, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpadd(f.rd(b), f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpadd(
    std::uint32_t a0, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpadd(a0, b, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpaddi(std::uint32_t imm, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpaddi(ckernel::instrn_buffer, f.rd(d), imm, 0, 0, m));
}
[[gnu::always_inline]] inline void sfpaddi(std::uint32_t imm, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpaddi(imm, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpmuli(std::uint32_t imm, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpmuli(ckernel::instrn_buffer, f.rd(d), imm, 0, 0, m));
}
[[gnu::always_inline]] inline void sfpmuli(std::uint32_t imm, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpmuli(imm, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfplutfp32(std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfplutfp32_6r(f.rd(0), f.rd(1), f.rd(2), f.rd(4), f.rd(5), f.rd(6), f.rd(3), m));
}
[[gnu::always_inline]] inline void sfplutfp32(std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfplutfp32(d, m, f);
}

// Modifier 9 (VD takes the max, VC the min) has no SFPI value; it is modifier 1 with the operands exchanged.
template <class R>
[[gnu::always_inline]] inline void sfpswap(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m == 9) {
        auto pair = __builtin_rvtt_sfpswap(f.rd(c), f.rd(d), 1);
        f.wr_if_writable(c, __builtin_rvtt_sfpselect2(pair, 0));
        f.wr_if_writable(d, __builtin_rvtt_sfpselect2(pair, 1));
    } else {
        auto pair = __builtin_rvtt_sfpswap(f.rd(d), f.rd(c), m);
        f.wr_if_writable(d, __builtin_rvtt_sfpselect2(pair, 0));
        f.wr_if_writable(c, __builtin_rvtt_sfpselect2(pair, 1));
    }
}
[[gnu::always_inline]] inline void sfpswap(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpswap(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpsetcc(std::uint32_t, std::uint32_t c, std::uint32_t, std::uint32_t m, R& f) {
    __builtin_rvtt_sfpsetcc(f.rd(c), m);
}
[[gnu::always_inline]] inline void sfpsetcc(std::uint32_t a0, std::uint32_t c, std::uint32_t a2, std::uint32_t m) {
    Pinned f;
    sfpsetcc(a0, c, a2, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpencc(std::uint32_t imm, std::uint32_t, std::uint32_t, std::uint32_t m, R&) {
    __builtin_rvtt_sfpencc(m, imm);
}
[[gnu::always_inline]] inline void sfpencc(std::uint32_t imm, std::uint32_t a1, std::uint32_t a2, std::uint32_t m) {
    Pinned f;
    sfpencc(imm, a1, a2, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpcompc(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, R&) {
    __builtin_rvtt_sfpcompc();
}
[[gnu::always_inline]] inline void sfpcompc(std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3) {
    Pinned f;
    sfpcompc(a0, a1, a2, a3, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpexman(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpexman(f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpexman(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpexman(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpexexp(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpexexp(f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpexexp(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpexexp(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpabs(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpabs(f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpabs(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpabs(a0, c, d, m, f);
}

// SFPI has no modifier for the plain copy; it is an assignment.
template <class R>
[[gnu::always_inline]] inline void sfpmov(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m == 0) {
        f.wr(d, f.rd(c));
    } else {
        f.wr(d, __builtin_rvtt_sfpmov(f.rd(c), m));
    }
}
[[gnu::always_inline]] inline void sfpmov(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpmov(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpcast(std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpcast(f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpcast(std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpcast(c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpand(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t, R& f) {
    f.wr(d, __builtin_rvtt_sfpand(f.rd(d), f.rd(c)));
}
[[gnu::always_inline]] inline void sfpand(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t a3) {
    Pinned f;
    sfpand(a0, c, d, a3, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpor(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t, R& f) {
    f.wr(d, __builtin_rvtt_sfpor(f.rd(d), f.rd(c)));
}
[[gnu::always_inline]] inline void sfpor(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t a3) {
    Pinned f;
    sfpor(a0, c, d, a3, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpxor(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t, R& f) {
    f.wr(d, __builtin_rvtt_sfpxor(f.rd(d), f.rd(c)));
}
[[gnu::always_inline]] inline void sfpxor(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t a3) {
    Pinned f;
    sfpxor(a0, c, d, a3, f);
}

// Bit 0 of the raw modifier selects the immediate operand, which SFPI encodes as a separate builtin.
template <class R>
[[gnu::always_inline]] inline void sfpsetexp(
    std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m & 1) {
        f.wr(d, __builtin_rvtt_sfpsetexp_i(f.rd(c), imm, m & ~1u));
    } else {
        f.wr(d, __builtin_rvtt_sfpsetexp_v(f.rd(c), f.rd(d), m));
    }
}
[[gnu::always_inline]] inline void sfpsetexp(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpsetexp(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpsetman(
    std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m & 1) {
        f.wr(d, __builtin_rvtt_sfpsetman_i(f.rd(c), imm, m & ~1u));
    } else {
        f.wr(d, __builtin_rvtt_sfpsetman_v(f.rd(c), f.rd(d), m));
    }
}
[[gnu::always_inline]] inline void sfpsetman(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpsetman(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpsetsgn(
    std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m & 1) {
        f.wr(d, __builtin_rvtt_sfpsetsgn_i(f.rd(c), imm, m & ~1u));
    } else {
        f.wr(d, __builtin_rvtt_sfpsetsgn_v(f.rd(c), f.rd(d), m));
    }
}
[[gnu::always_inline]] inline void sfpsetsgn(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpsetsgn(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpshft(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m & 1) {
        // With the immediate, bit 2 shifts VC rather than VD; SFPI takes the source as the operand.
        f.wr(d, __builtin_rvtt_sfpshft_i(f.rd(m & 4 ? c : d), imm, m & ~5u));
    } else {
        // Bit 2 is reserved without the immediate; silicon ignores it.
        f.wr(d, __builtin_rvtt_sfpshft_v(f.rd(d), f.rd(c), m & ~4u));
    }
}
[[gnu::always_inline]] inline void sfpshft(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpshft(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpiadd(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    if (m & 1) {
        f.wr(d, __builtin_rvtt_sfpiadd_i(f.rd(c), imm, m & ~1u));
    } else {
        f.wr(d, __builtin_rvtt_sfpiadd_v(f.rd(d), f.rd(c), m));
    }
}
[[gnu::always_inline]] inline void sfpiadd(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpiadd(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpdivp2(
    std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpdivp2(f.rd(c), imm, m));
}
[[gnu::always_inline]] inline void sfpdivp2(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpdivp2(imm, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfp_stoch_rnd(
    std::uint32_t rnd, std::uint32_t imm8, std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpstochrnd_i(f.rd(c), imm8, m, rnd));
}
[[gnu::always_inline]] inline void sfp_stoch_rnd(
    std::uint32_t rnd, std::uint32_t imm8, std::uint32_t a2, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfp_stoch_rnd(rnd, imm8, a2, c, d, m, f);
}

// A lane mask naming all eight lanes (every even bit of the immediate) masks nothing.
template <class R>
[[gnu::always_inline]] inline void sfpconfig(std::uint32_t imm, std::uint32_t dest, std::uint32_t m, R& f) {
    if ((m & 8) && (imm & 0x5555u) == 0x5555u && !(m & 1)) {
        m &= ~8u;
    }
    if (m & 1) {
        (__builtin_rvtt_sfpwriteconfig_i)(ckernel::instrn_buffer, imm, 0, 0, m & ~1u, dest);
    } else {
        __builtin_rvtt_sfpwriteconfig_v(f.rd(0), m, dest);
    }
}
[[gnu::always_inline]] inline void sfpconfig(std::uint32_t imm, std::uint32_t dest, std::uint32_t m) {
    Pinned f;
    sfpconfig(imm, dest, m, f);
}

#if defined(ARCH_BLACKHOLE)
template <class R>
[[gnu::always_inline]] inline void sfparecip(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfparecip(f.rd(c), m));
}
[[gnu::always_inline]] inline void sfparecip(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfparecip(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfple(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfple(f.rd(d), f.rd(c), m));
}
[[gnu::always_inline]] inline void sfple(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfple(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpgt(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f) {
    f.wr(d, __builtin_rvtt_sfpgt(f.rd(d), f.rd(c), m));
}
[[gnu::always_inline]] inline void sfpgt(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m) {
    Pinned f;
    sfpgt(a0, c, d, m, f);
}

#endif
}  // namespace ckernel::sfpu::bf16_sfpi
