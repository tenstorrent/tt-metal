// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stub for sfpi.h (clang-tidy analysis only).
//
// The real sfpi.h requires TT RISC-V compiler builtins (__builtin_rvtt_*)
// that host clang does not have.  This stub provides enough type declarations
// and no-op function stubs for ckernel_sfpu_*.h headers to parse successfully
// under host clang without any __builtin_rvtt_* calls.
//
// sfpi_constants.h contains only constexpr integer/float definitions — no
// builtins — so we include it directly.
//
// Notes on the type system:
//   • All vector types derive from __vBase (mirrors real sfpi.h hierarchy).
//   • Constructors that are implicit in the real sfpi.h are kept implicit here
//     so that cross-type assignments in ckernel_sfpu_*.h parse correctly.
//   • Cross-type conversion (vFloat→vInt etc.) is handled by __vBase-accepting
//     constructors, but those are EXPLICIT to prevent uint32_t→vUInt→vInt
//     ambiguities; instead we provide separate implicit vFloat/vUInt constructors
//     on vInt and vice versa.
//   • int32_t/uint32_t overloads are OMITTED: on riscv32 these are the same
//     types as int/unsigned int, causing "redeclaration" errors.

#pragma once

#include <cstdint>
#include <type_traits>
#include "sfpi_constants.h"

namespace sfpi {

#define sfpi_inline inline __attribute__((always_inline))

// ── Hardware register placeholder ─────────────────────────────────────────
struct __rvtt_vec_t {
    unsigned int val = 0;
    constexpr __rvtt_vec_t() = default;
    constexpr explicit __rvtt_vec_t(unsigned v) : val(v) {}
};

// ── Half-precision types (subset of sfpi_fp16.h) ──────────────────────────
struct s2vFloat16 {
    s2vFloat16() = default;
    s2vFloat16(float) {}
    s2vFloat16(int) {}
    s2vFloat16(unsigned int) {}
};

// sFloat16b is referenced in LLK sfpu headers — it is a subclass of
// s2vFloat16 in sfpi_fp16.h; alias it here so the name resolves.
using sFloat16b = s2vFloat16;
using sFloat16a = s2vFloat16;

// Forward declarations
struct vFloat;
struct vInt;
struct vUInt;
struct __vConstFloat;
struct __vConstIntBase;

// ── Condition type ─────────────────────────────────────────────────────────
// In the real sfpi.h __vCond is also implicitly convertible to vInt so that
// "return (v == 0.0f)" in an sfpi::vInt-returning function compiles.
struct vInt;  // forward again (before __vCond definition)
struct __vCond {
    __vCond() = default;
    __vCond operator!() const { return {}; }
    __vCond operator&&(const __vCond&) const { return {}; }
    __vCond operator||(const __vCond&) const { return {}; }
    // Allow returning __vCond from vInt-returning functions
    operator vInt() const;  // defined after vInt
};

// ── Base register type — all vector registers inherit from this ────────────
struct __vBase {
    __rvtt_vec_t v;
    constexpr __vBase() = default;
    constexpr explicit __vBase(const __rvtt_vec_t& r) : v(r) {}
    sfpi_inline __rvtt_vec_t get() const { return v; }

    // Comparison helpers
    __vCond operator==(const __vBase&) const { return {}; }
    __vCond operator!=(const __vBase&) const { return {}; }
    __vCond operator<(const __vBase&) const { return {}; }
    __vCond operator<=(const __vBase&) const { return {}; }
    __vCond operator>(const __vBase&) const { return {}; }
    __vCond operator>=(const __vBase&) const { return {}; }
    __vCond operator==(float) const { return {}; }
    __vCond operator!=(float) const { return {}; }
    __vCond operator<(float) const { return {}; }
    __vCond operator<=(float) const { return {}; }
    __vCond operator>(float) const { return {}; }
    __vCond operator>=(float) const { return {}; }
    __vCond operator==(int) const { return {}; }
    __vCond operator!=(int) const { return {}; }
    __vCond operator<(int) const { return {}; }
    __vCond operator<=(int) const { return {}; }
    __vCond operator>(int) const { return {}; }
    __vCond operator>=(int) const { return {}; }
};

// Integer vector base
struct __vIntBase : __vBase {
    constexpr __vIntBase() = default;
    constexpr explicit __vIntBase(const __rvtt_vec_t& r) : __vBase(r) {}
};

// ── vFloat ─────────────────────────────────────────────────────────────────
struct vFloat : __vBase {
    vFloat() = default;
    vFloat(float) {}              // implicit (matches real sfpi.h)
    vFloat(const s2vFloat16&) {}  // implicit
    explicit vFloat(const __rvtt_vec_t& r) : __vBase(r) {}
    // Cross-type: vInt/vUInt → vFloat via reinterpret
    explicit vFloat(const __vIntBase&) {}

    vFloat& operator=(const vFloat&) = default;
    vFloat& operator=(float) { return *this; }
    vFloat operator+(const vFloat&) const { return {}; }
    vFloat operator-(const vFloat&) const { return {}; }
    vFloat operator*(const vFloat&) const { return {}; }
    vFloat operator-() const { return {}; }
    vFloat& operator+=(const vFloat&) { return *this; }
    vFloat& operator-=(const vFloat&) { return *this; }
    vFloat& operator*=(const vFloat&) { return *this; }
    vFloat operator+(float) const { return {}; }
    vFloat operator-(float) const { return {}; }
    vFloat operator*(float) const { return {}; }
    vFloat& operator+=(float) { return *this; }
    vFloat& operator-=(float) { return *this; }
    vFloat& operator*=(float) { return *this; }
    vFloat operator++(int) { return {}; }
    vFloat& operator++() { return *this; }
    vFloat operator--(int) { return {}; }
    vFloat& operator--() { return *this; }
};

inline vFloat operator+(float, const vFloat&) { return {}; }
inline vFloat operator-(float, const vFloat&) { return {}; }
inline vFloat operator*(float, const vFloat&) { return {}; }

// ── vInt ───────────────────────────────────────────────────────────────────
struct vInt : __vIntBase {
    vInt() = default;
    vInt(int) {}  // implicit
    // vInt(short) is intentionally omitted: it causes ambiguity with vInt(int)
    // for unsigned int arguments.  Use explicit cast: vInt((int)short_val).
    // NOTE: vInt(unsigned int) is intentionally NOT defined here.
    // The ambiguity between vInt(int) and vUInt→vInt for uint32_t arguments
    // is prevented by keeping only the signed int constructor implicit.
    // Explicit construction from unsigned int is possible via vInt(vUInt(x)).
    explicit vInt(unsigned int) {}
    explicit vInt(const __rvtt_vec_t& r) : __vIntBase(r) {}
    // Cross-type implicit constructors (matches real sfpi.h assignment semantics)
    vInt(const vFloat& f) {}  // vFloat → vInt (e.g. vInt x = exexp(v))
    vInt(const vUInt& u);     // vUInt → vInt (defined after vUInt)

    vInt& operator=(const vInt&) = default;
    vInt& operator=(int) { return *this; }
    vInt operator+(const vInt&) const { return {}; }
    vInt operator-(const vInt&) const { return {}; }
    vInt operator&(const vInt&) const { return {}; }
    vInt operator|(const vInt&) const { return {}; }
    vInt operator^(const vInt&) const { return {}; }
    vInt operator~() const { return {}; }
    vInt operator-() const { return {}; }
    vInt operator+(int) const { return {}; }
    vInt operator-(int) const { return {}; }
    vInt operator&(int) const { return {}; }
    vInt operator|(int) const { return {}; }
    vInt operator^(int) const { return {}; }
    vInt& operator+=(const vInt&) { return *this; }
    vInt& operator-=(const vInt&) { return *this; }
    vInt& operator&=(const vInt&) { return *this; }
    vInt& operator|=(const vInt&) { return *this; }
    vInt& operator^=(const vInt&) { return *this; }
    vInt& operator+=(int) { return *this; }
    vInt& operator-=(int) { return *this; }
    vInt& operator&=(int) { return *this; }
    vInt& operator|=(int) { return *this; }
    vInt& operator^=(int) { return *this; }
    vInt operator<<(int) const { return {}; }
    vInt operator>>(int) const { return {}; }
    vInt operator<<(const vInt&) const { return {}; }
    vInt operator>>(const vInt&) const { return {}; }
    vInt& operator<<=(int) { return *this; }
    vInt& operator>>=(int) { return *this; }
    vInt operator++(int) { return {}; }
    vInt& operator++() { return *this; }
    vInt operator--(int) { return {}; }
    vInt& operator--() { return *this; }
};

inline vInt operator+(int, const vInt&) { return {}; }
inline vInt operator+(unsigned int, const vInt&) { return {}; }  // disambiguate 127U + vInt
inline vInt operator-(int, const vInt&) { return {}; }
inline vInt operator-(unsigned int, const vInt&) { return {}; }
inline vInt operator&(int, const vInt&) { return {}; }
inline vInt operator|(int, const vInt&) { return {}; }

// ── vUInt ──────────────────────────────────────────────────────────────────
struct vUInt : __vIntBase {
    vUInt() = default;
    vUInt(unsigned int) {}  // implicit
    vUInt(int) {}           // implicit
    vUInt(short) {}
    vUInt(unsigned short) {}
    explicit vUInt(const __rvtt_vec_t& r) : __vIntBase(r) {}
    // Cross-type implicit constructors
    vUInt(const vFloat& f) {}  // vFloat → vUInt
    vUInt(const vInt& i) {}    // vInt → vUInt

    vUInt& operator=(const vUInt&) = default;
    vUInt& operator=(unsigned int) { return *this; }
    vUInt& operator=(int) { return *this; }
    vUInt operator+(const vUInt&) const { return {}; }
    vUInt operator-(const vUInt&) const { return {}; }
    vUInt operator&(const vUInt&) const { return {}; }
    vUInt operator|(const vUInt&) const { return {}; }
    vUInt operator^(const vUInt&) const { return {}; }
    vUInt operator~() const { return {}; }
    vUInt operator+(int) const { return {}; }
    vUInt operator-(int) const { return {}; }
    vUInt operator+(unsigned int) const { return {}; }
    vUInt operator-(unsigned int) const { return {}; }
    vUInt operator&(unsigned int) const { return {}; }
    vUInt operator|(unsigned int) const { return {}; }
    vUInt operator^(unsigned int) const { return {}; }
    vUInt& operator+=(const vUInt&) { return *this; }
    vUInt& operator-=(const vUInt&) { return *this; }
    vUInt& operator&=(const vUInt&) { return *this; }
    vUInt& operator|=(const vUInt&) { return *this; }
    vUInt& operator^=(const vUInt&) { return *this; }
    vUInt& operator+=(int) { return *this; }
    vUInt& operator-=(int) { return *this; }
    vUInt& operator+=(unsigned int) { return *this; }
    vUInt& operator-=(unsigned int) { return *this; }
    vUInt& operator&=(unsigned int) { return *this; }
    vUInt& operator|=(unsigned int) { return *this; }
    vUInt& operator^=(unsigned int) { return *this; }
    vUInt operator<<(int) const { return {}; }
    vUInt operator>>(int) const { return {}; }
    vUInt operator<<(unsigned int) const { return {}; }
    vUInt operator>>(unsigned int) const { return {}; }
    vUInt operator<<(const vInt&) const { return {}; }
    vUInt operator>>(const vInt&) const { return {}; }
    vUInt& operator<<=(int) { return *this; }
    vUInt& operator>>=(int) { return *this; }
    vUInt& operator<<=(unsigned int) { return *this; }
    vUInt& operator>>=(unsigned int) { return *this; }
    vUInt operator++(int) { return {}; }
    vUInt& operator++() { return *this; }
    vUInt operator--(int) { return {}; }
    vUInt& operator--() { return *this; }
};

// Define deferred vInt constructor that requires vUInt to be complete
inline vInt::vInt(const vUInt& u) {}

// Define deferred __vCond → vInt conversion
inline __vCond::operator vInt() const { return {}; }

inline vUInt operator+(int, const vUInt&) { return {}; }
inline vUInt operator+(unsigned int, const vUInt&) { return {}; }
inline vUInt operator-(int, const vUInt&) { return {}; }

// ── Const register types ───────────────────────────────────────────────────
// These are constexpr objects representing hardware constant registers.
// They must be usable in constexpr contexts → base must be constexpr-constructible.
// We store the register index as an int directly (not via __vBase) to keep constexpr.

struct __vConstFloat {
    int reg;
    constexpr explicit __vConstFloat(int r) : reg(r) {}
    sfpi_inline int get() const { return reg; }

    operator vFloat() const { return {}; }

    // Arithmetic with vFloat — result is always vFloat
    vFloat operator+(const vFloat&) const { return {}; }
    vFloat operator-(const vFloat&) const { return {}; }
    vFloat operator*(const vFloat&) const { return {}; }
    vFloat operator+(float) const { return {}; }
    vFloat operator-(float) const { return {}; }
    vFloat operator*(float) const { return {}; }
    // Assignment (e.g. vConstFloatPrgm1 = sFloat16b(x))
    void operator=(const s2vFloat16&) const {}
    void operator=(float) const {}

    __vCond operator==(const vFloat&) const { return {}; }
    __vCond operator!=(const vFloat&) const { return {}; }
    __vCond operator<(const vFloat&) const { return {}; }
    __vCond operator<=(const vFloat&) const { return {}; }
    __vCond operator>(const vFloat&) const { return {}; }
    __vCond operator>=(const vFloat&) const { return {}; }
};

inline vFloat operator+(float, const __vConstFloat&) { return {}; }
inline vFloat operator*(float, const __vConstFloat&) { return {}; }
inline vFloat operator+(const vFloat&, const __vConstFloat&) { return {}; }
inline vFloat operator*(const vFloat&, const __vConstFloat&) { return {}; }

struct __vConstIntBase {
    int reg;
    constexpr explicit __vConstIntBase(int r) : reg(r) {}
    sfpi_inline int get() const { return reg; }

    operator vInt() const { return {}; }
    operator vUInt() const { return {}; }

    // Arithmetic — template enables matching __vIntBase-derived types
    vInt operator+(int) const { return {}; }
    vInt operator-(int) const { return {}; }
    template <typename T, typename std::enable_if_t<std::is_base_of<__vIntBase, T>::value>* = nullptr>
    T operator+(const T&) const {
        return {};
    }
    template <typename T, typename std::enable_if_t<std::is_base_of<__vIntBase, T>::value>* = nullptr>
    T operator-(const T&) const {
        return {};
    }
    template <typename T, typename std::enable_if_t<std::is_base_of<__vIntBase, T>::value>* = nullptr>
    T operator&(const T&) const {
        return {};
    }
    void operator=(int) const {}
};

// ── Dest register ──────────────────────────────────────────────────────────
struct __vDReg {
    int idx = 0;
    explicit __vDReg(int i) : idx(i) {}
    operator vFloat() const { return {}; }
    operator vInt() const { return {}; }  // needed: vInt x = dst_reg[0]
    operator vUInt() const { return {}; }
    __vDReg& operator=(const vFloat&) { return *this; }
    __vDReg& operator=(const vInt&) { return *this; }
    __vDReg& operator=(const vUInt&) { return *this; }
    __vDReg& operator=(float) { return *this; }
    __vDReg& operator=(int) { return *this; }
    __vDReg& operator=(unsigned int) { return *this; }
    __vDReg& operator+=(const vFloat&) { return *this; }
    __vDReg& operator-=(const vFloat&) { return *this; }
    __vDReg& operator*=(const vFloat&) { return *this; }
    vFloat operator+(const vFloat&) const { return {}; }
    vFloat operator*(const vFloat&) const { return {}; }
    vFloat operator-() const { return {}; }  // -dst_reg[i]
    vInt operator+(const vInt&) const { return {}; }
    __vCond operator==(int) const { return {}; }
    __vCond operator!=(int) const { return {}; }
    __vCond operator==(const vFloat&) const { return {}; }
    __vCond operator!=(const vFloat&) const { return {}; }
    __vCond operator<(const vFloat&) const { return {}; }
    __vCond operator<=(const vFloat&) const { return {}; }
    __vCond operator>(const vFloat&) const { return {}; }
    __vCond operator>=(const vFloat&) const { return {}; }
};

struct __DestReg {
    __vDReg operator[](int i) const { return __vDReg(i); }
    __vDReg operator[](unsigned int i) const { return __vDReg(static_cast<int>(i)); }
    sfpi_inline void operator++() const {}
    sfpi_inline void operator++(int) const {}
    sfpi_inline __DestReg& operator+=(int) { return *this; }
};

// ── Local register file ────────────────────────────────────────────────────
// LRegs enum — the real sfpi.h has 8 lregs on newer hardware
enum LRegs {
    LReg0 = 0,
    LReg1 = 1,
    LReg2 = 2,
    LReg3 = 3,
    LReg4 = 4,
    LReg5 = 5,
    LReg6 = 6,
    LReg7 = 7,
    LRegCount = SFP_LREG_COUNT,
};

struct __vLReg {
    int idx = 0;
    explicit __vLReg(int i) : idx(i) {}
    operator vFloat() const { return {}; }
    operator vInt() const { return {}; }
    operator vUInt() const { return {}; }
    __vLReg& operator=(const vFloat&) { return *this; }
    __vLReg& operator=(const vInt&) { return *this; }
    __vLReg& operator=(const vUInt&) { return *this; }
};

struct __LReg {
    __vLReg operator[](int i) const { return __vLReg(i); }
    __vLReg operator[](LRegs r) const { return __vLReg(static_cast<int>(r)); }
};

// ── Global register and constant objects ───────────────────────────────────
inline __DestReg dst_reg;
inline __LReg l_reg;

// Constant register file (mirrors sfpi.h)
constexpr __vConstFloat vConst0(CREG_IDX_0);
constexpr __vConstFloat vConst1(CREG_IDX_1);
constexpr __vConstFloat vConstNeg1(CREG_IDX_NEG_1);
constexpr __vConstFloat vConst0p8373(CREG_IDX_0P837300003);
constexpr __vConstFloat vConstFloatPrgm0(CREG_IDX_PRGM1);
constexpr __vConstFloat vConstFloatPrgm1(CREG_IDX_PRGM2);
constexpr __vConstFloat vConstFloatPrgm2(CREG_IDX_PRGM3);
constexpr __vConstIntBase vConstTileId(CREG_IDX_TILEID);
constexpr __vConstIntBase vConstIntPrgm0(CREG_IDX_PRGM1);
constexpr __vConstIntBase vConstIntPrgm1(CREG_IDX_PRGM2);
constexpr __vConstIntBase vConstIntPrgm2(CREG_IDX_PRGM3);

// ── Template reinterpret (sfpi_lib.h) ─────────────────────────────────────
template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType reinterpret(const __vBase& v) {
    return vType(v.v);
}

// Overload for __vConstFloat (not derived from __vBase in our stub)
template <typename vType, typename std::enable_if_t<std::is_base_of<__vIntBase, vType>::value>* = nullptr>
sfpi_inline vType reinterpret(const __vConstFloat& v) {
    return {};
}

// ── Template free functions ────────────────────────────────────────────────

// setsgn: set sign bit; return same type as first argument
template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType setsgn(const vType, int) {
    return {};
}

template <
    typename vTypeA,
    typename vTypeB,
    typename std::enable_if_t<std::is_base_of<__vBase, vTypeA>::value>* = nullptr,
    typename std::enable_if_t<std::is_base_of<__vBase, vTypeB>::value>* = nullptr>
sfpi_inline vTypeA setsgn(const vTypeA, const vTypeB) {
    return {};
}

// setexp / setman / addexp
template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType setexp(const vType, unsigned int) {
    return {};
}

template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType setexp(const vType, const vInt&) {
    return {};
}

template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType setman(const vType, const __vIntBase&) {
    return {};
}

template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vType setman(const vType, unsigned int) {
    return {};
}

// setman also accepts __vConstFloat (constant regs)
sfpi_inline vFloat setman(const __vConstFloat&, const vInt&) { return {}; }
sfpi_inline vFloat setman(const __vConstFloat&, unsigned int) { return {}; }

template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vFloat addexp(const vType, int) {
    return {};
}

// lz: leading zeros
template <typename vType, typename std::enable_if_t<std::is_base_of<__vBase, vType>::value>* = nullptr>
sfpi_inline vInt lz(const vType) {
    return {};
}

// ── Non-template free functions ────────────────────────────────────────────
// exexp/exman return vInt (they extract integer fields from float registers)
sfpi_inline vInt exexp(const vFloat&) { return {}; }
sfpi_inline vInt exexp_nodebias(const vFloat&) { return {}; }
sfpi_inline vInt exman8(const vFloat&) { return {}; }
sfpi_inline vInt exman9(const vFloat&) { return {}; }

sfpi_inline vFloat abs(const vFloat&) { return {}; }
sfpi_inline vInt abs(const vInt&) { return {}; }
sfpi_inline vUInt abs(const vUInt&) { return {}; }
sfpi_inline vFloat float_to_fp16b(const vFloat&, int = 0) { return {}; }
sfpi_inline vFloat float_to_fp16a(const vFloat&, int = 0) { return {}; }
sfpi_inline vInt float_to_int16(const vFloat&, int = 0) { return {}; }
sfpi_inline vFloat int32_to_float(const vInt&, int = 0) { return {}; }
sfpi_inline vFloat int32_to_float(const vUInt&, int = 0) { return {}; }

// shft — exactly matches sfpi_lib.h signatures
sfpi_inline vUInt shft(const vUInt, const vInt) { return {}; }
sfpi_inline vUInt shft(const vUInt, int) { return {}; }
sfpi_inline vInt shft(const vInt, const vInt) { return {}; }
sfpi_inline vInt shft(const vInt, int) { return {}; }

// lut variants — matches sfpi_lib.h overload set
sfpi_inline vFloat lut(const vFloat&, const vUInt&, const vUInt&, const vUInt&) { return {}; }
sfpi_inline vFloat lut_sign(const vFloat&, const vUInt&, const vUInt&, const vUInt&) { return {}; }
// 4-arg with int mode
sfpi_inline vFloat lut(const vFloat&, const vFloat&, const vFloat&, int) { return {}; }
sfpi_inline vFloat lut_sign(const vFloat&, const vFloat&, const vFloat&, int) { return {}; }
// lut2: 4-arg vFloat version and 8-arg vUInt version (7 vUInt + int mode)
sfpi_inline vFloat lut2(const vFloat&, const vFloat&, const vFloat&, const vFloat&) { return {}; }
sfpi_inline vFloat lut2_sign(const vFloat&, const vFloat&, const vFloat&, const vFloat&) { return {}; }
// 7-arg vFloat version
sfpi_inline vFloat
lut2(const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&) {
    return {};
}
sfpi_inline vFloat
lut2_sign(const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&, const vFloat&) {
    return {};
}
// 7-arg vUInt version + optional int mode (for fp16 LUT tables)
sfpi_inline vFloat
lut2(const vFloat&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, int = 1) {
    return {};
}
sfpi_inline vFloat
lut2_sign(const vFloat&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, const vUInt&, int = 1) {
    return {};
}
sfpi_inline void load_config() {}
sfpi_inline vFloat mad(const vFloat&, const vFloat&, const vFloat&) { return {}; }
sfpi_inline vFloat mul(const vFloat&, const vFloat&) { return {}; }
sfpi_inline vFloat add(const vFloat&, const vFloat&) { return {}; }
sfpi_inline vFloat sub(const vFloat&, const vFloat&) { return {}; }
sfpi_inline void set_expected_ftz(int) {}

// vec_min_max (sfpi_lib.h lines 324-338)
sfpi_inline void vec_min_max(vFloat&, vFloat&) {}
sfpi_inline void vec_min_max(__vIntBase&, __vIntBase&) {}

// ── Conditional execution macros ───────────────────────────────────────────
// Simplistic passthrough — just makes the code parse; clang-tidy analyses
// the call sites and control-flow structure.

#ifndef v_if
#define v_if(cond) if (true)
#define v_elseif(cond) else if (true)
#define v_else else
#define v_endif
#endif

#ifndef v_and
// In real sfpi.h, v_and(x) is used inside v_block to add conditions.
// Map to a no-op expression that parses and doesn't introduce dead code.
#define v_and(x) (void)(x)
#endif

}  // namespace sfpi
