# Unary SFPU Operations Using Low-Level Instructions

This document lists unary SFPU kernel implementations that use low-level TTI_*/TT_* instructions instead of high-level SFPI intrinsics.

## Wormhole B0 Hardware Kernels

### 1. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`

**Function:** `calculate_abs_int32()`
**Key Instructions:** `TT_SFPLOAD`, `TTI_SFPABS`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 4, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, 4, 3, 0);
        dst_reg++;
    }
}
```

---

### 2. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bitwise_not.h`

**Function:** `calculate_bitwise_not()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPNOT`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_3, 0);
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

---

### 3. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h`

**Function:** `calculate_logical_not_unary_uint16()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPSETCC`, `TTI_SFPLOADI`, `TTI_SFPENCC`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logical_not_unary_uint16() {
    for (int d = 0; d < ITERATIONS; d++) {
        // full tile size
        constexpr int tile_size = 64;
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);

        // TTI_SFPENCC(IMM12_MATH, LREG_C, LREG_DEST, INSTR_MOD1);
        // IMM12_MATH: optional immediate value for math operations
        // LREG_C: unused
        // LREG_DEST: unused
        // INSTR_MOD1: 0 => condition code enable reg is not modified.
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

---

### 4. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h`

**Function:** `cast_fp32_to_fp16a()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFP_STOCH_RND`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void cast_fp32_to_fp16a() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // vFloat val = dst_reg[0];
        // dst_reg[0] = float_to_fp16a(val, 0);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 8);
        TTI_SFPSTORE(0, 1, 3, 0);
        dst_reg++;
    }
}
```

---

### 5. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h`

**Function:** `calculate_unary_max_min_int32_body()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPMOV`, `TTI_SFPSWAP`, `TTI_SFPSTORE`

```cpp
sfpi_inline void load_value_param_int(uint value) {
    int scalar = value;
    if (scalar < 0) {  // To convert from 2's complement to sign+magnitude
        scalar = -scalar;
        scalar = 0x80000000 | (scalar & 0x7FFFFFFF);
    }
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
}

template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_int32_body() {
    // Load input tensor to lreg0
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);

    // Copy value param to lreg2 to lreg1
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);

    // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

    // Store the result
    if constexpr (IS_MAX_OP) {
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
    } else {
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
    }
}

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min_int32(uint value) {
    load_value_param_int(value);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_max_min_int32_body<IS_MAX_OP>();
        sfpi::dst_reg++;
    }
}
```

---

### 6. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h`

**Function:** `calculate_signbit_int32()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPSHFT`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

---

### 7. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h`

**Function:** `mul_int32()`
**Key Instructions:** `TT_SFPLOAD`, `TTI_SFPSHFT2`, `TTI_SFPAND`, `TTI_SFPCAST`, `TTI_SFPMAD`, `TTI_SFPEXMAN`, `TTI_SFPSHFT`, `TTI_SFPIADD`, `TT_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        // Split the 32-bit input values into 11-bit chunks:
        //
        //   a = (a2 << 22) | (a1 << 11) | a0
        //   b = (b2 << 22) | (b1 << 11) | b0
        //
        // This allows us to cast these values to fp32 without loss of
        // precision, and furthermore, we can compute:
        //
        //   a * b = (top << 22) + (mid << 11) + low
        //
        // Where:
        //
        //   top = a0*b2 + a1*b1 + a2*b0 (maximum 23 bits)
        //   mid = a0*b1 + a1*b0         (maximum 23 bits)
        //   low = a0*b0                 (maximum 22 bits)
        //
        // We cannot use SFPSTOCHRND to convert FP32 to INT32, as the values
        // are larger than 16 bits; instead we use the trick:
        //   fp32_to_u23(x) = mantissa_bits(x + 2**23)
        // This is exact for 23-bit integers.

        // a0
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // a1
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5);
        // a2
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5);

        // a1 = (a1 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);

        // a2 = a2 as fp32
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);

        // a0 = (a0 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // b0
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // b1
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5);
        // b2
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5);

        // b2 = b2 as fp32
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // top = a0*b2 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

        // b1 = (b1 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // top += a1*b1
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // b0 = (b0 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // top += a2*b0
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // mid = a0*b1 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);

        // low = a0*b0 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);

        // mid += a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

        // extract integers from mantissas
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);

        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);  // top <<= 22
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);  // mid <<= 11

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;
    sfpi::vConstIntPrgm1 = -11;
    sfpi::vConstFloatPrgm2 = 8388608.0f;  // 2**23
}
```

---

### 8. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h`

**Function:** `rand()`
**Key Instructions:** `TT_SFPLOADI`, `TTI_SFPMOV`, `TTI_SFPSETSGN`, `TTI_SFPSETEXP`, `TTI_SFPADDI`, `TTI_SFPMAD`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed);
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) {
    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);

        // Unset sign bit and Set exponent to 127 to ensure the float is within the range [1, 2).
        // lreg0.sign = 0
        // lreg0 = {sign: 0, exponent: 127, mantissa: lreg0.mantissa}
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1);

        // -1 to ensure the float is within the range [0, 1).
        // lreg0 = lreg0 - 1
        TTI_SFPADDI(0xbf80 /*-1*/, p_sfpu::LREG0, 0);
        TTI_SFPNOP;

        // Scale the float from [0, 1) to [from, from + scale)
        // lreg0 = lreg0 * scale + from
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPNOP;

        TTI_SFPSTORE(0, 3, 3, 0);
        dst_reg++;
    }
}
```

---

### 9. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rsub_int32.h`

**Function:** `calculate_rsub_scalar_int32()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPIADD`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_rsub_scalar_int32(uint32_t scalar) {
    int int_scalar = scalar;
    // Load scalar value param to lreg2
    _sfpu_load_imm32_(p_sfpu::LREG1, int_scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // Uses 6 as imod. Performs integer addition between LREG specified in lreg_c and the 2's complement (4) of LREG
        // specified in lreg_dest. The condition code register is not modified (2).
        TTI_SFPIADD(
            0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

---

### 10. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h`

**Functions:** `calculate_comp_uint16()`, `calculate_eqz_uint32()`, `calculate_nez_uint32()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPMOV`, `TTI_SFPSETCC`, `TTI_SFPLOADI`, `TTI_SFPENCC`, `TTI_SFPLZ`, `TTI_SFPSHFT`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
    constexpr int check = ((COMP_MODE == SfpuType::equal_zero) ? SFPSETCC_MOD1_LREG_EQ0 : SFPSETCC_MOD1_LREG_NE0);
    for (int d = 0; d < ITERATIONS; d++) {
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, check);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    int scalar = -5;  // used for shift operation
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        TTI_SFPLZ(0, 0, 1, 4);    // result in lreg1 is leading zero count
        TTI_SFPSHFT(0, 2, 1, 0);  // 32 >> 5 = 1 else 0
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 != 0)
        TTI_SFPSETCC(0, 0, 0, SFPSETCC_MOD1_LREG_NE0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

---

### 11. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_left_shift.h`

**Function:** `calculate_left_shift()`
**Key Instructions:** `TTI_SFPLOAD`, `TT_SFPSHFT`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(0,4,3,0);
        TT_SFPSHFT(shift_amt,0,0,1);
        TTI_SFPSTORE(0,4,3,0);
        dst_reg++;
    }
}
```

---

## Third-Party tt_llk Kernels

### 12. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_square.h`

**Function:** `_calculate_square_()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPMUL`, `TTI_SFPNOP`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_square_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load input from destination into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // Multiply LREG0 * LREG0, store result in LREG0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPNOP;
        // Store result back to destination
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

---

### 13. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`

**Functions:** `_calculate_floor_()`, `_calculate_ceil_()`, `_calculate_trunc_()`, `_calculate_frac_()`
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPLOADI`, `TTI_SFPEXEXP`, `TTI_SFPIADD`, `TTI_SFPSHFT2`, `TTI_SFPSETCC`, `TTI_SFPMAD`, `TTI_SFPENCC`, `TTI_SFPSTORE`

```cpp
// computes L1=trunc(L0).
inline void _trunc_body_()
{
    // set L3=23.  TODO: this could be stored in a constant register, but use by rdiv prevents this for now.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23); // 1
    // mask = 0x8000_0000
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000); // 2
    // disable lanes where exp < 0
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP); // 3
    // mask = 0xffff_ffff
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff); // 4
    // exp = 23 - exp
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0); // 5
    // mask <<= exp
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // 6
    // reset lanes
    TTI_SFPENCC(0, 0, 0, 0); // 7
    // apply mask
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // 8
}

// computes L1=floor(L0).
inline void _floor_body_()
{
    _trunc_body_();
    // if v>u, set v=v-1; this only happens for negative values.
    // on Wormhole, we don't have SFPGT, so use u<0 and (v-u)<0 instead.
    // First, ensure u<0.
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    // Then, ensure (v-u)<0 (two's complement).
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

// computes L1=ceil(L0).
inline void _ceil_body_()
{
    _trunc_body_();
    // if v<u, set v=v+1.
    // on Wormhole, we don't have SFPGT, so use u>=0 and (v-u)<0 instead.
    // First, ensure u>=0.
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    // Then, ensure (v-u)<0 (two's complement).
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_floor_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _floor_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_ceil_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _ceil_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _trunc_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_frac_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _trunc_body_();
        // frac(x) = x - trunc(x)
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

---

### 14. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`

**Functions:** Multiple typecast variants (fp32↔int32, fp32↔uint16, etc.)
**Key Instructions:** `TT_SFPLOADMACRO`, `TT_SFPABS`, `TTI_SFPLOAD`, `TTI_SFPLOADI`, `TTI_SFPEXEXP`, `TTI_SFPSHFT2`, `TTI_SFPCAST`, `TTI_SFPSETSGN`, `TTI_SFPMAD`, `TTI_SFP_STOCH_RND`, `TTI_SFPSTORE`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = INT_MIN
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result (two's complement)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

---

### 15. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_welfords.h`

**Functions:** Welford's algorithm helpers
**Key Instructions:** `TTI_SFPLOAD`, `TTI_SFPMAD`, `TTI_SFPNOP`, `TTI_SFPMOV`, `TTI_SFPSTORE`, `TTI_SFPTRANSP`, `TT_SFPLOADI`

```cpp
template <uint32_t I, uint32_t J>
sfpi_inline void _welfords_load_block_()
{
    constexpr uint32_t tile_offset    = 0; // offset for tile 0 in dst
    constexpr uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr uint32_t offset0        = dst_reg_offset;
    constexpr uint32_t offset1        = dst_reg_offset + 2;
    constexpr uint32_t offset2        = dst_reg_offset + 16;
    constexpr uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset2);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset3);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <uint32_t input_lreg>
sfpi_inline void _compute_welfords_row_()
{
    // mean calculation
    // ----------------
    // mean_{N_+1} = mean_{N} + ((1/N+1) * (x_{N+1} - mean_{N}))
    // Let α = x_{N+1} - mean_{N} and β = 1/N+1
    // Then mean_{N+1} = mean_{N} + α * β

    // 1. Calculate α = x_{N+1} - mean_{N}
    // LREG6 = -1 * LREG4 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG6, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG6 (2-cycle operation)

    // 2. Calculate α * β + mean_{N}
    // LREG6 = LREG6 * LREG7 + LREG4
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG6, 0);

    // m2 calculation
    // ---------------
    // m2_{N+1} = m2_{N} + (x_{N+1} - mean_{N}) * (x_{N+1} - mean_{N+1})
    // Let α = x_{N+1} - mean_{N} and β = x_{N+1} - mean_{N+1}
    // Then m2_{N+1} = m2_{N} + α * β

    // 1. Re-calculate α in lREG4 since LREG6 now contains the new mean
    // LREG4 = -1 * LREG4 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG4, 0);

    // 2. Calculate β = x_{N+1} - mean_{N+1}
    // input_lreg = -1 * LREG6 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG6, input_lreg, input_lreg, 0);
    TTI_SFPNOP; // Next cycle cannot read from input_lreg (2-cycle operation)

    // 3. Calculate m2_{N+1} = α * β + m2_{N}
    // LREG5 = LREG4 * input_lreg + LREG5
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);

    // Moves mean to LREG4 from LREG6 since it now is considered the past mean
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG4, 0);
}

sfpi_inline void _clear_previous_mean_and_m2_()
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
}

sfpi_inline void _store_mean_m2_to_dst_()
{
    constexpr uint32_t mean_tile_offset = 0;  // offset for the mean tile in dst
    constexpr uint32_t m2_tile_offset   = 64; // offset for the m2 tile in dst

    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, m2_tile_offset);
}
```

---

## Functions with Both SFPI and Low-Level Implementations

This section highlights functions where the same operation exists in both:
- **SFPI implementation** (high-level, typically for `float`)
- **Low-level implementation** (TTI_/TT_, typically for `int32`, `uint16`, `uint32`)

---

### 1. Absolute Value (`abs`)

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`

#### SFPI Version (float)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}
```

#### Low-Level Version (int32)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 4, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, 4, 3, 0);
        dst_reg++;
    }
}
```

**Why the difference?** The SFPI `sfpi::abs()` intrinsic works natively with `vFloat`. For `int32`, the hardware requires explicit load/store with `INT32` format modifier, and the `TTI_SFPABS` instruction operates on the sign-magnitude representation.

---

### 2. Sign Bit Extraction (`signbit`)

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h`

#### SFPI Version (float)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        v_if(val < 0.0f) { val = 1.0f; }
        v_else { val = 0.0f; }
        v_endif;
        dst_reg[0] = val;

        dst_reg++;
    }
}
```

#### Low-Level Version (int32)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

**Why the difference?** For float, SFPI uses conditional logic (`v_if`/`v_else`). For int32, extracting the sign bit is a simple arithmetic right shift by 31 positions (`TTI_SFPSHFT` with -31), which is more efficient than conditional branching.

---

### 3. Comparison Operations (`comp`)

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h`

#### SFPI Version (float)
```cpp
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp(uint exponent_size_8) {
    const vFloat zero = 0.0f;
    const vFloat one = 1.0f;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(_sfpu_is_fp16_zero_(v, exponent_size_8)) { v = one; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            v_if(_sfpu_is_fp16_zero_(v, exponent_size_8)) { v = zero; }
            v_else { v = one; }
            v_endif;
        }

        // ... other comparisons (less_than_zero, greater_than_zero, etc.)

        dst_reg[0] = v;
        dst_reg++;
    }
}
```

#### SFPI Version (int32) - Also uses SFPI!
```cpp
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // ... other comparisons

        dst_reg[0] = v;
        dst_reg++;
    }
}
```

#### Low-Level Version (uint16)
```cpp
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
    constexpr int check = ((COMP_MODE == SfpuType::equal_zero) ? SFPSETCC_MOD1_LREG_EQ0 : SFPSETCC_MOD1_LREG_NE0);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, 0, 0, check);
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

#### Low-Level Version (uint32 - equal zero)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    int scalar = -5;  // used for shift operation
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        TTI_SFPLZ(0, 0, 1, 4);    // result in lreg1 is leading zero count
        TTI_SFPSHFT(0, 2, 1, 0);  // 32 >> 5 = 1 else 0
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
```

**Why the difference?**
- `float` and `int32` (sign-magnitude) can use SFPI's `vFloat`/`vInt` types with `v_if`
- `uint16` requires `LO16` load/store format not directly supported by SFPI vector types
- `uint32` equal-zero uses a clever trick: count leading zeros (`TTI_SFPLZ`) - if all 32 bits are zero, the count is 32, then shift right by 5 to get 1; otherwise 0

---

### 4. Logical NOT

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h`

#### SFPI Version (templated for vFloat/vInt)
```cpp
template <typename V, typename T>
inline void calculate_logical_not_unary() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        V v = sfpi::dst_reg[0];
        v_if(v == 0) { sfpi::dst_reg[0] = T(1); }
        v_else { sfpi::dst_reg[0] = T(0); }
        v_endif;
        sfpi::dst_reg++;
    }
}
```

#### Low-Level Version (uint16)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logical_not_unary_uint16() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, 0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
```

**Why the difference?** The templated SFPI version works with `vFloat` or `vInt`. For `uint16`, the `LO16` data format requires explicit low-level load/store instructions.

---

### 5. Unary Max/Min

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h`

**Note:** Both float and int32 versions use low-level instructions because `TTI_SFPSWAP` provides an efficient hardware primitive for min/max that SFPI doesn't abstract.

#### Low-Level Version (float)
```cpp
template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_float_body() {
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);  // Hardware min/max swap

    if constexpr (IS_MAX_OP) {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
    } else {
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    }
}
```

#### Low-Level Version (int32)
```cpp
template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_int32_body() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);  // Hardware min/max swap

    if constexpr (IS_MAX_OP) {
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
    } else {
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
    }
}
```

**Why both use low-level?** The `TTI_SFPSWAP` instruction is a hardware-efficient way to compute min/max in a single instruction. SFPI doesn't expose this directly, so both data types benefit from using the low-level instruction.

---

## Summary: When to Use Low-Level vs SFPI

| Scenario | Use SFPI | Use Low-Level |
|----------|----------|---------------|
| Float operations | ✅ `vFloat`, `sfpi::abs()` | Only for special HW features |
| Int32 (sign-magnitude) | ✅ `vInt` with `v_if` | When bit manipulation is simpler |
| Uint16 | ❌ No native support | ✅ `LO16` format required |
| Uint32 | ❌ Limited support | ✅ `INT32` format with unsigned semantics |
| Hardware primitives (SWAP, STOCH_RND) | ❌ Not exposed | ✅ Direct access |
| Bit shifts for integers | Possible but verbose | ✅ `TTI_SFPSHFT` is direct |

---

## Key Differences: Low-Level vs SFPI

### Low-Level (TTI_/TT_) Approach

```cpp
// Direct register references
p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2

// Explicit load/store with address modes
TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index * dst_tile_size);
TTI_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index * dst_tile_size);

// Manual condition code management
TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

// Direct arithmetic instructions
TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);
TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
```

### High-Level (SFPI) Approach

```cpp
// Vector types with implicit register allocation
vInt input = dst_reg[0];
vUInt val = reinterpret<vUInt>(input);
vFloat result;

// Structured conditionals
v_if(input < 0) {
    val = setsgn(val - 1, 0);
}
v_endif;

// Implicit store via assignment
dst_reg[0] = res;
dst_reg++;
```

## When Low-Level Instructions Are Used

Low-level TTI_/TT_ instructions are typically used when:

1. **Integer operations**: Operations on `int32`, `uint16`, `uint32` types that SFPI doesn't natively support well
2. **Bit manipulation**: Direct bit shifts, masks, and logical operations
3. **Type casting**: Complex format conversions requiring precise bit-level control
4. **Performance-critical paths**: Where explicit instruction ordering matters
5. **Special hardware features**: Accessing features like stochastic rounding (`TTI_SFP_STOCH_RND`)
