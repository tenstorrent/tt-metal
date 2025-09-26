// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef CKERNEL_DEST_H
#define CKERNEL_DEST_H 1

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_vector.h"
#include "tensix_types.h"
#include "tt_t6_trisc_map.h"

// #define RISCV_DEST_START_ADDR 0xFFBD8000
#define RISCV_DEST_START_ADDR DEST_REGS_BASE // TEN-1404

#define RISC_DEST_FMT_FP32  0b000
#define RISC_DEST_FMT_INT32 0b001
#define RISC_DEST_FMT_FP16A 0b010
#define RISC_DEST_FMT_FP16B 0b011
#define RISC_DEST_FMT_INT16 0b100
#define RISC_DEST_FMT_INT8  0b101

#define DEST_32B_WORDS (16 * 512)
#define DEST_16B_WORDS (16 * 1024)
// 8-bit words still use up 16 bits of space in dest, so the capacity is
// the same as for 16-bit types
#define DEST_8B_WORDS (16 * 1024)

namespace ckernel
{

template <int t>
inline void set_dest_fmt(uint fmt)
{
    // FWLOG1("Setting RISC-dest access format to %d", fmt);
    static_assert(t >= 0 and t < 3, "Thread must be 0, 1, or 2");

    if (t == 0)
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_fmt_RMW, fmt);
    }
    else if (t == 1)
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_fmt_RMW, fmt);
    }
    else
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW, fmt);
    }

    tensix_sync();
}

inline void set_dest_fmt(uint fmt, int t)
{
    // FWLOG1("Setting RISC-dest access format to %d", fmt);
    if (t == 0)
    {
        set_dest_fmt<0>(fmt);
    }
    else if (t == 1)
    {
        set_dest_fmt<1>(fmt);
    }
    else
    {
        set_dest_fmt<2>(fmt);
    }

    tensix_sync();
}

template <int t, bool is_signed>
inline void set_dest_int8_int16_signed()
{
    static_assert(t >= 0 and t < 3, "Thread must be 0, 1, or 2");
    if (is_signed)
    {
        // FWLOG0("Setting RISC-dest int8 access mode to signed");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int8 access mode to unsigned");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 1);
        }
    }
}

template <int t>
inline void set_dest_int8_int16_signed(bool const is_signed)
{
    static_assert(t >= 0 and t < 3, "Thread must be 0, 1, or 2");
    if (is_signed)
    {
        // FWLOG0("Setting RISC-dest int8 access mode to signed");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int8 access mode to unsigned");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 1);
        }
    }
}

inline void set_dest_int8_int16_signed(int const t, bool const is_signed)
{
    if (is_signed)
    {
        // FWLOG0("Setting RISC-dest int8 access mode to signed");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int8 access mode to unsigned");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 1);
        }
    }
}

template <int t, bool enable>
inline void set_dest_enable_swizzling()
{
    static_assert(t >= 0 and t < 3, "Thread must be 0, 1, or 2");
    if (enable)
    {
        // In unswizzled mode, values are written into dest as-is with
        // no saturation checks and no bit shuffling. This means they
        // are incompatible with the FPU, but it could be useful for
        // debugging
        // FWLOG0("Setting RISC-dest int access mode to unswizzled mode");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int access mode to swizzled mode (normal)");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 1);
        }
    }
}

template <int t>
inline void set_dest_enable_swizzling(bool const enable)
{
    static_assert(t >= 0 and t < 3, "Thread must be 0, 1, or 2");
    if (enable)
    {
        // In unswizzled mode, values are written into dest as-is with
        // no saturation checks and no bit shuffling. This means they
        // are incompatible with the FPU, but it could be useful for
        // debugging
        // FWLOG0("Setting RISC-dest int access mode to unswizzled mode");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int access mode to swizzled mode (normal)");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 1);
        }
    }
}

inline void set_dest_enable_swizzling(int const t, bool const enable)
{
    if (enable)
    {
        // In unswizzled mode, values are written into dest as-is with
        // no saturation checks and no bit shuffling. This means they
        // are incompatible with the FPU, but it could be useful for
        // debugging
        // FWLOG0("Setting RISC-dest int access mode to unswizzled mode");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 0);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 0);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int access mode to swizzled mode (normal)");
        if (t == 0)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 1);
        }
        else if (t == 1)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 1);
        }
        else
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 1);
        }
    }
}

struct fp16a
{
    uint16_t impl;

    inline fp16a()
    {
    }

    inline fp16a(float f) : impl(float_to_u16(f))
    {
    }

    // Stupid C++ volatile rules are different for structs vs builtin types
    template <typename T>
    inline fp16a(T&& other) : impl(other.impl)
    {
    }

    // I always think "Oh I'll just take a minute to make a nice C++ struct"
    // and then I end up wasting an hour scouring cppreference for some
    // hint about why the compiler is throwing these ridiculous errors...
    // In this case it turns out that once you need the "this" pointer to
    // be volatile all kinds of things become difficult
    inline void operator=(fp16a other)
    {
        impl = other.impl;
    }

    inline void operator=(fp16a other) volatile
    {
        impl = other.impl;
    }

    inline bool operator==(fp16a const& other)
    {
        return impl == other.impl;
    }

    inline bool operator==(fp16a const& other) volatile
    {
        return impl == other.impl;
    }

    inline operator float() const
    {
        uint32_t sign = (impl & 0x8000) << 16;

        uint32_t exp_rebiased = ((impl & 0x7C00) >> 10) - 15 + 127;

        uint32_t exp = (exp_rebiased << 23) & 0x7F800000;

        uint32_t man = (impl & 0x3FF) << 13;

        uint32_t ret = (impl == 0) ? 0 : (sign | exp | man); // Annoying corner case for 0 (TODO: check for -0, inf, and -inf)

        float* sorry_gcc = reinterpret_cast<float*>(&ret);
        return *sorry_gcc;
    }

    inline operator float() volatile
    {
        return float(*const_cast<fp16a const*>(this));
    }

    inline operator float()
    {
        return float(*const_cast<fp16a const*>(this));
    }

    explicit inline operator uint32_t() const
    {
        return static_cast<uint32_t>(impl);
    }

    explicit inline operator uint32_t() volatile
    {
        return uint32_t(*const_cast<fp16a const*>(this));
    }

    explicit inline operator uint32_t()
    {
        return uint32_t(*const_cast<fp16a const*>(this));
    }

    inline fp16a operator+(fp16a volatile& other) volatile
    {
        return float(*const_cast<fp16a const*>(this)) + float(other);
    }

    // 1 sign bit, 5 exponent bits (bias = 15), 10 mantissa bits
    static uint16_t float_to_u16(float val)
    {
        // Evil floating point bit level hacking (though less pithy due to the C++ compiler rules...)
        uint32_t* sorry_gcc = reinterpret_cast<uint32_t*>(&val);
        uint32_t val_bits   = *sorry_gcc;

        uint16_t sign = static_cast<uint16_t>((val_bits >> 16) & 0x8000);

        uint32_t exp_rebiased = ((val_bits & 0x7F800000) >> 23) - 127 + 15; // Wraparound is okay here

        // Check for saturation
        if (exp_rebiased & (1 << 31))
        {
            exp_rebiased = 0;
        }
        else if (exp_rebiased > 31)
        {
            exp_rebiased = 31;
        }

        uint16_t exp = (static_cast<uint16_t>(exp_rebiased) << 10) & 0x7C00;

        uint16_t man = static_cast<uint16_t>((val_bits >> 13) & 0x3FF);

        return sign | exp | man;
    }
};

struct fp16b
{
    uint16_t impl;

    fp16b()
    {
    }

    inline fp16b(float f) : impl(float_to_u16(f))
    {
        // FWEVENT("Construct from float");
    }

    // Stupid C++ volatile rules are different for structs vs builtin types
    template <typename T>
    inline fp16b(T&& other) : impl(other.impl)
    {
    }

    inline void operator=(fp16b other)
    {
        // FWEVENT("operator= nonvolatile");
        impl = other.impl;
    }

    inline void operator=(fp16b other) volatile
    {
        // FWEVENT("operator= volatile");
        impl = other.impl;
    }

    inline bool operator==(fp16b const& other)
    {
        return impl == other.impl;
    }

    inline bool operator==(fp16b const& other) volatile
    {
        return impl == other.impl;
    }

    inline operator float() const
    {
        // FWEVENT("Converting to float (const)");
        uint32_t ret = static_cast<uint32_t>(impl) << 16;

        float* sorry_gcc = reinterpret_cast<float*>(&ret);
        return *sorry_gcc;
    }

    inline operator float() volatile
    {
        // FWEVENT("Converting to float (volatile)");
        return float(*const_cast<fp16b const*>(this));
    }

    inline operator float()
    {
        // FWEVENT("Converting to float (nonvolatile)");
        return float(*const_cast<fp16b const*>(this));
    }

    explicit inline operator uint32_t() const
    {
        // FWEVENT("Converting to uint32_t");
        return static_cast<uint32_t>(impl);
    }

    explicit inline operator uint32_t() volatile
    {
        // FWEVENT("Converting to uint32_t");
        return uint32_t(*const_cast<fp16b const*>(this));
    }

    explicit inline operator uint32_t()
    {
        // FWEVENT("Converting to uint32_t");
        return uint32_t(*const_cast<fp16b const*>(this));
    }

    inline fp16b operator+(fp16b volatile& other) volatile
    {
        return float(*const_cast<fp16b const*>(this)) + float(other);
    }

    // 1 sign bit, 8 exponent bits (bias = 127), 7 mantissa bits
    static inline uint16_t float_to_u16(float val)
    {
        uint32_t* sorry_gcc = reinterpret_cast<uint32_t*>(&val);
        uint32_t val_bits   = *sorry_gcc;

        return static_cast<uint16_t>(val_bits >> 16);
    }
};

inline float absf(float x)
{
    return x < 0.0f ? -x : x;
}

uint8_t fmt_to_dest_type(DataFormat fmt)
{
    switch (fmt)
    {
        case DataFormat::Float32:
            return RISC_DEST_FMT_FP32;
        case DataFormat::Float16:
            return RISC_DEST_FMT_FP16A;
        case DataFormat::Float16_b:
            return RISC_DEST_FMT_FP16B;
        case DataFormat::Int32:
            return RISC_DEST_FMT_INT32;
        case DataFormat::Int16:
            return RISC_DEST_FMT_INT16;
        case DataFormat::Int8:
            return RISC_DEST_FMT_INT8;
        default:
            // FWASSERT(0, "Unsupported dest format");
            return RISC_DEST_FMT_INT16;
    }
}

template <int t>
inline void set_dest_fmt(DataFormat fmt)
{
    set_dest_fmt<t>(fmt_to_dest_type(fmt));
}

template <typename T>
struct meta_from_dest_type
{
};

#define mk_dest_meta_type(T, _fmt, _is_32b)    \
    template <>                                \
    struct meta_from_dest_type<T>              \
    {                                          \
        static DataFormat const fmt = _fmt;    \
        static bool const is_32b    = _is_32b; \
    }

mk_dest_meta_type(uint32_t, DataFormat::Int32, true);
mk_dest_meta_type(int32_t, DataFormat::Int32, false);
mk_dest_meta_type(uint16_t, DataFormat::Int16, false);
mk_dest_meta_type(int16_t, DataFormat::Int16, false);
mk_dest_meta_type(uint8_t, DataFormat::Int8, false);
mk_dest_meta_type(int8_t, DataFormat::Int8, false);
mk_dest_meta_type(float, DataFormat::Float32, true);
mk_dest_meta_type(fp16a, DataFormat::Float16, false);
mk_dest_meta_type(fp16b, DataFormat::Float16_b, false);

#undef mk_dest_meta_type

#define dest_type_to_fmt(T) (meta_from_dest_type<T>::fmt)
#define dest_type_is_32b(T) (meta_from_dest_type<T>::is_32b)

} // namespace ckernel

template <>
struct _type_to_datasz<ckernel::fp16a>
{
    static vdatasz const sz = E16;
};

template <>
struct _type_to_datasz<ckernel::fp16b>
{
    static vdatasz const sz = E16;
};
#endif
