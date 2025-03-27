// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "sfpi.h"

#pragma once

// Inlining the tests may make the tests that use a parameter fail to test the
// non-imm path as compiling w/ -flto will fill in the value as an immediate
#if defined(__GNUC__) && !defined(__clang__)
#define sfpi_test_noinline __attribute__((noinline))
#else
#define sfpi_test_noinline
#endif

using namespace ckernel;

namespace sfpi_test
{

sfpi_inline void copy_result_to_dreg0(int addr)
{
    sfpi::dst_reg[0] = sfpi::dst_reg[addr];
}

// Test infrastructure is set up to test float values, not ints
// Viewing the ints as floats leads to a mess (eg, denorms)
// Instead, compare in the kernel to the expected result and write a sentinal
// value for "pass" and the sfpi::vInt v value for "fail"
// Assumes this code is called in an "inner" if
sfpi_inline void set_expected_result(int addr, float sentinel, int expected, sfpi::vInt v)
{
    // Poor man's equals
    // Careful, the register is 19 bits and the immediate is sign extended 12
    // bits so comparing bit patterns w/ the MSB set won't work
    v_if (v >= expected && v < expected + 1)
    {
        sfpi::dst_reg[addr] = sentinel;
    }
    v_else
    {
        sfpi::dst_reg[addr] = v;
    }
    v_endif;
}

sfpi_inline sfpi::vInt test_interleaved_scalar_vector_cond(bool scalar_bool, sfpi::vFloat vec, float a, float b)
{
    if (scalar_bool)
    {
        return vec == a;
    }
    else
    {
        return vec == b;
    }
}

template <class vType>
sfpi_inline vType reduce_bool4(vType a, vType b, vType c, vType d, int reference)
{
    vType result1 = 0;
    v_if (a == reference && b == reference)
    {
        result1 = 1;
    }
    v_endif;

    vType result2 = 0;
    v_if (c == reference && d == reference)
    {
        result2 = 1;
    }
    v_endif;

    sfpi::vUInt result = 0;
    v_if (result1 == 1 && result2 == 1)
    {
        result = 1;
    }
    v_endif;

    return result;
}

sfpi_test_noinline void test1()
{
    // Test SFPLOADI, SFPSTORE
    sfpi::dst_reg[0] = 1.3f;
}

sfpi_test_noinline void test2()
{
    // Test SFPLOAD, SFPMOV
    sfpi::dst_reg[2] = -sfpi::dst_reg[0];

    // Out: ramp down from 0 to -63
    copy_result_to_dreg0(2);
}

sfpi_test_noinline void test3()
{
    // Test SFPENCC, SFPSETCC, SFPCOMPC, LOADI, MAD (in conditionals)
    // Note: WH complains about the integer tests storing into float formated
    // sfpi::dst_reg w/ exponent of 0, so some tests use SFPOR to pass the result
    // through violating the spirit of testing one thing at a time

    v_if (sfpi::dst_reg[0] == 0.0F)
    {
        // 1 load
        sfpi::dst_reg[3] = 10.0F;
    }
    v_else
    {
        // 1 load
        sfpi::dst_reg[3] = 20.0F;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 2.0F)
    {
        // 1 load
        sfpi::vFloat a   = 30.0F;
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 3.0F)
    {
        // 2 loads
        sfpi::dst_reg[3] = 1.005f;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 4.0F)
    {
        // 2 loads
        sfpi::vFloat a   = 1.005f;
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 5.0F)
    {
        // This will be a short w/ 1 load
        sfpi::vInt a = 0x3F80;
        a |= 0x3f800000;
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 6.0F)
    {
        // This will be an int w/ 2 loads
        sfpi::vInt a     = 0x3F80A3D7; // 1.005
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 7.0F)
    {
        // This will be an int w/ 1 load (not sign extended)
        sfpi::vInt a     = 0x8F80;
        sfpi::dst_reg[3] = a | 0x3f800000;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 8.0F)
    {
        // This will be a ushort w/ 1 load (not sign extended)
        sfpi::vUInt a    = 0x8F80U;
        sfpi::dst_reg[3] = a | 0x3f800000;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 9.0F)
    {
        // This will be an int w/ 2 loads
        sfpi::dst_reg[3] = 0x3F80A3D7; // 1.005
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::vUInt a    = static_cast<unsigned short>(0x3f80);
        sfpi::dst_reg[3] = a | 0x3f800000;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 11.0F)
    {
        // This will be a short w/ 1 load (sign extended)
        sfpi::dst_reg[3] = static_cast<short>(0x8f80);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 12.0F)
    {
        sfpi::vUInt a    = 0x3F80A3D7;
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 13.0F)
    {
        sfpi::dst_reg[3] = sfpi::s2vFloat16b(0.005f);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 14.0F)
    {
        sfpi::dst_reg[3] = sfpi::s2vFloat16a(0x3c05);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 15.0F)
    {
        sfpi::dst_reg[3] = 25.0; // double
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 16.0F)
    {
        sfpi::vFloat a   = 28.0; // double
        sfpi::dst_reg[3] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 17.0F)
    {
        sfpi::dst_reg[3] = vConst0p8373;
    }
    v_endif;

    // Below are from the limits test.  Test the compiler's ability to use
    // fp16a, fp16b or fp32 as needed

    v_if (sfpi::dst_reg[0] == 18.0F)
    {
        sfpi::dst_reg[3] = 1.9921875f; // 0x3fff0000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 19.0F)
    {
        sfpi::dst_reg[3] = 1.99609375f; // 0x3fff8000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 20.0F)
    {
        // This is fp16b w/ large exp, with pass_offset != 0 the mantissa will overflow, use fp32
        sfpi::dst_reg[3] = 130560.0f; // 0x47ff0000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 21.0F)
    {
        // This is fp16b w/ large exp, with pass_offset != 0 the mantissa will overflow, use fp32
        sfpi::dst_reg[3] = 130592.0f; // 0x47ff1000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 22.0F)
    {
        // This is fp16a w/ largest exp, with pass_offset != 0 the exponent will overflow, use fp32
        sfpi::dst_reg[3] = 65408.0f; // 0x477f8000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 23.0F)
    {
        // This is fp16a w/ largest exp, with pass_offset != 0 the exponent will overflow, use fp32
        sfpi::dst_reg[3] = 130816.0f; // 0x47ff8000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 24.0F)
    {
        // This is fp16a w/ smallest exp, with pass offset != 0 the exponent will underflow, use fp32
        sfpi::dst_reg[3] = 0.000121831894f; // 0x38ff8000
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 25.0F)
    {
        // This is fp16a w/ smallest exp, with pass offset != 0 the exponent will underflow, use fp32
        sfpi::dst_reg[3] = 0.000060915947f; // 0x387f8000
    }
    v_endif;

    // [0] = 10.0
    // [1] = 20.0
    // [2] = 30.0
    // [3] = 1.005
    // [4] = 1.005
    // [5] = 0x3f80
    // [6] = 1.005
    // [7] = 0x8F80
    // [8] = 0x8F80
    // [9] = 1.005
    // [10] = 0x3f80
    // [11] = 0xFFFF8f80
    // [12] = 1.005
    // [13] = 1.0
    // [14] = 1.875
    // [15] = 25.0
    // [16] = 28.0
    // [17] = 0.837300003
    // [18] on 20.0F

    copy_result_to_dreg0(3);
}

sfpi_test_noinline void test4()
{
    // Test SFPPUSHCC, SFPPOPCC, SFPMAD (in conditionals)
    // Test vector loads
    // Operators &&, ||, !

    sfpi::vFloat v = sfpi::dst_reg[0];

    sfpi::dst_reg[4] = v;

    v_if (v < 2.0F)
    {
        sfpi::dst_reg[4] = 64.0F;
    }
    v_endif;
    // [0,1] = 64.0

    v_if (v < 6.0F)
    {
        v_if (v >= 2.0F)
        {
            v_if (v >= 3.0F)
            {
                sfpi::dst_reg[4] = 65.0F;
            }
            v_else
            {
                sfpi::dst_reg[4] = 66.0F;
            }
            v_endif;

            v_if (v == 5.0F)
            {
                sfpi::dst_reg[4] = 67.0F;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
    // [2] = 66.0
    // [3, 4] = 65.0
    // [5] = 67.0

    v_if (v >= 6.0F)
    {
        v_if (v < 9.0F)
        {
            v_if (v == 6.0F)
            {
                sfpi::dst_reg[4] = 68.0F;
            }
            v_elseif (v != 8.0F)
            {
                sfpi::dst_reg[4] = 69.0F;
            }
            v_else
            {
                sfpi::dst_reg[4] = 70.0F;
            }
            v_endif;
        }
        v_elseif (v == 9.0F)
        {
            sfpi::dst_reg[4] = 71.0F;
        }
        v_elseif (v == 10.0F)
        {
            sfpi::dst_reg[4] = 72.0F;
        }

        v_endif;
    }
    v_endif;

    v_if (v >= 11.0F)
    {
        v_if (v < 18.0F && v >= 12.0F && v != 15.0F)
        {
            sfpi::dst_reg[4] = 120.0F;
        }
        v_else
        {
            sfpi::dst_reg[4] = -sfpi::dst_reg[0];
        }
        v_endif;
    }
    v_endif;

    v_if (v >= 18.0F && v < 23.0F)
    {
        v_if (v == 19.0F || v == 21.0F)
        {
            sfpi::dst_reg[4] = 160.0F;
        }
        v_else
        {
            sfpi::dst_reg[4] = 180.0F;
        }
        v_endif;
    }
    v_endif;

    // Test ! on OP
    v_if (!(v != 23.0F))
    {
        sfpi::dst_reg[4] = 200.0F;
    }
    v_endif;

    v_if (!(v >= 25.0F) && !(v < 24.0F))
    {
        sfpi::dst_reg[4] = 220.0F;
    }
    v_endif;

    // Test ! on Boolean
    v_if (!((v < 25.0F) || (v >= 26.0F)))
    {
        sfpi::dst_reg[4] = 240.0F;
    }
    v_endif;

    v_if ((v >= 26.0F) && (v < 29.0F))
    {
        sfpi::dst_reg[4] = 260.0F;
        v_if (!((v >= 27.0F) && (v < 28.0F)))
        {
            sfpi::dst_reg[4] = 270.0F;
        }
        v_endif;
    }
    v_endif;

    // Test || after && to be sure PUSHC works properly
    v_if ((v >= 28.0F) && (v == 29.0F || v == 30.0F || v == 31.0F))
    {
        sfpi::vFloat x = 30.0F;
        sfpi::vFloat y = 280.0F;
        v_if (v < x)
        {
            y += 10.0F;
        }
        v_endif;
        v_if (v == x)
        {
            y += 20.0F;
        }
        v_endif;
        v_if (v >= x)
        {
            y += 40.0F;
        }
        v_endif;
        sfpi::dst_reg[4] = y;
    }
    v_endif;

    // [7] = 69.0
    // [8] = 70.0
    // [9] = 71.0
    // [10] = 72.0
    // [11] = -11.0
    // [12] = 120.0
    // [13] = 120.0
    // [14] = 120.0
    // [15] = -15.0
    // [16] = 120.0
    // [17] = 120.0
    // [18] = 180.0
    // [19] = 160.0
    // [20] = 180.0
    // [21] = 160.0
    // [22] = 180.0
    // [23] = 200.0
    // [24] = 220.0
    // [25] = 240.0
    // [26] = 270.0
    // [27] = 260.0
    // [28] = 270.0
    // [29] = 290.0
    // [30] = 340.0
    // [31] = 320.0

    // Remainder is -ramp
    copy_result_to_dreg0(4);
}

sfpi_test_noinline void test5()
{
    // Test SFPMAD, SFPMOV, vConsts
    sfpi::dst_reg[5] = -sfpi::dst_reg[0];

    vConstFloatPrgm0 = .5F;
    vConstFloatPrgm1 = 1.5F;
    vConstIntPrgm2   = 0xBFC00000; // -1.5F

    v_if (sfpi::dst_reg[0] == 0.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConst0p8373;
    }
    v_elseif (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConst0;
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConstNeg1;
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConstFloatPrgm0;
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConstFloatPrgm1;
    }
    v_elseif (sfpi::dst_reg[0] == 5.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConstFloatPrgm2;
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::dst_reg[5] = vConst0 * vConst0 + vConst1;
    }
    v_endif;
    // [0] = 0.8373
    // [1] = 0.0
    // [2] = -1.0
    // [3] = .5
    // [4] = 1.5
    // [5] = -1.5
    // [6] = 1.0

    // Fill holes in the tests; grayskull tested other const regs
    v_if (sfpi::dst_reg[0] == 7.0F)
    {
        sfpi::dst_reg[5] = vConst0;
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::dst_reg[5] = vConst0;
    }
    v_elseif (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::dst_reg[5] = vConst0;
    }
    v_elseif (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::dst_reg[5] = vConst0;
    }
    v_elseif (sfpi::dst_reg[0] == 11.0F)
    {
        sfpi::dst_reg[5] = vConst0p8373 * vConstNeg1 + vConst1;
    }
    v_endif;
    // [7] = 0.0
    // [8] = 0.0
    // [9] = 0.0
    // [10] = 0.0
    // [11] = .1627

    sfpi::vFloat a = sfpi::dst_reg[0];
    sfpi::vFloat b = 20.0F;

    // Note: loading sfpi::dst_reg[0] takes a reg and comparing against a float const
    // takes a reg so can't store A, B and C across the condtionals

    v_if (sfpi::dst_reg[0] == 12.0F)
    {
        sfpi::dst_reg[5] = a * b;
    }
    v_elseif (sfpi::dst_reg[0] == 13.0F)
    {
        sfpi::dst_reg[5] = a + b;
    }
    v_elseif (sfpi::dst_reg[0] == 14.0F)
    {
        sfpi::dst_reg[5] = a * b + 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 15.0F)
    {
        sfpi::dst_reg[5] = a + b + 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 16.0F)
    {
        sfpi::dst_reg[5] = a * b - 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 17.0F)
    {
        sfpi::dst_reg[5] = a + b - 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 18.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = a * b + c;
    }
    v_endif;
    // [12] = 240.0
    // [13] = 33.0
    // [14] = 280.5
    // [15] = 35.5
    // [16] = 319.5
    // [17] = 36.5
    // [18] = 355.0

    v_if (sfpi::dst_reg[0] == 19.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = a * b + c + 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 20.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = a * b + c - 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 21.0F)
    {
        sfpi::vFloat c = -5.0F;
        sfpi::vFloat d;
        d                = a * b + c - 0.5F;
        sfpi::dst_reg[5] = d;
    }
    v_elseif (sfpi::dst_reg[0] == 22.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = a * b - c;
    }
    v_elseif (sfpi::dst_reg[0] == 23.0F)
    {
        sfpi::dst_reg[5] = a * b + vConst1;
    }
    v_elseif (sfpi::dst_reg[0] == 24.0F)
    {
        sfpi::dst_reg[5] = vConst1 * b + vConst1;
    }
    v_endif;
    // [19] = 375.5
    // [20] = 394.5
    // [21] = 414.5
    // [22] = 445.0
    // [23] = 461.0
    // [24] = 21.0

    v_if (sfpi::dst_reg[0] == 25.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = sfpi::dst_reg[0] * b + c;
    }
    v_elseif (sfpi::dst_reg[0] == 26.0F)
    {
        sfpi::vFloat c   = -5.0F;
        sfpi::dst_reg[5] = b * sfpi::dst_reg[0] + c;
    }
    v_elseif (sfpi::dst_reg[0] == 27.0F)
    {
        sfpi::dst_reg[5] = a * b + sfpi::dst_reg[0];
    }
    v_elseif (sfpi::dst_reg[0] == 28.0F)
    {
        sfpi::dst_reg[5] = a * b - sfpi::dst_reg[0];
    }
    v_endif;
    // [25] = 495.0
    // [26] = 515.0
    // [27] = 567.0
    // [28] = 532.0

    v_if (sfpi::dst_reg[0] == 29.0F)
    {
        sfpi::dst_reg[5] = a - b;
    }
    v_elseif (sfpi::dst_reg[0] == 30.0F)
    {
        sfpi::dst_reg[5] = a - b - 0.5F;
    }
    v_elseif (sfpi::dst_reg[0] == 31.0F)
    {
        sfpi::dst_reg[5] = sfpi::dst_reg[0] - b + 0.5F;
    }
    v_endif;
    // [29] = 9.0
    // [30] = 9.5
    // [31] = 11.5

    copy_result_to_dreg0(5);
}

sfpi_test_noinline void test6()
{
    // Note: set_expected_result uses SFPIADD so can't really be used early in
    // this routine w/o confusing things

    // SFPIADD

    sfpi::dst_reg[6] = -sfpi::dst_reg[0];

    v_if (sfpi::dst_reg[0] < 3.0F)
    {
        v_if (sfpi::dst_reg[0] >= 0.0F)
        {
            sfpi::dst_reg[6] = 256.0F;

            sfpi::vInt a;
            v_if (sfpi::dst_reg[0] == 0.0F)
            {
                a = 28;
            }
            v_elseif (sfpi::dst_reg[0] == 1.0F)
            {
                a = 29;
            }
            v_elseif (sfpi::dst_reg[0] == 2.0F)
            {
                a = 30;
            }
            v_endif;

            sfpi::vInt b;
            // IADD imm
            b = a - 29;
            v_if (b >= 0)
            {
                sfpi::dst_reg[6] = 1024.0F;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] < 6.0F)
    {
        v_if (sfpi::dst_reg[0] >= 3.0F)
        {
            sfpi::dst_reg[6] = 256.0F;

            sfpi::vInt a;
            v_if (sfpi::dst_reg[0] == 3.0F)
            {
                a = 28;
            }
            v_elseif (sfpi::dst_reg[0] == 4.0F)
            {
                a = 29;
            }
            v_elseif (sfpi::dst_reg[0] == 5.0F)
            {
                a = 30;
            }
            v_endif;

            sfpi::vInt b = -29;
            // IADD reg
            b = a + b;
            v_if (b < 0)
            {
                sfpi::dst_reg[6] = 1024.0F;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] < 9.0F)
    {
        v_if (sfpi::dst_reg[0] >= 6.0F)
        {
            sfpi::dst_reg[6] = 16.0F;

            sfpi::vInt a = 3;
            v_if (sfpi::dst_reg[0] == 6.0F)
            {
                a = 28;
            }
            v_elseif (sfpi::dst_reg[0] == 7.0F)
            {
                a = 29;
            }
            v_elseif (sfpi::dst_reg[0] == 8.0F)
            {
                a = 30;
            }
            v_endif;

            sfpi::vFloat b = 128.0F;
            v_if (a >= 29)
            {
                b = 256.0F;
            }
            v_endif;

            v_if (a < 29)
            {
                b = 512.0F;
            }
            v_elseif (a >= 30)
            {
                b = 1024.0F;
            }
            v_endif;

            sfpi::dst_reg[6] = b;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] < 12.0F)
    {
        v_if (sfpi::dst_reg[0] >= 9.0F)
        {
            sfpi::dst_reg[6] = 16.0F;

            sfpi::vInt a = 3;
            v_if (sfpi::dst_reg[0] == 9.0F)
            {
                a = 28;
            }
            v_elseif (sfpi::dst_reg[0] == 10.0F)
            {
                a = 29;
            }
            v_elseif (sfpi::dst_reg[0] == 11.0F)
            {
                a = 30;
            }
            v_endif;

            sfpi::vFloat b = 128.0F;
            sfpi::vInt c   = 29;
            v_if (a >= c)
            {
                b = 256.0F;
            }
            v_endif;

            v_if (a < c)
            {
                b = 512.0F;
            }
            v_elseif (a >= 30)
            {
                b = 1024.0F;
            }
            v_endif;

            sfpi::dst_reg[6] = b;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 12.0F)
    {
        sfpi::vInt v = 25;
        set_expected_result(6, 4.0F, 25, v);
    }
    v_elseif (sfpi::dst_reg[0] == 13.0F)
    {
        sfpi::vInt a = 20;
        a            = a + 12;
        set_expected_result(6, 8.0F, 32, a);
    }
    v_elseif (sfpi::dst_reg[0] == 14.0F)
    {
        sfpi::vInt a = 18;
        sfpi::vInt b = -6;
        a            = a + b;
        set_expected_result(6, 16.0F, 12, a);
    }
    v_elseif (sfpi::dst_reg[0] == 15.0F)
    {
        sfpi::vInt a = 14;
        sfpi::vInt b = -5;
        a            = b + a;
        set_expected_result(6, 32.0F, 9, a);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 16.0F)
    {
        sfpi::vInt v = 25;
        set_expected_result(6, 4.0F, 25, v);
    }
    v_elseif (sfpi::dst_reg[0] == 17.0F)
    {
        sfpi::vInt a = 20;
        a            = a - 12;
        set_expected_result(6, 8.0F, 8, a);
    }
    v_elseif (sfpi::dst_reg[0] == 18.0F)
    {
        sfpi::vInt a = 18;
        sfpi::vInt b = 6;
        a            = a - b;
        set_expected_result(6, 16.0F, 12, a);
    }
    v_elseif (sfpi::dst_reg[0] == 19.0F)
    {
        sfpi::vInt a = 14;
        sfpi::vInt b = 5;
        a            = b - a;
        set_expected_result(6, 32.0F, -9, a);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 20.0F)
    {
        sfpi::vUInt v = 25;
        set_expected_result(6, 4.0F, 25, reinterpret<sfpi::vInt>(v));
    }
    v_elseif (sfpi::dst_reg[0] == 21.0F)
    {
        sfpi::vUInt a = 20;
        a             = a - 12;
        set_expected_result(6, 8.0F, 8, reinterpret<sfpi::vInt>(a));
    }
    v_elseif (sfpi::dst_reg[0] == 22.0F)
    {
        sfpi::vUInt a = 18;
        sfpi::vUInt b = 6;
        a             = a - b;
        set_expected_result(6, 16.0F, 12, reinterpret<sfpi::vInt>(a));
    }
    v_elseif (sfpi::dst_reg[0] == 23.0F)
    {
        sfpi::vUInt a = 14;
        sfpi::vUInt b = 5;
        a             = b - a;
        set_expected_result(6, 32.0F, -9, reinterpret<sfpi::vInt>(a));
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 24.0F)
    {
        sfpi::vInt a = 10;
        sfpi::vInt b = 20;
        a -= b;
        set_expected_result(6, 64.0F, -10, a);
    }
    v_elseif (sfpi::dst_reg[0] == 25.0F)
    {
        sfpi::vInt a = 10;
        sfpi::vInt b = 20;
        a += b;
        set_expected_result(6, 128.0F, 30, a);
    }
    v_endif;

    // Pseudo-16 bit via hidden loadi
    v_if (sfpi::dst_reg[0] == 26.0F)
    {
        sfpi::vInt a = 10;
        a += 4096;
        set_expected_result(6, 256.0F, 4106, a);
    }
    v_elseif (sfpi::dst_reg[0] == 27.0F)
    {
        sfpi::vInt a = 4096;
        v_if (a >= 4096)
        {
            sfpi::dst_reg[6] = 512.0f;
        }
        v_else
        {
            sfpi::dst_reg[6] = 0.0f;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] >= 28.0F)
    {
        sfpi::vInt a = vConstTileId;
        v_if (sfpi::dst_reg[0] == 28.0F)
        {
            set_expected_result(6, 256.0F, 56, a);
        }
        v_elseif (sfpi::dst_reg[0] == 29.0F)
        {
            set_expected_result(6, 256.0F, 58, a);
        }
        v_elseif (sfpi::dst_reg[0] == 30.0F)
        {
            set_expected_result(6, 256.0F, 60, a);
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 31.0F)
    {
        sfpi::vFloat x = 3.0f;
        v_if (!!(x == 3.0f && x != 4.0f))
        {
            sfpi::dst_reg[6] = 16.0;
        }
        v_else
        {
            sfpi::dst_reg[6] = 32.0;
        }
        v_endif;
    }
    v_endif;

    // [0] = 256.0
    // [1] = 1024.0
    // [2] = 1024.0
    // [3] = 1024.0
    // [4] = 256.0
    // [5] = 256.0
    // [6] = 512.0
    // [7] = 256.0
    // [8] = 1024.0
    // [9] = 512.0
    // [10] = 256.0
    // [11] = 1024.0
    // [12] = 4.0
    // [13] = 8.0
    // [14] = 16.0
    // [15] = 32.0
    // [16] = 4.0
    // [17] = 8.0
    // [18] = 16.0
    // [19] = 32.0
    // [20] = 4.0
    // [21] = 8.0
    // [22] = 16.0
    // [23] = 32.0
    // [24] = 64.0
    // [25] = 128.0
    // [26] = 256.0
    // [27] = 512.0
    // [28] = 256.0
    // [29] = 256.0
    // [30] = 256.0
    // [31] = 16.0

    copy_result_to_dreg0(6);
}

sfpi_test_noinline void test7()
{
    // SFPEXMAN, SFPEXEXP, SFPSETEXP, SFPSETMAN
    // Plus a little more && ||

    sfpi::dst_reg[7] = -sfpi::dst_reg[0];
    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::vFloat tmp = 124.05F;
        set_expected_result(7, 30.0F, 0xF8199A, exman8(tmp));
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::vFloat tmp = 124.05F;
        set_expected_result(7, 32.0F, 0x78199A, exman9(tmp));
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vFloat tmp = 65536.0F * 256.0F;
        set_expected_result(7, 33.0F, 0x18, exexp(tmp));
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vFloat tmp = 65536.0F * 256.0F;
        set_expected_result(7, 34.0F, 0x97, exexp_nodebias(tmp));
    }
    v_elseif (sfpi::dst_reg[0] < 8.0F)
    {
        sfpi::vFloat tmp;
        v_if (sfpi::dst_reg[0] == 5.0F)
        {
            // Exp < 0 for 5.0
            tmp = 0.5F;
        }
        v_elseif (sfpi::dst_reg[0] < 8.0F)
        {
            // Exp > 0 for 6.0, 7.0
            tmp = 512.0F;
        }
        v_endif;

        sfpi::vInt v;
        v = exexp(tmp);
        v_if (v < 0)
        {
            sfpi::dst_reg[7] = 32.0F;
        }
        v_else
        {
            sfpi::dst_reg[7] = 64.0F;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 7.0F)
        {
            // Exponent is 9, save it
            set_expected_result(7, 35.0F, 9, v);
        }
        v_endif;
        // [0] = 64.0
        // [1] = 30.0
        // [2] = 32.0
        // [3] = 33.0
        // [4] = 34.0
        // [5] = 32.0
        // [6] = 64.0
        // [7] = 35.0 (exponent(512) = 8)
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::vFloat tmp = 1.0F;
        sfpi::vFloat v   = setexp(tmp, 137);
        sfpi::dst_reg[7] = v;
    }
    v_elseif (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::vInt exp       = 0x007F;   // Exponent in low bits
        sfpi::vFloat sgn_man = -1664.0F; // 1024 + 512 + 128 or 1101
        sgn_man              = setexp(sgn_man, exp);
        sfpi::dst_reg[7]     = sgn_man;
    }
    v_endif;

    // [8] = 1024.0
    // [9] = -1.625

    v_if (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::vFloat tmp = 1024.0F;
        sfpi::vFloat b   = setman(tmp, 0x75019a);
        sfpi::dst_reg[7] = b;
    }
    v_elseif (sfpi::dst_reg[0] == 11.0F)
    {
        sfpi::vFloat tmp  = 1024.0F;
        sfpi::vInt man    = 0x75019a;
        sfpi::vFloat tmp2 = setman(tmp, man);
        sfpi::dst_reg[7]  = tmp2;
    }
    v_endif;

    // [10] = 1960.050049
    // [11] = 1960.050049

    sfpi::vFloat v = sfpi::dst_reg[0];
    v_if ((v >= 12.0f && v < 14.0f) || (v >= 15.0f && v < 17.0f))
    {
        sfpi::dst_reg[7] = -128.0f;
    }
    v_endif;
    // [12] = -128.0
    // [13] = -128.0
    // [14] = -14.0
    // [15] = -128.0
    // [16] = -128.0

    v_if (((v >= 17.0f && v < 18.0f) || (v >= 19.0f && v < 20.0f)) || ((v >= 21.0f && v < 22.0f) || (v >= 23.0f && v < 24.0f)))
    {
        sfpi::dst_reg[7] = -256.0f;
    }
    v_endif;
    // [17] = -256.0
    // [18] = -18.0
    // [19] = -256.0
    // [20] = -20.0
    // [21] = -256.0
    // [22] = -22.0
    // [23] = -256.0
    // [24] = -24.0

    v_if (v >= 25.0f && v < 29.0f)
    {
        v_if (!(v >= 25.0f && v < 26.0f) && !(v >= 27.0f && v < 28.0f))
        {
            sfpi::dst_reg[7] = -1024.0f;
        }
        v_endif;
    }
    v_endif;
    // [25] = -25.0
    // [26] = -1024.0
    // [27] = -27.0
    // [28] = -1024.0

    // <= and > are compound statements in the compiler, <= uses a compc
    // and things get flipped around when joined by ||
    v_if (v >= 29.0f && v < 32.0f)
    {
        sfpi::vInt t       = vConstTileId >> 1;
        sfpi::vFloat total = 16.0F;

        v_if (t <= 30)
        {
            total += 32.0F;
        }
        v_endif;
        v_if (t > 30)
        {
            total += 64.0F;
        }
        v_endif;
        v_if (!(t > 30))
        {
            total += 128.0F;
        }
        v_endif;
        v_if (!(t <= 30))
        {
            total += 256.0F;
        }
        v_endif;
        v_if (t <= 29 || t > 30)
        {
            total += 512.0F;
        }
        v_endif;
        v_if (t > 30 || t <= 29)
        {
            total += 1024.0F;
        }
        v_endif;

        sfpi::dst_reg[7] = total;
    }
    v_endif;
    // [29] = 1712.0
    // [30] = 176.0
    // [31] = 1872.0

    copy_result_to_dreg0(7);
}

sfpi_test_noinline void test8()
{
    // SFPAND, SFPOR, SFPNOT, SFPABS
    // Atypical usage of conditionals
    // More conditionals (short v compares)

    sfpi::dst_reg[8] = -sfpi::dst_reg[0];
    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::vUInt a = 0x05FF;
        sfpi::vUInt b = 0x0AAA;
        b &= a;
        set_expected_result(8, 16.0F, 0x00AA, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::vUInt a = 0x05FF;
        sfpi::vUInt b = 0x0AAA;
        sfpi::vUInt c = a & b;
        set_expected_result(8, 16.0F, 0x00AA, static_cast<sfpi::vInt>(c));
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vInt a = 0x05FF;
        sfpi::vInt b = 0x0AAA;
        b &= a;
        set_expected_result(8, 16.0F, 0x00AA, b);
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vInt a = 0x05FF;
        sfpi::vInt b = 0x0AAA;
        sfpi::vInt c = a & b;
        set_expected_result(8, 16.0F, 0x00AA, c);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 5.0F)
    {
        sfpi::vUInt a = 0x0111;
        sfpi::vUInt b = 0x0444;
        b |= a;
        set_expected_result(8, 20.0F, 0x0555, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::vUInt a = 0x0111;
        sfpi::vUInt b = 0x0444;
        sfpi::vUInt c = b | a;
        set_expected_result(8, 20.0F, 0x0555, static_cast<sfpi::vInt>(c));
    }
    v_elseif (sfpi::dst_reg[0] == 7.0F)
    {
        sfpi::vInt a = 0x0111;
        sfpi::vInt b = 0x0444;
        b |= a;
        set_expected_result(8, 20.0F, 0x0555, b);
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::vInt a = 0x0111;
        sfpi::vInt b = 0x0444;
        sfpi::vInt c = b | a;
        set_expected_result(8, 20.0F, 0x0555, c);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::vUInt a = 0x0AAA;
        a             = ~a;
        a &= 0x0FFF; // Tricky since ~ flips upper bits that immediates can't access
        set_expected_result(8, 22.0F, 0x0555, static_cast<sfpi::vInt>(a));
    }
    v_elseif (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::vFloat a   = 100.0F;
        sfpi::dst_reg[8] = sfpi::abs(a);
    }
    v_elseif (sfpi::dst_reg[0] == 11.0F)
    {
        sfpi::vFloat a   = -100.0F;
        sfpi::dst_reg[8] = sfpi::abs(a);
    }
    v_elseif (sfpi::dst_reg[0] == 12.0F)
    {
        sfpi::vInt a = 100;
        set_expected_result(8, 24.0F, 100, sfpi::abs(a));
    }
    v_elseif (sfpi::dst_reg[0] == 13.0F)
    {
        sfpi::vInt a = -100;
        set_expected_result(8, 26.0F, 100, sfpi::abs(a));
    }
    v_endif;

    v_if (test_interleaved_scalar_vector_cond(true, sfpi::dst_reg[0], 14.0F, 15.0F))
    {
        sfpi::dst_reg[8] = 32.0F;
    }
    v_elseif (test_interleaved_scalar_vector_cond(false, sfpi::dst_reg[0], 14.0F, 15.0F))
    {
        sfpi::dst_reg[8] = 16.0F;
    }
    v_endif;

    sfpi::vFloat tmp = sfpi::dst_reg[8];
    v_block
    {
        v_and(sfpi::dst_reg[0] >= 16.0F);

        for (int x = 0; x < 4; x++)
        {
            v_and(sfpi::dst_reg[0] < 20.0F - x);
            tmp += 16.0F;
        }
    }
    v_endblock;
    sfpi::dst_reg[8] = tmp;

    // <= and > are compound statements in the compiler, <= uses a compc
    // and things get flipped around when joined by ||
    v_if (sfpi::dst_reg[0] >= 20.0f && sfpi::dst_reg[0] < 23.0f)
    {
        sfpi::vInt t    = vConstTileId >> 1;
        sfpi::vInt low  = 20;
        sfpi::vInt high = 21;

        sfpi::dst_reg[8] = 16.0f;

        v_if (t <= high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 32.0F;
        }
        v_endif;
        v_if (t > high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 64.0F;
        }
        v_endif;
        v_if (!(t > high))
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 128.0F;
        }
        v_endif;
        v_if (!(t <= high))
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 256.0F;
        }
        v_endif;
        v_if (t <= low || t > high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 512.0F;
        }
        v_endif;
        v_if (t > high || t <= low)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 1024.0F;
        }
        v_endif;
    }
    v_endif;

    // Do the same tests as above, but for floats
    v_if (sfpi::dst_reg[0] >= 23.0f && sfpi::dst_reg[0] < 26.0f)
    {
        sfpi::vFloat t     = sfpi::dst_reg[0];
        sfpi::vFloat total = 16.0F;

        v_if (t <= 24.0f)
        {
            total += 32.0F;
        }
        v_endif;
        v_if (t > 24.0f)
        {
            total += 64.0F;
        }
        v_endif;
        v_if (!(t > 24.0f))
        {
            total += 128.0F;
        }
        v_endif;
        v_if (!(t <= 24.0f))
        {
            total += 256.0F;
        }
        v_endif;
        v_if (t <= 23.0f || t > 24.0f)
        {
            total += 512.0F;
        }
        v_endif;
        v_if (t > 24.0f || t <= 23.0f)
        {
            total += 1024.0F;
        }
        v_endif;

        sfpi::dst_reg[8] = total;
    }
    v_endif;

    // More of the same, again for floats.  Reloads for reg pressure
    v_if (sfpi::dst_reg[0] >= 26.0f && sfpi::dst_reg[0] < 29.0f)
    {
        sfpi::vFloat low  = 26.0f;
        sfpi::vFloat high = 27.0f;

        sfpi::dst_reg[8] = 16.0f;

        sfpi::vFloat t = sfpi::dst_reg[0];
        v_if (t <= high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 32.0F;
        }
        v_endif;
        t = sfpi::dst_reg[0];
        v_if (t > high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 64.0F;
        }
        v_endif;
        t = sfpi::dst_reg[0];
        v_if (!(t > high))
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 128.0F;
        }
        v_endif;
        t = sfpi::dst_reg[0];
        v_if (!(t <= high))
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 256.0F;
        }
        v_endif;
        t = sfpi::dst_reg[0];
        v_if (t <= low || t > high)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 512.0F;
        }
        v_endif;
        t   = sfpi::dst_reg[0];
        low = 26.0f;
        v_if (t > high || t <= low)
        {
            sfpi::dst_reg[8] = sfpi::dst_reg[8] + 1024.0F;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 29.0F)
    {
        sfpi::vInt a = 0xA5A5;
        sfpi::vInt b = 0xFF00;
        sfpi::vInt c = a ^ b;
        set_expected_result(8, 32.0F, 0x5AA5, c);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 30.0F)
    {
        sfpi::vUInt a = 0xA5A5;
        sfpi::vUInt b = 0xFF00;
        sfpi::vUInt c = a ^ b;
        set_expected_result(8, 64.0F, 0x5AA5, c);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 31.0F)
    {
        sfpi::vInt a = 0xA5A5;
        sfpi::vInt b = 0xFF00;
        b ^= a;
        set_expected_result(8, 32.0F, 0x5AA5, b);
    }
    v_endif;

    // [0] = 0
    // [1] = 16.0
    // [2] = 16.0
    // [3] = 16.0
    // [4] = 16.0
    // [5] = 20.0
    // [6] = 20.0
    // [7] = 20.0
    // [8] = 20.0
    // [9] = 22.0
    // [10] = 100.0
    // [11] = 100.0
    // [12] = 24.0
    // [13] = 26.0
    // [14] = 32.0
    // [15] = 16.0
    // [16] = 48.0
    // [17] = 31.0
    // [18] = 14.0
    // [19] = -3.0
    // [20] = 1712.0
    // [21] = 176.0
    // [22] = 1872.0
    // [23] = 1712.0
    // [24] = 176.0
    // [25] = 1872.0
    // [26] = 1712.0
    // [27] = 176.0
    // [28] = 1872.0
    // [29] = 32.0
    // [30] = 64.0
    // [31] = 32.0

    copy_result_to_dreg0(8);
}

sfpi_test_noinline void test9()
{
    // SFPMULI, SFPADDI, SFPDIVP2, SFPLZ
    // More conditional tests

    sfpi::dst_reg[9] = -sfpi::dst_reg[0];
    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::vFloat a   = 20.0F;
        sfpi::dst_reg[9] = a * 30.0F;
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::vFloat a = 20.0F;
        a *= 40.0F;
        sfpi::dst_reg[9] = a;
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vFloat a   = 20.0F;
        sfpi::dst_reg[9] = a + 30.0F;
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vFloat a = 20.0F;
        a += 40.0F;
        sfpi::dst_reg[9] = a;
    }
    v_elseif (sfpi::dst_reg[0] == 5.0F)
    {
        sfpi::vFloat a   = 16.0F;
        sfpi::dst_reg[9] = addexp(a, 4);
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::vFloat a   = 256.0F;
        sfpi::dst_reg[9] = addexp(a, -4);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 7.0F)
    {
        sfpi::vInt a = 0;
        sfpi::vInt b = lz(a);
        set_expected_result(9, 38.0F, 0x20, b);
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::vInt a = 0xFFFFFFFF;
        sfpi::vInt b = lz(a);
        set_expected_result(9, 55.0F, 0x0, b);
    }
    v_elseif (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::vUInt a = 0xFFFFU;
        sfpi::vInt b  = lz(a);
        set_expected_result(9, 30.0F, 0x10, b);
    }
    v_elseif (sfpi::dst_reg[0] < 13.0F)
    {
        sfpi::vFloat a = sfpi::dst_reg[0] - 11.0F;
        sfpi::vUInt b;

        // Relies on if chain above...
        v_if (sfpi::dst_reg[0] >= 7.0F)
        {
            b = sfpi::reinterpret<sfpi::vUInt>(lz(a));
            v_if (b != 32)
            {
                sfpi::dst_reg[9] = 60.0F;
            }
            v_else
            {
                sfpi::dst_reg[9] = 40.0F;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 13.0F)
    {
        sfpi::vFloat x = 1.0F;

        x *= 2.0f;
        x *= -3.0f;
        x += 4.0f;
        x += -4.0f;

        sfpi::dst_reg[9] = x;
    }
    v_elseif (sfpi::dst_reg[0] == 14.0F)
    {
        // MULI/ADDI don't accept fp16a
        // Ensure this goes to MAD

        sfpi::vFloat x = 1.0F;

        x *= sfpi::s2vFloat16a(2.0);
        x *= sfpi::s2vFloat16a(-3.0);
        x += sfpi::s2vFloat16a(4.0);
        x += sfpi::s2vFloat16a(-4.0);

        sfpi::dst_reg[9] = x;
    }
    v_endif;

    // Test more boolean expressions
    v_if (sfpi::dst_reg[0] >= 15.0F && sfpi::dst_reg[0] < 19.0)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if ((v <= 16.0f && v != 15.0f) || (v == 18.0f))
        {
            sfpi::dst_reg[9] = 32.0f;
        }
        v_endif;
    }
    v_endif;

    // Same as above, but flip the order of the top level OR
    v_if (sfpi::dst_reg[0] >= 19.0F && sfpi::dst_reg[0] < 23.0)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if ((v == 22.0f) || (v <= 20.0f && v != 19.0f))
        {
            sfpi::dst_reg[9] = 32.0f;
        }
        v_endif;
    }
    v_endif;

    v_if (
        (sfpi::dst_reg[0] == 23.0 || sfpi::dst_reg[0] == 24.0 || sfpi::dst_reg[0] == 25.0 || sfpi::dst_reg[0] == 26.0 || sfpi::dst_reg[0] == 27.0 ||
         sfpi::dst_reg[0] == 28.0) &&
        (sfpi::dst_reg[0] != 23.0 && sfpi::dst_reg[0] != 25.0 && sfpi::dst_reg[0] != 27.0f))
    {
        sfpi::dst_reg[9] = 64.0f;
    }
    v_endif;

    // [1] = 600.0
    // [2] = 800.0
    // [3] = 50.0
    // [4] = 60.0
    // [5] = 256.0
    // [6] = 16.0
    // [7] = 38.0
    // [8] = 55.0
    // [9] = 30.0
    // [10] = 60.0
    // [11] = 40.0
    // [12] = 60.0
    // [13] = -6.0
    // [14] = -6.0
    // [15] = -15.0
    // [16] = 32.0
    // [17] = -17.0
    // [18] = 32.0
    // [19] = -19.0
    // [20] = 32.0
    // [21] = -21.0
    // [22] = 32.0
    // [23] = -23.0
    // [24] = 64.0
    // [25] = -25.0
    // [26] = 64.0
    // [27] = -27.0
    // [28] = 64.0

    copy_result_to_dreg0(9);
}

sfpi_test_noinline void test10()
{
    // SFPSHFT, SFTSETSGN
    sfpi::dst_reg[10] = -sfpi::dst_reg[0];
    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::vUInt a    = 0x015;
        sfpi::vInt shift = 6;
        sfpi::vUInt b    = shft(a, shift);
        // Could write better tests if we could return and test the int result
        set_expected_result(10, 20.0F, 0x0540, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::vUInt a = 0x2AAA;
        sfpi::vUInt b = shft(a, -4);
        set_expected_result(10, 22.0F, 0x02AA, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vUInt a    = 0xAAAAU;
        sfpi::vInt shift = -6;
        sfpi::vUInt b    = shft(a, shift);
        set_expected_result(10, 24.0F, 0x02AA, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vUInt a = 0x005A;
        sfpi::vUInt b = shft(a, 4);
        set_expected_result(10, 26.0F, 0x05A0, static_cast<sfpi::vInt>(b));
    }
    v_elseif (sfpi::dst_reg[0] == 5.0F)
    {
        sfpi::vInt a = 25;
        a            = a + 5;
        a += 7;
        set_expected_result(10, 28.0F, 0x25, a);
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::vInt a = 28;
        sfpi::vInt b = 100;
        a += b;
        set_expected_result(10, 30.0F, 0x80, a);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 7.0F)
    {
        sfpi::vFloat a    = sfpi::dst_reg[0];
        sfpi::dst_reg[10] = sfpi::setsgn(a, 1);
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = -128.0;
        sfpi::vFloat r = sfpi::setsgn(b, a);

        sfpi::dst_reg[10] = r;
    }
    v_elseif (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::vFloat a    = -256.0F;
        sfpi::dst_reg[10] = sfpi::setsgn(a, 0);
    }
    v_elseif (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        a += 20.0f;
        sfpi::vFloat b = -512.0F;
        sfpi::vFloat r = sfpi::setsgn(a, b);

        sfpi::dst_reg[10] = r;
    }
    v_endif;

    // [1] = 20.0
    // [2] = 22.0
    // [3] = 24.0
    // [4] = 26.0
    // [5] = 28.0
    // [6] = 30.0
    // [7] = -7.0
    // [8] = 128.0
    // [9] = 256.0
    // [10] = -30.0
    copy_result_to_dreg0(10);
}

sfpi_test_noinline void test11()
{
    // SFPLUT, SFPLOADL<n>
    sfpi::dst_reg[11] = -sfpi::dst_reg[0];

    sfpi::vUInt l0a = 0xFF30; // Multiply by 0.0, add 0.125
    sfpi::vUInt l1a = 0X3020; // Multiply by 0.125, add 0.25
    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        // Use L0
        sfpi::vFloat h    = -0.3F;
        sfpi::vUInt l2a   = 0xA010; // Mulitply by -0.25, add 0.5
        h                 = lut_sign(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        // Use L0
        sfpi::vFloat h    = -0.3F;
        sfpi::vUInt l2a   = 0xA010; // Mulitply by -0.25, add 0.5
        h                 = lut(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        // Use L0
        sfpi::vFloat h  = -0.3F;
        sfpi::vUInt l2a = 0xA010; // Mulitply by -0.25, add 0.5
        // Test used a bias on Grayskull, not supported on Wormhole
        h                 = lut_sign(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        // Use L0
        sfpi::vFloat h  = -0.3F;
        sfpi::vUInt l2a = 0xA010; // Mulitply by -0.25, add 0.5
        // Test used a bias on Grayskull, not supported on Wormhole
        h                 = lut(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_elseif (sfpi::dst_reg[0] == 5.0F)
    {
        // Use L1
        sfpi::vFloat h  = 1.0F;
        sfpi::vUInt l2a = 0xA010; // Mulitply by -0.25, add 0.5
        // Test used a bias on Grayskull, not supported on Wormhole
        h                 = lut(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        // Use L2
        sfpi::vFloat h    = 4.0F;
        sfpi::vUInt l2a   = 0xA010; // Mulitply by -0.25, add 0.5
        h                 = lut_sign(h, l0a, l1a, l2a);
        sfpi::dst_reg[11] = h;
    }
    v_endif;

    {
        // Clear out the LUT, re-load it w/ ASM instructions, the pull it into
        // variables for the SFPLUT
        l0a = 0;
        l1a = 0;

        // These are fakedout w/ emule
        TTI_SFPLOADI(0, SFPLOADI_MOD0_USHORT, 0xFF20); // Mulitply by 0.0, add 0.25
        TTI_SFPLOADI(1, SFPLOADI_MOD0_USHORT, 0x2010); // Mulitply by 0.25, add 0.5
        sfpi::vUInt l0b, l1b;
        l0b = l_reg[LRegs::LReg0];
        l1b = l_reg[LRegs::LReg1];

        v_if (sfpi::dst_reg[0] == 7.0F)
        {
            // Use L0
            sfpi::vFloat h    = -0.3F;
            sfpi::vUInt l2b   = 0x9000;
            h                 = lut_sign(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_elseif (sfpi::dst_reg[0] == 8.0F)
        {
            // Use L0
            sfpi::vFloat h    = -0.3F;
            sfpi::vUInt l2b   = 0x9000;
            h                 = lut(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_elseif (sfpi::dst_reg[0] == 9.0F)
        {
            // Use L0
            sfpi::vFloat h  = -0.3F;
            sfpi::vUInt l2b = 0x9000;
            // Test used a bias on Grayskull, not supported on Wormhole
            h                 = lut_sign(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_elseif (sfpi::dst_reg[0] == 10.0F)
        {
            // Use L0
            sfpi::vFloat h  = -0.3F;
            sfpi::vUInt l2b = 0x9000;
            // Test used a bias on Grayskull, not supported on Wormhole
            h                 = lut(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_elseif (sfpi::dst_reg[0] == 11.0F)
        {
            // Use L1
            sfpi::vFloat h  = 1.0F;
            sfpi::vUInt l2b = 0x9000;
            // Test used a bias on Grayskull, not supported on Wormhole
            h                 = lut(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_elseif (sfpi::dst_reg[0] == 12.0F)
        {
            // Use L2
            sfpi::vFloat h    = 4.0F;
            sfpi::vUInt l2b   = 0x9000;
            h                 = lut_sign(h, l0b, l1b, l2b);
            sfpi::dst_reg[11] = h;
        }
        v_endif;
    }

    // lut2 3 entry 16 bit
    {
        sfpi::vUInt l0 = (sfpi::s2vFloat16a(2.0f).get() << 16) | sfpi::s2vFloat16a(3.0f).get();
        sfpi::vUInt l1 = (sfpi::s2vFloat16a(4.0f).get() << 16) | sfpi::s2vFloat16a(5.0f).get();
        sfpi::vUInt l2 = (sfpi::s2vFloat16a(6.0f).get() << 16) | sfpi::s2vFloat16a(7.0f).get();
        v_if (sfpi::dst_reg[0] == 13.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2(h, l0, l1, l2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = -3.5 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 14.0f)
        {
            sfpi::vFloat h    = 1.25f;
            h                 = lut2(h, l0, l1, l2);
            sfpi::dst_reg[11] = h; // 1.25 * 4 + 5 = 10 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 15.0f)
        {
            sfpi::vFloat h    = -2.25f;
            h                 = lut2(h, l0, l1, l2);
            sfpi::dst_reg[11] = h; // 2.25 * 6 + 7 = 13.5 + 7 = -20.5 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 16.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2_sign(h, l0, l1, l2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = 3.5 (sign update)
        }
        v_endif;
    }

    // lut2 3 entry 32 bit
    {
        sfpi::vFloat a0 = 2.0f;
        sfpi::vFloat a1 = 4.0f;
        sfpi::vFloat a2 = 6.0f;
        sfpi::vFloat b0 = 3.0f;
        sfpi::vFloat b1 = 5.0f;
        sfpi::vFloat b2 = 7.0f;
        v_if (sfpi::dst_reg[0] == 17.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2(h, a0, a1, a2, b0, b1, b2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = -3.5 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 18.0f)
        {
            sfpi::vFloat h    = 1.25f;
            h                 = lut2(h, a0, a1, a2, b0, b1, b2);
            sfpi::dst_reg[11] = h; // 1.25 * 4 + 5 = 10 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 19.0f)
        {
            sfpi::vFloat h    = -3.0f;
            h                 = lut2(h, a0, a1, a2, b0, b1, b2);
            sfpi::dst_reg[11] = h; // 3 * 6 + 7 = 18 + 7 = -25.0 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 20.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2_sign(h, a0, a1, a2, b0, b1, b2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = 3.5 (sign update)
        }
        v_endif;
    }

    // lut2 6 entry 16 bit mode 1
    {
        sfpi::vUInt a01 = (sfpi::s2vFloat16a(4.0f).get() << 16) | sfpi::s2vFloat16a(2.0f).get();
        sfpi::vUInt a23 = (sfpi::s2vFloat16a(8.0f).get() << 16) | sfpi::s2vFloat16a(6.0f).get();
        ;
        sfpi::vUInt a34 = (sfpi::s2vFloat16a(12.0f).get() << 16) | sfpi::s2vFloat16a(10.0f).get();
        sfpi::vUInt b01 = (sfpi::s2vFloat16a(5.0f).get() << 16) | sfpi::s2vFloat16a(3.0f).get();
        sfpi::vUInt b23 = (sfpi::s2vFloat16a(9.0f).get() << 16) | sfpi::s2vFloat16a(7.0f).get();
        sfpi::vUInt b34 = (sfpi::s2vFloat16a(13.0f).get() << 16) | sfpi::s2vFloat16a(11.0f).get();
        v_if (sfpi::dst_reg[0] == 21.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = -3.5 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 22.0f)
        {
            sfpi::vFloat h    = 0.75f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // .75 * 4 + 5 = 8
        }
        v_elseif (sfpi::dst_reg[0] == 23.0f)
        {
            sfpi::vFloat h    = -1.25f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // 1.25 * 6 + 7 = 7.5 + 7 = -14.5 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 24.0f)
        {
            sfpi::vFloat h    = -1.75f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // 1.75 * 8 + 9 = 14 + 9 = -23 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 25.0f)
        {
            sfpi::vFloat h    = 2.5f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // 2.5 * 10 + 11 = 25 + 11 = 36.0
        }
        v_elseif (sfpi::dst_reg[0] == 26.0f)
        {
            sfpi::vFloat h    = 3.5f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // 3.5 * 12 + 13 = 42 + 13 = 55.0
        }
        v_elseif (sfpi::dst_reg[0] == 27.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2_sign(h, a01, a23, a34, b01, b23, b34);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = 3.5 (sign update)
        }
        v_endif;
    }

    // lut2 6 entry 16 bit mode 2
    {
        sfpi::vUInt a01 = (sfpi::s2vFloat16a(4.0f).get() << 16) | sfpi::s2vFloat16a(2.0f).get();
        sfpi::vUInt a23 = (sfpi::s2vFloat16a(8.0f).get() << 16) | sfpi::s2vFloat16a(6.0f).get();
        ;
        sfpi::vUInt a34 = (sfpi::s2vFloat16a(12.0f).get() << 16) | sfpi::s2vFloat16a(10.0f).get();
        sfpi::vUInt b01 = (sfpi::s2vFloat16a(5.0f).get() << 16) | sfpi::s2vFloat16a(3.0f).get();
        sfpi::vUInt b23 = (sfpi::s2vFloat16a(9.0f).get() << 16) | sfpi::s2vFloat16a(7.0f).get();
        sfpi::vUInt b34 = (sfpi::s2vFloat16a(13.0f).get() << 16) | sfpi::s2vFloat16a(11.0f).get();

        // Can't fit all the tests into 32 elements, skipping a few that are
        // the most redundant to prior tests here
#if 0
        v_if(sfpi::dst_reg[0] == 28.0f) {
            sfpi::vFloat h = -0.25f;
            h = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = -3.5 (sign retain)
        } v_elseif(sfpi::dst_reg[0] == 29.0f) {
            sfpi::vFloat h = 0.75f;
            h = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // .75 * 4 + 5 = 8
        } v_elseif(sfpi::dst_reg[0] == 30.0f) {
            sfpi::vFloat h = -1.25f;
            h = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // 1.25 * 6 + 7 = 7.5 + 7 = -14.5 (sign retain)
        }
#endif
        v_if (sfpi::dst_reg[0] == 28.0f)
        {
            sfpi::vFloat h    = -1.75f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // 1.75 * 8 + 9 = 14 + 9 = -23 (sign retain)
        }
        v_elseif (sfpi::dst_reg[0] == 29.0f)
        {
            sfpi::vFloat h    = 3.5f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // 3.5 * 10 + 11 = 35 + 11 = 46.0
        }
        v_elseif (sfpi::dst_reg[0] == 30.0f)
        {
            sfpi::vFloat h    = 4.5f;
            h                 = lut2(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // 4.5 * 12 + 13 = 54 + 13 = 67.0
        }
        v_elseif (sfpi::dst_reg[0] == 31.0f)
        {
            sfpi::vFloat h    = -0.25f;
            h                 = lut2_sign(h, a01, a23, a34, b01, b23, b34, 2);
            sfpi::dst_reg[11] = h; // .25 * 2 + 3 = 3.5 (sign update)
        }
        v_endif;
    }

    // [1] = 0.125
    // [2] = -0.125
    // [3] = 0.125
    // [4] = -0.125
    // [5] = 0.375
    // [6] = -0.5
    // [7] = 0.25
    // [8] = -0.25
    // [9] = 0.25
    // [10] = -0.25
    // [11] = 0.25
    // [12] = -1.0
    // [13] = -3.5
    // [14] = 10.0
    // [15] = -20.5
    // [16] = 3.5
    // [17] = -3.5
    // [18] = 10.0
    // [19] = -25.0
    // [20] = 3.5
    // [21] = -3.5
    // [22] = 8.0
    // [23] = -14.5
    // [24] = -23.0
    // [25] = 36.0
    // [26] = 55.0
    // [27] = 3.5
    // [28] = -23.0
    // [29] = 46.0
    // [30] = 67.0
    // [31] = 3.5

    copy_result_to_dreg0(11);
}

sfpi_test_noinline void test12(int imm)
{
    // imm is 35
    // Test immediate forms of SFPLOAD, SFPLOADI, SFPSTORE, SFPIADD_I, SFPADDI
    // SFPMULI, SFPSHFT, SFPDIVP2, SFPSETEXP, SFPSETMAN, SFPSETSGN,
    // Tries to cover both positive and negative params (sign extension)
    sfpi::dst_reg[12] = -sfpi::dst_reg[0];

    v_if (sfpi::dst_reg[0] == 1.0F)
    {
        sfpi::dst_reg[12] = static_cast<float>(imm); // SFPLOADI
    }
    v_elseif (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::dst_reg[12] = static_cast<float>(-imm); // SFPLOADI
    }
    v_elseif (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vInt a = 5;
        a += imm; // SFPIADD_I
        set_expected_result(12, 25.0F, 40, a);
    }
    v_elseif (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vInt a = 5;
        a -= imm; // SFPIADD_I
        set_expected_result(12, -25.0F, -30, a);
    }
    v_elseif (sfpi::dst_reg[0] == 5.0F)
    {
        sfpi::vFloat a = 3.0F;
        a += static_cast<float>(imm); // SFPADDI
        sfpi::dst_reg[12] = a;
    }
    v_elseif (sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::vFloat a = 3.0F;
        a += static_cast<float>(-imm); // SFPADDI
        sfpi::dst_reg[12] = a;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 7.0F)
    {
        sfpi::vUInt a = 0x4000;
        a >>= imm - 25;
        set_expected_result(12, 64.0F, 0x0010, reinterpret<sfpi::vInt>(a));
    }
    v_elseif (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::vUInt a = 1;
        a <<= imm - 25;
        set_expected_result(12, 128.0F, 0x0400, reinterpret<sfpi::vInt>(a));
    }
    v_elseif (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::vFloat a    = 256.0F;
        sfpi::dst_reg[12] = addexp(a, imm - 31);
    }
    v_elseif (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::vFloat a    = 256.0F;
        sfpi::dst_reg[12] = addexp(a, imm - 39);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 11.0F)
    {
        sfpi::vFloat a    = 128.0;
        sfpi::vFloat r    = sfpi::setsgn(a, imm - 36);
        sfpi::dst_reg[12] = r;
    }
    v_elseif (sfpi::dst_reg[0] == 12.0F)
    {
        sfpi::vFloat tmp  = 1024.0F;
        int man           = 0x75019a + 35 - imm;
        sfpi::vFloat tmp2 = setman(tmp, man);
        sfpi::dst_reg[12] = tmp2;
    }
    v_elseif (sfpi::dst_reg[0] == 13.0F)
    {
        int exp              = 0x007F + 35 - imm; // Exponent in low bits
        sfpi::vFloat sgn_man = -1664.0F;          // 1024 + 512 + 128 or 1101
        sgn_man              = setexp(sgn_man, exp);
        sfpi::dst_reg[12]    = sgn_man;
    }
    v_endif;

    sfpi::dst_reg[30 + 35 - imm]     = 30.0F; // SFPSTORE
    sfpi::dst_reg[30 + 35 - imm + 1] = vConstNeg1;

    v_if (sfpi::dst_reg[0] == 14.0F)
    {
        sfpi::dst_reg[12] = sfpi::dst_reg[30 + 35 - imm]; // SFPLOAD
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 15.0F)
    {
        sfpi::dst_reg[12] = sfpi::dst_reg[30 + 35 - imm + 1]; // SFPLOAD
    }
    v_endif;

    // Test for store/load nops, imm store non-imm load
    // Need to use the semaphores to get TRISC to run ahead for non-imm loads

    v_if (sfpi::dst_reg[0] == 16.0F)
    {
        // imm store, non-imm load
        sfpi::vFloat a = 120.0F;

        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_3);
        TTI_SEMWAIT(p_stall::STALL_MATH, p_stall::SEMAPHORE_3, p_stall::STALL_ON_ZERO);

        sfpi::dst_reg[12] = a;
        __builtin_rvtt_sfpnop(); // XXXXXX remove me when compiler is fixed
        a = sfpi::dst_reg[imm - 23];

        semaphore_post(3);

        sfpi::dst_reg[12] = a + 1.0F;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 17.0F)
    {
        // non-imm store, imm load
        sfpi::vFloat a          = 130.0F;
        sfpi::dst_reg[imm - 23] = a;
        __builtin_rvtt_sfpnop(); // XXXXXX remove me when compiler is fixed
        a                 = sfpi::dst_reg[12];
        sfpi::dst_reg[12] = a + 1.0F;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 18.0F)
    {
        // non-imm store, non-imm load
        sfpi::vFloat a = 140.0F;

        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_3);
        TTI_SEMWAIT(p_stall::STALL_MATH, p_stall::SEMAPHORE_3, p_stall::STALL_ON_ZERO);

        sfpi::dst_reg[imm - 23] = a;
        __builtin_rvtt_sfpnop(); // XXXXXX remove me when compiler is fixed
        a = sfpi::dst_reg[imm - 23];

        semaphore_post(3);

        sfpi::dst_reg[12] = a + 1.0F;
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 19.0F)
    {
        sfpi::vFloat a = 3.0F;
        a *= static_cast<float>(imm); // SFPADDI
        sfpi::dst_reg[12] = a;
    }
    v_elseif (sfpi::dst_reg[0] == 20.0F)
    {
        sfpi::vFloat a = 3.0F;
        a *= static_cast<float>(-imm); // SFPADDI
        sfpi::dst_reg[12] = a;
    }
    v_endif;

    // [1] = 35.0F
    // [2] = -35.0F
    // [3] = 25.0F
    // [4] = -25.0F
    // [5] = 38.0F
    // [6] = -32.0F
    // [7] = 64.0F
    // [8] = 128.0F
    // [9] = 4096.0F
    // [10] = 16.0F
    // [11] = -128.0F
    // [12] = 1960.050049
    // [13] = -1.625
    // [14] = 30.0F
    // [15] = 1.0
    // [16] = 121.0F
    // [17] = 131.0F
    // [18] = 141.0F
    // [19] = 105.0F
    // [20] = -105.0F

    copy_result_to_dreg0(12);
}

// Test 13 covers variable liveness, ie, keeping a variable "alive" across a
// CC narrowing instruction.  Touches every affected instruction except LOAD,
// LOADI, IADD (those are covered in random tests above) across a SETCC
sfpi_test_noinline void test13(int imm)
{
    // Test variable liveness

    sfpi::dst_reg[13] = -sfpi::dst_reg[0];

    // ABS liveness across SETCC
    {
        sfpi::vFloat x = -20.0F;
        sfpi::vFloat y = -30.0F;
        v_if (sfpi::dst_reg[0] == 0.0F)
        {
            y = sfpi::abs(x);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 0.0F || sfpi::dst_reg[0] == 1.0F)
        {
            sfpi::dst_reg[13] = y;
        }
        v_endif;
    }
    // [0] = 20.0F
    // [1] = -30.0F

    // NOT liveness across SETCC
    {
        sfpi::vInt a = 0xFAAA;
        sfpi::vInt b = 0x07BB;
        v_if (sfpi::dst_reg[0] == 2.0F)
        {
            b = ~a;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 2.0F || sfpi::dst_reg[0] == 3.0F)
        {
            v_if (sfpi::dst_reg[0] == 2.0F)
            {
                set_expected_result(13, 40.0F, 0xFFFF0555, b);
            }
            v_endif;
            v_if (sfpi::dst_reg[0] == 3.0F)
            {
                set_expected_result(13, 50.0F, 0x07BB, b);
            }
            v_endif;
        }
        v_endif;
    }
    // [2] = 40.0F
    // [3] = 50.0F

    // LZ liveness across SETCC
    {
        sfpi::vInt a = 0x0080;
        sfpi::vInt b = 0x07BB;
        v_if (sfpi::dst_reg[0] == 4.0F)
        {
            b = lz(a);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 4.0F || sfpi::dst_reg[0] == 5.0F)
        {
            v_if (sfpi::dst_reg[0] == 4.0F)
            {
                set_expected_result(13, 60.0F, 24, b);
            }
            v_endif;
            v_if (sfpi::dst_reg[0] == 5.0F)
            {
                set_expected_result(13, 70.0F, 0x07BB, b);
            }
            v_endif;
        }
        v_endif;
    }
    // [4] = 60.0F
    // [5] = 70.0F

    // MAD liveness across SETCC
    {
        sfpi::vFloat a = 90.0F;
        sfpi::vFloat b = 110.0F;
        v_if (sfpi::dst_reg[0] == 6.0F)
        {
            b = a * a + 10.0;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 6.0F || sfpi::dst_reg[0] == 7.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [6] = 8110.0F
    // [7] = 110.0F

    // MOV liveness across SETCC
    {
        sfpi::vFloat a = 120.0F;
        sfpi::vFloat b = 130.0F;
        v_if (sfpi::dst_reg[0] == 8.0F)
        {
            b = -a;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 8.0F || sfpi::dst_reg[0] == 9.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [8] = -120.0F
    // [9] = 130.0F;

    // DIVP2 liveness across SETCC
    {
        sfpi::vFloat a = 140.0F;
        sfpi::vFloat b = 150.0F;
        v_if (sfpi::dst_reg[0] == 10.0F)
        {
            b = addexp(a, 1);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 10.0F || sfpi::dst_reg[0] == 11.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [10] = 280.0F
    // [11] = 150.0F

    // EXEXP liveness across SETCC
    {
        sfpi::vFloat a = 160.0F;
        sfpi::vInt b   = 128;
        v_if (sfpi::dst_reg[0] == 12.0F)
        {
            b = exexp_nodebias(a);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 12.0F || sfpi::dst_reg[0] == 13.0F)
        {
            sfpi::vFloat tmp  = 1.0F;
            sfpi::dst_reg[13] = setexp(tmp, b);
        }
        v_endif;
    }
    // [12] = 128.0F
    // [13] = 2.0F

    // EXMAN liveness across SETCC
    {
        sfpi::vFloat a = 160.0F;
        sfpi::vInt b   = 0x80000;
        v_if (sfpi::dst_reg[0] == 14.0F)
        {
            b = exman8(a);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 14.0F || sfpi::dst_reg[0] == 15.0F)
        {
            sfpi::vFloat tmp  = 128.0F;
            sfpi::dst_reg[13] = setman(tmp, b);
        }
        v_endif;
    }
    // [14] = 160.0F
    // [15] = 136.0F

    // SETEXP_I liveness across SETCC
    {
        sfpi::vFloat a = 170.0F;
        sfpi::vFloat b = 180.0F;
        v_if (sfpi::dst_reg[0] == 16.0F)
        {
            b = setexp(a, 132);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 16.0F || sfpi::dst_reg[0] == 17.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [16] = 42.5F
    // [17] = 180.0F

    // SETMAN_I liveness across SETCC
    {
        sfpi::vFloat a = 190.0F;
        sfpi::vFloat b = 200.0F;
        v_if (sfpi::dst_reg[0] == 18.0F)
        {
            b = setman(a, 0x75019a);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 18.0F || sfpi::dst_reg[0] == 19.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [18] = 245.06256F
    // [19] = 200.0F

    // SETSGN_I liveness across SETCC
    {
        sfpi::vFloat a = 210.0F;
        sfpi::vFloat b = 220.0F;
        v_if (sfpi::dst_reg[0] == 20.0F)
        {
            b = sfpi::setsgn(a, 1);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 20.0F || sfpi::dst_reg[0] == 21.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [20] = -210.0F
    // [21] = 220.0F

    // nonimm_dst_src using DIVP2 liveness across SETCC
    {
        sfpi::vFloat a = 140.0F;
        sfpi::vFloat b = 150.0F;
        v_if (sfpi::dst_reg[0] == 22.0F)
        {
            b = addexp(a, imm - 34);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 22.0F || sfpi::dst_reg[0] == 23.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [22] = 280.0F
    // [23] = 150.0F

    // nonimm_dst using LOADI liveness across SETCC
    {
        sfpi::vFloat b = 240.0F;
        v_if (sfpi::dst_reg[0] == 24.0F)
        {
            b = static_cast<float>(-imm);
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 24.0F || sfpi::dst_reg[0] == 25.0F)
        {
            sfpi::dst_reg[13] = b;
        }
        v_endif;
    }
    // [24] = -35.0F
    // [25] = 240.0F

    copy_result_to_dreg0(13);
}

sfpi_test_noinline void test14(int imm)
{
    // Test13 tests various builtins for liveness across a SETCC
    // Below test MOV liveness across COMPC, LZ, EXEXP, IADD

    sfpi::dst_reg[14] = -sfpi::dst_reg[0];

    // MOV liveness across COMPC
    {
        sfpi::vFloat a = 250.0F;
        sfpi::vFloat b = 260.0F;
        v_if (sfpi::dst_reg[0] != 0.0F)
        {
            b = 160.0F;
        }
        v_else
        {
            sfpi::vFloat c = vConst0 * vConst0 + vConst0;
            b              = -a;
            a              = c;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 0.0F || sfpi::dst_reg[0] == 1.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [0] = -250.0F
    // [1] = 160.0F;

    // MOV liveness across LZ
    {
        sfpi::vFloat a = 250.0F;
        sfpi::vFloat b = 260.0F;
        sfpi::vInt tmp;

        v_if (sfpi::dst_reg[0] == 2.0F)
        {
            v_if ((tmp = lz(a)) != 0)
            {
                sfpi::vFloat c = vConst0 * vConst0 + vConst0;
                b              = -a;
                a              = c;
            }
            v_endif;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 2.0F || sfpi::dst_reg[0] == 3.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [2] = -250.0F;
    // [3] = 260.0F

    // MOV liveness across EXEXP
    {
        sfpi::vFloat a = 270.0F;
        sfpi::vFloat b = 280.0F;
        sfpi::vInt tmp;

        v_if (sfpi::dst_reg[0] == 4.0F)
        {
            v_if ((tmp = exexp(a)) >= 0)
            {
                sfpi::vFloat c = vConst0 * vConst0 + vConst0;
                b              = -a;
                a              = c;
            }
            v_endif;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 4.0F || sfpi::dst_reg[0] == 5.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [4] = -270.0F;
    // [5] = 280.0F

    // Below 2 tests are incidentally covered by tests 1..12
    // MOV liveness across IADD
    {
        sfpi::vFloat b = 300.0F;
        sfpi::vInt tmp = 5;

        v_if (sfpi::dst_reg[0] == 6.0F)
        {
            sfpi::vFloat a = 290.0F;
            v_if (tmp >= 2)
            {
                sfpi::vFloat c = vConst0 * vConst0 + vConst0;
                b              = -a;
                a              = c;
            }
            v_endif;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 6.0F || sfpi::dst_reg[0] == 7.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [6] = -290.0F
    // [7] = 300.0F

    // IADD_I liveness
    {
        sfpi::vInt a = 10;
        sfpi::vInt b = 20;
        v_if (sfpi::dst_reg[0] == 8.0F)
        {
            b = a + 30;
        }
        v_endif;
        v_if (sfpi::dst_reg[0] == 8.0F || sfpi::dst_reg[0] == 9.0F)
        {
            v_if (sfpi::dst_reg[0] == 8.0F)
            {
                set_expected_result(14, -310.0F, 40, b);
            }
            v_endif;
            v_if (sfpi::dst_reg[0] == 9.0F)
            {
                set_expected_result(14, 320.0F, 20, b);
            }
            v_endif;
        }
        v_endif;
    }
    // [8] = -310.0F
    // [9] = 320.0F

    // Test various issues with move/assign. Unfortunately, compiler generated
    // moves are hard/impossible to induce and not all scenarios are testable
    // w/ explicit code afaict.  The case #s below come from the Predicated
    // Variable Liveness document and similar code exists in live.cc

    // Case 2a
    // Assignment resulting in register rename
    {
        sfpi::vFloat a = -20.0f;
        sfpi::vFloat b = 30.0f;
        v_if (sfpi::dst_reg[0] == 10.0f)
        {
            b = a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 10.0F || sfpi::dst_reg[0] == 11.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [10] = -20.0
    // [11] = 30.0

    // Case 2b
    // Assignment requiring move
    // This straddles case 2a and 3 - both values need to be preserved but the
    // compiler doesn't know that, solving case2a will solve this case as well
    {
        sfpi::vFloat a = -40.0f;
        sfpi::vFloat b = 50.0f;
        v_if (sfpi::dst_reg[0] == 12.0f)
        {
            b = a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 100.0f)
        { // always fail
            sfpi::dst_reg[14] = a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 12.0F || sfpi::dst_reg[0] == 13.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;
    }
    // [12] = -40.0
    // [13] = 50.0

    // Case 3
    // Assignment requiring move (both a and b need to be preserved)
    {
        sfpi::vFloat a = -60.0f;
        sfpi::vFloat b = 70.0f;
        v_if (sfpi::dst_reg[0] == 14.0f)
        {
            b = a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 100.0f)
        { // always fail
            sfpi::dst_reg[14] = a + 1.0f;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 14.0F || sfpi::dst_reg[0] == 15.0F)
        {
            sfpi::dst_reg[14] = b + 1.0f;
        }
        v_endif;
    }
    // [14] = -59.0
    // [15] = 71.0

    // Case 4a
    // Destination as source, 2 arguments in the wrong order
    // Confirm b is correct
    {
        sfpi::vInt a = 10;
        sfpi::vInt b = 20;
        v_if (sfpi::dst_reg[0] == 16.0f)
        {
            b = b - a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 100.0f)
        { // always fail
            sfpi::dst_reg[14] = a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 16.0F)
        {
            set_expected_result(14, -80.0F, 10, b);
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 17.0F)
        {
            set_expected_result(14, 90.0F, 20, b);
        }
        v_endif;
    }
    // [16] = -80.0
    // [17] = 90.0

    // Case 4b
    // Destination as source, 2 arguments in the wrong order
    // Confirm a is correct
    {
        sfpi::vInt a = 10;
        sfpi::vInt b = 20;
        v_if (sfpi::dst_reg[0] == 16.0f)
        {
            b = b - a;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 100.0f)
        { // always fail
            sfpi::dst_reg[14] = b;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 18.0F)
        {
            set_expected_result(14, -90.0F, 10, a);
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 19.0F)
        {
            set_expected_result(14, 100.0F, 10, a);
        }
        v_endif;
    }
    // [18] = -90.0
    // [19] = 100.0

    // Case 4c
    // Destination as source 3 arguments
    // Confirm c is correct
    {
        // Out of regs doing this the typical way
        sfpi::vFloat condition = sfpi::dst_reg[0] - 20.0F;
        sfpi::vInt a           = 10;
        sfpi::vInt b           = 20;
        sfpi::vInt c           = 30;

        v_if (condition == 0.0F)
        {
            c = a - b;
        }
        v_endif;

        v_if (vConst0p8373 == sfpi::dst_reg[0])
        { // always fail
            sfpi::dst_reg[14] = a;
            sfpi::dst_reg[14] = b;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 20.0F)
        {
            set_expected_result(14, -100.0F, -10, c);
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 21.0F)
        {
            set_expected_result(14, 110.0F, 30, c);
        }
        v_endif;
    }
    // [20] = -100.0
    // [21] = 110.0

    // Case 4c
    // Destination as source 3 arguments
    // Confirm a is correct
    {
        // Out of regs doing this the typical way
        sfpi::vFloat condition = sfpi::dst_reg[0] - 22.0F;
        sfpi::vInt a           = 10;
        sfpi::vInt b           = 20;
        sfpi::vInt c           = 30;

        v_if (condition == 0.0F)
        {
            c = a - b;
        }
        v_endif;

        v_if (vConst0p8373 == sfpi::dst_reg[0])
        { // always fail
            sfpi::dst_reg[14] = a;
            sfpi::dst_reg[14] = c;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 22.0F)
        {
            set_expected_result(14, -110.0F, 10, a);
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 23.0F)
        {
            set_expected_result(14, 120.0F, 10, a);
        }
        v_endif;
    }
    // [22] = -110.0
    // [23] = 120.0

    // Case 4c
    // Destination as source 3 arguments
    // Confirm b is correct
    {
        // Out of regs doing this the typical way
        sfpi::vFloat condition = sfpi::dst_reg[0] - 24.0F;
        sfpi::vInt a           = 10;
        sfpi::vInt b           = 20;
        sfpi::vInt c           = 30;

        v_if (condition == 0.0F)
        {
            c = a - b;
        }
        v_endif;

        v_if (vConst0p8373 == sfpi::dst_reg[0])
        { // always fail
            sfpi::dst_reg[14] = c;
            sfpi::dst_reg[14] = b;
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 24.0F)
        {
            set_expected_result(14, -120.0F, 20, b);
        }
        v_endif;

        v_if (sfpi::dst_reg[0] == 25.0F)
        {
            set_expected_result(14, 130.0F, 20, b);
        }
        v_endif;
    }
    // [24] = -120.0
    // [25] = 130.0

    // The code below tests the case where we descend down a CC cascade, pop
    // back up, then back down w/ different CC bits set.  Does the variable
    // stay live when assigned at the same CC level but in a different
    // cascade, ie, across generations?
    {
        sfpi::vFloat a;
        sfpi::vFloat b;
        sfpi::vFloat dr = sfpi::dst_reg[0];

        v_if (dr == 26.0F || dr == 27.0F)
        {
            b = -90.0F;
        }
        v_endif;

        v_if (dr == 26.0F)
        {
            a = 100.0F;
        }
        v_endif;

        v_if (dr == 27.0F)
        {
            a = 110.0F;
        }
        v_endif;

        v_if (dr == 27.0F)
        {
            b = a;
        }
        v_endif;

        v_if (dr == 26.0F || dr == 27.0F)
        {
            sfpi::dst_reg[14] = b;
        }
        v_endif;

        v_if (dr == 500.0F)
        {
            sfpi::dst_reg[14] = a;
        }
        v_endif;
    }
    // [26] = -90.0F
    // [27] = 110.0F;

    // Test a little basic block liveness madness
    // NOTE: the test below hit a riscv gcc compiler bug where the float
    // library conversions were wrong where:
    //    (30.0f - i) != static_cast<float>(30 - i)
    // and not just due to rounding (off by orders of magnitude)
    {
        sfpi::vFloat a = 200.0F;
        sfpi::vFloat b = 1.0F;

        // unroll forces the compiler into multiple basic blocks
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 0
#endif
        for (int i = 0; i < imm - 30; i++)
        { // 0..4
            v_if (sfpi::dst_reg[0] == 28.0F)
            {
                switch (i)
                {
                    case 0:
                        b = 2.0f;
                        break;
                    case 1:
                        b = 4.0f;
                        break;
                    case 2:
                        b = 8.0f;
                        break;
                    default:
                        b = b * 4.0F;
                }
            }
            v_elseif (sfpi::dst_reg[0] >= static_cast<float>(30 - i))
            {
                if (i % 2 == 0)
                {
                    b = 10.0F;
                }
                else
                {
                    b = 20.0F;
                }
            }
            v_endif;

            a = a + a * b;
        }

        v_if (sfpi::dst_reg[0] == 28.0F || sfpi::dst_reg[0] == 29.0F)
        {
            sfpi::dst_reg[14] = a;
        }
        v_endif;
    }
    // [28] = 200+200*2, 600+600*4, 3000+3000*8, 27000+27000*32, 89100+89100*128 =
    //        114939000.0F
    // [29] = 200+200*1, 400+400*20, 4400+4400*20, 92400+92400*10, 1016400+1016400*20 =
    //        21344400.0F

    copy_result_to_dreg0(14);
}

sfpi_test_noinline void test15()
{
    // SFPTRANSP, SFPSHFT2

    sfpi::dst_reg[15] = -sfpi::dst_reg[0];
    {
        sfpi::vUInt a = vConstTileId + 0x100;
        sfpi::vUInt b = vConstTileId + 0x200;
        sfpi::vUInt c = vConstTileId + 0x300;
        sfpi::vUInt d = vConstTileId + 0x400;

        subvec_transp(a, b, c, d);

        sfpi::vUInt base = vConstTileId >> 4;
        base <<= 8;
        base += 0x100;

        // Load expected value, subtract actual value. result is 0 if correct
        sfpi::vUInt eff  = 0xF;
        sfpi::vUInt cmpa = base | (vConstTileId & eff);
        cmpa -= a;
        sfpi::vUInt cmpb = base | ((vConstTileId & eff) + 0x10);
        cmpb -= b;
        sfpi::vUInt cmpc = base | ((vConstTileId & eff) + 0x20);
        cmpc -= c;
        sfpi::vUInt cmpd = base | ((vConstTileId & eff) + 0x30);
        cmpd -= d;

        // The above completes this test, now to make the results reportable
        // in less than 4 full width vectors

        // Reduce across a, b, c, d
        sfpi::vUInt result = reduce_bool4(cmpa, cmpb, cmpc, cmpd, 0);

        // We care about xyz
        // Use the thing we're testing to test the result by putting xyz result
        // 4 8-wide subvectors in 4 variables
        subvec_transp(result, cmpb, cmpc, cmpd);

        // Reduce result (only care about first subbvec, rest along for the ride)
        sfpi::vUInt final = reduce_bool4(result, cmpb, cmpc, cmpd, 1);

        v_if (sfpi::dst_reg[0] < 8.0F)
        {
            set_expected_result(15, 8.0F, 1, final);
        }
        v_endif;
    }

    {
        // subvec_shflror1
        sfpi::vUInt src = vConstTileId;
        sfpi::vUInt dst = subvec_shflror1(src);

        sfpi::vUInt cmpdst = vConstTileId - 2;
        // first element in the subvec
        v_if ((vConstTileId & 0xF) == 0)
        {
            cmpdst += 0x10;
        }
        v_endif;
        dst -= cmpdst;

        sfpi::vUInt tmp1 = 1;
        sfpi::vUInt tmp2 = 1;
        sfpi::vUInt tmp3 = 1;
        subvec_transp(tmp1, dst, tmp2, tmp3);

        sfpi::vUInt final = reduce_bool4(dst, tmp1, tmp2, tmp3, 0);
        v_if (sfpi::dst_reg[0] >= 8.0F && sfpi::dst_reg[0] < 16.0F)
        {
            set_expected_result(15, 16.0F, 1, final);
        }
        v_endif;
    }

    {
        // subvec_shflshr1
        sfpi::vUInt src = vConstTileId;
        sfpi::vUInt dst = subvec_shflshr1(src);

        sfpi::vUInt cmpdst = vConstTileId - 2;
        // first element in the subvec
        v_if ((vConstTileId & 0xF) == 0)
        {
            cmpdst = 0;
        }
        v_endif;
        dst -= cmpdst;

        sfpi::vUInt tmp1 = 1;
        sfpi::vUInt tmp2 = 1;
        sfpi::vUInt tmp3 = 1;
        subvec_transp(tmp1, tmp2, dst, tmp3);

        sfpi::vUInt final = reduce_bool4(tmp1, dst, tmp2, tmp3, 0);
        v_if (sfpi::dst_reg[0] >= 16.0F && sfpi::dst_reg[0] < 24.0F)
        {
            set_expected_result(15, 24.0F, 1, final);
        }
        v_endif;
    }
#if 0
    // Decided not to implement these at this time.  These insns are only
    // interesting if/when we implement LOADMACRO
    v_if (sfpi::dst_reg[0] == 16.0F) {
        // Wrapper doesn't emit shft2 bit shift, test directly
        sfpi::vUInt a = 0x005A;

        a.get() = __builtin_rvtt_sfpshft2_i(a.get(), 4);
        set_expected_result(16, 10.0F, 0x05A0, a);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 17.0F) {
        // Wrapper doesn't emit shft2 bit shift, test directly
        sfpi::vUInt a = 0x005A;
        sfpi::vUInt b = 4;

        a.get() = __builtin_rvtt_sfpshft2_v(a.get(), b.get());
        set_expected_result(16, 20.0F, 0x05A0, a);
    }
    v_endif;
#endif

    // [0..31] = 1
    copy_result_to_dreg0(15);
}

void test16()
{
    // SFPSWAP, SFPCAST, SFPSTOCHRND
    sfpi::dst_reg[16] = -sfpi::dst_reg[0];

    // Tests are all 2 results per row, allowing 4 independent tests only 2 of
    // which are used
    sfpi::vFloat x = 2.0f;
    sfpi::vFloat y = 3.0f;

    v_if (sfpi::dst_reg[0] < 8.0F)
    {
        vec_swap(x, y);
        v_if (sfpi::dst_reg[0] >= 4.0f)
        {
            vec_min_max(x, y);
        }
        v_endif;

        v_if (((vConstTileId >> 1) & 1) == 0)
        {
            sfpi::dst_reg[16] = x;
        }
        v_else
        {
            sfpi::dst_reg[16] = y;
        }
        v_endif;
    }
    v_endif;
    // [0] = 3.0
    // [1] = 2.0
    // [2] = 3.0
    // [3] = 2.0
    // [4] = 2.0
    // [5] = 3.0
    // [6] = 2.0
    // [7] = 3.0

    // These are really crappy "touch" tests
    v_if (sfpi::dst_reg[0] == 8.0F)
    {
        sfpi::dst_reg[16] = int32_to_float(0xABBAAB);
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 9.0F)
    {
        sfpi::dst_reg[16] = int32_to_float(0xABBAAB, 0);
    }
    v_endif;

    v_if (sfpi::dst_reg[0] == 10.0F)
    {
        sfpi::dst_reg[16] = float_to_fp16a(1.32332);
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 11.0F)
    {
        sfpi::dst_reg[16] = float_to_fp16b(1.32332);
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 12.0F)
    {
        set_expected_result(16, 48.0f, 24, float_to_uint8(23.3));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 13.0F)
    {
        set_expected_result(16, 64.0f, 24, float_to_int8(23.3));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 14.0F)
    {
        sfpi::vUInt descale = 8;
        set_expected_result(16, 80.0f, 0xeb, int32_to_uint8(0xea00, descale));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 15.0F)
    {
        set_expected_result(16, 96.0f, 0xf, int32_to_uint8(0xea0, 8));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 16.0F)
    {
        sfpi::vUInt descale = 8;
        set_expected_result(16, 112.0f, 0xf, int32_to_int8(0xea0, descale));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 17.0F)
    {
        set_expected_result(16, 128.0f, 0xf, int32_to_int8(0xea0, 8));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 18.0F)
    {
        set_expected_result(16, 130.0f, 0x7eb1, float_to_int16(32432.0f));
    }
    v_endif;
    v_if (sfpi::dst_reg[0] == 19.0F)
    {
        set_expected_result(16, 132.0f, 0x7eb1, float_to_uint16(32432.0f));
    }
    v_endif;

    copy_result_to_dreg0(16);
}

void test17()
{
    // more SFPSWAP
    sfpi::dst_reg[17] = -sfpi::dst_reg[0];

    // Test sign-magnitude for ints
    v_if (sfpi::dst_reg[0] == 2.0F)
    {
        sfpi::vUInt x = -1;
        sfpi::vUInt y = -2;
        vec_min_max(x, y);
        set_expected_result(17, 23.0f, -1, x);
    }
    v_endif;
    // [2] = 23.0f

    v_if (sfpi::dst_reg[0] == 3.0F)
    {
        sfpi::vFloat x = -1.0F;
        sfpi::vFloat y = -2.0F;
        vec_min_max(x, y);
        sfpi::dst_reg[17] = x;
    }
    v_endif;
    // [3] = -2.0

    v_if (sfpi::dst_reg[0] == 4.0F)
    {
        sfpi::vFloat x = 1.0F;
        sfpi::vFloat y = 2.0F;
        vec_min_max(x, y);
        sfpi::dst_reg[17] = x;
    }
    v_endif;
    // [4] = 1.0

    v_if (sfpi::dst_reg[0] == 5.0F || sfpi::dst_reg[0] == 6.0F)
    {
        sfpi::vFloat x = -1.0F;
        sfpi::vFloat y = 1.0F;

        v_if (sfpi::dst_reg[0] == 5.0F)
        {
            set_expected_result(17, 20.0F, 2, lz_nosgn(x));
        }
        v_else
        {
            set_expected_result(17, 20.0F, 2, lz_nosgn(y));
        }
        v_endif;
    }
    v_endif;
    // [5] = 20.0F
    // [6] = 20.0F

    copy_result_to_dreg0(17);
}

//////////////////////////////////////////////////////////////////////////////
// These tests are designed to be incremental so that if a test fails the
// earlier tests should be examined/fixed prior to the latter tests.
//
template <SfpiTestType operation>
inline void calculate_sfpi(uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0)
{
    if constexpr (operation == SfpiTestType::test1)
    {
        test1();
    }
    else if constexpr (operation == SfpiTestType::test2)
    {
        test2();
    }
    else if constexpr (operation == SfpiTestType::test3)
    {
        test3();
    }
    else if constexpr (operation == SfpiTestType::test4)
    {
        test4();
    }
    else if constexpr (operation == SfpiTestType::test5)
    {
        test5();
    }
    else if constexpr (operation == SfpiTestType::test6)
    {
        test6();
    }
    else if constexpr (operation == SfpiTestType::test7)
    {
        test7();
    }
    else if constexpr (operation == SfpiTestType::test8)
    {
        test8();
    }
    else if constexpr (operation == SfpiTestType::test9)
    {
        test9();
    }
    else if constexpr (operation == SfpiTestType::test10)
    {
        test10();
    }
    else if constexpr (operation == SfpiTestType::test11)
    {
        test11();
    }
    else if constexpr (operation == SfpiTestType::test12)
    {
        test12(param0);
    }
    else if constexpr (operation == SfpiTestType::test13)
    {
        test13(param0);
    }
    else if constexpr (operation == SfpiTestType::test14)
    {
        test14(param0);
    }
    else if constexpr (operation == SfpiTestType::test15)
    {
        test15();
    }
    else if constexpr (operation == SfpiTestType::test16)
    {
        test16();
    }
    else if constexpr (operation == SfpiTestType::test17)
    {
        test17();
    }
}

} // namespace sfpi_test
