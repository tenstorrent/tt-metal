// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Build-only: compile-time checks on data_format_derive.h (the Blackhole llk_api header the 2.0
// compute API uses to derive register formats from L1 formats). Every assertion here is a
// static_assert, so the only observable is whether this driver builds; test_data_format_derive.py
// builds it and never runs it.
//
// The two-operand exponent-family rule must match the host rule it mirrors
// (jit_build/data_format.cpp, check_consistent_format_across_buffers / get_data_exp_precision):
// Float32 and the integer formats the host skips (UInt32, UInt16, UInt8, Int32) carry no
// exponent family of their own, adopt the other operand's family, and default to the 8-bit "b"
// family when neither operand has one. Two formats are deliberately NOT wildcards: Int8 (not on
// the host's skip list) and Fp8_e4m3 (the host skips it, but its source-register datum is Float16,
// an "a" family operand -- see the header).

#include "data_format_derive.h"

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
using ckernel::common_exp_is_b;
using ckernel::infer_unpack_dst_format_2op;

// A Float32 operand next to an integer operand: the host derives the B family (nothing float is
// left to choose from), so without fp32 dest accumulation the Float32 operand unpacks to Float16_b,
// not the 5-bit-exponent Float16 that silently saturates |x| > 65504.
static_assert(
    infer_unpack_dst_format_2op<DataFormat::Float32, DataFormat::Int32>(false) == DataFormat::Float16_b,
    "Float32 paired with Int32 must unpack to Float16_b (host picks the B family when no float operand remains)");
static_assert(
    infer_unpack_dst_format_2op<DataFormat::Float32, DataFormat::UInt16>(false) == DataFormat::Float16_b,
    "Float32 paired with UInt16 must unpack to Float16_b");
static_assert(
    infer_unpack_dst_format_2op<DataFormat::Float32, DataFormat::Int32>(true) == DataFormat::Float32,
    "fp32 dest accumulation keeps Float32 in the source register");

// An integer operand next to a b-family float: the host accepts the pairing (integers are skipped
// when the family is chosen), so the device must not reject it as mixed exponent widths.
static_assert(common_exp_is_b<DataFormat::UInt16, DataFormat::Float16_b>(), "UInt16 x Float16_b is a b-family pairing");
static_assert(common_exp_is_b<DataFormat::Int32, DataFormat::Bfp8_b>(), "Int32 x Bfp8_b is a b-family pairing");
static_assert(common_exp_is_b<DataFormat::Tf32, DataFormat::UInt8>(), "Tf32 x UInt8 is a b-family pairing");
static_assert(common_exp_is_b<DataFormat::Int32, DataFormat::UInt32>(), "two integer operands default to the B family");

// Unchanged behaviour, pinned: an a-family float still wins over a wildcard, and a non-Float32
// self operand keeps its own register format.
static_assert(!common_exp_is_b<DataFormat::Float32, DataFormat::Float16>(), "Float32 x Float16 stays in the a family");
static_assert(!common_exp_is_b<DataFormat::UInt16, DataFormat::Bfp8>(), "UInt16 x Bfp8 stays in the a family");
static_assert(
    infer_unpack_dst_format_2op<DataFormat::Float32, DataFormat::Float16>(false) == DataFormat::Float16,
    "Float32 next to an a-family float still unpacks to Float16");
static_assert(infer_unpack_dst_format_2op<DataFormat::Int32, DataFormat::Float32>(false) == DataFormat::Int32, "an Int32 self operand keeps Int32");
static_assert(common_exp_is_b<DataFormat::Float16_b, DataFormat::Float16_b>(), "matching b-family operands");

// Deliberately NOT wildcards. Each is paired with Float32 so the check discriminates: a wildcard partner of
// Float32 makes both operands wild and the family "b", so the assertion would fail.
static_assert(!common_exp_is_b<DataFormat::Int8, DataFormat::Float32>(), "Int8 is not a wildcard: it is an a-family operand, as on the host");
static_assert(
    !common_exp_is_b<DataFormat::Float32, DataFormat::Fp8_e4m3>(), "Fp8_e4m3 is not a wildcard: its source-register datum is Float16, an a-family operand");
static_assert(
    infer_unpack_dst_format_2op<DataFormat::Float32, DataFormat::Fp8_e4m3>(false) == DataFormat::Float16,
    "Float32 next to Fp8_e4m3 unpacks to Float16 so both source registers hold fp16 datums");
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
