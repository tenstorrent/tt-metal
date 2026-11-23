// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "llk_assert.h"
#include "llk_sfpu/ckernel_sfpu_srcs.h"
#include "sfpi.h"
#include "sfpu_impl.h"

// Test-only: the harness knows the SrcS format at runtime, a kernel knows it at compile time and names
// srcs_op_t<OP, LAYOUT> directly. Not part of any production SFPU API.
namespace ckernel::sfpu
{

/**
 * @brief Call fn(op) with the SrcS op type of the op tag @p OP for the LAYOUT matching the runtime SrcS format,
 *        once outside the tile loop.
 *
 * Only the layouts OP takes are compiled in (@ref op_takes_layout). The SrcS register format is the
 * unpack destination and the pack source; both are passed so a mismatch is caught here.
 *
 * @tparam OP: Op tag (ckernel::op).
 * @tparam IMPL: Sfpi or LoadMacro, for every layout.
 * @param unpack_S_dst_format: SrcS format the unpacker writes.
 * @param pack_S_src_format: SrcS format the packer reads; must equal unpack_S_dst_format.
 * @param fn: Callable taking the op type by value; use decltype(op)::init / run.
 */
template <typename OP, SfpuImpl IMPL = SfpuImpl::Sfpi, typename Fn>
sfpi_inline void dispatch_sfpu_srcs_op(const DataFormat unpack_S_dst_format, const DataFormat pack_S_src_format, Fn&& fn)
{
    LLK_ASSERT(unpack_S_dst_format == pack_S_src_format, "SrcS ops require matching unpack destination and pack source formats");
    bool handled          = false;
    const auto try_layout = [&](auto op)
    {
        using Op = decltype(op);
        if (unpack_S_dst_format == Op::Layout::format)
        {
            fn(op);
            handled = true;
        }
    };
    if constexpr (op_takes_layout<OP, sfpi::DataLayout::F16a>)
    {
        try_layout(srcs_op_t<OP, sfpi::DataLayout::F16a, IMPL> {});
    }
    if constexpr (op_takes_layout<OP, sfpi::DataLayout::F16b>)
    {
        try_layout(srcs_op_t<OP, sfpi::DataLayout::F16b, IMPL> {});
    }
    if constexpr (op_takes_layout<OP, sfpi::DataLayout::F32>)
    {
        try_layout(srcs_op_t<OP, sfpi::DataLayout::F32, IMPL> {});
    }
    if constexpr (op_takes_layout<OP, sfpi::DataLayout::I32>)
    {
        try_layout(srcs_op_t<OP, sfpi::DataLayout::I32, IMPL> {});
    }
    LLK_ASSERT(handled, "SrcS format not supported by this op");
}

} // namespace ckernel::sfpu
