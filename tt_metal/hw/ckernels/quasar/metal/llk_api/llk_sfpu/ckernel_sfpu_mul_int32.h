// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpu/ckernel_sfpu_mul_int32.h"

namespace ckernel {
namespace sfpu {

// Op class for an elementwise integer multiply of two tiles. Same name and leading template parameters as on
// Wormhole/Blackhole; the kernel lives in tt-llk (_mul_int32_) and Quasar supports Int32 only. Int8 copy_tile +
// fp32_dest_acc FPU writes sign-magnitude Int32 into Dest, so the kernel runs with SIGN_MAGNITUDE_FORMAT=true, as
// the compute API did.
template <
    bool APPROXIMATION_MODE,
    DataFormat data_format,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct MulInt : SfpuBinaryOp<MulInt<APPROXIMATION_MODE, data_format, ITERATIONS, SLOT>, SLOT> {
    static_assert(data_format == DataFormat::Int32, "Unsupported data format for mul_int on Quasar. Supported: Int32");
    static constexpr auto& calculate =
        _mul_int32_<APPROXIMATION_MODE, ITERATIONS, true /* SIGN_MAGNITUDE_FORMAT */, SLOT>;
};

}  // namespace sfpu
}  // namespace ckernel
