// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "internal/debug/fw_debug.h"
#include "llk_param_structs.h"

#ifdef UCK_CHLKC_MATH
// clang-format off
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math_fidelity.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_math.cpp"
// clang-format on
#endif

#ifdef UCK_CHLKC_PACK
// clang-format off
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_pack_data_format.h"
#include "chlkc_pack_tile_dims.h"
#include "chlkc_pack.cpp"
// clang-format on
#endif

#ifdef UCK_CHLKC_UNPACK
// clang-format off
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_unpack.cpp"
// clang-format on
#endif

std::uint32_t run_kernel() {
#ifdef UCK_CHLKC_MATH
    ckernel::zeroacc();
    chlkc_math::math_main();
#endif

#ifdef UCK_CHLKC_PACK
    chlkc_pack::pack_main();
#endif

#ifdef UCK_CHLKC_UNPACK
    ckernel::zerosrc();
    chlkc_unpack::unpack_main();
#endif

    return 0;
}
