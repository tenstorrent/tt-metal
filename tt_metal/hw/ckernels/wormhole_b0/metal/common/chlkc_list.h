// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "debug/fw_debug.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "llk_param_structs.h"

using namespace ckernel;

#ifdef UCK_CHLKC_MATH
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_math_fidelity.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_math.cpp"
#endif

#ifdef UCK_CHLKC_PACK
#include "chlkc_pack_data_format.h"
#include "chlkc_pack_tile_dims.h"
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_pack.cpp"
#endif

#ifdef UCK_CHLKC_UNPACK
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_dst_accum_mode.h"
#include "chlkc_dst_sync_mode.h"
#include "chlkc_unpack.cpp"
#endif

uint run_kernel() {
#ifdef UCK_CHLKC_MATH
    zeroacc();
    chlkc_math::math_main();
#endif

#ifdef UCK_CHLKC_PACK
    chlkc_pack::pack_main();
#endif

#ifdef UCK_CHLKC_UNPACK
    zerosrc();
    chlkc_unpack::unpack_main();
#endif

    return 0;
}
