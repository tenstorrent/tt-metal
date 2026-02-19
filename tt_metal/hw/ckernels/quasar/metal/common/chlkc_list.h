// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "internal/debug/fw_debug.h"

#ifdef UCK_CHLKC_MATH
#include "chlkc_descriptors.h"
#include "chlkc_math.cpp"
#endif

#ifdef UCK_CHLKC_PACK
#include "chlkc_descriptors.h"
#include "chlkc_pack.cpp"
#endif

#ifdef UCK_CHLKC_UNPACK
#include "chlkc_descriptors.h"
#include "chlkc_unpack.cpp"
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
