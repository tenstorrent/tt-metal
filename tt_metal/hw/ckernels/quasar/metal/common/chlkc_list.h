// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "internal/debug/fw_debug.h"

using namespace ckernel;

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

#ifdef UCK_CHLKC_ISOLATE_SFPU
#include "chlkc_descriptors.h"
#include "chlkc_isolate_sfpu.cpp"
#endif

std::uint32_t run_kernel() {
#ifdef UCK_CHLKC_MATH
    if constexpr (!UnpackToDestEn) {
        ckernel::zeroacc();
    }
#endif

#ifdef UCK_CHLKC_UNPACK
    // With unpack-to-dest the unpack thread is DEST's first writer, so it clears DEST itself: ZEROACC acts at
    // instruction issue (assembly.yaml), ahead of this thread's UNPACRs; issued from math it would race them.
    if constexpr (UnpackToDestEn) {
        ckernel::zeroacc();
    }
    ckernel::zerosrc();
#endif

    kernel_main();

    return 0;
}
