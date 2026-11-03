// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chlkc_list_prefix.h"

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

uint run_kernel() {
#ifdef UCK_CHLKC_MATH
    zeroacc();
#endif

#ifdef UCK_CHLKC_UNPACK
    zerosrc();
#endif

    kernel_main();

    return 0;
}
