// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// #include "compute_kernel_api/common.h"
// #ifdef TRISC_UNPACK
// #include "llk_unpack_untilize_api.h"
// #endif
#include "compute_kernel_api.h"
#include "debug/ttsim_dump.h"

namespace NAMESPACE {

void MAIN {
    TTSIM_TENSIX_DUMP("BEFORE llk_unpack_untilize_init(0)", TTSIM_DUMP_DST);
    UNPACK((llk_unpack_untilize_init(0)));
    TTSIM_TENSIX_DUMP("AFTER llk_unpack_untilize_init(0)", TTSIM_DUMP_DST);

    TTSIM_TENSIX_DUMP("BEFORE llk_unpack_untilize_uninit(0)", false);
    UNPACK((llk_unpack_untilize_uninit(0)));
    TTSIM_TENSIX_DUMP("AFTER llk_unpack_untilize_uninit(0)", false);
}

}  // namespace NAMESPACE
