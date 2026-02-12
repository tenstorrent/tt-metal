// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
#ifdef TRISC_PACK
    int32_t A = 1;
    int32_t B = 2;

    DPRINT << "TEST packer" << ENDL();
    DPRINT << A + B << ENDL();
#endif

#ifdef TRISC_UNPACK
    int32_t A = 2;
    int32_t B = 2;

    DPRINT << "TEST unpacker" << ENDL();
    DPRINT << A + B << ENDL();
#endif

#ifdef TRISC_MATH
    int32_t A = 3;
    int32_t B = 2;

    DPRINT << "TEST math" << ENDL();
    DPRINT << A + B << ENDL();
#endif
}
