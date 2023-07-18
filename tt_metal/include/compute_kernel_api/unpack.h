#pragma once


#include "common_globals.h"
#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif

namespace ckernel {

/**
 * Helper function to reconfigure unpacker srca and srcb input data formats.
 */
ALWI void unpack_reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format(srca_new_operand, srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

}
