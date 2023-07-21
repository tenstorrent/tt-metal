#pragma once


#include "common_globals.h"

namespace ckernel {

ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb)
{
    PACK((  llk_pack<false, SYNC, false >(ifrom_dst, icb)  ));
}

/**
 * Helper function to reconfigure packer output data format.
 */
ALWI void pack_reconfig_data_format(const uint32_t new_operand) {
    #ifdef ARCH_GRAYSKULL
        PACK(( llk_pack_reconfig_data_format(new_operand) ));
    #endif
    // NOTE: For wormhole_b0, packer data format reconfig functions don;t yet exist. So skip.
}

}
