#pragma once

#define SYNC SyncHalf

#include "common_globals.h"
#include "compute_kernel_api/llk_pack_includes.h"

namespace ckernel {

ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb)
{
    PACK((  llk_pack<false, SYNC, false >(ifrom_dst, icb)  ));
}


}
