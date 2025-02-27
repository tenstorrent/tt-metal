// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "include/compute_kernel_api/common.h"

namespace NAMESPACE {

void MAIN {
    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.
    // uint32_t rt = get_arg_val<uint32_t>(0);
    uint32_t rt0 = get_common_arg_val<uint32_t>(0);
    uint32_t rt1 = get_common_arg_val<uint32_t>(1);
    DPRINT << HEX() << rt0 << ", " << HEX() << rt1 << ENDL();
}

}  // namespace NAMESPACE
