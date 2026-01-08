// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "mcast_utils.hpp"

void kernel_main() {
    uint32_t arg_idx = 0;
    DEFINE_PERSISTENT_MCAST_SENDER_VARS(mcast);
    DEFINE_MCAST_SENDER_VARS(mcast, mcast0, arg_idx);
    INIT_PERSISTENT_MCAST_SENDER(mcast);
    MCAST_SEND_DATA_WITH_STATE(mcast, mcast0);
    TEARDOWN_PERSISTENT_MCAST_SENDER(mcast);
}
