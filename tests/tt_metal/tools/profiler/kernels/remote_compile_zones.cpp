// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    DeviceZoneScopedN("REMOTE-COMPILE-OUTER");
    {
        DeviceZoneScopedN("REMOTE-COMPILE-INNER");
    }
}
