// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC consumer for a CrossNodeDFB relay CB.
//
// CrossNode resets the shared ring every program, so TRISC does not need a
// runtime align against the remote iface (that pattern is GlobalDFB-only).
//
// Compile-time parameters:
//   [0] relay_cb_id
//   [1] num_entries

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint32_t relay_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(1);

    for (uint32_t i = 0; i < num_entries; ++i) {
        cb_wait_front(relay_cb_id, 1);
        cb_pop_front(relay_cb_id, 1);
    }
#endif
}
