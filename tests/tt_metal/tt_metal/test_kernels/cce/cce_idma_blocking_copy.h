// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "cce_test_common.h"
#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

namespace cce_test {

// As kPollBudgetCycles in dram/mimir_dm_common.hpp.
inline constexpr uint64_t kIdmaWaitBudgetCycles = 1ull << 22;

// One iDMA copy on command buffer 0, then a wait until `trid` has nothing pending. The VC-space wait and the ack wait
// share `deadline`; returns false when it passes first. The trid pending counters are shared by every hart of a CCE,
// so harts that copy at the same time need distinct trids.
inline bool idma_copy_and_wait(
    uint64_t source_address,
    uint64_t destination_address,
    uint32_t len_bytes,
    uint32_t backend_vc,
    uint32_t trid,
    const WaitDeadline& deadline) {
    overlay::reset_cmdbuf<overlay::CMDBUF_0>();
    overlay::idma_setup_as_copy_cmdbuf<overlay::CMDBUF_0>(/*wrapping=*/false);
    overlay::setup_ongoing_cmdbuf<overlay::CMDBUF_0>({});
    overlay::setup_wrapping_vcs_cmdbuf<overlay::CMDBUF_0>(
        {.start = backend_vc, .end = backend_vc},
        {.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
    overlay::setup_trids_cmdbuf<overlay::CMDBUF_0>(trid);
    overlay::setup_max_bytes_in_packet_cmdbuf<overlay::CMDBUF_0>(NOC_OVERLAY_MAX_BYTES_IN_PACKET);
    overlay::set_src_cmdbuf<overlay::CMDBUF_0>(source_address);
    overlay::set_dest_cmdbuf<overlay::CMDBUF_0>(destination_address);
    overlay::set_len_cmdbuf<overlay::CMDBUF_0>(len_bytes);
    while (overlay::idma_free_space_cmdbuf<overlay::CMDBUF_0>(backend_vc) == 0) {
        if (deadline.expired()) {
            return false;
        }
    }
    overlay::issue_cmdbuf<overlay::CMDBUF_0>();
    while (!overlay::idma_acked_cmdbuf<overlay::CMDBUF_0>(trid)) {
        if (deadline.expired()) {
            return false;
        }
    }
    return true;
}

}  // namespace cce_test
