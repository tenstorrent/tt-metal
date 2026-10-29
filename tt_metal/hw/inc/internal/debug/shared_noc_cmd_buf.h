// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Two dedicated-NOC data movement kernels that share one NOC (built with DEDICATED_NOC_SHARED, see
// DataMovement1XXConfig::dedicated_noc_sharing) must program disjoint command buffers on it. Under the
// watcher's NoC sanitizer, every command they issue records its command buffer and checks the other RISC's
// record. The hook sits in NOC_CMD_BUF_WRITE_REG (noc_nonblocking_api.h); the check is defined in
// internal/debug/sanitize.h, which includes that header.
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE) && !defined(FORCE_WATCHER_OFF) && \
    defined(DEDICATED_NOC_SHARED) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC))
#define DEBUG_SANITIZE_SHARED_NOC_CMD_BUF
void debug_sanitize_shared_noc_cmd_buf(uint32_t noc, uint32_t cmd_buf);
#endif
