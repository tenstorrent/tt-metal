// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "ckernel_trisc_common.h"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"

void kernel_main() {
#ifdef TRISC_PACK
    constexpr std::uint32_t tensix_only_tc = 16;
    constexpr std::uint32_t posted_increment = 7;

    // Keep the trigger independent of DFB allocation. The bug is in the RTL
    // T6->overlay update path, so one direct legal T6 counter update is enough.
    auto* remapper_control = reinterpret_cast<volatile std::uint32_t*>(REMAP_GLOBAL_CONTROL_REG_ADDR32);
    *remapper_control = 1;
    while ((*remapper_control & 1) == 0) {
    }

    const std::uint32_t overlay_tc0_before = overlay::llk_intf_get_posted(0, 0);
    DPRINT(
        "BEFORE: T6 TC{} is private; overlay TC0 posted={}\n"
        "ACTION: reset T6 TC{}, set capacity=32, then post {} tiles\n",
        tensix_only_tc,
        overlay_tc0_before,
        tensix_only_tc,
        posted_increment);

    ckernel::trisc::tile_counters[tensix_only_tc].f.reset = 1;
    ckernel::trisc::tile_counters[tensix_only_tc].f.buf_capacity = 32;
    ckernel::trisc::tile_counters[tensix_only_tc].f.posted = posted_increment;

    // DPRINT above also gives the remote T6->overlay update enough time to settle
    // before this diagnostic read. The host-side read remains the assertion.
    const std::uint32_t overlay_tc0_after = overlay::llk_intf_get_posted(0, 0);
    DPRINT(
        "AFTER:  overlay TC0 posted={} (expected {}; observed {} means TC16 was truncated to TC0)\n",
        overlay_tc0_after,
        overlay_tc0_before,
        posted_increment);
#endif
}
