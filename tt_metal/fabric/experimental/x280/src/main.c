// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * main.c - Homogeneous four-hart X280 active-session sentinel.
 *
 * Runs one host-supplied value and returns through the acknowledged
 * resident-idle handoff.
 *
 * This firmware is loaded at X280_ACTIVE_FW_LOAD_ADDR (ld/x280.ld);
 * the idle FW owns the lowest 4 KiB of LIM. See x280_address_map.h,
 * x280_hart_protocol.h, and x280/src/lim_idle.c.
 */
#include "x280_hart_protocol.h"
#include "rt/x280_idle.h"

/* entry.S passes a0 = mhartid to main() for all four harts. Echoing the run
 * value is all this image does. */
int main(uint64_t hartid) {
    x280_hart_set_result(hartid, x280_hart_control(hartid)->run_value);
    return 0;
}
