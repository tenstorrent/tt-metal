// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry.c - experimental X280 telemetry active image.
 *
 * Linked behind the lifecycle harness (src/x280_active_start.c), which calls
 * main on all four harts and returns them to resident idle afterwards.
 * For now each hart just reports its hart id as the result.
 */
#include "rt/x280_idle.h"

int main(uint64_t hartid) {
    x280_hart_set_result(hartid, hartid);
    return 0;
}
