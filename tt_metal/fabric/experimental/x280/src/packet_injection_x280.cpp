// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// src/packet_injection_x280.cpp -- X280 active firmware entry point
// for packet injection characterization, shared by every producer target.

#include <stdint.h>
#include <stddef.h>

#include "rt/x280_idle.h"
#include "packet_injection/packet_injection_abi.h"
#include "packet_injection/x280_consumer.hpp"

int main(uint64_t hartid) {
    if (hartid != 0) {
        x280_hart_set_result(hartid, X280_PINNED_INPUT_RESULT_NOOP_HART);
        return 0;
    }

    const uint64_t run_value = x280_hart_control(hartid)->run_value;
    const uintptr_t block = (uintptr_t)x280_hart_args(hartid);
    const uint64_t block_bytes = x280_hart_args_bytes();

    if (block == 0u || (block % PI_ALIGNOF(pi_x280_args_t)) != 0 || block_bytes != sizeof(pi_x280_args_t)) {
        x280_hart_set_result(hartid, (uint64_t)PI_ERROR_BAD_ARGS);
        return 0;
    }

    const pi_x280_args_t* args = reinterpret_cast<const pi_x280_args_t*>(block);
    pi_error_t err = packet_injection::run_consumer(hartid, run_value, *args);
    x280_hart_set_result(hartid, (uint64_t)err);
    return 0;
}
