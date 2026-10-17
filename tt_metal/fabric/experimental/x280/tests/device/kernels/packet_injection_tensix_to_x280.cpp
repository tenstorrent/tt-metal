// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// tests/device/kernels/packet_injection_tensix_to_x280.cpp -- Tensix kernel
// entry point for packet injection characterization.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "packet_injection_abi.h"
#include "tensix_producer.hpp"

void kernel_main() {
    // Reading straight into the union's words leaves the reconstruction as 18
    // stores and no struct copy at all.
    pi_producer_arg_words_t block;
    block.words[0] = get_arg_val<uint32_t>(0);
    block.words[1] = get_arg_val<uint32_t>(1);
    block.words[2] = get_arg_val<uint32_t>(2);
    block.words[3] = get_arg_val<uint32_t>(3);
    block.words[4] = get_arg_val<uint32_t>(4);
    block.words[5] = get_arg_val<uint32_t>(5);
    block.words[6] = get_arg_val<uint32_t>(6);
    block.words[7] = get_arg_val<uint32_t>(7);
    block.words[8] = get_arg_val<uint32_t>(8);
    block.words[9] = get_arg_val<uint32_t>(9);
    block.words[10] = get_arg_val<uint32_t>(10);
    block.words[11] = get_arg_val<uint32_t>(11);
    block.words[12] = get_arg_val<uint32_t>(12);
    block.words[13] = get_arg_val<uint32_t>(13);
    block.words[14] = get_arg_val<uint32_t>(14);
    block.words[15] = get_arg_val<uint32_t>(15);
    block.words[16] = get_arg_val<uint32_t>(16);
    block.words[17] = get_arg_val<uint32_t>(17);

    if (!pi_abi_tag_matches(block.args.abi_tag, PACKET_INJECTION_MODE, PACKET_INJECTION_INCLUDE_CREATION)) {
        return;
    }

    packet_injection::run_producer(block.args);
}
