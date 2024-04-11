// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

inline const char *packet_queue_test_status_to_string(uint32_t status) {
    switch (status) {
    case PACKET_QUEUE_TEST_STARTED:
        return "STARTED";
    case PACKET_QUEUE_TEST_PASS:
        return "DONE/OK";
    case PACKET_QUEUE_TEST_TIMEOUT:
        return "TIMEOUT";
    case PACKET_QUEUE_TEST_DATA_MISMATCH:
        return "DATA_MISMATCH";
    default:
        return "UNKNOWN";
    }
}

inline uint64_t get_64b_result(uint32_t* buf, uint32_t index) {
    return (((uint64_t)buf[index]) << 32) | buf[index+1];
}

inline uint64_t get_64b_result(const std::vector<uint32_t>& vec, uint32_t index) {
    return (((uint64_t)vec[index]) << 32) | vec[index+1];
}
