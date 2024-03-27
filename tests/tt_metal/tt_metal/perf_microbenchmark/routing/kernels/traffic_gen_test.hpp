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
        return "PASS";
    case PACKET_QUEUE_TEST_TIMEOUT:
        return "TIMEOUT";
    case PACKET_QUEUE_TEST_DATA_MISMATCH:
        return "DATA_MISMATCH";
    default:
        return "UNKNOWN";
    }
}
