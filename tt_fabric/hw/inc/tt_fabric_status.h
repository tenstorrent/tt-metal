// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t TT_FABRIC_STAUS_MASK = 0xabc00000;
constexpr uint32_t TT_FABRIC_STATUS_STARTED = TT_FABRIC_STAUS_MASK | 0x0;
constexpr uint32_t TT_FABRIC_STATUS_PASS = TT_FABRIC_STAUS_MASK | 0x1;
constexpr uint32_t TT_FABRIC_STATUS_TIMEOUT = TT_FABRIC_STAUS_MASK | 0xdead0;
constexpr uint32_t TT_FABRIC_STATUS_BAD_HEADER = TT_FABRIC_STAUS_MASK | 0xdead1;
constexpr uint32_t TT_FABRIC_STATUS_DATA_MISMATCH = TT_FABRIC_STAUS_MASK | 0x3;

// indexes of return values in test results buffer
constexpr uint32_t TT_FABRIC_STATUS_INDEX = 0;
constexpr uint32_t TT_FABRIC_WORD_CNT_INDEX = 2;
constexpr uint32_t TT_FABRIC_CYCLES_INDEX = 4;
constexpr uint32_t TT_FABRIC_ITER_INDEX = 6;
constexpr uint32_t TT_FABRIC_MISC_INDEX = 16;

inline const char* tt_fabric_status_to_string(uint32_t status) {
    switch (status) {
        case TT_FABRIC_STATUS_STARTED: return "STARTED";
        case TT_FABRIC_STATUS_PASS: return "DONE/OK";
        case TT_FABRIC_STATUS_TIMEOUT: return "TIMEOUT";
        case TT_FABRIC_STATUS_BAD_HEADER: return "BAD_PACKET_HEADER";
        case TT_FABRIC_STATUS_DATA_MISMATCH: return "DATA_MISMATCH";
        default: return "UNKNOWN";
    }
}

#define TX_TEST_IDX_TOT_DATA_WORDS TT_FABRIC_MISC_INDEX + 1
#define TX_TEST_IDX_NPKT TT_FABRIC_MISC_INDEX + 3
#define TX_TEST_IDX_WORDS_FLUSHED TT_FABRIC_MISC_INDEX + 5
#define TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER TT_FABRIC_MISC_INDEX + 7
#define TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER TT_FABRIC_MISC_INDEX + 9
#define TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER TT_FABRIC_MISC_INDEX + 11
// #define TX_TEST_IDX_ TT_FABRIC_MISC_INDEX +
// #define TX_TEST_IDX_ TT_FABRIC_MISC_INDEX +

enum class pkt_dest_size_choices_t {
    RANDOM = 0,
    SAME_START_RNDROBIN_FIX_SIZE = 1  // max packet size used
};
