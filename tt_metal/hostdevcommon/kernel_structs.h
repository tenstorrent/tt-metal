// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace tt {

enum CBIndex : std::uint8_t {
    c_0 = 0,
    c_1 = 1,
    c_2 = 2,
    c_3 = 3,
    c_4 = 4,
    c_5 = 5,
    c_6 = 6,
    c_7 = 7,
    c_8 = 8,
    c_9 = 9,
    c_10 = 10,
    c_11 = 11,
    c_12 = 12,
    c_13 = 13,
    c_14 = 14,
    c_15 = 15,
    c_16 = 16,
    c_17 = 17,
    c_18 = 18,
    c_19 = 19,
    c_20 = 20,
    c_21 = 21,
    c_22 = 22,
    c_23 = 23,
    c_24 = 24,
    c_25 = 25,
    c_26 = 26,
    c_27 = 27,
    c_28 = 28,
    c_29 = 29,
    c_30 = 30,
    c_31 = 31,
    SIZE = 32
};

// Deprecated and to be deleted.
enum CB : std::uint8_t {
    // Designed to be used as compute inputs, or dataflow in/out
    c_in0 = 0,
    c_in1 = 1,
    c_in2 = 2,
    c_in3 = 3,
    c_in4 = 4,
    c_in5 = 5,
    c_in6 = 6,
    c_in7 = 7,

    // Dataflow in/out only
    dataflow0 = 8,
    dataflow1 = 9,
    dataflow2 = 10,
    dataflow3 = 11,
    dataflow4 = 12,
    dataflow5 = 13,
    dataflow6 = 14,
    dataflow7 = 15,

    // Designed to be used as compute outputs, or dataflow in/out
    c_out0 = 16,
    c_out1 = 17,
    c_out2 = 18,
    c_out3 = 19,
    c_out4 = 20,
    c_out5 = 21,
    c_out6 = 22,
    c_out7 = 23,

    // Designed to be used as compute intermediates, or dataflow in/out
    c_intermed0 = 24,
    c_intermed1 = 25,
    c_intermed2 = 26,
    c_intermed3 = 27,
    c_intermed4 = 28,
    c_intermed5 = 29,
    c_intermed6 = 30,
    c_intermed7 = 31,
};

/////////////////////////////
// end of user facing APIs //
/////////////////////////////

enum DstMode : std::uint8_t {
    Full = 0,
    Half = 1,
    Tile = 2,
    NUM_DST_MODES = 3,
};

// To be deprecated: the old enum from which CBs evolved
enum HlkOperand : std::uint8_t {
    in0 = 0,
    in1 = 1,
    in2 = 2,
    in3 = 3,
    in4 = 4,
    in5 = 5,
    in6 = 6,
    in7 = 7,

    param0 = 8,
    param1 = 9,
    param2 = 10,
    param3 = 11,
    param4 = 12,
    param5 = 13,
    param6 = 14,
    param7 = 15,

    out0 = 16,
    out1 = 17,
    out2 = 18,
    out3 = 19,
    out4 = 20,
    out5 = 21,
    out6 = 22,
    out7 = 23,

    intermed0 = 24,
    intermed1 = 25,
    intermed2 = 26,
    intermed3 = 27,
    intermed4 = 28,
    intermed5 = 29,
    intermed6 = 30,
    intermed7 = 31,
};

constexpr std::uint32_t NUM_MAX_IN_BUFFERS_PER_CORE = HlkOperand::in7 - HlkOperand::in0 + 1;
constexpr std::uint32_t NUM_MAX_PARAM_BUFFERS_PER_CORE = HlkOperand::param7 - HlkOperand::param0 + 1;
constexpr std::uint32_t NUM_MAX_OUT_BUFFERS_PER_CORE = HlkOperand::out7 - HlkOperand::out0 + 1;
constexpr std::uint32_t NUM_MAX_INTERMED_BUFFERS_PER_CORE = HlkOperand::intermed7 - HlkOperand::intermed0 + 1;

}  //  namespace tt
