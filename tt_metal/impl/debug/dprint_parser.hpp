// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Parses debug print data from device buffers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "hostdevcommon/dprint_common.h"

namespace tt::tt_metal {

class DPrintParser {
public:
    struct ParseResult {
        std::vector<std::string> completed_lines;
        size_t bytes_consumed{};
    };

    explicit DPrintParser(std::string line_prefix = "");
    ParseResult parse(const uint8_t* data, size_t len);
    std::string flush();

private:
    std::string line_prefix_;
    std::ostringstream intermediate_stream_;
    DPrintTypeID prev_type_{DPrintTypeID_Count};
    char most_recent_setw_{0};

    // Helper methods (from dprint_server.cpp anonymous namespace)
    static float bfloat16_to_float(uint16_t bfloat_val);
    static float make_float(uint8_t exp_bit_count, uint8_t mantissa_bit_count, uint32_t data);
    static void AssertSize(uint8_t sz, uint8_t expected_sz);
    static bool StreamEndsWithNewlineChar(const std::ostringstream* stream);
    static void ResetStream(std::ostringstream* stream);

    void PrintTileSlice(const uint8_t* ptr);
    void PrintTensixRegisterData(int setwidth, uint32_t datum, uint16_t data_format);
    void PrintTypedUint32Array(
        int setwidth,
        uint32_t raw_element_count,
        const uint32_t* data,
        TypedU32_ARRAY_Format force_array_type = TypedU32_ARRAY_Format_INVALID);

    std::string get_completed_line();
};

}  // namespace tt::tt_metal
