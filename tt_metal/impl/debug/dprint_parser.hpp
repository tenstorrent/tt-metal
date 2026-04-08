// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Parses debug print data from device buffers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "device/device_impl.hpp"
#include "hostdevcommon/dprint_common.h"
#include "hostdev/device_print_common.h"

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

// Abstract interface for parsing DEVICE_PRINT format strings from ELF files.
// Concrete implementations are templated on pointer size (4 or 8 bytes) to handle
// 32-bit and 64-bit ELFs at compile time via DevicePrintStringInfo32/DevicePrintStringInfo64.
class DevicePrintParser {
public:
    virtual ~DevicePrintParser() = default;

    struct TileSliceDynamic {
        TileSliceHostDev<0> header;
        std::vector<uint8_t> data;
    };

    struct TypedArray {
        tt::DataFormat type;
        std::vector<uint32_t> data;
    };

    using ArgumentValue = std::variant<
        bool,
        int8_t,
        uint8_t,
        int16_t,
        uint16_t,
        int32_t,
        uint32_t,
        int64_t,
        uint64_t,
        float,
        double,
        TileSliceDynamic,
        TypedArray>;
    struct FormatMessageBuffer {
        fmt::memory_buffer buffer;
        std::vector<ArgumentValue> argument_values;
    };

    // Factory: reads EI_CLASS from the ELF header and returns the correct 32-bit or 64-bit parser.
    static std::shared_ptr<DevicePrintParser> get_parser_for_elf(const std::string& elf_path);

    virtual std::string_view format_message(
        uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) = 0;

private:
    static std::map<std::string, std::weak_ptr<DevicePrintParser>> parser_cache;
    friend struct DevicePrintParserDeleter;
};

}  // namespace tt::tt_metal
