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
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "device/device_impl.hpp"
#include "hostdevcommon/dprint_common.h"
#include "hostdev/device_print_common.h"

namespace tt::tt_metal {

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

    struct TopCallstackInfo {
        // uint64_t fits both LP32 and LP64.
        uint64_t pc;
        uint64_t ra;
        size_t skip_frames;
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
        TypedArray,
        TopCallstackInfo>;
    struct FormatMessageBuffer {
        fmt::memory_buffer buffer;
        std::vector<ArgumentValue> argument_values;
    };

    // Factory: reads EI_CLASS from the ELF header and returns the correct 32-bit or 64-bit parser.
    static std::shared_ptr<DevicePrintParser> get_parser_for_elf(const std::string& elf_path);

    virtual std::string_view format_message(
        uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) = 0;

    virtual const std::string& get_elf_path() const = 0;

private:
    static std::map<std::string, std::weak_ptr<DevicePrintParser>> parser_cache;
    friend struct DevicePrintParserDeleter;
};

}  // namespace tt::tt_metal
