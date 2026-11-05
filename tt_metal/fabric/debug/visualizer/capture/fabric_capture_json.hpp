// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::tt_fabric::capture {

// Reads typed values out of a JSON document by JSON pointer. A missing key or a value of the wrong type fails, naming
// the document and the key's pointer, such as "the manifest has no /arch/l1/address".
class JsonReader {
public:
    using json = nlohmann::ordered_json;
    using json_pointer = json::json_pointer;

    // `document_name` is how errors name the document, such as "manifest".
    JsonReader(const json& document, std::string document_name);

    const json& find_required(const json_pointer& pointer) const;
    bool is_null(const json_pointer& pointer) const;
    const json& read_object(const json_pointer& pointer) const;
    const json& read_array(const json_pointer& pointer) const;
    uint64_t read_unsigned(const json_pointer& pointer) const;
    uint32_t read_uint32(const json_pointer& pointer) const;
    std::string read_string(const json_pointer& pointer) const;
    bool read_bool(const json_pointer& pointer) const;
    // A chip's asic id, written as hex, such as "0x00000000deadbeef".
    uint64_t read_asic_id(const json_pointer& pointer) const;
    // A core's coordinates, written as [x, y].
    CoreXY read_core(const json_pointer& pointer) const;

private:
    const json& document_;
    std::string document_name_;
};

// An asic id as the manifest writes it: "0x" and 16 hex digits.
std::string make_asic_id_text(uint64_t asic_id);

// A core's coordinates as [x, y].
nlohmann::ordered_json make_core_json(const CoreXY& core);

}  // namespace tt::tt_fabric::capture
