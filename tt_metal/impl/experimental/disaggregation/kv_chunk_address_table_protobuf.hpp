// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Protobuf export/import for KvChunkAddressTable.
//
// Consumers must:
//   1. Generate the proto with GENERATE_PROTO_FILES() and add the .pb.cc to their target
//   2. Add the protobuf .cpp to their target sources
//   3. Add appropriate include directories for the generated .pb.h
//   4. Link protobuf::libprotobuf

#pragma once

#include <string>

#include "tt_metal/api/tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp"

namespace tt::tt_metal::experimental::disaggregation {

// --- Binary wire format ---

std::string export_to_protobuf(const KvChunkAddressTable& table);
void export_to_protobuf_file(const KvChunkAddressTable& table, const std::string& path);
KvChunkAddressTable import_from_protobuf(const std::string& data);
KvChunkAddressTable import_from_protobuf_file(const std::string& path);

// --- Human-readable text format (debug only) ---

std::string export_to_protobuf_text(const KvChunkAddressTable& table);
void export_to_protobuf_text_file(const KvChunkAddressTable& table, const std::string& path);
KvChunkAddressTable import_from_protobuf_text(const std::string& text);
KvChunkAddressTable import_from_protobuf_text_file(const std::string& path);

}  // namespace tt::tt_metal::experimental::disaggregation
