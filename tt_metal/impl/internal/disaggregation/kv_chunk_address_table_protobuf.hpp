// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Protobuf export/import for CacheAddressTable.
//
// Consumers must:
//   1. Generate the proto with GENERATE_PROTO_FILES() and add the .pb.cc to their target
//   2. Add the protobuf .cpp to their target sources
//   3. Add appropriate include directories for the generated .pb.h
//   4. Link protobuf::libprotobuf

#pragma once

#include <string>

#include <internal/disaggregation/cache_address_table.hpp>

namespace tt::tt_metal::internal::disaggregation {

// --- Binary wire format ---

std::string export_to_protobuf(const CacheAddressTable& table);
void export_to_protobuf_file(const CacheAddressTable& table, const std::string& path);
CacheAddressTable import_from_protobuf(const std::string& data);
CacheAddressTable import_from_protobuf_file(const std::string& path);

// --- Human-readable text format (debug only) ---

std::string export_to_protobuf_text(const CacheAddressTable& table);
void export_to_protobuf_text_file(const CacheAddressTable& table, const std::string& path);
CacheAddressTable import_from_protobuf_text(const std::string& text);
CacheAddressTable import_from_protobuf_text_file(const std::string& path);

}  // namespace tt::tt_metal::internal::disaggregation
