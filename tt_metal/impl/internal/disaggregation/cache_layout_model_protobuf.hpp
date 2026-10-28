// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Protobuf export/import for CacheLayoutModel (protobuf/cache_layout_model.proto). Consumers build it the
// way kv_chunk_address_table_protobuf.hpp describes.

#pragma once

#include <string>

#include <internal/disaggregation/to_chunk_map.hpp>

namespace tt::tt_metal::internal::disaggregation {

std::string export_to_protobuf(const CacheLayoutModel& model);
void export_to_protobuf_file(const CacheLayoutModel& model, const std::string& path);
CacheLayoutModel import_cache_layout_model_from_protobuf(const std::string& data);
CacheLayoutModel import_cache_layout_model_from_protobuf_file(const std::string& path);

}  // namespace tt::tt_metal::internal::disaggregation
