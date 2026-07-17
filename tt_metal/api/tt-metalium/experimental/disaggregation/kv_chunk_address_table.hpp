// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED: KvChunkAddressTable and its related core types moved out of the
// experimental API into the internal API:
//
//   #include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>
//   namespace tt::tt_metal::internal::disaggregation
//
// This header re-exports the types under their old experimental namespace as
// deprecated aliases so existing consumers keep compiling while they migrate.
// Update includes and namespaces to the internal home; this shim will be
// removed once tt-llm-engine and tt-blaze are scrubbed.

#include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>

namespace tt::tt_metal::experimental::disaggregation {

using DeviceGroupIndex
    [[deprecated("moved to tt::tt_metal::internal::disaggregation; include <tt-metalium/internal/"
                 "disaggregation/kv_chunk_address_table.hpp>")]] = internal::disaggregation::DeviceGroupIndex;

using DeviceGroup
    [[deprecated("moved to tt::tt_metal::internal::disaggregation; include <tt-metalium/internal/"
                 "disaggregation/kv_chunk_address_table.hpp>")]] = internal::disaggregation::DeviceGroup;

using KvCacheLocation
    [[deprecated("moved to tt::tt_metal::internal::disaggregation; include <tt-metalium/internal/"
                 "disaggregation/kv_chunk_address_table.hpp>")]] = internal::disaggregation::KvCacheLocation;

using KvChunkAddressTableConfig
    [[deprecated("moved to tt::tt_metal::internal::disaggregation; include <tt-metalium/internal/"
                 "disaggregation/kv_chunk_address_table.hpp>")]] = internal::disaggregation::KvChunkAddressTableConfig;

using KvChunkAddressTable
    [[deprecated("moved to tt::tt_metal::internal::disaggregation; include <tt-metalium/internal/"
                 "disaggregation/kv_chunk_address_table.hpp>")]] = internal::disaggregation::KvChunkAddressTable;

}  // namespace tt::tt_metal::experimental::disaggregation
