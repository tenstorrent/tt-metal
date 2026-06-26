// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED: these types have moved to the internal API surface.
// Include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp> and use
// tt::tt_metal::internal::disaggregation instead. This header provides deprecated
// aliases so existing callers keep compiling while they are migrated.

#include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>

namespace tt::tt_metal::experimental::disaggregation {

using DeviceGroupIndex
    [[deprecated("Moved to tt::tt_metal::internal::disaggregation::DeviceGroupIndex")]] =
        tt::tt_metal::internal::disaggregation::DeviceGroupIndex;

using DeviceGroup [[deprecated("Moved to tt::tt_metal::internal::disaggregation::DeviceGroup")]] =
    tt::tt_metal::internal::disaggregation::DeviceGroup;

using KvCacheLocation
    [[deprecated("Moved to tt::tt_metal::internal::disaggregation::KvCacheLocation")]] =
        tt::tt_metal::internal::disaggregation::KvCacheLocation;

using KvChunkAddressTableConfig
    [[deprecated("Moved to tt::tt_metal::internal::disaggregation::KvChunkAddressTableConfig")]] =
        tt::tt_metal::internal::disaggregation::KvChunkAddressTableConfig;

using KvChunkAddressTable
    [[deprecated("Moved to tt::tt_metal::internal::disaggregation::KvChunkAddressTable")]] =
        tt::tt_metal::internal::disaggregation::KvChunkAddressTable;

}  // namespace tt::tt_metal::experimental::disaggregation
