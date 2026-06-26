// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED: this header has moved to the internal API surface.
// Include "tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"
// and use tt::tt_metal::internal::disaggregation instead. This shim forwards the
// serializer free-functions into the old experimental namespace so existing callers
// keep compiling while they are migrated.

#include "tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"

namespace tt::tt_metal::experimental::disaggregation {

using tt::tt_metal::internal::disaggregation::export_to_protobuf;
using tt::tt_metal::internal::disaggregation::export_to_protobuf_file;
using tt::tt_metal::internal::disaggregation::import_from_protobuf;
using tt::tt_metal::internal::disaggregation::import_from_protobuf_file;

using tt::tt_metal::internal::disaggregation::export_to_protobuf_text;
using tt::tt_metal::internal::disaggregation::export_to_protobuf_text_file;
using tt::tt_metal::internal::disaggregation::import_from_protobuf_text;
using tt::tt_metal::internal::disaggregation::import_from_protobuf_text_file;

}  // namespace tt::tt_metal::experimental::disaggregation
