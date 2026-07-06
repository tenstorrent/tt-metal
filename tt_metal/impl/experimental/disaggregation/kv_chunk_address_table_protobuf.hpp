// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEPRECATED: this header moved to
//   #include "tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"
// and the free functions now live in namespace tt::tt_metal::internal::disaggregation.
//
// This shim re-exports them under their old experimental namespace so existing
// consumers keep compiling while they migrate. It will be removed once
// tt-llm-engine and tt-blaze are scrubbed.

#pragma once

#include "tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"

namespace tt::tt_metal::experimental::disaggregation {

using internal::disaggregation::export_to_protobuf;
using internal::disaggregation::export_to_protobuf_file;
using internal::disaggregation::import_from_protobuf;
using internal::disaggregation::import_from_protobuf_file;

using internal::disaggregation::export_to_protobuf_text;
using internal::disaggregation::export_to_protobuf_text_file;
using internal::disaggregation::import_from_protobuf_text;
using internal::disaggregation::import_from_protobuf_text_file;

}  // namespace tt::tt_metal::experimental::disaggregation
