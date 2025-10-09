// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer_types_generated.h"
#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include <circular_buffer_config.hpp>

namespace tt::tt_metal {

flatbuffer::BufferType to_flatbuffer(BufferType type);
flatbuffer::TensorMemoryLayout to_flatbuffer(TensorMemoryLayout layout);

flatbuffers::Offset<flatbuffer::CircularBufferConfig> to_flatbuffer(
    const CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder);

flatbuffer::ShardOrientation to_flatbuffer(ShardOrientation orientation);
flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::BufferDistributionSpec> to_flatbuffer(
    const std::optional<BufferDistributionSpec>& spec, flatbuffers::FlatBufferBuilder& builder);

flatbuffers::Offset<flatbuffer::ShardSpecBuffer> to_flatbuffer(
    const std::optional<ShardSpecBuffer>& shard_parameters, ::flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
