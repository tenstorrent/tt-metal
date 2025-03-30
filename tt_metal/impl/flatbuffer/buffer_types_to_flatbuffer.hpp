// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <circular_buffer_types.hpp>
#include <flatbuffers/buffer.h>
#include <flatbuffers/flatbuffer_builder.h>
#include <optional>

#include "buffer_types_generated.h"
#include "flatbuffer/base_types_to_flatbuffer.hpp"

namespace tt {
namespace tt_metal {
enum class BufferType;
enum class ShardMode;
enum class ShardOrientation;
enum class TensorMemoryLayout;
struct ShardSpec;
struct ShardSpecBuffer;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

flatbuffer::BufferType to_flatbuffer(BufferType type);
flatbuffer::TensorMemoryLayout to_flatbuffer(TensorMemoryLayout layout);

flatbuffers::Offset<flatbuffer::CircularBufferConfig> to_flatbuffer(
    const CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder);

flatbuffer::ShardOrientation to_flatbuffer(ShardOrientation orientation);
flatbuffer::ShardMode to_flatbuffer(ShardMode shard_mode);
flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder);

flatbuffers::Offset<flatbuffer::ShardSpecBuffer> to_flatbuffer(
    const std::optional<ShardSpecBuffer>& shard_parameters, ::flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
