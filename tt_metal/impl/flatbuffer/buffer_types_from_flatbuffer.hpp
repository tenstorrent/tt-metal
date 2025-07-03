// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer_types_generated.h"
#include <circular_buffer_config.hpp>
#include "flatbuffer/base_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

BufferType from_flatbuffer(flatbuffer::BufferType type);

CircularBufferConfig from_flatbuffer(
    const flatbuffer::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer);

TensorMemoryLayout from_flatbuffer(flatbuffer::TensorMemoryLayout layout);

ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation);
ShardMode from_flatbuffer(flatbuffer::ShardMode mode);
ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec);
std::optional<ShardSpecBuffer> from_flatbuffer(const flatbuffer::ShardSpecBuffer* fb_shard_spec);

}  // namespace tt::tt_metal
