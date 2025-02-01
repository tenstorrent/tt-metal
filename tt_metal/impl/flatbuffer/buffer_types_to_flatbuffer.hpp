// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer_types_generated.h"
#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include <circular_buffer_types.hpp>

namespace tt::tt_metal {

flatbuffer::BufferType to_flatbuffer(BufferType type);
flatbuffer::TensorMemoryLayout to_flatbuffer(TensorMemoryLayout layout);

flatbuffers::Offset<flatbuffer::CircularBufferConfig> to_flatbuffer(
    const CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
