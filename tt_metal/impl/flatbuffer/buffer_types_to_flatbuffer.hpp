// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "buffer_types_generated.h"
#include <circular_buffer_types.hpp>

namespace tt::tt_metal {

inline flatbuffers::Offset<tt::tt_metal::flatbuffer::CircularBufferConfig> to_flatbuffer(
    const tt::tt_metal::CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
