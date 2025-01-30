// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer_types_generated.h"
#include <circular_buffer_types.hpp>
#include "flatbuffer/base_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

CircularBufferConfig from_flatbuffer(
    const tt::tt_metal::flatbuffer::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer);

}  // namespace tt::tt_metal
