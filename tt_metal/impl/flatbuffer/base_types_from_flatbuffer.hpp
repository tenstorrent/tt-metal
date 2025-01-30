// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_types_generated.h"
#include <buffer_constants.hpp>
#include <kernel_types.hpp>
#include <data_types.hpp>
#include <tt_backend_api_types.hpp>
#include <tile.hpp>
#include <circular_buffer_constants.h>

namespace tt::tt_metal {

BufferType from_flatbuffer(tt::tt_metal::flatbuffer::BufferType type);

tt::tt_metal::DataMovementProcessor from_flatbuffer(tt::tt_metal::flatbuffer::DataMovementProcessor in);

tt::tt_metal::NOC from_flatbuffer(tt::tt_metal::flatbuffer::NOC in);
tt::tt_metal::NOC_MODE from_flatbuffer(tt::tt_metal::flatbuffer::NOC_MODE in);
tt::tt_metal::Eth from_flatbuffer(tt::tt_metal::flatbuffer::Eth in);

MathFidelity from_flatbuffer(tt::tt_metal::flatbuffer::MathFidelity input);
UnpackToDestMode from_flatbuffer(tt::tt_metal::flatbuffer::UnpackToDestMode input);
tt::DataFormat from_flatbuffer(tt::tt_metal::flatbuffer::DataFormat input);

Tile from_flatbuffer(const tt::tt_metal::flatbuffer::Tile* tile_fb);

std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>>* tiles_fb);

}  // namespace tt::tt_metal
