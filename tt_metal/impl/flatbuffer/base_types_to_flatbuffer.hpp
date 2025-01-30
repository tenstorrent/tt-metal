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

tt::tt_metal::flatbuffer::BufferType to_flatbuffer(BufferType type);

tt::tt_metal::flatbuffer::TensorMemoryLayout to_flatbuffer(TensorMemoryLayout layout);
tt::tt_metal::flatbuffer::DataMovementProcessor to_flatbuffer(tt::tt_metal::DataMovementProcessor in);

tt::tt_metal::flatbuffer::NOC to_flatbuffer(tt::tt_metal::NOC in);
tt::tt_metal::flatbuffer::NOC_MODE to_flatbuffer(tt::tt_metal::NOC_MODE in);
tt::tt_metal::flatbuffer::Eth to_flatbuffer(tt::tt_metal::Eth in);

tt::tt_metal::flatbuffer::MathFidelity to_flatbuffer(MathFidelity input);
tt::tt_metal::flatbuffer::UnpackToDestMode to_flatbuffer(UnpackToDestMode input);
tt::tt_metal::flatbuffer::DataFormat to_flatbuffer(tt::DataFormat input);

flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile> to_flatbuffer(
    const Tile& tile, flatbuffers::FlatBufferBuilder& builder);

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>>> to_flatbuffer(
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles, flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
