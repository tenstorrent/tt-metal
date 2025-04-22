// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_types_generated.h"
#include <buffer_types.hpp>
#include <kernel_types.hpp>
#include <data_types.hpp>
#include <tt_backend_api_types.hpp>
#include <tile.hpp>
#include <circular_buffer_constants.h>

namespace tt::tt_metal {

flatbuffer::DataMovementProcessor to_flatbuffer(DataMovementProcessor in);
flatbuffer::NOC to_flatbuffer(NOC in);
flatbuffer::NOC_MODE to_flatbuffer(NOC_MODE in);
flatbuffer::EthMode to_flatbuffer(Eth in);

flatbuffer::MathFidelity to_flatbuffer(MathFidelity input);
flatbuffer::UnpackToDestMode to_flatbuffer(UnpackToDestMode input);
flatbuffer::DataFormat to_flatbuffer(tt::DataFormat input);

flatbuffer::Tile to_flatbuffer(const Tile& tile);

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::Tile>>> to_flatbuffer(
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles, flatbuffers::FlatBufferBuilder& builder);

}  // namespace tt::tt_metal
