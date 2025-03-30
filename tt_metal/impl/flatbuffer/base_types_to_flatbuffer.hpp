// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <buffer_constants.hpp>
#include <circular_buffer_constants.h>
#include <data_types.hpp>
#include <flatbuffers/buffer.h>
#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/vector.h>
#include <kernel_types.hpp>
#include <tile.hpp>
#include <tt_backend_api_types.hpp>
#include <array>
#include <optional>

#include "base_types_generated.h"

enum class MathFidelity : uint8_t;
namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
enum Eth : uint8_t;
enum class DataMovementProcessor;
struct Tile;
}  // namespace tt_metal
}  // namespace tt

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
