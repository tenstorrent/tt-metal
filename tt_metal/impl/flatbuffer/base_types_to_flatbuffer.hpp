// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_types_generated.h"
#include <tt-metalium/memory/buffer/buffer_types.hpp>
#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/data_formats/data_format.hpp>
#include <tt-metalium/data_formats/tile.hpp>
#include <tt-metalium/workload/legacy/circular_buffer_constants.h>
#include "kernels/kernel.hpp"

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
