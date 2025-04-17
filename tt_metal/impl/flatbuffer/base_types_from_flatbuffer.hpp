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

DataMovementProcessor from_flatbuffer(flatbuffer::DataMovementProcessor in);

NOC from_flatbuffer(flatbuffer::NOC in);
NOC_MODE from_flatbuffer(flatbuffer::NOC_MODE in);
Eth from_flatbuffer(flatbuffer::EthMode in);

MathFidelity from_flatbuffer(flatbuffer::MathFidelity input);
UnpackToDestMode from_flatbuffer(flatbuffer::UnpackToDestMode input);
tt::DataFormat from_flatbuffer(flatbuffer::DataFormat input);

Tile from_flatbuffer(const flatbuffer::Tile& tile_fb);

}  // namespace tt::tt_metal
