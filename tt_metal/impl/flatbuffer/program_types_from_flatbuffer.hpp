// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "program_types_generated.h"
#include <core_coord.hpp>
#include <kernel_types.hpp>
#include <sub_device_types.hpp>

namespace tt::tt_metal {

std::variant<CoreCoord, CoreRange, CoreRangeSet> from_flatbuffer(
    const flatbuffer::CoreSpec core_spec, const void* flatbuffer_union);

DataMovementConfig from_flatbuffer(const flatbuffer::DataMovementConfig* fb_config);
ComputeConfig from_flatbuffer(const flatbuffer::ComputeConfig* fb_config);
EthernetConfig from_flatbuffer(const flatbuffer::EthernetConfig* fb_config);

std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> from_flatbuffer(
    const flatbuffer::KernelConfig config_type, const void* flatbuffer_union);

std::vector<SubDeviceId> from_flatbuffer(const flatbuffers::Vector<uint8_t>* fb_sub_device_ids);

}  // namespace tt::tt_metal
