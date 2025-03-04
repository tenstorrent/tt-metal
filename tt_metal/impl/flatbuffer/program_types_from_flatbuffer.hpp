// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "program_types_generated.h"
#include <core_coord.hpp>
#include <kernel_types.hpp>
#include <sub_device_types.hpp>

namespace tt::tt_metal {

DataMovementConfig from_flatbuffer(const flatbuffer::DataMovementConfig* fb_config);
ComputeConfig from_flatbuffer(const flatbuffer::ComputeConfig* fb_config);
EthernetConfig from_flatbuffer(const flatbuffer::EthernetConfig* fb_config);
std::vector<SubDeviceId> from_flatbuffer(const flatbuffers::Vector<uint8_t>* fb_sub_device_ids);

std::vector<CoreCoord> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffer::CoreCoord>>* core_spec_fbs);
std::vector<std::vector<uint32_t>> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffer::UInt32Vector>>* vec_of_vec_fbs);

CoreCoord from_flatbuffer(const flatbuffer::CoreCoord* fb_core_coord);
CoreRange from_flatbuffer(const flatbuffer::CoreRange* fb_core_range);
CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* fb_core_range_set);

template <typename CommandType>
std::variant<CoreCoord, CoreRange, CoreRangeSet> core_spec_from_flatbuffer(const CommandType* cmd) {
    switch (cmd->core_spec_type()) {
        case flatbuffer::CoreSpec::CoreCoord: return from_flatbuffer(cmd->core_spec_as_CoreCoord());
        case flatbuffer::CoreSpec::CoreRange: return from_flatbuffer(cmd->core_spec_as_CoreRange());
        case flatbuffer::CoreSpec::CoreRangeSet: return from_flatbuffer(cmd->core_spec_as_CoreRangeSet());
        case flatbuffer::CoreSpec::NONE: TT_THROW("Invalid CoreSpec type. NONE cannot be processed.");
    }
    TT_THROW("Unhandled CoreSpec type in from_flatbuffer.");
}

template <typename CommandType>
std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> kernel_config_from_flatbuffer(const CommandType* cmd) {
    switch (cmd->kernel_config_type()) {
        case flatbuffer::KernelConfig::DataMovementConfig:
            return from_flatbuffer(cmd->kernel_config_as_DataMovementConfig());
        case flatbuffer::KernelConfig::ComputeConfig: return from_flatbuffer(cmd->kernel_config_as_ComputeConfig());
        case flatbuffer::KernelConfig::EthernetConfig: return from_flatbuffer(cmd->kernel_config_as_EthernetConfig());
        case flatbuffer::KernelConfig::NONE: TT_THROW("Unhandled KernelConfig type in from_flatbuffer.");
    }
    TT_THROW("Unhandled KernelConfig type in from_flatbuffer.");
}

}  // namespace tt::tt_metal
