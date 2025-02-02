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

template <typename CommandType>
std::variant<CoreCoord, CoreRange, CoreRangeSet> core_spec_from_flatbuffer(const CommandType* cmd) {
    switch (cmd->core_spec_type()) {
        case flatbuffer::CoreSpec::CoreCoord: {
            const auto* core_coord = cmd->core_spec_as_CoreCoord();
            TT_FATAL(core_coord, "Invalid CoreCoord data from flatbuffer.");
            return CoreCoord{core_coord->x(), core_coord->y()};
        }
        case flatbuffer::CoreSpec::CoreRange: {
            const auto* core_range = cmd->core_spec_as_CoreRange();
            TT_FATAL(core_range, "Invalid CoreRange data from flatbuffer.");
            return CoreRange{
                {core_range->start()->x(), core_range->start()->y()}, {core_range->end()->x(), core_range->end()->y()}};
        }
        case flatbuffer::CoreSpec::CoreRangeSet: {
            const auto* core_range_set = cmd->core_spec_as_CoreRangeSet();
            TT_FATAL(core_range_set, "Invalid CoreRangeSet data from flatbuffer.");

            std::vector<CoreRange> ranges;
            for (const auto* range : *core_range_set->ranges()) {
                ranges.emplace_back(
                    CoreCoord{range->start()->x(), range->start()->y()},
                    CoreCoord{range->end()->x(), range->end()->y()});
            }
            return CoreRangeSet{ranges};
        }
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
        case flatbuffer::KernelConfig::NONE:
            throw std::runtime_error("Unhandled KernelConfig type in from_flatbuffer.");
    }
    TT_THROW("Unhandled KernelConfig type in from_flatbuffer.");
}

}  // namespace tt::tt_metal
