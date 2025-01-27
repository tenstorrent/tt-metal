// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "program_types_generated.h"

namespace tt::tt_metal {
inline namespace v0 {

inline std::variant<CoreCoord, CoreRange, CoreRangeSet> FromFlatbuffer(
    const tt::tt_metal::flatbuffer::CoreSpec core_spec, const void* flatbuffer_union) {
    switch (core_spec) {
        case tt::tt_metal::flatbuffer::CoreSpec::CoreCoord: {
            auto core_coord = static_cast<const tt::tt_metal::flatbuffer::CoreCoord*>(flatbuffer_union);
            if (!core_coord) {
                throw std::runtime_error("Invalid CoreCoord data");
            }
            return CoreCoord{core_coord->x(), core_coord->y()};
        }
        case tt::tt_metal::flatbuffer::CoreSpec::CoreRange: {
            auto core_range = static_cast<const tt::tt_metal::flatbuffer::CoreRange*>(flatbuffer_union);
            if (!core_range) {
                throw std::runtime_error("Invalid CoreRange data");
            }
            return CoreRange{
                {core_range->start()->x(), core_range->start()->y()}, {core_range->end()->x(), core_range->end()->y()}};
        }
        case tt::tt_metal::flatbuffer::CoreSpec::CoreRangeSet: {
            auto core_range_set = static_cast<const tt::tt_metal::flatbuffer::CoreRangeSet*>(flatbuffer_union);
            if (!core_range_set) {
                throw std::runtime_error("Invalid CoreRangeSet data");
            }
            std::vector<CoreRange> ranges;
            for (const auto range : *core_range_set->ranges()) {
                ranges.emplace_back(
                    CoreCoord{range->start()->x(), range->start()->y()},
                    CoreCoord{range->end()->x(), range->end()->y()});
            }
            return CoreRangeSet{ranges};
        }
        default: throw std::runtime_error("Unhandled CoreSpec type in FromFlatbuffer");
    }
}

inline DataMovementConfig FromFlatbuffer(const tt::tt_metal::flatbuffer::DataMovementConfig* fb_config) {
    DataMovementConfig config;

    // Extract processor, noc, and noc_mode
    config.processor = FromFlatbuffer(fb_config->processor());
    config.noc = FromFlatbuffer(fb_config->noc());
    config.noc_mode = FromFlatbuffer(fb_config->noc_mode());

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline ComputeConfig FromFlatbuffer(const tt::tt_metal::flatbuffer::ComputeConfig* fb_config) {
    ComputeConfig config;

    // Extract math_fidelity and boolean flags
    config.math_fidelity = FromFlatbuffer(fb_config->math_fidelity());
    config.fp32_dest_acc_en = fb_config->fp32_dest_acc_en();
    config.dst_full_sync_en = fb_config->dst_full_sync_en();
    config.bfp8_pack_precise = fb_config->bfp8_pack_precise();
    config.math_approx_mode = fb_config->math_approx_mode();

    // Extract unpack_to_dest_mode
    auto fb_unpack_modes = fb_config->unpack_to_dest_mode();
    config.unpack_to_dest_mode.reserve(fb_unpack_modes->size());
    for (auto fb_mode : *fb_unpack_modes) {
        config.unpack_to_dest_mode.push_back(FromFlatbuffer(fb_mode));
    }

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline EthernetConfig FromFlatbuffer(const tt::tt_metal::flatbuffer::EthernetConfig* fb_config) {
    EthernetConfig config;

    // Extract eth_mode, noc, and processor
    config.eth_mode = FromFlatbuffer(fb_config->eth_mode());
    config.noc = FromFlatbuffer(fb_config->noc());
    config.processor = FromFlatbuffer(fb_config->processor());

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> FromFlatbuffer(
    const tt::tt_metal::flatbuffer::KernelConfig config_type, const void* flatbuffer_union) {
    switch (config_type) {
        case tt::tt_metal::flatbuffer::KernelConfig::DataMovementConfig:
            return FromFlatbuffer(static_cast<const tt::tt_metal::flatbuffer::DataMovementConfig*>(flatbuffer_union));
        case tt::tt_metal::flatbuffer::KernelConfig::ComputeConfig:
            return FromFlatbuffer(static_cast<const tt::tt_metal::flatbuffer::ComputeConfig*>(flatbuffer_union));
        case tt::tt_metal::flatbuffer::KernelConfig::EthernetConfig:
            return FromFlatbuffer(static_cast<const tt::tt_metal::flatbuffer::EthernetConfig*>(flatbuffer_union));
        default: throw std::runtime_error("Unhandled KernelConfig type in FromFlatbuffer.");
    }
}

// Convert from FB vector to Span of SubDeviceIds
tt::stl::Span<const SubDeviceId> FromFlatBuffer(const flatbuffers::Vector<uint8_t>* fb_sub_device_ids) {
    std::vector<SubDeviceId> sub_device_ids(fb_sub_device_ids ? fb_sub_device_ids->size() : 0);

    for (size_t i = 0; i < sub_device_ids.size(); ++i) {
        sub_device_ids[i] = SubDeviceId{(*fb_sub_device_ids)[i]};
    }

    return sub_device_ids;
}

}  // namespace v0
}  // namespace tt::tt_metal
