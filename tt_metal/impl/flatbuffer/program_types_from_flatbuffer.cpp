// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/program_types_from_flatbuffer.hpp"
#include "flatbuffer/base_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

std::variant<CoreCoord, CoreRange, CoreRangeSet> from_flatbuffer(
    const flatbuffer::CoreSpec core_spec, const void* flatbuffer_union) {
    switch (core_spec) {
        case flatbuffer::CoreSpec::CoreCoord: {
            auto core_coord = static_cast<const flatbuffer::CoreCoord*>(flatbuffer_union);
            TT_FATAL(core_coord, "Invalid CoreCoord data");
            return CoreCoord{core_coord->x(), core_coord->y()};
        }
        case flatbuffer::CoreSpec::CoreRange: {
            auto core_range = static_cast<const flatbuffer::CoreRange*>(flatbuffer_union);
            TT_FATAL(core_range, "Invalid CoreRange data");
            return CoreRange{
                {core_range->start()->x(), core_range->start()->y()}, {core_range->end()->x(), core_range->end()->y()}};
        }
        case flatbuffer::CoreSpec::CoreRangeSet: {
            auto core_range_set = static_cast<const flatbuffer::CoreRangeSet*>(flatbuffer_union);
            TT_FATAL(core_range_set, "Invalid CoreRangeSet data");
            std::vector<CoreRange> ranges;
            for (const auto range : *core_range_set->ranges()) {
                ranges.emplace_back(
                    CoreCoord{range->start()->x(), range->start()->y()},
                    CoreCoord{range->end()->x(), range->end()->y()});
            }
            return CoreRangeSet{ranges};
        }
        default: throw std::runtime_error("Unhandled CoreSpec type in from_flatbuffer");
    }
}

DataMovementConfig from_flatbuffer(const flatbuffer::DataMovementConfig* fb_config) {
    DataMovementConfig config;

    // Extract processor, noc, and noc_mode
    config.processor = from_flatbuffer(fb_config->processor());
    config.noc = from_flatbuffer(fb_config->noc());
    config.noc_mode = from_flatbuffer(fb_config->noc_mode());

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

ComputeConfig from_flatbuffer(const flatbuffer::ComputeConfig* fb_config) {
    ComputeConfig config;

    // Extract math_fidelity and boolean flags
    config.math_fidelity = from_flatbuffer(fb_config->math_fidelity());
    config.fp32_dest_acc_en = fb_config->fp32_dest_acc_en();
    config.dst_full_sync_en = fb_config->dst_full_sync_en();
    config.bfp8_pack_precise = fb_config->bfp8_pack_precise();
    config.math_approx_mode = fb_config->math_approx_mode();

    // Extract unpack_to_dest_mode
    auto fb_unpack_modes = fb_config->unpack_to_dest_mode();
    config.unpack_to_dest_mode.reserve(fb_unpack_modes->size());
    for (auto fb_mode : *fb_unpack_modes) {
        config.unpack_to_dest_mode.push_back(from_flatbuffer(fb_mode));
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

EthernetConfig from_flatbuffer(const flatbuffer::EthernetConfig* fb_config) {
    EthernetConfig config;

    // Extract eth_mode, noc, and processor
    config.eth_mode = from_flatbuffer(fb_config->eth_mode());
    config.noc = from_flatbuffer(fb_config->noc());
    config.processor = from_flatbuffer(fb_config->processor());

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

std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> from_flatbuffer(
    const flatbuffer::KernelConfig config_type, const void* flatbuffer_union) {
    switch (config_type) {
        case flatbuffer::KernelConfig::DataMovementConfig:
            return from_flatbuffer(static_cast<const flatbuffer::DataMovementConfig*>(flatbuffer_union));
        case flatbuffer::KernelConfig::ComputeConfig:
            return from_flatbuffer(static_cast<const flatbuffer::ComputeConfig*>(flatbuffer_union));
        case flatbuffer::KernelConfig::EthernetConfig:
            return from_flatbuffer(static_cast<const flatbuffer::EthernetConfig*>(flatbuffer_union));
        case flatbuffer::KernelConfig::NONE:
            throw std::runtime_error("Unhandled KernelConfig type in from_flatbuffer.");
    }
    TT_THROW("Unhandled KernelConfig type in from_flatbuffer.");
}

std::vector<SubDeviceId> from_flatbuffer(const flatbuffers::Vector<uint8_t>* fb_sub_device_ids) {
    std::vector<SubDeviceId> sub_device_ids(fb_sub_device_ids ? fb_sub_device_ids->size() : 0);

    for (size_t i = 0; i < sub_device_ids.size(); ++i) {
        sub_device_ids[i] = SubDeviceId{(*fb_sub_device_ids)[i]};
    }

    return sub_device_ids;
}

}  // namespace tt::tt_metal
