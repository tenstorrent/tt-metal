// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/program_types_from_flatbuffer.hpp"
#include "flatbuffer/base_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

DataMovementConfig from_flatbuffer(const flatbuffer::DataMovementConfig* fb_config) {
    TT_FATAL(fb_config, "Invalid DataMovementConfig data from flatbuffer.");
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
    TT_FATAL(fb_config, "Invalid ComputeConfig data from flatbuffer.");
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
    TT_FATAL(fb_config, "Invalid EthernetConfig data from flatbuffer.");
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

std::vector<SubDeviceId> from_flatbuffer(const flatbuffers::Vector<uint8_t>* fb_sub_device_ids) {
    if (!fb_sub_device_ids) {
        return {};
    }

    std::vector<SubDeviceId> sub_device_ids;
    sub_device_ids.reserve(fb_sub_device_ids->size());
    for (size_t i = 0; i < sub_device_ids.size(); ++i) {
        sub_device_ids.push_back(SubDeviceId{(*fb_sub_device_ids)[i]});
    }

    return sub_device_ids;
}

std::vector<CoreCoord> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffer::CoreCoord>>* core_spec_fbs) {
    TT_FATAL(core_spec_fbs, "Invalid Vector of CoreCoord data from flatbuffer.");

    std::vector<CoreCoord> core_spec(core_spec_fbs->size());
    for (const auto* coord_fbs : *core_spec_fbs) {
        core_spec.emplace_back(coord_fbs->x(), coord_fbs->y());
    }
    return core_spec;
}

std::vector<std::vector<uint32_t>> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffer::UInt32Vector>>* vec_of_vec_fbs) {
    TT_FATAL(vec_of_vec_fbs, "Invalid FlatBuffer data: expected a vector of vector of uint32_t.");

    std::vector<std::vector<uint32_t>> result(vec_of_vec_fbs->size());
    for (const auto* sub_vector_fbs : *vec_of_vec_fbs) {
        std::vector<uint32_t> sub_vector(sub_vector_fbs->values()->begin(), sub_vector_fbs->values()->end());
        result.push_back(std::move(sub_vector));
    }
    return result;
}

CoreCoord from_flatbuffer(const flatbuffer::CoreCoord* fb_core_coord) {
    TT_FATAL(fb_core_coord, "Invalid CoreCoord data from flatbuffer.");
    return CoreCoord{fb_core_coord->x(), fb_core_coord->y()};
}

CoreRange from_flatbuffer(const flatbuffer::CoreRange* fb_core_range) {
    TT_FATAL(
        fb_core_range && fb_core_range->start() && fb_core_range->end(), "Invalid CoreRange data from flatbuffer.");
    return CoreRange{
        from_flatbuffer(fb_core_range->start()),  // Reuse CoreCoord deserialization
        from_flatbuffer(fb_core_range->end())};
}

CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* fb_core_range_set) {
    TT_FATAL(fb_core_range_set, "Invalid CoreRangeSet data from flatbuffer.");

    std::vector<CoreRange> ranges;
    for (const auto* range : *fb_core_range_set->ranges()) {
        ranges.emplace_back(from_flatbuffer(range));  // Reuse CoreRange deserialization
    }
    return CoreRangeSet{ranges};
}

}  // namespace tt::tt_metal
