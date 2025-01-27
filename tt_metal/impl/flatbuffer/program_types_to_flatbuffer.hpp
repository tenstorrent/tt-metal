// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "program_types_generated.h"
#include "lightmetal_capture.hpp"  // For LightMetalCaptureContext

namespace tt::tt_metal {

// Original types defined in core_coord.hpp
inline std::pair<tt::tt_metal::flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec) {
    return std::visit(
        [&](auto&& spec) -> std::pair<tt::tt_metal::flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
            using T = std::decay_t<decltype(spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                auto core_coord = tt::tt_metal::flatbuffer::CreateCoreCoord(builder, spec.x, spec.y);
                return {tt::tt_metal::flatbuffer::CoreSpec::CoreCoord, core_coord.Union()};
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                auto start = tt::tt_metal::flatbuffer::CreateCoreCoord(builder, spec.start_coord.x, spec.start_coord.y);
                auto end = tt::tt_metal::flatbuffer::CreateCoreCoord(builder, spec.end_coord.x, spec.end_coord.y);
                auto core_range = tt::tt_metal::flatbuffer::CreateCoreRange(builder, start, end);
                return {tt::tt_metal::flatbuffer::CoreSpec::CoreRange, core_range.Union()};
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::CoreRange>> range_offsets;
                for (const auto& range : spec.ranges()) {
                    auto start =
                        tt::tt_metal::flatbuffer::CreateCoreCoord(builder, range.start_coord.x, range.start_coord.y);
                    auto end = tt::tt_metal::flatbuffer::CreateCoreCoord(builder, range.end_coord.x, range.end_coord.y);
                    range_offsets.push_back(tt::tt_metal::flatbuffer::CreateCoreRange(builder, start, end));
                }
                auto ranges_vector = builder.CreateVector(range_offsets);
                auto core_range_set = tt::tt_metal::flatbuffer::CreateCoreRangeSet(builder, ranges_vector);
                return {tt::tt_metal::flatbuffer::CoreSpec::CoreRangeSet, core_range_set.Union()};
            } else {
                throw std::runtime_error("Unhandled variant type in ToFlatbuffer");
            }
        },
        core_spec);
}

// Original types defined in kernel_types.hpp
inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const DataMovementConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(tt::tt_metal::flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    // Convert compile_args to FlatBuffer format
    auto compile_args_offset = builder.CreateVector(config.compile_args);

    // Create the FlatBuffer DataMovementConfig object
    auto config_offset = tt::tt_metal::flatbuffer::CreateDataMovementConfig(
        builder,
        ToFlatbuffer(config.processor),
        ToFlatbuffer(config.noc),
        ToFlatbuffer(config.noc_mode),
        compile_args_offset,
        defines_offset);

    return {tt::tt_metal::flatbuffer::KernelConfig::DataMovementConfig, config_offset.Union()};
}

inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ComputeConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(tt::tt_metal::flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    // Convert unpack_to_dest_mode to FlatBuffer format
    std::vector<tt::tt_metal::flatbuffer::UnpackToDestMode> unpack_modes;
    for (const auto& mode : config.unpack_to_dest_mode) {
        unpack_modes.push_back(ToFlatbuffer(mode));
    }
    auto unpack_modes_offset = builder.CreateVector(unpack_modes);

    // Convert compile_args to FlatBuffer format
    auto compile_args_offset = builder.CreateVector(config.compile_args);

    // Create the FlatBuffer ComputeConfig object
    auto config_offset = tt::tt_metal::flatbuffer::CreateComputeConfig(
        builder,
        ToFlatbuffer(config.math_fidelity),
        config.fp32_dest_acc_en,
        config.dst_full_sync_en,
        unpack_modes_offset,
        config.bfp8_pack_precise,
        config.math_approx_mode,
        compile_args_offset,
        defines_offset);

    return {tt::tt_metal::flatbuffer::KernelConfig::ComputeConfig, config_offset.Union()};
}

inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const EthernetConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(tt::tt_metal::flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    // Convert compile_args to FlatBuffer format
    auto compile_args_offset = builder.CreateVector(config.compile_args);

    // Create the FlatBuffer EthernetConfig object
    auto config_offset = tt::tt_metal::flatbuffer::CreateEthernetConfig(
        builder,
        ToFlatbuffer(config.eth_mode),
        ToFlatbuffer(config.noc),
        ToFlatbuffer(config.processor),
        compile_args_offset,
        defines_offset);

    return {tt::tt_metal::flatbuffer::KernelConfig::EthernetConfig, config_offset.Union()};
}

// Generic function for variant, specialized for each type above.
inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    return std::visit(
        [&](auto&& cfg) -> std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (
                std::is_same_v<T, DataMovementConfig> || std::is_same_v<T, ComputeConfig> ||
                std::is_same_v<T, EthernetConfig>) {
                return ToFlatbuffer(builder, cfg);
            } else {
                throw std::runtime_error("Unhandled config type in ToFlatbuffer.");
            }
        },
        config);
}

inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ReaderDataMovementConfig& config) {
    const DataMovementConfig& base_config = config;  // Cast to base
    return ToFlatbuffer(builder, base_config);
}

inline std::pair<tt::tt_metal::flatbuffer::KernelConfig, flatbuffers::Offset<void>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const WriterDataMovementConfig& config) {
    const DataMovementConfig& base_config = config;  // Cast to base
    return ToFlatbuffer(builder, base_config);
}

inline flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg> createRuntimeArg(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<Buffer*, uint32_t>& arg) {
    flatbuffers::Offset<void> value_offset;
    tt::tt_metal::flatbuffer::RuntimeArgValue value_type;

    if (std::holds_alternative<uint32_t>(arg)) {
        // Create UInt32Value table
        uint32_t value = std::get<uint32_t>(arg);
        auto uint32_offset = tt::tt_metal::flatbuffer::CreateUInt32Value(builder, value);
        value_offset = uint32_offset.Union();
        value_type = tt::tt_metal::flatbuffer::RuntimeArgValue::UInt32Value;
    } else if (std::holds_alternative<Buffer*>(arg)) {
        // Create BufferGlobalId table
        Buffer* buffer = std::get<Buffer*>(arg);
        auto& ctx = LightMetalCaptureContext::Get();
        uint32_t buffer_global_id = ctx.GetGlobalId(buffer);
        auto buffer_offset = tt::tt_metal::flatbuffer::CreateBufferGlobalId(builder, buffer_global_id);
        value_offset = buffer_offset.Union();
        value_type = tt::tt_metal::flatbuffer::RuntimeArgValue::BufferGlobalId;
    } else {
        throw std::runtime_error("Unexpected variant type in createRuntimeArg");
    }

    // Create RuntimeArg
    return tt::tt_metal::flatbuffer::CreateRuntimeArg(builder, value_type, value_offset);
}

inline flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::shared_ptr<RuntimeArgs>& runtime_args) {
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>> arg_offsets;

    for (const auto& arg : *runtime_args) {
        auto runtime_arg_offset = createRuntimeArg(builder, arg);
        arg_offsets.push_back(runtime_arg_offset);
    }

    return builder.CreateVector(arg_offsets);
}

// Convert SubDeviceId::Id to uint8 array for FlatBuffer.
inline flatbuffers::Offset<flatbuffers::Vector<uint8_t>> ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::vector<uint8_t> fb_sub_device_ids(sub_device_ids.size());
    for (size_t i = 0; i < sub_device_ids.size(); ++i) {
        fb_sub_device_ids[i] = sub_device_ids[i].id;
    }
    return builder.CreateVector(fb_sub_device_ids);
}

}  // namespace tt::tt_metal
