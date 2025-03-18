// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "lightmetal/lightmetal_capture.hpp"  // For LightMetalCaptureContext
#include <tt_stl/overloaded.hpp>

namespace tt::tt_metal {

using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

flatbuffers::Offset<flatbuffer::CoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreCoord& coord) {
    return flatbuffer::CreateCoreCoord(builder, coord.x, coord.y);
}

flatbuffers::Offset<flatbuffer::CoreRange> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRange& range) {
    auto start = to_flatbuffer(builder, range.start_coord);
    auto end = to_flatbuffer(builder, range.end_coord);
    return flatbuffer::CreateCoreRange(builder, start, end);
}

flatbuffers::Offset<flatbuffer::CoreRangeSet> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRangeSet& range_set) {
    std::vector<flatbuffers::Offset<flatbuffer::CoreRange>> range_offsets;
    for (const auto& range : range_set.ranges()) {
        range_offsets.push_back(to_flatbuffer(builder, range));
    }
    auto ranges_vector = builder.CreateVector(range_offsets);
    return flatbuffer::CreateCoreRangeSet(builder, ranges_vector);
}

std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec) {
    return std::visit(
        tt::stl::overloaded{
            [&](const CoreCoord& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                return {flatbuffer::CoreSpec::CoreCoord, to_flatbuffer(builder, spec).Union()};
            },
            [&](const CoreRange& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                return {flatbuffer::CoreSpec::CoreRange, to_flatbuffer(builder, spec).Union()};
            },
            [&](const CoreRangeSet& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                return {flatbuffer::CoreSpec::CoreRangeSet, to_flatbuffer(builder, spec).Union()};
            }},
        core_spec);
}

FlatbufferCoreCoordVector to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<CoreCoord>& core_spec) {
    std::vector<flatbuffers::Offset<flatbuffer::CoreCoord>> core_offsets;
    for (const auto& coord : core_spec) {
        core_offsets.push_back(flatbuffer::CreateCoreCoord(builder, coord.x, coord.y));
    }
    return builder.CreateVector(core_offsets);
}

FlatbufferUInt32VecOfVec to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<std::vector<uint32_t>>& vec_of_vec) {
    std::vector<flatbuffers::Offset<flatbuffer::UInt32Vector>> vec_offsets;

    for (const auto& sub_vector : vec_of_vec) {
        auto values_offset = builder.CreateVector(sub_vector);
        vec_offsets.push_back(flatbuffer::CreateUInt32Vector(builder, values_offset));
    }

    return builder.CreateVector(vec_offsets);
}

// Original types defined in kernel_types.hpp
std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const DataMovementConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    auto compile_args_offset = builder.CreateVector(config.compile_args);
    auto config_offset = flatbuffer::CreateDataMovementConfig(
        builder,
        to_flatbuffer(config.processor),
        to_flatbuffer(config.noc),
        to_flatbuffer(config.noc_mode),
        compile_args_offset,
        defines_offset);

    return {flatbuffer::KernelConfig::DataMovementConfig, config_offset.Union()};
}

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ComputeConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    // Convert unpack_to_dest_mode to FlatBuffer format
    std::vector<flatbuffer::UnpackToDestMode> unpack_modes;
    for (const auto& mode : config.unpack_to_dest_mode) {
        unpack_modes.push_back(to_flatbuffer(mode));
    }
    auto unpack_modes_offset = builder.CreateVector(unpack_modes);

    auto compile_args_offset = builder.CreateVector(config.compile_args);
    auto config_offset = flatbuffer::CreateComputeConfig(
        builder,
        to_flatbuffer(config.math_fidelity),
        config.fp32_dest_acc_en,
        config.dst_full_sync_en,
        unpack_modes_offset,
        config.bfp8_pack_precise,
        config.math_approx_mode,
        compile_args_offset,
        defines_offset);

    return {flatbuffer::KernelConfig::ComputeConfig, config_offset.Union()};
}

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const EthernetConfig& config) {
    // Convert defines (map) to FlatBuffer format
    std::vector<flatbuffers::Offset<flatbuffer::DefineEntry>> defines_vector;
    for (const auto& [key, value] : config.defines) {
        auto key_offset = builder.CreateString(key);
        auto value_offset = builder.CreateString(value);
        defines_vector.push_back(flatbuffer::CreateDefineEntry(builder, key_offset, value_offset));
    }
    auto defines_offset = builder.CreateVector(defines_vector);

    auto compile_args_offset = builder.CreateVector(config.compile_args);
    auto config_offset = flatbuffer::CreateEthernetConfig(
        builder,
        to_flatbuffer(config.eth_mode),
        to_flatbuffer(config.noc),
        to_flatbuffer(config.processor),
        compile_args_offset,
        defines_offset);

    return {flatbuffer::KernelConfig::EthernetConfig, config_offset.Union()};
}

// Generic function for variant, specialized for each type above.
std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    return std::visit(
        [&](auto&& cfg) {
            using T = std::decay_t<decltype(cfg)>;
            static_assert(
                std::is_same_v<T, DataMovementConfig> || std::is_same_v<T, ComputeConfig> ||
                    std::is_same_v<T, EthernetConfig>,
                "Unhandled config type in to_flatbuffer.");
            return to_flatbuffer(builder, cfg);
        },
        config);
}

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ReaderDataMovementConfig& config) {
    const DataMovementConfig& base_config = config;  // Cast to base
    return to_flatbuffer(builder, base_config);
}

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const WriterDataMovementConfig& config) {
    const DataMovementConfig& base_config = config;  // Cast to base
    return to_flatbuffer(builder, base_config);
}

flatbuffers::Offset<flatbuffer::RuntimeArg> create_runtime_arg(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<Buffer*, uint32_t>& arg) {
    flatbuffer::RuntimeArgValue value_type;

    flatbuffers::Offset<void> value_offset = std::visit(
        tt::stl::overloaded{
            [&](uint32_t arg_value) -> flatbuffers::Offset<void> {
                value_type = flatbuffer::RuntimeArgValue::UInt32Value;
                return builder.CreateStruct(tt_metal::flatbuffer::UInt32Value{arg_value}).Union();
            },
            [&](Buffer* arg_value) -> flatbuffers::Offset<void> {
                auto& ctx = LightMetalCaptureContext::get();
                uint32_t buffer_global_id = ctx.get_global_id(arg_value);
                value_type = flatbuffer::RuntimeArgValue::BufferGlobalId;
                return builder.CreateStruct(tt_metal::flatbuffer::BufferGlobalId{buffer_global_id}).Union();
            }},
        arg);

    return flatbuffer::CreateRuntimeArg(builder, value_type, value_offset);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::RuntimeArg>>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::shared_ptr<RuntimeArgs>& runtime_args) {
    std::vector<flatbuffers::Offset<flatbuffer::RuntimeArg>> arg_offsets;

    for (const auto& arg : *runtime_args) {
        arg_offsets.push_back(create_runtime_arg(builder, arg));
    }

    return builder.CreateVector(arg_offsets);
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::vector<uint8_t> fb_sub_device_ids(sub_device_ids.size());
    for (size_t i = 0; i < sub_device_ids.size(); ++i) {
        fb_sub_device_ids[i] = *sub_device_ids[i];
    }
    return builder.CreateVector(fb_sub_device_ids);
}

}  // namespace tt::tt_metal
