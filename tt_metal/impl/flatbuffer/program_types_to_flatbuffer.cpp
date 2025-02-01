// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include <overloaded.hpp>
namespace tt::tt_metal {

// Original types defined in core_coord.hpp
std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec) {
    return std::visit(
        tt::stl::overloaded{
            [&](const CoreCoord& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                auto core_coord = flatbuffer::CreateCoreCoord(builder, spec.x, spec.y);
                return {flatbuffer::CoreSpec::CoreCoord, core_coord.Union()};
            },
            [&](const CoreRange& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                auto start = flatbuffer::CreateCoreCoord(builder, spec.start_coord.x, spec.start_coord.y);
                auto end = flatbuffer::CreateCoreCoord(builder, spec.end_coord.x, spec.end_coord.y);
                auto core_range = flatbuffer::CreateCoreRange(builder, start, end);
                return {flatbuffer::CoreSpec::CoreRange, core_range.Union()};
            },
            [&](const CoreRangeSet& spec) -> std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> {
                std::vector<flatbuffers::Offset<flatbuffer::CoreRange>> range_offsets;
                for (const auto& range : spec.ranges()) {
                    auto start = flatbuffer::CreateCoreCoord(builder, range.start_coord.x, range.start_coord.y);
                    auto end = flatbuffer::CreateCoreCoord(builder, range.end_coord.x, range.end_coord.y);
                    range_offsets.push_back(flatbuffer::CreateCoreRange(builder, start, end));
                }
                auto ranges_vector = builder.CreateVector(range_offsets);
                auto core_range_set = flatbuffer::CreateCoreRangeSet(builder, ranges_vector);
                return {flatbuffer::CoreSpec::CoreRangeSet, core_range_set.Union()};
            }},
        core_spec);
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
                // auto& ctx = LightMetalCaptureContext::Get();
                // uint32_t buffer_global_id = ctx.GetGlobalId(arg_value);
                // TODO (kmabee) - Uncomment above code once capture library is merged. Temp hack here for now.
                uint32_t buffer_global_id = 0;
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
        fb_sub_device_ids[i] = sub_device_ids[i].id;
    }
    return builder.CreateVector(fb_sub_device_ids);
}

}  // namespace tt::tt_metal
