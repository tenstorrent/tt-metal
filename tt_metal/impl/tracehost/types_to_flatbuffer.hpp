// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//////////////////////////////////////////////////////////////
// To-flatbuffer helper functions                           //
//////////////////////////////////////////////////////////////

namespace tt::tt_metal {

// Original types defined in buffer_constants.hpp
inline tt::tt_metal::flatbuffer::BufferType ToFlatbuffer(BufferType type) {
    switch (type) {
        case BufferType::DRAM: return tt::tt_metal::flatbuffer::BufferType::DRAM;
        case BufferType::L1: return tt::tt_metal::flatbuffer::BufferType::L1;
        case BufferType::SYSTEM_MEMORY: return tt::tt_metal::flatbuffer::BufferType::SystemMemory;
        case BufferType::L1_SMALL: return tt::tt_metal::flatbuffer::BufferType::L1Small;
        case BufferType::TRACE: return tt::tt_metal::flatbuffer::BufferType::Trace;
        default: throw std::invalid_argument("Unknown BufferType value in ToFlatbuffer()");
    }
}

// Original types defined in buffer_constants.hpp
inline tt::tt_metal::flatbuffer::TensorMemoryLayout ToFlatbuffer(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::Interleaved;
        case TensorMemoryLayout::SINGLE_BANK: return tt::tt_metal::flatbuffer::TensorMemoryLayout::SingleBank;
        case TensorMemoryLayout::HEIGHT_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::HeightSharded;
        case TensorMemoryLayout::WIDTH_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::WidthSharded;
        case TensorMemoryLayout::BLOCK_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::BlockSharded;
        default: throw std::invalid_argument("Unknown TensorMemoryLayout value in ToFlatbuffer()");
    }
}

// Original types defined in data_types.hpp
inline tt::tt_metal::flatbuffer::DataMovementProcessor ToFlatbuffer(tt::tt_metal::DataMovementProcessor in) {
    switch (in) {
        case tt::tt_metal::DataMovementProcessor::RISCV_0:
            return tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_0;
        case tt::tt_metal::DataMovementProcessor::RISCV_1:
            return tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_1;
        default: throw std::invalid_argument("Unknown DataMovementProcessor value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::NOC ToFlatbuffer(tt::tt_metal::NOC in) {
    switch (in) {
        case tt::tt_metal::NOC::NOC_0: return tt::tt_metal::flatbuffer::NOC::NOC_0;
        case tt::tt_metal::NOC::NOC_1: return tt::tt_metal::flatbuffer::NOC::NOC_1;
        default: throw std::invalid_argument("Invalid NOC value passed to ToFlatbuffer");
    }
}

inline tt::tt_metal::flatbuffer::NOC_MODE ToFlatbuffer(tt::tt_metal::NOC_MODE in) {
    switch (in) {
        case tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC: return tt::tt_metal::flatbuffer::NOC_MODE::DM_DEDICATED_NOC;
        case tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC: return tt::tt_metal::flatbuffer::NOC_MODE::DM_DYNAMIC_NOC;
        default: throw std::invalid_argument("Unknown NOC_MODE value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::Eth ToFlatbuffer(tt::tt_metal::Eth in) {
    switch (in) {
        case tt::tt_metal::Eth::SENDER: return tt::tt_metal::flatbuffer::Eth::SENDER;
        case tt::tt_metal::Eth::RECEIVER: return tt::tt_metal::flatbuffer::Eth::RECEIVER;
        case tt::tt_metal::Eth::IDLE: return tt::tt_metal::flatbuffer::Eth::IDLE;
        default: throw std::invalid_argument("Unknown Eth value in ToFlatbuffer()");
    }
}

// Original types defined in base_types.hpp
inline tt::tt_metal::flatbuffer::MathFidelity ToFlatbuffer(MathFidelity input) {
    switch (input) {
        case MathFidelity::LoFi: return tt::tt_metal::flatbuffer::MathFidelity::LoFi;
        case MathFidelity::HiFi2: return tt::tt_metal::flatbuffer::MathFidelity::HiFi2;
        case MathFidelity::HiFi3: return tt::tt_metal::flatbuffer::MathFidelity::HiFi3;
        case MathFidelity::HiFi4: return tt::tt_metal::flatbuffer::MathFidelity::HiFi4;
        case MathFidelity::Invalid: return tt::tt_metal::flatbuffer::MathFidelity::Invalid;
        default: throw std::invalid_argument("Unknown MathFidelity value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::UnpackToDestMode ToFlatbuffer(UnpackToDestMode input) {
    switch (input) {
        case UnpackToDestMode::UnpackToDestFp32: return tt::tt_metal::flatbuffer::UnpackToDestMode::UnpackToDestFp32;
        case UnpackToDestMode::Default: return tt::tt_metal::flatbuffer::UnpackToDestMode::Default;
        default: throw std::invalid_argument("Invalid UnpackToDestMode value passed to ToFlatbuffer");
    }
}

// Original types defined in tt_backend_api_types.hpp
inline tt::tt_metal::flatbuffer::DataFormat ToFlatbuffer(tt::DataFormat input) {
    switch (input) {
        case tt::DataFormat::Float32: return tt::tt_metal::flatbuffer::DataFormat::Float32;
        case tt::DataFormat::Float16: return tt::tt_metal::flatbuffer::DataFormat::Float16;
        case tt::DataFormat::Bfp8: return tt::tt_metal::flatbuffer::DataFormat::Bfp8;
        case tt::DataFormat::Bfp4: return tt::tt_metal::flatbuffer::DataFormat::Bfp4;
        case tt::DataFormat::Bfp2: return tt::tt_metal::flatbuffer::DataFormat::Bfp2;
        case tt::DataFormat::Float16_b: return tt::tt_metal::flatbuffer::DataFormat::Float16_b;
        case tt::DataFormat::Bfp8_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp8_b;
        case tt::DataFormat::Bfp4_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp4_b;
        case tt::DataFormat::Bfp2_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp2_b;
        case tt::DataFormat::Lf8: return tt::tt_metal::flatbuffer::DataFormat::Lf8;
        case tt::DataFormat::Fp8_e4m3: return tt::tt_metal::flatbuffer::DataFormat::Fp8_e4m3;
        case tt::DataFormat::Int8: return tt::tt_metal::flatbuffer::DataFormat::Int8;
        case tt::DataFormat::Tf32: return tt::tt_metal::flatbuffer::DataFormat::Tf32;
        case tt::DataFormat::UInt8: return tt::tt_metal::flatbuffer::DataFormat::UInt8;
        case tt::DataFormat::UInt16: return tt::tt_metal::flatbuffer::DataFormat::UInt16;
        case tt::DataFormat::Int32: return tt::tt_metal::flatbuffer::DataFormat::Int32;
        case tt::DataFormat::UInt32: return tt::tt_metal::flatbuffer::DataFormat::UInt32;
        case tt::DataFormat::RawUInt8: return tt::tt_metal::flatbuffer::DataFormat::RawUInt8;
        case tt::DataFormat::RawUInt16: return tt::tt_metal::flatbuffer::DataFormat::RawUInt16;
        case tt::DataFormat::RawUInt32: return tt::tt_metal::flatbuffer::DataFormat::RawUInt32;
        case tt::DataFormat::Invalid: return tt::tt_metal::flatbuffer::DataFormat::Invalid;
        default: throw std::invalid_argument("Unknown DataFormat value in ToFlatbuffer()");
    }
}

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

inline flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile> ToFlatbuffer(
    const Tile& tile, flatbuffers::FlatBufferBuilder& builder) {
    auto tile_shape_fb = builder.CreateVector(tile.get_tile_shape().data(), tile.get_tile_shape().size());
    auto face_shape_fb = builder.CreateVector(tile.get_face_shape().data(), tile.get_face_shape().size());

    return tt::tt_metal::flatbuffer::CreateTile(
        builder,
        tile_shape_fb,
        face_shape_fb,
        tile.get_tile_hw(),
        tile.get_face_hw(),
        tile.get_num_faces(),
        tile.get_partial_face(),
        tile.get_narrow_tile(),
        tile.get_transpose_within_face(),
        tile.get_transpose_of_faces());
}

inline flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>>> ToFlatbuffer(
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles, flatbuffers::FlatBufferBuilder& builder) {
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>> tiles_fb;
    for (const auto& tile_opt : tiles) {
        if (tile_opt) {
            tiles_fb.push_back(ToFlatbuffer(*tile_opt, builder));
        }
    }

    return builder.CreateVector(tiles_fb);
}

inline flatbuffers::Offset<tt::tt_metal::flatbuffer::CircularBufferConfig> ToFlatbuffer(
    const tt::tt_metal::CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder) {
    // Note: std::optional not supported by FlatBuffers, so serialize it as a uint32_t with a default val 0
    auto global_address = config.globally_allocated_address_ ? *config.globally_allocated_address_ : 0;

    // Note: std::optional data_formats array represented as vec of k (idx) v (format) pairs
    std::vector<tt::tt_metal::flatbuffer::CBConfigDataFormat> data_formats_vec;
    for (size_t i = 0; i < config.data_formats_.size(); i++) {
        if (config.data_formats_[i]) {
            data_formats_vec.push_back({i, ToFlatbuffer(*config.data_formats_[i])});
        }
    }
    auto data_formats_fb = builder.CreateVectorOfStructs(data_formats_vec);

    // Note: std::optional page_sizes array represented as vec of k (idx) v (size) pairs
    std::vector<tt::tt_metal::flatbuffer::CBConfigPageSize> page_sizes_vec;
    for (size_t i = 0; i < config.page_sizes_.size(); i++) {
        if (config.page_sizes_[i]) {
            page_sizes_vec.push_back({i, *config.page_sizes_[i]});
        }
    }
    auto page_sizes_fb = builder.CreateVectorOfStructs(page_sizes_vec);
    auto tiles_fb = ToFlatbuffer(config.tiles_, builder);

    // Optional shadow buffer for dynamically allocated CBs, get global_id or use 0 as none/nullptr.
    auto& ctx = LightMetalCaptureContext::Get();
    auto shadow_buf_global_id = config.shadow_global_buffer ? ctx.GetGlobalId(config.shadow_global_buffer) : 0;

    // Serialize buffer_indices_ and variants as a FlatBuffer vector
    std::vector<uint8_t> buf_ind_vec(config.buffer_indices_.begin(), config.buffer_indices_.end());
    auto buffer_indices_fb = builder.CreateVector(buf_ind_vec);
    std::vector<uint8_t> local_buf_ind_vec(config.local_buffer_indices_.begin(), config.local_buffer_indices_.end());
    auto local_buffer_indices_fb = builder.CreateVector(local_buf_ind_vec);
    std::vector<uint8_t> remote_buf_ind_vec(config.remote_buffer_indices_.begin(), config.remote_buffer_indices_.end());
    auto remote_buffer_indices_fb = builder.CreateVector(remote_buf_ind_vec);

    // Create the FlatBuffer object
    return tt::tt_metal::flatbuffer::CreateCircularBufferConfig(
        builder,
        config.total_size_,
        global_address,
        data_formats_fb,
        page_sizes_fb,
        tiles_fb,
        shadow_buf_global_id,
        buffer_indices_fb,
        local_buffer_indices_fb,
        remote_buffer_indices_fb,
        config.dynamic_cb_,
        config.max_size_,
        config.buffer_size_);
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
