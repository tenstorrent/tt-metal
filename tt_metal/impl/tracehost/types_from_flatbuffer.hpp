// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//////////////////////////////////////////////////////////////
// From-flatbuffer helper functions                         //
//////////////////////////////////////////////////////////////

namespace tt::tt_metal {
inline namespace v0 {

inline BufferType FromFlatbuffer(tt::target::BufferType type) {
    switch (type) {
        case tt::target::BufferType::DRAM: return BufferType::DRAM;
        case tt::target::BufferType::L1: return BufferType::L1;
        case tt::target::BufferType::SystemMemory: return BufferType::SYSTEM_MEMORY;
        case tt::target::BufferType::L1Small: return BufferType::L1_SMALL;
        case tt::target::BufferType::Trace: return BufferType::TRACE;
        default: throw std::invalid_argument("Unknown BufferType value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::DataMovementProcessor FromFlatbuffer(tt::target::DataMovementProcessor in) {
    switch (in) {
        case tt::target::DataMovementProcessor::RISCV_0: return tt::tt_metal::DataMovementProcessor::RISCV_0;
        case tt::target::DataMovementProcessor::RISCV_1: return tt::tt_metal::DataMovementProcessor::RISCV_1;
        default: throw std::invalid_argument("Unknown DataMovementProcessor value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::NOC FromFlatbuffer(tt::target::NOC in) {
    switch (in) {
        case tt::target::NOC::NOC_0: return tt::tt_metal::NOC::NOC_0;
        case tt::target::NOC::NOC_1: return tt::tt_metal::NOC::NOC_1;
        default: throw std::invalid_argument("Invalid NOC value passed to FromFlatbuffer");
    }
}

inline tt::tt_metal::NOC_MODE FromFlatbuffer(tt::target::NOC_MODE in) {
    switch (in) {
        case tt::target::NOC_MODE::DM_DEDICATED_NOC: return tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
        case tt::target::NOC_MODE::DM_DYNAMIC_NOC: return tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC;
        default: throw std::invalid_argument("Unknown NOC_MODE value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::Eth FromFlatbuffer(tt::target::Eth in) {
    switch (in) {
        case tt::target::Eth::SENDER: return tt::tt_metal::Eth::SENDER;
        case tt::target::Eth::RECEIVER: return tt::tt_metal::Eth::RECEIVER;
        case tt::target::Eth::IDLE: return tt::tt_metal::Eth::IDLE;
        default: throw std::invalid_argument("Unknown Eth value in FromFlatbuffer()");
    }
}

inline MathFidelity FromFlatbuffer(tt::target::MathFidelity input) {
    switch (input) {
        case tt::target::MathFidelity::LoFi: return MathFidelity::LoFi;
        case tt::target::MathFidelity::HiFi2: return MathFidelity::HiFi2;
        case tt::target::MathFidelity::HiFi3: return MathFidelity::HiFi3;
        case tt::target::MathFidelity::HiFi4: return MathFidelity::HiFi4;
        case tt::target::MathFidelity::Invalid: return MathFidelity::Invalid;
        default: throw std::invalid_argument("Unknown MathFidelity value in FromFlatbuffer()");
    }
}

inline UnpackToDestMode FromFlatbuffer(tt::target::UnpackToDestMode input) {
    switch (input) {
        case tt::target::UnpackToDestMode::UnpackToDestFp32: return UnpackToDestMode::UnpackToDestFp32;
        case tt::target::UnpackToDestMode::Default: return UnpackToDestMode::Default;
        default: throw std::invalid_argument("Invalid UnpackToDestMode value passed to FromFlatbuffer");
    }
}

inline tt::DataFormat FromFlatbuffer(tt::target::DataFormat input) {
    switch (input) {
        case tt::target::DataFormat::Float32: return tt::DataFormat::Float32;
        case tt::target::DataFormat::Float16: return tt::DataFormat::Float16;
        case tt::target::DataFormat::Bfp8: return tt::DataFormat::Bfp8;
        case tt::target::DataFormat::Bfp4: return tt::DataFormat::Bfp4;
        case tt::target::DataFormat::Bfp2: return tt::DataFormat::Bfp2;
        case tt::target::DataFormat::Float16_b: return tt::DataFormat::Float16_b;
        case tt::target::DataFormat::Bfp8_b: return tt::DataFormat::Bfp8_b;
        case tt::target::DataFormat::Bfp4_b: return tt::DataFormat::Bfp4_b;
        case tt::target::DataFormat::Bfp2_b: return tt::DataFormat::Bfp2_b;
        case tt::target::DataFormat::Lf8: return tt::DataFormat::Lf8;
        case tt::target::DataFormat::Fp8_e4m3: return tt::DataFormat::Fp8_e4m3;
        case tt::target::DataFormat::Int8: return tt::DataFormat::Int8;
        case tt::target::DataFormat::Tf32: return tt::DataFormat::Tf32;
        case tt::target::DataFormat::UInt8: return tt::DataFormat::UInt8;
        case tt::target::DataFormat::UInt16: return tt::DataFormat::UInt16;
        case tt::target::DataFormat::Int32: return tt::DataFormat::Int32;
        case tt::target::DataFormat::UInt32: return tt::DataFormat::UInt32;
        case tt::target::DataFormat::RawUInt8: return tt::DataFormat::RawUInt8;
        case tt::target::DataFormat::RawUInt16: return tt::DataFormat::RawUInt16;
        case tt::target::DataFormat::RawUInt32: return tt::DataFormat::RawUInt32;
        case tt::target::DataFormat::Invalid: return tt::DataFormat::Invalid;
        default: throw std::invalid_argument("Unknown DataFormat value in FromFlatbuffer()");
    }
}

inline std::variant<CoreCoord, CoreRange, CoreRangeSet> FromFlatbuffer(
    const tt::target::CoreSpec core_spec, const void* flatbuffer_union) {
    switch (core_spec) {
        case tt::target::CoreSpec::CoreCoord: {
            auto core_coord = static_cast<const tt::target::CoreCoord*>(flatbuffer_union);
            if (!core_coord) {
                throw std::runtime_error("Invalid CoreCoord data");
            }
            return CoreCoord{core_coord->x(), core_coord->y()};
        }
        case tt::target::CoreSpec::CoreRange: {
            auto core_range = static_cast<const tt::target::CoreRange*>(flatbuffer_union);
            if (!core_range) {
                throw std::runtime_error("Invalid CoreRange data");
            }
            return CoreRange{
                {core_range->start()->x(), core_range->start()->y()}, {core_range->end()->x(), core_range->end()->y()}};
        }
        case tt::target::CoreSpec::CoreRangeSet: {
            auto core_range_set = static_cast<const tt::target::CoreRangeSet*>(flatbuffer_union);
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

inline DataMovementConfig FromFlatbuffer(const tt::target::DataMovementConfig* fb_config) {
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

inline ComputeConfig FromFlatbuffer(const tt::target::ComputeConfig* fb_config) {
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

inline EthernetConfig FromFlatbuffer(const tt::target::EthernetConfig* fb_config) {
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
    const tt::target::KernelConfig config_type, const void* flatbuffer_union) {
    switch (config_type) {
        case tt::target::KernelConfig::DataMovementConfig:
            return FromFlatbuffer(static_cast<const tt::target::DataMovementConfig*>(flatbuffer_union));
        case tt::target::KernelConfig::ComputeConfig:
            return FromFlatbuffer(static_cast<const tt::target::ComputeConfig*>(flatbuffer_union));
        case tt::target::KernelConfig::EthernetConfig:
            return FromFlatbuffer(static_cast<const tt::target::EthernetConfig*>(flatbuffer_union));
        default: throw std::runtime_error("Unhandled KernelConfig type in FromFlatbuffer.");
    }
}

inline Tile FromFlatbuffer(const tt::target::Tile* tile_fb) {
    if (!tile_fb) {
        throw std::runtime_error("Invalid Tile FlatBuffer object");
    }

    // Convert FlatBuffer vectors to std::array
    std::array<uint32_t, 2> tile_shape = {tile_fb->tile_shape()->Get(0), tile_fb->tile_shape()->Get(1)};
    std::array<uint32_t, 2> face_shape = {tile_fb->face_shape()->Get(0), tile_fb->face_shape()->Get(1)};

    // Create and return the Tile object, explicitly initializing the members
    Tile tile;
    tile.tile_shape = tile_shape;
    tile.face_shape = face_shape;
    tile.tile_hw = tile_fb->tile_hw();
    tile.face_hw = tile_fb->face_hw();
    tile.num_faces = tile_fb->num_faces();
    tile.partial_face = tile_fb->partial_face();
    tile.narrow_tile = tile_fb->narrow_tile();
    tile.transpose_within_face = tile_fb->transpose_within_face();
    tile.transpose_of_faces = tile_fb->transpose_of_faces();

    return tile;
}

inline std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> FromFlatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::Tile>>* tiles_fb) {
    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles = {};
    if (tiles_fb) {
        for (size_t i = 0; i < tiles_fb->size() && i < NUM_CIRCULAR_BUFFERS; ++i) {
            tiles[i] = FromFlatbuffer(tiles_fb->Get(i));
        }
    }
    return tiles;
}

inline CircularBufferConfig FromFlatbuffer(
    const tt::target::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer) {
    if (!config_fb) {
        throw std::runtime_error("Invalid CircularBufferConfig FlatBuffer object");
    }

    // Create a CircularBufferConfig. Constructor doesn't matter much, since we serialized all
    // members, will deserialize them here to get fully formed object.
    CircularBufferConfig config(0, {});
    config.total_size_ = config_fb->total_size();

    // Note: std::optional is not supported by FlatBuffers, so nullopt was serialized as value 0 in FlatBuffer.
    config.globally_allocated_address_ = config_fb->globally_allocated_address() == 0
                                             ? std::nullopt
                                             : std::optional<uint32_t>(config_fb->globally_allocated_address());

    if (config_fb->data_formats()) {
        for (auto entry : *config_fb->data_formats()) {
            config.data_formats_[entry->index()] = FromFlatbuffer(entry->format());
        }
    }

    if (config_fb->page_sizes()) {
        for (auto entry : *config_fb->page_sizes()) {
            config.page_sizes_[entry->index()] = entry->size();
        }
    }

    config.tiles_ = FromFlatbuffer(config_fb->tiles());
    config.shadow_global_buffer = shadow_global_buffer;

    if (config_fb->buffer_indices()) {
        config.buffer_indices_.insert(config_fb->buffer_indices()->begin(), config_fb->buffer_indices()->end());
    }

    if (config_fb->local_buffer_indices()) {
        config.local_buffer_indices_.insert(
            config_fb->local_buffer_indices()->begin(), config_fb->local_buffer_indices()->end());
    }

    if (config_fb->remote_buffer_indices()) {
        config.remote_buffer_indices_.insert(
            config_fb->remote_buffer_indices()->begin(), config_fb->remote_buffer_indices()->end());
    }

    config.dynamic_cb_ = config_fb->dynamic_cb();
    config.max_size_ = config_fb->max_size();
    config.buffer_size_ = config_fb->buffer_size();

    return config;
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
