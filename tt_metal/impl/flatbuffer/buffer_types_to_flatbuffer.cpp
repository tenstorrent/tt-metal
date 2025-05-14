// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/buffer_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "lightmetal/lightmetal_capture.hpp"  // For LightMetalCaptureContext

namespace tt::tt_metal {

// Original types defined in buffer_types.hpp
flatbuffer::BufferType to_flatbuffer(BufferType type) {
    switch (type) {
        case BufferType::DRAM: return flatbuffer::BufferType::DRAM;
        case BufferType::L1: return flatbuffer::BufferType::L1;
        case BufferType::SYSTEM_MEMORY: return flatbuffer::BufferType::SystemMemory;
        case BufferType::L1_SMALL: return flatbuffer::BufferType::L1Small;
        case BufferType::TRACE: return flatbuffer::BufferType::Trace;
    }
    TT_THROW("Unsupported BufferType to flatbuffer.");
}

// Original types defined in buffer_types.hpp
flatbuffer::TensorMemoryLayout to_flatbuffer(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return flatbuffer::TensorMemoryLayout::Interleaved;
        case TensorMemoryLayout::SINGLE_BANK: return flatbuffer::TensorMemoryLayout::SingleBank;
        case TensorMemoryLayout::HEIGHT_SHARDED: return flatbuffer::TensorMemoryLayout::HeightSharded;
        case TensorMemoryLayout::WIDTH_SHARDED: return flatbuffer::TensorMemoryLayout::WidthSharded;
        case TensorMemoryLayout::BLOCK_SHARDED: return flatbuffer::TensorMemoryLayout::BlockSharded;
    }
    TT_THROW("Unsupported TensorMemoryLayout to flatbuffer.");
}

// For page sizes, keep lambda usage consistent across types.
static inline uint32_t to_flatbuffer(const uint32_t& value) { return value; }

// Original type defined in circular_buffer_config.hpp
flatbuffers::Offset<flatbuffer::CircularBufferConfig> to_flatbuffer(
    const CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder) {
    // Convert optional arrays of various types to Flatbuffers vectors.
    auto create_fb_vec_of_structs = [&](const auto& array, auto fb_type_tag) {
        using FlatBufferType = decltype(fb_type_tag);
        std::vector<FlatBufferType> vec;
        for (size_t i = 0; i < array.size(); i++) {
            if (array[i]) {
                vec.push_back(FlatBufferType{i, to_flatbuffer(*array[i])});
            }
        }
        return builder.CreateVectorOfStructs(vec);
    };

    // Convert unordered_set of uint8_t to FlatBuffer vector
    auto create_fb_vec_of_uint8 = [&](const auto& set) {
        return builder.CreateVector(std::vector<uint8_t>(set.begin(), set.end()));
    };

    // Optional shadow buffer for dynamically allocated CBs.
    auto& ctx = LightMetalCaptureContext::get();
    auto shadow_buf_global_id_offset =
        config.shadow_global_buffer
            ? flatbuffer::CreateUint32Optional(builder, ctx.get_global_id(config.shadow_global_buffer))
            : 0;

    auto globally_allocated_address =
        config.globally_allocated_address()
            ? flatbuffer::CreateUint32Optional(builder, *config.globally_allocated_address())
            : 0;

    // Create the FlatBuffer object
    return flatbuffer::CreateCircularBufferConfig(
        builder,
        config.total_size(),
        globally_allocated_address,
        create_fb_vec_of_structs(config.data_formats(), flatbuffer::CBConfigDataFormat{}),
        create_fb_vec_of_structs(config.page_sizes(), flatbuffer::CBConfigPageSize{}),
        create_fb_vec_of_structs(config.tiles(), flatbuffer::CBConfigTile{}),
        shadow_buf_global_id_offset,
        create_fb_vec_of_uint8(config.buffer_indices()),
        create_fb_vec_of_uint8(config.local_buffer_indices()),
        create_fb_vec_of_uint8(config.remote_buffer_indices()),
        config.dynamic_cb(),
        config.max_size(),
        config.buffer_size());
}

// TODO: Opportunity to share with TTNN. This was straight up copied from tensor_spec_flatbuffer.cpp

flatbuffer::ShardOrientation to_flatbuffer(ShardOrientation orientation) {
    switch (orientation) {
        case ShardOrientation::ROW_MAJOR: return flatbuffer::ShardOrientation::RowMajor;
        case ShardOrientation::COL_MAJOR: return flatbuffer::ShardOrientation::ColMajor;
    }
    TT_THROW("Unsupported ShardOrientation to flatbuffer.");
}

flatbuffer::ShardMode to_flatbuffer(ShardMode shard_mode) {
    switch (shard_mode) {
        case ShardMode::LOGICAL: return flatbuffer::ShardMode::Logical;
        case ShardMode::PHYSICAL: return flatbuffer::ShardMode::Physical;
    }
    TT_THROW("Unsupported ShardMode to flatbuffer.");
}

flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<flatbuffer::ShardShape> physical_shard_shape = 0;
    if (spec.physical_shard_shape.has_value()) {
        const auto& phys_shape = *spec.physical_shard_shape;
        physical_shard_shape = flatbuffer::CreateShardShape(builder, phys_shape[0], phys_shape[1]);
    }
    return flatbuffer::CreateShardSpec(
        builder,
        to_flatbuffer(builder, spec.grid),
        spec.shape[0],
        spec.shape[1],
        to_flatbuffer(spec.orientation),
        to_flatbuffer(spec.mode),
        physical_shard_shape);
}

flatbuffers::Offset<flatbuffer::ShardSpecBuffer> to_flatbuffer(
    const std::optional<ShardSpecBuffer>& shard_parameters, ::flatbuffers::FlatBufferBuilder& builder) {
    if (!shard_parameters.has_value()) {
        return 0;
    }

    return flatbuffer::CreateShardSpecBuffer(
        builder,
        to_flatbuffer(shard_parameters->tensor_shard_spec, builder),
        shard_parameters->page_shape[0],
        shard_parameters->page_shape[1],
        shard_parameters->tensor2d_shape_in_pages[0],
        shard_parameters->tensor2d_shape_in_pages[1]);
}

}  // namespace tt::tt_metal
