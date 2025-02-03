// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/buffer_types_to_flatbuffer.hpp"

namespace tt::tt_metal {

// Original types defined in buffer_constants.hpp
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

// Original types defined in buffer_constants.hpp
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

// Original type defined in circular_buffer_types.hpp
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

    // Optional shadow buffer for dynamically allocated CBs, get global_id or use 0 as none/nullptr.
    // auto& ctx = LightMetalCaptureContext::Get();
    // auto shadow_buf_global_id = config.shadow_global_buffer ? ctx.GetGlobalId(config.shadow_global_buffer) : 0;
    // TODO (kmabee) - Uncomment above code once capture library is merged. Temp hack here for now.
    uint32_t shadow_buf_global_id = 0;

    // Create the FlatBuffer object
    return flatbuffer::CreateCircularBufferConfig(
        builder,
        config.total_size(),
        config.globally_allocated_address().value_or(0),  // Optional, default 0 if nullopt.
        create_fb_vec_of_structs(config.data_formats(), flatbuffer::CBConfigDataFormat{}),
        create_fb_vec_of_structs(config.page_sizes(), flatbuffer::CBConfigPageSize{}),
        create_fb_vec_of_structs(config.tiles(), flatbuffer::CBConfigTile{}),
        shadow_buf_global_id,
        create_fb_vec_of_uint8(config.buffer_indices()),
        create_fb_vec_of_uint8(config.local_buffer_indices()),
        create_fb_vec_of_uint8(config.remote_buffer_indices()),
        config.dynamic_cb(),
        config.max_size(),
        config.buffer_size());
}

}  // namespace tt::tt_metal
