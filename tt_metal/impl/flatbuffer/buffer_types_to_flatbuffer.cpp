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
}

flatbuffers::Offset<flatbuffer::CircularBufferConfig> to_flatbuffer(
    const CircularBufferConfig& config, flatbuffers::FlatBufferBuilder& builder) {
    // Note: std::optional not supported by FlatBuffers, so serialize it as a uint32_t with a default val 0
    auto global_address = config.globally_allocated_address_ ? *config.globally_allocated_address_ : 0;

    // Note: std::optional data_formats array represented as vec of k (idx) v (format) pairs
    std::vector<flatbuffer::CBConfigDataFormat> data_formats_vec;
    for (size_t i = 0; i < config.data_formats_.size(); i++) {
        if (config.data_formats_[i]) {
            data_formats_vec.push_back({i, to_flatbuffer(*config.data_formats_[i])});
        }
    }
    auto data_formats_fb = builder.CreateVectorOfStructs(data_formats_vec);

    // Note: std::optional page_sizes array represented as vec of k (idx) v (size) pairs
    std::vector<flatbuffer::CBConfigPageSize> page_sizes_vec;
    for (size_t i = 0; i < config.page_sizes_.size(); i++) {
        if (config.page_sizes_[i]) {
            page_sizes_vec.push_back({i, *config.page_sizes_[i]});
        }
    }
    auto page_sizes_fb = builder.CreateVectorOfStructs(page_sizes_vec);

    // Note: std::optional Tiles array represented as vec of k (idx) v (Tile) pairs
    std::vector<flatbuffer::CBConfigTile> tiles_vec;
    for (size_t i = 0; i < config.tiles_.size(); i++) {
        if (config.tiles_[i]) {
            tiles_vec.push_back({i, to_flatbuffer(*config.tiles_[i])});
        }
    }
    auto tiles_fb = builder.CreateVectorOfStructs(tiles_vec);

    // Optional shadow buffer for dynamically allocated CBs, get global_id or use 0 as none/nullptr.
    // auto& ctx = LightMetalCaptureContext::Get();
    // auto shadow_buf_global_id = config.shadow_global_buffer ? ctx.GetGlobalId(config.shadow_global_buffer) : 0;
    // TODO (kmabee) - Uncomment above code once capture library is merged. Temp hack here for now.
    uint32_t shadow_buf_global_id = 0;

    // Serialize buffer_indices_ and variants as a FlatBuffer vector
    std::vector<uint8_t> buf_ind_vec(config.buffer_indices_.begin(), config.buffer_indices_.end());
    auto buffer_indices_fb = builder.CreateVector(buf_ind_vec);
    std::vector<uint8_t> local_buf_ind_vec(config.local_buffer_indices_.begin(), config.local_buffer_indices_.end());
    auto local_buffer_indices_fb = builder.CreateVector(local_buf_ind_vec);
    std::vector<uint8_t> remote_buf_ind_vec(config.remote_buffer_indices_.begin(), config.remote_buffer_indices_.end());
    auto remote_buffer_indices_fb = builder.CreateVector(remote_buf_ind_vec);

    // Create the FlatBuffer object
    return flatbuffer::CreateCircularBufferConfig(
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

}  // namespace tt::tt_metal
