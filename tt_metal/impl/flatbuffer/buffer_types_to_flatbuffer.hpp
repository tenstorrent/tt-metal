// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "buffer_types_generated.h"
#include "lightmetal_capture.hpp"  // For LightMetalCaptureContext

namespace tt::tt_metal {

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

}  // namespace tt::tt_metal
