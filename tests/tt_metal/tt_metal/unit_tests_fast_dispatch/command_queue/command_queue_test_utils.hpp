/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"

struct BufferConfig {
    u32 num_pages;
    u32 page_size;
    BufferStorage buftype;
};

inline pair<Buffer, vector<u32>> EnqueueWriteBuffer_prior_to_wrap(Device* device, CommandQueue& cq, const BufferConfig& config) {
    // This function just enqueues a buffer (which should be large in the config)
    // write as a precursor to testing the wrap mechanism
    size_t buf_size = config.num_pages * config.page_size;
    Buffer buffer(device, buf_size, config.page_size, config.buftype);

    vector<u32> src = create_random_vector_of_bfloat16(
      buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    EnqueueWriteBuffer(cq, buffer, src, false);
    return std::make_pair(std::move(buffer), src);
}
