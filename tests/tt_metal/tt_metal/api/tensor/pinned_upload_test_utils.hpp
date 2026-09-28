// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Sets TT_METAL_PINNED_UPLOAD_THREADS for the guard's lifetime. With 0, large uploads pin each shard whole through
// PinnedMemoryCache instead of pinning chunks.
class ScopedPinnedUploadThreads {
public:
    explicit ScopedPinnedUploadThreads(uint32_t num_threads) :
        previous_(MetalContext::instance().rtoptions().get_pinned_upload_threads()) {
        MetalContext::instance().rtoptions().set_pinned_upload_threads(num_threads);
    }
    ~ScopedPinnedUploadThreads() { MetalContext::instance().rtoptions().set_pinned_upload_threads(previous_); }

    ScopedPinnedUploadThreads(const ScopedPinnedUploadThreads&) = delete;
    ScopedPinnedUploadThreads& operator=(const ScopedPinnedUploadThreads&) = delete;

private:
    uint32_t previous_ = 0;
};

}  // namespace tt::tt_metal
