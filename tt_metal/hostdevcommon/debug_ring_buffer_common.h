// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// We use a magic value to initialize the ring buffer to, so that we can avoid printing it to the
// watcher log if no ring buffer data has been written. Choose -1 so that we can increment it to
// 0 and immediately use it as an index for the first write.
constexpr static int DEBUG_RING_BUFFER_STARTING_INDEX = -1;

constexpr static int RING_BUFFER_ELEMENTS = RING_BUFFER_SIZE / sizeof(uint32_t) - 1;
struct DebugRingBufMemLayout {
    int32_t current_ptr;
    uint32_t data[RING_BUFFER_ELEMENTS];
};
static_assert(sizeof(DebugRingBufMemLayout) == RING_BUFFER_SIZE);
