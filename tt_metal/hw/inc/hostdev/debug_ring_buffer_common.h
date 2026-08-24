// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// We use a magic value to initialize the ring buffer to, so that we can avoid printing it to the
// watcher log if no ring buffer data has been written. Choose -1 so that we can increment it to
// 0 and immediately use it as an index for the first write.
constexpr int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;
constexpr int DEBUG_RING_BUFFER_SPSC_ELEMENTS = 32;

struct debug_spsc_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_SPSC_ELEMENTS];
};

// Writers share one head, so a chatty writer evicts the others: capacity scales with the number of
// concurrent writers per arch, not just with how much history is wanted.
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR = 128;
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE = 32;

struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // thread_idx + 1; 0 means never written
};

template <int Capacity>
struct debug_mpsc_ring_buf_msg_tmpl_t {
    uint32_t head;
    debug_mpsc_ring_buf_slot_t slots[Capacity];
};

using debug_mpsc_ring_buf_msg_quasar_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR>;
using debug_mpsc_ring_buf_msg_blackhole_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE>;

// Host doesn't know the target arch at compile time; it reads through the largest variant.
static_assert(
    DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE <= DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR,
    "host view must alias the largest MPSC variant");
using debug_mpsc_ring_buf_view_t = debug_mpsc_ring_buf_msg_quasar_t;

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

// TODO: re-verify on Quasar ERISC/DRISC once runtime support for those cores lands -
// they may need to fall back to SPSC.
#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
#define DEBUG_RING_BUFFER_MPSC 1
#endif

#if defined(DEBUG_RING_BUFFER_MPSC)
#if defined(ARCH_QUASAR)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_quasar_t;
#else
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_blackhole_t;
#endif
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
constexpr uint32_t DEBUG_RING_BUFFER_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;
#else
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
