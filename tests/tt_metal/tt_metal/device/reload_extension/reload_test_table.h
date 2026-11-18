// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Table walked by Metal's test firmware extension (runtime_reload.h next to this file). Shared by the
// host test and the extension; one table per reload core, at the same L1 address on each.

#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t RELOAD_TEST_MAX_IMAGES = 4;
constexpr uint32_t RELOAD_TEST_ADVANCE = 1;
constexpr uint32_t RELOAD_TEST_STOP = 2;

// reload_test_table_t::phase
constexpr uint32_t RELOAD_TEST_PHASE_ENTERED = 1;
constexpr uint32_t RELOAD_TEST_PHASE_QUIESCED = 2;
constexpr uint32_t RELOAD_TEST_PHASE_BARRIER1 = 3;
constexpr uint32_t RELOAD_TEST_PHASE_FETCHED = 4;
constexpr uint32_t RELOAD_TEST_PHASE_BARRIER2 = 5;
constexpr uint32_t RELOAD_TEST_PHASE_INSTALLED = 6;
constexpr uint32_t RELOAD_TEST_PHASE_STOPPED = 7;

struct reload_test_image_t {
    uint32_t block_src;   // L1 address of the image's kernel-config block on the holder core
    uint32_t block_bytes;
    uint32_t config_src;  // L1 address of its captured kernel_config_msg_t on the holder core
    uint32_t config_bytes;
};

struct reload_test_table_t {
    uint32_t num_images;
    uint32_t num_rounds;  // bounded walks only
    uint32_t src_x;  // virtual coordinates of the core holding the images
    uint32_t src_y;
    uint32_t scratch_addr;  // 16 B aligned landing area for a captured kernel_config_msg_t
    uint32_t is_coord;
    uint32_t num_cores;  // barrier arrivals, the coordinator included
    uint32_t coord_x;
    uint32_t coord_y;
    uint32_t mcast_start_x;
    uint32_t mcast_start_y;
    uint32_t mcast_end_x;
    uint32_t mcast_end_y;
    uint32_t mcast_dests;  // cores the release multicast reaches, the coordinator excluded
    uint32_t installs;
    uint32_t persistent;  // walk until the host sets stop, read by the coordinator at each round boundary
    // Written from outside this core while it runs, each in its own 16 B.
    uint32_t arrive;
    uint32_t pad1[3];
    uint32_t release;
    uint32_t pad2[3];
    uint32_t stop;
    uint32_t phase;  // the last step reached, for the host to read when a walk stalls
    uint32_t pad3[2];
    reload_test_image_t images[RELOAD_TEST_MAX_IMAGES];
};

static_assert(offsetof(reload_test_table_t, arrive) % 16 == 0);
static_assert(offsetof(reload_test_table_t, release) % 16 == 0);
static_assert(offsetof(reload_test_table_t, stop) % 16 == 0);
