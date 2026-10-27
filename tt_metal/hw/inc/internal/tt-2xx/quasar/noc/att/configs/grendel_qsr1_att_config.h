// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att.h"
#include "internal/tt-2xx/quasar/noc/att/att_address.h"

/**
 * @file
 * @brief The QSR1 boot map, transcribed: the mask-table windows tt-metal uses
 * (worker slot 4, GDDR slot 5, loopback-scratch slot 13, per-tile config
 * slot 14) and their selector/endpoint tables, copied row-for-row from the
 * generated grendelemulation tile-NIU payload.
 *
 * Data only. Address resolution over this data lives in noc/att/att_address.h.
 */
namespace grendel_qsr1_att_config {

// The worker frame below is the coordinate frame kernels receive today: the
// soc descriptor's physical functional_workers positions (quasar_32_arch.yaml,
// workers 2-2..9-5), passed through unchanged because Quasar NOC translation
// is the identity placeholder (tenstorrent/tt-umd#2494). Revisit together with
// that issue if the
// finalized translated scheme moves the worker frame.
constexpr std::uint32_t ATT_WORKER_API_ORIGIN_X = 2;
constexpr std::uint32_t ATT_WORKER_API_ORIGIN_Y = 2;
constexpr std::uint32_t ATT_WORKER_GRID_X = 8;
constexpr std::uint32_t ATT_WORKER_GRID_Y = 4;

// Frame offset from the soc descriptor's coordinates (what kernels receive) to
// the live package frame of the endpoint words and NOC_NODE_ID: worker 2-2
// reads NOC_NODE_ID (4,4). Same value as UMD's package offset for this map.
constexpr std::uint32_t ATT_NODE_ID_OFFSET_X = 2;
constexpr std::uint32_t ATT_NODE_ID_OFFSET_Y = 2;

// Worker selectors are row-major logical coordinates.
// clang-format off
constexpr std::uint8_t ATT_WORKER_SELECTORS[] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};

// The contents of hardware endpoint-table rows 128..159 (the worker window),
// one packed (y << 6) | x coordinate word per selector, in the package-NoC
// frame. The hardware only ever uses its table in the forward direction
// (selector -> tile); software needs this transcription for the two things
// the hardware cannot answer:
//  - the INVERSE lookup: a core knows itself only by its physical NOC_NODE_ID,
//    so recognizing "my own selector" (is_local on worker-window operands)
//    requires searching coordinate -> selector. The physical layout is
//    irregular, so this is a table by nature, never arithmetic.
//  - verification: the image test compares these words row-for-row against the
//    generated boot image, so the config and what boot actually programs
//    cannot silently disagree.
constexpr std::uint16_t ATT_WORKER_ENDPOINT_WORDS[] = {
    0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10a, 0x10b,
    0x144, 0x145, 0x146, 0x147, 0x148, 0x149, 0x14a, 0x14b,
    0x184, 0x185, 0x186, 0x187, 0x188, 0x189, 0x18a, 0x18b,
    0x1c4, 0x1c5, 0x1c6, 0x1c7, 0x1c8, 0x1c9, 0x1ca, 0x1cb,
};

// The contents of hardware endpoint-table rows 256..315 (the full-tile config
// window), indexed by the full-tile selector: the same 32 workers plus the
// perimeter/Dispatch-capable initiators. Needed for the same inverse lookup as
// the worker table (a perimeter core resolving its own identity can only
// appear here) and cross-checked against the boot image by the image test.
constexpr std::uint16_t ATT_FULL_TILE_ENDPOINT_WORDS[] = {
    0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10a, 0x10b,
    0x144, 0x145, 0x146, 0x147, 0x148, 0x149, 0x14a, 0x14b,
    0x184, 0x185, 0x186, 0x187, 0x188, 0x189, 0x18a, 0x18b,
    0x1c4, 0x1c5, 0x1c6, 0x1c7, 0x1c8, 0x1c9, 0x1ca, 0x1cb,
    0x103, 0x143, 0x183, 0x1c3, 0x204, 0x205, 0x206, 0x207,
    0x208, 0x209, 0x20a, 0x20b, 0x1cc, 0x18c, 0x14c, 0x10c,
    0x0cb, 0x0ca, 0x0c9, 0x0c8, 0x0c7, 0x0c6, 0x0c5, 0x0c4,
    0x20c, 0x0cc, 0x203, 0x0c3,
};
// clang-format on

constexpr noc_att::Window LOOPBACK_SCRATCH_WINDOW{
    .compare = 0x100000ull,
    .mask_bits = 20,
    .endpoint_shift = 0,
    .endpoint_size = 0,
    .endpoint_table_offset = 256,
    .translate_address = false,
};  // mask-table slot 13

constexpr noc_att::Window WORKER_WINDOW{
    .compare = 0x10000000000ull,
    .mask_bits = 30,
    .endpoint_shift = 24,
    .endpoint_size = 6,
    .endpoint_table_offset = 128,
    .translate_address = true,
};  // mask-table slot 4

constexpr noc_att::Window DRAM_WINDOW{
    .compare = 0x1000000000000ull,
    .mask_bits = 38,
    .endpoint_shift = 33,
    .endpoint_size = 5,
    .endpoint_table_offset = 96,
    .translate_address = false,
};  // mask-table slot 5

constexpr noc_att::Window TILE_WINDOW{
    .compare = 0x1800000000ull,
    .mask_bits = 33,
    .endpoint_shift = 27,
    .endpoint_size = 6,
    .endpoint_table_offset = 256,
    .translate_address = true,
};  // mask-table slot 14

// The local (self) window: endpoint 256 (full-tile selector 0) is boot-patched
// per tile to the tile's own coordinate, so this initiator's L1 is reached
// through the tile window at selector 0.
constexpr noc_att::Window LOCAL_WINDOW = TILE_WINDOW;

// Every operand for this initiator's own L1 is LOCAL_WINDOW_BASE | local_address.
// The ATT enablement surfaces this value as the NOC_ATT_LOCAL_WINDOW_BASE macro
// that noc_nonblocking_api_v3.h requires, with a static_assert against this
// constant so the two cannot drift apart.
constexpr std::uint64_t LOCAL_WINDOW_BASE = LOCAL_WINDOW.make_address(/*selector*/ 0, /*local_address*/ 0);
static_assert(LOCAL_WINDOW_BASE == 0x1800000000ull);

// DRAM window selector per logical DRAM channel: boot rows 96..99 are the Mimir
// d2d0 ingress nodes of channels 0..3 in that order, so channel N is selector N.
// The allocator binds one interleaved bank per channel, so logical bank N is
// selector N too.
constexpr std::uint8_t ATT_LOGICAL_DRAM_SELECTORS[] = {0, 1, 2, 3};

// DRAM endpoint words by DRAM-window selector (boot rows 96..127): lane A (d2d0
// ingress) at selectors 0..3, lane B (d2d1) at 16..19, the rest unprogrammed.
// A host coordinate naming a DRAM tile resolves through these to the DRAM window.
// clang-format off
constexpr std::uint16_t ATT_DRAM_ENDPOINT_WORDS[] = {
    0x246, 0x24a, 0x089, 0x085, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x247, 0x24b, 0x088, 0x084, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
// clang-format on

// Dispatch-capable tiles, keyed by the descriptor-frame coordinate the host
// publishes (go-message master coordinates, the dispatch core descriptors).
// The three DE tiles reach their L1 through the full-tile window (boot rows
// 312..314 = full-tile selectors 56..58); the interim Tensix dispatch tile is
// an ordinary worker-window selector. Each entry is checked against the
// endpoint word its selector programs.
inline constexpr noc_att::MapData::DispatchEntry DISPATCH_ENTRIES[] = {
    {.x = 10, .y = 6, .selector = 56, .window = noc_att::WindowClass::FullTile},
    {.x = 10, .y = 1, .selector = 57, .window = noc_att::WindowClass::FullTile},
    {.x = 1, .y = 6, .selector = 58, .window = noc_att::WindowClass::FullTile},
    {.x = 9, .y = 5, .selector = 31, .window = noc_att::WindowClass::Worker},
    {.x = 2, .y = 5, .selector = 24, .window = noc_att::WindowClass::Worker},
    {.x = 9, .y = 2, .selector = 7, .window = noc_att::WindowClass::Worker},
};

constexpr bool dispatch_entries_match_endpoint_words() {
    for (const noc_att::MapData::DispatchEntry& entry : DISPATCH_ENTRIES) {
        const std::uint32_t live_word = ((entry.y + ATT_NODE_ID_OFFSET_Y) << 6) | (entry.x + ATT_NODE_ID_OFFSET_X);
        const std::uint32_t programmed = entry.window == noc_att::WindowClass::Worker
                                             ? ATT_WORKER_ENDPOINT_WORDS[entry.selector]
                                             : ATT_FULL_TILE_ENDPOINT_WORDS[entry.selector];
        if (programmed != live_word) {
            return false;
        }
    }
    return true;
}
static_assert(dispatch_entries_match_endpoint_words(), "a dispatch entry's selector does not program its tile");

// The declarative map: everything the shared resolver needs, as data.
// Windows are indexed by WindowClass (LoopbackScratch, Worker, Dram, FullTile, Local).
inline constexpr noc_att::MapData MAP{
    .windows = {{LOOPBACK_SCRATCH_WINDOW, WORKER_WINDOW, DRAM_WINDOW, TILE_WINDOW, LOCAL_WINDOW}},
    .local_window_class = noc_att::WindowClass::Local,  // boot-patched ep256 = self at selector 0
    .worker_origin_x = ATT_WORKER_API_ORIGIN_X,
    .worker_origin_y = ATT_WORKER_API_ORIGIN_Y,
    .worker_grid_x = ATT_WORKER_GRID_X,
    .worker_grid_y = ATT_WORKER_GRID_Y,
    .worker_selectors = {ATT_WORKER_SELECTORS},
    .node_id_offset_x = ATT_NODE_ID_OFFSET_X,
    .node_id_offset_y = ATT_NODE_ID_OFFSET_Y,
    .worker_endpoint_words = {ATT_WORKER_ENDPOINT_WORDS},
    .full_tile_endpoint_words = {ATT_FULL_TILE_ENDPOINT_WORDS},
    .dram_endpoint_words = {ATT_DRAM_ENDPOINT_WORDS},
    .dram_selectors = {ATT_LOGICAL_DRAM_SELECTORS},
    .dispatch_entries = {DISPATCH_ENTRIES},
};

}  // namespace grendel_qsr1_att_config
