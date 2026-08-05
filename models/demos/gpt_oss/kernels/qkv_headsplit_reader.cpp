// SPDX-License-Identifier: Apache-2.0
//
// gpt-oss decode QKV head split, multi-core, producing the SAME packed layout as
// ttnn.experimental.nlp_create_qkv_heads_decode.
//
// WHY: the stock op runs on 1 core of 130 (it shards its output by LOGICAL batch,
// and decode is batch=1) at 23.28 us/op x 24 layers = 0.558 ms/tok. It is not
// bandwidth bound: an earlier tile-granular kernel of mine moved 32x the data in
// 1/3 the time, so the cost is overhead. Spreading the work over several cores
// should therefore win.
//
// LAYOUT (what a first attempt got wrong):
//   input  xqkv : [1, 1, 32(pad), 5120] TILE; only batch row 0 holds real data
//   output q    : [1, 1, 64, 64]           = one ROW per head
//           k,v : [1, 1, 8, 64]
// Head h is a single head_dim-wide ROW inside a tile, not a whole tile. Placing it
// needs face-row (sub-tile) addressing, taken from the stock reader: a 32x32 bf16
// tile is four 16x16 faces, so row r of a tile sits at
//     r <  16 : r * SUBTILE_LINE_BYTES
//     r >= 16 : (r-16) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE
// and each 16-column face segment is one phase, so head_dim 64 = head_tiles
// tiles x 2 phases.
//
// WORK DECOMPOSITION (the second thing to get right): heads 0..31 of q all live in
// output tile-row 0, so they SHARE tiles. A work unit therefore cannot be one head
// (two cores would write the same tile through separate CB pages). The work unit
// is one OUTPUT TILE-ROW per tensor:
//     q tile-row 0 (heads 0..31), q tile-row 1 (heads 32..63), k tile-row 0, v tile-row 0
// = 4 work units, each gathering up to 32 head rows into head_tiles tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_q_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_k_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_v_id = get_compile_time_arg_val(2);
constexpr uint32_t num_q_heads = get_compile_time_arg_val(3);
constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
constexpr uint32_t head_tiles = get_compile_time_arg_val(5);          // head_dim / 32
constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(6);  // 16 elems * elem size
constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(7);
constexpr uint32_t ct_in = 8;

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t HALF_TILE_ELEMENTS = 512;                            // 32*32/2
constexpr uint32_t PHASE_OFFSET_BYTES = 256 * ELEMENT_SIZE;             // one face

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);

    constexpr auto in_args = TensorAccessorArgs<ct_in>();
    const auto in = TensorAccessor(in_args, in_addr);

    Noc noc;

    // Work-unit table, built at compile time from the head counts.
    //   unit 0..(q_tile_rows-1) -> q
    //   next                    -> k
    //   next                    -> v
    constexpr uint32_t q_tile_rows = (num_q_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    constexpr uint32_t k_tile_rows = (num_kv_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t u = 0; u < unit_count; ++u) {
        const uint32_t unit = unit_start + u;

        uint32_t cb_id;
        uint32_t tile_row;        // which tile-row of the destination
        uint32_t src_head_base;   // first input head for this unit
        uint32_t n_rows;          // how many real head rows this unit covers

        if (unit < q_tile_rows) {
            cb_id = cb_q_id;
            tile_row = unit;
            src_head_base = unit * TILE_HEIGHT;
            const uint32_t left = num_q_heads - src_head_base;
            n_rows = left < TILE_HEIGHT ? left : TILE_HEIGHT;
        } else if (unit < q_tile_rows + k_tile_rows) {
            cb_id = cb_k_id;
            tile_row = unit - q_tile_rows;
            src_head_base = num_q_heads + tile_row * TILE_HEIGHT;
            const uint32_t done = tile_row * TILE_HEIGHT;
            const uint32_t left = num_kv_heads - done;
            n_rows = left < TILE_HEIGHT ? left : TILE_HEIGHT;
        } else {
            cb_id = cb_v_id;
            tile_row = unit - q_tile_rows - k_tile_rows;
            src_head_base = num_q_heads + num_kv_heads + tile_row * TILE_HEIGHT;
            const uint32_t done = tile_row * TILE_HEIGHT;
            const uint32_t left = num_kv_heads - done;
            n_rows = left < TILE_HEIGHT ? left : TILE_HEIGHT;
        }

        CircularBuffer cb(cb_id);
        cb.reserve_back(head_tiles);
        const uint32_t wptr = cb.get_write_ptr();

        for (uint32_t r = 0; r < n_rows; ++r) {
            const uint32_t head = src_head_base + r;
            const uint32_t src_tile = head * head_tiles;

            // Destination row r within this tile-row.
            const uint32_t offset_in_tile =
                r < FACE_HEIGHT
                    ? r * SUBTILE_LINE_BYTES
                    : (r - FACE_HEIGHT) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE;

            for (uint32_t phase = 0; phase < 2; ++phase) {
                const uint32_t phase_off = phase * PHASE_OFFSET_BYTES;
                uint32_t waddr = wptr + offset_in_tile + phase_off;
                for (uint32_t t = 0; t < head_tiles; ++t) {
                    noc.async_read(
                        in,
                        CoreLocalMem<uint32_t>(waddr),
                        SUBTILE_LINE_BYTES,
                        {.page_id = src_tile + t, .offset_bytes = phase_off},
                        {});
                    // next destination tile, same row: one whole tile onward
                    waddr += TILE_HEIGHT * TILE_HEIGHT * ELEMENT_SIZE;
                }
            }
        }
        // One barrier per work unit (batched-barrier), not one per row.
        noc.async_read_barrier();
        cb.push_back(head_tiles);
    }
}
