// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Model-local copy (custom_ops/topk_rows) of the shared topk_large_indices kernel with per-row valid lengths:
// valid_length_tensor is [rows, 1] (one 4-byte page per input row), the reader derives and pushes the bounds per row.
// Segment mode (SEG_ROWS_PER_USER > 0, the two-stage top-k's stage 1 under the live bound, custom_ops/sparse_bound.py):
// row r is segment r % SEG of user r / SEG, valid_length_tensor is [users, 1] and the row's bound is the user's bound
// clamped into its segment, clamp(valid[u] - (r % SEG) * SEG_COLS, 0, SEG_COLS); an EMPTY segment is one chunk of -inf
// synthesized in L1 (the columns beyond the scorer's live bound are never read from DRAM).

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
// Shared metadata read invalidates the reused L1 address before loading the next value.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(topk_common_args::input_address);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::search_length);
    const uint32_t input_page_bytes = get_common_arg_val<uint32_t>(topk_common_args::input_row_bytes);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t element_bytes = sizeof(uint16_t);  // input is validated as BFLOAT16
    constexpr uint32_t llk_k = chunk_bytes / element_bytes;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr uint32_t metadata_args_base = input_args.next_compile_time_args_offset();
    // THE metadata-mode selector for this kernel. One flag, not one per metadata argument: the host
    // validates the metadata tensors as a group, so inside this mode every metadata read is legal.
    constexpr bool metadata_mode = get_compile_time_arg_val(metadata_args_base) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(metadata_args_base + 1);
    constexpr uint32_t meta_offset = get_compile_time_arg_val(metadata_args_base + 2);
    constexpr auto meta_args = TensorAccessorArgs<metadata_args_base + 3>();
    // Real-token-end block, appended after the accessor above so those offsets are untouched.
    constexpr uint32_t vend_args_base = meta_args.next_compile_time_args_offset();
    constexpr auto vend_args = TensorAccessorArgs<vend_args_base>();
    // Per-row metadata: valid_length_tensor is [rows, 1] (one 4-byte page per input row), each row ranked over
    // its own prefix (the users of a batched decode step). The bounds are then pushed per row.
    constexpr uint32_t per_row_base = vend_args.next_compile_time_args_offset();
    constexpr bool per_row_metadata = get_compile_time_arg_val(per_row_base) != 0;
    // Segment mode (see the header comment): rows per user (0 = plain per-row bounds) and the segment width.
    constexpr uint32_t seg_rows_per_user = get_compile_time_arg_val(per_row_base + 1);
    constexpr uint32_t seg_cols = get_compile_time_arg_val(per_row_base + 2);
    constexpr uint32_t seg_div = seg_rows_per_user ? seg_rows_per_user : 1;  // a constant divisor of 0 is a build error

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    const uint32_t input_width = input_page_bytes / element_bytes;
    CircularBuffer input_cb(cb_in);
    Noc noc;

    TopkMetadataBounds bounds{0, 0};  // set below (once, or per row in per-row mode)
    uint32_t valid_end_cap = 0;       // per-row mode: the (optional) real-token-end cap, read once
    if constexpr (metadata_mode && per_row_metadata) {
        const uint32_t valid_end_addr = get_common_arg_val<uint32_t>(topk_common_args::valid_end_address);
        if (valid_end_addr != 0) {
            CircularBuffer meta_cb_obj(meta_cb);
            meta_cb_obj.reserve_back(1);
            const uint32_t scratch = meta_cb_obj.get_write_ptr();
            const uint32_t valid_end =
                trace_metadata::read_metadata_scalar_u32(noc, vend_args, valid_end_addr, scratch);
            valid_end_cap = ((valid_end + 31) / 32) * 32;
        }
    } else if constexpr (metadata_mode) {
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        const uint32_t scratch = meta_cb_obj.get_write_ptr();
        const uint32_t metadata_length = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(topk_common_args::metadata_address), scratch);
        // Validate before addition so the offset cannot wrap and no malformed metadata can produce a NoC read
        // outside the input row. ASSERT is watcher-gated, so it compiles to nothing in a normal build; pair it
        // with a clamp, as bounded_kv_actual_isl / bounded_cache_batch_idx do, or the bound would hold only
        // under the watcher. Clamping to input_width also matches the score kernel's coupled derivation, which
        // saturates its own extent (kv_len_tiles is clamped to k_len_tiles) rather than reading past the row.
        const bool metadata_valid = meta_offset <= input_width && metadata_length <= input_width - meta_offset &&
                                    (metadata_length != 0 || meta_offset != 0);
        ASSERT(metadata_valid);
        search_len = metadata_valid ? metadata_length + meta_offset : input_width;
        // 0 = no bound supplied (full chunk): the host zeroes the slot rather than compiling a second
        // variant, so presence costs a compare here instead of a program hash split.
        const uint32_t valid_end_addr = get_common_arg_val<uint32_t>(topk_common_args::valid_end_address);
        if (valid_end_addr != 0) {
            // Same scratch page as the read above: the helper invalidates the L1 line before loading, which
            // is exactly what it documents for a reused address.
            const uint32_t valid_end =
                trace_metadata::read_metadata_scalar_u32(noc, vend_args, valid_end_addr, scratch);
            const uint32_t capped = ((valid_end + 31) / 32) * 32;
            if (capped < search_len) {
                search_len = capped;
            }
        }
        bounds = calculate_topk_bounds(search_len, llk_k);
        CoreLocalMem<TopkMetadataBounds> mailbox(scratch);
        mailbox->num_chunks = bounds.num_chunks;
        mailbox->tail_elements = bounds.tail_elements;
        clobber_all_memory();
        meta_cb_obj.push_back(1);
    } else {
        bounds = calculate_topk_bounds(search_len, llk_k);
    }
    uint32_t tail_chunk_bytes = bounds.tail_elements * element_bytes;
    bool synthesized = false;  // segment mode: this row is one all--inf chunk filled in L1 (no DRAM read)

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        if constexpr (metadata_mode && per_row_metadata) {
            // this row's bound: page `row` of the [rows, 1] metadata tensor (same validation / clamp as above); in
            // segment mode page `row / SEG` (the user's bound over SEG * SEG_COLS columns), clamped into the segment
            CircularBuffer meta_cb_obj(meta_cb);
            meta_cb_obj.reserve_back(1);
            const uint32_t scratch = meta_cb_obj.get_write_ptr();
            const auto meta_acc =
                TensorAccessor(meta_args, get_common_arg_val<uint32_t>(topk_common_args::metadata_address));
            const uint32_t meta_page = seg_rows_per_user ? row / seg_div : row;
            noc.async_read(meta_acc, CoreLocalMem<uint8_t>(scratch), sizeof(uint32_t), {.page_id = meta_page}, {});
            noc.async_read_barrier();
            invalidate_l1_cache();
            const uint32_t metadata_length = CoreLocalMem<volatile uint32_t>(scratch)[0];
            uint32_t row_len;
            synthesized = false;
            if constexpr (seg_rows_per_user > 0) {
                constexpr uint32_t user_width = seg_rows_per_user * seg_cols;
                const uint32_t user_len = metadata_length < user_width ? metadata_length : user_width;
                const uint32_t seg_start = (row % seg_div) * seg_cols;
                row_len = user_len > seg_start ? user_len - seg_start : 0;
                if (row_len > seg_cols) {
                    row_len = seg_cols;
                }
                if (row_len == 0) {
                    synthesized = true;  // one full chunk of -inf: the unbounded path's content of such a segment
                    row_len = llk_k;
                }
            } else {
                const bool metadata_valid = meta_offset <= input_width &&
                                            metadata_length <= input_width - meta_offset &&
                                            (metadata_length != 0 || meta_offset != 0);
                ASSERT(metadata_valid);
                row_len = metadata_valid ? metadata_length + meta_offset : input_width;
                if (valid_end_cap != 0 && valid_end_cap < row_len) {
                    row_len = valid_end_cap;
                }
            }
            bounds = calculate_topk_bounds(row_len, llk_k);
            tail_chunk_bytes = bounds.tail_elements * element_bytes;
            CoreLocalMem<TopkMetadataBounds> mailbox(scratch);
            mailbox->num_chunks = bounds.num_chunks;
            mailbox->tail_elements = bounds.tail_elements;
            clobber_all_memory();
            meta_cb_obj.push_back(1);
        }
        for (uint32_t chunk = 0; chunk < bounds.num_chunks; ++chunk) {
            const uint32_t active_chunk_bytes = (chunk + 1 == bounds.num_chunks) ? tail_chunk_bytes : chunk_bytes;
            input_cb.reserve_back(tiles_per_chunk);
            if (synthesized) {
                // the whole page -inf (bf16 0xFF80): what the scorer's mask holds in such a segment on the unbounded
                // path
                volatile tt_l1_ptr uint32_t* fill =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_cb.get_write_ptr());
                for (uint32_t w = 0; w < tiles_per_chunk * tile_bytes / sizeof(uint32_t); ++w) {
                    fill[w] = 0xFF80FF80u;
                }
                input_cb.push_back(tiles_per_chunk);
                continue;
            }
            for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                const uint32_t tile_offset = tile * tile_bytes;
                const uint32_t read_bytes =
                    tile_offset < active_chunk_bytes
                        ? (active_chunk_bytes - tile_offset < tile_bytes ? active_chunk_bytes - tile_offset
                                                                         : tile_bytes)
                        : 0;
                if (read_bytes != 0) {
                    noc.async_read(
                        input,
                        input_cb,
                        read_bytes,
                        {.page_id = row, .offset_bytes = chunk * chunk_bytes + tile_offset},
                        {.offset_bytes = tile_offset});
                }
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
        }
    }
    noc.async_full_barrier();
}
