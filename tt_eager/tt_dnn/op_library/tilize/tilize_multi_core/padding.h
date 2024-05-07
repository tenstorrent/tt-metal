// # SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// # SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace tt::tt_metal {
// BlockRep represents a repeated sequence of data blocks, mixed blocks, and padding blocks.
// It is convient to pass to the device kernels because a single data structure made of 4 ints
// can represent pure data rows, pure padding rows or a mixture thereof.
struct BlockRep {
    // number of data blocks
    uint32_t n_data;
    // number of mixed data rows in a mixed block, 0 means no mixed block
    uint32_t n_mixed;
    // number of padding blocks
    uint32_t n_pads;
    // total repeat times
    uint32_t times;

    BlockRep(uint32_t n_data, uint32_t n_mixed, uint32_t n_pads, uint32_t times)
        : n_data(n_data), n_mixed(n_mixed), n_pads(n_pads), times(times) {
        if (n_data == 0 && n_mixed == 0) {
            n_pads *= times;
            times = 1;
        } else if (n_pads == 0 && n_mixed == 0) {
            n_data *= times;
            times = 1;
        }
    }

    bool has_mixed_block() const { return n_mixed > 0; }

    uint32_t single_rep() const { return n_data + has_mixed_block() + n_pads; }

    uint32_t block_count() const { return single_rep() * times; }

    uint32_t data_row_count() const { return (n_data * 32 + n_mixed) * times; }

    std::pair<std::vector<BlockRep>, std::vector<BlockRep>> split_at(uint32_t idx) const {
        // TT_ASSERT(idx <= block_count());

        std::vector<BlockRep> first;
        std::vector<BlockRep> second;

        int rep_idx = idx / single_rep();
        if (rep_idx > 0) {
            first.emplace_back(n_data, n_mixed, n_pads, rep_idx);
        }

        int within_rep_idx = idx % single_rep();
        bool is_within_rep = within_rep_idx > 0;
        if (is_within_rep) {
            if (within_rep_idx <= n_data) {
                first.emplace_back(within_rep_idx, 0, 0, 1);
                second.emplace_back(n_data - within_rep_idx, n_mixed, n_pads, 1);
            } else if (within_rep_idx == n_data + 1 && has_mixed_block()) {
                first.emplace_back(n_data, n_mixed, 0, 1);
                second.emplace_back(0, 0, n_pads, 1);
            } else {
                within_rep_idx -= n_data + has_mixed_block();
                first.emplace_back(n_data, n_mixed, within_rep_idx, 1);
                second.emplace_back(0, 0, n_pads - within_rep_idx, 1);
            }
        }

        int remaining_times = times - rep_idx - is_within_rep;
        if (remaining_times > 0) {
            second.emplace_back(n_data, n_mixed, n_pads, remaining_times);
        }

        return {first, second};
    }
};

// FullRep is a repeated sequence of data rows followed by pure padding. It represents the row
// pattern seen from the outer-most dimension of a 4D tensor when padding is added to the second
// or the thrird dimension.
struct FullRep {
    uint32_t n_rows;
    uint32_t n_pads;
    uint32_t times;

    uint32_t pads_mul;
    uint32_t times_total;

    BlockRep rep;
    BlockRep pad;

    FullRep(uint32_t n_rows, uint32_t n_pads, uint32_t times, uint32_t pads_mul, uint32_t times_total)
        : n_rows(n_rows), n_pads(n_pads), times(times), pads_mul(pads_mul),
          times_total(times_total), rep{n_rows / 32, n_rows % 32, n_pads / 32, times},
          pad{0, 0, (n_rows + n_pads) * pads_mul, 1} {
        // TT_ASSERT((n_rows + n_pads) % 32 == 0 && "total rows must be divisible by 32");
    }

    std::vector<BlockRep> to_block_reps() const {
        std::vector<BlockRep> block_reps;
        block_reps.reserve(2 * times_total);

        for (int i = 0; i < times_total; ++i) {
            block_reps.push_back(rep);
            block_reps.push_back(pad);
        }

        return block_reps;
    }
};

} // namespace tt::tt_metal
