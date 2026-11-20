// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host run of kernels/attn_idx_rows.hpp (tests/test_idx_rows_host.py writes the input and checks the output).
//   host_idx_rows <rows> <in> <out>
// in: the landing (uint32 words, idx_rows::landing_words(rows)), then the head-weight row tile (16 * rows words).
// out: per token row: its query (2,048 words), then its 16 multiplier words.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../kernels/attn_idx_rows.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::atoi(argv[1]));
    std::FILE* in = std::fopen(argv[2], "rb");
    std::FILE* out = std::fopen(argv[3], "wb");
    if (in == nullptr || out == nullptr || (rows != 8 && rows != 16)) {
        return 2;
    }
    std::vector<uint32_t> landing(idx_rows::landing_words(rows)), weights(idx_rows::row_tile_words(rows));
    if (std::fread(landing.data(), 4, landing.size(), in) != landing.size() ||
        std::fread(weights.data(), 4, weights.size(), in) != weights.size()) {
        return 3;
    }
    std::vector<uint32_t> query(idx_rows::QUERY_WORDS);
    uint32_t words[16];
    for (uint32_t u = 0; u < rows; ++u) {
        idx_rows::assemble(landing.data(), rows, u, query.data());
        idx_rows::multipliers(weights.data(), rows, u, words);
        std::fwrite(query.data(), 4, query.size(), out);
        std::fwrite(words, 4, 16, out);
    }
    std::fclose(in);
    std::fclose(out);
    return 0;
}
