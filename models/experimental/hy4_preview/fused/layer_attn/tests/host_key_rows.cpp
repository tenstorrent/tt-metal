// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host run of the index-key append's row writer (kernels/attn_key_write.cpp) with the kernels' own header
// (kernels/attn_key_rows.hpp): the same sequence, read every group's block once, patch the token rows ascending,
// write every group's block once, on a "bank" given by the test. tests/test_key_rows_host.py writes the cases and
// compares the result with a model of its own.
//
//   host_key_rows <in> <out>
// in, per case: where[8], row[8] (uint32), the packed tiles (4,352 B), then 8 x (address uint32, block 4,352 B):
//     the store. where[u] is one of the 8 addresses or NONE.
// out, per case: reads[8], writes[8] (uint32, per store block), same_row's result (uint32), 3 zero words, then the 8
// blocks after the pass.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../kernels/attn_key_rows.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    constexpr uint32_t USERS = 8, STORE = 8;
    std::FILE* in = std::fopen(argv[1], "rb");
    std::FILE* out = std::fopen(argv[2], "wb");
    if (in == nullptr || out == nullptr) {
        return 2;
    }
    uint32_t where[USERS], row_of[USERS], address[STORE];
    std::vector<uint8_t> packed(key_rows::BLOCK_BYTES), store(STORE * key_rows::BLOCK_BYTES);
    std::vector<uint8_t> blocks(USERS * key_rows::BLOCK_BYTES);
    uint32_t cases = 0;
    while (std::fread(where, 4, USERS, in) == USERS) {
        if (std::fread(row_of, 4, USERS, in) != USERS ||
            std::fread(packed.data(), 1, packed.size(), in) != packed.size()) {
            return 3;
        }
        for (uint32_t b = 0; b < STORE; ++b) {
            if (std::fread(&address[b], 4, 1, in) != 1 ||
                std::fread(&store[b * key_rows::BLOCK_BYTES], 1, key_rows::BLOCK_BYTES, in) != key_rows::BLOCK_BYTES) {
                return 3;
            }
        }
        uint32_t reads[STORE] = {}, writes[STORE] = {};
        auto slot = [&](uint32_t w) -> uint32_t {
            for (uint32_t b = 0; b < STORE; ++b) {
                if (address[b] == w) {
                    return b;
                }
            }
            return key_rows::NONE;
        };
        uint32_t group[USERS], leader[USERS];
        const uint32_t groups = key_rows::group_rows(where, group, leader);
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t b = slot(where[leader[g]]);
            if (b == key_rows::NONE) {
                return 4;
            }
            ++reads[b];
            std::memcpy(&blocks[g * key_rows::BLOCK_BYTES], &store[b * key_rows::BLOCK_BYTES], key_rows::BLOCK_BYTES);
        }
        for (uint32_t u = 0; u < USERS; ++u) {
            if (group[u] != key_rows::NONE) {
                key_rows::move_row(packed.data(), u, &blocks[group[u] * key_rows::BLOCK_BYTES], row_of[u]);
            }
        }
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t b = slot(where[leader[g]]);
            ++writes[b];
            std::memcpy(&store[b * key_rows::BLOCK_BYTES], &blocks[g * key_rows::BLOCK_BYTES], key_rows::BLOCK_BYTES);
        }
        const uint32_t tail[4] = {key_rows::same_row(where, row_of), 0, 0, 0};
        std::fwrite(reads, 4, STORE, out);
        std::fwrite(writes, 4, STORE, out);
        std::fwrite(tail, 4, 4, out);
        std::fwrite(store.data(), 1, store.size(), out);
        ++cases;
    }
    std::fclose(in);
    std::fclose(out);
    std::printf("cases %u\n", cases);
    return 0;
}
