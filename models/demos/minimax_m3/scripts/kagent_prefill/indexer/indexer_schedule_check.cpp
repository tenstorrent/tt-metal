// Host-side invariant check of the unfused indexer_score schedule (indexer_schedule.hpp::for_core_bounded).
// g++ -std=c++17 -I<tt-metal>/ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels
// indexer_schedule_check.cpp && ./a.out
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <vector>
#include "indexer_schedule.hpp"
#include "indexer_score_work_split.hpp"
using namespace ttnn::operations::experimental::indexer_score;
int main() {
    long checks = 0;
    for (uint32_t grid_x : {11u, 13u, 14u}) {
        for (uint32_t grid_y : {10u}) {
            for (uint32_t Sqt : {2u, 4u, 8u, 16u, 20u, 32u, 40u, 64u}) {
                for (uint32_t QC : {1u, 2u}) {
                    for (uint32_t KC : {8u, 16u, 32u}) {
                        for (uint32_t Tt : {32u, 64u, 256u, 1760u, 1920u, 4096u, 32640u}) {
                            if (Sqt % QC) {
                                continue;
                            }
                            uint32_t G = Sqt / QC;
                            uint32_t U = units_in_group(KC, Tt);
                            uint32_t gr = rows_for_groups(G, grid_y), cols = cols_for_bands(U, grid_x),
                                     nb = band_row_blocks(G, U, grid_x, grid_y);
                            uint32_t rows = gr * nb;
                            uint32_t maxb_ct = ((U + nb - 1) / nb + cols - 1) / cols;
                            for (uint32_t kv : {0u, 1u, 5u, 31u, 32u, 33u, 100u, 1760u, 1761u, Tt - 1, Tt}) {
                                if (kv > Tt) {
                                    continue;
                                }
                                uint32_t du = indexer_schedule::dealt_units(U, nb, cols, kv, KC);
                                uint32_t valid_units = (kv + KC - 1) / KC;
                                uint32_t maxb = indexer_schedule::widest_cell_bands(du, nb, cols);
                                assert(maxb <= maxb_ct);
                                if (kv == Tt) {
                                    assert(du == U);
                                }
                                std::vector<int> cover(U, 0);
                                uint32_t real_max = 0;
                                for (uint32_t core = 0; core < rows * cols; ++core) {
                                    auto s = indexer_schedule::for_core_bounded<false>(
                                        core, gr, {1, U, 0, 0, nb, cols, false}, kv, KC);
                                    assert(s.band_count >= 1);
                                    real_max = s.band_count > real_max ? s.band_count : real_max;
                                    assert(s.band_start + s.band_count <= U);
                                    bool seen_invalid = false;
                                    for (uint32_t b = 0; b < s.band_count; ++b) {
                                        uint32_t band = s.band_start + b;
                                        bool valid = band * KC < kv;
                                        if (!valid) {
                                            seen_invalid = true;
                                        } else {
                                            assert(!seen_invalid);  // valid prefix
                                        }
                                        if (s.row_group == 0 && s.block < nb) {
                                            cover[band]++;  // count once per block-row 0
                                        }
                                    }
                                }
                                assert(real_max == maxb);
                                for (uint32_t b = 0; b < U; ++b) {
                                    if (b < valid_units) {
                                        assert(cover[b] == 1);  // each valid band exactly once (over rows with
                                                                // row_group 0 of each block)
                                    } else {
                                        assert(cover[b] <= 1);
                                    }
                                }
                                // load: max valid bands per core <= ceil(valid/ (cols*nb)) when valid >= cells
                                ++checks;
                            }
                        }
                    }
                }
            }
        }
    }
    printf("ok %ld\n", checks);
}
