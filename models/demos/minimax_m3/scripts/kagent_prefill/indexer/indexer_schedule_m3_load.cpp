// Per-core unit load (critical path) of the unfused indexer_score schedule, old (capacity-dealt) vs bounded, M3 shapes.
// g++ -std=c++17 -I<tt-metal>/ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels
// indexer_schedule_m3_load.cpp
#include <initializer_list>
#include <cstdio>
#include "indexer_schedule.hpp"
#include "indexer_score_work_split.hpp"
using namespace ttnn::operations::experimental::indexer_score;
// per-core valid units (groups x bands with k work) for the M3 shapes: old (capacity) vs new vs fitted
struct R {
    uint32_t max_units, sum_units, cores;
};
R run(uint32_t Sqt, uint32_t QC, uint32_t KC, uint32_t Tt, uint32_t kv_tiles, bool bounded, uint32_t gx, uint32_t gy) {
    uint32_t G = Sqt / QC, U = units_in_group(KC, Tt);
    uint32_t gr = rows_for_groups(G, gy), cols = cols_for_bands(U, gx), nb = band_row_blocks(G, U, gx, gy);
    uint32_t phases = G / gr, rows = gr * nb;
    R r{0, 0, rows * cols};
    for (uint32_t core = 0; core < rows * cols; ++core) {
        indexer_ring_schedule::Geometry g{1, U, 0, 0, nb, cols, false};
        auto s = bounded ? indexer_schedule::for_core_bounded<false>(core, gr, g, kv_tiles, KC)
                         : indexer_schedule::for_core<false>(core, gr, g);
        uint32_t v = 0;
        for (uint32_t b = 0; b < s.band_count; ++b) {
            if ((s.band_start + b) * KC < kv_tiles) {
                ++v;
            }
        }
        v *= phases;
        r.sum_units += v;
        if (v > r.max_units) {
            r.max_units = v;
        }
    }
    return r;
}
int main() {
    for (uint32_t gx : {13u, 14u, 11u}) {
        const uint32_t gy = 10;
        struct C {
            const char* n;
            uint32_t S, T, kv;
        } cs[] = {
            {"chunk5120 @1044480", 1280, 1044480, 56320},
            {"chunk5120 @61440", 1280, 61440, 56320},
            {"chunk5120 fitted56320", 1280, 56320, 56320},
            {"chunk4096 @1048576 kv=55296", 1024, 1048576, 55296},
            {"chunk4096 @1048576 kv=8192", 1024, 1048576, 8192},
            {"chunk5120 @1044480 kv=1044480", 1280, 1044480, 1044480}};
        for (auto& c : cs) {
            uint32_t Sqt = c.S / 32, Tt = c.T / 32, kv = c.kv / 32;
            R o = run(Sqt, 2, 32, Tt, kv, false, gx, gy), n = run(Sqt, 2, 32, Tt, kv, true, gx, gy);
            printf(
                "grid %ux%u %-30s cores %u  old max %3u units (sum %u)  new max %3u units (sum %u)\n",
                gx,
                gy,
                c.n,
                o.cores,
                o.max_units,
                o.sum_units,
                n.max_units,
                n.sum_units);
        }
    }
}
