//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include "tile_math.hpp"
#include "core_coord.h"

#include "tt_metal/host_api.hpp"


namespace tt {
namespace tt_metal {

// splits the tiles evenly between num_cores,
// with option of padding where necessary
struct TilesSplit {
    int num_cores_;
    int total_tiles_;
    int tpc_; // unclipped tiles per core

    inline TilesSplit(int num_cores, int total_tiles) : num_cores_(num_cores), total_tiles_(total_tiles) {
        tpc_ = divup(total_tiles_, num_cores_);
    }

    // number of tiles per core for divup split
    inline uint32_t get_tpc() const { return tpc_; }

    // number of tiles per core for close to even split with multiples of 8 going to each core
    inline uint32_t get_clipped_tpc(int icore) const {
        auto result = ( tpc_*(icore+1) > total_tiles_ ) ? ( total_tiles_ - tpc_*(icore+1) ) : tpc_;
        return result;
    }
};

struct CoreGridDesc {
    uint32_t x_, y_;
    CoreGridDesc(Device* dev) { auto gs = dev->logical_grid_size(); x_ = gs.x; y_ = gs.y; TT_ASSERT(x_ > 0 && y_ > 0); }
    uint32_t total_cores() const { return x_*y_; }
    CoreCoord wrap_core(int icore) const {
        TT_ASSERT(icore < total_cores());
        CoreCoord core = {(std::size_t) icore % x_, (std::size_t) icore / x_};
        return core;
    }

    int numcores_dividing_numtiles(int num_tiles, int block_size = 1) {
        // since we will be splitting num_tiles into num_cores we need to find num_cores such that
        // num_tiles % num_cores = 0, so that it's evenly divided since we don't support leftovers at the moment
        // TODO(AP): optimize if needed, O(max_cores) atm
        uint32_t max_cores = total_cores();
        TT_ASSERT(max_cores % block_size == 0 || max_cores == 1);
        if (max_cores > num_tiles)
            max_cores = num_tiles;
        for (int j = max_cores; j >= 1; j--)
            if (num_tiles % j == 0)
                return j;
        return 1;
    }
};

// Finds the maximum even divisor of val starting at start_max_div and below
inline int find_max_divisor(uint32_t val, uint32_t start_max_div) {
    int result = 1;
    for (int find_divisor = start_max_div; find_divisor >= 1; find_divisor--) {
        if (find_divisor == 7 || find_divisor == 5)
            continue;
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

} } // namespace tt::tt_metal
