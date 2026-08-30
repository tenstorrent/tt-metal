// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "ttnn/operations/experimental/indexer_score/device/ring_indexer_score_schedule.hpp"
#include "ttnn/operations/experimental/indexer_score/device/kernels/indexer_score_work_split.hpp"

namespace ttnn::operations::experimental::indexer_score::program::ring_schedule {
namespace {

constexpr uint32_t kKcTiles = 10;
constexpr uint32_t kBlockCyclicChunkTiles = 5;
constexpr uint32_t kGlmCacheCapacityTiles = 1'044'480 / 32;

uint32_t units_in_shard(uint32_t tiles_per_shard) { return (tiles_per_shard + kKcTiles - 1) / kKcTiles; }

WorkList legacy_work_list(
    const ArrivalWaves& waves,
    uint32_t units_per_shard,
    uint32_t tiles_per_shard,
    uint32_t num_blocks,
    uint32_t cols_used) {
    WorkList work_list(num_blocks, std::vector<std::vector<uint32_t>>(cols_used));
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (const auto& wave : waves) {
            for (uint32_t col = 0; col < cols_used; ++col) {
                for (uint32_t unit_slot = col;; unit_slot += cols_used) {
                    const uint32_t unit = block + unit_slot * num_blocks;
                    if (unit >= units_per_shard) {
                        break;
                    }
                    for (uint32_t shard : wave) {
                        work_list[block][col].push_back(shard * tiles_per_shard + unit * kKcTiles);
                    }
                }
            }
        }
    }
    return work_list;
}

void expect_exact_coverage_and_wave_order(
    const WorkList& work_list,
    const ArrivalWaves& waves,
    uint32_t units_per_shard,
    uint32_t tiles_per_shard,
    uint32_t num_blocks,
    uint32_t cols_used,
    bool rotated) {
    const uint32_t ring_size = std::accumulate(waves.begin(), waves.end(), 0u, [](uint32_t sum, const auto& wave) {
        return sum + static_cast<uint32_t>(wave.size());
    });
    std::vector<uint32_t> coverage(ring_size * units_per_shard, 0);
    const uint32_t lane_count = num_blocks * cols_used;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            const auto& starts = work_list[block][col];
            uint32_t index = 0;
            for (uint32_t wave = 0; wave < waves.size(); ++wave) {
                const uint32_t column_shift =
                    rotated && waves[wave].size() > 1 ? wave_column_shift(wave, waves.size(), cols_used) : 0;
                const uint32_t source_col = (col + cols_used - column_shift) % cols_used;
                const uint32_t source_lane = block + source_col * num_blocks;
                for (uint32_t unit = source_lane; unit < units_per_shard; unit += lane_count) {
                    for (uint32_t shard : waves[wave]) {
                        ASSERT_LT(index, starts.size());
                        const uint32_t physical_start = starts[index++];
                        EXPECT_EQ(physical_start, shard * tiles_per_shard + unit * kKcTiles);
                        EXPECT_LT(physical_start, (shard + 1) * tiles_per_shard);
                        EXPECT_EQ((physical_start - shard * tiles_per_shard) % kKcTiles, 0u);
                        coverage[shard * units_per_shard + unit]++;
                    }
                }
            }
            EXPECT_EQ(index, starts.size());
        }
    }
    EXPECT_TRUE(std::all_of(coverage.begin(), coverage.end(), [](uint32_t count) { return count == 1; }));
}

uint32_t valid_tiles_in_unit(
    uint32_t physical_start,
    uint32_t tiles_per_shard,
    uint32_t valid_k_tiles,
    uint32_t ring_size,
    uint32_t chunk_local_tiles = kBlockCyclicChunkTiles) {
    const uint32_t shard = physical_start / tiles_per_shard;
    const uint32_t shard_offset = physical_start - shard * tiles_per_shard;
    const uint32_t capacity = std::min(kKcTiles, tiles_per_shard - shard_offset);
    uint32_t valid = 0;
    for (uint32_t col = 0; col < capacity; ++col) {
        const uint32_t local = shard_offset + col;
        const uint32_t slab = local / chunk_local_tiles;
        const uint32_t slab_offset = local - slab * chunk_local_tiles;
        const uint32_t logical_tile = (slab * ring_size + shard) * chunk_local_tiles + slab_offset;
        if (logical_tile >= valid_k_tiles) {
            break;
        }
        ++valid;
    }
    return valid;
}

std::vector<uint32_t> nonempty_loads(
    const WorkList& work_list, uint32_t tiles_per_shard, uint32_t valid_k_tiles, uint32_t ring_size) {
    const uint32_t num_blocks = work_list.size();
    const uint32_t cols_used = work_list.front().size();
    std::vector<uint32_t> loads(num_blocks * cols_used, 0);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            const uint32_t lane = block + col * num_blocks;
            for (uint32_t physical_start : work_list[block][col]) {
                loads[lane] += valid_tiles_in_unit(physical_start, tiles_per_shard, valid_k_tiles, ring_size) != 0;
            }
        }
    }
    return loads;
}

std::vector<uint32_t> valid_tile_loads(
    const WorkList& work_list, uint32_t tiles_per_shard, uint32_t valid_k_tiles, uint32_t ring_size) {
    const uint32_t num_blocks = work_list.size();
    const uint32_t cols_used = work_list.front().size();
    std::vector<uint32_t> loads(num_blocks * cols_used, 0);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            const uint32_t lane = block + col * num_blocks;
            for (uint32_t physical_start : work_list[block][col]) {
                loads[lane] += valid_tiles_in_unit(physical_start, tiles_per_shard, valid_k_tiles, ring_size);
            }
        }
    }
    return loads;
}

double max_to_mean(const std::vector<uint32_t>& loads) {
    const uint32_t total = std::accumulate(loads.begin(), loads.end(), 0u);
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(*std::max_element(loads.begin(), loads.end())) * loads.size() / total;
}

struct PerWaveTailRatios {
    std::vector<double> nonempty;
    std::vector<double> valid_tiles;
};

PerWaveTailRatios per_wave_tail_ratios(
    const WorkList& work_list,
    const ArrivalWaves& waves,
    uint32_t tiles_per_shard,
    uint32_t valid_k_tiles,
    uint32_t ring_size) {
    const uint32_t num_blocks = work_list.size();
    const uint32_t cols_used = work_list.front().size();
    const uint32_t lane_count = num_blocks * cols_used;
    std::vector<uint32_t> shard_wave(ring_size);
    for (uint32_t wave = 0; wave < waves.size(); ++wave) {
        for (uint32_t shard : waves[wave]) {
            shard_wave[shard] = wave;
        }
    }

    PerWaveTailRatios ratios;
    ratios.nonempty.reserve(waves.size());
    ratios.valid_tiles.reserve(waves.size());
    for (uint32_t first_wave = 0; first_wave < waves.size(); ++first_wave) {
        std::vector<uint32_t> nonempty(lane_count, 0);
        std::vector<uint32_t> valid_tiles(lane_count, 0);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t col = 0; col < cols_used; ++col) {
                const uint32_t lane = block + col * num_blocks;
                for (uint32_t physical_start : work_list[block][col]) {
                    if (shard_wave[physical_start / tiles_per_shard] < first_wave) {
                        continue;
                    }
                    const uint32_t valid =
                        valid_tiles_in_unit(physical_start, tiles_per_shard, valid_k_tiles, ring_size);
                    nonempty[lane] += valid != 0;
                    valid_tiles[lane] += valid;
                }
            }
        }
        ratios.nonempty.push_back(max_to_mean(nonempty));
        ratios.valid_tiles.push_back(max_to_mean(valid_tiles));
    }
    return ratios;
}

std::string comma_separated(const std::vector<double>& values) {
    std::ostringstream stream;
    stream.precision(3);
    stream << std::fixed;
    for (uint32_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

std::string decimal(double value) {
    std::ostringstream stream;
    stream.precision(6);
    stream << std::fixed << value;
    return stream.str();
}

struct PrefixProfile {
    std::vector<uint32_t> max_nonempty;
    std::vector<uint32_t> max_valid_tiles;
    std::vector<uint32_t> total_nonempty;
    std::vector<uint32_t> total_valid_tiles;
};

PrefixProfile prefix_profile(
    const WorkList& work_list, uint32_t tiles_per_shard, uint32_t capacity_tiles, uint32_t ring_size) {
    const uint32_t num_blocks = work_list.size();
    const uint32_t cols_used = work_list.front().size();
    const uint32_t lane_count = num_blocks * cols_used;
    std::vector<std::vector<uint16_t>> nonempty_events(capacity_tiles + 1, std::vector<uint16_t>(lane_count));
    std::vector<std::vector<uint16_t>> valid_tile_events(capacity_tiles + 1, std::vector<uint16_t>(lane_count));

    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            const uint32_t lane = block + col * num_blocks;
            for (uint32_t physical_start : work_list[block][col]) {
                const uint32_t shard = physical_start / tiles_per_shard;
                const uint32_t shard_offset = physical_start - shard * tiles_per_shard;
                const uint32_t unit_capacity = std::min(kKcTiles, tiles_per_shard - shard_offset);
                uint32_t first_logical_tile = std::numeric_limits<uint32_t>::max();
                for (uint32_t col_in_unit = 0; col_in_unit < unit_capacity; ++col_in_unit) {
                    const uint32_t local = shard_offset + col_in_unit;
                    const uint32_t slab = local / kBlockCyclicChunkTiles;
                    const uint32_t slab_offset = local - slab * kBlockCyclicChunkTiles;
                    const uint32_t logical_tile = (slab * ring_size + shard) * kBlockCyclicChunkTiles + slab_offset;
                    EXPECT_LT(logical_tile, capacity_tiles);
                    first_logical_tile = std::min(first_logical_tile, logical_tile);
                    valid_tile_events[logical_tile + 1][lane]++;
                }
                EXPECT_NE(first_logical_tile, std::numeric_limits<uint32_t>::max());
                nonempty_events[first_logical_tile + 1][lane]++;
            }
        }
    }

    PrefixProfile profile{
        .max_nonempty = std::vector<uint32_t>(capacity_tiles + 1),
        .max_valid_tiles = std::vector<uint32_t>(capacity_tiles + 1),
        .total_nonempty = std::vector<uint32_t>(capacity_tiles + 1),
        .total_valid_tiles = std::vector<uint32_t>(capacity_tiles + 1)};
    std::vector<uint32_t> nonempty(lane_count, 0);
    std::vector<uint32_t> valid_tiles(lane_count, 0);
    for (uint32_t prefix = 1; prefix <= capacity_tiles; ++prefix) {
        for (uint32_t lane = 0; lane < lane_count; ++lane) {
            nonempty[lane] += nonempty_events[prefix][lane];
            valid_tiles[lane] += valid_tile_events[prefix][lane];
        }
        profile.max_nonempty[prefix] = *std::max_element(nonempty.begin(), nonempty.end());
        profile.max_valid_tiles[prefix] = *std::max_element(valid_tiles.begin(), valid_tiles.end());
        profile.total_nonempty[prefix] = std::accumulate(nonempty.begin(), nonempty.end(), 0u);
        profile.total_valid_tiles[prefix] = std::accumulate(valid_tiles.begin(), valid_tiles.end(), 0u);
    }
    return profile;
}

TEST(RingIndexerScoreSchedule, ShiftZeroExactlyMatchesLegacySchedule) {
    EXPECT_EQ(wave_column_shift(0, 3, 10), 0u);
    for (auto topology : {ttnn::ccl::Topology::Linear, ttnn::ccl::Topology::Ring}) {
        for (uint32_t ring_size : {2u, 4u, 8u}) {
            for (uint32_t rank = 0; rank < ring_size; ++rank) {
                const auto waves = arrival_waves(ring_size, rank, topology);
                const auto legacy = legacy_work_list(waves, 44, 440, 2, 10);
                const auto unrotated = make_work_list(waves, 44, 440, kKcTiles, 2, 10, false);
                EXPECT_EQ(unrotated, legacy);
            }
        }
    }
    EXPECT_FALSE(rotation_enabled(ttnn::ccl::Topology::Linear, 8));
    EXPECT_FALSE(rotation_enabled(ttnn::ccl::Topology::Ring, 2));
    EXPECT_TRUE(rotation_enabled(ttnn::ccl::Topology::Ring, 4));
}

TEST(RingIndexerScoreSchedule, RotationHasExactCoverageAndMonotonicPairedWaveOrder) {
    for (const auto [ring_size, lanes, tiles_per_shard] :
         {std::tuple{4u, 20u, 440u},
          std::tuple{8u, 22u, 230u},
          std::tuple{32u, 20u, 1'020u},
          std::tuple{4u, 20u, 103u}}) {
        const uint32_t units = units_in_shard(tiles_per_shard);
        for (uint32_t rank = 0; rank < ring_size; ++rank) {
            const auto waves = arrival_waves(ring_size, rank, ttnn::ccl::Topology::Ring);
            ASSERT_EQ(waves.front(), std::vector<uint32_t>{rank});
            const auto work_list = make_work_list(waves, units, tiles_per_shard, kKcTiles, 2, lanes / 2, true);
            expect_exact_coverage_and_wave_order(work_list, waves, units, tiles_per_shard, 2, lanes / 2, true);
        }
    }
}

TEST(RingIndexerScoreSchedule, SmallCapacityLaneGeometryHasNoEmptyWorkList) {
    constexpr uint32_t grid_x = 10;
    constexpr uint32_t grid_y = 10;
    constexpr uint32_t group_count = 20;
    static_assert(group_count / rows_for_groups(group_count, grid_y) > 1);
    for (uint32_t ring_size : {4u, 8u, 32u}) {
        const auto waves = arrival_waves(ring_size, 0, ttnn::ccl::Topology::Ring);
        for (uint32_t units_per_shard = 1; units_per_shard <= 2 * grid_x; ++units_per_shard) {
            const uint32_t cols_used = cols_for_bands(units_per_shard, grid_x);
            const uint32_t num_blocks = band_row_blocks(group_count, units_per_shard, grid_x, grid_y);
            ASSERT_LE(num_blocks * cols_used, units_per_shard);
            const auto work_list = make_work_list(
                waves, units_per_shard, units_per_shard * kKcTiles, kKcTiles, num_blocks, cols_used, true);
            for (const auto& block : work_list) {
                for (const auto& lane : block) {
                    EXPECT_FALSE(lane.empty()) << "ring_size=" << ring_size << " units_per_shard=" << units_per_shard;
                }
            }
        }
    }
}

TEST(RingIndexerScoreSchedule, NamedQbAndLbLoadRatiosMatchPlan) {
    struct Case {
        std::string name;
        uint32_t ring_size;
        uint32_t lanes;
        uint32_t kv_len_tiles;
        uint32_t expected_current_max;
        uint32_t expected_rotated_max;
    };
    const std::vector<Case> cases = {
        {"qb_55k", 4, 20, 56'320 / 32, 12, 10},
        {"qb_512k", 4, 20, 524'288 / 32, 84, 84},
        {"lb_55k", 8, 22, 56'320 / 32, 8, 8},
        {"lb_58880", 8, 22, 58'880 / 32, 16, 10},
        {"lb_512k", 8, 22, 524'288 / 32, 80, 80},
    };
    for (const auto& test_case : cases) {
        const uint32_t tiles_per_shard = kGlmCacheCapacityTiles / test_case.ring_size;
        const auto waves = arrival_waves(test_case.ring_size, 0, ttnn::ccl::Topology::Ring);
        const auto current = make_work_list(
            waves, units_in_shard(tiles_per_shard), tiles_per_shard, kKcTiles, 2, test_case.lanes / 2, false);
        const auto rotated = make_work_list(
            waves, units_in_shard(tiles_per_shard), tiles_per_shard, kKcTiles, 2, test_case.lanes / 2, true);
        const auto current_loads =
            nonempty_loads(current, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        const auto rotated_loads =
            nonempty_loads(rotated, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        const auto current_valid_tiles =
            valid_tile_loads(current, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        const auto rotated_valid_tiles =
            valid_tile_loads(rotated, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        EXPECT_EQ(*std::max_element(current_loads.begin(), current_loads.end()), test_case.expected_current_max);
        EXPECT_EQ(*std::max_element(rotated_loads.begin(), rotated_loads.end()), test_case.expected_rotated_max);
        EXPECT_EQ(
            std::accumulate(current_loads.begin(), current_loads.end(), 0u),
            std::accumulate(rotated_loads.begin(), rotated_loads.end(), 0u));
        EXPECT_EQ(
            std::accumulate(current_valid_tiles.begin(), current_valid_tiles.end(), 0u),
            std::accumulate(rotated_valid_tiles.begin(), rotated_valid_tiles.end(), 0u));

        const auto current_tails =
            per_wave_tail_ratios(current, waves, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        const auto rotated_tails =
            per_wave_tail_ratios(rotated, waves, tiles_per_shard, test_case.kv_len_tiles, test_case.ring_size);
        RecordProperty(test_case.name + "_current_nonempty_max_to_mean", decimal(max_to_mean(current_loads)));
        RecordProperty(test_case.name + "_rotated_nonempty_max_to_mean", decimal(max_to_mean(rotated_loads)));
        RecordProperty(test_case.name + "_current_valid_tiles_max_to_mean", decimal(max_to_mean(current_valid_tiles)));
        RecordProperty(test_case.name + "_rotated_valid_tiles_max_to_mean", decimal(max_to_mean(rotated_valid_tiles)));
        RecordProperty(test_case.name + "_current_tail_nonempty", comma_separated(current_tails.nonempty));
        RecordProperty(test_case.name + "_rotated_tail_nonempty", comma_separated(rotated_tails.nonempty));
        RecordProperty(test_case.name + "_current_tail_valid_tiles", comma_separated(current_tails.valid_tiles));
        RecordProperty(test_case.name + "_rotated_tail_valid_tiles", comma_separated(rotated_tails.valid_tiles));
    }
}

TEST(RingIndexerScoreSchedule, PrefixSweepRecordsUnitTileAndArrivalWaveBalance) {
    struct Sweep {
        std::string name;
        uint32_t ring_size;
        uint32_t lanes;
        std::vector<uint32_t> prefixes_tokens;
    };
    std::vector<uint32_t> qb_prefixes;
    for (uint32_t prefix = 51'200; prefix <= 75'520; prefix += 1'280) {
        qb_prefixes.push_back(prefix);
    }
    qb_prefixes.push_back(524'288);
    std::vector<uint32_t> lb_prefixes;
    for (uint32_t prefix = 56'320; prefix <= 110'080; prefix += 2'560) {
        lb_prefixes.push_back(prefix);
    }
    lb_prefixes.push_back(524'288);

    for (const Sweep& sweep :
         {Sweep{"qb", 4, 20, std::move(qb_prefixes)}, Sweep{"lb", 8, 22, std::move(lb_prefixes)}}) {
        const uint32_t tiles_per_shard = kGlmCacheCapacityTiles / sweep.ring_size;
        const uint32_t units_per_shard = units_in_shard(tiles_per_shard);
        for (uint32_t prefix_tokens : sweep.prefixes_tokens) {
            const uint32_t valid_k_tiles = prefix_tokens / 32;
            double current_nonempty = 0.0;
            double rotated_nonempty = 0.0;
            double current_valid_tiles = 0.0;
            double rotated_valid_tiles = 0.0;
            std::vector<double> current_tail_nonempty(sweep.ring_size / 2 + 1, 0.0);
            std::vector<double> rotated_tail_nonempty(sweep.ring_size / 2 + 1, 0.0);
            std::vector<double> current_tail_valid_tiles(sweep.ring_size / 2 + 1, 0.0);
            std::vector<double> rotated_tail_valid_tiles(sweep.ring_size / 2 + 1, 0.0);
            for (uint32_t rank = 0; rank < sweep.ring_size; ++rank) {
                const auto waves = arrival_waves(sweep.ring_size, rank, ttnn::ccl::Topology::Ring);
                const auto current =
                    make_work_list(waves, units_per_shard, tiles_per_shard, kKcTiles, 2, sweep.lanes / 2, false);
                const auto rotated =
                    make_work_list(waves, units_per_shard, tiles_per_shard, kKcTiles, 2, sweep.lanes / 2, true);
                const auto current_nonempty_loads =
                    nonempty_loads(current, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                const auto rotated_nonempty_loads =
                    nonempty_loads(rotated, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                const auto current_valid_tile_loads =
                    valid_tile_loads(current, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                const auto rotated_valid_tile_loads =
                    valid_tile_loads(rotated, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                current_nonempty = std::max(current_nonempty, max_to_mean(current_nonempty_loads));
                rotated_nonempty = std::max(rotated_nonempty, max_to_mean(rotated_nonempty_loads));
                current_valid_tiles = std::max(current_valid_tiles, max_to_mean(current_valid_tile_loads));
                rotated_valid_tiles = std::max(rotated_valid_tiles, max_to_mean(rotated_valid_tile_loads));

                const auto current_tails =
                    per_wave_tail_ratios(current, waves, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                const auto rotated_tails =
                    per_wave_tail_ratios(rotated, waves, tiles_per_shard, valid_k_tiles, sweep.ring_size);
                for (uint32_t wave = 0; wave < waves.size(); ++wave) {
                    current_tail_nonempty[wave] = std::max(current_tail_nonempty[wave], current_tails.nonempty[wave]);
                    rotated_tail_nonempty[wave] = std::max(rotated_tail_nonempty[wave], rotated_tails.nonempty[wave]);
                    current_tail_valid_tiles[wave] =
                        std::max(current_tail_valid_tiles[wave], current_tails.valid_tiles[wave]);
                    rotated_tail_valid_tiles[wave] =
                        std::max(rotated_tail_valid_tiles[wave], rotated_tails.valid_tiles[wave]);
                }
            }

            const std::string prefix = sweep.name + "_" + std::to_string(prefix_tokens) + "_";
            const uint32_t valid_units_per_shard =
                (valid_k_tiles + sweep.ring_size * kKcTiles - 1) / (sweep.ring_size * kKcTiles);
            RecordProperty(prefix + "lanes", sweep.lanes);
            RecordProperty(prefix + "valid_units_per_shard", valid_units_per_shard);
            RecordProperty(prefix + "unit_remainder", valid_units_per_shard % sweep.lanes);
            RecordProperty(prefix + "current_nonempty_max_to_mean", decimal(current_nonempty));
            RecordProperty(prefix + "rotated_nonempty_max_to_mean", decimal(rotated_nonempty));
            RecordProperty(prefix + "current_valid_tiles_max_to_mean", decimal(current_valid_tiles));
            RecordProperty(prefix + "rotated_valid_tiles_max_to_mean", decimal(rotated_valid_tiles));
            RecordProperty(prefix + "current_tail_nonempty", comma_separated(current_tail_nonempty));
            RecordProperty(prefix + "rotated_tail_nonempty", comma_separated(rotated_tail_nonempty));
            RecordProperty(prefix + "current_tail_valid_tiles", comma_separated(current_tail_valid_tiles));
            RecordProperty(prefix + "rotated_tail_valid_tiles", comma_separated(rotated_tail_valid_tiles));
        }
    }
}

TEST(RingIndexerScoreSchedule, Ring32EveryRankAndTileAlignedPrefixIsNeverWorse) {
    constexpr uint32_t ring_size = 32;
    constexpr uint32_t tiles_per_shard = kGlmCacheCapacityTiles / ring_size;
    static_assert(tiles_per_shard * ring_size == kGlmCacheCapacityTiles);
    const uint32_t units_per_shard = units_in_shard(tiles_per_shard);

    for (uint32_t lanes : {20u, 22u}) {
        double worst_rotated_to_current = 0.0;
        double worst_current_max_to_mean = 0.0;
        double worst_rotated_max_to_mean = 0.0;
        double worst_current_valid_tiles_max_to_mean = 0.0;
        double worst_rotated_valid_tiles_max_to_mean = 0.0;
        double worst_rotated_valid_tiles_to_current = 0.0;
        uint32_t worst_ratio_prefix = 0;
        uint32_t worst_ratio_rank = 0;
        uint32_t worst_rotated_max_to_mean_prefix = 0;
        uint32_t worst_rotated_max_to_mean_rank = 0;
        uint32_t worst_rotated_valid_tiles_prefix = 0;
        uint32_t worst_rotated_valid_tiles_rank = 0;
        uint32_t full_prefix_current_max = 0;
        uint32_t full_prefix_rotated_max = 0;
        for (uint32_t rank = 0; rank < ring_size; ++rank) {
            SCOPED_TRACE(::testing::Message() << "lanes=" << lanes << " rank=" << rank);
            const auto waves = arrival_waves(ring_size, rank, ttnn::ccl::Topology::Ring);
            ASSERT_EQ(waves.size(), 17u);
            ASSERT_EQ(waves.front(), std::vector<uint32_t>{rank});
            for (uint32_t wave = 1; wave + 1 < waves.size(); ++wave) {
                ASSERT_EQ(waves[wave].size(), 2u);
            }
            ASSERT_EQ(waves.back().size(), 1u);

            const auto current = make_work_list(waves, units_per_shard, tiles_per_shard, kKcTiles, 2, lanes / 2, false);
            const auto rotated = make_work_list(waves, units_per_shard, tiles_per_shard, kKcTiles, 2, lanes / 2, true);
            expect_exact_coverage_and_wave_order(rotated, waves, units_per_shard, tiles_per_shard, 2, lanes / 2, true);

            const auto current_profile = prefix_profile(current, tiles_per_shard, kGlmCacheCapacityTiles, ring_size);
            const auto rotated_profile = prefix_profile(rotated, tiles_per_shard, kGlmCacheCapacityTiles, ring_size);
            for (uint32_t prefix = 1; prefix <= kGlmCacheCapacityTiles; ++prefix) {
                ASSERT_LE(rotated_profile.max_nonempty[prefix], current_profile.max_nonempty[prefix])
                    << "prefix_tiles=" << prefix;
                ASSERT_EQ(rotated_profile.total_nonempty[prefix], current_profile.total_nonempty[prefix]);
                ASSERT_EQ(rotated_profile.total_valid_tiles[prefix], current_profile.total_valid_tiles[prefix]);
                const double ratio = static_cast<double>(rotated_profile.max_nonempty[prefix]) /
                                     static_cast<double>(current_profile.max_nonempty[prefix]);
                const double current_mean =
                    static_cast<double>(current_profile.total_nonempty[prefix]) / static_cast<double>(lanes);
                const double rotated_mean =
                    static_cast<double>(rotated_profile.total_nonempty[prefix]) / static_cast<double>(lanes);
                const double current_max_to_mean =
                    static_cast<double>(current_profile.max_nonempty[prefix]) / current_mean;
                const double rotated_max_to_mean =
                    static_cast<double>(rotated_profile.max_nonempty[prefix]) / rotated_mean;
                worst_current_max_to_mean = std::max(worst_current_max_to_mean, current_max_to_mean);
                if (rotated_max_to_mean > worst_rotated_max_to_mean) {
                    worst_rotated_max_to_mean = rotated_max_to_mean;
                    worst_rotated_max_to_mean_prefix = prefix;
                    worst_rotated_max_to_mean_rank = rank;
                }
                const double current_valid_tiles_mean =
                    static_cast<double>(current_profile.total_valid_tiles[prefix]) / static_cast<double>(lanes);
                const double rotated_valid_tiles_mean =
                    static_cast<double>(rotated_profile.total_valid_tiles[prefix]) / static_cast<double>(lanes);
                const double current_valid_tiles_max_to_mean =
                    static_cast<double>(current_profile.max_valid_tiles[prefix]) / current_valid_tiles_mean;
                const double rotated_valid_tiles_max_to_mean =
                    static_cast<double>(rotated_profile.max_valid_tiles[prefix]) / rotated_valid_tiles_mean;
                worst_current_valid_tiles_max_to_mean =
                    std::max(worst_current_valid_tiles_max_to_mean, current_valid_tiles_max_to_mean);
                if (rotated_valid_tiles_max_to_mean > worst_rotated_valid_tiles_max_to_mean) {
                    worst_rotated_valid_tiles_max_to_mean = rotated_valid_tiles_max_to_mean;
                    worst_rotated_valid_tiles_prefix = prefix;
                    worst_rotated_valid_tiles_rank = rank;
                }
                worst_rotated_valid_tiles_to_current = std::max(
                    worst_rotated_valid_tiles_to_current,
                    static_cast<double>(rotated_profile.max_valid_tiles[prefix]) /
                        static_cast<double>(current_profile.max_valid_tiles[prefix]));
                if (ratio > worst_rotated_to_current) {
                    worst_rotated_to_current = ratio;
                    worst_ratio_prefix = prefix;
                    worst_ratio_rank = rank;
                }
            }
            full_prefix_current_max = std::max(full_prefix_current_max, current_profile.max_nonempty.back());
            full_prefix_rotated_max = std::max(full_prefix_rotated_max, rotated_profile.max_nonempty.back());
        }
        const std::string prefix = "lanes_" + std::to_string(lanes) + "_";
        RecordProperty(prefix + "worst_rotated_to_current", decimal(worst_rotated_to_current));
        RecordProperty(prefix + "worst_ratio_prefix_tiles", worst_ratio_prefix);
        RecordProperty(prefix + "worst_ratio_rank", worst_ratio_rank);
        RecordProperty(prefix + "worst_current_max_to_mean", decimal(worst_current_max_to_mean));
        RecordProperty(prefix + "worst_rotated_max_to_mean", decimal(worst_rotated_max_to_mean));
        RecordProperty(prefix + "worst_rotated_max_to_mean_prefix_tiles", worst_rotated_max_to_mean_prefix);
        RecordProperty(prefix + "worst_rotated_max_to_mean_rank", worst_rotated_max_to_mean_rank);
        RecordProperty(
            prefix + "worst_current_valid_tiles_max_to_mean", decimal(worst_current_valid_tiles_max_to_mean));
        RecordProperty(
            prefix + "worst_rotated_valid_tiles_max_to_mean", decimal(worst_rotated_valid_tiles_max_to_mean));
        RecordProperty(prefix + "worst_rotated_valid_tiles_to_current", decimal(worst_rotated_valid_tiles_to_current));
        RecordProperty(prefix + "worst_rotated_valid_tiles_prefix_tiles", worst_rotated_valid_tiles_prefix);
        RecordProperty(prefix + "worst_rotated_valid_tiles_rank", worst_rotated_valid_tiles_rank);
        RecordProperty(prefix + "full_prefix_current_max", full_prefix_current_max);
        RecordProperty(prefix + "full_prefix_rotated_max", full_prefix_rotated_max);
        EXPECT_EQ(full_prefix_current_max, lanes == 20 ? 192u : 160u);
        EXPECT_EQ(full_prefix_rotated_max, lanes == 20 ? 164u : 152u);
    }
}

}  // namespace
}  // namespace ttnn::operations::experimental::indexer_score::program::ring_schedule
