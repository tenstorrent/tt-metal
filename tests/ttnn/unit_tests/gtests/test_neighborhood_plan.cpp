// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for 3D neighborhood attention geometry. No device.
//
// The oracle here derives each context window by SEARCHING all placements rather than by
// computing an origin, so it cannot share an off-by-one with the implementation it checks.

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_plan.hpp"

namespace ttnn::transformer::neighborhood {
namespace {

// The NATTEN rule, by brute force: of every in-bounds placement of a window of `extent` that
// contains `site_index`, take the one most centred on it.
uint32_t oracle_window_origin(uint32_t site_index, uint32_t volume_extent_sites, uint32_t context_window_extent_sites) {
    const uint32_t window_extent_sites = std::min(context_window_extent_sites, volume_extent_sites);

    uint32_t best_origin = 0;
    uint32_t best_distance_to_centre = UINT32_MAX;
    for (uint32_t candidate_origin = 0; candidate_origin + window_extent_sites <= volume_extent_sites;
         ++candidate_origin) {
        const bool contains_site =
            site_index >= candidate_origin && site_index < candidate_origin + window_extent_sites;
        if (!contains_site) {
            continue;
        }
        const uint32_t centre_site = candidate_origin + window_extent_sites / 2;
        const uint32_t distance_to_centre =
            centre_site > site_index ? centre_site - site_index : site_index - centre_site;
        if (distance_to_centre < best_distance_to_centre) {
            best_distance_to_centre = distance_to_centre;
            best_origin = candidate_origin;
        }
    }
    return best_origin;
}

NeighborhoodConfig make_config(Extent3 volume, Extent3 context_window, Extent3 stride, Extent3 brick) {
    return NeighborhoodConfig{volume, context_window, stride, brick};
}

// Small enough to brute force, awkward enough to catch edge bugs: no axis is a multiple of
// the brick on every case, and one axis is shorter than the window.
const std::vector<NeighborhoodConfig>& awkward_configs() {
    static const std::vector<NeighborhoodConfig> configs = {
        make_config(Extent3::of(8, 12, 12), Extent3::of(5, 5, 5), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4)),
        make_config(Extent3::of(7, 13, 11), Extent3::of(5, 5, 5), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4)),
        make_config(Extent3::of(3, 12, 12), Extent3::of(5, 7, 7), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4)),
        make_config(Extent3::of(8, 12, 12), Extent3::of(5, 5, 5), Extent3::of(2, 4, 4), Extent3::of(2, 4, 4)),
        make_config(Extent3::of(8, 12, 12), Extent3::of(5, 5, 5), Extent3::of(2, 2, 2), Extent3::of(2, 4, 4)),
        make_config(Extent3::of(6, 8, 16), Extent3::of(3, 7, 7), Extent3::of(1, 1, 1), Extent3::of(1, 4, 8)),
    };
    return configs;
}

// ---------------------------------------------------------------------------
// choose_brick
// ---------------------------------------------------------------------------

TEST(NeighborhoodChooseBrick, MatchesExhaustiveSearch) {
    const std::vector<Extent3> context_windows = {
        Extent3::of(11, 11, 11),
        Extent3::of(3, 7, 7),
        Extent3::of(3, 5, 5),
        Extent3::of(1, 11, 11),
        Extent3::of(7, 7, 7),
        Extent3::of(9, 3, 3),
    };

    for (const Extent3& context_window : context_windows) {
        const Extent3 chosen_brick = choose_brick(context_window);
        ASSERT_EQ(chosen_brick.sites(), SITES_PER_BRICK) << "brick must be exactly one tile";

        uint64_t chosen_union_sites = 1;
        for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
            chosen_union_sites *= context_window.by_axis[axis_index] + chosen_brick.by_axis[axis_index] - 1;
        }

        for (uint32_t time_extent = 1; time_extent <= SITES_PER_BRICK; time_extent *= 2) {
            for (uint32_t height_extent = 1; time_extent * height_extent <= SITES_PER_BRICK; height_extent *= 2) {
                if (SITES_PER_BRICK % (time_extent * height_extent) != 0) {
                    continue;
                }
                const uint32_t width_extent = SITES_PER_BRICK / (time_extent * height_extent);
                uint64_t candidate_union_sites = 1;
                candidate_union_sites *= context_window.time() + time_extent - 1;
                candidate_union_sites *= context_window.height() + height_extent - 1;
                candidate_union_sites *= context_window.width() + width_extent - 1;

                EXPECT_LE(chosen_union_sites, candidate_union_sites)
                    << "a better brick exists for window " << context_window.time() << "x" << context_window.height()
                    << "x" << context_window.width();
            }
        }
    }
}

TEST(NeighborhoodChooseBrick, FollowsWindowShape) {
    // A cubic window wants a cubic brick.
    EXPECT_EQ(choose_brick(Extent3::of(11, 11, 11)), Extent3::of(2, 4, 4));
    // A window flat in time wants a brick flat in time -- this is why it cannot be a constant.
    EXPECT_EQ(choose_brick(Extent3::of(1, 11, 11)), Extent3::of(1, 4, 8));
}

// ---------------------------------------------------------------------------
// context_window_for
// ---------------------------------------------------------------------------

TEST(NeighborhoodContextWindow, MatchesOracleAtStrideOne) {
    const NeighborhoodConfig config =
        make_config(Extent3::of(7, 13, 11), Extent3::of(5, 5, 5), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4));

    for (uint32_t site_time = 0; site_time < config.volume.time(); ++site_time) {
        for (uint32_t site_height = 0; site_height < config.volume.height(); ++site_height) {
            for (uint32_t site_width = 0; site_width < config.volume.width(); ++site_width) {
                const Site query_site = Site::at(site_time, site_height, site_width);
                const ContextWindow window = context_window_for(query_site, config);

                for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                    const uint32_t expected_origin = oracle_window_origin(
                        query_site.by_axis[axis_index],
                        config.volume.by_axis[axis_index],
                        config.context_window.by_axis[axis_index]);
                    EXPECT_EQ(window.origin.by_axis[axis_index], expected_origin)
                        << "axis " << axis_index << " at site " << site_time << "," << site_height << "," << site_width;
                }
            }
        }
    }
}

TEST(NeighborhoodContextWindow, KeepsExtentAndStaysInBounds) {
    for (const NeighborhoodConfig& config : awkward_configs()) {
        for (uint32_t site_time = 0; site_time < config.volume.time(); ++site_time) {
            for (uint32_t site_height = 0; site_height < config.volume.height(); ++site_height) {
                for (uint32_t site_width = 0; site_width < config.volume.width(); ++site_width) {
                    const ContextWindow window =
                        context_window_for(Site::at(site_time, site_height, site_width), config);

                    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                        const uint32_t volume_extent = config.volume.by_axis[axis_index];
                        const uint32_t expected_extent =
                            std::min(config.context_window.by_axis[axis_index], volume_extent);

                        // The window slides inward at a boundary; it never shrinks.
                        EXPECT_EQ(window.extent.by_axis[axis_index], expected_extent);
                        EXPECT_LE(window.origin.by_axis[axis_index] + window.extent.by_axis[axis_index], volume_extent);
                    }
                }
            }
        }
    }
}

TEST(NeighborhoodContextWindow, ContainsItsOwnQueryGroup) {
    for (const NeighborhoodConfig& config : awkward_configs()) {
        for (uint32_t site_time = 0; site_time < config.volume.time(); ++site_time) {
            for (uint32_t site_height = 0; site_height < config.volume.height(); ++site_height) {
                for (uint32_t site_width = 0; site_width < config.volume.width(); ++site_width) {
                    const Site query_site = Site::at(site_time, site_height, site_width);
                    const ContextWindow window = context_window_for(query_site, config);

                    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                        const uint32_t origin = window.origin.by_axis[axis_index];
                        const uint32_t extent = window.extent.by_axis[axis_index];
                        const uint32_t site = query_site.by_axis[axis_index];
                        EXPECT_GE(site, origin) << "query fell below its own window on axis " << axis_index;
                        EXPECT_LT(site, origin + extent) << "query fell above its own window on axis " << axis_index;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Natural <-> Bricked
// ---------------------------------------------------------------------------

TEST(NeighborhoodBrickedOrder, RoundTripsAndIsInjective) {
    for (const NeighborhoodConfig& config : awkward_configs()) {
        std::set<uint32_t> seen_bricked_indices;

        for (uint32_t site_time = 0; site_time < config.volume.time(); ++site_time) {
            for (uint32_t site_height = 0; site_height < config.volume.height(); ++site_height) {
                for (uint32_t site_width = 0; site_width < config.volume.width(); ++site_width) {
                    const Site original_site = Site::at(site_time, site_height, site_width);
                    const uint32_t bricked_index = site_to_bricked_index(original_site, config);

                    EXPECT_TRUE(seen_bricked_indices.insert(bricked_index).second)
                        << "two sites collided on bricked index " << bricked_index;
                    EXPECT_EQ(bricked_index_to_site(bricked_index, config), original_site);
                }
            }
        }
        EXPECT_EQ(seen_bricked_indices.size(), config.volume.sites());
    }
}

TEST(NeighborhoodBrickedOrder, PacksOneBrickContiguously) {
    // The whole point of bricking: 32 consecutive bricked indices are one compact 3D box.
    const NeighborhoodConfig config =
        make_config(Extent3::of(8, 12, 12), Extent3::of(5, 5, 5), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4));

    for (uint32_t brick_index = 0; brick_index < 6; ++brick_index) {
        const Site brick_origin = brick_index_to_origin(brick_index, config);

        for (uint32_t site_index_in_brick = 0; site_index_in_brick < SITES_PER_BRICK; ++site_index_in_brick) {
            const Site site = bricked_index_to_site(brick_index * SITES_PER_BRICK + site_index_in_brick, config);

            for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                EXPECT_GE(site.by_axis[axis_index], brick_origin.by_axis[axis_index]);
                EXPECT_LT(
                    site.by_axis[axis_index], brick_origin.by_axis[axis_index] + config.brick.by_axis[axis_index]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build_plan
// ---------------------------------------------------------------------------

TEST(NeighborhoodPlanBuild, GatherCoversEveryQueryWindowInTheBrick) {
    // The load-bearing invariant: whatever one brick gathers must contain the context window
    // of every query inside it. If this fails, the kernel silently reads the wrong keys.
    for (const NeighborhoodConfig& config : awkward_configs()) {
        const NeighborhoodPlan plan = build_plan(config);

        for (uint32_t brick_index = 0; brick_index < plan.brick_count; ++brick_index) {
            const Site brick_origin = brick_index_to_origin(brick_index, config);
            const Site gather_origin = plan.gather_origin_by_brick[brick_index];

            for (uint32_t site_in_brick_time = 0; site_in_brick_time < config.brick.time(); ++site_in_brick_time) {
                for (uint32_t site_in_brick_height = 0; site_in_brick_height < config.brick.height();
                     ++site_in_brick_height) {
                    for (uint32_t site_in_brick_width = 0; site_in_brick_width < config.brick.width();
                         ++site_in_brick_width) {
                        const Site query_site = Site::at(
                            brick_origin.time() + site_in_brick_time,
                            brick_origin.height() + site_in_brick_height,
                            brick_origin.width() + site_in_brick_width);

                        // Bricks may overhang a volume that is not a brick multiple.
                        const bool inside_volume = query_site.time() < config.volume.time() &&
                                                   query_site.height() < config.volume.height() &&
                                                   query_site.width() < config.volume.width();
                        if (!inside_volume) {
                            continue;
                        }

                        // Checked against the WHOLE-BRICK span, because that is what a
                        // tile-granular read actually fetches.
                        const ContextWindow window = context_window_for(query_site, config);
                        for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                            const uint32_t brick_span_sites =
                                plan.gather_bricks.by_axis[axis_index] * config.brick.by_axis[axis_index];
                            EXPECT_GE(window.origin.by_axis[axis_index], gather_origin.by_axis[axis_index])
                                << "gather starts after the window on axis " << axis_index << ", brick " << brick_index;
                            EXPECT_LE(
                                window.origin.by_axis[axis_index] + window.extent.by_axis[axis_index],
                                gather_origin.by_axis[axis_index] + brick_span_sites)
                                << "gather ends before the window on axis " << axis_index << ", brick " << brick_index;
                        }
                    }
                }
            }
        }
    }
}

TEST(NeighborhoodPlanBuild, GatherStaysInBoundsAndIsConstantSize) {
    for (const NeighborhoodConfig& config : awkward_configs()) {
        const NeighborhoodPlan plan = build_plan(config);

        EXPECT_EQ(plan.gather_sites, plan.gather_extent.sites());
        EXPECT_EQ(plan.gather_origin_by_brick.size(), plan.brick_count);

        for (const Site& gather_origin : plan.gather_origin_by_brick) {
            for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                const uint32_t brick_extent_sites = config.brick.by_axis[axis_index];
                const uint32_t brick_span_sites = plan.gather_bricks.by_axis[axis_index] * brick_extent_sites;
                const uint32_t padded_volume_sites = plan.volume_bricks.by_axis[axis_index] * brick_extent_sites;

                EXPECT_EQ(gather_origin.by_axis[axis_index] % brick_extent_sites, 0u)
                    << "gather origin is not on a brick boundary, axis " << axis_index;
                EXPECT_LE(gather_origin.by_axis[axis_index] + brick_span_sites, padded_volume_sites)
                    << "whole-brick gather ran off the padded volume on axis " << axis_index;
            }
        }
    }
}

TEST(NeighborhoodPlanBuild, StrideEqualToBrickGathersExactlyOneContextWindow) {
    // The payoff of striding: one window per brick, no union, nothing to mask.
    const NeighborhoodConfig config =
        make_config(Extent3::of(8, 12, 12), Extent3::of(5, 5, 5), Extent3::of(2, 4, 4), Extent3::of(2, 4, 4));
    const NeighborhoodPlan plan = build_plan(config);

    EXPECT_EQ(plan.gather_extent, Extent3::of(5, 5, 5));
    EXPECT_EQ(plan.gather_sites, 125u);
}

TEST(NeighborhoodPlanBuild, StrideOneGathersTheUnionOfTheBricksWindows) {
    // At stride 1 the 32 queries have 32 distinct windows: window + brick - 1 on each axis.
    const NeighborhoodConfig config =
        make_config(Extent3::of(16, 24, 24), Extent3::of(5, 5, 5), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4));
    const NeighborhoodPlan plan = build_plan(config);

    EXPECT_EQ(plan.gather_extent, Extent3::of(6, 8, 8));
}

TEST(NeighborhoodPlanBuild, RejectsUnbuildableConfigs) {
    // A brick that is not one tile.
    EXPECT_THROW(
        build_plan(make_config(Extent3::of(8, 8, 8), Extent3::of(3, 3, 3), Extent3::of(1, 1, 1), Extent3::of(2, 2, 2))),
        std::invalid_argument);

    // A stride wider than the window would put a query outside its own context.
    EXPECT_THROW(
        build_plan(make_config(Extent3::of(8, 8, 8), Extent3::of(3, 3, 3), Extent3::of(4, 1, 1), Extent3::of(2, 4, 4))),
        std::invalid_argument);

    // A zero extent.
    EXPECT_THROW(
        build_plan(make_config(Extent3::of(8, 0, 8), Extent3::of(3, 3, 3), Extent3::of(1, 1, 1), Extent3::of(2, 4, 4))),
        std::invalid_argument);
}

}  // namespace
}  // namespace ttnn::transformer::neighborhood
