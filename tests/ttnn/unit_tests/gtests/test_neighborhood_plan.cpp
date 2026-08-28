// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for 3D neighborhood attention geometry. No device.
//
// The oracle here derives each context window by SEARCHING all placements rather than by
// computing an origin, so it cannot share an off-by-one with the implementation it checks.

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_plan.hpp"

namespace ttnn::transformer::neighborhood {

// ---- the point types, pinned at compile time ----
//
// A position is Point3<Scalar, Unit>, where Unit is a PHANTOM tag: it distinguishes a site from a
// brick from a chunk in the type system while adding nothing to the object. Both halves of that
// claim are load-bearing and neither is visible at a call site, so both are asserted here.
//
// Before this existed, a position was spelled five different ways -- Site, Offset3,
// SignedAxisOffsets, SiteInBrick, BrickCoordinate -- which were distinct types by NAME only.
// Assigning a brick coordinate into a site compiled cleanly and was wrong by a factor of the
// brick shape, and the symptom was a mask that read the wrong keys and still returned plausible
// video. These assertions are what make that a build failure instead.

// The tag costs nothing: same size, same layout, still trivially copyable into a kernel argument.
static_assert(sizeof(Site) == AXIS_COUNT * sizeof(uint32_t));
static_assert(sizeof(BrickPoint) == sizeof(Site));
static_assert(sizeof(ChunkPoint) == sizeof(Site));
static_assert(sizeof(SiteOffset) == AXIS_COUNT * sizeof(int32_t));
static_assert(std::is_trivially_copyable_v<Site>);
static_assert(std::is_trivially_copyable_v<SiteOffset>);

// The tag bites: units do not convert into one another, in either direction.
static_assert(!std::is_assignable_v<Site&, BrickPoint>);
static_assert(!std::is_assignable_v<BrickPoint&, Site>);
static_assert(!std::is_assignable_v<BrickPoint&, ChunkPoint>);
static_assert(!std::is_assignable_v<ChunkPoint&, BrickPoint>);
// Nor does signedness, which is why the shard origin cannot be a Site: see SiteOffset.
static_assert(!std::is_assignable_v<Site&, SiteOffset>);
static_assert(!std::is_assignable_v<SiteOffset&, Site>);
// Same unit still assigns and compares, or nothing above would be usable.
static_assert(std::is_assignable_v<Site&, Site>);
static_assert(Site::at(1, 2, 3) == Site::at(1, 2, 3));
static_assert(!(Site::at(1, 2, 3) == Site::at(1, 2, 4)));

// The conversions are the ONLY route between units, and they fold at compile time. Worked with
// the shipped brick (2, 4, 4): brick (3, 1, 2) begins at site (6, 4, 8), and every site in that
// brick -- (7, 5, 9) among them -- maps back to it.
static_assert(first_site_of(BrickPoint::at(3, 1, 2), BrickShapeInSites::of(2, 4, 4)) == Site::at(6, 4, 8));
static_assert(containing_brick(Site::at(6, 4, 8), BrickShapeInSites::of(2, 4, 4)) == BrickPoint::at(3, 1, 2));
static_assert(containing_brick(Site::at(7, 5, 9), BrickShapeInSites::of(2, 4, 4)) == BrickPoint::at(3, 1, 2));
static_assert(first_brick_of(ChunkPoint::at(2, 1, 0), ChunkShapeInBricks::of(4, 2, 2)) == BrickPoint::at(8, 2, 0));

// ---- and the SCALE argument is typed too ----
//
// A tagged point with an untyped extent would check only half of each call, and the open half is
// the one that silently rescales a position: `first_site_of(brick_point, config.volume)` once
// compiled and multiplied a brick coordinate by the whole volume. A unit ratio is not an ShapeInSites,
// and the two ratios are not each other -- brick shape is sites-per-brick, chunk shape is
// bricks-per-chunk, and they sit in the same argument slot of adjacent functions.
template <typename PointT, typename ScaleT>
concept ScalesToSite = requires(PointT point, ScaleT scale) { first_site_of(point, scale); };
template <typename PointT, typename ScaleT>
concept ScalesToBrick = requires(PointT point, ScaleT scale) { first_brick_of(point, scale); };
template <typename PointT, typename ScaleT>
concept DividesToBrick = requires(PointT point, ScaleT scale) { containing_brick(point, scale); };

static_assert(ScalesToSite<BrickPoint, BrickShapeInSites>);
static_assert(!ScalesToSite<BrickPoint, ShapeInSites>);        // a size is not a scale
static_assert(!ScalesToSite<BrickPoint, ChunkShapeInBricks>);  // bricks-per-chunk is not sites-per-brick
static_assert(!ScalesToSite<ChunkPoint, BrickShapeInSites>);   // and a chunk is not a brick

static_assert(ScalesToBrick<ChunkPoint, ChunkShapeInBricks>);
static_assert(!ScalesToBrick<ChunkPoint, BrickShapeInSites>);
static_assert(!ScalesToBrick<ChunkPoint, ShapeInSites>);

static_assert(DividesToBrick<Site, BrickShapeInSites>);
static_assert(!DividesToBrick<Site, ShapeInSites>);
static_assert(!DividesToBrick<Site, ChunkShapeInBricks>);

// product() rather than sites(), because on a ChunkShapeInBricks the answer is a BRICK count. ShapeInSites
// keeps sites() and is still used for every genuine size.
// count() rather than sites(): the old name asserted a unit it could not know, and was false at
// every one of its non-site call sites -- volume_bricks.sites() returned a BRICK count.
static_assert(BrickShapeInSites::of(2, 4, 4).count() == SITES_PER_BRICK);
static_assert(ChunkShapeInBricks::of(4, 2, 2).count() == 16);
static_assert(ShapeInBricks::of(3, 4, 5).count() == 60);

// ---- what unifying Extent3, AxisExtents and UnitRatio into Shape newly makes checkable ----
//
// A region shape now carries the unit it counts, so a brick grid cannot be handed where a site
// region belongs. The old untagged Extent3 could not say this: gather_extent (sites) compared
// equal to gather_bricks (bricks), and either could be passed to the other's function.
static_assert(!std::is_assignable_v<ShapeInSites&, ShapeInBricks>);
static_assert(!std::is_assignable_v<ShapeInBricks&, ShapeInSites>);
static_assert(!std::is_assignable_v<ShapeInBricks&, ShapeInChunks>);
static_assert(!std::is_assignable_v<ShapeInChunks&, ShapeInBricks>);

// A region is not a unit shape, in either direction: PER distinguishes "how big is this" from
// "what do I scale by".
static_assert(!std::is_assignable_v<ShapeInSites&, BrickShapeInSites>);
static_assert(!std::is_assignable_v<BrickShapeInSites&, ShapeInSites>);
static_assert(!std::is_assignable_v<ShapeInBricks&, ChunkShapeInBricks>);
static_assert(!std::is_assignable_v<BrickShapeInSites&, ChunkShapeInBricks>);

// The PER tag is phantom: unifying three structs into one template costs no space.
static_assert(sizeof(ShapeInSites) == AXIS_COUNT * sizeof(uint32_t));
static_assert(sizeof(BrickShapeInSites) == sizeof(ShapeInSites));
static_assert(sizeof(ShapeInChunks) == sizeof(ShapeInSites));
static_assert(std::is_trivially_copyable_v<ShapeInSites>);
static_assert(std::is_trivially_copyable_v<BrickShapeInSites>);

// ONE conversion pair serves both levels, with both units read off the unit shape.
static_assert(first_point_of(BrickPoint::at(3, 1, 2), BrickShapeInSites::of(2, 4, 4)) == Site::at(6, 4, 8));
static_assert(first_point_of(ChunkPoint::at(2, 1, 0), ChunkShapeInBricks::of(4, 2, 2)) == BrickPoint::at(8, 2, 0));
static_assert(containing_unit(Site::at(7, 5, 9), BrickShapeInSites::of(2, 4, 4)) == BrickPoint::at(3, 1, 2));

// Axis indexing is a plain array load on shapes too, which is what retires the per-axis hoists
// the mask generator used to need.
static_assert(ShapeInSites::of(4, 5, 6)[Axis::Height] == 5);
static_assert(ALL_AXES.size() == AXIS_COUNT);

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

NeighborhoodConfig make_config(
    ShapeInSites volume, ShapeInSites context_window, ShapeInSites stride, BrickShapeInSites brick) {
    return NeighborhoodConfig{volume, context_window, stride, brick};
}

// Small enough to brute force, awkward enough to catch edge bugs: no axis is a multiple of
// the brick on every case, and one axis is shorter than the window.
const std::vector<NeighborhoodConfig>& awkward_configs() {
    static const std::vector<NeighborhoodConfig> configs = {
        make_config(
            ShapeInSites::of(8, 12, 12),
            ShapeInSites::of(5, 5, 5),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(2, 4, 4)),
        make_config(
            ShapeInSites::of(7, 13, 11),
            ShapeInSites::of(5, 5, 5),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(2, 4, 4)),
        make_config(
            ShapeInSites::of(3, 12, 12),
            ShapeInSites::of(5, 7, 7),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(2, 4, 4)),
        make_config(
            ShapeInSites::of(8, 12, 12),
            ShapeInSites::of(5, 5, 5),
            ShapeInSites::of(2, 4, 4),
            BrickShapeInSites::of(2, 4, 4)),
        make_config(
            ShapeInSites::of(8, 12, 12),
            ShapeInSites::of(5, 5, 5),
            ShapeInSites::of(2, 2, 2),
            BrickShapeInSites::of(2, 4, 4)),
        make_config(
            ShapeInSites::of(6, 8, 16),
            ShapeInSites::of(3, 7, 7),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(1, 4, 8)),
    };
    return configs;
}

// ---------------------------------------------------------------------------
// choose_brick
// ---------------------------------------------------------------------------

TEST(NeighborhoodChooseBrick, MatchesExhaustiveSearch) {
    const std::vector<ShapeInSites> context_windows = {
        ShapeInSites::of(11, 11, 11),
        ShapeInSites::of(3, 7, 7),
        ShapeInSites::of(3, 5, 5),
        ShapeInSites::of(1, 11, 11),
        ShapeInSites::of(7, 7, 7),
        ShapeInSites::of(9, 3, 3),
    };

    for (const ShapeInSites& context_window : context_windows) {
        const BrickShapeInSites chosen_brick = choose_brick(context_window);
        ASSERT_EQ(chosen_brick.count(), SITES_PER_BRICK) << "brick must be exactly one tile";

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
    EXPECT_EQ(choose_brick(ShapeInSites::of(11, 11, 11)), BrickShapeInSites::of(2, 4, 4));
    // A window flat in time wants a brick flat in time -- this is why it cannot be a constant.
    EXPECT_EQ(choose_brick(ShapeInSites::of(1, 11, 11)), BrickShapeInSites::of(1, 4, 8));
}

// ---------------------------------------------------------------------------
// context_window_for
// ---------------------------------------------------------------------------

TEST(NeighborhoodContextWindow, MatchesOracleAtStrideOne) {
    const NeighborhoodConfig config = make_config(
        ShapeInSites::of(7, 13, 11),
        ShapeInSites::of(5, 5, 5),
        ShapeInSites::of(1, 1, 1),
        BrickShapeInSites::of(2, 4, 4));

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
// build_plan
// ---------------------------------------------------------------------------

TEST(NeighborhoodPlanBuild, GatherCoversEveryQueryWindowInTheChunk) {
    // The load-bearing invariant: whatever one chunk gathers must contain the context window of
    // every query inside it. If this fails, the kernel silently reads the wrong keys.
    //
    // The gather table is indexed by CHUNK, and a chunk is a box of query_chunk_bricks bricks
    // sharing one gather -- so the scan below is over the chunk's whole site extent, not one
    // brick's. The chunk origin is decoded here rather than borrowed from the planner, on the same
    // principle as the window oracle above: an independent transcription cannot share a bug with
    // the thing it checks.
    for (const NeighborhoodConfig& config : awkward_configs()) {
        const NeighborhoodPlan plan = build_plan(config);
        const ShapeInSites chunk_sites = config.query_chunk_sites();

        for (uint32_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
            const uint32_t chunks_per_time_slice = plan.volume_chunks.height() * plan.volume_chunks.width();
            const uint32_t remainder = chunk_index % chunks_per_time_slice;
            const Site chunk_origin = Site::at(
                (chunk_index / chunks_per_time_slice) * chunk_sites.time(),
                (remainder / plan.volume_chunks.width()) * chunk_sites.height(),
                (remainder % plan.volume_chunks.width()) * chunk_sites.width());
            const Site gather_origin = plan.gather_origin_by_chunk[chunk_index];

            for (uint32_t site_in_chunk_time = 0; site_in_chunk_time < chunk_sites.time(); ++site_in_chunk_time) {
                for (uint32_t site_in_chunk_height = 0; site_in_chunk_height < chunk_sites.height();
                     ++site_in_chunk_height) {
                    for (uint32_t site_in_chunk_width = 0; site_in_chunk_width < chunk_sites.width();
                         ++site_in_chunk_width) {
                        const Site query_site = Site::at(
                            chunk_origin.time() + site_in_chunk_time,
                            chunk_origin.height() + site_in_chunk_height,
                            chunk_origin.width() + site_in_chunk_width);

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
                                << "gather starts after the window on axis " << axis_index << ", chunk " << chunk_index;
                            EXPECT_LE(
                                window.origin.by_axis[axis_index] + window.extent.by_axis[axis_index],
                                gather_origin.by_axis[axis_index] + brick_span_sites)
                                << "gather ends before the window on axis " << axis_index << ", chunk " << chunk_index;
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

        EXPECT_EQ(plan.gather_sites, plan.gather_extent.count());
        EXPECT_EQ(plan.gather_origin_by_chunk.size(), plan.chunk_count);

        for (const Site& gather_origin : plan.gather_origin_by_chunk) {
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
    const NeighborhoodConfig config = make_config(
        ShapeInSites::of(8, 12, 12),
        ShapeInSites::of(5, 5, 5),
        ShapeInSites::of(2, 4, 4),
        BrickShapeInSites::of(2, 4, 4));
    const NeighborhoodPlan plan = build_plan(config);

    EXPECT_EQ(plan.gather_extent, ShapeInSites::of(5, 5, 5));
    EXPECT_EQ(plan.gather_sites, 125u);
}

TEST(NeighborhoodPlanBuild, StrideOneGathersTheUnionOfTheBricksWindows) {
    // At stride 1 the 32 queries have 32 distinct windows: window + brick - 1 on each axis.
    const NeighborhoodConfig config = make_config(
        ShapeInSites::of(16, 24, 24),
        ShapeInSites::of(5, 5, 5),
        ShapeInSites::of(1, 1, 1),
        BrickShapeInSites::of(2, 4, 4));
    const NeighborhoodPlan plan = build_plan(config);

    EXPECT_EQ(plan.gather_extent, ShapeInSites::of(6, 8, 8));
}

TEST(NeighborhoodPlanBuild, RejectsUnbuildableConfigs) {
    // A brick that is not one tile.
    EXPECT_THROW(
        build_plan(make_config(
            ShapeInSites::of(8, 8, 8),
            ShapeInSites::of(3, 3, 3),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(2, 2, 2))),
        std::invalid_argument);

    // A stride wider than the window would put a query outside its own context.
    EXPECT_THROW(
        build_plan(make_config(
            ShapeInSites::of(8, 8, 8),
            ShapeInSites::of(3, 3, 3),
            ShapeInSites::of(4, 1, 1),
            BrickShapeInSites::of(2, 4, 4))),
        std::invalid_argument);

    // A zero extent.
    EXPECT_THROW(
        build_plan(make_config(
            ShapeInSites::of(8, 0, 8),
            ShapeInSites::of(3, 3, 3),
            ShapeInSites::of(1, 1, 1),
            BrickShapeInSites::of(2, 4, 4))),
        std::invalid_argument);
}

}  // namespace
}  // namespace ttnn::transformer::neighborhood
