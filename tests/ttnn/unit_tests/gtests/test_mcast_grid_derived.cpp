// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grid-derived multicast arguments: fixed-sender groups that tile a rectangle, with their senders at one offset or
// staggered along lines, carry no per-core words; each core derives its roles, its sender and its group's NoC
// rectangle from a common-arg NoC table.

#include "mcast_host_test_common.hpp"

#include <functional>
#include <string>

namespace ttnn::kernel_lib::host::test {
namespace {
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

struct CoreView {
    uint32_t roles = 0;
    std::vector<uint32_t> sender;  // Receivers: the sender's NoC x, y.
    std::vector<uint32_t> bounds;  // Senders: the NoC-ordered rectangle.
    bool operator==(const CoreView&) const = default;
};

uint32_t named(const KernelDescriptor& kernel, const std::string& name) {
    for (const auto& [key, value] : kernel.named_compile_time_args) {
        if (key == name) {
            return value;
        }
    }
    ADD_FAILURE() << "missing named compile-time arg " << name;
    return 0;
}

wire::ArgumentMetadata attached_metadata(const KernelDescriptor& kernel, const std::string& prefix) {
    return wire::decode_compile_time_metadata(
        kernel.compile_time_args.data() + named(kernel, prefix + "_ct_offset"), true);
}

// What a core of `kernel` reads for channel `prefix`: from the grid table when the channel is grid-derived, else
// from its per-core words (fixed sender, one rectangle).
CoreView attached_view(const KernelDescriptor& kernel, const std::string& prefix, CoreCoord core) {
    const auto metadata = attached_metadata(kernel, prefix);
    CoreView view;
    if (metadata.grid.derived) {
        const auto& grid = metadata.grid;
        const std::vector<uint32_t> table(
            kernel.common_runtime_args.begin() + grid.common_base,
            kernel.common_runtime_args.begin() + grid.common_base + grid.columns + grid.rows);
        const auto position = wire::grid_position(grid, core.x, core.y);
        view.roles = metadata.kernel.roles == wire::DYNAMIC_ROLES ? wire::grid_roles(position) : metadata.kernel.roles;
        if (view.roles & wire::CAN_RECEIVE) {
            view.sender = {
                wire::grid_sender_coordinate(table, grid, position, wire::SENDER_X),
                wire::grid_sender_coordinate(table, grid, position, wire::SENDER_Y)};
        }
        if (view.roles & wire::CAN_SEND) {
            const auto b = wire::grid_bounds(table, grid, position, (metadata.mcast.flags & wire::NOC1) != 0);
            view.bounds = {b.sx, b.sy, b.ex, b.ey};
        }
        return view;
    }
    const wire::RuntimeLayout layout(metadata);
    std::vector<uint32_t> words;
    for (const auto& [placed, args] : kernel.runtime_args) {
        if (placed == core) {
            words.assign(args.begin() + named(kernel, prefix + "_rt_offset"), args.end());
        }
    }
    EXPECT_EQ(words.size(), layout.words);
    view.roles = layout.roles == wire::OMITTED ? metadata.kernel.roles : words.at(layout.roles);
    if ((view.roles & wire::CAN_RECEIVE) && layout.sender_coordinates != wire::OMITTED) {
        view.sender = {words.at(layout.sender_coordinates), words.at(layout.sender_coordinates + 1)};
    }
    if ((view.roles & wire::CAN_SEND) && layout.rectangle_bounds != wire::OMITTED) {
        const uint32_t base = layout.rectangles + layout.rectangle_bounds;
        view.bounds = {words.at(base), words.at(base + 1), words.at(base + 2), words.at(base + 3)};
    }
    return view;
}

// Independent oracle: each group is a logical rectangle with one sender; map it with the device directly.
struct ExpectedGroup {
    CoreRange receivers;
    CoreCoord sender;
};

CoreView expected_view(
    tt::tt_metal::distributed::MeshDevice* device, const std::vector<ExpectedGroup>& groups, NOC noc, CoreCoord core) {
    for (const auto& group : groups) {
        if (!group.receivers.contains(core)) {
            continue;
        }
        const auto sender = device->worker_core_from_logical_core(group.sender);
        if (core != group.sender) {
            return {.roles = wire::CAN_RECEIVE, .sender = {uint32_t(sender.x), uint32_t(sender.y)}};
        }
        const auto lo = device->worker_core_from_logical_core(group.receivers.start_coord);
        const auto hi = device->worker_core_from_logical_core(group.receivers.end_coord);
        const std::vector<uint32_t> bounds =
            noc == NOC::NOC_0 ? std::vector<uint32_t>{uint32_t(lo.x), uint32_t(lo.y), uint32_t(hi.x), uint32_t(hi.y)}
                              : std::vector<uint32_t>{uint32_t(hi.x), uint32_t(hi.y), uint32_t(lo.x), uint32_t(lo.y)};
        return {.roles = wire::CAN_SEND, .bounds = bounds};
    }
    return {};
}

KernelDescriptor make_kernel(const CoreRangeSet& cores, NOC noc) {
    KernelDescriptor kernel;
    kernel.core_ranges = cores;
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = noc};
    return kernel;
}

std::vector<uint32_t> noc_table(tt::tt_metal::distributed::MeshDevice* device, CoreRange grid) {
    std::vector<uint32_t> table;
    for (auto x = grid.start_coord.x; x <= grid.end_coord.x; ++x) {
        table.push_back(device->worker_core_from_logical_core({x, grid.start_coord.y}).x);
    }
    for (auto y = grid.start_coord.y; y <= grid.end_coord.y; ++y) {
        table.push_back(device->worker_core_from_logical_core({grid.start_coord.x, y}).y);
    }
    return table;
}

struct Layout {
    const char* name;
    CoreRange receivers;
    uint32_t group_size;
    McastCoreOrder order;
    uint32_t sender_index;
    std::vector<ExpectedGroup> groups;
    wire::GridMetadata grid;
    McastSenderPlacement placement = McastSenderPlacement::Uniform;
};

std::vector<Layout> layouts() {
    return {
        // One group per column, the sender in the second row (AGMM's in1 pattern with a moved sender).
        {"columns",
         CoreRange({0, 0}, {2, 3}),
         4,
         McastCoreOrder::ColumnMajor,
         1,
         {{CoreRange({0, 0}, {0, 3}), {0, 1}},
          {CoreRange({1, 0}, {1, 3}), {1, 1}},
          {CoreRange({2, 0}, {2, 3}), {2, 1}}},
         {.derived = true,
          .origin_x = 0,
          .origin_y = 0,
          .columns = 3,
          .rows = 4,
          .group_width = 1,
          .group_height = 4,
          .sender_dx = 0,
          .sender_dy = 1}},
        // One group per row, offset from the grid origin.
        {"rows",
         CoreRange({1, 1}, {4, 2}),
         4,
         McastCoreOrder::RowMajor,
         2,
         {{CoreRange({1, 1}, {4, 1}), {3, 1}}, {CoreRange({1, 2}, {4, 2}), {3, 2}}},
         {.derived = true,
          .origin_x = 1,
          .origin_y = 1,
          .columns = 4,
          .rows = 2,
          .group_width = 4,
          .group_height = 1,
          .sender_dx = 2,
          .sender_dy = 0}},
        // Half rows: two groups per row.
        {"half_rows",
         CoreRange({0, 0}, {3, 1}),
         2,
         McastCoreOrder::RowMajor,
         0,
         {{CoreRange({0, 0}, {1, 0}), {0, 0}},
          {CoreRange({2, 0}, {3, 0}), {2, 0}},
          {CoreRange({0, 1}, {1, 1}), {0, 1}},
          {CoreRange({2, 1}, {3, 1}), {2, 1}}},
         {.derived = true,
          .origin_x = 0,
          .origin_y = 0,
          .columns = 4,
          .rows = 2,
          .group_width = 2,
          .group_height = 1,
          .sender_dx = 0,
          .sender_dy = 0}},
        // One group over the whole rectangle (a 1D multicast).
        {"one_group",
         CoreRange({0, 0}, {2, 1}),
         6,
         McastCoreOrder::RowMajor,
         4,
         {{CoreRange({0, 0}, {2, 1}), {1, 1}}},
         {.derived = true,
          .origin_x = 0,
          .origin_y = 0,
          .columns = 3,
          .rows = 2,
          .group_width = 3,
          .group_height = 2,
          .sender_dx = 1,
          .sender_dy = 1}},
        // Staggered rows: row n's sender is column (1 + n) % 4, wrapping on the last row (MMRS's diagonal senders).
        {"staggered_rows",
         CoreRange({1, 0}, {4, 3}),
         4,
         McastCoreOrder::RowMajor,
         1,
         {{CoreRange({1, 0}, {4, 0}), {2, 0}},
          {CoreRange({1, 1}, {4, 1}), {3, 1}},
          {CoreRange({1, 2}, {4, 2}), {4, 2}},
          {CoreRange({1, 3}, {4, 3}), {1, 3}}},
         {.derived = true,
          .staggered = true,
          .origin_x = 1,
          .origin_y = 0,
          .columns = 4,
          .rows = 4,
          .group_width = 4,
          .group_height = 1,
          .sender_dx = 1,
          .sender_dy = 0},
         McastSenderPlacement::Staggered},
        // Staggered columns, starting past the line length: column n's sender is row (5 + n) % 4.
        {"staggered_columns",
         CoreRange({0, 1}, {2, 4}),
         4,
         McastCoreOrder::ColumnMajor,
         5,
         {{CoreRange({0, 1}, {0, 4}), {0, 2}},
          {CoreRange({1, 1}, {1, 4}), {1, 3}},
          {CoreRange({2, 1}, {2, 4}), {2, 4}}},
         {.derived = true,
          .staggered = true,
          .origin_x = 0,
          .origin_y = 1,
          .columns = 3,
          .rows = 4,
          .group_width = 1,
          .group_height = 4,
          .sender_dx = 0,
          .sender_dy = 1},
         McastSenderPlacement::Staggered},
        // Staggered half rows: lines are numbered row by row, two per row.
        {"staggered_half_rows",
         CoreRange({0, 0}, {5, 1}),
         3,
         McastCoreOrder::RowMajor,
         0,
         {{CoreRange({0, 0}, {2, 0}), {0, 0}},
          {CoreRange({3, 0}, {5, 0}), {4, 0}},
          {CoreRange({0, 1}, {2, 1}), {2, 1}},
          {CoreRange({3, 1}, {5, 1}), {3, 1}}},
         {.derived = true,
          .staggered = true,
          .origin_x = 0,
          .origin_y = 0,
          .columns = 6,
          .rows = 2,
          .group_width = 3,
          .group_height = 1,
          .sender_dx = 0,
          .sender_dy = 0},
         McastSenderPlacement::Staggered},
    };
}

}  // namespace

// Each core of a grid-derived channel reads what the per-core path would give it and what the device mapping says.
TEST_F(McastHostFixture, GridDerivedDescriptorMatchesPerCoreAndDeviceMapping) {
    for (const auto& layout : layouts()) {
        for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
            SCOPED_TRACE(::testing::Message() << layout.name << " noc=" << noc);
            // One kernel over the bounding grid plus a column outside it (inactive cores), and a kernel per role.
            const CoreRange placed(
                layout.receivers.start_coord, {layout.receivers.end_coord.x + 1, layout.receivers.end_coord.y});
            std::vector<CoreCoord> senders;
            for (const auto& group : layout.groups) {
                senders.push_back(group.sender);
            }
            const auto sender_cores = cores(senders);
            const auto receiver_cores = CoreRangeSet(layout.receivers).subtract(sender_cores);
            for (const bool derive : {true, false}) {
                Mcast mcast(
                    *device_,
                    McastConfig{.noc = noc, .derive_regular_grid = derive},
                    CoreRangeSet(layout.receivers),
                    layout.group_size,
                    McastFixedSenderConfig{.sender_index = layout.sender_index, .placement = layout.placement},
                    layout.order);
                ProgramDescriptor descriptor;
                auto all = make_kernel(CoreRangeSet(placed), noc);
                all.common_runtime_args = {71};
                all.runtime_args = {{layout.receivers.start_coord, {5}}};
                auto sender = make_kernel(sender_cores, noc);
                auto receiver = make_kernel(receiver_cores, noc);
                mcast.attach(descriptor, "channel", std::array{std::ref(all), std::ref(sender), std::ref(receiver)}, 0);
                for (const auto* kernel : {&all, &sender, &receiver}) {
                    const auto metadata = attached_metadata(*kernel, "channel");
                    EXPECT_EQ(metadata.grid.derived, derive);
                    for (const auto& core : tt::tt_metal::corerange_to_cores(kernel->core_ranges)) {
                        EXPECT_EQ(
                            attached_view(*kernel, "channel", core), expected_view(device_, layout.groups, noc, core))
                            << "core " << core.str()
                            << (kernel == &all      ? " (all)"
                                : kernel == &sender ? " (sender)"
                                                    : " (receiver)");
                    }
                }
                if (!derive) {
                    continue;
                }
                auto expected_grid = layout.grid;
                expected_grid.common_base = 1;
                const auto grid = attached_metadata(all, "channel").grid;
                EXPECT_EQ(grid.staggered, expected_grid.staggered);
                EXPECT_EQ(
                    std::vector<uint32_t>(
                        {grid.origin_x,
                         grid.origin_y,
                         grid.columns,
                         grid.rows,
                         grid.group_width,
                         grid.group_height,
                         grid.sender_dx,
                         grid.sender_dy,
                         grid.common_base}),
                    std::vector<uint32_t>(
                        {expected_grid.origin_x,
                         expected_grid.origin_y,
                         expected_grid.columns,
                         expected_grid.rows,
                         expected_grid.group_width,
                         expected_grid.group_height,
                         expected_grid.sender_dx,
                         expected_grid.sender_dy,
                         expected_grid.common_base}));
                // The caller's words stay; the table follows them; no core gains per-core words or padding.
                auto expected_common = std::vector<uint32_t>{71};
                const auto table = noc_table(device_, layout.receivers);
                expected_common.insert(expected_common.end(), table.begin(), table.end());
                EXPECT_EQ(all.common_runtime_args, expected_common);
                EXPECT_EQ(all.runtime_args, (KernelDescriptor::RuntimeArgs{{layout.receivers.start_coord, {5}}}));
                EXPECT_TRUE(sender.runtime_args.empty());
                EXPECT_TRUE(receiver.runtime_args.empty());
                EXPECT_EQ(sender.common_runtime_args, table);
                EXPECT_EQ(attached_metadata(sender, "channel").kernel.roles, wire::CAN_SEND);
                EXPECT_EQ(attached_metadata(receiver, "channel").kernel.roles, wire::CAN_RECEIVE);
                EXPECT_EQ(attached_metadata(all, "channel").kernel.roles, wire::DYNAMIC_ROLES);
            }
        }
    }
}

// Two channels on one kernel each append their own table after the previous one.
TEST_F(McastHostFixture, GridDerivedChannelsStackTheirTables) {
    const CoreRange receivers({0, 0}, {2, 1});
    ProgramDescriptor descriptor;
    auto kernel = make_kernel(CoreRangeSet(receivers), NOC::NOC_0);
    kernel.common_runtime_args = {7, 9};
    Mcast rows(*device_, {}, CoreRangeSet(receivers), 3);
    Mcast columns(*device_, {}, CoreRangeSet(receivers), 2, McastFixedSenderConfig{}, McastCoreOrder::ColumnMajor);
    rows.attach(descriptor, "rows", std::array{std::ref(kernel)}, 0);
    columns.attach(descriptor, "columns", std::array{std::ref(kernel)}, rows.next_semaphore_id());
    EXPECT_EQ(attached_metadata(kernel, "rows").grid.common_base, 2u);
    EXPECT_EQ(attached_metadata(kernel, "columns").grid.common_base, 7u);
    const auto table = noc_table(device_, receivers);
    auto expected = std::vector<uint32_t>{7, 9};
    expected.insert(expected.end(), table.begin(), table.end());
    expected.insert(expected.end(), table.begin(), table.end());
    EXPECT_EQ(kernel.common_runtime_args, expected);
    EXPECT_TRUE(kernel.runtime_args.empty());
}

// The spec adapter appends the same table to the common varargs, after the caller's.
TEST_F(McastHostFixture, GridDerivedSpecAppendsCommonVarargs) {
    const CoreRange receivers({0, 0}, {3, 0});
    const std::array targets{m2::KernelSpecName{"test"}};
    const auto make_spec = [&] {
        m2::ProgramSpec spec{
            .kernels =
                {{.unique_id = targets.front(),
                  .source = m2::KernelSpec::SourceCode{"void kernel_main() {}"},
                  .hw_config =
                      m2::DataMovementHardwareConfig{
                          .config_1xx =
                              m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                                  .processor = tt::tt_metal::DataMovementProcessor::RISCV_0}}}},
            .work_units = {{.name = "test", .kernels = {targets.front()}, .target_nodes = CoreRangeSet(receivers)}}};
        spec.kernels.front().advanced_options.num_runtime_varargs = 1;
        spec.kernels.front().advanced_options.num_common_runtime_varargs = 1;
        return spec;
    };
    const auto make_args = [&](std::vector<uint32_t> common) {
        m2::ProgramRunArgs args;
        args.kernel_run_args.push_back({.kernel = targets.front()});
        auto& run = args.kernel_run_args.front().advanced_options;
        run.common_runtime_varargs = std::move(common);
        for (const auto& core : tt::tt_metal::corerange_to_cores(CoreRangeSet(receivers))) {
            run.runtime_varargs[core] = {uint32_t(core.x)};
        }
        return args;
    };
    const auto attached_grid = [](const m2::KernelSpec& kernel) {
        const auto ct_base = kernel.compile_time_args.get("channel_mcast_ct_base").value().to_uint32();
        return wire::decode_compile_time_metadata(kernel.advanced_options.compile_time_varargs.data() + ct_base, false);
    };

    auto spec = make_spec();
    auto args = make_args({33});
    const auto before = args;
    Mcast mcast(*device_, {}, CoreRangeSet(receivers), 4, McastFixedSenderConfig{.sender_index = 1});
    mcast.attach(spec, args, "channel", targets);
    const auto& kernel = spec.kernels.front();
    const auto metadata = attached_grid(kernel);
    EXPECT_TRUE(metadata.grid.derived);
    EXPECT_EQ(metadata.grid.common_base, 1u);
    EXPECT_EQ(metadata.grid.sender_dx, 1u);
    EXPECT_EQ(metadata.kernel.roles, wire::DYNAMIC_ROLES);
    EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, 1u);
    EXPECT_EQ(kernel.advanced_options.num_common_runtime_varargs, 6u);
    auto expected = std::vector<uint32_t>{33};
    const auto table = noc_table(device_, receivers);
    expected.insert(expected.end(), table.begin(), table.end());
    EXPECT_EQ(args.kernel_run_args.front().advanced_options.common_runtime_varargs, expected);
    EXPECT_EQ(
        args.kernel_run_args.front().advanced_options.runtime_varargs,
        before.kernel_run_args.front().advanced_options.runtime_varargs);

    // Common varargs declared but not yet given values: the table has no fixed place, so the channel stays per-core.
    auto pending_spec = make_spec();
    auto pending_args = make_args({});
    Mcast pending(*device_, {}, CoreRangeSet(receivers), 4, McastFixedSenderConfig{.sender_index = 1});
    pending.attach(pending_spec, pending_args, "channel", targets);
    EXPECT_FALSE(attached_grid(pending_spec.kernels.front()).grid.derived);
    EXPECT_EQ(pending_spec.kernels.front().advanced_options.num_common_runtime_varargs, 1u);
    EXPECT_GT(pending_spec.kernels.front().advanced_options.num_runtime_varargs, 1u);
}

// Layouts a core cannot derive keep per-core words.
TEST_F(McastHostFixture, GridDerivedFallsBackToPerCoreWords) {
    struct Case {
        const char* name;
        std::function<Mcast()> make;
        CoreRangeSet placement;
    };
    const CoreRange rows({0, 0}, {3, 1});
    const std::vector<Case> cases{
        {"rotating",
         [&] { return Mcast(*device_, {}, CoreRangeSet(rows), 4, McastRotatingSenderConfig{}); },
         CoreRangeSet(rows)},
        {"sender_outside",
         [&] {
             return Mcast(
                 *device_, {}, CoreRangeSet(CoreRange({0, 0}, {3, 0})), 4, McastExplicitFixedSenderConfig{{{0, 1}}});
         },
         CoreRangeSet(CoreRange({0, 0}, {3, 1}))},
        // Staggered 2x2 blocks: the sender moves within a block, which only lines can derive.
        {"staggered_blocks",
         [&] {
             return Mcast(
                 *device_,
                 {},
                 CoreRangeSet(CoreRange({0, 0}, {1, 3})),
                 4,
                 McastFixedSenderConfig{.sender_index = 0, .placement = McastSenderPlacement::Staggered});
         },
         CoreRangeSet(CoreRange({0, 0}, {1, 3}))},
        {"irregular",
         [&] { return Mcast(*device_, {}, cores({{0, 0}, {2, 0}, {0, 1}, {2, 1}}), 2); },
         CoreRangeSet(rows)},
        {"varying_acks",
         [&] { return Mcast(*device_, McastConfig{.handshake_cores = cores({{1, 0}})}, CoreRangeSet(rows), 4); },
         CoreRangeSet(rows)},
        {"opted_out",
         [&] { return Mcast(*device_, McastConfig{.derive_regular_grid = false}, CoreRangeSet(rows), 4); },
         CoreRangeSet(rows)},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        auto mcast = c.make();
        ProgramDescriptor descriptor;
        auto kernel = make_kernel(c.placement, NOC::NOC_0);
        mcast.attach(descriptor, "channel", std::array{std::ref(kernel)}, 0);
        EXPECT_FALSE(attached_metadata(kernel, "channel").grid.derived);
        EXPECT_TRUE(kernel.common_runtime_args.empty());
        EXPECT_FALSE(kernel.runtime_args.empty());
    }
    // A direct Program has no place for the table.
    Mcast mcast(*device_, {}, CoreRangeSet(rows), 4);
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram(*device_);
    tt::tt_metal::DataMovementConfig config{};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> runtime_args;
    mcast.attach(program, CoreRangeSet(rows), config, "channel", runtime_args);
    EXPECT_FALSE(wire::decode_compile_time_metadata(config.compile_args.data(), true).grid.derived);
    EXPECT_EQ(runtime_args.size(), rows.size());
}

}  // namespace ttnn::kernel_lib::host::test
