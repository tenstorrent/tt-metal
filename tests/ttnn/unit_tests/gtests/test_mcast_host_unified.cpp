// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <set>
#include <type_traits>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host_unified.hpp"
#include "ttnn_test_fixtures.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace ttnn::kernel_lib::host::test {
// Shared independent test decoder; compact CT goldens pin its bit/field positions.
dataflow_kernel_lib::mcast_wire::ArgumentMetadata emitted_metadata(const std::vector<uint32_t>& ct, uint32_t base);
}  // namespace ttnn::kernel_lib::host::test

namespace ttnn::kernel_lib::host::unified_test {
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
namespace wire = dataflow_kernel_lib::mcast_wire;
namespace m2 = tt::tt_metal::experimental;

template <typename T>
concept ExposesCollectionOrPreparation = requires(T value) {
    value.add_group(CoreRangeSet{}, std::vector<CoreCoord>{});
} || requires(T value) { value.prepare_arguments(); };
static_assert(!ExposesCollectionOrPreparation<Mcast>);
static_assert(std::is_constructible_v<
              Mcast,
              tt::tt_metal::IDevice*,
              const McastUnifiedConfig&,
              const CoreRangeSet&,
              uint32_t,
              McastCoreOrder>);
static_assert(!std::is_constructible_v<
              Mcast,
              tt::tt_metal::IDevice*,
              const McastUnifiedConfig&,
              const std::vector<CoreCoord>&,
              uint32_t,
              McastCoreOrder>);

class McastUnifiedFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastUnifiedFixture> {};

CoreRangeSet grid(CoreCoord start, CoreCoord end) { return CoreRangeSet(CoreRange(start, end)); }
CoreRangeSet core_set(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(std::move(ranges));
}

template <typename Family>
KernelDescriptor emitted(const Family& family, NOC noc = NOC::NOC_0, const std::vector<uint32_t>& adopted_ids = {}) {
    tt::tt_metal::ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.core_ranges = family.participating_cores();
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc};
    for (const auto id : adopted_ids) {
        descriptor.semaphores.push_back({.id = id, .core_ranges = kernel.core_ranges, .initial_value = 0});
    }
    family.attach(descriptor, "channel", std::array{std::ref(kernel)});
    return kernel;
}

void expect_pattern(
    const Mcast& channel,
    tt::tt_metal::IDevice* device,
    const std::vector<std::vector<CoreCoord>>& receivers,
    const std::vector<std::vector<CoreCoord>>& senders,
    const std::vector<std::vector<uint32_t>>& acks,
    NOC noc = NOC::NOC_0) {
    const auto kernel = emitted(channel, noc);
    const auto metadata = test::emitted_metadata(kernel.compile_time_args, 0);
    const wire::RuntimeLayout layout(metadata);
    std::vector<CoreCoord> physical_workers;
    const auto size = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < size.y; ++y) {
        for (uint32_t x = 0; x < size.x; ++x) {
            physical_workers.push_back(device->worker_core_from_logical_core({x, y}));
        }
    }
    std::set<CoreCoord> expected_participants, expected_receivers;
    for (const auto& group : receivers) {
        expected_receivers.insert(group.begin(), group.end());
        expected_participants.insert(group.begin(), group.end());
    }
    for (const auto& group : senders) {
        expected_participants.insert(group.begin(), group.end());
    }
    EXPECT_EQ(channel.participating_cores().num_cores(), expected_participants.size());
    for (const auto& core : expected_participants) {
        EXPECT_TRUE(channel.participating_cores().contains(core));
        EXPECT_EQ(channel.sender_only_cores().contains(core), !expected_receivers.contains(core));
    }
    for (const auto& [core, words] : kernel.runtime_args) {
        const uint32_t roles = layout.roles == wire::OMITTED ? metadata.kernel.roles : words.at(layout.roles);
        bool found = false;
        for (size_t group = 0; group < receivers.size(); ++group) {
            const auto sender = std::find(senders[group].begin(), senders[group].end(), core);
            const bool can_send = sender != senders[group].end();
            const bool receives =
                std::find(receivers[group].begin(), receivers[group].end(), core) != receivers[group].end();
            if (!can_send && !receives) {
                continue;
            }
            ASSERT_FALSE(found);
            found = true;
            EXPECT_EQ(roles & 1u, can_send ? 1u : 0u);
            EXPECT_EQ(roles & 2u, receives && (!can_send || senders[group].size() > 1) ? 2u : 0u);
            if (can_send) {
                const size_t phase = sender - senders[group].begin();
                EXPECT_EQ(layout.sender_phase == wire::OMITTED ? 0u : words.at(layout.sender_phase), phase);
                EXPECT_EQ(
                    layout.ack == wire::OMITTED ? metadata.family.ack_count : words.at(layout.ack),
                    (metadata.family.flags & wire::PRE_HANDSHAKE) ? acks[group][phase] : 0u);
                // Independently enumerate the emitted worker destinations, not
                // merely the frontend's participant list. Partial handshakes
                // must not shrink the destination or add bounding-box holes.
                std::set<CoreCoord> expected, actual;
                for (const auto& receiver : receivers[group]) {
                    expected.insert(device->worker_core_from_logical_core(receiver));
                }
                if (layout.rectangle_bounds == wire::OMITTED) {
                    actual.insert(device->worker_core_from_logical_core(core));
                } else {
                    const uint32_t count =
                        layout.rectangle_count == wire::OMITTED ? 1u : words.at(layout.rectangle_count);
                    for (uint32_t rectangle = 0; rectangle < count; ++rectangle) {
                        const uint32_t base =
                            layout.rectangles + rectangle * layout.rectangle_stride + layout.rectangle_bounds;
                        const auto sx = words.at(base), sy = words.at(base + 1);
                        const auto ex = words.at(base + 2), ey = words.at(base + 3);
                        EXPECT_EQ(sx, noc == NOC::NOC_0 ? std::min(sx, ex) : std::max(sx, ex));
                        EXPECT_EQ(sy, noc == NOC::NOC_0 ? std::min(sy, ey) : std::max(sy, ey));
                        for (const auto& worker : physical_workers) {
                            if (worker.x >= std::min(sx, ex) && worker.x <= std::max(sx, ex) &&
                                worker.y >= std::min(sy, ey) && worker.y <= std::max(sy, ey)) {
                                EXPECT_TRUE(actual.insert(worker).second) << "Overlapping multicast rectangles";
                            }
                        }
                    }
                }
                EXPECT_EQ(actual, expected);
            }
            if (layout.sender_coordinates != wire::OMITTED) {
                const auto payload = std::span(words).subspan(layout.sender_coordinates, layout.coordinate_words);
                for (size_t phase = 0; phase < senders[group].size(); ++phase) {
                    const auto expected = device->worker_core_from_logical_core(senders[group][phase]);
                    EXPECT_EQ(wire::sender_coordinate(payload, metadata.coordinates, phase, 0), expected.x);
                    EXPECT_EQ(wire::sender_coordinate(payload, metadata.coordinates, phase, 1), expected.y);
                }
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(McastUnifiedFixture, RowsColumnsAndWholeGrid) {
    const auto receivers = grid({0, 0}, {2, 1});
    for (const auto order : {McastCoreOrder::RowMajor, McastCoreOrder::ColumnMajor}) {
        const bool rows = order == McastCoreOrder::RowMajor;
        const std::vector<std::vector<CoreCoord>> groups =
            rows ? std::vector<std::vector<CoreCoord>>{{{0, 0}, {1, 0}, {2, 0}}, {{0, 1}, {1, 1}, {2, 1}}}
                 : std::vector<std::vector<CoreCoord>>{{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}};
        for (bool rotating : {false, true}) {
            Mcast channel(
                device_,
                {},
                receivers,
                rows ? 3 : 2,
                order,
                rotating ? McastSenderConfig{McastRotatingSenderConfig{}}
                         : McastSenderConfig{McastFixedSenderConfig{}});
            McastFamily legacy(device_);
            std::vector<std::vector<CoreCoord>> senders;
            std::vector<std::vector<uint32_t>> acks;
            for (const auto& group : groups) {
                senders.push_back(rotating ? group : std::vector<CoreCoord>{group.front()});
                acks.emplace_back(senders.back().size(), group.size() - 1);
                legacy.add_group(core_set(group), senders.back());
            }
            expect_pattern(channel, device_, groups, senders, acks);
            EXPECT_EQ(emitted(channel).compile_time_args, emitted(legacy).compile_time_args);
            EXPECT_EQ(emitted(channel).runtime_args, emitted(legacy).runtime_args);
        }
    }
    Mcast all(device_, {}, receivers, 6, McastCoreOrder::RowMajor);
    expect_pattern(all, device_, {{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}}}, {{{0, 0}}}, {{5}});
    Mcast local(device_, {}, grid({0, 0}, {1, 0}), 1, McastCoreOrder::RowMajor);
    expect_pattern(local, device_, {{{0, 0}}, {{1, 0}}}, {{{0, 0}}, {{1, 0}}}, {{0}, {0}});
}

TEST_F(McastUnifiedFixture, StaggeredAndSortedReceiverOrder) {
    const auto receivers = grid({0, 0}, {3, 1});
    for (const auto order : {McastCoreOrder::RowMajor, McastCoreOrder::ColumnMajor}) {
        const bool rows = order == McastCoreOrder::RowMajor;
        const std::vector<std::vector<CoreCoord>> groups =
            rows ? std::vector<
                       std::vector<CoreCoord>>{{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {{0, 1}, {1, 1}, {2, 1}, {3, 1}}}
                 : std::vector<std::vector<CoreCoord>>{
                       {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {{2, 0}, {2, 1}, {3, 0}, {3, 1}}};
        const std::vector<std::vector<CoreCoord>> senders =
            rows ? std::vector<std::vector<CoreCoord>>{{{3, 0}}, {{0, 1}}}
                 : std::vector<std::vector<CoreCoord>>{{{1, 1}}, {{2, 0}}};
        Mcast channel(
            device_,
            {},
            receivers,
            4,
            order,
            McastFixedSenderConfig{.sender_index = UINT32_MAX, .placement = McastSenderPlacement::Staggered});
        expect_pattern(channel, device_, groups, senders, {{3}, {3}});
        Mcast rotating(device_, {}, receivers, 4, order, McastRotatingSenderConfig{});
        expect_pattern(rotating, device_, groups, groups, {{3, 3, 3, 3}, {3, 3, 3, 3}});
    }
}

TEST_F(McastUnifiedFixture, SenderGridOrderingIsNotSpatialInference) {
    const auto receivers = grid({0, 0}, {1, 1});
    Mcast channel(
        device_,
        {},
        receivers,
        2,
        McastCoreOrder::RowMajor,
        McastSenderGridConfig{.sender_cores = grid({3, 0}, {4, 1}), .sender_order = McastCoreOrder::ColumnMajor});
    expect_pattern(
        channel, device_, {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}}, {{{3, 0}, {3, 1}}, {{4, 0}, {4, 1}}}, {{2, 2}, {2, 2}});
    Mcast fixed(
        device_,
        {},
        receivers,
        2,
        McastCoreOrder::RowMajor,
        McastSenderGridConfig{.sender_cores = grid({3, 0}, {4, 0})});
    expect_pattern(fixed, device_, {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}}, {{{3, 0}}, {{4, 0}}}, {{2}, {2}});
}

TEST_F(McastUnifiedFixture, HandshakeSubsetOwnsDataAndVariesBySender) {
    const auto receivers = grid({0, 0}, {3, 1});
    auto handshake = core_set({{0, 0}, {1, 0}, {1, 1}});
    McastUnifiedConfig config{.handshake_cores = &handshake, .base_sem_id = 5};
    McastExplicitSenderConfig senders{.senders_per_group = {{{0, 0}, {3, 0}}, {{0, 1}, {1, 1}}}};
    Mcast channel(device_, config, receivers, 4, McastCoreOrder::RowMajor, senders);
    handshake = CoreRangeSet{};
    config.base_sem_id = 0;
    senders.senders_per_group.clear();
    auto copy = channel;
    expect_pattern(
        copy,
        device_,
        {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {{0, 1}, {1, 1}, {2, 1}, {3, 1}}},
        {{{0, 0}, {3, 0}}, {{0, 1}, {1, 1}}},
        {{1, 2}, {1, 0}});
    const auto snapshot = emitted(copy);
    EXPECT_EQ(snapshot.compile_time_args.at(1), 5u);
    EXPECT_EQ(test::emitted_metadata(snapshot.compile_time_args, 0).family.ack_count, UINT32_MAX);
    tt::tt_metal::Program program;
    copy.append_semaphores(program);
    std::vector<uint32_t> ct, first_rt, second_rt;
    copy.append_compile_time_args_to(ct);
    copy.append_runtime_args_to(first_rt, {0, 0});
    copy.append_runtime_args_to(second_rt, {3, 0});
    EXPECT_EQ(test::emitted_metadata(ct, 0).family.ack_count, UINT32_MAX);
    EXPECT_EQ(first_rt.at(2), 1u);  // Generic roles, phase, then per-sender ACK.
    EXPECT_EQ(second_rt.at(2), 2u);
}

TEST_F(McastUnifiedFixture, DefaultEmptyAndDisabledHandshakes) {
    const auto receivers = grid({0, 0}, {1, 0});
    const CoreRangeSet empty;
    Mcast no_acks(
        device_,
        McastUnifiedConfig{.handshake_cores = &empty},
        receivers,
        2,
        McastCoreOrder::RowMajor,
        McastRotatingSenderConfig{});
    expect_pattern(no_acks, device_, {{{0, 0}, {1, 0}}}, {{{0, 0}, {1, 0}}}, {{0, 0}});
    Mcast default_acks(device_, {}, receivers, 2, McastCoreOrder::RowMajor, McastRotatingSenderConfig{});
    expect_pattern(default_acks, device_, {{{0, 0}, {1, 0}}}, {{{0, 0}, {1, 0}}}, {{1, 1}});
    Mcast disabled(device_, McastUnifiedConfig{.handshake = false}, receivers, 2, McastCoreOrder::RowMajor);
    const auto snapshot = emitted(disabled);
    EXPECT_EQ(test::emitted_metadata(snapshot.compile_time_args, 0).family.flags & 1u, 0u);
    EXPECT_EQ(wire::CompileTimeLayout(snapshot.compile_time_args.front()).consumer_ready, wire::OMITTED);
    EXPECT_ANY_THROW((Mcast(
        device_,
        McastUnifiedConfig{.handshake = false, .handshake_cores = &empty},
        receivers,
        2,
        McastCoreOrder::RowMajor)));
}

TEST_F(McastUnifiedFixture, RejectsInvalidGroupingAndSchedules) {
    const auto receivers = grid({0, 0}, {3, 0});
    for (uint32_t size : {0u, 3u, 5u}) {
        EXPECT_ANY_THROW((Mcast(device_, {}, receivers, size, McastCoreOrder::RowMajor)));
    }
    EXPECT_ANY_THROW((Mcast(device_, {}, CoreRangeSet{}, 1, McastCoreOrder::RowMajor)));
    EXPECT_ANY_THROW(
        (Mcast(device_, {}, receivers, 2, McastCoreOrder::RowMajor, McastFixedSenderConfig{.sender_index = 2})));
    for (const auto& lists : std::vector<std::vector<std::vector<CoreCoord>>>{
             {},
             {{{0, 0}}},
             {{{0, 0}}, {}},
             {{{0, 0}, {0, 0}}, {{2, 0}, {3, 0}}},
             {{{0, 0}}, {{2, 0}, {3, 0}}},
             {{{2, 0}}, {{3, 0}}}}) {
        EXPECT_ANY_THROW(
            (Mcast(device_, {}, receivers, 2, McastCoreOrder::RowMajor, McastExplicitSenderConfig{lists})));
    }
    for (const auto& senders : {CoreRangeSet{}, grid({0, 1}, {2, 1})}) {
        EXPECT_ANY_THROW((Mcast(device_, {}, receivers, 2, McastCoreOrder::RowMajor, McastSenderGridConfig{senders})));
    }
    const auto outside = grid({0, 1}, {0, 1});
    EXPECT_ANY_THROW(
        (Mcast(device_, McastUnifiedConfig{.handshake_cores = &outside}, receivers, 4, McastCoreOrder::RowMajor)));
}

TEST_F(McastUnifiedFixture, ChainSelectionPreservesParticipationLimits) {
    const auto irregular = core_set({{0, 0}, {2, 0}});
    const auto partial = grid({0, 0}, {0, 0});
    const CoreRangeSet empty;
    McastUnifiedConfig config{.irregular_receiver_set_mode = dataflow_kernel_lib::TransferMode::ChainUnicast};
    Mcast legacy_equivalent(device_, config, irregular, 2, McastCoreOrder::RowMajor);
    auto expected = McastFamily(
        device_, McastConfig{.irregular_receiver_set_mode = dataflow_kernel_lib::TransferMode::ChainUnicast});
    expected.add_group(irregular, {{0, 0}});
    EXPECT_EQ(emitted(legacy_equivalent).compile_time_args, emitted(expected).compile_time_args);
    EXPECT_EQ(emitted(legacy_equivalent).runtime_args, emitted(expected).runtime_args);
    config.handshake_cores = &irregular;
    EXPECT_NO_THROW((Mcast(device_, config, irregular, 2, McastCoreOrder::RowMajor)));
    for (const auto* subset : {&partial, &empty}) {
        config.handshake_cores = subset;
        EXPECT_ANY_THROW((Mcast(device_, config, irregular, 2, McastCoreOrder::RowMajor)));
        EXPECT_NO_THROW((Mcast(device_, config, grid({0, 0}, {1, 0}), 2, McastCoreOrder::RowMajor)));
    }
    config.handshake_cores = nullptr;
    EXPECT_ANY_THROW((Mcast(device_, config, irregular, 2, McastCoreOrder::RowMajor, McastRotatingSenderConfig{})));
    config.handshake = false;
    EXPECT_ANY_THROW((Mcast(device_, config, irregular, 2, McastCoreOrder::RowMajor)));
}

TEST_F(McastUnifiedFixture, WrappedGroupsAndRectangleLimit) {
    const std::vector<std::vector<CoreCoord>> groups{
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}, {0, 2}},
        {{1, 2}, {2, 2}, {3, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 3}, {0, 4}, {1, 4}}};
    auto ordered = groups.front();
    ordered.insert(ordered.end(), groups.back().begin(), groups.back().end());
    Mcast wrapped(device_, {}, core_set(ordered), 9, McastCoreOrder::RowMajor);
    expect_pattern(wrapped, device_, groups, {{{0, 0}}, {{1, 2}}}, {{8}, {8}});
    const auto snapshot = emitted(wrapped);
    const wire::RuntimeLayout layout(test::emitted_metadata(snapshot.compile_time_args, 0));
    EXPECT_EQ(test::emitted_metadata(snapshot.compile_time_args, 0).family.rectangle_capacity, 3u);
    for (const auto& [core, words] : snapshot.runtime_args) {
        if (core == CoreCoord{0, 0}) {
            EXPECT_EQ(words.at(layout.rectangle_count), 2u);
        } else if (core == CoreCoord{1, 2}) {
            EXPECT_EQ(words.at(layout.rectangle_count), 3u);
        }
    }
    const auto too_fragmented = core_set({{0, 0}, {2, 0}, {4, 0}, {6, 0}});
    EXPECT_ANY_THROW((Mcast(device_, {}, too_fragmented, 4, McastCoreOrder::RowMajor)));
}

TEST_F(McastUnifiedFixture, DescriptorSpecAndDirectAttachmentParity) {
    const auto receivers = grid({0, 0}, {3, 0});
    const auto handshake = grid({0, 0}, {1, 0});
    Mcast channel(
        device_,
        McastUnifiedConfig{.handshake_cores = &handshake},
        receivers,
        4,
        McastCoreOrder::RowMajor,
        McastExplicitSenderConfig{{{{0, 0}, {3, 0}}}});
    tt::tt_metal::ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.core_ranges = receivers;
    kernel.compile_time_args = {73};
    kernel.config =
        tt::tt_metal::DataMovementConfigDescriptor{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0};
    kernel.runtime_args = {{{0, 0}, {11}}, {{1, 0}, {13, 17}}, {{2, 0}, {}}, {{3, 0}, {19}}};
    auto direct_rt = kernel.runtime_args;
    auto direct_ct = kernel.compile_time_args;
    channel.attach(descriptor, "channel", std::array{std::ref(kernel)});
    auto direct = channel;
    tt::tt_metal::Program program;
    direct.append_semaphores(program);
    const auto offsets = direct.append_kernel_args_to(direct_ct, direct_rt, receivers);
    EXPECT_EQ(offsets.compile_time, 1u);
    EXPECT_EQ(offsets.runtime, 2u);
    EXPECT_EQ(direct_ct, kernel.compile_time_args);
    EXPECT_EQ(direct_rt, kernel.runtime_args);
    EXPECT_EQ(kernel.runtime_args.front().second.at(1), 0u);

    const std::array targets{m2::KernelSpecName{"test"}};
    m2::ProgramSpec spec{
        .kernels =
            {{.unique_id = targets.front(),
              .source = m2::KernelSpec::SourceCode{"void kernel_main() {}"},
              .hw_config = m2::DataMovementGen1Config{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0}}},
        .work_units = {{.name = "test", .kernels = {targets.front()}, .target_nodes = receivers}}};
    spec.kernels.front().advanced_options.num_runtime_varargs = 2;
    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back({.kernel = targets.front()});
    for (const auto& [core, words] : direct_rt) {
        args.kernel_run_args.front().advanced_options.runtime_varargs[core] = {words[0], words[1]};
    }
    auto adopted_spec = spec;
    auto adopted_args = args;
    const std::array adopted{m2::SemaphoreSpecName{"publication"}, m2::SemaphoreSpecName{"acknowledgments"}};
    for (const auto& name : adopted) {
        adopted_spec.semaphores.push_back({.unique_id = name, .target_nodes = receivers});
    }
    channel.attach(adopted_spec, adopted_args, "channel", targets, adopted);
    ASSERT_EQ(adopted_spec.semaphores.size(), 2u);
    ASSERT_EQ(adopted_spec.kernels.front().semaphore_bindings.size(), 2u);
    for (size_t i = 0; i < adopted.size(); ++i) {
        EXPECT_EQ(adopted_spec.kernels.front().semaphore_bindings[i].semaphore_spec_name, adopted[i]);
    }
    // Explicit numeric configuration is owned too, but remains inappropriate for
    // native named-resource attachment, as with the legacy backend.
    McastUnifiedConfig numeric_config{.sem_ids = std::vector<uint32_t>{4, 7}};
    Mcast numeric(device_, numeric_config, receivers, 4, McastCoreOrder::RowMajor);
    numeric_config.sem_ids->clear();
    const auto numeric_snapshot = emitted(numeric, NOC::NOC_0, {4, 7});
    EXPECT_EQ(numeric_snapshot.compile_time_args.at(1), 4u);
    EXPECT_EQ(numeric_snapshot.compile_time_args.at(2), 7u);
    EXPECT_ANY_THROW(numeric.attach(spec, args, "numeric", targets));
    EXPECT_TRUE(spec.semaphores.empty());
    channel.attach(spec, args, "channel", targets);
    EXPECT_EQ(spec.semaphores.size(), 2u);
    const auto& spec_kernel = spec.kernels.front();
    const auto spec_ct_base = spec_kernel.compile_time_args.get("channel_mcast_ct_base").value();
    EXPECT_EQ(
        wire::decode_compile_time_metadata(
            spec_kernel.advanced_options.compile_time_varargs.data() + spec_ct_base, false)
            .family.ack_count,
        UINT32_MAX);
    for (const auto& [core, words] : kernel.runtime_args) {
        EXPECT_EQ(args.kernel_run_args.front().advanced_options.runtime_varargs.get(core).value(), words);
    }
    auto copied_spec = spec;
    auto copied_args = args;
    Mcast second(device_, McastUnifiedConfig{.handshake = false}, receivers, 4, McastCoreOrder::RowMajor);
    second.attach(copied_spec, copied_args, "second", targets);
    EXPECT_EQ(copied_spec.semaphores.size(), 3u);
    EXPECT_EQ(spec.semaphores.size(), 2u);
    for (const auto& [core, words] : args.kernel_run_args.front().advanced_options.runtime_varargs) {
        const auto expanded = copied_args.kernel_run_args.front().advanced_options.runtime_varargs.get(core).value();
        EXPECT_GT(expanded.size(), words.size());
        EXPECT_TRUE(std::equal(words.begin(), words.end(), expanded.begin()));
    }
}

TEST_F(McastUnifiedFixture, OperationCommunicationFeasibility) {
    struct Pattern {
        const char* name;
        std::vector<std::vector<CoreCoord>> receivers;
        std::vector<std::vector<CoreCoord>> senders;
        std::vector<std::vector<uint32_t>> acks;
        std::optional<CoreRangeSet> handshake;
        bool handshake_enabled = true;
        dataflow_kernel_lib::DataReadySignal signal = dataflow_kernel_lib::DataReadySignal::Flag;
        McastCoreOrder order = McastCoreOrder::RowMajor;
    };
    const std::vector<CoreCoord> block{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}};
    const auto workers = core_set({{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}});
    const std::vector<Pattern> cases{
        {"Matmul fixed in0", {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}}, {{{0, 0}}}, {{3}}},
        {"Matmul rotating rows",
         {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}},
         {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}},
         {{1, 1}, {1, 1}}},
        {"DRAM storage/compute overlap",
         {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}},
         {{{0, 0}, {3, 0}}},
         {{1, 2}},
         grid({0, 0}, {1, 0})},
        {"Conv2D rotating activation, passive landing",
         {block},
         {{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}}},
         {{5, 5, 5, 5, 5, 5}},
         workers},
        {"Conv3D weight strip, passive tail", {block}, {{{0, 0}}}, {{5}}, workers},
        {"LayerNorm readiness, bounding-box holes", {block}, {{{0, 0}}}, {{5}}, workers},
        {"LayerNorm final statistics, no handshake",
         {block},
         {{{0, 0}}},
         {{7}},
         std::nullopt,
         false,
         dataflow_kernel_lib::DataReadySignal::Counter},
        // Sharded GroupNorm keeps whole channel groups per shard. For N=1,
        // spatial=64, C=128, groups=4, and 32x32 shards, reductions span only
        // the spatial axis: columns on a row-major 4x2 grid, rows when transposed.
        {.name = "GroupNorm block-sharded column reductions",
         .receivers = {{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}, {{3, 0}, {3, 1}}},
         .senders = {{{0, 0}}, {{1, 0}}, {{2, 0}}, {{3, 0}}},
         .acks = {{1}, {1}, {1}, {1}},
         .order = McastCoreOrder::ColumnMajor},
        {"GroupNorm block-sharded row reductions",
         {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}, {{0, 2}, {1, 2}}, {{0, 3}, {1, 3}}},
         {{{0, 0}}, {{0, 1}}, {{0, 2}}, {{0, 3}}},
         {{1}, {1}, {1}, {1}}},
        {"Attention sender-only storage", {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}}, {block}, {{3, 3, 3, 3, 4, 4, 4, 4}}},
        {"TopK external readiness sender", {{{0, 0}, {1, 0}, {2, 0}}}, {{{3, 0}}}, {{3}}, std::nullopt, false}};
    for (const auto& pattern : cases) {
        SCOPED_TRACE(pattern.name);
        std::vector<CoreCoord> ordered;
        for (const auto& group : pattern.receivers) {
            ordered.insert(ordered.end(), group.begin(), group.end());
        }
        for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
            Mcast channel(
                device_,
                McastUnifiedConfig{
                    .noc = noc,
                    .handshake = pattern.handshake_enabled,
                    .handshake_cores = pattern.handshake ? &*pattern.handshake : nullptr,
                    .data_ready = pattern.signal},
                core_set(ordered),
                pattern.receivers.front().size(),
                pattern.order,
                McastExplicitSenderConfig{pattern.senders});
            expect_pattern(channel, device_, pattern.receivers, pattern.senders, pattern.acks, noc);
            const auto flags = test::emitted_metadata(emitted(channel, noc).compile_time_args, 0).family.flags;
            EXPECT_EQ(flags & 1u, pattern.handshake_enabled ? 1u : 0u);
            EXPECT_EQ(flags & 2u, pattern.signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u);
        }
    }
}

void run_mixed_ack_device(tt::tt_metal::distributed::MeshDevice& device, NOC noc, uint32_t rounds) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;
    const auto receivers = grid({0, 0}, {3, 0});
    const auto workers = grid({0, 0}, {1, 0});
    Mcast channel(
        &device,
        McastUnifiedConfig{
            .noc = noc, .handshake_cores = &workers, .data_ready = dataflow_kernel_lib::DataReadySignal::Counter},
        receivers,
        4,
        McastCoreOrder::RowMajor,
        McastExplicitSenderConfig{{{{0, 0}, {3, 0}}}});
    expect_pattern(channel, &device, {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}}, {{{0, 0}, {3, 0}}}, {{1, 2}}, noc);
    const std::array targets{m2::KernelSpecName{"mixed"}};
    m2::KernelSpec kernel{
        .unique_id = targets.front(),
        .source = "tests/ttnn/unit_tests/kernel_lib/kernels/mcast_unified.cpp",
        .compile_time_args = {{"rounds", rounds}},
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "acknowledges", "report_addr"}},
        .hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = noc}};
    kernel.scratchpad_bindings.push_back(
        {.scratchpad_spec_name = m2::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    m2::ProgramSpec spec{
        .kernels = {kernel},
        .scratchpads = {{.unique_id = m2::ScratchpadSpecName{"pad"}, .size_per_node = 2048}},
        .work_units = {{.name = "mixed", .kernels = {targets.front()}, .target_nodes = receivers}}};
    m2::ProgramRunArgs args;
    channel.attach(spec, args, "channel", targets);
    auto workload = m2::MakeMeshWorkloadFromSpec(device, spec);
    auto* physical = device.get_devices().front();
    for (uint32_t run = 0; run < 3; ++run) {
        auto invocation = args;
        const uint32_t seed = 73 + 101 * run;
        const uint32_t report_addr = 100 * 1024 + run * 4;
        std::vector<uint32_t> zero{0};
        for (const auto& core : corerange_to_cores(receivers)) {
            m2::AddRuntimeArgsForNode(
                invocation.kernel_run_args.front().runtime_arg_values,
                core,
                {{"seed", seed}, {"acknowledges", workers.contains(core) ? 1u : 0u}, {"report_addr", report_addr}});
            tt::tt_metal::detail::WriteToDeviceL1(physical, core, report_addr, zero);
        }
        for (auto& [region, program] : workload.get_programs()) {
            m2::SetProgramRunArgs(program, invocation);
        }
        EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
        for (const auto& core : corerange_to_cores(receivers)) {
            std::vector<uint32_t> address, results;
            tt::tt_metal::detail::ReadFromDeviceL1(physical, core, report_addr, 4, address);
            ASSERT_EQ(address.size(), 1u);
            ASSERT_NE(address.front(), 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(physical, core, address.front() + 1024, rounds * 4, results);
            ASSERT_EQ(results.size(), rounds);
            for (uint32_t round = 0; round < rounds; ++round) {
                EXPECT_EQ(results[round], 16 * (seed + 100 * round) + 120)
                    << "core=(" << core.x << "," << core.y << ") round=" << round;
            }
        }
    }
}

TEST_F(McastUnifiedFixture, DeviceSmoke) { run_mixed_ack_device(*device_, NOC::NOC_0, 2); }
TEST_F(McastUnifiedFixture, DeviceMatrix) {
    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        run_mixed_ack_device(*device_, noc, 8);
    }
}

}  // namespace ttnn::kernel_lib::host::unified_test
