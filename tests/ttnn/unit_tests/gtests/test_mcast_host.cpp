// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <set>
#include <vector>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"
#include "ttnn_test_fixtures.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"

namespace ttnn::kernel_lib::host::test {
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
namespace wire = dataflow_kernel_lib::mcast_wire;
using dataflow_kernel_lib::SenderMcastMode;
using dataflow_kernel_lib::TransferMode;
template <typename T>
concept ExposesInternalMcastAccessors =
    requires(T value) { value.compile_time_args(); } || requires(T value) { value.runtime_args(CoreCoord{}); } ||
    requires(T value) { value.owned_semaphores(); } || requires(T value) { value.num_semaphores(); } ||
    requires(T value) { value.next_base_sem_id(); } || requires(T value) { value.rotating(); } ||
    requires(T value) { value.num_senders(); } || requires(T value) { value.num_receivers(CoreCoord{}); } ||
    requires(T value) { value.has_remote_receivers(); } || requires(T value) { value.ack_count_override(); } ||
    requires(T value) { value.ack_count(); } || requires(T value) { value.ack_count(CoreCoord{}); } ||
    requires(T value) { value.num_rectangles(); } || requires(T value) { value.num_rectangles(CoreCoord{}); } ||
    requires(T value) { value.is_sender(CoreCoord{}); } || requires(T value) { value.receiver_cores(); } ||
    requires(T value) { value.rectangle_capacity(); } || requires(T value) { value.sender_in_rect(); };
static_assert(!ExposesInternalMcastAccessors<McastFamily>);
static_assert(!ExposesInternalMcastAccessors<Mcast1D>);
static_assert(!ExposesInternalMcastAccessors<Mcast2D>);

class McastHostFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastHostFixture> {};

CoreRangeSet grid(CoreCoord start, CoreCoord end) { return CoreRangeSet(CoreRange(start, end)); }
CoreRangeSet cores(const std::vector<CoreCoord>& values) {
    std::vector<CoreRange> ranges;
    ranges.reserve(values.size());
    for (auto core : values) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(std::move(ranges));
}
McastConfig chain_config(McastConfig cfg = {}) {
    cfg.irregular_receiver_set_mode = TransferMode::ChainUnicast;
    return cfg;
}

// Test-owned inputs for the independent destination oracle; no helper state or behavior.
struct GroupInput {
    CoreRangeSet receivers;
    std::vector<CoreCoord> senders;
    std::optional<uint32_t> ack;
    GroupInput(CoreRangeSet r, std::vector<CoreCoord> s, std::optional<uint32_t> a = std::nullopt) :
        receivers(std::move(r)), senders(std::move(s)), ack(a) {}
};

McastFamily make_family(
    tt::tt_metal::IDevice* device, const std::vector<GroupInput>& inputs, const McastConfig& cfg = {}) {
    McastFamily family(device, cfg);
    for (const auto& input : inputs) {
        family.add_group(input.receivers, input.senders, input.ack);
    }
    family.prepare_arguments();
    return family;
}

// Golden wire tests use the regular Program path. Adoption fixtures explicitly seed
// the requested existing resources; every other family resolves its own allocation.
template <typename Family>
void bind_for_inspection(Family& family, tt::tt_metal::Program& program, const std::vector<uint32_t>& existing) {
    if constexpr (requires { family.participating_cores(); }) {
        for (auto id : existing) {
            program.impl().add_semaphore(family.participating_cores(), id, 0, tt::CoreType::WORKER);
        }
    } else {
        TT_FATAL(existing.empty(), "Wrapper adoption fixtures must provide explicit placement");
    }
    family.append_semaphores(program);
}

template <typename Family>
std::vector<uint32_t> compile_args(Family family, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(family, program, existing);
    std::vector<uint32_t> args;
    family.append_compile_time_args_to(args);
    return args;
}

template <typename Family>
std::vector<uint32_t> runtime_args(Family family, CoreCoord core, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(family, program, existing);
    std::vector<uint32_t> args;
    family.append_runtime_args_to(args, core);
    return args;
}

template <typename Family>
std::vector<tt::tt_metal::SemaphoreDescriptor> allocated_semaphores(
    Family family, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(family, program, existing);
    std::vector<tt::tt_metal::SemaphoreDescriptor> result;
    const auto& semaphores = program.impl().semaphores();
    for (const auto& sem : semaphores) {
        if (std::find(existing.begin(), existing.end(), sem.id()) != existing.end()) {
            continue;
        }
        result.push_back({.id = sem.id(), .core_ranges = sem.core_range_set(), .initial_value = sem.initial_value()});
    }
    return result;
}

using Coordinates = std::set<std::pair<uint32_t, uint32_t>>;

Coordinates worker_coordinates(tt::tt_metal::IDevice* device) {
    Coordinates result;
    const auto size = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < size.y; ++y) {
        for (uint32_t x = 0; x < size.x; ++x) {
            const auto w = device->worker_core_from_logical_core({x, y});
            result.emplace(w.x, w.y);
        }
    }
    return result;
}

// Independent oracle: enumerate the emitted destinations and compare with each mapped logical
// receiver, without using the helper's decomposition or a bounding-box reference.
void check_group(
    tt::tt_metal::IDevice* device,
    const McastFamily& family,
    const GroupInput& group,
    const std::vector<uint32_t>& existing = {}) {
    const auto workers = worker_coordinates(device);
    const auto ct = compile_args(family, existing);
    ASSERT_EQ(ct.size(), 11u);
    ASSERT_EQ(ct[0], 1u);
    const uint32_t count = group.senders.size();
    EXPECT_EQ(ct[6], (count > 1) ? count : 0u);
    Coordinates expected;
    for (auto c : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        auto w = device->worker_core_from_logical_core(c);
        expected.emplace(w.x, w.y);
    }
    for (uint32_t phase = 0; phase < count; ++phase) {
        const auto sender = group.senders[phase];
        const auto worker = device->worker_core_from_logical_core(sender);
        const auto rt = runtime_args(family, sender, existing);
        ASSERT_EQ(rt.size(), 2u + 2u * count + 7u * ct[10] + 2u);
        EXPECT_EQ(rt[rt.size() - 1], phase);
        EXPECT_EQ(rt[rt.size() - 2], 1u | (count > 1 && group.receivers.contains(sender) ? 2u : 0u));
        Coordinates actual;
        uint32_t remote_total = 0;
        for (uint32_t i = 0; i < rt[0]; ++i) {
            const auto* r = rt.data() + 2 + 2 * count + 7 * i;
            const auto xlo = std::min(r[0], r[2]), xhi = std::max(r[0], r[2]);
            const auto ylo = std::min(r[1], r[3]), yhi = std::max(r[1], r[3]);
            EXPECT_EQ(r[0], (ct[5] & 4) ? xhi : xlo);
            EXPECT_EQ(r[1], (ct[5] & 4) ? yhi : ylo);
            uint32_t area = 0;
            bool inside = false;
            for (uint32_t y = ylo; y <= yhi; ++y) {
                for (uint32_t x = xlo; x <= xhi; ++x) {
                    if (!workers.contains({x, y})) {
                        continue;
                    }
                    EXPECT_TRUE(actual.emplace(x, y).second) << "overlapping rectangles";
                    inside |= x == worker.x && y == worker.y;
                    ++area;
                }
            }
            const uint32_t remote = area - inside;
            EXPECT_EQ(r[4], remote);
            EXPECT_EQ(r[5], remote + 1u);
            EXPECT_EQ(r[6], remote == 0 ? 1u : inside ? 3u : 2u);
            if (ct[7] != 4) {
                EXPECT_EQ(r[6], ct[7]);
            }
            remote_total += remote;
        }
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(remote_total, expected.size() - expected.count({worker.x, worker.y}));
        for (uint32_t i = 0; i < count; ++i) {
            const auto w = device->worker_core_from_logical_core(group.senders[i]);
            EXPECT_EQ(rt[2 + 2 * i], w.x);
            EXPECT_EQ(rt[3 + 2 * i], w.y);
        }
        EXPECT_EQ(rt, runtime_args(family, sender, existing));
    }
    for (auto core : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        if (std::find(group.senders.begin(), group.senders.end(), core) != group.senders.end()) {
            continue;
        }
        const auto rt = runtime_args(family, core, existing);
        EXPECT_EQ(rt[rt.size() - 2], 2u);
        EXPECT_EQ(rt.back(), wire::NO_SENDER_ROUND);
        EXPECT_EQ(runtime_args(family, core, existing)[wire::ACK], 0u);
    }
}

template <typename Wrapper>
void check_wrapper(const Wrapper& wrapper, const McastFamily& family) {
    EXPECT_EQ(compile_args(wrapper), compile_args(family));
    EXPECT_EQ(allocated_semaphores(wrapper).size(), allocated_semaphores(family).size());
    std::vector<CoreCoord> participants =
        tt::tt_metal::corerange_to_cores(family.participating_cores(), std::nullopt, true);
    participants.emplace_back(0, 0);  // Outside the offset grids below.
    for (auto core : participants) {
        EXPECT_EQ(runtime_args(wrapper, core), runtime_args(family, core));
        std::vector<uint32_t> appended{42};
        detail::append_args_to(appended, runtime_args(wrapper, core));
        auto expected = runtime_args(family, core);
        expected.insert(expected.begin(), 42);
        EXPECT_EQ(appended, expected);
    }
    struct Appendable {
        std::vector<uint32_t> words;
        void append(const std::vector<uint32_t>& a) { words.insert(words.end(), a.begin(), a.end()); }
    } ct;
    detail::append_args_to(ct, compile_args(wrapper));
    append_absent_mcast_compile_time_args_to(ct);
    detail::append_args_to(ct, compile_args(family));
    auto expected = compile_args(family);
    expected.push_back(0);
    const auto tail = compile_args(family);
    expected.insert(expected.end(), tail.begin(), tail.end());
    EXPECT_EQ(ct.words, expected);
    const auto sems = allocated_semaphores(wrapper);
    ASSERT_EQ(sems.size(), allocated_semaphores(family).size());
    for (size_t i = 0; i < sems.size(); ++i) {
        EXPECT_EQ(sems[i].id, allocated_semaphores(family)[i].id);
        EXPECT_EQ(sems[i].core_ranges, family.participating_cores());
        EXPECT_EQ(sems[i].initial_value, 0u);
    }
}

TEST(GroupNormMcastGeometry, ZeroEdgeRectangle) {
    std::vector<CoreCoord> group = {CoreCoord(0, 0), CoreCoord(1, 0), CoreCoord(2, 0)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_TRUE(first.empty());
    EXPECT_EQ(middle, group);
    EXPECT_TRUE(last.empty());
}

TEST(GroupNormMcastGeometry, OneEdgeWrappedSequence) {
    std::vector<CoreCoord> group = {CoreCoord(7, 0), CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_EQ(first, std::vector<CoreCoord>({CoreCoord(7, 0)}));
    EXPECT_EQ(middle, std::vector<CoreCoord>({CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)}));
    EXPECT_TRUE(last.empty());
}

TEST(GroupNormMcastGeometry, TwoEdgeWrappedSequence) {
    std::vector<CoreCoord> group = {
        CoreCoord(7, 0), CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1), CoreCoord(0, 2)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_EQ(first, std::vector<CoreCoord>({CoreCoord(7, 0)}));
    EXPECT_EQ(middle, std::vector<CoreCoord>({CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)}));
    EXPECT_EQ(last, std::vector<CoreCoord>({CoreCoord(0, 2)}));
}

TEST(McastHostWire, AbsentFamilyIsOneWord) {
    std::vector<uint32_t> args{17};
    append_absent_mcast_compile_time_args_to(args);
    EXPECT_EQ(args, (std::vector<uint32_t>{17, 0}));
}

TEST_F(McastHostFixture, NamedOffsetsPreserveSerializedWireValues) {
    // Literal positions are intentional: this oracle must catch coordinated encoder/decoder
    // changes that accidentally alter the established wire format.
    EXPECT_EQ(uint32_t(TransferMode::Multicast), 0u);
    EXPECT_EQ(uint32_t(TransferMode::ChainUnicast), 1u);
    EXPECT_EQ(uint32_t(SenderMcastMode::Invalid), 0u);
    EXPECT_EQ(uint32_t(SenderMcastMode::LocalCopy), 1u);
    EXPECT_EQ(uint32_t(SenderMcastMode::MulticastExcludeSource), 2u);
    EXPECT_EQ(uint32_t(SenderMcastMode::MulticastIncludeSource), 3u);
    EXPECT_EQ(uint32_t(SenderMcastMode::Unknown), 4u);
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    cfg.data_ready = dataflow_kernel_lib::DataReadySignal::Counter;
    cfg.sem_ids = std::vector<uint32_t>{11, 13};
    cfg.ack_count_override = 5;
    McastFamily family(device_, cfg);
    const std::vector<CoreCoord> senders{{6, 1}, {1, 6}};
    family.add_group(grid({2, 3}, {4, 5}), senders);
    family.prepare_arguments();
    const std::vector<uint32_t> expected_ct{1, 1, 11, 13, 5, 7, 2, 2, 9, 10, 1};
    EXPECT_EQ(compile_args(family, {11, 13}), expected_ct);
    auto face_ct = expected_ct;
    face_ct[5] = 6;
    cfg.handshake = false;
    cfg.sem_ids = std::vector<uint32_t>{11};
    auto passive = make_family(device_, {GroupInput(grid({2, 3}, {4, 5}), senders)}, cfg);
    face_ct[3] = UNUSED_SEM_ID;
    EXPECT_EQ(compile_args(passive, {11}), face_ct);
    auto mapped = [&](CoreCoord core) { return device_->worker_core_from_logical_core(core); };
    const auto a = mapped(senders[0]), b = mapped(senders[1]);
    const auto lo = mapped({2, 3}), hi = mapped({4, 5});
    for (uint32_t phase = 0; phase < senders.size(); ++phase) {
        const std::vector<uint32_t> expected_rt{
            1,
            5,
            uint32_t(a.x),
            uint32_t(a.y),
            uint32_t(b.x),
            uint32_t(b.y),
            uint32_t(hi.x),
            uint32_t(hi.y),
            uint32_t(lo.x),
            uint32_t(lo.y),
            9,
            10,
            2,
            1,
            phase};
        EXPECT_EQ(runtime_args(family, senders[phase], {11, 13}), expected_rt);
    }
    const std::vector<uint32_t> receiver_rt{
        0, 0, uint32_t(a.x), uint32_t(a.y), uint32_t(b.x), uint32_t(b.y), 0, 0, 0, 0, 0, 0, 0, 2, 0xFFFFFFFFu};
    EXPECT_EQ(runtime_args(family, {3, 4}, {11, 13}), receiver_rt);
    std::vector<uint32_t> inactive(15, 0);
    inactive[14] = 0xFFFFFFFFu;
    EXPECT_EQ(runtime_args(family, {0, 0}, {11, 13}), inactive);
    const auto sender_only = family.sender_only_cores();
    EXPECT_EQ(sender_only.num_cores(), senders.size());
    for (const auto& sender : senders) {
        EXPECT_TRUE(sender_only.contains(sender));
    }
}

TEST_F(McastHostFixture, WrapperRowsColumnsFixedAndRotating) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (auto shape : {Mcast1DShape::PerRow, Mcast1DShape::PerColumn}) {
            for (auto end : {CoreCoord(4, 3), CoreCoord(2, 2)}) {
                const auto receivers = grid({2, 2}, end);
                const bool row = shape == Mcast1DShape::PerRow;
                const uint32_t lines = (row ? end.y : end.x) - 1;
                const uint32_t span = (row ? end.x : end.y) - 1;
                McastConfig cfg;
                cfg.noc = noc;
                cfg.base_sem_id = 0;
                for (auto placement : {Mcast1DSenderPlacement::Uniform, Mcast1DSenderPlacement::Diagonal}) {
                    std::vector<GroupInput> groups;
                    for (uint32_t i = 0; i < lines; ++i) {
                        const uint32_t phase =
                            placement == Mcast1DSenderPlacement::Diagonal ? (span - 1 + i) % span : span - 1;
                        const CoreCoord lo(row ? 2 : 2 + i, row ? 2 + i : 2),
                            hi(row ? end.x : 2 + i, row ? 2 + i : end.y);
                        const CoreCoord sender(row ? 2 + phase : 2 + i, row ? 2 + i : 2 + phase);
                        groups.emplace_back(grid(lo, hi), std::vector<CoreCoord>{sender});
                    }
                    auto family = make_family(device_, groups, cfg);
                    Mcast1D wrapper(device_, receivers, shape, Mcast1DFixedSenderConfig{span - 1, placement}, cfg);
                    check_wrapper(wrapper, family);
                    EXPECT_EQ(wrapper.participating_cores(), receivers);
                    EXPECT_EQ(wrapper.participating_cores(), family.participating_cores());
                    EXPECT_EQ(wrapper.sender_only_cores(), family.sender_only_cores());
                    const auto owned = allocated_semaphores(wrapper);
                    ASSERT_FALSE(owned.empty());
                    EXPECT_EQ(owned.back().id + 1, 2u);
                    for (auto& group : groups) {
                        check_group(device_, family, group);
                    }
                }
                // Default rotation and explicit aligned senders extending outside the receiver set.
                for (bool outside : {false, true}) {
                    std::vector<CoreCoord> all_senders;
                    std::vector<GroupInput> groups;
                    for (uint32_t i = 0; i < lines; ++i) {
                        std::vector<CoreCoord> senders;
                        senders.reserve(span + uint32_t(outside));
                        for (uint32_t j = 0; j < span + uint32_t(outside); ++j) {
                            senders.emplace_back(row ? 2 + j : 2 + i, row ? 2 + i : 2 + j);
                        }
                        all_senders.insert(all_senders.end(), senders.begin(), senders.end());
                        groups.emplace_back(
                            grid({row ? 2 : 2 + i, row ? 2 + i : 2}, {row ? end.x : 2 + i, row ? 2 + i : end.y}),
                            senders);
                    }
                    auto family = make_family(device_, groups, cfg);
                    Mcast1D wrapper(
                        device_,
                        receivers,
                        shape,
                        Mcast1DRotatingSenderConfig{outside ? std::optional(cores(all_senders)) : std::nullopt},
                        cfg);
                    check_wrapper(wrapper, family);
                    for (auto& group : groups) {
                        check_group(device_, family, group);
                    }
                }
            }
        }
    }
}

TEST_F(McastHostFixture, SparseSenderLinesPreserveOrderAndAlignment) {
    for (auto shape : {Mcast1DShape::PerRow, Mcast1DShape::PerColumn}) {
        const bool per_row = shape == Mcast1DShape::PerRow;
        const auto coord = [per_row](uint32_t along, uint32_t line) {
            return per_row ? CoreCoord{along, line} : CoreCoord{line, along};
        };
        const auto receivers = grid(coord(2, 2), coord(4, 3));
        // Deliberately unordered, sparse, with one external sender on each line.
        const auto senders = cores({coord(4, 3), coord(0, 2), coord(2, 3), coord(4, 2), coord(0, 3), coord(2, 2)});
        for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
            McastConfig cfg;
            cfg.noc = noc;
            std::vector<GroupInput> groups;
            for (uint32_t line : {2u, 3u}) {
                groups.emplace_back(
                    grid(coord(2, line), coord(4, line)),
                    std::vector<CoreCoord>{coord(0, line), coord(2, line), coord(4, line)});
            }
            Mcast1D wrapper(device_, receivers, shape, Mcast1DRotatingSenderConfig{senders}, cfg);
            auto family = make_family(device_, groups, cfg);
            check_wrapper(wrapper, family);
            for (const auto& group : groups) {
                check_group(device_, family, group);
            }
        }
        EXPECT_ANY_THROW(Mcast1D(
            device_, receivers, shape, Mcast1DRotatingSenderConfig{cores({coord(2, 2), coord(4, 2), coord(2, 3)})}));
        EXPECT_ANY_THROW(Mcast1D(device_, receivers, shape, Mcast1DRotatingSenderConfig{cores({coord(2, 2)})}));
        EXPECT_ANY_THROW(Mcast1D(
            device_, receivers, shape, Mcast1DRotatingSenderConfig{cores({coord(2, 2), coord(2, 3), coord(2, 4)})}));
        EXPECT_ANY_THROW(Mcast1D(device_, receivers, shape, Mcast1DFixedSenderConfig{3}));
    }
}

TEST_F(McastHostFixture, Wrapper2DFixedAndRotatingOrder) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        const auto receivers = grid({2, 2}, {3, 3});
        for (auto sender : {CoreCoord(2, 2), CoreCoord(3, 3), CoreCoord(4, 2)}) {
            GroupInput group(receivers, std::vector<CoreCoord>{sender});
            auto family = make_family(device_, {group}, cfg);
            Mcast2D wrapper(device_, receivers, Mcast2DFixedSenderConfig{sender}, cfg);
            check_wrapper(wrapper, family);
            check_group(device_, family, group);
        }
        for (auto order : {Mcast2DSenderOrder::RowMajor, Mcast2DSenderOrder::ColumnMajor}) {
            for (bool outside : {false, true}) {
                std::vector<CoreCoord> senders;
                for (uint32_t a = 2; a <= 3; ++a) {
                    for (uint32_t b = 2; b <= (outside ? 4u : 3u); ++b) {
                        senders.emplace_back(
                            order == Mcast2DSenderOrder::RowMajor ? b : a,
                            order == Mcast2DSenderOrder::RowMajor ? a : b);
                    }
                }
                GroupInput group(receivers, senders);
                auto family = make_family(device_, {group}, cfg);
                Mcast2D wrapper(
                    device_,
                    receivers,
                    Mcast2DRotatingSenderConfig{outside ? std::optional(cores(senders)) : std::nullopt, order},
                    cfg);
                check_wrapper(wrapper, family);
                check_group(device_, family, group);
            }
        }
    }
}

TEST_F(McastHostFixture, ExactStaircaseAndDifferentGroupSizes) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        GroupInput staircase(cores({{7, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}}), std::vector<CoreCoord>{{7, 0}});
        GroupInput rectangle(grid({3, 3}, {5, 3}), std::vector<CoreCoord>{{3, 3}});
        auto family = make_family(device_, {staircase, rectangle}, cfg);
        EXPECT_EQ(compile_args(family)[wire::RECTANGLE_CAPACITY], 3u);
        EXPECT_EQ(runtime_args(family, {3, 3})[wire::NUM_RECTANGLES], 1u);
        EXPECT_EQ(runtime_args(family, {7, 0})[wire::ACK], 4u);
        EXPECT_EQ(runtime_args(family, {3, 3})[wire::ACK], 2u);
        EXPECT_EQ(compile_args(family)[4], ACK_EQUALS_FANOUT);
        check_group(device_, family, staircase);
        check_group(device_, family, rectangle);
    }
}

TEST_F(McastHostFixture, FullWidthMappedCoverage) {
    const auto size = device_->compute_with_storage_grid_size();
    const auto receivers = grid({0, 0}, {size.x - 1, 1});
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        GroupInput group(receivers, std::vector<CoreCoord>{{0, 0}, {size.x - 1, 1}});
        auto family = make_family(device_, {group}, cfg);
        check_group(device_, family, group);
        EXPECT_EQ(compile_args(family)[wire::RECTANGLE_CAPACITY], 1u);
    }
}

TEST_F(McastHostFixture, FullWidthGroupsUseLogicalRectangles) {
    const auto size = device_->compute_with_storage_grid_size();
    if (size.x < 11 || size.y < 9) {
        GTEST_SKIP() << "Requires an 11-column worker grid";
    }
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        std::vector<GroupInput> groups;
        for (uint32_t group = 0; group < 4; ++group) {
            std::vector<CoreCoord> members;
            for (uint32_t slot = group * 24; slot < (group + 1) * 24; ++slot) {
                members.emplace_back(slot % 11, slot / 11);
            }
            groups.emplace_back(cores(members), std::vector<CoreCoord>{members.front()});
        }
        auto multicast = make_family(device_, groups, cfg);
        auto chain = make_family(device_, groups, chain_config(cfg));
        EXPECT_EQ(compile_args(multicast)[wire::RECTANGLE_CAPACITY], 3u);
        EXPECT_EQ(compile_args(chain)[wire::RECTANGLE_CAPACITY], 0u);
        for (const auto& group : groups) {
            check_group(device_, multicast, group);
            EXPECT_EQ(runtime_args(chain, group.senders.front())[wire::ACK], 1u);
        }
    }
}

TEST_F(McastHostFixture, LocalAndNonparticipantRoles) {
    GroupInput local(grid({3, 3}, {3, 3}), std::vector<CoreCoord>{{3, 3}});
    auto family = make_family(device_, {local});
    check_group(device_, family, local);
    EXPECT_EQ(runtime_args(family, {3, 3})[0], 1u);
    EXPECT_EQ(compile_args(family)[wire::HAS_RECEIVERS], 0u);
    auto outside = runtime_args(family, {0, 0});
    EXPECT_EQ(outside.back(), wire::NO_SENDER_ROUND);
    outside.pop_back();
    EXPECT_TRUE(std::all_of(outside.begin(), outside.end(), [](auto v) { return v == 0; }));
    Mcast2D wrapper(device_, grid({3, 3}, {3, 3}), Mcast2DFixedSenderConfig{{3, 3}});
    check_wrapper(wrapper, make_family(device_, {local}));
}

TEST_F(McastHostFixture, IrregularReceiverSetPolicySelectsFamilyTransport) {
    const GroupInput dense(grid({0, 0}, {2, 0}), {{1, 0}});
    const GroupInput irregular(cores({{0, 2}, {2, 2}, {4, 2}}), {{2, 2}});
    const GroupInput local(cores({{6, 0}}), {{6, 0}});
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig config;
        config.noc = noc;
        auto hardware = make_family(device_, {dense}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(compile_args(hardware)[wire::FLAGS]), TransferMode::Multicast);
        check_group(device_, hardware, dense);
        auto chain = make_family(device_, {irregular}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(compile_args(chain)[wire::FLAGS]), TransferMode::ChainUnicast);
        auto rectangles = make_family(device_, {dense, local}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(compile_args(rectangles)[wire::FLAGS]), TransferMode::Multicast);
        for (const auto& groups :
             {std::vector<GroupInput>{dense, irregular, local}, std::vector<GroupInput>{irregular, local, dense}}) {
            auto family = make_family(device_, groups, chain_config(config));
            auto multiple_mcast = make_family(device_, groups, config);
            EXPECT_EQ(wire::transfer_mode(compile_args(family)[wire::FLAGS]), TransferMode::ChainUnicast);
            EXPECT_EQ(wire::transfer_mode(compile_args(multiple_mcast)[wire::FLAGS]), TransferMode::Multicast);
            EXPECT_EQ(compile_args(family)[wire::RECTANGLE_CAPACITY], 0u);
            EXPECT_EQ(compile_args(multiple_mcast)[wire::RECTANGLE_CAPACITY], 3u);
            EXPECT_EQ(runtime_args(family, {1, 0})[wire::ACK], 1u);  // Dense group also uses per-hop readiness.
            EXPECT_EQ(runtime_args(family, {2, 2})[wire::ACK], 1u);
            EXPECT_EQ(runtime_args(family, {6, 0})[wire::ACK], 0u);
            for (auto core : tt::tt_metal::corerange_to_cores(family.participating_cores())) {
                EXPECT_EQ(runtime_args(family, core).size(), wire::runtime_words(0, 0, TransferMode::ChainUnicast));
            }
            EXPECT_EQ(runtime_args(multiple_mcast, {7, 7}).size(), wire::runtime_words(0, 3, TransferMode::Multicast));
        }
    }
}

TEST_F(McastHostFixture, FlagsSemaphoresAndAckPrecedence) {
    const auto receivers = grid({2, 2}, {4, 2});
    for (auto signal : {dataflow_kernel_lib::DataReadySignal::Flag, dataflow_kernel_lib::DataReadySignal::Counter}) {
        for (bool handshake : {false, true}) {
            for (auto config_ack :
                 {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{1}}) {
                for (auto group_ack :
                     {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2}}) {
                    McastConfig cfg;
                    cfg.data_ready = signal;
                    cfg.handshake = handshake;
                    cfg.ack_count_override = config_ack;
                    cfg.base_sem_id = 0;
                    GroupInput group(receivers, std::vector<CoreCoord>{{2, 2}}, group_ack);
                    auto family = make_family(device_, {group}, cfg);
                    EXPECT_EQ(runtime_args(family, {2, 2})[wire::ACK], group_ack.value_or(config_ack.value_or(2)));
                    EXPECT_EQ(
                        compile_args(family)[5],
                        uint32_t(handshake) + (signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u));
                    auto passive_cfg = cfg;
                    passive_cfg.handshake = false;
                    EXPECT_EQ(
                        compile_args(make_family(device_, {group}, passive_cfg))[5],
                        signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u);
                    EXPECT_EQ(allocated_semaphores(family).size(), handshake ? 2u : 1u);
                    const auto owned = allocated_semaphores(family);
                    ASSERT_FALSE(owned.empty());
                    EXPECT_EQ(owned.back().id + 1, handshake ? 2u : 1u);
                    cfg.sem_ids = handshake ? std::vector<uint32_t>{6, 7} : std::vector<uint32_t>{6};
                    auto adopted = make_family(device_, {group}, cfg);
                    EXPECT_TRUE(allocated_semaphores(adopted, *cfg.sem_ids).empty());
                    EXPECT_EQ(compile_args(adopted, *cfg.sem_ids)[2], 6u);
                    EXPECT_EQ(compile_args(adopted, *cfg.sem_ids)[3], handshake ? 7u : UNUSED_SEM_ID);
                }
            }
        }
    }
}

TEST_F(McastHostFixture, SenderListsAndSingletonWrapperSchedules) {
    const std::vector<CoreCoord> ordered = {{3, 2}, {2, 2}};
    GroupInput rotating(grid({2, 2}, {3, 2}), ordered);
    check_group(device_, make_family(device_, {rotating}), rotating);

    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (auto signal :
             {dataflow_kernel_lib::DataReadySignal::Flag, dataflow_kernel_lib::DataReadySignal::Counter}) {
            McastConfig cfg;
            cfg.noc = noc;
            cfg.data_ready = signal;
            for (auto shape : {Mcast1DShape::PerRow, Mcast1DShape::PerColumn}) {
                const bool row = shape == Mcast1DShape::PerRow;
                const auto receivers = grid({2, 2}, row ? CoreCoord{3, 2} : CoreCoord{2, 3});
                for (auto sender : {CoreCoord{2, 2}, row ? CoreCoord{4, 2} : CoreCoord{2, 4}}) {
                    GroupInput group(receivers, {sender});
                    auto family = make_family(device_, {group}, cfg);
                    EXPECT_EQ(compile_args(family)[6], 0u);
                    check_group(device_, family, group);
                    const auto sender_grid = grid(sender, sender);
                    check_wrapper(
                        Mcast1D(device_, receivers, shape, Mcast1DRotatingSenderConfig{sender_grid}, cfg), family);
                    check_wrapper(Mcast2D(device_, receivers, Mcast2DRotatingSenderConfig{sender_grid}, cfg), family);
                    check_wrapper(Mcast2D(device_, receivers, Mcast2DFixedSenderConfig{sender}, cfg), family);
                }
            }
        }
    }
}

TEST_F(McastHostFixture, InvalidGroupsAndWrappers) {
    const auto receivers = grid({2, 2}, {3, 2});
    GroupInput fixed(receivers, std::vector<CoreCoord>{{2, 2}});
    GroupInput rotating(grid({2, 3}, {3, 3}), std::vector<CoreCoord>{{2, 3}, {3, 3}});
    GroupInput longer(grid({2, 4}, {3, 4}), std::vector<CoreCoord>{{2, 4}, {3, 4}, {4, 4}});
    // Four isolated mapped destinations cannot be represented within the three-rectangle limit.
    GroupInput fragmented(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), std::vector<CoreCoord>{{0, 0}});
    EXPECT_THROW(make_family(device_, {fragmented}), std::exception);
    EXPECT_ANY_THROW(McastFamily(device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}}));
    EXPECT_ANY_THROW(McastFamily(device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}, {3, 2}}));
    EXPECT_ANY_THROW(make_family(device_, {}));
    EXPECT_ANY_THROW(make_family(nullptr, {fixed}));
    EXPECT_ANY_THROW(McastFamily(device_).add_group(receivers, std::vector<CoreCoord>{}));
    EXPECT_ANY_THROW(McastFamily(device_).add_group(receivers, std::vector<CoreCoord>{{2, 2}, {2, 2}}));
    EXPECT_ANY_THROW(make_family(device_, {fixed, fixed}));
    EXPECT_ANY_THROW(make_family(device_, {fixed, rotating}));
    EXPECT_ANY_THROW(make_family(device_, {rotating, longer}));
    // Disjoint receivers are insufficient when groups share a sender.
    EXPECT_ANY_THROW(make_family(device_, {fixed, GroupInput(grid({4, 4}, {4, 4}), std::vector<CoreCoord>{{2, 2}})}));
    McastConfig cfg;
    cfg.ack_count_override = 2;
    EXPECT_ANY_THROW(make_family(device_, {fixed}, cfg));
    EXPECT_ANY_THROW(Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg));
    EXPECT_ANY_THROW(Mcast2D(device_, receivers, Mcast2DFixedSenderConfig{{2, 2}}, cfg));
    for (const auto& ids :
         {std::vector<uint32_t>{}, std::vector<uint32_t>{0}, std::vector<uint32_t>{0, UNUSED_SEM_ID}}) {
        cfg = {};
        cfg.sem_ids = ids;
        EXPECT_ANY_THROW(make_family(device_, {fixed}, cfg));
    }
    cfg = {};
    cfg.handshake = false;
    EXPECT_ANY_THROW(Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{2}));
    EXPECT_ANY_THROW(
        Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{grid({2, 3}, {3, 3})}));
    EXPECT_ANY_THROW(Mcast2D(device_, receivers, Mcast2DRotatingSenderConfig{CoreRangeSet{}}));
}

TEST_F(McastHostFixture, FamilySerializationAndRouting) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool rotating : {false, true}) {
            McastConfig cfg;
            cfg.noc = noc;
            cfg.data_ready = dataflow_kernel_lib::DataReadySignal::Counter;
            cfg.sem_ids = std::vector<uint32_t>{6, 7};
            cfg.ack_count_override = 0;
            const std::vector<CoreCoord> first = {{2, 2}, {0, 0}, {0, 4}};
            const std::vector<CoreCoord> outside = {{4, 2}, {4, 0}, {6, 4}};
            const std::vector<CoreRangeSet> receivers = {
                grid({2, 2}, {3, 2}), cores({{0, 0}, {2, 0}}), cores({{0, 4}, {2, 4}, {4, 4}})};
            std::vector<GroupInput> groups;
            for (size_t i = 0; i < receivers.size(); ++i) {
                auto senders =
                    rotating ? std::vector<CoreCoord>{outside[i], first[i]} : std::vector<CoreCoord>{first[i]};
                groups.emplace_back(receivers[i], senders, i == 2 ? std::optional<uint32_t>{1} : std::nullopt);
            }
            auto family = make_family(device_, groups, cfg);
            ASSERT_EQ(compile_args(family, {6, 7})[wire::RECTANGLE_CAPACITY], 3u);
            EXPECT_TRUE(allocated_semaphores(family, {6, 7}).empty());
            for (size_t i = 0; i < groups.size(); ++i) {
                check_group(device_, family, groups[i], {6, 7});
                EXPECT_EQ(runtime_args(family, first[i], {6, 7})[wire::NUM_RECTANGLES], i + 1u);
                EXPECT_EQ(runtime_args(family, first[i], {6, 7})[wire::ACK], i == 2 ? 1u : 0u);
                std::vector<uint32_t> appended{99};
                detail::append_args_to(appended, runtime_args(family, first[i], {6, 7}));
                auto expected = runtime_args(family, first[i], {6, 7});
                expected.insert(expected.begin(), 99);
                EXPECT_EQ(appended, expected);
            }
            EXPECT_EQ(family.sender_only_cores(), rotating ? cores(outside) : CoreRangeSet{});
            std::vector<uint32_t> appended{42};
            detail::append_args_to(appended, compile_args(family, {6, 7}));
            auto expected = compile_args(family, {6, 7});
            expected.insert(expected.begin(), 42);
            EXPECT_EQ(appended, expected);
            auto inactive = runtime_args(family, {7, 7}, {6, 7});
            EXPECT_EQ(inactive.size(), runtime_args(family, first[0], {6, 7}).size());
            EXPECT_EQ(inactive.back(), wire::NO_SENDER_ROUND);
            inactive.pop_back();
            EXPECT_TRUE(std::all_of(inactive.begin(), inactive.end(), [](auto word) { return word == 0; }));
        }
    }
}

void check_building_queries(const McastFamily& family) {
    auto unprepared = family;
    tt::tt_metal::Program program;
    EXPECT_ANY_THROW(unprepared.append_semaphores(program));
    EXPECT_TRUE(program.impl().semaphores().empty());
    EXPECT_ANY_THROW(family.participating_cores());
    EXPECT_ANY_THROW(family.sender_only_cores());
    std::vector<uint32_t> unchanged{42};
    EXPECT_ANY_THROW(family.append_compile_time_args_to(unchanged));
    EXPECT_ANY_THROW(family.append_runtime_args_to(unchanged, {0, 0}));
    EXPECT_EQ(unchanged, std::vector<uint32_t>{42});
}

TEST_F(McastHostFixture, CollectionAndArgumentPreparationLifecycle) {
    McastFamily family(device_);
    check_building_queries(family);
    EXPECT_ANY_THROW(family.prepare_arguments());
    family.add_group(grid({2, 2}, {3, 2}), {{2, 2}});
    check_building_queries(family);
    EXPECT_ANY_THROW(family.add_group(CoreRangeSet{}, {{0, 0}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {2, 3}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {3, 3}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 2}}));
    family.add_group(grid({4, 2}, {4, 2}), {{4, 2}});
    family.prepare_arguments();
    EXPECT_EQ(family.participating_cores(), grid({2, 2}, {4, 2}));
    EXPECT_EQ(allocated_semaphores(family).size(), 2u);
    const auto ct = compile_args(family);
    const auto rt = runtime_args(family, {2, 2});
    const auto* participants = &family.participating_cores();
    family.prepare_arguments();
    EXPECT_EQ(compile_args(family), ct);
    EXPECT_EQ(runtime_args(family, {2, 2}), rt);
    EXPECT_EQ(&family.participating_cores(), participants);
    EXPECT_ANY_THROW(family.add_group(grid({6, 2}, {6, 2}), {{6, 2}}));
    EXPECT_EQ(family.participating_cores().num_cores(), 3u);
}

TEST_F(McastHostFixture, FailedArgumentPreparationAndLateTransportSelection) {
    // Fail after groups have been mapped, then retry and verify that queries stay blocked.
    McastConfig invalid;
    invalid.sem_ids = std::vector<uint32_t>{0};
    McastFamily failed(device_, invalid);
    failed.add_group(grid({0, 0}, {2, 0}), {{0, 0}});
    for (int attempt = 0; attempt < 2; ++attempt) {
        EXPECT_ANY_THROW(failed.prepare_arguments());
        check_building_queries(failed);
    }
    // A failed prepare_arguments must retain input overlap validation, even with partially prepared groups.
    EXPECT_ANY_THROW(failed.add_group(grid({0, 0}, {1, 0}), {{0, 0}}));
    failed.add_group(grid({4, 0}, {5, 0}), {{4, 0}});
    EXPECT_ANY_THROW(failed.prepare_arguments());
    check_building_queries(failed);

    // A late irregular group changes the common transport of the earlier dense group.
    McastFamily late(device_, chain_config());
    late.add_group(grid({0, 0}, {2, 0}), {{1, 0}});
    check_building_queries(late);
    late.add_group(cores({{0, 2}, {2, 2}}), {{0, 2}});
    late.prepare_arguments();
    EXPECT_EQ(wire::transfer_mode(compile_args(late)[wire::FLAGS]), TransferMode::ChainUnicast);
    EXPECT_EQ(compile_args(late)[wire::RECTANGLE_CAPACITY], 0u);
    EXPECT_EQ(runtime_args(late, {1, 0})[wire::ACK], 1u);
    EXPECT_EQ(runtime_args(late, {1, 0}).size(), 11u);
    EXPECT_EQ(runtime_args(late, {2, 2}).size(), 11u);
}

TEST_F(McastHostFixture, FamilyValueSemanticsAndConfigSnapshot) {
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    cfg.sem_ids = std::vector<uint32_t>{6, 7};
    McastFamily building(device_, cfg);
    cfg.noc = NOC::NOC_0;
    (*cfg.sem_ids)[0] = 1;
    building.add_group(grid({2, 2}, {2, 2}), {{2, 2}});
    auto copy = building;
    copy.add_group(grid({4, 2}, {5, 2}), {{4, 2}});
    copy.prepare_arguments();
    building.prepare_arguments();
    EXPECT_EQ(building.participating_cores().num_cores(), 1u);
    EXPECT_EQ(copy.participating_cores().num_cores(), 3u);
    EXPECT_EQ(compile_args(copy, {6, 7})[wire::DATA_READY], 6u);
    EXPECT_NE(compile_args(copy, {6, 7})[wire::FLAGS] & wire::NOC1, 0u);
    EXPECT_EQ(compile_args(building, {6, 7})[wire::HAS_RECEIVERS], 0u);
    EXPECT_EQ(compile_args(copy, {6, 7})[wire::HAS_RECEIVERS], 1u);
    const auto ct = compile_args(copy, {6, 7});
    const auto rt = runtime_args(copy, {4, 2}, {6, 7});
    auto moved = [&] {
        auto original = copy;
        return McastFamily(std::move(original));
    }();
    copy = building;
    moved.prepare_arguments();
    EXPECT_EQ(compile_args(moved, {6, 7}), ct);
    EXPECT_EQ(runtime_args(moved, {4, 2}, {6, 7}), rt);
    McastFamily assigned(device_);
    assigned = moved;
    EXPECT_EQ(runtime_args(assigned, {4, 2}, {6, 7}), rt);
    auto moving_building = McastFamily(device_);
    moving_building.add_group(grid({4, 2}, {5, 2}), {{4, 2}});
    assigned = std::move(moving_building);
    check_building_queries(assigned);
    assigned.prepare_arguments();
    EXPECT_EQ(assigned.participating_cores(), grid({4, 2}, {5, 2}));
}

TEST_F(McastHostFixture, IrregularReceiverSetPolicyDoesNotAffectRegularFamilies) {
    const GroupInput dense(grid({1, 1}, {3, 1}), {{1, 1}});
    McastConfig config;
    config.handshake = false;
    config.ack_count_override = 0;
    auto ordinary = make_family(device_, {dense}, config);
    auto chain_link_policy = make_family(device_, {dense}, chain_config(config));
    EXPECT_EQ(wire::transfer_mode(compile_args(ordinary)[wire::FLAGS]), dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(compile_args(chain_link_policy), compile_args(ordinary));
    EXPECT_EQ(runtime_args(chain_link_policy, {1, 1}), runtime_args(ordinary, {1, 1}));
    config.handshake = true;
    config.ack_count_override.reset();
    auto requested = make_family(device_, {dense}, chain_config(config));
    EXPECT_EQ(wire::transfer_mode(compile_args(requested)[wire::FLAGS]), dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(compile_args(requested)[wire::RECTANGLE_CAPACITY], 1u);
    EXPECT_EQ(runtime_args(requested, {1, 1})[wire::NUM_RECTANGLES], 1u);
    EXPECT_EQ(runtime_args(requested, {1, 1})[wire::ACK], 2u);
    auto rotating = make_family(device_, {GroupInput(dense.receivers, {{1, 1}, {2, 1}})}, chain_config());
    EXPECT_EQ(wire::transfer_mode(compile_args(rotating)[wire::FLAGS]), dataflow_kernel_lib::TransferMode::Multicast);
    const GroupInput irregular(cores({{0, 0}, {2, 0}, {4, 0}}), {{0, 0}});
    auto multicast = make_family(device_, {irregular}, {});
    EXPECT_EQ(compile_args(multicast)[wire::RECTANGLE_CAPACITY], 3u);
    EXPECT_EQ(wire::transfer_mode(compile_args(multicast)[wire::FLAGS]), dataflow_kernel_lib::TransferMode::Multicast);
    check_group(device_, multicast, irregular);
}

TEST_F(McastHostFixture, ChainTopologyIsExactSenderFirstAndMapped) {
    using dataflow_kernel_lib::NO_CHAIN_NEIGHBOR;
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig config;
        config.noc = noc;
        // Sender in the middle of row-major order, then outside the set.
        const auto receivers = cores({{0, 1}, {2, 1}, {0, 2}});
        for (auto sender : {CoreCoord{2, 1}, CoreCoord{4, 2}}) {
            auto family = make_family(device_, {GroupInput(receivers, {sender})}, chain_config(config));
            EXPECT_EQ(compile_args(family)[wire::RECTANGLE_CAPACITY], 0u);
            EXPECT_EQ(runtime_args(family, sender)[wire::ACK], 1u);
            const auto ct = compile_args(family);
            EXPECT_EQ(ct.size(), 12u);
            EXPECT_EQ(ct[11], 2u);
            EXPECT_EQ(wire::transfer_mode(ct[wire::FLAGS]), dataflow_kernel_lib::TransferMode::ChainUnicast);
            const std::vector<CoreCoord> expected = receivers.contains(sender)
                                                        ? std::vector<CoreCoord>{sender, {0, 1}, {0, 2}}
                                                        : std::vector<CoreCoord>{sender, {0, 1}, {2, 1}, {0, 2}};
            for (size_t i = 0; i < expected.size(); ++i) {
                auto rt = runtime_args(family, expected[i]);
                ASSERT_EQ(rt.size(), 11u);  // Two header, two head coords, five chain, two roles.
                auto mapped = [&](CoreCoord c) { return device_->worker_core_from_logical_core(c); };
                const auto predecessor = i ? mapped(expected[i - 1]) : CoreCoord{NO_CHAIN_NEIGHBOR, NO_CHAIN_NEIGHBOR};
                const auto successor =
                    i + 1 < expected.size() ? mapped(expected[i + 1]) : CoreCoord{NO_CHAIN_NEIGHBOR, NO_CHAIN_NEIGHBOR};
                EXPECT_EQ(rt[4], predecessor.x);
                EXPECT_EQ(rt[5], predecessor.y);
                EXPECT_EQ(rt[6], successor.x);
                EXPECT_EQ(rt[7], successor.y);
                EXPECT_EQ(rt[8], receivers.contains(sender));
                EXPECT_EQ(rt[9], i == 0 ? wire::CAN_SEND : wire::CAN_RECEIVE);
                EXPECT_EQ(rt[1], i == 0 ? 1u : 0u);
            }
            const auto inactive = runtime_args(family, {7, 7});
            EXPECT_EQ(inactive.size(), 11u);
            EXPECT_EQ(inactive[inactive.size() - 2], 0u);
            EXPECT_EQ(inactive.back(), wire::NO_SENDER_ROUND);
        }
    }
}

TEST_F(McastHostFixture, ChainFamilyUsesOneCompileTimeTransportForEveryGeometry) {
    const GroupInput dense(grid({0, 0}, {2, 0}), {{0, 0}});
    const GroupInput irregular(cores({{0, 2}, {2, 2}, {4, 2}}), {{0, 2}});
    const GroupInput local(cores({{6, 0}}), {{6, 0}});
    auto family = make_family(device_, {dense, irregular, local}, chain_config());
    EXPECT_EQ(compile_args(family)[wire::RECTANGLE_CAPACITY], 0u);
    const auto ct = compile_args(family);
    EXPECT_EQ(wire::transfer_mode(ct[wire::FLAGS]), dataflow_kernel_lib::TransferMode::ChainUnicast);
    EXPECT_EQ(ct[wire::ACK_COUNT], ACK_EQUALS_FANOUT);  // One successor, except for the local-only group.
    EXPECT_EQ(runtime_args(family, {6, 0})[wire::ACK], 0u);
    const auto local_rt = runtime_args(family, {6, 0});
    EXPECT_EQ(local_rt[4], dataflow_kernel_lib::NO_CHAIN_NEIGHBOR);
    EXPECT_EQ(local_rt[6], dataflow_kernel_lib::NO_CHAIN_NEIGHBOR);
    EXPECT_EQ(local_rt[8], 1u);
    for (auto core : std::vector<CoreCoord>{{0, 0}, {1, 0}, {0, 2}, {2, 2}, {6, 0}, {7, 7}}) {
        const auto rt = runtime_args(family, core);
        ASSERT_EQ(rt.size(), 11u);  // Header, head coordinates, neighbors and roles; no transport selector.
        std::vector<uint32_t> args{123};
        detail::append_args_to(args, runtime_args(family, core));
        detail::append_args_to(args, runtime_args(family, core));
        EXPECT_EQ(args.size(), 1 + 2 * rt.size());
        EXPECT_TRUE(std::equal(rt.begin(), rt.end(), args.begin() + 1));
        EXPECT_TRUE(std::equal(rt.begin(), rt.end(), args.begin() + 1 + rt.size()));
    }
    std::vector<uint32_t> appended;
    detail::append_args_to(appended, compile_args(family));
    EXPECT_EQ(appended, ct);
    EXPECT_EQ(allocated_semaphores(family).size(), 3u);
    EXPECT_EQ(allocated_semaphores(family)[2].core_ranges, family.participating_cores());
}

TEST_F(McastHostFixture, ChainRejectsUnsupportedProtocolsAndGeometry) {
    const auto receivers = cores({{0, 0}, {2, 0}});
    const GroupInput group(receivers, {{0, 0}});
    McastConfig config;
    config.handshake = false;
    EXPECT_ANY_THROW(make_family(device_, {group}, chain_config(config)));
    config.handshake = true;
    for (uint32_t ack : {0u, 1u}) {
        config.ack_count_override = ack;
        EXPECT_ANY_THROW(make_family(device_, {group}, chain_config(config)));
        EXPECT_ANY_THROW(make_family(device_, {GroupInput(receivers, {{0, 0}}, ack)}, chain_config()));
    }
    EXPECT_ANY_THROW(make_family(device_, {GroupInput(receivers, {{0, 0}, {2, 0}})}, chain_config()));
    EXPECT_ANY_THROW(
        make_family(device_, {GroupInput(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), {{0, 0}})}, chain_config()));
    EXPECT_ANY_THROW(make_family(device_, {group, group}, chain_config()));
    config.ack_count_override.reset();
    config.sem_ids = std::vector<uint32_t>{3, 4, 5};
    auto adopted = make_family(device_, {group}, chain_config(config));
    EXPECT_TRUE(allocated_semaphores(adopted, *config.sem_ids).empty());
    EXPECT_EQ(compile_args(adopted, *config.sem_ids)[wire::DATA_READY], 3u);
    EXPECT_EQ(compile_args(adopted, *config.sem_ids)[wire::CONSUMER_READY], 4u);
    EXPECT_EQ(compile_args(adopted, *config.sem_ids)[11], 5u);
}

TEST_F(McastHostFixture, SameInputsCanPrepareDifferentFamilyTransports) {
    const GroupInput group(cores({{0, 0}, {2, 0}}), {{0, 0}});
    auto chain = make_family(device_, {group}, chain_config());
    auto multicast = make_family(device_, {group});
    EXPECT_EQ(compile_args(multicast), compile_args(make_family(device_, {group})));
    check_group(device_, multicast, group);
    auto chain_again = make_family(device_, {group}, chain_config());
    EXPECT_EQ(compile_args(chain_again), compile_args(chain));
    EXPECT_EQ(runtime_args(chain_again, {2, 0}), runtime_args(chain, {2, 0}));
}

TEST_F(McastHostFixture, ChainSignalSourceAllocationAndWire) {
    const GroupInput group(cores({{0, 0}, {2, 0}}), {{0, 0}});
    auto cfg = chain_config();
    cfg.base_sem_id = 0;
    auto family = make_family(device_, {group}, cfg);
    const std::vector<uint32_t> expected{1, 1, 0, 1, 1, 9, 0, 4, 1, 2, 0, 2};
    EXPECT_EQ(compile_args(family), expected);
    EXPECT_EQ(allocated_semaphores(family).size(), 3u);
    const auto owned = allocated_semaphores(family);
    ASSERT_FALSE(owned.empty());
    EXPECT_EQ(owned.back().id + 1, 3u);
    const auto semaphores = allocated_semaphores(family);
    ASSERT_EQ(semaphores.size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(semaphores[i].id, i);
        EXPECT_EQ(semaphores[i].initial_value, 0u);
        EXPECT_EQ(semaphores[i].core_ranges, family.participating_cores());
    }
    for (const auto& ids : std::vector<std::vector<uint32_t>>{
             {3, 4}, {3, 4, UNUSED_SEM_ID}, {3, 4, 3}, {3, 4, 4}, {3, 3, 5}, {UNUSED_SEM_ID, 4, 5}}) {
        cfg.sem_ids = ids;
        EXPECT_ANY_THROW(make_family(device_, {group}, cfg));
    }
    cfg.sem_ids = std::vector<uint32_t>{3, 4, 5};
    const auto adopted = make_family(device_, {group}, cfg);
    auto adopted_expected = expected;
    adopted_expected[wire::DATA_READY] = 3;
    adopted_expected[wire::CONSUMER_READY] = 4;
    adopted_expected[wire::SIGNAL_SOURCE] = 5;
    EXPECT_EQ(compile_args(adopted, {3, 4, 5}), adopted_expected);
    EXPECT_TRUE(allocated_semaphores(adopted, {3, 4, 5}).empty());
    EXPECT_EQ(allocated_semaphores(adopted, {3, 4, 5}).size(), 0u);
    // A rectangular family resolves to multicast and needs only two semaphore IDs.
    cfg.sem_ids = std::vector<uint32_t>{3, 4};
    const auto dense = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})}, cfg);
    EXPECT_EQ(compile_args(dense, {3, 4}).size(), 11u);
}

TEST_F(McastHostFixture, DescriptorAppendPadsPerKernelAndPreservesBindings) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {GroupInput(participants, {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = grid({0, 0}, {2, 0});
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {17, 19};
    kernel.named_compile_time_args = {{"op", 99}};
    kernel.runtime_args = {{{0, 0}, {21}}, {{1, 0}, {31, 33, 35}}};
    kernel.buffer_bindings = {{.core = {1, 0}, .arg_idx = 2}};
    kernel.common_runtime_args = {71};
    // Direct references work both before and after moving a kernel into the descriptor.
    family.attach(desc, "first", kernel);
    EXPECT_EQ(
        kernel.named_compile_time_args,
        (KernelDescriptor::NamedCompileTimeArgs{{"op", 99}, {"first_ct_offset", 2}, {"first_rt_offset", 3}}));
    EXPECT_EQ(kernel.buffer_bindings[0].arg_idx, 2u);
    EXPECT_EQ(kernel.common_runtime_args, (std::vector<uint32_t>{71}));
    ASSERT_EQ(kernel.runtime_args.size(), 3u);
    for (size_t i = 0; i < kernel.runtime_args.size(); ++i) {
        const auto& [core, args] = kernel.runtime_args[i];
        std::vector<uint32_t> expected;
        if (i == 0) {
            expected = {21, 0, 0};
        } else if (i == 1) {
            expected = {31, 33, 35};
        } else {
            expected = {0, 0, 0};
        }
        const auto payload = runtime_args(family, core);
        expected.insert(expected.end(), payload.begin(), payload.end());
        EXPECT_EQ(args, expected);
    }
    auto expected_ct = std::vector<uint32_t>{17, 19};
    const auto ct = compile_args(family);
    expected_ct.insert(expected_ct.end(), ct.begin(), ct.end());
    EXPECT_EQ(kernel.compile_time_args, expected_ct);
    desc.kernels.push_back(std::move(kernel));
    const auto old_ct_size = desc.kernels[0].compile_time_args.size();
    const auto old_rt_size = desc.kernels[0].runtime_args[0].second.size();
    family.attach(desc, "second", desc.kernels[0]);
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[3].second, old_ct_size);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[4].second, old_rt_size);
    const auto before = desc.kernels[0].compile_time_args;
    EXPECT_ANY_THROW(family.attach(desc, "second", desc.kernels[0]));
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].compile_time_args, before);
    attach_absent(desc.kernels[0], "absent");
    EXPECT_EQ(desc.kernels[0].compile_time_args.back(), 0u);
    EXPECT_EQ(desc.kernels[0].runtime_args[0].second.size(), old_rt_size + runtime_args(family, {0, 0}).size());
}

TEST_F(McastHostFixture, DescriptorAppendFailurePreservesAllTargets) {
    using namespace tt::tt_metal;
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor first;
    first.core_ranges = grid({0, 0}, {1, 0});
    first.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    first.runtime_args = {{{0, 0}, {7}}, {{1, 0}, {8, 9}}};
    auto second = first;
    // Padding must not turn an out-of-bounds binding into a valid one.
    second.buffer_bindings = {{.core = {0, 0}, .arg_idx = 1}};
    const std::array targets{std::ref(first), std::ref(second)};
    EXPECT_ANY_THROW(family.attach(desc, "channel", targets));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_TRUE(first.compile_time_args.empty());
    EXPECT_TRUE(first.named_compile_time_args.empty());
    EXPECT_EQ(first.runtime_args[0].second, (std::vector<uint32_t>{7}));
    EXPECT_TRUE(second.named_compile_time_args.empty());
    const std::array duplicate{std::ref(first), std::ref(first)};
    EXPECT_ANY_THROW(family.attach(desc, "channel", duplicate));
}

TEST_F(McastHostFixture, DescriptorAttachAppendsAfterPrefixesAndPreservesBufferBindings) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {GroupInput(participants, {{0, 0}})});
    ProgramDescriptor desc;
    // Different cores occupy different slots: allocation must consider their union.
    desc.semaphores.push_back({.id = 0, .core_ranges = grid({0, 0}, {0, 0})});
    desc.semaphores.push_back({.id = 2, .core_ranges = grid({1, 0}, {1, 0})});
    KernelDescriptor kernel;
    kernel.core_ranges = grid({0, 0}, {2, 0});
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {17, 19};
    kernel.common_runtime_args = {71};
    kernel.runtime_args = {{{0, 0}, {21, 23}}, {{1, 0}, {31, 33, 35}}, {{2, 0}, {41, 43}}};
    kernel.buffer_bindings = {{.core = {0, 0}, .arg_idx = 0}, {.core = {1, 0}, .arg_idx = 1}};
    kernel.common_buffer_bindings = {{.arg_idx = 0}};
    desc.kernels.push_back(kernel);
    const std::array points{std::ref(desc.kernels[0])};
    family.attach(desc, "mcast", points);
    ASSERT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.semaphores[2].id, 1u);
    EXPECT_EQ(desc.semaphores[3].id, 3u);
    const auto& attached = desc.kernels[0];
    const std::vector<uint32_t> expected_ct{17, 19, 1, 1, 1, 3, 1, 1, 0, 3, 1, 2, 1};
    EXPECT_EQ(attached.compile_time_args, expected_ct);
    const auto a = device_->worker_core_from_logical_core({0, 0});
    const auto b = device_->worker_core_from_logical_core({1, 0});
    const std::vector<uint32_t> expected_sender{21, 23, 0, 1, 1, a.x, a.y, a.x, a.y, b.x, b.y, 1, 2, 3, 1, 0};
    const std::vector<uint32_t> expected_receiver{31, 33, 35, 0, 0, a.x, a.y, 0, 0, 0, 0, 0, 0, 0, 2, 0xFFFFFFFFu};
    const std::vector<uint32_t> expected_inactive{41, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFFu};
    EXPECT_EQ(attached.runtime_args[0].second, expected_sender);
    EXPECT_EQ(attached.runtime_args[1].second, expected_receiver);
    EXPECT_EQ(attached.runtime_args[2].second, expected_inactive);
    EXPECT_EQ(attached.buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.buffer_bindings[1].arg_idx, 1u);
    EXPECT_EQ(attached.common_buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.common_runtime_args, std::vector<uint32_t>{71});
    // Automatic resource assignment belongs to the descriptor, not the prepared family.
    EXPECT_EQ(compile_args(family)[2], 0u);
    EXPECT_EQ(compile_args(family)[3], 1u);
}

TEST_F(McastHostFixture, DescriptorAttachValidatesAllKernelsBeforeMutation) {
    using namespace tt::tt_metal;
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor sender;
    sender.core_ranges = grid({0, 0}, {0, 0});
    sender.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    sender.compile_time_args = {17};
    sender.runtime_args = {{{0, 0}, {23}}};
    KernelDescriptor receiver;
    receiver.core_ranges = grid({1, 0}, {1, 0});
    receiver.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};  // Pure receivers may use the other NoC.
    receiver.compile_time_args = {29};
    desc.kernels = {sender, receiver};
    const std::array bad_points{std::ref(desc.kernels[0]), std::ref(desc.kernels[1])};
    desc.kernels[1].buffer_bindings = {{.core = {1, 0}, .arg_idx = 0}};
    EXPECT_ANY_THROW(family.attach(desc, "mcast", bad_points));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_EQ(desc.kernels[0].compile_time_args, sender.compile_time_args);
    EXPECT_EQ(desc.kernels[0].runtime_args, sender.runtime_args);
    EXPECT_TRUE(desc.kernels[1].runtime_args.empty());
    auto points = bad_points;
    desc.kernels[1].buffer_bindings.clear();
    EXPECT_NO_THROW(family.attach(desc, "mcast", points));
    EXPECT_EQ(desc.kernels[1].runtime_args.size(), 1u);
}

TEST_F(McastHostFixture, DescriptorAttachAdoptionAndAbsence) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(
        device_,
        {GroupInput(participants, {{0, 0}})},
        McastConfig{.handshake = false, .sem_ids = std::vector<uint32_t>{5}});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {31};
    desc.kernels.push_back(kernel);
    const std::array points{std::ref(desc.kernels[0])};
    EXPECT_ANY_THROW(family.attach(desc, "mcast", points));
    EXPECT_EQ(desc.kernels[0].compile_time_args, kernel.compile_time_args);
    desc.semaphores.push_back({.id = 5, .core_ranges = participants, .initial_value = 1});
    EXPECT_ANY_THROW(family.attach(desc, "mcast", points));
    desc.semaphores[0].initial_value = 0;
    family.attach(desc, "mcast", points);
    EXPECT_EQ(desc.semaphores.size(), 1u);
    EXPECT_EQ(desc.kernels[0].compile_time_args[3], 5u);
    EXPECT_EQ(desc.kernels[0].compile_time_args[4], 0xFFFFFFFFu);
    EXPECT_EQ(desc.kernels[0].compile_time_args[6], 0u);
    const auto runtime = desc.kernels[0].runtime_args;
    attach_absent(desc.kernels[0], "absent_mcast");
    EXPECT_EQ(desc.kernels[0].compile_time_args[12], 0u);
    EXPECT_EQ(desc.kernels[0].compile_time_args[0], 31u);
    EXPECT_EQ(desc.kernels[0].runtime_args, runtime);
    EXPECT_EQ(desc.semaphores.size(), 1u);
}

TEST_F(McastHostFixture, DescriptorAttachExactAllocationAndIndependentFamilies) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    const GroupInput group(participants, {{0, 0}});
    auto exact = make_family(device_, {group}, McastConfig{.base_sem_id = 4});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {97};
    desc.kernels.push_back(kernel);
    const std::array first{std::ref(desc.kernels[0])};
    desc.semaphores.push_back({.id = 5, .core_ranges = participants});
    EXPECT_ANY_THROW(exact.attach(desc, "mcast", first));
    EXPECT_EQ(desc.semaphores.size(), 1u);
    EXPECT_EQ(desc.kernels[0].compile_time_args, kernel.compile_time_args);
    EXPECT_TRUE(desc.kernels[0].runtime_args.empty());
    desc.semaphores.clear();
    exact.attach(desc, "mcast", first);
    EXPECT_EQ(desc.semaphores[0].id, 4u);
    EXPECT_EQ(desc.semaphores[1].id, 5u);

    // The second channel allocates its own slots, appending after the first helper block.
    auto automatic = make_family(device_, {group});
    const std::array second{std::ref(desc.kernels[0])};
    automatic.attach(desc, "second_mcast", second);
    ASSERT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.semaphores[2].id, 0u);
    EXPECT_EQ(desc.semaphores[3].id, 1u);
    const std::vector<uint32_t> expected{97, 1, 1, 4, 5, 1, 1, 0, 3, 1, 2, 1, 1, 1, 0, 1, 1, 1, 0, 3, 1, 2, 1};
    EXPECT_EQ(desc.kernels[0].compile_time_args, expected);
    for (const auto& [core, args] : desc.kernels[0].runtime_args) {
        ASSERT_EQ(args.size(), 26u);
        EXPECT_TRUE(std::equal(args.begin(), args.begin() + 13, args.begin() + 13));
    }
    auto exhausted_base = make_family(device_, {group}, McastConfig{.base_sem_id = 15});
    const auto before = desc.kernels[0];
    EXPECT_ANY_THROW(exhausted_base.attach(desc, "third_mcast", second));
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].compile_time_args, before.compile_time_args);
    EXPECT_EQ(desc.kernels[0].runtime_args, before.runtime_args);
}

TEST_F(McastHostFixture, DescriptorAttachChainRequiresForwarderNocAndThreeResources) {
    using namespace tt::tt_metal;
    const auto participants = cores({{0, 0}, {2, 0}});
    auto family = make_family(device_, {GroupInput(participants, {{0, 0}})}, chain_config());
    ProgramDescriptor desc;
    KernelDescriptor head;
    head.core_ranges = grid({0, 0}, {0, 0});
    head.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    KernelDescriptor forwarder;
    forwarder.core_ranges = grid({2, 0}, {2, 0});
    forwarder.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    desc.kernels = {head, forwarder};
    const std::array points{std::ref(desc.kernels[0]), std::ref(desc.kernels[1])};
    EXPECT_ANY_THROW(family.attach(desc, "mcast", points));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_TRUE(desc.kernels[0].compile_time_args.empty());
    desc.kernels[1].config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    family.attach(desc, "mcast", points);
    ASSERT_EQ(desc.semaphores.size(), 3u);
    const std::vector<uint32_t> expected{1, 1, 0, 1, 1, 9, 0, 4, 1, 2, 0, 2};
    for (const auto& attached : desc.kernels) {
        EXPECT_EQ(attached.compile_time_args, expected);
        ASSERT_EQ(attached.runtime_args.size(), 1u);
        EXPECT_EQ(attached.runtime_args[0].second.size(), 11u);
    }
}

TEST_F(McastHostFixture, DescriptorAttachUsesNativeReaderWriterNocDefaults) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    const GroupInput group(participants, {{0, 0}});
    for (bool reader : {false, true}) {
        // Resolve through the actual native config constructors, independently of attach.
        const auto native_noc = reader ? ReaderDataMovementConfig{}.noc : WriterDataMovementConfig{}.noc;
        auto family = make_family(device_, {group}, McastConfig{.noc = native_noc});
        ProgramDescriptor desc;
        KernelDescriptor kernel;
        kernel.core_ranges = participants;
        if (reader) {
            kernel.config = ReaderConfigDescriptor{};
        } else {
            kernel.config = WriterConfigDescriptor{};
        }
        desc.kernels.push_back(kernel);
        const std::array points{std::ref(desc.kernels[0])};
        EXPECT_NO_THROW(family.attach(desc, "mcast", points));

        const auto other_noc = native_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
        auto mismatched = make_family(device_, {group}, McastConfig{.noc = other_noc});
        desc.kernels = {kernel};
        desc.semaphores.clear();
        EXPECT_ANY_THROW(mismatched.attach(desc, "mcast", points));
        EXPECT_TRUE(desc.kernels[0].compile_time_args.empty());
        EXPECT_TRUE(desc.semaphores.empty());
    }
}

}  // namespace ttnn::kernel_lib::host::test

namespace ttnn::kernel_lib::host::test {
namespace m2 = tt::tt_metal::experimental;

m2::ProgramSpec spec_pair(uint32_t prefix = 0) {
    m2::ProgramSpec spec;
    for (const auto& name : {"sender", "receiver"}) {
        m2::KernelSpec kernel{
            .unique_id = m2::KernelSpecName{name},
            .source = m2::KernelSpec::SourceCode{"void kernel_main() {}"},
            .compile_time_args = {{"kept_ct", 73}},
            .runtime_arg_schema = {.runtime_arg_names = {"kept_rt"}, .common_runtime_arg_names = {"kept_common"}},
            .hw_config = m2::DataMovementHardwareConfig{m2::CreateReaderGen1DataMovementConfig()}};
        kernel.advanced_options.num_runtime_varargs = prefix;
        spec.kernels.push_back(std::move(kernel));
    }
    spec.work_units = {
        {.name = "sender", .kernels = {m2::KernelSpecName{"sender"}}, .target_nodes = CoreCoord{0, 0}},
        {.name = "receiver", .kernels = {m2::KernelSpecName{"receiver"}}, .target_nodes = grid({1, 0}, {2, 0})}};
    return spec;
}
const std::array spec_targets{m2::KernelSpecName{"sender"}, m2::KernelSpecName{"receiver"}};

TEST_F(McastHostFixture, SpecAttachPopulatesNamedMetadataResourcesAndRuntimePrefixes) {
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair(2);
    m2::ProgramRunArgs args;
    for (size_t i = 0; i < spec_targets.size(); ++i) {
        m2::KernelRunArgs values{.kernel = spec_targets[i], .common_runtime_arg_values = {{"kept_common", 97}}};
        for (uint32_t x = i == 0 ? 0 : 1; x <= (i == 0 ? 0 : 2); ++x) {
            values.advanced_options.runtime_varargs[{x, 0}] = {41 + x, 51 + x};
            values.runtime_arg_values["kept_rt"][{x, 0}] = 83 + x;
        }
        args.kernel_run_args.push_back(std::move(values));
    }
    family.attach(spec, args, "channel", spec_targets);
    ASSERT_EQ(spec.semaphores.size(), 2u);
    EXPECT_EQ(spec.semaphores[0].unique_id, m2::SemaphoreSpecName{"channel_mcast_data_ready"});
    EXPECT_EQ(spec.semaphores[1].unique_id, m2::SemaphoreSpecName{"channel_mcast_consumer_ready"});
    const auto sender = device_->worker_core_from_logical_core({0, 0});
    const auto end = device_->worker_core_from_logical_core({1, 0});
    for (size_t i = 0; i < spec.kernels.size(); ++i) {
        const auto& kernel = spec.kernels[i];
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, 15u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_rt_base").value(), 2u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_flags").value(), 1u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_rectangle_capacity").value(), 1u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_uniform_remote_count").value(), 1u);
        EXPECT_EQ(kernel.compile_time_args.get("kept_ct").value(), 73u);
        EXPECT_EQ(
            kernel.compiler_options.defines.get("channel_mcast_data_ready_type").value(),
            "sem::channel_mcast_data_ready_t");
        EXPECT_EQ(kernel.compiler_options.defines.get("channel_mcast_signal_source_type").value(), "std::nullptr_t");
        EXPECT_EQ(args.kernel_run_args[i].common_runtime_arg_values.get("kept_common").value(), 97u);
    }
    const std::vector<uint32_t> sender_expected{
        41, 51, 1, 1, sender.x, sender.y, sender.x, sender.y, end.x, end.y, 1, 2, 3, 1, 0};
    EXPECT_EQ(args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value(), sender_expected);
    EXPECT_EQ(args.kernel_run_args[1].runtime_arg_values.get("kept_rt").value().get({1, 0}).value(), 84u);
    const auto& inactive = args.kernel_run_args[1].advanced_options.runtime_varargs.get({2, 0}).value();
    EXPECT_EQ(inactive[13], 0u);
    EXPECT_EQ(inactive[14], 0xffffffffu);
}

TEST_F(McastHostFixture, SpecAttachComposesAndNativeRunArgsCopiesKeepPayloads) {
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair();
    m2::ProgramRunArgs args;
    family.attach(spec, args, "first", spec_targets);
    const auto payload = args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value();
    family.attach(spec, args, "second", spec_targets);
    EXPECT_EQ(spec.semaphores.size(), 4u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("second_mcast_rt_base").value(), 13u);
    EXPECT_EQ(spec.kernels[0].advanced_options.num_runtime_varargs, 26u);
    auto copied = args;
    copied.kernel_run_args[0].runtime_arg_values["kept_rt"][{0, 0}] = 123;
    auto expected = payload;
    expected.insert(expected.end(), payload.begin(), payload.end());
    EXPECT_EQ(copied.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value(), expected);
    EXPECT_TRUE(args.kernel_run_args[0].runtime_arg_values.empty());
    EXPECT_ANY_THROW(family.attach(spec, args, "first", spec_targets));
    EXPECT_EQ(spec.semaphores.size(), 4u);
}

TEST_F(McastHostFixture, SpecAttachFailuresLeaveBothObjectsUnchanged) {
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    for (const auto* const violation :
         {"missing-prefix",
          "trailing-values",
          "duplicate-run-entry",
          "unplaced-run-node",
          "wrong-sender-noc",
          "duplicate-target",
          "duplicate-prefix"}) {
        auto spec = spec_pair(1);
        m2::ProgramRunArgs args;
        for (const auto& name : spec_targets) {
            args.kernel_run_args.push_back({.kernel = name});
        }
        args.kernel_run_args[0].advanced_options.runtime_varargs[{0, 0}] = {19};
        args.kernel_run_args[1].advanced_options.runtime_varargs[{1, 0}] = {23};
        args.kernel_run_args[1].advanced_options.runtime_varargs[{2, 0}] = {29};
        if (std::string_view(violation) == "missing-prefix") {
            args.kernel_run_args[1].advanced_options.runtime_varargs.erase({2, 0});
        }
        if (std::string_view(violation) == "trailing-values") {
            args.kernel_run_args[1].advanced_options.runtime_varargs[{2, 0}].push_back(31);
        }
        if (std::string_view(violation) == "duplicate-run-entry") {
            args.kernel_run_args.push_back(args.kernel_run_args[0]);
        }
        if (std::string_view(violation) == "unplaced-run-node") {
            args.kernel_run_args[1].advanced_options.runtime_varargs[{3, 0}] = {37};
        }
        if (std::string_view(violation) == "wrong-sender-noc") {
            std::get<m2::DataMovementGen1Config>(std::get<m2::DataMovementHardwareConfig>(spec.kernels[0].hw_config))
                .noc = NOC::NOC_1;
        }
        if (std::string_view(violation) == "duplicate-prefix") {
            spec.kernels[1].compile_time_args["channel_mcast_flags"] = 55;
        }
        auto targets = spec_targets;
        if (std::string_view(violation) == "duplicate-target") {
            targets[1] = targets[0];
        }
        const auto before = args;
        const auto before_spec = spec;
        EXPECT_ANY_THROW(family.attach(spec, args, "channel", targets)) << violation;
        EXPECT_TRUE(spec.semaphores.empty()) << violation;
        ASSERT_EQ(args.kernel_run_args.size(), before.kernel_run_args.size());
        for (size_t i = 0; i < spec.kernels.size(); ++i) {
            EXPECT_EQ(spec.kernels[i].compile_time_args, before_spec.kernels[i].compile_time_args) << violation;
            EXPECT_EQ(spec.kernels[i].advanced_options.num_runtime_varargs, 1u) << violation;
            EXPECT_TRUE(spec.kernels[i].semaphore_bindings.empty()) << violation;
            EXPECT_TRUE(spec.kernels[i].compiler_options.defines.empty()) << violation;
        }
        for (size_t i = 0; i < args.kernel_run_args.size(); ++i) {
            EXPECT_EQ(
                args.kernel_run_args[i].advanced_options.runtime_varargs,
                before.kernel_run_args[i].advanced_options.runtime_varargs)
                << violation;
        }
    }
}

TEST_F(McastHostFixture, SpecAttachAdoptsNamedResourcesAndRejectsNumericConfiguration) {
    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {GroupInput(participants, {{0, 0}})}, McastConfig{.handshake = false});
    auto spec = spec_pair();
    m2::ProgramRunArgs args;
    const std::array adopted{m2::SemaphoreSpecName{"existing"}};
    EXPECT_ANY_THROW(family.attach(spec, args, "channel", spec_targets, adopted));
    spec.semaphores.push_back({.unique_id = adopted[0], .target_nodes = CoreCoord{0, 0}});
    EXPECT_ANY_THROW(family.attach(spec, args, "channel", spec_targets, adopted));
    spec.semaphores[0].target_nodes = participants;
    spec.semaphores[0].advanced_options.initial_value = 1;
    EXPECT_ANY_THROW(family.attach(spec, args, "channel", spec_targets, adopted));
    spec.semaphores[0].advanced_options.initial_value = 0;
    family.attach(spec, args, "channel", spec_targets, adopted);
    ASSERT_EQ(spec.semaphores.size(), 1u);
    ASSERT_EQ(spec.kernels[0].semaphore_bindings.size(), 1u);
    EXPECT_EQ(spec.kernels[0].semaphore_bindings[0].semaphore_spec_name, adopted[0]);
    EXPECT_EQ(
        spec.kernels[0].compiler_options.defines.get("channel_mcast_consumer_ready_type").value(), "std::nullptr_t");
    auto numeric = make_family(device_, {GroupInput(participants, {{0, 0}})}, McastConfig{.base_sem_id = 0});
    auto fresh = spec_pair();
    EXPECT_ANY_THROW(numeric.attach(fresh, args, "numeric", spec_targets));
    EXPECT_TRUE(fresh.semaphores.empty());
}

TEST_F(McastHostFixture, SpecAttachValidatesUniformVarargSchemaAndNormalizesOverrides) {
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair();
    spec.kernels[1].advanced_options.num_runtime_varargs_per_node.emplace(CoreCoord{1, 0}, 2);
    m2::ProgramRunArgs args;
    EXPECT_ANY_THROW(family.attach(spec, args, "channel", spec_targets));
    EXPECT_TRUE(args.kernel_run_args.empty());
    spec.kernels[1].advanced_options.num_runtime_varargs_per_node.emplace(CoreCoord{2, 0}, 2);
    args.kernel_run_args.push_back({.kernel = spec_targets[1]});
    args.kernel_run_args[0].advanced_options.runtime_varargs[{1, 0}] = {31, 37};
    args.kernel_run_args[0].advanced_options.runtime_varargs[{2, 0}] = {41, 43};
    family.attach(spec, args, "channel", spec_targets);
    EXPECT_EQ(spec.kernels[1].advanced_options.num_runtime_varargs, 15u);
    EXPECT_TRUE(spec.kernels[1].advanced_options.num_runtime_varargs_per_node.empty());
    EXPECT_EQ(spec.kernels[1].compile_time_args.get("channel_mcast_rt_base").value(), 2u);
}

TEST_F(McastHostFixture, SpecAbsentNeedsNoResourcesOrRunArgumentObject) {
    auto spec = spec_pair(3);
    attach_absent(spec, "none", spec_targets);
    EXPECT_TRUE(spec.semaphores.empty());
    for (const auto& kernel : spec.kernels) {
        EXPECT_EQ(kernel.compile_time_args.get("none_mcast_tag").value(), 0u);
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, 3u);
        EXPECT_TRUE(kernel.semaphore_bindings.empty());
        EXPECT_EQ(kernel.compiler_options.defines.size(), 3u);
        for (const auto& [name, value] : kernel.compiler_options.defines) {
            EXPECT_EQ(value, "std::nullptr_t");
        }
    }
    EXPECT_ANY_THROW(attach_absent(spec, "none", spec_targets));
}

TEST_F(McastHostFixture, SpecAttachAllowsOtherNocOnlyOnPureMulticastReceivers) {
    const auto participants = cores({{0, 0}, {2, 0}});
    auto multicast = make_family(device_, {GroupInput(participants, {{0, 0}})});
    auto chain = make_family(device_, {GroupInput(participants, {{0, 0}})}, chain_config());
    auto spec = spec_pair();
    std::get<m2::DataMovementGen1Config>(std::get<m2::DataMovementHardwareConfig>(spec.kernels[1].hw_config)).noc =
        NOC::NOC_1;
    m2::ProgramRunArgs args;
    EXPECT_ANY_THROW(chain.attach(spec, args, "chain", spec_targets));
    EXPECT_TRUE(args.kernel_run_args.empty());
    EXPECT_NO_THROW(multicast.attach(spec, args, "multicast", spec_targets));
}
}  // namespace ttnn::kernel_lib::host::test

namespace ttnn::kernel_lib::host::test {

void run_spec_device_contract(
    tt::tt_metal::distributed::MeshDevice& device,
    NOC noc,
    bool counter,
    bool rotating,
    bool control,
    bool chain,
    bool handshake,
    bool local) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;
    SCOPED_TRACE(
        ::testing::Message() << "noc=" << noc << " counter=" << counter << " rotating=" << rotating << " control="
                             << control << " chain=" << chain << " handshake=" << handshake << " local=" << local);
    std::vector<CoreCoord> active;
    if (local) {
        active = {{0, 0}};
    } else if (chain) {
        active = {{0, 0}, {1, 0}, {0, 1}};
    } else {
        active = {{0, 0}, {1, 0}, {2, 0}};
    }
    auto placed = active;
    placed.push_back({3, 0});  // Placed kernel outside either family must get inactive role data.
    const auto participants = cores(active);
    const auto placement = cores(placed);
    const uint32_t rounds = handshake ? 4 : 1;
    const std::array targets{m2::KernelSpecName{"contract"}};
    m2::KernelSpec kernel{
        .unique_id = targets.front(),
        .source = "tests/ttnn/unit_tests/kernel_lib/kernels/mcast_spec.cpp",
        .compile_time_args = {{"rounds", rounds}, {"control", control ? 1u : 0u}},
        .hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = noc}};
    kernel.advanced_options.num_runtime_varargs = 2;
    kernel.scratchpad_bindings.push_back(
        {.scratchpad_spec_name = m2::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    m2::ProgramSpec spec{
        .kernels = {kernel},
        .scratchpads = {{.unique_id = m2::ScratchpadSpecName{"pad"}, .size_per_node = 256}},
        .work_units = {{.name = "contract", .kernels = {targets.front()}, .target_nodes = placement}}};
    m2::ProgramRunArgs populated;
    populated.kernel_run_args.push_back({.kernel = targets.front()});
    for (auto node : placed) {
        populated.kernel_run_args.front().advanced_options.runtime_varargs.emplace(
            node, std::vector<uint32_t>{0x1234, 0x5678});
    }
    McastConfig cfg{
        .noc = noc,
        .handshake = handshake,
        .data_ready =
            counter ? dataflow_kernel_lib::DataReadySignal::Counter : dataflow_kernel_lib::DataReadySignal::Flag,
        .irregular_receiver_set_mode = chain ? TransferMode::ChainUnicast : TransferMode::Multicast};
    std::vector<CoreCoord> senders{{0, 0}};
    if (rotating) {
        senders.push_back({2, 0});
    }
    auto family = make_family(&device, {GroupInput(participants, senders)}, cfg);
    family.attach(spec, populated, "channel", targets);
    auto second = make_family(&device, {GroupInput(participants, {{0, 0}})}, McastConfig{.noc = noc});
    second.attach(spec, populated, "second", targets);
    attach_absent(spec, "absent", targets);
    // Named RT values may be declared after attach: generated get_vararg must use
    // the final named-argument offset, preserving both the caller prefix and helper slices.
    spec.kernels.front().runtime_arg_schema.runtime_arg_names = {"seed", "report_addr"};
    auto workload = m2::MakeMeshWorkloadFromSpec(device, spec);
    auto* physical = device.get_devices().front();
    for (uint32_t run = 0; run < 3; ++run) {
        const uint32_t seed = 19 + run * 173;
        const uint32_t report_addr = 100 * 1024 + run * sizeof(uint32_t);
        // Native value copies retain the topology, while operation values change each launch.
        auto invocation = populated;
        for (auto node : placed) {
            m2::AddRuntimeArgsForNode(
                invocation.kernel_run_args.front().runtime_arg_values,
                node,
                {{"seed", seed}, {"report_addr", report_addr}});
            std::vector<uint32_t> zero{0};
            tt::tt_metal::detail::WriteToDeviceL1(physical, node, report_addr, zero);
        }
        for (auto& [region, program] : workload.get_programs()) {
            m2::SetProgramRunArgs(program, invocation);
        }
        EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
        for (auto node : placed) {
            SCOPED_TRACE(::testing::Message() << "run=" << run << " node=(" << node.x << "," << node.y << ")");
            std::vector<uint32_t> base, result;
            tt::tt_metal::detail::ReadFromDeviceL1(physical, node, report_addr, 4, base);
            ASSERT_EQ(base.size(), 1u);
            ASSERT_NE(base.front(), 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(physical, node, base.front() + 128, 48, result);
            ASSERT_EQ(result.size(), 12u);
            const bool inside = node != CoreCoord{3, 0};
            for (uint32_t round = 0; round < rounds; ++round) {
                uint32_t expected = 136 * (seed + round * 100) + 1360;
                if (control) {
                    expected = counter ? round + 1 : 1;
                }
                EXPECT_EQ(result[round], inside ? expected : 0u);
            }
            EXPECT_EQ(result[8], inside ? 136 * (seed + 1000) + 1360 : 0u);
            EXPECT_EQ(result[9], 0x1234u);
            EXPECT_EQ(result[10], 0x5678u);
            EXPECT_EQ(result[11], 0xA55Au);
        }
    }
}

TEST_F(McastHostFixture, SpecDeviceSmoke) {
    run_spec_device_contract(*device_, NOC::NOC_0, false, false, false, false, true, false);
}

TEST_F(McastHostFixture, SpecDeviceMatrix) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool counter : {false, true}) {
            for (bool control : {false, true}) {
                for (bool rotating : {false, true}) {
                    run_spec_device_contract(*device_, noc, counter, rotating, control, false, true, false);
                }
                run_spec_device_contract(*device_, noc, counter, false, control, true, true, false);
                run_spec_device_contract(*device_, noc, counter, false, control, false, false, false);
                run_spec_device_contract(*device_, noc, counter, false, control, false, true, true);
            }
        }
    }
}
TEST_F(McastHostFixture, ProgramBindingRejectsRepeatedAppendAndPreservesResolvedIds) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {{participants, {{0, 0}}}});
    Program program;
    program.impl().add_semaphore(participants, 0, 7, tt::CoreType::WORKER);
    program.impl().add_semaphore(participants, 2, 9, tt::CoreType::WORKER);
    std::vector<uint32_t> ct{71}, rt{73};
    EXPECT_ANY_THROW(family.append_compile_time_args_to(ct));
    EXPECT_ANY_THROW(family.append_runtime_args_to(rt, {0, 0}));
    EXPECT_EQ(ct, (std::vector<uint32_t>{71}));
    EXPECT_EQ(rt, (std::vector<uint32_t>{73}));
    family.append_semaphores(program);
    family.append_compile_time_args_to(ct);
    family.append_runtime_args_to(rt, {0, 0});
    EXPECT_EQ(ct[1 + wire::DATA_READY], 1u);
    EXPECT_EQ(ct[1 + wire::CONSUMER_READY], 3u);
    EXPECT_EQ(rt.front(), 73u);
    EXPECT_GT(rt.size(), 1u);
    ASSERT_EQ(program.impl().semaphores().size(), 4u);
    EXPECT_EQ(program.impl().semaphores()[0].initial_value(), 7u);
    EXPECT_ANY_THROW(family.append_semaphores(program));
    auto copy = family;
    EXPECT_ANY_THROW(copy.append_semaphores(program));
    EXPECT_EQ(program.impl().semaphores().size(), 4u);
    Program other;
    EXPECT_ANY_THROW(copy.append_semaphores(other));
    EXPECT_TRUE(other.impl().semaphores().empty());
    Program moved = std::move(program);
    EXPECT_ANY_THROW(family.append_semaphores(moved));
    EXPECT_EQ(moved.impl().semaphores().size(), 4u);
    std::vector<uint32_t> copied_ct;
    copy.append_compile_time_args_to(copied_ct);
    EXPECT_EQ(copied_ct[wire::DATA_READY], 1u);
    EXPECT_EQ(copied_ct[wire::CONSUMER_READY], 3u);
    ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    EXPECT_ANY_THROW(family.attach(descriptor, "weights", kernel));
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    EXPECT_ANY_THROW(family.attach(spec, run_args, "weights", {}));
}

TEST_F(McastHostFixture, ProgramBindingChecksAllocatedIdsAndSupportsAdoption) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto exact = make_family(device_, {{participants, {{0, 0}}}}, {.base_sem_id = 2});
    Program mismatch;
    EXPECT_ANY_THROW(exact.append_semaphores(mismatch));
    // CreateSemaphore chooses the next free ID; an expected-ID mismatch is detected afterwards.
    ASSERT_EQ(mismatch.impl().semaphores().size(), 1u);
    EXPECT_EQ(mismatch.impl().semaphores().front().id(), 0u);
    std::vector<uint32_t> ct;
    EXPECT_ANY_THROW(exact.append_compile_time_args_to(ct));

    Program program;
    EXPECT_EQ(CreateSemaphore(program, participants, 7), 0u);
    EXPECT_EQ(CreateSemaphore(program, participants, 9), 1u);
    exact.append_semaphores(program);
    exact.append_compile_time_args_to(ct);
    EXPECT_EQ(ct[wire::DATA_READY], 2u);
    EXPECT_EQ(ct[wire::CONSUMER_READY], 3u);
    ASSERT_EQ(program.impl().semaphores().size(), 4u);
    EXPECT_EQ(program.impl().semaphores()[2].initial_value(), 0u);
    EXPECT_EQ(program.impl().semaphores()[3].initial_value(), 0u);

    Mcast2D passive(
        device_,
        participants,
        Mcast2DFixedSenderConfig{{0, 0}},
        {.handshake = false, .sem_ids = std::vector<uint32_t>{2}});
    passive.append_semaphores(program);
    ct.clear();
    passive.append_compile_time_args_to(ct);
    EXPECT_EQ(ct[wire::DATA_READY], 2u);
    EXPECT_EQ(ct[wire::CONSUMER_READY], UNUSED_SEM_ID);
    EXPECT_EQ(program.impl().semaphores().size(), 4u);

    Mcast1D another(device_, participants, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{});
    another.append_semaphores(program);
    ct.clear();
    another.append_compile_time_args_to(ct);
    EXPECT_EQ(ct[wire::DATA_READY], 4u);
    EXPECT_EQ(ct[wire::CONSUMER_READY], 5u);
    EXPECT_EQ(program.impl().semaphores().size(), 6u);

    for (const auto& ids : std::vector<std::vector<uint32_t>>{{2, 2}, {2, NUM_SEMAPHORES}, {2, 3, 4}}) {
        auto invalid = make_family(device_, {{participants, {{0, 0}}}}, {.sem_ids = ids});
        EXPECT_ANY_THROW(invalid.append_semaphores(program));
        EXPECT_EQ(program.impl().semaphores().size(), 6u);
    }
    auto invalid_base = make_family(device_, {{participants, {{0, 0}}}}, {.base_sem_id = NUM_SEMAPHORES - 1});
    EXPECT_ANY_THROW(invalid_base.append_semaphores(program));
    EXPECT_EQ(program.impl().semaphores().size(), 6u);
}

TEST_F(McastHostFixture, ProgramBindingCoversChainExhaustionAndCompiledPrograms) {
    using namespace tt::tt_metal;
    auto chain = make_family(device_, {{cores({{0, 0}, {1, 1}, {2, 0}}), {{0, 0}}}}, chain_config());
    Program chain_program;
    chain.append_semaphores(chain_program);
    std::vector<uint32_t> ct;
    chain.append_compile_time_args_to(ct);
    ASSERT_EQ(chain_program.impl().semaphores().size(), 3u);
    EXPECT_EQ(ct[wire::SIGNAL_SOURCE], 2u);

    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {{participants, {{0, 0}}}});
    Program full;
    for (uint32_t id = 0; id + 1 < NUM_SEMAPHORES; ++id) {
        full.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
    }
    EXPECT_ANY_THROW(family.append_semaphores(full));
    // The public allocator has no transaction/rollback: the first role consumes the last slot.
    EXPECT_EQ(full.impl().semaphores().size(), NUM_SEMAPHORES);
    EXPECT_ANY_THROW(family.append_compile_time_args_to(ct));
    Program fresh;
    family.append_semaphores(fresh);
    ct.clear();
    family.append_compile_time_args_to(ct);
    EXPECT_EQ(ct[wire::DATA_READY], 0u);
    EXPECT_EQ(ct[wire::CONSUMER_READY], 1u);
    ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.kernel_source = "void kernel_main() {}";
    kernel.source_type = KernelDescriptor::SourceType::SOURCE_CODE;
    kernel.core_ranges = grid({0, 0}, {0, 0});
    kernel.config = ReaderConfigDescriptor{};
    descriptor.kernels.push_back(kernel);
    Program compiled(descriptor);
    compiled.impl().compile(device_);
    ASSERT_TRUE(compiled.impl().is_compiled());
    EXPECT_ANY_THROW(family.append_semaphores(compiled));
    EXPECT_TRUE(compiled.impl().semaphores().empty());
}

}  // namespace ttnn::kernel_lib::host::test
