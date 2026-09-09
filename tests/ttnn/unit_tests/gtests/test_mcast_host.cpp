// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.hpp"
#include <set>
#include <vector>
#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::kernel_lib::host::test {
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
namespace wire = dataflow_kernel_lib::mcast_wire;
using dataflow_kernel_lib::SenderTransferMode;
class McastHostFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastHostFixture> {};

CoreRangeSet grid(CoreCoord start, CoreCoord end) { return CoreRangeSet(CoreRange(start, end)); }
CoreRangeSet cores(const std::vector<CoreCoord>& values) {
    std::vector<CoreRange> ranges;
    for (auto core : values) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(std::move(ranges));
}
using Coordinates = std::set<std::pair<uint32_t, uint32_t>>;

// Independent oracle: enumerate the emitted destinations and compare with each mapped logical
// receiver, without using the helper's decomposition or a bounding-box reference.
void check_group(tt::tt_metal::IDevice* device, const McastFamily& family, const McastGroup& group) {
    const auto ct = family.compile_time_args();
    ASSERT_EQ(ct.size(), 11u);
    ASSERT_EQ(ct[0], 1u);
    const uint32_t count = group.senders().size();
    EXPECT_EQ(ct[6], group.rotating() ? count : 0u);
    Coordinates expected;
    for (auto c : tt::tt_metal::corerange_to_cores(group.receiver_cores(), std::nullopt, true)) {
        auto w = device->worker_core_from_logical_core(c);
        expected.emplace(w.x, w.y);
    }
    for (uint32_t phase = 0; phase < count; ++phase) {
        const auto sender = group.senders()[phase];
        const auto worker = device->worker_core_from_logical_core(sender);
        const auto rt = family.runtime_args(sender);
        ASSERT_EQ(rt.size(), 2u + 2u * count + 7u * ct[10] + 2u);
        EXPECT_EQ(rt[0], family.num_rectangles(sender));
        EXPECT_EQ(rt[1], family.ack_count(sender));
        EXPECT_EQ(rt[rt.size() - 1], phase);
        EXPECT_EQ(
            rt[rt.size() - 2],
            1u | (group.rotating() && count > 1 && group.receiver_cores().contains(sender) ? 2u : 0u));
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
            const auto w = device->worker_core_from_logical_core(group.senders()[i]);
            EXPECT_EQ(rt[2 + 2 * i], w.x);
            EXPECT_EQ(rt[3 + 2 * i], w.y);
        }
        EXPECT_EQ(rt, family.runtime_args(sender));
    }
    for (auto core : tt::tt_metal::corerange_to_cores(group.receiver_cores(), std::nullopt, true)) {
        if (family.is_sender(core)) {
            continue;
        }
        const auto rt = family.runtime_args(core);
        EXPECT_EQ(rt[rt.size() - 2], 2u);
        EXPECT_EQ(rt.back(), wire::NO_SENDER_ROUND);
        EXPECT_EQ(family.num_receivers(core), 0u);
        EXPECT_EQ(family.ack_count(core), 0u);
    }
}

template <typename Wrapper>
void check_wrapper(const Wrapper& wrapper, const McastFamily& family) {
    EXPECT_EQ(wrapper.compile_time_args(), family.compile_time_args());
    EXPECT_EQ(wrapper.compile_time_args(false), family.compile_time_args(false));
    EXPECT_EQ(wrapper.num_senders(), family.num_senders());
    EXPECT_EQ(wrapper.num_semaphores(), family.num_semaphores());
    EXPECT_EQ(wrapper.has_remote_receivers(), family.has_remote_receivers());
    EXPECT_EQ(wrapper.ack_count(), family.compile_time_args()[4]);
    std::vector<CoreCoord> participants =
        tt::tt_metal::corerange_to_cores(family.participating_cores(), std::nullopt, true);
    participants.emplace_back(0, 0);  // Outside the offset grids below.
    for (auto core : participants) {
        EXPECT_EQ(wrapper.runtime_args(core), family.runtime_args(core));
        EXPECT_EQ(wrapper.is_sender(core), family.is_sender(core));
        EXPECT_EQ(wrapper.num_receivers(core), family.num_receivers(core));
        std::vector<uint32_t> appended{42};
        wrapper.append_runtime_args_to(appended, core);
        auto expected = family.runtime_args(core);
        expected.insert(expected.begin(), 42);
        EXPECT_EQ(appended, expected);
    }
    struct Appendable {
        std::vector<uint32_t> words;
        void append(const std::vector<uint32_t>& a) { words.insert(words.end(), a.begin(), a.end()); }
    } ct;
    wrapper.append_compile_time_args_to(ct);
    append_absent_mcast_compile_time_args_to(ct);
    family.append_compile_time_args_to(ct);
    auto expected = family.compile_time_args();
    expected.push_back(0);
    const auto tail = family.compile_time_args();
    expected.insert(expected.end(), tail.begin(), tail.end());
    EXPECT_EQ(ct.words, expected);
    const auto sems = wrapper.owned_semaphores();
    ASSERT_EQ(sems.size(), family.owned_semaphores().size());
    for (size_t i = 0; i < sems.size(); ++i) {
        EXPECT_EQ(sems[i].id, family.owned_semaphores()[i].id);
        EXPECT_EQ(sems[i].core_ranges, family.participating_cores());
        EXPECT_EQ(sems[i].initial_value, 0u);
    }
}

TEST(McastHostWire, AbsentFamilyIsOneWord) { EXPECT_EQ(absent_mcast_compile_time_args(), std::vector<uint32_t>{0}); }

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
                cfg.base_sem_id = 2;
                for (auto placement : {Mcast1DSenderPlacement::Uniform, Mcast1DSenderPlacement::Diagonal}) {
                    std::vector<McastGroup> groups;
                    for (uint32_t i = 0; i < lines; ++i) {
                        const uint32_t phase =
                            placement == Mcast1DSenderPlacement::Diagonal ? (span - 1 + i) % span : span - 1;
                        const CoreCoord lo(row ? 2 : 2 + i, row ? 2 + i : 2),
                            hi(row ? end.x : 2 + i, row ? 2 + i : end.y);
                        const CoreCoord sender(row ? 2 + phase : 2 + i, row ? 2 + i : 2 + phase);
                        groups.emplace_back(grid(lo, hi), std::vector<CoreCoord>{sender});
                    }
                    McastFamily family(device_, groups, cfg);
                    Mcast1D wrapper(device_, receivers, shape, Mcast1DFixedSenderConfig{span - 1, placement}, cfg);
                    check_wrapper(wrapper, family);
                    EXPECT_EQ(wrapper.receiver_cores(), receivers);
                    EXPECT_EQ(wrapper.participating_cores(), family.participating_cores());
                    EXPECT_EQ(wrapper.sender_only_cores(), family.sender_only_cores());
                    EXPECT_EQ(wrapper.next_base_sem_id(), 4u);
                    for (auto& group : groups) {
                        check_group(device_, family, group);
                    }
                }
                // Default rotation and explicit aligned senders extending outside the receiver set.
                for (bool outside : {false, true}) {
                    std::vector<CoreCoord> all_senders;
                    std::vector<McastGroup> groups;
                    for (uint32_t i = 0; i < lines; ++i) {
                        std::vector<CoreCoord> senders;
                        for (uint32_t j = 0; j < span + uint32_t(outside); ++j) {
                            senders.emplace_back(row ? 2 + j : 2 + i, row ? 2 + i : 2 + j);
                        }
                        all_senders.insert(all_senders.end(), senders.begin(), senders.end());
                        groups.emplace_back(
                            grid({row ? 2 : 2 + i, row ? 2 + i : 2}, {row ? end.x : 2 + i, row ? 2 + i : end.y}),
                            senders);
                    }
                    McastFamily family(device_, groups, cfg);
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

TEST_F(McastHostFixture, Wrapper2DFixedAndRotatingOrder) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        const auto receivers = grid({2, 2}, {3, 3});
        for (auto sender : {CoreCoord(2, 2), CoreCoord(3, 3), CoreCoord(4, 2)}) {
            McastGroup group(receivers, std::vector<CoreCoord>{sender});
            McastFamily family(device_, {group}, cfg);
            Mcast2D wrapper(device_, receivers, Mcast2DFixedSenderConfig{sender}, cfg);
            check_wrapper(wrapper, family);
            check_group(device_, family, group);
            EXPECT_EQ(wrapper.sender_in_rect(), receivers.contains(sender));
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
                McastGroup group(receivers, senders);
                McastFamily family(device_, {group}, cfg);
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
        McastGroup staircase(cores({{7, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}}), std::vector<CoreCoord>{{7, 0}});
        McastGroup rectangle(grid({3, 3}, {5, 3}), std::vector<CoreCoord>{{3, 3}});
        McastFamily family(device_, {staircase, rectangle}, cfg);
        EXPECT_EQ(family.rectangle_capacity(), 3u);
        EXPECT_EQ(family.num_rectangles({3, 3}), 1u);
        EXPECT_EQ(family.ack_count({7, 0}), 4u);
        EXPECT_EQ(family.ack_count({3, 3}), 2u);
        EXPECT_EQ(family.compile_time_args()[4], ACK_EQUALS_FANOUT);
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
        McastGroup group(receivers, std::vector<CoreCoord>{{0, 0}, {size.x - 1, 1}});
        McastFamily family(device_, {group}, cfg);
        check_group(device_, family, group);
        const auto left = device_->worker_core_from_logical_core({0, 0});
        const auto right = device_->worker_core_from_logical_core({size.x - 1, 0});
        if (right.x - left.x + 1 > size.x) {
            EXPECT_GE(family.rectangle_capacity(), 2u);
        }
    }
}

TEST_F(McastHostFixture, LocalAndNonparticipantRoles) {
    McastGroup local(grid({3, 3}, {3, 3}), std::vector<CoreCoord>{{3, 3}});
    McastFamily family(device_, {local});
    check_group(device_, family, local);
    EXPECT_EQ(family.runtime_args({3, 3})[0], 1u);
    EXPECT_FALSE(family.has_remote_receivers());
    auto outside = family.runtime_args({0, 0});
    EXPECT_EQ(outside.back(), wire::NO_SENDER_ROUND);
    outside.pop_back();
    EXPECT_TRUE(std::all_of(outside.begin(), outside.end(), [](auto v) { return v == 0; }));
    Mcast2D wrapper(device_, grid({3, 3}, {3, 3}), Mcast2DFixedSenderConfig{{3, 3}});
    check_wrapper(wrapper, McastFamily(device_, {local}));
}

TEST_F(McastHostFixture, FlagsSemaphoresAndAckPrecedence) {
    const auto receivers = grid({2, 2}, {4, 2});
    for (auto signal : {DataReadyMode::Flag, DataReadyMode::Counter}) {
        for (bool handshake : {false, true}) {
            for (auto config_ack :
                 {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{1}}) {
                for (auto group_ack :
                     {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2}}) {
                    McastConfig cfg;
                    cfg.data_ready = signal;
                    cfg.handshake = handshake;
                    cfg.ack_count_override = config_ack;
                    cfg.base_sem_id = 4;
                    McastGroup group(receivers, std::vector<CoreCoord>{{2, 2}}, group_ack);
                    McastFamily family(device_, {group}, cfg);
                    EXPECT_EQ(family.ack_count({2, 2}), group_ack.value_or(config_ack.value_or(2)));
                    EXPECT_EQ(
                        family.compile_time_args()[5],
                        uint32_t(handshake) + (signal == DataReadyMode::Counter ? 2u : 0u));
                    EXPECT_EQ(family.compile_time_args(false)[5], signal == DataReadyMode::Counter ? 2u : 0u);
                    EXPECT_EQ(family.num_semaphores(), handshake ? 2u : 1u);
                    EXPECT_EQ(family.next_base_sem_id(), handshake ? 6u : 5u);
                    cfg.sem_ids = std::vector<uint32_t>{6, 7};
                    McastFamily adopted(device_, {group}, cfg);
                    EXPECT_TRUE(adopted.owned_semaphores().empty());
                    EXPECT_EQ(adopted.compile_time_args()[2], 6u);
                    EXPECT_EQ(adopted.compile_time_args()[3], handshake ? 7u : UNUSED_SEM_ID);
                    EXPECT_ANY_THROW(adopted.next_base_sem_id());
                }
            }
        }
    }
}

TEST_F(McastHostFixture, SenderListsAndSingletonWrapperSchedules) {
    const std::vector<CoreCoord> ordered = {{3, 2}, {2, 2}};
    McastGroup rotating(grid({2, 2}, {3, 2}), ordered);
    EXPECT_TRUE(rotating.rotating());
    EXPECT_EQ(rotating.senders(), ordered);
    check_group(device_, McastFamily(device_, {rotating}), rotating);

    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (auto signal : {DataReadyMode::Flag, DataReadyMode::Counter}) {
            McastConfig cfg;
            cfg.noc = noc;
            cfg.data_ready = signal;
            for (auto shape : {Mcast1DShape::PerRow, Mcast1DShape::PerColumn}) {
                const bool row = shape == Mcast1DShape::PerRow;
                const auto receivers = grid({2, 2}, row ? CoreCoord{3, 2} : CoreCoord{2, 3});
                for (auto sender : {CoreCoord{2, 2}, row ? CoreCoord{4, 2} : CoreCoord{2, 4}}) {
                    McastGroup group(receivers, {sender});
                    EXPECT_FALSE(group.rotating());
                    McastFamily family(device_, {group}, cfg);
                    EXPECT_EQ(family.compile_time_args()[6], 0u);
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
    McastGroup fixed(receivers, std::vector<CoreCoord>{{2, 2}});
    McastGroup rotating(grid({2, 3}, {3, 3}), std::vector<CoreCoord>{{2, 3}, {3, 3}});
    McastGroup longer(grid({2, 4}, {3, 4}), std::vector<CoreCoord>{{2, 4}, {3, 4}, {4, 4}});
    // Four isolated mapped destinations cannot be represented within the three-rectangle limit.
    McastGroup fragmented(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), std::vector<CoreCoord>{{0, 0}});
    EXPECT_THROW(McastFamily(device_, {fragmented}), std::exception);
    EXPECT_ANY_THROW(McastGroup(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}}));
    EXPECT_ANY_THROW(McastGroup(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}, {3, 2}}));
    EXPECT_ANY_THROW(McastFamily(device_, {}));
    EXPECT_ANY_THROW(McastFamily(nullptr, {fixed}));
    EXPECT_ANY_THROW(McastGroup(receivers, std::vector<CoreCoord>{}));
    EXPECT_ANY_THROW(McastGroup(receivers, std::vector<CoreCoord>{{2, 2}, {2, 2}}));
    EXPECT_ANY_THROW(McastFamily(device_, {fixed, fixed}));
    EXPECT_ANY_THROW(McastFamily(device_, {fixed, rotating}));
    EXPECT_ANY_THROW(McastFamily(device_, {rotating, longer}));
    // Disjoint receivers are insufficient when groups share a sender.
    EXPECT_ANY_THROW(McastFamily(device_, {fixed, McastGroup(grid({4, 4}, {4, 4}), std::vector<CoreCoord>{{2, 2}})}));
    McastConfig cfg;
    cfg.ack_count_override = 2;
    EXPECT_ANY_THROW(McastFamily(device_, {fixed}, cfg));
    EXPECT_ANY_THROW(Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg));
    EXPECT_ANY_THROW(Mcast2D(device_, receivers, Mcast2DFixedSenderConfig{{2, 2}}, cfg));
    for (auto ids : {std::vector<uint32_t>{}, std::vector<uint32_t>{0}, std::vector<uint32_t>{0, UNUSED_SEM_ID}}) {
        cfg = {};
        cfg.sem_ids = ids;
        EXPECT_ANY_THROW(McastFamily(device_, {fixed}, cfg));
    }
    cfg = {};
    cfg.handshake = false;
    EXPECT_ANY_THROW(McastFamily(device_, {fixed}, cfg).compile_time_args(true));
    EXPECT_ANY_THROW(Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{2}));
    EXPECT_ANY_THROW(
        Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{grid({2, 3}, {3, 3})}));
    EXPECT_ANY_THROW(Mcast2D(device_, receivers, Mcast2DRotatingSenderConfig{CoreRangeSet{}}));
}

TEST_F(McastHostFixture, GroupSerializationAndStrictRouting) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool rotating : {false, true}) {
            McastConfig cfg;
            cfg.noc = noc;
            cfg.data_ready = DataReadyMode::Counter;
            cfg.sem_ids = std::vector<uint32_t>{6, 7};
            cfg.ack_count_override = 0;
            const std::vector<CoreCoord> first = {{2, 2}, {0, 0}, {0, 4}};
            const std::vector<CoreCoord> outside = {{4, 2}, {4, 0}, {6, 4}};
            const std::vector<CoreRangeSet> receivers = {
                grid({2, 2}, {3, 2}), cores({{0, 0}, {2, 0}}), cores({{0, 4}, {2, 4}, {4, 4}})};
            std::vector<McastGroup> groups;
            for (size_t i = 0; i < receivers.size(); ++i) {
                auto senders =
                    rotating ? std::vector<CoreCoord>{outside[i], first[i]} : std::vector<CoreCoord>{first[i]};
                groups.emplace_back(receivers[i], senders, i == 2 ? std::optional<uint32_t>{1} : std::nullopt);
            }
            McastFamily family(device_, groups, cfg);
            ASSERT_EQ(family.rectangle_capacity(), 3u);
            EXPECT_TRUE(family.owned_semaphores().empty());
            for (uint32_t i = 0; i < groups.size(); ++i) {
                const auto& group = family.group(i);
                EXPECT_EQ(group.receiver_cores(), receivers[i]);
                EXPECT_EQ(group.senders(), groups[i].senders());
                EXPECT_EQ(group.num_senders(), rotating ? 2u : 1u);
                EXPECT_EQ(group.num_rectangles(), i + 1u);
                EXPECT_TRUE(group.has_remote_receivers());
                EXPECT_EQ(group.ack_count(first[i]), i == 2 ? 1u : 0u);
                EXPECT_EQ(group.compile_time_args(), family.compile_time_args());
                EXPECT_EQ(group.compile_time_args(false), family.compile_time_args(false));
                EXPECT_EQ(group.compile_time_args(true), family.compile_time_args(true));
                EXPECT_EQ(group.sender_only_cores(), rotating ? cores({outside[i]}) : CoreRangeSet{});
                check_group(device_, family, group);
                struct Appendable {
                    std::vector<uint32_t> words{42};
                    void append(const std::vector<uint32_t>& args) {
                        words.insert(words.end(), args.begin(), args.end());
                    }
                } ct;
                group.append_compile_time_args_to(ct, false);
                auto expected_ct = family.compile_time_args(false);
                expected_ct.insert(expected_ct.begin(), 42);
                EXPECT_EQ(ct.words, expected_ct);
                for (const auto& core :
                     tt::tt_metal::corerange_to_cores(group.participating_cores(), std::nullopt, true)) {
                    const auto rt = group.runtime_args(core);
                    EXPECT_EQ(rt, family.runtime_args(core));
                    EXPECT_EQ(group.is_sender(core), family.is_sender(core));
                    EXPECT_EQ(group.num_receivers(core), family.num_receivers(core));
                    EXPECT_EQ(group.ack_count(core), family.ack_count(core));
                    std::vector<uint32_t> appended{99};
                    group.append_runtime_args_to(appended, core);
                    auto expected_rt = rt;
                    expected_rt.insert(expected_rt.begin(), 99);
                    EXPECT_EQ(appended, expected_rt);
                    const auto start = wire::rectangles_offset(rotating ? 2u : 0u);
                    const auto padding = start + rt[wire::NUM_RECTANGLES] * wire::RECT_WORDS;
                    const auto end = wire::roles_offset(rotating ? 2u : 0u, 3u);
                    EXPECT_TRUE(
                        std::all_of(rt.begin() + padding, rt.begin() + end, [](auto word) { return word == 0; }));
                }
                // Both another group's sender and a core outside the family must be rejected.
                for (const auto& core : {first[(i + 1) % groups.size()], CoreCoord{7, 7}}) {
                    EXPECT_THROW(group.runtime_args(core), std::exception);
                    std::vector<uint32_t> unchanged{123};
                    EXPECT_THROW(group.append_runtime_args_to(unchanged, core), std::exception);
                    EXPECT_EQ(unchanged, std::vector<uint32_t>{123});
                }
            }
            EXPECT_THROW(family.group(groups.size()), std::exception);
            auto inactive = family.runtime_args({7, 7});
            EXPECT_EQ(inactive.size(), family.group(0).runtime_args(first[0]).size());
            EXPECT_EQ(inactive.back(), wire::NO_SENDER_ROUND);
            inactive.pop_back();
            EXPECT_TRUE(std::all_of(inactive.begin(), inactive.end(), [](auto word) { return word == 0; }));
        }
    }
}

TEST_F(McastHostFixture, GroupPreparationAndValueSemantics) {
    McastGroup local(grid({2, 2}, {2, 2}), {{2, 2}});
    McastGroup remote(grid({4, 2}, {5, 2}), {{4, 2}});
    EXPECT_FALSE(local.has_remote_receivers());
    EXPECT_EQ(local.participating_cores(), local.receiver_cores());
    EXPECT_THROW(local.compile_time_args(), std::exception);
    EXPECT_THROW(local.runtime_args({2, 2}), std::exception);
    EXPECT_THROW(local.ack_count({2, 2}), std::exception);
    EXPECT_THROW(local.num_rectangles(), std::exception);
    McastFamily family(device_, {local, remote});
    EXPECT_FALSE(family.group(0).has_remote_receivers());
    EXPECT_TRUE(family.group(1).has_remote_receivers());
    EXPECT_EQ(family.group(0).compile_time_args(), family.group(1).compile_time_args());
    EXPECT_EQ(family.group(0).compile_time_args()[1], 1u);  // Family-wide remote presence.
    EXPECT_EQ(family.group(0).num_receivers({2, 2}), 0u);
    EXPECT_EQ(family.group(0).runtime_args({2, 2}), family.runtime_args({2, 2}));
    EXPECT_THROW(local.compile_time_args(), std::exception);  // Family prepares its owned copies.

    // Copying and moving must not leave groups pointing at the original family's layout.
    auto moved = [&] {
        McastFamily original(device_, {remote});
        auto copy = original;
        return McastFamily(std::move(copy));
    }();
    EXPECT_EQ(moved.group(0).runtime_args({4, 2}), McastFamily(device_, {remote}).runtime_args({4, 2}));
    const auto old_ct = moved.compile_time_args();
    const auto old_rt = moved.runtime_args({4, 2});
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    cfg.handshake = false;
    cfg.data_ready = DataReadyMode::Counter;
    cfg.base_sem_id = 4;
    cfg.ack_count_override = 0;
    McastFamily rebound(device_, {moved.group(0)}, cfg);
    McastFamily expected(device_, {remote}, cfg);
    EXPECT_EQ(rebound.group(0).compile_time_args(), expected.compile_time_args());
    EXPECT_EQ(rebound.group(0).runtime_args({4, 2}), expected.runtime_args({4, 2}));
    EXPECT_EQ(rebound.group(0).ack_count({4, 2}), 0u);
    EXPECT_THROW(rebound.group(0).compile_time_args(true), std::exception);
    EXPECT_EQ(moved.compile_time_args(), old_ct);
    EXPECT_EQ(moved.runtime_args({4, 2}), old_rt);
}

TEST_F(McastHostFixture, GroupNormFactoryEmitsExactDestinations) {
    using namespace tt::tt_metal;
    using Operation = ttnn::prim::GroupNormDeviceOperation;
    const auto available = device_->compute_with_storage_grid_size();
    if (available.x < 9 || available.y < 9) {
        GTEST_SKIP() << "Requires both 7x9 and 9x7 shard grids";
    }
    for (bool wrapped : {false, true}) {
        const uint32_t batches = wrapped ? 7u : 1u;
        const uint32_t capacity = wrapped ? 3u : 1u;
        for (bool column_major : {false, true}) {
            const CoreCoord dimensions = !wrapped ? CoreCoord{3, 3} : column_major ? CoreCoord{9, 7} : CoreCoord{7, 9};
            const auto shard_cores = grid({0, 0}, {dimensions.x - 1, dimensions.y - 1});
            const MemoryConfig memory{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec(
                    shard_cores,
                    std::array<uint32_t, 2>{32, 128},
                    column_major ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR)};
            const TensorSpec spec(
                ttnn::Shape({batches, 1, 288, 128}),
                TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory));
            auto input = ttnn::create_device_tensor(spec, device_);
            const auto logical_core = [&](uint32_t index) {
                return column_major ? CoreCoord{index / dimensions.y, index % dimensions.y}
                                    : CoreCoord{index % dimensions.x, index / dimensions.x};
            };
            for (bool welford : {false, true}) {
                ttnn::prim::GroupNormParams params{
                    .eps = 1e-5f,
                    .num_groups = 16,
                    .output_mem_config = memory,
                    .program_config =
                        ttnn::prim::GroupNormShardedMultiCoreProgramConfig{
                            .compute_with_storage_grid_size = dimensions,
                            .im_data_format = DataType::BFLOAT16,
                            .out_data_format = DataType::BFLOAT16,
                            .inplace = false,
                            .output_layout = Layout::TILE},
                    .compute_kernel_config = ttnn::init_device_compute_kernel_config(device_->arch(), std::nullopt),
                    .use_welford = welford};
                ttnn::prim::GroupNormInputs inputs{.input = input};
                auto factory = Operation::select_program_factory(params, inputs);
                ASSERT_TRUE(std::holds_alternative<Operation::GroupNormShardedProgramFactory>(factory));
                auto output = Operation::create_output_tensors(params, inputs);
                const auto descriptor = std::visit(
                    [&](auto selected) { return decltype(selected)::create_descriptor(params, inputs, output); },
                    factory);
                const auto it =
                    std::find_if(descriptor.kernels.begin(), descriptor.kernels.end(), [](const auto& kernel) {
                        return kernel.kernel_source.find("reader_mcast_sender_unary_sharded_gn_v2.cpp") !=
                               std::string::npos;
                    });
                ASSERT_NE(it, descriptor.kernels.end());
                ASSERT_GE(it->compile_time_args.size(), 11u);
                EXPECT_EQ(it->compile_time_args[it->compile_time_args.size() - 11], wire::FAMILY);
                EXPECT_EQ(it->compile_time_args.back(), capacity);
                ASSERT_EQ(it->runtime_args.size(), batches);
                std::set<uint32_t> counts;
                for (const auto& [sender, args] : it->runtime_args) {
                    uint32_t batch = 0;
                    while (batch < batches && logical_core(batch * 9) != sender) {
                        ++batch;
                    }
                    ASSERT_LT(batch, batches);
                    // Nine contributors' X/Y coordinates precede the family's sender block.
                    constexpr uint32_t rt_base = 18;
                    ASSERT_GE(args.size(), rt_base + wire::roles_offset(0, capacity) + 2u);
                    const uint32_t rectangles = args[rt_base + wire::NUM_RECTANGLES];
                    ASSERT_GE(rectangles, 1u);
                    ASSERT_LE(rectangles, 3u);
                    counts.insert(rectangles);
                    EXPECT_EQ(args[rt_base + wire::ACK], 8u);
                    EXPECT_EQ(args[args.size() - 2], wire::CAN_SEND);
                    Coordinates expected, actual;
                    for (uint32_t i = 0; i < 9; ++i) {
                        const auto worker = device_->worker_core_from_logical_core(logical_core(batch * 9 + i));
                        expected.emplace(worker.x, worker.y);
                        EXPECT_EQ(args[i], worker.x);
                        EXPECT_EQ(args[9 + i], worker.y);
                    }
                    for (uint32_t i = 0; i < rectangles; ++i) {
                        const auto base = rt_base + wire::rectangles_offset(0) + i * wire::RECT_WORDS;
                        const auto sx = args[base + wire::SX], sy = args[base + wire::SY];
                        const auto ex = args[base + wire::EX], ey = args[base + wire::EY];
                        for (uint32_t y = std::min(sy, ey); y <= std::max(sy, ey); ++y) {
                            for (uint32_t x = std::min(sx, ex); x <= std::max(sx, ex); ++x) {
                                EXPECT_TRUE(actual.emplace(x, y).second) << "Overlapping multicast rectangles";
                            }
                        }
                    }
                    EXPECT_EQ(actual, expected) << "Batch " << batch;
                    if (wrapped && batch == 3) {
                        EXPECT_EQ(rectangles, 3u) << "The middle batch must exercise the staircase";
                    }
                }
                if (wrapped) {
                    EXPECT_TRUE(counts.contains(2u));
                    EXPECT_TRUE(counts.contains(3u));
                } else {
                    EXPECT_EQ(counts, std::set<uint32_t>{1u});
                }
            }
        }
    }
}

TEST_F(McastHostFixture, GroupNormUsesExactFamilies) {
    const std::vector<std::vector<CoreCoord>> shapes = {
        {{0, 0}, {1, 0}, {2, 0}}, {{7, 0}, {0, 1}, {1, 1}, {2, 1}}, {{7, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}}, {{0, 0}}};
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig config;
        config.noc = noc;
        config.sem_ids = std::vector<uint32_t>{0, 1};
        for (const auto& shape : shapes) {
            const std::vector<CoreCoord> second = {{4, 3}, {5, 3}};
            const auto family = ttnn::prim::make_group_norm_mcast_family(device_, {shape, second}, config);
            check_group(device_, family, McastGroup(cores(shape), std::vector<CoreCoord>{shape.front()}));
            check_group(device_, family, McastGroup(cores(second), std::vector<CoreCoord>{second.front()}));
            EXPECT_TRUE(family.owned_semaphores().empty());
            EXPECT_EQ(family.compile_time_args(false)[5] & 1, 0u);
            EXPECT_EQ(family.compile_time_args(true)[5] & 1, 1u);
        }
        EXPECT_ANY_THROW(ttnn::prim::make_group_norm_mcast_family(device_, {{}}, config));
    }
}

}  // namespace ttnn::kernel_lib::host::test
