// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host_test_common.hpp"

#include <set>

namespace ttnn::kernel_lib::host::test {
namespace {

struct DecodedMcast {
    uint32_t roles = 0;
    uint32_t phase = wire::NO_SENDER_ROUND;
    uint32_t ack = 0;
    std::vector<dataflow_kernel_lib::RectangleRuntimeArguments> rectangles;
    std::vector<uint32_t> coordinates;
};

DecodedMcast decode_multicast(const wire::ArgumentMetadata& metadata, std::span<const uint32_t> words) {
    const wire::RuntimeLayout layout(metadata);
    EXPECT_EQ(words.size(), layout.words);
    DecodedMcast decoded;
    decoded.roles = layout.roles == wire::OMITTED ? metadata.kernel.roles : words[layout.roles];
    if (decoded.roles & wire::CAN_SEND) {
        decoded.phase = layout.sender_phase == wire::OMITTED ? 0 : words[layout.sender_phase];
        decoded.ack = layout.ack == wire::OMITTED ? metadata.mcast.ack_count : words[layout.ack];
        const uint32_t count = layout.rectangle_count == wire::OMITTED ? 1 : words[layout.rectangle_count];
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t base = layout.rectangles + i * layout.rectangle_stride;
            dataflow_kernel_lib::RectangleRuntimeArguments rectangle{};
            if (layout.rectangle_bounds != wire::OMITTED) {
                const uint32_t bounds = base + layout.rectangle_bounds;
                rectangle.bounds = {words[bounds], words[bounds + 1], words[bounds + 2], words[bounds + 3]};
            }
            rectangle.remote_count = layout.rectangle_remote == wire::OMITTED ? metadata.mcast.uniform_remote_count
                                                                              : words[base + layout.rectangle_remote];
            rectangle.loopback_count = rectangle.remote_count + 1;
            rectangle.sender_mcast_mode = layout.rectangle_mode == wire::OMITTED
                                              ? metadata.mcast.sender_mcast_mode
                                              : static_cast<SenderMcastMode>(words[base + layout.rectangle_mode]);
            decoded.rectangles.push_back(rectangle);
        }
    }
    if ((decoded.roles & wire::CAN_RECEIVE) && layout.sender_coordinates != wire::OMITTED) {
        const uint32_t count = std::max(1u, metadata.mcast.rotating_span);
        const auto payload = words.subspan(layout.sender_coordinates, layout.coordinate_words);
        // Independent expansion: build axis vectors, then walk the CT traversal.
        if (metadata.coordinates.encoding == wire::SenderCoordinateEncoding::ExplicitPairs) {
            decoded.coordinates.assign(payload.begin(), payload.end());
        } else {
            std::vector<uint32_t> xs, ys;
            for (uint32_t range = 0; range < metadata.coordinates.x_ranges + metadata.coordinates.y_ranges; ++range) {
                auto& axis = range < metadata.coordinates.x_ranges ? xs : ys;
                for (uint32_t value = payload[2 * range]; value <= payload[2 * range + 1]; ++value) {
                    axis.push_back(value);
                }
            }
            for (uint32_t phase = 0; phase < count; ++phase) {
                const bool rows = metadata.coordinates.encoding == wire::SenderCoordinateEncoding::RowMajorRanges;
                decoded.coordinates.push_back(xs.at(rows ? phase % xs.size() : phase / ys.size()));
                decoded.coordinates.push_back(ys.at(rows ? phase / xs.size() : phase % ys.size()));
            }
        }
    }
    return decoded;
}

template <typename McastType>
DecodedMcast decoded_args(const McastType& mcast, CoreCoord core, const std::vector<uint32_t>& existing = {}) {
    return decode_multicast(emitted_metadata(compile_args(mcast, existing)), runtime_args(mcast, core, existing));
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
template <typename McastType>
void check_group(
    tt::tt_metal::IDevice* device,
    const McastType& mcast,
    const GroupInput& group,
    const std::vector<uint32_t>& existing = {}) {
    const auto workers = worker_coordinates(device);
    const auto ct = compile_args(mcast, existing);
    ASSERT_EQ(ct.size(), decode_emitted_ct(ct).words);
    ASSERT_EQ(ct[0] & 15u, 3u);
    const uint32_t count = group.senders.size();
    EXPECT_EQ(emitted_metadata(ct).mcast.rotating_span, (count > 1) ? count : 0u);
    Coordinates expected;
    for (auto c : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        auto w = device->worker_core_from_logical_core(c);
        expected.emplace(w.x, w.y);
    }
    for (uint32_t phase = 0; phase < count; ++phase) {
        const auto sender = group.senders[phase];
        const auto worker = device->worker_core_from_logical_core(sender);
        const auto rt = runtime_args(mcast, sender, existing);
        const auto decoded = decode_multicast(emitted_metadata(ct), rt);
        EXPECT_EQ(decoded.phase, phase);
        EXPECT_EQ(decoded.roles, 1u | (count > 1 && group.receivers.contains(sender) ? 2u : 0u));
        Coordinates actual;
        uint32_t remote_total = 0;
        for (const auto& rectangle : decoded.rectangles) {
            if (uint32_t(emitted_metadata(ct).mcast.sender_mcast_mode) == 1) {
                EXPECT_EQ(rectangle.remote_count, 0u);
                EXPECT_EQ(rectangle.loopback_count, 1u);
                actual.emplace(worker.x, worker.y);
                continue;
            }
            const auto& r = rectangle.bounds;
            const auto xlo = std::min(r.sx, r.ex), xhi = std::max(r.sx, r.ex);
            const auto ylo = std::min(r.sy, r.ey), yhi = std::max(r.sy, r.ey);
            EXPECT_EQ(r.sx, (emitted_metadata(ct).mcast.flags & 4) ? xhi : xlo);
            EXPECT_EQ(r.sy, (emitted_metadata(ct).mcast.flags & 4) ? yhi : ylo);
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
            EXPECT_EQ(rectangle.remote_count, remote);
            EXPECT_EQ(rectangle.loopback_count, remote + 1u);
            EXPECT_EQ(uint32_t(rectangle.sender_mcast_mode), remote == 0 ? 1u : inside ? 3u : 2u);
            if (uint32_t(emitted_metadata(ct).mcast.sender_mcast_mode) != 4) {
                EXPECT_EQ(
                    uint32_t(rectangle.sender_mcast_mode), uint32_t(emitted_metadata(ct).mcast.sender_mcast_mode));
            }
            remote_total += remote;
        }
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(remote_total, expected.size() - expected.count({worker.x, worker.y}));
        for (uint32_t i = 0; i < decoded.coordinates.size() / 2; ++i) {
            const auto w = device->worker_core_from_logical_core(group.senders[i]);
            EXPECT_EQ(decoded.coordinates[2 * i], w.x);
            EXPECT_EQ(decoded.coordinates[2 * i + 1], w.y);
        }
        EXPECT_EQ(rt, runtime_args(mcast, sender, existing));
    }
    for (auto core : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        if (std::find(group.senders.begin(), group.senders.end(), core) != group.senders.end()) {
            continue;
        }
        const auto decoded = decoded_args(mcast, core, existing);
        EXPECT_EQ(decoded.roles, 2u);
        EXPECT_EQ(decoded.phase, wire::NO_SENDER_ROUND);
        EXPECT_EQ(decoded.ack, 0u);
        for (uint32_t i = 0; i < count; ++i) {
            const auto mapped = device->worker_core_from_logical_core(group.senders[i]);
            EXPECT_EQ(decoded.coordinates[2 * i], mapped.x);
            EXPECT_EQ(decoded.coordinates[2 * i + 1], mapped.y);
        }
    }
}

}  // namespace

TEST(McastHostWire, AbsentMcastIsOneWord) {
    std::vector<uint32_t> args{17};
    append_absent_mcast_compile_time_args_to(args);
    EXPECT_EQ(args, (std::vector<uint32_t>{17, 0}));
}

constexpr wire::ArgumentMetadata fixed_metadata(uint32_t roles) {
    return {
        .mcast =
            {.rectangle_capacity = 1,
             .ack_count = 7,
             .uniform_remote_count = 7,
             .remote_count_known = true,
             .sender_mcast_mode = SenderMcastMode::MulticastIncludeSource,
             .has_remote_receivers = true,
             .flags = 1},
        .kernel = {.roles = roles, .capabilities = roles}};
}

TEST(McastHostWire, CompactCompileTimeCountsAndFullWidthValues) {
    constexpr auto sender = fixed_metadata(1);
    constexpr uint32_t control = wire::compile_time_control(sender);
    static_assert(control == 0x0244AE13u);
    static_assert(wire::CompileTimeLayout(control).words == 4);         // + two named offsets = 6.
    static_assert(wire::CompileTimeLayout(control, false).words == 2);  // Native bindings, no numeric IDs.
    constexpr std::array<uint32_t, 4> literal{0x0244AE13u, 11, 13, 7};
    constexpr auto decoded = wire::decode_compile_time_metadata(literal);
    static_assert(decoded.mcast.ack_count == 7 && decoded.mcast.uniform_remote_count == 7);
    static_assert(decoded.kernel.roles == 1 && decoded.kernel.capabilities == 1);
    EXPECT_EQ(decode_emitted_ct(std::vector<uint32_t>(literal.begin(), literal.end())).words, 4u);
    EXPECT_FALSE(wire::valid_compile_time_control(1));
    EXPECT_FALSE(wire::valid_compile_time_control(2));
    constexpr std::array<uint32_t, 1> absent{0};
    static_assert(wire::CompileTimeLayout(0).words == 1);
    static_assert(wire::decode_compile_time_metadata(absent).mcast.rotating_span == 0);

    auto receiver = fixed_metadata(2);
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(receiver)).words + 2, 5u);
    receiver.mcast.flags = 0;
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(receiver)).words + 2, 4u);
    auto custom_ack = sender;
    custom_ack.mcast.ack_count = 0;  // Known zero must remain a separate constant, not mean "use fanout".
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(custom_ack)).words + 2, 7u);

    auto large = sender;
    large.kernel = {.roles = wire::DYNAMIC_ROLES, .capabilities = 3};
    large.mcast.uniform_remote_count = 0x12345678;
    large.mcast.ack_count = 0x01234567;
    large.mcast.rotating_span = 0x10001;
    large.coordinates = {wire::SenderCoordinateEncoding::ColumnMajorRanges, 0x10002, 0x10003, 0x10004, 0x10005};
    for (const bool ids : {false, true}) {
        const wire::CompileTimeLayout layout(wire::compile_time_control(large), ids);
        std::vector<uint32_t> words(layout.words);
        wire::encode_compile_time_metadata(words, large, ids);
        const auto result = wire::decode_compile_time_metadata(words, ids);
        const auto independent = decode_emitted_ct(words, 0, ids);
        EXPECT_EQ(independent.words, words.size());
        EXPECT_EQ(result.mcast.uniform_remote_count, large.mcast.uniform_remote_count);
        EXPECT_EQ(result.mcast.ack_count, large.mcast.ack_count);
        EXPECT_EQ(result.mcast.rotating_span, large.mcast.rotating_span);
        EXPECT_EQ(result.coordinates.columns, large.coordinates.columns);
        EXPECT_EQ(result.coordinates.rows, large.coordinates.rows);
        EXPECT_EQ(result.coordinates.x_ranges, large.coordinates.x_ranges);
        EXPECT_EQ(result.coordinates.y_ranges, large.coordinates.y_ranges);
        EXPECT_EQ(independent.metadata.mcast.rotating_span, large.mcast.rotating_span);
        EXPECT_EQ(independent.metadata.coordinates.y_ranges, large.coordinates.y_ranges);
    }
}

TEST(McastHostWire, CompactCompileTimePreservesEveryRuntimeOffset) {
    const auto offsets = [](const wire::ArgumentMetadata& metadata) {
        const wire::RuntimeLayout r(metadata);
        return std::array{
            r.roles,
            r.sender_phase,
            r.rectangle_count,
            r.ack,
            r.sender_coordinates,
            r.coordinate_words,
            r.rectangles,
            r.rectangle_bounds,
            r.rectangle_remote,
            r.rectangle_mode,
            r.rectangle_stride,
            r.chain_neighbors,
            r.words};
    };
    const auto verify = [&](const wire::ArgumentMetadata& metadata) {
        for (const bool ids : {false, true}) {
            const wire::CompileTimeLayout layout(wire::compile_time_control(metadata), ids);
            std::vector<uint32_t> words(layout.words);
            wire::encode_compile_time_metadata(words, metadata, ids);
            const auto decoded = wire::decode_compile_time_metadata(words, ids);
            EXPECT_EQ(offsets(decoded), offsets(metadata));
            EXPECT_EQ(offsets(decode_emitted_ct(words, 0, ids).metadata), offsets(metadata));
            std::vector<uint32_t> again(layout.words);
            wire::encode_compile_time_metadata(again, decoded, ids);
            EXPECT_EQ(again, words);
        }
    };
    for (const uint32_t flags : {0u, 1u, 3u, 5u, 7u}) {
        for (const auto kernel :
             {wire::KernelMetadata{0, 0},
              wire::KernelMetadata{1, 1},
              wire::KernelMetadata{2, 2},
              wire::KernelMetadata{3, 3},
              wire::KernelMetadata{wire::DYNAMIC_ROLES, 1},
              wire::KernelMetadata{wire::DYNAMIC_ROLES, 2},
              wire::KernelMetadata{}}) {
            for (const uint32_t rectangles : {1u, 2u, 3u}) {
                for (const uint32_t ack : {0u, 7u, ACK_EQUALS_FANOUT}) {
                    for (const bool known : {false, true}) {
                        for (const auto mode :
                             {SenderMcastMode::LocalCopy,
                              SenderMcastMode::MulticastExcludeSource,
                              SenderMcastMode::MulticastIncludeSource,
                              SenderMcastMode::Unknown}) {
                            auto metadata = fixed_metadata(1);
                            metadata.kernel = kernel;
                            metadata.mcast.flags = flags;
                            metadata.mcast.rectangle_capacity = rectangles;
                            metadata.mcast.ack_count = ack;
                            metadata.mcast.remote_count_known = known;
                            metadata.mcast.sender_mcast_mode = mode;
                            verify(metadata);
                            metadata.mcast.rotating_span = 61;
                            verify(metadata);  // Explicit coordinate fallback.
                            metadata.coordinates = {wire::SenderCoordinateEncoding::RowMajorRanges, 8, 8, 2, 1};
                            verify(metadata);  // Partial traversal, including a gap in mapped X coordinates.
                            metadata.coordinates.encoding = wire::SenderCoordinateEncoding::ColumnMajorRanges;
                            verify(metadata);
                        }
                    }
                }
            }
        }
    }
    auto chain = fixed_metadata(3);
    chain.mcast.flags = 9;
    chain.mcast.rectangle_capacity = 0;
    chain.mcast.sender_mcast_mode = SenderMcastMode::Unknown;
    verify(chain);
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(chain)).words + 2, 6u);
}

TEST(McastHostWire, CompactOffsetsAreConstexprAndKnownZeroIsDistinct) {
    constexpr wire::RuntimeLayout sender(fixed_metadata(1)), receiver(fixed_metadata(2)), idle(fixed_metadata(0));
    static_assert(sender.words == 4 && sender.rectangles == 0 && sender.rectangle_stride == 4);
    static_assert(sender.roles == wire::OMITTED && sender.sender_phase == wire::OMITTED);
    static_assert(sender.sender_coordinates == wire::OMITTED && sender.rectangle_remote == wire::OMITTED);
    static_assert(receiver.words == 2 && receiver.sender_coordinates == 0 && receiver.rectangles == wire::OMITTED);
    static_assert(idle.words == 0);
    static_assert(wire::CompileTimeLayout(wire::compile_time_control(fixed_metadata(1))).words == 4);
    static_assert(wire::CompileTimeLayout(wire::compile_time_control(fixed_metadata(2))).words == 3);

    auto local = fixed_metadata(1);
    local.mcast.sender_mcast_mode = SenderMcastMode::LocalCopy;
    local.mcast.uniform_remote_count = 0;
    local.mcast.ack_count = 0;
    EXPECT_EQ(wire::RuntimeLayout(local).words, 0u);
    local.mcast.remote_count_known = false;
    EXPECT_EQ(wire::RuntimeLayout(local).words, 1u);
    auto multiple = fixed_metadata(1);
    multiple.mcast.rectangle_capacity = 2;
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 11u);
    EXPECT_EQ(wire::RuntimeLayout(multiple).rectangle_remote, 4u);  // Equal total fanout is not per-rectangle fanout.
    multiple.mcast.ack_count = 0xFFFFFFFFu;
    multiple.mcast.sender_mcast_mode = SenderMcastMode::Unknown;
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 14u);
    multiple.mcast.flags = 0;  // No ACK field without handshakes.
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 13u);
    multiple.mcast.flags = 9;
    multiple.mcast.rectangle_capacity = 0;
    const wire::RuntimeLayout chain(multiple);
    EXPECT_EQ(chain.words, 5u);
    EXPECT_EQ(chain.chain_neighbors, 0u);
    EXPECT_EQ(chain.roles, wire::OMITTED);
    multiple.kernel = {.roles = 2, .capabilities = 2};
    const wire::RuntimeLayout chain_receiver(multiple);
    EXPECT_EQ(chain_receiver.words, 7u);
    EXPECT_EQ(chain_receiver.sender_coordinates, 0u);
    EXPECT_EQ(chain_receiver.chain_neighbors, 2u);
    multiple.kernel = {.roles = wire::DYNAMIC_ROLES, .capabilities = 3};
    const wire::RuntimeLayout chain_mixed(multiple);
    EXPECT_EQ(chain_mixed.words, 8u);
    EXPECT_EQ(chain_mixed.roles, 0u);
    EXPECT_EQ(chain_mixed.sender_coordinates, 1u);
    EXPECT_EQ(chain_mixed.chain_neighbors, 3u);
}

TEST(McastHostWire, CoordinateRangesPreserveGapsOrdersAndPartialTraversal) {
    // Independent literal payload: X=3..6,8..11; Y=13..20. 8x8 with one gap.
    constexpr std::array<uint32_t, 6> ranges{3, 6, 8, 11, 13, 20};
    constexpr wire::SenderCoordinateMetadata rows{wire::SenderCoordinateEncoding::RowMajorRanges, 8, 8, 2, 1};
    constexpr wire::SenderCoordinateMetadata columns{wire::SenderCoordinateEncoding::ColumnMajorRanges, 8, 8, 2, 1};
    static_assert(wire::sender_coordinate(ranges, rows, 4, 0) == 8);
    static_assert(wire::sender_coordinate(ranges, columns, 4, 1) == 17);
    for (const auto metadata : {rows, columns}) {
        for (uint32_t count : {1u, 8u, 61u, 64u}) {
            for (uint32_t phase = 0; phase < count; ++phase) {
                const uint32_t x = metadata.encoding == rows.encoding ? phase % 8 : phase / 8;
                const uint32_t y = metadata.encoding == rows.encoding ? phase / 8 : phase % 8;
                EXPECT_EQ(wire::sender_coordinate(ranges, metadata, phase, 0), 3u + x + (x >= 4));
                EXPECT_EQ(wire::sender_coordinate(ranges, metadata, phase, 1), 13u + y);
            }
        }
    }
    auto metadata = fixed_metadata(2);
    metadata.mcast.rotating_span = 64;
    metadata.coordinates = rows;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 6u);
    metadata.coordinates.x_ranges = 1;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 4u);
    metadata.mcast.rotating_span = 8;
    metadata.coordinates.rows = 1;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 4u);
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
    McastImpl mcast(*device_, cfg);
    const std::vector<CoreCoord> senders{{6, 1}, {1, 6}};
    mcast.add_group(grid({2, 3}, {4, 5}), senders, 5);
    const std::vector<uint32_t> expected_ct{0x01CE2A73u, 0, 1, 9, 5, 2};
    EXPECT_EQ(compile_args(mcast), expected_ct);
    cfg.handshake = false;
    auto passive = make_mcast(device_, {GroupInput(grid({2, 3}, {4, 5}), senders, 5)}, cfg);
    const std::vector<uint32_t> face_ct{0x00CE2A63u, 0, 9, 2};
    EXPECT_EQ(compile_args(passive), face_ct);
    auto mapped = [&](CoreCoord core) { return device_->worker_core_from_logical_core(core); };
    const auto a = mapped(senders[0]), b = mapped(senders[1]);
    const auto lo = mapped({2, 3}), hi = mapped({4, 5});
    for (uint32_t phase = 0; phase < senders.size(); ++phase) {
        const std::vector<uint32_t> expected_rt{
            1,
            phase,
            uint32_t(a.x),
            uint32_t(a.y),
            uint32_t(b.x),
            uint32_t(b.y),
            uint32_t(hi.x),
            uint32_t(hi.y),
            uint32_t(lo.x),
            uint32_t(lo.y)};
        EXPECT_EQ(runtime_args(mcast, senders[phase]), expected_rt);
    }
    const std::vector<uint32_t> receiver_rt{
        2, 0xFFFFFFFFu, uint32_t(a.x), uint32_t(a.y), uint32_t(b.x), uint32_t(b.y), 0, 0, 0, 0};
    EXPECT_EQ(runtime_args(mcast, {3, 4}), receiver_rt);
    std::vector<uint32_t> inactive(10, 0);
    inactive[1] = 0xFFFFFFFFu;
    EXPECT_EQ(runtime_args(mcast, {0, 0}, {11, 13}), inactive);
    const auto sender_only = mcast.sender_only_cores();
    EXPECT_EQ(sender_only.num_cores(), senders.size());
    for (const auto& sender : senders) {
        EXPECT_TRUE(sender_only.contains(sender));
    }
}

TEST_F(McastHostFixture, CompactCoordinatesMatchEveryMappedSenderAndFallbackPerPlacement) {
    if (device_->compute_with_storage_grid_size().x < 9 || device_->compute_with_storage_grid_size().y < 9) {
        GTEST_SKIP() << "Coordinate matrix requires a 9x9 worker grid";
    }
    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (const bool column_major : {false, true}) {
            for (const uint32_t count : {8u, 61u, 64u}) {
                std::vector<CoreCoord> senders;
                senders.reserve(count);
                for (uint32_t index = 0; index < count; ++index) {
                    senders.emplace_back(
                        1 + (column_major ? index / 8 : index % 8), 1 + (column_major ? index % 8 : index / 8));
                }
                const GroupInput input(grid({1, 1}, {8, 8}), senders);
                auto mcast = make_mcast(device_, {input}, {.noc = noc});
                const auto ct = compile_args(mcast);
                ASSERT_NE(
                    uint32_t(emitted_metadata(ct).coordinates.encoding),
                    0u);  // Regular mapped grids/lines save words even across worker-coordinate gaps.
                const auto metadata = emitted_metadata(ct);
                for (uint32_t phase = 0; phase < count; ++phase) {
                    const auto decoded = decoded_args(mcast, senders[phase]);
                    EXPECT_EQ(decoded.phase, phase);
                    ASSERT_EQ(decoded.coordinates.size(), 2 * count);
                    for (uint32_t expected_phase = 0; expected_phase < count; ++expected_phase) {
                        const auto mapped = device_->worker_core_from_logical_core(senders[expected_phase]);
                        EXPECT_EQ(decoded.coordinates[2 * expected_phase], mapped.x);
                        EXPECT_EQ(decoded.coordinates[2 * expected_phase + 1], mapped.y);
                    }
                }
                EXPECT_LT(wire::RuntimeLayout(metadata).coordinate_words, 2u * count);
                EXPECT_EQ(metadata.coordinates.columns * metadata.coordinates.rows, count == 61 ? 64u : count);
            }
        }
        // Distinct compatible widths and traversal dimensions may not share one
        // CT description. Each separately placed kernel can still compress.
        const GroupInput line(grid({0, 0}, {3, 0}), {{0, 0}, {1, 0}, {2, 0}, {3, 0}});
        const GroupInput square(grid({0, 2}, {1, 3}), {{0, 2}, {1, 2}, {0, 3}, {1, 3}});
        auto mixed = make_mcast(device_, {line, square}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(mixed)).coordinates.encoding), 0u);
        for (const auto& input : {line, square}) {
            tt::tt_metal::ProgramDescriptor descriptor;
            tt::tt_metal::KernelDescriptor kernel;
            kernel.core_ranges = input.receivers;
            kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc};
            mixed.attach(descriptor, "channel", std::array{std::ref(kernel)}, 0);
            EXPECT_NE(uint32_t(emitted_metadata(kernel.compile_time_args).coordinates.encoding), 0u);
            EXPECT_EQ(wire::RuntimeLayout(emitted_metadata(kernel.compile_time_args)).coordinate_words, 4u);
        }
        const GroupInput custom(grid({0, 0}, {2, 1}), {{0, 0}, {1, 1}, {2, 0}, {0, 1}});
        auto sparse = make_mcast(device_, {custom}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(sparse)).coordinates.encoding), 0u);
        check_group(device_, sparse, custom);
        const GroupInput external(grid({0, 4}, {3, 4}), line.senders);
        auto outside = make_mcast(device_, {external}, {.noc = noc});
        EXPECT_NE(uint32_t(emitted_metadata(compile_args(outside)).coordinates.encoding), 0u);
        check_group(device_, outside, external);
        const GroupInput external_custom(external.receivers, custom.senders);
        auto outside_custom = make_mcast(device_, {external_custom}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(outside_custom)).coordinates.encoding), 0u);
        check_group(device_, outside_custom, external_custom);

        std::vector<CoreCoord> eight_senders;
        eight_senders.reserve(8);
        for (uint32_t x = 0; x < 8; ++x) {
            eight_senders.emplace_back(x, 0);
        }
        const GroupInput split_receivers(
            CoreRangeSet(std::set{CoreRange({0, 0}, {7, 0}), CoreRange({0, 2}, {3, 2})}), eight_senders);
        auto split = make_mcast(device_, {split_receivers}, {.noc = noc});
        const auto split_metadata = emitted_metadata(compile_args(split));
        EXPECT_NE(split_metadata.coordinates.encoding, wire::SenderCoordinateEncoding::ExplicitPairs);
        EXPECT_EQ(split_metadata.mcast.rectangle_capacity, 2u);
        EXPECT_EQ(split_metadata.mcast.sender_mcast_mode, SenderMcastMode::Unknown);
        const wire::RuntimeLayout split_layout(split_metadata);
        EXPECT_EQ(split_layout.rectangle_stride, 6u);    // Bounds, per-rectangle remote, mode.
        EXPECT_EQ(split_layout.sender_coordinates, 3u);  // Roles, rotating phase, rectangle count.
        check_group(device_, split, split_receivers);

        for (const bool compatible : {false, true}) {
            const GroupInput external_group(
                grid({0, 4}, {3, 4}),
                compatible ? std::vector<CoreCoord>{{0, 6}, {1, 6}, {2, 6}, {3, 6}}
                           : std::vector<CoreCoord>{{0, 6}, {1, 6}, {0, 7}, {1, 7}});
            auto receiver_and_sender_groups = make_mcast(device_, {line, external_group}, {.noc = noc});
            tt::tt_metal::ProgramDescriptor descriptor;
            tt::tt_metal::KernelDescriptor kernel;
            kernel.core_ranges = line.receivers.merge(cores(external_group.senders));
            kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc};
            receiver_and_sender_groups.attach(descriptor, "channel", std::array{std::ref(kernel)}, 0);
            const auto metadata = emitted_metadata(kernel.compile_time_args);
            EXPECT_EQ(metadata.coordinates.encoding != wire::SenderCoordinateEncoding::ExplicitPairs, compatible);
            const wire::RuntimeLayout layout(metadata);
            // A mixed placement must carry the correct schedule even on its
            // sender-only cores, for both compatible and fallback encodings.
            for (const auto& [core, args] : kernel.runtime_args) {
                const auto& schedule = core.y >= 6 ? external_group.senders : line.senders;
                for (uint32_t phase = 0; phase < schedule.size(); ++phase) {
                    const auto mapped = device_->worker_core_from_logical_core(schedule[phase]);
                    const auto* coordinates = args.data() + layout.sender_coordinates;
                    EXPECT_EQ(
                        wire::sender_coordinate(coordinates, metadata.coordinates, phase, wire::SENDER_X), mapped.x);
                    EXPECT_EQ(
                        wire::sender_coordinate(coordinates, metadata.coordinates, phase, wire::SENDER_Y), mapped.y);
                }
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
        auto mcast = make_mcast(device_, {staircase, rectangle}, cfg);
        EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 3u);
        EXPECT_EQ(decoded_args(mcast, {3, 3}).rectangles.size(), 1u);
        EXPECT_EQ(decoded_args(mcast, {7, 0}).ack, 4u);
        EXPECT_EQ(decoded_args(mcast, {3, 3}).ack, 2u);
        EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.ack_count, ACK_EQUALS_FANOUT);
        check_group(device_, mcast, staircase);
        check_group(device_, mcast, rectangle);
    }
}

TEST_F(McastHostFixture, FullWidthMappedCoverage) {
    const auto size = device_->compute_with_storage_grid_size();
    const auto receivers = grid({0, 0}, {size.x - 1, 1});
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        GroupInput group(receivers, std::vector<CoreCoord>{{0, 0}, {size.x - 1, 1}});
        auto mcast = make_mcast(device_, {group}, cfg);
        check_group(device_, mcast, group);
        EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 1u);
    }
}

TEST_F(McastHostFixture, CompactConv3dGroupsUseLogicalRectangles) {
    const auto size = device_->compute_with_storage_grid_size();
    if (size.x < 11 || size.y < 9) {
        GTEST_SKIP() << "Requires the compact Conv3D 11-column placement";
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
        auto multicast = make_mcast(device_, groups, cfg);
        auto chain = make_mcast(device_, groups, chain_config(cfg));
        EXPECT_EQ(emitted_metadata(compile_args(multicast)).mcast.rectangle_capacity, 3u);
        EXPECT_EQ(emitted_metadata(compile_args(chain)).mcast.rectangle_capacity, 0u);
        const wire::RuntimeLayout chain_layout(emitted_metadata(compile_args(chain)));
        for (const auto& group : groups) {
            check_group(device_, multicast, group);
            EXPECT_NE(
                runtime_args(chain, group.senders.front())[chain_layout.chain_neighbors + wire::SUCCESSOR_X],
                dataflow_kernel_lib::NO_CHAIN_NEIGHBOR);
        }
    }
}

TEST_F(McastHostFixture, LocalAndNonparticipantRoles) {
    GroupInput local(grid({3, 3}, {3, 3}), std::vector<CoreCoord>{{3, 3}});
    auto mcast = make_mcast(device_, {local});
    check_group(device_, mcast, local);
    EXPECT_EQ(runtime_args(mcast, {3, 3})[0], 1u);
    EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.has_remote_receivers, 0u);
    auto outside = runtime_args(mcast, {0, 0});
    EXPECT_EQ(outside.size(), 3u);  // Dynamic role + fixed coordinate pair; no local-copy rectangle fields.
    EXPECT_TRUE(std::all_of(outside.begin(), outside.end(), [](auto v) { return v == 0; }));
}

TEST_F(McastHostFixture, IrregularReceiverSetPolicySelectsTransport) {
    const GroupInput dense(grid({0, 0}, {2, 0}), {{1, 0}});
    const GroupInput irregular(cores({{0, 2}, {2, 2}, {4, 2}}), {{2, 2}});
    const GroupInput local(cores({{6, 0}}), {{6, 0}});
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig config;
        config.noc = noc;
        auto hardware = make_mcast(device_, {dense}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(hardware)).mcast.flags), TransferMode::Multicast);
        check_group(device_, hardware, dense);
        auto chain = make_mcast(device_, {irregular}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(chain)).mcast.flags), TransferMode::ChainUnicast);
        auto rectangles = make_mcast(device_, {dense, local}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(rectangles)).mcast.flags), TransferMode::Multicast);
        for (const auto& groups :
             {std::vector<GroupInput>{dense, irregular, local}, std::vector<GroupInput>{irregular, local, dense}}) {
            auto mcast = make_mcast(device_, groups, chain_config(config));
            auto multiple_mcast = make_mcast(device_, groups, config);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(compile_args(mcast)).mcast.flags), TransferMode::ChainUnicast);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(compile_args(multiple_mcast)).mcast.flags),
                TransferMode::Multicast);
            EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 0u);
            EXPECT_EQ(emitted_metadata(compile_args(multiple_mcast)).mcast.rectangle_capacity, 3u);
            for (auto core : tt::tt_metal::corerange_to_cores(mcast.participating_cores())) {
                EXPECT_EQ(runtime_args(mcast, core).size(), 8u);
            }
            EXPECT_EQ(
                runtime_args(multiple_mcast, {7, 7}).size(), 23u);  // Roles, count, ACK, coords, 3 * 6 rectangle words.
        }
    }
}

TEST_F(McastHostFixture, FlagsSemaphoresAndAckPrecedence) {
    const auto receivers = grid({2, 2}, {4, 2});
    for (auto signal : {dataflow_kernel_lib::DataReadySignal::Flag, dataflow_kernel_lib::DataReadySignal::Counter}) {
        for (bool handshake : {false, true}) {
            for (auto group_ack : {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2}}) {
                McastConfig cfg;
                cfg.data_ready = signal;
                cfg.handshake = handshake;
                GroupInput group(receivers, std::vector<CoreCoord>{{2, 2}}, group_ack);
                auto mcast = make_mcast(device_, {group}, cfg);
                EXPECT_EQ(decoded_args(mcast, {2, 2}).ack, handshake ? group_ack.value_or(2) : 0u);
                EXPECT_EQ(
                    emitted_metadata(compile_args(mcast)).mcast.flags,
                    uint32_t(handshake) + (signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u));
                auto passive_cfg = cfg;
                passive_cfg.handshake = false;
                EXPECT_EQ(
                    emitted_metadata(compile_args(make_mcast(device_, {group}, passive_cfg))).mcast.flags,
                    signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u);
                EXPECT_EQ(allocated_semaphores(mcast).size(), handshake ? 2u : 1u);
                const auto owned = allocated_semaphores(mcast);
                ASSERT_FALSE(owned.empty());
                EXPECT_EQ(owned.back().id + 1, handshake ? 2u : 1u);
            }
        }
    }
}

TEST_F(McastHostFixture, SenderListsPreserveOrder) {
    const std::vector<CoreCoord> ordered = {{3, 2}, {2, 2}};
    GroupInput rotating(grid({2, 2}, {3, 2}), ordered);
    check_group(device_, make_mcast(device_, {rotating}), rotating);
}

TEST_F(McastHostFixture, McastSerializationAndRouting) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool rotating : {false, true}) {
            McastConfig cfg;
            cfg.noc = noc;
            cfg.data_ready = dataflow_kernel_lib::DataReadySignal::Counter;
            const std::vector<CoreCoord> first = {{2, 2}, {0, 0}, {0, 4}};
            const std::vector<CoreCoord> outside = {{4, 2}, {4, 0}, {6, 4}};
            const std::vector<CoreRangeSet> receivers = {
                grid({2, 2}, {3, 2}), cores({{0, 0}, {2, 0}}), cores({{0, 4}, {2, 4}, {4, 4}})};
            std::vector<GroupInput> groups;
            for (size_t i = 0; i < receivers.size(); ++i) {
                auto senders =
                    rotating ? std::vector<CoreCoord>{outside[i], first[i]} : std::vector<CoreCoord>{first[i]};
                groups.emplace_back(receivers[i], senders, i == 2 ? 1u : 0u);
            }
            auto mcast = make_mcast(device_, groups, cfg);
            ASSERT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 3u);
            EXPECT_EQ(allocated_semaphores(mcast).size(), 2u);
            for (size_t i = 0; i < groups.size(); ++i) {
                check_group(device_, mcast, groups[i]);
                EXPECT_EQ(decoded_args(mcast, first[i]).rectangles.size(), i + 1u);
                EXPECT_EQ(decoded_args(mcast, first[i]).ack, i == 2 ? 1u : 0u);
                std::vector<uint32_t> appended{99};
                detail::append_args_to(appended, runtime_args(mcast, first[i]));
                auto expected = runtime_args(mcast, first[i]);
                expected.insert(expected.begin(), 99);
                EXPECT_EQ(appended, expected);
            }
            EXPECT_EQ(mcast.sender_only_cores(), rotating ? cores(outside) : CoreRangeSet{});
            std::vector<uint32_t> appended{42};
            detail::append_args_to(appended, compile_args(mcast));
            auto expected = compile_args(mcast);
            expected.insert(expected.begin(), 42);
            EXPECT_EQ(appended, expected);
            auto inactive = runtime_args(mcast, {7, 7});
            EXPECT_EQ(inactive.size(), runtime_args(mcast, first[0]).size());
            if (rotating) {
                EXPECT_EQ(inactive[1], wire::NO_SENDER_ROUND);
                inactive[1] = 0;
            }
            EXPECT_TRUE(std::all_of(inactive.begin(), inactive.end(), [](auto word) { return word == 0; }));
        }
    }
}

TEST_F(McastHostFixture, PublicGroupingUsesExplicitGroupSizeAndOrder) {
    struct Case {
        CoreRangeSet receivers;
        uint32_t group_size;
        McastCoreOrder order;
        std::vector<std::vector<CoreCoord>> groups;
    };
    const std::vector<Case> cases = {
        {grid({0, 0}, {5, 0}), 3, McastCoreOrder::RowMajor, {{{0, 0}, {1, 0}, {2, 0}}, {{3, 0}, {4, 0}, {5, 0}}}},
        {CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {7, 0}), CoreRange({0, 1}, {1, 1})}),
         5,
         McastCoreOrder::RowMajor,
         {{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}, {{5, 0}, {6, 0}, {7, 0}, {0, 1}, {1, 1}}}},
        {grid({0, 0}, {1, 1}), 2, McastCoreOrder::ColumnMajor, {{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}}},
        {grid({0, 0}, {1, 0}), 1, McastCoreOrder::RowMajor, {{{0, 0}}, {{1, 0}}}}};
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig config;
        config.noc = noc;
        for (const auto& test : cases) {
            const Mcast mcast(*device_, config, test.receivers, test.group_size, McastFixedSenderConfig{}, test.order);
            for (const auto& group : test.groups) {
                check_group(device_, mcast, GroupInput(cores(group), std::vector<CoreCoord>{group.front()}));
            }
            EXPECT_EQ(allocated_semaphores(mcast).size(), 2u);
            EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.flags & 1, 1u);
            config.handshake = false;
            const Mcast passive(
                *device_, config, test.receivers, test.group_size, McastFixedSenderConfig{}, test.order);
            EXPECT_EQ(emitted_metadata(compile_args(passive)).mcast.flags & 1u, 0u);
            config.handshake = true;
        }
        EXPECT_ANY_THROW(Mcast(*device_, config, CoreRangeSet{}, 1));
        EXPECT_ANY_THROW(Mcast(*device_, config, grid({0, 0}, {2, 0}), 2));
    }
}

TEST_F(McastHostFixture, IrregularReceiverSetPolicyDoesNotAffectRegularMcasts) {
    const GroupInput dense(grid({1, 1}, {3, 1}), {{1, 1}});
    McastConfig config;
    config.handshake = false;
    const GroupInput passive_dense(dense.receivers, dense.senders, 0);
    auto ordinary = make_mcast(device_, {passive_dense}, config);
    auto chain_link_policy = make_mcast(device_, {passive_dense}, chain_config(config));
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(ordinary)).mcast.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(compile_args(chain_link_policy), compile_args(ordinary));
    EXPECT_EQ(runtime_args(chain_link_policy, {1, 1}), runtime_args(ordinary, {1, 1}));
    config.handshake = true;
    auto requested = make_mcast(device_, {dense}, chain_config(config));
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(requested)).mcast.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(emitted_metadata(compile_args(requested)).mcast.rectangle_capacity, 1u);
    EXPECT_EQ(decoded_args(requested, {1, 1}).rectangles.size(), 1u);
    EXPECT_EQ(decoded_args(requested, {1, 1}).ack, 2u);
    auto rotating = make_mcast(device_, {GroupInput(dense.receivers, {{1, 1}, {2, 1}})}, chain_config());
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(rotating)).mcast.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    const GroupInput irregular(cores({{0, 0}, {2, 0}, {4, 0}}), {{0, 0}});
    auto multicast = make_mcast(device_, {irregular}, {});
    EXPECT_EQ(emitted_metadata(compile_args(multicast)).mcast.rectangle_capacity, 3u);
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(multicast)).mcast.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
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
            auto mcast = make_mcast(device_, {GroupInput(receivers, {sender})}, chain_config(config));
            EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 0u);
            const auto ct = compile_args(mcast);
            EXPECT_EQ(ct.size(), 4u);
            EXPECT_EQ(emitted_semaphore(ct, wire::SIGNAL_SOURCE), 2u);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(ct).mcast.flags), dataflow_kernel_lib::TransferMode::ChainUnicast);
            const std::vector<CoreCoord> expected = receivers.contains(sender)
                                                        ? std::vector<CoreCoord>{sender, {0, 1}, {0, 2}}
                                                        : std::vector<CoreCoord>{sender, {0, 1}, {2, 1}, {0, 2}};
            const wire::RuntimeLayout layout(emitted_metadata(ct));
            for (size_t i = 0; i < expected.size(); ++i) {
                auto rt = runtime_args(mcast, expected[i]);
                ASSERT_EQ(rt.size(), 8u);  // Dynamic role, head coordinates, and five chain words.
                auto mapped = [&](CoreCoord c) { return device_->worker_core_from_logical_core(c); };
                const auto predecessor = i ? mapped(expected[i - 1]) : CoreCoord{NO_CHAIN_NEIGHBOR, NO_CHAIN_NEIGHBOR};
                const auto successor =
                    i + 1 < expected.size() ? mapped(expected[i + 1]) : CoreCoord{NO_CHAIN_NEIGHBOR, NO_CHAIN_NEIGHBOR};
                EXPECT_EQ(rt[layout.chain_neighbors + wire::PREDECESSOR_X], predecessor.x);
                EXPECT_EQ(rt[layout.chain_neighbors + wire::PREDECESSOR_Y], predecessor.y);
                EXPECT_EQ(rt[layout.chain_neighbors + wire::SUCCESSOR_X], successor.x);
                EXPECT_EQ(rt[layout.chain_neighbors + wire::SUCCESSOR_Y], successor.y);
                EXPECT_EQ(rt[layout.chain_neighbors + wire::INCLUDES_SENDER], receivers.contains(sender));
                EXPECT_EQ(rt[layout.roles], i == 0 ? wire::CAN_SEND : wire::CAN_RECEIVE);
            }
            const auto inactive = runtime_args(mcast, {7, 7});
            EXPECT_EQ(inactive.size(), 8u);
            EXPECT_EQ(inactive[layout.roles], 0u);
        }
    }
}

TEST_F(McastHostFixture, ChainMcastUsesOneCompileTimeTransportForEveryGeometry) {
    const GroupInput dense(grid({0, 0}, {2, 0}), {{0, 0}});
    const GroupInput irregular(cores({{0, 2}, {2, 2}, {4, 2}}), {{0, 2}});
    const GroupInput local(cores({{6, 0}}), {{6, 0}});
    auto mcast = make_mcast(device_, {dense, irregular, local}, chain_config());
    EXPECT_EQ(emitted_metadata(compile_args(mcast)).mcast.rectangle_capacity, 0u);
    const auto ct = compile_args(mcast);
    EXPECT_EQ(wire::transfer_mode(emitted_metadata(ct).mcast.flags), dataflow_kernel_lib::TransferMode::ChainUnicast);
    EXPECT_EQ(emitted_metadata(ct).mcast.ack_count, 0u);
    const wire::RuntimeLayout layout(emitted_metadata(ct));
    const auto local_rt = runtime_args(mcast, {6, 0});
    EXPECT_EQ(local_rt[layout.chain_neighbors + wire::PREDECESSOR_X], dataflow_kernel_lib::NO_CHAIN_NEIGHBOR);
    EXPECT_EQ(local_rt[layout.chain_neighbors + wire::SUCCESSOR_X], dataflow_kernel_lib::NO_CHAIN_NEIGHBOR);
    EXPECT_EQ(local_rt[layout.chain_neighbors + wire::INCLUDES_SENDER], 1u);
    for (auto core : std::vector<CoreCoord>{{0, 0}, {1, 0}, {0, 2}, {2, 2}, {6, 0}, {7, 7}}) {
        const auto rt = runtime_args(mcast, core);
        ASSERT_EQ(rt.size(), 8u);  // Dynamic role, head coordinates, and neighbors; no transport selector.
        std::vector<uint32_t> args{123};
        detail::append_args_to(args, runtime_args(mcast, core));
        detail::append_args_to(args, runtime_args(mcast, core));
        EXPECT_EQ(args.size(), 1 + 2 * rt.size());
        EXPECT_TRUE(std::equal(rt.begin(), rt.end(), args.begin() + 1));
        EXPECT_TRUE(std::equal(rt.begin(), rt.end(), args.begin() + 1 + rt.size()));
    }
    std::vector<uint32_t> appended;
    detail::append_args_to(appended, compile_args(mcast));
    EXPECT_EQ(appended, ct);
    EXPECT_EQ(allocated_semaphores(mcast).size(), 3u);
    EXPECT_EQ(allocated_semaphores(mcast)[2].core_ranges, mcast.participating_cores());
}

TEST_F(McastHostFixture, SameInputsCanPrepareDifferentMcastTransports) {
    const GroupInput group(cores({{0, 0}, {2, 0}}), {{0, 0}});
    auto chain = make_mcast(device_, {group}, chain_config());
    auto multicast = make_mcast(device_, {group});
    EXPECT_EQ(compile_args(multicast), compile_args(make_mcast(device_, {group})));
    check_group(device_, multicast, group);
    auto chain_again = make_mcast(device_, {group}, chain_config());
    EXPECT_EQ(compile_args(chain_again), compile_args(chain));
    EXPECT_EQ(runtime_args(chain_again, {2, 0}), runtime_args(chain, {2, 0}));
}

TEST_F(McastHostFixture, ChainSignalSourceAllocationAndWire) {
    const GroupInput group(cores({{0, 0}, {2, 0}}), {{0, 0}});
    auto cfg = chain_config();
    auto mcast = make_mcast(device_, {group}, cfg);
    const std::vector<uint32_t> expected{0x000E1293u, 0, 1, 2};
    EXPECT_EQ(compile_args(mcast), expected);
    EXPECT_EQ(allocated_semaphores(mcast).size(), 3u);
    const auto owned = allocated_semaphores(mcast);
    ASSERT_FALSE(owned.empty());
    EXPECT_EQ(owned.back().id + 1, 3u);
    const auto semaphores = allocated_semaphores(mcast);
    ASSERT_EQ(semaphores.size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(semaphores[i].id, i);
        EXPECT_EQ(semaphores[i].initial_value, 0u);
        EXPECT_EQ(semaphores[i].core_ranges, mcast.participating_cores());
    }
    // A rectangular mcast resolves to multicast and needs only two semaphore IDs.
    const auto dense = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})}, cfg);
    EXPECT_EQ(compile_args(dense).size(), 4u);
}

}  // namespace ttnn::kernel_lib::host::test
