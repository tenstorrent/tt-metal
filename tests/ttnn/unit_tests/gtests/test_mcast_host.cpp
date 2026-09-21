// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.hpp"
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
template <typename T>
concept ExposesPreparation = requires(T value) { value.prepare_arguments(); };
static_assert(!ExposesPreparation<McastFamily>);
template <typename T>
concept AcceptsSingleKernel =
    requires(T value, tt::tt_metal::ProgramDescriptor descriptor, tt::tt_metal::KernelDescriptor kernel) {
        value.attach(descriptor, "channel", kernel);
    };
static_assert(!AcceptsSingleKernel<McastFamily>);
static_assert(!AcceptsSingleKernel<Mcast1D>);
static_assert(!AcceptsSingleKernel<Mcast2D>);

class McastHostFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastHostFixture> {};

CoreRangeSet grid(CoreCoord start, CoreCoord end) { return CoreRangeSet(CoreRange(start, end)); }
CoreRangeSet cores(const std::vector<CoreCoord>& values) {
    std::vector<CoreRange> ranges;
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
    return family;
}

void attach_for_inspection(const McastFamily& family, const McastConfig& cfg = {}) {
    tt::tt_metal::ProgramDescriptor descriptor;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.core_ranges = family.participating_cores();
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = cfg.noc};
    if (cfg.sem_ids) {
        for (const auto id : *cfg.sem_ids) {
            descriptor.semaphores.push_back({.id = id, .core_ranges = kernel.core_ranges, .initial_value = 0});
        }
    }
    const std::array targets{std::ref(kernel)};
    family.attach(descriptor, "inspection", targets);
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

struct DecodedCompileTime {
    wire::ArgumentMetadata metadata;
    std::array<uint32_t, 3> semaphores{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    uint32_t words = 1;
};

DecodedCompileTime decode_emitted_ct(const std::vector<uint32_t>& ct, uint32_t base = 0, bool ids = true) {
    // Independent literal v3 decoder; complete-block goldens below pin the bits
    // and optional-field order without using the production codec or its layout.
    DecodedCompileTime result;
    const uint32_t control = ct.at(base);
    if (control == 0) {
        return result;
    }
    EXPECT_EQ(control & 15u, 3u);
    auto& m = result.metadata;
    m.family.flags = (control >> 4) & 31u;
    m.family.has_remote_receivers = (control >> 9) & 1u;
    m.family.sender_mcast_mode = SenderMcastMode((control >> 10) & 7u);
    m.family.rectangle_capacity = (control >> 13) & 3u;
    m.kernel.roles = control & (1u << 17) ? 0xFFFFFFFFu : (control >> 15) & 3u;
    m.kernel.capabilities = (control >> 18) & 3u;
    m.coordinates.encoding = wire::SenderCoordinateEncoding((control >> 20) & 3u);
    auto next = [&]() { return ct.at(base + result.words++); };
    if (ids) {
        result.semaphores[0] = next();
        if (m.family.flags & 1u) {
            result.semaphores[1] = next();
        }
        if ((m.family.flags >> 3) & 3u) {
            result.semaphores[2] = next();
        }
    }
    m.family.remote_count_known = (control >> 22) & 1u;
    if (m.family.remote_count_known) {
        m.family.uniform_remote_count = next();
    }
    const uint32_t ack = (control >> 24) & 3u;
    m.family.ack_count = ack == 1 ? next() : ack == 2 ? m.family.uniform_remote_count : ack == 3 ? 0xFFFFFFFFu : 0u;
    if (control & (1u << 23)) {
        m.family.rotating_span = next();
    }
    if (uint32_t(m.coordinates.encoding) != 0) {
        m.coordinates.columns = next();
        m.coordinates.rows = next();
        m.coordinates.x_ranges = next();
        m.coordinates.y_ranges = next();
    }
    return result;
}

wire::ArgumentMetadata emitted_metadata(const std::vector<uint32_t>& ct, uint32_t base = 0) {
    return decode_emitted_ct(ct, base).metadata;
}

uint32_t emitted_semaphore(const std::vector<uint32_t>& ct, wire::SemaphoreRole role, uint32_t base = 0) {
    return decode_emitted_ct(ct, base).semaphores[role];
}

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
        decoded.ack = layout.ack == wire::OMITTED ? metadata.family.ack_count : words[layout.ack];
        const uint32_t count = layout.rectangle_count == wire::OMITTED ? 1 : words[layout.rectangle_count];
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t base = layout.rectangles + i * layout.rectangle_stride;
            dataflow_kernel_lib::RectangleRuntimeArguments rectangle{};
            if (layout.rectangle_bounds != wire::OMITTED) {
                const uint32_t bounds = base + layout.rectangle_bounds;
                rectangle.bounds = {words[bounds], words[bounds + 1], words[bounds + 2], words[bounds + 3]};
            }
            rectangle.remote_count = layout.rectangle_remote == wire::OMITTED ? metadata.family.uniform_remote_count
                                                                              : words[base + layout.rectangle_remote];
            rectangle.loopback_count = rectangle.remote_count + 1;
            rectangle.sender_mcast_mode = layout.rectangle_mode == wire::OMITTED
                                              ? metadata.family.sender_mcast_mode
                                              : static_cast<SenderMcastMode>(words[base + layout.rectangle_mode]);
            decoded.rectangles.push_back(rectangle);
        }
    }
    if ((decoded.roles & wire::CAN_RECEIVE) && layout.sender_coordinates != wire::OMITTED) {
        const uint32_t count = std::max(1u, metadata.family.rotating_span);
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

DecodedMcast decoded_args(const McastFamily& family, CoreCoord core, const std::vector<uint32_t>& existing = {}) {
    return decode_multicast(emitted_metadata(compile_args(family, existing)), runtime_args(family, core, existing));
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
    ASSERT_EQ(ct.size(), decode_emitted_ct(ct).words);
    ASSERT_EQ(ct[0] & 15u, 3u);
    const uint32_t count = group.senders.size();
    EXPECT_EQ(emitted_metadata(ct).family.rotating_span, (count > 1) ? count : 0u);
    Coordinates expected;
    for (auto c : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        auto w = device->worker_core_from_logical_core(c);
        expected.emplace(w.x, w.y);
    }
    for (uint32_t phase = 0; phase < count; ++phase) {
        const auto sender = group.senders[phase];
        const auto worker = device->worker_core_from_logical_core(sender);
        const auto rt = runtime_args(family, sender, existing);
        const auto decoded = decode_multicast(emitted_metadata(ct), rt);
        EXPECT_EQ(decoded.phase, phase);
        EXPECT_EQ(decoded.roles, 1u | (count > 1 && group.receivers.contains(sender) ? 2u : 0u));
        Coordinates actual;
        uint32_t remote_total = 0;
        for (const auto& rectangle : decoded.rectangles) {
            if (uint32_t(emitted_metadata(ct).family.sender_mcast_mode) == 1) {
                EXPECT_EQ(rectangle.remote_count, 0u);
                EXPECT_EQ(rectangle.loopback_count, 1u);
                actual.emplace(worker.x, worker.y);
                continue;
            }
            const auto& r = rectangle.bounds;
            const auto xlo = std::min(r.sx, r.ex), xhi = std::max(r.sx, r.ex);
            const auto ylo = std::min(r.sy, r.ey), yhi = std::max(r.sy, r.ey);
            EXPECT_EQ(r.sx, (emitted_metadata(ct).family.flags & 4) ? xhi : xlo);
            EXPECT_EQ(r.sy, (emitted_metadata(ct).family.flags & 4) ? yhi : ylo);
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
            if (uint32_t(emitted_metadata(ct).family.sender_mcast_mode) != 4) {
                EXPECT_EQ(
                    uint32_t(rectangle.sender_mcast_mode), uint32_t(emitted_metadata(ct).family.sender_mcast_mode));
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
        EXPECT_EQ(rt, runtime_args(family, sender, existing));
    }
    for (auto core : tt::tt_metal::corerange_to_cores(group.receivers, std::nullopt, true)) {
        if (std::find(group.senders.begin(), group.senders.end(), core) != group.senders.end()) {
            continue;
        }
        const auto decoded = decoded_args(family, core, existing);
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

TEST(McastHostWire, AbsentFamilyIsOneWord) {
    std::vector<uint32_t> args{17};
    append_absent_mcast_compile_time_args_to(args);
    EXPECT_EQ(args, (std::vector<uint32_t>{17, 0}));
}

constexpr wire::ArgumentMetadata fixed_metadata(uint32_t roles) {
    return {
        .family =
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
    static_assert(decoded.family.ack_count == 7 && decoded.family.uniform_remote_count == 7);
    static_assert(decoded.kernel.roles == 1 && decoded.kernel.capabilities == 1);
    EXPECT_EQ(decode_emitted_ct(std::vector<uint32_t>(literal.begin(), literal.end())).words, 4u);
    EXPECT_FALSE(wire::valid_compile_time_control(1));
    EXPECT_FALSE(wire::valid_compile_time_control(2));
    constexpr std::array<uint32_t, 1> absent{0};
    static_assert(wire::CompileTimeLayout(0).words == 1);
    static_assert(wire::decode_compile_time_metadata(absent).family.rotating_span == 0);

    auto receiver = fixed_metadata(2);
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(receiver)).words + 2, 5u);
    receiver.family.flags = 0;
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(receiver)).words + 2, 4u);
    auto custom_ack = sender;
    custom_ack.family.ack_count = 0;  // Known zero must remain a separate constant, not mean "use fanout".
    EXPECT_EQ(wire::CompileTimeLayout(wire::compile_time_control(custom_ack)).words + 2, 7u);

    auto large = sender;
    large.kernel = {.roles = wire::DYNAMIC_ROLES, .capabilities = 3};
    large.family.uniform_remote_count = 0x12345678;
    large.family.ack_count = 0x01234567;
    large.family.rotating_span = 0x10001;
    large.coordinates = {wire::SenderCoordinateEncoding::ColumnMajorRanges, 0x10002, 0x10003, 0x10004, 0x10005};
    for (const bool ids : {false, true}) {
        const wire::CompileTimeLayout layout(wire::compile_time_control(large), ids);
        std::vector<uint32_t> words(layout.words);
        wire::encode_compile_time_metadata(words, large, ids);
        const auto result = wire::decode_compile_time_metadata(words, ids);
        const auto independent = decode_emitted_ct(words, 0, ids);
        EXPECT_EQ(independent.words, words.size());
        EXPECT_EQ(result.family.uniform_remote_count, large.family.uniform_remote_count);
        EXPECT_EQ(result.family.ack_count, large.family.ack_count);
        EXPECT_EQ(result.family.rotating_span, large.family.rotating_span);
        EXPECT_EQ(result.coordinates.columns, large.coordinates.columns);
        EXPECT_EQ(result.coordinates.rows, large.coordinates.rows);
        EXPECT_EQ(result.coordinates.x_ranges, large.coordinates.x_ranges);
        EXPECT_EQ(result.coordinates.y_ranges, large.coordinates.y_ranges);
        EXPECT_EQ(independent.metadata.family.rotating_span, large.family.rotating_span);
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
                            metadata.family.flags = flags;
                            metadata.family.rectangle_capacity = rectangles;
                            metadata.family.ack_count = ack;
                            metadata.family.remote_count_known = known;
                            metadata.family.sender_mcast_mode = mode;
                            verify(metadata);
                            metadata.family.rotating_span = 61;
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
    chain.family.flags = 9;
    chain.family.rectangle_capacity = 0;
    chain.family.sender_mcast_mode = SenderMcastMode::Unknown;
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
    local.family.sender_mcast_mode = SenderMcastMode::LocalCopy;
    local.family.uniform_remote_count = 0;
    local.family.ack_count = 0;
    EXPECT_EQ(wire::RuntimeLayout(local).words, 0u);
    local.family.remote_count_known = false;
    EXPECT_EQ(wire::RuntimeLayout(local).words, 1u);
    auto multiple = fixed_metadata(1);
    multiple.family.rectangle_capacity = 2;
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 11u);
    EXPECT_EQ(wire::RuntimeLayout(multiple).rectangle_remote, 4u);  // Equal total fanout is not per-rectangle fanout.
    multiple.family.ack_count = 0xFFFFFFFFu;
    multiple.family.sender_mcast_mode = SenderMcastMode::Unknown;
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 14u);
    multiple.family.flags = 0;  // No ACK field without handshakes.
    EXPECT_EQ(wire::RuntimeLayout(multiple).words, 13u);
    multiple.family.flags = 9;
    multiple.family.rectangle_capacity = 0;
    const wire::RuntimeLayout chain(multiple);
    EXPECT_EQ(chain.words, 11u);
    EXPECT_EQ(chain.chain_neighbors, 4u);
    EXPECT_EQ(chain.roles, 9u);
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
    metadata.family.rotating_span = 64;
    metadata.coordinates = rows;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 6u);
    metadata.coordinates.x_ranges = 1;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 4u);
    metadata.family.rotating_span = 8;
    metadata.coordinates.rows = 1;
    EXPECT_EQ(wire::RuntimeLayout(metadata).coordinate_words, 4u);
}

TEST_F(McastHostFixture, CompactPlacementGoldensAndDirectDescriptorParity) {
    using namespace tt::tt_metal;
    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool handshake : {false, true}) {
            auto family =
                make_family(device_, {{grid({1, 2}, {4, 2}), {{1, 2}}}}, {.noc = noc, .handshake = handshake});
            ProgramDescriptor descriptor;
            KernelDescriptor sender, receiver, inactive;
            sender.core_ranges = cores({{1, 2}});
            receiver.core_ranges = grid({2, 2}, {4, 2});
            inactive.core_ranges = cores({{0, 0}});
            for (auto* kernel : {&sender, &receiver, &inactive}) {
                kernel->config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = noc};
            }
            family.attach(descriptor, "channel", std::array{std::ref(sender), std::ref(receiver), std::ref(inactive)});
            const auto lo = device_->worker_core_from_logical_core({1, 2});
            const auto hi = device_->worker_core_from_logical_core({4, 2});
            const std::vector<uint32_t> bounds = noc == NOC::NOC_0 ? std::vector<uint32_t>{lo.x, lo.y, hi.x, hi.y}
                                                                   : std::vector<uint32_t>{hi.x, hi.y, lo.x, lo.y};
            EXPECT_EQ(sender.runtime_args.front().second, bounds);
            for (const auto& [core, args] : receiver.runtime_args) {
                EXPECT_EQ(args, (std::vector<uint32_t>{lo.x, lo.y}));
            }
            EXPECT_TRUE(inactive.runtime_args.front().second.empty());
            EXPECT_EQ(sender.compile_time_args.size() + sender.named_compile_time_args.size(), handshake ? 6u : 5u);
            EXPECT_EQ(receiver.compile_time_args.size() + receiver.named_compile_time_args.size(), handshake ? 5u : 4u);
            EXPECT_EQ(emitted_metadata(sender.compile_time_args).kernel.roles, 1u);
            EXPECT_EQ(emitted_metadata(receiver.compile_time_args).kernel.roles, 2u);
            EXPECT_EQ(emitted_metadata(inactive.compile_time_args).kernel.roles, 0u);
            auto bound = family;
            Program program;
            bound.append_semaphores(program);
            for (const auto* expected : {&sender, &receiver, &inactive}) {
                std::vector<uint32_t> ct;
                KernelDescriptor::RuntimeArgs rt;
                const auto offsets = bound.append_kernel_args_to(ct, rt, expected->core_ranges);
                EXPECT_EQ(offsets.compile_time, 0u);
                EXPECT_EQ(offsets.runtime, 0u);
                EXPECT_EQ(ct, expected->compile_time_args);
                EXPECT_EQ(rt, expected->runtime_args);
            }
            std::vector<uint32_t> ct{17};
            KernelDescriptor::RuntimeArgs rt{{{7, 7}, {19}}};
            const auto before = rt;
            EXPECT_ANY_THROW(bound.append_kernel_args_to(ct, rt, sender.core_ranges));
            EXPECT_EQ(ct, (std::vector<uint32_t>{17}));
            EXPECT_EQ(rt, before);
            // Placement specialization must not mutate the conservative paired API.
            EXPECT_EQ(emitted_metadata(compile_args(family)).kernel.roles, 0xFFFFFFFFu);
        }
    }
    auto local = make_family(device_, {{cores({{2, 3}}), {{2, 3}}}});
    tt::tt_metal::ProgramDescriptor descriptor;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.core_ranges = cores({{2, 3}});
    kernel.config =
        tt::tt_metal::DataMovementConfigDescriptor{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0};
    local.attach(descriptor, "local", std::array{std::ref(kernel)});
    EXPECT_TRUE(kernel.runtime_args.front().second.empty());
    EXPECT_EQ(kernel.compile_time_args[0] & 15u, 3u);
    EXPECT_EQ(emitted_metadata(kernel.compile_time_args).family.remote_count_known, 1u);
    tt::tt_metal::KernelDescriptor empty;
    empty.config = kernel.config;
    local.attach(descriptor, "empty", std::array{std::ref(empty)});
    EXPECT_EQ(empty.compile_time_args[0] & 15u, 3u);
    EXPECT_EQ(emitted_metadata(empty.compile_time_args).kernel.roles, 0xFFFFFFFFu);
    EXPECT_EQ(emitted_metadata(empty.compile_time_args).kernel.capabilities, 3u);
    EXPECT_TRUE(empty.runtime_args.empty());
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
    const std::vector<uint32_t> expected_ct{0x01CE2A73u, 11, 13, 9, 5, 2};
    EXPECT_EQ(compile_args(family, {11, 13}), expected_ct);
    cfg.handshake = false;
    cfg.sem_ids = std::vector<uint32_t>{11};
    auto passive = make_family(device_, {GroupInput(grid({2, 3}, {4, 5}), senders)}, cfg);
    const std::vector<uint32_t> face_ct{0x00CE2A63u, 11, 9, 2};
    EXPECT_EQ(compile_args(passive, {11}), face_ct);
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
        EXPECT_EQ(runtime_args(family, senders[phase], {11, 13}), expected_rt);
    }
    const std::vector<uint32_t> receiver_rt{
        2, 0xFFFFFFFFu, uint32_t(a.x), uint32_t(a.y), uint32_t(b.x), uint32_t(b.y), 0, 0, 0, 0};
    EXPECT_EQ(runtime_args(family, {3, 4}, {11, 13}), receiver_rt);
    std::vector<uint32_t> inactive(10, 0);
    inactive[1] = 0xFFFFFFFFu;
    EXPECT_EQ(runtime_args(family, {0, 0}, {11, 13}), inactive);
    const auto sender_only = family.sender_only_cores();
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
                for (uint32_t index = 0; index < count; ++index) {
                    senders.emplace_back(
                        1 + (column_major ? index / 8 : index % 8), 1 + (column_major ? index % 8 : index / 8));
                }
                const GroupInput input(grid({1, 1}, {8, 8}), senders);
                auto family = make_family(device_, {input}, {.noc = noc});
                const auto ct = compile_args(family);
                ASSERT_NE(
                    uint32_t(emitted_metadata(ct).coordinates.encoding),
                    0u);  // Regular mapped grids/lines save words even across worker-coordinate gaps.
                const auto metadata = emitted_metadata(ct);
                for (uint32_t phase = 0; phase < count; ++phase) {
                    const auto decoded = decoded_args(family, senders[phase]);
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
        auto mixed = make_family(device_, {line, square}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(mixed)).coordinates.encoding), 0u);
        for (const auto& input : {line, square}) {
            tt::tt_metal::ProgramDescriptor descriptor;
            tt::tt_metal::KernelDescriptor kernel;
            kernel.core_ranges = input.receivers;
            kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc};
            mixed.attach(descriptor, "channel", std::array{std::ref(kernel)});
            EXPECT_NE(uint32_t(emitted_metadata(kernel.compile_time_args).coordinates.encoding), 0u);
            EXPECT_EQ(wire::RuntimeLayout(emitted_metadata(kernel.compile_time_args)).coordinate_words, 4u);
        }
        const GroupInput custom(grid({0, 0}, {2, 1}), {{0, 0}, {1, 1}, {2, 0}, {0, 1}});
        auto sparse = make_family(device_, {custom}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(sparse)).coordinates.encoding), 0u);
        check_group(device_, sparse, custom);
        const GroupInput external(grid({0, 4}, {3, 4}), line.senders);
        auto outside = make_family(device_, {external}, {.noc = noc});
        EXPECT_NE(uint32_t(emitted_metadata(compile_args(outside)).coordinates.encoding), 0u);
        check_group(device_, outside, external);
        const GroupInput external_custom(external.receivers, custom.senders);
        auto outside_custom = make_family(device_, {external_custom}, {.noc = noc});
        EXPECT_EQ(uint32_t(emitted_metadata(compile_args(outside_custom)).coordinates.encoding), 0u);
        check_group(device_, outside_custom, external_custom);

        std::vector<CoreCoord> eight_senders;
        for (uint32_t x = 0; x < 8; ++x) {
            eight_senders.emplace_back(x, 0);
        }
        const GroupInput split_receivers(
            CoreRangeSet(std::set{CoreRange({0, 0}, {7, 0}), CoreRange({0, 2}, {3, 2})}), eight_senders);
        auto split = make_family(device_, {split_receivers}, {.noc = noc});
        const auto split_metadata = emitted_metadata(compile_args(split));
        EXPECT_NE(split_metadata.coordinates.encoding, wire::SenderCoordinateEncoding::ExplicitPairs);
        EXPECT_EQ(split_metadata.family.rectangle_capacity, 2u);
        EXPECT_EQ(split_metadata.family.sender_mcast_mode, SenderMcastMode::Unknown);
        const wire::RuntimeLayout split_layout(split_metadata);
        EXPECT_EQ(split_layout.rectangle_stride, 6u);    // Bounds, per-rectangle remote, mode.
        EXPECT_EQ(split_layout.sender_coordinates, 3u);  // Roles, rotating phase, rectangle count.
        check_group(device_, split, split_receivers);

        for (const bool compatible : {false, true}) {
            const GroupInput external_group(
                grid({0, 4}, {3, 4}),
                compatible ? std::vector<CoreCoord>{{0, 6}, {1, 6}, {2, 6}, {3, 6}}
                           : std::vector<CoreCoord>{{0, 6}, {1, 6}, {0, 7}, {1, 7}});
            auto receiver_and_sender_groups = make_family(device_, {line, external_group}, {.noc = noc});
            tt::tt_metal::ProgramDescriptor descriptor;
            tt::tt_metal::KernelDescriptor kernel;
            kernel.core_ranges = line.receivers.merge(cores(external_group.senders));
            kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc};
            receiver_and_sender_groups.attach(descriptor, "channel", std::array{std::ref(kernel)});
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
                    EXPECT_EQ(owned.back().id + 1, 4u);
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
        EXPECT_EQ(emitted_metadata(compile_args(family)).family.rectangle_capacity, 3u);
        EXPECT_EQ(decoded_args(family, {3, 3}).rectangles.size(), 1u);
        EXPECT_EQ(decoded_args(family, {7, 0}).ack, 4u);
        EXPECT_EQ(decoded_args(family, {3, 3}).ack, 2u);
        EXPECT_EQ(emitted_metadata(compile_args(family)).family.ack_count, ACK_EQUALS_FANOUT);
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
        EXPECT_EQ(emitted_metadata(compile_args(family)).family.rectangle_capacity, 1u);
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
        auto multicast = make_family(device_, groups, cfg);
        auto chain = make_family(device_, groups, chain_config(cfg));
        EXPECT_EQ(emitted_metadata(compile_args(multicast)).family.rectangle_capacity, 3u);
        EXPECT_EQ(emitted_metadata(compile_args(chain)).family.rectangle_capacity, 0u);
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
    EXPECT_EQ(emitted_metadata(compile_args(family)).family.has_remote_receivers, 0u);
    auto outside = runtime_args(family, {0, 0});
    EXPECT_EQ(outside.size(), 3u);  // Dynamic role + fixed coordinate pair; no local-copy rectangle fields.
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
        EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(hardware)).family.flags), TransferMode::Multicast);
        check_group(device_, hardware, dense);
        auto chain = make_family(device_, {irregular}, chain_config(config));
        EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(chain)).family.flags), TransferMode::ChainUnicast);
        auto rectangles = make_family(device_, {dense, local}, chain_config(config));
        EXPECT_EQ(
            wire::transfer_mode(emitted_metadata(compile_args(rectangles)).family.flags), TransferMode::Multicast);
        for (const auto& groups :
             {std::vector<GroupInput>{dense, irregular, local}, std::vector<GroupInput>{irregular, local, dense}}) {
            auto family = make_family(device_, groups, chain_config(config));
            auto multiple_mcast = make_family(device_, groups, config);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(compile_args(family)).family.flags), TransferMode::ChainUnicast);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(compile_args(multiple_mcast)).family.flags),
                TransferMode::Multicast);
            EXPECT_EQ(emitted_metadata(compile_args(family)).family.rectangle_capacity, 0u);
            EXPECT_EQ(emitted_metadata(compile_args(multiple_mcast)).family.rectangle_capacity, 3u);
            EXPECT_EQ(runtime_args(family, {1, 0})[wire::ACK], 1u);  // Dense group also uses per-hop readiness.
            EXPECT_EQ(runtime_args(family, {2, 2})[wire::ACK], 1u);
            EXPECT_EQ(runtime_args(family, {6, 0})[wire::ACK], 0u);
            for (auto core : tt::tt_metal::corerange_to_cores(family.participating_cores())) {
                EXPECT_EQ(runtime_args(family, core).size(), wire::CHAIN_RUNTIME_WORDS);
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
            for (auto config_ack :
                 {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{1}}) {
                for (auto group_ack :
                     {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2}}) {
                    McastConfig cfg;
                    cfg.data_ready = signal;
                    cfg.handshake = handshake;
                    cfg.ack_count_override = config_ack;
                    cfg.base_sem_id = 4;
                    GroupInput group(receivers, std::vector<CoreCoord>{{2, 2}}, group_ack);
                    auto family = make_family(device_, {group}, cfg);
                    EXPECT_EQ(
                        decoded_args(family, {2, 2}).ack, handshake ? group_ack.value_or(config_ack.value_or(2)) : 0u);
                    EXPECT_EQ(
                        emitted_metadata(compile_args(family)).family.flags,
                        uint32_t(handshake) + (signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u));
                    auto passive_cfg = cfg;
                    passive_cfg.handshake = false;
                    EXPECT_EQ(
                        emitted_metadata(compile_args(make_family(device_, {group}, passive_cfg))).family.flags,
                        signal == dataflow_kernel_lib::DataReadySignal::Counter ? 2u : 0u);
                    EXPECT_EQ(allocated_semaphores(family).size(), handshake ? 2u : 1u);
                    const auto owned = allocated_semaphores(family);
                    ASSERT_FALSE(owned.empty());
                    EXPECT_EQ(owned.back().id + 1, handshake ? 6u : 5u);
                    cfg.sem_ids = handshake ? std::vector<uint32_t>{6, 7} : std::vector<uint32_t>{6};
                    auto adopted = make_family(device_, {group}, cfg);
                    EXPECT_TRUE(allocated_semaphores(adopted, *cfg.sem_ids).empty());
                    EXPECT_EQ(emitted_semaphore(compile_args(adopted, *cfg.sem_ids), wire::DATA_READY), 6u);
                    EXPECT_EQ(
                        emitted_semaphore(compile_args(adopted, *cfg.sem_ids), wire::CONSUMER_READY),
                        handshake ? 7u : UNUSED_SEM_ID);
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
                    EXPECT_EQ(emitted_metadata(compile_args(family)).family.rotating_span, 0u);
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
    EXPECT_THROW(compile_args(make_family(device_, {fragmented})), std::exception);
    EXPECT_ANY_THROW(McastFamily(device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}}));
    EXPECT_ANY_THROW(McastFamily(device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}, {3, 2}}));
    EXPECT_ANY_THROW(compile_args(make_family(device_, {})));
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
    EXPECT_ANY_THROW(compile_args(make_family(device_, {fixed}, cfg)));
    EXPECT_ANY_THROW(Mcast1D(device_, receivers, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg));
    EXPECT_ANY_THROW(Mcast2D(device_, receivers, Mcast2DFixedSenderConfig{{2, 2}}, cfg));
    for (auto ids : {std::vector<uint32_t>{}, std::vector<uint32_t>{0}, std::vector<uint32_t>{0, UNUSED_SEM_ID}}) {
        cfg = {};
        cfg.sem_ids = ids;
        EXPECT_ANY_THROW(compile_args(make_family(device_, {fixed}, cfg)));
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
            ASSERT_EQ(emitted_metadata(compile_args(family, {6, 7})).family.rectangle_capacity, 3u);
            EXPECT_TRUE(allocated_semaphores(family, {6, 7}).empty());
            for (size_t i = 0; i < groups.size(); ++i) {
                check_group(device_, family, groups[i], {6, 7});
                EXPECT_EQ(decoded_args(family, first[i], {6, 7}).rectangles.size(), i + 1u);
                EXPECT_EQ(decoded_args(family, first[i], {6, 7}).ack, i == 2 ? 1u : 0u);
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
            if (rotating) {
                EXPECT_EQ(inactive[1], wire::NO_SENDER_ROUND);
                inactive[1] = 0;
            }
            EXPECT_TRUE(std::all_of(inactive.begin(), inactive.end(), [](auto word) { return word == 0; }));
        }
    }
}

void check_building_queries(const McastFamily& family) {
    EXPECT_NO_THROW(family.participating_cores());
    EXPECT_NO_THROW(family.sender_only_cores());
    std::vector<uint32_t> unchanged{42};
    EXPECT_ANY_THROW(family.append_compile_time_args_to(unchanged));
    EXPECT_ANY_THROW(family.append_runtime_args_to(unchanged, {0, 0}));
    EXPECT_EQ(unchanged, std::vector<uint32_t>{42});
}

TEST_F(McastHostFixture, CollectionAndArgumentPreparationLifecycle) {
    McastFamily family(device_);
    check_building_queries(family);
    EXPECT_TRUE(family.participating_cores().empty());
    EXPECT_ANY_THROW(attach_for_inspection(family));
    family.add_group(grid({2, 2}, {3, 2}), {{2, 2}});
    check_building_queries(family);
    EXPECT_ANY_THROW(family.add_group(CoreRangeSet{}, {{0, 0}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {2, 3}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {3, 3}}));
    EXPECT_ANY_THROW(family.add_group(grid({2, 3}, {3, 3}), {{2, 2}}));
    family.add_group(grid({4, 2}, {4, 2}), {{4, 2}});
    EXPECT_EQ(family.participating_cores(), grid({2, 2}, {4, 2}));
    attach_for_inspection(family);
    EXPECT_EQ(allocated_semaphores(family).size(), 2u);
    const auto ct = compile_args(family);
    const auto rt = runtime_args(family, {2, 2});
    const auto* participants = &family.participating_cores();
    attach_for_inspection(family);
    EXPECT_EQ(compile_args(family), ct);
    EXPECT_EQ(runtime_args(family, {2, 2}), rt);
    EXPECT_EQ(&family.participating_cores(), participants);
    EXPECT_ANY_THROW(family.add_group(grid({6, 2}, {6, 2}), {{6, 2}}));
    EXPECT_EQ(family.participating_cores().num_cores(), 3u);
}

TEST_F(McastHostFixture, FailedArgumentPreparationAndLateTransportSelection) {
    // Failed preparation preserves topology queries and leaves argument emission unavailable.
    McastConfig invalid;
    invalid.sem_ids = std::vector<uint32_t>{0};
    McastFamily failed(device_, invalid);
    failed.add_group(grid({0, 0}, {2, 0}), {{0, 0}});
    for (int attempt = 0; attempt < 2; ++attempt) {
        EXPECT_ANY_THROW(attach_for_inspection(failed, invalid));
        check_building_queries(failed);
        EXPECT_EQ(failed.participating_cores(), grid({0, 0}, {2, 0}));
    }
    // Failed preparation retains input overlap validation, even with partially prepared groups.
    EXPECT_ANY_THROW(failed.add_group(grid({0, 0}, {1, 0}), {{0, 0}}));
    failed.add_group(grid({4, 0}, {5, 0}), {{4, 0}});
    EXPECT_ANY_THROW(attach_for_inspection(failed, invalid));
    check_building_queries(failed);
    EXPECT_EQ(failed.participating_cores().num_cores(), 5u);

    // A late irregular group changes the common transport of the earlier dense group.
    McastFamily late(device_, chain_config());
    late.add_group(grid({0, 0}, {2, 0}), {{1, 0}});
    check_building_queries(late);
    late.add_group(cores({{0, 2}, {2, 2}}), {{0, 2}});
    attach_for_inspection(late);
    EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(late)).family.flags), TransferMode::ChainUnicast);
    EXPECT_EQ(emitted_metadata(compile_args(late)).family.rectangle_capacity, 0u);
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
    const McastConfig original_cfg{.noc = NOC::NOC_1, .sem_ids = std::vector<uint32_t>{6, 7}};
    attach_for_inspection(copy, original_cfg);
    attach_for_inspection(building, original_cfg);
    EXPECT_EQ(building.participating_cores().num_cores(), 1u);
    EXPECT_EQ(copy.participating_cores().num_cores(), 3u);
    EXPECT_EQ(emitted_semaphore(compile_args(copy, {6, 7}), wire::DATA_READY), 6u);
    EXPECT_NE(emitted_metadata(compile_args(copy, {6, 7})).family.flags & wire::NOC1, 0u);
    EXPECT_EQ(emitted_metadata(compile_args(building, {6, 7})).family.has_remote_receivers, 0u);
    EXPECT_EQ(emitted_metadata(compile_args(copy, {6, 7})).family.has_remote_receivers, 1u);
    const auto ct = compile_args(copy, {6, 7});
    const auto rt = runtime_args(copy, {4, 2}, {6, 7});
    auto moved = [&] {
        auto original = copy;
        return McastFamily(std::move(original));
    }();
    copy = building;
    attach_for_inspection(moved, original_cfg);
    EXPECT_EQ(compile_args(moved, {6, 7}), ct);
    EXPECT_EQ(runtime_args(moved, {4, 2}, {6, 7}), rt);
    McastFamily assigned(device_);
    assigned = moved;
    EXPECT_EQ(runtime_args(assigned, {4, 2}, {6, 7}), rt);
    auto moving_building = McastFamily(device_);
    moving_building.add_group(grid({4, 2}, {5, 2}), {{4, 2}});
    assigned = std::move(moving_building);
    check_building_queries(assigned);
    attach_for_inspection(assigned);
    EXPECT_EQ(assigned.participating_cores(), grid({4, 2}, {5, 2}));
}

TEST_F(McastHostFixture, GroupNormFactoryEmitsExactDestinations) {
    using namespace tt::tt_metal;
    using Operation = ttnn::prim::GroupNormDeviceOperation;
    const auto available = device_->compute_with_storage_grid_size();
    if (available.x < 7 || available.y < 7) {
        GTEST_SKIP() << "Requires both 5x7 and 7x5 shard grids";
    }
    for (bool wrapped : {false, true}) {
        const uint32_t batches = wrapped ? 5u : 1u;
        const uint32_t contributors = wrapped ? 7u : 9u;
        const uint32_t capacity = wrapped ? 3u : 1u;
        for (bool column_major : {false, true}) {
            // Seven contributors wrap across five-core lines, including a partial/full/partial staircase.
            // Both orientations fit on Wormhole as well as Blackhole.
            const CoreCoord dimensions = !wrapped ? CoreCoord{3, 3} : column_major ? CoreCoord{7, 5} : CoreCoord{5, 7};
            const auto shard_cores = grid({0, 0}, {dimensions.x - 1, dimensions.y - 1});
            const MemoryConfig memory{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec(
                    shard_cores,
                    std::array<uint32_t, 2>{32, 128},
                    column_major ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR)};
            const TensorSpec spec(
                ttnn::Shape({batches, 1, contributors * 32, 128}),
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
                const auto offset = std::find_if(
                    it->named_compile_time_args.begin(), it->named_compile_time_args.end(), [](const auto& arg) {
                        return arg.first == "reduction_mcast_ct_offset";
                    });
                ASSERT_NE(offset, it->named_compile_time_args.end());
                const auto metadata = emitted_metadata(it->compile_time_args, offset->second);
                EXPECT_EQ(it->compile_time_args[offset->second] & 15u, 3u);
                EXPECT_EQ(metadata.family.rectangle_capacity, capacity);
                ASSERT_EQ(it->runtime_args.size(), batches);
                std::set<uint32_t> counts;
                for (const auto& [sender, args] : it->runtime_args) {
                    uint32_t batch = 0;
                    while (batch < batches && logical_core(batch * contributors) != sender) {
                        ++batch;
                    }
                    ASSERT_LT(batch, batches);
                    // Contributors' X/Y coordinates precede the family's sender block.
                    const uint32_t rt_base = 2 * contributors;
                    ASSERT_GE(args.size(), rt_base);
                    const auto decoded = decode_multicast(metadata, std::span(args).subspan(rt_base));
                    const uint32_t rectangles = decoded.rectangles.size();
                    ASSERT_GE(rectangles, 1u);
                    ASSERT_LE(rectangles, 3u);
                    counts.insert(rectangles);
                    EXPECT_EQ(metadata.family.flags & wire::PRE_HANDSHAKE, 0u);
                    EXPECT_EQ(decoded.ack, 0u);  // Gather readiness is independent; no multicast ACK field.
                    EXPECT_EQ(decoded.roles, wire::CAN_SEND);
                    Coordinates expected, actual;
                    for (uint32_t i = 0; i < contributors; ++i) {
                        const auto worker =
                            device_->worker_core_from_logical_core(logical_core(batch * contributors + i));
                        expected.emplace(worker.x, worker.y);
                        EXPECT_EQ(args[i], worker.x);
                        EXPECT_EQ(args[contributors + i], worker.y);
                    }
                    for (uint32_t i = 0; i < rectangles; ++i) {
                        const auto& bounds = decoded.rectangles[i].bounds;
                        const auto sx = bounds.sx, sy = bounds.sy;
                        const auto ex = bounds.ex, ey = bounds.ey;
                        for (uint32_t y = std::min(sy, ey); y <= std::max(sy, ey); ++y) {
                            for (uint32_t x = std::min(sx, ex); x <= std::max(sx, ex); ++x) {
                                if (!worker_coordinates(device_).contains({x, y})) {
                                    continue;
                                }
                                EXPECT_TRUE(actual.emplace(x, y).second) << "Overlapping multicast rectangles";
                            }
                        }
                    }
                    EXPECT_EQ(actual, expected) << "Batch " << batch;
                    if (wrapped && batch == batches / 2) {
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
            check_group(device_, family, GroupInput(cores(shape), std::vector<CoreCoord>{shape.front()}), {0, 1});
            check_group(device_, family, GroupInput(cores(second), std::vector<CoreCoord>{second.front()}), {0, 1});
            EXPECT_TRUE(allocated_semaphores(family, {0, 1}).empty());
            EXPECT_EQ(emitted_metadata(compile_args(family, {0, 1})).family.flags & 1, 1u);
            config.handshake = false;
            config.sem_ids = std::vector<uint32_t>{0};
            auto passive = ttnn::prim::make_group_norm_mcast_family(device_, {shape, second}, config);
            EXPECT_EQ(emitted_metadata(compile_args(passive, {0})).family.flags & 1u, 0u);
            config.handshake = true;
            config.sem_ids = std::vector<uint32_t>{0, 1};
        }
        EXPECT_ANY_THROW(ttnn::prim::make_group_norm_mcast_family(device_, {{}}, config));
    }
}

TEST_F(McastHostFixture, IrregularReceiverSetPolicyDoesNotAffectRegularFamilies) {
    const GroupInput dense(grid({1, 1}, {3, 1}), {{1, 1}});
    McastConfig config;
    config.handshake = false;
    config.ack_count_override = 0;
    auto ordinary = make_family(device_, {dense}, config);
    auto chain_link_policy = make_family(device_, {dense}, chain_config(config));
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(ordinary)).family.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(compile_args(chain_link_policy), compile_args(ordinary));
    EXPECT_EQ(runtime_args(chain_link_policy, {1, 1}), runtime_args(ordinary, {1, 1}));
    config.handshake = true;
    config.ack_count_override.reset();
    auto requested = make_family(device_, {dense}, chain_config(config));
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(requested)).family.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    EXPECT_EQ(emitted_metadata(compile_args(requested)).family.rectangle_capacity, 1u);
    EXPECT_EQ(decoded_args(requested, {1, 1}).rectangles.size(), 1u);
    EXPECT_EQ(decoded_args(requested, {1, 1}).ack, 2u);
    auto rotating = make_family(device_, {GroupInput(dense.receivers, {{1, 1}, {2, 1}})}, chain_config());
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(rotating)).family.flags),
        dataflow_kernel_lib::TransferMode::Multicast);
    const GroupInput irregular(cores({{0, 0}, {2, 0}, {4, 0}}), {{0, 0}});
    auto multicast = make_family(device_, {irregular}, {});
    EXPECT_EQ(emitted_metadata(compile_args(multicast)).family.rectangle_capacity, 3u);
    EXPECT_EQ(
        wire::transfer_mode(emitted_metadata(compile_args(multicast)).family.flags),
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
            auto family = make_family(device_, {GroupInput(receivers, {sender})}, chain_config(config));
            EXPECT_EQ(emitted_metadata(compile_args(family)).family.rectangle_capacity, 0u);
            EXPECT_EQ(runtime_args(family, sender)[wire::ACK], 1u);
            const auto ct = compile_args(family);
            EXPECT_EQ(ct.size(), 4u);
            EXPECT_EQ(emitted_semaphore(ct, wire::SIGNAL_SOURCE), 2u);
            EXPECT_EQ(
                wire::transfer_mode(emitted_metadata(ct).family.flags),
                dataflow_kernel_lib::TransferMode::ChainUnicast);
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
    EXPECT_EQ(emitted_metadata(compile_args(family)).family.rectangle_capacity, 0u);
    const auto ct = compile_args(family);
    EXPECT_EQ(wire::transfer_mode(emitted_metadata(ct).family.flags), dataflow_kernel_lib::TransferMode::ChainUnicast);
    EXPECT_EQ(emitted_metadata(ct).family.ack_count, 0u);  // Chain ACKs remain in the unchanged per-core RT block.
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
    EXPECT_ANY_THROW(compile_args(make_family(device_, {group}, chain_config(config))));
    config.handshake = true;
    for (uint32_t ack : {0u, 1u}) {
        config.ack_count_override = ack;
        EXPECT_ANY_THROW(compile_args(make_family(device_, {group}, chain_config(config))));
        EXPECT_ANY_THROW(compile_args(make_family(device_, {GroupInput(receivers, {{0, 0}}, ack)}, chain_config())));
    }
    EXPECT_ANY_THROW(compile_args(make_family(device_, {GroupInput(receivers, {{0, 0}, {2, 0}})}, chain_config())));
    EXPECT_ANY_THROW(compile_args(
        make_family(device_, {GroupInput(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), {{0, 0}})}, chain_config())));
    EXPECT_ANY_THROW(make_family(device_, {group, group}, chain_config()));
    config.ack_count_override.reset();
    config.sem_ids = std::vector<uint32_t>{3, 4, 5};
    auto adopted = make_family(device_, {group}, chain_config(config));
    EXPECT_TRUE(allocated_semaphores(adopted, *config.sem_ids).empty());
    EXPECT_EQ(emitted_semaphore(compile_args(adopted, *config.sem_ids), wire::DATA_READY), 3u);
    EXPECT_EQ(emitted_semaphore(compile_args(adopted, *config.sem_ids), wire::CONSUMER_READY), 4u);
    EXPECT_EQ(emitted_semaphore(compile_args(adopted, *config.sem_ids), wire::SIGNAL_SOURCE), 5u);
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
    cfg.base_sem_id = 3;
    auto family = make_family(device_, {group}, cfg);
    const std::vector<uint32_t> expected{0x000E1293u, 3, 4, 5};
    EXPECT_EQ(compile_args(family), expected);
    EXPECT_EQ(allocated_semaphores(family).size(), 3u);
    const auto owned = allocated_semaphores(family);
    ASSERT_FALSE(owned.empty());
    EXPECT_EQ(owned.back().id + 1, 6u);
    const auto semaphores = allocated_semaphores(family);
    ASSERT_EQ(semaphores.size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(semaphores[i].id, 3u + i);
        EXPECT_EQ(semaphores[i].initial_value, 0u);
        EXPECT_EQ(semaphores[i].core_ranges, family.participating_cores());
    }
    for (auto ids : std::vector<std::vector<uint32_t>>{
             {3, 4}, {3, 4, UNUSED_SEM_ID}, {3, 4, 3}, {3, 4, 4}, {3, 3, 5}, {UNUSED_SEM_ID, 4, 5}}) {
        cfg.sem_ids = ids;
        EXPECT_ANY_THROW(compile_args(make_family(device_, {group}, cfg)));
    }
    cfg.sem_ids = std::vector<uint32_t>{3, 4, 5};
    const auto adopted = make_family(device_, {group}, cfg);
    EXPECT_EQ(compile_args(adopted, {3, 4, 5}), expected);
    EXPECT_TRUE(allocated_semaphores(adopted, {3, 4, 5}).empty());
    EXPECT_EQ(allocated_semaphores(adopted, {3, 4, 5}).size(), 0u);
    // A rectangular family resolves to multicast and needs only two semaphore IDs.
    cfg.sem_ids = std::vector<uint32_t>{3, 4};
    const auto dense = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})}, cfg);
    EXPECT_EQ(compile_args(dense, {3, 4}).size(), 4u);
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
    family.attach(desc, "first", std::array{std::ref(kernel)});
    EXPECT_EQ(
        kernel.named_compile_time_args,
        (KernelDescriptor::NamedCompileTimeArgs{{"op", 99}, {"first_ct_offset", 2}, {"first_rt_offset", 3}}));
    EXPECT_EQ(kernel.buffer_bindings[0].arg_idx, 2u);
    EXPECT_EQ(kernel.common_runtime_args, (std::vector<uint32_t>{71}));
    ASSERT_EQ(kernel.runtime_args.size(), 3u);
    for (size_t i = 0; i < kernel.runtime_args.size(); ++i) {
        const auto& [core, args] = kernel.runtime_args[i];
        auto expected = i == 0   ? std::vector<uint32_t>{21, 0, 0}
                        : i == 1 ? std::vector<uint32_t>{31, 33, 35}
                                 : std::vector<uint32_t>{0, 0, 0};
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
    family.attach(desc, "second", std::array{std::ref(desc.kernels[0])});
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[3].second, old_ct_size);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[4].second, old_rt_size);
    const auto before = desc.kernels[0].compile_time_args;
    EXPECT_ANY_THROW(family.attach(desc, "second", std::array{std::ref(desc.kernels[0])}));
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
    const std::vector<uint32_t> expected_ct{17, 19, 0x024E2E13u, 1, 3, 1};
    EXPECT_EQ(attached.compile_time_args, expected_ct);
    const auto a = device_->worker_core_from_logical_core({0, 0});
    const auto b = device_->worker_core_from_logical_core({1, 0});
    const std::vector<uint32_t> expected_sender{21, 23, 0, 1, a.x, a.y, a.x, a.y, b.x, b.y};
    const std::vector<uint32_t> expected_receiver{31, 33, 35, 2, a.x, a.y, 0, 0, 0, 0};
    const std::vector<uint32_t> expected_inactive{41, 43, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(attached.runtime_args[0].second, expected_sender);
    EXPECT_EQ(attached.runtime_args[1].second, expected_receiver);
    EXPECT_EQ(attached.runtime_args[2].second, expected_inactive);
    EXPECT_EQ(attached.buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.buffer_bindings[1].arg_idx, 1u);
    EXPECT_EQ(attached.common_buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.common_runtime_args, std::vector<uint32_t>{71});
    // Automatic resource assignment belongs to the descriptor, not the prepared family.
    EXPECT_EQ(emitted_semaphore(compile_args(family), wire::DATA_READY), 0u);
    EXPECT_EQ(emitted_semaphore(compile_args(family), wire::CONSUMER_READY), 1u);
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
    EXPECT_EQ(emitted_semaphore(desc.kernels[0].compile_time_args, wire::DATA_READY, 1), 5u);
    EXPECT_EQ(emitted_semaphore(desc.kernels[0].compile_time_args, wire::CONSUMER_READY, 1), UNUSED_SEM_ID);
    EXPECT_EQ(emitted_metadata(desc.kernels[0].compile_time_args, 1).family.flags, 0u);
    const auto runtime = desc.kernels[0].runtime_args;
    attach_absent(desc.kernels[0], "absent_mcast");
    EXPECT_EQ(desc.kernels[0].compile_time_args.back(), 0u);
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
    const std::vector<uint32_t> expected{97, 0x024E2E13u, 4, 5, 1, 0x024E2E13u, 0, 1, 1};
    EXPECT_EQ(desc.kernels[0].compile_time_args, expected);
    for (const auto& [core, args] : desc.kernels[0].runtime_args) {
        ASSERT_EQ(args.size(), 14u);
        EXPECT_TRUE(std::equal(args.begin(), args.begin() + 7, args.begin() + 7));
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
    const std::vector<uint32_t> expected{0x000E1293u, 0, 1, 2};
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
        kernel.advanced_options.compile_time_varargs = {101, 103};
        spec.kernels.push_back(std::move(kernel));
    }
    spec.work_units = {
        {.name = "sender", .kernels = {m2::KernelSpecName{"sender"}}, .target_nodes = CoreCoord{0, 0}},
        {.name = "receiver", .kernels = {m2::KernelSpecName{"receiver"}}, .target_nodes = grid({1, 0}, {2, 0})}};
    return spec;
}
const std::array spec_targets{m2::KernelSpecName{"sender"}, m2::KernelSpecName{"receiver"}};

TEST_F(McastHostFixture, NoHandshakeLeavesExchangeCreditAcrossConstructionPaths) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    for (const auto signal :
         {dataflow_kernel_lib::DataReadySignal::Flag, dataflow_kernel_lib::DataReadySignal::Counter}) {
        auto family = make_family(device_, {{participants, {{0, 0}}}}, {.handshake = false, .data_ready = signal});
        Program program;
        ProgramDescriptor descriptor;
        auto spec = spec_pair();
        for (uint32_t id = 0; id < 14; ++id) {
            program.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
            descriptor.semaphores.push_back({.id = id, .core_ranges = participants, .initial_value = 0});
            spec.semaphores.push_back(
                {.unique_id = m2::SemaphoreSpecName{"exchange_" + std::to_string(id)}, .target_nodes = participants});
        }
        KernelDescriptor kernel;
        kernel.core_ranges = participants;
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        const std::array targets{std::ref(kernel)};
        family.attach(descriptor, "payload", targets);
        ASSERT_EQ(descriptor.semaphores.size(), 15u);
        EXPECT_EQ(descriptor.semaphores.back().id, 14u);
        EXPECT_EQ(emitted_semaphore(kernel.compile_time_args, wire::CONSUMER_READY), UNUSED_SEM_ID);

        m2::ProgramRunArgs args;
        family.attach(spec, args, "payload", spec_targets);
        ASSERT_EQ(spec.semaphores.size(), 15u);
        for (const auto& target : spec.kernels) {
            EXPECT_EQ(target.semaphore_bindings.size(), 1u);
        }
        family.append_semaphores(program);
        ASSERT_EQ(program.impl().semaphores().size(), 15u);
        EXPECT_EQ(program.impl().semaphores().back().id(), 14u);
        // The final hardware slot remains available for the operation's exchange credit.
        EXPECT_EQ(CreateSemaphore(program, participants, 0), 15u);
    }
}

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
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, i == 0 ? 6u : 5u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_rt_base").value(), 2u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_ct_base").value(), 2u);
        const auto decoded = decode_emitted_ct(kernel.advanced_options.compile_time_varargs, 2, false);
        EXPECT_EQ(decoded.metadata.family.flags, 1u);
        EXPECT_EQ(decoded.metadata.family.rectangle_capacity, 1u);
        EXPECT_EQ(decoded.metadata.family.uniform_remote_count, i == 0 ? 1u : 0u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs.size(), i == 0 ? 4u : 3u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs[0], 101u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs[1], 103u);
        EXPECT_EQ(kernel.compile_time_args.size(), 3u);  // Existing arg plus CT/RT offsets.
        EXPECT_EQ(kernel.compile_time_args.get("kept_ct").value(), 73u);
        EXPECT_EQ(
            kernel.compiler_options.defines.get("channel_mcast_data_ready_type").value(),
            "sem::channel_mcast_data_ready_t");
        EXPECT_EQ(kernel.compiler_options.defines.get("channel_mcast_signal_source_type").value(), "std::nullptr_t");
        EXPECT_EQ(args.kernel_run_args[i].common_runtime_arg_values.get("kept_common").value(), 97u);
    }
    const std::vector<uint32_t> sender_expected{41, 51, sender.x, sender.y, end.x, end.y};
    EXPECT_EQ(args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value(), sender_expected);
    EXPECT_EQ(args.kernel_run_args[1].runtime_arg_values.get("kept_rt").value().get({1, 0}).value(), 84u);
    const auto& inactive = args.kernel_run_args[1].advanced_options.runtime_varargs.get({2, 0}).value();
    EXPECT_EQ(inactive, (std::vector<uint32_t>{43, 53, 0, 0, 0}));
}

TEST_F(McastHostFixture, SpecAttachComposesAndNativeRunArgsCopiesKeepPayloads) {
    auto family = make_family(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair();
    m2::ProgramRunArgs args;
    family.attach(spec, args, "first", spec_targets);
    const auto payload = args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value();
    family.attach(spec, args, "second", spec_targets);
    EXPECT_EQ(spec.semaphores.size(), 4u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("second_mcast_rt_base").value(), 4u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("first_mcast_ct_base").value(), 2u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("second_mcast_ct_base").value(), 4u);
    EXPECT_EQ(spec.kernels[1].compile_time_args.get("second_mcast_ct_base").value(), 3u);
    EXPECT_EQ(spec.kernels[0].advanced_options.num_runtime_varargs, 8u);
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
    for (const auto violation :
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
            spec.kernels[1].compile_time_args["channel_mcast_ct_base"] = 55;
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
            EXPECT_EQ(
                spec.kernels[i].advanced_options.compile_time_varargs,
                before_spec.kernels[i].advanced_options.compile_time_varargs)
                << violation;
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
    EXPECT_EQ(spec.kernels[1].advanced_options.num_runtime_varargs, 5u);
    EXPECT_TRUE(spec.kernels[1].advanced_options.num_runtime_varargs_per_node.empty());
    EXPECT_EQ(spec.kernels[1].compile_time_args.get("channel_mcast_rt_base").value(), 2u);
}

TEST_F(McastHostFixture, SpecAbsentNeedsNoResourcesOrRunArgumentObject) {
    auto spec = spec_pair(3);
    attach_absent(spec, "none", spec_targets);
    EXPECT_TRUE(spec.semaphores.empty());
    for (const auto& kernel : spec.kernels) {
        EXPECT_EQ(kernel.compile_time_args.get("none_mcast_ct_base").value(), 2u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs, (std::vector<uint32_t>{101, 103, 0}));
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
    const std::vector<CoreCoord> active = local   ? std::vector<CoreCoord>{{0, 0}}
                                          : chain ? std::vector<CoreCoord>{{0, 0}, {1, 0}, {0, 1}}
                                                  : std::vector<CoreCoord>{{0, 0}, {1, 0}, {2, 0}};
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
                const uint32_t expected = control ? (counter ? round + 1 : 1) : 136 * (seed + round * 100) + 1360;
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

TEST_F(McastHostFixture, SpecDeviceOldTag) {
    using namespace tt::tt_metal;
    const std::array targets{m2::KernelSpecName{"old_tag"}};
    m2::ProgramSpec spec{
        .kernels =
            {{.unique_id = targets.front(),
              .source = m2::KernelSpec::SourceCode{R"(
                #include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"
                void kernel_main() { constexpr auto channel = MCAST_SPEC_ARGS(channel); }
            )"},
              .hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}}},
        .work_units = {{.name = "old_tag", .kernels = {targets.front()}, .target_nodes = CoreCoord{0, 0}}}};
    auto family = make_family(device_, {{grid({0, 0}, {1, 0}), {{0, 0}}}});
    m2::ProgramRunArgs args;
    family.attach(spec, args, "channel", targets);
    const auto ct_base = spec.kernels.front().compile_time_args.get("channel_mcast_ct_base").value();
    spec.kernels.front().advanced_options.compile_time_varargs[ct_base] = 2;
    try {
        auto workload = m2::MakeMeshWorkloadFromSpec(*device_, spec);
        for (auto& [region, program] : workload.get_programs()) {
            m2::SetProgramRunArgs(program, args);
        }
        tt::tt_metal::distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, true);
        FAIL() << "The native decoder accepted an obsolete multicast tag";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("Unsupported multicast wire tag"), std::string::npos);
    }
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
TEST_F(McastHostFixture, ProgramBindingAllocatesOnceAndAppendsResolvedIds) {
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
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY, 1), 1u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY, 1), 3u);
    EXPECT_EQ(rt.front(), 73u);
    EXPECT_GT(rt.size(), 1u);
    ASSERT_EQ(program.impl().semaphores().size(), 4u);
    EXPECT_EQ(program.impl().semaphores()[0].initial_value(), 7u);
    family.append_semaphores(program);
    auto copy = family;
    copy.append_semaphores(program);
    EXPECT_EQ(program.impl().semaphores().size(), 4u);
    Program other;
    EXPECT_ANY_THROW(copy.append_semaphores(other));
    EXPECT_TRUE(other.impl().semaphores().empty());
    Program moved = std::move(program);
    family.append_semaphores(moved);
    EXPECT_EQ(moved.impl().semaphores().size(), 4u);
    ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    EXPECT_ANY_THROW(family.attach(descriptor, "weights", std::array{std::ref(kernel)}));
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    EXPECT_ANY_THROW(family.attach(spec, run_args, "weights", {}));
}

TEST_F(McastHostFixture, ProgramBindingValidatesAllRolesBeforeMutationAndSupportsAdoption) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    Program collision;
    collision.impl().add_semaphore(participants, 3, 0, tt::CoreType::WORKER);
    auto exact = make_family(device_, {{participants, {{0, 0}}}}, {.base_sem_id = 2});
    EXPECT_ANY_THROW(exact.append_semaphores(collision));
    ASSERT_EQ(collision.impl().semaphores().size(), 1u);
    std::vector<uint32_t> ct;
    EXPECT_ANY_THROW(exact.append_compile_time_args_to(ct));
    Program program;
    exact.append_semaphores(program);
    exact.append_compile_time_args_to(ct);
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY), 2u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY), 3u);

    Mcast2D passive(
        device_,
        participants,
        Mcast2DFixedSenderConfig{{0, 0}},
        {.handshake = false, .sem_ids = std::vector<uint32_t>{2}});
    passive.append_semaphores(program);
    ct.clear();
    passive.append_compile_time_args_to(ct);
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY), 2u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY), UNUSED_SEM_ID);
    EXPECT_EQ(program.impl().semaphores().size(), 2u);

    Mcast1D another(device_, participants, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{});
    another.append_semaphores(program);
    ct.clear();
    another.append_compile_time_args_to(ct);
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY), 0u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY), 1u);
    EXPECT_EQ(program.impl().semaphores().size(), 4u);

    auto missing = make_family(device_, {{participants, {{0, 0}}}}, {.sem_ids = std::vector<uint32_t>{4, 5}});
    EXPECT_ANY_THROW(missing.append_semaphores(program));
    EXPECT_EQ(program.impl().semaphores().size(), 4u);
    Program nonzero;
    nonzero.impl().add_semaphore(participants, 4, 1, tt::CoreType::WORKER);
    nonzero.impl().add_semaphore(participants, 5, 0, tt::CoreType::WORKER);
    EXPECT_ANY_THROW(missing.append_semaphores(nonzero));
    EXPECT_EQ(nonzero.impl().semaphores().size(), 2u);
}

TEST_F(McastHostFixture, ProgramBindingCoversChainExhaustionAndUnsupportedPrograms) {
    using namespace tt::tt_metal;
    auto chain = make_family(device_, {{cores({{0, 0}, {1, 1}, {2, 0}}), {{0, 0}}}}, chain_config());
    Program chain_program;
    chain.append_semaphores(chain_program);
    std::vector<uint32_t> ct;
    chain.append_compile_time_args_to(ct);
    ASSERT_EQ(chain_program.impl().semaphores().size(), 3u);
    EXPECT_EQ(emitted_semaphore(ct, wire::SIGNAL_SOURCE), 2u);

    const auto participants = grid({0, 0}, {1, 0});
    auto family = make_family(device_, {{participants, {{0, 0}}}});
    Program full;
    for (uint32_t id = 0; id + 1 < NUM_SEMAPHORES; ++id) {
        full.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
    }
    EXPECT_ANY_THROW(family.append_semaphores(full));
    EXPECT_EQ(full.impl().semaphores().size(), NUM_SEMAPHORES - 1);
    Program partial;
    partial.impl().add_semaphore(grid({0, 0}, {0, 0}), 0, 0, tt::CoreType::WORKER);
    partial.impl().add_semaphore(participants, 1, 0, tt::CoreType::WORKER);
    auto adopted = make_family(device_, {{participants, {{0, 0}}}}, {.sem_ids = std::vector<uint32_t>{0, 1}});
    EXPECT_ANY_THROW(adopted.append_semaphores(partial));
    EXPECT_EQ(partial.impl().semaphores().size(), 2u);

    Program spec_program;
    spec_program.impl().mark_created_from_spec();
    EXPECT_ANY_THROW(family.append_semaphores(spec_program));
    EXPECT_TRUE(spec_program.impl().semaphores().empty());
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
