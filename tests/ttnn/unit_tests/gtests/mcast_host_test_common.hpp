// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <vector>
#include "ttnn/kernel_lib/mcast/host/mcast_host_impl.hpp"
#include "ttnn/kernel_lib/mcast/mcast_compile_time_args.hpp"
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
namespace m2 = tt::tt_metal::experimental;
using dataflow_kernel_lib::SenderMcastMode;
using dataflow_kernel_lib::TransferMode;
class McastHostFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastHostFixture> {};

inline CoreRangeSet grid(CoreCoord start, CoreCoord end) { return CoreRangeSet(CoreRange(start, end)); }
inline CoreRangeSet cores(const std::vector<CoreCoord>& values) {
    std::vector<CoreRange> ranges;
    ranges.reserve(values.size());
    for (auto core : values) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(std::move(ranges));
}
inline McastConfig chain_config(McastConfig cfg = {}) {
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

inline McastImpl make_mcast(
    tt::tt_metal::IDevice* device, const std::vector<GroupInput>& inputs, const McastConfig& cfg = {}) {
    McastImpl mcast(*device, cfg);
    for (const auto& input : inputs) {
        mcast.add_group(input.receivers, input.senders, input.ack);
    }
    return mcast;
}

// Golden wire tests use the regular Program path. Existing resources may be seeded
// to verify that automatic allocation skips occupied IDs.
template <typename McastType>
void bind_for_inspection(McastType& mcast, tt::tt_metal::Program& program, const std::vector<uint32_t>& existing) {
    if constexpr (requires { mcast.participating_cores(); }) {
        for (auto id : existing) {
            program.impl().add_semaphore(mcast.participating_cores(), id, 0, tt::CoreType::WORKER);
        }
    } else {
        TT_FATAL(existing.empty(), "Argument allocation fixtures must provide explicit placement");
    }
    mcast.append_semaphores(program);
}

template <typename McastType>
std::vector<uint32_t> compile_args(McastType mcast, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(mcast, program, existing);
    std::vector<uint32_t> args;
    mcast.append_compile_time_args_to(args);
    return args;
}

template <typename McastType>
std::vector<uint32_t> runtime_args(McastType mcast, CoreCoord core, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(mcast, program, existing);
    std::vector<uint32_t> args;
    mcast.append_runtime_args_to(args, core);
    return args;
}

struct DecodedCompileTime {
    wire::ArgumentMetadata metadata;
    std::array<uint32_t, 3> semaphores{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    uint32_t words = 1;
};

inline DecodedCompileTime decode_emitted_ct(const std::vector<uint32_t>& ct, uint32_t base = 0, bool ids = true) {
    // Independent literal v3 decoder; complete-block goldens below pin the bits
    // and optional-field order without using the production codec or its layout.
    DecodedCompileTime result;
    const uint32_t control = ct.at(base);
    if (control == 0) {
        return result;
    }
    EXPECT_EQ(control & 15u, 3u);
    auto& m = result.metadata;
    m.mcast.flags = (control >> 4) & 31u;
    m.mcast.has_remote_receivers = (control >> 9) & 1u;
    m.mcast.sender_mcast_mode = SenderMcastMode((control >> 10) & 7u);
    m.mcast.rectangle_capacity = (control >> 13) & 3u;
    m.kernel.roles = control & (1u << 17) ? 0xFFFFFFFFu : (control >> 15) & 3u;
    m.kernel.capabilities = (control >> 18) & 3u;
    m.coordinates.encoding = wire::SenderCoordinateEncoding((control >> 20) & 3u);
    auto next = [&]() { return ct.at(base + result.words++); };
    if (ids) {
        result.semaphores[0] = next();
        if (m.mcast.flags & 1u) {
            result.semaphores[1] = next();
        }
        if ((m.mcast.flags >> 3) & 3u) {
            result.semaphores[2] = next();
        }
    }
    m.mcast.remote_count_known = (control >> 22) & 1u;
    if (m.mcast.remote_count_known) {
        m.mcast.uniform_remote_count = next();
    }
    const uint32_t ack = (control >> 24) & 3u;
    switch (ack) {
        case 1: m.mcast.ack_count = next(); break;
        case 2: m.mcast.ack_count = m.mcast.uniform_remote_count; break;
        case 3: m.mcast.ack_count = 0xFFFFFFFFu; break;
        default: m.mcast.ack_count = 0u; break;
    }
    if (control & (1u << 23)) {
        m.mcast.rotating_span = next();
    }
    if (uint32_t(m.coordinates.encoding) != 0) {
        m.coordinates.columns = next();
        m.coordinates.rows = next();
        m.coordinates.x_ranges = next();
        m.coordinates.y_ranges = next();
    }
    return result;
}

inline wire::ArgumentMetadata emitted_metadata(const std::vector<uint32_t>& ct, uint32_t base = 0) {
    return decode_emitted_ct(ct, base).metadata;
}

inline uint32_t emitted_semaphore(const std::vector<uint32_t>& ct, wire::SemaphoreRole role, uint32_t base = 0) {
    return decode_emitted_ct(ct, base).semaphores[role];
}

template <typename McastType>
std::vector<tt::tt_metal::SemaphoreDescriptor> allocated_semaphores(
    McastType mcast, const std::vector<uint32_t>& existing = {}) {
    tt::tt_metal::Program program;
    bind_for_inspection(mcast, program, existing);
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

inline m2::ProgramSpec spec_pair(uint32_t prefix = 0) {
    m2::ProgramSpec spec;
    for (const auto& name : {"sender", "receiver"}) {
        m2::KernelSpec kernel{
            .unique_id = m2::KernelSpecName{name},
            .source = m2::KernelSpec::SourceCode{"void kernel_main() {}"},
            .compile_time_args = {{"kept_ct", 73}},
            .runtime_arg_schema = {.runtime_arg_names = {"kept_rt"}, .common_runtime_arg_names = {"kept_common"}},
            .hw_config = m2::CreateReaderDataMovementConfig()};
        kernel.advanced_options.num_runtime_varargs = prefix;
        kernel.advanced_options.compile_time_varargs = {101, 103};
        spec.kernels.push_back(std::move(kernel));
    }
    spec.work_units = {
        {.name = "sender", .kernels = {m2::KernelSpecName{"sender"}}, .target_nodes = CoreCoord{0, 0}},
        {.name = "receiver", .kernels = {m2::KernelSpecName{"receiver"}}, .target_nodes = grid({1, 0}, {2, 0})}};
    return spec;
}
inline const std::array spec_targets{m2::KernelSpecName{"sender"}, m2::KernelSpecName{"receiver"}};

}  // namespace ttnn::kernel_lib::host::test
