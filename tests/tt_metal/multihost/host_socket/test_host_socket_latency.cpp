// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Latency for HostMeshSocket, measured without any cross-host clock sync.
//
// Two measurements, each timed on a single clock:
//
//  1. Idle device-to-device round trip. Rank 0 sends a page and waits for rank 1
//     to echo it, timing the pair with rank 0's own Tensix clock. Half of that is
//     the one-way transit estimate, assuming the two directions are symmetric.
//
//  2. The streaming workload's ack round trip, timed on the sending host: from
//     handing a batch to the NIC until the peer credits it, i.e. out through the
//     far host, into the far device, and back. Subtracting the idle round trip's
//     half removes the return transit and leaves the forward path under load --
//     so the gap between that and the idle half is queueing delay, not distance.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include "host_socket_test_utils.hpp"

#include <algorithm>
#include <vector>

namespace tt::tt_metal::distributed::host_socket_test {
namespace {

// Round-trip latency: one page out, the same page back, timed on the device.
TEST(HostSocketLatencyTest, RoundTrip) {
    const auto context = multihost::DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2) << "needs exactly 2 ranks";

    Params params = params_from_env();
    const uint32_t page_size = params.page_size;
    const uint32_t fifo_size = params.fifo_size();
    const uint32_t iterations = static_cast<uint32_t>(env_or("TT_HOST_SOCKET_LAT_ITERS", 200));

    auto device = MeshDevice::create_unit_mesh(static_cast<int>(params.device_id));
    if (!host_sockets_supported(device)) {
        GTEST_SKIP() << "pinned host memory cannot be mapped to the NOC; enable vIOMMU";
    }

    const MeshCoordinate coord(0, 0);
    const CoreCoord core(0, 0);
    SocketConnection connection{MeshCoreCoord(coord, core), MeshCoreCoord(coord, core)};
    SocketMemoryConfig mem_config(BufferType::L1, fifo_size);

    HostMeshSocket::TransportConfig transport;
    transport.page_size = page_size;
    // Depth 1 on the wire: a latency probe should not batch.
    transport.max_batch_pages = 1;

    // Two sockets, forward and reverse. Constructed in the same order on both
    // ranks so their handshakes cannot be confused for one another.
    SocketConfig forward({connection}, mem_config, kSenderRank, kReceiverRank, context);
    SocketConfig reverse({connection}, mem_config, kReceiverRank, kSenderRank, context);
    HostMeshSocket fwd(device, forward, transport);
    HostMeshSocket rev(device, reverse, transport);

    const bool is_initiator = context->rank() == kSenderRank;
    auto payload = make_core_l1_buffer(device.get(), core, page_size);
    auto measurement = make_core_l1_buffer(device.get(), core, iterations * sizeof(uint64_t));

    auto program = CreateProgram();
    if (is_initiator) {
        CreateKernel(
            program,
            kPingPongKernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(fwd.get_config_buffer_address()),
                    static_cast<uint32_t>(rev.get_config_buffer_address()),
                    page_size,
                    static_cast<uint32_t>(payload->address()),
                    static_cast<uint32_t>(measurement->address()),
                    iterations,
                    kModeHostTransport,
                }});
    } else {
        CreateKernel(
            program,
            kEchoKernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(fwd.get_config_buffer_address()),
                    static_cast<uint32_t>(rev.get_config_buffer_address()),
                    page_size,
                    static_cast<uint32_t>(payload->address()),
                    iterations,
                    kModeHostTransport,
                }});
    }

    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(coord), std::move(program));
    context->barrier();
    EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
    Finish(device->mesh_command_queue());
    context->barrier();

    if (!is_initiator) {
        return;
    }

    std::vector<uint32_t> raw(iterations * sizeof(uint64_t) / sizeof(uint32_t));
    ReadShard(device->mesh_command_queue(), raw, measurement, coord, true);
    const auto* cycles = reinterpret_cast<const uint64_t*>(raw.data());
    const double cycles_per_us = get_cycles_per_us(*device);
    ASSERT_GT(cycles_per_us, 0.0);

    std::vector<double> rtt_us;
    rtt_us.reserve(iterations);
    for (uint32_t i = 0; i < iterations; i++) {
        rtt_us.push_back(static_cast<double>(cycles[i]) / cycles_per_us);
    }
    const auto rtt = summarize(rtt_us);

    GTEST_LOG_(INFO) << "round trip @" << page_size << " B: min " << rtt.min << " p50 " << rtt.p50 << " avg " << rtt.avg
                     << " p99 " << rtt.p99 << " max " << rtt.max << " us (" << rtt.count << " samples)";
    GTEST_LOG_(INFO) << "one-way estimate (p50/2): " << rtt.p50 / 2.0 << " us";
    record_latency("d2d_round_trip", page_size, rtt);

    Percentiles one_way = rtt;
    one_way.min /= 2.0;
    one_way.p50 /= 2.0;
    one_way.avg /= 2.0;
    one_way.p99 /= 2.0;
    one_way.max /= 2.0;
    record_latency("d2d_one_way_estimate", page_size, one_way);
}

// Ack round trip under a streaming load, timed on the sending host, plus the
// forward-path figure that remains once the idle return transit is removed.
TEST(HostSocketLatencyTest, StreamingAckLatency) {
    const auto context = multihost::DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2) << "needs exactly 2 ranks";

    // One long iteration on purpose. With several iterations the relay also
    // samples across a kernel relaunch, where the far device is not consuming at
    // all, and those samples measure "wait for the next launch" -- hundreds of
    // milliseconds -- which swamps the average. A single streaming pass keeps
    // every sample in steady state.
    Params params = params_from_env(Params{
        .page_size = 14336,
        .fifo_pages = 64,
        .num_cores = 1,
        .bytes_per_core = 14336ull * 4096,
        .iterations = 1,
    });
    const double idle_half_us = env_or_double("TT_HOST_SOCKET_IDLE_RTT_US", 0.0) / 2.0;

    double gbps = 0.0;
    std::vector<uint64_t> samples;
    run_transfer(params, /*verify=*/false, &gbps, &samples);

    if (context->rank() != kSenderRank || samples.empty()) {
        return;
    }
    std::vector<double> us;
    us.reserve(samples.size());
    for (uint64_t ns : samples) {
        us.push_back(static_cast<double>(ns) / 1000.0);
    }
    // Drop the opening batches: the ring starts empty, so the first credits come
    // back without the queueing the steady state has.
    const size_t skip = std::min<size_t>(us.size() / 10, 16);
    us.erase(us.begin(), us.begin() + static_cast<long>(skip));
    const auto ack = summarize(us);
    GTEST_LOG_(INFO) << "streaming ack round trip @" << params.page_size << " B: min " << ack.min << " p50 " << ack.p50
                     << " avg " << ack.avg << " p99 " << ack.p99 << " max " << ack.max << " us (" << ack.count
                     << " samples, " << gbps << " GB/s)";
    record_latency("stream_ack_round_trip", params.page_size, ack);

    if (idle_half_us > 0.0) {
        Percentiles fwd = ack;
        fwd.min -= idle_half_us;
        fwd.p50 -= idle_half_us;
        fwd.avg -= idle_half_us;
        fwd.p99 -= idle_half_us;
        fwd.max -= idle_half_us;
        GTEST_LOG_(INFO) << "forward path under load (ack - idle_rtt/2 = " << idle_half_us << " us): p50 " << fwd.p50
                         << " avg " << fwd.avg << " us";
        record_latency("stream_forward_under_load", params.page_size, fwd);
    } else {
        GTEST_LOG_(INFO) << "set TT_HOST_SOCKET_IDLE_RTT_US (from the RoundTrip test) to also report the "
                            "forward-path figure with the return transit removed";
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_socket_test
