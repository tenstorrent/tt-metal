// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Burst latency: how long until the credit for the LAST of a burst of messages
// comes back, as a function of message size and burst size. Timed on the
// initiator's Tensix clock, so no cross-host clock sync is needed.
//
// Two credits are reported per burst:
//   local   bytes_acked covers the last page. The local host relay released the
//           ring slot, i.e. the NIC finished reading it. This is what a sender
//           kernel blocks on, so it is the back-pressure latency.
//   remote  the receiver kernel consumed the last page and its one-page ack has
//           arrived over the reverse socket. This is the end-to-end figure.
//
// The sweep is one page size per run (TT_HOST_SOCKET_PAGE_SIZE) over a list of
// burst sizes (TT_HOST_SOCKET_BURSTS), so the launcher can drive the outer loop.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include "host_socket_test_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace tt::tt_metal::distributed::host_socket_test {
namespace {

std::vector<uint32_t> bursts_from_env(uint32_t ring_pages) {
    std::vector<uint32_t> bursts;
    const char* raw = std::getenv("TT_HOST_SOCKET_BURSTS");
    if (raw != nullptr && *raw != '\0') {
        std::string s(raw);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t next = s.find(',', pos);
            if (next == std::string::npos) {
                next = s.size();
            }
            const auto v = static_cast<uint32_t>(std::strtoul(s.substr(pos, next - pos).c_str(), nullptr, 0));
            if (v != 0) {
                bursts.push_back(v);
            }
            pos = next + 1;
        }
    } else {
        for (uint32_t b = 1; b <= ring_pages; b *= 2) {
            bursts.push_back(b);
        }
    }
    // A burst must fit the ring, or socket_reserve_pages never returns.
    bursts.erase(
        std::remove_if(bursts.begin(), bursts.end(), [ring_pages](uint32_t b) { return b > ring_pages; }),
        bursts.end());
    return bursts;
}

TEST(HostSocketLatencyTest, BurstToLastCredit) {
    const auto& context = multihost::DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2) << "needs exactly 2 ranks";

    Params params = params_from_env(Params{.page_size = 14336, .fifo_pages = 64});
    const uint32_t page_size = params.page_size;
    // The receiver mirrors D2D and allocates fifo_size of L1 as a landing buffer,
    // so the ring is capped in bytes; bursts that no longer fit are dropped.
    const uint32_t ring_cap = static_cast<uint32_t>(env_or("TT_HOST_SOCKET_RING_CAP_BYTES", 1u << 20));
    const uint32_t ring_pages = std::max(1u, std::min(params.fifo_pages, ring_cap / page_size));
    const uint32_t fifo_size = page_size * ring_pages;
    const uint32_t iterations = static_cast<uint32_t>(env_or("TT_HOST_SOCKET_BURST_ITERS", 100));
    // The ack is the smallest page the D2H path accepts; it should cost as
    // little as possible so the remote figure is the forward path plus one
    // minimal return.
    const uint32_t ack_page_size = static_cast<uint32_t>(env_or("TT_HOST_SOCKET_ACK_PAGE_SIZE", 64));
    const uint32_t ack_fifo_size = ack_page_size * 16;
    const auto bursts = bursts_from_env(ring_pages);
    ASSERT_FALSE(bursts.empty());

    auto device = MeshDevice::create_unit_mesh(static_cast<int>(params.device_id));
    if (!host_sockets_supported(device)) {
        GTEST_SKIP() << "pinned host memory cannot be mapped to the NOC; enable vIOMMU";
    }

    const MeshCoordinate coord(0, 0);
    const CoreCoord core(0, 0);
    SocketConnection connection{MeshCoreCoord(coord, core), MeshCoreCoord(coord, core)};
    SocketMemoryConfig fwd_mem(BufferType::L1, fifo_size);
    SocketMemoryConfig rev_mem(BufferType::L1, ack_fifo_size);

    HostMeshSocket::TransportConfig fwd_transport;
    fwd_transport.page_size = page_size;
    fwd_transport.max_batch_pages = static_cast<uint32_t>(env_or("TT_HOST_SOCKET_BATCH_PAGES", 8));
    HostMeshSocket::TransportConfig rev_transport;
    rev_transport.page_size = ack_page_size;
    rev_transport.max_batch_pages = 1;

    // Same construction order on both ranks, else the handshakes cross.
    SocketConfig forward({connection}, fwd_mem, kSenderRank, kReceiverRank, context);
    SocketConfig reverse({connection}, rev_mem, kReceiverRank, kSenderRank, context);
    HostMeshSocket fwd(device, forward, fwd_transport);
    HostMeshSocket rev(device, reverse, rev_transport);
    // Host-side hop stamps for the same run, so the device-side figures can be
    // decomposed.
    fwd.set_latency_sampling(true);

    const bool is_initiator = context->rank() == kSenderRank;
    auto payload = make_core_l1_buffer(device.get(), core, page_size);
    auto ack = make_core_l1_buffer(device.get(), core, ack_page_size);
    const uint32_t num_bursts = static_cast<uint32_t>(bursts.size());
    const uint32_t measurement_bytes = num_bursts * iterations * 2 * sizeof(uint64_t);
    auto measurement = make_core_l1_buffer(device.get(), core, measurement_bytes);

    auto program = CreateProgram();
    KernelHandle kernel = 0;
    if (is_initiator) {
        kernel = CreateKernel(
            program,
            kBurstSenderKernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(fwd.get_config_buffer_address()),
                    static_cast<uint32_t>(rev.get_config_buffer_address()),
                    page_size,
                    ack_page_size,
                    static_cast<uint32_t>(payload->address()),
                    static_cast<uint32_t>(ack->address()),
                    static_cast<uint32_t>(measurement->address()),
                    iterations,
                    num_bursts,
                    kModeHostTransport,
                }});
    } else {
        kernel = CreateKernel(
            program,
            kBurstReceiverKernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(fwd.get_config_buffer_address()),
                    static_cast<uint32_t>(rev.get_config_buffer_address()),
                    page_size,
                    ack_page_size,
                    static_cast<uint32_t>(payload->address()),
                    static_cast<uint32_t>(ack->address()),
                    iterations,
                    num_bursts,
                    kModeHostTransport,
                }});
    }
    SetRuntimeArgs(program, kernel, core, bursts);

    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(coord), std::move(program));
    context->barrier();
    EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
    Finish(device->mesh_command_queue());
    context->barrier();

    // The receiver's hop samples live on the other rank; bring them over so one
    // rank reports everything.
    auto hops = gather_hop_samples(fwd, context, kSenderRank, kReceiverRank);

    if (!is_initiator) {
        return;
    }

    std::vector<uint32_t> raw(measurement_bytes / sizeof(uint32_t));
    ReadShard(device->mesh_command_queue(), raw, measurement, coord, true);
    const auto* cycles = reinterpret_cast<const uint64_t*>(raw.data());
    const double cycles_per_us = get_cycles_per_us(*device);
    ASSERT_GT(cycles_per_us, 0.0);

    GTEST_LOG_(INFO) << "burst to last credit @" << page_size << " B pages, ring " << ring_pages << " pages, "
                     << iterations << " iterations per burst, ack page " << ack_page_size << " B";
    for (uint32_t b = 0; b < num_bursts; b++) {
        std::vector<double> local_us;
        std::vector<double> remote_us;
        local_us.reserve(iterations);
        remote_us.reserve(iterations);
        for (uint32_t it = 0; it < iterations; it++) {
            const uint64_t* sample = cycles + (static_cast<size_t>(b) * iterations + it) * 2;
            local_us.push_back(static_cast<double>(sample[0]) / cycles_per_us);
            remote_us.push_back(static_cast<double>(sample[1]) / cycles_per_us);
        }
        const auto local = summarize(local_us);
        const auto remote = summarize(remote_us);
        const uint32_t burst = bursts[b];
        const double burst_bytes = static_cast<double>(page_size) * burst;
        GTEST_LOG_(INFO) << "  burst " << burst << " (" << burst_bytes / 1024.0 << " KiB):"
                         << " local p50 " << local.p50 << " p99 " << local.p99
                         << " | remote p50 " << remote.p50 << " p99 " << remote.p99 << " us"
                         << " | per msg " << remote.p50 / burst << " us"
                         << " | " << burst_bytes / remote.p50 / 1e3 << " GB/s";
        record_burst(page_size, burst, ring_pages, local, remote);
    }
    report_hop_samples("burst", page_size, hops);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_socket_test
