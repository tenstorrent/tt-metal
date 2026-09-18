// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_socket_test_utils.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <span>
#include <string_view>

namespace tt::tt_metal::distributed::host_socket_test {

namespace {

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    return std::strtoull(raw, nullptr, 0);
}

// One buffer sharded across the cores, one page each. A buffer per core would
// reserve its range on *every* core under the default allocator and exhaust L1.
std::shared_ptr<MeshBuffer> make_sharded_l1_buffer(MeshDevice* device, uint32_t num_cores, uint32_t page_size) {
    auto grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)));
    auto shard = ShardSpecBuffer(grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    const DeviceLocalBufferConfig local{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return MeshBuffer::create(
        ReplicatedBufferConfig{.size = static_cast<DeviceAddr>(page_size) * num_cores}, local, device);
}

std::vector<SocketConnection> make_connections(uint32_t num_cores) {
    std::vector<SocketConnection> connections;
    connections.reserve(num_cores);
    const MeshCoordinate device_coord(0, 0);
    for (uint32_t i = 0; i < num_cores; i++) {
        connections.emplace_back(
            MeshCoreCoord(device_coord, CoreCoord(i, 0)), MeshCoreCoord(device_coord, CoreCoord(i, 0)));
    }
    return connections;
}

}  // namespace

std::shared_ptr<MeshBuffer> make_core_l1_buffer(MeshDevice* device, const CoreCoord& core, uint32_t size) {
    auto shard =
        ShardSpecBuffer(CoreRangeSet(CoreRange(core, core)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig local{
        .page_size = size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return MeshBuffer::create(ReplicatedBufferConfig{.size = size}, local, device);
}

uint64_t env_or(const char* name, uint64_t fallback) { return env_u64(name, fallback); }

double env_or_double(const char* name, double fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    return std::strtod(raw, nullptr);
}

double get_cycles_per_us(const MeshDevice& mesh_device) {
    const auto chip_id = mesh_device.get_device(MeshCoordinate(0, 0))->id();
    return static_cast<double>(MetalContext::instance().get_cluster().get_device_aiclk(chip_id));
}

Params params_from_env(Params defaults) {
    Params p = defaults;
    p.page_size = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_PAGE_SIZE", p.page_size));
    p.fifo_pages = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_FIFO_PAGES", p.fifo_pages));
    p.num_cores = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_NUM_CORES", p.num_cores));
    p.bytes_per_core = env_u64("TT_HOST_SOCKET_BYTES", p.bytes_per_core);
    p.device_id = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_DEVICE_ID", p.device_id));
    p.iterations = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_ITERS", p.iterations));
    p.min_seconds = static_cast<double>(env_u64("TT_HOST_SOCKET_SOAK_SECONDS", 0));
    return p;
}

bool host_sockets_supported(const std::shared_ptr<MeshDevice>& mesh_device) {
    return experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc;
}

std::vector<uint32_t> payload_for_core(uint32_t core_index, uint64_t size_bytes) {
    std::vector<uint32_t> data(size_bytes / sizeof(uint32_t));
    // Distinct per core and per word: a page delivered to the wrong place must
    // not verify as correct.
    uint32_t word = 0x1000u * (core_index + 1);
    for (auto& v : data) {
        v = word;
        word = word * 1664525u + 1013904223u;
    }
    return data;
}

void run_transfer(const Params& params, bool verify, double* gbps_out, std::vector<uint64_t>* ack_latency_ns_out) {
    const auto context = multihost::DistributedContext::get_current_world();
    const auto rank = context->rank();
    const bool is_sender = rank == kSenderRank;

    auto mesh_device = MeshDevice::create_unit_mesh(static_cast<int>(params.device_id));
    if (!host_sockets_supported(mesh_device)) {
        GTEST_SKIP() << "pinned host memory cannot be mapped to the NOC; enable vIOMMU";
    }

    const uint32_t page_size = params.page_size;
    const uint32_t fifo_size = params.fifo_size();
    const uint64_t data_size = params.data_size();
    TT_FATAL(data_size % page_size == 0, "data_size must be a whole number of pages");

    SocketMemoryConfig mem_config(BufferType::L1, fifo_size);
    SocketConfig config(make_connections(params.num_cores), mem_config, kSenderRank, kReceiverRank, context);

    HostMeshSocket::TransportConfig transport;
    transport.page_size = page_size;
    transport.max_batch_pages = static_cast<uint32_t>(env_u64("TT_HOST_SOCKET_BATCH_PAGES", 8));

    HostMeshSocket socket(mesh_device, config, transport);
    if (ack_latency_ns_out != nullptr) {
        socket.set_latency_sampling(true);
    }

    const uint32_t buffer_size = static_cast<uint32_t>(verify ? data_size : page_size);
    const MeshCoordinate device_coord(0, 0);
    auto data_buffer = make_sharded_l1_buffer(mesh_device.get(), params.num_cores, buffer_size);

    const size_t words_per_core = buffer_size / sizeof(uint32_t);
    std::vector<uint32_t> staging(words_per_core * params.num_cores);
    for (uint32_t i = 0; i < params.num_cores; i++) {
        const auto payload = payload_for_core(i, buffer_size);
        for (size_t w = 0; w < words_per_core; w++) {
            // Receiver starts from the complement, so a page that never arrives
            // cannot pass.
            staging[i * words_per_core + w] = is_sender ? payload[w] : ~payload[w];
        }
    }
    WriteShard(mesh_device->mesh_command_queue(), data_buffer, staging, device_coord, true);

    auto program = CreateProgram();
    for (uint32_t i = 0; i < params.num_cores; i++) {
        const std::vector<uint32_t> compile_args{
            static_cast<uint32_t>(socket.get_config_buffer_address()),
            static_cast<uint32_t>(data_buffer->address()),
            page_size,
            static_cast<uint32_t>(data_size),
            kModeHostTransport,
            buffer_size,
        };
        CreateKernel(
            program,
            is_sender ? kSenderKernel : kReceiverKernel,
            CoreCoord(i, 0),
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args});
    }

    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(device_coord), std::move(program));

    // Launch together so the measurement covers transfer only.
    context->barrier();
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    uint64_t completed = 0;

    // Socket counters persist in L1, so re-enqueueing continues the same stream.
    for (uint64_t iter = 0;; iter++) {
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        Finish(mesh_device->mesh_command_queue());
        socket.barrier();
        completed++;

        if (verify && !is_sender) {
            std::vector<uint32_t> actual(staging.size());
            ReadShard(mesh_device->mesh_command_queue(), actual, data_buffer, device_coord, true);
            for (uint32_t i = 0; i < params.num_cores; i++) {
                const auto expected = payload_for_core(i, buffer_size);
                const auto* got = actual.data() + i * words_per_core;
                for (size_t w = 0; w < words_per_core; w++) {
                    ASSERT_EQ(got[w], expected[w]) << "core (" << i << ",0) word " << w << ", iteration " << iter;
                }
            }
            // Restore the complement for the next iteration.
            WriteShard(mesh_device->mesh_command_queue(), data_buffer, staging, device_coord, true);
        }

        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        // Both ranks must agree when to stop; the sender's verdict wins.
        uint32_t keep_going = (completed < params.iterations || elapsed < params.min_seconds) ? 1u : 0u;
        context->broadcast(
            std::span<std::byte>(reinterpret_cast<std::byte*>(&keep_going), sizeof(keep_going)), kSenderRank);
        if (keep_going == 0u) {
            break;
        }
    }
    context->barrier();

    if (ack_latency_ns_out != nullptr) {
        *ack_latency_ns_out = socket.take_latency_samples_ns();
    }

    const double total_bytes = static_cast<double>(data_size) * params.num_cores * static_cast<double>(completed);
    if (gbps_out != nullptr) {
        *gbps_out = (is_sender && elapsed > 0.0) ? total_bytes / elapsed / 1e9 : 0.0;
    }
}

Percentiles summarize(std::vector<double> samples) {
    Percentiles out;
    if (samples.empty()) {
        return out;
    }
    std::sort(samples.begin(), samples.end());
    out.count = samples.size();
    out.min = samples.front();
    out.max = samples.back();
    out.p50 = samples[samples.size() / 2];
    out.p99 = samples[std::min(samples.size() - 1, static_cast<size_t>(samples.size() * 99 / 100))];
    out.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    return out;
}

void record_latency(const std::string& label, uint32_t page_size, const Percentiles& us) {
    const char* path = std::getenv("TT_HOST_SOCKET_CSV_LATENCY");
    if (path == nullptr || *path == '\0') {
        return;
    }
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    if (fresh) {
        out << "label,page_size,samples,min_us,p50_us,avg_us,p99_us,max_us\n";
    }
    out << label << ',' << page_size << ',' << us.count << ',' << us.min << ',' << us.p50 << ',' << us.avg << ','
        << us.p99 << ',' << us.max << '\n';
}

void record_result(const Params& params, double gbps) {
    const char* path = std::getenv("TT_HOST_SOCKET_CSV");
    if (path == nullptr || *path == '\0') {
        return;
    }
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    if (fresh) {
        out << "page_size,fifo_pages,num_cores,bytes_per_core,iterations,gbps\n";
    }
    out << params.page_size << ',' << params.fifo_pages << ',' << params.num_cores << ',' << params.data_size() << ','
        << params.iterations << ',' << gbps << '\n';
}

}  // namespace tt::tt_metal::distributed::host_socket_test
