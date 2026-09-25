// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Times uploading file-backed host memory to device DRAM the way ttnn.load_tensor does: mmap the file read-only,
// then write it to a DRAM buffer. Compares the copy path, a single whole-file pin, and the production tensor upload
// (tt_metal/impl/tensor/pinned_upload.cpp), which pins in chunks from a worker pool.
//
// Usage: bench_pinned_file_upload <mode> <cold|warm> <file>...
//   mode:
//     copy            non-pinned enqueue_write_shards (the command queue copies the data)
//     pin             PinnedMemory::Create(ReadOnly) of the whole file, blocking pinned write, unpin
//     tensor          MeshCommandQueue::enqueue_write_tensor of the mapping marked device-immutable, then finish()
//     tensor-mutable  like tensor, without the device-immutable mark (the upload blocks until the device read it)
//   cold: posix_fadvise(DONTNEED) each file before its upload. warm: read each file fully before its upload.
//   Files must be a multiple of 4 KiB. VERIFY=1 reads each upload back and compares it with the file.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fmt/base.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/memory_pin_access.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Clock = std::chrono::steady_clock;

namespace {

// One 4 KiB page per row of 1024 uint32, so every mode writes the same page size.
constexpr uint32_t k_row_words = 1024;
constexpr size_t k_row_bytes = k_row_words * sizeof(uint32_t);

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

size_t file_size(const std::string& path) {
    struct stat st{};
    TT_FATAL(stat(path.c_str(), &st) == 0, "stat {} failed: {}", path, strerror(errno));
    return st.st_size;
}

void evict(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "open {} failed: {}", path, strerror(errno));
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

void warm(const std::string& path) {
    static std::vector<uint8_t> scratch(64 << 20);
    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "open {} failed: {}", path, strerror(errno));
    while (read(fd, scratch.data(), scratch.size()) > 0) {
    }
    close(fd);
}

std::shared_ptr<void> map_file(const std::string& path, size_t size) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd >= 0, "open {} failed: {}", path, strerror(errno));
    void* addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    TT_FATAL(addr != MAP_FAILED, "mmap {} failed: {}", path, strerror(errno));
    return std::shared_ptr<void>(addr, [size](void* a) { munmap(a, size); });
}

TensorSpec file_tensor_spec(size_t size) {
    return TensorSpec(
        tt::tt_metal::Shape{1, 1, static_cast<uint32_t>(size / k_row_bytes), k_row_words},
        TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
}

void verify(MeshCommandQueue& cq, const std::shared_ptr<MeshBuffer>& device_buffer, const void* expected, size_t size) {
    std::vector<uint8_t> readback(size);
    auto transfer = ShardDataTransfer(MeshCoordinate(0, 0)).host_data(readback.data()).region(BufferRegion(0, size));
    cq.enqueue_read_shards({transfer}, device_buffer, /*blocking=*/true);
    TT_FATAL(std::memcmp(readback.data(), expected, size) == 0, "readback mismatch");
}

void verify(MeshCommandQueue& cq, const MeshTensor& device_tensor, const void* expected, size_t size) {
    HostTensor result = cq.enqueue_read_tensor(device_tensor);
    auto shard = result.buffer().get_shard(MeshCoordinate(0, 0));
    TT_FATAL(shard->view_bytes().size() == size, "readback size {} != {}", shard->view_bytes().size(), size);
    TT_FATAL(std::memcmp(shard->view_bytes().data(), expected, size) == 0, "readback mismatch");
}

}  // namespace

int main(int argc, char** argv) {
    TT_FATAL(argc >= 4, "usage: {} <copy|pin|tensor|tensor-mutable> <cold|warm> <file>...", argv[0]);
    const std::string mode = argv[1];
    TT_FATAL(mode == "copy" || mode == "pin" || mode == "tensor" || mode == "tensor-mutable", "unknown mode {}", mode);
    const std::string cache = argv[2];
    TT_FATAL(cache == "cold" || cache == "warm", "cache must be cold or warm, got {}", cache);
    const std::vector<std::string> files(argv + 3, argv + argc);
    const bool check = std::getenv("VERIFY") != nullptr && std::string(std::getenv("VERIFY")) != "0";

    size_t max_size = 0;
    for (const auto& f : files) {
        const size_t size = file_size(f);
        TT_FATAL(size % k_row_bytes == 0, "{} is {} B, not a multiple of {} B", f, size, k_row_bytes);
        max_size = std::max(max_size, size);
    }

    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    const auto coord = MeshCoordinate(0, 0);
    std::shared_ptr<MeshBuffer> device_buffer;
    if (mode == "copy" || mode == "pin") {
        device_buffer = MeshBuffer::create(
            ReplicatedBufferConfig{max_size},
            DeviceLocalBufferConfig{.page_size = k_row_bytes, .buffer_type = BufferType::DRAM},
            mesh_device.get());
    }

    double total_ms = 0;
    size_t total_bytes = 0;
    for (const auto& path : files) {
        const size_t size = file_size(path);
        if (cache == "cold") {
            evict(path);
        } else {
            warm(path);
        }

        auto t = Clock::now();
        auto mapping = map_file(path, size);
        auto* data = static_cast<uint8_t*>(mapping.get());
        double pin_ms = 0;
        if (mode == "copy" || mode == "pin") {
            auto transfer = ShardDataTransfer(coord).host_data(data).region(BufferRegion(0, size));
            std::shared_ptr<experimental::PinnedMemory> pinned;
            if (mode == "pin") {
                HostBuffer host_buffer(ttsl::Span<uint8_t>(data, size), MemoryPin(mapping));
                const auto pin_start = Clock::now();
                pinned = experimental::PinnedMemory::Create(
                    *mesh_device,
                    MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord)),
                    host_buffer,
                    /*map_to_noc=*/true,
                    experimental::PinnedMemoryDeviceAccess::ReadOnly);
                experimental::HostBufferSetPinnedMemory(host_buffer, nullptr);
                pin_ms = ms_since(pin_start);
                experimental::ShardDataTransferSetPinnedMemory(transfer, pinned);
            }
            cq.enqueue_write_shards(device_buffer, {transfer}, /*blocking=*/true);
            transfer = ShardDataTransfer(coord);
            pinned.reset();
            const double ms = ms_since(t);
            if (check) {
                verify(cq, device_buffer, data, size);
            }
            fmt::print("file {} size_mb {:.1f} pin_ms {:.1f} total_ms {:.1f}\n", path, size / 1e6, pin_ms, ms);
            total_ms += ms;
        } else {
            MemoryPin pin(mapping);
            if (mode == "tensor") {
                experimental::MemoryPinMarkDeviceImmutable(pin);
            }
            auto dhb = DistributedHostBuffer::create(mesh_device->shape());
            dhb.emplace_shard(coord, [&]() { return HostBuffer(ttsl::Span<uint8_t>(data, size), pin); });
            auto host_tensor = host_tensor_from_buffer_with_topology(
                std::move(dhb),
                file_tensor_spec(size),
                TensorTopology::create_fully_replicated_tensor_topology(mesh_device->shape()));
            MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
            cq.finish();
            const double ms = ms_since(t);
            if (check) {
                verify(cq, device_tensor, data, size);
            }
            fmt::print("file {} size_mb {:.1f} total_ms {:.1f}\n", path, size / 1e6, ms);
            total_ms += ms;
        }
        if (check) {
            fmt::print("verified {}\n", path);
        }
        total_bytes += size;
    }
    fmt::print(
        "SUMMARY mode {} cache {} files {} bytes {} total_ms {:.1f} GBps {:.2f}\n",
        mode,
        cache,
        files.size(),
        total_bytes,
        total_ms,
        total_ms > 0 ? total_bytes / (total_ms * 1e6) : 0.0);

    device_buffer.reset();
    mesh_device->close();
    return 0;
}
