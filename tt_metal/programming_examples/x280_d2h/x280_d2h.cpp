// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// X280 -> host D2H socket prototype (host side).
//
// Two modes:
//   selftest  (default) -- runs the standard D2H flow with a *Tensix* sender
//                           kernel and verifies the host receives the data.
//                           This proves, on this exact box, that:
//                             * host pinned memory works (settles the vIOMMU
//                               question empirically -- ctor would TT_FATAL
//                               otherwise),
//                             * a device -> host NoC write through the PCIe
//                               tile lands in the pinned buffer,
//                             * host D2HSocket::read() drains it correctly.
//                           It is the safe de-risk for the X280 writer: the
//                           X280 just has to replicate what the Tensix sender
//                           kernel (pcie_socket_sender.cpp) does.
//
//   listen               -- creates the D2HSocket, prints the NoC target the
//                           X280 must write to (pcie_xy_enc + 64-bit host data
//                           addr + 64-bit host bytes_sent addr), the sender
//                           core's config-buffer L1 address/coords (so the X280
//                           can read the same metadata over NoC), exports the
//                           descriptor, then polls read() and prints whatever
//                           arrives. Run this during the X280 write window.
//
// On multi-chip boxes metal's logical device id != PCIe id. The X280 is booted
// on /dev/tenstorrent/N = PCIe id N, and its PCIe tile can only reach host
// memory mapped to its OWN physical chip, so open the logical id that maps to
// that PCIe id (X280 on PCIe 0 => logical device 3 on bh-qb-05).
//
// Launch with TT_METAL_SKIP_DRAM_TLBS=1 so device init doesn't collide with the
// tt-bh-linux console's DRAM TLB windows.

#include <fmt/ostream.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

// Config-buffer word layout on the sender core (see tt_metal/hw/inc/hostdev/socket.h
// and D2HSocket::write_socket_metadata). L1 alignment on Blackhole is 16 B, so:
//   sender_socket_md (7 u32) padded to 32 B  -> words [0..7]
//   bytes_acked[1]            padded to 16 B  -> words [8..11]
//   sender_downstream_encoding (3 u32) @ 48 B -> words [12..14]
// Fixed-part fields we care about:
constexpr uint32_t kWordWritePtr = 2;          // sender_socket_md.write_ptr
constexpr uint32_t kWordBytesSentAddrLo = 3;   // downstream_bytes_sent_addr
constexpr uint32_t kWordFifoAddrLo = 4;        // downstream_fifo_addr (host data buf, low 32)
constexpr uint32_t kWordFifoTotalSize = 5;     // downstream_fifo_total_size
constexpr uint32_t kWordIsD2H = 6;             // is_d2h
constexpr uint32_t kWordBytesSentAddrHi = 12;  // d2h.bytes_sent_addr_hi
constexpr uint32_t kWordDataAddrHi = 13;       // d2h.data_addr_hi
constexpr uint32_t kWordPcieXyEnc = 14;        // d2h.pcie_xy_enc
constexpr uint32_t kConfigReadBytes = 64;      // 16 words covers the whole config

struct SocketTarget {
    uint64_t host_data_addr;        // 64-bit PCIe-NoC addr of the host data FIFO
    uint64_t host_bytes_sent_addr;  // 64-bit PCIe-NoC addr of the host bytes_sent word
    uint32_t pcie_xy_enc;           // pcie64-encoded NoC xy of the PCIe tile
    uint32_t fifo_total_size;
    uint32_t write_ptr;
    uint32_t is_d2h;
};

SocketTarget read_socket_target(IDevice* dev, const CoreCoord& sender_logical, uint32_t config_addr) {
    std::vector<uint32_t> cfg;
    tt::tt_metal::detail::ReadFromDeviceL1(
        dev, sender_logical, config_addr, kConfigReadBytes, cfg, tt::CoreType::WORKER);
    SocketTarget t{};
    t.host_data_addr = (static_cast<uint64_t>(cfg[kWordDataAddrHi]) << 32) | cfg[kWordFifoAddrLo];
    t.host_bytes_sent_addr = (static_cast<uint64_t>(cfg[kWordBytesSentAddrHi]) << 32) | cfg[kWordBytesSentAddrLo];
    t.pcie_xy_enc = cfg[kWordPcieXyEnc];
    t.fifo_total_size = cfg[kWordFifoTotalSize];
    t.write_ptr = cfg[kWordWritePtr];
    t.is_d2h = cfg[kWordIsD2H];
    return t;
}

void print_target(
    int device_id,
    IDevice* dev,
    const CoreCoord& sender_logical,
    uint32_t config_addr,
    uint32_t fifo_size,
    uint32_t page_size,
    const SocketTarget& t) {
    const CoreCoord noc0 = dev->worker_core_from_logical_core(sender_logical);
    fmt::print("\n==================== D2H SOCKET TARGET (for X280) ====================\n");
    fmt::print("metal logical device id : {}\n", device_id);
    fmt::print("metal physical chip id  : {}\n", dev->id());
    fmt::print("sender core logical     : ({}, {})\n", sender_logical.x, sender_logical.y);
    fmt::print("sender core NOC0 virtual: ({}, {})\n", noc0.x, noc0.y);
    fmt::print(
        "config buffer L1 addr   : 0x{:x}  ({} bytes; X280 can read it over NoC)\n",
        config_addr,
        D2HSocket::required_config_buffer_size());
    fmt::print("fifo_size / page_size   : {} / {}\n", fifo_size, page_size);
    fmt::print("--- decoded socket metadata (the X280 write target) ---\n");
    fmt::print("pcie_xy_enc             : 0x{:08x}\n", t.pcie_xy_enc);
    fmt::print("host data FIFO addr     : 0x{:016x}  (64-bit PCIe-NoC addr)\n", t.host_data_addr);
    fmt::print("host bytes_sent addr    : 0x{:016x}\n", t.host_bytes_sent_addr);
    fmt::print("fifo_total_size         : {}\n", t.fifo_total_size);
    fmt::print("write_ptr (initial)     : {}\n", t.write_ptr);
    fmt::print("is_d2h                  : {}\n", t.is_d2h);
    fmt::print("======================================================================\n\n");
    std::fflush(stdout);
}

// ---- selftest: standard D2H flow with a Tensix sender kernel ----
int run_selftest(
    const std::shared_ptr<MeshDevice>& mesh_device,
    int device_id,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    uint32_t page_size,
    uint32_t data_size) {
    auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
    socket.set_page_size(page_size);

    IDevice* dev = mesh_device->get_devices().at(0);
    const SocketTarget t = read_socket_target(dev, sender_core.core_coord, socket.get_config_buffer_address());
    print_target(device_id, dev, sender_core.core_coord, socket.get_config_buffer_address(), fifo_size, page_size, t);

    // Sender data buffer on the sender core's L1, filled with iota.
    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto sender_shard =
        ShardSpecBuffer(CoreRangeSet(sender_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig sender_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_local_config, mesh_device.get());

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
            }});

    const uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst(data_size / sizeof(uint32_t), 0xdeadbeef);
    std::iota(src.begin(), src.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src, sender_core.device_coord, true);

    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(sender_core.device_coord), std::move(send_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    fmt::print("[selftest] launched Tensix sender; reading {} pages of {} B...\n", num_pages, page_size);
    std::fflush(stdout);

    const uint32_t page_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_pages; i++) {
        socket.read(dst.data() + i * page_words, 1);
    }
    socket.barrier(5000);

    bool ok = (src == dst);
    fmt::print("[selftest] data integrity: {}\n", ok ? "PASS ✓" : "FAIL ✗");
    if (!ok) {
        for (uint32_t i = 0; i < dst.size() && i < 16; i++) {
            fmt::print("  word[{}] got 0x{:08x} want 0x{:08x}\n", i, dst[i], src[i]);
        }
    }
    return ok ? 0 : 1;
}

// ---- listen: wait for an external (X280) writer ----
int run_listen(
    const std::shared_ptr<MeshDevice>& mesh_device,
    int device_id,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    uint32_t page_size,
    uint32_t pages_to_read,
    uint32_t timeout_s,
    const std::string& socket_id) {
    auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
    socket.set_page_size(page_size);

    IDevice* dev = mesh_device->get_devices().at(0);
    const SocketTarget t = read_socket_target(dev, sender_core.core_coord, socket.get_config_buffer_address());
    print_target(device_id, dev, sender_core.core_coord, socket.get_config_buffer_address(), fifo_size, page_size, t);

    const std::string desc_path = socket.export_descriptor(socket_id);
    fmt::print("[listen] exported descriptor: {}\n", desc_path);
    fmt::print(
        "[listen] waiting up to {}s for {} page(s) of {} B from the X280...\n", timeout_s, pages_to_read, page_size);
    std::fflush(stdout);

    const uint32_t page_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> page(page_words);
    auto start = std::chrono::steady_clock::now();
    uint32_t got = 0;
    while (got < pages_to_read) {
        if (socket.has_data()) {
            socket.read(page.data(), 1);
            fmt::print("[listen] page {} received:", got);
            for (uint32_t i = 0; i < page_words && i < 8; i++) {
                fmt::print(" 0x{:08x}", page[i]);
            }
            fmt::print("{}\n", page_words > 8 ? " ..." : "");
            std::fflush(stdout);
            got++;
            continue;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() >=
            timeout_s) {
            fmt::print("[listen] TIMEOUT after {}s; received {}/{} page(s)\n", timeout_s, got, pages_to_read);
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    fmt::print("[listen] received all {} page(s) ✓\n", pages_to_read);
    return 0;
}

// ---- hold: create the socket + keep it alive, but DO NOT read ----
// For the raw X280->host PCIe write-bandwidth test: the X280 blasts writes into
// the pinned FIFO region (overwriting freely, no flow control), and we just need
// the pinned memory to stay mapped. The host does not read, so it never competes
// for the PCIe path -- the measured rate is purely the X280's write throughput.
int run_hold(
    const std::shared_ptr<MeshDevice>& mesh_device,
    int device_id,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    uint32_t page_size,
    uint32_t secs) {
    auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
    socket.set_page_size(page_size);
    IDevice* dev = mesh_device->get_devices().at(0);
    const SocketTarget t = read_socket_target(dev, sender_core.core_coord, socket.get_config_buffer_address());
    print_target(device_id, dev, sender_core.core_coord, socket.get_config_buffer_address(), fifo_size, page_size, t);
    socket.export_descriptor("x280");
    fmt::print("[hold] socket alive for {}s; host will NOT read (pure write-BW test)\n", secs);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(secs));
    fmt::print("[hold] done\n");
    return 0;
}

// ---- serve: drain the FIFO continuously, report host read throughput ----
// For the integrated poll+stream test: the X280 streams flow-controlled pages;
// the host read()s them as fast as it can and counts bytes. Timing starts at the
// first page so we report steady-state drain throughput.
int run_serve(
    const std::shared_ptr<MeshDevice>& mesh_device,
    int device_id,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    uint32_t page_size,
    uint32_t secs) {
    auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
    socket.set_page_size(page_size);
    IDevice* dev = mesh_device->get_devices().at(0);
    const SocketTarget t = read_socket_target(dev, sender_core.core_coord, socket.get_config_buffer_address());
    print_target(device_id, dev, sender_core.core_coord, socket.get_config_buffer_address(), fifo_size, page_size, t);
    socket.export_descriptor("x280");
    fmt::print("[serve] draining for {}s once data starts...\n", secs);
    std::fflush(stdout);

    const uint32_t page_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> buf(page_words);
    uint64_t pages = 0, bytes = 0;
    bool started = false;
    std::chrono::steady_clock::time_point t0;
    auto wall0 = std::chrono::steady_clock::now();
    while (true) {
        if (socket.has_data()) {
            if (!started) {
                started = true;
                t0 = std::chrono::steady_clock::now();
            }
            socket.read(buf.data(), 1);
            pages++;
            bytes += page_size;
            if ((pages & 0x3fff) == 0) {  // periodic stop check w/o per-page clock reads
                if (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t0)
                        .count() >= secs) {
                    break;
                }
            }
            continue;
        }
        auto nowt = std::chrono::steady_clock::now();
        if (started && std::chrono::duration_cast<std::chrono::duration<double>>(nowt - t0).count() >= secs) {
            break;
        }
        if (!started && std::chrono::duration_cast<std::chrono::duration<double>>(nowt - wall0).count() >= 60) {
            fmt::print("[serve] no data within 60s; giving up\n");
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    double dt =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t0).count();
    fmt::print(
        "[serve] drained {} pages ({} B) in {:.3f}s -> {:.1f} MB/s ({:.0f} pages/s)\n",
        pages,
        bytes,
        dt,
        bytes / 1e6 / dt,
        pages / dt);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 3;
    const std::string mode = argc > 2 ? argv[2] : "selftest";

    auto mesh_device = MeshDevice::create_unit_mesh(device_id);
    const MeshCoreCoord sender_core{MeshCoordinate(0, 0), CoreCoord(0, 0)};

    fmt::print("x280_d2h: device {} (logical), mode '{}'\n", device_id, mode);
    std::fflush(stdout);

    int rc = 0;
    if (mode == "selftest") {
        rc = run_selftest(mesh_device, device_id, sender_core, /*fifo*/ 4096, /*page*/ 1024, /*data*/ 4096);
    } else if (mode == "listen") {
        const uint32_t pages_to_read = argc > 3 ? std::atoi(argv[3]) : 1;
        const uint32_t timeout_s = argc > 4 ? std::atoi(argv[4]) : 120;
        rc = run_listen(mesh_device, device_id, sender_core, 4096, 1024, pages_to_read, timeout_s, "x280");
    } else if (mode == "hold" || mode == "serve") {
        // hold/serve <fifo_size> <page_size> <secs>
        const uint32_t fifo_size = argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 0) : (256u * 1024);
        const uint32_t page_size = argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 0) : 4096;
        const uint32_t secs = argc > 5 ? std::atoi(argv[5]) : 5;
        if (mode == "hold") {
            rc = run_hold(mesh_device, device_id, sender_core, fifo_size, page_size, secs);
        } else {
            rc = run_serve(mesh_device, device_id, sender_core, fifo_size, page_size, secs);
        }
    } else {
        fmt::print("unknown mode '{}' (use selftest|listen|hold|serve)\n", mode);
        rc = 64;
    }
    return rc;
}
