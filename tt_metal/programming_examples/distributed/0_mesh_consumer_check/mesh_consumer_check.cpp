// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two-thread example: sender streams LLM-decode-style data to device via H2D socket,
// receiver reads it back via D2H socket (loopback on device), prints and verifies.
// Data format: 128 entries × (Token ID + User ID + Position ID) = 3 words = 12 bytes per entry.
// Build and run from repo root (see README in this directory).
// Related: https://github.com/tenstorrent/tt-metal/issues/34274

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <iostream>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <thread>
#include <random>
#include <cstring>

namespace {

constexpr uint32_t kNumEntries = 128;
constexpr uint32_t kWordsPerEntry = 3;
constexpr uint32_t kBytesPerEntry = kWordsPerEntry * sizeof(uint32_t);
constexpr uint32_t kDataSizeBytes = kNumEntries * kBytesPerEntry;

struct DecodeEntry {
    uint32_t token_id;
    uint32_t user_id;
    uint32_t position_id;
};
static_assert(sizeof(DecodeEntry) == kBytesPerEntry, "DecodeEntry must be 12 bytes");

}  // namespace

int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    size_t num_available = GetNumAvailableDevices();
    size_t num_pcie = GetNumPCIeDevices();
    std::cout << "Available devices: " << num_available << ", PCIe devices: " << num_pcie << std::endl;

    if (num_available == 0) {
        std::cout << "No devices found. Check drivers and TT_VISIBLE_DEVICES.\n";
        return 0;
    }

    std::cout << "Opening 1x1 mesh device..." << std::endl;
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    std::cout << "Mesh device opened. Shape: " << mesh_device->num_rows() << "x" << mesh_device->num_cols()
              << std::endl;

    const MeshCoreCoord socket_core(MeshCoordinate(0, 0), CoreCoord(0, 0));
    constexpr uint32_t page_size = 64;
    constexpr uint32_t data_size = kDataSizeBytes;
    constexpr uint32_t num_iterations = 1;
    constexpr uint32_t fifo_size = 2048;

    std::cerr << "[h2d_socket] Creating H2D socket...\n" << std::flush;
    auto input_socket = std::make_unique<H2DSocket>(
        mesh_device, socket_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
    input_socket->set_page_size(page_size);

    std::cerr << "[d2h_socket] Creating D2H socket...\n" << std::flush;
    auto output_socket = std::make_unique<D2HSocket>(mesh_device, socket_core, fifo_size);
    output_socket->set_page_size(page_size);

    std::cout << "Creating loopback kernel (H2D -> D2H)..." << std::endl;
    auto loopback_program = CreateProgram();
    CreateKernel(
        loopback_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket->get_config_buffer_address()),
                static_cast<uint32_t>(output_socket->get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
                static_cast<uint32_t>(false),
            }});

    MeshWorkload mesh_workload;
    mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord), std::move(loopback_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    std::cout << "Loopback kernel enqueued. Streaming " << kNumEntries << " entries ("
              << kDataSizeBytes << " bytes) with sender and receiver threads." << std::endl;

    std::vector<DecodeEntry> ref_data(kNumEntries);
    std::vector<DecodeEntry> readback(kNumEntries);

    const uint32_t num_pages = data_size / page_size;
    const uint32_t page_size_words = page_size / sizeof(uint32_t);

    std::thread sender([&]() {
        std::mt19937 rng{42};
        std::uniform_int_distribution<uint32_t> dist(0, 0xFFFF);
        for (uint32_t i = 0; i < kNumEntries; i++) {
            ref_data[i].token_id = dist(rng);
            ref_data[i].user_id = dist(rng);
            ref_data[i].position_id = dist(rng);
        }
        const auto* raw = reinterpret_cast<const uint32_t*>(ref_data.data());
        for (uint32_t j = 0; j < num_pages; j++) {
            input_socket->write(const_cast<void*>(static_cast<const void*>(raw + (j * page_size_words))), 1);
        }
        input_socket->barrier();
    });

    std::thread receiver([&]() {
        auto* raw = reinterpret_cast<uint32_t*>(readback.data());
        for (uint32_t j = 0; j < num_pages; j++) {
            output_socket->read(raw + (j * page_size_words), 1);
        }
        output_socket->barrier();
    });

    sender.join();
    receiver.join();

    std::cout << "Receiver readback (first 5 and last 2 entries):" << std::endl;
    for (uint32_t i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] token_id=" << readback[i].token_id << " user_id=" << readback[i].user_id
                  << " position_id=" << readback[i].position_id << std::endl;
    }
    std::cout << "  ..." << std::endl;
    for (uint32_t i = kNumEntries - 2; i < kNumEntries; i++) {
        std::cout << "  [" << i << "] token_id=" << readback[i].token_id << " user_id=" << readback[i].user_id
                  << " position_id=" << readback[i].position_id << std::endl;
    }

    bool ok = (std::memcmp(ref_data.data(), readback.data(), kDataSizeBytes) == 0);
    if (!ok) {
        std::cout << "Mismatch: data read via D2H does not match data sent via H2D." << std::endl;
        for (uint32_t i = 0; i < kNumEntries && i < 20; i++) {
            if (ref_data[i].token_id != readback[i].token_id || ref_data[i].user_id != readback[i].user_id ||
                ref_data[i].position_id != readback[i].position_id) {
                std::cout << "  First diff at entry " << i << ": ref(" << ref_data[i].token_id << ","
                          << ref_data[i].user_id << "," << ref_data[i].position_id << ") vs recv("
                          << readback[i].token_id << "," << readback[i].user_id << "," << readback[i].position_id
                          << ")" << std::endl;
                break;
            }
        }
        Finish(mesh_device->mesh_command_queue());
        mesh_device->close();
        return 1;
    }
    std::cout << "OK: " << kNumEntries << " entries streamed via H2D and read back via D2H; verified." << std::endl;

    Finish(mesh_device->mesh_command_queue());
    mesh_device->close();
    std::cout << "Done." << std::endl;
    return 0;
}
