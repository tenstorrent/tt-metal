// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

void warn_if_dprint_not_enabled() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see kernel output."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, DmHelloWorld) {
    warn_if_dprint_not_enabled();

    auto mesh_device = devices_[0];
    const CoreRange cluster_range = CoreRange(CoreCoord{0, 0}, CoreCoord{1, 0});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_hello_world.cpp",
        cluster_range,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
}

namespace {

struct KernelPerfResult {
    uint32_t write_avg;  // cycles per iteration in the memory-write phase (0 for per-iter kernel)
    uint32_t noc_avg;    // cycles per iteration in the NOC-issue phase   (0 for per-iter kernel)
    uint32_t total_avg;  // write_avg + noc_avg (two-phase), or combined loop avg (per-iter)
};

static constexpr const char* kPerfResultsFile = "dm_noc_perf_results.txt";

// Print a clearly visible banner to stdout and append one CSV row to kPerfResultsFile.
// The file is created with a header line on first write; subsequent runs append rows.
void print_and_log_perf(const KernelPerfResult& r) {
    const char* test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    std::cout << "\n\n================================================================\n"
              << "  [PERF]  " << test_name << "\n"
              << "  write_avg=" << r.write_avg << "  noc_avg=" << r.noc_avg << "  total_avg=" << r.total_avg
              << "  cycles\n"
              << "================================================================\n\n";

    // Write CSV header if the file does not yet exist.
    {
        std::ifstream check(kPerfResultsFile);
        if (!check.good()) {
            std::ofstream hdr(kPerfResultsFile);
            hdr << "test_name,write_avg,noc_avg,total_avg\n";
        }
    }
    std::ofstream out(kPerfResultsFile, std::ios::app);
    out << test_name << "," << r.write_avg << "," << r.noc_avg << "," << r.total_avg << "\n";
}

// Two-phase kernel: all memory writes first (separate loop), then all NOC writes.
//   write_mode   0 = write to L1 cache + flush L2 to SRAM, then NOC write
//   write_mode   1 = write directly to SRAM bypassing cache (+0x400000), then NOC write
//   barrier_mode 0 = single barrier after all NOC issues
//   barrier_mode 1 = barrier after every NOC issue (serialised)
KernelPerfResult run_two_phase_noc_test(
    distributed::MeshDevice* mesh_device,
    uint32_t write_mode,
    uint32_t barrier_mode = 0,
    uint32_t update_noc_addr = 0) {
    IDevice* device = mesh_device->get_devices()[0];

    constexpr CoreCoord src_core = {0, 0};
    constexpr CoreCoord dst_core = {1, 0};

    constexpr uint32_t packet_size_bytes = 3 * 64;
    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);  // 48 words = 192 bytes
    constexpr uint32_t num_iterations = 10;
    constexpr uint32_t stride_bytes = packet_size_bytes;
    // NOC always writes to the same dst address — only one packet worth of data lands there
    constexpr uint32_t total_words = packet_words;
    constexpr uint32_t total_bytes = packet_size_bytes;

    // Keep all buffers in a high, non-overlapping region of L1 on their respective cores.
    constexpr uint32_t src_l1_address = 1000 * 1024;
    constexpr uint32_t dst_l1_address = 1200 * 1024;
    constexpr uint32_t results_l1_address = 1400 * 1024;  // 12 bytes on src_core for timing readback

    std::vector<uint32_t> src_init(total_words, 0);
    std::vector<uint32_t> dst_init(total_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, src_core, src_l1_address, src_init);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, dst_l1_address, dst_init);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    const CoreCoord physical_dst_core = device->worker_core_from_logical_core(dst_core);
    const uint32_t packed_physical_dst_core = (physical_dst_core.x << 16) | (physical_dst_core.y & 0xFFFF);
    std::vector<uint32_t> compile_args = {
        src_l1_address,
        dst_l1_address,
        num_iterations,
        packet_size_bytes,
        stride_bytes,
        packed_physical_dst_core,
        write_mode,
        barrier_mode,
        results_l1_address,
        update_noc_addr};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_write_noc_two_phase.cpp",
        src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = compile_args,
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> observed(total_words, 0);
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, dst_l1_address, total_bytes, observed);

    std::vector<uint32_t> perf_words(3, 0);
    tt_metal::detail::ReadFromDeviceL1(device, src_core, results_l1_address, 3 * sizeof(uint32_t), perf_words);

    // Kernel writes 3 identical 64B packets then NOCs them to dst (same address every iteration).
    // Destination has the last iteration's 3x64B: each packet has [hdr, src, dst, 3,4,5,6,7].
    constexpr uint32_t kDmaTypeWrite = 1;
    constexpr uint32_t kEnDataInDescWriteToDst = 0;
    constexpr uint32_t kPacketTarget3b = 0x3;
    constexpr uint32_t kCompletionSw2b = 1;
    constexpr uint64_t kPacketDummySrcAddrBase = 0x100000000ULL;
    constexpr uint64_t kPacketDummyDstAddrBase = 0x200000000ULL;
    const uint32_t transfer_size_19b = packet_size_bytes & 0x7FFFF;
    const uint32_t final_iter = num_iterations - 1;

    auto lo32 = [](uint64_t v) { return static_cast<uint32_t>(v & 0xFFFFFFFFULL); };
    auto hi32 = [](uint64_t v) { return static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFULL); };

    const uint32_t req_word0 =
        ((kDmaTypeWrite & 0x1) << 8) | ((kEnDataInDescWriteToDst & 0x1) << 9) | (transfer_size_19b << 10);
    const uint32_t req_word1 = (kPacketTarget3b & 0x7) | ((kCompletionSw2b & 0x3) << 8);
    const uint64_t hdr = static_cast<uint64_t>(req_word0) | (static_cast<uint64_t>(req_word1) << 32);
    const uint64_t final_src_addr = kPacketDummySrcAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;
    const uint64_t final_dst_addr = kPacketDummyDstAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;

    // Each 64B packet is 16 uint32 words: [hdr(2), src(2), dst(2), 3(2), 4(2), 5(2), 6(2), 7(2)]
    auto fill_packet = [&](std::vector<uint32_t>& buf, uint32_t word_offset) {
        buf[word_offset + 0] = lo32(hdr);
        buf[word_offset + 1] = hi32(hdr);
        buf[word_offset + 2] = lo32(final_src_addr);
        buf[word_offset + 3] = hi32(final_src_addr);
        buf[word_offset + 4] = lo32(final_dst_addr);
        buf[word_offset + 5] = hi32(final_dst_addr);
        buf[word_offset + 6] = 3;
        buf[word_offset + 7] = 0;
        buf[word_offset + 8] = 4;
        buf[word_offset + 9] = 0;
        buf[word_offset + 10] = 5;
        buf[word_offset + 11] = 0;
        buf[word_offset + 12] = 6;
        buf[word_offset + 13] = 0;
        buf[word_offset + 14] = 7;
        buf[word_offset + 15] = 0;
    };

    std::vector<uint32_t> expected(total_words, 0);
    fill_packet(expected, 0);   // pkt0 at offset 0
    fill_packet(expected, 16);  // pkt1 at offset 64B = 16 words
    fill_packet(expected, 32);  // pkt2 at offset 128B = 32 words
    EXPECT_EQ(observed, expected);

    return KernelPerfResult{perf_words[0], perf_words[1], perf_words[2]};
}

}  // namespace

// Write to L1 cache, flush L2 to SRAM, then NOC write — all writes before all NOC sends
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/0);
    print_and_log_perf(r);
}

// Write directly to SRAM bypassing cache (+0x400000), then NOC write — all writes before all NOC sends
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/0);
    print_and_log_perf(r);
}

// Same as above but with a barrier after every NOC issue (serialised round-trip)
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/1);
    print_and_log_perf(r);
}

namespace {

// Per-iteration kernel: each iteration does memory write + NOC issue together in one loop.
//   barrier_mode 0 = single barrier after the full loop
//   barrier_mode 1 = barrier after every NOC issue (serialised)
KernelPerfResult run_per_iter_noc_test(
    distributed::MeshDevice* mesh_device,
    uint32_t write_mode,
    uint32_t barrier_mode = 0,
    uint32_t update_noc_addr = 0) {
    IDevice* device = mesh_device->get_devices()[0];

    constexpr CoreCoord src_core = {0, 0};
    constexpr CoreCoord dst_core = {1, 0};

    constexpr uint32_t packet_size_bytes = 3 * 64;
    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);
    constexpr uint32_t num_iterations = 10;
    constexpr uint32_t stride_bytes = packet_size_bytes;
    constexpr uint32_t total_words = packet_words;
    constexpr uint32_t total_bytes = packet_size_bytes;

    constexpr uint32_t src_l1_address = 1000 * 1024;
    constexpr uint32_t dst_l1_address = 1200 * 1024;
    constexpr uint32_t results_l1_address = 1400 * 1024;  // 12 bytes on src_core for timing readback

    std::vector<uint32_t> src_init(total_words, 0);
    std::vector<uint32_t> dst_init(total_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, src_core, src_l1_address, src_init);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, dst_l1_address, dst_init);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    const CoreCoord physical_dst_core = device->worker_core_from_logical_core(dst_core);
    const uint32_t packed_physical_dst_core = (physical_dst_core.x << 16) | (physical_dst_core.y & 0xFFFF);
    std::vector<uint32_t> compile_args = {
        src_l1_address,
        dst_l1_address,
        num_iterations,
        packet_size_bytes,
        stride_bytes,
        packed_physical_dst_core,
        write_mode,
        barrier_mode,
        results_l1_address,
        update_noc_addr};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_write_noc_per_iter.cpp",
        src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = compile_args,
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> observed(total_words, 0);
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, dst_l1_address, total_bytes, observed);

    std::vector<uint32_t> perf_words(3, 0);
    tt_metal::detail::ReadFromDeviceL1(device, src_core, results_l1_address, 3 * sizeof(uint32_t), perf_words);

    constexpr uint32_t kDmaTypeWrite = 1;
    constexpr uint32_t kEnDataInDescWriteToDst = 0;
    constexpr uint32_t kPacketTarget3b = 0x3;
    constexpr uint32_t kCompletionSw2b = 1;
    constexpr uint64_t kPacketDummySrcAddrBase = 0x100000000ULL;
    constexpr uint64_t kPacketDummyDstAddrBase = 0x200000000ULL;
    const uint32_t transfer_size_19b = packet_size_bytes & 0x7FFFF;
    const uint32_t final_iter = num_iterations - 1;

    auto lo32 = [](uint64_t v) { return static_cast<uint32_t>(v & 0xFFFFFFFFULL); };
    auto hi32 = [](uint64_t v) { return static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFULL); };

    const uint32_t req_word0 =
        ((kDmaTypeWrite & 0x1) << 8) | ((kEnDataInDescWriteToDst & 0x1) << 9) | (transfer_size_19b << 10);
    const uint32_t req_word1 = (kPacketTarget3b & 0x7) | ((kCompletionSw2b & 0x3) << 8);
    const uint64_t hdr = static_cast<uint64_t>(req_word0) | (static_cast<uint64_t>(req_word1) << 32);
    const uint64_t final_src_addr = kPacketDummySrcAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;
    const uint64_t final_dst_addr = kPacketDummyDstAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;

    auto fill_packet = [&](std::vector<uint32_t>& buf, uint32_t word_offset) {
        buf[word_offset + 0] = lo32(hdr);
        buf[word_offset + 1] = hi32(hdr);
        buf[word_offset + 2] = lo32(final_src_addr);
        buf[word_offset + 3] = hi32(final_src_addr);
        buf[word_offset + 4] = lo32(final_dst_addr);
        buf[word_offset + 5] = hi32(final_dst_addr);
        buf[word_offset + 6] = 3;
        buf[word_offset + 7] = 0;
        buf[word_offset + 8] = 4;
        buf[word_offset + 9] = 0;
        buf[word_offset + 10] = 5;
        buf[word_offset + 11] = 0;
        buf[word_offset + 12] = 6;
        buf[word_offset + 13] = 0;
        buf[word_offset + 14] = 7;
        buf[word_offset + 15] = 0;
    };

    std::vector<uint32_t> expected(total_words, 0);
    fill_packet(expected, 0);
    fill_packet(expected, 16);
    fill_packet(expected, 32);
    EXPECT_EQ(observed, expected);

    return KernelPerfResult{perf_words[0], perf_words[1], perf_words[2]};
}

}  // namespace

// Write to L1 cache, flush L2 to SRAM, then NOC write — all in one loop per iteration
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/0);
    print_and_log_perf(r);
}

// Write directly to SRAM bypassing cache (+0x400000), then NOC write — all in one loop per iteration
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/0);
    print_and_log_perf(r);
}

// Same as above but with a barrier after every NOC issue (serialised round-trip)
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/1);
    print_and_log_perf(r);
}

// Address register writes before every NOC issue (same src/dst values) — measures ROCC write overhead.
// Full combinatorial matrix: 2 write_modes × 2 barrier_modes × 2 kernels.
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/0, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/0, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/1, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr) {
    auto r = run_two_phase_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/1, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/0, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/0, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/0, /*barrier_mode=*/1, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}

TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr) {
    auto r = run_per_iter_noc_test(devices_[0].get(), /*write_mode=*/1, /*barrier_mode=*/1, /*update_noc_addr=*/1);
    print_and_log_perf(r);
}
