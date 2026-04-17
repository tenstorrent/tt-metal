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

// Build the expected 24-byte descriptor buffer that every kernel variant writes to dst.
// Descriptor layout (little-endian, matches kernel writes):
//   bytes  0- 3: req_word0
//   bytes  4-11: SrcAddr (64b)
//   bytes 12-19: DestAddr (64b)
//   bytes 20-23: req_word1
// = 3 qwords = 6 uint32 words.
// The dst buffer is overwritten each iteration; we check the final state (last iteration).
static std::vector<uint32_t> build_expected_packet_buffer() {
    constexpr uint32_t kDescriptorBytes = 3 * sizeof(uint64_t);             // 24
    constexpr uint32_t packet_words = kDescriptorBytes / sizeof(uint32_t);  // 6
    constexpr uint32_t num_iterations = 100;
    constexpr uint32_t kDmaTypeWrite = 1;
    constexpr uint32_t kEnDataInDescWriteToDst = 0;
    constexpr uint32_t kPacketTarget3b = 0x3;
    constexpr uint32_t kCompletionSw2b = 1;
    constexpr uint64_t kPacketDummySrcAddrBase = 0x100000000ULL;
    constexpr uint64_t kPacketDummyDstAddrBase = 0x200000000ULL;
    const uint32_t transfer_size_19b = kDescriptorBytes & 0x7FFFF;
    const uint32_t final_iter = num_iterations - 1;

    auto lo32 = [](uint64_t v) { return static_cast<uint32_t>(v & 0xFFFFFFFFULL); };
    auto hi32 = [](uint64_t v) { return static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFULL); };

    const uint32_t req_word0 =
        ((kDmaTypeWrite & 0x1) << 8) | ((kEnDataInDescWriteToDst & 0x1) << 9) | (transfer_size_19b << 10);
    const uint32_t req_word1 = (kPacketTarget3b & 0x7) | ((kCompletionSw2b & 0x3) << 8);
    const uint64_t final_src = kPacketDummySrcAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;
    const uint64_t final_dst = kPacketDummyDstAddrBase + static_cast<uint64_t>(final_iter) * transfer_size_19b;

    std::vector<uint32_t> buf(packet_words, 0);
    buf[0] = req_word0;        // bytes  0- 3
    buf[1] = lo32(final_src);  // bytes  4- 7
    buf[2] = hi32(final_src);  // bytes  8-11
    buf[3] = lo32(final_dst);  // bytes 12-15
    buf[4] = hi32(final_dst);  // bytes 16-19
    buf[5] = req_word1;        // bytes 20-23
    return buf;
}

// Two-phase kernel: all memory writes first (separate loop), then all NOC writes.
//   write_mode        0 = write to L1 cache + flush L2 to SRAM, then NOC write
//   write_mode        1 = write directly to SRAM bypassing cache (+0x400000), then NOC write
//   barrier_mode      0 = single barrier after all NOC issues
//   barrier_mode      1 = barrier after every NOC issue (serialised)
//   num_active_cores  N cores run the same kernel in parallel; result is averaged across cores.
//                     all N DM threads run within the same cluster at {0,0}; dst cluster is {0,1}.
KernelPerfResult run_two_phase_noc_test(
    distributed::MeshDevice* mesh_device,
    uint32_t write_mode,
    uint32_t barrier_mode = 0,
    uint32_t update_noc_addr = 0,
    uint32_t num_active_cores = 1) {
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t packet_size_bytes = 3 * sizeof(uint64_t);             // 24 bytes — one descriptor
    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);  // 6 words
    constexpr uint32_t num_iterations = 100;
    constexpr uint32_t stride_bytes = packet_size_bytes;
    constexpr uint32_t total_words = packet_words;
    constexpr uint32_t total_bytes = packet_size_bytes;
    // Each hart writes to src_l1_address + hart_id*64 (one cache line per core, max 8 cores).
    constexpr uint32_t kHartStride = 64;
    constexpr uint32_t kMaxHarts = 8;
    constexpr uint32_t preinit_words = kMaxHarts * kHartStride / sizeof(uint32_t);  // 128 words
    constexpr uint32_t src_l1_address = 1000 * 1024;
    constexpr uint32_t dst_l1_address = 1200 * 1024;
    constexpr uint32_t results_l1_address = 1400 * 1024;

    // All num_active_cores DM threads run inside the same cluster at {0,0}; dst cluster is {1,0}.
    // num_threads_per_cluster controls how many of the 8 DM cores within that cluster execute.
    const CoreCoord src_core = {0, 0};
    const CoreCoord dst_core = {1, 0};

    std::vector<uint32_t> zeros(preinit_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, src_core, src_l1_address, zeros);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, dst_l1_address, zeros);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const CoreCoord physical_dst = device->worker_core_from_logical_core(dst_core);
    const uint32_t packed_dst = (physical_dst.x << 16) | (physical_dst.y & 0xFFFF);
    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_write_noc_two_phase.cpp",
        src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = num_active_cores,
            .compile_args =
                {src_l1_address,
                 dst_l1_address,
                 num_iterations,
                 packet_size_bytes,
                 stride_bytes,
                 packed_dst,
                 write_mode,
                 barrier_mode,
                 results_l1_address,
                 update_noc_addr},
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    const std::vector<uint32_t> expected = build_expected_packet_buffer();
    std::vector<uint32_t> observed(total_words, 0);
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, dst_l1_address, total_bytes, observed);
    EXPECT_EQ(observed, expected);

    // Read all num_active_cores slots (hart_id 0..N-1), each 3 words, and average.
    const uint32_t results_total_bytes = num_active_cores * 3 * sizeof(uint32_t);
    std::vector<uint32_t> all_perf(num_active_cores * 3, 0);
    tt_metal::detail::ReadFromDeviceL1(device, src_core, results_l1_address, results_total_bytes, all_perf);
    uint32_t wa_sum = 0, na_sum = 0, ta_sum = 0;
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        wa_sum += all_perf[i * 3 + 0];
        na_sum += all_perf[i * 3 + 1];
        ta_sum += all_perf[i * 3 + 2];
    }
    return KernelPerfResult{wa_sum / num_active_cores, na_sum / num_active_cores, ta_sum / num_active_cores};
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
//   barrier_mode      0 = single barrier after the full loop
//   barrier_mode      1 = barrier after every NOC issue (serialised)
//   num_active_cores  N cores run the same kernel in parallel; result is averaged across cores.
//                     all N DM threads run within the same cluster at {0,0}; dst cluster is {0,1}.
KernelPerfResult run_per_iter_noc_test(
    distributed::MeshDevice* mesh_device,
    uint32_t write_mode,
    uint32_t barrier_mode = 0,
    uint32_t update_noc_addr = 0,
    uint32_t num_active_cores = 1) {
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t packet_size_bytes = 3 * sizeof(uint64_t);             // 24 bytes — one descriptor
    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);  // 6 words
    constexpr uint32_t num_iterations = 100;
    constexpr uint32_t stride_bytes = packet_size_bytes;
    constexpr uint32_t total_words = packet_words;
    constexpr uint32_t total_bytes = packet_size_bytes;
    // Each hart writes to src_l1_address + hart_id*64 (one cache line per core, max 8 cores).
    constexpr uint32_t kHartStride = 64;
    constexpr uint32_t kMaxHarts = 8;
    constexpr uint32_t preinit_words = kMaxHarts * kHartStride / sizeof(uint32_t);  // 128 words
    constexpr uint32_t src_l1_address = 1000 * 1024;
    constexpr uint32_t dst_l1_address = 1200 * 1024;
    constexpr uint32_t results_l1_address = 1400 * 1024;

    // All num_active_cores DM threads run inside the same cluster at {0,0}; dst cluster is {1,0}.
    // num_threads_per_cluster controls how many of the 8 DM cores within that cluster execute.
    const CoreCoord src_core = {0, 0};
    const CoreCoord dst_core = {1, 0};

    std::vector<uint32_t> zeros(preinit_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, src_core, src_l1_address, zeros);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, dst_l1_address, zeros);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const CoreCoord physical_dst = device->worker_core_from_logical_core(dst_core);
    const uint32_t packed_dst = (physical_dst.x << 16) | (physical_dst.y & 0xFFFF);
    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_write_noc_per_iter.cpp",
        src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = num_active_cores,
            .compile_args =
                {src_l1_address,
                 dst_l1_address,
                 num_iterations,
                 packet_size_bytes,
                 stride_bytes,
                 packed_dst,
                 write_mode,
                 barrier_mode,
                 results_l1_address,
                 update_noc_addr},
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    const std::vector<uint32_t> expected = build_expected_packet_buffer();
    std::vector<uint32_t> observed(total_words, 0);
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, dst_l1_address, total_bytes, observed);
    EXPECT_EQ(observed, expected);

    // Read all num_active_cores slots (hart_id 0..N-1), each 3 words, and average.
    const uint32_t results_total_bytes = num_active_cores * 3 * sizeof(uint32_t);
    std::vector<uint32_t> all_perf(num_active_cores * 3, 0);
    tt_metal::detail::ReadFromDeviceL1(device, src_core, results_l1_address, results_total_bytes, all_perf);
    uint32_t wa_sum = 0, na_sum = 0, ta_sum = 0;
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        wa_sum += all_perf[i * 3 + 0];
        na_sum += all_perf[i * 3 + 1];
        ta_sum += all_perf[i * 3 + 2];
    }
    return KernelPerfResult{wa_sum / num_active_cores, na_sum / num_active_cores, ta_sum / num_active_cores};
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

// ── Multi-core variants ───────────────────────────────────────────────────────
// Each test runs the same kernel simultaneously on N cores (src[i]={0,i}, dst[i]={1,i}).
// Results are averaged across all active cores.
// Naming: existing single-core tests have no suffix; multi-core tests append _NCores.

// ── 2 cores ──────────────────────────────────────────────────────────────────
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_2Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 0, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 1, 2);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_2Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 1, 2);
    print_and_log_perf(r);
}

// ── 3 cores ──────────────────────────────────────────────────────────────────
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_3Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 0, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 1, 3);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_3Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 1, 3);
    print_and_log_perf(r);
}

// ── 4 cores ──────────────────────────────────────────────────────────────────
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 0, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 0, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 0, 1, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_4Cores) {
    auto r = run_two_phase_noc_test(devices_[0].get(), 1, 1, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 0, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 0, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 0, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 0, 1, 1, 4);
    print_and_log_perf(r);
}
TEST_F(MeshDeviceSingleCardFixture, DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_4Cores) {
    auto r = run_per_iter_noc_test(devices_[0].get(), 1, 1, 1, 4);
    print_and_log_perf(r);
}
