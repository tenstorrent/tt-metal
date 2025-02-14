// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// TT_METAL_LOGGER_LEVEL=FATAL ./build/test/tt_metal/perf_microbenchmark/3_pcie_transfer/pcie_bench_wormhole_b0
// --benchmark_out=benchdata.json --benchmark_out_format=json
//

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <span>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/rtoptions.hpp>
#include <tt-metalium/memcpy.hpp>
#include <tt-metalium/helpers.hpp>
#include "bfloat16.hpp"
#include "buffer.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "hal.hpp"
#include "host_api.hpp"
#include "kernel_types.hpp"
#include "program_impl.hpp"
#include "tt_cluster.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include <benchmark/benchmark.h>
#include <cstdint>

using namespace tt::tt_metal;
using namespace std::chrono_literals;  // s

using Timepoint = decltype(std::chrono::high_resolution_clock::now());
using StartEndTime = std::pair<Timepoint, Timepoint>;

static constexpr uint32_t k_MemCpyAlignment = sizeof(__m256i);  // Largest instruction in memcpy impl

template <bool debug_sync = false>
static inline void my_memcpy_to_device(void* __restrict dst, const void* __restrict src, size_t n) {
    TT_ASSERT((uintptr_t)dst % MEMCPY_ALIGNMENT == 0);

    static constexpr uint32_t inner_loop = 8;
    static constexpr uint32_t inner_blk_size = inner_loop * sizeof(__m256i);

    uint8_t* src8 = (uint8_t*)src;
    uint8_t* dst8 = (uint8_t*)dst;

    if (size_t num_lines = n / inner_blk_size) {
        for (size_t i = 0; i < num_lines; ++i) {
            for (size_t j = 0; j < inner_loop; ++j) {
                __m256i blk = _mm256_loadu_si256((const __m256i*)src8);
                _mm256_stream_si256((__m256i*)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= inner_blk_size;
        }
    }

    if (n > 0) {
        if (size_t num_lines = n / sizeof(__m256i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m256i blk = _mm256_loadu_si256((const __m256i*)src8);
                _mm256_stream_si256((__m256i*)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= num_lines * sizeof(__m256i);
        }
        if (size_t num_lines = n / sizeof(__m128i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m128i blk = _mm_loadu_si128((const __m128i*)src8);
                _mm_stream_si128((__m128i*)dst8, blk);
                src8 += sizeof(__m128i);
                dst8 += sizeof(__m128i);
            }
            n -= n / sizeof(__m128i) * sizeof(__m128i);
        }
        if (n > 0) {
            for (size_t i = 0; i < n / sizeof(int32_t); ++i) {
                _mm_stream_si32((int32_t*)dst8, *(int32_t*)src8);
                src8 += sizeof(int32_t);
                dst8 += sizeof(int32_t);
            }
            n -= n / sizeof(int32_t) * sizeof(int32_t);
            // Copying the last few bytes (n < 4).
            // Overrunning dst buffer is safe, because the actual allocated space for dst is guaranteed to be at least 4
            // byte aligned.
            if (n > 0) {
                int32_t val = 0;
                std::memcpy(&val, src8, n);
                _mm_stream_si32((int32_t*)dst8, val);
            }
        }
    }
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

double get_current_time_seconds() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Uses low level APIs to benchmark Pcie transfer.
// Fast dispatch needs to be disabled because this benchmark will write into hugepage.
// For better benchmark outputs, run it with TT_METAL_LOGGER_LEVEL=FATAL
class MemCpyPcieBench : public benchmark::Fixture {
public:
    static constexpr std::string_view k_PcieBenchKernel =
        "tests/tt_metal/tt_metal/perf_microbenchmark/3_pcie_transfer/kernels/pcie_bench.cpp";

    struct PcieTransferResults {
        std::chrono::duration<double> host_hugepage_writing_duration;
        int64_t host_hugepage_bytes_processed;

        std::chrono::duration<double> host_wait_for_kernels_duration;

        std::chrono::duration<double> kernel_duration;
        int64_t kernel_bytes_rd;
        int64_t kernel_bytes_wr;
    };

    // Mini Mem Map
    struct DeviceAddresses {
        uint32_t cycles;
        uint32_t rd_bytes;
        uint32_t wr_bytes;
        uint32_t unreserved;
    };

    // Device under test
    IDevice* device;

    // Get pointer to the host hugepage
    void* GetHostHugePage(uint32_t base_offset) const {
        const auto dut_id = this->device->id();  // device under test
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);
        return (void*)(tt::Cluster::instance().host_dma_address(base_offset, mmio_device_id, channel));
    }

    // Get size of the host hugepage
    uint32_t GetHostHugePageSize(chip_id_t dut_id) const {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);
        return tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
    }

    // Generate my_memcpy_to_device aligned source data
    vector_memcpy_aligned<uint32_t> GenSrcData(uint32_t num_bytes) const {
        std::uniform_int_distribution<uint32_t> distribution(
            std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
        std::default_random_engine generator;

        vector_memcpy_aligned<uint32_t> vec(num_bytes / sizeof(uint32_t));
        std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });

        return vec;
    }

    // Read a word at the address from each core in the core range
    std::vector<int64_t> GetWordsFromDevice(const CoreRange& core, uint32_t addr) {
        std::vector<int64_t> data;
        for (int xi = core.start_coord.x; xi <= core.end_coord.x; ++xi) {
            for (int yi = core.start_coord.y; yi <= core.end_coord.y; ++yi) {
                std::vector<uint32_t> single_data;
                detail::ReadFromDeviceL1(device, CoreCoord{xi, yi}, addr, sizeof(uint32_t), single_data);
                data.push_back(single_data[0]);
            }
        }
        return data;
    }

    // Returns the simple device address map required for this benchmark
    DeviceAddresses GetDevAddrMap() const {
        const auto l1_alignment = hal.get_alignment(HalMemType::L1);
        DeviceAddresses addrs;
        addrs.cycles = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
        addrs.rd_bytes = align_addr(addrs.cycles + sizeof(uint32_t), l1_alignment);
        addrs.wr_bytes = align_addr(addrs.rd_bytes + sizeof(uint32_t), l1_alignment);
        addrs.unreserved = align_addr(addrs.wr_bytes + sizeof(uint32_t), l1_alignment);
        return addrs;
    }

    // Perform a host copy to hugepage
    // repeating_src_vector means the same data at the start of the src_data will be copied. The src_data read address
    // will not be incremented. Repeatedly reading the same data should keep the src data hot in the cache to expose
    // faster theoretical workload speeds... to simulate the workload having all data perfectly prefetched and in the
    // cache...etc.
    template <bool repeating_src_vector>
    std::chrono::duration<double> HostWriteHP(
        void* hugepage_base,
        uint32_t hugepage_size,
        std::span<uint32_t> src_data,
        size_t total_size,
        size_t page_size) {
        uint64_t hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
        uint64_t hugepage_end = hugepage_addr + hugepage_size;
        uint64_t src_addr = reinterpret_cast<uint64_t>(src_data.data());
        size_t num_pages;
        if (!page_size) {
            num_pages = 1;
            page_size = total_size;
        } else {
            num_pages = total_size / page_size;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_pages; ++i) {
            my_memcpy_to_device<false /*fence*/>((void*)(hugepage_addr), (void*)(src_addr), page_size);

            hugepage_addr += page_size;

            if constexpr (!repeating_src_vector) {
                src_addr += page_size;
            }

            // This may exceed the maximum hugepage
            if (hugepage_addr >= hugepage_end) {
                hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    }

    // Configure a range of kernels to pull from PCIe. Returns the range of kernels that
    // was configured if any. Number of kernels needs to be less than 1 row or a multiple of rows.
    std::optional<CoreRange> ConfigureKernels(
        Program& program,
        const DeviceAddresses& dev_addrs,
        uint32_t start_y,
        uint32_t num_kernels,
        bool is_writer,
        uint32_t total_size,
        uint32_t page_size,
        uint32_t pcie_size,
        uint32_t pcie_offset = 0) const {
        if (!page_size) {
            page_size = total_size;
        }
        if (!num_kernels) {
            return {};
        }

        const auto grid_size = device->logical_grid_size();
        const auto max_x = grid_size.x;
        const auto max_y = grid_size.y;

        // Number readers either less than one row
        // or a multiple of the rows
        CoreCoord start_coord{0, start_y};
        CoreCoord end_coord;
        if (num_kernels <= max_x) {
            end_coord.x = start_coord.x + num_kernels - 1;
            end_coord.y = start_coord.y;
        } else {
            const auto number_of_rows = num_kernels / max_x;
            const auto last_row_width = (num_kernels % max_x) ? num_kernels % max_x : max_x;
            end_coord.x = start_coord.x + last_row_width - 1;
            end_coord.y = number_of_rows - 1;
        }
        CoreRange core_range{start_coord, end_coord};

        std::vector<uint32_t> pcie_bench_compile_args(12, 0);
        if (is_writer) {
            pcie_bench_compile_args[5] = 0;                   // reserved_0
            pcie_bench_compile_args[6] = pcie_offset;         // pcie_wr_base
            pcie_bench_compile_args[7] = pcie_size;           // pcie_wr_size
            pcie_bench_compile_args[8] = page_size;           // pcie_wr_transfer_size
            pcie_bench_compile_args[9] = dev_addrs.wr_bytes;  // my_bytes_wr_addr
        } else {
            // reader
            pcie_bench_compile_args[0] = dev_addrs.unreserved,  // my_rd_dst_addr
                pcie_bench_compile_args[1] = pcie_offset;       // pcie_rd_base
            pcie_bench_compile_args[2] = pcie_size;             // pcie_rd_size
            pcie_bench_compile_args[3] = page_size;             // pcie_rd_transfer_size
            pcie_bench_compile_args[4] = dev_addrs.rd_bytes;    // my_bytes_rd_addr
        }
        pcie_bench_compile_args[10] = total_size;
        pcie_bench_compile_args[11] = dev_addrs.cycles;

        [[maybe_unused]] KernelHandle kernel = CreateKernel(
            program,
            std::string{k_PcieBenchKernel},
            core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC_0,
                .compile_args = pcie_bench_compile_args,
                .defines = {},
            });

        return core_range;
    }

    MemCpyPcieBench() : benchmark::Fixture{} {
        UseManualTime();
        Iterations(5);
    }

    void SetUp(benchmark::State& state) override {
        const chip_id_t target_device_id = 0;
        this->device = CreateDevice(target_device_id, 1);
        if (!this->device->is_mmio_capable()) {
            state.SkipWithMessage("MemCpyPcieBench can only be run on a MMIO capable device");
        }

        if (this->device->using_fast_dispatch()) {
            state.SkipWithMessage(
                "MemCpyPcieBench can only be run with slow dispatch enabled. It conflicts with fast dispatch because "
                "it needs to read/write into HugePages");
        } else if (!tt::llrt::RunTimeOptions::get_instance().get_clear_l1()) {
            state.SkipWithMessage("export TT_METAL_CLEAR_L1=1 is required");
        }
    }

    void TearDown(const benchmark::State& state) override {
        tt::DevicePool::instance().close_device(this->device->id());
    }
};

// Host writing to hugepage while there are N kernels reading from it
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Readers)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto page_size = state.range(1);
    const auto pages = total_size / page_size;
    const auto num_readers = state.range(2);
    const auto cached_vector = static_cast<bool>(state.range(3));
    const auto enable_host_write = static_cast<bool>(state.range(4));
    const auto hp_size = this->GetHostHugePageSize(this->device->id());
    const auto hp_base = this->GetHostHugePage(0);  // Already aligned
    const auto hp_end = reinterpret_cast<uint64_t>(hp_base) + hp_size;
    const auto dev_addrs = this->GetDevAddrMap();

    double total_device_time = 0;
    double total_device_bytes = 0;
    double total_iteration_time = 0;

    for (auto _ : state) {
        auto src_data = this->GenSrcData(total_size);  // Already aligned
        auto program = Program();
        auto configured_readers =
            ConfigureKernels(program, dev_addrs, 0, num_readers, false, total_size, page_size, hp_size);

        std::atomic<bool> start_flag{false};
        std::atomic<int> thread_ready{0};
        std::chrono::duration<double> program_time;

        auto thread = std::thread([&]() {
            thread_ready++;
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            auto launch_start = std::chrono::high_resolution_clock::now();
            detail::LaunchProgram(this->device, program, true);
            auto launch_end = std::chrono::high_resolution_clock::now();
            program_time = std::chrono::duration_cast<std::chrono::duration<double>>(launch_end - launch_start);
        });

        while (!thread_ready.load()) {
            std::this_thread::yield();
        }

        start_flag.store(true);

        std::chrono::duration<double> hp_duration{1};
        if (enable_host_write) {
            if (cached_vector) {
                hp_duration = HostWriteHP<true>(hp_base, hp_size, src_data, total_size, page_size);
            } else {
                hp_duration = HostWriteHP<false>(hp_base, hp_size, src_data, total_size, page_size);
            }
        }
        thread.join();

        if (configured_readers.has_value()) {
            auto dev_cycles = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.cycles);
            auto dev_bytes_read = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.rd_bytes);
            auto dev_clk = tt::Cluster::instance().get_device_aiclk(device->id()) * 1e6;  // Hz
            double all_cores_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end());
            double all_cores_bytes_read = std::reduce(dev_bytes_read.begin(), dev_bytes_read.end());
            std::chrono::duration<double> kernel_duration{};

            total_device_time += all_cores_cycles / dev_clk;
            total_device_bytes += all_cores_bytes_read;
        }

        state.SetIterationTime(hp_duration.count());
        total_iteration_time += hp_duration.count();
    }

    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["dev_bandwidth_per_second"] = (total_device_bytes / total_device_time);
    state.counters["dev_bytes"] = total_device_bytes;
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = page_size;
    state.counters["num_readers"] = num_readers;
}

// Host writing to hugepage using N threads while there's a kernel reading from the same hugepage. Total
// data to write to HP is split among N threads
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Threads)(benchmark::State& state) {
    static_assert((k_MemCpyAlignment & ((k_MemCpyAlignment)-1)) == 0);

    const auto total_size = state.range(0);
    const auto page_size = state.range(1);
    const auto cached_vector = static_cast<bool>(state.range(3));
    const auto num_threads = static_cast<size_t>(state.range(4));

    const auto hp_base_addr = this->GetHostHugePage(0);
    const auto hp_size = this->GetHostHugePageSize(this->device->id());
    const auto hp_end = reinterpret_cast<uint64_t>(hp_base_addr) + hp_size;
    const auto bytes_per_thread = ((total_size / num_threads) + (k_MemCpyAlignment)-1) & -(k_MemCpyAlignment);
    const auto last_thread_bytes = total_size - (bytes_per_thread * (num_threads - 1));

    for (auto _ : state) {
        auto src_data = this->GenSrcData(total_size);  // Already aligned

        std::vector<std::thread> threads(num_threads);
        std::atomic<bool> start_flag{false};
        std::atomic<int> threads_ready{0};
        std::vector<double> thread_durations(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads[i] = std::thread(
                [&](size_t thread_i) {
                    // Get subarray for this thread
                    uint64_t thread_dst = (uint64_t)hp_base_addr + (thread_i * bytes_per_thread);
                    uint64_t thread_bytes = (thread_i == num_threads - 1) ? last_thread_bytes : bytes_per_thread;
                    std::span<uint32_t> thread_src{src_data};
                    thread_src = thread_src.subspan(
                        (thread_i * bytes_per_thread) / sizeof(uint32_t), thread_bytes / sizeof(uint32_t));

                    threads_ready++;
                    while (!start_flag.load()) {
                        std::this_thread::yield();
                    }

                    auto start = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> hp_duration;
                    if (cached_vector) {
                        hp_duration =
                            this->HostWriteHP<true>((void*)thread_dst, hp_size, thread_src, thread_bytes, page_size);
                    } else {
                        hp_duration =
                            this->HostWriteHP<false>((void*)thread_dst, hp_size, thread_src, thread_bytes, page_size);
                    }
                    auto end = std::chrono::high_resolution_clock::now();

                    thread_durations[thread_i] = std::chrono::duration<double>(end - start).count();
                },
                i);  // Need to copy i into local thread
        }

        while (threads_ready.load() < num_threads) {
            std::this_thread::yield();
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        start_flag.store(true);
        for (auto& thread : threads) {
            thread.join();
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        state.SetIterationTime(
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count());
    }

    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = page_size;
    state.counters["num_threads"] = num_threads;
}

// Try to match real workload. Singlethread memcpy_to_device is called many times. Max size is
// usually 32K. UNET tensors are ~10KB
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_BatchSizing)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto batch_size = state.range(1);
    const auto pages = total_size / batch_size;

    for (auto _ : state) {
        auto rnd_data = this->GenSrcData(total_size);
        auto hugepage_ptr = this->GetHostHugePage(0);
        auto hugepage_size = this->GetHostHugePageSize(this->device->id());
        auto src_ptr = rnd_data.data();
        state.SetIterationTime(
            this->HostWriteHP<false>(hugepage_ptr, hugepage_size, rnd_data, total_size, batch_size).count());
    }
    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = state.range(1);
}

// Host writing to a hugepage while the device reads from another hugepage
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_DifferentPage)(benchmark::State& state) {
    constexpr uint32_t k_NumReaders = 1;
    constexpr uint32_t k_TotalSize = 1_GB;
    constexpr uint32_t k_PageSize = 32_KB;

    double total_device_duration = 0;
    double total_device_bytes = 0;

    for (auto _ : state) {
        const auto dut_id = this->device->id();
        const auto dev_addrs = this->GetDevAddrMap();
        auto src_data = this->GenSrcData(k_TotalSize);  // Already aligned

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);

        // First page for reader kernels to pull
        auto first_hp = (void*)(tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel));
        auto first_hp_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
        // Second page for host to write
        auto second_hp = (void*)(tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel + 1));
        auto second_hp_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel + 1);

        auto program = Program();
        auto configured_readers =
            ConfigureKernels(program, dev_addrs, 0, k_NumReaders, false, k_TotalSize, k_PageSize, first_hp_size);

        tt::log_info("channel {} hugepage = {:#x}", channel, (uint64_t)first_hp);
        tt::log_info("channel {} hugepage = {:#x}", channel + 1, (uint64_t)second_hp);

        std::thread kernel_thread;
        std::atomic<bool> start_flag{false};
        std::atomic<int> threads_ready{0};
        std::chrono::duration<double> host_program_duration;

        // Thread to wait for kernel to finish
        auto thread = std::thread([&]() {
            threads_ready++;
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            auto launch_start = std::chrono::high_resolution_clock::now();
            detail::LaunchProgram(this->device, program, true);
            auto launch_end = std::chrono::high_resolution_clock::now();
            host_program_duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(launch_end - launch_start);
        });

        // Wait for thread to prepare
        while (!threads_ready.load()) {
            std::this_thread::yield();
        }

        // Start kernel thread
        start_flag.store(true);

        // Do host write to other page
        std::chrono::duration<double> hp_duration =
            HostWriteHP<false>(second_hp, second_hp_size, src_data, k_TotalSize, k_PageSize);

        thread.join();

        // Collect results from device
        auto dev_cycles = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.cycles);
        auto dev_bytes_read = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.rd_bytes);
        auto dev_clk = tt::Cluster::instance().get_device_aiclk(device->id()) * 1e6;  // Hz
        double all_cores_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end());
        double all_cores_bytes_read = std::reduce(dev_bytes_read.begin(), dev_bytes_read.end());

        total_device_duration += all_cores_cycles / dev_clk;
        total_device_bytes += all_cores_bytes_read;

        state.SetIterationTime(hp_duration.count());
    }

    state.SetBytesProcessed(k_TotalSize * state.iterations());

    state.counters["dev_bandwidth_per_second"] = (total_device_bytes / total_device_duration);
    state.counters["dev_bytes"] = total_device_bytes;
    state.counters["total_size"] = k_TotalSize;
    state.counters["page_size"] = k_PageSize;
    state.counters["num_readers"] = 1;
}

// Host writing to one half of hugepage while device writes to another half
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Writers)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto page_size = state.range(1);
    const auto pages = total_size / page_size;
    const auto num_writers = state.range(2);
    const auto cached_vector = static_cast<bool>(state.range(3));
    const auto enable_host_write = static_cast<bool>(state.range(4));
    const auto hp_size = this->GetHostHugePageSize(this->device->id());
    const auto hp_base = this->GetHostHugePage(0);  // Already aligned
    const auto hp_end = reinterpret_cast<uint64_t>(hp_base) + hp_size;
    const auto dev_addrs = this->GetDevAddrMap();

    // Device will write to second half of hugepage
    const auto pcie_offset = total_size;
    const auto remaining_pcie_size = hp_size - pcie_offset;

    double total_device_time = 0;
    double total_device_bytes = 0;
    double total_iteration_time = 0;

    for (auto _ : state) {
        auto src_data = this->GenSrcData(total_size);  // Already aligned
        auto program = Program();
        auto configured_writers = ConfigureKernels(
            program, dev_addrs, 0, num_writers, true, total_size, page_size, remaining_pcie_size, pcie_offset);

        std::atomic<bool> start_flag{false};
        std::atomic<int> thread_ready{0};
        std::chrono::duration<double> program_time;

        auto thread = std::thread([&]() {
            thread_ready++;
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            auto launch_start = std::chrono::high_resolution_clock::now();
            detail::LaunchProgram(this->device, program, true);
            auto launch_end = std::chrono::high_resolution_clock::now();
            program_time = std::chrono::duration_cast<std::chrono::duration<double>>(launch_end - launch_start);
        });

        while (!thread_ready.load()) {
            std::this_thread::yield();
        }

        start_flag.store(true);

        std::chrono::duration<double> hp_duration{1};
        if (enable_host_write) {
            if (cached_vector) {
                hp_duration = HostWriteHP<true>(hp_base, hp_size, src_data, total_size, page_size);
            } else {
                hp_duration = HostWriteHP<false>(hp_base, hp_size, src_data, total_size, page_size);
            }
        }
        thread.join();

        if (configured_writers.has_value()) {
            auto dev_cycles = this->GetWordsFromDevice(configured_writers.value(), dev_addrs.cycles);
            auto dev_bytes_write = this->GetWordsFromDevice(configured_writers.value(), dev_addrs.wr_bytes);
            auto dev_clk = tt::Cluster::instance().get_device_aiclk(device->id()) * 1e6;  // Hz
            double all_cores_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end());
            double all_cores_bytes_read = std::reduce(dev_bytes_write.begin(), dev_bytes_write.end());

            total_device_time += all_cores_cycles / dev_clk;
            total_device_bytes += all_cores_bytes_read;
        }

        state.SetIterationTime(hp_duration.count());
        total_iteration_time += hp_duration.count();
    }

    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["dev_bandwidth_per_second"] = (total_device_bytes / total_device_time);
    state.counters["dev_bytes"] = total_device_bytes;
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = page_size;
    state.counters["num_writers"] = num_writers;
}

// 2 writer kernels writing to different hugepages
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_2_Writers_DifferentPage)(benchmark::State& state) {
    for (auto _ : state) {
    }
}

// 2 MMIO devices reading their hugepage. Host memory will probably get saturated.
// NOTE: From the device perspective, hugepage is at 0. It's not explicit in this benchmark, but
// each MMIO device will be reading a different page.
// The address gets mapped to the addr with the BAR which is configured by UMD.
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_2_MMIO_Devices_Reading_DifferentPage)(benchmark::State& state) {
    constexpr uint32_t k_ReadersEachDevice = 1;
    constexpr uint32_t k_NumThreads = k_ReadersEachDevice * 2;  // Two devices
    constexpr uint32_t k_TotalSize = 1_GB;
    constexpr uint32_t k_PageSize = 32_KB;

    const auto dev_addrs = this->GetDevAddrMap();

    double total_device_bytes = 0;

    // Open another device
    if (tt::Cluster::instance().number_of_pci_devices() < 2) {
        state.SkipWithMessage("At least two MMIO devices are required for this benchmark");
        return;
    }

    std::vector<IDevice*> devices{
        this->device,
        CreateDevice(this->device->id() + 1, 1),
    };

    for (auto _ : state) {
        auto earliest_start = std::numeric_limits<double>::max();
        auto latest_end = std::numeric_limits<double>::min();

        std::vector<std::thread> threads(k_NumThreads);
        std::atomic<bool> start_flag{false};
        std::atomic<int> threads_ready{0};
        std::vector<double> start_times(k_NumThreads);
        std::vector<double> end_times(k_NumThreads);
        std::vector<CoreRange> reader_core_ranges;

        auto make_program_thread = [](int i,
                                      std::atomic<int>& thread_started_counter,
                                      std::atomic<bool>& start_signal,
                                      IDevice* device,
                                      Program& pgm,
                                      std::vector<double>& start_times,
                                      std::vector<double>& end_times) {
            return std::thread([&]() {
                thread_started_counter++;
                while (!start_signal.load()) {
                    std::this_thread::yield();
                }

                start_times[i] = get_current_time_seconds();
                detail::LaunchProgram(device, pgm, true);
                end_times[i] = get_current_time_seconds();
            });
        };

        for (int i = 0; i < devices.size(); ++i) {
            auto program = CreateProgram();
            const auto hp_size = GetHostHugePageSize(devices[i]->id());
            reader_core_ranges.push_back(
                ConfigureKernels(program, dev_addrs, 0, k_ReadersEachDevice, false, k_TotalSize, k_PageSize, hp_size)
                    .value());
            threads.push_back(
                make_program_thread(i, threads_ready, start_flag, devices[i], program, start_times, end_times));
        }

        while (!threads_ready.load()) {
            std::this_thread::yield();
        }

        start_flag.store(true);

        for (auto& thread : threads) {
            thread.join();
        }

        double iteration_time = 0;
        for (int i = 0; i < devices.size(); ++i) {
            auto dev_bytes_read = this->GetWordsFromDevice(reader_core_ranges[i], dev_addrs.rd_bytes);
            auto dev_clk = tt::Cluster::instance().get_device_aiclk(devices[i]->id()) * 1e6;  // Hz
            total_device_bytes += std::reduce(dev_bytes_read.begin(), dev_bytes_read.end());
            earliest_start = std::max(earliest_start, start_times[i]);
            latest_end = std::max(latest_end, end_times[i]);
        }

        state.SetIterationTime(latest_end - earliest_start);
    }

    state.SetBytesProcessed(total_device_bytes);

    // Close devices not managed by the bench fixture
    for (auto& device_ptr : devices) {
        if (device_ptr->id() != this->device->id()) {
            tt::tt_metal::CloseDevice(device_ptr);
        }
    }
}

// 2 MMIO devices writing the same hugepage. Host memory will probably get saturated
// NOTE: From the device perspective, hugepage is at 0. It's not explicit in this benchmark, but
// each MMIO device will be reading a different page.
// The address gets mapped to the addr with the BAR which is configured by UMD.
BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_2_MMIO_Devices_Writing_DifferentPage)(benchmark::State& state) {}

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("0_Host_Write_HP_N_Readers")
    ->ArgsProduct({
        {1_GB},                                // Total size
        {4_KB, 16_KB, 32_KB},                  // Page size
        {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32},  // Num kernels
        {0},                                   // Cached vector
        {1},                                   // Host Copy
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("1_Host_Write_HP_N_Readers_Cached_Vector")
    ->ArgsProduct({
        {1_GB},                                // Total size
        {4_KB, 16_KB, 32_KB},                  // Page size
        {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32},  // Num kernels
        {1},                                   // Cached vector
        {1},                                   // Host copy
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("2_N_Readers_No_Host_Copy")
    ->ArgsProduct({
        {1_GB},                                // Total size
        {4_KB, 16_KB, 32_KB},                  // Page size
        {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32},  // Num kernels
        {0},                                   // Cached vector
        {0},                                   // Host copy
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Threads)
    ->Name("3_Host_Write_HP_N_Threads_No_Kernels")
    ->ArgsProduct({
        {1_GB},                                                   // Total size
        {4_KB, 16_KB, 32_KB},                                     // Page size
        {0},                                                      // Num kernels
        {0},                                                      // Cached vector
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Thread count
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Threads)
    ->Name("4_Host_Write_HP_N_Threads_Cached_Vector_No_Kernels")
    ->ArgsProduct({
        {1_GB},                                                   // Total size
        {4_KB, 16_KB, 32_KB},                                     // Page size
        {0},                                                      // Num kernels
        {1},                                                      // Cached vector
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Thread count
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_BatchSizing)
    ->Name("5_MyMemcpyToDevice_Sizing")
    ->ArgsProduct({
        {1_GB},                                                             // Total size
        {4, 8, 16, 32, 64, 128, 256, 512, 1_KB, 4_KB, 8_KB, 16_KB, 32_KB},  // Size
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_DifferentPage)->Name("6_HostHP_1_Reader_DifferentPage");

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Writers)
    ->Name("7_Host_Write_HP_1_Writer")
    ->ArgsProduct({
        {512_MB},              // Total size. Half of 1GB hugepage.
        {4_KB, 16_KB, 32_KB},  // Page size
        {1},                   // Num kernels
        {0},                   // Cached vector
        {1},                   // Host Copy
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Writers)
    ->Name("8_Writer_Kernel_Only")
    ->ArgsProduct({
        {512_MB},              // Total size. Half of 1GB hugepage.
        {4_KB, 16_KB, 32_KB},  // Page size
        {1, 2, 3, 4},          // Num kernels
        {0},                   // Cached vector
        {0},                   // Host Copy
    });

BENCHMARK_MAIN();
