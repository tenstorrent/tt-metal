// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// TT_METAL_LOGGER_LEVEL=FATAL ./build/test/tt_metal/perf_microbenchmark/3_pcie_transfer/pcie_bench_wormhole_b0
// --benchmark_out=benchdata.txt --benchmark_out_format=csv
//

#include <cassert>
#include <chrono>
#include <future>
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
using namespace tt::tt_metal::dispatch;  // _KB, _MB, _GB
using namespace std::chrono_literals;    // s

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
    uint32_t GetHostHugePageSize() const {
        const auto dut_id = this->device->id();
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dut_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(dut_id);
        return tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
    }

    vector_memcpy_aligned<uint32_t> GenSrcData(uint32_t num_bytes) const {
        std::uniform_int_distribution<uint32_t> distribution(
            std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
        std::default_random_engine generator;

        vector_memcpy_aligned<uint32_t> vec(num_bytes / sizeof(uint32_t));
        std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });

        return vec;
    }

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

    DeviceAddresses GetDevAddrMap() const {
        const auto l1_alignment = hal.get_alignment(HalMemType::L1);
        DeviceAddresses addrs;
        addrs.cycles = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
        addrs.rd_bytes = align_addr(addrs.cycles + sizeof(uint32_t), l1_alignment);
        addrs.unreserved = align_addr(addrs.rd_bytes + sizeof(uint32_t), l1_alignment);
        return addrs;
    }

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

    std::optional<CoreRange> ConfigureReaderKernels(
        Program& program,
        const DeviceAddresses& dev_addrs,
        uint32_t start_y,
        uint32_t num_readers,
        uint32_t total_size,
        uint32_t page_size,
        uint32_t pcie_size,
        uint32_t pcie_offset = 0) const {
        if (!page_size) {
            page_size = total_size;
        }
        if (!num_readers) {
            return {};
        }

        const auto grid_size = device->logical_grid_size();
        const auto max_x = grid_size.x;
        const auto max_y = grid_size.y;

        // Number readers either less than one row
        // or a multiple of the rows
        CoreCoord start_coord{0, start_y};
        CoreCoord end_coord;
        if (num_readers <= max_x) {
            end_coord.x = start_coord.x + num_readers - 1;
            end_coord.y = start_coord.y;
        } else {
            const auto number_of_rows = num_readers / max_x;
            const auto last_row_width = (num_readers % max_x) ? num_readers % max_x : max_x;
            end_coord.x = start_coord.x + last_row_width - 1;
            end_coord.y = number_of_rows - 1;
        }
        CoreRange core_range{start_coord, end_coord};

        [[maybe_unused]] KernelHandle read_kernel = CreateKernel(
            program,
            std::string{k_PcieBenchKernel},
            core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC_0,
                .compile_args =
                    {
                        dev_addrs.unreserved,  // my_rd_dst_addr
                        pcie_offset,  // pcie_rd_base. From the device's perspective the pcie base is 0. device bar is
                                      // mapped to hugepage.
                        pcie_size,    // pcie_rd_size
                        page_size,    // pcie_rd_transfer_size
                        dev_addrs.rd_bytes,  // my_bytes_rd_addr

                        0,  // my_wr_src_addr
                        0,  // pcie_wr_base
                        0,  // pcie_wr_size
                        0,  // pcie_wr_transfer_size
                        0,  // my_bytes_wr_addr

                        total_size,        // total_bytes
                        dev_addrs.cycles,  // cycles
                    },
                .defines = {},
            });

        return core_range;
    }

    MemCpyPcieBench() : benchmark::Fixture{} {
        UseManualTime();
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
        }
    }

    void TearDown(const benchmark::State& state) override {
        tt::DevicePool::instance().close_device(this->device->id());
    }

    ///
    /// @brief Host writing to hugepage with N reader kernels reading from it
    /// @param state Benchmark state
    /// @tparam caching_src_vector Repeat source vector writes to hugepage
    ///
    template <bool caching_src_vector = false>
    PcieTransferResults HostHP_N_Readers_Impl(benchmark::State& state) {
        const auto total_size = state.range(0);
        const auto page_size = state.range(1);
        const auto pages = total_size / page_size;
        const auto num_readers = state.range(2);
        auto src_data = this->GenSrcData(total_size);  // Already aligned
        const auto hp_size = this->GetHostHugePageSize();
        const auto hp_base = this->GetHostHugePage(0);  // Already aligned
        const auto hp_end = reinterpret_cast<uint64_t>(hp_base) + hp_size;
        const auto dev_addrs = this->GetDevAddrMap();
        PcieTransferResults results;

        auto program = Program();
        auto configured_readers =
            ConfigureReaderKernels(program, dev_addrs, 0, num_readers, total_size, page_size, hp_size);

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
        auto hp_duration = HostWriteHP<caching_src_vector>(hp_base, hp_size, src_data, total_size, page_size);
        thread.join();

        if (configured_readers.has_value()) {
            results.host_wait_for_kernels_duration = program_time;

            auto dev_cycles = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.cycles);
            auto dev_bytes_read = this->GetWordsFromDevice(configured_readers.value(), dev_addrs.rd_bytes);
            auto dev_clk = tt::Cluster::instance().get_device_aiclk(device->id()) * 1e6;  // Hz
            double all_cores_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end());
            double all_cores_bytes_read = std::reduce(dev_bytes_read.begin(), dev_bytes_read.end());
            std::chrono::duration<double> kernel_duration{all_cores_cycles / dev_clk};

            results.kernel_duration = kernel_duration;
            results.kernel_bytes_rd = all_cores_bytes_read;
        } else {
            // No readers. Set to 0
            results.kernel_duration = std::chrono::duration<double>{1};
            results.kernel_bytes_rd = 0;
        }

        results.host_hugepage_writing_duration = hp_duration;
        results.host_hugepage_bytes_processed = total_size;

        return results;
    }

    ///
    /// @brief Host writing to hugepage with N threads and M reader kernels reading from it. N threads to split doing
    /// the
    //         work to complete the copy. Each thread will copy Total Size / N amount of bytes.
    /// @param state Benchmark state
    /// @tparam caching_src_vector Repeat source vector writes to hugepage
    ///
    template <bool caching_src_vector = false>
    PcieTransferResults HostHP_N_Threads_M_ReadersImpl(benchmark::State& state) {
        using Timepoint = decltype(std::chrono::high_resolution_clock::now());
        using StartEndTime = std::pair<Timepoint, Timepoint>;

        static_assert((k_MemCpyAlignment & ((k_MemCpyAlignment)-1)) == 0);

        PcieTransferResults results;

        const auto hp_base_addr = this->GetHostHugePage(0);
        const auto hp_size = this->GetHostHugePageSize();
        const auto hp_end = reinterpret_cast<uint64_t>(hp_base_addr) + hp_size;
        const auto total_size = state.range(0);
        const auto page_size = state.range(1);
        auto src_data = this->GenSrcData(total_size);  // Already aligned
        const auto num_threads = static_cast<size_t>(state.range(4));
        const auto bytes_per_thread = ((total_size / num_threads) + (k_MemCpyAlignment)-1) & -(k_MemCpyAlignment);
        const auto last_thread_bytes = total_size - (bytes_per_thread * (num_threads - 1));

        std::vector<std::thread> threads(num_threads);
        std::atomic<bool> start_flag{false};
        std::atomic<int> threads_ready{0};
        std::vector<double> thread_durations(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads[i] = std::thread(
                [&](size_t thread_i) {
                    uint64_t thread_dst = (uint64_t)hp_base_addr + (thread_i * bytes_per_thread);
                    uint64_t thread_bytes = (thread_i == num_threads - 1) ? last_thread_bytes : bytes_per_thread;
                    std::span<uint32_t> thread_src{src_data};
                    thread_src = thread_src.subspan(
                        (thread_i * bytes_per_thread) / sizeof(uint32_t), thread_bytes / sizeof(uint32_t));

                    // Signal ready and wait for start
                    threads_ready++;
                    while (!start_flag.load()) {
                        std::this_thread::yield();
                    }

                    // Actual timed operation
                    auto start = std::chrono::high_resolution_clock::now();
                    this->HostWriteHP<caching_src_vector>(
                        (void*)thread_dst, hp_size, thread_src, thread_bytes, page_size);
                    auto end = std::chrono::high_resolution_clock::now();

                    thread_durations[thread_i] = std::chrono::duration<double>(end - start).count();
                },
                i);
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

        results.host_hugepage_writing_duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        results.host_hugepage_bytes_processed = total_size;

        return results;
    }
};

BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Readers)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto cached_vector = static_cast<bool>(state.range(3));
    double total_device_time = 0;
    double total_device_bytes = 0;
    double total_iteration_time = 0;
    for (auto _ : state) {
        PcieTransferResults res;
        if (cached_vector) {
            res = this->HostHP_N_Readers_Impl<true>(state);
        } else {
            res = this->HostHP_N_Readers_Impl<false>(state);
        }
        state.SetIterationTime(res.host_hugepage_writing_duration.count());
        total_device_time += res.kernel_duration.count();
        total_device_bytes += res.kernel_bytes_rd;
        total_iteration_time += res.host_hugepage_writing_duration.count();
    }

    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["dev_bandwidth_per_second"] = benchmark::Counter(
        (total_device_bytes / total_device_time) *
            total_iteration_time,  // Multiply by total_iteration_time to negate kIsRate to pretty print
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1024);
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = state.range(1);
    state.counters["num_readers"] = state.range(2);
}

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("Host_Write_HP_N_Readers")
    ->ArgsProduct({
        /*Total Size*/ {1_GB},
        /*Page Size*/ {4_KB, 16_KB, 32_KB},
        /*N Reader Kernels*/ {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32},
        /*Cached Vector*/ {0},
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Readers)
    ->Name("Host_Write_HP_N_Readers_HotVector")
    ->ArgsProduct({
        /*Total Size*/ {1_GB},
        /*Page Size*/ {4_KB, 16_KB, 32_KB},
        /*N Reader Kernels*/ {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32},
        /*Cached Vector*/ {1},
    });

BENCHMARK_DEFINE_F(MemCpyPcieBench, BM_HostHP_N_Threads)(benchmark::State& state) {
    const auto total_size = state.range(0);
    const auto cached_vector = static_cast<bool>(state.range(3));
    for (auto _ : state) {
        PcieTransferResults res;
        if (cached_vector) {
            res = this->HostHP_N_Threads_M_ReadersImpl<true>(state);
        } else {
            res = this->HostHP_N_Threads_M_ReadersImpl<false>(state);
        }
        state.SetIterationTime(res.host_hugepage_writing_duration.count());
    }
    state.SetBytesProcessed(total_size * state.iterations());
    state.counters["total_size"] = total_size;
    state.counters["page_size"] = state.range(1);
    state.counters["num_threads"] = state.range(4);
}

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Threads)
    ->Name("Host_Write_HP_N_Threads")
    ->ArgsProduct({
        /*Total Size*/ {1_GB},
        /*Page Size*/ {4_KB, 16_KB, 32_KB},
        /*M Reader Kernels*/ {0},
        /*Cached Vector*/ {0},
        /*N Threads*/ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    });

BENCHMARK_REGISTER_F(MemCpyPcieBench, BM_HostHP_N_Threads)
    ->Name("Host_Write_HP_N_Threads_HotVector")
    ->ArgsProduct({
        /*Total Size*/ {1_GB},
        /*Page Size*/ {4_KB, 16_KB, 32_KB},
        /*M Reader Kernels*/ {0},
        /*Cached Vector*/ {1},
        /*N Threads*/ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    });

BENCHMARK_MAIN();
