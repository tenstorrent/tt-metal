// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <chrono>
#include <ctime>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"

// TODO: To keep sizing perfectly in sync with production,
//  consider using DeviceCommandCalculator in tests for size
//  computation before allocating the issue queue (mirrors how
//  assemble_device_commands sizes in production).

// TODO: clean up these globals
bool debug_g = false;
bool use_coherent_data_g = false;
uint32_t dispatch_buffer_page_size_g = 4096;
uint32_t min_xfer_size_bytes_g = 16;
uint32_t max_xfer_size_bytes_g = 4096;
bool send_to_all_g = false;
bool perf_test_g = false;
uint32_t hugepage_issue_buffer_size_g;
#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"

namespace tt::tt_dispatch {
namespace dispatcher_tests {

struct LinearWriteParams {
    uint32_t transfer_size_bytes;
    uint32_t num_iterations;
    uint32_t dram_data_size_words;
    bool is_mcast;
};

struct PagedWriteParams {
    uint32_t page_size;
    uint32_t num_pages;
    uint32_t num_iterations;
    uint32_t dram_data_size_words;
    bool is_dram;
};

// Forward declare the accessor if not already available
class FDMeshCQTestAccessor {
public:
    static tt_metal::SystemMemoryManager& sysmem(tt_metal::distributed::FDMeshCommandQueue& cq) {
        return cq.reference_sysmem_manager();
    }
};

constexpr uint32_t DEFAULT_ITERATIONS = 3;
constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

// Using UnitMeshCQSingleCardFixture fixture for this test
// UnitMeshCQSingleCardFixture is a special case of MeshDispatchFixture
// UnitMeshCQSingleCardFixture creates a single card of unit meshes
// and only works with fast dispatch
// MeshDispatchFixture uses all available devices
// and works with fast and slow dispatch
class DispatchLinearWriteTestFixture : public tt_metal::UnitMeshCQSingleCardFixture,
                                       public ::testing::WithParamInterface<LinearWriteParams> {
public:
    uint32_t transfer_size_bytes_;
    uint32_t num_iterations_;
    uint32_t dram_data_size_words_;
    bool is_mcast_;

    void SetUp() override {
        // This test requires Fast Dispatch mode
        // if (this->IsSlowDispatch()) {
        //     GTEST_SKIP() << "This test requires Fast Dispatch (unset TT_METAL_SLOW_DISPATCH_MODE)";
        // }

        tt_metal::UnitMeshCQSingleCardFixture::SetUp();

        // TODO: This tests needs to transfer 12288 x 3 words = 36864 words = 147456 bytes = 147.456 KB
        //  of data. But the below settings and max size of prefetch command buffer is too small.
        //  We need to increase the max size of prefetch command buffer to 147.456 KB
        //  to match the original test.

        // // Override BEFORE base SetUp initializes MetalContext/CQs
        // Pick a value just above your need (147.456 KB) but small enough to fit L1.
        // cmddat must be >= 2 * max_cmd. Defaults: scratch ~128KB, ringbuffer ~1024KB, dispatch ~512KB.
        // constexpr uint32_t new_max = 192 * 1024; // 192KB
        // constexpr uint32_t new_cmddat = new_max * 2; // 384KB

        // for (auto core_type : {tt::CoreType::WORKER, tt::CoreType::ETH}) {
        //     auto& s = tt::tt_metal::DispatchSettings::get(core_type, /*num_hw_cqs*/ 1);
        //     s.prefetch_max_cmd_size(new_max)
        //     .prefetch_cmddat_q_size(new_cmddat)
        //     // If you still overflow L1, reduce ringbuffer a bit to make room:
        //     // .prefetch_ringbuffer_size(512 * 1024)
        //     .build();
        // }

        const auto params = GetParam();
        this->transfer_size_bytes_ = params.transfer_size_bytes;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
        this->is_mcast_ = params.is_mcast;

        auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        // ASSERT_LE(new_max, max_fetch);
        log_info(tt::LogTest, "Max fetch: {}", max_fetch);

        // Initialize random seed
        uint32_t seed = static_cast<uint32_t>(std::time(nullptr));
        std::srand(seed);
    }

    uint32_t get_transfer_size_bytes() const { return transfer_size_bytes_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
    bool get_is_mcast() const { return is_mcast_; }
};

// Paged Writes to L1/DRAM
class DispatchPagedWriteTestFixture : public tt_metal::UnitMeshCQSingleCardFixture,
                                      public ::testing::WithParamInterface<PagedWriteParams> {
public:
    uint32_t page_size_;
    uint32_t num_pages_;
    uint32_t num_iterations_;
    uint32_t dram_data_size_words_;
    bool is_dram_;

    void SetUp() override {
        // This test requires Fast Dispatch mode
        // if (this->IsSlowDispatch()) {
        //     GTEST_SKIP() << "This test requires Fast Dispatch (unset TT_METAL_SLOW_DISPATCH_MODE)";
        // }

        tt_metal::UnitMeshCQSingleCardFixture::SetUp();

        // // TODO: This tests needs to transfer 12288 x 3 words = 36864 words = 147456 bytes = 147.456 KB
        // //  of data. But the below settings and max size of prefetch command buffer is too small.
        // //  We need to increase the max size of prefetch command buffer to 147.456 KB
        // //  to match the original test.
        // // BEFORE base SetUp
        // auto& ctx = tt::tt_metal::MetalContext::instance();
        // tt::tt_metal::DispatchSettings::initialize(ctx.get_cluster());

        // Pick a value just above your need (147.456 KB) but small enough to fit L1.
        // cmddat must be >= 2 * max_cmd. Defaults: scratch ~128KB, ringbuffer ~1024KB, dispatch ~512KB.
        // constexpr uint32_t new_max = 192 * 1024; // 192KB
        // constexpr uint32_t new_cmddat = new_max * 2; // 384KB

        // for (auto core_type : {tt::CoreType::WORKER, tt::CoreType::ETH}) {
        //     auto& s = tt::tt_metal::DispatchSettings::get(core_type, /*num_hw_cqs*/ 1);
        //     s.prefetch_max_cmd_size(new_max)
        //     .prefetch_cmddat_q_size(new_cmddat)
        //     // If you still overflow L1, reduce ringbuffer a bit to make room:
        //     // .prefetch_ringbuffer_size(512 * 1024)
        //     .build();
        // }

        const auto params = GetParam();
        this->page_size_ = params.page_size;
        this->num_pages_ = params.num_pages;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
        this->is_dram_ = params.is_dram;

        auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        // ASSERT_LE(new_max, max_fetch);
        log_info(tt::LogTest, "Max fetch: {}", max_fetch);

        // Initialize random seed
        uint32_t seed = static_cast<uint32_t>(std::time(nullptr));
        std::srand(seed);
    }

    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_num_pages() const { return num_pages_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
    bool get_is_dram() const { return is_dram_; }
};

class DispatchPackedWriteTestFixture : public tt_metal::UnitMeshCQSingleCardFixture,
                                       public ::testing::WithParamInterface<LinearWriteParams> {
public:
    uint32_t transfer_size_bytes_;
    uint32_t num_iterations_;
    uint32_t dram_data_size_words_;
    bool is_mcast_;

    void SetUp() override {
        // This test requires Fast Dispatch mode
        // if (this->IsSlowDispatch()) {
        //     GTEST_SKIP() << "This test requires Fast Dispatch (unset TT_METAL_SLOW_DISPATCH_MODE)";
        // }

        tt_metal::UnitMeshCQSingleCardFixture::SetUp();

        // TODO: This tests needs to transfer 12288 x 3 words = 36864 words = 147456 bytes = 147.456 KB
        //  of data. But the below settings and max size of prefetch command buffer is too small.
        //  We need to increase the max size of prefetch command buffer to 147.456 KB
        //  to match the original test.

        // // Override BEFORE base SetUp initializes MetalContext/CQs
        // Pick a value just above your need (147.456 KB) but small enough to fit L1.
        // cmddat must be >= 2 * max_cmd. Defaults: scratch ~128KB, ringbuffer ~1024KB, dispatch ~512KB.
        // constexpr uint32_t new_max = 192 * 1024; // 192KB
        // constexpr uint32_t new_cmddat = new_max * 2; // 384KB

        // for (auto core_type : {tt::CoreType::WORKER, tt::CoreType::ETH}) {
        //     auto& s = tt::tt_metal::DispatchSettings::get(core_type, /*num_hw_cqs*/ 1);
        //     s.prefetch_max_cmd_size(new_max)
        //     .prefetch_cmddat_q_size(new_cmddat)
        //     // If you still overflow L1, reduce ringbuffer a bit to make room:
        //     // .prefetch_ringbuffer_size(512 * 1024)
        //     .build();
        // }

        const auto params = GetParam();
        this->transfer_size_bytes_ = params.transfer_size_bytes;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
        this->is_mcast_ = params.is_mcast;

        auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        // ASSERT_LE(new_max, max_fetch);
        log_info(tt::LogTest, "Max fetch: {}", max_fetch);

        // Initialize random seed
        uint32_t seed = static_cast<uint32_t>(std::time(nullptr));
        std::srand(seed);
    }

    uint32_t get_transfer_size_bytes() const { return transfer_size_bytes_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
    bool get_is_mcast() const { return is_mcast_; }
};

using namespace tt::tt_metal;

// Linear Write Unicast/Multicast
TEST_P(DispatchLinearWriteTestFixture, LinearWrite) {
    log_info(tt::LogTest, "DispatchLinearWriteTestFixture - LinearWrite (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto mesh_device = this->devices_[0];
    auto& mcq = mesh_device->mesh_command_queue();
    auto& fdcq = dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
    // Borrow SystemMemoryManager from existing FDMeshCommandQueue
    auto& mgr = FDMeshCQTestAccessor::sysmem(fdcq);
    // Get the first device
    auto device = mesh_device->get_devices()[0];

    // Test parameters
    uint32_t transfer_size = this->get_transfer_size_bytes();
    uint32_t num_iterations = this->get_num_iterations();
    uint32_t dram_data_size_words = this->get_dram_data_size_words();
    // Convert transfer size to words
    uint32_t transfer_size_words = transfer_size / sizeof(uint32_t);
    bool is_mcast = this->get_is_mcast();

    log_info(tt::LogTest, "Transfer size: {} bytes, Iterations: {}", transfer_size, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    CoreCoord last_worker = first_worker;
    if (is_mcast) {
        last_worker = {first_worker.x + 1, first_worker.y + 1};
    }
    // Needs a worker core range because DeviceData needs to work
    // for unicast and multicast writes
    // for unicast, the worker range is just the first worker
    CoreRange worker_range = {first_worker, last_worker};

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
    // Compute NOC for virtual worker on the dispatch downstream NOC
    // There was a mistake here
    CoreCoord first_virt_worker = device->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy = device->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
    if (is_mcast) {
        CoreCoord last_virt_worker = device->virtual_core_from_logical_core(last_worker, CoreType::WORKER);
        noc_xy = device->get_noc_multicast_encoding(
            k_dispatch_downstream_noc, CoreRange(first_virt_worker, last_virt_worker));
    }

    // Setup DeviceData for validation
    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    // Calculate command buffer size
    // take a loop at device command calculator
    uint32_t cmd_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    uint32_t write_size = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge) + transfer_size, cmd_alignment);

    uint32_t total_cmd_bytes = num_iterations * (write_size);
    log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

    ASSERT_LE(total_cmd_bytes, mgr.get_issue_queue_limit(fdcq.id()))
        << "Test requires " << total_cmd_bytes << " B, but issue queue limit is "
        << mgr.get_issue_queue_limit(fdcq.id()) << " B";

    // Reserve space from SystemMemoryManager
    void* cmd_buffer_base = mgr.issue_queue_reserve(total_cmd_bytes, fdcq.id());
    ASSERT_TRUE(cmd_buffer_base != nullptr) << "Failed to reserve issue queue space";

    // Use DeviceCommand helper (HugepageDeviceCommand)
    HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

    // Generate commands using DeviceCommand helper
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

        // Generate random payload
        auto host_payload =
            test_utils::generate_uniform_random_vector<uint32_t>(0, 1000, transfer_size_words, std::rand());

        // Add dispatch write command using helper (no manual struct creation)
        // false, false means: no flush prefetch, no inline data -> test fails
        // true, false means: flush prefetch, no inline data -> test hangs
        // false, true means: no flush prefetch, inline data -> test fails and 2nd test hangs
        // true, true means: flush prefetch, inline data -> test passes
        dc.add_dispatch_write_linear<true, true>(
            is_mcast ? worker_range.size() : 0,  // num_mcast_dests (0 = unicast)
            noc_xy,                              // NOC coordinates
            l1_addr,                             // destination address
            transfer_size,                       // data size
            host_payload.data()                  // payload data
        );

        // Add a barrier wait command to ensure the write is
        // complete before the next iteration starts
        //  dc.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);

        // what does the below exactly do?
        // works with and without the below
        // dc.align_write_offset();

        // Track expected data for validation
        for (size_t j = 0; j < host_payload.size(); j++) {
            if (is_mcast) {
                device_data.push_range(worker_range, host_payload[j], true);
            } else {
                device_data.push_one(first_worker, host_payload[j]);
            }
        }
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(write_size, max_fetch);

    // Important: remove this terminate command
    // This stops the dispatcher so the finish event sequence
    // never runs. Dont send terminate command if you want to
    // test the finish event sequence.
    // dc.add_dispatch_terminate();

    ASSERT_LE(dc.write_offset_bytes(), total_cmd_bytes)
        << "HugepageDeviceCommand wrote more bytes (" << dc.write_offset_bytes() << ") than reserved ("
        << total_cmd_bytes << ")";

    // Submit commands to issue queue on the host side
    // Host side memory (hugepages) is used to store the commands
    mgr.issue_queue_push_back(dc.write_offset_bytes(), fdcq.id());
    // Make sure there's enough space in the device-side fetch queue
    mgr.fetch_queue_reserve_back(fdcq.id());
    // after moving the issue queue write pointer,
    // we also need post a prefetch queue entry so the prefetcher
    // actually pulls the commands
    // This resides on the Device L1 memory
    // Write the commands to the device-side fetch queue, notifying the prefetcher
    // that there are new commands to fetch from the issue queue
    auto start = std::chrono::steady_clock::now();
    mgr.fetch_queue_write(dc.write_offset_bytes(), fdcq.id());
    // Wait for completion
    // Manual thing: Completion queue is empty (need to handle manual thing)
    distributed::Finish(mesh_device->mesh_command_queue());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    log_info(tt::LogTest, "Command queue finished");

    // Validate results
    bool pass = device_data.validate(device);
    EXPECT_TRUE(pass) << "Dispatcher Linear Write test failed validation";
    // Restore the original settings
    // DispatchSettings::initialize(original_copy);

    log_info(tt::LogTest, "Ran in {}us (for total iterations: {})", elapsed.count() * 1000 * 1000, num_iterations);
    log_info(tt::LogTest, "Ran in {}us per iteration", elapsed.count() * 1000 * 1000 / num_iterations);

    // Report performance
    if (pass) {
        float total_words = device_data.size();
        log_info(LogTest, "Total words: {}", total_words);
        // total_words *= num_iterations;
        float bw = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << bw;
        log_info(
            LogTest,
            "BW: {} GB/s (from total_words: {} size: {} MB via host_iter: {} prefetcher_iter: {} for num_cores: "
            "{})",
            ss.str(),
            total_words,
            total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
            num_iterations,
            1,
            worker_range.size());
    }
}

// Paged Write CMD to DRAM
TEST_P(DispatchPagedWriteTestFixture, LinearWritePagedDRAM) {
    log_info(tt::LogTest, "DispatchPagedWriteTestFixture - LinearWritePagedDRAM (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto mesh_device = this->devices_[0];
    auto& mcq = mesh_device->mesh_command_queue();
    auto& fdcq = dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
    // Borrow SystemMemoryManager from existing FDMeshCommandQueue
    auto& mgr = FDMeshCQTestAccessor::sysmem(fdcq);
    // Get the first device
    auto device = mesh_device->get_devices()[0];

    // Test parameters
    uint32_t num_iterations = this->get_num_iterations();
    uint32_t dram_data_size_words = this->get_dram_data_size_words();
    uint32_t start_page = 0;
    // lps = log page size = 4 in run_cpp_fd2_tests.sh
    uint32_t page_size = this->get_page_size();
    // Set the Number of pages
    uint32_t num_pages = this->get_num_pages();
    // Convert page size to words
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    // TODO: should there be a bool here for each if(is_dram_uint8) ?
    uint8_t is_dram_uint8 = static_cast<uint8_t>(this->get_is_dram());

    log_info(tt::LogTest, "Page size: {} bytes, Iterations: {}", page_size, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    // Multicast write range
    CoreRange worker_range = {first_worker, last_worker};

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup DeviceData for validation
    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, true, dram_data_size_words);

    // Calculate command buffer size
    // take a look at device command calculator
    uint32_t cmd_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    auto buf_type = is_dram_uint8 ? BufferType::DRAM : BufferType::L1;
    uint32_t page_size_alignment_bytes = device->allocator()->get_alignment(buf_type);
    uint32_t num_banks = device->allocator()->get_num_banks(buf_type);
    tt::CoreType core_type = is_dram_uint8 ? tt::CoreType::DRAM : tt::CoreType::WORKER;

    // Size of one CQ + inline payload for all pages
    uint32_t data_size_bytes = page_size * num_pages;
    uint32_t write_size = tt::align(
        sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + data_size_bytes,
        cmd_alignment);  // CQPrefetchCmd + CQDispatchCmd + data_size_bytes
    uint32_t total_cmd_bytes = num_iterations * (write_size);
    log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

    ASSERT_LE(total_cmd_bytes, mgr.get_issue_queue_limit(fdcq.id()))
        << "Test requires " << total_cmd_bytes << " B, but issue queue limit is "
        << mgr.get_issue_queue_limit(fdcq.id()) << " B";

    // Reserve space from SystemMemoryManager
    void* cmd_buffer_base = mgr.issue_queue_reserve(total_cmd_bytes, fdcq.id());
    ASSERT_TRUE(cmd_buffer_base != nullptr) << "Failed to reserve issue queue space";

    // Use DeviceCommand helper (HugepageDeviceCommand)
    HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);
    uint32_t absolute_start_page = start_page;
    CoreCoord bank_core;
    // Generate commands using DeviceCommand helper
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // For the CMD generation, start_page is 8 bits, so much wrap around, and increase base_addr instead based on
        // page size, which assumes page size never changed between calls to this function (checked above).
        uint32_t bank_offset = tt::align(page_size, page_size_alignment_bytes) * (absolute_start_page / num_banks);
        // TODO: make this take the latest address, change callers to not manage this
        uint32_t base_addr = device_data.get_base_result_addr(core_type) + bank_offset;
        uint16_t start_page_cmd = absolute_start_page % num_banks;

        // Build payload for all pages
        std::vector<uint32_t> host_payload;
        host_payload.reserve(page_size_words * num_pages);
        // Note: the dst address marches in unison regardless of whether or not a core is written to
        for (size_t page = 0; page < num_pages; ++page) {
            uint32_t page_id = absolute_start_page + page;
            uint32_t bank_id = page_id % num_banks;

            // Determine logical DRAM bank core for expected model

            if (is_dram_uint8) {
                auto dram_channel = device->allocator()->get_dram_channel_from_bank_id(bank_id);
                bank_core = device->logical_core_from_dram_channel(dram_channel);
            } else {
                bank_core = device->allocator()->get_logical_core_from_bank_id(bank_id);
            }

            for (size_t word = 0; word < page_size_words; ++word) {
                uint32_t datum = static_cast<uint32_t>(std::rand());
                host_payload.push_back(datum);
                device_data.push_one(bank_core, bank_id, datum);
            }

            // Padding ensures expected data respects alignment like device buffer
            device_data.pad(bank_core, bank_id, page_size_alignment_bytes);
        }

        // Emit one CQ_WRITE_PAGED covering all pages (inline data)
        // false: no hugepage_write -> test hangs
        // true : do hugepage_write -> test passes
        dc.add_dispatch_write_paged<true>(
            true,  // flush prefetch: determines whether data is immediately flushed to the dispatcher or assembled
                   // out-of-line
            is_dram_uint8,       // is_dram
            start_page_cmd,      // start_page
            base_addr,           // base_addr
            page_size,           // page_size
            num_pages,           // pages
            host_payload.data()  // payload data
        );
        absolute_start_page += num_pages;

        // Add a barrier wait command to ensure the write is
        // complete before the next iteration starts
        //  dc.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);

        // what does the below exactly do?
        // works with and without the below0
        // below is redundant when inline_data is true
        // dc.align_write_offset();
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(write_size, max_fetch);

    // Important: remove this terminate command
    // This stops the dispatcher so the finish event sequence
    // never runs. Dont send terminate command if you want to
    // test the finish event sequence.
    // dc.add_dispatch_terminate();

    ASSERT_LE(dc.write_offset_bytes(), total_cmd_bytes)
        << "HugepageDeviceCommand wrote more bytes (" << dc.write_offset_bytes() << ") than reserved ("
        << total_cmd_bytes << ")";

    // Submit commands to issue queue on the host side
    // Host side memory (hugepages) is used to store the commands
    mgr.issue_queue_push_back(dc.write_offset_bytes(), fdcq.id());
    // Make sure there's enough space in the device-side fetch queue
    mgr.fetch_queue_reserve_back(fdcq.id());
    // after moving the issue queue write pointer,
    // we also need post a prefetch queue entry so the prefetcher
    // actually pulls the commands
    // This resides on the Device L1 memory
    // Write the commands to the device-side fetch queue, notifying the prefetcher
    // that there are new commands to fetch from the issue queue
    auto start = std::chrono::steady_clock::now();
    mgr.fetch_queue_write(dc.write_offset_bytes(), fdcq.id());
    // Wait for completion
    // Manual thing: Completion queue is empty (need to handle manual thing)
    distributed::Finish(mesh_device->mesh_command_queue());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    log_info(tt::LogTest, "Command queue finished");

    // Validate results
    bool pass = device_data.validate(device);
    EXPECT_TRUE(pass) << "Dispatcher Paged Write DRAM test failed validation";
    // Restore the original settings
    // DispatchSettings::initialize(original_copy);

    log_info(tt::LogTest, "Ran in {}us (for total iterations: {})", elapsed.count() * 1000 * 1000, num_iterations);
    log_info(tt::LogTest, "Ran in {}us per iteration", elapsed.count() * 1000 * 1000 / num_iterations);

    // Report performance
    if (pass) {
        float total_words = device_data.size();
        log_info(LogTest, "Total words: {}", total_words);
        // total_words *= num_iterations;
        float bw = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << bw;
        log_info(
            LogTest,
            "BW: {} GB/s (from total_words: {} size: {} MB via host_iter: {} prefetcher_iter: {} for num_cores: "
            "{})",
            ss.str(),
            total_words,
            total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
            num_iterations,
            1,
            worker_range.size());
    }
}

// Packed Write
TEST_P(DispatchPackedWriteTestFixture, LinearWritePackedWrite) {
    log_info(tt::LogTest, "DispatchPackedWriteTestFixture - LinearWritePackedWrite (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto mesh_device = this->devices_[0];
    auto& mcq = mesh_device->mesh_command_queue();
    auto& fdcq = dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
    // Borrow SystemMemoryManager from existing FDMeshCommandQueue
    auto& mgr = FDMeshCQTestAccessor::sysmem(fdcq);
    // Get the first device
    auto device = mesh_device->get_devices()[0];

    // Test parameters
    uint32_t num_iterations = this->get_num_iterations();
    uint32_t dram_data_size_words = this->get_dram_data_size_words();
    uint32_t size_bytes = this->get_transfer_size_bytes();
    // Convert transfer size to words
    uint32_t size_words = size_bytes / sizeof(uint32_t);

    log_info(tt::LogTest, "Transfer size: {} bytes, Iterations: {}", size_bytes, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    // Multicast write range
    CoreRange worker_range = {first_worker, last_worker};

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup DeviceData for validation
    DeviceData device_data(
        device, worker_range, l1_base, dram_base, nullptr, /*is_banked*/ false, dram_data_size_words);

    // Randomly pick a non-empty set of worker cores to receive the write
    std::vector<CoreCoord> worker_cores;
    for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; ++y) {
        for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; ++x) {
            if (std::rand() % 2) {
                worker_cores.push_back({x, y});
            }
        }
    }
    if (worker_cores.empty()) {
        worker_cores.push_back(first_worker);
    }
    // Optionally use no_stride (repeat) semantics
    const bool repeat = (std::rand() % 2) != 0;

    // Max subcmds bound from device (same logic as prod)
    const uint32_t packed_write_max_unicast_sub_cmds =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    ASSERT_LE(worker_cores.size(), packed_write_max_unicast_sub_cmds);

    // Build packed subcmds (unicast variant)
    std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds;
    sub_cmds.reserve(worker_cores.size());
    for (const auto& core : worker_cores) {
        const CoreCoord phys = device->worker_core_from_logical_core(core);
        CQDispatchWritePackedUnicastSubCmd s{};
        s.noc_xy_addr = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(phys.x, phys.y);
        sub_cmds.push_back(s);
    }

    // Generate payload to write and expected data (same data for each core)
    std::vector<uint32_t> payload;
    payload.reserve(size_words);
    uint32_t coherent_count = 0;
    const auto& fw = worker_cores[0];
    for (uint32_t i = 0; i < size_words; ++i) {
        uint32_t datum = use_coherent_data_g ? ((fw.x << 16) | (fw.y << 24) | coherent_count++) : std::rand();
        payload.push_back(datum);
    }

    // Add expected results; pad per-core to L1 alignment like device buffer
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    device_data.relevel(tt::CoreType::WORKER);
    // In packed write, we need to capture the write address before
    // updating expected model, so device and expected match
    const uint32_t common_addr = device_data.get_result_data_addr(fw);
    for (const auto& core : worker_cores) {
        for (uint32_t i = 0; i < size_words; ++i) {
            device_data.push_one(core, /*bank*/ 0, payload[i]);
        }
        device_data.pad(core, /*bank*/ 0, l1_alignment);
    }

    // Sizing: inline prefetch (CQPrefetchCmd) + CQ_DISPATCH_CMD_WRITE_PACKED + subcmds + data (1 or N copies)
    const uint32_t pcie_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    const uint32_t sub_cmds_sizeB = worker_cores.size() * sizeof(CQDispatchWritePackedUnicastSubCmd);
    const uint32_t sub_cmds_paddedB = tt::align(sub_cmds_sizeB, l1_alignment);
    const uint32_t data_strideB = tt::align(size_bytes, l1_alignment);
    const uint32_t data_copies = repeat ? 1u : static_cast<uint32_t>(worker_cores.size());
    const uint32_t data_sectionB = data_copies * data_strideB;
    const uint32_t payload_sizeB = tt::align(sizeof(CQDispatchCmd) + sub_cmds_paddedB + data_sectionB, l1_alignment);
    const uint32_t write_sizeB = tt::align(sizeof(CQPrefetchCmd) + payload_sizeB, pcie_alignment);
    const uint32_t total_cmd_bytes = num_iterations * write_sizeB;
    log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

    ASSERT_LE(total_cmd_bytes, mgr.get_issue_queue_limit(fdcq.id()))
        << "Test requires " << total_cmd_bytes << " B, but issue queue limit is "
        << mgr.get_issue_queue_limit(fdcq.id()) << " B";

    // Build data_collection for the helper
    // - no_stride=false: duplicate pointers per subcmd
    // - no_stride=true : only one copy
    std::vector<std::pair<const void*, uint32_t>> data_collection;
    if (repeat) {
        data_collection.emplace_back(payload.data(), size_bytes);
    } else {
        data_collection.resize(worker_cores.size(), {payload.data(), size_bytes});
    }

    // Reserve, write commands
    void* cmd_buffer_base = mgr.issue_queue_reserve(total_cmd_bytes, fdcq.id());
    ASSERT_TRUE(cmd_buffer_base != nullptr) << "Failed to reserve issue queue space";
    HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        dc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
            0,                                           // type
            static_cast<uint16_t>(worker_cores.size()),  // num_sub_cmds
            common_addr,                                 // common_addr
            static_cast<uint16_t>(size_bytes),           // packed_data_sizeB
            payload_sizeB,                               // payload_sizeB
            sub_cmds,                                    // sub_cmds
            data_collection,                             // data_collection
            packed_write_max_unicast_sub_cmds,           // packed_write_max_unicast_sub_cmds
            0,                                           // offset_idx
            repeat,                                      // no_stride
            0);                                          // write_offset_index
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(write_sizeB, max_fetch);

    // Important: remove this terminate command
    // This stops the dispatcher so the finish event sequence
    // never runs. Dont send terminate command if you want to
    // test the finish event sequence.
    // dc.add_dispatch_terminate();

    ASSERT_LE(dc.write_offset_bytes(), total_cmd_bytes)
        << "HugepageDeviceCommand wrote more bytes (" << dc.write_offset_bytes() << ") than reserved ("
        << total_cmd_bytes << ")";

    // Submit commands to issue queue on the host side
    // Host side memory (hugepages) is used to store the commands
    mgr.issue_queue_push_back(dc.write_offset_bytes(), fdcq.id());
    // Make sure there's enough space in the device-side fetch queue
    mgr.fetch_queue_reserve_back(fdcq.id());
    // after moving the issue queue write pointer,
    // we also need post a prefetch queue entry so the prefetcher
    // actually pulls the commands
    // This resides on the Device L1 memory
    // Write the commands to the device-side fetch queue, notifying the prefetcher
    // that there are new commands to fetch from the issue queue
    auto start = std::chrono::steady_clock::now();
    mgr.fetch_queue_write(dc.write_offset_bytes(), fdcq.id());
    // Wait for completion
    // Manual thing: Completion queue is empty (need to handle manual thing)
    distributed::Finish(mesh_device->mesh_command_queue());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    log_info(tt::LogTest, "Command queue finished");

    // Validate results
    bool pass = device_data.validate(device);
    EXPECT_TRUE(pass) << "Dispatcher Packed Write test failed validation";
    // Restore the original settings
    // DispatchSettings::initialize(original_copy);

    log_info(tt::LogTest, "Ran in {}us (for total iterations: {})", elapsed.count() * 1000 * 1000, num_iterations);
    log_info(tt::LogTest, "Ran in {}us per iteration", elapsed.count() * 1000 * 1000 / num_iterations);

    // Report performance
    if (pass) {
        float total_words = device_data.size();
        log_info(LogTest, "Total words: {}", total_words);
        // total_words *= num_iterations;
        float bw = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << bw;
        log_info(
            LogTest,
            "BW: {} GB/s (from total_words: {} size: {} MB via host_iter: {} prefetcher_iter: {} for num_cores: "
            "{})",
            ss.str(),
            total_words,
            total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
            num_iterations,
            1,
            worker_range.size());
    }
}

// Large Packed Write
TEST_P(DispatchPackedWriteTestFixture, LargePackedWrite) {
    log_info(tt::LogTest, "DispatchPackedWriteTestFixture - LargePackedWrite (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto mesh_device = this->devices_[0];
    auto& mcq = mesh_device->mesh_command_queue();
    auto& fdcq = dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
    // Borrow SystemMemoryManager from existing FDMeshCommandQueue
    auto& mgr = FDMeshCQTestAccessor::sysmem(fdcq);
    // Get the first device
    auto device = mesh_device->get_devices()[0];

    // Test parameters
    uint32_t num_iterations = this->get_num_iterations();
    uint32_t dram_data_size_words = this->get_dram_data_size_words();
    uint32_t size_bytes = this->get_transfer_size_bytes();
    // Convert transfer size to words
    uint32_t size_words = size_bytes / sizeof(uint32_t);

    log_info(tt::LogTest, "Transfer size: {} bytes, Iterations: {}", size_bytes, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    // Multicast write range
    CoreRange worker_range = {first_worker, last_worker};

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup DeviceData for validation
    DeviceData device_data(
        device, worker_range, l1_base, dram_base, nullptr, /*is_banked*/ false, dram_data_size_words);

    // Randomly pick a non-empty set of worker cores to receive the write
    std::vector<CoreCoord> worker_cores;
    for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; ++y) {
        for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; ++x) {
            if (std::rand() % 2) {
                worker_cores.push_back({x, y});
            }
        }
    }
    if (worker_cores.empty()) {
        worker_cores.push_back(first_worker);
    }

    // Max subcmds bound from device (same logic as prod)
    const uint32_t packed_write_max_unicast_sub_cmds =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    ASSERT_LE(worker_cores.size(), packed_write_max_unicast_sub_cmds);

    // Generate payload to write and expected data (same data for each core)
    std::vector<uint32_t> payload;
    payload.reserve(size_words);
    uint32_t coherent_count = 0;
    const auto& fw = worker_cores[0];
    for (uint32_t i = 0; i < size_words; ++i) {
        uint32_t datum = use_coherent_data_g ? ((fw.x << 16) | (fw.y << 24) | coherent_count++) : std::rand();
        payload.push_back(datum);
    }

    // Add expected results; pad per-core to L1 alignment like device buffer
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // In packed-large, we prepare multiple transactions. Capture each tx address,
    // then immediately advance the expected model so the next tx uses the next address.
    const uint32_t ntransactions = 2;  // simple deterministic choice

    const uint32_t pcie_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    const uint32_t sub_cmds_sizeB = ntransactions * sizeof(CQDispatchWritePackedLargeSubCmd);
    const uint32_t data_collection_sizeB = ntransactions * size_bytes;
    const uint32_t payload_sizeB = tt::align(
        tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, l1_alignment) + data_collection_sizeB, l1_alignment);
    const uint32_t write_sizeB = tt::align(sizeof(CQPrefetchCmd) + payload_sizeB, pcie_alignment);
    const uint32_t total_cmd_bytes = num_iterations * write_sizeB;
    log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

    ASSERT_LE(total_cmd_bytes, mgr.get_issue_queue_limit(fdcq.id()))
        << "Test requires " << total_cmd_bytes << " B, but issue queue limit is "
        << mgr.get_issue_queue_limit(fdcq.id()) << " B";

    // Reserve, write commands
    void* cmd_buffer_base = mgr.issue_queue_reserve(total_cmd_bytes, fdcq.id());
    ASSERT_TRUE(cmd_buffer_base != nullptr) << "Failed to reserve issue queue space";
    HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        device_data.relevel(tt::CoreType::WORKER);
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds;
        sub_cmds.reserve(ntransactions);

        for (uint32_t t = 0; t < ntransactions; ++t) {
            CoreCoord physical_start = device->worker_core_from_logical_core(worker_range.start_coord);
            CoreCoord physical_end = device->worker_core_from_logical_core(worker_range.end_coord);

            // 1) Capture destination address for this transaction
            uint32_t common_addr = device_data.get_result_data_addr(worker_range.start_coord);

            // 2) Build sub-cmd with this address
            CQDispatchWritePackedLargeSubCmd s{};
            s.noc_xy_addr = tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
                physical_start.x, physical_start.y, physical_end.x, physical_end.y);
            s.addr = common_addr;
            s.length = size_bytes;
            s.num_mcast_dests = worker_range.size();
            s.flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
            sub_cmds.push_back(s);

            // 3) Advance expected model for this tx (so next tx gets the next addr)
            for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; ++y) {
                for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; ++x) {
                    CoreCoord core = {x, y};
                    for (uint32_t i = 0; i < size_words; ++i) {
                        device_data.push_one(core, /*bank*/ 0, payload[i]);
                    }
                    device_data.pad(core, /*bank*/ 0, l1_alignment);
                }
            }
        }
        // Build data as spans of bytes, one blob per transaction
        std::vector<std::vector<uint8_t>> payload_bytes;
        payload_bytes.resize(ntransactions);
        for (uint32_t t = 0; t < ntransactions; ++t) {
            payload_bytes[t].resize(size_bytes);
            std::memcpy(payload_bytes[t].data(), payload.data(), size_bytes);
        }
        std::vector<tt::stl::Span<const uint8_t>> data_spans;
        data_spans.reserve(ntransactions);
        for (uint32_t t = 0; t < ntransactions; ++t) {
            data_spans.emplace_back(payload_bytes[t].data(), payload_bytes[t].size());
        }

        // If you vary sizes/addresses per-iter, rebuild sub_cmds/data and expected model here.
        dc.add_dispatch_write_packed_large(
            /*type*/ 0,
            /*alignment*/ static_cast<uint16_t>(l1_alignment),
            /*num_sub_cmds*/ static_cast<uint16_t>(ntransactions),
            /*sub_cmds*/ sub_cmds,
            /*data_collection*/ data_spans,
            /*data_collection_buffer_ptr*/ nullptr,
            /*offset_idx*/ 0,
            /*write_offset_index*/ 0);
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(write_sizeB, max_fetch);

    // Important: remove this terminate command
    // This stops the dispatcher so the finish event sequence
    // never runs. Dont send terminate command if you want to
    // test the finish event sequence.
    // dc.add_dispatch_terminate();

    ASSERT_LE(dc.write_offset_bytes(), total_cmd_bytes)
        << "HugepageDeviceCommand wrote more bytes (" << dc.write_offset_bytes() << ") than reserved ("
        << total_cmd_bytes << ")";

    // Submit commands to issue queue on the host side
    // Host side memory (hugepages) is used to store the commands
    mgr.issue_queue_push_back(dc.write_offset_bytes(), fdcq.id());
    // Make sure there's enough space in the device-side fetch queue
    mgr.fetch_queue_reserve_back(fdcq.id());
    // after moving the issue queue write pointer,
    // we also need post a prefetch queue entry so the prefetcher
    // actually pulls the commands
    // This resides on the Device L1 memory
    // Write the commands to the device-side fetch queue, notifying the prefetcher
    // that there are new commands to fetch from the issue queue
    auto start = std::chrono::steady_clock::now();
    mgr.fetch_queue_write(dc.write_offset_bytes(), fdcq.id());
    // Wait for completion
    // Manual thing: Completion queue is empty (need to handle manual thing)
    distributed::Finish(mesh_device->mesh_command_queue());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    log_info(tt::LogTest, "Command queue finished");

    // Validate results
    bool pass = device_data.validate(device);
    EXPECT_TRUE(pass) << "Dispatcher Large Packed Write test failed validation";
    // Restore the original settings
    // DispatchSettings::initialize(original_copy);

    log_info(tt::LogTest, "Ran in {}us (for total iterations: {})", elapsed.count() * 1000 * 1000, num_iterations);
    log_info(tt::LogTest, "Ran in {}us per iteration", elapsed.count() * 1000 * 1000 / num_iterations);

    // Report performance
    if (pass) {
        float total_words = device_data.size();
        log_info(LogTest, "Total words: {}", total_words);
        // total_words *= num_iterations;
        float bw = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << bw;
        log_info(
            LogTest,
            "BW: {} GB/s (from total_words: {} size: {} MB via host_iter: {} prefetcher_iter: {} for num_cores: "
            "{})",
            ss.str(),
            total_words,
            total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
            num_iterations,
            1,
            worker_range.size());
    }
}

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchLinearWriteTestFixture,
    ::testing::Values(
        LinearWriteParams{256, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false},
        LinearWriteParams{1024, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false},
        LinearWriteParams{256, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, true},
        LinearWriteParams{1024, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, true}),
    [](const testing::TestParamInfo<LinearWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_" +
               (info.param.is_mcast ? "mcast" : "unicast");
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPagedWriteTestFixture,
    ::testing::Values(
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false},
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, true},
        PagedWriteParams{
            512, 128, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false},  // TODO: increase page size to 2048
        PagedWriteParams{
            512, 128, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, true}),  // TODO: increase page size to 2048
    [](const testing::TestParamInfo<PagedWriteParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.is_dram ? "DRAM" : "L1");
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPackedWriteTestFixture,
    ::testing::Values(
        LinearWriteParams{256, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false},
        LinearWriteParams{1024, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS, false}),
    [](const testing::TestParamInfo<LinearWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_" +
               (info.param.is_mcast ? "mcast" : "unicast");
    });

}  // namespace dispatcher_tests
}  // namespace tt::tt_dispatch
