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

struct DispatcherTestParams {
    uint32_t transfer_size_bytes;
    uint32_t num_iterations;
    uint32_t dram_data_size_words;
};

// Forward declare the accessor if not already available
class FDMeshCQTestAccessor {
public:
    static tt_metal::SystemMemoryManager& sysmem(tt_metal::distributed::FDMeshCommandQueue& cq) {
        return cq.reference_sysmem_manager();
    }
};

constexpr uint32_t DEFAULT_ITERATIONS = 1;
constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

// Using UnitMeshCQSingleCardFixture fixture for this test
// UnitMeshCQSingleCardFixture is a special case of MeshDispatchFixture
// UnitMeshCQSingleCardFixture creates a single card of unit meshes
// and only works with fast dispatch
// MeshDispatchFixture uses all available devices
// and works with fast and slow dispatch
class DispatcherTestFixture : public tt_metal::UnitMeshCQSingleCardFixture,
                              public ::testing::WithParamInterface<DispatcherTestParams> {
public:
    uint32_t transfer_size_bytes_;
    uint32_t num_iterations_;
    uint32_t dram_data_size_words_;

    void SetUp() override {
        // This test requires Fast Dispatch mode
        // if (this->IsSlowDispatch()) {
        //     GTEST_SKIP() << "This test requires Fast Dispatch (unset TT_METAL_SLOW_DISPATCH_MODE)";
        // }

        // TODO: This tests needs to transfer 12288 x 3 words = 36864 words = 147456 bytes = 147.456 KB
        //  of data. But the below settings and max size of prefetch command buffer is too small.
        //  We need to increase the max size of prefetch command buffer to 147.456 KB
        //  to match the original test.

        // // Override BEFORE base SetUp initializes MetalContext/CQs
        // auto& ctx = tt::tt_metal::MetalContext::instance();
        // tt::tt_metal::DispatchSettings::initialize(ctx.get_cluster());
        // auto& s = tt::tt_metal::DispatchSettings::get(CoreType::WORKER, /*num_hw_cqs*/ 1);

        // uint32_t new_max = 3200000;
        // s.prefetch_max_cmd_size(new_max)
        // .prefetch_cmddat_q_size(new_max * 2)  // required: cmddat >= 2 * max_cmd
        // .build();

        tt_metal::UnitMeshCQSingleCardFixture::SetUp();
        const auto params = GetParam();
        this->transfer_size_bytes_ = params.transfer_size_bytes;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;

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
};

using namespace tt::tt_metal;

// Linear Write Unicast
TEST_P(DispatcherTestFixture, LinearWriteUnicast) {
    log_info(tt::LogTest, "DispatcherTestFixture - LinearWriteUnicast (Fast Dispatch) - Test Start");

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

    log_info(tt::LogTest, "Transfer size: {} bytes, Iterations: {}", transfer_size, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    // Needs a worker core range because DeviceData needs to work
    // for unicast and multicast writes
    // for unicast, the worker range is just the first worker
    CoreRange worker_range = {first_worker, first_worker};
    // CoreCoord phys_target_core = device->worker_core_from_logical_core(first_worker);

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
    // Compute NOC for virtual worker on the dispatch downstream NOC
    // There was a mistake here
    CoreCoord virt_worker = device->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy = device->get_noc_unicast_encoding(k_dispatch_downstream_noc, virt_worker);

    // Setup DeviceData for validation
    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    // Calculate command buffer size
    // take a loop at device command calculator
    uint32_t cmd_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    uint32_t write_size = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge) + transfer_size, cmd_alignment);
    // uint32_t wait_size  = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), cmd_alignment);
    // uint32_t term_size  = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), cmd_alignment);

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
            0,                   // num_mcast_dests (0 = unicast)
            noc_xy,              // NOC coordinates
            l1_addr,             // destination address
            transfer_size,       // data size
            host_payload.data()  // payload data
        );

        // Add a barrier wait command to ensure the write is
        // complete before the next iteration starts
        //  dc.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);

        // what does the below exactly do?
        // works with and without the below
        // dc.align_write_offset();

        // Track expected data for validation
        for (size_t j = 0; j < host_payload.size(); j++) {
            device_data.push_one(first_worker, host_payload[j]);
        }
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(dc.write_offset_bytes(), max_fetch);

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
    EXPECT_TRUE(pass) << "Dispatcher Linear Write Unicast test failed validation";
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

// Linear Write Multicast
TEST_P(DispatcherTestFixture, LinearWriteMulticast) {
    log_info(tt::LogTest, "DispatcherTestFixture - LinearWriteMulticast (Fast Dispatch) - Test Start");

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

    log_info(tt::LogTest, "Transfer size: {} bytes, Iterations: {}", transfer_size, num_iterations);

    // Setup target worker core
    CoreCoord first_worker = {0, 1};
    CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    // Multicast write range
    CoreRange worker_range = {first_worker, last_worker};

    // Get L1 base address (use allocator base for FD mode)
    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
    // Compute NOC for virtual worker on the dispatch downstream NOC
    // There was a mistake here
    CoreCoord first_virt_worker = device->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    CoreCoord last_virt_worker = device->virtual_core_from_logical_core(last_worker, CoreType::WORKER);
    uint32_t noc_xy =
        device->get_noc_multicast_encoding(k_dispatch_downstream_noc, CoreRange(first_virt_worker, last_virt_worker));

    // Setup DeviceData for validation
    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    // Calculate command buffer size
    // take a loop at device command calculator
    uint32_t cmd_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    uint32_t write_size = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge) + transfer_size, cmd_alignment);
    // uint32_t wait_size  = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), cmd_alignment);
    // uint32_t term_size  = tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), cmd_alignment);

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
            worker_range.size(),  // num_mcast_dests
            noc_xy,               // NOC coordinates
            l1_addr,              // destination address
            transfer_size,        // data size
            host_payload.data()   // payload data
        );

        // Add a barrier wait command to ensure the write is
        // complete before the next iteration starts
        //  dc.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);

        // what does the below exactly do?
        // works with and without the below
        // dc.align_write_offset();

        // Track expected data for validation
        for (size_t j = 0; j < host_payload.size(); j++) {
            device_data.push_range(worker_range, host_payload[j], true);
        }
    }

    auto max_fetch = tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    ASSERT_LE(dc.write_offset_bytes(), max_fetch);

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
    EXPECT_TRUE(pass) << "Dispatcher Linear Write Multicast test failed validation";
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
TEST_P(DispatcherTestFixture, LinearWritePagedDRAM) {
    log_info(tt::LogTest, "DispatcherTestFixture - LinearWritePagedDRAM (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto mesh_device = this->devices_[0];
    auto& mcq = mesh_device->mesh_command_queue();
    auto& fdcq = dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
    // Borrow SystemMemoryManager from existing FDMeshCommandQueue
    auto& mgr = FDMeshCQTestAccessor::sysmem(fdcq);
    // Get the first device
    auto device = mesh_device->get_devices()[0];

    // Test parameters
    // uint32_t page_size = this->get_transfer_size_bytes();
    uint32_t num_iterations = this->get_num_iterations();
    uint32_t dram_data_size_words = this->get_dram_data_size_words();
    uint32_t start_page = 0;
    // lps = log page size = 4 in run_cpp_fd2_tests.sh
    uint32_t page_size = 16;
    // Set the Number of pages
    uint32_t num_pages = 512;
    // Convert page size to words
    uint32_t page_size_words = page_size / sizeof(uint32_t);

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
    uint32_t page_size_alignment_bytes = device->allocator()->get_alignment(BufferType::DRAM);
    uint32_t num_banks = device->allocator()->get_num_banks(BufferType::DRAM);

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
    // Generate commands using DeviceCommand helper
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // For the CMD generation, start_page is 8 bits, so much wrap around, and increase base_addr instead based on
        // page size, which assumes page size never changed between calls to this function (checked above).
        uint32_t bank_offset = tt::align(page_size, page_size_alignment_bytes) * (absolute_start_page / num_banks);
        // TODO: make this take the latest address, change callers to not manage this
        uint32_t base_addr = device_data.get_base_result_addr(tt::CoreType::DRAM) + bank_offset;
        uint16_t start_page_cmd = absolute_start_page % num_banks;

        // Build payload for all pages
        std::vector<uint32_t> host_payload;
        host_payload.reserve(page_size_words * num_pages);
        // Note: the dst address marches in unison regardless of whether or not a core is written to
        for (size_t page = 0; page < num_pages; ++page) {
            uint32_t page_id = absolute_start_page + page;
            uint32_t bank_id = page_id % num_banks;

            // Determine logical DRAM bank core for expected model
            auto dram_channel = device->allocator()->get_dram_channel_from_bank_id(bank_id);
            CoreCoord dram_core = device->logical_core_from_dram_channel(dram_channel);

            for (size_t word = 0; word < page_size_words; ++word) {
                uint32_t datum = static_cast<uint32_t>(std::rand());
                host_payload.push_back(datum);
                device_data.push_one(dram_core, bank_id, datum);
            }

            // Padding ensures expected data respects alignment like device buffer
            device_data.pad(dram_core, bank_id, page_size_alignment_bytes);
        }

        // Emit one CQ_WRITE_PAGED covering all pages (inline data)
        // false: no hugepage_write -> test hangs
        // true : do hugepage_write -> test passes
        dc.add_dispatch_write_paged<true>(
            true,  // flush prefetch: determines whether data is immediately flushed to the dispatcher or assembled
                   // out-of-line
            1,     // is_dram
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
    ASSERT_LE(dc.write_offset_bytes(), max_fetch);

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

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatcherTestFixture,
    ::testing::Values(
        DispatcherTestParams{256, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS},
        DispatcherTestParams{1024, DEFAULT_ITERATIONS, DRAM_DATA_SIZE_WORDS}),
    [](const testing::TestParamInfo<DispatcherTestParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words";
    });

}  // namespace dispatcher_tests
}  // namespace tt::tt_dispatch
