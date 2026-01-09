// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <chrono>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"
#include <impl/dispatch/dispatch_query_manager.hpp>

/*
 * FAST PREFETCHER MICROBENCHMARK SUITE
 *
 * Architecture Overview:
 * This test suite validates the Fast Dispatcher (FD) kernel mechanisms by bypassing the
 * standard high-level Enqueue APIs and directly constructing low-level command sequences.
 *
 * The test flow follows a "Shadow Model" pattern:
 * 1. Plan: Determine transfer sizes, destinations, and command types.
 * 2. Shadow: Update `Common::DeviceData` (the expectation model) to reflect what should happen.
 * 3. Build: Use `DeviceCommand` and `DeviceCommandCalculator` to construct binary
 *    command packets (HostMemDeviceCommand) exactly as the runtime would.
 * 4. Execute and Validate: Push these raw commands directly into the Issue Queue and notify the hardware.
 *    Read back device memory and compare against `Common::DeviceData` to validate the correctness of the command
 * execution.
 *
 * Key Concepts:
 * - Issue Queue: Host-resident ring buffer where commands are written.
 * - Fetch Queue: Device-resident ring buffer (pointers) telling the Prefetcher where to look.
 * - Prefetcher: Kernel that pulls commands from Host/DRAM and relays them.
 * - Dispatcher: Kernel that parses commands and issues writes/signals to Worker cores.
 */

namespace tt::tt_metal::tt_dispatch_tests::prefetcher_tests {

constexpr uint32_t DEFAULT_ITERATIONS = 5;
constexpr uint32_t DEFAULT_ITERATIONS_SMOKE_RANDOM = 1000;
constexpr uint32_t DEVICE_DATA_SIZE = 768 * 1024;
constexpr uint32_t DRAM_PAGE_SIZE_DEFAULT = 1024;
constexpr uint32_t DRAM_PAGES_TO_READ_DEFAULT = 16;
constexpr uint32_t DEFAULT_SCRATCH_DB_SIZE = 16 * 1024;

// Params that control the data volume, iteration count
// for the packed / large packed write test
struct PagedReadParams {
    uint32_t page_size{};       // Page size in bytes
    uint32_t num_pages{};       // Number of pages
    uint32_t num_iterations{};  // Number of iterations for the test
    uint32_t dram_data_size_words{};
    bool use_exec_buf{};  // Whether to use exec buff
};

namespace DeviceDataUpdater {

void update_paged_dram_read(
    const CoreRange& workers,
    Common::DeviceData& device_data,
    const CoreCoord& bank_core,
    uint32_t bank_id,
    uint32_t bank_offset,
    uint32_t page_size_words) {
    for (uint32_t j = 0; j < page_size_words; j++) {
        uint32_t datum = device_data.at(bank_core, bank_id, bank_offset + j);
        device_data.push_range(workers, datum, false);
    }
}

// Write the expected dispatch info into the device data for validation
// This should match the data inside completion queue on the host side
// The shadow model must have command header + generated payload for validation since dispatch
// copies both into completion buffer
void update_host_data(Common::DeviceData& device_data, const std::vector<uint32_t>& data, uint32_t data_size_bytes) {
    uint32_t data_size_words = data_size_bytes / sizeof(uint32_t);
    CQDispatchCmd expected_cmd{};
    expected_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
    // Include cmd in transfer
    expected_cmd.write_linear_host.length = data_size_bytes + sizeof(CQDispatchCmd);
    expected_cmd.write_linear_host.is_event = false;
    expected_cmd.write_linear_host.pad1 = 0;
    expected_cmd.write_linear_host.pad2 = HostMemDeviceCommand::random_padding_value();

    uint32_t* cmd_as_words = reinterpret_cast<uint32_t*>(&expected_cmd);
    for (uint32_t i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
        device_data.push_one(device_data.get_host_core(), 0, cmd_as_words[i]);
    }

    for (uint32_t i = 0; i < data_size_words; i++) {
        device_data.push_one(device_data.get_host_core(), 0, data[i]);
    }
}

// Mirrors a read-from-source operation
// Looks up existing data in device_data and pushes those into destination workers
// Used when hardware will read from DRAM/L1 and copy whatever is already modeled there
void update_read(
    const CoreCoord& worker_core,
    Common::DeviceData& device_data,
    const CoreCoord& bank_core,
    uint32_t bank_id,
    uint32_t bank_offset,
    uint32_t page_size_words) {
    for (uint32_t j = 0; j < page_size_words; j++) {
        uint32_t datum = device_data.at(bank_core, bank_id, bank_offset + j);
        device_data.push_one(worker_core, datum);
    }
}
}  // namespace DeviceDataUpdater

// Forward declare for CommandBuilder::build_prefetch_ringbuffer_relay
class PrefetcherRingbufferReadTestFixture;

// Host-side helpers used by tests to emit the same CQ commands
// that prefetcher/dispatcher code emits. This namespace replicates the production code's command generation logic
// for testing purposes.
namespace CommandBuilder {

HostMemDeviceCommand build_dispatch_terminate() {
    bool dispatch_sub_enabled = MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled();
    DeviceCommandCalculator calc;
    calc.add_dispatch_wait();
    calc.add_dispatch_terminate();
    if (dispatch_sub_enabled) {
        calc.add_dispatch_terminate();
    }
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();
    HostMemDeviceCommand cmd(total_cmd_bytes);
    cmd.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
    cmd.add_dispatch_terminate();
    if (dispatch_sub_enabled) {
        cmd.add_dispatch_terminate(DispatcherSelect::DISPATCH_SUBORDINATE);
    }

    return cmd;
}

HostMemDeviceCommand build_prefetch_terminate() {
    DeviceCommandCalculator calc;
    calc.add_prefetch_terminate();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();
    HostMemDeviceCommand cmd(total_cmd_bytes);

    cmd.add_prefetch_terminate();

    return cmd;
}

HostMemDeviceCommand build_dispatch_prefetch_stall() {
    DeviceCommandCalculator calc;
    calc.add_dispatch_wait_with_prefetch_stall();
    const uint32_t command_size_bytes = calc.write_offset_bytes();
    HostMemDeviceCommand cmd(command_size_bytes);

    cmd.add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER | CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY, 0, 0, 0);

    return cmd;
}

HostMemDeviceCommand build_dispatch_write_offset(tt::stl::Span<const uint32_t> write_offsets) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_set_write_offsets(write_offsets.size());
    const uint32_t command_size_bytes = calc.write_offset_bytes();

    HostMemDeviceCommand cmd(command_size_bytes);

    cmd.add_dispatch_set_write_offsets(write_offsets);

    return cmd;
}

template <bool inline_data>
HostMemDeviceCommand build_prefetch_relay_linear_host(uint32_t noc_xy, uint32_t addr, uint32_t data_size_bytes) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear_host();
    calc.add_prefetch_relay_linear();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    HostMemDeviceCommand cmd(total_cmd_bytes);
    cmd.add_dispatch_write_host<inline_data>(
        false,            // flush_prefetch
        data_size_bytes,  // data_sizeB
        false,            // is_event
        0,                // pad1
        nullptr           // data
    );

    // Relay data from L1 to dispatcher using NOC
    cmd.add_prefetch_relay_linear(noc_xy, data_size_bytes, addr);

    return cmd;
}

template <bool inline_data>
HostMemDeviceCommand build_relay_inline_host(const std::vector<uint32_t>& payload, uint32_t data_size_bytes) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear_host_event(data_size_bytes);
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    HostMemDeviceCommand cmd(total_cmd_bytes);
    cmd.add_dispatch_write_host<inline_data>(
        true,             // flush_prefetch
        data_size_bytes,  // data_sizeB
        false,            // is_event
        0,                // pad1
        payload.data()    // data
    );

    return cmd;
}

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_relay_linear_read(
    uint32_t noc_xy, uint32_t dst_addr, uint32_t src_addr, uint32_t transfer_size) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear<flush_prefetch, inline_data>(transfer_size);
    calc.add_prefetch_relay_linear();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    HostMemDeviceCommand cmd(total_cmd_bytes);
    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        0,              // num_mcast_dests
        noc_xy,         // NOC coordinates
        dst_addr,       // destination address
        transfer_size,  // data size
        nullptr         // payload data
    );

    // Relay data from L1 to dispatcher using NOC
    cmd.add_prefetch_relay_linear(noc_xy, transfer_size, src_addr);

    return cmd;
}

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_relay_paged(
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t start_page,
    uint32_t base_addr,
    uint32_t page_size_bytes,
    uint32_t pages_in_chunk,
    uint32_t length_adjust = 0) {
    DeviceCommandCalculator calc;
    uint32_t transfer_size = page_size_bytes * pages_in_chunk;
    calc.add_dispatch_write_linear<flush_prefetch, inline_data>(transfer_size);
    calc.add_prefetch_relay_paged();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);

    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        0,              // num_mcast_dests
        noc_xy,         // NOC coordinates
        addr,           // destination address
        transfer_size,  // data size
        nullptr         // payload data
    );

    cmd.add_prefetch_relay_paged(
        /*is_dram=*/true, start_page, base_addr, page_size_bytes, pages_in_chunk, length_adjust);

    return cmd;
}

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_relay_paged_packed(
    const std::vector<CQPrefetchRelayPagedPackedSubCmd>& sub_cmds,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t total_length) {
    const uint32_t n_sub_cmds = sub_cmds.size();
    // Calculate the command size using DeviceCommandCalculator
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear<flush_prefetch, inline_data>(total_length);
    calc.add_prefetch_relay_paged_packed(n_sub_cmds);
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);

    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        0,             // num_mcast_dests
        noc_xy,        // NOC coordinates
        addr,          // destination address
        total_length,  // data size
        nullptr        // payload data
    );

    // Add the prefetch relay paged packed
    cmd.add_prefetch_relay_paged_packed(total_length, sub_cmds, n_sub_cmds);

    return cmd;
}

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_ringbuffer_relay(
    const std::vector<CQPrefetchRelayRingbufferSubCmd>& sub_cmds,
    const std::vector<uint32_t>& lengths,
    Common::DeviceData& device_data,
    PrefetcherRingbufferReadTestFixture& fixture,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t total_length,
    uint32_t ringbuffer_read_page_size_log2);
}  // namespace CommandBuilder

class BasePrefetcherTestFixture : public Common::BaseTestFixture,
                                  public ::testing::WithParamInterface<PagedReadParams> {
protected:
    // Common constants
    static constexpr uint32_t HOST_DATA_DIRTY_PATTERN = 0xbaadf00d;
    static constexpr uint32_t PCIE_TRANSFER_SIZE_DEFAULT = 4096;
    static constexpr uint32_t MIN_READ_SIZE = 128;  // Minimum meaningful transfer size, aligns with DRAM
    static constexpr uint32_t DEFAULT_PREFETCH_Q_ENTRIES = 1024;
    static constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_BASE_ADDR = 0x1f400000;  // Magic, half of dram
    static constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE = 10;

    // Default values for inline data and flush prefetch for prefetcher tests
    static constexpr bool inline_data_ = false;
    static constexpr bool flush_prefetch_ = false;
    static constexpr bool hugepage_write_ = true;

    uint32_t dram_base_{};
    uint32_t num_banks_{};
    uint32_t l1_alignment_{};
    uint32_t packed_write_max_unicast_sub_cmds_{};

    // Exec Buf Configuration
    bool use_exec_buf_{};

    void SetUp() override {
        BaseTestFixture::SetUp();
        dram_base_ = device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);
        num_banks_ = device_->allocator_impl()->get_num_banks(BufferType::DRAM);
        l1_alignment_ = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
        packed_write_max_unicast_sub_cmds_ =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;

        // Init Test params
        const auto params = GetParam();
        page_size_ = params.page_size;
        num_pages_ = params.num_pages;
        num_iterations_ = params.num_iterations;
        dram_data_size_words_ = params.dram_data_size_words;
        use_exec_buf_ = params.use_exec_buf;
    }

    void execute_generated_commands(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        Common::DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool wait_for_completion = true,
        bool wait_for_host_writes = false) override {
        // exe buff execution in BasePrefetcherTestFixture
        if (use_exec_buf_) {
            execute_generated_commands_exec_buff(
                commands_per_iteration,
                device_data,
                num_cores_to_log,
                num_iterations,
                wait_for_completion,
                wait_for_host_writes,
                *this);
            return;
        }
        // Non exec buff execution in BaseTestFixture
        BaseTestFixture::execute_generated_commands(
            commands_per_iteration, device_data, num_cores_to_log, num_iterations, wait_for_completion);
    }

    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_num_pages() const { return num_pages_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }

private:
    uint32_t page_size_{};
    uint32_t num_pages_{};
    uint32_t num_iterations_{};
    uint32_t dram_data_size_words_{};

    // Helper function to execute generated commands via exec buff on device
    // Orchestrates the command buffer reservation, writing, and submission.
    // Executing commands via using exec buf in DRAM involves below steps:
    // 1. Concat all commands to be executed via exec buf in a buffer.
    // 2. Append a barrier, and then add exec buf end command to switch back to issue queue
    // 3. Write exec buf data into dram
    // 4. execute exec buf start command from issue queue
    void execute_generated_commands_exec_buff(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        Common::DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool wait_for_completion,
        bool wait_for_host_writes,
        BasePrefetcherTestFixture& fixture) {
        // 1. Add all commands to be uploaded into exec buff into a single vector
        std::vector<uint8_t> exec_buf_data;
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (const auto& cmd : commands_per_iteration) {
                const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(cmd.data());
                const uint8_t* end_ptr = reinterpret_cast<uint8_t*>(cmd.data()) + (cmd.size_bytes());
                exec_buf_data.insert(exec_buf_data.end(), src_ptr, end_ptr);
            }
        }

        // Append a barrier wait command after all commands across all iterations
        // Helpful to ensure all commands are completed flush before exec_buf_end
        DeviceCommandCalculator wait_calc;
        wait_calc.add_dispatch_wait();
        HostMemDeviceCommand wait_cmd(wait_calc.write_offset_bytes());
        wait_cmd.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
        const uint8_t* wptr = reinterpret_cast<const uint8_t*>(wait_cmd.data());
        const uint8_t* wend = wptr + (wait_cmd.size_bytes());
        exec_buf_data.insert(exec_buf_data.end(), wptr, wend);

        // 2. Append exec_buf_end command (terminate the trace execution and switch back to issue queue)
        DeviceCommandCalculator exec_buf_end_calc;
        exec_buf_end_calc.add_prefetch_exec_buf_end();
        DeviceCommand exec_terminate(exec_buf_end_calc.write_offset_bytes());
        exec_terminate.add_prefetch_exec_buf_end();
        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(exec_terminate.data());
        const uint8_t* end_ptr =
            reinterpret_cast<const uint8_t*>(exec_terminate.data()) + (exec_terminate.size_bytes());
        exec_buf_data.insert(exec_buf_data.end(), src_ptr, end_ptr);

        // 3. Write exec buff data into DRAM
        const uint32_t page_size = 1 << fixture.DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE;
        const uint32_t exec_buf_base_addr = fixture.DRAM_EXEC_BUF_DEFAULT_BASE_ADDR;

        // Pad data to full page alignment
        size_t size_bytes = exec_buf_data.size();
        log_info(tt::LogTest, "Total exec buff bytes: {}", size_bytes);
        size_t padded_size_bytes = tt::align(size_bytes, page_size);
        exec_buf_data.resize(padded_size_bytes);
        log_info(tt::LogTest, "Padded total exec buff bytes: {}", padded_size_bytes);

        uint32_t num_pages = padded_size_bytes / page_size;
        uint32_t data_idx = 0;

        // Create pages of exec buff data and write to DRAM
        for (uint32_t page_idx = 0; page_idx < num_pages; page_idx++) {
            uint32_t bank_id = page_idx % fixture.num_banks_;
            uint32_t bank_offset = (page_idx / fixture.num_banks_) * page_size;
            uint32_t addr = exec_buf_base_addr + bank_offset;

            std::vector<uint8_t> page_data(
                exec_buf_data.begin() + data_idx, exec_buf_data.begin() + data_idx + (page_size));

            detail::WriteToDeviceDRAMChannel(device_, bank_id, addr, page_data);

            data_idx += (page_size);
        }

        // Ensure DRAM writes are visible to device
        MetalContext::instance().get_cluster().dram_barrier(device_->id());

        // 4. Reserve and Write exec_buff command
        DeviceCommandCalculator exec_buf_calc;
        exec_buf_calc.add_prefetch_exec_buf();
        uint32_t cmd_size = exec_buf_calc.write_offset_bytes();
        void* cmd_buffer_base = mgr_->issue_queue_reserve(cmd_size, fdcq_->id());
        // Use DeviceCommand helper (HugepageDeviceCommand) to write to the issue queue memory
        HugepageDeviceCommand exec_cmd(cmd_buffer_base, cmd_size);
        exec_cmd.add_prefetch_exec_buf(exec_buf_base_addr, fixture.DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE, num_pages);

        // Verifies destination memory bounds
        device_data.overflow_check(device_);

        // Submit and execute commands
        mgr_->issue_queue_push_back(exec_cmd.write_offset_bytes(), fdcq_->id());

        // Write the commands to the device-side fetch queue with STALL FLAG
        bool stall_prefetcher = true;  // true with exec buff execution
        const auto start = std::chrono::steady_clock::now();
        mgr_->fetch_queue_reserve_back(fdcq_->id());
        mgr_->fetch_queue_write(cmd_size, fdcq_->id(), stall_prefetcher);

        // Wait for completion of the issued commands
        if (wait_for_completion) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        } else if (wait_for_host_writes) {
            uint32_t total_expected_cq_payload = device_data.size() * sizeof(uint32_t);
            // For host writes, wait until expected data is written into completion queue by dispatcher
            wait_for_completion_queue_bytes(total_expected_cq_payload, wait_completion_timeout);
        }
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        log_info(tt::LogTest, "Ran in {:.3f} ms (for {} iterations)", elapsed.count() * 1000.0, num_iterations);

        // Validate results
        const bool pass = device_data.validate(device_);
        EXPECT_TRUE(pass) << "Dispatcher test failed validation";

        // Report performance
        if (pass) {
            report_performance(device_data, num_cores_to_log, elapsed, num_iterations);
        }
    }

public:
    std::vector<HostMemDeviceCommand> generate_paged_read_commands(
        const CoreRange& worker_range, uint32_t page_size_bytes, uint32_t num_pages, Common::DeviceData& device_data) {
        std::vector<HostMemDeviceCommand> commands_per_iteration;
        uint32_t absolute_start_page = 0;
        const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
        uint32_t remaining_bytes = DEVICE_DATA_SIZE;
        const auto first_worker = worker_range.start_coord;

        // Compute NOC encoding once
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

        while (remaining_bytes > 0) {
            // Calculate how many pages fit in this chunk
            uint32_t max_bytes_in_chunk = num_pages * page_size_bytes;
            uint32_t bytes_in_chunk = std::min(remaining_bytes, max_bytes_in_chunk);
            uint32_t pages_in_chunk = bytes_in_chunk / page_size_bytes;

            uint32_t start_page = absolute_start_page % num_banks_;
            uint32_t base_addr = (absolute_start_page / num_banks_) * page_size_bytes + dram_base_;

            // Capture address before updating device_data
            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            // Relay paged from prefetcher -> linear write from dispatcher to L1
            HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged<flush_prefetch_, inline_data_>(
                noc_xy, l1_addr, start_page, base_addr, page_size_bytes, pages_in_chunk);

            // Shadow model updated per page
            for (uint32_t page = 0; page < pages_in_chunk; ++page) {
                const uint32_t page_id = absolute_start_page + page;
                const uint32_t bank_id = page_id % num_banks_;
                uint32_t bank_offset = page_size_words * (page_id / num_banks_);

                // Get the logical core for this bank
                const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
                const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

                // Update Common::DeviceData for paged read
                DeviceDataUpdater::update_paged_dram_read(
                    worker_range, device_data, bank_core, bank_id, bank_offset, page_size_words);
            }

            commands_per_iteration.push_back(std::move(cmd));

            absolute_start_page += pages_in_chunk;
            remaining_bytes -= bytes_in_chunk;
        }

        return commands_per_iteration;
    }

    // Helper function to generate paged writes wtih random payload and
    // relay paged reads commands to dispatcher to write to L1
    // for the End-To-End Paged/Interleaved Write+Read test
    std::vector<HostMemDeviceCommand> generate_paged_end_to_end_commands(
        const CoreRange& worker_range, uint32_t page_size_bytes, uint32_t num_pages, Common::DeviceData& device_data) {
        const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
        const uint32_t page_size_alignment_bytes = device_->allocator_impl()->get_alignment(BufferType::DRAM);
        const auto first_worker = worker_range.start_coord;

        // Compute NOC encoding once
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = DEVICE_DATA_SIZE;
        uint32_t absolute_start_page = 0;

        while (remaining_bytes > 0) {
            // Calculate how many pages fit in this chunk
            uint32_t max_bytes_in_chunk = num_pages * page_size_bytes;
            uint32_t bytes_in_chunk = std::min(remaining_bytes, max_bytes_in_chunk);
            uint32_t pages_in_chunk = bytes_in_chunk / page_size_bytes;

            // Generate payload & update expectations for this chunk
            std::vector<uint32_t> chunk_payload;
            chunk_payload.reserve(pages_in_chunk * page_size_words);

            for (uint32_t page = 0; page < pages_in_chunk; ++page) {
                const uint32_t page_id = absolute_start_page + page;
                const uint32_t bank_id = page_id % num_banks_;

                // Get the logical core for this bank
                const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
                const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

                // Generate payload with page id
                std::vector<uint32_t> page_payload =
                    payload_generator_->generate_payload_with_page_id(page_size_words, page_id);

                // Update Common::DeviceData for paged write
                Common::DeviceDataUpdater::update_paged_write(
                    page_payload, device_data, bank_core, bank_id, page_size_alignment_bytes);

                // Append page payload to chunk payload
                chunk_payload.insert(chunk_payload.end(), page_payload.begin(), page_payload.end());
            }

            // Calculate base address for the command
            const uint32_t bank_offset =
                tt::align(page_size_bytes, page_size_alignment_bytes) * (absolute_start_page / num_banks_);
            const uint32_t base_addr = device_data.get_base_result_addr(tt::CoreType::DRAM) + bank_offset;
            // Calculate start page for the command
            const uint16_t start_page_cmd = absolute_start_page % num_banks_;

            //  Step 1: Paged Write of host data to DRAM banks by dispatcher cmd
            HostMemDeviceCommand cmd_dispatch_dram = Common::CommandBuilder::build_paged_write_command<hugepage_write_>(
                chunk_payload, base_addr, page_size_bytes, pages_in_chunk, start_page_cmd, true);
            commands_per_iteration.push_back(std::move(cmd_dispatch_dram));

            // Followed by stall to avoid RAW hazard
            HostMemDeviceCommand cmd_stall = CommandBuilder::build_dispatch_prefetch_stall();
            commands_per_iteration.push_back(std::move(cmd_stall));

            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            // Step 2: Paged Read of DRAM banks by prefetcher, relay data to dispatcher for linear write to L1
            HostMemDeviceCommand cmd_prefetch =
                CommandBuilder::build_prefetch_relay_paged<flush_prefetch_, inline_data_>(
                    noc_xy, l1_addr, start_page_cmd, base_addr, page_size_bytes, pages_in_chunk);
            commands_per_iteration.push_back(std::move(cmd_prefetch));

            for (uint32_t page = 0; page < pages_in_chunk; ++page) {
                const uint32_t page_id = absolute_start_page + page;
                const uint32_t bank_id = page_id % num_banks_;
                // Add dram_data_size_words since we're reading after the pre-populated DRAM data
                uint32_t bank_offset = dram_data_size_words_ + page_size_words * (page_id / num_banks_);

                // Get the logical core for this bank
                const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
                const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

                // Update Common::DeviceData for paged read
                DeviceDataUpdater::update_paged_dram_read(
                    worker_range, device_data, bank_core, bank_id, bank_offset, page_size_words);
            }

            // Update loop state
            remaining_bytes -= bytes_in_chunk;
            absolute_start_page += pages_in_chunk;
        }

        return commands_per_iteration;
    }
};

class PrefetcherHostTextFixture : virtual public BasePrefetcherTestFixture {
protected:
    void pad_host_data(Common::DeviceData& device_data) {
        Common::one_core_data_t& host_data = device_data.get_data()[device_data.get_host_core()][0];

        int pad =
            dispatch_buffer_page_size_ - ((host_data.data.size() * sizeof(uint32_t)) % dispatch_buffer_page_size_);
        pad = pad % dispatch_buffer_page_size_;

        for (int i = 0; i < pad / sizeof(uint32_t); i++) {
            device_data.push_one(device_data.get_host_core(), 0, HOST_DATA_DIRTY_PATTERN);
        }
    }

    void dirty_host_completion_buffer(void* completion_queue_buffer, uint32_t size_bytes) {
        uint32_t* buffer = static_cast<uint32_t*>(completion_queue_buffer);
        uint32_t size_words = size_bytes / sizeof(uint32_t);

        for (uint32_t i = 0; i < size_words; i++) {
            buffer[i] = HOST_DATA_DIRTY_PATTERN;
        }

        tt_driver_atomics::sfence();
    }

    std::vector<HostMemDeviceCommand> generate_host_write_commands(
        const std::vector<uint32_t>& data,
        const CoreCoord first_worker,
        const uint32_t l1_base,
        Common::DeviceData& device_data) {
        const uint32_t max_data_size_words = data.size();
        // Compute NOC encoding once
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

        std::vector<HostMemDeviceCommand> commands_per_iteration;
        uint32_t max_limit = max_data_size_words / 100;

        // Vary host write sizes up to 1% - 100% of max_data_size_words
        for (uint32_t count = 1; count < 100; count++) {
            uint32_t data_size_words = payload_generator_->get_rand<uint32_t>(0, max_limit - 1) * count + 1;
            uint32_t data_size_bytes = data_size_words * sizeof(uint32_t);

            HostMemDeviceCommand cmd =
                CommandBuilder::build_prefetch_relay_linear_host<inline_data_>(noc_xy, l1_base, data_size_bytes);

            commands_per_iteration.push_back(std::move(cmd));

            DeviceDataUpdater::update_host_data(device_data, data, data_size_bytes);

            // The completion queue is page-aligned (4KB pages)
            // Each write reserves full pages but only writes actual data,
            // leaving the remainder untouched (still HOST_DATA_DIRTY_PATTERN)
            // This ensures padding areas retain the sentinel value for validation
            pad_host_data(device_data);
        }

        return commands_per_iteration;
    }
};

class PrefetcherPackedReadTestFixture : virtual public BasePrefetcherTestFixture {
    const bool relay_max_packed_paged_submcds = true;  // TODO: randomize?
protected:
    std::vector<CQPrefetchRelayPagedPackedSubCmd> build_sub_cmds(
        const std::vector<uint32_t>& lengths,
        Common::DeviceData& device_data,
        uint32_t log_packed_read_page_size,
        uint32_t n_sub_cmds) {
        uint32_t count = 0;
        uint32_t page_size_bytes = 1 << log_packed_read_page_size;
        std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds;
        sub_cmds.reserve(n_sub_cmds);
        for (auto length : lengths) {
            CQPrefetchRelayPagedPackedSubCmd sub_cmd{};
            sub_cmd.start_page = 0;
            sub_cmd.log_page_size = log_packed_read_page_size;
            sub_cmd.base_addr = dram_base_ + count * page_size_bytes;
            sub_cmd.length = length;
            sub_cmds.push_back(sub_cmd);
            count++;

            // Model the packed paged read in this function by updating worker data with interleaved/paged DRAM data,
            // for validation later.
            uint32_t length_words = length / sizeof(uint32_t);
            uint32_t base_addr_words = (sub_cmd.base_addr - dram_base_) / sizeof(uint32_t);
            uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);

            // Get data from DRAM map, add to all workers, but only set valid for cores included in workers range.
            uint32_t page_idx = sub_cmd.start_page;
            for (uint32_t i = 0; i < length_words; i += page_size_words) {
                uint32_t dram_bank_id = page_idx % num_banks_;
                auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(dram_bank_id);
                CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);
                uint32_t bank_offset = base_addr_words + (page_size_words * (page_idx / num_banks_));

                uint32_t words = (page_size_words > length_words - i) ? length_words - i : page_size_words;

                DeviceDataUpdater::update_read(
                    default_worker_start, device_data, bank_core, dram_bank_id, bank_offset, words);

                page_idx++;
            }
        }

        return sub_cmds;
    }

    std::vector<HostMemDeviceCommand> generate_packed_read_commands(
        const CoreCoord first_worker, const uint32_t dram_alignment, Common::DeviceData& device_data) {
        // Compute NOC encoding once
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = DEVICE_DATA_SIZE;

        constexpr uint32_t max_size128b = (DEFAULT_SCRATCH_DB_SIZE / 2) >> 7;
        while (remaining_bytes > 0) {
            uint32_t packed_read_page_size =
                payload_generator_->get_rand<uint32_t>(0, 2) + 9;  // log2 values. i.e., 512, 1024, 2048
            uint32_t n_sub_cmds = relay_max_packed_paged_submcds ? CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS
                                                                 : payload_generator_->get_rand<uint32_t>(0, 6) + 1;
            uint32_t max_read_size = std::min((1 << packed_read_page_size) * num_banks_, remaining_bytes);

            std::vector<uint32_t> lengths;
            lengths.reserve(n_sub_cmds);
            uint32_t total_length = 0;
            for (uint32_t i = 0; i < n_sub_cmds; i++) {
                // limit the length to min and max read size
                uint32_t raw = (payload_generator_->get_rand<uint32_t>(0, max_size128b - 1)) << 7;
                uint32_t clamped = std::min(max_read_size, std::max(MIN_READ_SIZE, raw));
                uint32_t length = tt::align(clamped, dram_alignment);
                total_length += length;
                lengths.push_back(length);
            }

            // If we're about to exceed DEVICE_DATA_SIZE, then exit
            if (device_data.size() * sizeof(uint32_t) + total_length > DEVICE_DATA_SIZE) {
                break;
            }

            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            // Create n_sub_cmds
            std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds =
                build_sub_cmds(lengths, device_data, packed_read_page_size, n_sub_cmds);

            HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged_packed<flush_prefetch_, inline_data_>(
                sub_cmds, noc_xy, l1_addr, total_length);

            commands_per_iteration.push_back(std::move(cmd));
            remaining_bytes -= total_length;
        }

        return commands_per_iteration;
    }
};

class PrefetcherRingbufferReadTestFixture : virtual public BasePrefetcherTestFixture {
    static constexpr uint32_t MAX_PAGE_OFFSET = 5;

public:
    // Set ring buffer offset to arbitrary number
    static constexpr uint32_t RING_BUFFER_TEST_OFFSET = 1234;

    void populate_ringbuffer_from_dram(
        HostMemDeviceCommand& cmd,
        const std::vector<uint32_t>& lengths,
        Common::DeviceData& device_data,
        uint32_t ringbuffer_read_page_size_log2,
        uint32_t n_sub_cmds) {
        bool reset = false;
        uint32_t page_size_bytes = 1 << ringbuffer_read_page_size_log2;
        uint32_t count = 0;

        for (auto length : lengths) {
            uint8_t wraparound_flag = 0;
            if (!reset) {
                wraparound_flag = CQ_PREFETCH_PAGED_TO_RING_BUFFER_FLAG_RESET_TO_START;
                reset = true;
            }

            CQPrefetchPagedToRingbufferCmd ringbuffer_cmd{};
            ringbuffer_cmd.flags = wraparound_flag;
            ringbuffer_cmd.log2_page_size = ringbuffer_read_page_size_log2;
            // RECOMMENDED: Use uint16_t instead of uint8_t for std::uniform_int_distribution, then cast it to uint8_t
            ringbuffer_cmd.start_page =
                static_cast<uint8_t>(payload_generator_->get_rand<uint16_t>(0, MAX_PAGE_OFFSET - 1));
            ringbuffer_cmd.wp_offset_update = length;
            ringbuffer_cmd.base_addr = dram_base_ + count * page_size_bytes;
            ringbuffer_cmd.length = length;

            cmd.add_prefetch_paged_to_ringbuffer(ringbuffer_cmd);
            count++;

            // Model the paged to ringbuffer read in this function by updating worker data with interleaved/paged DRAM
            // data, for validation later.
            uint32_t length_words = length / sizeof(uint32_t);
            uint32_t base_addr_words = (ringbuffer_cmd.base_addr - dram_base_) / sizeof(uint32_t);
            uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);

            // Get data from DRAM map, add to worker.
            uint32_t page_idx = ringbuffer_cmd.start_page;
            for (uint32_t i = 0; i < length_words; i += page_size_words) {
                uint32_t dram_bank_id = page_idx % num_banks_;
                auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(dram_bank_id);
                CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);
                uint32_t bank_offset = base_addr_words + (page_size_words * (page_idx / num_banks_));

                uint32_t words = (page_size_words > length_words - i) ? length_words - i : page_size_words;
                DeviceDataUpdater::update_read(
                    default_worker_start, device_data, bank_core, dram_bank_id, bank_offset, words);

                page_idx++;
            }
        }
    }

    std::vector<CQPrefetchRelayRingbufferSubCmd> build_sub_cmds(
        const std::vector<uint32_t>& lengths, uint32_t n_sub_cmds) {
        std::vector<CQPrefetchRelayRingbufferSubCmd> sub_cmds;
        sub_cmds.reserve(n_sub_cmds);

        uint32_t current_offset = 0;
        for (auto length : lengths) {
            CQPrefetchRelayRingbufferSubCmd sub_cmd{};
            sub_cmd.start = current_offset -
                            RING_BUFFER_TEST_OFFSET;  // Since set_ringbuffer_offset is set to RING_BUFFER_TEST_OFFSET
            sub_cmd.length = length;
            current_offset += length;
            sub_cmds.push_back(sub_cmd);
        }

        return sub_cmds;
    }

    std::vector<HostMemDeviceCommand> generate_ringbuffer_relay_commands(
        const CoreCoord first_worker, const uint32_t dram_alignment, Common::DeviceData& device_data) {
        // Compute NOC encoding once
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = DEVICE_DATA_SIZE;

        while (remaining_bytes > 0) {
            uint32_t ringbuffer_read_page_size_log2 =
                payload_generator_->get_rand<uint32_t>(0, 2) + 9;  // log2 values. i.e., 512, 1024, 2048
            uint32_t n_sub_cmds = payload_generator_->get_rand<uint32_t>(0, 6) + 1;
            uint32_t max_read_size = std::min((1 << ringbuffer_read_page_size_log2) * num_banks_, remaining_bytes);

            std::vector<uint32_t> lengths;
            lengths.reserve(n_sub_cmds);

            uint32_t total_length = 0;
            for (uint32_t i = 0; i < n_sub_cmds; i++) {
                // limit the length to min and max read size
                uint32_t raw = payload_generator_->get_rand<uint32_t>(0, DEFAULT_SCRATCH_DB_SIZE - 1);
                uint32_t clamped = std::min(max_read_size, std::max(MIN_READ_SIZE, raw));
                uint32_t length = tt::align(clamped, dram_alignment);
                total_length += length;
                lengths.push_back(length);
            }

            // If we're about to exceed DEVICE_DATA_SIZE, then exit
            if (device_data.size() * sizeof(uint32_t) + total_length > DEVICE_DATA_SIZE) {
                break;
            }

            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            // We build the sub commands to relay them from the ringbuffer to the dispatcher
            std::vector<CQPrefetchRelayRingbufferSubCmd> sub_cmds = build_sub_cmds(lengths, n_sub_cmds);

            HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_ringbuffer_relay<flush_prefetch_, inline_data_>(
                sub_cmds, lengths, device_data, *this, noc_xy, l1_addr, total_length, ringbuffer_read_page_size_log2);

            commands_per_iteration.push_back(std::move(cmd));
            remaining_bytes -= total_length;
        }

        return commands_per_iteration;
    }
};

namespace CommandBuilder {

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_ringbuffer_relay(
    const std::vector<CQPrefetchRelayRingbufferSubCmd>& sub_cmds,
    const std::vector<uint32_t>& lengths,
    Common::DeviceData& device_data,
    PrefetcherRingbufferReadTestFixture& fixture,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t total_length,
    uint32_t ringbuffer_read_page_size_log2) {
    const uint32_t n_sub_cmds = sub_cmds.size();
    // Calculate the command size using DeviceCommandCalculator
    DeviceCommandCalculator calc;
    for (uint32_t i = 0; i < n_sub_cmds; i++) {
        calc.add_prefetch_paged_to_ringbuffer();
    }
    calc.add_prefetch_set_ringbuffer_offset();
    calc.add_dispatch_write_linear<false, false>(total_length);
    calc.add_prefetch_relay_ringbuffer(n_sub_cmds);
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);

    // First, we populate the ringbuffer
    fixture.populate_ringbuffer_from_dram(cmd, lengths, device_data, ringbuffer_read_page_size_log2, n_sub_cmds);

    // Second, we set the read offset in the ring buffer
    cmd.add_prefetch_set_ringbuffer_offset(fixture.RING_BUFFER_TEST_OFFSET);

    // Third, we write the dispatch command to the dispatcher
    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        0,             // num_mcast_dests
        noc_xy,        // NOC coordinates
        addr,          // destination address
        total_length,  // data size
        nullptr        // payload data
    );

    // Lastly, relay the subcommands from ringbuffer in L1 to dispatcher
    cmd.add_prefetch_relay_ringbuffer(n_sub_cmds, sub_cmds);

    return cmd;
}
}  // namespace CommandBuilder

// This fixture is useful for commands like CQ_PREFETCH_CMD_RELAY_LINEAR_H,
// where we need multi-chip device environment and needs PREFETCH_H / PREFETCH_D split
class PrefetchRelayLinearHTestFixture : public BasePrefetcherTestFixture {
protected:
    static constexpr uint32_t MIN_READ_SIZE = 32;

    tt_metal::distributed::MeshDevice::IDevice* mmio_device_ = nullptr;
    tt_metal::distributed::MeshDevice::IDevice* remote_device_ = nullptr;

    void SetUp() override {
        BasePrefetcherTestFixture::SetUp();
        if (mesh_device_->num_devices() < 2) {
            GTEST_SKIP() << "Skipping RelayLinearHTest: need MMIO+remote pair in mesh";
        }
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        // Identify the MMIO device in the mesh
        for (auto* dev : mesh_device_->get_devices()) {
            if (cluster.get_associated_mmio_device(dev->id()) == dev->id()) {
                mmio_device_ = dev;
                break;
            }
        }

        if (!mmio_device_) {
            GTEST_SKIP() << "Skipping RelayLinearHTest: need MMIO+remote pair in mesh";
        }

        // Next, identify a remote device associated with the MMIO device
        for (auto* dev : mesh_device_->get_devices()) {
            if (dev != mmio_device_ && (cluster.get_associated_mmio_device(dev->id()) == mmio_device_->id())) {
                remote_device_ = dev;
                break;
            }
        }

        // Skip if no suitable pair was found since this test
        // targets PREFETCH_H (MMIO) -> PREFETCH_D (Remote) -> DISPATCHER (Remote)
        if (!remote_device_) {
            GTEST_SKIP() << "Skipping RelayLinearHTest: need MMIO+remote pair in mesh";
        }

        // Override the base class device_ to use remote device
        device_ = remote_device_;
        // We need access to remote chips issue queue to execute commands like
        // CQ_PREFETCH_CMD_RELAY_LINEAR_H in PREFETCH_H (channel 1 region)
        mgr_ = &remote_device_->sysmem_manager();

        // Execution through exec_buf is always disabled for standalone commands like
        // CQ_PREFETCH_CMD_RELAY_LINEAR_H
        use_exec_buf_ = false;
    }

public:
    std::vector<HostMemDeviceCommand> generate_prefetch_relay_h_commands(
        const CoreCoord first_worker,
        const uint32_t mmio_dram_base,
        const uint32_t dram_alignment,
        Common::DeviceData& device_data) {
        const uint32_t max_read_size =
            std::min(
                DEFAULT_SCRATCH_DB_SIZE,
                DEFAULT_PREFETCH_Q_ENTRIES * (uint32_t)sizeof(DispatchSettings::prefetch_q_entry_type)) -
            64;

        // Compute NOC encoding: Destination (Worker on Remote Device) -> Use device_ (Chip 1)
        const CoreCoord first_virt_worker =
            remote_device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
        const uint32_t noc_xy = remote_device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = DEVICE_DATA_SIZE;

        while (remaining_bytes > 0) {
            uint32_t length = std::min(
                tt::align(
                    std::max(MIN_READ_SIZE, payload_generator_->get_rand<uint32_t>(0, max_read_size - 1)),
                    dram_alignment),
                remaining_bytes);

            // Capture address before updating device_data
            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            DeviceCommandCalculator calc;
            calc.add_dispatch_write_linear<false, false>(length);
            const uint32_t dispatch_cmd_size = calc.write_offset_bytes();

            // Create the HostMemDeviceCommand with pre-calculated size
            HostMemDeviceCommand cmd1(dispatch_cmd_size);

            cmd1.add_dispatch_write_linear<false, false>(
                0,        // num_mcast_dests
                noc_xy,   // NOC coordinates
                l1_addr,  // destination address
                length,   // data size
                nullptr   // payload data
            );

            commands_per_iteration.push_back(std::move(cmd1));

            const uint32_t total_cmd_bytes = tt::align(sizeof(CQPrefetchCmdLarge), host_alignment_);

            // Create the the relay linear H command as a separate command as it must be
            // the only entry in fetchQ
            HostMemDeviceCommand cmd2(total_cmd_bytes);

            // Create the relay linear H command
            CQPrefetchCmdLarge cmd{};
            std::memset(&cmd, 0, sizeof(CQPrefetchCmdLarge));
            cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR_H;

            // Source (DRAM on MMIO Device) -> Use mmio_device_ (Chip 0)
            const uint32_t dram_bank_id = 0;
            auto dram_channel = mmio_device_->allocator_impl()->get_dram_channel_from_bank_id(dram_bank_id);
            CoreCoord dram_logical_core = mmio_device_->logical_core_from_dram_channel(dram_channel);
            CoreCoord dram_physical_core = MetalContext::instance()
                                               .get_cluster()
                                               .get_soc_desc(mmio_device_->id())
                                               .get_preferred_worker_core_for_dram_view(dram_channel, NOC::NOC_0);
            cmd.relay_linear_h.noc_xy_addr =
                mmio_device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, dram_physical_core);

            [[maybe_unused]] auto offset =
                mmio_device_->allocator_impl()->get_bank_offset(BufferType::DRAM, dram_bank_id);
            // Read from DRAM result data address where data is stored
            // Common::DeviceData uses the logical coordinates as keys
            cmd.relay_linear_h.addr = mmio_dram_base + offset;
            cmd.relay_linear_h.length = length;
            cmd.relay_linear_h.pad1 = 0;
            cmd.relay_linear_h.pad2 = 0;

            uint32_t length_words = length / sizeof(uint32_t);

            // Use reserve_space to properly update cmd_write_offsetB
            CQPrefetchCmdLarge* cmd_ptr = cmd2.reserve_space<CQPrefetchCmdLarge*>(sizeof(CQPrefetchCmdLarge));
            std::memcpy(cmd_ptr, &cmd, sizeof(CQPrefetchCmdLarge));

            // Align the write offset for any padding needed
            cmd2.align_write_offset();

            commands_per_iteration.push_back(std::move(cmd2));

            DeviceDataUpdater::update_read(
                default_worker_start, device_data, dram_logical_core, dram_bank_id, 0, length_words);

            remaining_bytes -= length;
        }

        return commands_per_iteration;
    }
};

class RandomTestFixture : public BasePrefetcherTestFixture {
    bool big_chunk_ = true;                                // TODO: make this parameterized
    static constexpr uint32_t MAX_PAGE_SIZE = 256 * 1024;  // TODO: make it equal to actual scratch_db_page_size

protected:
    // L1 read to L1 write in same core
    // Copies from worker core's start of data back to the end of data
    std::optional<HostMemDeviceCommand> gen_random_linear_read_cmd(
        Common::DeviceData& device_data, const CoreCoord& worker_core, uint32_t noc_xy, uint32_t& remaining_bytes) {
        const uint32_t bank_id = 0;

        const uint32_t avail_bytes = device_data.size_at(worker_core, bank_id) * sizeof(uint32_t);
        if (avail_bytes < l1_alignment_ || avail_bytes == 0) {
            return std::nullopt;  // nothing to read yet
        }
        // Random length (bytes), aligned to L1, capped by remaining
        uint32_t data_size_bytes = payload_generator_->get_rand<uint32_t>(l1_alignment_, avail_bytes);
        data_size_bytes = tt::align(data_size_bytes, l1_alignment_);
        data_size_bytes = std::min({data_size_bytes, avail_bytes, remaining_bytes});
        if (data_size_bytes == 0 || data_size_bytes > avail_bytes) {
            return std::nullopt;
        }
        // Offset within the available data, aligned to L1, ensure it fits
        const uint32_t max_offset = avail_bytes - data_size_bytes;
        uint32_t offset_bytes = tt::align(payload_generator_->get_rand<uint32_t>(0, max_offset), l1_alignment_);
        if (offset_bytes + data_size_bytes > avail_bytes) {
            return std::nullopt;
        }
        // Destination address and payload
        uint32_t dst_addr = device_data.get_result_data_addr(worker_core, 0);
        // Source address
        uint32_t src_addr = device_data.get_base_result_addr(device_data.get_core_type(worker_core)) + offset_bytes;

        // Shadow model: copy existing L1 data from src offset
        const uint32_t data_size_words = data_size_bytes / sizeof(uint32_t);
        const uint32_t offset_words = offset_bytes / sizeof(uint32_t);
        DeviceDataUpdater::update_read(worker_core, device_data, worker_core, bank_id, offset_words, data_size_words);
        device_data.pad(worker_core, bank_id, MetalContext::instance().hal().get_alignment(HalMemType::L1));

        // Build command (dispatcher linear write (dest) + relay linear read (src))
        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_linear_read<flush_prefetch_, inline_data_>(
            noc_xy, dst_addr, src_addr, data_size_bytes);

        remaining_bytes -= data_size_bytes;
        return cmd;
    }

    std::optional<HostMemDeviceCommand> gen_random_dram_paged_cmd(
        Common::DeviceData& device_data,
        const CoreCoord& worker_core,
        const CoreRange& worker_range,
        uint32_t noc_xy,
        uint32_t& remaining_bytes) {
        const uint32_t dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

        // Get max pages
        uint32_t max_page_size = std::min(big_chunk_ ? MAX_PAGE_SIZE : 4096, remaining_bytes);
        if (max_page_size < dram_alignment) {
            return std::nullopt;
        }
        // Pick a random page size, align up and then clamp down if we overshot remaining bytes
        uint32_t page_size = tt::align(payload_generator_->get_rand<uint32_t>(1, max_page_size), dram_alignment);
        if (page_size > remaining_bytes) {
            page_size -= dram_alignment;
        }

        // Pick random number of pages
        uint32_t max_pages = remaining_bytes / page_size;
        uint32_t max_data_limit = big_chunk_ ? DEVICE_DATA_SIZE : DEVICE_DATA_SIZE / 8;
        uint32_t cmd_limit_pages = std::max(1u, max_data_limit / page_size);
        uint32_t pages = payload_generator_->get_rand<uint32_t>(1, std::min(max_pages, cmd_limit_pages));

        // Randomize source DRAM location
        uint32_t total_bytes = pages * page_size;
        uint32_t random_offset =
            tt::align(payload_generator_->get_rand<uint32_t>(0, Common::DRAM_DATA_SIZE_BYTES / 2), dram_alignment);

        if (random_offset + total_bytes > Common::DRAM_DATA_SIZE_BYTES) {
            random_offset = 0;
        }

        uint32_t start_page = payload_generator_->get_rand<uint32_t>(0, num_banks_ - 1);

        // TODO: Randomize length adjust
        uint32_t length_adjust = 0;

        // Calculate absolute address for command
        uint32_t cmd_base_addr = dram_base_ + random_offset;
        uint32_t addr = device_data.get_result_data_addr(worker_core, 0);
        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged<flush_prefetch_, inline_data_>(
            noc_xy, addr, start_page, cmd_base_addr, page_size, pages, length_adjust);

        // Update Common::DeviceData for paged read
        uint32_t page_size_words = page_size / sizeof(uint32_t);
        uint32_t base_addr_words = random_offset / sizeof(uint32_t);
        uint32_t length_adjust_words = length_adjust / sizeof(uint32_t);
        uint32_t last_page = start_page + pages;

        for (uint32_t page_idx = start_page; page_idx < last_page; ++page_idx) {
            const uint32_t bank_id = page_idx % num_banks_;
            uint32_t bank_offset = base_addr_words + page_size_words * (page_idx / num_banks_);

            // Get the logical core for this bank
            const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
            const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

            uint32_t words_to_read = page_size_words;
            if (page_idx == last_page - 1) {
                words_to_read -= length_adjust_words;
            }

            DeviceDataUpdater::update_paged_dram_read(
                worker_range, device_data, bank_core, bank_id, bank_offset, words_to_read);
        }

        uint32_t actual_data_size = total_bytes - length_adjust;
        remaining_bytes -= actual_data_size;
        return cmd;
    }

    std::optional<HostMemDeviceCommand> gen_random_inline_cmd(
        Common::DeviceData& device_data, const CoreRange& worker_range, uint32_t noc_xy, uint32_t& remaining_bytes) {
        // Randomize the dispatcher command we choose to relay
        uint32_t random_dispatch_cmd = payload_generator_->get_rand<uint32_t>(0, 1);

        switch (random_dispatch_cmd) {
            case 0:
                // Unicast Write
                {
                    uint32_t cmd_size_bytes = host_alignment_;

                    // New implementation using get_random_size:
                    uint32_t max_prefetch_command_size = max_fetch_bytes_;
                    uint32_t max_size = big_chunk_ ? max_prefetch_command_size : max_prefetch_command_size / 16;

                    // Subtract overhead to get max payload size allowed
                    uint32_t max_payload_bytes = max_size - cmd_size_bytes;
                    uint32_t max_xfer_size_16b = max_payload_bytes >> 4;

                    // get_random_size handles:
                    uint32_t xfer_size_bytes = payload_generator_->get_random_size(
                        max_xfer_size_16b,   // max_allowed (in 16B units)
                        bytes_per_16B_unit,  // bytes_per_unit
                        remaining_bytes      // remaining_bytes
                    );

                    if (xfer_size_bytes < 4) {
                        return std::nullopt;
                    }

                    // Capture address before updating device_data
                    uint32_t addr = device_data.get_result_data_addr(worker_range.start_coord, 0);

                    // Generate payload
                    std::vector<uint32_t> payload = payload_generator_->generate_payload(xfer_size_bytes);

                    // Update Common::DeviceData for linear write
                    Common::DeviceDataUpdater::update_linear_write(payload, device_data, worker_range, false);

                    // Build the command: Dispatch Write Linear (Unicast) wrapped in Prefetch Relay Inline
                    HostMemDeviceCommand cmd = Common::CommandBuilder::build_linear_write_command<true, true>(
                        payload, worker_range, false, noc_xy, addr, xfer_size_bytes);

                    remaining_bytes -= xfer_size_bytes;
                    return cmd;
                }
                break;
            case 1:
                // Packed Unicast Write
                {
                    // Randomly pick worker cores once for all commands
                    std::vector<CoreCoord> worker_cores;

                    for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; ++y) {
                        for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; ++x) {
                            if (send_to_all_ || payload_generator_->get_rand_bool()) {
                                worker_cores.push_back({x, y});
                            }
                        }
                    }

                    if (worker_cores.empty()) {
                        worker_cores.push_back(default_worker_start);
                    }
                    const uint32_t num_sub_cmds = static_cast<uint32_t>(worker_cores.size());
                    const uint32_t sub_cmds_bytes =
                        tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment_);

                    // Build sub-commands
                    std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds =
                        Common::PackedWriteUtils::build_sub_cmds(device_, worker_cores, k_dispatch_downstream_noc);
                    // Relevel once before generating commands
                    device_data.relevel(tt::CoreType::WORKER);

                    // Size and Clamp
                    uint32_t xfer_size_bytes =
                        payload_generator_->get_random_size(dispatch_buffer_page_size_, 1, remaining_bytes);

                    bool no_stride = payload_generator_->get_rand_bool();

                    // Clamp for dispatch page size (no_stride mode)
                    if (no_stride) {
                        const uint32_t max_allowed =
                            dispatch_buffer_page_size_ - sizeof(CQDispatchCmd) - sub_cmds_bytes;
                        if (xfer_size_bytes > max_allowed) {
                            log_warning(tt::LogTest, "Clamping packed_write cmd w/ no_stride to fit dispatch page");
                            xfer_size_bytes = max_allowed;
                        }
                    }

                    xfer_size_bytes = Common::PackedWriteUtils::clamp_to_max_fetch(
                        max_fetch_bytes_,
                        xfer_size_bytes,
                        num_sub_cmds,
                        packed_write_max_unicast_sub_cmds_,
                        no_stride,
                        l1_alignment_);

                    const CoreCoord& fw = worker_cores[0];
                    // Capture address before updating device_data
                    uint32_t addr = device_data.get_result_data_addr(fw, 0);
                    // Generate Payload
                    std::vector<uint32_t> payload = payload_generator_->generate_payload_with_core(fw, xfer_size_bytes);
                    // Update expected device_data for all cores
                    Common::DeviceDataUpdater::update_packed_write(payload, device_data, worker_cores, l1_alignment_);

                    // Build Command
                    HostMemDeviceCommand cmd = Common::CommandBuilder::build_packed_write_command(
                        payload, sub_cmds, addr, l1_alignment_, packed_write_max_unicast_sub_cmds_, no_stride);

                    remaining_bytes -= xfer_size_bytes;
                    return cmd;
                }
                break;
            default: TT_THROW("Invalid random_dispatch_cmd {} in gen_random_inline_cmd", random_dispatch_cmd);
        }
        return std::nullopt;
    }
};

struct HelperInfo {
    uint32_t dram_base_{};
    uint32_t num_banks_{};
    uint32_t l1_alignment_{};
    uint32_t packed_write_max_unicast_sub_cmds_{};
    uint32_t dispatch_buffer_page_size_{};
};

// TODO: add CQ_PREFETCH_CMD_DEBUG
class SmokeTestHelper {
    tt_metal::distributed::MeshDevice::IDevice* device_;
    Common::DeviceData& device_data_;
    std::vector<HostMemDeviceCommand>& cmds_;
    std::unique_ptr<Common::DispatchPayloadGenerator> payload_generator_;

    HelperInfo info_{};

public:
    SmokeTestHelper(
        tt_metal::distributed::MeshDevice::IDevice* device,
        Common::DeviceData& device_data,
        std::vector<HostMemDeviceCommand>& cmds,
        HelperInfo info) :
        device_(device), device_data_(device_data), cmds_(cmds), info_(info) {
        // Initialize simple payload generator
        Common::DispatchPayloadGenerator::Config cfg;
        cfg.seed = 1234;  // Deterministic seed for smoke test
        payload_generator_ = std::make_unique<Common::DispatchPayloadGenerator>(cfg);
    }

    void add_unicast_write(const CoreRange worker_range, uint32_t length) {
        const auto worker = worker_range.start_coord;
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(worker, CoreType::WORKER);
        uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
        const bool is_mcast_ = false;
        const bool flush_prefetch = true;
        const bool inline_data = true;  // send data inline
        // Capture address before updating device_data
        uint32_t addr = device_data_.get_result_data_addr(worker, 0);
        // Generate payload
        std::vector<uint32_t> payload = payload_generator_->generate_payload(length);
        // Update Common::DeviceData for linear write
        Common::DeviceDataUpdater::update_linear_write(payload, device_data_, worker_range, is_mcast_);
        // Create the HostMemDeviceCommand
        HostMemDeviceCommand cmd = Common::CommandBuilder::build_linear_write_command<flush_prefetch, inline_data>(
            payload, worker_range, is_mcast_, noc_xy, addr, length);

        cmds_.push_back(std::move(cmd));
    }

    // Merge multiple of unicast writes into a single fetchQ command entry
    void add_merged_unicast_writes(const CoreRange worker_range, const std::vector<uint32_t>& lengths) {
        const auto worker = worker_range.start_coord;
        const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(worker, CoreType::WORKER);
        uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
        const bool is_mcast_ = false;
        const bool flush_prefetch = true;
        const bool inline_data = true;

        // Calculate size of a single merged entry of all lengths
        DeviceCommandCalculator cmd_calc;
        for (const auto length : lengths) {
            cmd_calc.add_dispatch_write_linear<flush_prefetch, inline_data>(length);
        }

        const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

        // Create the HostMemDeviceCommand for single merged entry
        HostMemDeviceCommand cmd(command_size_bytes);

        // Add all lengths into single cmd
        for (const auto length : lengths) {
            // Capture address before updating device_data
            uint32_t addr = device_data_.get_result_data_addr(worker, 0);
            // Generate payload
            std::vector<uint32_t> payload = payload_generator_->generate_payload(length);
            // Update Common::DeviceData for linear write
            Common::DeviceDataUpdater::update_linear_write(payload, device_data_, worker_range, is_mcast_);
            // Add the dispatch write linear command
            cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
                is_mcast_ ? worker_range.size() : 0,  // num_mcast_dests
                noc_xy,                               // NOC coordinates
                addr,                                 // destination address
                length,                               // data size
                payload.data()                        // payload data
            );
        }

        cmds_.push_back(std::move(cmd));
    }

    void add_packed_dram_read(
        const CoreRange worker_range,
        const uint32_t log_packed_read_page_size,
        const std::vector<uint32_t>& lengths,
        const std::function<std::vector<CQPrefetchRelayPagedPackedSubCmd>(
            const std::vector<uint32_t>& lengths, Common::DeviceData&, uint32_t log_page_sz, uint32_t n_sub_cmds)>&
            build_packed) {
        const auto worker = worker_range.start_coord;
        const CoreCoord last_virt_worker = device_->virtual_core_from_logical_core(worker, CoreType::WORKER);
        uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, last_virt_worker);

        const uint32_t total_length = std::accumulate(lengths.begin(), lengths.end(), 0u);
        const uint32_t n_sub_cmds = lengths.size();
        const bool flush_prefetch = false;
        const bool inline_data = false;

        uint32_t l1_addr = device_data_.get_result_data_addr(worker, 0);

        std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds =
            build_packed(lengths, device_data_, log_packed_read_page_size, n_sub_cmds);

        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged_packed<flush_prefetch, inline_data>(
            sub_cmds, noc_xy, l1_addr, total_length);

        cmds_.push_back(std::move(cmd));
    }

    void add_paged_dram_read(
        const CoreRange worker_range,
        uint32_t start_page,
        uint32_t base_addr_offset,
        uint32_t page_size_bytes,
        uint32_t num_pages,
        uint32_t length_adjust) {
        const auto worker = worker_range.start_coord;
        const CoreCoord first_worker = device_->virtual_core_from_logical_core(worker, CoreType::WORKER);
        uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_worker);
        const bool flush_prefetch = false;
        const bool inline_data = false;

        const uint32_t max_bytes_in_chunk = num_pages * page_size_bytes;
        const uint32_t pages_in_chunk = max_bytes_in_chunk / page_size_bytes;
        const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
        const uint32_t base_addr_words = base_addr_offset / sizeof(uint32_t);
        uint32_t base_addr = info_.dram_base_ + base_addr_offset;

        // Capture address before updating device_data
        uint32_t l1_addr = device_data_.get_result_data_addr(worker, 0);

        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged<flush_prefetch, inline_data>(
            noc_xy, l1_addr, start_page, base_addr, page_size_bytes, pages_in_chunk);

        for (uint32_t page = 0; page < pages_in_chunk; ++page) {
            const uint32_t page_id = start_page + page;
            const uint32_t bank_id = page_id % info_.num_banks_;
            uint32_t bank_offset = base_addr_words + (page_size_words * (page_id / info_.num_banks_));

            // Get the logical core for this bank
            const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
            const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

            // Update Common::DeviceData for paged read
            DeviceDataUpdater::update_paged_dram_read(
                worker_range, device_data_, bank_core, bank_id, bank_offset, page_size_words);
        }

        cmds_.push_back(std::move(cmd));
    }

    // Length should always be less than max_fetch_bytes_
    void add_packed_write(const std::vector<CoreCoord>& worker_cores, uint32_t length, bool no_stride = false) {
        const uint32_t num_sub_cmds = static_cast<uint32_t>(worker_cores.size());
        const uint32_t sub_cmds_bytes =
            tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), info_.l1_alignment_);

        // Build sub-commands
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds =
            Common::PackedWriteUtils::build_sub_cmds(device_, worker_cores, k_dispatch_downstream_noc);
        // Relevel once before generating commands
        device_data_.relevel(tt::CoreType::WORKER);

        // Clamp for dispatch page size (no_stride mode)
        if (no_stride) {
            const uint32_t max_allowed = info_.dispatch_buffer_page_size_ - sizeof(CQDispatchCmd) - sub_cmds_bytes;
            if (length > max_allowed) {
                log_warning(tt::LogTest, "Clamping packed_write cmd w/ no_stride to fit dispatch page");
                length = max_allowed;
            }
        }

        const CoreCoord& fw = worker_cores[0];
        // Capture address before updating device_data
        uint32_t addr = device_data_.get_result_data_addr(fw);
        // Generate Payload
        auto payload = payload_generator_->generate_payload_with_core(fw, length);
        // Update expected device_data for all cores
        Common::DeviceDataUpdater::update_packed_write(payload, device_data_, worker_cores, info_.l1_alignment_);

        // Build Command
        HostMemDeviceCommand cmd = Common::CommandBuilder::build_packed_write_command(
            payload, sub_cmds, addr, info_.l1_alignment_, info_.packed_write_max_unicast_sub_cmds_, no_stride);

        cmds_.push_back(std::move(cmd));
    }

    // L1 read to L1 write in same core: copies from worker core's start of data back to the end of data
    // For this function to run, there needs to prior data present in L1 already
    // Note: This function needs to run after L1 writes by previous commands
    void add_linear_read(const CoreRange worker_range, uint32_t length, uint32_t offset_from_current = 0) {
        const auto worker = worker_range.start_coord;
        const uint32_t bank_id = 0;
        // Ensure we have data to read
        const uint32_t avail_bytes = device_data_.size_at(worker, bank_id) * sizeof(uint32_t);
        if (length == 0 || offset_from_current >= avail_bytes) {
            return;  // nothing to copy
        }

        const CoreCoord first_worker = device_->virtual_core_from_logical_core(worker, CoreType::WORKER);
        uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_worker);

        constexpr bool flush_prefetch = false;
        constexpr bool inline_data = false;

        // Copy already existing data from exisitng addresses src core's L1 region
        const uint32_t src_addr =
            device_data_.get_base_result_addr(device_data_.get_core_type(worker)) + offset_from_current;
        // Write data to the next free/append position in modeled L1 stream
        const uint32_t dst_addr = device_data_.get_result_data_addr(worker, 0);  // append point

        // Shadow model: copy existing L1 data from src offset
        const uint32_t words = length / sizeof(uint32_t);
        const uint32_t offset_words = offset_from_current / sizeof(uint32_t);
        DeviceDataUpdater::update_read(worker, device_data_, worker, bank_id, offset_words, words);
        device_data_.pad(worker, bank_id, MetalContext::instance().hal().get_alignment(HalMemType::L1));

        // Barrier/stall to avoid RAW hazards
        HostMemDeviceCommand stall_cmd = CommandBuilder::build_dispatch_prefetch_stall();
        cmds_.push_back(std::move(stall_cmd));

        // Blit: dispatcher linear write (dest) + relay linear read (src)
        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_linear_read<flush_prefetch, inline_data>(
            noc_xy, dst_addr, src_addr, length);

        cmds_.push_back(std::move(cmd));
    }

    // Relay inline dispatcher to host writes smoke test helper
    void add_host_write(uint32_t length, const std::function<void(Common::DeviceData&)>& pad_host_data) {
        constexpr bool inline_data = true;
        std::vector<uint32_t> payload = payload_generator_->generate_payload(length);

        DeviceDataUpdater::update_host_data(device_data_, payload, length);

        // Create the HostMemDeviceCommand with pre-calculated size
        HostMemDeviceCommand cmd = CommandBuilder::build_relay_inline_host<inline_data>(payload, length);

        cmds_.push_back(std::move(cmd));

        // The completion queue is page-aligned (4KB pages)
        // Each write reserves full pages but only writes actual data,
        // leaving the remainder untouched (still HOST_DATA_DIRTY_PATTERN)
        // This ensures padding areas retain the sentinel value for validation
        pad_host_data(device_data_);
    }
};

// In this we, we test the terminate command by adding a linear write unicast
// with a small payload followed by commands to terminate prefetcher and dispatcher.
// exec_buf enabled execution needs special handling
TEST_P(BasePrefetcherTestFixture, TestTerminate) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - TestTerminate - Test Start");

    // Test parameters
    constexpr uint32_t xfer_size_bytes = 16;  // Very small write size
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    // Capture address before updating device_data
    uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

    // Generate payload
    std::vector<uint32_t> payload = payload_generator_->generate_payload(xfer_size_bytes);

    // PHASE 1: Generate terminate command metadata
    std::vector<HostMemDeviceCommand> work_cmds;
    HostMemDeviceCommand linear_write_cmd = Common::CommandBuilder::build_linear_write_command<true, true>(
        payload, worker_range, false, noc_xy, l1_addr, xfer_size_bytes);
    work_cmds.push_back(std::move(linear_write_cmd));

    std::vector<HostMemDeviceCommand> terminate_cmds;
    HostMemDeviceCommand dispatch_term_cmd = CommandBuilder::build_dispatch_terminate();
    terminate_cmds.push_back(std::move(dispatch_term_cmd));
    HostMemDeviceCommand prefetch_term_cmd = CommandBuilder::build_prefetch_terminate();
    terminate_cmds.push_back(std::move(prefetch_term_cmd));

    // PHASE 2, 3, 4: Execute and Validate
    // Don't wait for Finish()
    bool wait_for_completion = false;

    // If exec_buf is enabled, we must split execution
    // 1. Run workload (executes in exec_buf)
    // 2. Run terminate (must execute in issue queue, not exec_buf.
    // Executing terminate in exec_buf leads to a hang since exec_buf_end is never reached)
    if (use_exec_buf_) {
        // Step 1: Execute workload in exec_buf
        execute_generated_commands(
            work_cmds,
            device_data,
            worker_range.size(),
            num_iterations,
            false);  // Don't wait (device not done yet)

        // Step 2: Switch to Issue Queue for termination
        use_exec_buf_ = false;
        execute_generated_commands(
            terminate_cmds,
            device_data,
            0,
            1,
            wait_for_completion);  // Don't wait (device terminates)
    } else {
        // Standard flow: All commands in one Issue Queue stream
        work_cmds.insert(work_cmds.end(), terminate_cmds.begin(), terminate_cmds.end());

        execute_generated_commands(work_cmds, device_data, worker_range.size(), num_iterations, wait_for_completion);
    }
}

// Relay pre-populated data from DRAM using prefetcher to dispatcher, write it to L1 and validate it
TEST_P(BasePrefetcherTestFixture, DRAMToL1PagedRead) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - DRAMToL1PagedRead - Test Start");

    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t page_size_bytes = get_page_size();
    const uint32_t num_pages = get_num_pages();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    // No L1 -> Host writes, so pcie_data_addr is nullptr
    // We test DRAM -> L1
    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate paged read command metadata
    auto commands_per_iteration = generate_paged_read_commands(first_worker, page_size_bytes, num_pages, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// In this test, the prefetcher reads from L1 and relays it to dispatcher. Dispatcher then
// writes the data to the host completion queue.
// Note: Since we're writing into completion queue, we skip distributed::Finish
TEST_P(PrefetcherHostTextFixture, HostTest) {
    log_info(tt::LogTest, "PrefetcherHostTextFixture - HostTest - Test Start");

    const uint32_t max_data_size = DEVICE_DATA_SIZE;
    const uint32_t max_data_size_words = max_data_size / sizeof(uint32_t);
    const uint32_t dram_data_size_words = this->get_dram_data_size_words();
    const uint32_t num_iterations = this->get_num_iterations();

    std::vector<uint32_t> data(max_data_size_words);
    for (uint32_t i = 0; i < max_data_size_words; i++) {
        data[i] = i;
    }

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    CoreCoord phys_worker_core = device_->worker_core_from_logical_core(first_worker);
    // Write data into L1 for prefetcher to read it later
    MetalContext::instance().get_cluster().write_core(device_->id(), phys_worker_core, data, l1_base);
    MetalContext::instance().get_cluster().l1_barrier(device_->id());

    // Get completion queue buffer pointer
    void* completion_queue_buffer = mgr_->get_completion_queue_ptr(fdcq_->id());
    uint32_t completion_queue_size = mgr_->get_completion_queue_size(fdcq_->id());
    // Pre-fill with dirty pattern:
    // The dispatcher writes commands and data but doesn't overwrite padding regions
    // Pre-filling ensures padding areas retain the sentinel value for validation
    dirty_host_completion_buffer(completion_queue_buffer, completion_queue_size);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, completion_queue_buffer, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate host write command metadata
    auto commands_per_iteration = generate_host_write_commands(data, first_worker, l1_base, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    // Note: Skip distributed::Finish since we are manually writing into the completion queue
    // which Finish doesn't expect
    bool wait_for_completion = false;
    // For host writes, we need to wait for all writes to be written into the completion queue
    bool wait_for_host_writes = true;
    execute_generated_commands(
        commands_per_iteration,
        device_data,
        worker_range.size(),
        num_iterations,
        wait_for_completion,
        wait_for_host_writes);
}

// This tests relay of packed paged data using prefetcher to dispacher
// with multiple sub commands
TEST_P(PrefetcherPackedReadTestFixture, PackedReadTest) {
    log_info(tt::LogTest, "PrefetcherPackedReadTestFixture - PackedReadTest - Test Start");

    const uint32_t num_iterations = 1;
    const uint32_t dram_data_size_words = Common::DRAM_DATA_SIZE_WORDS;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate packed read command metadata
    auto commands_per_iteration = generate_packed_read_commands(first_worker, dram_alignment, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Ring Buffer operates differently than others
// Data is first staged (cached) into Ringbuffer (L1) from DRAM
// Then, we relay command header + data into dispatcher
// Note: Ring buffer is stateful as we set the ringbuffer offset on first load
TEST_P(PrefetcherRingbufferReadTestFixture, RingbufferReadTest) {
    log_info(tt::LogTest, "PrefetcherRingbufferReadTestFixture - RingbufferReadTest - Test Start");

    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate ringbuffer command metadata
    auto commands_per_iteration = generate_ringbuffer_relay_commands(first_worker, dram_alignment, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// End-To-End Paged/Interleaved Write+Read test that does the following:
//  1. Paged Write of host data to DRAM banks by dispatcher, followed by stall to avoid RAW hazard
//  2. Paged Read of DRAM banks by prefetcher, relay data to dispatcher for linear write to L1.
//  3. Do previous 2 steps in a loop, reading and writing new data until DEVICE_DATA_SIZE bytes is written to worker
//  core.
TEST_P(BasePrefetcherTestFixture, PagedReadWriteTest) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - PagedReadWriteTest - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t page_size_bytes = get_page_size();
    const uint32_t num_pages = get_num_pages();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;  // {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate paged end to end read + write command metadata
    auto commands_per_iteration =
        generate_paged_end_to_end_commands(first_worker, page_size_bytes, num_pages, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// This tests random configurations of commands like CQ_PREFETCH_CMD_RELAY_LINEAR, CQ_PREFETCH_CMD_RELAY_PAGED,
// CQ_PREFETCH_CMD_RELAY_INLINE etc
TEST_P(RandomTestFixture, RandomTest) {
    log_info(tt::LogTest, "RandomTestFixture - RandomTest - Test Start");

    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate random command metadata
    std::vector<HostMemDeviceCommand> commands_per_iteration;

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    while (remaining_bytes > 0) {
        // Assumes terminate is the last command...
        uint32_t cmd = payload_generator_->get_rand<uint32_t>(0, CQ_PREFETCH_CMD_TERMINATE - 1);
        const uint32_t limit_x = (worker_range.end_coord.x - first_worker.x - 1);
        const uint32_t limit_y = (worker_range.end_coord.y - first_worker.y - 1);
        uint32_t x = payload_generator_->get_rand<uint32_t>(0, limit_x);
        uint32_t y = payload_generator_->get_rand<uint32_t>(0, limit_y);

        CoreCoord worker_core(first_worker.x + x, first_worker.y + y);
        // Compute NOC encoding once
        const CoreCoord worker_core_virt = device_->virtual_core_from_logical_core(worker_core, CoreType::WORKER);
        const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, worker_core_virt);

        switch (cmd) {
            case CQ_PREFETCH_CMD_RELAY_LINEAR:
            // TODO: Temporarily dropping the linear relay from random mix.
            // Having this test enabled leads to intermittent validation failures
            // Issue: Randomization is probably landing on holes the hardware never wrote (padding/gaps in the model?),
            // so we're reading old L1 content and causing intermittent validation failures
            // Note: this was disabled in legacy as well

            /* if (has_l1_data) {
            //     CoreRange single_worker_range = {worker_core, worker_core};
            //     auto result = gen_random_linear_read_cmd(device_data, worker_core, noc_xy, remaining_bytes);
            //     if(result.has_value()){
            //         HostMemDeviceCommand& cmd = *result;
            //         commands_per_iteration.push_back(std::move(cmd));
            //     }
            //     break;
            // }*/
            case CQ_PREFETCH_CMD_RELAY_PAGED: {
                CoreRange single_worker_range = {worker_core, worker_core};
                auto result =
                    gen_random_dram_paged_cmd(device_data, worker_core, single_worker_range, noc_xy, remaining_bytes);
                if (result.has_value()) {
                    HostMemDeviceCommand& cmd = *result;
                    commands_per_iteration.push_back(std::move(cmd));
                }
                break;
            }
            case CQ_PREFETCH_CMD_RELAY_INLINE: {
                const CoreCoord last_worker = {worker_core.x + 1, worker_core.y + 1};
                CoreRange multi_worker_range = {worker_core, last_worker};
                auto result = gen_random_inline_cmd(device_data, multi_worker_range, noc_xy, remaining_bytes);
                if (result.has_value()) {
                    HostMemDeviceCommand& cmd = *result;
                    commands_per_iteration.push_back(std::move(cmd));
                }
                break;
            }
            case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            case CQ_PREFETCH_CMD_STALL:
            case CQ_PREFETCH_CMD_DEBUG:
            default: break;
        }
    }

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// This test targets relay linear H command: PREFETCH_H (MMIO) -> PREFETCH_D (Remote) -> DISPATCHER (Remote)
// Note: CQ_PREFETCH_CMD_RELAY_LINEAR_H needs to a standalone entry in fetchQ
TEST_P(PrefetchRelayLinearHTestFixture, RelayLinearHTest) {
    log_info(tt::LogTest, "PrefetchRelayLinearHTestFixture - RelayLinearHTest - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = mmio_device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    // Source (DRAM on MMIO Device) -> Use mmio_device_ (Chip 0)
    auto mmio_dram_base = mmio_device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);
    Common::DeviceData device_data(
        mmio_device_, worker_range, l1_base, mmio_dram_base, nullptr, false, dram_data_size_words, cfg_);

    auto commands_per_iteration =
        generate_prefetch_relay_h_commands(first_worker, mmio_dram_base, dram_alignment, device_data);

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Smoke test of prefetcher/dispatcher commands except add_dispatch_write_host
TEST_P(PrefetcherPackedReadTestFixture, SmokeTest) {
    log_info(tt::LogTest, "PrefetcherPackedReadTestFixture - SmokeTest - Test Start");

    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate smoke test command metadata
    std::vector<HostMemDeviceCommand> commands_per_iteration;

    HelperInfo info{
        dram_base_, num_banks_, l1_alignment_, packed_write_max_unicast_sub_cmds_, dispatch_buffer_page_size_};
    // Instantiate Helper
    SmokeTestHelper helper(device_, device_data, commands_per_iteration, info);

    // Section 1: Unicast
    // Important: reset device_data from prior tests if smoke test isn't the first
    helper.add_unicast_write(worker_range, 32);
    helper.add_unicast_write(worker_range, 1026);
    helper.add_unicast_write(worker_range, 8448);
    helper.add_unicast_write(worker_range, dispatch_buffer_page_size_);
    helper.add_unicast_write(worker_range, dispatch_buffer_page_size_ - sizeof(CQDispatchCmdLarge));
    helper.add_unicast_write(worker_range, 2 * dispatch_buffer_page_size_);
    helper.add_unicast_write(worker_range, (2 * dispatch_buffer_page_size_) - sizeof(CQDispatchCmdLarge));

    // Section 2: Merged Unicast Writes
    helper.add_merged_unicast_writes(worker_range, {112, 608, 64, 96});

    // Section 3: packed read
    constexpr uint32_t log_packed_read_page_size = 10;
    const uint32_t dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    const uint32_t length_to_read = tt::align(2080, dram_alignment);
    const uint32_t length_to_read_sdb = tt::align(DEFAULT_SCRATCH_DB_SIZE / 8, dram_alignment);
    // Create and pass PrefetcherPackedReadTestFixture::build_sub_cmds callable
    auto build_packed = [this](
                            const std::vector<uint32_t>& lengths,
                            Common::DeviceData& device_data,
                            uint32_t log_page_sz,
                            uint32_t n_sub_cmds) {
        return PrefetcherPackedReadTestFixture::build_sub_cmds(lengths, device_data, log_page_sz, n_sub_cmds);
    };
    // Uses last_worker as first_worker is filling up
    helper.add_packed_dram_read(worker_range, log_packed_read_page_size, {256, 512}, build_packed);
    helper.add_packed_dram_read(worker_range, log_packed_read_page_size, {1024, 2048}, build_packed);
    helper.add_packed_dram_read(worker_range, log_packed_read_page_size, {length_to_read}, build_packed);
    helper.add_packed_dram_read(
        worker_range, log_packed_read_page_size + 1, {length_to_read, length_to_read}, build_packed);

    std::vector<uint32_t> lengths{
        length_to_read_sdb,
        length_to_read_sdb,
        length_to_read_sdb,
        tt::align(DEFAULT_SCRATCH_DB_SIZE / 4, dram_alignment),   // won't fit in first pass
        tt::align(DEFAULT_SCRATCH_DB_SIZE / 2, dram_alignment)};  // won't fit in second pass
    helper.add_packed_dram_read(worker_range, log_packed_read_page_size + 1, lengths, build_packed);

    lengths.clear();
    lengths.push_back(tt::align((DEFAULT_SCRATCH_DB_SIZE / 4) + (2 * 1024) + 32, dram_alignment));
    lengths.push_back(tt::align((DEFAULT_SCRATCH_DB_SIZE / 4) + (3 * 1024) + 32, dram_alignment));
    lengths.push_back(tt::align(DEFAULT_SCRATCH_DB_SIZE / 2, dram_alignment));
    lengths.push_back(tt::align((DEFAULT_SCRATCH_DB_SIZE / 8) + (5 * 1024) + 96, dram_alignment));
    helper.add_packed_dram_read(worker_range, log_packed_read_page_size, lengths, build_packed);

    // Section 4: Read from dram, write to worker
    //                              start_page,                      base addr,      page_size,           pages,
    //                              length_adjust
    helper.add_paged_dram_read(worker_range, 0, 0, dram_alignment, num_banks_, 0);
    helper.add_paged_dram_read(worker_range, 4, dram_alignment, dram_alignment * 2, num_banks_, 0);
    helper.add_paged_dram_read(worker_range, 4, dram_alignment, dram_alignment * 2, num_banks_, 0);
    helper.add_paged_dram_read(worker_range, 0, 0, 128, 128, 0);
    helper.add_paged_dram_read(worker_range, 4, dram_alignment, 2048, num_banks_ + 4, 0);
    helper.add_paged_dram_read(worker_range, 5, dram_alignment, 2048, (num_banks_ * 3) + 1, 0);
    helper.add_paged_dram_read(worker_range, 3, tt::align(128, dram_alignment), 6144, num_banks_ - 1, 0);
    helper.add_paged_dram_read(worker_range, 3, tt::align(128, dram_alignment), 6144, num_banks_ - 1, 0);
    helper.add_paged_dram_read(worker_range, 0, 0, 128, 128, 32);
    helper.add_paged_dram_read(worker_range, 4, dram_alignment, 2048, num_banks_ * 2, 1536);
    helper.add_paged_dram_read(worker_range, 5, dram_alignment, 2048, (num_banks_ * 2) + 1, 256);
    helper.add_paged_dram_read(worker_range, 3, tt::align(128, dram_alignment), 6144, num_banks_ - 1, 640);
    // Large pages
    helper.add_paged_dram_read(worker_range, 0, 0, DEFAULT_SCRATCH_DB_SIZE / 2 + dram_alignment, 2, 128);
    helper.add_paged_dram_read(worker_range, 0, 0, DEFAULT_SCRATCH_DB_SIZE, 2, 0);
    // Forces length_adjust to back into prior read.  Device reads pages, shouldn't be a problem...
    uint32_t page_size = 256 + dram_alignment;
    uint32_t length = (DEFAULT_SCRATCH_DB_SIZE / 2 / page_size * page_size) + page_size;
    helper.add_paged_dram_read(worker_range, 3, 128, page_size, length / page_size, 160);

    // Section 5: Inline packed writes
    const CoreCoord first_worker_2x2 = default_worker_start;
    const CoreCoord last_worker_2x2 = {first_worker_2x2.x + 1, first_worker_2x2.y + 1};
    const CoreRange worker_range_2x2 = {first_worker_2x2, last_worker_2x2};
    // Pick worker cores once for all commands
    std::vector<CoreCoord> worker_cores{first_worker_2x2};
    helper.add_packed_write(worker_cores, 4);
    worker_cores.clear();
    for (uint32_t y = worker_range_2x2.start_coord.y; y <= worker_range_2x2.end_coord.y; ++y) {
        for (uint32_t x = worker_range_2x2.start_coord.x; x <= worker_range_2x2.end_coord.x; ++x) {
            worker_cores.push_back({x, y});
        }
    }
    helper.add_packed_write(worker_cores, 12);
    helper.add_packed_write(worker_cores, 12, true);
    worker_cores.clear();
    worker_cores.push_back(first_worker_2x2);
    worker_cores.push_back(last_worker_2x2);
    helper.add_packed_write(worker_cores, 156);

    // Section 6: Linear read -> Linear write test
    // Note: this test is dependent on already written data in L1
    // So it reads the above data from prior commands and needs to run after those
    helper.add_linear_read(worker_range, 32);
    helper.add_linear_read(worker_range, 65 * 1024);
    helper.add_linear_read(worker_range, dispatch_buffer_page_size_ - sizeof(CQDispatchCmdLarge));
    helper.add_linear_read(worker_range, (2 * dispatch_buffer_page_size_) - sizeof(CQDispatchCmdLarge));
    helper.add_linear_read(worker_range, (2 * dispatch_buffer_page_size_));
    // Skipping CQ_DISPATCH_CMD_DELAY from the legacy test here
    helper.add_unicast_write(worker_range, 1024);
    // Barrier/stall to avoid RAW hazards
    HostMemDeviceCommand stall_cmd = CommandBuilder::build_dispatch_prefetch_stall();
    commands_per_iteration.push_back(std::move(stall_cmd));
    uint32_t length_bytes = 32;
    uint32_t offset_words = device_data.size_at(worker_range.start_coord, 0) - (length_bytes / sizeof(uint32_t));
    uint32_t offset_bytes = offset_words * sizeof(uint32_t);
    helper.add_linear_read(worker_range, length_bytes, offset_bytes);

    // Section 7: Write offset
    // Simple Smoke test: do a change write_offset_idx to 48 and back to 0
    std::array<uint32_t, CQ_DISPATCH_MAX_WRITE_OFFSETS> write_offset1 = {48, 0, 0, 0};
    std::array<uint32_t, CQ_DISPATCH_MAX_WRITE_OFFSETS> write_offset2 = {};
    HostMemDeviceCommand write_offset_cmd1 = CommandBuilder::build_dispatch_write_offset(write_offset1);
    HostMemDeviceCommand write_offset_cmd2 = CommandBuilder::build_dispatch_write_offset(write_offset2);
    commands_per_iteration.push_back(std::move(write_offset_cmd1));
    commands_per_iteration.push_back(std::move(write_offset_cmd2));

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Smoke test for writes to Host from dispatcher
// This needed to be separate since it executes differently than others
// (we skip distributed::Finish)
TEST_P(PrefetcherHostTextFixture, HostSmokeTest) {
    log_info(tt::LogTest, "PrefetcherHostTextFixture - HostSmokeTest - Test Start");

    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);

    // Get completion queue buffer pointer
    void* completion_queue_buffer = mgr_->get_completion_queue_ptr(fdcq_->id());
    uint32_t completion_queue_size = mgr_->get_completion_queue_size(fdcq_->id());
    // Pre-fill with dirty pattern:
    // The dispatcher writes commands and data but doesn't overwrite padding regions
    // Pre-filling ensures padding areas retain the sentinel value for validation
    dirty_host_completion_buffer(completion_queue_buffer, completion_queue_size);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, completion_queue_buffer, false, dram_data_size_words, cfg_);

    // PHASE 1: Generate host smoke test command metadata
    std::vector<HostMemDeviceCommand> commands_per_iteration;

    HelperInfo info{
        dram_base_, num_banks_, l1_alignment_, packed_write_max_unicast_sub_cmds_, dispatch_buffer_page_size_};
    // Instantiate Helper
    SmokeTestHelper helper(device_, device_data, commands_per_iteration, info);

    auto add_host_write_func = [this](Common::DeviceData& device_data) {
        PrefetcherHostTextFixture::pad_host_data(device_data);
    };

    for (uint32_t multiplier = 1; multiplier < 3; multiplier++) {
        helper.add_host_write(multiplier * 32, add_host_write_func);
        helper.add_host_write(multiplier * 36, add_host_write_func);
        helper.add_host_write(multiplier * 1024, add_host_write_func);
        helper.add_host_write(
            (multiplier * dispatch_buffer_page_size_) - (2 * sizeof(CQDispatchCmd)), add_host_write_func);
        helper.add_host_write((multiplier * dispatch_buffer_page_size_) - sizeof(CQDispatchCmd), add_host_write_func);
        helper.add_host_write((multiplier * dispatch_buffer_page_size_), add_host_write_func);
        helper.add_host_write((multiplier * dispatch_buffer_page_size_) + sizeof(CQDispatchCmd), add_host_write_func);
    }

    // PHASE 2, 3, 4: Execute and Validate
    // Skip distributed::Finish since we are manually writing into the completion queue
    // which Finish doesn't expect
    bool wait_for_completion = false;
    // For host writes, we need to wait for all writes to be written into the completion queue
    bool wait_for_host_writes = true;
    execute_generated_commands(
        commands_per_iteration,
        device_data,
        worker_range.size(),
        num_iterations,
        wait_for_completion,
        wait_for_host_writes);
}

// All BasePrefetcherTestFixture tests with exec buff enabled / disabled
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    BasePrefetcherTestFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            true}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

// PrefetcherPackedReadTestFixture tests with exec buff enabled / disabled
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    PrefetcherPackedReadTestFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT, DRAM_PAGES_TO_READ_DEFAULT, DEFAULT_ITERATIONS, Common::DRAM_DATA_SIZE_WORDS, true},
        // With exec buf disabled + higher iterations
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS_SMOKE_RANDOM,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled + higher iterations
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS_SMOKE_RANDOM,
            Common::DRAM_DATA_SIZE_WORDS,
            true}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

// PrefetcherHostTextFixture test with exec buff enabled / disabled
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    PrefetcherHostTextFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            true}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

// PrefetcherRingbufferReadTestFixture test with exec buff enabled / disabled
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    PrefetcherRingbufferReadTestFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            true}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

// RandomTestFixture test with exec buff enabled / disabled
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    RandomTestFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT, DRAM_PAGES_TO_READ_DEFAULT, DEFAULT_ITERATIONS, Common::DRAM_DATA_SIZE_WORDS, true},
        // With exec buf disabled + higher iterations
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS_SMOKE_RANDOM,
            Common::DRAM_DATA_SIZE_WORDS,
            false},
        // With exec buf enabled + higher iterations
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS_SMOKE_RANDOM,
            Common::DRAM_DATA_SIZE_WORDS,
            true}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

// Runs only with exec buff disabled - RELAY_LINEAR_H must be standalone
INSTANTIATE_TEST_SUITE_P(
    PrefetcherTests,
    PrefetchRelayLinearHTestFixture,
    ::testing::Values(
        // With exec buf disabled
        PagedReadParams{
            DRAM_PAGE_SIZE_DEFAULT,
            DRAM_PAGES_TO_READ_DEFAULT,
            DEFAULT_ITERATIONS,
            Common::DRAM_DATA_SIZE_WORDS,
            false}),
    [](const testing::TestParamInfo<PagedReadParams>& info) {
        return std::to_string(info.param.page_size) + "B_" + std::to_string(info.param.num_pages) + "pages_" +
               std::to_string(info.param.num_iterations) + "iter_" + std::to_string(info.param.dram_data_size_words) +
               "words_" + (info.param.use_exec_buf ? "use_exec_buf_enabled" : "use_exec_buf_disabled");
    });

}  // namespace tt::tt_metal::tt_dispatch_tests::prefetcher_tests
