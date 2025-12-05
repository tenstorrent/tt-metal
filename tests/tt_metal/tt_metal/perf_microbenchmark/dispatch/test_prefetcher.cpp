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

bool use_coherent_data_g = false;  // Use sequential test data vs random
uint32_t dispatch_buffer_page_size_g =
    1 << tt::tt_metal::DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;  // Dispatch buffer page size (bytes)
uint32_t min_xfer_size_bytes_g = 16;                                     // Min transfer size for random commands
uint32_t max_xfer_size_bytes_g = 4096;                                   // Max transfer size for random commands
bool send_to_all_g = true;                                               // Send to all cores vs random subset
bool perf_test_g = false;                                                // Perf mode: use consistent sizes
constexpr uint32_t DEFAULT_HUGEPAGE_ISSUE_BUFFER_SIZE = 256 * 1024 * 1024;
uint32_t hugepage_issue_buffer_size_g = DEFAULT_HUGEPAGE_ISSUE_BUFFER_SIZE;

namespace tt::tt_dispatch {
namespace dispatcher_tests {

// constexpr uint32_t DEFAULT_ITERATIONS_LINEAR_WRITE = 3;
// constexpr uint32_t DEFAULT_ITERATIONS_PAGED_WRITE = 1;
// constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE = 1;
// constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE_LARGE = 1;
constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

// This will be ported to common.h when test_prefetcher.cpp is refactored
namespace DeviceDataUpdater {

void update_paged_dram_read(
    const CoreRange& workers,
    DeviceData& device_data,
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
void update_host_data(DeviceData& device_data, const std::vector<uint32_t>& data, uint32_t data_size_bytes) {
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

void update_read(
    const CoreCoord& worker_core,
    DeviceData& device_data,
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

// TODO: Remove this:
/*
Type 0 (Terminate)
Type 4 (DRAM Read)
Type 6 (L1 to Host)
Type 3 (PCIe)
Type 7 (Packed Read)
Type 8 (Ringbuffer)
Type 5 (Write + Read)
Type 1 (Smoke)  <- remaining
Type 2 (Random) <- remaining
Type 9 (Relay Linear H) <- hangs (skip for now)
*/

class BasePrefetcherTestFixture : public BaseTestFixture {
protected:
    // Common constants
    static constexpr uint32_t DEVICE_DATA_SIZE = 768 * 1024;
    static constexpr uint32_t MAX_PAGE_SIZE = 256 * 1024;  // bigger than scratch_db_page_size
    static constexpr uint32_t DRAM_PAGE_SIZE_DEFAULT = 1024;
    static constexpr uint32_t DRAM_PAGES_TO_READ_DEFAULT = 16;
    static constexpr uint32_t HOST_DATA_DIRTY_PATTERN = 0xbaadf00d;
    static constexpr uint32_t PCIE_TRANSFER_SIZE_DEFAULT = 4096;
    static constexpr uint32_t DEFAULT_SCRATCH_DB_SIZE = 16 * 1024;
    static constexpr uint32_t MIN_READ_SIZE = 128;  // Minimum meaningful transfer size, aligns with DRAM
    static constexpr uint32_t DEFAULT_PREFETCH_Q_ENTRIES = 1024;
    static constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_BASE_ADDR = 0x1f400000;  // magic, half of dram
    static constexpr uint32_t DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE = 10;

    // Default values for inline data and flush prefetch for linear write commands
    static constexpr bool inline_data_ = false;
    static constexpr bool flush_prefetch_ = false;
    static constexpr bool hugepage_write_ = true;
    // Exec Buf Configuration
    // TODO: parameterize
    bool use_exec_buf_ = true;

    uint32_t dram_base_;
    uint32_t num_banks_;
    uint32_t host_alignment_;

    void SetUp() override {
        BaseTestFixture::SetUp();
        dram_base_ = device_->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        num_banks_ = device_->allocator()->get_num_banks(BufferType::DRAM);
        host_alignment_ = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    }

    void execute_generated_commands(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool wait_for_completion = true,
        bool notify_host = false) override {
        if (use_exec_buf_) {
            execute_generated_commands_exec_buff(
                commands_per_iteration,
                device_data,
                num_cores_to_log,
                num_iterations,
                wait_for_completion,
                notify_host,
                *this);
            return;
        }

        BaseTestFixture::execute_generated_commands(
            commands_per_iteration, device_data, num_cores_to_log, num_iterations, wait_for_completion, notify_host);
    }

private:
    // Helper function to execute generated commands via exec buff on device
    // Orchestrates the command buffer reservation, writing, and submission
    void execute_generated_commands_exec_buff(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool wait_for_completion,
        bool notify_host,
        BasePrefetcherTestFixture& fixture) {
        // 1. Add all commands to be uploaded into exec buff into a single vector
        std::vector<uint32_t> exec_buf_data;
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (const auto& cmd : commands_per_iteration) {
                const uint32_t* src_ptr = reinterpret_cast<uint32_t*>(cmd.data());
                const uint32_t* end_ptr =
                    reinterpret_cast<uint32_t*>(cmd.data()) + (cmd.size_bytes() / sizeof(uint32_t));
                exec_buf_data.insert(exec_buf_data.end(), src_ptr, end_ptr);
            }
        }

        // 2. Append exec_buf_end command (terminate the trace execution and switch back to issue queue)
        const uint64_t exec_terminate_cmd_size =
            tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), fixture.host_alignment_);
        DeviceCommand exec_terminate(exec_terminate_cmd_size);
        exec_terminate.add_prefetch_exec_buf_end();
        const uint32_t* src_ptr = reinterpret_cast<uint32_t*>(exec_terminate.data());
        const uint32_t* end_ptr =
            reinterpret_cast<uint32_t*>(exec_terminate.data()) + (exec_terminate.size_bytes() / sizeof(uint32_t));
        exec_buf_data.insert(exec_buf_data.end(), src_ptr, end_ptr);

        // 3. Write exec buff data into DRAM
        const uint32_t page_size = 1 << fixture.DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE;
        const uint32_t exec_buf_base_addr = fixture.DRAM_EXEC_BUF_DEFAULT_BASE_ADDR;

        // Pad data to full page alignment
        size_t size_bytes = exec_buf_data.size() * sizeof(uint32_t);
        size_t padded_size_bytes = tt::align(size_bytes, page_size);
        exec_buf_data.resize(padded_size_bytes / sizeof(uint32_t));

        uint32_t num_pages = padded_size_bytes / page_size;
        uint32_t data_idx = 0;

        // Create pages of exec buff data and write to DRAM
        for (uint32_t page_idx = 0; page_idx < num_pages; page_idx++) {
            uint32_t bank_id = page_idx % fixture.num_banks_;
            uint32_t bank_offset = (page_idx / fixture.num_banks_) * page_size;
            uint32_t addr = exec_buf_base_addr + bank_offset;

            std::vector<uint32_t> page_data(
                exec_buf_data.begin() + data_idx, exec_buf_data.begin() + data_idx + (page_size / sizeof(uint32_t)));

            tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, bank_id, addr, page_data);

            data_idx += (page_size / sizeof(uint32_t));
        }

        // Ensure DRAM writes are visible to device
        tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device_->id());

        // 4. Reserve and Write exec_buff command
        // TODO: should we add below to DeviceCommandCalculator?
        uint32_t cmd_size = tt::align(sizeof(CQPrefetchCmd), fixture.host_alignment_);
        void* cmd_buffer_base = mgr_->issue_queue_reserve(cmd_size, fdcq_->id());
        // Use DeviceCommand helper (HugepageDeviceCommand) to write to the issue queue memory
        HugepageDeviceCommand exec_cmd(cmd_buffer_base, cmd_size);
        exec_cmd.add_prefetch_exec_buf(exec_buf_base_addr, fixture.DRAM_EXEC_BUF_DEFAULT_LOG_PAGE_SIZE, num_pages);

        // Verifies destination memory bounds
        device_data.overflow_check(device_);

        // Submit and execute commands
        mgr_->issue_queue_push_back(exec_cmd.write_offset_bytes(), fdcq_->id());

        // Write the commands to the device-side fetch queue with STALL FLAG
        const auto start = std::chrono::steady_clock::now();
        mgr_->fetch_queue_reserve_back(fdcq_->id());
        mgr_->fetch_queue_write(cmd_size, fdcq_->id(), true);  // stall_prefetcher = true with exec buff execution

        // Wait for completion of the issued commands
        if (wait_for_completion) {
            if (notify_host) {
                // For host write tests: wait for completion queue writes without using events
                // This polls the completion queue write pointer updated by the dispatcher
                // std::atomic<bool> exit_condition{false};
                // mgr_->completion_queue_wait_front(fdcq_->id(), exit_condition);

                // Small delay to ensure all async writes are complete
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                // Normal path: use Finish() which handles events properly
                distributed::Finish(mesh_device_->mesh_command_queue());
            }
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
};

class PrefecherHostTextFixture : virtual public BasePrefetcherTestFixture {
protected:
    void pad_host_data(DeviceData& device_data) {
        one_core_data_t& host_data = device_data.get_data()[device_data.get_host_core()][0];

        int pad =
            dispatch_buffer_page_size_ - ((host_data.data.size() * sizeof(uint32_t)) % dispatch_buffer_page_size_);
        pad = pad % dispatch_buffer_page_size_;

        for (int i = 0; i < pad / sizeof(uint32_t); i++) {
            device_data.push_one(device_data.get_host_core(), 0, HOST_DATA_DIRTY_PATTERN);  // ← Use 0xbaadf00d
        }
    }

    void dirty_host_completion_buffer(void* completion_queue_buffer, uint32_t size_bytes) {
        uint32_t* buffer = static_cast<uint32_t*>(completion_queue_buffer);
        uint32_t size_words = size_bytes / sizeof(uint32_t);

        for (uint32_t i = 0; i < size_words; i++) {
            buffer[i] = HOST_DATA_DIRTY_PATTERN;  // 0xbaadf00d
        }

        tt_driver_atomics::sfence();
    }
};

class PrefetcherPackedReadTestFixture : virtual public BasePrefetcherTestFixture {
protected:
    std::vector<CQPrefetchRelayPagedPackedSubCmd> build_sub_cmds(
        const std::vector<uint32_t>& lengths,
        DeviceData& device_data,
        uint32_t packed_read_page_size,
        uint32_t n_sub_cmds) {
        int count = 0;
        uint32_t page_size_bytes = 1 << packed_read_page_size;
        std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds;
        sub_cmds.reserve(n_sub_cmds);
        for (auto length : lengths) {
            TT_ASSERT((length & (MetalContext::instance().hal().get_alignment(HalMemType::DRAM) - 1)) == 0);
            CQPrefetchRelayPagedPackedSubCmd sub_cmd{};
            sub_cmd.start_page = 0;  // TODO: randomize?
            sub_cmd.log_page_size = packed_read_page_size;
            sub_cmd.base_addr = dram_base_ + count * page_size_bytes;
            sub_cmd.length = length;
            sub_cmds.push_back(sub_cmd);
            count++;

            // Model the packed paged read in this function by updating worker data with interleaved/paged DRAM data,
            // for validation later.
            // TODO: see if this whole thing can fit in update_read()
            uint32_t length_words = length / sizeof(uint32_t);
            uint32_t base_addr_words = (sub_cmd.base_addr - dram_base_) / sizeof(uint32_t);
            uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);

            // Get data from DRAM map, add to all workers, but only set valid for cores included in workers range.
            uint32_t page_idx = sub_cmd.start_page;
            for (uint32_t i = 0; i < length_words; i += page_size_words) {
                uint32_t dram_bank_id = page_idx % num_banks_;
                auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(dram_bank_id);
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
};

class PrefetcherRingbufferReadTestFixture : virtual public BasePrefetcherTestFixture {
    static constexpr uint32_t MAX_PAGE_OFFSET = 5;

public:
    // Set ring buffer offset to arbitrary number
    static constexpr uint32_t WRITE_OFFSET = 1234;

    void populate_ringbuffer_from_dram(
        HostMemDeviceCommand& cmd,
        const std::vector<uint32_t>& lengths,
        DeviceData& device_data,
        uint32_t ringbuffer_read_page_size_log2,
        uint32_t n_sub_cmds) {
        bool reset = false;
        uint32_t page_size_bytes = 1 << ringbuffer_read_page_size_log2;
        int count = 0;

        for (auto length : lengths) {
            uint8_t wraparound_flag = 0;
            if (!reset) {
                wraparound_flag = CQ_PREFETCH_PAGED_TO_RING_BUFFER_FLAG_RESET_TO_START;
                reset = true;
            }

            CQPrefetchPagedToRingbufferCmd ringbuffer_cmd{};
            ringbuffer_cmd.flags = wraparound_flag;
            ringbuffer_cmd.log2_page_size =
                ringbuffer_read_page_size_log2;  // uint16_t(HostMemDeviceCommand::LOG2_PROGRAM_PAGE_SIZE),
            // RECOMMENDED: Use uint16_t instead of uint8_t for std::uniform_int_distribution, then cast it to uint8_t
            ringbuffer_cmd.start_page = static_cast<uint8_t>(
                payload_generator_->get_rand<uint16_t>(0, MAX_PAGE_OFFSET - 1));  // std::rand() % MAX_PAGE_OFFSET;
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
                auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(dram_bank_id);
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
            sub_cmd.start = current_offset - WRITE_OFFSET;  // Since set_ringbuffer_offset is set to WRITE_OFFSET
            sub_cmd.length = length;
            current_offset += length;
            sub_cmds.push_back(sub_cmd);
        }

        return sub_cmds;
    }
};

class PrefetcherSmokeTestFixture : public PrefetcherRingbufferReadTestFixture,
                                   public PrefetcherPackedReadTestFixture,
                                   public PrefecherHostTextFixture {
protected:
    void SetUp() override { PrefetcherPackedReadTestFixture::SetUp(); }
};

// Host-side helpers used by tests to emit the same CQ commands
// that prefetcher/dispatcher code emits. This namespace replicates the production code's command generation logic
// for testing purposes.
namespace CommandBuilder {

HostMemDeviceCommand build_dispatch_terminate() {
    bool dispatch_sub_enabled = tt_metal::MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled();
    // Calculate total command buffer size needed
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

HostMemDeviceCommand build_dispatch_prefetch_stall() {
    // Calculate the command size
    DeviceCommandCalculator calc;
    calc.add_dispatch_wait_with_prefetch_stall();
    const uint32_t command_size_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    cmd.add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER | CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY, 0, 0, 0);

    return cmd;
}

HostMemDeviceCommand build_prefetch_relay_linear(uint32_t noc_xy, uint32_t l1_buf_base, uint32_t data_size_bytes) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear_host();
    calc.add_prefetch_relay_linear();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);
    // Dispatcher command to write data from prefetcher into completion buffer
    cmd.add_dispatch_write_host(
        false,            // flush_prefetch
        data_size_bytes,  // data_sizeB
        false,            // is_event
        0,                // pad1
        nullptr           // data
    );

    // Relay data from L1 to dispatcher
    cmd.add_prefetch_relay_linear(noc_xy, data_size_bytes, l1_buf_base);

    return cmd;
}

HostMemDeviceCommand build_prefetch_terminate() {
    // Calculate total command buffer size needed
    DeviceCommandCalculator calc;
    calc.add_prefetch_terminate();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();
    HostMemDeviceCommand cmd(total_cmd_bytes);
    cmd.add_prefetch_terminate();

    return cmd;
}

template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_prefetch_relay_paged(
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t start_page,
    uint32_t base_addr,
    uint32_t page_size_bytes,
    uint32_t pages_in_chunk) {
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear<flush_prefetch, inline_data>(0);
    calc.add_prefetch_relay_paged();
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);

    uint32_t transfer_size = page_size_bytes * pages_in_chunk;
    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        0,              // num_mcast_dests
        noc_xy,         // NOC coordinates
        addr,           // destination address
        transfer_size,  // data size
        nullptr         // payload data
    );

    cmd.add_prefetch_relay_paged(true, start_page, base_addr, page_size_bytes, pages_in_chunk, 0);

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
    // Pre-calculate the exact size to allocate correct amount of memory in HostMemDeviceCommand buffer
    DeviceCommandCalculator calc;
    calc.add_dispatch_write_linear<flush_prefetch, inline_data>(0);
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
    DeviceData& device_data,
    PrefetcherRingbufferReadTestFixture& fixture,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t total_length,
    uint32_t ringbuffer_read_page_size_log2) {
    const uint32_t n_sub_cmds = sub_cmds.size();
    // Calculate the command size using DeviceCommandCalculator
    // Pre-calculate the exact size to allocate correct amount of memory in HostMemDeviceCommand buffer
    DeviceCommandCalculator calc;
    for (uint32_t i = 0; i < n_sub_cmds; i++) {
        calc.add_prefetch_paged_to_ringbuffer();
    }
    calc.add_prefetch_set_ringbuffer_offset();
    calc.add_dispatch_write_linear<false, false>(0);
    calc.add_prefetch_relay_ringbuffer(n_sub_cmds);
    const uint32_t total_cmd_bytes = calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(total_cmd_bytes);

    // First, we populate the ringbuffer
    fixture.populate_ringbuffer_from_dram(cmd, lengths, device_data, ringbuffer_read_page_size_log2, n_sub_cmds);

    // Second, we set the read offset in the ring buffer
    cmd.add_prefetch_set_ringbuffer_offset(fixture.WRITE_OFFSET);

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

using namespace tt::tt_metal;

TEST_F(BasePrefetcherTestFixture, TestTerminate) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - TestTerminate - Test Start");

    // Test parameters
    const uint32_t xfer_size_bytes = 16;                         // Very small write size
    const uint32_t num_iterations = 1;                           // this->get_num_iterations();
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;  // this->get_dram_data_size_words();

    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    // Capture address before updating device_data
    uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

    // Generate payload
    std::vector<uint32_t> payload = payload_generator_->generate_payload(xfer_size_bytes);

    std::vector<HostMemDeviceCommand> work_cmds;
    work_cmds.push_back(CommandBuilder::Common::build_linear_write_command<true, true>(
        payload, worker_range, false, noc_xy, l1_addr, xfer_size_bytes));

    std::vector<HostMemDeviceCommand> terminate_cmds;
    terminate_cmds.push_back(CommandBuilder::build_dispatch_terminate());
    terminate_cmds.push_back(CommandBuilder::build_prefetch_terminate());

    // If exec_buf is enabled, we must split execution
    // 1. Run workload (executes in exec_buf)
    // 2. Run terminate (must execute in issue queue, not exec_buf.
    // Executing terminate in exec_buf leads to a hang since exec_buf_end is never reached)
    if (this->use_exec_buf_) {
        // Step 1: Execute workload in exec_buf
        execute_generated_commands(
            work_cmds,
            device_data,
            worker_range.size(),
            num_iterations,
            false);  // Don't wait (device not done yet)

        // Step 2: Switch to Issue Queue for termination
        this->use_exec_buf_ = false;
        execute_generated_commands(terminate_cmds, device_data, 0, 1,
                                   false);  // Don't wait (device terminates)
    } else {
        // Standard flow: All commands in one Issue Queue stream
        work_cmds.insert(
            work_cmds.end(),
            std::make_move_iterator(terminate_cmds.begin()),
            std::make_move_iterator(terminate_cmds.end()));

        execute_generated_commands(work_cmds, device_data, worker_range.size(), num_iterations, false, false);
    }
}

// Read pre-populated data from DRAM, Write it to L1 and Validate it
TEST_F(BasePrefetcherTestFixture, DRAMToL1PagedRead) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - DRAMToL1PagedRead - Test Start");

    // TODO: look at parameters in run_cpp_fd2_tests.sh and make the below
    //  parameters configurable
    const uint32_t num_iterations = 5;
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;
    const uint32_t page_size_bytes = DRAM_PAGE_SIZE_DEFAULT;
    const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
    const uint32_t num_pages = DRAM_PAGES_TO_READ_DEFAULT;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    // No L1 -> Host writes, so pcie_data_addr is nullptr
    // We test DRAM -> L1
    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    std::vector<HostMemDeviceCommand> commands_per_iteration;
    uint32_t absolute_start_page = 0;
    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    while (remaining_bytes > 0) {
        // Calculate how many pages fit in this chunk
        uint32_t max_bytes_in_chunk = num_pages * page_size_bytes;
        uint32_t bytes_in_chunk = std::min(remaining_bytes, max_bytes_in_chunk);
        uint32_t pages_in_chunk = bytes_in_chunk / page_size_bytes;

        uint32_t start_page = absolute_start_page % num_banks_;
        uint32_t base_addr = (absolute_start_page / num_banks_) * page_size_bytes + dram_base_;

        // Capture address before updating device_data
        uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged<flush_prefetch_, inline_data_>(
            noc_xy, l1_addr, start_page, base_addr, page_size_bytes, pages_in_chunk);

        for (uint32_t page = 0; page < pages_in_chunk; ++page) {
            const uint32_t page_id = absolute_start_page + page;
            const uint32_t bank_id = page_id % num_banks_;
            uint32_t bank_offset = page_size_words * (page_id / num_banks_);

            // Get the logical core for this bank
            const auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(bank_id);
            const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

            // Update DeviceData for paged read
            DeviceDataUpdater::update_paged_dram_read(
                worker_range, device_data, bank_core, bank_id, bank_offset, page_size_words);
        }

        commands_per_iteration.push_back(cmd);

        absolute_start_page += pages_in_chunk;
        remaining_bytes -= bytes_in_chunk;
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// TODO: think about if we need a barrier in execute_generated_commands()
//  if (notify_host) {
//      // For host write tests: barrier ensures writes complete
//      std::this_thread::sleep_for(std::chrono::milliseconds(100));
//  } else {
//      // Normal path: use Finish() for event synchronization
//      distributed::Finish(mesh_device_->mesh_command_queue());
//  }
//  For now, seems to work without it
//  In this test, the prefetcher reads from L1 and relays it to dispatcher. Dispatcher then
//  writes the data to the host completion queue
TEST_F(PrefecherHostTextFixture, HostTest) {
    log_info(tt::LogTest, "PrefecherHostTextFixture - HostTest - Test Start");
    const uint32_t max_data_size = DEVICE_DATA_SIZE;
    const uint32_t max_data_size_words = max_data_size / sizeof(uint32_t);
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;  // this->get_dram_data_size_words();
    const uint32_t num_iterations = 1;                           // this->get_num_iterations();

    std::vector<uint32_t> data;
    data.reserve(max_data_size_words);
    for (uint32_t i = 0; i < max_data_size_words; i++) {
        data.push_back(i);
    }

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    // TODO: check if this is still needed?
    uint32_t l1_base_aligned = tt::align(l1_base, (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE));
    CoreCoord phys_worker_core = device_->worker_core_from_logical_core(first_worker);
    uint32_t l1_buf_base = l1_base_aligned + (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);  // Reserve a page.
    // Write data into L1 for prefetcher to read it later
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(device_->id(), phys_worker_core, data, l1_buf_base);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_->id());

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    // Get completion queue buffer pointer
    void* completion_queue_buffer = mgr_->get_completion_queue_ptr(fdcq_->id());
    uint32_t completion_queue_size = mgr_->get_completion_queue_size(fdcq_->id());
    // Pre-fill with dirty pattern:
    // The dispatcher writes commands and data but doesn't overwrite padding regions
    // Pre-filling ensures padding areas retain the sentinel value for validation
    dirty_host_completion_buffer(completion_queue_buffer, completion_queue_size);

    DeviceData device_data(
        device_, worker_range, l1_base, dram_base_, completion_queue_buffer, false, dram_data_size_words);

    std::vector<HostMemDeviceCommand> commands_per_iteration;

    for (int count = 1; count < 100; count++) {
        // TODO: replace with DispatchPayloadGenerator
        uint32_t max_limit = max_data_size_words / 100;
        uint32_t data_size_words = payload_generator_->get_rand<uint32_t>(0, max_limit - 1) * count +
                                   1;  // (std::rand() % (max_data_size_words / 100) * count) + 1;
        uint32_t data_size_bytes = data_size_words * sizeof(uint32_t);

        // Create the HostMemDeviceCommand with pre-calculated size
        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_linear(noc_xy, l1_buf_base, data_size_bytes);

        commands_per_iteration.push_back(cmd);

        DeviceDataUpdater::update_host_data(device_data, data, data_size_bytes);

        // The completion queue is page-aligned (4KB pages)
        // Each write reserves full pages but only writes actual data,
        // leaving the remainder untouched (still 0xbaadf00d)
        // This ensures padding areas retain the sentinel value for validation
        pad_host_data(device_data);
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations, true, true);
}

// This is the same as linear write tests in dispatcher. Not sure if this adds any value here
// TODO: think about removing this
TEST_F(BasePrefetcherTestFixture, PCIEToL1Read) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - PCIEToL1Read - Test Start");

    // Test parameters
    const uint32_t num_iterations = 1;                           // this->get_num_iterations();
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;  // this->get_dram_data_size_words();
    const bool is_mcast = false;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    // This vector stores commands related information for each iteration
    std::vector<HostMemDeviceCommand> commands_per_iteration;

    // This loop generates fixed-sized (PCIE_TRANSFER_SIZE_DEFAULT) linear write
    // commands until all transfer bytes are consumed
    while (remaining_bytes > 0) {
        // Generate random transfer size
        uint32_t xfer_size_bytes = std::min(remaining_bytes, PCIE_TRANSFER_SIZE_DEFAULT);

        // Capture address before updating device_data
        uint32_t addr = device_data.get_result_data_addr(first_worker, 0);

        // Generate payload
        std::vector<uint32_t> payload = payload_generator_->generate_payload(xfer_size_bytes);

        // Update DeviceData for linear write
        DeviceDataUpdater::Common::update_linear_write(payload, device_data, worker_range, is_mcast);

        // Create the HostMemDeviceCommand
        HostMemDeviceCommand cmd = CommandBuilder::Common::build_linear_write_command<true, true>(
            payload, worker_range, is_mcast, noc_xy, addr, xfer_size_bytes);

        commands_per_iteration.push_back(std::move(cmd));
        remaining_bytes -= xfer_size_bytes;
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

TEST_F(PrefetcherPackedReadTestFixture, PackedReadTest) {
    log_info(tt::LogTest, "PrefetcherPackedReadTestFixture - PackedReadTest - Test Start");

    // TODO: look at parameters in run_cpp_fd2_tests.sh and make the below
    //  parameters configurable
    const uint32_t num_iterations = 1;
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;
    const bool relay_max_packed_paged_submcds = true;  // TODO: randomize?

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    std::vector<HostMemDeviceCommand> commands_per_iteration;

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    constexpr uint32_t max_size128b = (DEFAULT_SCRATCH_DB_SIZE / 2) >> 7;
    while (remaining_bytes > 0) {
        // TODO: replace all std::rand() with DispatchPayloadGenerator
        uint32_t packed_read_page_size = payload_generator_->get_rand<uint32_t>(0, 2) +
                                         9;  // (std::rand() % 3) + 9;  // log2 values. i.e., 512, 1024, 2048
        uint32_t n_sub_cmds = relay_max_packed_paged_submcds
                                  ? CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS
                                  : payload_generator_->get_rand<uint32_t>(0, 6) + 1;  // (std::rand() % 7) + 1;
        uint32_t max_read_size = (1 << packed_read_page_size) * num_banks_;

        std::vector<uint32_t> lengths;
        lengths.reserve(n_sub_cmds);
        uint32_t total_length = 0;
        for (uint32_t i = 0; i < n_sub_cmds; i++) {
            // limit the length to min and max read size
            uint32_t length = tt::align(
                std::min(
                    std::max(MIN_READ_SIZE, (payload_generator_->get_rand<uint32_t>(0, max_size128b - 1)) << 7),
                    max_read_size),
                dram_alignment);  //(std::rand() % max_size128b) << 7), max_read_size), dram_alignment);
            total_length += length;
            lengths.push_back(length);
        }

        // TODO: Can we use Calculator to check if this will exceed DEVICE_DATA_SIZE?
        if (device_data.size() * sizeof(uint32_t) + total_length > DEVICE_DATA_SIZE) {
            break;
        }

        uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

        std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds =
            build_sub_cmds(lengths, device_data, packed_read_page_size, n_sub_cmds);

        HostMemDeviceCommand cmd = CommandBuilder::build_prefetch_relay_paged_packed<flush_prefetch_, inline_data_>(
            sub_cmds, noc_xy, l1_addr, total_length);

        commands_per_iteration.push_back(std::move(cmd));
        remaining_bytes -= total_length;
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Ring Buffer operates differently than others
// Data is first staged (cached) into Ringbuffer (L1) from DRAM
// Then, we relay command header + data into dispatcher
// Note: Ring buffer is stateful as we set the ringbuffer offset on first load
TEST_F(PrefetcherRingbufferReadTestFixture, RingbufferReadTest) {
    log_info(tt::LogTest, "PrefetcherRingbufferReadTestFixture - RingbufferReadTest - Test Start");

    // TODO: look at parameters in run_cpp_fd2_tests.sh and make the below
    //  parameters configurable
    const uint32_t num_iterations = 1;
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    std::vector<HostMemDeviceCommand> commands_per_iteration;

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    while (remaining_bytes > 0) {
        // TODO: replace all std::rand() with DispatchPayloadGenerator
        uint32_t ringbuffer_read_page_size_log2 = payload_generator_->get_rand<uint32_t>(0, 2) +
                                                  9;  // (std::rand() % 3) + 9;  // log2 values. i.e., 512, 1024, 2048
        uint32_t n_sub_cmds = payload_generator_->get_rand<uint32_t>(0, 6) + 1;  // (std::rand() % 7) + 1;
        uint32_t max_read_size = (1 << ringbuffer_read_page_size_log2) * num_banks_;

        std::vector<uint32_t> lengths;
        lengths.reserve(n_sub_cmds);

        uint32_t total_length = 0;
        for (uint32_t i = 0; i < n_sub_cmds; i++) {
            // limit the length to min and max read size
            uint32_t length = tt::align(
                std::min(
                    max_read_size,
                    std::max(MIN_READ_SIZE, payload_generator_->get_rand<uint32_t>(0, DEFAULT_SCRATCH_DB_SIZE - 1))),
                dram_alignment);  // std::rand() % DEFAULT_SCRATCH_DB_SIZE)), dram_alignment);
            total_length += length;
            lengths.push_back(length);
        }

        // Can we use Calculator to check if this will exceed DEVICE_DATA_SIZE?
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

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// End-To-End Paged/Interleaved Write+Read test that does the following:
//  1. Paged Write of host data to DRAM banks by dispatcher cmd, followed by stall to avoid RAW hazard
//  2. Paged Read of DRAM banks by prefetcher, relay data to dispatcher for linear write to L1.
//  3. Do previous 2 steps in a loop, reading and writing new data until DEVICE_DATA_SIZE bytes is written to worker
//  core.
TEST_F(BasePrefetcherTestFixture, PagedReadWriteTest) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - PagedReadWriteTest - Test Start");

    // TODO: look at parameters in run_cpp_fd2_tests.sh and make the below
    //  parameters configurable
    const uint32_t num_iterations = 5;
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;
    const uint32_t page_size_bytes = DRAM_PAGE_SIZE_DEFAULT;
    const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
    const uint32_t num_pages = DRAM_PAGES_TO_READ_DEFAULT;
    bool is_dram = true;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;  // {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    const auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    const uint32_t page_size_alignment_bytes = device_->allocator()->get_alignment(buf_type);
    const tt::CoreType core_type = is_dram ? tt::CoreType::DRAM : tt::CoreType::WORKER;

    std::vector<HostMemDeviceCommand> commands_per_iteration;

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;
    uint32_t absolute_start_page = 0;

    // This loop generates commands, payloads, and updates expectations all at once
    // Each iteration represents one "chunk" that fits in max_payload_per_cmd_bytes
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
            const auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(bank_id);
            const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

            // Generate payload with page id
            std::vector<uint32_t> page_payload =
                payload_generator_->generate_payload_with_page_id(page_size_words, page_id);

            // Update DeviceData for paged write
            DeviceDataUpdater::Common::update_paged_write(
                page_payload, device_data, bank_core, bank_id, page_size_alignment_bytes);

            // Append page payload to chunk payload
            chunk_payload.insert(chunk_payload.end(), page_payload.begin(), page_payload.end());
        }

        // Calculate base address for the command
        const uint32_t bank_offset =
            tt::align(page_size_bytes, page_size_alignment_bytes) * (absolute_start_page / num_banks_);
        const uint32_t base_addr = device_data.get_base_result_addr(core_type) + bank_offset;
        // Calculate start page for the command
        const uint16_t start_page_cmd = absolute_start_page % num_banks_;

        //  Step 1: Paged Write of host data to DRAM banks by dispatcher cmd
        HostMemDeviceCommand cmd_dispatch_dram = CommandBuilder::Common::build_paged_write_command<hugepage_write_>(
            chunk_payload, base_addr, page_size_bytes, pages_in_chunk, start_page_cmd, is_dram);
        commands_per_iteration.push_back(std::move(cmd_dispatch_dram));

        // Followed by stall to avoid RAW hazard
        HostMemDeviceCommand cmd_stall = CommandBuilder::build_dispatch_prefetch_stall();
        commands_per_iteration.push_back(std::move(cmd_stall));

        uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

        // Step 2: Paged Read of DRAM banks by prefetcher, relay data to dispatcher for linear write to L1
        HostMemDeviceCommand cmd_prefetch = CommandBuilder::build_prefetch_relay_paged<flush_prefetch_, inline_data_>(
            noc_xy, l1_addr, start_page_cmd, base_addr, page_size_bytes, pages_in_chunk);
        commands_per_iteration.push_back(std::move(cmd_prefetch));

        for (uint32_t page = 0; page < pages_in_chunk; ++page) {
            const uint32_t page_id = absolute_start_page + page;
            const uint32_t bank_id = page_id % num_banks_;
            // Add dram_data_size_words since we're reading after the pre-populated DRAM data
            uint32_t bank_offset = dram_data_size_words + page_size_words * (page_id / num_banks_);

            // Get the logical core for this bank
            const auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(bank_id);
            const CoreCoord bank_core = device_->logical_core_from_dram_channel(dram_channel);

            // Update DeviceData for paged read
            DeviceDataUpdater::update_paged_dram_read(
                worker_range, device_data, bank_core, bank_id, bank_offset, page_size_words);
        }

        // Update loop state
        remaining_bytes -= bytes_in_chunk;
        absolute_start_page += pages_in_chunk;
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

TEST_F(BasePrefetcherTestFixture, RandomTest) {
    log_info(tt::LogTest, "BasePrefetcherTestFixture - RandomTest - Test Start");
}

// TODO: This test hangs even on multi-chips, so skipping it for now
TEST_F(BasePrefetcherTestFixture, RelayLinearHTest) {
    GTEST_SKIP() << "Skipping RelayLinearHTest due to hangs that need to be resolved";
    log_info(tt::LogTest, "BasePrefetcherTestFixture - RelayLinearHTest - Test Start");

    // Check if we are physically capable of running PREFETCH_H (Multi-chip MMIO)
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Even on multi-chip, we must ensure we are testing the MMIO device which acts as PREFETCH_H
    // AND that the firmware was actually compiled/loaded as PREFETCH_H (which is automatic for MMIO servicing remote).

    // Simplest check: If we are on a single chip setup, we are definitely PREFETCH_HD.
    if (cluster.number_of_devices() == 1) {
        GTEST_SKIP()
            << "Skipping: CQ_PREFETCH_CMD_RELAY_LINEAR_H is not supported on single-chip PREFETCH_HD firmware.";
    }

    constexpr uint32_t min_read_size = 32;
    const uint32_t max_read_size =
        std::min(
            DEFAULT_SCRATCH_DB_SIZE,
            DEFAULT_PREFETCH_Q_ENTRIES * (uint32_t)sizeof(DispatchSettings::prefetch_q_entry_type)) -
        64;

    // TODO: look at parameters in run_cpp_fd2_tests.sh and make the below
    //  parameters configurable
    const uint32_t num_iterations = 1;
    const uint32_t dram_data_size_words = DRAM_DATA_SIZE_WORDS;

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = first_worker;  // {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    // const auto host_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    const uint32_t noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);

    DeviceData device_data(device_, worker_range, l1_base, dram_base_, nullptr, false, dram_data_size_words);

    std::vector<HostMemDeviceCommand> commands_per_iteration;

    uint32_t remaining_bytes = DEVICE_DATA_SIZE;

    while (remaining_bytes > 0) {
        uint32_t length = tt::align(
            std::max(min_read_size, payload_generator_->get_rand<uint32_t>(0, max_read_size - 1)), dram_alignment);
        log_info(tt::LogTest, "remaining_bytes: {}", remaining_bytes);
        if (device_data.size() * sizeof(uint32_t) + length > DEVICE_DATA_SIZE) {
            break;
        }

        // Capture address before updating device_data
        uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

        DeviceCommandCalculator calc;
        calc.add_dispatch_write_linear<false, false>(0);
        calc.add_prefetch_relay_linear();
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

        // commands_per_iteration.push_back(cmd1);

        // const uint32_t total_cmd_bytes = tt::align(sizeof(CQPrefetchCmdLarge), host_alignment);

        // // Create the HostMemDeviceCommand with pre-calculated size
        // HostMemDeviceCommand cmd2(total_cmd_bytes);

        // Create the relay linear H command
        CQPrefetchCmdLarge cmd{};
        std::memset(&cmd, 0, sizeof(CQPrefetchCmdLarge));  // Explicit zero-initialization
        cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR_H;

        // Set up the source NOC address - we'll read from DRAM where data is initialized
        // Use DRAM bank 0 for simplicity
        const uint32_t dram_bank_id = 0;
        auto dram_channel = device_->allocator()->get_dram_channel_from_bank_id(dram_bank_id);
        CoreCoord dram_logical_core = device_->logical_core_from_dram_channel(dram_channel);
        CoreCoord dram_physical_core = tt::tt_metal::MetalContext::instance()
                                           .get_cluster()
                                           .get_soc_desc(device_->id())
                                           .get_preferred_worker_core_for_dram_view(dram_channel, NOC::NOC_0);
        cmd.relay_linear_h.noc_xy_addr = device_->get_noc_unicast_encoding(NOC::NOC_0, dram_physical_core);

        [[maybe_unused]] auto offset = device_->allocator()->get_bank_offset(BufferType::DRAM, dram_bank_id);
        // Read from DRAM result data address where data is stored
        // DeviceData uses the logical coordinates as keys
        cmd.relay_linear_h.addr = dram_base_ + offset;
        cmd.relay_linear_h.length = length;
        cmd.relay_linear_h.pad1 = 0;
        cmd.relay_linear_h.pad2 = 0;

        // Update device data to simulate the data that would be written
        // For relay linear H, we're reading from DRAM which should already be populated.
        // We need to simulate writing that DRAM data to the target worker core.
        uint32_t length_words = length / sizeof(uint32_t);

        // Ensure DRAM data exists
        TT_FATAL(
            device_data.core_and_bank_present(dram_logical_core, dram_bank_id),
            "DRAM core {} bank {} not present in device data",
            dram_logical_core.str(),
            dram_bank_id);
        TT_FATAL(
            length_words <= device_data.size_at(dram_logical_core, dram_bank_id),
            "Requested length {} words exceeds available DRAM data {} words at core {} bank {}",
            length_words,
            device_data.size_at(dram_logical_core, dram_bank_id),
            dram_logical_core.str(),
            dram_bank_id);

        // Use reserve_space to properly update cmd_write_offsetB
        CQPrefetchCmdLarge* cmd_ptr = cmd1.reserve_space<CQPrefetchCmdLarge*>(sizeof(CQPrefetchCmdLarge));
        std::memcpy(cmd_ptr, &cmd, sizeof(CQPrefetchCmdLarge));

        // // Align the write offset for any padding needed
        cmd1.align_write_offset();

        commands_per_iteration.push_back(cmd1);

        DeviceDataUpdater::update_read(
            default_worker_start, device_data, dram_logical_core, dram_bank_id, 0, length_words);

        remaining_bytes -= length;
    }

    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

TEST_F(PrefetcherSmokeTestFixture, PagedReadWriteTest) {}
}  // namespace dispatcher_tests
}  // namespace tt::tt_dispatch
