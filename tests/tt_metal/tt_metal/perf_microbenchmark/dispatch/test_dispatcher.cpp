// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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

/*
 * FAST DISPATCHER MICROBENCHMARK SUITE
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

namespace tt::tt_metal::tt_dispatch_tests::dispatcher_tests {

constexpr uint32_t DEFAULT_ITERATIONS_LINEAR_WRITE = 3;
constexpr uint32_t DEFAULT_ITERATIONS_PAGED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE_LARGE = 1;

// Params that control the data volume, iteration count, and multicast/unicast
// for the linear write test
struct LinearWriteParams {
    uint32_t transfer_size_bytes{};  // Total transfer size in bytes for the test iteration
    uint32_t num_iterations{};       // Number of iterations for the test
    uint32_t dram_data_size_words{};
    bool is_mcast{};  // Whether to use multicast or unicast
};

// Params that control the data volume, iteration count, and DRAM/L1
// for the paged write test
struct PagedWriteParams {
    uint32_t page_size{};       // Page size in bytes
    uint32_t num_pages{};       // Number of pages
    uint32_t num_iterations{};  // Number of iterations for the test
    uint32_t dram_data_size_words{};
    bool is_dram{};  // Whether to use DRAM or L1
};

// Params that control the data volume, iteration count
// for the packed / large packed write test
struct PackedWriteParams {
    uint32_t transfer_size_bytes{};  // Total transfer size in bytes for the test iteration
    uint32_t num_iterations{};       // Number of iterations for the test
    uint32_t dram_data_size_words{};
};

namespace DeviceDataUpdater {

// Update Common::DeviceData for packed large write
// Populates Common::DeviceData for the packed-large multicast path
void update_packed_large_write(
    const std::vector<uint32_t>& payload,
    Common::DeviceData& device_data,
    const CoreRange& worker_range,
    uint32_t l1_alignment) {
    // Update expected data model for all cores in range
    for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; y++) {
        for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; x++) {
            const CoreCoord core = {x, y};
            for (const uint32_t datum : payload) {
                device_data.push_one(core, 0, datum);
            }
            device_data.pad(core, 0, l1_alignment);
        }
    }
}
}  // namespace DeviceDataUpdater

// Host-side helpers used by tests to emit the same CQ commands
// that dispatcher code emits. This namespace replicates the production code's command generation logic
// for testing purposes.
namespace CommandBuilder {

//  Builds a multi-transaction packed-large command
//  payload spans map 1:1 with the sub-command list
HostMemDeviceCommand build_packed_large_write_command(
    const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
    const std::vector<std::vector<uint32_t>>& payloads,
    uint32_t cumulative_payload_bytes,
    uint32_t l1_alignment) {
    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_packed_large(sub_cmds.size(), cumulative_payload_bytes);
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Build data spans pointing to the generated payloads
    std::vector<tt::stl::Span<const uint8_t>> data_spans;
    data_spans.reserve(payloads.size());

    for (const auto& payload : payloads) {
        data_spans.emplace_back(reinterpret_cast<const uint8_t*>(payload.data()), payload.size() * sizeof(uint32_t));
    }

    // Add write packed large command with data inlined
    cmd.add_dispatch_write_packed_large(
        CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN,
        static_cast<uint16_t>(l1_alignment),
        sub_cmds.size(),
        sub_cmds,
        data_spans,
        nullptr);

    return cmd;
}
}  // namespace CommandBuilder

class BaseDispatchTestFixture : public Common::BaseTestFixture {
protected:
    // Common constants
    static constexpr uint32_t MAX_XFER_SIZE_16B = 4 * 1024;  // Shouldn't exceed max_fetch_bytes_
};

class DispatchLinearWriteTestFixture : public BaseDispatchTestFixture,
                                       public ::testing::WithParamInterface<LinearWriteParams> {
    uint32_t transfer_size_bytes_{};
    uint32_t num_iterations_{};
    uint32_t dram_data_size_words_{};
    bool is_mcast_{};

protected:
    // Default values for inline data and flush prefetch for linear write commands
    static constexpr bool inline_data_ = true;
    static constexpr bool flush_prefetch_ = true;

public:
    void SetUp() override {
        BaseDispatchTestFixture::SetUp();

        const auto params = GetParam();
        transfer_size_bytes_ = params.transfer_size_bytes;
        num_iterations_ = params.num_iterations;
        dram_data_size_words_ = params.dram_data_size_words;
        is_mcast_ = params.is_mcast;
    }

    // Splits the requested transfer into randomly sized chunks that
    // respect max-fetch limits, generates payloads, and updates expected
    // results before emitting HostMemDeviceCommands
    std::vector<HostMemDeviceCommand> generate_linear_write_commands(
        const CoreRange& worker_range,
        uint32_t noc_xy,
        uint32_t max_payload_per_cmd_bytes,
        Common::DeviceData& device_data  // Pass by ref to update the expectation model
    ) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = get_transfer_size_bytes();
        const CoreCoord first_worker = worker_range.start_coord;

        // Relevel once for multicast before generating commands
        if (is_mcast_) {
            device_data.relevel(tt::CoreType::WORKER);
        }

        // Chunking logic:
        // The prefetcher has a buffer limit (max_fetch_bytes_) which restricts the size of each command
        // Thus, each chunk's payload is clamped to max_payload_per_cmd_bytes (= max_fetch_bytes_ - overhead)
        // This loop generates random-sized linear write commands until all transfer bytes are consumed
        while (remaining_bytes > 0) {
            // Generate random transfer size
            uint32_t xfer_size_bytes =
                payload_generator_->get_random_size(MAX_XFER_SIZE_16B, bytes_per_16B_unit, remaining_bytes);

            // Clamp to max_payload_per_cmd_bytes constraints
            // This ensures the command fits within the prefetcher's buffer limit
            xfer_size_bytes = std::min(xfer_size_bytes, max_payload_per_cmd_bytes);

            // Capture address before updating device_data
            uint32_t addr = device_data.get_result_data_addr(first_worker, 0);

            // Generate payload
            std::vector<uint32_t> payload = payload_generator_->generate_payload(xfer_size_bytes);

            // Update Common::DeviceData for linear write
            Common::DeviceDataUpdater::update_linear_write(payload, device_data, worker_range, is_mcast_);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd =
                Common::CommandBuilder::build_linear_write_command<flush_prefetch_, inline_data_>(
                    payload, worker_range, is_mcast_, noc_xy, addr, xfer_size_bytes);

            commands_per_iteration.push_back(std::move(cmd));
            remaining_bytes -= xfer_size_bytes;
        }

        log_info(
            tt::LogTest,
            "Generated {} linear write commands totaling {} bytes",
            commands_per_iteration.size(),
            transfer_size_bytes_ - remaining_bytes);

        return commands_per_iteration;
    }

    uint32_t get_transfer_size_bytes() const { return transfer_size_bytes_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
    bool get_is_mcast() const { return is_mcast_; }
};

// Paged Writes to L1/DRAM
class DispatchPagedWriteTestFixture : public BaseDispatchTestFixture,
                                      public ::testing::WithParamInterface<PagedWriteParams> {
    uint32_t page_size_{};
    uint32_t num_pages_{};
    uint32_t num_iterations_{};
    uint32_t dram_data_size_words_{};
    bool is_dram_{};

    // Get the logical core for this bank
    CoreCoord get_bank_core(uint32_t bank_id) const {
        // If DRAM, get the logical core from the DRAM channel
        if (is_dram_) {
            const auto dram_channel = device_->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
            return device_->logical_core_from_dram_channel(dram_channel);
        }

        // If L1, get logical core from the bank id
        return device_->allocator_impl()->get_logical_core_from_bank_id(bank_id);
    }

protected:
    static constexpr bool inline_data_ = true;

public:
    void SetUp() override {
        BaseDispatchTestFixture::SetUp();

        const auto params = GetParam();
        page_size_ = params.page_size;
        num_pages_ = params.num_pages;
        num_iterations_ = params.num_iterations;
        dram_data_size_words_ = params.dram_data_size_words;
        is_dram_ = params.is_dram;
    }

    // Tiles the requested page count into fetch-sized chunks,
    // preserving bank order, and emits one paged-write command per chunk
    std::vector<HostMemDeviceCommand> generate_paged_write_commands(
        uint32_t page_size_bytes,
        uint32_t page_size_alignment_bytes,
        uint32_t num_banks,
        uint32_t max_payload_per_cmd_bytes,
        tt::CoreType core_type,
        Common::DeviceData& device_data  // Pass by ref to update the expectation model
    ) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;
        const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);

        uint32_t remaining_pages = num_pages_;
        uint32_t absolute_start_page = 0;

        // This loop generates commands, payloads, and updates expectations all at once
        // Each iteration represents one "chunk" that fits in max_payload_per_cmd_bytes
        while (remaining_pages > 0) {
            // Calculate how many pages fit in this chunk
            const uint32_t max_pages_in_chunk = max_payload_per_cmd_bytes / page_size_bytes;
            const uint32_t pages_in_chunk = std::min(remaining_pages, max_pages_in_chunk);

            // Generate payload & update expectations *for this chunk*
            std::vector<uint32_t> chunk_payload;
            chunk_payload.reserve(pages_in_chunk * page_size_words);

            for (uint32_t page = 0; page < pages_in_chunk; ++page) {
                const uint32_t page_id = absolute_start_page + page;
                const uint32_t bank_id = page_id % num_banks;

                // Get the logical core for this bank
                CoreCoord bank_core = get_bank_core(bank_id);

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
                tt::align(page_size_bytes, page_size_alignment_bytes) * (absolute_start_page / num_banks);
            const uint32_t base_addr = device_data.get_base_result_addr(core_type) + bank_offset;
            // Calculate start page for the command
            const uint16_t start_page_cmd = absolute_start_page % num_banks;

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd = Common::CommandBuilder::build_paged_write_command<inline_data_>(
                chunk_payload, base_addr, page_size_bytes, pages_in_chunk, start_page_cmd, is_dram_);

            commands_per_iteration.push_back(std::move(cmd));

            // Update loop state
            remaining_pages -= pages_in_chunk;
            absolute_start_page += pages_in_chunk;
        }

        log_info(
            tt::LogTest,
            "Generated {} paged write command chunks for {} total pages",
            commands_per_iteration.size(),
            num_pages_);

        return commands_per_iteration;
    }

    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_num_pages() const { return num_pages_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
    bool get_is_dram() const { return is_dram_; }
};

class DispatchPackedWriteTestFixture : public BaseDispatchTestFixture,
                                       public ::testing::WithParamInterface<PackedWriteParams> {
    uint32_t transfer_size_bytes_{};
    uint32_t num_iterations_{};
    uint32_t dram_data_size_words_{};

    // Clamp xfer_size to fit within max_fetch_bytes_
    uint32_t clamp_to_max_fetch(
        uint32_t xfer_size_bytes,
        uint32_t num_sub_cmds,
        uint32_t packed_write_max_unicast_sub_cmds,
        bool no_stride,
        uint32_t l1_alignment) {
        return Common::PackedWriteUtils::clamp_to_max_fetch(
            max_fetch_bytes_,
            xfer_size_bytes,
            num_sub_cmds,
            packed_write_max_unicast_sub_cmds,
            no_stride,
            l1_alignment);
    };

public:
    void SetUp() override {
        BaseDispatchTestFixture::SetUp();

        const auto params = GetParam();
        transfer_size_bytes_ = params.transfer_size_bytes;
        num_iterations_ = params.num_iterations;
        dram_data_size_words_ = params.dram_data_size_words;
    }

    // Picks a random subset of workers, then emits packed-unicast
    // commands with or without stride,
    // clamping sizes to both max-fetch and dispatch-page limits
    std::vector<HostMemDeviceCommand> generate_packed_write_commands(
        const std::vector<CoreCoord>& worker_cores,
        uint32_t l1_alignment,
        uint32_t packed_write_max_unicast_sub_cmds,
        Common::DeviceData& device_data) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = get_transfer_size_bytes();

        const uint32_t num_sub_cmds = static_cast<uint32_t>(worker_cores.size());
        const uint32_t sub_cmds_bytes =
            tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment);

        // Build subcmds once - reused for all commands
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds =
            Common::PackedWriteUtils::build_sub_cmds(device_, worker_cores, k_dispatch_downstream_noc);

        // Relevel once before generating commands
        device_data.relevel(tt::CoreType::WORKER);

        // Generate random-sized packed write commands until transfer_size_bytes_ is consumed
        // Each command is constrained by:
        // 1. Command entry size <= max_fetch_bytes_ : the whole packet must within prefetcher buffer limits
        // 2. Payload <= dispatch_cb_page_size_bytes : when 'no_stride' mode is used (data replication),
        // the payload + header + sub-commands must fit within the dispatch buffer page size
        while (remaining_bytes > 0) {
            // Generate random transfer size
            uint32_t xfer_size_bytes =
                payload_generator_->get_random_size(dispatch_buffer_page_size_, 1, remaining_bytes);

            // Random no_stride flag
            bool no_stride = payload_generator_->get_rand_bool();

            // Clamp for dispatch page size (no_stride mode)
            if (no_stride) {
                const uint32_t max_allowed = dispatch_buffer_page_size_ - sizeof(CQDispatchCmd) - sub_cmds_bytes;
                if (xfer_size_bytes > max_allowed) {
                    log_warning(tt::LogTest, "Clamping packed_write cmd w/ no_stride to fit dispatch page");
                    xfer_size_bytes = max_allowed;
                }
            }

            // Clamp to fit within max_fetch_bytes_
            xfer_size_bytes = clamp_to_max_fetch(
                xfer_size_bytes, num_sub_cmds, packed_write_max_unicast_sub_cmds, no_stride, l1_alignment);

            const CoreCoord& fw = worker_cores[0];
            // Capture address before updating device_data
            uint32_t common_addr = device_data.get_result_data_addr(fw);

            // Generate payload
            std::vector<uint32_t> payload = payload_generator_->generate_payload_with_core(fw, xfer_size_bytes);

            // Update expected device_data for all cores
            Common::DeviceDataUpdater::update_packed_write(payload, device_data, worker_cores, l1_alignment);

            HostMemDeviceCommand cmd = Common::CommandBuilder::build_packed_write_command(
                payload, sub_cmds, common_addr, l1_alignment, packed_write_max_unicast_sub_cmds, no_stride);

            // Add command to batch
            commands_per_iteration.push_back(std::move(cmd));
            remaining_bytes -= xfer_size_bytes;
        }

        log_info(
            tt::LogTest,
            "Generated {} packed write commands totaling {} bytes",
            commands_per_iteration.size(),
            transfer_size_bytes_ - remaining_bytes);

        return commands_per_iteration;
    }

    uint32_t get_transfer_size_bytes() const { return transfer_size_bytes_; }
    uint32_t get_num_iterations() const { return num_iterations_; }
    uint32_t get_dram_data_size_words() const { return dram_data_size_words_; }
};

class DispatchPackedWriteLargeTestFixture : public DispatchPackedWriteTestFixture {
    static constexpr uint32_t max_pages_per_transaction = 4;

    struct TransactionBatch {
        std::vector<uint32_t> sizes;
        uint32_t total_payload_bytes;
    };

    // Helper function to generate random-sized transactions
    // until remaining_bytes is exhausted or max_transactions is reached
    // Packed large is used when sub-commands exceed the limits of packed write
    // Packed large supports large payloads by splitting them into multiple transactions
    // with a single command
    TransactionBatch generate_packed_large_transactions(
        uint32_t& remaining_bytes, uint32_t max_transactions, uint32_t l1_alignment) {
        std::vector<uint32_t> transaction_sizes;
        transaction_sizes.reserve(max_transactions);

        // Track payload size for this command
        uint32_t cumulative_payload_bytes = 0;

        for (int i = 0; i < max_transactions && remaining_bytes > 0; i++) {
            // Generate a random transfer size
            // We're first converting the max payload allowed by the packed large format into alignment units
            // get_random_size will clamp the size to remaining_bytes
            uint32_t max_allowed = dispatch_buffer_page_size_ * max_pages_per_transaction / l1_alignment;
            uint32_t xfer_size_bytes =
                payload_generator_->get_random_size(max_allowed, bytes_per_16B_unit, remaining_bytes);

            // Verify adding this transaction won't exceed max_fetch_bytes_
            // Use a projected command size to see if we would exceed the max_fetch_bytes_
            // We speculatively calculate the command size to ensure we don't overflow
            DeviceCommandCalculator cmd_calc;
            cmd_calc.add_dispatch_write_packed_large(
                transaction_sizes.size() + 1, cumulative_payload_bytes + xfer_size_bytes);
            const uint32_t projected_cmd_size = cmd_calc.write_offset_bytes();

            if (projected_cmd_size > max_fetch_bytes_) {
                log_info(
                    tt::LogTest,
                    "Command would exceed max_fetch_bytes ({} > {}), finalizing with {} transactions",
                    projected_cmd_size,
                    max_fetch_bytes_,
                    transaction_sizes.size());
                break;  // This transaction would make command too large
            }

            transaction_sizes.push_back(xfer_size_bytes);
            cumulative_payload_bytes += xfer_size_bytes;
            remaining_bytes -= xfer_size_bytes;
        }

        return TransactionBatch{transaction_sizes, cumulative_payload_bytes};
    }

    void build_sub_cmd(
        std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
        uint32_t xfer_size_bytes,
        uint32_t addr,
        const CoreCoord& /*worker_coord*/,
        const CoreCoord& virtual_start,
        const CoreCoord& virtual_end,
        uint32_t num_mcast_dests) {
        // Build sub-command
        CQDispatchWritePackedLargeSubCmd sub_cmd{};
        sub_cmd.noc_xy_addr =
            device_->get_noc_multicast_encoding(k_dispatch_downstream_noc, CoreRange(virtual_start, virtual_end));
        sub_cmd.addr = addr;
        sub_cmd.length_minus1 = static_cast<uint16_t>(xfer_size_bytes) - 1;
        sub_cmd.num_mcast_dests = num_mcast_dests;
        sub_cmd.flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;

        sub_cmds.push_back(sub_cmd);
    }

protected:
    // Builds multi-transaction packed-large commands by sampling random transaction
    // sizes until max-fetch would be exceeded, updating Common::DeviceData for every multicast target
    std::vector<HostMemDeviceCommand> generate_packed_large_write_commands(
        const CoreRange& worker_range, uint32_t l1_alignment, Common::DeviceData& device_data) {
        // Generate multiple packed-large commands with random transactions
        std::vector<HostMemDeviceCommand> commands_per_iteration;
        uint32_t remaining_bytes = get_transfer_size_bytes();

        const CoreCoord virtual_start =
            device_->virtual_core_from_logical_core(worker_range.start_coord, CoreType::WORKER);
        const CoreCoord virtual_end = device_->virtual_core_from_logical_core(worker_range.end_coord, CoreType::WORKER);
        const uint32_t num_mcast_dests = (worker_range.end_coord.x - worker_range.start_coord.x + 1) *
                                         (worker_range.end_coord.y - worker_range.start_coord.y + 1);

        // Relevel once at start (all transactions target same fixed range)
        device_data.relevel(worker_range);

        // This loop generates random-sized packed-large commands until remaining_bytes is exhausted
        while (remaining_bytes > 0) {
            // Random number of transactions per command (1-16)
            const int max_transactions =
                payload_generator_->get_rand<int>(1, CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS);

            // These are temporary containers for building one multi-transaction command
            std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds;
            std::vector<std::vector<uint32_t>> payloads;

            TransactionBatch transaction_batch =
                generate_packed_large_transactions(remaining_bytes, max_transactions, l1_alignment);
            // Exit if no transactions could be generated
            if (transaction_batch.sizes.empty()) {
                break;
            }

            // Build sub-commands and payloads for each transaction
            sub_cmds.reserve(transaction_batch.sizes.size());
            payloads.reserve(transaction_batch.sizes.size());

            for (const uint32_t xfer_size_bytes : transaction_batch.sizes) {
                const CoreCoord& fw = worker_range.start_coord;
                uint32_t addr = device_data.get_result_data_addr(fw);

                // Build sub-command
                build_sub_cmd(sub_cmds, xfer_size_bytes, addr, fw, virtual_start, virtual_end, num_mcast_dests);

                // Generate payload
                std::vector<uint32_t> payload = payload_generator_->generate_payload_with_core(fw, xfer_size_bytes);

                // Update expected data model for all cores in range
                DeviceDataUpdater::update_packed_large_write(payload, device_data, worker_range, l1_alignment);

                payloads.push_back(std::move(payload));
            }

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd = CommandBuilder::build_packed_large_write_command(
                sub_cmds, payloads, transaction_batch.total_payload_bytes, l1_alignment);

            log_info(
                tt::LogTest,
                "Generated packed-large command {} with {} transactions, {} bytes",
                commands_per_iteration.size(),
                sub_cmds.size(),
                transaction_batch.total_payload_bytes);

            commands_per_iteration.push_back(std::move(cmd));
        }

        log_info(tt::LogTest, "Generated {} packed-large commands total", commands_per_iteration.size());

        return commands_per_iteration;
    }
};

// Linear Write Unicast/Multicast
TEST_P(DispatchLinearWriteTestFixture, LinearWrite) {
    log_info(tt::LogTest, "DispatchLinearWriteTestFixture - LinearWrite (Fast Dispatch) - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t total_target_bytes = get_transfer_size_bytes();  // Total bytes to transfer per iteration
    const bool is_mcast = get_is_mcast();

    ASSERT_EQ(total_target_bytes % 16, 0) << "Require 16B alignment for write payload";

    log_info(
        tt::LogTest,
        "Target total: {} bytes, Iterations: {}, Multicast: {}",
        total_target_bytes,
        num_iterations,
        is_mcast);

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    CoreCoord last_worker = first_worker;
    if (is_mcast) {
        last_worker = {first_worker.x + 1, first_worker.y + 1};
    }
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words, cfg_);

    // Calculate the overhead for a linear write command using DeviceCommandCalculator
    // Substracting the overhead from max_fetch_bytes_ gives the max allowed payload size per command
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_linear<flush_prefetch_, inline_data_>(0);
    const uint32_t overhead_bytes = cmd_calc.write_offset_bytes();
    const uint32_t max_payload_per_cmd_bytes = max_fetch_bytes_ - overhead_bytes;

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device_->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy;
    if (is_mcast) {
        const CoreCoord last_virt_worker = device_->virtual_core_from_logical_core(last_worker, CoreType::WORKER);
        noc_xy = device_->get_noc_multicast_encoding(
            k_dispatch_downstream_noc, CoreRange(first_virt_worker, last_virt_worker));
    } else {
        noc_xy = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
    }

    // PHASE 1: Generate random-sized linear write commands metadata
    auto commands_per_iteration =
        generate_linear_write_commands(worker_range, noc_xy, max_payload_per_cmd_bytes, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Paged Write CMD to DRAM/L1
TEST_P(DispatchPagedWriteTestFixture, PagedWrite) {
    log_info(tt::LogTest, "DispatchPagedWriteTestFixture - PagedWrite (Fast Dispatch) - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t num_pages_per_cmd = get_num_pages();
    const uint32_t page_size_bytes_param = get_page_size();
    const bool is_dram = get_is_dram();

    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base, nullptr, true, dram_data_size_words, cfg_);

    const auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    const uint32_t page_size_alignment_bytes = device_->allocator_impl()->get_alignment(buf_type);
    const uint32_t num_banks = device_->allocator_impl()->get_num_banks(buf_type);
    const tt::CoreType core_type = is_dram ? tt::CoreType::DRAM : tt::CoreType::WORKER;

    // Generate random page size
    uint32_t max_allowed = MAX_XFER_SIZE_16B - 1;
    uint32_t page_size_bytes =
        payload_generator_->get_random_size(max_allowed, bytes_per_16B_unit, page_size_bytes_param);

    // Calculate overhead using DeviceCommandCalculator in CommandSizeHelper
    // 0 pages for overhead only
    // Substracting the overhead from max_fetch_bytes_ gives the max allowed payload size per command
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_paged<inline_data_>(page_size_bytes, 0);
    const uint32_t overhead_bytes = cmd_calc.write_offset_bytes();
    const uint32_t max_payload_per_cmd_bytes = max_fetch_bytes_ - overhead_bytes;

    log_info(
        tt::LogTest,
        "Paged Write test to {} - random page_size: {} bytes, num_pages_per_cmd: {}, iterations: {}",
        is_dram ? "DRAM" : "L1",
        page_size_bytes,
        num_pages_per_cmd,
        num_iterations);

    // PHASE 1: Generate paged write command metadata
    auto commands_per_iteration = generate_paged_write_commands(
        page_size_bytes, page_size_alignment_bytes, num_banks, max_payload_per_cmd_bytes, core_type, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Packed Write Unicast
// TODO: Add multicast support
TEST_P(DispatchPackedWriteTestFixture, WritePackedUnicast) {
    log_info(tt::LogTest, "DispatchPackedWriteTestFixture - WritePackedUnicast (Fast Dispatch) - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t total_target_bytes = get_transfer_size_bytes();

    log_info(tt::LogTest, "Target total: {} bytes, Iterations: {}", total_target_bytes, num_iterations);

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);

    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words, cfg_);

    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t packed_write_max_unicast_sub_cmds =
        device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;

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

    ASSERT_LE(worker_cores.size(), packed_write_max_unicast_sub_cmds);

    // PHASE 1: Generate random-sized packed write commands metadata
    auto commands_per_iteration =
        generate_packed_write_commands(worker_cores, l1_alignment, packed_write_max_unicast_sub_cmds, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Large Packed Write - Multiple Commands with Random Transactions
TEST_P(DispatchPackedWriteLargeTestFixture, WriteLargePackedMulticast) {
    log_info(
        tt::LogTest, "DispatchPackedWriteLargeTestFixture - WriteLargePackedMulticast (Fast Dispatch) - Test Start");

    // Test parameters
    const uint32_t num_iterations = get_num_iterations();
    const uint32_t dram_data_size_words = get_dram_data_size_words();
    const uint32_t total_target_bytes = get_transfer_size_bytes();

    log_info(tt::LogTest, "Max transfer: {} bytes, Iterations: {}", total_target_bytes, num_iterations);

    // Setup worker core range (fixed - no variation)
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    // Get memory base addresses
    const uint32_t l1_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator_impl()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup Common::DeviceData for validation
    Common::DeviceData device_data(
        device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words, cfg_);

    // Get alignment requirements
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // PHASE 1: Generate packed-large write commands metadata
    auto commands_per_iteration = generate_packed_large_write_commands(worker_range, l1_alignment, device_data);

    // PHASE 2, 3, 4: Execute and Validate
    execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchLinearWriteTestFixture,
    ::testing::Values(
        // Testcase: 49152 bytes (Unicast)
        LinearWriteParams{49152, DEFAULT_ITERATIONS_LINEAR_WRITE, Common::DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 196608 bytes (Unicast)
        LinearWriteParams{196608, DEFAULT_ITERATIONS_LINEAR_WRITE, Common::DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 49152 bytes (Multicast)
        LinearWriteParams{49152, DEFAULT_ITERATIONS_LINEAR_WRITE, Common::DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 196608 bytes (Multicast)
        LinearWriteParams{196608, DEFAULT_ITERATIONS_LINEAR_WRITE, Common::DRAM_DATA_SIZE_WORDS, true}),
    [](const testing::TestParamInfo<LinearWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_" +
               (info.param.is_mcast ? "mcast" : "unicast");
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPagedWriteTestFixture,
    ::testing::Values(
        // Testcase: 512 pages x 16 bytes (DRAM)
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 512 pages x 16 bytes (L1)
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 128 pages x 2048 bytes (DRAM)
        PagedWriteParams{2048, 128, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 128 pages x 2048 bytes (L1)
        PagedWriteParams{2048, 128, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 10 pages x 4128 bytes (not 4K-aligned) (DRAM)
        PagedWriteParams{4128, 10, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 13 pages x 16 bytes (arbitrary non-even numbers) (DRAM)
        PagedWriteParams{16, 13, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 13 pages x 16 bytes (arbitrary non-even numbers) (L1)
        PagedWriteParams{16, 13, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 100 pages x 8192 bytes (high BW) (DRAM)
        PagedWriteParams{8192, 100, DEFAULT_ITERATIONS_PAGED_WRITE, Common::DRAM_DATA_SIZE_WORDS, true}),
    [](const testing::TestParamInfo<PagedWriteParams>& info) {
        std::stringstream ss;
        ss << "page_size" << info.param.page_size << "_np" << info.param.num_pages << "_iter"
           << info.param.num_iterations << "_" << (info.param.is_dram ? "DRAM" : "L1");
        return ss.str();
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPackedWriteTestFixture,
    ::testing::Values(
        // Testcase: 786432 bytes (Unicast)
        PackedWriteParams{786432, DEFAULT_ITERATIONS_PACKED_WRITE, Common::DRAM_DATA_SIZE_WORDS},
        // Testcase: 819200 bytes (Unicast)
        PackedWriteParams{819200, DEFAULT_ITERATIONS_PACKED_WRITE, Common::DRAM_DATA_SIZE_WORDS}),
    [](const testing::TestParamInfo<PackedWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_";
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPackedWriteLargeTestFixture,
    ::testing::Values(
        // Testcase: 40960 bytes
        PackedWriteParams{40960, DEFAULT_ITERATIONS_PACKED_WRITE_LARGE, Common::DRAM_DATA_SIZE_WORDS},
        // Testcase: 409600 bytes
        PackedWriteParams{409600, DEFAULT_ITERATIONS_PACKED_WRITE_LARGE, Common::DRAM_DATA_SIZE_WORDS}),
    [](const testing::TestParamInfo<PackedWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_";
    });

}  // namespace tt::tt_metal::tt_dispatch_tests::dispatcher_tests
