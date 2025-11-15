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
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "dispatch/device_command_calculator.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"

// TODO: clean up these globals
// They are used in common.h and test_prefetcher.cpp
// In the next refactor, they will be encapsulated in a struct DispatchTestContext
// that can be passed to common.h and test fixtures.
bool use_coherent_data_g = false;  // Use sequential test data vs random
uint32_t dispatch_buffer_page_size_g =
    1 << tt::tt_metal::DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;  // Dispatch buffer page size (bytes)
uint32_t min_xfer_size_bytes_g = 16;                                     // Min transfer size for random commands
uint32_t max_xfer_size_bytes_g = 4096;                                   // Max transfer size for random commands
bool send_to_all_g = true;                                               // Send to all cores vs random subset
bool perf_test_g = false;                                                // Perf mode: use consistent sizes
uint32_t hugepage_issue_buffer_size_g;                                   // Hugepage issue buffer size (runtime)

namespace tt::tt_dispatch {
namespace dispatcher_tests {

struct LinearWriteParams {
    uint32_t transfer_size_bytes{};  // Total transfer size in bytes for the test iteration
    uint32_t num_iterations{};       // Number of iterations for the test
    uint32_t dram_data_size_words{};
    bool is_mcast{};  // Whether to use multicast or unicast
};

struct PagedWriteParams {
    uint32_t page_size{};       // Page size in bytes
    uint32_t num_pages{};       // Number of pages
    uint32_t num_iterations{};  // Number of iterations for the test
    uint32_t dram_data_size_words{};
    bool is_dram{};  // Whether to use DRAM or L1
};

struct PackedWriteParams {
    uint32_t transfer_size_bytes{};  // Total transfer size in bytes for the test iteration
    uint32_t num_iterations{};       // Number of iterations for the test
    uint32_t dram_data_size_words{};
};

// Forward declare the accessor
// This accessor class provides test access to private members
// of FDMeshCommandQueue
class FDMeshCQTestAccessor {
public:
    static tt_metal::SystemMemoryManager& sysmem(tt_metal::distributed::FDMeshCommandQueue& cq) {
        return cq.reference_sysmem_manager();
    }
};

constexpr uint32_t DEFAULT_ITERATIONS_LINEAR_WRITE = 3;
constexpr uint32_t DEFAULT_ITERATIONS_PAGED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE_LARGE = 1;
constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

class BaseDispatchTestFixture : public tt_metal::UnitMeshCQSingleCardFixture {
protected:
    // DeviceCommandCalculator for command size calculations
    DeviceCommandCalculator cmd_calc_{};

    // Common constants
    static constexpr uint32_t MAX_XFER_SIZE_16B = 4 * 1024;
    static constexpr CoreCoord default_worker_start = {0, 1};
    static constexpr uint32_t bytes_per_16B_unit = 16;

    // Common setup for all dispatch tests
    tt_metal::distributed::MeshDevice* mesh_device_ = nullptr;
    tt_metal::distributed::FDMeshCommandQueue* fdcq_ = nullptr;
    tt_metal::SystemMemoryManager* mgr_ = nullptr;
    tt_metal::distributed::MeshDevice::IDevice* device_ = nullptr;

    // HW properties
    uint32_t host_alignment_ = 0;
    uint32_t max_fetch_bytes_ = 0;

    // Knobs from globals
    bool use_coherent_data_ = false;
    bool perf_test_ = false;
    uint32_t min_xfer_size_bytes_ = 0;
    uint32_t max_xfer_size_bytes_ = 0;
    uint32_t dispatch_buffer_page_size_ = 0;
    bool send_to_all_ = false;

    // Random number generation using C++11 facilities
    std::mt19937 rng_;
    // Distributions for random number generation
    std::uniform_int_distribution<int> bool_dist{0, 1};
    std::uniform_int_distribution<uint32_t> uint32_dist{
        std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max()};

    // Helper for random number generation in a range [min, max]
    template <typename T>
    T get_rand(T min, T max) {
        static_assert(std::is_integral<T>::value, "T must be an integral type");
        std::uniform_int_distribution<T> dist(min, max);
        return dist(this->rng_);
    }

    // Helper for generating a random boolean (replaces std::rand() % 2)
    bool get_rand_bool() { return (this->bool_dist(this->rng_) != 0); }

    // Helper for generating full-range random data (replaces static_cast<uint32_t>(std::rand()))
    uint32_t get_rand_data() { return this->uint32_dist(this->rng_); }

    void SetUp() override {
        tt_metal::UnitMeshCQSingleCardFixture::SetUp();

        // Seed the PER-FIXTURE RNG. This runs for every test
        std::random_device rd;
        uint32_t seed = rd();
        this->rng_.seed(seed);
        log_info(tt::LogTest, "Random seed set to {}", seed);

        // Initialize common pointers
        this->mesh_device_ = this->devices_[0].get();
        auto& mcq = this->mesh_device_->mesh_command_queue();
        this->fdcq_ = &dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
        this->mgr_ = &FDMeshCQTestAccessor::sysmem(*this->fdcq_);
        this->device_ = this->mesh_device_->get_devices()[0];

        // Initialize common HW properties
        this->host_alignment_ = tt_metal::MetalContext::instance().hal().get_alignment(tt_metal::HalMemType::HOST);
        this->max_fetch_bytes_ = tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();

        // For now, we'll use the globals
        // TODO: remove this once test_prefetcher.cpp is refactored and common globals are removed
        this->use_coherent_data_ = use_coherent_data_g;
        this->perf_test_ = perf_test_g;
        this->min_xfer_size_bytes_ = min_xfer_size_bytes_g;
        this->max_xfer_size_bytes_ = max_xfer_size_bytes_g;
        this->dispatch_buffer_page_size_ = dispatch_buffer_page_size_g;
        this->send_to_all_ = send_to_all_g;
    }

    uint32_t get_wait_barrier_command_size() {
        this->cmd_calc_.clear();
        this->cmd_calc_.add_dispatch_wait();
        return this->cmd_calc_.write_offset_bytes();
    }

    uint64_t calculate_total_command_size(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        uint32_t num_iterations,
        bool add_barrier_per_iteration = false) {
        // Barrier wait command
        const uint32_t wait_bytes = add_barrier_per_iteration ? this->get_wait_barrier_command_size() : 0;

        uint64_t per_iter_total = 0;
        for (const auto& cmd : commands_per_iteration) {
            per_iter_total += cmd.size_bytes();
        }

        per_iter_total += wait_bytes;

        return num_iterations * per_iter_total;
    }

    // Check if the total command size fits in the issue queue
    void check_cmd_buffer_size(uint64_t total_cmd_bytes) const {
        ASSERT_LE(total_cmd_bytes, this->mgr_->get_issue_queue_limit(this->fdcq_->id()))
            << "Test requires " << total_cmd_bytes << " B, but issue queue limit is "
            << this->mgr_->get_issue_queue_limit(this->fdcq_->id()) << " B";
    }

    // Reserve the command buffer space
    void* reserve_cmd_buffer(uint64_t total_cmd_bytes) const {
        void* cmd_buffer_base = this->mgr_->issue_queue_reserve(total_cmd_bytes, this->fdcq_->id());
        return cmd_buffer_base;
    }

    // Helper function to submit commands to the issue queue
    // Returns the elapsed time in seconds
    std::chrono::duration<double> submit_commands(
        const HugepageDeviceCommand& dc, const std::vector<uint32_t>& entry_sizes) {
        // Submit commands to issue queue on the host side
        // Host side memory (hugepages) is used to store the commands
        this->mgr_->issue_queue_push_back(dc.write_offset_bytes(), this->fdcq_->id());

        // Write the commands to the device-side fetch queue, notifying the prefetcher
        // that there are new commands to fetch from the issue queue
        // Post per-chunk entries to fetch queue
        const auto start = std::chrono::steady_clock::now();
        for (const uint32_t sz : entry_sizes) {
            this->mgr_->fetch_queue_reserve_back(this->fdcq_->id());
            this->mgr_->fetch_queue_write(sz, this->fdcq_->id());
        }

        // Wait for completion
        distributed::Finish(this->mesh_device_->mesh_command_queue());
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        return elapsed;
    }

    // Helper function to validate the results
    void validate_results(
        DeviceData& device_data,
        size_t num_cores_to_log,
        std::chrono::duration<double> elapsed,
        uint32_t num_iterations) {
        // Validate results
        const bool pass = device_data.validate(this->device_);
        EXPECT_TRUE(pass) << "Dispatcher test failed validation";

        // Report performance
        if (pass) {
            const float total_words = static_cast<float>(device_data.size()) * num_iterations;
            const float bw_gbps = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

            log_info(
                LogTest,
                "BW: {:.3f} GB/s (total_words: {:.0f}, size: {:.2f} MB, iterations: {}, cores: {})",
                bw_gbps,
                total_words,
                total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
                num_iterations,
                num_cores_to_log);
        }
    }

    // Helper function to verify if the command buffer size is sufficient
    // and reserve the command buffer
    void* plan_command_buffer(uint64_t total_cmd_bytes) {
        this->check_cmd_buffer_size(total_cmd_bytes);
        return this->reserve_cmd_buffer(total_cmd_bytes);
    }

    // Helper function to write commands to the command buffer
    void write_commands_to_buffer(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        HugepageDeviceCommand& dc,
        uint32_t num_iterations,
        std::vector<uint32_t>& entry_sizes,
        bool add_barrier_per_iteration = false) {
        // Calculate the size of the wait command only if needed
        const uint32_t wait_bytes = add_barrier_per_iteration ? this->get_wait_barrier_command_size() : 0;

        // Write commands for all iterations
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            for (const auto& cmd : commands_per_iteration) {
                // Add the command data to the command buffer
                dc.add_data(cmd.data(), cmd.size_bytes(), cmd.size_bytes());
                entry_sizes.push_back(cmd.size_bytes());
            }

            // Add barrier wait command if requested
            if (add_barrier_per_iteration) {
                dc.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
                entry_sizes.push_back(wait_bytes);
            }
        }
    }

    // Helper function for execution and validation
    void execute_and_validate(
        const std::vector<uint32_t>& entry_sizes,
        const HugepageDeviceCommand& dc,
        DeviceData& device_data,
        uint64_t total_cmd_bytes,
        size_t num_cores_to_log,
        uint32_t num_iterations) {
        // Final check to ensure the command buffer is not overflowed
        ASSERT_LE(dc.write_offset_bytes(), total_cmd_bytes)
            << "Command buffer overflow: wrote " << dc.write_offset_bytes() << " bytes, reserved " << total_cmd_bytes
            << " bytes";

        // Verifies destination memory bounds
        device_data.overflow_check(this->device_);

        // Submit commands to issue queue
        auto elapsed = this->submit_commands(dc, entry_sizes);
        log_info(tt::LogTest, "Ran in {:.3f} ms (for {} iterations)", elapsed.count() * 1000.0, num_iterations);

        // Validate results
        this->validate_results(device_data, num_cores_to_log, elapsed, num_iterations);
    }

    // Helper function to execute generated commands
    // Orchestrates the command buffer planning, writing, and submission
    void execute_generated_commands(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool add_barrier_per_iteration = false) {
        // ============================================================
        // PHASE 2: Calculate total command buffer size
        // ============================================================

        const uint64_t total_cmd_bytes =
            calculate_total_command_size(commands_per_iteration, num_iterations, add_barrier_per_iteration);
        log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

        // ============================================================
        // PHASE 3: Reserve and write commands
        // ============================================================

        void* cmd_buffer_base = this->plan_command_buffer(total_cmd_bytes);
        ASSERT_TRUE(cmd_buffer_base != nullptr) << "Failed to reserve issue queue space";

        // Use DeviceCommand helper (HugepageDeviceCommand)
        HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

        // Store the size of each command entry (per-chunk)
        std::vector<uint32_t> entry_sizes;

        // Calculate the total number of entries to reserve
        // +1 for barrier wait if requested
        size_t total_num_entries =
            num_iterations * (commands_per_iteration.size() + (add_barrier_per_iteration ? 1 : 0));
        entry_sizes.reserve(total_num_entries);

        // Write commands to the command buffer
        this->write_commands_to_buffer(
            commands_per_iteration, dc, num_iterations, entry_sizes, add_barrier_per_iteration);

        // ============================================================
        // PHASE 4: Submit and execute
        // ============================================================

        this->execute_and_validate(entry_sizes, dc, device_data, total_cmd_bytes, num_cores_to_log, num_iterations);
    }
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
        this->transfer_size_bytes_ = params.transfer_size_bytes;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
        this->is_mcast_ = params.is_mcast;
    }

    // Helper function to generate HostMemDeviceCommands for linear writes
    // Encapsulates all logic for chunking, payload generation,
    // expectation model (DeviceData) updates
    std::vector<HostMemDeviceCommand> generate_linear_write_commands(
        const CoreRange& worker_range,
        uint32_t noc_xy,
        uint32_t max_payload_per_cmd_bytes,
        DeviceData& device_data  // Pass by ref to update the expectation model
    ) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = this->get_transfer_size_bytes();
        uint32_t coherent_count = 0;
        const CoreCoord first_worker = worker_range.start_coord;

        const auto chunk_size_calculator = [&](uint32_t size_bytes) -> uint32_t {
            cmd_calc_.clear();
            cmd_calc_.add_dispatch_write_linear<flush_prefetch_, inline_data_>(size_bytes);
            return cmd_calc_.write_offset_bytes();
        };

        // Relevel once for multicast before generating commands
        if (is_mcast_) {
            device_data.relevel(tt::CoreType::WORKER);
        }

        // This loop generates random-sized linear write commands until all transfer bytes are consumed
        // Each chunk's payload is clamped to max_payload_per_cmd_bytes (= max_fetch_bytes - overhead)
        while (remaining_bytes > 0) {
            // Generate random transfer size
            uint32_t xfer_size_16B = this->get_rand<uint32_t>(1, MAX_XFER_SIZE_16B - 1);
            uint32_t xfer_size_bytes = xfer_size_16B * bytes_per_16B_unit;  // Convert 16B units to bytes

            // Clamp to remaining bytes
            xfer_size_bytes = std::min(xfer_size_bytes, remaining_bytes);

            // Apply perf_test_ constraints if enabled
            if (this->perf_test_) {
                xfer_size_bytes = std::clamp(xfer_size_bytes, this->min_xfer_size_bytes_, this->max_xfer_size_bytes_);
            }

            // Clamp to max_fetch_bytes constraints
            xfer_size_bytes = std::min(xfer_size_bytes, max_payload_per_cmd_bytes);

            if (xfer_size_bytes == 0) {
                break;
            }

            // Capture address before updating device_data
            uint32_t l1_addr = device_data.get_result_data_addr(first_worker, 0);

            // Generate payload
            const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
            std::vector<uint32_t> payload;
            payload.reserve(size_words);

            for (uint32_t i = 0; i < size_words; ++i) {
                const uint32_t datum = this->use_coherent_data_ ? coherent_count++ : this->get_rand_data();
                payload.push_back(datum);
            }

            // Update expected device_data
            if (is_mcast_) {
                for (const uint32_t datum : payload) {
                    device_data.push_range(worker_range, datum, true);
                }
            } else {
                for (const uint32_t datum : payload) {
                    device_data.push_one(first_worker, 0, datum);
                }
            }

            // Relevel for next multicast command
            if (is_mcast_) {
                device_data.relevel(tt::CoreType::WORKER);
            }

            // Calculate the command size
            const uint32_t command_size_bytes = chunk_size_calculator(xfer_size_bytes);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd(command_size_bytes);

            // Add the dispatch write linear command
            cmd.add_dispatch_write_linear<flush_prefetch_, inline_data_>(
                is_mcast_ ? worker_range.size() : 0,  // num_mcast_dests
                noc_xy,                               // NOC coordinates
                l1_addr,                              // destination address
                xfer_size_bytes,                      // data size
                payload.data()                        // payload data
            );

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

    // Start offset to avoid 0x0 which matches DRAM prefill
    static constexpr uint32_t COHERENT_DATA_START_VALUE = 0x100;

protected:
    static constexpr bool hugepage_write_ = true;

public:
    void SetUp() override {
        BaseDispatchTestFixture::SetUp();

        const auto params = GetParam();
        this->page_size_ = params.page_size;
        this->num_pages_ = params.num_pages;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
        this->is_dram_ = params.is_dram;
    }

    // Helper function to generate HostMemDeviceCommands for paged writes
    // Encapsulates all logic for chunking (to respect max_fetch_bytes),
    // payload generation, expectation model (DeviceData) updates
    std::vector<HostMemDeviceCommand> generate_paged_write_commands(
        uint32_t page_size_bytes,
        uint32_t page_size_alignment_bytes,
        uint32_t num_banks,
        uint32_t max_payload_per_cmd_bytes,
        tt::CoreType core_type,
        DeviceData& device_data  // Pass by ref to update the expectation model
    ) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;
        const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);

        // Arbitrary starting value from original test to avoid 0x0
        uint32_t coherent_count = COHERENT_DATA_START_VALUE;

        // Helper lambda: Calculate the size of one chunk
        const auto chunk_size_calculator = [&](uint32_t num_pages) -> uint32_t {
            cmd_calc_.clear();
            cmd_calc_.add_dispatch_write_paged<hugepage_write_>(page_size_bytes, num_pages);
            return cmd_calc_.write_offset_bytes();
        };

        // This loop generates commands, payloads, and updates expectations all at once
        // Each iteration represents one "chunk" that fits in max_fetch_bytes
        uint32_t remaining_pages = this->num_pages_;
        uint32_t absolute_start_page = 0;

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

                // Get the logical core for this bank (same logic as original)
                CoreCoord bank_core;
                if (is_dram_) {
                    const auto dram_channel = this->device_->allocator()->get_dram_channel_from_bank_id(bank_id);
                    bank_core = this->device_->logical_core_from_dram_channel(dram_channel);
                } else {
                    bank_core = this->device_->allocator()->get_logical_core_from_bank_id(bank_id);
                }

                // Generate payload data and update expectation model
                for (uint32_t word = 0; word < page_size_words; ++word) {
                    uint32_t datum = (this->use_coherent_data_) ? (((page_id & 0xFF) << 24) | coherent_count++)
                                                                : this->get_rand_data();
                    device_data.push_one(bank_core, bank_id, datum);
                    chunk_payload.push_back(datum);
                }
                device_data.pad(bank_core, bank_id, page_size_alignment_bytes);
            }

            // Calculate base address, start page for the command
            const uint32_t bank_offset =
                tt::align(page_size_bytes, page_size_alignment_bytes) * (absolute_start_page / num_banks);
            const uint32_t base_addr = device_data.get_base_result_addr(core_type) + bank_offset;
            const uint16_t start_page_cmd = absolute_start_page % num_banks;

            // Calculate the command size
            const uint32_t command_size_bytes = chunk_size_calculator(pages_in_chunk);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd(command_size_bytes);
            cmd.add_dispatch_write_paged<hugepage_write_>(
                true,                            // flush_prefetch (inline data)
                static_cast<uint8_t>(is_dram_),  // is_dram
                start_page_cmd,                  // start_page
                base_addr,                       // base_addr
                page_size_bytes,                 // page_size
                pages_in_chunk,                  // pages
                chunk_payload.data()             // payload for this chunk
            );

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

public:
    void SetUp() override {
        BaseDispatchTestFixture::SetUp();

        const auto params = GetParam();
        this->transfer_size_bytes_ = params.transfer_size_bytes;
        this->num_iterations_ = params.num_iterations;
        this->dram_data_size_words_ = params.dram_data_size_words;
    }

    // Helper to generate payload data for a given core
    // Pass count by-reference to increment it
    // Pass core_id to use for coherent data generation
    void generate_payload(
        std::vector<uint32_t>& payload,
        uint32_t size_words,
        uint32_t& coherent_count,  // Pass by reference
        const CoreCoord& core_id)  // Pass the core to use
    {
        for (uint32_t i = 0; i < size_words; ++i) {
            const uint32_t datum = this->use_coherent_data_ ? ((core_id.x << 16) | (core_id.y << 24) | coherent_count++)
                                                            : this->get_rand_data();
            payload.push_back(datum);
        }
    }

    // Helper function to generate HostMemDeviceCommands for packed unicast writes
    // Encapsulates all logic for chunking, payload generation,
    // expectation model (DeviceData) updates
    std::vector<HostMemDeviceCommand> generate_packed_write_commands(
        const std::vector<CoreCoord>& worker_cores,
        uint32_t l1_alignment,
        uint32_t packed_write_max_unicast_sub_cmds,
        DeviceData& device_data) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = this->get_transfer_size_bytes();
        uint32_t coherent_count = 0;

        const uint32_t num_sub_cmds = static_cast<uint32_t>(worker_cores.size());

        // Build subcmds once - reused for all commands
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds;
        sub_cmds.reserve(worker_cores.size());
        for (const auto& core : worker_cores) {
            const CoreCoord virtual_core = this->device_->virtual_core_from_logical_core(core, CoreType::WORKER);
            CQDispatchWritePackedUnicastSubCmd sub_cmd{};
            sub_cmd.noc_xy_addr = this->device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, virtual_core);
            sub_cmds.push_back(sub_cmd);
        }

        const uint32_t sub_cmds_bytes =
            tt::align(worker_cores.size() * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment);

        // Helper lambda: Calculate total entry size for a command
        const auto chunk_size_calculator = [&](uint32_t payload_size_bytes, bool no_stride) -> uint32_t {
            cmd_calc_.clear();
            cmd_calc_.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_sub_cmds,        // num_sub_cmds
                payload_size_bytes,  // packed_data_sizeB
                packed_write_max_unicast_sub_cmds,
                no_stride  // no_stride
            );
            return cmd_calc_.write_offset_bytes();
        };

        // Helper lambda: Clamp xfer_size to fit within max_fetch_bytes
        auto clamp_to_max_fetch = [&](uint32_t xfer_size_bytes, bool no_stride) -> uint32_t {
            if (chunk_size_calculator(xfer_size_bytes, no_stride) <= this->max_fetch_bytes_) {
                return xfer_size_bytes;
            }

            // Linear decrement by alignment until it fits
            uint32_t result = xfer_size_bytes;
            while (result > 0 && chunk_size_calculator(result, no_stride) > this->max_fetch_bytes_) {
                result -= l1_alignment;
            }

            return result;
        };

        // Relevel once before generating commands
        device_data.relevel(tt::CoreType::WORKER);

        // Generate random-sized packed write commands until transfer_size_bytes_ is consumed
        // Each command is constrained by:
        // 1. Command entry size <= max_fetch_bytes (enforced by clamp_to_max_fetch)
        // 2. Payload <= dispatch_cb_page_size_bytes - overhead (no_stride mode only)
        while (remaining_bytes > 0) {
            // Generate random transfer size
            const uint32_t max_words_per_page = this->dispatch_buffer_page_size_ / sizeof(uint32_t);
            uint32_t xfer_size_words = this->get_rand<uint32_t>(1, max_words_per_page);
            uint32_t xfer_size_bytes = xfer_size_words * sizeof(uint32_t);

            // Clamp to remaining bytes
            xfer_size_bytes = std::min(xfer_size_bytes, remaining_bytes);

            // Apply perf_test_ constraints if enabled
            if (this->perf_test_) {
                xfer_size_bytes = std::clamp(xfer_size_bytes, this->min_xfer_size_bytes_, this->max_xfer_size_bytes_);
            }

            // Random no_stride flag
            bool no_stride = this->get_rand_bool();
            uint32_t data_copies = no_stride ? 1u : static_cast<uint32_t>(worker_cores.size());

            // Clamp for dispatch page size (no_stride mode)
            if (no_stride) {
                const uint32_t max_allowed = this->dispatch_buffer_page_size_ - sizeof(CQDispatchCmd) - sub_cmds_bytes;
                if (xfer_size_bytes > max_allowed) {
                    log_warning(tt::LogTest, "Clamping packed_write cmd w/ no_stride to fit dispatch page");
                    xfer_size_bytes = max_allowed;
                }
            }

            // Clamp to fit within max_fetch_bytes
            xfer_size_bytes = clamp_to_max_fetch(xfer_size_bytes, no_stride);

            if (xfer_size_bytes == 0) {
                break;
            }

            // Capture address before updating device_data
            uint32_t common_addr = device_data.get_result_data_addr(worker_cores[0]);

            // Generate payload
            const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
            std::vector<uint32_t> payload;
            payload.reserve(size_words);

            const CoreCoord& fw = worker_cores[0];
            // Generate payload
            this->generate_payload(payload, size_words, coherent_count, fw);

            // Update expected device_data for all cores
            for (const auto& core : worker_cores) {
                for (const uint32_t datum : payload) {
                    device_data.push_one(core, 0, datum);
                }
                device_data.pad(core, 0, l1_alignment);
            }

            // Re-relevel for next command
            device_data.relevel(tt::CoreType::WORKER);

            // Pre-calculate all sizes needed
            const uint32_t payload_size_bytes = xfer_size_bytes;
            const uint32_t data_bytes = data_copies * tt::align(payload_size_bytes, l1_alignment);
            const uint32_t payload_bytes = tt::align(sizeof(CQDispatchCmd) + sub_cmds_bytes, l1_alignment) + data_bytes;

            // Calculate the command size
            const uint32_t command_size_bytes = chunk_size_calculator(payload_size_bytes, no_stride);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd(command_size_bytes);

            // Build data_collection pointing to the payload
            std::vector<std::pair<const void*, uint32_t>> data_collection;
            const void* payload_data = payload.data();

            if (no_stride) {
                data_collection.emplace_back(payload_data, payload_size_bytes);
            } else {
                data_collection.resize(num_sub_cmds, {payload_data, payload_size_bytes});
            }

            cmd.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                0,                                          // type
                num_sub_cmds,                               // num_sub_cmds
                common_addr,                                // common_addr
                static_cast<uint16_t>(payload_size_bytes),  // packed_data_sizeB
                payload_bytes,                              // payload_sizeB
                sub_cmds,                                   // sub_cmds
                data_collection,                            // data_collection
                packed_write_max_unicast_sub_cmds,          // packed_write_max_unicast_sub_cmds
                0,                                          // offset_idx
                no_stride);                                 // no_stride

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

protected:
    // Helper function to generate HostMemDeviceCommands for packed large writes
    // Encapsulates all logic for chunking, payload generation,
    // expectation model (DeviceData) updates
    std::vector<HostMemDeviceCommand> generate_packed_large_write_commands(
        const CoreRange& worker_range, uint32_t l1_alignment, DeviceData& device_data) {
        // Generate multiple packed-large commands with random transactions
        std::vector<HostMemDeviceCommand> commands_per_iteration;
        uint32_t remaining_bytes = this->get_transfer_size_bytes();
        uint32_t coherent_count = 0;
        const CoreCoord virtual_start =
            this->device_->virtual_core_from_logical_core(worker_range.start_coord, CoreType::WORKER);
        const CoreCoord virtual_end =
            this->device_->virtual_core_from_logical_core(worker_range.end_coord, CoreType::WORKER);
        const uint32_t num_mcast_dests = (worker_range.end_coord.x - worker_range.start_coord.x + 1) *
                                         (worker_range.end_coord.y - worker_range.start_coord.y + 1);

        // Helper lambda: Calculate total command size for a multi-transaction command
        const auto calculate_command_size = [&](uint32_t num_txns, uint32_t total_payload_bytes) -> uint32_t {
            cmd_calc_.clear();
            cmd_calc_.add_dispatch_write_packed_large(num_txns, total_payload_bytes);
            return cmd_calc_.write_offset_bytes();
        };

        // Relevel once at start (all transactions target same fixed range)
        device_data.relevel(worker_range);

        // Generate commands until remaining_bytes is exhausted
        // This loop generates random-sized packed-large commands until remaining_bytes is exhausted
        while (remaining_bytes > 0) {
            // Random number of transactions per command (1-16)
            const int max_transactions = this->get_rand<int>(1, CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS);

            // These are temporary containers for building *one* multi-transaction command
            std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds;
            std::vector<std::vector<uint32_t>> payloads;
            std::vector<uint32_t> transaction_sizes;
            transaction_sizes.reserve(max_transactions);

            // Track payload size for this command
            uint32_t cumulative_payload_bytes = 0;

            // Generate random-sized transactions until remaining_bytes is exhausted or max_transactions is reached
            for (int i = 0; i < max_transactions && remaining_bytes > 0; i++) {
                // Generate a random transfer size constrained by max pages and alignment
                uint32_t xfer_size_16B = this->get_rand<uint32_t>(
                    1, (this->dispatch_buffer_page_size_ * max_pages_per_transaction / l1_alignment));
                uint32_t xfer_size_bytes = xfer_size_16B * bytes_per_16B_unit;  // Convert 16B units to bytes

                // Apply perf test constraints if enabled
                if (this->perf_test_) {
                    xfer_size_bytes =
                        std::clamp(xfer_size_bytes, this->min_xfer_size_bytes_, this->max_xfer_size_bytes_);
                }

                // Clamp to remaining space to ensure at least one transaction fits
                xfer_size_bytes = std::min(xfer_size_bytes, remaining_bytes);

                if (xfer_size_bytes == 0) {
                    break;  // No more space available
                }

                // Verify adding this transaction won't exceed max_fetch_bytes
                const uint32_t projected_cmd_size =
                    calculate_command_size(transaction_sizes.size() + 1, cumulative_payload_bytes + xfer_size_bytes);

                if (projected_cmd_size > this->max_fetch_bytes_) {
                    log_info(
                        tt::LogTest,
                        "Command would exceed max_fetch_bytes ({} > {}), finalizing with {} transactions",
                        projected_cmd_size,
                        this->max_fetch_bytes_,
                        transaction_sizes.size());
                    break;  // This transaction would make command too large
                }

                transaction_sizes.push_back(xfer_size_bytes);
                cumulative_payload_bytes += xfer_size_bytes;
                remaining_bytes -= xfer_size_bytes;
            }

            // Exit if no transactions could be generated
            if (transaction_sizes.empty()) {
                break;
            }

            // Build sub-commands and payloads for each transaction
            sub_cmds.reserve(transaction_sizes.size());
            payloads.reserve(transaction_sizes.size());

            for (const uint32_t xfer_size_bytes : transaction_sizes) {
                const uint32_t xfer_size_words = xfer_size_bytes / sizeof(uint32_t);

                // Build sub-command
                CQDispatchWritePackedLargeSubCmd sub_cmd{};
                sub_cmd.noc_xy_addr = this->device_->get_noc_multicast_encoding(
                    k_dispatch_downstream_noc, CoreRange(virtual_start, virtual_end));
                sub_cmd.addr = device_data.get_result_data_addr(worker_range.start_coord);
                sub_cmd.length = static_cast<uint16_t>(xfer_size_bytes);
                sub_cmd.num_mcast_dests = num_mcast_dests;
                sub_cmd.flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;

                sub_cmds.push_back(sub_cmd);

                // Generate random payload
                std::vector<uint32_t> payload;
                payload.reserve(xfer_size_words);

                const CoreCoord& fw = worker_range.start_coord;
                // Generate payload
                this->generate_payload(payload, xfer_size_words, coherent_count, fw);

                payloads.push_back(std::move(payload));

                // Update expected data model for all cores in range
                for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; y++) {
                    for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; x++) {
                        const CoreCoord core = {x, y};
                        for (uint32_t j = 0; j < xfer_size_words; j++) {
                            device_data.push_one(core, 0, payloads.back()[j]);
                        }
                        device_data.pad(core, 0, l1_alignment);
                    }
                }
            }

            // Calculate and validate final command size
            const uint32_t command_size_bytes = calculate_command_size(sub_cmds.size(), cumulative_payload_bytes);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd(command_size_bytes);

            // Build data spans pointing to the generated payloads
            std::vector<tt::stl::Span<const uint8_t>> data_spans;
            data_spans.reserve(payloads.size());

            for (const auto& pld : payloads) {
                data_spans.emplace_back(reinterpret_cast<const uint8_t*>(pld.data()), pld.size() * sizeof(uint32_t));
            }

            // Add write packed large command with data inlined
            cmd.add_dispatch_write_packed_large(
                CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN,
                static_cast<uint16_t>(l1_alignment),
                sub_cmds.size(),
                sub_cmds,
                data_spans,
                nullptr);

            log_info(
                tt::LogTest,
                "Generated packed-large command {} with {} transactions, {} bytes (cmd size: {})",
                commands_per_iteration.size() + 1,
                sub_cmds.size(),
                cumulative_payload_bytes,
                command_size_bytes);

            commands_per_iteration.push_back(std::move(cmd));
        }

        log_info(tt::LogTest, "Generated {} packed-large commands total", commands_per_iteration.size());

        return commands_per_iteration;
    }
};

using namespace tt::tt_metal;

// Linear Write Unicast/Multicast
TEST_P(DispatchLinearWriteTestFixture, LinearWrite) {
    log_info(tt::LogTest, "DispatchLinearWriteTestFixture - LinearWrite (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto device = this->device_;

    // Test parameters
    const uint32_t num_iterations = this->get_num_iterations();
    const uint32_t dram_data_size_words = this->get_dram_data_size_words();
    const uint32_t total_target_bytes = this->get_transfer_size_bytes();  // Total bytes to transfer per iteration
    const bool is_mcast = this->get_is_mcast();

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

    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    const uint32_t max_fetch_bytes = this->max_fetch_bytes_;

    // Calculate overhead using DeviceCommandCalculator
    cmd_calc_.clear();
    cmd_calc_.add_dispatch_write_linear<flush_prefetch_, inline_data_>(0);
    const uint32_t overhead_bytes = cmd_calc_.write_offset_bytes();

    ASSERT_GT(max_fetch_bytes, overhead_bytes)
        << "max_fetch_bytes " << max_fetch_bytes << " must be greater than overhead_bytes " << overhead_bytes;
    const uint32_t max_payload_per_cmd_bytes = max_fetch_bytes - overhead_bytes;

    // Compute NOC encoding once
    const CoreCoord first_virt_worker = device->virtual_core_from_logical_core(first_worker, CoreType::WORKER);
    uint32_t noc_xy;
    if (is_mcast) {
        const CoreCoord last_virt_worker = device->virtual_core_from_logical_core(last_worker, CoreType::WORKER);
        noc_xy = device->get_noc_multicast_encoding(
            k_dispatch_downstream_noc, CoreRange(first_virt_worker, last_virt_worker));
    } else {
        noc_xy = device->get_noc_unicast_encoding(k_dispatch_downstream_noc, first_virt_worker);
    }

    // ============================================================
    // PHASE 1: Generate random-sized linear write commands metadata
    // ============================================================

    auto commands_per_iteration =
        generate_linear_write_commands(worker_range, noc_xy, max_payload_per_cmd_bytes, device_data);

    // ============================================================
    // PHASE 2, 3, 4: Execute and Validate
    // ============================================================

    this->execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Paged Write CMD to DRAM/L1
TEST_P(DispatchPagedWriteTestFixture, PagedWrite) {
    log_info(tt::LogTest, "DispatchPagedWriteTestFixture - PagedWrite (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto device = this->device_;

    // Test parameters
    const uint32_t num_iterations = this->get_num_iterations();
    const uint32_t dram_data_size_words = this->get_dram_data_size_words();
    const uint32_t num_pages_per_cmd = this->get_num_pages();
    const uint32_t page_size_bytes_param = this->get_page_size();
    const bool is_dram = this->get_is_dram();

    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, true, dram_data_size_words);

    const auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    const uint32_t page_size_alignment_bytes = device->allocator()->get_alignment(buf_type);
    const uint32_t num_banks = device->allocator()->get_num_banks(buf_type);
    const tt::CoreType core_type = is_dram ? tt::CoreType::DRAM : tt::CoreType::WORKER;

    const uint32_t max_fetch_bytes = this->max_fetch_bytes_;

    // Generate random page size
    uint32_t xfer_size_16B = this->get_rand<uint32_t>(1, MAX_XFER_SIZE_16B - 1);
    uint32_t page_size_bytes = xfer_size_16B * bytes_per_16B_unit;  // Convert 16B units to bytes

    // Clamp by test parameters (matching page_size parameter)
    page_size_bytes = std::min(page_size_bytes, page_size_bytes_param);
    // Apply perf_test_ constraints if needed
    if (this->perf_test_) {
        page_size_bytes = std::clamp(page_size_bytes, this->min_xfer_size_bytes_, this->max_xfer_size_bytes_);
    }

    // Calculate overhead using DeviceCommandCalculator
    cmd_calc_.clear();
    cmd_calc_.add_dispatch_write_paged<hugepage_write_>(page_size_bytes, 0);  // 0 pages for overhead only
    const uint32_t overhead_bytes = cmd_calc_.write_offset_bytes();

    ASSERT_GT(max_fetch_bytes, overhead_bytes)
        << "max_fetch_bytes " << max_fetch_bytes << " must be greater than overhead_bytes " << overhead_bytes;
    const uint32_t max_payload_per_cmd_bytes = max_fetch_bytes - overhead_bytes;

    // Ensure page_size fits within max_fetch_bytes constraints
    ASSERT_LE(page_size_bytes, max_payload_per_cmd_bytes)
        << "page_size_bytes " << page_size_bytes << " must be less than max_payload_per_cmd_bytes "
        << max_payload_per_cmd_bytes;

    log_info(
        tt::LogTest,
        "Paged Write test to {} - random page_size: {} bytes, num_pages_per_cmd: {}, iterations: {}",
        is_dram ? "DRAM" : "L1",
        page_size_bytes,
        num_pages_per_cmd,
        num_iterations);

    // ============================================================
    // PHASE 1: Generate paged write command metadata
    // ============================================================

    auto commands_per_iteration = this->generate_paged_write_commands(
        page_size_bytes, page_size_alignment_bytes, num_banks, max_payload_per_cmd_bytes, core_type, device_data);

    // ============================================================
    // PHASE 2, 3, 4: Execute and Validate
    // ============================================================

    this->execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations);
}

// Packed Write Unicast
// TODO: Add multicast support
TEST_P(DispatchPackedWriteTestFixture, WritePackedUnicast) {
    log_info(tt::LogTest, "DispatchPackedWriteTestFixture - WritePackedUnicast (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto device = this->device_;

    // Test parameters
    const uint32_t num_iterations = this->get_num_iterations();
    const uint32_t dram_data_size_words = this->get_dram_data_size_words();
    const uint32_t total_target_bytes = this->get_transfer_size_bytes();

    log_info(tt::LogTest, "Target total: {} bytes, Iterations: {}", total_target_bytes, num_iterations);

    // Setup target worker cores
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t packed_write_max_unicast_sub_cmds =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;

    // Randomly pick worker cores once for all commands
    std::vector<CoreCoord> worker_cores;
    while (worker_cores.empty()) {
        for (uint32_t y = worker_range.start_coord.y; y <= worker_range.end_coord.y; ++y) {
            for (uint32_t x = worker_range.start_coord.x; x <= worker_range.end_coord.x; ++x) {
                if (this->send_to_all_ || this->get_rand_bool()) {
                    worker_cores.push_back({x, y});
                }
            }
        }
    }
    ASSERT_LE(worker_cores.size(), packed_write_max_unicast_sub_cmds);

    // ============================================================
    // PHASE 1: Generate random-sized packed write commands metadata
    // ============================================================

    auto commands_per_iteration = this->generate_packed_write_commands(
        worker_cores, l1_alignment, packed_write_max_unicast_sub_cmds, device_data);

    // ============================================================
    // PHASE 2, 3, 4: Execute and Validate
    // ============================================================

    this->execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations, true);
}

// Large Packed Write - Multiple Commands with Random Transactions
TEST_P(DispatchPackedWriteLargeTestFixture, WriteLargePackedMulticast) {
    log_info(
        tt::LogTest, "DispatchPackedWriteLargeTestFixture - WriteLargePackedMulticast (Fast Dispatch) - Test Start");

    // Get mesh device and command queue
    auto device = this->device_;

    // Test parameters
    const uint32_t num_iterations = this->get_num_iterations();
    const uint32_t dram_data_size_words = this->get_dram_data_size_words();
    const uint32_t total_target_bytes = this->get_transfer_size_bytes();

    log_info(tt::LogTest, "Max transfer: {} bytes, Iterations: {}", total_target_bytes, num_iterations);

    // Get hardware limits
    const uint32_t max_fetch_bytes = this->max_fetch_bytes_;
    log_info(tt::LogTest, "Max prefetch command size: {} bytes", max_fetch_bytes);

    // Setup worker core range (fixed - no variation)
    const CoreCoord first_worker = default_worker_start;
    const CoreCoord last_worker = {first_worker.x + 1, first_worker.y + 1};
    const CoreRange worker_range = {first_worker, last_worker};

    // Get memory base addresses
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup DeviceData for validation
    DeviceData device_data(device, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

    // Get alignment requirements
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // ============================================================
    // PHASE 1: Generate packed-large write commands metadata
    // ============================================================

    auto commands_per_iteration = this->generate_packed_large_write_commands(worker_range, l1_alignment, device_data);

    // ============================================================
    // PHASE 2, 3, 4: Execute and Validate
    // ============================================================

    this->execute_generated_commands(commands_per_iteration, device_data, worker_range.size(), num_iterations, true);
}

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchLinearWriteTestFixture,
    ::testing::Values(
        // Testcase: 49152 bytes (Unicast)
        LinearWriteParams{49152, DEFAULT_ITERATIONS_LINEAR_WRITE, DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 196608 bytes (Unicast)
        LinearWriteParams{196608, DEFAULT_ITERATIONS_LINEAR_WRITE, DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 49152 bytes (Multicast)
        LinearWriteParams{49152, DEFAULT_ITERATIONS_LINEAR_WRITE, DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 196608 bytes (Multicast)
        LinearWriteParams{196608, DEFAULT_ITERATIONS_LINEAR_WRITE, DRAM_DATA_SIZE_WORDS, true}),
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
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 512 pages x 16 bytes (L1)
        PagedWriteParams{16, 512, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 128 pages x 2048 bytes (DRAM)
        PagedWriteParams{2048, 128, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 128 pages x 2048 bytes (L1)
        PagedWriteParams{2048, 128, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 10 pages x 4128 bytes (not 4K-aligned) (DRAM)
        PagedWriteParams{4128, 10, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 13 pages x 16 bytes (arbitrary non-even numbers) (DRAM)
        PagedWriteParams{16, 13, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, true},
        // Testcase: 13 pages x 16 bytes (arbitrary non-even numbers) (L1)
        PagedWriteParams{16, 13, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, false},
        // Testcase: 100 pages x 8192 bytes (high BW) (DRAM)
        PagedWriteParams{8192, 100, DEFAULT_ITERATIONS_PAGED_WRITE, DRAM_DATA_SIZE_WORDS, true}),
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
        PackedWriteParams{786432, DEFAULT_ITERATIONS_PACKED_WRITE, DRAM_DATA_SIZE_WORDS},
        // Testcase: 819200 bytes (Unicast)
        PackedWriteParams{819200, DEFAULT_ITERATIONS_PACKED_WRITE, DRAM_DATA_SIZE_WORDS}),
    [](const testing::TestParamInfo<PackedWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_";
    });

INSTANTIATE_TEST_SUITE_P(
    DispatcherTests,
    DispatchPackedWriteLargeTestFixture,
    ::testing::Values(
        // Testcase: 40960 bytes
        PackedWriteParams{40960, DEFAULT_ITERATIONS_PACKED_WRITE_LARGE, DRAM_DATA_SIZE_WORDS},
        // Testcase: 409600 bytes
        PackedWriteParams{409600, DEFAULT_ITERATIONS_PACKED_WRITE_LARGE, DRAM_DATA_SIZE_WORDS}),
    [](const testing::TestParamInfo<PackedWriteParams>& info) {
        return std::to_string(info.param.transfer_size_bytes) + "B_" + std::to_string(info.param.num_iterations) +
               "iter_" + std::to_string(info.param.dram_data_size_words) + "words_";
    });

}  // namespace dispatcher_tests
}  // namespace tt::tt_dispatch
