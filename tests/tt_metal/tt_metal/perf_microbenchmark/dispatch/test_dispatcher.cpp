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
#include <impl/dispatch/dispatch_mem_map.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "dispatch/device_command_calculator.hpp"
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
 * 2. Shadow: Update `DeviceData` (the expectation model) to reflect what *should* happen.
 * 3. Build: Use `DeviceCommand` and `DeviceCommandCalculator` to construct binary
 *    command packets (HostMemDeviceCommand) exactly as the runtime would.
 * 4. Execute and Validate: Push these raw commands directly into the Issue Queue and notify the hardware.
 *    Read back device memory and compare against `DeviceData` to validate the correctness of the command execution.
 *
 * Key Concepts:
 * - Issue Queue: Host-resident ring buffer where commands are written.
 * - Fetch Queue: Device-resident ring buffer (pointers) telling the Prefetcher where to look.
 * - Prefetcher: Kernel that pulls commands from Host/DRAM and relays them.
 * - Dispatcher: Kernel that parses commands and issues writes/signals to Worker cores.
 */

// Temporary globals shared with legacy test_prefetcher.cpp and common.h
// In the refactor of test_prefetcher.cpp, they will be encapsulated in a struct DispatchTestContext
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

constexpr uint32_t DEFAULT_ITERATIONS_LINEAR_WRITE = 3;
constexpr uint32_t DEFAULT_ITERATIONS_PAGED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE = 1;
constexpr uint32_t DEFAULT_ITERATIONS_PACKED_WRITE_LARGE = 1;
constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

// Forward declare the accessor
// Exposes the internal system memory manager of the FDMeshCommandQueue
class FDMeshCQTestAccessor {
public:
    static tt_metal::SystemMemoryManager& sysmem(tt_metal::distributed::FDMeshCommandQueue& cq) {
        return cq.reference_sysmem_manager();
    }
};

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

// This will be ported to common.h when test_prefetcher.cpp is refactored
namespace DeviceDataUpdater {

// Update DeviceData for linear write
// Mirrors a dispatcher linear-write transaction into the DeviceData expectation model
void update_linear_write(
    const std::vector<uint32_t>& payload, DeviceData& device_data, const CoreRange& worker_range, bool is_mcast) {
    // Update expected device_data
    if (is_mcast) {
        for (const uint32_t datum : payload) {
            device_data.push_range(worker_range, datum, true);
        }
    } else {
        for (const uint32_t datum : payload) {
            device_data.push_one(worker_range.start_coord, 0, datum);
        }
    }
    // Relevel for next multicast command
    if (is_mcast) {
        device_data.relevel(tt::CoreType::WORKER);
    }
}

// Update DeviceData for paged write
// Tracks page-wise writes so validate() can check DRAM/L1 bank contents after the test runs
void update_paged_write(
    const std::vector<uint32_t>& payload,
    DeviceData& device_data,
    const CoreCoord& bank_core,
    uint32_t bank_id,
    uint32_t page_alignment) {
    for (const uint32_t datum : payload) {
        device_data.push_one(bank_core, bank_id, datum);
    }
    device_data.pad(bank_core, bank_id, page_alignment);
}

// Update DeviceData for packed write
// Applies packed write payloads to every selected worker
void update_packed_write(
    const std::vector<uint32_t>& payload,
    DeviceData& device_data,
    const std::vector<CoreCoord>& worker_cores,
    uint32_t l1_alignment) {
    // Update expected device_data for all cores
    for (const auto& core : worker_cores) {
        for (const uint32_t datum : payload) {
            device_data.push_one(core, 0, datum);
        }
        device_data.pad(core, 0, l1_alignment);
    }

    // Re-relevel for next command
    device_data.relevel(tt::CoreType::WORKER);
}

// Update DeviceData for packed large write
// Populates DeviceData for the packed-large multicast path
void update_packed_large_write(
    const std::vector<uint32_t>& payload,
    DeviceData& device_data,
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
};  // namespace DeviceDataUpdater

// Host-side helpers used by tests to emit the same CQ commands
// that dispatcher code emits. This namespace replicates the production code's command generation logic
// for testing purposes.
// This will be ported to common.h when test_prefetcher.cpp is refactored
namespace CommandBuilder {

// Emits a single linear write, optionally multicast, with inline data
template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_linear_write_command(
    const std::vector<uint32_t>& payload,
    const CoreRange& worker_range,
    bool is_mcast,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t xfer_size_bytes) {
    // Calculate the command size using DeviceCommandCalculator
    // Pre-calculate the exact size to allocate correct amount of memory in HostMemDeviceCommand buffer
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_linear<flush_prefetch, inline_data>(xfer_size_bytes);
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Add the dispatch write linear command
    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        is_mcast ? worker_range.size() : 0,  // num_mcast_dests
        noc_xy,                              // NOC coordinates
        addr,                                // destination address
        xfer_size_bytes,                     // data size
        payload.data()                       // payload data
    );

    return cmd;
}

// Emits a paged write (DRAM or L1) chunk
// payload is already stitched together for all pages in the chunk
template <bool hugepage_write>
HostMemDeviceCommand build_paged_write_command(
    const std::vector<uint32_t>& payload,
    uint32_t base_addr,
    uint32_t page_size_bytes,
    uint32_t pages_in_chunk,
    uint16_t start_page_cmd,
    bool is_dram) {
    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_paged<hugepage_write>(page_size_bytes, pages_in_chunk);
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Add the dispatch write paged command
    cmd.add_dispatch_write_paged<hugepage_write>(
        true,                           // flush_prefetch (inline data)
        static_cast<uint8_t>(is_dram),  // is_dram
        start_page_cmd,                 // start_page
        base_addr,                      // base_addr
        page_size_bytes,                // page_size
        pages_in_chunk,                 // pages
        payload.data()                  // payload for this chunk
    );

    return cmd;
}

// Serializes a packed-unicast command including sub-command table
// and optional replicated payloads when stride is enabled
HostMemDeviceCommand build_packed_write_command(
    const std::vector<uint32_t>& payload,
    const std::vector<CQDispatchWritePackedUnicastSubCmd>& sub_cmds,
    uint32_t common_addr,
    uint32_t l1_alignment,
    uint32_t packed_write_max_unicast_sub_cmds,
    bool no_stride) {
    const uint32_t num_sub_cmds = static_cast<uint32_t>(sub_cmds.size());
    const uint32_t sub_cmds_bytes = tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment);
    uint32_t num_data_copies = no_stride ? 1u : static_cast<uint32_t>(num_sub_cmds);

    // Pre-calculate all sizes needed
    const uint32_t payload_size_bytes = payload.size() * sizeof(uint32_t);
    const uint32_t data_bytes = num_data_copies * tt::align(payload_size_bytes, l1_alignment);
    const uint32_t payload_bytes = tt::align(sizeof(CQDispatchCmd) + sub_cmds_bytes, l1_alignment) + data_bytes;

    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_sub_cmds,        // num_sub_cmds
        payload_size_bytes,  // packed_data_sizeB
        packed_write_max_unicast_sub_cmds,
        no_stride  // no_stride
    );
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Build data_collection pointing to the payload
    std::vector<std::pair<const void*, uint32_t>> data_collection;
    const void* payload_data = payload.data();

    if (no_stride) {
        data_collection.emplace_back(payload_data, payload_size_bytes);
    } else {
        data_collection.resize(num_sub_cmds, {payload_data, payload_size_bytes});
    }

    // Add the dispatch write packed command
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

    return cmd;
}

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
};  // namespace CommandBuilder

// This will be ported to common.h when test_prefetcher.cpp is refactored
// DispatchPayloadGenerator is used to generate payloads for the tests
class DispatchPayloadGenerator {
public:
    struct Config {
        bool use_coherent_data = false;
        uint32_t coherent_start_val = COHERENT_DATA_START_VALUE;
        uint32_t seed = 0;

        // Perf test configuration
        bool perf_test = false;
        uint32_t min_xfer_size_bytes = 0;
        uint32_t max_xfer_size_bytes = 0;
    };

    DispatchPayloadGenerator(const Config& cfg) : config_(cfg), coherent_count_(cfg.coherent_start_val) {
        if (config_.seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        } else {
            rng_.seed(config_.seed);
        }
    }

    // Getter to log the seed used
    uint32_t get_seed() const { return config_.seed; }

    // Helper for random number generation in a range [min, max]
    template <typename T>
    T get_rand(T min, T max) {
        static_assert(std::is_integral<T>::value, "T must be an integral type");
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }

    // Helper for generating a random boolean (replaces std::rand() % 2)
    bool get_rand_bool() { return (bool_dist(rng_) != 0); }

    // Generates either deterministic (coherent) or random 32-bit words
    // for the requested byte count
    // In coherent mode, the counter is incremented so validation knows
    // the exact pattern
    std::vector<uint32_t> generate_payload(uint32_t xfer_size_bytes) {
        const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
        std::vector<uint32_t> payload;
        payload.reserve(size_words);

        for (uint32_t i = 0; i < size_words; ++i) {
            const uint32_t datum = config_.use_coherent_data ? coherent_count_++ : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Generate payload with page id
    // Pass page_id to use for coherent data generation
    std::vector<uint32_t> generate_payload_with_page_id(uint32_t page_size_words, uint32_t page_id) {
        std::vector<uint32_t> payload;
        payload.reserve(page_size_words);

        for (uint32_t i = 0; i < page_size_words; ++i) {
            const uint32_t datum = config_.use_coherent_data
                                       ? (((page_id & 0xFF) << 24) | (coherent_count_++ & 0xFFFFFF))
                                       : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Helper to generate payload data for a given core
    // Pass core_id to use for coherent data generation
    std::vector<uint32_t> generate_payload_with_core(
        const CoreCoord& core_id,  // Pass the core to use
        uint32_t xfer_size_bytes) {
        const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
        std::vector<uint32_t> payload;
        payload.reserve(size_words);

        for (uint32_t i = 0; i < size_words; ++i) {
            const uint32_t datum =
                config_.use_coherent_data
                    ? (((core_id.x & 0xFF) << 16) | ((core_id.y & 0xFF) << 24) | (coherent_count_++ & 0xFFFF))
                    : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Chooses a payload size in 16B units, respecting perf mode clamps and remaining budget
    uint32_t get_random_size(uint32_t max_allowed, uint32_t bytes_per_unit, uint32_t remaining_bytes) {
        // Generate random transfer size
        std::uniform_int_distribution<uint32_t> dist(1, max_allowed);
        uint32_t xfer_size_16B = dist(rng_);
        uint32_t xfer_size_bytes = xfer_size_16B * bytes_per_unit;  // Convert 16B units to bytes

        // Clamp to remaining bytes
        xfer_size_bytes = std::min(xfer_size_bytes, remaining_bytes);

        // Apply perf_test_ constraints if enabled
        if (config_.perf_test) {
            xfer_size_bytes = std::clamp(xfer_size_bytes, config_.min_xfer_size_bytes, config_.max_xfer_size_bytes);
        }

        return xfer_size_bytes;
    }

private:
    // Start offset to avoid 0x0 which matches DRAM prefill
    static constexpr uint32_t COHERENT_DATA_START_VALUE = 0x100;
    Config config_{};
    uint32_t coherent_count_ = COHERENT_DATA_START_VALUE;

    // Random number generation
    std::mt19937 rng_;
    // Distributions for random number generation
    std::uniform_int_distribution<int> bool_dist{0, 1};
    std::uniform_int_distribution<uint32_t> uint32_dist{
        std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max()};
};

class BaseDispatchTestFixture : public tt_metal::UnitMeshCQSingleCardFixture {
protected:
    // DispatchPayloadGenerator for generating payloads
    std::unique_ptr<DispatchPayloadGenerator> payload_generator_;

    // Common constants
    static constexpr uint32_t MAX_XFER_SIZE_16B = 4 * 1024;  // Shouldn't exceed max_fetch_bytes_
    static constexpr CoreCoord default_worker_start = {0, 1};
    static constexpr uint32_t bytes_per_16B_unit = 16;  // conversion factor to convert 16-byte "chunks" to bytes

    // Common setup for all dispatch tests
    // Provides shared wiring for mesh device access,
    // and command-buffer helpers so derived fixtures
    // only implement workload-specific planning
    tt_metal::distributed::MeshDevice* mesh_device_ = nullptr;
    tt_metal::distributed::FDMeshCommandQueue* fdcq_ = nullptr;
    tt_metal::SystemMemoryManager* mgr_ = nullptr;
    tt_metal::distributed::MeshDevice::IDevice* device_ = nullptr;

    // HW properties
    uint32_t host_alignment_ = 0;
    uint32_t max_fetch_bytes_ = 0;

    // Knobs from globals
    uint32_t dispatch_buffer_page_size_ = 0;
    bool send_to_all_ = false;

    void SetUp() override {
        tt_metal::UnitMeshCQSingleCardFixture::SetUp();

        // Setup Config
        DispatchPayloadGenerator::Config cfg;
        cfg.use_coherent_data = use_coherent_data_g;  // derived from globals
        cfg.perf_test = perf_test_g;
        cfg.min_xfer_size_bytes = min_xfer_size_bytes_g;
        cfg.max_xfer_size_bytes = max_xfer_size_bytes_g;

        // Handle Seeding
        std::random_device rd;
        cfg.seed = rd();

        // Initialize Generator
        payload_generator_ = std::make_unique<DispatchPayloadGenerator>(cfg);
        log_info(tt::LogTest, "Random seed set to {}", cfg.seed);

        // These are used for test logic (loops, alignment, etc.) rather than generation
        dispatch_buffer_page_size_ = dispatch_buffer_page_size_g;
        send_to_all_ = send_to_all_g;

        // Initialize common pointers
        mesh_device_ = devices_[0].get();
        auto& mcq = mesh_device_->mesh_command_queue();
        fdcq_ = &dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
        mgr_ = &FDMeshCQTestAccessor::sysmem(*fdcq_);
        device_ = mesh_device_->get_devices()[0];

        // Initialize common HW properties
        host_alignment_ = tt_metal::MetalContext::instance().hal().get_alignment(tt_metal::HalMemType::HOST);
        max_fetch_bytes_ = tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    }

    // Helper function to report performance
    void report_performance(
        DeviceData& device_data,
        size_t num_cores_to_log,
        std::chrono::duration<double> elapsed,
        uint32_t num_iterations) {
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

    // Helper function to execute generated commands
    // Orchestrates the command buffer reservation, writing, and submission
    void execute_generated_commands(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations) {
        // PHASE 2: Calculate total command buffer size
        uint64_t per_iter_total = 0;
        for (const auto& cmd : commands_per_iteration) {
            per_iter_total += cmd.size_bytes();
        }

        const uint64_t total_cmd_bytes = num_iterations * per_iter_total;
        log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

        // PHASE 3: Reserve and write commands
        // Reserve a continuous block in the system memory issue queue for all commands across all iterations
        // This memory is mapped and visible to the device's prefetcher kernel
        void* cmd_buffer_base = mgr_->issue_queue_reserve(total_cmd_bytes, fdcq_->id());

        // Use DeviceCommand helper (HugepageDeviceCommand) to write to the issue queue memory
        // Two stage command construction:
        // 1. Staging (HostMemDeviceCommand):
        //    - commands_per_iteration: vector of HostMemDeviceCommand objects which holds a deep copy
        //      of each command header + payload assembled offline without holding issue queue space
        // 2. Writing (HugepageDeviceCommand):
        //    - wraps a pointer that points directly to the issue queue memory (cmd_buffer_base)
        //    - the loop below copies staged commands into the issue queue memory
        HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

        // Store the size of each command entry (per-chunk)
        std::vector<uint32_t> entry_sizes;

        // Calculate the total number of entries to reserve
        size_t total_num_entries = num_iterations * commands_per_iteration.size();
        entry_sizes.reserve(total_num_entries);

        // Write commands to the command buffer for all iterations
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            for (const auto& cmd : commands_per_iteration) {
                // Add the command data to the command buffer
                dc.add_data(cmd.data(), cmd.size_bytes(), cmd.size_bytes());
                entry_sizes.push_back(cmd.size_bytes());
            }
        }

        // Add barrier wait command after all commands across all iterations
        // Helpful to ensure all commands are completed flush before terminating
        // Without this, there can be occasional timeouts in MetalContext::initialize_and_launch_firmware()
        // between test fixtures possibly because the previously issued commands
        // are not completed before next firmware launch
        DeviceCommandCalculator cmd_calc;
        cmd_calc.add_dispatch_wait();
        HostMemDeviceCommand cmd(cmd_calc.write_offset_bytes());
        cmd.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
        dc.add_data(cmd.data(), cmd.size_bytes(), cmd.size_bytes());
        entry_sizes.push_back(cmd.size_bytes());

        // Verifies destination memory bounds
        device_data.overflow_check(device_);

        // PHASE 4: Submit and execute commands
        // Update host-side write pointer
        // Tells the SystemMemoryManager that valid data exists in the issue queue upto this point
        mgr_->issue_queue_push_back(dc.write_offset_bytes(), fdcq_->id());

        // Write the commands to the device-side fetch queue
        // This updates the read/write pointers in the Device's L1 memory, effectively
        // Signals to the prefetcher kernel that new commands are available to fetch
        const auto start = std::chrono::steady_clock::now();
        for (const uint32_t sz : entry_sizes) {
            mgr_->fetch_queue_reserve_back(fdcq_->id());
            mgr_->fetch_queue_write(sz, fdcq_->id());
        }

        // Wait for completion of the issued commands
        distributed::Finish(mesh_device_->mesh_command_queue());
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
        DeviceData& device_data  // Pass by ref to update the expectation model
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

            // Update DeviceData for linear write
            DeviceDataUpdater::update_linear_write(payload, device_data, worker_range, is_mcast_);

            // Create the HostMemDeviceCommand
            HostMemDeviceCommand cmd = CommandBuilder::build_linear_write_command<flush_prefetch_, inline_data_>(
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
    static constexpr bool hugepage_write_ = true;

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
        DeviceData& device_data  // Pass by ref to update the expectation model
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

                // Update DeviceData for paged write
                DeviceDataUpdater::update_paged_write(
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
            HostMemDeviceCommand cmd = CommandBuilder::build_paged_write_command<hugepage_write_>(
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

    // Build subcmds once - reused for all commands
    std::vector<CQDispatchWritePackedUnicastSubCmd> build_sub_cmds(const std::vector<CoreCoord>& worker_cores) {
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds;
        sub_cmds.reserve(worker_cores.size());
        for (const auto& core : worker_cores) {
            const CoreCoord virtual_core = device_->virtual_core_from_logical_core(core, CoreType::WORKER);
            CQDispatchWritePackedUnicastSubCmd sub_cmd{};
            sub_cmd.noc_xy_addr = device_->get_noc_unicast_encoding(k_dispatch_downstream_noc, virtual_core);
            sub_cmds.push_back(sub_cmd);
        }
        return sub_cmds;
    }

    // Clamp xfer_size to fit within max_fetch_bytes_
    uint32_t clamp_to_max_fetch(
        uint32_t xfer_size_bytes,
        uint32_t num_sub_cmds,
        uint32_t packed_write_max_unicast_sub_cmds,
        bool no_stride,
        uint32_t l1_alignment) {
        // Calculate the command size
        DeviceCommandCalculator cmd_calc;
        cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
            num_sub_cmds,     // num_sub_cmds
            xfer_size_bytes,  // packed_data_sizeB
            packed_write_max_unicast_sub_cmds,
            no_stride  // no_stride
        );
        uint32_t command_size_bytes = cmd_calc.write_offset_bytes();
        // If the command size is less than max_fetch_bytes_, return the transfer size
        if (command_size_bytes <= max_fetch_bytes_) {
            return xfer_size_bytes;
        }

        // Else, linearly decrement by alignment until it fits
        // We can use binary search to speed this up
        uint32_t result = xfer_size_bytes;
        while (result > 0 && command_size_bytes > max_fetch_bytes_) {
            result -= l1_alignment;
            cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_sub_cmds,  // num_sub_cmds
                result,        // packed_data_sizeB
                packed_write_max_unicast_sub_cmds,
                no_stride  // no_stride
            );
            command_size_bytes = cmd_calc.write_offset_bytes();
        }

        return result;
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
        DeviceData& device_data) {
        // This vector stores commands related information for each iteration
        std::vector<HostMemDeviceCommand> commands_per_iteration;

        uint32_t remaining_bytes = get_transfer_size_bytes();

        const uint32_t num_sub_cmds = static_cast<uint32_t>(worker_cores.size());
        const uint32_t sub_cmds_bytes =
            tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment);

        // Build subcmds once - reused for all commands
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds = build_sub_cmds(worker_cores);

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
            DeviceDataUpdater::update_packed_write(payload, device_data, worker_cores, l1_alignment);

            HostMemDeviceCommand cmd = CommandBuilder::build_packed_write_command(
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
        const CoreCoord& worker_coord,
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
    // sizes until max-fetch would be exceeded, updating DeviceData for every multicast target
    std::vector<HostMemDeviceCommand> generate_packed_large_write_commands(
        const CoreRange& worker_range, uint32_t l1_alignment, DeviceData& device_data) {
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

using namespace tt::tt_metal;

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

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

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

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device_, worker_range, l1_base, dram_base, nullptr, true, dram_data_size_words);

    const auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    const uint32_t page_size_alignment_bytes = device_->allocator()->get_alignment(buf_type);
    const uint32_t num_banks = device_->allocator()->get_num_banks(buf_type);
    const tt::CoreType core_type = is_dram ? tt::CoreType::DRAM : tt::CoreType::WORKER;

    // Generate random page size
    uint32_t max_allowed = MAX_XFER_SIZE_16B - 1;
    uint32_t page_size_bytes =
        payload_generator_->get_random_size(max_allowed, bytes_per_16B_unit, page_size_bytes_param);

    // Calculate overhead using DeviceCommandCalculator in CommandSizeHelper
    // 0 pages for overhead only
    // Substracting the overhead from max_fetch_bytes_ gives the max allowed payload size per command
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_paged<hugepage_write_>(page_size_bytes, 0);
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

    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    DeviceData device_data(device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

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
    const uint32_t l1_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_base = device_->allocator()->get_base_allocator_addr(HalMemType::DRAM);

    // Setup DeviceData for validation
    DeviceData device_data(device_, worker_range, l1_base, dram_base, nullptr, false, dram_data_size_words);

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
