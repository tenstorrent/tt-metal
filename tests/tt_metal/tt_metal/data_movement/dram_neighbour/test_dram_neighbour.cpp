// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram_neighbour {

uint32_t runtime_host_id = 0;

typedef struct IndexRange {
    uint32_t start;
    uint32_t end;
} IndexRange;

// Test config
struct DramNeighbourConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_banks = 0;
    uint32_t pages_per_bank = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    std::reference_wrapper<const std::map<uint32_t, uint32_t>> core_dram_map;
    std::reference_wrapper<const std::map<uint32_t, IndexRange>> dram_index_map;

    DramNeighbourConfig(
        uint32_t test_id_,
        uint32_t num_of_transactions_,
        uint32_t num_banks_,
        uint32_t pages_per_bank_,
        uint32_t page_size_bytes_,
        DataFormat l1_data_format_,
        const std::map<uint32_t, uint32_t>& core_dram_map_,
        const std::map<uint32_t, IndexRange>& dram_index_map_) :
        test_id(test_id_),
        num_of_transactions(num_of_transactions_),
        num_banks(num_banks_),
        pages_per_bank(pages_per_bank_),
        page_size_bytes(page_size_bytes_),
        l1_data_format(l1_data_format_),
        core_dram_map(core_dram_map_),
        dram_index_map(dram_index_map_) {}
};

void print_detailed_comparison(const vector<uint32_t>& packed_golden, const vector<uint32_t>& packed_output);

/// @brief Reads from DRAM to L1 with each core reading only its adjacent bank
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes
bool run_dm_neighbour(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramNeighbourConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    uint32_t num_pages = test_config.num_banks * test_config.pages_per_bank;
    const size_t total_size_bytes = num_pages * test_config.page_size_bytes;
    const map<uint32_t, uint32_t>& core_dram_map = test_config.core_dram_map.get();
    const map<uint32_t, IndexRange>& dram_index_map = test_config.dram_index_map.get();

    std::vector<CoreCoord> dram_cores;
    unordered_set<uint32_t> dram_visited;
    for (const auto& [key, value] : core_dram_map) {
        if (!dram_visited.contains(value)) {
            dram_cores.push_back(CoreCoord{static_cast<uint16_t>(value), 0});
            dram_visited.insert(value);
        }
    }

    // Buffer sharding: each bank's data to corresponding core
    BufferDistributionSpec shard_spec = BufferDistributionSpec(
        Shape{1, num_pages},                   // tensor shape in pages
        Shape{1, test_config.pages_per_bank},  // shard shape per core
        dram_cores);

    uint32_t single_tile_size = test_config.page_size_bytes;
    distributed::DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec)};

    Shape2D global_shape = {32, 32 * num_pages};
    distributed::ShardedBufferConfig sharded_buffer_config{
        .global_size = total_size_bytes,
        .global_buffer_shape = global_shape,
        .shard_shape = global_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer =
        distributed::MeshBuffer::create(sharded_buffer_config, per_device_buffer_config, mesh_device.get());
    uint32_t input_buffer_address = mesh_buffer->address();  // need to read from different dram starting point

    // Generate input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_golden = packed_input;

    // Compile-time arguments
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.pages_per_bank,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Worker cores
    vector<CoreCoord> worker_cores;
    for (const auto [key, value] : core_dram_map) {
        CoreCoord core{static_cast<uint16_t>(key >> 16), static_cast<uint16_t>(key & 0xFFFF)};
        worker_cores.push_back(core);
    }

    CoreRangeSet core_range_set(worker_cores);

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_neighbour/kernels/dram_neighbour_read.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    // ===== Barrier synchronization setup =====
    // CreateSemaphore allocates semaphores on all specified cores (same ID maps to same L1 offset).
    // We only use the coordinator's semaphore - all cores increment it via NOC and poll until num_cores.
    // Creating on all cores ensures get_semaphore(id) works correctly on every core.
    CoreCoord coordinator_core = worker_cores[0];
    CoreCoord coordinator_phys = device->worker_core_from_logical_core(coordinator_core);

    uint32_t reader_barrier_sem_id = 0;
    reader_barrier_sem_id = CreateSemaphore(program, core_range_set, 0);

    vector<uint32_t> l1_addr;
    uint32_t num_cores = worker_cores.size();
    for (uint32_t i = 0; i < num_cores; i++) {
        l1_addr.push_back(get_l1_address_and_size(mesh_device, worker_cores[i]).base_address);
    }

    // Set runtime args: each core reads its adjacent bank
    for (uint32_t i = 0; i < num_cores; i++) {
        uint32_t dram_bank_id = core_dram_map.at(
            (static_cast<uint32_t>(worker_cores[i].x) << 16) | static_cast<uint32_t>(worker_cores[i].y));
        uint32_t local_barrier_addr = l1_addr[i] + total_size_bytes;
        std::vector<uint32_t> core_runtime_args = {
            input_buffer_address,
            l1_addr[i],
            dram_bank_id,
            reader_barrier_sem_id,  // Semaphore ID, kernel will call get_semaphore() to get address
            coordinator_phys.x,
            coordinator_phys.y,
            num_cores,
            local_barrier_addr};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, worker_cores[i], core_runtime_args);
    }

    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, runtime_host_id);
    program.set_runtime_id(runtime_host_id++);

    // LAUNCH PROGRAM - Use mesh workload approach
    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, mesh_buffer, packed_input);

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    vector<uint32_t> packed_output;
    vector<uint32_t> cur_output;
    uint32_t per_core_output_size_bytes = total_size_bytes / test_config.num_banks;

    for (uint32_t i = 0; i < num_cores; i++) {
        // golden data for current core's adjacent bank
        uint32_t key = (static_cast<uint32_t>(worker_cores[i].x) << 16) | static_cast<uint32_t>(worker_cores[i].y);
        uint32_t dram_bank_id = core_dram_map.at(key);
        IndexRange cur_indices = dram_index_map.at(dram_bank_id);
        for (int j = cur_indices.start; j < cur_indices.end; j++) {
            packed_output.push_back(packed_golden[j]);
        }

        detail::ReadFromDeviceL1(device, worker_cores[i], l1_addr[i], per_core_output_size_bytes, cur_output);

        // Verify results
        bool is_equal = (packed_output == cur_output);
        if (!is_equal) {
            log_error(
                tt::LogTest,
                "Equality Check failed at index {} for Core ({}, {}) reading from DRAM bank {}. Test ID: {}, Run ID: "
                "{}",
                i,
                worker_cores[i].x,
                worker_cores[i].y,
                dram_bank_id,
                test_config.test_id,
                runtime_host_id - 1);
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(packed_golden));
            log_info(tt::LogTest, "Packed Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(packed_output));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(cur_output));

            return is_equal;
        }

        packed_output.clear();
        cur_output.clear();
    }

    return true;
}

std::map<uint32_t, uint32_t> core_dram_mapping_ideal(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t num_dram_banks) {
    map<uint32_t, uint32_t> mapping;
    const vector<CoreCoord> dram_bank2core_coords =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);

    for (uint32_t i = 0; i < dram_bank2core_coords.size(); i++) {
        if (i == num_dram_banks) {
            break;
        }
        const CoreCoord& coord = dram_bank2core_coords[i];
        uint32_t key = (static_cast<uint32_t>(coord.x) << 16) | static_cast<uint32_t>(coord.y);
        mapping[key] = i;
    }

    return mapping;
}

std::map<uint32_t, IndexRange> get_golden_index_ranges(
    const std::map<uint32_t, uint32_t>& core_dram_map, uint32_t total_size_bytes, uint32_t num_banks) {
    std::map<uint32_t, IndexRange> index_ranges;

    uint32_t per_core_output_size_bytes = total_size_bytes / num_banks;
    uint32_t per_core_output_size_elements = per_core_output_size_bytes / sizeof(uint32_t);

    unordered_set<uint32_t> visited;
    uint32_t i = 0;
    for (const auto& [key, dram_bank_id] : core_dram_map) {
        if (!visited.contains(dram_bank_id)) {
            uint32_t start_index = i * per_core_output_size_elements;
            uint32_t end_index = start_index + per_core_output_size_elements;
            index_ranges[dram_bank_id] = IndexRange{start_index, end_index};
            i++;
            visited.insert(dram_bank_id);
        }
    }

    return index_ranges;
}

std::map<uint32_t, uint32_t> add_neighbour_cores_dram_mapping(
    const std::map<uint32_t, uint32_t>& core_dram_map, const shared_ptr<distributed::MeshDevice>& mesh_device) {
    std::map<uint32_t, uint32_t> updated_map = core_dram_map;
    CoreCoord grid_size = mesh_device->logical_grid_size();

    const auto& dispatch_cores =
        tt::tt_metal::MetalContext::instance().get_dispatch_query_manager().get_logical_dispatch_cores_on_user_chips();

    std::unordered_set<CoreCoord> dispatch_set(dispatch_cores.begin(), dispatch_cores.end());

    for (const auto& [key, value] : core_dram_map) {
        uint32_t cur_x = static_cast<uint32_t>(key >> 16);
        uint32_t cur_y = static_cast<uint32_t>(key & 0xFFFF);
        uint32_t dram_bank_id = value;

        uint32_t right_core_x = cur_x >= grid_size.x - 1 ? 0 : cur_x + 1;
        uint32_t left_core_x = cur_x == 0 ? grid_size.x - 1 : cur_x - 1;

        uint32_t right_core_key = (right_core_x << 16) | cur_y;
        if (!core_dram_map.contains(right_core_key) &&
            !dispatch_set.contains(CoreCoord{static_cast<uint16_t>(right_core_x), static_cast<uint16_t>(cur_y)})) {
            updated_map[right_core_key] = dram_bank_id;
        }

        uint32_t left_core_key = (left_core_x << 16) | cur_y;
        if (!core_dram_map.contains(left_core_key) &&
            !dispatch_set.contains(CoreCoord{static_cast<uint16_t>(left_core_x), static_cast<uint16_t>(cur_y)})) {
            updated_map[left_core_key] = dram_bank_id;
        }
    }
    return updated_map;
}

std::map<uint32_t, uint32_t> add_single_row_cores_dram_mapping(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    std::map<uint32_t, uint32_t> mapping = core_dram_mapping_ideal(mesh_device, 1);
    CoreCoord grid_size = mesh_device->logical_grid_size();

    const auto& dispatch_cores =
        tt::tt_metal::MetalContext::instance().get_dispatch_query_manager().get_logical_dispatch_cores_on_user_chips();

    std::unordered_set<CoreCoord> dispatch_set(dispatch_cores.begin(), dispatch_cores.end());

    auto it = mapping.begin();
    uint32_t cur_x = static_cast<uint32_t>(it->first >> 16);
    uint32_t cur_y = static_cast<uint32_t>(it->first & 0xFFFF);
    uint32_t dram_bank_id = it->second;

    uint32_t next_x = cur_x >= grid_size.x - 1 ? 0 : cur_x + 1;
    while (next_x != cur_x) {
        uint32_t next_key = (next_x << 16) | cur_y;
        if (!dispatch_set.contains(CoreCoord{static_cast<uint16_t>(next_x), static_cast<uint16_t>(cur_y)})) {
            mapping[next_key] = dram_bank_id;
        }
        next_x = next_x >= grid_size.x - 1 ? 0 : next_x + 1;
    }

    return mapping;
}

bool run_single_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    uint32_t num_of_transactions,
    uint32_t num_banks,
    uint32_t pages_per_bank,
    uint32_t page_size_bytes,
    DataFormat l1_data_format,
    const std::map<uint32_t, uint32_t>& core_dram_map = {}) {
    std::map<uint32_t, IndexRange> dram_index_map =
        get_golden_index_ranges(core_dram_map, num_banks * pages_per_bank * page_size_bytes, num_banks);

    auto test_config = DramNeighbourConfig(
        test_id,
        num_of_transactions,
        num_banks,
        pages_per_bank,
        page_size_bytes,
        l1_data_format,
        core_dram_map,
        dram_index_map);

    return run_dm_neighbour(mesh_device, test_config);
}

bool run_sweep_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    uint32_t max_transactions,
    uint32_t num_banks,
    uint32_t max_num_pages,
    uint32_t page_size_bytes,
    DataFormat l1_data_format,
    const std::map<uint32_t, uint32_t>& core_dram_map = {}) {
    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_pages = 1; num_pages <= max_num_pages; num_pages *= 2) {
            EXPECT_TRUE(run_single_test(
                mesh_device,
                test_id,
                num_of_transactions,
                num_banks,
                num_pages,
                page_size_bytes,
                l1_data_format,
                core_dram_map));
        }
    }
    return true;
}

bool run_bank_sweep_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    uint32_t max_transactions,
    uint32_t max_num_banks,
    uint32_t pages_per_bank,
    uint32_t page_size_bytes,
    DataFormat l1_data_format) {
    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_banks = 1; num_banks <= max_num_banks; num_banks++) {
            std::map<uint32_t, uint32_t> core_dram_map =
                add_neighbour_cores_dram_mapping(core_dram_mapping_ideal(mesh_device, num_banks), mesh_device);
            EXPECT_TRUE(run_single_test(
                mesh_device,
                test_id,
                num_of_transactions,
                num_banks,
                pages_per_bank,
                page_size_bytes,
                l1_data_format,
                core_dram_map));
        }
    }
    return true;
}

}  // namespace unit_tests::dm::dram_neighbour

using unit_tests::dm::dram_neighbour::add_neighbour_cores_dram_mapping;
using unit_tests::dm::dram_neighbour::add_single_row_cores_dram_mapping;
using unit_tests::dm::dram_neighbour::core_dram_mapping_ideal;
using unit_tests::dm::dram_neighbour::run_bank_sweep_test;
using unit_tests::dm::dram_neighbour::run_single_test;
using unit_tests::dm::dram_neighbour::run_sweep_test;

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDramNeighbourDirectedIdeal) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 502;
    uint32_t num_of_transactions = 1;
    uint32_t num_banks = mesh_device->num_dram_channels();
    uint32_t pages_per_bank = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    std::map<uint32_t, uint32_t> core_dram_map =
        add_neighbour_cores_dram_mapping(core_dram_mapping_ideal(mesh_device, num_banks), mesh_device);

    EXPECT_TRUE(run_single_test(
        mesh_device,
        test_id,
        num_of_transactions,
        num_banks,
        pages_per_bank,
        page_size_bytes,
        l1_data_format,
        core_dram_map));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDramNeighbourNumPagesSweep) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 503;
    uint32_t num_banks = mesh_device->num_dram_channels();
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    std::map<uint32_t, uint32_t> core_dram_map =
        add_neighbour_cores_dram_mapping(core_dram_mapping_ideal(mesh_device, num_banks), mesh_device);

    EXPECT_TRUE(run_sweep_test(
        mesh_device,
        test_id,
        max_transactions,
        num_banks,
        max_num_pages,
        page_size_bytes,
        l1_data_format,
        core_dram_map));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDramNeighbourNumBankSweep) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 504;
    uint32_t max_num_banks = mesh_device->num_dram_channels();
    uint32_t num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);

    EXPECT_TRUE(run_bank_sweep_test(
        mesh_device, test_id, max_transactions, max_num_banks, num_pages, page_size_bytes, l1_data_format));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDramNeighbourSingleRowSweep) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 505;
    uint32_t num_banks = 1;
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);

    std::map<uint32_t, uint32_t> core_dram_map = add_single_row_cores_dram_mapping(mesh_device);

    EXPECT_TRUE(run_sweep_test(
        mesh_device,
        test_id,
        max_transactions,
        num_banks,
        max_num_pages,
        page_size_bytes,
        l1_data_format,
        core_dram_map));
}

}  // namespace tt::tt_metal
