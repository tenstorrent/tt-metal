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



namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram_neighbour {

uint32_t runtime_host_id = 0;

// Test config
struct DramNeighbourConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_banks = 0;
    uint32_t pages_per_bank = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    std::reference_wrapper<const std::map<uint32_t, uint32_t>> core_dram_map;

    DramNeighbourConfig(
        uint32_t test_id_,
        uint32_t num_of_transactions_,
        uint32_t num_banks_,
        uint32_t pages_per_bank_,
        uint32_t page_size_bytes_,
        DataFormat l1_data_format_,
        const std::map<uint32_t, uint32_t>& core_dram_map_)
        : test_id(test_id_),
          num_of_transactions(num_of_transactions_),
          num_banks(num_banks_),
          pages_per_bank(pages_per_bank_),
          page_size_bytes(page_size_bytes_),
          l1_data_format(l1_data_format_),
          core_dram_map(core_dram_map_) {}
};

void print_detailed_comparision(const vector<uint32_t>& packed_golden, const vector<uint32_t>& packed_output);


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

    std::vector<CoreCoord> dram_cores;
    for(const auto& [key, value] : core_dram_map) {
        dram_cores.push_back(CoreCoord{static_cast<uint16_t>(value), 0});
        // log_info(tt::LogTest, "line 67: dram_cores vector - DRAM bank {}", value);
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
    uint32_t input_buffer_address = mesh_buffer->address(); // need to read from different dram starting point

    // Generate input
    // vector<uint32_t> packed_input = generate_packed_constant_vector<uint32_t, bfloat16>(100.0f, total_size_bytes / sizeof(bfloat16));

    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_golden = packed_input;

    // Compile-time arguments
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.pages_per_bank,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Worker cores
    vector<CoreCoord> worker_cores;
    for(const auto [key, value] : core_dram_map) {
        CoreCoord core{static_cast<uint16_t>(key >> 16), static_cast<uint16_t>(key & 0xFFFF)};
        worker_cores.push_back(core);
        // log_info(tt::LogTest, "line 114: worker_cores vector - Core ({}, {})", core.x, core.y);
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

    vector<uint32_t> l1_addr;
    for(uint32_t i = 0; i < worker_cores.size(); i++) {
        l1_addr.push_back(get_l1_address_and_size(mesh_device, worker_cores[i]).base_address);
    }


    // Set runtime args: each core reads its adjacent bank
    for(uint32_t i = 0; i < worker_cores.size(); i++) {
        uint32_t dram_bank_id = core_dram_map.at((static_cast<uint32_t>(worker_cores[i].x) << 16) | static_cast<uint32_t>(worker_cores[i].y));
        // log_info(tt::LogTest, "line 130: Core ({}, {}), dram_bank_id {}",
        //  worker_cores[i].x, worker_cores[i].y, dram_bank_id);
        std::vector<uint32_t> core_runtime_args = {input_buffer_address, l1_addr[i], dram_bank_id};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, worker_cores[i], core_runtime_args);
    }

    // log_info(tt::LogTest, "Running Neighbour Read Test ID: {}, Run ID: {}", test_config.test_id, runtime_host_id);
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

    for (uint32_t i = 0; i < worker_cores.size(); i++) {
        detail::ReadFromDeviceL1(device, worker_cores[i], l1_addr[i], total_size_bytes/test_config.num_banks, cur_output);
        // log_info(tt::LogTest, "line 170: Read from L1 of Core ({}, {})", worker_cores[i].x, worker_cores[i].y);
        packed_output.insert(packed_output.end(), cur_output.begin(), cur_output.end());
        cur_output.clear();
    }

    // erase zeros
    packed_output.erase(remove(packed_output.begin(), packed_output.end(), 0), packed_output.end());

    // Verify results
    bool is_equal = (packed_output == packed_golden);
    if (!is_equal) {
        log_error(tt::LogTest, "Equality Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_golden));
        log_info(tt::LogTest, "Output vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_output));

        unit_tests::dm::dram_neighbour::print_detailed_comparision(packed_golden, packed_output);
        return is_equal;
    }

    return is_equal;
}

std::map<uint32_t, uint32_t> core_dram_mapping_ideal(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t num_dram_banks) {
    map<uint32_t, uint32_t> mapping;
    const vector<CoreCoord> dram_bank2core_coords =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);

    for(uint32_t i = 0; i < dram_bank2core_coords.size(); i++) {
        if(i == num_dram_banks) {
            break;
        }
        const CoreCoord& coord = dram_bank2core_coords[i];
        uint32_t key = (static_cast<uint32_t>(coord.x) << 16) | static_cast<uint32_t>(coord.y);
        mapping[key] = i;
    }

    return mapping;
}

void print_detailed_comparision(const vector<uint32_t>& packed_golden, const vector<uint32_t>& packed_output) {
    log_info(tt::LogTest, "\n\nDetailed Comparison:");
    log_info(tt::LogTest, "Total elements in Golden: {}, Total elements in Output: {}", packed_golden.size(), packed_output.size());
    size_t min_size = min(packed_golden.size(), packed_output.size());
    for (size_t i = 0; i < min_size; i++) {
        if (packed_golden[i] != packed_output[i]) {
            log_info(tt::LogTest, "Index {}: Golden = {}, Output = {}", i, packed_golden[i], packed_output[i]);
        }
    }
}


}  // namespace unit_tests::dm::dram_neighbour

TEST_F(GenericMeshDeviceFixture, printLogical2PhysicalMapping) {
    shared_ptr<distributed::MeshDevice> device = get_mesh_device();
    auto mesh_device = device->get_devices().front();

    CoreCoord grid_size = mesh_device->logical_grid_size();

    for (uint32_t x = 0; x < grid_size.x; x++) {
        for (uint32_t y = 0; y < grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            CoreCoord physical_core = mesh_device->worker_core_from_logical_core(logical_core);
            CoreCoord noc0_coord = mesh_device->virtual_noc0_coordinate(0, physical_core);
            log_info(tt::LogTest, "Logical Core: ({}, {}), Physical Core: ({}, {}), Noc0 Core: ({}, {}) ",
                     logical_core.x, logical_core.y, physical_core.x, physical_core.y, noc0_coord.x, noc0_coord.y);
        }
    }

    std::vector<CoreCoord> dram_physical_coords;
    uint32_t num_dram_banks = mesh_device->num_dram_channels();
    for (uint32_t bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        uint32_t dram_channel = mesh_device->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
        CoreCoord logical_dram_core = mesh_device->logical_core_from_dram_channel(dram_channel);
        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(mesh_device->id());
        CoreCoord physical_dram_core = soc_desc.get_physical_dram_core_from_logical(logical_dram_core);
        dram_physical_coords.push_back(physical_dram_core);
        log_info(tt::LogTest, "Bank ID: {}, Dram Channel: {}, Logical DRAM Core: ({}, {}), Physical DRAM Core: ({}, {}) ",
                 bank_id, dram_channel, logical_dram_core.x, logical_dram_core.y, physical_dram_core.x, physical_dram_core.y);
    }

    EXPECT_TRUE(true);
}


TEST_F(GenericMeshDeviceFixture, idealClosestNeighbourTest) {

    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 501;
    uint32_t num_of_transactions = 1;
    uint32_t num_banks = mesh_device->num_dram_channels();
    uint32_t pages_per_bank = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    std::map<uint32_t, uint32_t> core_dram_map =
        unit_tests::dm::dram_neighbour::core_dram_mapping_ideal(mesh_device, num_banks);

    for(const auto& [key, value] : core_dram_map) {
        CoreCoord core{static_cast<uint16_t>(key >> 16), static_cast<uint16_t>(key & 0xFFFF)};
        // log_info(tt::LogTest, "line 268: Core ({}, {}) assigned to DRAM bank {}", core.x, core.y, value);
    }


    unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
        test_id,
        num_of_transactions,
        num_banks,
        pages_per_bank,
        page_size_bytes,
        l1_data_format,
        core_dram_map);

    EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));
}


TEST_F(GenericMeshDeviceFixture, numPagesSweepClosestNeighbourTest) {

    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 502;
    uint32_t num_banks = mesh_device->num_dram_channels();
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    std::map<uint32_t, uint32_t> core_dram_map =
        unit_tests::dm::dram_neighbour::core_dram_mapping_ideal(mesh_device, num_banks);

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_pages = 1; num_pages <= max_num_pages; num_pages *= 2) {
            unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
                test_id,
                num_of_transactions,
                num_banks,
                num_pages,
                page_size_bytes,
                l1_data_format,
                core_dram_map);

            EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));
        }
    }

}

TEST_F(GenericMeshDeviceFixture, numBankSweepClosestNeighbourTest) {

    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    uint32_t test_id = 503;
    uint32_t max_num_banks = mesh_device->num_dram_channels();
    uint32_t num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_banks = 1; num_banks <= max_num_banks; num_banks++) {

            std::map<uint32_t, uint32_t> core_dram_map =
                unit_tests::dm::dram_neighbour::core_dram_mapping_ideal(mesh_device, num_banks);

            // Test config
            unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
                test_id,
                num_of_transactions,
                num_banks,
                num_pages,
                page_size_bytes,
                l1_data_format,
                core_dram_map);

            EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));
        }
    }

}

TEST_F(GenericMeshDeviceFixture, randomCoreToDramAssignmentTest) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    // Test parameters
    uint32_t test_id = 504;
    uint32_t num_of_transactions = 1;
    uint32_t pages_per_bank = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);

    auto grid_size = mesh_device->logical_grid_size();
    uint32_t max_dram_banks = mesh_device->num_dram_channels();

    for(uint32_t num_banks = 1; num_banks <= max_dram_banks; num_banks++) {
        std::map<uint32_t, uint32_t> core_dram_map;
        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<CoreCoord> available_cores;
        for (uint32_t x = 0; x < grid_size.x && available_cores.size() < num_banks; x++) {
            for (uint32_t y = 0; y < grid_size.y && available_cores.size() < num_banks; y++) {
                available_cores.push_back(CoreCoord(x, y));
            }
        }

        std::shuffle(available_cores.begin(), available_cores.end(), gen);

        std::vector<uint32_t> available_banks(num_banks);
        std::iota(available_banks.begin(), available_banks.end(), 0);
        std::shuffle(available_banks.begin(), available_banks.end(), gen);

        for (uint32_t i = 0; i < num_banks; i++) {
            const auto& core = available_cores[i];
            uint32_t packed_core = (static_cast<uint32_t>(core.x) << 16) | static_cast<uint32_t>(core.y);
            core_dram_map[packed_core] = available_banks[i];

            log_info(tt::LogTest, "Random assignment: Core ({}, {}) -> DRAM bank {}",
                    core.x, core.y, available_banks[i]);
        }

        unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
            test_id,
            num_of_transactions,
            num_banks,
            pages_per_bank,
            page_size_bytes,
            l1_data_format,
            core_dram_map);

        EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));

    }
}

TEST_F(GenericMeshDeviceFixture, randomCoreToDramAssignmentSweepTest) {
    GTEST_SKIP() << "Takes Too Long to Run";
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();

    // Test parameters
    uint32_t test_id = 505;
    uint32_t max_transactions = 256;
    uint32_t max_num_pages = 32;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);

    auto grid_size = mesh_device->logical_grid_size();
    uint32_t max_dram_banks = mesh_device->num_dram_channels();

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_pages = 1; num_pages <= max_num_pages; num_pages *= 2) {
            for(uint32_t num_banks = 1; num_banks <= max_dram_banks; num_banks++) {
                std::map<uint32_t, uint32_t> core_dram_map;
                std::random_device rd;
                std::mt19937 gen(rd());

                std::vector<CoreCoord> available_cores;
                for (uint32_t x = 0; x < grid_size.x && available_cores.size() < num_banks; x++) {
                    for (uint32_t y = 0; y < grid_size.y && available_cores.size() < num_banks; y++) {
                        available_cores.push_back(CoreCoord(x, y));
                    }
                }

                std::shuffle(available_cores.begin(), available_cores.end(), gen);

                std::vector<uint32_t> available_banks(num_banks);
                std::iota(available_banks.begin(), available_banks.end(), 0);
                std::shuffle(available_banks.begin(), available_banks.end(), gen);

                for (uint32_t i = 0; i < num_banks; i++) {
                    const auto& core = available_cores[i];
                    uint32_t packed_core = (static_cast<uint32_t>(core.x) << 16) | static_cast<uint32_t>(core.y);
                    core_dram_map[packed_core] = available_banks[i];

                    log_info(tt::LogTest, "Random assignment: Core ({}, {}) -> DRAM bank {}",
                            core.x, core.y, available_banks[i]);
                }

                unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
                    test_id,
                    num_of_transactions,
                    num_banks,
                    num_pages,
                    page_size_bytes,
                    l1_data_format,
                    core_dram_map);

                EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));

            }
        }
    }
}


}  // namespace tt::tt_metal
