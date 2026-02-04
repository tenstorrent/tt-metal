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
    std::reference_wrapper<const std::unordered_map<uint32_t, uint32_t>> core_dram_map;

    DramNeighbourConfig(
        uint32_t test_id_,
        uint32_t num_of_transactions_,
        uint32_t num_banks_,
        uint32_t pages_per_bank_,
        uint32_t page_size_bytes_,
        DataFormat l1_data_format_,
        const std::unordered_map<uint32_t, uint32_t>& core_dram_map_)
        : test_id(test_id_),
          num_of_transactions(num_of_transactions_),
          num_banks(num_banks_),
          pages_per_bank(pages_per_bank_),
          page_size_bytes(page_size_bytes_),
          l1_data_format(l1_data_format_),
          core_dram_map(core_dram_map_) {}
};

/// @brief Reads from DRAM to L1 with each core reading only its adjacent bank
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes
bool run_dm_neighbour(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramNeighbourConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    uint32_t num_pages = test_config.num_banks * test_config.pages_per_bank;
    const size_t total_size_bytes = num_pages * test_config.page_size_bytes;

    // DRAM coords: y=0, x=[0, num_banks-1]
    CoreRange dram_bank_range({0, 0}, {test_config.num_banks - 1, 0});
    std::vector<CoreCoord> dram_cores = corerange_to_cores(dram_bank_range);

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
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_golden = packed_input;

    // Compile-time arguments
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.pages_per_bank,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Worker cores
    vector<CoreCoord> worker_cores;
    const unordered_map<uint32_t, uint32_t>& core_dram_map = test_config.core_dram_map.get();
    for(const auto [key, value] : core_dram_map) {
        worker_cores.push_back(CoreCoord{static_cast<uint16_t>(key >> 16), static_cast<uint16_t>(key & 0xFFFF)});
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

    uint32_t l1_addr = get_l1_address_and_size(mesh_device, worker_cores[0]).base_address;

    // Set runtime args: each core reads its adjacent bank
    for(uint32_t i = 0; i < worker_cores.size(); i++) {
        uint32_t dram_bank_id = core_dram_map.at((static_cast<uint32_t>(worker_cores[i].x) << 16) | static_cast<uint32_t>(worker_cores[i].y));
        std::vector<uint32_t> core_runtime_args = {input_buffer_address, l1_addr, dram_bank_id};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, worker_cores[i], core_runtime_args);
    }


    log_info(tt::LogTest, "Running Neighbour Read Test ID: {}, Run ID: {}", test_config.test_id, runtime_host_id);
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
    bool is_equal = false;

    for (const CoreCoord& logical_core : worker_cores) {
        detail::ReadFromDeviceL1(device, logical_core, l1_addr, total_size_bytes, packed_output);

        // Verify results
        is_equal = (packed_output == packed_golden);
        if (!is_equal) {
            log_error(tt::LogTest, "Equality Check failed");
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(packed_golden));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(packed_output));
            return is_equal;
        }

    }

    return is_equal;
}

unordered_map<uint32_t, uint32_t> core_dram_mapping_ideal(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    unordered_map<uint32_t, uint32_t> mapping;
    const vector<CoreCoord> dram_bank2core_coords =   
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(  
            tt::tt_metal::NOC::RISCV_0_default); 
    
    for(uint32_t i = 0; i < dram_bank2core_coords.size(); i++) {
        const CoreCoord& coord = dram_bank2core_coords[i];
        uint32_t key = (static_cast<uint32_t>(coord.x) << 16) | static_cast<uint32_t>(coord.y);
        mapping[key] = i;
    }

    return mapping;
}

}  // namespace unit_tests::dm::dram_neighbour

TEST_F(GenericMeshDeviceFixture, test1) {
    log_info(tt::LogTest, "GOT HERE!");
    auto mesh_device = get_mesh_device();

    // DataFormat l1_data_format = DataFormat::Float16_b;
    // uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    // uint32_t num_of_transactions = 256;

    // unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config = {
    //     .test_id = 100,
    //     .num_of_transactions = num_of_transactions,
    //     .num_banks = mesh_device->num_dram_channels(),
    //     .pages_per_bank = 32,
    //     .page_size_bytes = page_size_bytes,
    //     .l1_data_format = l1_data_format};


    for(uint32_t i = 0; i < mesh_device->num_dram_channels(); i++) {
        CoreCoord coord = mesh_device->allocator_impl()->get_logical_core_from_bank_id(i);
        log_info(tt::LogTest, "Bank id: {}, Core Coord: ({}, {}) \n", i, coord.x, coord.y);
    }
    
    EXPECT_TRUE(true);
}


TEST_F(GenericMeshDeviceFixture, test2) {
    shared_ptr<distributed::MeshDevice> device = get_mesh_device();
    auto mesh_device = device->get_devices().front();

    CoreCoord grid_size = mesh_device->logical_grid_size();

    for (uint32_t x = 0; x < grid_size.x; x++) {  
        for (uint32_t y = 0; y < grid_size.y; y++) {  
            CoreCoord logical_core(x, y);  
            CoreCoord physical_core = mesh_device->worker_core_from_logical_core(logical_core);  
            CoreCoord noc0_coord = mesh_device->virtual_noc0_coordinate(0, physical_core);
            log_info(tt::LogTest, "Logical Core: ({}, {}), Physical Core: ({}, {}), Noc0 Core: ({}, {}) \n",  
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
        log_info(tt::LogTest, "Bank ID: {}, Dram Channel: {}, Logical DRAM Core: ({}, {}), Physical DRAM Core: ({}, {}) \n",  
                 bank_id, dram_channel, logical_dram_core.x, logical_dram_core.y, physical_dram_core.x, physical_dram_core.y);
    }  
    
    EXPECT_TRUE(true);
}


TEST_F(GenericMeshDeviceFixture, idealTest) {

    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();  
  
    unordered_map<uint32_t, uint32_t> core_dram_map =   
        unit_tests::dm::dram_neighbour::core_dram_mapping_ideal(mesh_device);

    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    uint32_t num_of_transactions = 256;

    unit_tests::dm::dram_neighbour::DramNeighbourConfig test_config(
        110,
        num_of_transactions,
        mesh_device->num_dram_channels(),
        32,
        page_size_bytes,
        l1_data_format,
        core_dram_map);    

    // const uint32_t num_cores = dram_bank2core_coords.size();  
    // auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);  
  
    // for (uint32_t bank_id = 0; bank_id < dram_bank2core_coords.size(); bank_id++) {  
    //     const auto& core_coord = dram_bank2core_coords[bank_id];  
    //     CoreCoord physical_core = mesh_device->worker_core_from_logical_core(core_coord); 
    //     log_info(tt::LogTest, "DRAM bank {} -> Optimal worker core ({}, {}) (Physical Core: ({}, {}))",   
    //              bank_id, core_coord.x, core_coord.y, physical_core.x, physical_core.y);  
    // }  

    EXPECT_TRUE(unit_tests::dm::dram_neighbour::run_dm_neighbour(mesh_device, test_config));
}


}  // namespace tt::tt_metal