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
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::one_d_matmul {

uint32_t runtime_host_id = 0;

// Test config
struct one_d_Matmul_Config {
    uint32_t test_id = 0;
    CoreCoord origin_logical_core;
    CoreCoord start_logical_core;
    CoreCoord end_logical_core;
    uint32_t num_subblocks_r_dim = 2;  // how many subblocks in r dim
    uint32_t num_subblocks_c_dim = 2;
    uint32_t num_subblocks_k_dim = 1;
    uint32_t subblock_r_dim = 1;  // how many pages each subblock in r dim takes up
    uint32_t subblock_c_dim = 1;
    uint32_t subblock_k_dim = 1;
    uint32_t page_size_bytes = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t dram_bank_id = 0;  // dram bank that all cores will read from
};

/// @brief Reads from DRAM to L1 with each core reading only its adjacent bank
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes
bool run_dm_1d_matmul(const shared_ptr<distributed::MeshDevice>& mesh_device, const one_d_Matmul_Config& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    CoreRangeSet matmul_cores({CoreRange(test_config.start_logical_core, test_config.end_logical_core)});
    vector<CoreCoord> matmul_cores_list = corerange_to_cores(matmul_cores);

    // Logical core sets for in0. The first column of cores will need the host writing the content into L1.
    CoreCoord in0_logical_start_coord = test_config.start_logical_core;
    CoreCoord in0_logical_end_coord = CoreCoord(test_config.start_logical_core.x, test_config.end_logical_core.y);
    CoreRangeSet in0_cores({CoreRange(in0_logical_start_coord, in0_logical_end_coord)});
    vector<CoreCoord> in0_cores_list = corerange_to_cores(in0_cores);

    // Logical core sets for in1. All cores in matmul_cores will read in1 from DRAM.
    // vector<CoreCoord> in1_cores_list = corerange_to_cores(matmul_cores);

    uint32_t in0_pages = (test_config.num_subblocks_r_dim * test_config.subblock_r_dim) *
                         (test_config.num_subblocks_k_dim * test_config.subblock_k_dim);
    uint32_t in1_pages = (test_config.num_subblocks_k_dim * test_config.subblock_k_dim) *
                         (test_config.num_subblocks_c_dim * test_config.subblock_c_dim);
    uint32_t in0_pages_bytes = in0_pages * test_config.page_size_bytes;
    uint32_t in1_pages_bytes = in1_pages * test_config.page_size_bytes;

    uint32_t l1_base_address = unit_tests::dm::get_l1_address_and_size(mesh_device, in0_cores_list[0]).base_address;
    log_info(
        tt::LogTest,
        "L1 base address for in0 cores starting at logical core {}: {:#x}",
        in0_cores_list[0].str(),
        l1_base_address);
    log_info(tt::LogTest, "Size of a Page: {} bytes", test_config.page_size_bytes);
    log_info(
        tt::LogTest,
        "Each subblock in r dim takes up {} pages, total in0 pages: {}",
        test_config.subblock_r_dim,
        in0_pages);
    log_info(tt::LogTest, "Total in0 size in bytes: {} bytes", in0_pages_bytes);
    log_info(tt::LogTest, "Size of bfloat16: {} bytes", sizeof(bfloat16));

    // in0 Input
    // vector<uint32_t> in0_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
    //     -100.0f,
    //     100.0f,
    //     in0_pages_bytes / sizeof(bfloat16),
    //     chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> in0_input =
        generate_packed_constant_vector<uint32_t, bfloat16>(1.0f, in0_pages_bytes / sizeof(bfloat16));

    // Write in0 input to L1 of the first column of cores
    log_info(tt::LogTest, "in0_input size (in uint32_t): {}", in0_input.size());
    vector<uint32_t> in0_per_core_pages;
    uint32_t pages_per_core = in0_input.size() / in0_cores_list.size();
    uint32_t pages_per_core_size_bytes = pages_per_core * sizeof(uint32_t);
    log_info(tt::LogTest, "Number of in0 cores: {}", in0_cores_list.size());
    log_info(tt::LogTest, "Each core in the first column will read {} elements (in uint32_t)", pages_per_core);
    for (uint32_t i = 0; i < in0_cores_list.size(); i++) {
        // in0_per_core_pages should contain the pages that the i-th core in in0_cores_list will read
        for (uint32_t j = 0; j < pages_per_core; j++) {
            in0_per_core_pages.push_back(in0_input[i * pages_per_core + j]);
        }
        log_info(
            tt::LogTest,
            "Writing to L1 of core {} with {} bytes of data",
            in0_cores_list[i].str(),
            in0_per_core_pages.size() * sizeof(uint32_t));
        detail::WriteToDeviceL1(device, in0_cores_list[i], l1_base_address, in0_per_core_pages);
        log_info(
            tt::LogTest,
            "Content inside in0_per_core_pages: {}, {}, {},...",
            in0_per_core_pages[0],
            in0_per_core_pages[1],
            in0_per_core_pages[2]);
        in0_per_core_pages.clear();
    }

    // in1 Input
    // vector<uint32_t> in1_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
    //     -100.0f,
    //     100.0f,
    //     in1_pages_bytes / sizeof(bfloat16),
    //     chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> in1_input =
        generate_packed_constant_vector<uint32_t, bfloat16>(1.0f, in1_pages_bytes / sizeof(bfloat16));

    // DRAM Address
    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();
    uint32_t input_dram_address = dram_info.base_address;

    // Write Input to DRAM
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_bank_id, input_dram_address, in1_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    // in1_per_core_read_size_bytes is how much each core should read from the DRAM bank
    uint32_t in1_per_core_read_size_bytes = in1_pages_bytes / test_config.num_subblocks_c_dim;
    // in1_per_core_read_addr stores the memory address where each core should read from DRAM bank
    vector<uint32_t> in1_per_core_read_addr;
    for (uint32_t i = 0; i < test_config.num_subblocks_c_dim; i++) {
        in1_per_core_read_addr.push_back(input_dram_address + i * in1_per_core_read_size_bytes);
    }
    // in0_mcast_output_addr is the memory address where each core will leave the in0 mcast output in L1
    // uint32_t in0_mcast_output_addr = l1_base_address + pages_per_core_size_bytes + 0x10;
    // in1_output_addr is the memory address where each core will leave the in1 read output in L1. It is placed after
    // in0 mcast output in L1.
    uint32_t in1_output_addr = l1_base_address + ((pages_per_core_size_bytes + 0x10) << 1);

    log_info(tt::LogTest, "\n\n\nEach core will read {} bytes from DRAM", in1_per_core_read_size_bytes);
    log_info(
        tt::LogTest,
        "Each core will read from DRAM address: {:#x}, {:#x},...",
        in1_per_core_read_addr[0],
        in1_per_core_read_addr[1]);
    log_info(tt::LogTest, "Each core will write in1 read output to L1 address starting from: {:#x}", in1_output_addr);

    vector<uint32_t> risc1_compile_args = {
        test_config.test_id,      // Test ID
        test_config.dram_bank_id  // DRAM bank that all cores will read from
    };

    // Kernels
    // auto risc0_kernel = CreateKernel(
    //     program,
    //     "tests/tt_metal/tt_metal/data_movement/1d_matmul/kernels/2cluster-1d_matmul.cpp",
    //     matmul_cores,
    //     DataMovementConfig{
    //         .processor = DataMovementProcessor::RISCV_0,
    //         .noc = NOC::RISCV_0_default,
    //         .compile_args = risc0_compile_args});

    auto risc1_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/1d_matmul/kernels/in1_kernel.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = risc1_compile_args});

    // Assign Runtime Args
    for (int i = 0; i < matmul_cores_list.size(); i++) {
        vector<uint32_t> risc1_core_runtime_args = {
            in1_per_core_read_addr[(
                matmul_cores_list[i].x %
                test_config.num_subblocks_c_dim)],  // Each core reads from addr based on its column
            in1_per_core_read_size_bytes,           // Each core reads the same amount of data
            in1_output_addr                         // Each core writes to the same address in L1
        };
        tt::tt_metal::SetRuntimeArgs(program, risc1_kernel, matmul_cores_list[i], risc1_core_runtime_args);
    }

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Verify in1 read output in L1 for each core
    vector<uint32_t> golden_in1_read_output;
    uint32_t total_in1_read_elements = in1_per_core_read_size_bytes / sizeof(uint32_t);
    for (int i = 0; i < matmul_cores_list.size(); i++) {
        vector<uint32_t> in1_read_output;
        detail::ReadFromDeviceL1(
            device, matmul_cores_list[i], in1_output_addr, in1_per_core_read_size_bytes, in1_read_output);
        log_info(
            tt::LogTest,
            "Core {}: Read {} bytes from DRAM, content (in uint32_t): {}, {}, {},...",
            matmul_cores_list[i].str(),
            in1_per_core_read_size_bytes,
            in1_read_output[0],
            in1_read_output[1],
            in1_read_output[2]);
        uint32_t cur_c_dim = matmul_cores_list[i].x % test_config.num_subblocks_c_dim;
        for (uint32_t j = 0; j < total_in1_read_elements; j++) {
            golden_in1_read_output.push_back(in1_input[cur_c_dim * total_in1_read_elements + j]);
        }
        bool is_equal = (golden_in1_read_output == in1_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in1 read output does not match golden output!", matmul_cores_list[i].str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in1_input));
            log_info(tt::LogTest, "Packed Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(golden_in1_read_output));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in1_read_output));

            return is_equal;
        } else {
            log_info(tt::LogTest, "Core {}: in1 read output matches golden output!", matmul_cores_list[i].str());
        }
        golden_in1_read_output.clear();
    }

    return true;
}

}  // namespace unit_tests::dm::one_d_matmul

TEST_F(GenericMeshDeviceFixture, Test1DMatmulIdeal) {
    auto mesh_device = get_mesh_device();

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    unit_tests::dm::one_d_matmul::one_d_Matmul_Config test_config{
        .test_id = 520,
        .origin_logical_core = CoreCoord(0, 0),
        .start_logical_core = CoreCoord(0, 0),
        .end_logical_core = CoreCoord(1, 1),
        .num_subblocks_r_dim = 2,
        .num_subblocks_c_dim = 2,
        .num_subblocks_k_dim = 1,
        .subblock_r_dim = 1,
        .subblock_c_dim = 1,
        .subblock_k_dim = 1,
        .page_size_bytes = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .dram_bank_id = 0};

    EXPECT_TRUE(unit_tests::dm::one_d_matmul::run_dm_1d_matmul(mesh_device, test_config));
}

}  // namespace tt::tt_metal
