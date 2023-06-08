#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "bfloat16.hpp"

#include "catch.hpp"

using namespace tt;

// Reader reads from 1 DRAM into single core
bool read_single_dram_to_single_core(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& dram_channel,
    const size_t& dram_byte_address,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core
) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto input_dram_buffer = tt_metal::Buffer(
        device, byte_size, dram_byte_address, dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(reader_core);
    auto l1_buffer = tt_metal::Buffer(
        device, byte_size, l1_byte_address, l1_bank_ids.at(0), byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/dram_to_l1_copy.cpp",
        reader_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = tt::test_utils::generate_uniform_int_random_vector<uint32_t>(0, 100, byte_size);
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        }
    );
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_core_data;
    tt_metal::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
    pass &= (dest_core_data == inputs);
    if (not pass) {
        INFO("Mismatch at Core: " << reader_core.str());
    }
    return pass;
}

// Writer reads from 1 DRAM into single core
bool write_single_core_to_single_dram(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& dram_channel,
    const size_t& dram_byte_address,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core
) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto output_dram_buffer = tt_metal::Buffer(
        device, byte_size, dram_byte_address, dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto dram_noc_xy = output_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(writer_core);
    auto l1_buffer = tt_metal::Buffer(
        device, byte_size, l1_byte_address, l1_bank_ids.at(0), byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/l1_to_dram_copy.cpp",
        writer_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = tt::test_utils::generate_uniform_int_random_vector<uint32_t>(0, 100, byte_size);
    tt_metal::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        }
    );
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        INFO("Mismatch at Core: " << writer_core.str());
    }
    return pass;
}

// Reader reads from single dram to core, writer synchronizes with datacopy kernel and writes to dram
// DRAM --> (Reader Core CB using reader RISCV)
// Reader Core --> Datacopy --> Reader Core
// Reader Core --> Writes to Dram
bool read_write_single_core_to_single_dram(
    tt_metal::Device* device,
    const size_t& num_tiles,
    const size_t& tile_byte_size,
    const size_t& output_dram_channel,
    const size_t& output_dram_byte_address,
    const size_t& input_dram_channel,
    const size_t& input_dram_byte_address,
    const size_t& local_core_input_byte_address,
    const tt::DataFormat& local_core_input_data_format,
    const size_t& local_core_output_byte_address,
    const tt::DataFormat& local_core_output_data_format,
    const CoreCoord& core
) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = num_tiles*tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = tt_metal::Buffer(
        device, byte_size, input_dram_byte_address, input_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device, byte_size, output_dram_byte_address, output_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        0,
        core,
        num_tiles,
        byte_size,
        local_core_input_byte_address,
        local_core_input_data_format
        );
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        16,
        core,
        num_tiles,
        byte_size,
        local_core_output_byte_address,
        local_core_output_data_format
        );

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto datacopy_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = create_random_vector_of_bfloat16(
        byte_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)num_tiles,
        }
    );
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)num_tiles,
        }
    );
    pass &= tt_metal::LaunchKernels(device, program);


    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    tt::test_utils::print_vec(inputs);
    tt::test_utils::print_vec(dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}


TEST_CASE(
    "single_core_reader", "[dram][single_core][reader]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        read_single_dram_to_single_core(
            device,
            2*1024,
            0,
            0,
            UNRESERVED_BASE,
            {.x=0, .y=0}
        )
    );
    tt_metal::CloseDevice(device);
}
TEST_CASE(
    "single_core_writer", "[dram][single_core][writer]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        write_single_core_to_single_dram(
            device,
            2*1024,
            0,
            0,
            UNRESERVED_BASE,
            {.x=0, .y=0}
        )
    );
    tt_metal::CloseDevice(device);
}
TEST_CASE(
    "single_core_reader_datacopy_writer", "[dram][single_core][writer][reader]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        read_write_single_core_to_single_dram(
            device,
            1,
            2*32*32,
            0,
            0,
            0,
            16*32*32,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0}
        )
    );
    tt_metal::CloseDevice(device);
}
