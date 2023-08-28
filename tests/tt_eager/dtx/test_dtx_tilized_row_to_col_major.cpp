#include <algorithm>
#include <functional>
#include <random>
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// 1. Host writes data to buffer in DRAM
// 2. Host generates DTX transformation node -> Producer : 1D vector sized 64 -> Consumer : 1D vector of same size but flips first 32 elements with last 32 elements
// 2. dram_to_l1_copy_with_address_map kernel on logical core {0, 0} BRISC copies data from buffer in step 1. to buffer in L1
// 3. Host reads from buffer written to in step 2.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////


        DataTransformations * dtx = new DataTransformations();
        TransformationNode * node0 = new TransformationNode("producer", 1);
        node0->groups[0]->shape = {64,64};
        dtx->transformations.push_back(node0);
        pass &= tilize_and_store(dtx, {0,1});
        dtx->print();
        DataTransformations * dtx_full = reverse_transformations(dtx);
        dtx_full->print();
        pass &= tilize_and_store(dtx_full, {1,0});
        dtx_full->print();
        pass &= collapse_transformations(dtx_full);
        dtx_full->print();

        pass &= generate_transfer_addresses_tiled_data(dtx_full);

        // copy transfer addresses into a vector
        std::vector<uint32_t> address_map;
        for(auto transfer : dtx_full->transformations.back()->groups[0]->transfers){
            address_map.push_back(transfer->src_address*2); // 2 for bfloat16
            address_map.push_back(transfer->dst_address*2);
            address_map.push_back(transfer->size*2);
        }
        std::cout << "Address Map - " << std::endl;
        for(auto i = 0; i < address_map.size(); i+=3) {
            std::cout << "Source address - " << address_map[i];
            std::cout << ", Destination address - " << address_map[i+1];
            std::cout << ", Size to transfer in bytes - " << address_map[i+2] << std::endl;
        }
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t dram_buffer_size = 2 * 64 * 64;
        uint32_t input_dram_buffer_addr = 0;
        uint32_t address_map_l1_addr = 500 * 1024;

        auto input_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, input_dram_buffer_addr, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto l1_b0 = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::L1);
        uint32_t l1_buffer_addr = l1_b0.address();

        auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();

        auto dram_to_l1_copy_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/dram_to_l1_copy_with_address_map.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto input_vector = unpack_uint32_vec_into_bfloat16_vec(input_vec);
        tt_metal::WriteToBuffer(input_dram_buffer, input_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::detail::WriteToDeviceL1(device, core, address_map_l1_addr, address_map);

        tt_metal::SetRuntimeArgs(
            program,
            dram_to_l1_copy_kernel,
            core,
            {input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            l1_buffer_addr,
            address_map_l1_addr,
            (std::uint32_t) address_map.size()});

        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_buffer_addr, dram_buffer_size, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_vector = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
