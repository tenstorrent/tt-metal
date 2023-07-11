#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/host_buffer.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "constants.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    bool pass = true;

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
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 32, 45, 64};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a =  tt::numpy::random::random(shape).to(device);
        Tensor b = tilize_with_zero_padding(a);
        Tensor c =  b.to(host);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        std::cout << "Moving src data to host to validate" << std::endl;
        Tensor host_a = a.to(host); // Move tensor a to host to validate
        // TODO: Update when tensor.pad_to_tile() function is added
        auto padded_shape = a.shape();
        padded_shape[2] = roundup(padded_shape[2], TILE_HEIGHT);
        padded_shape[3] = roundup(padded_shape[3], TILE_WIDTH);
        Tensor padded_host_a = host_a.pad(padded_shape, {0,0,0,0}, 0);
        Tensor golden = padded_host_a.to(Layout::TILE);
        auto golden_vec =  host_buffer::view_as<bfloat16>(golden);
        auto result_vec = host_buffer::view_as<bfloat16>(c);
        std::cout << "Validating " << std::endl;
         std::cout << "golden vec size " << golden_vec.size() << std::endl;
        std::cout << "result vec size " << result_vec.size() << std::endl;
        uint32_t num_errors = 0;
        for(uint32_t i = 0; i < result_vec.size() ; i++) {
            if(result_vec[i] != golden_vec[i]) {
                if(num_errors < 10)
                    std::cout << "Error at i=" << i << " result=" <<result_vec[i]<< " golden=" <<golden_vec[i] << std::endl;
                num_errors++;
            }
        }
        pass &= (result_vec == golden_vec);

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
