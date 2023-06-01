#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool nearly_equal(
  float a, float b,
  float epsilon = 1e-5, float abs_th = 1e-5) {
  auto diff = std::abs(a-b);
  auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  return diff < std::max(abs_th, epsilon * norm);
}


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
        std::array<uint32_t, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
        Tensor b = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);

        std::vector<bfloat16> a_vec = *reinterpret_cast<std::vector<bfloat16>*>(a.to(host).data_ptr());
        std::vector<bfloat16> b_vec = *reinterpret_cast<std::vector<bfloat16>*>(b.to(host).data_ptr());

        Tensor add_output = add(a, b).to(host);
        std::vector<bfloat16> add_output_vec = *reinterpret_cast<std::vector<bfloat16>*>(add_output.data_ptr());
        for (int i = 0; i < a_vec.size(); i++) {
            TT_ASSERT(nearly_equal(a_vec[i].to_float() + b_vec[i].to_float(), add_output_vec[i].to_float()), "EltwiseBinary Add: Comparison Failed");
        }

        Tensor sub_output = sub(a, b).to(host);
        std::vector<bfloat16> sub_output_vec = *reinterpret_cast<std::vector<bfloat16>*>(sub_output.data_ptr());
        for (int i = 0; i < a_vec.size(); i++) {
            TT_ASSERT(nearly_equal(a_vec[i].to_float() - b_vec[i].to_float(), sub_output_vec[i].to_float()), "EltwiseBinary Sub: Comparison Failed");
        }

        Tensor mul_output = mul(a, b).to(host);
        std::vector<bfloat16> mul_output_vec = *reinterpret_cast<std::vector<bfloat16>*>(mul_output.data_ptr());
        for (int i = 0; i < a_vec.size(); i++) {
            TT_ASSERT(nearly_equal(a_vec[i].to_float() * b_vec[i].to_float(), mul_output_vec[i].to_float(), 1e-2, 1e-3), "EltwiseBinary Mul: Comparison Failed");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.to(host); // Move tensor a to host to validate
        //pass &= (host_a.data() == dcAdd.data());
        //pass &= (host_a.data() == dcSub.data());
        //pass &= (host_a.data() == dcMul.data());

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
