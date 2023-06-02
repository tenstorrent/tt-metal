#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/transpose_rm/transpose_rm_op.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
Tensor perform_transpose_hc(Tensor& a) {
    TT_ASSERT(a.on_host());
    auto ashape = a.shape();
    TT_ASSERT(ashape.size() == 4);
    auto bshape = ashape;
    bshape[1] = ashape[2];
    bshape[2] = ashape[1];
    TT_ASSERT(a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This transpose assumes that the data layout is row major!");
    std::vector<bfloat16> a_vec = *reinterpret_cast<std::vector<bfloat16>*>(a.data_ptr());
    std::vector<bfloat16> b_vec(a_vec.size(), 0);
    auto N = ashape[0];
    auto C = ashape[1];
    auto H = ashape[2];
    auto W = ashape[3];
    for(auto n = 0; n < N; n++) {
        for(auto h = 0; h < H; h++) {
            for(auto c = 0; c < C; c++) {
                for(auto w = 0; w < W; w++) {
                    auto a_index = n*N*C*H*W + c*H*W + h*W + w;
                    auto b_index = n*N*C*H*W + h*C*W + c*W + w;
                    b_vec[b_index] = a_vec[a_index];
                }
            }
        }
    }
    //auto bdata = pack_bfloat16_vec_into_uint32_vec(b_vec);
    tt_metal::Tensor b = tt_metal::Tensor(b_vec, bshape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR);
    return b;
}

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

        pass &= InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 2, 2, 512};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::ROW_MAJOR, device);

        tt_metal::Tensor c = tt_metal::transpose_hc_rm(a);

        tt_metal::Tensor d = c.to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Tensor host_a = a.to(host); // Move tensor a to host to validate
        auto golden_vec = *reinterpret_cast<std::vector<bfloat16>*>(perform_transpose_hc(host_a).data_ptr());
        auto result_vec = *reinterpret_cast<std::vector<bfloat16>*>(d.data_ptr());
        pass &= (golden_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);

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
