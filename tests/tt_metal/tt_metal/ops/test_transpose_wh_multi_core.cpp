#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/host_buffer.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

Tensor perform_transpose_wh(Tensor& input_tensor) {
    TT_ASSERT(input_tensor.storage_type() == StorageType::HOST);
    auto ashape = input_tensor.shape();
    TT_ASSERT(ashape.size() == 4);
    auto bshape = ashape;
    bshape[2] = ashape[3];
    bshape[3] = ashape[2];
    TT_ASSERT(input_tensor.layout() == tt::tt_metal::Layout::TILE, "This transpose assumes that the data layout is tiled!");
    auto input_view = host_buffer::view_as<bfloat16>(input_tensor);
    auto output_buffer = host_buffer::create<bfloat16>(input_view.size());
    auto output_view = host_buffer::view_as<bfloat16>(output_buffer);
    auto N = ashape[0];
    auto C = ashape[1];
    auto H = ashape[2];
    auto W = ashape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;
    assert(H%TILE_HEIGHT == 0);
    assert(W%TILE_WIDTH == 0);
    for(auto n = 0; n < N; n++) {
        for(auto c = 0; c < C; c++) {
            for(auto ht = 0; ht < Ht; ht++) {
                for(auto wt = 0; wt < Wt; wt++) {
                    for(auto fh = 0; fh < 2; fh++) {
                        for(auto fw = 0; fw < 2; fw++) {
                            for(auto sth = 0; sth < 16; sth++) {
                                for(auto stw = 0; stw < 16; stw++) {
                                    auto input_index = n*C*H*W + c*H*W + ht*Wt*TILE_HEIGHT*TILE_WIDTH + wt*TILE_HEIGHT*TILE_WIDTH + fh*32*16 + fw*16*16 + sth*16 + stw;
                                    auto output_index = n*C*H*W + c*H*W + wt*Ht*TILE_HEIGHT*TILE_WIDTH + ht*TILE_HEIGHT*TILE_WIDTH + fw*16*32 + fh*16*16 + stw*16 + sth;
                                    output_view[output_index] = input_view[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return tt_metal::Tensor(HostStorage{output_buffer}, bshape, input_tensor.dtype(), tt::tt_metal::Layout::TILE);
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
        std::array<uint32_t, 4> shape = {1, 1, 10*TILE_HEIGHT, 12*TILE_WIDTH};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a =  tt::numpy::random::random(shape).to(Layout::TILE).to(device);

        tt_metal::Tensor c = tt_metal::transpose_wh(a);

        tt_metal::Tensor d = c.to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Tensor host_a = a.to(host); // Move tensor a to host to validate
        auto host_vec = host_buffer::view_as<bfloat16>(host_a);
        auto transposed_host_a = perform_transpose_wh(host_a);
        auto golden_vec = host_buffer::view_as<bfloat16>(transposed_host_a);
        auto result_vec = host_buffer::view_as<bfloat16>(d);
        if(golden_vec != result_vec) {
            assert(golden_vec.size() == result_vec.size());
            for(uint32_t i = 0; i < golden_vec.size(); i++) {
                if(golden_vec[i] != result_vec[i]) {
                    std::cout << "Error at i=" << i << ", golden=" << golden_vec[i]  << ", result=" << result_vec[i] << std::endl;
                }
            }
        }
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
