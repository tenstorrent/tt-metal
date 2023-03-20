#include "tt_metal/host_api.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/op_library/tilize/tilize_op.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

class ConvParameters {
    public:
        uint32_t R = 1;
        uint32_t S = 1;
        uint32_t U = 1;
        uint32_t V = 1;
        uint32_t PadH = 0;
        uint32_t PadW = 0;
        ConvParameters(uint32_t r, uint32_t s,  uint32_t u,  uint32_t v,  uint32_t padH,  uint32_t padW) :
        R(r), S(s), U(u), V(v), PadH(padH), PadW(padW) {
            TT_ASSERT(U > 0 and V > 0);
            TT_ASSERT(R > 0 and S > 0);
            TT_ASSERT(PadH >= 0 and PadW >= 0);
        }
        void print() {
            std::cout << "Printing conv params" << std::endl;
            std::cout << "R - " << R << " S - " << S << " U - " << U << " V - " << V << " PadH - " << PadH << " PadW - " << PadW << std::endl;
        }
};

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

std::vector<bfloat16> move_act_dram_to_l1(std::vector<bfloat16> &input_values,
                                    const std::array<uint32_t, 4>& in_shape,
                                    ConvParameters conv_params) {
    std::vector<bfloat16> output;

    std::vector<std::pair<int, int>> increments;
    for (auto r=0; r<conv_params.R; r++) {
        for(auto s =0; s<conv_params.S; s++) {
            increments.push_back(std::make_pair(s, r));
        }
    }
    for(int w = 0; w < in_shape[0]; w++) {
        for(int y = 0; y <= in_shape[1]-conv_params.R; y=y+conv_params.U) {
            for(int x = 0; x <= in_shape[2]-conv_params.S; x=x+conv_params.V) {
                for(auto increment: increments) {
                    auto x_new = x + increment.first;
                    auto y_new = y + increment.second;
                    for(int z = 0; z < in_shape[3]; z++) {
                        auto idx = z + x_new * in_shape[3] + y_new * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                        assert(idx >= 0 and idx < input_values.size());
                        output.push_back(input_values[idx]);
                    }
                }
            }
        }

    }

    return output;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 32, 4, 4};
        // Allocates a DRAM buffer on device populated with values specified by initialize

        Tensor a = Tensor(shape, Initialize::INCREMENT, DataType::BFLOAT16, Layout::CHANNELS_LAST, device, MemoryConfig{.interleaved=false, .dram_channel=0});
        Tensor b = tilize_conv_activation(a);
        Tensor c =  b.to(host);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        std::cout << "Moving src data to host to validate" << std::endl;
        Tensor host_a = a.to(host); // Move tensor a to host to validate
        auto host_vec_conv =  *reinterpret_cast<std::vector<bfloat16>*>(host_a.data_ptr());
        auto host_vec = move_act_dram_to_l1(host_vec_conv, {1,4,4,32}, ConvParameters(3,3,1,1,0,0));
        std::cout << "host vec size " << host_vec.size();
        assert(host_vec.size() == 4*32*9);
        std::array<uint32_t, 4> cl_shape = {1, 1, 4, 32*9};
        Tensor g = Tensor(host_vec, cl_shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
        Tensor golden = g.to(Layout::TILE);
        auto golden_vec =  *reinterpret_cast<std::vector<bfloat16>*>(golden.data_ptr());
        auto result_vec = *reinterpret_cast<std::vector<bfloat16>*>(c.data_ptr());
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
