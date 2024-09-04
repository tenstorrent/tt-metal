// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <tt_numpy/functions.hpp>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

Tensor perform_transpose_wh(Tensor& input_tensor) {
    TT_FATAL(input_tensor.storage_type() == StorageType::OWNED);
    auto ashape = input_tensor.get_legacy_shape();
    TT_FATAL(ashape.rank() == 4);
    auto bshape = ashape;
    bshape[2] = ashape[3];
    bshape[3] = ashape[2];
    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::TILE, "This transpose assumes that the data layout is tiled!");
    auto input_buffer = owned_buffer::get_as<bfloat16>(input_tensor);
    auto output_buffer = owned_buffer::create<bfloat16>(input_buffer.size());
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
                                    output_buffer[output_index] = input_buffer[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return tt_metal::Tensor(OwnedStorage{output_buffer}, bshape, input_tensor.get_dtype(), tt::tt_metal::Layout::TILE);
}

int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        Shape shape = {1, 1, 10*TILE_HEIGHT, 12*TILE_WIDTH};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a =  tt::numpy::random::random(shape).to(Layout::TILE).to(device);

        tt_metal::Tensor c = ttnn::transpose(a, -2, -1);

        tt_metal::Tensor d = c.cpu();

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Tensor host_a = a.cpu(); // Move tensor a to host to validate
        auto host_vec = owned_buffer::get_as<bfloat16>(host_a);
        auto transposed_host_a = perform_transpose_wh(host_a);
        auto golden_vec = owned_buffer::get_as<bfloat16>(transposed_host_a);
        auto result_vec = owned_buffer::get_as<bfloat16>(d);
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
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
