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


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

inline vector<uint32_t> gold_standard_tilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;
    for (int x = 0; x < num_rows; x += 32) {
        for (int y = 0; y < num_cols; y += 16) {
            int start = x * num_cols + y;

            // Top faces
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            // Bottom faces
            start += 16 * num_cols;
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }
        }
    }

    return dst_vec;
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
        std::array<uint32_t, 4> shape = {1, 64, 31, 32};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::CHANNELS_LAST, device);
        Tensor b = tilize_with_zero_padding(a).to(host);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.to(host); // Move tensor a to host to validate
        //vector<uint32_t> golden = gold_standard_tilize(src_vec, {num_tiles_r * 32, num_tiles_c * 32});
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
