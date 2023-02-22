#include "ll_buda/host_api.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/op_library/bmm/bmm_op.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace ll_buda;
using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        ll_buda::Host *host = ll_buda::GetHost();

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t Mt = 3;
        uint32_t Kt = 2;
        uint32_t Nt = 4;
        uint32_t B = 5;
        std::array<uint32_t, 4> shapea = {B, 1, Mt*TILE_HEIGHT, Kt*TILE_WIDTH};
        std::array<uint32_t, 4> shapeb = {B, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};
        std::array<uint32_t, 4> shapeb1 = {1, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shapea, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
        Tensor b = Tensor(shapeb, Initialize::ZEROS, DataType::BFLOAT16, Layout::TILE, device);
        Tensor b1 = Tensor(shapeb1, Initialize::ZEROS, DataType::BFLOAT16, Layout::TILE, device);

        Tensor mm = bmm(a, b).to(host);
        Tensor mm1 = matmul(a, b1).to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.to(host); // Move tensor a to host to validate

        pass &= ll_buda::CloseDevice(device);;

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
