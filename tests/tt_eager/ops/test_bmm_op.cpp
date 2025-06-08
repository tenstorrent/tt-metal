// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstring>
#include <exception>
#include <optional>

#include <tt-metalium/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device_owner = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto device = device_owner.get();

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t Mt = 3;
        uint32_t Kt = 2;
        uint32_t Nt = 4;
        uint32_t B = 5;
        ttnn::Shape shapea({B, 1, Mt * TILE_HEIGHT, Kt * TILE_WIDTH});
        ttnn::Shape shapeb({B, 1, Kt * TILE_HEIGHT, Nt * TILE_WIDTH});
        ttnn::Shape shapeb1({1, 1, Kt * TILE_HEIGHT, Nt * TILE_WIDTH});

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = ttnn::random::random(shapea).to_layout(Layout::TILE).to_device(device);
        Tensor b = ttnn::zeros(shapeb, DataType::BFLOAT16, Layout::TILE, *device);
        Tensor b1 = ttnn::zeros(shapeb1, DataType::BFLOAT16, Layout::TILE, *device);

        Tensor mm = ttnn::operations::matmul::matmul(
                        a,
                        b,
                        /*bias=*/std::nullopt,
                        ttnn::operations::matmul::Matmul{
                            /*program_config=*/std::nullopt,
                            /*bcast_batch=*/std::nullopt,
                            tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                            /*output_dtype=*/std::nullopt,
                            /*compute_kernel_config=*/std::nullopt,
                            /*untilize_out=*/false,
                            /*user_core_coord=*/std::nullopt,
                            /*user_fused_activation=*/std::nullopt,
                            /*user_run_batched=*/true})
                        .cpu();
        Tensor mm1 = ttnn::operations::matmul::matmul(a, b1).cpu();

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.cpu();  // Move tensor a to host to validate
    } catch (const std::exception& e) {
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

    TT_FATAL(pass, "Error");

    return 0;
}
