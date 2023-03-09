#include "tt_metal/op_library/reduce/reduce_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace {

// Only used when scaler is not 1.0
const char* dim_to_llk_kernel_name[] = {
    "kernels/compute/3T/reduce_h_sum",
    "kernels/compute/3T/reduce_w_sum",
    "kernels/compute/3T/reduce_hw_sum" };
}

using namespace tt::constants;
using u32 = std::uint32_t;

namespace tt {

namespace tt_metal {

Tensor reduce(const Tensor &a, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;
    if (reduce_dim == ReduceOpDim::HW)
        TT_ASSERT(scaler == 1.0f && "ReduceHW currently only works correctly with scaler == 1.0f!");

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto outshape = a.shape();
    switch(reduce_dim) {
        case ReduceOpDim::W: outshape[3] = 32; break;
        case ReduceOpDim::H: outshape[2] = 32; break;
        case ReduceOpDim::HW: outshape[2] = outshape[3] = 32; break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    tt_metal::Tensor output = tt_metal::Tensor(outshape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        reduce_dim == ReduceOpDim::H ?
            "kernels/dataflow/reader_unary_transpose_wh_8bank.cpp" :
            "kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
        uint(Ht), // Ht
        uint(Wt), // Wt
        uint(NC), // NC
    };
    tt_metal::ComputeKernelArgs *compute_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    TT_ASSERT(int(reduce_dim) >= 0 && int(reduce_dim) <= ReduceOpDim::all().size());

    string compute_kernel = ::dim_to_llk_kernel_name[int(reduce_dim)];

    auto reduce_h_compute_kernel = tt_metal::CreateComputeKernel(
        program,
        compute_kernel,
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    bool do_max = false;
    TT_ASSERT(do_max == false, "Max not yet supported");
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    reduce_h_compute_kernel->add_define("REDUCE_OP", reduce_op == ReduceOpMath::MAX ? 2 : 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CompileProgramNew(device, program);


    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device, reader_kernel, core,
            {
                a.buffer()->address(),
                0, // unused by multibank reader
                0, // unused by multibank reader
                num_tensor_tiles, NC, Ht, Wt, Ht*Wt
            }
        );

        uint32_t out_dim_divider = 1;
        switch (reduce_dim) {
            case ReduceOpDim::H: out_dim_divider = Ht; break;
            case ReduceOpDim::W: out_dim_divider = Wt; break;
            case ReduceOpDim::HW: out_dim_divider = Ht*Wt; break;
            default: TT_ASSERT(false && "Unsupported reduce_dim!");
        }

        tt_metal::WriteRuntimeArgsToDevice(
            device, writer_kernel, core,
            {
                output.buffer()->address(),
                0, // unused by multibank writer
                0, // unused by multibank writer
                num_tensor_tiles/out_dim_divider
            }
        );

    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
