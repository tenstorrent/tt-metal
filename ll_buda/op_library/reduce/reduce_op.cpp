#include "ll_buda/op_library/reduce/reduce_op.hpp"

#include "ll_buda/host_api.hpp"
#include "constants.hpp"

namespace reduce_args {
// FIXME:copy pasted the args here from the kernel file
struct hlk_args_t {
    // per-batch params
    int Ht; // number of tiles in H to expect (expected to be a full tensor by this kernel)
    int Wt; // number of tiles in W to expect (can be a partial tensor), always <= DSTt
    int NC;
    float scaler;
};
}

namespace {
const char* dim_to_kernel_name[] = {
    "kernels/compute/reduce_h.cpp",
    "kernels/compute/reduce_w.cpp",
    "kernels/compute/reduce_hw.cpp" };
}

using namespace tt::constants;
using u32 = std::uint32_t;

namespace tt {

namespace ll_buda {

Tensor reduce(const Tensor &a, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;
    if (reduce_dim == ReduceOpDim::HW)
        TT_ASSERT(scaler == 1.0f && "Reduce_HW currently only works correctly with scaler == 1.0f!");

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    auto outshape = a.shape();
    switch(reduce_dim) {
        case ReduceOpDim::W: outshape[3] = 32; break;
        case ReduceOpDim::H: outshape[2] = 32; break;
        case ReduceOpDim::HW: outshape[2] = outshape[3] = 32; break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    ll_buda::Tensor output = ll_buda::Tensor(outshape, a.dtype(), tt::ll_buda::Layout::TILE, device);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
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
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    ll_buda::DataMovementKernel *reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        reduce_dim == ReduceOpDim::H ?
            "kernels/dataflow/reader_unary_transpose_wh_8bank.cpp" :
            "kernels/dataflow/reader_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new reduce_args::hlk_args_t{ .Ht = int(Ht), .Wt = int(Wt), .NC = int(NC), .scaler = scaler };
    ll_buda::ComputeKernelArgs *compute_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(reduce_args::hlk_args_t));
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    TT_ASSERT(int(reduce_dim) >= 0 && int(reduce_dim) <= ReduceOpDim::all().size());
    auto reduce_h_compute_kernel = ll_buda::CreateComputeKernel(
        program,
        ::dim_to_kernel_name[int(reduce_dim)],
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    bool do_max = false;
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    reduce_h_compute_kernel->add_define("REDUCE_OP", reduce_op == ReduceOpMath::MAX ? 2 : 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
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

        ll_buda::WriteRuntimeArgsToDevice(
            device, writer_kernel, core,
            {
                output.buffer()->address(),
                0, // unused by multibank writer
                0, // unused by multibank writer
                num_tensor_tiles/out_dim_divider
            }
        );

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
