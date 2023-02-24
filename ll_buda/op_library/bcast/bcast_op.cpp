#include "ll_buda/op_library/bcast/bcast_op.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

#include "constants.hpp"

// TODO(AP): duplication
namespace bcast_op_params {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    uint32_t B;
    uint32_t Ht;
    uint32_t Wt;
};
}


using namespace tt::ll_buda;
using namespace tt::constants;
using u32 = std::uint32_t;

namespace {
const char* get_reader_name(BcastOpDim::Enum bcast_dim) {
    if (bcast_dim == BcastOpDim::H) {
        return "kernels/dataflow/reader_bcast_h_8bank.cpp";
    } else if (bcast_dim == BcastOpDim::W) {
        return "kernels/dataflow/reader_bcast_w_8bank.cpp";
    } if (bcast_dim == BcastOpDim::HW) {
        return "kernels/dataflow/reader_bcast_hw_8bank.cpp";
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "kernels/compute/bcast_hw.cpp";
        default:           TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

const char* math_to_op_define[] = { "add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast" };

}


namespace tt {

namespace ll_buda {

Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    u32 N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    u32 bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    u32 NC = N*C;
    u32 HW = H*W;

    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.volume() % TILE_HW == 0);

    // validate input dimensions
    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(N == bN && C == bC && H == bH && bW == TILE_WIDTH);
    if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(N == bN && C == bC && W == bW && bH == TILE_HEIGHT);
    if (bcast_dim == BcastOpDim::HW)
        TT_ASSERT(N == bN && C == bC && bW == TILE_WIDTH && bH == TILE_HEIGHT);

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    ll_buda::Tensor output = Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;
    auto cb_src1 = ll_buda::CreateCircularBuffer(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src1_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    const char* reader_name = ::get_reader_name(bcast_dim);
    ll_buda::DataMovementKernel *binary_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        reader_name,
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    // TODO(AP): add dimensions and op params
    void *hlk_args = new bcast_op_params::hlk_args_t { .B = NC, .Ht = Ht, .Wt = Wt };
    ll_buda::ComputeKernelArgs *compute_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(bcast_op_params::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    const char* compute_name = ::get_compute_name(bcast_dim);
    auto bcast_kernel = ll_buda::CreateComputeKernel(
        program,
        compute_name,
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    bcast_kernel->add_define("BCAST_OP", ::math_to_op_define[int(bcast_math)]);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {a.buffer()->address(), // 0
            0, // 1
            0, // 2
            num_tensor_tiles, // 3
            b.buffer()->address(), // 4
            0, // 5
            0, // 6
            num_btensor_tiles, NC*Ht*Wt, NC, Ht, Wt}); // 7 8 9 10 11

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            { output.buffer()->address(),
              0, 0, num_tensor_tiles});

    ll_buda::ConfigureDeviceWithProgram(device, program);

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
