#include "tt_dnn/op_library/reduce/reduce_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Program reduce_single_core(const Tensor &a, Tensor& output, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {

    const auto shape = a.shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    uint32_t HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    if (reduce_dim == ReduceOpDim::HW)
        TT_ASSERT(scaler == 1.0f && "ReduceHW currently only works correctly with scaler == 1.0f!");

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to reduce op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        CB::c_in2,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );
    // no need to create c_in2 buffer since we pass scaler=0 to reader

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        reduce_dim == ReduceOpDim::H ?
            "tt_metal/kernels/dataflow/reader_unary_transpose_wh_8bank.cpp" :
            "tt_metal/kernels/dataflow/reader_unary_8bank_reduce.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
        Ht, // Ht
        Wt, // Wt
        NC, // NC
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    TT_ASSERT(int(reduce_dim) >= 0 && int(reduce_dim) <= ReduceOpDim::all().size());

    string compute_kernel_name = reduce_op_utils::dim_to_kernel_name(reduce_dim, reduce_op);

    auto reduce_compute_kernel = tt_metal::CreateComputeKernel(
        program,
        compute_kernel_name,
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    reduce_op_utils::add_defines(reduce_compute_kernel, reduce_op, reduce_dim);

    tt_metal::WriteRuntimeArgsToDevice(
        device, reader_kernel, core,
        {
            a.buffer()->address(),
            0, // unused by multibank reader
            0, // unused by multibank reader
            num_tensor_tiles, NC, Ht, Wt, Ht*Wt,
            uint32_t(*reinterpret_cast<uint32_t*>(&scaler)),
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

    // output does not hold any data, contains pointer to buffer on device with the data
    return program;
}

}  // namespace tt_metal

}  // namespace tt
