#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks bcast_single_core(const Tensor &a, const Tensor &b, Tensor& output, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");

    auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    // TODO: CHANGE TO FUNCTION CONVERSION
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }

    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(cb_data_format), (uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(cb_data_format), (uint32_t)dst_is_dram};

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::SINGLE_CORE);
    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        reader_name,
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank_start_id.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // TODO(AP): add dimensions and op params
    vector<uint32_t> compute_kernel_args = {
        NC, // B
        Ht, // Ht
        Wt  // Wt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);
    auto bcast_kernel = tt_metal::CreateComputeKernel(
        program,
        compute_name,
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    bcast_op_utils::add_defines(bcast_kernel, bcast_dim, bcast_math);

    uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;
    tt_metal::SetRuntimeArgs(
        binary_reader_kernel,
        core,
        {
            a.buffer()->address(), // 0
            0, // 1
            0, // 2
            num_tensor_tiles, // 3
            b.buffer()->address(), // 4
            0, // 5
            0, // 6
            num_btensor_tiles, NC*Ht*Wt, NC, Ht, Wt, bnc1  // 7 8 9 10 11 12
        }
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {
            output.buffer()->address(),
            0, 0, num_tensor_tiles, 0
        }
    );

    auto override_runtime_args_callback = [
            binary_reader_kernel,
            unary_writer_kernel
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);

        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(binary_reader_kernel, core);
            runtime_args[0] = src_dram_buffer_a->address();
            runtime_args[4] = src_dram_buffer_b->address();
            SetRuntimeArgs(binary_reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(unary_writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            SetRuntimeArgs(unary_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
