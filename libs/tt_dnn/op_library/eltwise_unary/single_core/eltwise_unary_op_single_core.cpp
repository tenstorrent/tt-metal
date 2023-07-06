#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_unary_single_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param /* = {} */) {
    Program program{};

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );
    // no need to create c_in2 buffer since we pass scaler=0 to reader

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1 // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = eltwise_unary_op_utils::get_op_approx_mode(op_type);
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_unary_op_utils::add_defines(eltwise_unary_kernel, op_type, param);

    auto src_dram_buffer = a.buffer();
    auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

    auto dst_dram_buffer = output.buffer();
    auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

    SetRuntimeArgs(
        unary_reader_kernel,
        core,
        {
            src_dram_buffer->address(),
            uint32_t(src_dram_noc_xy.x),
            uint32_t(src_dram_noc_xy.y),
            num_tiles,
            0, 0, 0, 0, 0  // TODO(AP): [8] is scaler
        }
    );

    SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            uint32_t(dst_dram_noc_xy.x),
            uint32_t(dst_dram_noc_xy.y),
            num_tiles
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel, unary_writer_kernel](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(unary_reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = uint32_t(src_dram_noc_xy.x);
            runtime_args[2] = uint32_t(src_dram_noc_xy.y);
            SetRuntimeArgs(unary_reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(unary_writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
            runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
            SetRuntimeArgs(unary_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
