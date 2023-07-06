#include "tt_dnn/op_library/transpose/transpose_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks transpose_wh_single_core(const Tensor &a, Tensor& output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = a.volume()/TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();


    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

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

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_transpose_wh_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        Ht*Wt*NC // NHtWt
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/transpose_wh.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        reader_kernel,
        core,
        {
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tensor_tiles, NC, Ht, Wt, Ht*Wt,
            0 /*scaler*/
        }
    );

    tt_metal::SetRuntimeArgs(
        writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tensor_tiles
        }
    );

    auto override_runtime_args_callback = [reader_kernel, writer_kernel](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = uint32_t(src_dram_noc_xy.x);
            runtime_args[2] = uint32_t(src_dram_noc_xy.y);
            SetRuntimeArgs(reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
            runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
            SetRuntimeArgs(writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_hc_single_core(const Tensor &a, Tensor &output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    u32 HW = H*W;
    u32 CHW = C*H*W;

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;
    u32 Ct = C/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

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

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/transpose_hc_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        num_tensor_tiles // num_tensor_tiles
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        reader_kernel,
        core,
        {
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            W, H, C, HW, N, CHW
        }
    );

    tt_metal::SetRuntimeArgs(
        writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tensor_tiles
        }
    );

    auto override_runtime_args_callback = [reader_kernel, writer_kernel](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = uint32_t(src_dram_noc_xy.x);
            runtime_args[2] = uint32_t(src_dram_noc_xy.y);
            SetRuntimeArgs(reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
            runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
            SetRuntimeArgs(writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_cn_single_core(const Tensor &a, Tensor &output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

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

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/transpose_cn_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        num_tensor_tiles // num_tensor_tiles
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        reader_kernel,
        core,
        {
            src0_dram_buffer->address(),
            N, C, Ht, Wt, HtWt, CHtWt, NCHtWt
        }
    );

    tt_metal::SetRuntimeArgs(
        writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tensor_tiles
        }
    );

    auto override_runtime_args_callback = [reader_kernel, writer_kernel](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            SetRuntimeArgs(reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
            runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
            SetRuntimeArgs(writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


operation::ProgramWithCallbacks transpose_single_core(const Tensor &a, Tensor &output, TransposeOpDim::Enum transpose_dim) {
    if (transpose_dim == TransposeOpDim::WH){
        return transpose_wh_single_core(a, output);
    } else if (transpose_dim == TransposeOpDim::HC) {
        return transpose_hc_single_core(a, output);
    } else if (transpose_dim == TransposeOpDim::CN) {
        return transpose_cn_single_core(a, output);
    } else {
        TT_THROW("Unsupported Transpose Op Dim");
    }
}

}  // namespace tt_metal

}  // namespace tt
