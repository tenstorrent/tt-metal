// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
operation::ProgramWithCallbacks layernorm_(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    bool rms_norm = false
) {

    const auto shape = a.shape();
    uint32_t W = shape[3], H = shape[2];
    uint32_t HW = H*W;
    uint32_t NC = a.volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.dtype();

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    uint32_t block_size = find_max_divisor(Wt, 8);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_WIDTH : 0;
    }
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_WIDTH : 0;
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();


    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // TODO(AP): this will not work for all Wts possibly, but should work for Wt=8, 12, 16, 32
    // TODO(AP): can also add support for block_size=7 -> 63, 28
    uint32_t WtB    =  div_up(Wt, block_size)*block_size; // Wt padded to be divisible by block size
    uint32_t in0_t  =  WtB; // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t  =  block_size*2; // buffer for fused pre-add b tensor
    uint32_t out0_t =  block_size*2;
    uint32_t im0_t  =  WtB; // buffer for saving xmm
    uint32_t im3_t  =  WtB; // buffer for xmm^2
    uint32_t in5_t  =  WtB; // buffer for gamma
    uint32_t in6_t  =  WtB; // buffer for beta
    uint32_t im6_t  =  block_size*2; // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = WtB;
        //cout << "im6_t=WtB=" << WtB << endl;
        in0_t = 2*block_size;
    }
    uint32_t im5_t  =  2*block_size; // for buffering to/from *gamma/+beta
    uint32_t im4_t  =  8; // 8 just in case, 4 would prob suffice
    uint32_t im1_t  =  2;
    uint32_t in2_t  =  2; // scaler for reduce coming from reader
    uint32_t in3_t  =  2; // epsilon coming from reader
    uint32_t im2_t  =  2; //

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in1_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im3_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in5_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT(num_gamma_tiles % block_size == 0);
    TT_ASSERT(num_beta_tiles % block_size == 0);

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(a),
        (std::uint32_t) is_dram(b),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_size
    };

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)log2(gamma_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)log2(beta_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(output),
        (std::uint32_t) block_size
    };


    bool tile_dtype_is_bfloat16 = a.dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> reader_defines;
    std::map<string, string> eltwise_binary_defines;
    if (b) {
        reader_defines["FUSE_PRE_ADD"] = "1";
        eltwise_binary_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    auto use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    auto reader_kernels_id = CreateKernel(
        program,
        use_row_major_kernel ? "tt_eager/tt_dnn/op_library/layernorm/kernels/reader_unary_interleaved_ln_rm_gb.cpp" : "tt_eager/tt_dnn/op_library/layernorm/kernels/reader_unary_interleaved_ln.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines = reader_defines}
    );

    auto writer_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args}
    );

    vector<uint32_t> compute_args = { Wt, block_size, gamma.has_value(), beta.has_value() };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto compute_kernels_id = CreateKernel(
        program,
        rms_norm ? "tt_eager/tt_dnn/kernels/compute/rmsnorm.cpp" : "tt_eager/tt_dnn/kernels/compute/layernorm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args, .defines = eltwise_binary_defines}
    );

    // Create circular buffers
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_t*single_tile_size, {{CB::c_in0, cb_data_format}}).set_page_size(CB::c_in0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_src0_config );
    CircularBufferConfig cb_out0_config = CircularBufferConfig(out0_t*single_tile_size, {{CB::c_out0, cb_data_format}}).set_page_size(CB::c_out0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_out0_config );
    CircularBufferConfig cb_intermed1_config = CircularBufferConfig(im1_t*single_tile_size, {{CB::c_intermed1, cb_data_format}}).set_page_size(CB::c_intermed1, single_tile_size);
    CreateCircularBuffer( program, all_cores,  cb_intermed1_config );
    CircularBufferConfig cb_in2_config = CircularBufferConfig(in2_t*bfloat16_tile_size, {{CB::c_in2, DataFormat::Float16_b}}).set_page_size(CB::c_in2, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in2_config );
    CircularBufferConfig cb_in3_config = CircularBufferConfig(in3_t*bfloat16_tile_size, {{CB::c_in3, DataFormat::Float16_b}}).set_page_size(CB::c_in3, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in3_config );
    CircularBufferConfig cb_intermed2_config = CircularBufferConfig(im2_t*single_tile_size, {{CB::c_intermed2, cb_data_format}}).set_page_size(CB::c_intermed2, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_intermed2_config );
    if (!rms_norm) {
        CircularBufferConfig cb_intermed0_config = CircularBufferConfig(im0_t*single_tile_size, {{CB::c_intermed0, cb_data_format}}).set_page_size(CB::c_intermed0, single_tile_size);
        CreateCircularBuffer( program, all_cores, cb_intermed0_config );
    }
    CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t*single_tile_size, {{CB::c_intermed3, cb_data_format}}).set_page_size(CB::c_intermed3, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed3_config );
    CircularBufferConfig c_intermed4_config = CircularBufferConfig(im4_t*single_tile_size, {{CB::c_intermed4, cb_data_format}}).set_page_size(CB::c_intermed4, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed4_config );
    if (gamma.has_value() || beta.has_value()) {
        CircularBufferConfig c_intermed5_config = CircularBufferConfig(im5_t*single_tile_size, {{CB::c_intermed5, cb_data_format}}).set_page_size(CB::c_intermed5, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_intermed5_config );
    }
    if (gamma.has_value()) {
        uint32_t c_in5_page_size = gamma.value().layout() == Layout::ROW_MAJOR ? bfloat16_tile_size : single_tile_size;
        DataFormat c_in5_df = gamma.value().layout() == Layout::ROW_MAJOR ? DataFormat::Float16_b : cb_data_format;
        CircularBufferConfig c_in5_config = CircularBufferConfig(in5_t * c_in5_page_size, {{CB::c_in5, c_in5_df}})
            .set_page_size(CB::c_in5, c_in5_page_size);
        CreateCircularBuffer( program, all_cores, c_in5_config );
    }
    if (beta.has_value()) {
        uint32_t c_in6_page_size = beta.value().layout() == Layout::ROW_MAJOR ? bfloat16_tile_size : single_tile_size;
        DataFormat c_in6_df = beta.value().layout() == Layout::ROW_MAJOR ? DataFormat::Float16_b : cb_data_format;
        CircularBufferConfig c_in6_config = CircularBufferConfig(in6_t * c_in6_page_size, {{CB::c_in6, c_in6_df}})
            .set_page_size(CB::c_in6, c_in6_page_size);
        CreateCircularBuffer( program, all_cores, c_in6_config );
    }
    if (b) {
        // x = a+b in this notation
        // result = ln(x)*gamma + beta
        // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is used as x
        // b is buffered into c_in1
        CircularBufferConfig c_intermed6_config = CircularBufferConfig(im6_t*single_tile_size, {{CB::c_intermed6, cb_data_format}}).set_page_size(CB::c_intermed6, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_intermed6_config );
        // c_in1 is input buffer for b
        CircularBufferConfig c_in1_config = CircularBufferConfig(in1_t*single_tile_size, {{CB::c_in1, cb_data_format}}).set_page_size(CB::c_in1, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_in1_config);
    }

    uint32_t curr_row = 0;
    float winv = 1.0f / W; // bcast-w scaler
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps; // epsilon
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;

        SetRuntimeArgs(reader_kernels_id, core,
            { a_addr, num_tile_rows_per_core, Wt, tile_offset, packed_winv_value, e.u, // 0-5
            gamma_dram_addr, beta_dram_addr, b_dram_addr } // 6-8
        );
        SetRuntimeArgs(compute_kernels_id, core, { num_tile_rows_per_core });
        SetRuntimeArgs(writer_kernels_id, core, { dst_addr, num_tile_rows_per_core * Wt, tile_offset } );
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid_size
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_a_dram_buffer = input_buffers.at(0);
        auto src_b_dram_buffer = input_buffers.at(1);
        auto gamma_dram_buffer = input_buffers.at(2);
        auto beta_dram_buffer = input_buffers.at(3);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            {
                auto &runtime_args = GetRuntimeArgs(reader_kernel_id, core);
                runtime_args[0] = src_a_dram_buffer->address();
                if (src_b_dram_buffer != nullptr) {
                    runtime_args[8] = src_b_dram_buffer->address();
                }
                if (gamma_dram_buffer != nullptr) {
                    runtime_args[6] = gamma_dram_buffer->address();
                }
                if (beta_dram_buffer != nullptr) {
                    runtime_args[7] = beta_dram_buffer->address();
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void LayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {

    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_FATAL(a.layout() == Layout::TILE);
    TT_FATAL(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE);
        TT_FATAL(a.shape() == b.value().shape());
        TT_FATAL(a.device() == b.value().device());
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(gamma.value().layout() == beta.value().layout(), "Gamma and beta must have the same layout!");
    }

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(a.shape()[3] == gamma.value().shape()[3], fmt::format("{} != {}", a.shape()[3], gamma.value().shape()[3]));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR);
            TT_FATAL((gamma.value().shape()[3] == TILE_WIDTH && gamma.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().dtype() == DataType::BFLOAT16);
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(a.shape()[3] == beta.value().shape()[3]);
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR);
            TT_FATAL((beta.value().shape()[3] == TILE_WIDTH && beta.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().dtype() == DataType::BFLOAT16);
        }
    }

}

std::vector<Shape> LayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps);

}

tt::stl::reflection::Attributes LayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}


void RMSNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_FATAL(a.layout() == Layout::TILE);
    TT_FATAL(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE);
        TT_FATAL(a.shape() == b.value().shape());
        TT_FATAL(a.device() == b.value().device());
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }
    if (gamma.has_value()) {
        TT_FATAL(gamma.value().layout() == Layout::TILE);
        TT_FATAL(a.shape()[3] == gamma.value().shape()[3]);
        TT_FATAL(a.device() == gamma.value().device());
        TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(gamma.value().shape()[2] == TILE_HEIGHT);
    }
    if (beta.has_value()) {
        TT_FATAL(beta.value().layout() == Layout::TILE);
        TT_FATAL(a.shape()[3] == beta.value().shape()[3]);
        TT_FATAL(a.device() == beta.value().device());
        TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(beta.value().shape()[2] == TILE_HEIGHT);
    }

}

std::vector<Shape> RMSNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> RMSNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks RMSNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps, true);

}

tt::stl::reflection::Attributes RMSNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace ll_buda

}  // namespace tt
