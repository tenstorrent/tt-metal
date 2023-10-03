// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"

#include <optional>
#include <utility>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

namespace {
inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

inline void are_valid_normalized_dims(const std::vector<uint32_t>& normalized_dims) {
    // We assume that tensor is 4D.
    if (normalized_dims.size() == 1) {
        TT_ASSERT(normalized_dims.at(0) == 3);
    } else if (normalized_dims.size() == 2) {
        TT_ASSERT(normalized_dims.at(0) == 2);
        TT_ASSERT(normalized_dims.at(1) == 3);
    } else if (normalized_dims.size() == 3) {
        TT_ASSERT(normalized_dims.at(0) == 1);
        TT_ASSERT(normalized_dims.at(1) == 2);
        TT_ASSERT(normalized_dims.at(2) == 3);
    } else if (normalized_dims.size() == 4) {
        TT_ASSERT(normalized_dims.at(0) == 0);
        TT_ASSERT(normalized_dims.at(1) == 1);
        TT_ASSERT(normalized_dims.at(2) == 2);
        TT_ASSERT(normalized_dims.at(3) == 3);
    } else {
        TT_ASSERT("Not supported case yet.");
    }
}

inline uint32_t find_divisor_with_max_block_size(uint32_t val, uint32_t max_block_size) {
    uint32_t divisor{1};
    for (uint32_t current_divisor = max_block_size; current_divisor >= 1; current_divisor--) {
        if (val % current_divisor == 0) {
            divisor = current_divisor;
            break;
        }
    }
    return divisor;
}
}  // namespace

operation::ProgramWithCallbacks moreh_layernorm_(
    const Tensor& input,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    const std::vector<uint32_t>& normalized_dims) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device* device = input.device();
    Program program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.shape();

    are_valid_normalized_dims(normalized_dims);

    const bool is_lastdim_layernorm = normalized_dims.size() == 1;

    const auto input_shape_without_padding = input_shape.without_padding();

    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_H = input_shape_without_padding[2];
    const auto origin_W = input_shape_without_padding[3];

    const auto origin_Ht = tt::div_up(origin_H, TILE_HEIGHT);
    const auto origin_Wt = tt::div_up(origin_W, TILE_WIDTH);

    auto adjusted_input_shape = input_shape;
    if (normalized_dims.size() == 2) {
        // HW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, C, TILE_HEIGHT, Ht * Wt * TILE_WIDTH)
        adjusted_input_shape[2] = TILE_HEIGHT;
        adjusted_input_shape[3] = (input_shape[2] / TILE_HEIGHT) * input_shape[3];
    } else if (normalized_dims.size() == 3) {
        // CHW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, 1, TILE_HEIGHT, C * Ht * Wt * TILE_WIDTH)
        adjusted_input_shape[1] = 1;
        adjusted_input_shape[2] = TILE_HEIGHT;
        adjusted_input_shape[3] = input_shape[1] * (input_shape[2] / TILE_HEIGHT) * input_shape[3];
    } else if (normalized_dims.size() == 4) {
        // NCHW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (1, 1, TILE_HEIGHT, N * C * Ht * Wt * TILE_WIDTH)
        adjusted_input_shape[0] = 1;
        adjusted_input_shape[1] = 1;
        adjusted_input_shape[2] = TILE_HEIGHT;
        adjusted_input_shape[3] = input_shape[0] * input_shape[1] * (input_shape[2] / TILE_HEIGHT) * input_shape[3];
    } else {
        TT_ASSERT(is_lastdim_layernorm);
    }

    const auto N = adjusted_input_shape[0];
    const auto C = adjusted_input_shape[1];
    const auto H = adjusted_input_shape[2];
    const auto W = adjusted_input_shape[3];

    const auto NC = N * C;
    const auto HW = H * W;

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    // This could be inefficient.
    // If Wt is 65, the block_size will be 5. Then, the number of iteration is 13.
    // It can be 8 * 8 + 1, so the number of iterations is 9. It's more efficient.
    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = find_divisor_with_max_block_size(Wt, MAX_BLOCK_SIZE);

    const auto gamma_has_value = gamma.has_value();
    const auto beta_has_value = beta.has_value();

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    uint32_t in0_t = 2 * Wt;                                      // input
    const uint32_t in1_t = 2;                                     // scaler
    const uint32_t in2_t = 2;                                     // epsilon
    const uint32_t in3_t = gamma_has_value ? 2 * block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 2 * block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 2 : 0;                     // mask_h
    const uint32_t in6_t = do_mask_w ? 2 : 0;                     // mask_w

    const uint32_t out0_t = 2 * block_size;  // output

    const uint32_t im0_t = 2;                                                         // E[x]
    uint32_t im1_t = 2 * Wt;                                                          // x - E[x]
    uint32_t im2_t = 2 * Wt;                                                          // (x - E[x])^2
    uint32_t im3_t = 0;                                                               // Sum[(x - E[x])^2]
    const uint32_t im4_t = 2;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 8;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const uint32_t cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + im0_t + im1_t + im2_t +
                               im3_t + im4_t + im5_t + im6_t + im7_t) *
                              single_tile_size;
    const uint32_t available_L1 = device->l1_size() - L1_UNRESERVED_BASE;
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_layernorm algorithm is selected.");
        in0_t = 2 * block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
        im3_t = 2;
    } else {
        log_info(LogTest, "Small moreh_layernorm algorithm is selected.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto NCHt = NC * Ht;
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {.x = core_grid.x_, .y = num_cores_y};

    // core_group_2 works more.
    // If number of working cores is 108 and NCHt is 110,
    // core_group_2[(x=0, y=0), (x=0, y=1)] works for 2 rows. Others work for 1 row.
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, NCHt);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},        // input
            {CB::c_in1, in1_t},        // scaler
            {CB::c_in2, in2_t},        // epsilon
            {CB::c_in3, in3_t},        // gamma
            {CB::c_in4, in4_t},        // beta
            {CB::c_in5, in5_t},        // mask_h
            {CB::c_in6, in6_t},        // mask_w
            {CB::c_out0, out0_t},      // output
            {CB::c_intermed0, im0_t},  // E[x]
            {CB::c_intermed1, im1_t},  // x - E[x]
            {CB::c_intermed2, im2_t},  // (x - E[x])^2
            {CB::c_intermed3, im3_t},  // Sum[(x - E[x])^2]
            {CB::c_intermed4, im4_t},  // E[(x - E[x])^2] = Var[x]
            {CB::c_intermed5, im5_t},  // 1.0/(sqrt(Var[x] + eps))
            {CB::c_intermed6, im6_t},  // y * gamm + beta
            {CB::c_intermed7, im7_t},  // Sum[x]
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        (uint32_t)(is_dram(input)), (uint32_t)(is_dram(gamma)), (uint32_t)(is_dram(beta)), block_size};

    const std::vector<uint32_t> writer_compile_time_args{(uint32_t)(is_dram(output)), block_size};

    std::map<string, string> reader_defines{};
    if (gamma_has_value) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta_has_value) {
        reader_defines["FUSE_BETA"] = "1";
    }
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
    }
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }

    const auto reader_kernel_file =
        use_large_algorithm ? "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/reader_moreh_layernorm_large.cpp"
                            : "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/reader_moreh_layernorm_small.cpp";
    const auto writer_kernel_file = "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/writer_moreh_layernorm.cpp";

    const auto reader_kernels_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernels_id =
        tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    if (is_lastdim_layernorm) {
        compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    } else {
        compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    }

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_H,
        origin_W,
        Wt,
        block_size,
        gamma_has_value,
        beta_has_value,
        is_lastdim_layernorm};

    const auto compute_kernel_file =
        use_large_algorithm ? "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_large.cpp"
                            : "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_small.cpp";

    tt::operations::primary::CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_H,
            origin_W,
            Wt,
            block_size,
            gamma_has_value,
            beta_has_value,
            is_lastdim_layernorm};

        tt::operations::primary::CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f / static_cast<float>(origin_W);  // scaler

    const auto f_n = static_cast<float>(origin_N);
    const auto f_c = static_cast<float>(origin_C);
    const auto f_ht = static_cast<float>(origin_H) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_W) / static_cast<float>(TILE_WIDTH);

    if (normalized_dims.size() == 2) {
        // HW
        scaler.f = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_ht * f_wt));
    } else if (normalized_dims.size() == 3) {
        // CHW
        scaler.f = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_c * f_ht * f_wt));
    } else if (normalized_dims.size() == 4) {
        // NCHW
        scaler.f = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_n * f_c * f_ht * f_wt));
    }

    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;  // epsilon

    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

    const auto mask_h = origin_H % TILE_HEIGHT;
    const auto mask_w = origin_W % TILE_WIDTH;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            input_addr, num_rows_per_core, Wt, tile_offset, scaler.u, e.u, gamma_addr, beta_addr, mask_h, mask_w};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{output_addr, num_rows_per_core, Wt, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * Wt;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto input_buffer = input_buffers.at(0);
        auto gamma_buffer = input_buffers.at(1);
        auto beta_buffer = input_buffers.at(2);

        auto ouput_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = input_buffer->address();
                if (gamma_buffer != nullptr) {
                    runtime_args[6] = gamma_buffer->address();
                }
                if (beta_buffer != nullptr) {
                    runtime_args[7] = beta_buffer->address();
                }
                SetRuntimeArgs(program, reader_kernels_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = ouput_buffer->address();
                SetRuntimeArgs(program, writer_kernels_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void MorehLayerNorm::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 1 and optional_input_tensors.size() <= 2, "Must have between 1 to 3 input tensors");

    const auto& input = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);

    TT_ASSERT(input.layout() == Layout::TILE, "moreh_layernorm only supports tiled layout.");
    TT_ASSERT(input.dtype() == DataType::BFLOAT16, "moreh_layernorm only supports bfloat16.");
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Operands to moreh_layernorm need to be on device!");
    TT_ASSERT(input.buffer() != nullptr, "Operands to moreh_layernorm need to be allocated in buffers on device!");

    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().layout() == Layout::TILE, "moreh_layernorm only supports tiled layout.");
        TT_ASSERT(gamma.value().dtype() == input.dtype(), "moreh_layernorm only supports bfloat16.");
        TT_ASSERT(
            input.shape()[3] == gamma.value().shape()[3],
            fmt::format("{} != {}", input.shape()[3], gamma.value().shape()[3]));
        TT_ASSERT(
            input.shape().without_padding()[3] == gamma.value().shape().without_padding()[3],
            fmt::format("{} != {}", input.shape().without_padding()[3], gamma.value().shape().without_padding()[3]));
        TT_ASSERT(input.device() == gamma.value().device());
        TT_ASSERT(
            gamma.value().buffer() != nullptr,
            "Operands to moreh_layernorm need to be allocated in buffers on device!");
    }

    if (beta.has_value()) {
        TT_ASSERT(beta.value().layout() == Layout::TILE, "moreh_layernorm only supports tiled layout.");
        TT_ASSERT(beta.value().dtype() == input.dtype(), "moreh_layernorm only supports bfloat16.");
        TT_ASSERT(
            input.shape()[3] == beta.value().shape()[3],
            fmt::format("{} != {}", input.shape()[3], beta.value().shape()[3]));
        TT_ASSERT(
            input.shape().without_padding()[3] == beta.value().shape().without_padding()[3],
            fmt::format("{} != {}", input.shape().without_padding()[3], beta.value().shape().without_padding()[3]));
        TT_ASSERT(input.device() == beta.value().device());
        TT_ASSERT(
            beta.value().buffer() != nullptr, "Operands to moreh_layernorm need to be allocated in buffers on device!");
    }
}

std::vector<Shape> MorehLayerNorm::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> MorehLayerNorm::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehLayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    auto& output = output_tensors.at(0);
    return moreh_layernorm_(input, gamma, beta, output, this->eps, this->normalized_dims);
}

tt::stl::reflection::Attributes MorehLayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"normalized_dims", this->normalized_dims},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
