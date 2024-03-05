// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {
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

inline void check_tensor(const Tensor& tensor, const std::string& op_name) {
    TT_ASSERT(tensor.get_layout() == Layout::TILE, fmt::format("{} only supports tiled layout.", op_name));
    TT_ASSERT(tensor.get_dtype() == DataType::BFLOAT16, fmt::format("{} only supports bfloat16.", op_name));
    TT_ASSERT(
        tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to {} need to be on device!", op_name));
    TT_ASSERT(
        tensor.buffer() != nullptr, fmt::format("Operands to {} need to be allocated in buffers on device!", op_name));
}
}  // namespace

operation::ProgramWithCallbacks moreh_layernorm_impl(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    Tensor& output,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> beta,
    const std::optional<std::reference_wrapper<const Tensor>> mean,
    const std::optional<std::reference_wrapper<const Tensor>> rstd) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device* device = input.device();
    Program program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_legacy_shape();

    const bool is_lastdim_layernorm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto input_shape_without_padding = input_shape.without_padding();

    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_H = input_shape_without_padding[2];
    const auto origin_W = input_shape_without_padding[3];

    auto adjusted_input_shape = input_shape;
    if (normalized_dims == 2) {
        // HW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, C, TILE_HEIGHT, Ht * Wt * TILE_WIDTH)
        adjusted_input_shape[2] = TILE_HEIGHT;
        adjusted_input_shape[3] = (input_shape[2] / TILE_HEIGHT) * input_shape[3];
    } else if (normalized_dims == 3) {
        // CHW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, 1, TILE_HEIGHT, C * Ht * Wt * TILE_WIDTH)
        adjusted_input_shape[1] = 1;
        adjusted_input_shape[2] = TILE_HEIGHT;
        adjusted_input_shape[3] = input_shape[1] * (input_shape[2] / TILE_HEIGHT) * input_shape[3];
    } else if (normalized_dims == 4) {
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

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;

    // This could be inefficient.
    // If Wt is 65, the block_size will be 5. Then, the number of iteration is 13.
    // It can be 8 * 8 + 1, so the number of iterations is 9. It's more efficient.
    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = find_divisor_with_max_block_size(Wt, MAX_BLOCK_SIZE);

    const auto gamma_has_value = gamma.has_value();
    const auto beta_has_value = beta.has_value();
    const auto mean_has_value = mean.has_value();
    const auto rstd_has_value = rstd.has_value();

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    const auto mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto NCHt = N * C * Ht;
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {core_grid.x_, num_cores_y};

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
    uint32_t in0_t = Wt;                                          // input
    const uint32_t in1_t = 1;                                     // scaler
    const uint32_t in2_t = 1;                                     // epsilon
    const uint32_t in3_t = gamma_has_value ? 2 * block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 2 * block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                     // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                     // mask_w

    const uint32_t out0_t = 2 * block_size;          // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = Wt;                                                              // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const uint32_t cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t +
                               im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                              single_tile_size;
    const uint32_t available_L1 = device->l1_size_per_core() - L1_UNRESERVED_BASE;
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_layernorm algorithm is selected.");
        in0_t = 2 * block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(LogTest, "Small moreh_layernorm algorithm is selected.");
    }

    CreateCircularBuffer(
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
            {CB::c_out1, out1_t},      // mean
            {CB::c_out2, out2_t},      // rstd
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
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(gamma)),
        static_cast<uint32_t>(is_dram(beta)),
        block_size};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(output)),
        static_cast<uint32_t>(is_dram(mean)),
        static_cast<uint32_t>(is_dram(rstd)),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        block_size};

    std::map<string, string> reader_defines{};
    if (gamma_has_value) {
        reader_defines["GAMMA_HAS_VALUE"] = "1";
    }
    if (beta_has_value) {
        reader_defines["BETA_HAS_VALUE"] = "1";
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

    const auto reader_kernels_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

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
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    const auto compute_kernel_file =
        use_large_algorithm ? "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_large_kernel.cpp"
                            : "tt_eager/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_small_kernel.cpp";

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_H,
            origin_W,
            Wt,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
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

    if (normalized_dims == 2) {
        // HW
        scaler.f = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_ht * f_wt));
    } else if (normalized_dims == 3) {
        // CHW
        scaler.f = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_c * f_ht * f_wt));
    } else if (normalized_dims == 4) {
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

    const auto gamma_addr = gamma_has_value ? gamma->get().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta->get().buffer()->address() : 0;
    const auto mean_addr = mean_has_value ? mean->get().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd->get().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            input_addr, num_rows_per_core, Wt, tile_offset, scaler.u, e.u, gamma_addr, beta_addr, mask_h, mask_w};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            output_addr, mean_addr, rstd_addr, num_rows_per_core, Wt, tile_offset};
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
        auto mean_buffer = input_buffers.at(3);
        auto rstd_buffer = input_buffers.at(4);

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
                if (mean_buffer != nullptr) {
                    runtime_args[1] = mean_buffer->address();
                }
                if (rstd_buffer != nullptr) {
                    runtime_args[2] = rstd_buffer->address();
                }
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
        input_tensors.size() == 1 and optional_input_tensors.size() <= 4, "Must have between 1 to 5 input tensors");

    const auto& input = input_tensors.at(0);

    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& mean = optional_input_tensors.at(2);
    const auto& rstd = optional_input_tensors.at(3);

    check_tensor(input, "moreh_layernorm");

    TT_ASSERT(this->normalized_dims > 0);
    TT_ASSERT(this->normalized_dims <= input.get_legacy_shape().rank());

    if (gamma.has_value()) {
        check_tensor(gamma.value(), "moreh_layernorm");
        TT_ASSERT(
            input.get_legacy_shape()[3] == gamma.value().get_legacy_shape()[3],
            fmt::format("{} != {}", input.get_legacy_shape()[3], gamma.value().get_legacy_shape()[3]));
        TT_ASSERT(
            input.get_legacy_shape().without_padding()[3] == gamma.value().get_legacy_shape().without_padding()[3],
            fmt::format("{} != {}", input.get_legacy_shape().without_padding()[3], gamma.value().get_legacy_shape().without_padding()[3]));
        TT_ASSERT(input.device() == gamma.value().device());
    }

    if (beta.has_value()) {
        check_tensor(beta.value(), "moreh_layernorm");
        TT_ASSERT(
            input.get_legacy_shape()[3] == beta.value().get_legacy_shape()[3],
            fmt::format("{} != {}", input.get_legacy_shape()[3], beta.value().get_legacy_shape()[3]));
        TT_ASSERT(
            input.get_legacy_shape().without_padding()[3] == beta.value().get_legacy_shape().without_padding()[3],
            fmt::format("{} != {}", input.get_legacy_shape().without_padding()[3], beta.value().get_legacy_shape().without_padding()[3]));
        TT_ASSERT(input.device() == beta.value().device());
    }

    if (mean.has_value()) {
        check_tensor(mean.value(), "moreh_layernorm");
    }

    if (rstd.has_value()) {
        check_tensor(rstd.value(), "moreh_layernorm");
    }
}

std::vector<Shape> MorehLayerNorm::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> MorehLayerNorm::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehLayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input = input_tensors.at(0);

    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    auto& mean = optional_input_tensors.at(2);
    auto& rstd = optional_input_tensors.at(3);

    auto& output = output_tensors.at(0);

    return moreh_layernorm_impl(input, this->normalized_dims, this->eps, output, gamma, beta, mean, rstd);
}

Tensor moreh_layernorm(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> beta,
    const std::optional<std::reference_wrapper<const Tensor>> mean,
    const std::optional<std::reference_wrapper<const Tensor>> rstd,
    const MemoryConfig& output_mem_config) {
    return operation::run(
               MorehLayerNorm{
                   .normalized_dims = normalized_dims, .eps = eps, .output_mem_config = std::move(output_mem_config)},
               {input},
               {gamma, beta, mean, rstd})
        .at(0);
}

}  // namespace primary

}  // namespace operations

namespace tt_metal {

Tensor moreh_layernorm(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> beta,
    const std::optional<std::reference_wrapper<const Tensor>> mean,
    const std::optional<std::reference_wrapper<const Tensor>> rstd,
    const MemoryConfig& output_mem_config) {
    return operation::run_with_autoformat(
               operations::primary::MorehLayerNorm{
                   .normalized_dims = normalized_dims, .eps = eps, .output_mem_config = std::move(output_mem_config)},
               {input},
               {gamma, beta, mean, rstd})
        .at(0);
}

}  // namespace tt_metal

}  // namespace tt
