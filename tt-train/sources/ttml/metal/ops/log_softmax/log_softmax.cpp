// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax.hpp"

#include <map>
#include <optional>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <variant>

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

#ifndef TTML_LOG_SOFTMAX_KERNEL_ROOT
#define TTML_LOG_SOFTMAX_KERNEL_ROOT "tt-train/sources/ttml/metal/ops/log_softmax/kernels/"
#endif

namespace ttml::metal::log_softmax_device {
struct LogSoftmaxAttributes {
    bool backward;
};
struct LogSoftmaxInputs {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> upstream;
    std::optional<ttnn::Tensor> state;
};
using Tensors = std::vector<ttnn::Tensor>;
using Specs = std::vector<tt::tt_metal::TensorSpec>;
struct LogSoftmaxFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader, writer;
        uint32_t cores, cores_y;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const LogSoftmaxAttributes&, const LogSoftmaxInputs&, Tensors&);
    static void override_runtime_arguments(
        cached_program_t&, const LogSoftmaxAttributes&, const LogSoftmaxInputs&, Tensors&);
};
struct LogSoftmaxOperation {
    using operation_attributes_t = LogSoftmaxAttributes;
    using tensor_args_t = LogSoftmaxInputs;
    using tensor_return_value_t = Tensors;
    using spec_return_value_t = Specs;
    using program_factory_t = std::variant<LogSoftmaxFactory>;
    static void validate_on_program_cache_miss(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        check_device_tensor(
            t.input,
            "LogSoftmax",
            "input",
            {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
        TT_FATAL(
            t.input.layout() == ttnn::Layout::TILE && t.input.logical_shape().rank() == 4,
            "Fused log_softmax expects rank-four TILE input");
        TT_FATAL(
            t.input.tensor_spec().tile().get_height() == 32 && t.input.tensor_spec().tile().get_width() == 32,
            "Fused log_softmax currently supports 32x32 tiles");
        TT_FATAL(t.input.memory_config() == ttnn::DRAM_MEMORY_CONFIG, "Fused log_softmax expects interleaved DRAM");
        TT_FATAL(
            t.input.logical_shape()[-1] > 0 && t.input.logical_shape()[-1] <= (1U << 24), "Invalid reduction width");
        if (a.backward) {
            TT_FATAL(t.upstream && t.state, "Fused log_softmax backward requires upstream and row state");
            check_device_tensor(
                *t.upstream,
                "LogSoftmax",
                "upstream",
                {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
            check_device_tensor(
                *t.state,
                "LogSoftmax",
                "state",
                {.dtypes = {ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
            TT_FATAL(
                t.upstream->logical_shape() == t.input.logical_shape() &&
                    t.upstream->padded_shape() == t.input.padded_shape() &&
                    t.upstream->memory_config() == t.input.memory_config() &&
                    t.upstream->layout() == t.input.layout() &&
                    t.upstream->tensor_spec().tile() == t.input.tensor_spec().tile(),
                "Upstream shape, layout and memory configuration must match input");
            TT_FATAL(
                t.upstream->device() == t.input.device() && t.state->device() == t.input.device(), "Device mismatch");
            TT_FATAL(
                t.state->dtype() == ttnn::DataType::FLOAT32 && t.state->layout() == ttnn::Layout::TILE,
                "Expected FP32 TILE row state");
            const uint32_t rows = t.input.physical_volume() / t.input.padded_shape()[-1];
            TT_FATAL(
                t.state->logical_shape() == ttnn::Shape({1, 1, rows, 96}) &&
                    t.state->memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
                    t.state->tensor_spec().tile() == t.input.tensor_spec().tile(),
                "Invalid fused log_softmax row state");
        }
    }
    static void validate_on_program_cache_hit(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        validate_on_program_cache_miss(a, t);
    }
    static Specs compute_output_specs(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        Specs result{t.input.tensor_spec()};
        if (!a.backward) {
            const auto rows = t.input.physical_volume() / t.input.padded_shape()[-1];
            result.emplace_back(
                ttnn::Shape({1, 1, static_cast<uint32_t>(rows), 96}),
                tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
        }
        return result;
    }
    static Tensors create_output_tensors(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        Tensors result;
        for (const auto& spec : compute_output_specs(a, t))
            result.push_back(ttnn::create_device_tensor(spec, t.input.device()));
        return result;
    }
    static ttsl::hash::hash_t compute_program_hash(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        return tt::tt_metal::operation::hash_operation<LogSoftmaxOperation>(
            a,
            t.input.dtype(),
            t.input.logical_shape(),
            t.input.padded_shape(),
            t.input.memory_config(),
            t.upstream ? t.upstream->dtype() : t.input.dtype());
    }
};
LogSoftmaxFactory::cached_program_t LogSoftmaxFactory::create(
    const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    using namespace tt::tt_metal;
    const auto& input = t.input;
    auto* device = input.device();
    Program program;
    const uint32_t th = input.tensor_spec().tile().get_height(), tw = input.tensor_spec().tile().get_width();
    const uint32_t width_tiles = input.padded_shape()[-1] / tw;
    const uint32_t rows = input.physical_volume() / (width_tiles * th * tw);
    const auto grid = device->compute_with_storage_grid_size();
    auto [cores, all, group1, group2, rows1, rows2] = split_work_to_cores(grid, rows);
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    const auto native_bytes = tt::tile_size(fmt), fp32_bytes = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t block = get_block_size(width_tiles, 4);
    // Double-buffered bounded streams; scratch size is independent of vocabulary width.
    create_circular_buffer(program, all, 0, fmt, native_bytes, 2 * block);
    const auto grad_fmt = a.backward ? datatype_to_dataformat_converter(t.upstream->dtype()) : fmt;
    create_circular_buffer(program, all, 1, grad_fmt, tt::tile_size(grad_fmt), 2 * block);
    create_circular_buffer(program, all, 2, tt::DataFormat::Float32, fp32_bytes, 1);
    for (uint32_t cb = 3; cb <= 7; ++cb)
        create_circular_buffer(program, all, cb, tt::DataFormat::Float32, fp32_bytes, 1);
    create_circular_buffer(program, all, 8, tt::DataFormat::Float32, fp32_bytes, 3);
    create_circular_buffer(program, all, 9, tt::DataFormat::Float32, fp32_bytes, 3);
    create_circular_buffer(program, all, 16, fmt, native_bytes, 2 * block);
    const std::map<std::string, std::string> defines =
        a.backward ? std::map<std::string, std::string>{{"BACKWARD", "1"}} : std::map<std::string, std::string>{};
    std::vector<uint32_t> ra{width_tiles, block, th, tw};
    TensorAccessorArgs(input.buffer()).append_to(ra);
    if (a.backward) {
        TensorAccessorArgs(t.upstream->buffer()).append_to(ra);
        TensorAccessorArgs(t.state->buffer()).append_to(ra);
    }
    std::vector<uint32_t> wa{width_tiles, block};
    TensorAccessorArgs(out[0].buffer()).append_to(wa);
    if (!a.backward)
        TensorAccessorArgs(out[1].buffer()).append_to(wa);
    auto reader = create_reader_kernel(program, all, ra, defines, TTML_LOG_SOFTMAX_KERNEL_ROOT "dataflow/reader.cpp");
    auto writer = create_writer_kernel(program, all, wa, defines, TTML_LOG_SOFTMAX_KERNEL_ROOT "dataflow/writer.cpp");
    const std::string compute_path = TTML_LOG_SOFTMAX_KERNEL_ROOT "compute/log_softmax.cpp";
    // Route FP32 loads directly into DST: the FPU copy path truncates to TF32.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb = 2; cb <= 9; ++cb) unpack_modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    if (input.dtype() == ttnn::DataType::FLOAT32) {
        unpack_modes[0] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (a.backward && t.upstream->dtype() == ttnn::DataType::FLOAT32)
        unpack_modes[1] = UnpackToDestMode::UnpackToDestFp32;
    const auto make_compute = [&](const CoreRangeSet& group, uint32_t count) {
        if (!group.ranges().empty())
            CreateKernel(
                program,
                compute_path,
                group,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .unpack_to_dest_mode = unpack_modes,
                    .math_approx_mode = false,
                    .compile_args = {count, width_tiles, input.logical_shape()[-1], tw},
                    .defines = defines});
    };
    make_compute(group1, rows1);
    make_compute(group2, rows2);
    for_each_core_with_work(cores, grid.y, group1, group2, rows1, rows2, [&](const CoreWork& w) {
        SetRuntimeArgs(
            program,
            reader,
            w.core,
            {input.buffer()->address(),
             a.backward ? t.upstream->buffer()->address() : 0,
             a.backward ? t.state->buffer()->address() : 0,
             w.num_units,
             w.start});
        SetRuntimeArgs(
            program,
            writer,
            w.core,
            {out[0].buffer()->address(), a.backward ? 0 : out[1].buffer()->address(), w.num_units, w.start});
    });
    return {std::move(program), {reader, writer, cores, grid.y}};
}
void LogSoftmaxFactory::override_runtime_arguments(
    cached_program_t& cache, const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    auto& r = GetRuntimeArgs(cache.program, cache.shared_variables.reader);
    auto& w = GetRuntimeArgs(cache.program, cache.shared_variables.writer);
    for_each_core(cache.shared_variables.cores, cache.shared_variables.cores_y, [&](const tt::tt_metal::CoreCoord& c) {
        r[c.x][c.y][0] = t.input.buffer()->address();
        r[c.x][c.y][1] = a.backward ? t.upstream->buffer()->address() : 0;
        r[c.x][c.y][2] = a.backward ? t.state->buffer()->address() : 0;
        w[c.x][c.y][0] = out[0].buffer()->address();
        w[c.x][c.y][1] = a.backward ? 0 : out[1].buffer()->address();
    });
}
}  // namespace ttml::metal::log_softmax_device
namespace ttml::metal {
std::vector<ttnn::Tensor> log_softmax_fw(const ttnn::Tensor& x) {
    using namespace log_softmax_device;
    return ttnn::device_operation::launch<LogSoftmaxOperation>(
        LogSoftmaxAttributes{false}, LogSoftmaxInputs{x, std::nullopt, std::nullopt});
}
ttnn::Tensor log_softmax_bw(const ttnn::Tensor& x, const ttnn::Tensor& dy, const ttnn::Tensor& state) {
    using namespace log_softmax_device;
    return ttnn::device_operation::launch<LogSoftmaxOperation>(
        LogSoftmaxAttributes{true}, LogSoftmaxInputs{x, dy, state})[0];
}
}  // namespace ttml::metal
