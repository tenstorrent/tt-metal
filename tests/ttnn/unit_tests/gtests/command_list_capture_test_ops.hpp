// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/command_list_capture.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::test {

namespace m2 = tt::tt_metal::experimental;

// Both kernels split the first num_tile_rows rows of tiles evenly across the cores on the device, so one program
// serves every length up to the tensors' capacity.
inline constexpr const char* kCopyRowsKernelPrelude = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

struct TileRange {
    uint32_t begin;
    uint32_t end;
};

inline TileRange tiles_of_this_core() {
    const uint32_t num_tiles = get_arg(args::num_tile_rows) * get_arg(args::tiles_per_row);
    const uint32_t num_cores = get_arg(args::num_cores);
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t base = num_tiles / num_cores;
    const uint32_t extra = num_tiles % num_cores;
    const uint32_t begin = core_index * base + (core_index < extra ? core_index : extra);
    return {begin, begin + base + (core_index < extra ? 1u : 0u)};
}
)";

inline constexpr const char* kCopyRowsReaderBody = R"(
void kernel_main() {
    const TileRange range = tiles_of_this_core();
    Noc noc;
    TensorAccessor src(tensor::src);
    DataflowBuffer tiles(dfb::tiles);
    const uint32_t entry_size = tiles.get_entry_size();
    for (uint32_t tile = range.begin; tile < range.end; ++tile) {
        tiles.reserve_back(1);
        noc.async_read(src, tiles, entry_size, {.page_id = tile}, {});
        noc.async_read_barrier();
        tiles.push_back(1);
    }
}
)";

inline constexpr const char* kCopyRowsWriterBody = R"(
void kernel_main() {
    const TileRange range = tiles_of_this_core();
    Noc noc;
    TensorAccessor dst(tensor::dst);
    DataflowBuffer tiles(dfb::tiles);
    const uint32_t entry_size = tiles.get_entry_size();
    for (uint32_t tile = range.begin; tile < range.end; ++tile) {
        tiles.wait_front(1);
        noc.async_write(tiles, dst, entry_size, {}, {.page_id = tile});
        noc.async_write_barrier();
        tiles.pop_front(1);
    }
}
)";

// Copies the first num_tile_rows rows of tiles of a capacity-sized input; the rest of the output is left untouched.
// Runs on the 4x2 cores starting at row first_core_row.
struct CopyRowsOp {
    struct operation_attributes_t {
        uint32_t num_tile_rows = 0;
        uint32_t first_core_row = 0;
    };
    struct tensor_args_t {
        Tensor input;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    inline static const m2::KernelSpecName kReader{"reader"};
    inline static const m2::KernelSpecName kWriter{"writer"};
    inline static const m2::TensorParamName kSrc{"src"};
    inline static const m2::TensorParamName kDst{"dst"};
    inline static const m2::DFBSpecName kTiles{"tiles"};
    static constexpr uint32_t kCoreColumns = 4;
    static constexpr uint32_t kCoreRows = 2;
    static constexpr uint32_t kNumCores = kCoreColumns * kCoreRows;

    static uint32_t capacity(const tensor_args_t& tensor_args) {
        return tensor_args.input.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    }

    static CoreRangeSet cores(const operation_attributes_t& attrs) {
        return CoreRangeSet(
            CoreRange({0, attrs.first_core_row}, {kCoreColumns - 1, attrs.first_core_row + kCoreRows - 1}));
    }

    struct Factory {
        static device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
            const auto& input = tensor_args.input;
            const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
            const auto op_cores = cores(attrs);

            const auto make_kernel = [&](const m2::KernelSpecName& name,
                                         const char* body,
                                         const m2::TensorParamName& tensor,
                                         const char* accessor,
                                         m2::DFBEndpointType endpoint,
                                         m2::DataMovementHardwareConfig hw_config) {
                return m2::KernelSpec{
                    .unique_id = name,
                    .source = m2::KernelSpec::SourceCode{std::string(kCopyRowsKernelPrelude) + body},
                    .dfb_bindings = {m2::DFBBinding{
                        .dfb_spec_name = kTiles, .accessor_name = "tiles", .endpoint_type = endpoint}},
                    .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = tensor, .accessor_name = accessor}},
                    .compile_time_args =
                        {{"tiles_per_row", input.padded_shape()[-1] / tt::constants::TILE_WIDTH},
                         {"num_cores", kNumCores}},
                    .runtime_arg_schema =
                        {.runtime_arg_names = {"core_index"}, .common_runtime_arg_names = {"num_tile_rows"}},
                    .hw_config = std::move(hw_config),
                };
            };

            m2::ProgramSpec spec{
                .name = "copy_rows",
                .kernels =
                    {make_kernel(
                         kReader,
                         kCopyRowsReaderBody,
                         kSrc,
                         "src",
                         m2::DFBEndpointType::PRODUCER,
                         ttnn::create_reader_datamovement_config()),
                     make_kernel(
                         kWriter,
                         kCopyRowsWriterBody,
                         kDst,
                         "dst",
                         m2::DFBEndpointType::CONSUMER,
                         ttnn::create_writer_datamovement_config())},
                .dataflow_buffers = {m2::DataflowBufferSpec{
                    .unique_id = kTiles,
                    .entry_size = tt::tile_size(data_format),
                    .num_entries = 2,
                    .data_format_metadata = data_format,
                }},
                .tensor_parameters =
                    {m2::TensorParameter{.unique_id = kSrc, .spec = input.tensor_spec()},
                     m2::TensorParameter{.unique_id = kDst, .spec = output.tensor_spec()}},
                .work_units = {m2::WorkUnitSpec{
                    .name = "copy", .kernels = {kReader, kWriter}, .target_nodes = op_cores}},
            };

            auto run_args = override_runtime_arguments(attrs, tensor_args, output);
            uint32_t core_index = 0;
            for (const auto& core : corerange_to_cores(op_cores)) {
                for (auto& kernel_run_args : run_args.kernel_run_args) {
                    m2::AddRuntimeArgsForNode(kernel_run_args.runtime_arg_values, core, {{"core_index", core_index}});
                }
                ++core_index;
            }
            return device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
        }

        // The per-core indices never change, so a cache hit refreshes only the length and the tensors.
        static m2::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& = std::nullopt) {
            m2::ProgramRunArgs run_args;
            for (const auto& kernel : {kReader, kWriter}) {
                m2::ProgramRunArgs::KernelRunArgs kernel_run_args{.kernel = kernel};
                kernel_run_args.common_runtime_arg_values["num_tile_rows"] = attrs.num_tile_rows;
                run_args.kernel_run_args.push_back(std::move(kernel_run_args));
            }
            run_args.tensor_args = {
                {kSrc, m2::ProgramRunArgs::TensorArgument{tensor_args.input.mesh_tensor()}},
                {kDst, m2::ProgramRunArgs::TensorArgument{output.mesh_tensor()}},
            };
            return run_args;
        }
    };
    using program_factory_t = std::variant<Factory>;

    static std::vector<CommandListParameter> command_list_parameters(
        const operation_attributes_t&, const tensor_args_t& tensor_args) {
        const uint32_t tile_rows = capacity(tensor_args);
        return {
            {.name = "tile_rows", .kernel = kReader, .arg_name = "num_tile_rows", .capacity = tile_rows},
            {.name = "tile_rows", .kernel = kWriter, .arg_name = "num_tile_rows", .capacity = tile_rows},
        };
    }

    // The length is a runtime argument, so it stays out of the key.
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return ttsl::hash::hash_objects_with_default_seed(tensor_args.input.tensor_spec(), attrs.first_core_row);
    }

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        TT_FATAL(
            attrs.num_tile_rows <= capacity(tensor_args),
            "CopyRowsOp: {} tile rows exceed the capacity {}",
            attrs.num_tile_rows,
            capacity(tensor_args));
    }

    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        validate_on_program_cache_miss(attrs, tensor_args);
    }

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
};

static_assert(HasCommandListParameters<CopyRowsOp>);

// A pre-Metal 2.0 op, built from a raw Program.
struct LegacyOp {
    using operation_attributes_t = CopyRowsOp::operation_attributes_t;
    using tensor_args_t = CopyRowsOp::tensor_args_t;
    using spec_return_value_t = CopyRowsOp::spec_return_value_t;
    using tensor_return_value_t = CopyRowsOp::tensor_return_value_t;

    struct Factory {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            tt::tt_metal::Program program;
            tt::tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
                CoreCoord{0, 0},
                tt::tt_metal::DataMovementConfig{});
            return cached_program_t(std::move(program), shared_variables_t{});
        }

        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
};

}  // namespace ttnn::experimental::test
