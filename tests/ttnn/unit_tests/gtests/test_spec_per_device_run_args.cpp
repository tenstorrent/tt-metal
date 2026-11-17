// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MeshWorkloadSpecFactoryAdapter with per-device run args (`uses_per_device_run_args`): a Metal 2.0 op whose devices
// differ only in a few named kernel run args runs one program per range (merged), or one per coordinate (split, the
// fallback), and every call's base and per-device values reach each device. The op's kernel writes its args to L1,
// where the test reads them on every device.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <internal/mesh_device.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/per_device_run_args.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace spec_per_device_run_args_test {

using ttnn::Tensor;
namespace m2 = tt::tt_metal;
using tt::tt_metal::CoreCoord;
using ttnn::device_operation::DeviceRunArgs;
using ttnn::device_operation::MeshWorkloadArtifacts;
using ttnn::device_operation::RangeRunArgs;

constexpr uint32_t kAddress = 100 * 1024;
constexpr uint32_t kBaseSalt = 5;
constexpr uint32_t kBaseBias = 7;
const std::vector<CoreCoord> kCores = {{0, 0}, {1, 0}};
const char kMain[] = "main";
const char kExtra[] = "extra";

// Writes [salt, bias, arr[0], arr[1]] to `address` on its core.
constexpr const char* kMainSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    out[0] = get_arg(args::salt);
    out[1] = get_arg(args::bias);
    const auto arr = get_arg(args::arr);
    out[2] = arr[0];
    out[3] = arr[1];
}
)";
// Writes [salt] to `address` (an optional second kernel, on the first core only).
constexpr const char* kExtraSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    out[0] = get_arg(args::salt);
}
)";

uint32_t main_address(size_t core_index) { return kAddress + 256 * static_cast<uint32_t>(core_index); }
uint32_t extra_address() { return kAddress + 1024; }

m2::KernelSpec dm_kernel(const char* name, const char* source, tt::tt_metal::DataMovementProcessor processor) {
    return m2::KernelSpec{
        .unique_id = m2::KernelSpecName{name},
        .source = m2::KernelSpec::SourceCode{source},
        .num_threads = 1,
        .hw_config = m2::DataMovementHardwareConfig{
            .config_1xx = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = processor,
                .noc = processor == tt::tt_metal::DataMovementProcessor::RISCV_0 ? tt::tt_metal::NOC::NOC_0
                                                                                 : tt::tt_metal::NOC::NOC_1}}};
}

struct PerDeviceRunArgsOp {
    struct operation_attributes_t {
        // Hash-excluded: what this call sets.
        uint32_t call = 0;
        bool salt_per_device = true;
        std::vector<uint32_t> bias_cores = {0};  // cores whose bias is set per device
        bool array_per_device = false;
        bool base_omits_core1_bias = false;        // violates the adapter's rule when core 1's bias was set per device
        bool base_omits_arr = false;               // likewise for the array
        std::optional<uint32_t> skip_device;       // this device (row-major index) gets no per-device values
        std::optional<uint32_t> bad_array_device;  // this device's array has the wrong length
        std::optional<uint32_t> duplicate_kernel_device;  // this device lists the main kernel twice
        bool misspelled_arg = false;  // the base and every device also set an arg the kernel does not declare
        bool bad_base_row = false;    // on a hit, row 1's base sets an undeclared arg (metal rejects it as it applies)
        // Hashed: the program's shape.
        bool with_extra = false;  // a second kernel, on the first core
        // 0: one factory range per tensor range; 1: one per mesh row; 2: device (0, 0) alone (split), the rest of
        // row 0, and row 1 (merged).
        uint32_t ranges = 0;
    };
    struct tensor_args_t {
        Tensor input;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static m2::ProgramSpec spec(const operation_attributes_t& attrs) {
        auto main = dm_kernel(kMain, kMainSource, tt::tt_metal::DataMovementProcessor::RISCV_0);
        main.runtime_arg_schema = {
            .runtime_arg_names = {"address", "bias"},
            .common_runtime_arg_names = {"salt"},
            .runtime_array_args = {{.name = "arr", .size = 2}},
        };
        m2::ProgramSpec program{.name = "per_device_run_args_test", .kernels = {main}};
        if (attrs.with_extra) {
            auto extra = dm_kernel(kExtra, kExtraSource, tt::tt_metal::DataMovementProcessor::RISCV_1);
            extra.runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"salt"}};
            program.kernels.push_back(extra);
            // Work units do not overlap: the first core runs both kernels, the second only the main one.
            program.work_units = {
                m2::WorkUnitSpec{
                    .name = "first",
                    .kernels = {m2::KernelSpecName{kMain}, m2::KernelSpecName{kExtra}},
                    .target_nodes = kCores.front()},
                m2::WorkUnitSpec{
                    .name = "second", .kernels = {m2::KernelSpecName{kMain}}, .target_nodes = kCores.back()},
            };
        } else {
            program.work_units = {m2::WorkUnitSpec{
                .name = "cores",
                .kernels = {m2::KernelSpecName{kMain}},
                .target_nodes = m2::NodeRange(kCores.front(), kCores.back())}};
        }
        return program;
    }

    static m2::ProgramRunArgs base(const operation_attributes_t& attrs) {
        m2::ProgramRunArgs::KernelRunArgs main{.kernel = m2::KernelSpecName{kMain}};
        for (size_t c = 0; c < kCores.size(); ++c) {
            main.runtime_arg_values["address"][kCores[c]] = main_address(c);
            if (!(attrs.base_omits_core1_bias && c == 1)) {
                main.runtime_arg_values["bias"][kCores[c]] = kBaseBias;
            }
            if (!attrs.base_omits_arr) {
                main.runtime_array_arg_values["arr"][kCores[c]] = {11u, 12u};
            }
        }
        if (attrs.misspelled_arg) {
            main.runtime_arg_values["biass"][kCores[0]] = 1u;
        }
        main.common_runtime_arg_values = {{"salt", kBaseSalt}};
        m2::ProgramRunArgs args;
        args.kernel_run_args.push_back(std::move(main));
        if (attrs.with_extra) {
            m2::ProgramRunArgs::KernelRunArgs extra{.kernel = m2::KernelSpecName{kExtra}};
            extra.runtime_arg_values["address"][kCores.front()] = extra_address();
            extra.common_runtime_arg_values = {{"salt", kBaseSalt}};
            args.kernel_run_args.push_back(std::move(extra));
        }
        return args;
    }

    static uint32_t index_of(const ttnn::MeshCoordinate& coord, const ttnn::MeshShape& shape) {
        return static_cast<uint32_t>(coord.to_linear_index(shape));
    }
    static uint32_t device_salt(const operation_attributes_t& attrs, uint32_t index) {
        return 1000 * (index + 1) + attrs.call;
    }
    static uint32_t device_bias(const operation_attributes_t& attrs, uint32_t index, uint32_t core) {
        return 50000 + 100 * index + core + attrs.call;
    }
    static std::vector<uint32_t> device_array(const operation_attributes_t& attrs, uint32_t index) {
        return {70000 + index, 80000 + attrs.call};
    }

    static std::vector<DeviceRunArgs> per_device(
        const operation_attributes_t& attrs, const ttnn::MeshCoordinateRange& range, const ttnn::MeshShape& shape) {
        std::vector<DeviceRunArgs> out;
        for (const auto& coord : range) {
            const uint32_t index = index_of(coord, shape);
            if (attrs.skip_device == index) {
                continue;
            }
            m2::ProgramRunArgs::KernelRunArgs main{.kernel = m2::KernelSpecName{kMain}};
            if (attrs.salt_per_device) {
                main.common_runtime_arg_values = {{"salt", device_salt(attrs, index)}};
            }
            for (const uint32_t c : attrs.bias_cores) {
                main.runtime_arg_values["bias"][kCores[c]] = device_bias(attrs, index, c);
            }
            if (attrs.array_per_device) {
                const auto values = device_array(attrs, index);
                main.runtime_array_arg_values["arr"][kCores[0]] = attrs.bad_array_device == index
                                                                      ? m2::ArrayArgValue{values[0], values[1], 0u}
                                                                      : m2::ArrayArgValue{values[0], values[1]};
            }
            if (attrs.misspelled_arg) {
                main.runtime_arg_values["biass"][kCores[0]] = 1u;
            }
            const bool main_empty = main.common_runtime_arg_values.empty() && main.runtime_arg_values.empty() &&
                                    main.runtime_array_arg_values.empty();
            if (main_empty && !(attrs.with_extra && attrs.salt_per_device)) {
                continue;  // no per-device values: no entry (a call can leave the list empty)
            }
            m2::ProgramRunArgs values;
            if (attrs.duplicate_kernel_device == index) {
                values.kernel_run_args.push_back(main);
            }
            values.kernel_run_args.push_back(std::move(main));
            if (attrs.with_extra && attrs.salt_per_device) {
                m2::ProgramRunArgs::KernelRunArgs extra{.kernel = m2::KernelSpecName{kExtra}};
                extra.common_runtime_arg_values = {{"salt", device_salt(attrs, index) + 1}};
                values.kernel_run_args.push_back(std::move(extra));
            }
            out.push_back({coord, std::move(values)});
        }
        return out;
    }

    static std::vector<ttnn::MeshCoordinateRange> factory_ranges(
        const operation_attributes_t& attrs, const ttnn::MeshCoordinateRangeSet& tensor_coords) {
        std::vector<ttnn::MeshCoordinateRange> ranges;
        for (const auto& range : tensor_coords.ranges()) {
            if (attrs.ranges == 0) {
                ranges.push_back(range);
                continue;
            }
            const auto& start = range.start_coord();
            const auto& end = range.end_coord();
            for (uint32_t row = start[0]; row <= end[0]; ++row) {
                if (attrs.ranges == 2 && row == start[0] && start[1] < end[1]) {
                    ranges.emplace_back(ttnn::MeshCoordinate(row, start[1]), ttnn::MeshCoordinate(row, start[1]));
                    ranges.emplace_back(ttnn::MeshCoordinate(row, start[1] + 1), ttnn::MeshCoordinate(row, end[1]));
                } else {
                    ranges.emplace_back(ttnn::MeshCoordinate(row, start[1]), ttnn::MeshCoordinate(row, end[1]));
                }
            }
        }
        return ranges;
    }

    struct Factory {
        static constexpr bool uses_per_device_run_args = true;

        static MeshWorkloadArtifacts create_mesh_workload_artifacts(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& /*output*/,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            const auto shape = tensor_args.input.device()->shape();
            MeshWorkloadArtifacts artifacts;
            for (const auto& range : factory_ranges(attrs, tensor_coords)) {
                artifacts.programs.push_back({
                    .range = range,
                    .spec = spec(attrs),
                    .run_params = base(attrs),
                    .per_device_run_params = per_device(attrs, range, shape),
                });
            }
            return artifacts;
        }

        static RangeRunArgs override_runtime_arguments(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& /*output*/,
            const ttnn::MeshCoordinateRange& range) {
            auto range_base = base(attrs);
            if (attrs.bad_base_row && range.start_coord()[0] == 1) {
                range_base.kernel_run_args[0].common_runtime_arg_values["salty"] = 1u;
            }
            return {
                .base = std::move(range_base),
                .per_device = per_device(attrs, range, tensor_args.input.device()->shape())};
        }
    };
    // Without an override (opted in): the per-device values of the cache miss stay on every hit.
    struct FixedFactory {
        static constexpr bool uses_per_device_run_args = true;
        static constexpr bool fixed_per_device_run_args = true;

        static MeshWorkloadArtifacts create_mesh_workload_artifacts(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            return Factory::create_mesh_workload_artifacts(attrs, tensor_args, output, tensor_coords);
        }
    };
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t& attrs, const tensor_args_t&) {
        return ttsl::hash::hash_objects_with_default_seed(attrs.with_extra, attrs.ranges);
    }
};

struct FixedPerDeviceRunArgsOp : PerDeviceRunArgsOp {
    using program_factory_t = std::variant<PerDeviceRunArgsOp::FixedFactory>;
};

using MeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<PerDeviceRunArgsOp>;
using Adapter = MeshOp::MeshWorkloadSpecFactoryAdapter<PerDeviceRunArgsOp::Factory>;
using FixedMeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<FixedPerDeviceRunArgsOp>;
using FixedAdapter = FixedMeshOp::MeshWorkloadSpecFactoryAdapter<PerDeviceRunArgsOp::FixedFactory>;
static_assert(!Adapter::fixed_per_device_run_args() && FixedAdapter::fixed_per_device_run_args());

// A tensor-backed op: its kernel writes [salt, bias] into page 0 of the output tensor, which every call allocates
// anew. A cache hit must rebind that tensor on every device and keep (fixed) or refresh (override) the per-device
// values.
constexpr const char* kTensorSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    const uint32_t staging = get_arg(args::address);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
    TensorAccessor out(tensor::out);
    const uint32_t page_size = out.get_aligned_page_size();
    for (uint32_t i = 0; i < page_size / sizeof(uint32_t); ++i) {
        words[i] = 0;
    }
    words[0] = get_arg(args::salt);
    words[1] = get_arg(args::bias);
    noc_async_write(staging, out.get_noc_addr(0), page_size);
    noc_async_write_barrier();
}
)";

uint32_t tensor_salt(uint32_t index, uint32_t call) { return 1000 * (index + 1) + call; }
uint32_t tensor_bias(uint32_t index, uint32_t call) { return 50000 + 100 * index + call; }

struct TensorPerDeviceRunArgsOp {
    struct operation_attributes_t {
        uint32_t call = 0;    // hash-excluded: the per-device values this call sets
        uint32_t ranges = 0;  // hashed: as PerDeviceRunArgsOp::operation_attributes_t::ranges
    };
    using tensor_args_t = PerDeviceRunArgsOp::tensor_args_t;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static m2::ProgramSpec spec(const Tensor& output) {
        auto kernel = dm_kernel(kMain, kTensorSource, tt::tt_metal::DataMovementProcessor::RISCV_0);
        kernel.runtime_arg_schema = {.runtime_arg_names = {"address", "bias"}, .common_runtime_arg_names = {"salt"}};
        kernel.tensor_bindings = {
            m2::KernelSpec::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"}};
        return m2::ProgramSpec{
            .name = "per_device_run_args_tensor_test",
            .kernels = {kernel},
            .tensor_parameters = {m2::TensorParameter{
                .unique_id = m2::TensorParamName{"out"}, .spec = output.tensor_spec()}},
            .work_units = {m2::WorkUnitSpec{
                .name = "core", .kernels = {m2::KernelSpecName{kMain}}, .target_nodes = kCores.front()}},
        };
    }

    static m2::ProgramRunArgs base(const Tensor& output) {
        m2::ProgramRunArgs::KernelRunArgs main{.kernel = m2::KernelSpecName{kMain}};
        main.runtime_arg_values["address"][kCores.front()] = kAddress;
        main.runtime_arg_values["bias"][kCores.front()] = kBaseBias;
        main.common_runtime_arg_values = {{"salt", kBaseSalt}};
        m2::ProgramRunArgs args;
        args.kernel_run_args.push_back(std::move(main));
        args.tensor_args = {{m2::TensorParamName{"out"}, m2::ProgramRunArgs::TensorArgument{output.mesh_tensor()}}};
        return args;
    }

    static std::vector<DeviceRunArgs> per_device(
        const operation_attributes_t& attrs, const ttnn::MeshCoordinateRange& range, const ttnn::MeshShape& shape) {
        std::vector<DeviceRunArgs> out;
        for (const auto& coord : range) {
            const uint32_t index = PerDeviceRunArgsOp::index_of(coord, shape);
            m2::ProgramRunArgs::KernelRunArgs main{.kernel = m2::KernelSpecName{kMain}};
            main.common_runtime_arg_values = {{"salt", tensor_salt(index, attrs.call)}};
            main.runtime_arg_values["bias"][kCores.front()] = tensor_bias(index, attrs.call);
            m2::ProgramRunArgs values;
            values.kernel_run_args.push_back(std::move(main));
            out.push_back({coord, std::move(values)});
        }
        return out;
    }

    struct Factory {
        static constexpr bool uses_per_device_run_args = true;

        static MeshWorkloadArtifacts create_mesh_workload_artifacts(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            const auto shape = tensor_args.input.device()->shape();
            MeshWorkloadArtifacts artifacts;
            for (const auto& range : PerDeviceRunArgsOp::factory_ranges(
                     PerDeviceRunArgsOp::operation_attributes_t{.ranges = attrs.ranges}, tensor_coords)) {
                artifacts.programs.push_back({
                    .range = range,
                    .spec = spec(output),
                    .run_params = base(output),
                    .per_device_run_params = per_device(attrs, range, shape),
                });
            }
            return artifacts;
        }

        static RangeRunArgs override_runtime_arguments(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const ttnn::MeshCoordinateRange& range) {
            return {.base = base(output), .per_device = per_device(attrs, range, tensor_args.input.device()->shape())};
        }
    };
    struct FixedFactory {
        static constexpr bool uses_per_device_run_args = true;
        static constexpr bool fixed_per_device_run_args = true;

        static MeshWorkloadArtifacts create_mesh_workload_artifacts(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            return Factory::create_mesh_workload_artifacts(attrs, tensor_args, output, tensor_coords);
        }
    };
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t& attrs, const tensor_args_t&) {
        return ttsl::hash::hash_objects_with_default_seed(attrs.ranges);
    }
};

struct FixedTensorPerDeviceRunArgsOp : TensorPerDeviceRunArgsOp {
    using program_factory_t = std::variant<TensorPerDeviceRunArgsOp::FixedFactory>;
};

class SpecPerDeviceRunArgsTest : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    SpecPerDeviceRunArgsTest() :
        MeshDeviceFixtureBase(Config{.mesh_shape = ttnn::MeshShape{2, 4}, .num_cqs = 1, .trace_region_size = 1 << 20}) {
    }

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!tt::tt_metal::distributed::ArePerDeviceRuntimeArgsSupported(
                *mesh_device_, ttnn::MeshCoordinateRange(mesh_device_->shape()))) {
            GTEST_SKIP() << "per-device runtime args are not supported on this mesh";
        }
        mesh_device_->clear_program_cache();
    }

    PerDeviceRunArgsOp::tensor_args_t input() const {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, 1, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig{}));
        return {.input = Tensor::from_vector(std::vector<uint32_t>(32, 0), spec, mesh_device_.get())};
    }

    void launch(const PerDeviceRunArgsOp::operation_attributes_t& attrs) {
        (void)ttnn::device_operation::launch<PerDeviceRunArgsOp>(attrs, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    }

    // Enqueues the cached workload for `attrs` as it is (no override call).
    void enqueue_cached(const PerDeviceRunArgsOp::operation_attributes_t& attrs) {
        const auto args = input();
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<MeshOp>(attrs, args, mesh_device_.get());
        ASSERT_TRUE(lookup.hit);
        auto& workload = mesh_device_->get_program_cache()
                             .get(lookup.key)
                             .cached_program.template get<Adapter::cached_mesh_workload_t>()
                             .workload;
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);
    }

    size_t cached_programs(const PerDeviceRunArgsOp::operation_attributes_t& attrs) {
        const auto args = input();
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<MeshOp>(attrs, args, mesh_device_.get());
        EXPECT_TRUE(lookup.hit);
        return mesh_device_->get_program_cache()
            .get(lookup.key)
            .cached_program.template get<Adapter::cached_mesh_workload_t>()
            .workload.get_programs()
            .size();
    }

    std::vector<uint32_t> read(const ttnn::MeshCoordinate& coord, const CoreCoord& core, uint32_t address, uint32_t n) {
        std::vector<uint32_t> words;
        tt::tt_metal::detail::ReadFromDeviceL1(
            tt::tt_metal::internal::get_device(*mesh_device_, coord), core, address, n * sizeof(uint32_t), words);
        return words;
    }

    void clear_l1() {
        std::vector<uint32_t> zeros(4, 0);
        for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device_->shape())) {
            auto* device = tt::tt_metal::internal::get_device(*mesh_device_, coord);
            for (size_t c = 0; c < kCores.size(); ++c) {
                tt::tt_metal::detail::WriteToDeviceL1(device, kCores[c], main_address(c), zeros);
            }
            tt::tt_metal::detail::WriteToDeviceL1(device, kCores.front(), extra_address(), zeros);
        }
    }

    // Every device reads what `attrs` says: its own values where set per device, the base elsewhere.
    void expect_values(const PerDeviceRunArgsOp::operation_attributes_t& attrs) {
        const auto shape = mesh_device_->shape();
        for (const auto& coord : ttnn::MeshCoordinateRange(shape)) {
            const uint32_t index = PerDeviceRunArgsOp::index_of(coord, shape);
            const bool own = attrs.skip_device != index;  // whether this device has per-device values
            for (uint32_t c = 0; c < kCores.size(); ++c) {
                const bool bias_per_device =
                    own && std::find(attrs.bias_cores.begin(), attrs.bias_cores.end(), c) != attrs.bias_cores.end();
                const auto array = own && attrs.array_per_device && c == 0
                                       ? PerDeviceRunArgsOp::device_array(attrs, index)
                                       : std::vector<uint32_t>{11, 12};
                const std::vector<uint32_t> expected = {
                    own && attrs.salt_per_device ? PerDeviceRunArgsOp::device_salt(attrs, index) : kBaseSalt,
                    bias_per_device ? PerDeviceRunArgsOp::device_bias(attrs, index, c) : kBaseBias,
                    array[0],
                    array[1]};
                EXPECT_EQ(read(coord, kCores[c], main_address(c), 4), expected)
                    << "device " << coord << ", core " << kCores[c].str() << ", call " << attrs.call;
            }
            if (attrs.with_extra) {
                const uint32_t expected =
                    own && attrs.salt_per_device ? PerDeviceRunArgsOp::device_salt(attrs, index) + 1 : kBaseSalt;
                EXPECT_EQ(read(coord, kCores.front(), extra_address(), 1), std::vector<uint32_t>{expected})
                    << "device " << coord << ", extra kernel, call " << attrs.call;
            }
        }
    }

    void bad_values_rejected() {
        using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
        const Attrs first{.call = 0, .bias_cores = {}};
        launch(first);
        EXPECT_ANY_THROW(launch(Attrs{.call = 1, .bias_cores = {}, .array_per_device = true, .bad_array_device = 3}));
        EXPECT_ANY_THROW(launch(Attrs{.call = 2, .bias_cores = {1}, .misspelled_arg = true}));
        EXPECT_ANY_THROW(launch(Attrs{.call = 4, .bias_cores = {1}, .duplicate_kernel_device = 5}));
        clear_l1();
        enqueue_cached(first);
        expect_values(first);
        // Neither the array nor the misspelled arg became per device: a base without the array is fine.
        const Attrs without_array{.call = 3, .bias_cores = {0}, .base_omits_arr = true};
        clear_l1();
        launch(without_array);
        expect_values(without_array);
    }

    // A factory with `fixed_per_device_run_args` (no override): the miss sets the per-device values, and a hit
    // (whose attributes would set others) keeps them.
    void run_fixed(uint32_t ranges, size_t expected_programs) {
        using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
        const Attrs miss{.call = 0, .bias_cores = {1}, .array_per_device = true, .with_extra = true, .ranges = ranges};
        clear_l1();
        (void)ttnn::device_operation::launch<FixedPerDeviceRunArgsOp>(miss, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
        expect_values(miss);
        Attrs hit = miss;
        hit.call = 1;
        hit.bias_cores = {0};
        clear_l1();
        (void)ttnn::device_operation::launch<FixedPerDeviceRunArgsOp>(hit, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
        expect_values(miss);
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u);
        const auto args = input();
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<FixedMeshOp>(hit, args, mesh_device_.get());
        ASSERT_TRUE(lookup.hit);
        EXPECT_EQ(
            mesh_device_->get_program_cache()
                .get(lookup.key)
                .cached_program.template get<FixedAdapter::cached_mesh_workload_t>()
                .workload.get_programs()
                .size(),
            expected_programs);
    }

    // Each device's shard of `output` holds the [salt, bias] of `call`.
    void expect_tensor_values(const Tensor& output, uint32_t call) {
        const auto shape = mesh_device_->shape();
        const auto shards = ttnn::distributed::get_device_tensors(output);
        ASSERT_EQ(shards.size(), shape.mesh_size());
        for (const auto& shard : shards) {
            const auto coords = shard.device_storage().get_coords();
            ASSERT_EQ(coords.size(), 1u);
            const uint32_t index = PerDeviceRunArgsOp::index_of(coords.front(), shape);
            const auto words = shard.to_vector<uint32_t>();
            ASSERT_GE(words.size(), 2u);
            EXPECT_EQ(
                (std::vector<uint32_t>{words[0], words[1]}),
                (std::vector<uint32_t>{tensor_salt(index, call), tensor_bias(index, call)}))
                << "device " << coords.front() << ", call " << call;
        }
    }

    // Two calls on one cache entry, the first call's output kept alive so the hit's output lands at a new address:
    // the hit writes the new tensor on every device, with the miss's per-device values (fixed) or its own (override).
    template <typename Op>
    void run_tensor_hit(bool fixed, uint32_t ranges, size_t expected_programs) {
        using Attrs = TensorPerDeviceRunArgsOp::operation_attributes_t;
        const Tensor first = ttnn::device_operation::launch<Op>(Attrs{.call = 0, .ranges = ranges}, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
        expect_tensor_values(first, 0);
        const Tensor second = ttnn::device_operation::launch<Op>(Attrs{.call = 1, .ranges = ranges}, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u);
        ASSERT_NE(first.mesh_buffer().address(), second.mesh_buffer().address());
        expect_tensor_values(second, fixed ? 0 : 1);
        expect_tensor_values(first, 0);  // the hit did not write the miss's tensor
        using TensorAdapter = typename ttnn::device_operation::MeshDeviceOperationAdapter<
            Op>::template MeshWorkloadSpecFactoryAdapter<std::variant_alternative_t<0, typename Op::program_factory_t>>;
        const auto args = input();
        const auto lookup = ttnn::device_operation::detail::look_up_program_cache<
            ttnn::device_operation::MeshDeviceOperationAdapter<Op>>(
            Attrs{.call = 1, .ranges = ranges}, args, mesh_device_.get());
        ASSERT_TRUE(lookup.hit);
        EXPECT_EQ(
            mesh_device_->get_program_cache()
                .get(lookup.key)
                .cached_program.template get<typename TensorAdapter::cached_mesh_workload_t>()
                .workload.get_programs()
                .size(),
            expected_programs);
    }

    // A sequence of calls on one cache entry: per-device values change, a core and an array start and stop being set
    // per device, the per-device lists go empty and come back.
    void run_sequence(bool with_extra, uint32_t ranges, size_t expected_programs) {
        using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
        const std::vector<Attrs> calls = {
            Attrs{.call = 0, .salt_per_device = true, .bias_cores = {0}},
            Attrs{.call = 1, .salt_per_device = true, .bias_cores = {1}, .array_per_device = true},
            Attrs{.call = 2, .salt_per_device = true, .bias_cores = {1}, .array_per_device = true, .skip_device = 2},
            Attrs{.call = 3, .salt_per_device = false, .bias_cores = {}},  // empty per-device list
            Attrs{.call = 4, .salt_per_device = true, .bias_cores = {0, 1}},
        };
        for (auto attrs : calls) {
            attrs.with_extra = with_extra;
            attrs.ranges = ranges;
            clear_l1();
            launch(attrs);
            expect_values(attrs);
            EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u) << "call " << attrs.call;
            EXPECT_EQ(cached_programs(attrs), expected_programs) << "call " << attrs.call;
        }
    }
};

// A single-device range runs split next to merged ranges in one workload.
TEST_F(SpecPerDeviceRunArgsTest, MergedAndSplitRangesInOneWorkload) {
    run_sequence(/*with_extra=*/true, /*ranges=*/2, /*expected_programs=*/3);
}

TEST_F(SpecPerDeviceRunArgsTest, SplitMatchesMerged) {
    ttnn::device_operation::per_device_run_args::ScopedForceSplit split(*mesh_device_);
    run_sequence(/*with_extra=*/true, /*ranges=*/0, /*expected_programs=*/mesh_device_->shape().mesh_size());
}

TEST_F(SpecPerDeviceRunArgsTest, BaseMustListWhatThePreviousCallSetPerDevice) {
    using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
    const Attrs first{.call = 0, .bias_cores = {1}};
    launch(first);
    // Core 1's bias was set per device; a base without it would leave those values behind.
    EXPECT_ANY_THROW(launch(Attrs{.call = 1, .bias_cores = {}, .base_omits_core1_bias = true}));
    // Nothing changed: the cached workload, enqueued as it is, still holds the first call's values.
    clear_l1();
    enqueue_cached(first);
    expect_values(first);
    // The next valid call applies in full.
    const Attrs next{.call = 2, .bias_cores = {0}};
    clear_l1();
    launch(next);
    expect_values(next);
}

// With `fixed_per_device_run_args`, a hit keeps the cache miss's per-device values, merged or split.
TEST_F(SpecPerDeviceRunArgsTest, WithoutOverrideHitKeepsMissValues) {
    run_fixed(/*ranges=*/2, /*expected_programs=*/3);
}

TEST_F(SpecPerDeviceRunArgsTest, WithoutOverrideSplitKeepsMissValues) {
    ttnn::device_operation::per_device_run_args::ScopedForceSplit split(*mesh_device_);
    run_fixed(/*ranges=*/0, /*expected_programs=*/mesh_device_->shape().mesh_size());
}

// A hit with a new output tensor on every device of a 2x4 mesh: the tensor is rebound, merged or split, and the
// per-device values are the miss's (`fixed_per_device_run_args`) or the hit's own (override).
TEST_F(SpecPerDeviceRunArgsTest, FixedHitRebindsTensor) {
    run_tensor_hit<FixedTensorPerDeviceRunArgsOp>(/*fixed=*/true, /*ranges=*/2, /*expected_programs=*/3);
}

TEST_F(SpecPerDeviceRunArgsTest, FixedSplitHitRebindsTensor) {
    ttnn::device_operation::per_device_run_args::ScopedForceSplit split(*mesh_device_);
    run_tensor_hit<FixedTensorPerDeviceRunArgsOp>(
        /*fixed=*/true, /*ranges=*/0, /*expected_programs=*/mesh_device_->shape().mesh_size());
}

TEST_F(SpecPerDeviceRunArgsTest, OverrideHitRebindsTensor) {
    run_tensor_hit<TensorPerDeviceRunArgsOp>(/*fixed=*/false, /*ranges=*/2, /*expected_programs=*/3);
}

TEST_F(SpecPerDeviceRunArgsTest, OverrideSplitHitRebindsTensor) {
    ttnn::device_operation::per_device_run_args::ScopedForceSplit split(*mesh_device_);
    run_tensor_hit<TensorPerDeviceRunArgsOp>(
        /*fixed=*/false, /*ranges=*/0, /*expected_programs=*/mesh_device_->shape().mesh_size());
}

// Per-device values metal would reject (an array of the wrong length on one device, an arg the kernel does not
// declare, a kernel listed twice) are rejected before anything changes, so a hit cannot fail part-way on them and
// leaves no trace: the cached workload still holds the previous call's values, and later calls are unconstrained by the
// rejected args.
TEST_F(SpecPerDeviceRunArgsTest, BadPerDeviceValuesAreRejectedBeforeAnyChange) { bad_values_rejected(); }

// A hit can still fail part-way on something the adapter does not check (here: row 1's base sets an undeclared
// arg, which metal rejects as it applies it), possibly after row 0 got its new per-device values. Every range then
// keeps tracking its old and new per-device args: a base without the new array fails, one with it restores every
// device.
TEST_F(SpecPerDeviceRunArgsTest, HitFailingPartWayKeepsItsArgsTracked) {
    using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
    launch(Attrs{.call = 0, .bias_cores = {}, .ranges = 1});
    EXPECT_ANY_THROW(
        launch(Attrs{.call = 1, .bias_cores = {}, .array_per_device = true, .bad_base_row = true, .ranges = 1}));
    EXPECT_ANY_THROW(launch(Attrs{.call = 2, .bias_cores = {}, .base_omits_arr = true, .ranges = 1}));
    const Attrs next{.call = 3, .bias_cores = {0}, .ranges = 1};
    clear_l1();
    launch(next);
    expect_values(next);
}

}  // namespace spec_per_device_run_args_test
