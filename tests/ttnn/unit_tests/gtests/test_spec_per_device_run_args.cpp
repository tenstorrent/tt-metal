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
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
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
namespace m2 = tt::tt_metal::experimental;
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

using MeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<PerDeviceRunArgsOp>;
using Adapter = MeshOp::MeshWorkloadSpecFactoryAdapter<PerDeviceRunArgsOp::Factory>;

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
        if (!tt::tt_metal::distributed::PerDeviceRuntimeArgsSupported(
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
        clear_l1();
        enqueue_cached(first);
        expect_values(first);
        // Neither the array nor the misspelled arg became per device: a base without the array is fine.
        const Attrs without_array{.call = 3, .bias_cores = {0}, .base_omits_arr = true};
        clear_l1();
        launch(without_array);
        expect_values(without_array);
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

TEST_F(SpecPerDeviceRunArgsTest, MergedRangeGetsEveryCallsValues) {
    run_sequence(/*with_extra=*/false, /*ranges=*/0, /*expected_programs=*/1);
}

TEST_F(SpecPerDeviceRunArgsTest, OptionalKernelAndSeveralRanges) {
    run_sequence(/*with_extra=*/true, /*ranges=*/1, /*expected_programs=*/2);
}

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

// Per-device values metal would reject (an array of the wrong length on one device, an arg the kernel does not
// declare) are rejected before anything changes, so a hit cannot fail part-way on them and leaves no trace: the
// cached workload still holds the previous call's values, and later calls are unconstrained by the rejected args.
TEST_F(SpecPerDeviceRunArgsTest, BadPerDeviceValuesAreRejectedBeforeAnyChange) { bad_values_rejected(); }

TEST_F(SpecPerDeviceRunArgsTest, BadPerDeviceValuesAreRejectedBeforeAnyChangeSplit) {
    ttnn::device_operation::per_device_run_args::ScopedForceSplit split(*mesh_device_);
    bad_values_rejected();
}

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

TEST_F(SpecPerDeviceRunArgsTest, TraceKeepsCapturedValues) {
    using Attrs = PerDeviceRunArgsOp::operation_attributes_t;
    const Attrs captured{.call = 0, .bias_cores = {0}};
    launch(captured);  // compile and cache
    auto& cq = mesh_device_->mesh_command_queue();
    const auto args = input();
    const auto trace_id = mesh_device_->begin_mesh_trace(cq.id());
    auto traced_output = ttnn::device_operation::launch<PerDeviceRunArgsOp>(captured, args);
    mesh_device_->end_mesh_trace(cq.id(), trace_id);

    const Attrs later{.call = 5, .bias_cores = {1}, .array_per_device = true};
    launch(later);
    clear_l1();
    mesh_device_->replay_mesh_trace(cq.id(), trace_id, /*blocking=*/true);
    expect_values(captured);
    mesh_device_->release_mesh_trace(trace_id);
}

}  // namespace spec_per_device_run_args_test
