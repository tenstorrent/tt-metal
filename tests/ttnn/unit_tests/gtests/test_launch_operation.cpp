// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <gmock/gmock.h>
#include <type_traits>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operations/examples/example/device/example_device_operation.hpp"
#include "ttnn/operations/reduction/prod/device/prod_all_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::ttnn::device_operation::mesh_device_operation_utils::all_tensors_have_uniform_storage;
using ::ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates;
using ::ttnn::device_operation::mesh_device_operation_utils::filter_tensor_shards;

// Returns a dummy device tensor with `num_device_shards` populated.
Tensor make_tensor_with_num_shards(int num_device_shards, MeshDevice* mesh_device, int shard_dim = 0) {
    TT_FATAL(num_device_shards > 0 && num_device_shards <= mesh_device->num_devices(), "Invalid number of shards");

    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        ttsl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *distributed::shard_tensor_to_mesh_mapper(*mesh_device, shard_dim),
        *mesh_device);
}

// Returns a dummy device tensor distributed according to the `mapper_config`.
Tensor make_tensor_with_mapper_config(
    int num_device_shards, MeshDevice* mesh_device, const distributed::MeshMapperConfig& mapper_config) {
    auto mapper = distributed::create_mesh_mapper(*mesh_device, mapper_config);
    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        ttsl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *mapper,
        *mesh_device);
}

struct SharedVariables {};
struct OperationAttributes {};

// New-infra style program factory that uses the "create" method (non-heterogeneous dispatch)
struct NewInfraProgramFactory {
    using shared_variables_t = SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_program_t(tt::tt_metal::Program(), SharedVariables{});
    }

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

// New-infra style program factory that uses the "create_at" method (heterogeneous dispatch)
struct NewInfraWorkloadFactory {
    using shared_variables_t = SharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_mesh_workload_t(
            tt::tt_metal::distributed::MeshWorkload(),
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t>());
    }

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<NewInfraWorkloadFactory>);
static_assert(ttnn::device_operation::ProgramFactoryConcept<NewInfraProgramFactory>);

// ---------------------------------------------------------------------------
// ProgramSpecFactoryConcept (Metal 2.0 op-porting stepping stone)
//
// Real ops already use create_program_artifacts (typecast, prod_all, …). The
// checks below still (a) pin the concept's classification (it recognizes a
// create_program_artifacts factory and is mutually exclusive with the other
// factory concepts, per all_factories_valid), and (b) force-instantiate the
// adapter so its bodies are compiled even when a particular TU does not
// instantiate them through a real op.
//
// NOTE: runtime occupancy mapping (programs only on tensor_coords) is covered
// by ProgramSpecAdapterProgramsOnUniformTensorCoords and
// ProgramSpecAdapterProgramsOnlyOnUnevenTensorCoords below. Hit-path TensorArg
// patching / op-owned liveness remain covered by op-level program-cache tests.
// ---------------------------------------------------------------------------
struct ProgramSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const OperationAttributes& /*attrs*/, const Tensor& /*tensor_args*/, Tensor& /*tensor_return_value*/) {
        return ttnn::device_operation::ProgramArtifacts{};
    }
};

// Same, but additionally provides override_runtime_arguments -> classified as the
// custom variant, whose cache-hit path applies the returned ProgramRunArgs via
// UpdateProgramRunArgs.
struct CustomProgramSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const OperationAttributes& /*attrs*/, const Tensor& /*tensor_args*/, Tensor& /*tensor_return_value*/) {
        return ttnn::device_operation::ProgramArtifacts{};
    }
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/ = std::nullopt) {
        return {};
    }
};

// Programs vary across the mesh: one call returns workload-scoped resources plus a program per
// coordinate range.
struct MeshWorkloadSpecFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
        return ttnn::device_operation::MeshWorkloadArtifacts{};
    }
};

// Same, plus a per-range cache-hit run-args refresh.
struct MeshWorkloadSpecFactoryWithOverride {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
        return ttnn::device_operation::MeshWorkloadArtifacts{};
    }
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRange& /*range*/) {
        return {};
    }
};

// Minimal device operation supplying just the typedefs the adapter inherits.
struct ProgramSpecMinimalOp {
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
};

static_assert(ttnn::device_operation::ProgramSpecFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::MeshWorkloadFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramDescriptorFactoryConcept<ProgramSpecFactory>);

// The custom variant is mutually exclusive with the base one.
static_assert(ttnn::device_operation::CustomProgramSpecFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::MeshWorkloadFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramDescriptorFactoryConcept<CustomProgramSpecFactory>);

// A mesh-workload factory is its own flavor, and an override doesn't pull it into the others.
static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);

// Every flavor is exactly one alternative as far as a program_factory_t variant is concerned.
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<ProgramSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<CustomProgramSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<MeshWorkloadSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<MeshWorkloadSpecFactoryWithOverride>>);

template <typename Factory>
using WorkloadSpecAdapter =
    device_operation::MeshDeviceOperationAdapter<ProgramSpecMinimalOp>::MeshWorkloadSpecFactoryAdapter<Factory>;

// The cache-hit path forks on this; pin both directions.
static_assert(!WorkloadSpecAdapter<MeshWorkloadSpecFactory>::has_override_runtime_arguments());
static_assert(WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::has_override_runtime_arguments());

// Compile-coverage: taking the adapter methods' addresses ODR-uses them, forcing
// the (otherwise un-instantiated) bodies to compile. Never dispatched.
TEST(LaunchOperationTest, ProgramSpecAdapterCompiles) {
    using Adapter = device_operation::MeshDeviceOperationAdapter<
        ProgramSpecMinimalOp>::ProgramSpecMeshWorkloadFactoryAdapter<ProgramSpecFactory>;
    [[maybe_unused]] auto create = &Adapter::create_mesh_workload;
    [[maybe_unused]] auto apply = &Adapter::apply_descriptor;
    [[maybe_unused]] auto resolve = &Adapter::resolve_bindings;

    using CustomAdapter = device_operation::MeshDeviceOperationAdapter<
        ProgramSpecMinimalOp>::CustomProgramSpecMeshWorkloadFactoryAdapter<CustomProgramSpecFactory>;
    [[maybe_unused]] auto ccreate = &CustomAdapter::create_mesh_workload;
    [[maybe_unused]] auto capply = &CustomAdapter::apply_descriptor;

    // The mesh-workload adapter is a separate template; instantiate both its hit-path branches.
    [[maybe_unused]] auto wcreate = &WorkloadSpecAdapter<MeshWorkloadSpecFactory>::create_mesh_workload;
    [[maybe_unused]] auto wapply = &WorkloadSpecAdapter<MeshWorkloadSpecFactory>::apply_descriptor;
    [[maybe_unused]] auto wocreate = &WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::create_mesh_workload;
    [[maybe_unused]] auto woapply = &WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::apply_descriptor;
    SUCCEED();
}

// SupportsPerCoreAllocation gates whether launch() will accept a per-core allocated tensor. No op
// declares supports_per_core_allocation today, so the accept path is otherwise never exercised --
// a concept that could never match would look identical at runtime. Pin all three directions here:
// absent, true, and explicitly false.
namespace per_core_opt_in_test {
struct NoDeclaration {};
struct OptedIn {
    static constexpr bool supports_per_core_allocation = true;
};
struct OptedOut {
    static constexpr bool supports_per_core_allocation = false;
};
}  // namespace per_core_opt_in_test

static_assert(!device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::NoDeclaration>);
static_assert(device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::OptedIn>);
static_assert(!device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::OptedOut>);

TEST(LaunchOperationTest, MeshDeviceOperationAdapterGetName) {
    using ::ttnn::operations::examples::ExampleDeviceOperation;
    EXPECT_EQ(
        device_operation::MeshDeviceOperationAdapter<ExampleDeviceOperation>::get_type_name(
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42}),
        "ExampleDeviceOperation");
}

using LaunchOperation2x4Test = tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(LaunchOperation2x4Test, UniformTensor) {
    const tt::tt_metal::TensorSpec tensor_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = ttnn::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));

    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));
}

TEST_F(LaunchOperation2x4Test, UnevenTensor) {
    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());

    EXPECT_THAT(uneven_tensor.device_storage().get_coords(), SizeIs(2));

    EXPECT_FALSE(all_tensors_have_uniform_storage(uneven_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(uneven_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, FilterTensorShards) {
    const tt::tt_metal::TensorSpec tensor_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = ttnn::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first 2 shards and the last 3 shards.
    auto filtered_tensor = filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},
         ttnn::MeshCoordinate{0, 1},
         ttnn::MeshCoordinate{1, 1},
         ttnn::MeshCoordinate{1, 2},
         ttnn::MeshCoordinate{1, 3}},
        full_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(filtered_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first and the last shards.
    filtered_tensor = filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},  //
         ttnn::MeshCoordinate{1, 3}},
        filtered_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(filtered_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{1, 3}));

    // Filter the rest.
    filtered_tensor = filter_tensor_shards(/*tensor_coordinates=*/{}, filtered_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(extract_tensor_coordinates(filtered_tensor), IsEmpty());
}

TEST_F(LaunchOperation2x4Test, LaunchOpFilterTensorShards) {
    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_TRUE(all_tensors_have_uniform_storage(sum));
    EXPECT_THAT(
        extract_tensor_coordinates(sum),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(sum_uneven));
    EXPECT_THAT(
        extract_tensor_coordinates(sum_uneven),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, CachingHeterogeneousDispatch) {
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 0);

    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto sum2 = ttnn::add(full_tensor, full_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);

    auto sum3 = ttnn::add(uneven_tensor, uneven_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);
}

// Emits a program only for non-idle coordinates, so one create_mesh_workload covers the range split
// and the sharing of workload-scoped resources.
struct PerCoordProdAllFactory {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    static bool is_idle(const ttnn::MeshCoordinate& coord) { return coord[1] % 2 == 1; }

    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const Op::operation_attributes_t& attrs,
        const Op::tensor_args_t& tensor_args,
        Op::tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords) {
        ttnn::device_operation::MeshWorkloadArtifacts artifacts;
        for (const auto& range : tensor_coords.ranges()) {
            for (const auto& coord : range) {
                if (is_idle(coord)) {
                    continue;
                }
                auto program =
                    Op::ProdAllProgramFactory::create_program_artifacts(attrs, tensor_args, tensor_return_value);
                artifacts.programs.push_back(
                    {.range = ttnn::MeshCoordinateRange(coord),
                     .spec = std::move(program.spec),
                     .run_params = std::move(program.run_params)});
            }
        }
        return artifacts;
    }
};

TEST_F(LaunchOperation2x4Test, MeshWorkloadSpecAdapterMapsProgramsPerCoordinate) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter =
        device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<PerCoordProdAllFactory>;
    static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<PerCoordProdAllFactory>);

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    // One single-coordinate program per non-idle coordinate; idle ones contribute nothing.
    size_t expected = 0;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device_->shape())) {
        const ttnn::MeshCoordinateRange range(coord);
        const bool idle = PerCoordProdAllFactory::is_idle(coord);
        EXPECT_EQ(cached.workload.get_programs().contains(range), !idle) << "coord " << coord;
        EXPECT_EQ(cached.shared_variables.contains(range), !idle) << "coord " << coord;
        expected += idle ? 0 : 1;
    }
    EXPECT_GT(expected, 0u);
    EXPECT_EQ(cached.workload.get_programs().size(), expected);
    EXPECT_EQ(cached.shared_variables.size(), expected);
}

// Same per-coordinate mapping, but with an override so the cache-hit path takes the
// UpdateProgramRunArgs branch instead of the tensor-only refresh.
struct PerCoordProdAllFactoryWithOverride : PerCoordProdAllFactory {
    static std::atomic<int> override_calls;

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const Op::operation_attributes_t& attrs,
        const Op::tensor_args_t& tensor_args,
        Op::tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRange& /*range*/) {
        ++override_calls;
        return Op::ProdAllProgramFactory::create_program_artifacts(attrs, tensor_args, tensor_return_value).run_params;
    }
};
std::atomic<int> PerCoordProdAllFactoryWithOverride::override_calls{0};

TEST_F(LaunchOperation2x4Test, MeshWorkloadSpecAdapterAppliesDescriptorOnCacheHit) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter =
        device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<PerCoordProdAllFactory>;
    using OverrideAdapter = device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<
        PerCoordProdAllFactoryWithOverride>;
    static_assert(!Adapter::has_override_runtime_arguments());
    static_assert(OverrideAdapter::has_override_runtime_arguments());

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    // Tensor-refresh branch: a second apply must not disturb the per-coordinate mapping.
    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);
    const auto ranges_before = cached.workload.get_programs().size();
    Adapter::apply_descriptor(cached, attrs, tensor_args, output);
    EXPECT_EQ(cached.workload.get_programs().size(), ranges_before);
    for (const auto& [range, program] : cached.workload.get_programs()) {
        EXPECT_TRUE(cached.shared_variables.contains(range)) << "range " << range;
    }

    // Override branch: called once per range, not once for the whole workload.
    PerCoordProdAllFactoryWithOverride::override_calls = 0;
    auto override_cached = OverrideAdapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);
    const auto override_ranges = override_cached.workload.get_programs().size();
    EXPECT_GT(override_ranges, 1u);
    OverrideAdapter::apply_descriptor(override_cached, attrs, tensor_args, output);
    EXPECT_EQ(PerCoordProdAllFactoryWithOverride::override_calls.load(), static_cast<int>(override_ranges));
}

TEST_F(LaunchOperation2x4Test, ProgramSpecAdapterProgramsOnUniformTensorCoords) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter = device_operation::MeshDeviceOperationAdapter<Op>::ProgramSpecMeshWorkloadFactoryAdapter<
        Op::ProdAllProgramFactory>;

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    EXPECT_TRUE(all_tensors_have_uniform_storage(input));

    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    // Fast path in launch: one range covering the entire mesh.
    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    const ttnn::MeshCoordinateRange full_mesh(mesh_device_->shape());
    EXPECT_EQ(cached.workload.get_programs().size(), 1u);
    EXPECT_EQ(cached.shared_variables.size(), 1u);
    EXPECT_TRUE(cached.workload.get_programs().contains(full_mesh));
    EXPECT_TRUE(cached.shared_variables.contains(full_mesh));
}

TEST_F(LaunchOperation2x4Test, ProgramSpecAdapterProgramsOnlyOnUnevenTensorCoords) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter = device_operation::MeshDeviceOperationAdapter<Op>::ProgramSpecMeshWorkloadFactoryAdapter<
        Op::ProdAllProgramFactory>;

    auto input = make_tensor_with_num_shards(2, mesh_device_.get());
    EXPECT_FALSE(all_tensors_have_uniform_storage(input));
    EXPECT_THAT(
        extract_tensor_coordinates(input),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));

    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    // Slow path in launch: merge occupied coordinates one by one.
    ttnn::MeshCoordinateRangeSet tensor_coords;
    for (const auto& coord : extract_tensor_coordinates(input, mesh_device_.get())) {
        tensor_coords.merge(ttnn::MeshCoordinateRange(coord, coord));
    }

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    EXPECT_EQ(cached.workload.get_programs().size(), tensor_coords.ranges().size());
    EXPECT_EQ(cached.shared_variables.size(), tensor_coords.ranges().size());
    for (const auto& range : tensor_coords.ranges()) {
        EXPECT_TRUE(cached.workload.get_programs().contains(range));
        EXPECT_TRUE(cached.shared_variables.contains(range));
    }

    // Unpopulated mesh coordinates must not have a program associated with them.
    for (const auto& coord : {
             ttnn::MeshCoordinate{0, 2},
             ttnn::MeshCoordinate{0, 3},
             ttnn::MeshCoordinate{1, 0},
             ttnn::MeshCoordinate{1, 1},
             ttnn::MeshCoordinate{1, 2},
             ttnn::MeshCoordinate{1, 3},
         }) {
        for (const auto& [program_range, _] : cached.workload.get_programs()) {
            EXPECT_FALSE(program_range.contains(coord))
                << "unexpected program covering unpopulated mesh coordinate " << coord;
        }
    }
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopology) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get());

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyAugmentedDistribution) {
    auto config_1 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(2, 2),
    };
    auto input_tensor_1 = make_tensor_with_mapper_config(4, mesh_device_.get(), config_1);
    auto config_2 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(1, 4),
    };
    auto input_tensor_2 = make_tensor_with_mapper_config(8, mesh_device_.get(), config_2);
    auto config_3 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(8),
    };
    auto input_tensor_3 = make_tensor_with_mapper_config(16, mesh_device_.get(), config_3);

    auto sum_1 = ttnn::add(input_tensor_1, input_tensor_2);
    auto sum_2 = ttnn::add(input_tensor_2, input_tensor_1);
    auto sum_3 = ttnn::add(input_tensor_3, input_tensor_2);

    EXPECT_EQ(sum_1.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_1.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}}));
    EXPECT_EQ(sum_2.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_2.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
    EXPECT_EQ(sum_3.tensor_topology().distribution_shape(), MeshShape(1, 4));
    EXPECT_EQ(
        sum_3.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyMultipleShardDims) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get(), /*shard_dim=*/1);

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

}  // namespace
}  // namespace ttnn
