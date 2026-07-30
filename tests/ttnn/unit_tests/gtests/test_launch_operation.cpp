// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <type_traits>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operations/examples/example/device/example_device_operation.hpp"
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
// No real op uses create_program_artifacts yet, so the adapter's templated
// method bodies — including the op-owned tensor enumeration and parking added
// for this concept — are never instantiated by a normal build. The checks below
// (a) pin the concept's classification (it recognizes a create_program_artifacts
// factory and is mutually exclusive with the other factory concepts, per
// all_factories_valid), and (b) force-instantiate the adapter so its bodies are
// actually compiled.
//
// NOTE: runtime behavior (cache miss -> hit TensorArg patching, op-owned device
// allocation liveness across dispatches) is NOT exercised here — that requires a
// real op dispatched through the launch path, i.e. the first op port. Until then
// this is compile-coverage only.
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

// Keying: which half of the key each declaration moves, and that a relaxation loosens a TensorSpec
// comparison without letting two tensor_args layouts share a key.
namespace keying_hooks_test {

using tt::tt_metal::experimental::TensorSpecRelaxations;
using ::ttnn::device_operation::detail::compute_op_hash;

struct Attributes {
    uint32_t knob = 0;
    uint32_t seed = 0;
};

struct Inputs {
    Tensor input;
    std::optional<Tensor> first_optional;
    std::optional<Tensor> second_optional;
};

ttsl::SmallVector<TensorSpecRelaxations> strict_for_each_tensor(
    const Inputs& tensor_args, TensorSpecRelaxations input) {
    ttsl::SmallVector<TensorSpecRelaxations> relaxations{input};
    if (tensor_args.first_optional.has_value()) {
        relaxations.emplace_back();
    }
    if (tensor_args.second_optional.has_value()) {
        relaxations.emplace_back();
    }
    return relaxations;
}

struct DefaultKeyedOp {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
};

struct PaddedShapeOnlyOp {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    static ttsl::SmallVector<TensorSpecRelaxations> tensor_args_relaxations(const Inputs& tensor_args) {
        return strict_for_each_tensor(tensor_args, TensorSpecRelaxations{.match_padded_shape_only = true});
    }
};

// The rand pattern: one attribute out of the key.
struct SeedDroppingAttributes {
    uint32_t knob = 0;
    uint32_t seed = 0;

    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("seed");
};

struct SeedDroppingOp {
    using operation_attributes_t = SeedDroppingAttributes;
    using tensor_args_t = Inputs;
};

struct MiscountingOp {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    static ttsl::SmallVector<TensorSpecRelaxations> tensor_args_relaxations(const Inputs&) { return {}; }
};

static_assert(!device_operation::HasTensorArgsRelaxations<DefaultKeyedOp>);
static_assert(device_operation::HasTensorArgsRelaxations<PaddedShapeOnlyOp>);
static_assert(device_operation::HasExcludedAttributes<SeedDroppingAttributes>);
static_assert(device_operation::excluded_attributes_are_members<SeedDroppingAttributes>());
static_assert(device_operation::field_is_excluded<SeedDroppingAttributes, 1>());
static_assert(!device_operation::field_is_excluded<SeedDroppingAttributes, 0>());
static_assert(!device_operation::HasTensorArgsRelaxations<SeedDroppingOp>);
static_assert(!device_operation::HasLegacyProgramHash<SeedDroppingOp>);

// The adapter static_asserts on "spec factory AND legacy hash"; pin that conjunction is detectable.
// The adapter is deliberately not instantiated for the bad one.
struct SpecFactoryOp {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    using program_factory_t = std::variant<ProgramSpecFactory>;
};
struct SpecFactoryOpWithLegacyHash {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    using program_factory_t = std::variant<ProgramSpecFactory>;
    static ttsl::hash::hash_t compute_program_hash(const Attributes&, const Inputs&) { return 0; }
};
struct LegacyFactoryOpWithLegacyHash {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    using program_factory_t = std::variant<NewInfraProgramFactory>;
    static ttsl::hash::hash_t compute_program_hash(const Attributes&, const Inputs&) { return 0; }
};

static_assert(device_operation::HasSpecProgramFactory<SpecFactoryOp>);
static_assert(!device_operation::HasLegacyProgramHash<SpecFactoryOp>);
static_assert(device_operation::HasSpecProgramFactory<SpecFactoryOpWithLegacyHash>);
static_assert(device_operation::HasLegacyProgramHash<SpecFactoryOpWithLegacyHash>);
static_assert(!device_operation::HasSpecProgramFactory<LegacyFactoryOpWithLegacyHash>);
static_assert(device_operation::HasLegacyProgramHash<LegacyFactoryOpWithLegacyHash>);
static_assert(!device_operation::HasSpecProgramFactory<DefaultKeyedOp>);

Tensor make_host_tensor(const ttnn::Shape& logical_shape, Layout layout = Layout::TILE) {
    const tt::tt_metal::TensorSpec spec(
        logical_shape, tt::tt_metal::TensorLayout(DataType::FLOAT32, layout, MemoryConfig{}));
    return Tensor::from_vector(std::vector<float>(logical_shape.volume()), spec);
}

// Same padded_shape (TILE rounds both to 32x32), different logical_shape.
Tensor make_logical_30x32() { return make_host_tensor(ttnn::Shape{1, 1, 30, 32}); }
Tensor make_logical_32x32() { return make_host_tensor(ttnn::Shape{1, 1, 32, 32}); }

TEST(ProgramHashTest, DefaultKeysOnEverything) {
    const Inputs tensor_args{.input = make_logical_30x32()};
    const Inputs other_logical_shape{.input = make_logical_32x32()};

    EXPECT_NE(
        compute_op_hash<DefaultKeyedOp>(Attributes{.knob = 1, .seed = 1}, tensor_args),
        compute_op_hash<DefaultKeyedOp>(Attributes{.knob = 1, .seed = 2}, tensor_args));
    EXPECT_NE(
        compute_op_hash<DefaultKeyedOp>(Attributes{.knob = 1, .seed = 1}, tensor_args),
        compute_op_hash<DefaultKeyedOp>(Attributes{.knob = 1, .seed = 1}, other_logical_shape));
}

TEST(ProgramHashTest, AttributesHashDropsSeedButKeepsTheRest) {
    const Inputs tensor_args{.input = make_logical_30x32()};

    EXPECT_EQ(
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{.knob = 1, .seed = 1}, tensor_args),
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{.knob = 1, .seed = 2}, tensor_args));
    EXPECT_NE(
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{.knob = 1, .seed = 1}, tensor_args),
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{.knob = 2, .seed = 1}, tensor_args));
    EXPECT_NE(
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{}, tensor_args),
        compute_op_hash<SeedDroppingOp>(SeedDroppingAttributes{}, Inputs{.input = make_logical_32x32()}));
}

TEST(ProgramHashTest, PaddedShapeOnlyRelaxationSharesAKey) {
    const Attributes attrs{};
    const Inputs logical_30{.input = make_logical_30x32()};
    const Inputs logical_32{.input = make_logical_32x32()};

    EXPECT_NE(compute_op_hash<DefaultKeyedOp>(attrs, logical_30), compute_op_hash<DefaultKeyedOp>(attrs, logical_32));
    EXPECT_EQ(
        compute_op_hash<PaddedShapeOnlyOp>(attrs, logical_30), compute_op_hash<PaddedShapeOnlyOp>(attrs, logical_32));

    // Relaxing logical_shape relaxes nothing else.
    const Inputs row_major{.input = make_host_tensor(ttnn::Shape{1, 1, 32, 32}, Layout::ROW_MAJOR)};
    EXPECT_NE(
        compute_op_hash<PaddedShapeOnlyOp>(attrs, logical_32), compute_op_hash<PaddedShapeOnlyOp>(attrs, row_major));
}

TEST(ProgramHashTest, RelaxationPreservesTensorArgsStructure) {
    const Attributes attrs{};
    const Tensor tensor = make_logical_32x32();

    // Same tensor and count, different slot -> different program, so different key.
    const Inputs in_first_slot{.input = tensor, .first_optional = tensor};
    const Inputs in_second_slot{.input = tensor, .second_optional = tensor};
    EXPECT_NE(
        compute_op_hash<PaddedShapeOnlyOp>(attrs, in_first_slot),
        compute_op_hash<PaddedShapeOnlyOp>(attrs, in_second_slot));

    EXPECT_NE(
        compute_op_hash<PaddedShapeOnlyOp>(attrs, Inputs{.input = tensor}),
        compute_op_hash<PaddedShapeOnlyOp>(attrs, in_first_slot));
}

TEST(ProgramHashTest, MiscountedRelaxationsAreRejected) {
    EXPECT_ANY_THROW(compute_op_hash<MiscountingOp>(Attributes{}, Inputs{.input = make_logical_32x32()}));
}

}  // namespace keying_hooks_test

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

// Every way a Metal 2.0 op may shape its key, and the ways it must not. Fake ops, host tensors only.
namespace keying_gaps_test {

using keying_hooks_test::Attributes;
using keying_hooks_test::DefaultKeyedOp;
using keying_hooks_test::Inputs;
using keying_hooks_test::make_logical_32x32;
using ::ttnn::device_operation::detail::canonical_declared_attributes;
using ::ttnn::device_operation::detail::compute_op_hash;

// Presence-only: the key cares whether a semaphore was supplied, not which one. Store the bool, exclude
// the value (ccl/reduce_scatter_minimal_async keys barrier_semaphore.has_value() today).
struct PresenceAttributes {
    bool has_semaphore = false;
    std::optional<uint32_t> semaphore;

    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("semaphore");
};
struct PresenceOp {
    using operation_attributes_t = PresenceAttributes;
    using tensor_args_t = Inputs;
};

// Conditional neutralization: the kernels see an effective value, so store it and exclude the raw one
// (transformer/ring_joint_sdpa keys has_kv_pad_rotation() ? 0 : logical_n today).
struct EffectiveAttributes {
    bool pad_rotation = false;
    uint32_t raw_n = 0;
    uint32_t effective_n = 0;

    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("raw_n");
};
struct EffectiveOp {
    using operation_attributes_t = EffectiveAttributes;
    using tensor_args_t = Inputs;
};

// Sub-field projection: an op that keys some fields of a struct member holds those values flat and excludes
// the member (ccl/strided_all_gather_minimal_matmul_async projects 11 fields of one member today).
struct NestedParams {
    uint32_t keyed = 0;
    uint32_t ignored = 0;
};
struct ProjectedAttributes {
    NestedParams params;
    uint32_t keyed = 0;

    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("params");
};
struct ProjectedOp {
    using operation_attributes_t = ProjectedAttributes;
    using tensor_args_t = Inputs;
};

// A misspelled or renamed exclusion excludes nothing; the adapter static_asserts on this predicate.
struct TypoAttributes {
    uint32_t knob = 0;

    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("knobb");
};

// The two banned mechanisms, so the predicates the asserts use are pinned as detecting them.
struct ToHashAttributes {
    uint32_t knob = 0;
    ttsl::hash::hash_t to_hash() const { return 0; }
};
struct TupleAttributes {
    uint32_t knob = 0;

    static constexpr auto attribute_names = std::forward_as_tuple("knob");
    auto attribute_values() const { return std::forward_as_tuple(knob); }
};

static_assert(device_operation::excluded_attributes_are_members<PresenceAttributes>());
static_assert(device_operation::excluded_attributes_are_members<EffectiveAttributes>());
static_assert(!device_operation::excluded_attributes_are_members<TypoAttributes>());
static_assert(ttsl::reflection::detail::supports_to_hash_v<ToHashAttributes>);
static_assert(device_operation::HasAttributeNames<TupleAttributes>);
static_assert(!device_operation::HasAttributeNames<PresenceAttributes>);

TEST(KeyingGapsTest, PresenceKeysWhetherNotWhich) {
    const Inputs tensor_args{.input = make_logical_32x32()};
    const PresenceAttributes with_one{.has_semaphore = true, .semaphore = 1};
    PresenceAttributes with_another = with_one;
    with_another.semaphore = 2;
    const PresenceAttributes without{.has_semaphore = false, .semaphore = std::nullopt};

    EXPECT_EQ(
        compute_op_hash<PresenceOp>(with_one, tensor_args), compute_op_hash<PresenceOp>(with_another, tensor_args));
    EXPECT_NE(compute_op_hash<PresenceOp>(with_one, tensor_args), compute_op_hash<PresenceOp>(without, tensor_args));
}

TEST(KeyingGapsTest, EffectiveValueKeysInsteadOfRaw) {
    const Inputs tensor_args{.input = make_logical_32x32()};
    const EffectiveAttributes attrs{.pad_rotation = true, .raw_n = 8, .effective_n = 0};
    EffectiveAttributes other_raw = attrs;
    other_raw.raw_n = 99;
    EffectiveAttributes other_effective = attrs;
    other_effective.effective_n = 1;

    EXPECT_EQ(compute_op_hash<EffectiveOp>(attrs, tensor_args), compute_op_hash<EffectiveOp>(other_raw, tensor_args));
    EXPECT_NE(
        compute_op_hash<EffectiveOp>(attrs, tensor_args), compute_op_hash<EffectiveOp>(other_effective, tensor_args));
}

TEST(KeyingGapsTest, ProjectionKeysTheFlatCopyNotTheMember) {
    const Inputs tensor_args{.input = make_logical_32x32()};
    const ProjectedAttributes attrs{.params = {.keyed = 3, .ignored = 4}, .keyed = 3};
    ProjectedAttributes member_changed = attrs;
    member_changed.params.ignored = 99;
    member_changed.params.keyed = 99;
    ProjectedAttributes flat_changed = attrs;
    flat_changed.keyed = 9;

    EXPECT_EQ(
        compute_op_hash<ProjectedOp>(attrs, tensor_args), compute_op_hash<ProjectedOp>(member_changed, tensor_args));
    EXPECT_NE(
        compute_op_hash<ProjectedOp>(attrs, tensor_args), compute_op_hash<ProjectedOp>(flat_changed, tensor_args));
}

// The exact key must split exactly what the hash splits, or a 64-bit collision becomes a wrong cache hit.
TEST(KeyingGapsTest, CanonicalKeyMirrorsTheHash) {
    const PresenceAttributes with_one{.has_semaphore = true, .semaphore = 1};
    PresenceAttributes with_another = with_one;
    with_another.semaphore = 2;
    PresenceAttributes without = with_one;
    without.has_semaphore = false;
    EXPECT_EQ(canonical_declared_attributes(with_one), canonical_declared_attributes(with_another));
    EXPECT_NE(canonical_declared_attributes(with_one), canonical_declared_attributes(without));

    const ProjectedAttributes projected{.params = {.keyed = 3, .ignored = 4}, .keyed = 3};
    ProjectedAttributes projected_member_changed = projected;
    projected_member_changed.params.keyed = 99;
    ProjectedAttributes projected_flat_changed = projected;
    projected_flat_changed.keyed = 9;
    EXPECT_EQ(canonical_declared_attributes(projected), canonical_declared_attributes(projected_member_changed));
    EXPECT_NE(canonical_declared_attributes(projected), canonical_declared_attributes(projected_flat_changed));

    // An op that declares nothing keeps the plain encoding.
    const Attributes plain{.knob = 1, .seed = 7};
    EXPECT_EQ(canonical_declared_attributes(plain), ttsl::hash::canonical_key(plain));
}

// Declaring nothing must key exactly as before this mechanism existed.
TEST(KeyingGapsTest, UndeclaredOpsKeyByPlainReflection) {
    const Inputs tensor_args{.input = make_logical_32x32()};
    const Attributes attrs{.knob = 3, .seed = 4};
    EXPECT_EQ(
        compute_op_hash<DefaultKeyedOp>(attrs, tensor_args),
        ttsl::hash::hash_objects_with_default_seed(ttsl::hash::type_hash<DefaultKeyedOp>, attrs, tensor_args));
}

}  // namespace keying_gaps_test
}  // namespace
}  // namespace ttnn
