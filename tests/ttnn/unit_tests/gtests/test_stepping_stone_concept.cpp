// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/tensor/tensor.hpp"

// Compile-time contract for the Metal 2.0 "stepping stone" factory concept
// (IntermediateStepMetalV2FactoryConcept). The stepping stone lets a porter prove Metal 2.0 functional
// correctness with a single create_everything() method, before climbing the ladder to the real (fast)
// spec-factory concepts. These checks pin the concept's shape so a rename or stray edit to
// operation_concepts.hpp can't silently widen or break it. No hardware required: the static_asserts fire
// at compile time; the TEST is a runnable mirror. Synthetic types use an `Ss` prefix to stay clear of
// the unity-build neighbors in this directory.

namespace ttnn {
namespace {

namespace concepts = ttnn::device_operation;

struct SsOperationAttributes {};

// A valid stepping-stone factory: exactly one method, create_everything, returning the artifact
// (spec + run args + op-owned tensors). A declaration is enough for the concept's address-of check.
struct SsSteppingStoneFactory {
    // Defined (not just declared) so the forced adapter instantiation below can odr-use it.
    static concepts::MetalV2IntermediateStepArtifact create_everything(
        const SsOperationAttributes&, const Tensor&, Tensor&) {
        return {};
    }
};

// A prior-framework ProgramDescriptor factory — confirms the stepping-stone concept excludes the legacy
// factory shapes (and that legacy classification is unaffected by the concept rename).
struct SsDescriptorFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const SsOperationAttributes&, const Tensor&, Tensor&);
};

// A factory with no create_everything is not a stepping-stone factory.
struct SsNotAStepFactory {
    static int unrelated_method();
};

// The stepping-stone factory satisfies its concept...
static_assert(concepts::IntermediateStepMetalV2FactoryConcept<SsSteppingStoneFactory>);
// ...and is none of the legacy factory shapes (the concept explicitly excludes them).
static_assert(!concepts::ProgramFactoryConcept<SsSteppingStoneFactory>);
static_assert(!concepts::MeshWorkloadFactoryConcept<SsSteppingStoneFactory>);
static_assert(!concepts::ProgramDescriptorFactoryConcept<SsSteppingStoneFactory>);

// A factory missing create_everything does not satisfy the concept.
static_assert(!concepts::IntermediateStepMetalV2FactoryConcept<SsNotAStepFactory>);

// A legacy descriptor factory is classified as such, and is NOT a stepping-stone factory.
static_assert(concepts::ProgramDescriptorFactoryConcept<SsDescriptorFactory>);
static_assert(!concepts::IntermediateStepMetalV2FactoryConcept<SsDescriptorFactory>);

// AllFactoriesValid requires each variant alternative to satisfy exactly one factory concept. A single
// stepping-stone alternative qualifies, as does a single legacy descriptor alternative.
static_assert(concepts::AllFactoriesValid<std::variant<SsSteppingStoneFactory>>);
static_assert(concepts::AllFactoriesValid<std::variant<SsDescriptorFactory>>);

// --- Force the adapter's method bodies to compile -------------------------------------------------
// The stepping-stone adapter is a template that nothing instantiates yet (machinery-only), so without
// this the build would not type-check create_mesh_workload / apply_descriptor / build — the first real
// op-owned port would be their first compile. A minimal synthetic device-op supplies the five typedefs
// MeshDeviceOperationAdapter reads eagerly; taking the addresses of the adapter's static methods odr-uses
// them, instantiating their bodies. Nothing here executes — no device required, this is a compile check.
struct SsMinimalOp {
    using operation_attributes_t = SsOperationAttributes;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SsSteppingStoneFactory>;
};

[[maybe_unused]] void ss_force_instantiate_adapter() {
    using Adapter = concepts::MeshDeviceOperationAdapter<
        SsMinimalOp>::IntermediateMetalV2StepMeshWorkloadFactoryAdapter<SsSteppingStoneFactory>;
    auto build_ptr = &Adapter::build;
    auto create_ptr = &Adapter::create_mesh_workload;
    auto apply_ptr = &Adapter::apply_descriptor;
    (void)build_ptr;
    (void)create_ptr;
    (void)apply_ptr;
}

TEST(SteppingStoneConceptTest, ConceptClassification) {
    EXPECT_TRUE(concepts::IntermediateStepMetalV2FactoryConcept<SsSteppingStoneFactory>);
    EXPECT_FALSE(concepts::IntermediateStepMetalV2FactoryConcept<SsDescriptorFactory>);
    EXPECT_FALSE(concepts::IntermediateStepMetalV2FactoryConcept<SsNotAStepFactory>);
    EXPECT_TRUE((concepts::AllFactoriesValid<std::variant<SsSteppingStoneFactory>>));
}

}  // namespace
}  // namespace ttnn
