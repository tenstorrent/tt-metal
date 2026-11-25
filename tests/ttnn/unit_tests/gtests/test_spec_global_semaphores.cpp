// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphores in the spec adapters. Every spec factory may bind its caller's GlobalSemaphores from
// operation_attributes_t::global_semaphores, which every cache hit rebinds, on each of the adapters' hit paths. A
// MeshWorkloadSpecFactoryConcept factory may also allocate its own in MeshWorkloadArtifacts::global_semaphores, which
// the program cache keeps allocated, at the same address, across hits. The op's kernel up()s each GlobalSemaphore it
// binds once per call, and the test reads their values on every device.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/mesh/allocator.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <internal/mesh_device.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/command_list.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace spec_global_semaphores_test {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::GlobalSemaphore;
using ttnn::Tensor;
namespace m2 = tt::tt_metal;

const CoreCoord kCore{0, 0};
const char kKernel[] = "count";
const char kCaller[] = "caller";
const char kParked[] = "parked";

// up()s the caller's GlobalSemaphore once.
constexpr const char* kCallerSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    GlobalSemaphore caller(global_sem::caller);
    caller.up();
}
)";
// up()s the caller's GlobalSemaphore and the factory's parked one once each.
constexpr const char* kCallerAndParkedSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    GlobalSemaphore caller(global_sem::caller);
    caller.up();
    GlobalSemaphore parked(global_sem::parked);
    parked.up();
}
)";

struct Attrs {
    uint32_t call = 0;  // not hashed
    // The caller's GlobalSemaphores, bound by position: not hashed, rebound on every cache hit.
    std::vector<GlobalSemaphore> global_semaphores;
    // When set, the factory binds this instead of global_semaphores[0]. It is in neither place the adapter rebinds
    // from, so the cache miss throws unless the factory's override supplies its run args on every hit. Hashed (by
    // presence).
    std::optional<GlobalSemaphore> stray;
};
struct Args {
    Tensor input;
};

m2::ProgramSpec spec(bool with_parked) {
    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{kKernel},
        .source = m2::KernelSpec::SourceCode{with_parked ? kCallerAndParkedSource : kCallerSource},
        .num_threads = 1,
        .hw_config = m2::DataMovementHardwareConfig{
            .config_1xx = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0}}};
    m2::ProgramSpec program{.name = "spec_global_semaphores_test"};
    for (const char* name :
         with_parked ? std::vector<const char*>{kCaller, kParked} : std::vector<const char*>{kCaller}) {
        kernel.global_semaphore_bindings.push_back(m2::GlobalSemaphoreBinding{
            .global_semaphore_parameter_name = m2::GlobalSemaphoreParamName{name}, .accessor_name = name});
        program.global_semaphore_parameters.push_back(
            m2::GlobalSemaphoreParameter{.unique_id = m2::GlobalSemaphoreParamName{name}});
    }
    program.kernels.push_back(std::move(kernel));
    program.work_units = {
        m2::WorkUnitSpec{.name = "core", .kernels = {m2::KernelSpecName{kKernel}}, .target_nodes = kCore}};
    return program;
}

// Binds the caller's first GlobalSemaphore (or the stray one) and, if given, a parked one.
m2::ProgramRunArgs run_args(const Attrs& attrs, const GlobalSemaphore* parked) {
    m2::ProgramRunArgs args;
    args.global_semaphore_args.emplace(
        m2::GlobalSemaphoreParamName{kCaller},
        std::cref(attrs.stray.has_value() ? *attrs.stray : attrs.global_semaphores.at(0)));
    if (parked != nullptr) {
        args.global_semaphore_args.emplace(m2::GlobalSemaphoreParamName{kParked}, std::cref(*parked));
    }
    return args;
}

GlobalSemaphore make_semaphore(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    return ttnn::global_semaphore::create_global_semaphore(
        mesh_device, tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(kCore)), /*initial_value=*/0);
}

// SPMD: the caller's GlobalSemaphore only.
struct SpmdFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const Attrs& attrs, const Args& /*args*/, Tensor& /*output*/) {
        return {.spec = spec(/*with_parked=*/false), .run_params = run_args(attrs, nullptr)};
    }
};
// SPMD with a hit override that returns nothing: the adapter's rebinding is all that refreshes the semaphore.
struct CustomSpmdFactory : SpmdFactory {
    static m2::ProgramRunArgs override_runtime_arguments(
        const Attrs& /*attrs*/,
        const Args& /*args*/,
        Tensor& /*output*/,
        const std::optional<ttnn::MeshCoordinate>& /*coord*/ = std::nullopt) {
        return {};
    }
};

// MeshWorkload: allocates and parks one GlobalSemaphore, and binds it next to the caller's.
struct MeshFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const Attrs& attrs, const Args& args, Tensor& /*output*/, const ttnn::MeshCoordinateRangeSet& tensor_coords) {
        ttnn::device_operation::MeshWorkloadArtifacts artifacts;
        artifacts.global_semaphores.push_back(make_semaphore(args.input.device()));
        for (const auto& range : tensor_coords.ranges()) {
            artifacts.programs.push_back(
                {.range = range,
                 .spec = spec(/*with_parked=*/true),
                 .run_params = run_args(attrs, &artifacts.global_semaphores.front())});
        }
        return artifacts;
    }
};
struct MeshOverrideFactory : MeshFactory {
    static m2::ProgramRunArgs override_runtime_arguments(
        const Attrs& /*attrs*/, const Args& /*args*/, Tensor& /*output*/, const ttnn::MeshCoordinateRange& /*range*/) {
        return {};
    }
};
struct MeshFixedPerDeviceFactory : MeshFactory {
    static constexpr bool uses_per_device_run_args = true;
    static constexpr bool fixed_per_device_run_args = true;
};
struct MeshPerDeviceOverrideFactory : MeshFactory {
    static constexpr bool uses_per_device_run_args = true;
    static ttnn::device_operation::RangeRunArgs override_runtime_arguments(
        const Attrs& /*attrs*/, const Args& /*args*/, Tensor& /*output*/, const ttnn::MeshCoordinateRange& /*range*/) {
        return {};
    }
};

template <typename Factory>
struct GlobalSemaphoreOp {
    using operation_attributes_t = Attrs;
    using tensor_args_t = Args;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
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
        return ttsl::hash::hash_objects_with_default_seed(attrs.stray.has_value());
    }
};

class SpecGlobalSemaphoresTest : public tt::tt_metal::GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        mesh_device_->clear_program_cache();
    }

    Args input() const {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, 1, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig{}));
        return {.input = Tensor::from_vector(std::vector<uint32_t>(32, 0), spec, mesh_device_.get())};
    }

    GlobalSemaphore make_semaphore() const { return spec_global_semaphores_test::make_semaphore(mesh_device_.get()); }

    template <typename Factory>
    void launch(const Attrs& attrs) {
        (void)ttnn::device_operation::launch<GlobalSemaphoreOp<Factory>>(attrs, input());
        tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    }

    // The word at `address` on kCore of every device.
    std::vector<uint32_t> values_at(tt::tt_metal::DeviceAddr address) {
        std::vector<uint32_t> out;
        for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> word;
            tt::tt_metal::detail::ReadFromDeviceL1(
                tt::tt_metal::internal::get_device(*mesh_device_, coord),
                kCore,
                static_cast<uint32_t>(address),
                sizeof(uint32_t),
                word);
            out.push_back(word.at(0));
        }
        return out;
    }

    // The semaphore's value on kCore of every device.
    std::vector<uint32_t> values(const GlobalSemaphore& semaphore) { return values_at(semaphore.address()); }

    std::vector<uint32_t> all(uint32_t value) const {
        return std::vector<uint32_t>(mesh_device_->shape().mesh_size(), value);
    }

    // The GlobalSemaphore a MeshWorkload factory parked in the cache entry for `attrs`: every program shares it.
    template <typename Factory>
    const GlobalSemaphore& parked_semaphore(const Attrs& attrs) {
        using MeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<GlobalSemaphoreOp<Factory>>;
        using Adapter = typename MeshOp::template MeshWorkloadSpecFactoryAdapter<Factory>;
        const auto args = input();
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<MeshOp>(attrs, args, mesh_device_.get());
        EXPECT_TRUE(lookup.hit);
        const auto& shared_variables = mesh_device_->get_program_cache()
                                           .get(lookup.key)
                                           .cached_program.template get<typename Adapter::cached_mesh_workload_t>()
                                           .shared_variables;
        const auto& parked = shared_variables.begin()->second.global_semaphores;
        for (const auto& [range, sv] : shared_variables) {
            EXPECT_EQ(sv.global_semaphores, parked) << "range " << range;
        }
        EXPECT_NE(parked, nullptr);
        EXPECT_EQ(parked->size(), 1u);
        return parked->front();
    }

    // Three calls on one cache entry, alternating the caller's GlobalSemaphore: each call up()s the one it supplies,
    // and the factory's parked semaphore, if any, counts every call.
    template <typename Factory, bool kParks>
    void run_rebinding() {
        const GlobalSemaphore first = make_semaphore();
        const GlobalSemaphore second = make_semaphore();
        launch<Factory>(Attrs{.call = 0, .global_semaphores = {first}});
        EXPECT_EQ(values(first), all(1));
        launch<Factory>(Attrs{.call = 1, .global_semaphores = {second}});
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u);
        // Still bound to the miss's semaphore, the hit would have made this 2.
        EXPECT_EQ(values(first), all(1));
        EXPECT_EQ(values(second), all(1));
        launch<Factory>(Attrs{.call = 2, .global_semaphores = {first}});
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u);
        EXPECT_EQ(values(first), all(2));
        EXPECT_EQ(values(second), all(1));
        if constexpr (kParks) {
            EXPECT_EQ(values(parked_semaphore<Factory>(Attrs{.global_semaphores = {first}})), all(3));
        }
    }

    // A binding to a GlobalSemaphore that is neither the caller's (in global_semaphores) nor parked is rejected on the
    // cache miss, before any program runs.
    template <typename Factory>
    void run_stray() {
        const GlobalSemaphore caller = make_semaphore();
        const GlobalSemaphore stray = make_semaphore();
        EXPECT_THAT(
            [&] { launch<Factory>(Attrs{.global_semaphores = {caller}, .stray = stray}); },
            ThrowsMessage<std::runtime_error>(
                HasSubstr("must reference a GlobalSemaphore in operation_attributes_t::global_semaphores")));
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 0u);
        EXPECT_EQ(values(stray), all(0));
    }

    // A factory whose override_runtime_arguments supplies its run args on every hit (as generic_op does) may bind a
    // GlobalSemaphore the adapter cannot rebind: keeping it valid is the factory's business, as in legacy factories.
    template <typename Factory>
    void run_override_owned() {
        const GlobalSemaphore caller = make_semaphore();
        const GlobalSemaphore stray = make_semaphore();
        launch<Factory>(Attrs{.global_semaphores = {caller}, .stray = stray});
        launch<Factory>(Attrs{.call = 1, .global_semaphores = {caller}, .stray = stray});
        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u);
        EXPECT_EQ(values(stray), all(2));
        EXPECT_EQ(values(caller), all(0));
    }
};

TEST_F(SpecGlobalSemaphoresTest, SpmdHitRebindsCallerSemaphore) { run_rebinding<SpmdFactory, false>(); }
TEST_F(SpecGlobalSemaphoresTest, CustomSpmdHitRebindsCallerSemaphore) { run_rebinding<CustomSpmdFactory, false>(); }
TEST_F(SpecGlobalSemaphoresTest, MeshHitRebindsCallerAndKeepsParkedSemaphore) { run_rebinding<MeshFactory, true>(); }
TEST_F(SpecGlobalSemaphoresTest, MeshOverrideHitRebindsCallerAndKeepsParkedSemaphore) {
    run_rebinding<MeshOverrideFactory, true>();
}
TEST_F(SpecGlobalSemaphoresTest, FixedPerDeviceHitRebindsCallerAndKeepsParkedSemaphore) {
    run_rebinding<MeshFixedPerDeviceFactory, true>();
}
TEST_F(SpecGlobalSemaphoresTest, PerDeviceOverrideHitRebindsCallerAndKeepsParkedSemaphore) {
    run_rebinding<MeshPerDeviceOverrideFactory, true>();
}

TEST_F(SpecGlobalSemaphoresTest, SpmdMissRejectsStraySemaphore) { run_stray<SpmdFactory>(); }
TEST_F(SpecGlobalSemaphoresTest, MeshMissRejectsStraySemaphore) { run_stray<MeshFactory>(); }
TEST_F(SpecGlobalSemaphoresTest, CustomSpmdOverrideMayBindItsOwnSemaphore) { run_override_owned<CustomSpmdFactory>(); }
TEST_F(SpecGlobalSemaphoresTest, MeshOverrideMayBindItsOwnSemaphore) { run_override_owned<MeshOverrideFactory>(); }

// A recorded command list replays the parked semaphore's address, so it keeps the semaphore allocated until the list
// is deallocated, even after the program cache that parked it is cleared.
TEST_F(SpecGlobalSemaphoresTest, CommandListKeepsParkedSemaphoreAlive) {
    const GlobalSemaphore caller = make_semaphore();
    const Attrs attrs{.global_semaphores = {caller}};
    const Args args = input();
    ttnn::command_list::CommandListBuilder builder(*mesh_device_);
    {
        auto recording = builder.record_scope();
        (void)ttnn::device_operation::launch<GlobalSemaphoreOp<MeshFactory>>(attrs, args);
    }
    const tt::tt_metal::DeviceAddr parked_address = parked_semaphore<MeshFactory>(attrs).address();
    ttnn::command_list::CommandList command_list = builder.build(ttnn::QueueId{0});
    builder.deallocate();
    mesh_device_->clear_program_cache();

    command_list.enqueue(/*blocking=*/true);
    EXPECT_EQ(values(caller), all(1));
    EXPECT_EQ(values_at(parked_address), all(1));

    const auto l1_allocated = [&] {
        return mesh_device_->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    };
    const size_t l1_held = l1_allocated();
    command_list.deallocate();
    EXPECT_LT(l1_allocated(), l1_held) << "The list should hold the parked semaphore until it is deallocated";
}

// The caller's GlobalSemaphores are bound by position, so a hit must supply as many as the miss did.
TEST_F(SpecGlobalSemaphoresTest, HitWithAnotherNumberOfCallerSemaphoresThrows) {
    const GlobalSemaphore first = make_semaphore();
    const GlobalSemaphore second = make_semaphore();
    launch<MeshFactory>(Attrs{.global_semaphores = {first}});
    EXPECT_THAT(
        [&] { launch<MeshFactory>(Attrs{.call = 1, .global_semaphores = {second, first}}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("the program hash must cover how many there are")));
    EXPECT_EQ(values(first), all(1));
    EXPECT_EQ(values(second), all(0));
}

}  // namespace spec_global_semaphores_test
