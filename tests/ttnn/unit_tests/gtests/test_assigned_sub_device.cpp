// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/assigned_sub_device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/sub_device_scope.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace {

using testing::AllOf;
using testing::HasSubstr;
using testing::Not;
using testing::ThrowsMessage;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::SubDeviceId;
using tt::tt_metal::TensorMemoryLayout;

// Factories that place kernels on their sub-device's cores split their work over a CoreRangeSet where they used to
// split it over the grid's size. With no sub-device manager loaded that set is the whole grid, and their programs must
// not change: the same cores in each group, visited in the same order.
TEST(AssignedSubDevice, SplitWorkOverTheWholeGridMatchesSplitWorkOverItsCoreRangeSet) {
    for (const CoreCoord grid :
         {CoreCoord{1, 1}, CoreCoord{7, 3}, CoreCoord{8, 8}, CoreCoord{11, 10}, CoreCoord{13, 10}}) {
        const CoreRangeSet grid_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        for (const bool row_wise : {false, true}) {
            for (uint32_t units = 0; units <= (2 * grid.x * grid.y) + 3; ++units) {
                const auto [num_cores, all_cores, group_1, group_2, units_1, units_2] =
                    tt::tt_metal::split_work_to_cores(grid, units, row_wise);
                const auto [set_num_cores, set_all_cores, set_group_1, set_group_2, set_units_1, set_units_2] =
                    tt::tt_metal::split_work_to_cores(grid_cores, units, row_wise);
                const auto in_order = [row_wise](const CoreRangeSet& cores) {
                    return corerange_to_cores(cores, std::nullopt, row_wise);
                };
                SCOPED_TRACE(fmt::format("grid {}, {} units, row_wise {}", grid, units, row_wise));
                EXPECT_EQ(set_num_cores, num_cores);
                EXPECT_EQ(in_order(set_all_cores), in_order(all_cores));
                EXPECT_EQ(in_order(set_group_1), in_order(group_1));
                EXPECT_EQ(in_order(set_group_2), in_order(group_2));
                EXPECT_EQ(set_units_1, units_1);
                EXPECT_EQ(set_units_2, units_2);
            }
        }
    }
}

TEST(AssignedSubDevice, RectangleIn) {
    const CoreRange whole({0, 0}, {10, 9});
    EXPECT_EQ(rectangle_in(CoreRangeSet(whole)), whole);
    // Llama's Wormhole worker columns, around the prefetcher's senders.
    EXPECT_EQ(
        rectangle_in(CoreRangeSet(std::vector<CoreRange>{CoreRange({1, 0}, {3, 9}), CoreRange({5, 0}, {6, 9})})),
        CoreRange({1, 0}, {3, 9}));
    // An L whose two arms are equal: the first, in row-major order.
    EXPECT_EQ(
        rectangle_in(CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {3, 0}), CoreRange({0, 1}, {0, 3})})),
        CoreRange({0, 0}, {3, 0}));
    EXPECT_EQ(
        rectangle_in(CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {1, 1}), CoreRange({2, 0}, {4, 0})})),
        CoreRange({0, 0}, {4, 0}));
    EXPECT_EQ(
        rectangle_in(CoreRangeSet(std::vector<CoreRange>{CoreRange({4, 4}, {4, 4}), CoreRange({6, 2}, {7, 3})})),
        CoreRange({6, 2}, {7, 3}));
}

tt::tt_metal::Program program_on(const CoreRangeSet& cores) {
    tt::tt_metal::Program program;
    tt::tt_metal::CreateKernelFromString(program, "void kernel_main() {}", cores, tt::tt_metal::DataMovementConfig{});
    return program;
}

struct ProbeTensorArgs {
    const Tensor& input;
};

// The parts of a device operation the probes below share.
template <typename Attributes>
struct ProbeBase {
    using operation_attributes_t = Attributes;
    using tensor_args_t = ProbeTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static inline int outputs_created = 0;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        ++outputs_created;
        return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
    }
};

// Records its sub-device, puts a kernel on each of its cores, and, when asked to stray, on core (0, 0) as well; asked
// for a stray object, it puts that object on its cores and on core (0, 0). Its output is its input when in place.
enum class StrayObject : std::uint8_t { none, semaphore, circular_buffer };

struct RecordingProbeAttributes {
    bool stray = false;
    StrayObject stray_object = StrayObject::none;
    bool in_place = false;
    AssignedSubDevice assigned_sub_device;
};

struct RecordingProbe : ProbeBase<RecordingProbeAttributes> {
    static inline int programs_built = 0;

    static const AssignedSubDevice& assigned_sub_device(const operation_attributes_t& attributes) {
        return attributes.assigned_sub_device;
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return attributes.in_place ? tensor_args.input : ProbeBase::create_output_tensors(attributes, tensor_args);
    }

    struct Factory {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(const operation_attributes_t& attributes, const tensor_args_t&, Tensor&) {
            ++programs_built;
            const CoreRangeSet& cores = attributes.assigned_sub_device.worker_cores;
            const CoreRangeSet with_origin = cores.merge(CoreRangeSet(CoreRange({0, 0}, {0, 0})));
            tt::tt_metal::Program program = program_on(attributes.stray ? with_origin : cores);
            if (attributes.stray_object == StrayObject::semaphore) {
                tt::tt_metal::CreateSemaphore(program, with_origin, 0);
            } else if (attributes.stray_object == StrayObject::circular_buffer) {
                tt::tt_metal::CreateCircularBuffer(
                    program,
                    with_origin,
                    tt::tt_metal::CircularBufferConfig(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                        .set_page_size(tt::CBIndex::c_0, 2048));
            }
            return cached_program_t{std::move(program), {}};
        }
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, Tensor&) {}
    };
    using program_factory_t = std::variant<Factory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return Factory{};
    }
};

// Records no sub-device: it puts its kernel on core (0, 0) whatever it is asked.
struct SilentProbeAttributes {
    bool unused = false;
};

struct SilentProbe : ProbeBase<SilentProbeAttributes> {
    struct Factory {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, Tensor&) {
            return cached_program_t{program_on(CoreRangeSet(CoreRange({0, 0}, {0, 0}))), {}};
        }
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, Tensor&) {}
    };
    using program_factory_t = std::variant<Factory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return Factory{};
    }
};

// RecordingProbe whose own program hash leaves the record out, as an operation's compute_program_hash may.
struct UnhashedRecordProbe : RecordingProbe {
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&) { return 0; }
};

// SilentProbe under other names, so each one's once-per-operation log is its own.
struct WanderingProbe : SilentProbe {};
struct StrayingProbe : SilentProbe {};

static_assert(device_operation::DeviceOperationConcept<RecordingProbe>);
static_assert(device_operation::RecordsSubDevice<RecordingProbe>);
static_assert(device_operation::DeviceOperationWithCustomProgramCacheConcept<UnhashedRecordProbe>);
static_assert(device_operation::DeviceOperationConcept<SilentProbe>);
static_assert(!device_operation::RecordsSubDevice<SilentProbe>);

class AssignedSubDeviceFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Sub-device managers need fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
        input_ = tensor_in(MemoryConfig{});
        RecordingProbe::programs_built = 0;
        RecordingProbe::outputs_created = 0;
        SilentProbe::outputs_created = 0;
    }

    void TearDown() override {
        input_ = Tensor();
        if (!managers_.empty()) {
            tt::tt_metal::distributed::Synchronize(*device_, std::nullopt);
            device_->clear_loaded_sub_device_manager();
            for (const auto manager : managers_) {
                device_->remove_sub_device_manager(manager);
            }
        }
        TTNNFixtureWithDevice::TearDown();
    }

    Tensor tensor_in(const MemoryConfig& memory_config) const {
        return create_device_tensor(
            TensorSpec(
                Shape({1, 1, 32, 32}),
                tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), memory_config)),
            device_);
    }

    CoreRangeSet compute_grid() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    }
    CoreRangeSet all_but_last_column() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {grid.x - 2, grid.y - 1}));
    }
    CoreRangeSet last_column() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({grid.x - 1, 0}, {grid.x - 1, grid.y - 1}));
    }
    CoreRangeSet first_row() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, 0}));
    }

    tt::tt_metal::SubDeviceManagerId create_manager(const std::vector<CoreRangeSet>& sub_devices) {
        std::vector<tt::tt_metal::SubDevice> subs;
        for (const auto& cores : sub_devices) {
            subs.push_back(tt::tt_metal::SubDevice(std::array{cores}));
        }
        managers_.push_back(device_->create_sub_device_manager(subs, 0));
        return managers_.back();
    }

    // Sub-device 0: every column but the last; sub-device 1: the last column.
    void load_two_sub_devices() {
        device_->load_sub_device_manager(create_manager({all_but_last_column(), last_column()}));
    }

    Tensor launch_recording(const AssignedSubDevice& sub_device, bool stray = false, const Tensor* input = nullptr) {
        return device_operation::launch<RecordingProbe>(
            {.stray = stray, .assigned_sub_device = sub_device}, {.input = input != nullptr ? *input : input_});
    }

    Tensor input_;
    std::vector<tt::tt_metal::SubDeviceManagerId> managers_;
};

TEST_F(AssignedSubDeviceFixture, WithNoSubDeviceManagerAnOperationRunsOnTheWholeComputeGrid) {
    const auto sub_device = assign_sub_device(device_, {}, "probe");
    EXPECT_EQ(*sub_device.id, 0);
    EXPECT_EQ(sub_device.worker_cores, compute_grid());
    EXPECT_EQ(assign_sub_device(device_, {.data = last_column()}, "probe").worker_cores, compute_grid());
    EXPECT_EQ(assign_sub_device(device_, {.cores = last_column()}, "probe").worker_cores, compute_grid());
}

// Outside a scope: the caller's sub-device, else the one holding the caller's cores, else the one holding the data,
// else the only one.
TEST_F(AssignedSubDeviceFixture, ResolutionOrder) {
    load_two_sub_devices();
    EXPECT_EQ(
        *assign_sub_device(device_, {.sub_device = SubDeviceId{1}, .data = all_but_last_column()}, "probe").id, 1);
    EXPECT_EQ(*assign_sub_device(device_, {.sub_device = SubDeviceId{1}, .cores = last_column()}, "probe").id, 1);
    EXPECT_EQ(*assign_sub_device(device_, {.cores = last_column(), .data = all_but_last_column()}, "probe").id, 1);
    EXPECT_EQ(*assign_sub_device(device_, {.data = last_column()}, "probe").id, 1);
    EXPECT_EQ(assign_sub_device(device_, {.data = all_but_last_column()}, "probe").worker_cores, all_but_last_column());
    {
        const auto scope = with_sub_device(SubDeviceId{0});
        EXPECT_EQ(*assign_sub_device(device_, {.data = last_column()}, "probe").id, 0);
    }
    device_->load_sub_device_manager(create_manager({compute_grid()}));
    EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, compute_grid());
}

// Cores and data name the sub-device that holds all of them, not the first one they touch.
TEST_F(AssignedSubDeviceFixture, ContainmentNotIntersection) {
    load_two_sub_devices();
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.data = compute_grid()}, "probe"); },
        ThrowsMessage<std::runtime_error>(
            AllOf(HasSubstr("its L1 shards"), HasSubstr("span sub-devices 0 and 1, so they name none of them"))));
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.cores = first_row()}, "probe"); },
        ThrowsMessage<std::runtime_error>(AllOf(HasSubstr("its cores"), HasSubstr("span sub-devices 0 and 1;"))));
    // Cores partly in the only sub-device and partly in none.
    device_->load_sub_device_manager(create_manager({all_but_last_column()}));
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.cores = first_row()}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("do not all lie in one sub-device")));
}

TEST_F(AssignedSubDeviceFixture, AmbiguityIsRefusedOnASplitDevice) {
    load_two_sub_devices();
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {}, "probe"); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("probe does not know which of the 2 loaded sub-devices to run on")));
}

TEST_F(AssignedSubDeviceFixture, CallerCoresOutsideTheResultAreRefused) {
    load_two_sub_devices();
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.sub_device = SubDeviceId{0}, .cores = last_column()}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("outside sub-device 0")));
    const auto scope = with_sub_device(SubDeviceId{0});
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.cores = last_column()}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("outside sub-device 0")));
}

TEST_F(AssignedSubDeviceFixture, AnOutOfRangeIdIsRefused) {
    load_two_sub_devices();
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.sub_device = SubDeviceId{5}}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("the loaded sub-device manager has 2")));
    const auto scope = with_sub_device(SubDeviceId{5});
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("the loaded sub-device manager has 2")));
}

TEST_F(AssignedSubDeviceFixture, InsideAScopeAnOperationRunsOnTheScopesSubDevice) {
    load_two_sub_devices();
    {
        const auto outer = with_sub_device(SubDeviceId{0});
        EXPECT_EQ(assign_sub_device(device_, {.data = last_column()}, "probe").worker_cores, all_but_last_column());
        {
            const auto inner = with_sub_device(SubDeviceId{1});
            EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, last_column());
            EXPECT_EQ(*assign_sub_device(device_, {.sub_device = SubDeviceId{1}}, "probe").id, 1);
        }
        EXPECT_EQ(*assign_sub_device(device_, {}, "probe").id, 0);
    }
    EXPECT_FALSE(in_sub_device_scope());
}

TEST_F(AssignedSubDeviceFixture, ARequestForAnotherSubDeviceInsideAScopeIsRefused) {
    load_two_sub_devices();
    const auto guard = with_sub_device(SubDeviceId{1});
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {.sub_device = SubDeviceId{0}}, "probe"); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("probe was asked to run on sub-device 0 inside a sub-device scope on sub-device 1")));
}

TEST_F(AssignedSubDeviceFixture, ScopesExitedOutOfOrderAreRefused) {
    const auto outer = push_sub_device_for_thread(SubDeviceId{0});
    const auto inner = push_sub_device_for_thread(SubDeviceId{0});
    EXPECT_THAT(
        [&] { pop_sub_device_for_thread(outer); }, ThrowsMessage<std::runtime_error>(HasSubstr("out of order")));
    pop_sub_device_for_thread(inner);
    EXPECT_FALSE(in_sub_device_scope());
}

TEST_F(AssignedSubDeviceFixture, AManagerSwapInsideAScopeIsRefused) {
    load_two_sub_devices();
    const auto other = create_manager({last_column(), all_but_last_column()});
    const auto scope = with_sub_device(SubDeviceId{1});
    EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, last_column());
    tt::tt_metal::distributed::Synchronize(*device_, std::nullopt);
    device_->load_sub_device_manager(other);
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("earlier operations ran under another sub-device manager")));
}

// The scope binds to a manager when it first assigns an operation, so one loaded before that is the one it runs under.
TEST_F(AssignedSubDeviceFixture, AManagerLoadedBeforeAScopesFirstOperationIsTheOneItRunsUnder) {
    load_two_sub_devices();
    const auto other = create_manager({last_column(), all_but_last_column()});
    const auto scope = with_sub_device(SubDeviceId{1});
    device_->load_sub_device_manager(other);
    EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, all_but_last_column());
}

// A layer that loads a manager for an overlap window, runs scopes of its own inside, and clears it again: the enclosing
// scope is refused only within the window, where its id names other cores.
TEST_F(AssignedSubDeviceFixture, AScopeAroundALoadAndClearWindowRunsAgainOnceTheManagerIsCleared) {
    const auto outer = with_sub_device(SubDeviceId{0});
    EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, compute_grid());
    load_two_sub_devices();
    {
        const auto inner = with_sub_device(SubDeviceId{1});
        EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, last_column());
    }
    EXPECT_THAT(
        [&] { assign_sub_device(device_, {}, "probe"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("earlier operations ran under another sub-device manager")));
    device_->clear_loaded_sub_device_manager();
    EXPECT_EQ(assign_sub_device(device_, {}, "probe").worker_cores, compute_grid());
}

TEST_F(AssignedSubDeviceFixture, AnOperationRecordingNoSubDeviceIsRefusedInsideAScopeBeforeItAllocates) {
    device_operation::launch<SilentProbe>({}, {.input = input_});  // outside a scope it runs as before
    EXPECT_EQ(SilentProbe::outputs_created, 1);

    const auto guard = with_sub_device(SubDeviceId{0});
    EXPECT_THAT(
        [&] { device_operation::launch<SilentProbe>({}, {.input = input_}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("does not record the sub-device it runs on")));
    EXPECT_EQ(SilentProbe::outputs_created, 1);
}

TEST_F(AssignedSubDeviceFixture, AnOperationRecordingAnotherSubDeviceThanTheScopesIsRefused) {
    load_two_sub_devices();
    const auto first = assign_sub_device(device_, {.sub_device = SubDeviceId{0}}, "probe");
    const auto guard = with_sub_device(SubDeviceId{1});
    EXPECT_THAT(
        [&] { launch_recording(first); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("records sub-device 0, but runs inside a sub-device scope on sub-device 1")));
    EXPECT_EQ(RecordingProbe::programs_built, 0);
    EXPECT_EQ(RecordingProbe::outputs_created, 0);
    launch_recording(assign_sub_device(device_, {}, "probe"));
    EXPECT_EQ(RecordingProbe::programs_built, 1);
}

TEST_F(AssignedSubDeviceFixture, AnUnassignedRecordIsRefusedAtLaunch) {
    EXPECT_THAT(
        [&] { launch_recording({SubDeviceId{0}, {}}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("was launched without the sub-device it runs on")));
}

// A composite operation carries its sub-device to the operations it launches that record one; the others run as they
// would without it, and only a user's scope refuses them.
TEST_F(AssignedSubDeviceFixture, ACompositeCarriesItsSubDeviceToRecordingOpsAndLeavesOthersAsBefore) {
    load_two_sub_devices();
    {
        const CompositeSubDevice composite(device_, {.sub_device = SubDeviceId{1}}, "composite");
        EXPECT_EQ(*composite.assigned().id, 1);
        EXPECT_TRUE(in_sub_device_scope());
        const auto inner = assign_sub_device(device_, {}, "probe");
        EXPECT_EQ(inner.worker_cores, last_column());
        launch_recording(inner);
        device_operation::launch<SilentProbe>({}, {.input = input_});
        EXPECT_THAT(
            [&] { assign_sub_device(device_, {.sub_device = SubDeviceId{0}}, "probe"); },
            ThrowsMessage<std::runtime_error>(HasSubstr("inside a composite operation on sub-device 1")));
    }
    EXPECT_FALSE(in_sub_device_scope());
    // A user's scope decides; the composite carries nothing over it.
    const auto scope = with_sub_device(SubDeviceId{0});
    const CompositeSubDevice composite(device_, {}, "composite");
    EXPECT_EQ(*composite.assigned().id, 0);
}

// Inside a scope, a composite operation's own request is checked when its guard is built, before it launches anything;
// its entry binds the scope to the sub-device manager, as an operation's does.
TEST_F(AssignedSubDeviceFixture, ACompositesRequestIsCheckedAgainstTheScopeAtItsEntry) {
    load_two_sub_devices();
    const auto other = create_manager({last_column(), all_but_last_column()});
    const auto scope = with_sub_device(SubDeviceId{1});
    EXPECT_THAT(
        [&] { const CompositeSubDevice composite(device_, {.sub_device = SubDeviceId{0}}, "composite"); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("composite was asked to run on sub-device 0 inside a sub-device scope on sub-device 1")));
    EXPECT_THAT(
        [&] { const CompositeSubDevice composite(device_, {.cores = all_but_last_column()}, "composite"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("composite was given cores")));
    {
        const CompositeSubDevice composite(device_, {}, "composite");
        EXPECT_EQ(composite.assigned().worker_cores, last_column());
    }
    device_->load_sub_device_manager(other);
    EXPECT_THAT(
        [&] { const CompositeSubDevice composite(device_, {}, "composite"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("earlier operations ran under another sub-device manager")));
}

// Inside a scope, an operation that records no sub-device is refused where a composite operation launches it. The
// refusal names the outermost composite, the call its caller made, instead of telling them to launch an operation they
// never launched; the name leaves with the composite, on unwinding too.
TEST_F(AssignedSubDeviceFixture, AnUnrecordedOpRefusedWithinACompositeNamesTheComposite) {
    load_two_sub_devices();
    const auto scope = with_sub_device(SubDeviceId{1});
    {
        const CompositeSubDevice outer(device_, {}, "outer");
        const CompositeSubDevice inner(device_, {}, "inner");
        EXPECT_THAT(
            [&] { device_operation::launch<SilentProbe>({}, {.input = input_}); },
            ThrowsMessage<std::runtime_error>(AllOf(
                HasSubstr(", launched within outer, cannot run inside a sub-device scope (sub-device 1)"),
                HasSubstr("outer cannot run inside a sub-device scope on this path until the operations it launches "
                          "record their sub-device."),
                Not(HasSubstr("Launch it outside the scope")))));
    }
    EXPECT_ANY_THROW({
        const CompositeSubDevice unwound(device_, {}, "unwound");
        device_operation::launch<SilentProbe>({}, {.input = input_});
    });
    EXPECT_THAT(
        [&] { device_operation::launch<SilentProbe>({}, {.input = input_}); },
        ThrowsMessage<std::runtime_error>(AllOf(
            HasSubstr("cannot run inside a sub-device scope (sub-device 1)"),
            HasSubstr("Launch it outside the scope; if an operation you called launched it, that operation cannot run "
                      "inside a scope until"),
            Not(HasSubstr("launched within")))));
    EXPECT_EQ(SilentProbe::outputs_created, 0);
}

// An op that records no sub-device runs where its own code puts it, inside a composite too; when that is off the
// composite's sub-device, it is logged once, naming the call that carried the sub-device.
TEST_F(AssignedSubDeviceFixture, AnUnrecordedOpLeavingACompositesSubDeviceIsLoggedOnce) {
    load_two_sub_devices();
    const auto launch_logged = [&] {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        device_operation::launch<WanderingProbe>({}, {.input = input_});
        const std::string logged = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
        return logged.find("records no sub-device: its kernels landed on") != std::string::npos;
    };
    {
        // Its kernel on (0, 0) lies in sub-device 0: nothing to report.
        const CompositeSubDevice composite(device_, {.sub_device = SubDeviceId{0}}, "composite");
        EXPECT_FALSE(launch_logged());
    }
    const CompositeSubDevice composite(device_, {.sub_device = SubDeviceId{1}}, "composite");
    EXPECT_TRUE(launch_logged());
    EXPECT_FALSE(launch_logged());
    // A composite nested in the carry leaves the carrier's name; the warning asks for the fix.
    const CompositeSubDevice inner(device_, {}, "inner");
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    device_operation::launch<StrayingProbe>({}, {.input = input_});
    const std::string logged = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
    EXPECT_THAT(logged, HasSubstr("ran inside composite on sub-device 1, but records no sub-device"));
    EXPECT_THAT(logged, HasSubstr("record its sub-device. (Logged once per operation.)"));
    EXPECT_THAT(logged, Not(HasSubstr("to have it refused")));
}

// Interleaved L1 spans every sub-device's cores, where the others keep their circular buffers.
TEST_F(AssignedSubDeviceFixture, NewL1OutputsOffTheSubDeviceAreRefusedInAScopeAndLoggedOutside) {
    const Tensor l1_input = tensor_in(MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1));
    {
        const auto scope = with_sub_device(SubDeviceId{0});
        launch_recording(assign_sub_device(device_, {}, "probe"), false, &l1_input);  // one sub-device
    }
    load_two_sub_devices();
    // Outside a scope it runs, and is logged once per operation.
    for (int launch = 0; launch < 2; ++launch) {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        launch_recording(assign_sub_device(device_, {.sub_device = SubDeviceId{1}}, "probe"), false, &l1_input);
        const std::string logged = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
        EXPECT_EQ(
            logged.find("allocated an output in L1 (interleaved) on sub-device 1") != std::string::npos, launch == 0)
            << logged;
    }
    const auto scope = with_sub_device(SubDeviceId{1});
    EXPECT_THAT(
        [&] { launch_recording(assign_sub_device(device_, {}, "probe"), false, &l1_input); },
        ThrowsMessage<std::runtime_error>(HasSubstr("allocates an output in L1 (interleaved) inside a scope")));
    launch_recording(assign_sub_device(device_, {}, "probe"));  // DRAM
    device_operation::launch<RecordingProbe>(
        {.in_place = true, .assigned_sub_device = assign_sub_device(device_, {}, "probe")}, {.input = l1_input});
}

// Checked whenever an operation records its sub-device, scope or not. A refused program is not cached.
TEST_F(AssignedSubDeviceFixture, AProgramOffItsSubDeviceIsRefused) {
    device_->enable_program_cache();
    load_two_sub_devices();
    const auto last = assign_sub_device(device_, {.sub_device = SubDeviceId{1}}, "probe");
    const auto entries = device_->num_program_cache_entries();
    for (int attempt = 0; attempt < 2; ++attempt) {
        EXPECT_THAT(
            [&] { launch_recording(last, /*stray=*/true); },
            ThrowsMessage<std::runtime_error>(HasSubstr("outside sub-device 1")));
    }
    EXPECT_EQ(RecordingProbe::programs_built, 2);
    EXPECT_EQ(device_->num_program_cache_entries(), entries);
    launch_recording(last);
}

// Launch holds an operation's kernels to its record, and Metal keeps the program's semaphores and circular buffers off
// other sub-devices. Neither refusal is cached.
TEST_F(AssignedSubDeviceFixture, AnObjectOnAnotherSubDeviceIsRefusedByMetal) {
    device_->enable_program_cache();
    load_two_sub_devices();
    const auto last = assign_sub_device(device_, {.sub_device = SubDeviceId{1}}, "probe");
    for (const auto object : {StrayObject::semaphore, StrayObject::circular_buffer}) {
        const int built = RecordingProbe::programs_built;
        const auto entries = device_->num_program_cache_entries();
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_THAT(
                [&] {
                    device_operation::launch<RecordingProbe>(
                        {.stray_object = object, .assigned_sub_device = last}, {.input = input_});
                },
                ThrowsMessage<std::runtime_error>(
                    HasSubstr("in sub-device 0, outside sub-device 1 where its kernels run")));
        }
        EXPECT_EQ(RecordingProbe::programs_built, built + 2);
        EXPECT_EQ(device_->num_program_cache_entries(), entries);
    }
    launch_recording(last);
}

// The record is folded into the program-cache key whatever an operation's own hash covers, so a program built for one
// sub-device's cores is never reused on another's.
TEST_F(AssignedSubDeviceFixture, TheRecordKeysProgramsWhenTheOperationsOwnHashLeavesItOut) {
    device_->enable_program_cache();
    for (int round = 0; round < 2; ++round) {
        device_operation::launch<UnhashedRecordProbe>(
            {.assigned_sub_device = {SubDeviceId{0}, compute_grid()}}, {.input = input_});
        device_operation::launch<UnhashedRecordProbe>(
            {.assigned_sub_device = {SubDeviceId{0}, last_column()}}, {.input = input_});
    }
    EXPECT_EQ(RecordingProbe::programs_built, 2);
}

// The same sub-device id names other cores under another sub-device manager, so the key holds the cores.
TEST_F(AssignedSubDeviceFixture, TheSubDevicesCoresArePartOfTheProgramCacheKey) {
    device_->enable_program_cache();
    for (int round = 0; round < 2; ++round) {
        launch_recording({SubDeviceId{0}, compute_grid()});
        launch_recording({SubDeviceId{0}, last_column()});
    }
    EXPECT_EQ(RecordingProbe::programs_built, 2);
}

}  // namespace
}  // namespace ttnn
