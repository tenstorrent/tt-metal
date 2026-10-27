// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/worker_cores.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

// Factories that take worker cores split their work over a CoreRangeSet where they used to split it over the grid's
// size. Outside a scope that set is the whole grid, and their programs must not change: the same cores in each group,
// visited in the same order.
TEST(WorkerCores, SplitWorkOverTheWholeGridMatchesSplitWorkOverItsCoreRangeSet) {
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

// A device operation whose program factories exist to exercise launch()'s handling of worker cores: one does not take
// them, one puts a kernel on each, and one strays onto core (0, 0) as well.
struct WorkerCoresProbe {
    enum class Factory : uint32_t { WithoutWorkerCores, OnWorkerCores, OffWorkerCores };

    struct operation_attributes_t {
        Factory factory = Factory::OnWorkerCores;

        static constexpr auto attribute_names = std::forward_as_tuple("factory");
        auto attribute_values() const { return std::forward_as_tuple(factory); }
    };
    struct tensor_args_t {
        const Tensor& input;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // The worker cores the last program built was given, and how many programs were built.
    static inline std::optional<CoreRangeSet> last_worker_cores;
    static inline int programs_built = 0;

    static tt::tt_metal::Program program_on(const CoreRangeSet& cores) {
        ++programs_built;
        tt::tt_metal::Program program;
        tt::tt_metal::CreateKernelFromString(
            program, "void kernel_main() {}", cores, tt::tt_metal::DataMovementConfig{});
        return program;
    }

    struct WithoutWorkerCores {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            last_worker_cores.reset();
            return cached_program_t{program_on(CoreRangeSet(CoreRange({0, 0}, {0, 0}))), {}};
        }
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };

    struct OnWorkerCores {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t&,
            const CoreRangeSet& worker_cores) {
            last_worker_cores = worker_cores;
            return cached_program_t{program_on(worker_cores), {}};
        }
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };

    struct OffWorkerCores {
        struct shared_variables_t {};
        using cached_program_t = device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t&,
            const CoreRangeSet& worker_cores) {
            last_worker_cores = worker_cores;
            return cached_program_t{program_on(worker_cores.merge(CoreRangeSet(CoreRange({0, 0}, {0, 0})))), {}};
        }
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };

    using program_factory_t = std::variant<WithoutWorkerCores, OnWorkerCores, OffWorkerCores>;

    static program_factory_t select_program_factory(const operation_attributes_t& attributes, const tensor_args_t&) {
        switch (attributes.factory) {
            case Factory::WithoutWorkerCores: return WithoutWorkerCores{};
            case Factory::OnWorkerCores: return OnWorkerCores{};
            case Factory::OffWorkerCores: return OffWorkerCores{};
        }
        TT_THROW("Unknown factory");
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
    }
};

static_assert(device_operation::DeviceOperationConcept<WorkerCoresProbe>);

class WorkerCoresFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        TTNNFixtureWithDevice::SetUp();
        input_ = create_device_tensor(
            TensorSpec(
                Shape({1, 1, 32, 32}),
                tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            device_);
        WorkerCoresProbe::last_worker_cores.reset();
        WorkerCoresProbe::programs_built = 0;
    }

    void TearDown() override {
        input_ = Tensor();
        TTNNFixtureWithDevice::TearDown();
    }

    Tensor launch(WorkerCoresProbe::Factory factory) {
        return device_operation::launch<WorkerCoresProbe>({.factory = factory}, {.input = input_});
    }

    CoreRangeSet compute_grid() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    }

    // The last column of the compute grid: a rectangle away from (0, 0).
    CoreRangeSet last_column() const {
        const auto grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({grid.x - 1, 0}, {grid.x - 1, grid.y - 1}));
    }

    Tensor input_;
};

TEST_F(WorkerCoresFixture, OutsideAScopeAFactoryIsGivenTheWholeComputeGrid) {
    EXPECT_FALSE(get_scoped_worker_cores(*device_).has_value());
    EXPECT_EQ(get_worker_cores(*device_), compute_grid());

    launch(WorkerCoresProbe::Factory::OnWorkerCores);
    EXPECT_EQ(WorkerCoresProbe::last_worker_cores, compute_grid());
}

TEST_F(WorkerCoresFixture, InsideAScopeAFactoryIsGivenTheScopesCores) {
    const auto guard = with_worker_cores(last_column());
    EXPECT_EQ(get_scoped_worker_cores(*device_), last_column());
    EXPECT_EQ(get_worker_cores(*device_), last_column());

    launch(WorkerCoresProbe::Factory::OnWorkerCores);
    EXPECT_EQ(WorkerCoresProbe::last_worker_cores, last_column());
}

TEST_F(WorkerCoresFixture, InsideAScopeAFactoryThatDoesNotTakeWorkerCoresIsRefused) {
    // Outside a scope it runs where it likes.
    launch(WorkerCoresProbe::Factory::WithoutWorkerCores);

    const auto entries = device_->num_program_cache_entries();
    const auto guard = with_worker_cores(last_column());
    EXPECT_THAT(
        [&] { launch(WorkerCoresProbe::Factory::WithoutWorkerCores); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("does not take worker cores")));
    EXPECT_EQ(device_->num_program_cache_entries(), entries);
}

TEST_F(WorkerCoresFixture, InsideAScopeAProgramThatStraysOffTheScopeIsRefused) {
    // Given the whole grid, (0, 0) is one of its worker cores.
    launch(WorkerCoresProbe::Factory::OffWorkerCores);

    const auto entries = device_->num_program_cache_entries();
    const auto guard = with_worker_cores(last_column());
    EXPECT_THAT(
        [&] { launch(WorkerCoresProbe::Factory::OffWorkerCores); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("outside the worker-cores scope")));
    // A refused program is not cached, so the next launch is refused too rather than hitting it.
    EXPECT_EQ(device_->num_program_cache_entries(), entries);
    EXPECT_ANY_THROW(launch(WorkerCoresProbe::Factory::OffWorkerCores));
}

TEST_F(WorkerCoresFixture, TheScopesCoresArePartOfTheProgramCacheKey) {
    const auto first_column = CoreRangeSet(CoreRange({0, 0}, {0, device_->compute_with_storage_grid_size().y - 1}));
    const auto launch_on = [&](const std::optional<CoreRangeSet>& cores) {
        if (!cores.has_value()) {
            launch(WorkerCoresProbe::Factory::OnWorkerCores);
            return;
        }
        const auto guard = with_worker_cores(*cores);
        launch(WorkerCoresProbe::Factory::OnWorkerCores);
    };

    const auto entries = device_->num_program_cache_entries();
    for (int pass = 0; pass < 2; ++pass) {
        launch_on(std::nullopt);
        launch_on(last_column());
        launch_on(first_column);
    }
    EXPECT_EQ(device_->num_program_cache_entries(), entries + 3);
    // One program per placement; the second pass hit the cache each time.
    EXPECT_EQ(WorkerCoresProbe::programs_built, 3);
}

TEST_F(WorkerCoresFixture, ScopesNestAndTheInnermostApplies) {
    const auto first_column = CoreRangeSet(CoreRange({0, 0}, {0, device_->compute_with_storage_grid_size().y - 1}));
    {
        const auto outer = with_worker_cores(first_column);
        {
            const auto inner = with_worker_cores(last_column());
            launch(WorkerCoresProbe::Factory::OnWorkerCores);
            EXPECT_EQ(WorkerCoresProbe::last_worker_cores, last_column());
        }
        launch(WorkerCoresProbe::Factory::OnWorkerCores);
        EXPECT_EQ(WorkerCoresProbe::last_worker_cores, first_column);
    }
    EXPECT_FALSE(get_scoped_worker_cores(*device_).has_value());
}

TEST_F(WorkerCoresFixture, ASubDeviceScopeStandsForThatSubDevicesTensixCores) {
    if (!check_dispatch_mode()) {
        GTEST_SKIP() << "Sub-device managers need fast dispatch";
    }
    const CoreRangeSet rest = compute_grid().subtract(last_column());
    const auto manager = device_->create_sub_device_manager(
        {tt::tt_metal::SubDevice(std::array{rest}), tt::tt_metal::SubDevice(std::array{last_column()})}, 0);
    device_->load_sub_device_manager(manager);
    {
        const auto guard = with_sub_device(tt::tt_metal::SubDeviceId{1});
        EXPECT_EQ(get_worker_cores(*device_), last_column());
        launch(WorkerCoresProbe::Factory::OnWorkerCores);
        EXPECT_EQ(WorkerCoresProbe::last_worker_cores, last_column());
    }
    {
        const auto guard = with_sub_device(tt::tt_metal::SubDeviceId{0});
        launch(WorkerCoresProbe::Factory::OnWorkerCores);
        EXPECT_EQ(WorkerCoresProbe::last_worker_cores, rest);
    }
    tt::tt_metal::distributed::Synchronize(device_, std::nullopt);
    device_->clear_loaded_sub_device_manager();
    device_->remove_sub_device_manager(manager);
}

TEST_F(WorkerCoresFixture, AScopeReachingOutsideTheComputeGridIsRefused) {
    const auto grid = device_->compute_with_storage_grid_size();
    const auto guard = with_worker_cores(CoreRangeSet(CoreRange({0, 0}, {grid.x, grid.y - 1})));
    EXPECT_THAT(
        [&] { launch(WorkerCoresProbe::Factory::OnWorkerCores); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("outside the compute grid")));
}

}  // namespace
}  // namespace ttnn
