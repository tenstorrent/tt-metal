// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "env_lib.hpp"
#include "gmock/gmock.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/worker_config_buffer.hpp"
#include <tt_stl/span.hpp>
#include "tests/tt_metal/distributed/utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

struct CBConfig {
    uint32_t cb_id = 0;
    uint32_t num_pages = 0;
    uint32_t page_size = 0;
    tt::DataFormat data_format;
};

std::vector<CBHandle> initialize_dummy_circular_buffers(
    Program& program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs) {
    std::vector<CBHandle> cb_handles;
    for (const auto& cb_config : cb_configs) {
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config =
            CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

std::shared_ptr<Program> initialize_dummy_program(CoreCoord worker_grid_size) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreRange cr = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    initialize_dummy_kernels(*program, cr_set);
    initialize_dummy_circular_buffers(*program, cr_set, cb_config_vector);
    return program;
}

void verify_cb_config(
    std::shared_ptr<MeshDevice>& mesh_device,
    MeshWorkload& workload,
    std::vector<CBConfig>& golden_cb_config,
    CoreRangeSet& crs) {
    uint32_t max_cbs = MetalContext::instance().hal().get_arch_num_circular_buffers();
    std::vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = max_cbs * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const auto& [device_range, _] : workload.get_programs()) {
        for (const auto& coord : device_range) {
            if (!mesh_device->impl().is_local(coord)) {
                continue;
            }
            auto* device = mesh_device->impl().get_device(coord);
            uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
            for (const auto& core_range : crs.ranges()) {
                for (const auto& core_coord : core_range) {
                    ::tt::tt_metal::detail::ReadFromDeviceL1(
                        device,
                        core_coord,
                        workload.get_cb_base_addr(mesh_device, core_coord, CoreType::WORKER),
                        cb_config_buffer_size,
                        cb_config_vector);

                    uint32_t cb_addr = l1_unreserved_base;
                    for (const auto& config : golden_cb_config) {
                        const uint32_t index = config.cb_id * sizeof(uint32_t);
                        const uint32_t cb_num_pages = config.num_pages;
                        const uint32_t cb_size = cb_num_pages * config.page_size;
                        const bool addr_match = cb_config_vector.at(index) == cb_addr;
                        const bool size_match = cb_config_vector.at(index + 1) == cb_size;
                        const bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                        EXPECT_TRUE(addr_match);
                        EXPECT_TRUE(size_match);
                        EXPECT_TRUE(num_pages_match);
                        cb_addr += cb_size;
                    }
                }
            }
        }
    }
}

void validate_sems(
    std::shared_ptr<MeshDevice>& mesh_device,
    IDevice* device,
    CoreRange& crs,
    MeshWorkload& mesh_workload,
    std::vector<uint32_t>& expected_semaphore_values) {
    for (const auto& core : crs) {
        const uint32_t sem_buffer_size = mesh_workload.get_sem_size(mesh_device, core, CoreType::WORKER);
        const uint32_t sem_buffer_base = mesh_workload.get_sem_base_addr(mesh_device, core, CoreType::WORKER);
        std::vector<uint32_t> readback_sem_vals;
        ::tt::tt_metal::detail::ReadFromDeviceL1(device, core, sem_buffer_base, sem_buffer_size, readback_sem_vals);
        uint32_t sem_idx = 0;
        for (uint32_t i = 0; i < readback_sem_vals.size();
             i += (MetalContext::instance().hal().get_alignment(HalMemType::L1) / sizeof(uint32_t))) {
            EXPECT_EQ(readback_sem_vals[i], expected_semaphore_values[sem_idx]);
            sem_idx++;
        }
    }
}

using MeshWorkloadTest2x4 = MeshDevice2x4Fixture;
using MeshWorkloadTest4x8 = MeshDevice4x8Fixture;
using MeshWorkloadTestSuite = GenericMeshDeviceFixture;

// A worker still reading its kernel config must not have that config overwritten, including on devices left out of the
// workloads that follow. The host holds one device's worker while other devices run enough workloads to wrap the
// mesh-wide config ring back onto the held worker's region.
TEST_F(MeshWorkloadTest4x8, UnusedDeviceKernelConfigNotOverwritten) {
    if (mesh_device_->arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Host-to-L1 release handshake is not supported on Quasar data movement cores";
    }

    // Fillers stop as soon as the ring frees the held slot. Extra ones use up the launch-message slots, which would
    // fence the probe on its own account and hide any overwrite.
    constexpr uint32_t max_fillers = 5;
    constexpr uint32_t release_value = 0x67216721;
    constexpr uint32_t started_value = 0x5a5a5a5a;
    constexpr CoreCoord core = {0, 0};
    const CoreRangeSet core_set(CoreRange(core, core));
    const std::string blank_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";

    // The victim is the last device in the mesh; the fillers take every device outside its column.
    const uint32_t last_row = mesh_device_->num_rows() - 1;
    const uint32_t last_col = mesh_device_->num_cols() - 1;
    const MeshCoordinate victim_coord(last_row, last_col);
    const MeshCoordinateRange victim_range(victim_coord, victim_coord);
    const MeshCoordinateRange other_devices(MeshCoordinate(0, 0), MeshCoordinate(last_row, last_col - 1));
    auto& cq = mesh_device_->mesh_command_queue();
    auto* victim = mesh_device_->impl().get_device(victim_coord);

    // Program circular buffers start at the unreserved base and grow up, so the flags go just past the probe's.
    constexpr uint32_t cb_num_pages = 3;
    constexpr uint32_t cb_page_size = tile_size(tt::DataFormat::Float16_b);
    const uint32_t l1_unreserved_base = victim->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t release_addr = l1_unreserved_base + cb_num_pages * cb_page_size;
    const uint32_t started_addr = release_addr + sizeof(uint32_t);

    const uint32_t kernel_config_base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    const uint32_t ring_size = l1_unreserved_base - kernel_config_base;
    // Kernel::validate_runtime_args_size caps a Tensix kernel at max_runtime_args_tensix minus two watcher words.
    constexpr uint32_t max_rt_args_per_kernel = 4096 - 2;
    // Runtime args are the cheapest way to take up ring space, and a quarter of the ring per kernel makes the ring,
    // rather than the launch-message slots, run out first.
    const uint32_t rt_args_per_kernel = std::min<uint32_t>(ring_size / 4 / sizeof(uint32_t), max_rt_args_per_kernel);

    auto write_word = [&](uint32_t addr, uint32_t value) {
        std::vector<uint32_t> word = {value};
        ::tt::tt_metal::detail::WriteToDeviceL1(victim, core, addr, word);
    };
    // Kernels on one core store their runtime args back to back, so each one added here claims another quarter ring.
    auto add_padded_kernel =
        [&](Program& program, const std::string& path, DataMovementProcessor processor, std::vector<uint32_t> args) {
            args.resize(rt_args_per_kernel, 0);
            const auto kernel = CreateKernel(
                program,
                path,
                core_set,
                DataMovementConfig{
                    .processor = processor,
                    .noc = processor == DataMovementProcessor::RISCV_0 ? NOC::RISCV_0_default : NOC::RISCV_1_default});
            SetRuntimeArgs(program, kernel, core_set, args);
        };
    auto poll_until = [](std::chrono::seconds timeout, auto&& predicate) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    // Which ring slots the host still considers in use; the checks below use it to confirm the setup worked.
    auto queued_slots = [&]() {
        return cq.get_config_buffer_mgr(0).get_queued_entry_indices(
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX));
    };

    // Holds the victim's worker until the host releases it. Two kernels make it half the ring, so the filler that later
    // reuses that space covers only half of it and leaves room for the probe.
    Program wait_program;
    add_padded_kernel(
        wait_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/wait_for_host_l1_write.cpp",
        DataMovementProcessor::RISCV_0,
        {release_addr, release_value, started_addr, started_value});
    add_padded_kernel(wait_program, blank_kernel, DataMovementProcessor::RISCV_1, {});
    MeshWorkload waiting_workload;
    waiting_workload.add_program(victim_range, std::move(wait_program));

    // The probe is the workload whose config would land on the held worker's region: small enough to fit the freed
    // space without a wait of its own, and its circular buffer is the pattern the host looks for in L1.
    Program probe_program;
    initialize_dummy_circular_buffers(
        probe_program,
        core_set,
        {CBConfig{
            .cb_id = 0,
            .num_pages = cb_num_pages,
            .page_size = cb_page_size,
            .data_format = tt::DataFormat::Float16_b}});
    CreateKernel(
        probe_program,
        blank_kernel,
        core_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    MeshWorkload probe_workload;
    probe_workload.add_program(victim_range, std::move(probe_program));

    // Fillers skip the victim and are half the held workload's size, so a few of them use up the ring.
    std::vector<MeshWorkload> fillers(max_fillers);
    for (uint32_t i = 0; i < max_fillers; i++) {
        Program filler_program;
        add_padded_kernel(filler_program, blank_kernel, DataMovementProcessor::RISCV_0, {i + 1});
        fillers[i].add_program(other_devices, std::move(filler_program));
    }

    // First pass compiles and caches everything, so the real run below is pure dispatch. The release flag is set up
    // front, so the waiting kernel exits right away here.
    write_word(release_addr, release_value);
    EnqueueMeshWorkload(cq, waiting_workload, false);
    for (auto& filler : fillers) {
        EnqueueMeshWorkload(cq, filler, false);
    }
    EnqueueMeshWorkload(cq, probe_workload, false);
    Finish(cq);

    // Slots are only freed once the ring runs out of room, so the first pass leaves its own marked as in use. They
    // would sit ahead of the held workload and soak up the fillers instead.
    cq.get_config_buffer_mgr(0).mark_completely_full(0);

    // Zero the ring so any circular buffer pattern seen later must come from a new write.
    std::vector<uint32_t> ring_zeros(ring_size / sizeof(uint32_t), 0);
    ::tt::tt_metal::detail::WriteToDeviceL1(victim, core, kernel_config_base, ring_zeros);

    // A failed check must still release the worker, or the mesh stays stuck. Running the release twice is harmless.
    auto release = [&]() {
        write_word(release_addr, release_value);
        Finish(cq);
    };
    struct ReleaseOnExit {
        decltype(release)& fn;
        ~ReleaseOnExit() { fn(); }
    } release_on_exit{release};

    write_word(release_addr, 0);
    write_word(started_addr, 0);
    EnqueueMeshWorkload(cq, waiting_workload, false);
    // The reset above leaves the held workload as the ring's only slot in use.
    const auto held_slots = queued_slots();
    ASSERT_EQ(held_slots.size(), 1u) << "Held workload does not own the only ring slot";
    ASSERT_TRUE(poll_until(std::chrono::seconds(30), [&]() {
        std::vector<uint32_t> started;
        ::tt::tt_metal::detail::ReadFromDeviceL1(victim, core, started_addr, sizeof(uint32_t), started);
        return started.at(0) == started_value;
    })) << "Victim worker never started";

    // Enqueue fillers until the ring reuses the held slot. From here on the ring considers the victim's region free
    // even though its worker is still reading that region.
    bool held_slot_reused = false;
    for (auto& filler : fillers) {
        EnqueueMeshWorkload(cq, filler, false);
        const auto slots = queued_slots();
        held_slot_reused = std::find(slots.begin(), slots.end(), held_slots.front()) == slots.end();
        if (held_slot_reused) {
            break;
        }
    }
    ASSERT_TRUE(held_slot_reused) << "Ring never reused the held slot after " << max_fillers << " fillers; ring is "
                                  << ring_size << " bytes from " << kernel_config_base << " and reached "
                                  << cq.get_config_buffer_mgr(0).get_last_slot_addr(HalProgrammableCoreType::TENSIX);

    // The probe must fit the freed space without a wait of its own, since such a wait would hold its write back for
    // reasons of its own. A wait frees slots, so only the probe's slot may appear.
    const size_t slots_before_probe = queued_slots().size();
    EnqueueMeshWorkload(cq, probe_workload, false);
    ASSERT_EQ(queued_slots().size(), slots_before_probe + 1)
        << "Probe reserved with a wait of its own, so this run cannot show the missing wait";

    const uint32_t probe_cb_base = probe_workload.get_cb_base_addr(mesh_device_, core, CoreType::WORKER);
    auto probe_config_written = [&]() {
        std::vector<uint32_t> cb_config;
        ::tt::tt_metal::detail::ReadFromDeviceL1(
            victim, core, probe_cb_base, UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t), cb_config);
        return cb_config.at(1) == cb_num_pages * cb_page_size && cb_config.at(2) == cb_num_pages;
    };

    // The held worker is still reading this region, so nothing may write kernel config over it.
    const bool written_while_held = poll_until(std::chrono::seconds(2), probe_config_written);
    release();

    EXPECT_FALSE(written_while_held) << "Kernel config overwrote a region a busy worker was still reading";
    // The config does land once the worker is released, so the check above read the right address.
    EXPECT_TRUE(probe_config_written());
}

// Parameterized: runs once with either submesh (index 0 or 1) executing the program.
class MeshWorkloadTestSuiteSubmeshFixture : public MeshWorkloadTestSuite, public ::testing::WithParamInterface<int> {};

// The test succeeds if it completes without hanging.
TEST_P(MeshWorkloadTestSuiteSubmeshFixture, QuiesceSubmeshesAllowsAlternatingWorkloads) {
    if (mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Requires at least 2 devices";
    }

    // Create two evenly sized submeshes (split along columns if possible, else rows).
    MeshShape parent_shape = mesh_device_->shape();
    std::optional<MeshShape> sub_shape;
    if (parent_shape.dims() == 2 && (parent_shape[1] % 2 == 0)) {
        sub_shape = MeshShape(parent_shape[0], parent_shape[1] / 2);
    } else if (parent_shape.dims() == 2 && (parent_shape[0] % 2 == 0)) {
        sub_shape = MeshShape(parent_shape[0] / 2, parent_shape[1]);
    }

    if (!sub_shape.has_value()) {
        GTEST_SKIP() << "Mesh shape is not evenly splittable into two submeshes";
    }

    auto submeshes = mesh_device_->create_submeshes(*sub_shape);
    ASSERT_EQ(submeshes.size(), 2u);

    int submesh_index = GetParam();
    ASSERT_TRUE(submesh_index == 0 || submesh_index == 1);
    auto& submesh = submeshes[submesh_index];

    // Single-core no-op program for submesh
    Program submesh_program = CreateProgram();
    CoreCoord single_core = {0, 0};
    CreateKernel(
        submesh_program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", single_core, ComputeConfig{});
    MeshWorkload submesh_workload;
    submesh_workload.add_program(MeshCoordinateRange(submesh->shape()), std::move(submesh_program));

    // Single-core no-op program for parent mesh
    Program parent_program = CreateProgram();
    CreateKernel(
        parent_program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", single_core, ComputeConfig{});
    MeshWorkload parent_workload;
    parent_workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(parent_program));

    // 1) Run on submesh (non-blocking)
    EnqueueMeshWorkload(submesh->mesh_command_queue(), submesh_workload, /*blocking=*/false);

    // Enqueue a record event to ensure the last event ID is higher on the submesh than on the parent mesh.
    submesh->mesh_command_queue().enqueue_record_event();

    // 2) Quiesce all submeshes from the parent
    mesh_device_->quiesce_devices();

    // 3) Run on parent (non-blocking)
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), parent_workload, /*blocking=*/false);

    // 4) Quiesce again
    mesh_device_->quiesce_devices();

    // 5) Run again on the same submesh (non-blocking) and finish to ensure completion
    EnqueueMeshWorkload(submesh->mesh_command_queue(), submesh_workload, /*blocking=*/false);
    Finish(submesh->mesh_command_queue());
}

INSTANTIATE_TEST_SUITE_P(QuiesceSubmeshIndex, MeshWorkloadTestSuiteSubmeshFixture, ::testing::Values(0, 1));

TEST_F(MeshWorkloadTestSuite, TestMeshWorkloadOnActiveEth) {
    uint32_t num_workloads = 10;
    auto random_seed = 0;
    uint32_t num_iters = 500;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    std::vector<std::shared_ptr<MeshWorkload>> workloads = {};
    log_info(tt::LogTest, "Create {} workloads", num_workloads);
    for (int i = 0; i < num_workloads; i++) {
        std::shared_ptr<MeshWorkload> workload = std::make_shared<MeshWorkload>();
        for (const auto& device_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (mesh_device_->impl().is_local(device_coord)) {
                IDevice* device = mesh_device_->impl().get_device(device_coord);
                auto programs = utils::create_random_programs(
                    1, mesh_device_->compute_with_storage_grid_size(), seed, device->get_active_ethernet_cores(true));
                workload->add_program(MeshCoordinateRange(device_coord, device_coord), std::move(*programs[0]));
            }
        }
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        workloads.push_back(workload);
    }
    for (int i = 0; i < num_iters; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    Finish(mesh_device_->mesh_command_queue());
}

TEST_F(MeshWorkloadTestSuite, OverlappingProgramRanges) {
    MeshWorkload workload;

    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        /*num_programs=*/2, mesh_device_->compute_with_storage_grid_size(), /*seed=*/0);
    auto mesh_workload = MeshWorkload();

    MeshCoordinate zero_coord = MeshCoordinate::zero_coordinate(mesh_device_->shape().dims());
    MeshCoordinateRange devices_range = MeshCoordinateRange(zero_coord, zero_coord);

    mesh_workload.add_program(devices_range, std::move(*programs[0]));
    EXPECT_THAT(
        ([&]() { mesh_workload.add_program(devices_range, std::move(*programs[1])); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("overlaps with the previously added range")));
}

TEST_F(MeshWorkloadTest2x4, SimultaneousMeshWorkloads) {
    uint32_t num_programs = 100;
    uint32_t num_heterogeneous_programs = 64;
    uint32_t num_iterations = 1000;
    auto random_seed = 0;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    log_info(tt::LogTest, "Create MeshWorkloads with multiple programs each");

    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 2) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        if (i % 2) {
            MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{0, 3});
            MeshCoordinateRange devices_1(MeshCoordinate{1, 0}, MeshCoordinate{1, 3});
            random_workload->add_program(devices_0, std::move(*programs[i]));
            random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        } else {
            MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{1, 1});
            MeshCoordinateRange devices_1(MeshCoordinate{0, 2}, MeshCoordinate{1, 3});
            random_workload->add_program(devices_0, std::move(*programs[i]));
            random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        }
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_programs; i += 4) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{1, 0});
        MeshCoordinateRange devices_1(MeshCoordinate{0, 1}, MeshCoordinate{1, 1});
        MeshCoordinateRange devices_2(MeshCoordinate{0, 2}, MeshCoordinate{1, 2});
        MeshCoordinateRange devices_3(MeshCoordinate{0, 3}, MeshCoordinate{1, 3});
        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        random_workload->add_program(devices_2, std::move(*programs[i + 2]));
        random_workload->add_program(devices_3, std::move(*programs[i + 3]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_heterogeneous_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_heterogeneous_programs; i += 8) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});
        MeshCoordinateRange devices_1(MeshCoordinate{0, 1}, MeshCoordinate{0, 1});
        MeshCoordinateRange devices_2(MeshCoordinate{0, 2}, MeshCoordinate{0, 2});
        MeshCoordinateRange devices_3(MeshCoordinate{0, 3}, MeshCoordinate{0, 3});
        MeshCoordinateRange devices_4(MeshCoordinate{1, 0}, MeshCoordinate{1, 0});
        MeshCoordinateRange devices_5(MeshCoordinate{1, 1}, MeshCoordinate{1, 1});
        MeshCoordinateRange devices_6(MeshCoordinate{1, 2}, MeshCoordinate{1, 2});
        MeshCoordinateRange devices_7(MeshCoordinate{1, 3}, MeshCoordinate{1, 3});

        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        random_workload->add_program(devices_2, std::move(*programs[i + 2]));
        random_workload->add_program(devices_3, std::move(*programs[i + 3]));
        random_workload->add_program(devices_4, std::move(*programs[i + 4]));
        random_workload->add_program(devices_5, std::move(*programs[i + 5]));
        random_workload->add_program(devices_6, std::move(*programs[i + 6]));
        random_workload->add_program(devices_7, std::move(*programs[i + 7]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }

    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : mesh_workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    Finish(mesh_device_->mesh_command_queue());
}

// 4x8 benchmark: 32 unique programs on 32 devices (1:1 mapping)
// Expects parallel dispatch with thread pool.
// This replicates the deepseek blitz case of 32 independent programs dispatched across 32 devices on a GLX
TEST_F(MeshDeviceFixture4x8DispatchAgnostic, ParallelizationBenchmark) {
    constexpr uint32_t num_programs = 32;
    constexpr uint32_t num_devices = 32;

    auto programs = tt::tt_metal::distributed::test::utils::create_benchmark_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), true);

    std::vector<MeshCoordinateRange> devices;
    devices.reserve(num_devices);
    for (uint32_t d = 0; d < num_devices; d++) {
        devices.emplace_back(MeshCoordinate{d / 8, d % 8});
    }

    MeshWorkload workload;
    for (uint32_t d = 0; d < num_devices; d++) {
        workload.add_program(devices[d], std::move(*programs[d]));
    }

    auto t0 = std::chrono::steady_clock::now();
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
    auto t1 = std::chrono::steady_clock::now();
    Finish(mesh_device_->mesh_command_queue());
    auto t2 = std::chrono::steady_clock::now();

    auto enqueue_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto finish_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    log_info(tt::LogTest, "ParallelizationBenchmark: {} programs on {} devices", num_programs, num_devices);
    log_info(tt::LogTest, "Enqueue time: {} ms", enqueue_ms);
    log_info(tt::LogTest, "Finish time: {} ms", finish_ms);
    log_info(tt::LogTest, "Total time: {} ms", enqueue_ms + finish_ms);
}

TEST_F(MeshWorkloadTest4x8, SimultaneousMeshWorkloads) {
    uint32_t num_programs_0 = 16;
    uint32_t num_programs_1 = 24;
    uint32_t num_iterations = 1000;
    auto random_seed = 0;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    log_info(tt::LogTest, "Create MeshWorkloads with multiple programs each");

    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", 2 * (num_programs_0 + num_programs_1));

    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs_0, mesh_device_->compute_with_storage_grid_size(), seed);

    for (int i = 0; i < num_programs_0; i += 2) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{1, 7});
        MeshCoordinateRange devices_1(MeshCoordinate{2, 0}, MeshCoordinate{3, 7});
        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }

    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs_0, mesh_device_->compute_with_storage_grid_size(), seed);

    for (int i = 0; i < num_programs_0; i += 2) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{3, 3});
        MeshCoordinateRange devices_1(MeshCoordinate{0, 4}, MeshCoordinate{3, 7});
        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }

    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs_1, mesh_device_->compute_with_storage_grid_size(), seed);

    for (int i = 0; i < num_programs_1; i += 4) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{0, 7});
        MeshCoordinateRange devices_1(MeshCoordinate{1, 0}, MeshCoordinate{1, 7});
        MeshCoordinateRange devices_2(MeshCoordinate{2, 0}, MeshCoordinate{2, 7});
        MeshCoordinateRange devices_3(MeshCoordinate{3, 0}, MeshCoordinate{3, 7});

        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        random_workload->add_program(devices_2, std::move(*programs[i + 2]));
        random_workload->add_program(devices_3, std::move(*programs[i + 3]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }

    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs_1, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_programs_1; i += 8) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{3, 0});
        MeshCoordinateRange devices_1(MeshCoordinate{0, 1}, MeshCoordinate{3, 1});
        MeshCoordinateRange devices_2(MeshCoordinate{0, 2}, MeshCoordinate{3, 2});
        MeshCoordinateRange devices_3(MeshCoordinate{0, 3}, MeshCoordinate{3, 3});
        MeshCoordinateRange devices_4(MeshCoordinate{0, 4}, MeshCoordinate{3, 4});
        MeshCoordinateRange devices_5(MeshCoordinate{0, 5}, MeshCoordinate{3, 5});
        MeshCoordinateRange devices_6(MeshCoordinate{0, 6}, MeshCoordinate{3, 6});
        MeshCoordinateRange devices_7(MeshCoordinate{0, 7}, MeshCoordinate{3, 7});

        random_workload->add_program(devices_0, std::move(*programs[i]));
        random_workload->add_program(devices_1, std::move(*programs[i + 1]));
        random_workload->add_program(devices_2, std::move(*programs[i + 2]));
        random_workload->add_program(devices_3, std::move(*programs[i + 3]));
        random_workload->add_program(devices_4, std::move(*programs[i + 4]));
        random_workload->add_program(devices_5, std::move(*programs[i + 5]));
        random_workload->add_program(devices_6, std::move(*programs[i + 6]));
        random_workload->add_program(devices_7, std::move(*programs[i + 7]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : mesh_workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    Finish(mesh_device_->mesh_command_queue());
}

TEST_F(MeshWorkloadTestSuite, RandomizedMeshWorkload) {
    uint32_t num_programs = 60;
    uint32_t num_iterations = 1500;
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);
    log_info(tt::LogTest, "Create {} MeshWorkloads", num_programs);
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gen_col(1, mesh_device_->num_cols());
    std::uniform_int_distribution<int> gen_row(1, mesh_device_->num_rows());
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    // Create multiple mesh workloads on grids of random sizes.
    // Compile the workload (lower + send binaries to mesh device here as well)
    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 1) {
        // Choose a grid of random dimensions and run a MeshWorkload on it
        MeshCoordinateRange device_range(MeshCoordinate{0, 0}, MeshCoordinate{gen_row(rng) - 1, gen_col(rng) - 1});
        auto random_workload = std::make_shared<MeshWorkload>();
        random_workload->add_program(device_range, std::move(*programs[i]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : mesh_workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    log_info(tt::LogTest, "Calling Finish");
    Finish(mesh_device_->mesh_command_queue());
}

TEST_F(MeshWorkloadTestSuite, EltwiseBinaryMeshWorkload) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    std::vector<std::shared_ptr<MeshBuffer>> src0_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> src1_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    auto programs = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, src0_bufs, src1_bufs, output_bufs);
    uint32_t num_rows = mesh_device_->num_rows();
    uint32_t num_rows_in_mesh_workload = num_rows / 2;
    TT_FATAL(num_rows_in_mesh_workload > 0, "The MeshWorkload must be enqueued on at least one row.");
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_0(
        MeshCoordinate{0, 0}, MeshCoordinate{num_rows_in_mesh_workload - 1, mesh_device_->num_cols() - 1});
    MeshCoordinateRange devices_1(
        MeshCoordinate{num_rows_in_mesh_workload, 0}, MeshCoordinate{num_rows - 1, mesh_device_->num_cols() - 1});
    mesh_workload.add_program(devices_0, std::move(*programs[0]));
    mesh_workload.add_program(devices_1, std::move(*programs[1]));
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), 2);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src1_bufs[0]->size(), 3);

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src0_bufs[(col_idx * worker_grid_size.y) + row_idx], src0_vec);
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src1_bufs[(col_idx * worker_grid_size.y) + row_idx], src1_vec);
        }
    }

    // Run workload multiple times
    for (int i = 0; i < 1000; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    }

    for (const auto& device_coord : MeshCoordinateRange(mesh_device_->shape())) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                std::vector<bfloat16> dst_vec = {};
                ReadShard(
                    mesh_device_->mesh_command_queue(),
                    dst_vec,
                    output_bufs[(col_idx * worker_grid_size.y) + row_idx],
                    device_coord);
                if (device_coord[0] <= num_rows_in_mesh_workload - 1) {
                    for (auto val : dst_vec) {
                        EXPECT_EQ(static_cast<float>(val), 5);
                    }
                } else {
                    for (auto val : dst_vec) {
                        EXPECT_EQ(static_cast<float>(val), 6);
                    }
                }
            }
        }
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSanity) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    // Create buffers
    std::vector<std::shared_ptr<MeshBuffer>> input_buffers = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_buffers = {};

    ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = true};

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            input_buffers.push_back(
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get()));
            output_buffers.push_back(
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get()));
        }
    }

    // Create MeshWorkload
    Program program = CreateProgram();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    auto reader_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        full_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    auto scaling_sem_idx = CreateSemaphore(program, full_grid, sem_scaling_factor);
    uint32_t scaling_height_toggle = 16;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    uint32_t add_factor = 64;
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            CoreCoord curr_core = {col_idx, row_idx};
            SetRuntimeArgs(
                program,
                reader_writer_kernel,
                curr_core,
                {input_buffers.at((col_idx * worker_grid_size.y) + row_idx)->address(),
                 output_buffers.at((col_idx * worker_grid_size.y) + row_idx)->address(),
                 0, /* src_bank_id */
                 0, /* dst_bank_id */
                 add_factor,
                 constants::TILE_HEIGHT,
                 constants::TILE_WIDTH,
                 scaling_sem_idx,
                 scaling_height_toggle});
            CreateCircularBuffer(program, curr_core, cb_src0_config);
        }
    }
    auto program_1 = initialize_dummy_program(worker_grid_size);
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_0(MeshCoordinate{0, 0}, MeshCoordinate{0, mesh_device_->num_cols() - 1});
    MeshCoordinateRange devices_1(MeshCoordinate{1, 0}, MeshCoordinate{1, mesh_device_->num_cols() - 1});
    mesh_workload.add_program(devices_0, std::move(program));
    mesh_workload.add_program(devices_1, std::move(*program_1));

    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), input_buffers[(col_idx * worker_grid_size.y) + row_idx], src_vec);
        }
    }

    for (int iter = 0; iter < 100; iter++) {
        log_info(LogTest, "Run iter {}", iter);
        if (iter) {
            auto& program = mesh_workload.get_programs().at(devices_0);
            auto& rtas = GetRuntimeArgs(program, reader_writer_kernel);
            for (auto core : full_grid) {
                rtas[core.x][core.y].at(4) = ((iter % 2) + 1) * add_factor;
            }
        }
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
        for (const auto& device_coord : devices_0) {
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    std::vector<bfloat16> dst_vec = {};
                    ReadShard(
                        mesh_device_->mesh_command_queue(),
                        dst_vec,
                        output_buffers[(col_idx * worker_grid_size.y) + row_idx],
                        device_coord);
                    for (int i = 0; i < dst_vec.size(); i++) {
                        float ref_val = std::pow(2, (iter % 2) + 1);
                        if (i >= 512) {
                            ref_val = std::pow(2, 2 * ((iter % 2) + 1));
                        }
                        EXPECT_EQ(static_cast<float>(dst_vec[i]), ref_val);
                    }
                }
            }
        }
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadCBUpdate) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    CoreRange cr = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    const std::vector<CBHandle>& cb_handles = initialize_dummy_circular_buffers(*program, cr_set, cb_config_vector);
    initialize_dummy_kernels(*program, cr_set);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(mesh_device_->shape());

    mesh_workload.add_program(devices, std::move(*program));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());
    verify_cb_config(mesh_device_, mesh_workload, cb_config_vector, cr_set);

    std::vector<CBConfig> updated_cb_config_vector = cb_config_vector;
    for (uint32_t cb_id = 0; cb_id < cb_config_vector.size(); cb_id++) {
        CBConfig& cb_config = updated_cb_config_vector[cb_id];
        cb_config.num_pages *= 2;
        const uint32_t cb_size = cb_config.num_pages * cb_config.page_size;
        UpdateCircularBufferTotalSize(mesh_workload.get_programs().at(devices), cb_handles[cb_id], cb_size);
    }
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());
    verify_cb_config(mesh_device_, mesh_workload, updated_cb_config_vector, cr_set);
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSemaphoreSanity) {
    auto worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    Program program;
    std::vector<uint32_t> expected_semaphore_values;

    for (uint32_t sem = 0; sem < NUM_SEMAPHORES; sem++) {
        CreateSemaphore(program, full_grid, sem);
        expected_semaphore_values.push_back(sem);
    }
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(mesh_device_->shape());
    mesh_workload.add_program(devices, std::move(program));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());

    for (auto* const device : mesh_device_->get_devices()) {
        validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values);
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSemaphoreDifferentPrograms) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    auto worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    Program program0;
    Program program1;
    std::vector<uint32_t> expected_semaphore_values_0;
    std::vector<uint32_t> expected_semaphore_values_1;

    for (uint32_t sem = 0; sem < NUM_SEMAPHORES; sem++) {
        CreateSemaphore(program0, full_grid, sem);
        expected_semaphore_values_0.push_back(sem);

        CreateSemaphore(program1, full_grid, sem + 1);
        expected_semaphore_values_1.push_back(sem + 1);
    }
    uint32_t num_cols_in_workload = mesh_device_->num_cols() / 2;
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices_0({0, 0}, {mesh_device_->num_rows() - 1, num_cols_in_workload - 1});
    MeshCoordinateRange devices_1(
        {0, num_cols_in_workload}, {mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1});

    mesh_workload.add_program(devices_0, std::move(program0));
    mesh_workload.add_program(devices_1, std::move(program1));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());

    for (const auto& device_coord : devices_0) {
        if (!mesh_device_->impl().is_local(device_coord)) {
            continue;
        }
        auto* device = mesh_device_->impl().get_device(device_coord);
        validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values_0);
    }

    for (const auto& device_coord : devices_1) {
        if (!mesh_device_->impl().is_local(device_coord)) {
            continue;
        }
        auto* device = mesh_device_->impl().get_device(device_coord);
        validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values_1);
    }
}

TEST_F(MeshWorkloadTestSuite, RandomizedMeshWorkloadMultiThread) {
    uint32_t num_programs = 30;
    uint32_t num_iterations = 1500;
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);
    log_info(tt::LogTest, "Create {} MeshWorkloads", num_programs);

    std::vector<std::thread> threads;
    for (int thread_idx = 0; thread_idx < 2; thread_idx += 1) {
        threads.push_back(std::thread([&, thread_idx]() {
            auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
                num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> gen_col(1, mesh_device_->num_cols());
            std::uniform_int_distribution<int> gen_row(1, mesh_device_->num_rows());
            std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

            // Create multiple mesh workloads on grids of random sizes.
            // Compile the workload (lower + send binaries to mesh device here as well)
            log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
            for (int i = 0; i < num_programs; i += 1) {
                // Choose a grid of random dimensions and run a MeshWorkload on it
                MeshCoordinateRange device_range(
                    MeshCoordinate{0, 0}, MeshCoordinate{gen_row(rng) - 1, gen_col(rng) - 1});
                auto random_workload = std::make_shared<MeshWorkload>();
                random_workload->add_program(device_range, std::move(*programs[i]));
                EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
                mesh_workloads.push_back(random_workload);
            }
            for (int i = 0; i < num_iterations; i++) {
                if (i % 100 == 0) {
                    log_info(tt::LogTest, "Run MeshWorkloads thread {} for iteration {}", thread_idx, i);
                }
                for (auto& workload : mesh_workloads) {
                    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
                }
            }
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }

    log_info(tt::LogTest, "Calling Finish");
    Finish(mesh_device_->mesh_command_queue());
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
