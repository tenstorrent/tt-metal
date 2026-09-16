// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "command_queue_fixture.hpp"
#include "env_lib.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <gtest/gtest.h>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/buffers/semaphore.hpp"
#include "impl/kernels/kernel.hpp"
#include "dispatch_test_utils.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal {

class UnitMeshRandomProgramFixture : virtual public UnitMeshCQSingleCardProgramFixture {
protected:
    static const uint32_t MIN_KERNEL_SIZE_BYTES = 20;
    static const uint32_t MAX_KERNEL_SIZE_BYTES = 4096;

    static const uint32_t MIN_KERNEL_RUNTIME_MICROSECONDS = 100;
    static const uint32_t MAX_KERNEL_RUNTIME_MICROSECONDS = 20000;

    static const uint32_t MIN_NUM_RUNTIME_ARGS = 0;
    static const uint32_t MAX_NUM_RUNTIME_ARGS = max_runtime_args;
    static const uint32_t UNIQUE_RUNTIME_ARGS_VAL_OFFSET = 50;
    static const uint32_t COMMON_RUNTIME_ARGS_VAL_OFFSET = 100;

    static const uint32_t MIN_NUM_SEMS = 0;
    static const uint32_t MAX_NUM_SEMS = NUM_SEMAPHORES;
    static const uint32_t SEM_VAL = 1;

    // A DM-produced, Tensix-consumed DFB takes one of the 16 DM-visible tile counters on its
    // tensix, so that is the ceiling here rather than the 32 device slots Gen2 allows. It also
    // matches the length of the accessor ladder in dispatcher_kernel_size_and_runtime_2_0.cpp.
    static const uint32_t MAX_NUM_DFBS = 16;

    static const uint32_t MIN_NUM_CBS = 0;
    static const uint32_t MIN_CB_PAGE_SIZE = 16;
    static const uint32_t MAX_CB_PAGE_SIZE = 64;
    static const uint32_t MIN_CB_TOTAL_SIZE = MAX_CB_PAGE_SIZE;
    static const uint32_t MAX_CB_TOTAL_SIZE = 2048;

    struct KernelProperties {
        uint32_t min_kernel_size_bytes;
        uint32_t max_kernel_size_bytes;
        uint32_t min_kernel_runtime_microseconds;
        uint32_t max_kernel_runtime_microseconds;
        uint32_t min_num_rt_args;
        uint32_t max_num_rt_args;
        uint32_t min_num_sems;
        uint32_t max_num_sems;
        uint32_t min_num_cbs;
        uint32_t max_num_cbs{};
        KernelProperties() :
            min_kernel_size_bytes(MIN_KERNEL_SIZE_BYTES),
            max_kernel_size_bytes(MAX_KERNEL_SIZE_BYTES),
            min_kernel_runtime_microseconds(MIN_KERNEL_RUNTIME_MICROSECONDS),
            max_kernel_runtime_microseconds(MAX_KERNEL_RUNTIME_MICROSECONDS),
            min_num_rt_args(MIN_NUM_RUNTIME_ARGS),
            max_num_rt_args(MAX_NUM_RUNTIME_ARGS),
            min_num_sems(MIN_NUM_SEMS),
            max_num_sems(MAX_NUM_SEMS),
            min_num_cbs(MIN_NUM_CBS) {}
    };

    static const uint32_t NUM_WORKLOADS = 75;

    std::shared_ptr<distributed::MeshDevice> device_;

    void SetUp() override {
        UnitMeshCQSingleCardProgramFixture::SetUp();
        if (!::testing::Test::IsSkipped()) {
            // Parent may have skipped
            this->device_ = this->devices_[0];
            this->initialize_seed();
        }
    }

    void initialize_seed() {
        const uint32_t seed = tt::parse_env("TT_METAL_SEED", static_cast<uint32_t>(time(nullptr)));
        log_info(tt::LogTest, "Using seed: {}", seed);
        srand(seed);
    }
    // Gen2 programs are built from a ProgramSpec rather than having kernels added to an existing
    // Program, so callers that support both take the whole program from here.
    Program create_program_with_kernel(
        const CoreType kernel_core_type,
        const bool simple_kernel = false,
        KernelProperties kernel_properties = KernelProperties()) {
        if (device_->arch() != ARCH::QUASAR) {
            Program program = CreateProgram();
            this->create_kernel(program, kernel_core_type, simple_kernel, kernel_properties);
            return program;
        }
        return this->create_gen2_program(kernel_core_type, simple_kernel, kernel_properties);
    }

    void create_kernel(
        Program& program,
        const CoreType kernel_core_type,
        const bool simple_kernel = false,
        KernelProperties kernel_properties = KernelProperties()) {
        if (kernel_properties.max_num_cbs == 0) {
            kernel_properties.max_num_cbs = max_cbs_;
        }

        CoreRangeSet cores = this->get_cores(kernel_core_type);
        const bool create_eth_config = kernel_core_type == CoreType::ETH;

        if (simple_kernel) {
            this->create_kernel(program, cores, create_eth_config, 0, 0, 0, 0);
        } else {
            const std::vector<uint32_t> sem_ids = this->generate_semaphores(
                program, cores, kernel_core_type, kernel_properties.min_num_sems, kernel_properties.max_num_sems);

            std::vector<uint32_t> cb_page_sizes;
            if (!create_eth_config) {
                cb_page_sizes = this->generate_circular_buffers(
                    program, cores, kernel_properties.min_num_cbs, kernel_properties.max_num_cbs);
            }

            const auto [unique_rt_args, common_rt_args] = this->generate_runtime_args(
                sem_ids, cb_page_sizes, kernel_properties.min_num_rt_args, kernel_properties.max_num_rt_args);
            const uint32_t num_unique_rt_args = unique_rt_args.size() - sem_ids.size() - cb_page_sizes.size();

            KernelHandle kernel_id = this->create_kernel(
                program,
                cores,
                create_eth_config,
                sem_ids.size(),
                cb_page_sizes.size(),
                num_unique_rt_args,
                common_rt_args.size(),
                kernel_properties.min_kernel_size_bytes,
                kernel_properties.max_kernel_size_bytes,
                kernel_properties.min_kernel_runtime_microseconds,
                kernel_properties.max_kernel_runtime_microseconds);
            SetRuntimeArgs(program, kernel_id, cores, unique_rt_args);
            SetCommonRuntimeArgs(program, kernel_id, common_rt_args);
        }
    }

    std::vector<uint32_t> generate_semaphores(
        Program& program,
        const CoreRangeSet& cores,
        const CoreType core_type = CoreType::WORKER,
        const uint32_t min = MIN_NUM_SEMS,
        const uint32_t max = MAX_NUM_SEMS) {
        const uint32_t num_sems = this->generate_random_num(min, max);
        std::vector<uint32_t> sem_ids;
        for (uint32_t i = 0; i < num_sems; i++) {
            const uint32_t sem_id = CreateSemaphore(program, cores, SEM_VAL, core_type);
            sem_ids.push_back(sem_id);
        }
        return sem_ids;
    }

    std::vector<uint32_t> generate_circular_buffers(
        Program& program, const CoreRangeSet& cores, const uint32_t min, const uint32_t max) {
        const uint32_t num_cbs = this->generate_random_num(min, max);
        std::vector<uint32_t> cb_page_sizes;
        for (uint32_t cb_idx = 0; cb_idx < num_cbs; cb_idx++) {
            const uint32_t cb_page_size =
                this->generate_random_num(MIN_CB_PAGE_SIZE, MAX_CB_PAGE_SIZE, CIRCULAR_BUFFER_COMPUTE_WORD_SIZE);
            const uint32_t cb_total_size =
                this->generate_random_num(MIN_CB_TOTAL_SIZE, MAX_CB_TOTAL_SIZE, cb_page_size);
            CircularBufferConfig config = CircularBufferConfig(cb_total_size, {{cb_idx, tt::DataFormat::Float16_b}})
                                              .set_page_size(cb_idx, cb_page_size);
            CreateCircularBuffer(program, cores, config);
            cb_page_sizes.push_back(cb_page_size);
        }
        return cb_page_sizes;
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> generate_runtime_args(
        const std::vector<uint32_t>& sem_ids,
        const std::vector<uint32_t>& cb_page_sizes,
        const uint32_t min = MIN_NUM_RUNTIME_ARGS,
        const uint32_t max = MAX_NUM_RUNTIME_ARGS) {
        const uint32_t num_sems = sem_ids.size();
        const uint32_t num_cbs = cb_page_sizes.size();
        TT_FATAL(
            max >= num_sems + num_cbs,
            "Max number of runtime args to generate must be >= number of semaphores + number of circular buffers "
            "created");

        const uint32_t max_num_unique_rt_args = max - num_sems - num_cbs;
        const uint32_t min_num_unique_rt_args =
            static_cast<uint32_t>(std::max(static_cast<int>(min - num_sems - num_cbs), 0));
        const uint32_t num_unique_rt_args = this->generate_random_num(min_num_unique_rt_args, max_num_unique_rt_args);

        const uint32_t max_num_common_rt_args = max_num_unique_rt_args - num_unique_rt_args;
        const uint32_t num_common_rt_args = this->generate_random_num(0, max_num_common_rt_args);

        auto [unique_rt_args, common_rt_args] = create_runtime_args(
            num_unique_rt_args, num_common_rt_args, UNIQUE_RUNTIME_ARGS_VAL_OFFSET, COMMON_RUNTIME_ARGS_VAL_OFFSET);

        unique_rt_args.insert(unique_rt_args.end(), sem_ids.begin(), sem_ids.end());
        unique_rt_args.insert(unique_rt_args.end(), cb_page_sizes.begin(), cb_page_sizes.end());

        return {unique_rt_args, common_rt_args};
    }

    KernelProperties get_small_kernel_properties() {
        KernelProperties small_kernel_properties;
        small_kernel_properties.min_kernel_size_bytes = MIN_KERNEL_SIZE_BYTES;
        small_kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES * (2.0 / 10);
        small_kernel_properties.min_kernel_runtime_microseconds = MIN_KERNEL_RUNTIME_MICROSECONDS;
        small_kernel_properties.max_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS * (2.0 / 10);
        small_kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS * (3.0 / 10);
        small_kernel_properties.min_num_sems = MIN_NUM_SEMS;
        small_kernel_properties.max_num_sems = MAX_NUM_SEMS * (3.0 / 10);
        small_kernel_properties.min_num_cbs = MIN_NUM_CBS;
        small_kernel_properties.max_num_cbs = max_cbs_ * (3.0 / 10);
        small_kernel_properties.min_num_rt_args =
            small_kernel_properties.max_num_sems + small_kernel_properties.max_num_cbs;
        return small_kernel_properties;
    }

    KernelProperties get_large_kernel_properties() {
        KernelProperties large_kernel_properties;
        large_kernel_properties.min_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES * (9.0 / 10);
        large_kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES;
        large_kernel_properties.min_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS * (9.0 / 10);
        large_kernel_properties.max_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS;
        large_kernel_properties.min_num_rt_args = MAX_NUM_RUNTIME_ARGS * (9.0 / 10);
        large_kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS;
        large_kernel_properties.min_num_sems = MAX_NUM_SEMS * (8.0 / 10);
        large_kernel_properties.max_num_sems = MAX_NUM_SEMS;
        large_kernel_properties.min_num_cbs = max_cbs_ * (8.0 / 10);
        large_kernel_properties.max_num_cbs = max_cbs_;
        return large_kernel_properties;
    }

private:
    // Gen2 equivalent of the create_kernel path below. Semaphores are declared so dispatch still has
    // to place them, but are not value checked: Gen2 only allows a zero initial value.
    Program create_gen2_program(
        const CoreType kernel_core_type, const bool simple_kernel, KernelProperties kernel_properties) {
        using namespace tt::tt_metal::experimental;

        TT_FATAL(kernel_core_type == CoreType::WORKER, "Only worker cores are ported to Metal 2.0");
        if (kernel_properties.max_num_cbs == 0) {
            kernel_properties.max_num_cbs = max_cbs_;
        }
        // hal().get_arch_num_circular_buffers() reports 64 on Quasar, far more DFBs than a single
        // node has tile counters for, so clamp both ends of the range.
        kernel_properties.max_num_cbs = std::min(kernel_properties.max_num_cbs, MAX_NUM_DFBS);
        kernel_properties.min_num_cbs = std::min(kernel_properties.min_num_cbs, kernel_properties.max_num_cbs);
        const CoreRangeSet cores = this->get_cores(kernel_core_type);

        std::vector<uint32_t> sem_ids;
        std::vector<uint32_t> entry_sizes;
        std::vector<uint32_t> num_entries;
        std::vector<uint32_t> unique_rt_args;
        std::vector<uint32_t> common_rt_args;
        uint32_t num_unique_rt_args = 0;

        if (!simple_kernel) {
            const uint32_t num_sems =
                this->generate_random_num(kernel_properties.min_num_sems, kernel_properties.max_num_sems);
            for (uint32_t i = 0; i < num_sems; i++) {
                sem_ids.push_back(i);
            }

            const uint32_t num_dfbs =
                this->generate_random_num(kernel_properties.min_num_cbs, kernel_properties.max_num_cbs);
            for (uint32_t i = 0; i < num_dfbs; i++) {
                const uint32_t entry_size =
                    this->generate_random_num(MIN_CB_PAGE_SIZE, MAX_CB_PAGE_SIZE, CIRCULAR_BUFFER_COMPUTE_WORD_SIZE);
                const uint32_t total_size = this->generate_random_num(MIN_CB_TOTAL_SIZE, MAX_CB_TOTAL_SIZE, entry_size);
                entry_sizes.push_back(entry_size);
                num_entries.push_back(total_size / entry_size);
            }

            std::tie(unique_rt_args, common_rt_args) = this->generate_runtime_args(
                sem_ids, entry_sizes, kernel_properties.min_num_rt_args, kernel_properties.max_num_rt_args);
            num_unique_rt_args = unique_rt_args.size() - sem_ids.size() - entry_sizes.size();
        }

        const uint32_t kernel_size_bytes =
            this->generate_random_num(kernel_properties.min_kernel_size_bytes, kernel_properties.max_kernel_size_bytes);
        const uint32_t kernel_runtime_microseconds = this->generate_random_num(
            kernel_properties.min_kernel_runtime_microseconds, kernel_properties.max_kernel_runtime_microseconds);

        KernelSpec::CompilerOptions::Defines defines;
        defines.emplace("KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes));
        defines.emplace("KERNEL_RUNTIME_MICROSECONDS", std::to_string(kernel_runtime_microseconds));
        // NUM_DFBS would collide with the firmware's dfb::NUM_DFBS constant.
        defines.emplace("NUM_TEST_DFBS", std::to_string(entry_sizes.size()));

        // Gen2 rejects a data-movement kernel bound as both ends of a DFB, so the kernel under test
        // produces and a blank compute kernel consumes. Nothing is pushed through the buffers, as on
        // Gen1; the point is that dispatch has to deliver each buffer's config for the kernel to read
        // back. A compute endpoint requires the data format to be declared.
        Group<DataflowBufferSpec> dataflow_buffers;
        Group<KernelSpec::DFBBinding> producer_bindings;
        Group<KernelSpec::DFBBinding> consumer_bindings;
        for (uint32_t i = 0; i < entry_sizes.size(); i++) {
            const DFBSpecName dfb_name{"dfb_" + std::to_string(i)};
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = dfb_name,
                .entry_size = entry_sizes[i],
                .num_entries = num_entries[i],
                .data_format_metadata = tt::DataFormat::Float16_b,
            });
            producer_bindings.push_back(KernelSpec::DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = dfb_name.get(),
                .endpoint_type = KernelSpec::DFBBinding::EndpointType::PRODUCER,
            });
            consumer_bindings.push_back(KernelSpec::DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = dfb_name.get() + "_in",
                .endpoint_type = KernelSpec::DFBBinding::EndpointType::CONSUMER,
            });
        }

        Group<SemaphoreSpec> semaphores;
        for (uint32_t sem_id : sem_ids) {
            semaphores.push_back(SemaphoreSpec{
                .unique_id = SemaphoreSpecName{"sem_" + std::to_string(sem_id)},
                .target_nodes = cores,
            });
        }

        const KernelSpecName name{"dispatcher_kernel_size_and_runtime"};
        KernelSpec kernel_spec{
            .unique_id = name,
            .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/"
                                            "dispatcher_kernel_size_and_runtime_2_0.cpp"},
            .num_threads = 1,
            .compiler_options = {.defines = std::move(defines)},
            .dfb_bindings = producer_bindings,
            .compile_time_args =
                {{"num_unique_rt_args", num_unique_rt_args},
                 {"num_common_rt_args", static_cast<uint32_t>(common_rt_args.size())},
                 {"unique_rt_args_vals_offset", UNIQUE_RUNTIME_ARGS_VAL_OFFSET},
                 {"common_rt_args_vals_offset", COMMON_RUNTIME_ARGS_VAL_OFFSET},
                 {"num_sems", static_cast<uint32_t>(sem_ids.size())}},
            // Implicit sync would take a transaction id per DFB out of a pool of 24, and nothing is
            // ever pushed through these buffers for it to synchronize.
            .hw_config = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true},
            .advanced_options =
                KernelAdvancedOptions{
                    .num_runtime_varargs = static_cast<uint32_t>(unique_rt_args.size()),
                    .num_common_runtime_varargs = static_cast<uint32_t>(common_rt_args.size()),
                },
        };

        Group<KernelSpec> kernels{kernel_spec};
        Group<KernelSpecName> work_unit_kernels{name};
        const KernelSpecName consumer_name{"dfb_consumer"};
        if (!consumer_bindings.empty()) {
            kernels.push_back(KernelSpec{
                .unique_id = consumer_name,
                .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp"},
                .num_threads = 1,
                .dfb_bindings = consumer_bindings,
                .hw_config = ComputeHardwareConfig{ComputeGen2Config{}},
            });
            work_unit_kernels.push_back(consumer_name);
        }

        ProgramSpec spec{
            .name = "random_program",
            .kernels = kernels,
            .dataflow_buffers = dataflow_buffers,
            .semaphores = semaphores,
            .work_units = {WorkUnitSpec{
                .name = "work_unit",
                .kernels = work_unit_kernels,
                .target_nodes = cores,
            }},
        };
        Program program = MakeProgramFromSpec(*device_, spec);

        if (!unique_rt_args.empty() || !common_rt_args.empty()) {
            ProgramRunArgs run_args;
            ProgramRunArgs::KernelRunArgs kernel_run_args{.kernel = name};
            for (const CoreRange& core_range : cores.ranges()) {
                for (const CoreCoord& core_coord : core_range) {
                    kernel_run_args.advanced_options.runtime_varargs[core_coord] = unique_rt_args;
                }
            }
            kernel_run_args.advanced_options.common_runtime_varargs = common_rt_args;
            run_args.kernel_run_args.push_back(std::move(kernel_run_args));
            SetProgramRunArgs(program, run_args);
        }
        return program;
    }

    KernelHandle create_kernel(
        Program& program,
        const CoreRangeSet& cores,
        const bool create_eth_config,
        const uint32_t num_sems,
        const uint32_t num_cbs,
        const uint32_t num_unique_rt_args,
        const uint32_t num_common_rt_args,
        const uint32_t min_kernel_size_bytes = MIN_KERNEL_SIZE_BYTES,
        const uint32_t max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES,
        const uint32_t min_kernel_runtime_microseconds = MIN_KERNEL_RUNTIME_MICROSECONDS,
        const uint32_t max_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS) {
        std::vector<uint32_t> compile_args = {
            num_unique_rt_args,
            num_common_rt_args,
            UNIQUE_RUNTIME_ARGS_VAL_OFFSET,
            COMMON_RUNTIME_ARGS_VAL_OFFSET,
            num_sems,
            SEM_VAL,
            num_cbs};

        uint32_t divisible_by;
        if (create_eth_config) {
            divisible_by = 4;
        } else {
            divisible_by = 1;
        }

        const uint32_t kernel_size_bytes =
            this->generate_random_num(min_kernel_size_bytes, max_kernel_size_bytes, divisible_by);
        const uint32_t kernel_runtime_microseconds =
            this->generate_random_num(min_kernel_runtime_microseconds, max_kernel_runtime_microseconds);

        const std::map<std::string, std::string> defines = {
            {"KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes)},
            {"KERNEL_RUNTIME_MICROSECONDS", std::to_string(kernel_runtime_microseconds)}};

        std::variant<DataMovementConfig, EthernetConfig> config;
        if (create_eth_config) {
            compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
            const auto proc = this->get_processor(true);
            config = EthernetConfig{
                .noc = static_cast<NOC>(proc), .processor = proc, .compile_args = compile_args, .defines = defines};
            eth_test_common::set_arch_specific_eth_config(std::get<EthernetConfig>(config));
        } else {
            compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
            config = DataMovementConfig{
                .processor = this->get_processor(false), .compile_args = compile_args, .defines = defines};
        }

        KernelHandle kernel_id = std::visit(
            [&](const auto& cfg) -> KernelHandle {
                return CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/"
                    "dispatcher_kernel_size_and_runtime.cpp",
                    cores,
                    cfg);
            },
            config);
        return kernel_id;
    }

    // Generates a random number within the given bounds (inclusive) that is divisible by divisible_by
    uint32_t generate_random_num(const uint32_t min, const uint32_t max, const uint32_t divisible_by = 1) {
        TT_FATAL(max >= min, "max: {}, min: {} - max must be >= min", max, min);

        const uint32_t adjusted_min = ((min + divisible_by - 1) / divisible_by) * divisible_by;
        const uint32_t adjusted_max = (max / divisible_by) * divisible_by;

        TT_FATAL(
            adjusted_min <= adjusted_max,
            "There are no numbers between {} and {} that are divisible by {}",
            min,
            max,
            divisible_by);

        return adjusted_min + ((rand() % ((adjusted_max - adjusted_min) / divisible_by + 1)) * divisible_by);
    }

    DataMovementProcessor get_processor(bool is_eth) {
        int max_index = 1;
        int num = 0;

        if (is_eth) {
            max_index = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH) -
                        1;
        }
        num = this->generate_random_num(0, max_index);
        DataMovementProcessor processor;
        if (num == 0) {
            processor = DataMovementProcessor::RISCV_0;
        } else {
            processor = DataMovementProcessor::RISCV_1;
        }
        return processor;
    }

    CoreRangeSet get_cores(const CoreType core_type) {
        CoreRangeSet all_cores;
        if (core_type == CoreType::WORKER) {
            CoreCoord worker_grid_size = device_->compute_with_storage_grid_size();
            all_cores = CoreRangeSet({CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1})});
        } else {
            TT_FATAL(core_type == CoreType::ETH, "Unsupported core type");
            std::set<CoreRange> core_ranges;
            const std::unordered_set<CoreCoord> active_eth_cores =
                this->device_->get_devices()[0]->get_active_ethernet_cores(true);
            TT_FATAL(!active_eth_cores.empty(), "No active ethernet cores detected");
            for (CoreCoord eth_core : active_eth_cores) {
                core_ranges.emplace(eth_core);
            }
            all_cores = CoreRangeSet(core_ranges);
        }
        CoreRangeSet empty_crs;
        all_cores = empty_crs.merge(all_cores);

        CoreRangeSet cores;
        const uint32_t num = 0;  // this->generate_random_num(0, 2);
        switch (num) {
            case 0: cores = all_cores; break;
            case 1: cores = this->generate_subset_of_cores(all_cores, 2); break;
            case 2: cores = this->generate_subset_of_cores(all_cores, 4); break;
            default: TT_THROW("Invalid random core selection value {}", num);
        }

        TT_FATAL(!cores.empty(), "Generated cores cannot be empty");
        return cores;
    }

    CoreRangeSet generate_subset_of_cores(const CoreRangeSet& cores, const uint32_t resulting_ratio_of_cores) {
        std::set<CoreRange> cores_subset;
        const uint32_t num_cores = cores.num_cores();
        const uint32_t num_cores_to_include_in_subset =
            std::max(static_cast<uint32_t>(1), num_cores / resulting_ratio_of_cores);
        uint32_t num_cores_added = 0;
        while (num_cores_added != num_cores_to_include_in_subset) {
            for (CoreRange cr : cores.ranges()) {
                for (CoreCoord core : cr) {
                    const uint32_t random_num = this->generate_random_num(1, resulting_ratio_of_cores);
                    if (random_num == 1 && num_cores_added != num_cores_to_include_in_subset) {
                        cores_subset.emplace(core);
                        num_cores_added += 1;
                    }
                }
            }
        }

        CoreRangeSet empty_crs;
        CoreRangeSet resulting_cores = empty_crs.merge(cores_subset);
        return resulting_cores;
    }
};

class UnitMeshRandomProgramTraceFixture : virtual public UnitMeshRandomProgramFixture,
                                          virtual public UnitMeshCQSingleCardTraceFixture {
protected:
    static const uint32_t NUM_TRACE_ITERATIONS = 50;
    distributed::MeshWorkload workloads[NUM_WORKLOADS];

    void SetUp() override {
        UnitMeshCQSingleCardTraceFixture::SetUp();
        if (!::testing::Test::IsSkipped()) {
            // Parent may have skipped
            this->device_ = this->devices_[0];
            this->initialize_seed();
        }
    }

    distributed::MeshTraceId trace_programs() {
        log_info(tt::LogTest, "Starting trace capture");
        const distributed::MeshTraceId trace_id = this->capture_trace();
        log_info(tt::LogTest, "Trace capture complete, starting trace replay (50 iterations)");
        this->run_trace(trace_id);
        log_info(tt::LogTest, "Trace replay complete");
        return trace_id;
    }

private:
    distributed::MeshTraceId capture_trace() {
        auto& mesh_command_queue = this->device_->mesh_command_queue();

        // Create a zero coordinate and range for the device
        distributed::MeshCoordinate zero_coord =
            distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        const distributed::MeshTraceId trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
        for (auto& workload : this->workloads) {
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
        }
        log_info(tt::LogTest, "All workloads enqueued in trace, calling end_mesh_trace");
        this->device_->end_mesh_trace(mesh_command_queue, trace_id);
        log_info(tt::LogTest, "end_mesh_trace complete");
        return trace_id;
    }

    void run_trace(const distributed::MeshTraceId trace_id) {
        auto& mesh_command_queue = this->device_->mesh_command_queue();
        for (uint32_t i = 0; i < NUM_TRACE_ITERATIONS; i++) {
            if (i % 10 == 0) {
                log_info(tt::LogTest, "Replaying trace iteration {}", i);
            }
            this->device_->replay_mesh_trace(mesh_command_queue, trace_id, false);
        }
        log_info(tt::LogTest, "All trace iterations enqueued, calling Finish");
    }
};

}  // namespace tt::tt_metal
