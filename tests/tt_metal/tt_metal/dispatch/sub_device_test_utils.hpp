// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "sub_device.hpp"
#include "impl/sub_device/sub_device_impl.hpp"

namespace tt::tt_metal {

// One data-movement kernel on every core of `cores`, each core given the same named runtime args. The names are
// declared in order, so kernels that read their args positionally with get_arg_val see them in that order. On
// Wormhole and Blackhole the kernel runs on `processor` with the NoC of the same index.
inline Program create_single_kernel_program(
    distributed::MeshDevice& mesh_device,
    const std::string& name,
    const std::string& kernel_path,
    const CoreRangeSet& cores,
    DataMovementProcessor processor,
    std::initializer_list<std::pair<std::string, ArgValue>> runtime_args) {
    const KernelSpecName kernel_name{name};
    KernelSpec::RuntimeArgSchema runtime_arg_schema;
    for (const auto& [arg_name, arg_value] : runtime_args) {
        runtime_arg_schema.runtime_arg_names.push_back(arg_name);
    }
    const ProgramSpec spec{
        .name = name,
        .kernels = {KernelSpec{
            .unique_id = kernel_name,
            .source = kernel_path,
            .num_threads = 1,
            .runtime_arg_schema = std::move(runtime_arg_schema),
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = processor, .noc = static_cast<NOC>(processor)},
                }}},
        .work_units = {WorkUnitSpec{.name = name, .kernels = {kernel_name}, .target_nodes = cores}},
    };
    Program program = Program(mesh_device, spec);
    ProgramRunArgs::KernelRunArgs kernel_run_args{.kernel = kernel_name};
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        AddRuntimeArgsForNode(kernel_run_args.runtime_arg_values, core, runtime_args);
    }
    program.set_program_run_args(ProgramRunArgs{.kernel_run_args = {std::move(kernel_run_args)}});
    return program;
}

inline CoreRange worker_grid(const distributed::MeshDevice& mesh_device) {
    const CoreCoord worker_grid_size = mesh_device.compute_with_storage_grid_size();
    return CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
}

// The desired core sets where the worker grid holds both, otherwise the first and the second worker.
// The caller must have checked that the device has at least two workers.
inline std::pair<CoreRangeSet, CoreRangeSet> two_disjoint_core_sets(
    const distributed::MeshDevice& mesh_device, const CoreRangeSet& desired_first, const CoreRangeSet& desired_second) {
    const CoreRange workers = worker_grid(mesh_device);
    if (workers.contains(desired_first) && workers.contains(desired_second)) {
        return {desired_first, desired_second};
    }
    const std::vector<CoreCoord> first_two_workers = corerange_to_cores(CoreRangeSet(workers), 2, true);
    return {CoreRangeSet(CoreRange(first_two_workers.at(0))), CoreRangeSet(CoreRange(first_two_workers.at(1)))};
}

inline std::tuple<Program, CoreCoord, GlobalSemaphore> create_single_sync_program(
    distributed::MeshDevice* device, const SubDevice& sub_device) {
    auto syncer_coord = sub_device.impl()->cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto global_sem = GlobalSemaphore(*device, sub_device.impl()->cores(HalProgrammableCoreType::TENSIX), INVALID);

    Program syncer_program = create_single_kernel_program(
        *device,
        "syncer",
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementProcessor::RISCV_0,
        {{"sem_addr", global_sem.address()}});
    return {std::move(syncer_program), syncer_coord, std::move(global_sem)};
}

inline std::tuple<Program, Program, Program, GlobalSemaphore> create_basic_sync_program(
    distributed::MeshDevice* device, const SubDevice& sub_device_1, const SubDevice& sub_device_2) {
    auto waiter_coord = sub_device_2.impl()->cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto waiter_core_physical = device->worker_core_from_logical_core(waiter_coord);
    auto incrementer_cores = sub_device_1.impl()->cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = GlobalSemaphore(*device, all_cores, INVALID);

    Program waiter_program = create_single_kernel_program(
        *device,
        "persistent_waiter",
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/persistent_waiter.cpp",
        waiter_core,
        DataMovementProcessor::RISCV_0,
        {{"sem_addr", global_sem.address()},
         {"num_inc", incrementer_cores.num_cores()},
         {"sync_core_x", syncer_core_physical.x},
         {"sync_core_y", syncer_core_physical.y}});
    Program syncer_program = create_single_kernel_program(
        *device,
        "syncer",
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementProcessor::RISCV_0,
        {{"sem_addr", global_sem.address()}});
    Program incrementer_program = create_single_kernel_program(
        *device,
        "incrementer",
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/incrementer.cpp",
        incrementer_cores,
        DataMovementProcessor::RISCV_1,
        {{"sem_addr", global_sem.address()},
         {"waiter_core_x", waiter_core_physical.x},
         {"waiter_core_y", waiter_core_physical.y}});
    waiter_program.set_runtime_id(1);
    syncer_program.set_runtime_id(2);
    incrementer_program.set_runtime_id(3);
    return {
        std::move(waiter_program), std::move(syncer_program), std::move(incrementer_program), std::move(global_sem)};
}

inline std::tuple<Program, Program, Program, GlobalSemaphore> create_basic_eth_sync_program(
    distributed::MeshDevice* device, const SubDevice& sub_device_1, const SubDevice& sub_device_2, DataMovementProcessor dm_processor) {
    auto waiter_coord = sub_device_2.impl()->cores(HalProgrammableCoreType::ACTIVE_ETH).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto tensix_waiter_coord = sub_device_2.impl()->cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto tensix_waiter_core = CoreRangeSet(CoreRange(tensix_waiter_coord, tensix_waiter_coord));
    auto tensix_waiter_core_physical = device->worker_core_from_logical_core(tensix_waiter_coord);
    auto incrementer_cores = sub_device_1.impl()->cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = tensix_waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = GlobalSemaphore(*device, all_cores, INVALID);

    Program waiter_program = CreateProgram(*device);
    auto waiter_kernel = CreateKernel(
        waiter_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/persistent_remote_waiter.cpp",
        waiter_core,
        EthernetConfig{.noc = NOC::RISCV_0_default, .processor = dm_processor});
    std::array<uint32_t, 7> waiter_rt_args = {
        global_sem.address(),
        incrementer_cores.num_cores(),
        syncer_core_physical.x,
        syncer_core_physical.y,
        tensix_waiter_core_physical.x,
        tensix_waiter_core_physical.y,
        MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED)
    };
    SetRuntimeArgs(waiter_program, waiter_kernel, waiter_core, waiter_rt_args);

    Program syncer_program = CreateProgram(*device);
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem.address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);

    Program incrementer_program = CreateProgram(*device);
    auto incrementer_kernel = CreateKernel(
        incrementer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/incrementer.cpp",
        incrementer_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    std::array<uint32_t, 3> incrementer_rt_args = {
        global_sem.address(), tensix_waiter_core_physical.x, tensix_waiter_core_physical.y};
    SetRuntimeArgs(incrementer_program, incrementer_kernel, incrementer_cores, incrementer_rt_args);
    waiter_program.set_runtime_id(1);
    syncer_program.set_runtime_id(2);
    incrementer_program.set_runtime_id(3);
    return {
        std::move(waiter_program), std::move(syncer_program), std::move(incrementer_program), std::move(global_sem)};
}

} // namespace tt::tt_metal
