// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "impl/context/metal_context.hpp"
#include "sub_device.hpp"

namespace tt::tt_metal {

inline std::tuple<Program, CoreCoord, GlobalSemaphore> create_single_sync_program(
    IDevice* device, const SubDevice& sub_device) {
    auto syncer_coord = sub_device.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto global_sem = CreateGlobalSemaphore(device, sub_device.cores(HalProgrammableCoreType::TENSIX), INVALID);

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem.address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);
    return {std::move(syncer_program), std::move(syncer_coord), std::move(global_sem)};
}

inline std::tuple<Program, Program, Program, GlobalSemaphore> create_basic_sync_program(
    IDevice* device, const SubDevice& sub_device_1, const SubDevice& sub_device_2) {
    auto waiter_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto waiter_core_physical = device->worker_core_from_logical_core(waiter_coord);
    auto incrementer_cores = sub_device_1.cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = CreateGlobalSemaphore(device, all_cores, INVALID);

    Program waiter_program = CreateProgram();
    auto waiter_kernel = CreateKernel(
        waiter_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/persistent_waiter.cpp",
        waiter_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 4> waiter_rt_args = {
        global_sem.address(), incrementer_cores.num_cores(), syncer_core_physical.x, syncer_core_physical.y};
    SetRuntimeArgs(waiter_program, waiter_kernel, waiter_core, waiter_rt_args);

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem.address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);

    Program incrementer_program = CreateProgram();
    auto incrementer_kernel = CreateKernel(
        incrementer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/incrementer.cpp",
        incrementer_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    std::array<uint32_t, 3> incrementer_rt_args = {
        global_sem.address(), waiter_core_physical.x, waiter_core_physical.y};
    SetRuntimeArgs(incrementer_program, incrementer_kernel, incrementer_cores, incrementer_rt_args);
    waiter_program.set_runtime_id(1);
    syncer_program.set_runtime_id(2);
    incrementer_program.set_runtime_id(3);
    return {
        std::move(waiter_program), std::move(syncer_program), std::move(incrementer_program), std::move(global_sem)};
}

inline std::tuple<Program, Program, Program, GlobalSemaphore> create_basic_eth_sync_program(
    IDevice* device, const SubDevice& sub_device_1, const SubDevice& sub_device_2, DataMovementProcessor dm_processor) {
    auto waiter_coord = sub_device_2.cores(HalProgrammableCoreType::ACTIVE_ETH).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto waiter_core_physical = device->ethernet_core_from_logical_core(waiter_coord);
    auto tensix_waiter_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto tensix_waiter_core = CoreRangeSet(CoreRange(tensix_waiter_coord, tensix_waiter_coord));
    auto tensix_waiter_core_physical = device->worker_core_from_logical_core(tensix_waiter_coord);
    auto incrementer_cores = sub_device_1.cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = tensix_waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = CreateGlobalSemaphore(device, all_cores, INVALID);

    Program waiter_program = CreateProgram();
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

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem.address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);

    Program incrementer_program = CreateProgram();
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
