// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dram_core_kernels.hpp>

#include <algorithm>
#include <thread>
#include <utility>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/experimental/dram_core_kernels_internal.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/impl/device/device_impl.hpp"

namespace tt::tt_metal::experimental {

namespace {

// (the concrete device: experimental::Device is a namespace here)
using ConcreteDevice = ::tt::tt_metal::Device;

MetalContext& context_of(const distributed::MeshDevice& mesh_device) {
    return MetalContext::instance(mesh_device.impl().get_context_id());
}

ConcreteDevice& local_device(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord) {
    TT_FATAL(
        mesh_device.impl().is_local(mesh_coord),
        "DRAM core kernels: MeshCoordinate {} maps to a device this rank does not drive",
        mesh_coord);
    ConcreteDevice* device = mesh_device.impl().get_device(mesh_coord);
    TT_FATAL(device != nullptr, "DRAM core kernels: no device at MeshCoordinate {}", mesh_coord);
    return *device;
}

// The core's NoC-visible L1 address on the host side (the DRAM-core L1 sits behind the NoC offset).
tt_cxy_pair host_target(const ConcreteDevice& device, const CoreCoord& logical_core) {
    return tt_cxy_pair(device.id(), device.virtual_core_from_logical_core(logical_core, tt::CoreType::DRAM));
}

}  // namespace

bool AreDramCoreKernelsSupported(const distributed::MeshDevice& mesh_device) {
    const auto& metal = context_of(mesh_device);
    return !metal.rtoptions().get_streaming_profiler_enabled() &&
           metal.hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

DramCoreL1Region GetDramCoreKernelL1Region(const distributed::MeshDevice& mesh_device) {
    TT_FATAL(AreDramCoreKernelsSupported(mesh_device), "DRAM core kernels are not supported on this mesh device");
    const auto& arena = mesh_device.impl().drisc_l1_arena();
    return DramCoreL1Region{
        .base = static_cast<uint32_t>(arena.kernel_working_region_base()), .size = arena.kernel_working_region_size()};
}

std::vector<DramCoreInfo> GetDramCores(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, uint32_t bank) {
    const ConcreteDevice& device = local_device(mesh_device, mesh_coord);
    const auto& soc = context_of(mesh_device).get_cluster().get_soc_desc(device.id());
    TT_FATAL(bank < soc.dram_bank_endpoint_coords.size(), "GetDramCores: bank {} out of range", bank);
    const auto usable = soc.get_metal_dram_cores(tt::CoordSystem::LOGICAL);
    std::vector<DramCoreInfo> cores;
    for (uint32_t i = 0; i < soc.dram_bank_endpoint_coords[bank].size(); ++i) {
        const CoreCoord logical(bank, i);
        const CoreCoord translated = device.virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
        const auto noc0 = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED),
            tt::CoordSystem::NOC0);
        // MPFE P0 is tied off: the bank's subchannels 0..2 enter through P1..P3
        const auto subchannel = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED),
            tt::CoordSystem::LOGICAL);
        cores.push_back(DramCoreInfo{
            .logical_core = logical,
            .virtual_core = translated,
            .noc0_core = CoreCoord(noc0.x, noc0.y),
            .endpoint_noc_mask = soc.get_dram_endpoint_noc_mask(translated),
            .mpfe_port = static_cast<uint32_t>(subchannel.y) + 1,
            .usable = std::find(usable.begin(), usable.end(), logical) != usable.end()});
    }
    return cores;
}

CoreCoord GetWorkerNoc0Core(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_core) {
    const ConcreteDevice& device = local_device(mesh_device, mesh_coord);
    return context_of(mesh_device)
        .get_cluster()
        .get_soc_desc(device.id())
        .get_physical_tensix_core_from_logical(logical_core);
}

void DramCoreKernelsState::wait_done() {
    if (done) {
        return;
    }
    for (size_t d = 0; d < devices.size(); ++d) {
        slow_dispatch::WaitProgramDone(*devices[d], *programs[d]);
    }
    // (released while the devices are open)
    programs.clear();
    devices.clear();
    done = true;
}

namespace detail {
void StopDramCoreKernelsAtClose(std::vector<std::shared_ptr<DramCoreKernelsState>>& launches) {
    for (auto& state : launches) {
        if (state->done) {
            continue;
        }
        log_debug(tt::LogMetal, "MeshDevice close: stopping resident DRAM-core kernels");
        if (state->on_close) {
            state->on_close();
        }
        state->wait_done();
    }
    launches.clear();
}
}  // namespace detail

DramCoreKernels::DramCoreKernels(std::shared_ptr<DramCoreKernelsState> state) : state_(std::move(state)) {}

DramCoreKernels::~DramCoreKernels() = default;

bool DramCoreKernels::running() const { return !state_->done; }

void DramCoreKernels::wait_done() { state_->wait_done(); }

std::unique_ptr<DramCoreKernels> LaunchDramCoreKernels(
    distributed::MeshDevice& mesh_device,
    const std::function<std::vector<DramCoreKernelConfig>(const distributed::MeshCoordinate&)>& kernels,
    std::function<void()> on_close) {
    TT_FATAL(
        AreDramCoreKernelsSupported(mesh_device),
        "DRAM core kernels are not supported on this mesh device: programmable DRAM cores are unavailable (they "
        "auto-enable on Blackhole with firmware >= 19.12.0.0) or the streaming profiler holds them");
    auto state = std::make_shared<DramCoreKernelsState>();
    state->on_close = std::move(on_close);
    for (ConcreteDevice* local : mesh_device.impl().get_devices()) {
        ConcreteDevice& device = *local;
        const distributed::MeshCoordinate coord = mesh_device.get_view().find_device(device.id());
        auto program = std::make_unique<Program>();
        std::vector<CoreCoord> used;
        for (const auto& k : kernels(coord)) {
            TT_FATAL(
                std::find(used.begin(), used.end(), k.core) == used.end(),
                "LaunchDramCoreKernels: two kernels on DRAM core {}",
                k.core.str());
            used.push_back(k.core);
            const KernelHandle handle = CreateKernel(
                *program,
                k.kernel_file,
                k.core,
                DramConfig{.noc = k.noc, .compile_args = k.compile_args, .defines = k.defines});
            SetRuntimeArgs(*program, handle, k.core, k.runtime_args);
        }
        if (used.empty()) {
            continue;
        }
        slow_dispatch::LaunchProgramAsync(device, *program, /*force_slow_dispatch=*/true);
        state->devices.push_back(&device);
        state->programs.push_back(std::move(program));
    }
    auto& launches = mesh_device.impl().dram_core_kernels();
    std::erase_if(launches, [](const auto& s) { return s->done; });
    launches.push_back(state);
    return std::make_unique<DramCoreKernels>(std::move(state));
}

void WriteDramCoreL1(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const uint32_t> words) {
    TT_FATAL(address % 4 == 0, "WriteDramCoreL1: address {:#x} is not 4 B aligned", address);
    const auto& metal = context_of(mesh_device);
    const uint64_t noc_offset = metal.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    for (const ConcreteDevice* device : mesh_device.impl().get_devices()) {
        metal.get_cluster().write_core(
            words.data(), words.size_bytes(), host_target(*device, logical_core), noc_offset + address);
        metal.get_cluster().l1_barrier(device->id());
    }
}

std::vector<uint32_t> ReadDramCoreL1(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t num_words) {
    TT_FATAL(address % 4 == 0, "ReadDramCoreL1: address {:#x} is not 4 B aligned", address);
    const ConcreteDevice& device = local_device(mesh_device, mesh_coord);
    const auto& metal = context_of(mesh_device);
    const uint64_t noc_offset = metal.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    std::vector<uint32_t> words(num_words, 0);
    metal.get_cluster().read_core(
        words.data(), num_words * sizeof(uint32_t), host_target(device, logical_core), noc_offset + address);
    return words;
}

bool WaitForDramCoreL1Word(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t expected,
    std::chrono::milliseconds timeout) {
    TT_FATAL(address % 4 == 0, "WaitForDramCoreL1Word: address {:#x} is not 4 B aligned", address);
    const auto& metal = context_of(mesh_device);
    if (metal.get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return true;
    }
    const uint64_t noc_offset = metal.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const auto start = std::chrono::steady_clock::now();
    for (const ConcreteDevice* device : mesh_device.impl().get_devices()) {
        uint32_t word = ~expected;
        while (true) {
            metal.get_cluster().read_core(
                &word, sizeof(word), host_target(*device, logical_core), noc_offset + address);
            if (word == expected) {
                break;
            }
            if (std::chrono::steady_clock::now() - start > timeout) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return true;
}

}  // namespace tt::tt_metal::experimental
