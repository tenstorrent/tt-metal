// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt_metal/tt_metal/context/mimir_contract.hpp"

namespace tt::tt_metal::keraunos_session {

// CCE0 of one HSIO tile of an mmk Keraunos chiplet.
struct KeraunosCore {
    CoreCoord logical;
    uint32_t hsio = 0;
};

// A device word the host polls after launch: a kernel's status at flat SRAM @p offset of @p core.
struct StatusWord {
    KeraunosCore core;
    uint32_t offset = 0;
};

// A unit MeshDevice (device 0) on an mmk package, with host access to the CCE0 SRAM of its HSIO tiles through the
// IDLE_ETH view. Not thread-safe. Opening costs about 90 s on the emulator, so a test binary keeps one open.
class Session {
public:
    // Opens the device. A test's region is [base(), base() + region_bytes) of each core's SRAM; TT_FATAL when it does
    // not fit IDLE_ETH's unreserved region, or when the descriptor has no Keraunos tile_stride.
    explicit Session(uint32_t region_bytes) {
        mesh_ = distributed::MeshDevice::create_unit_mesh(0);
        device_ = &slow_dispatch::physical_device_from_unit_mesh(*mesh_);
        const auto& hal = MetalContext::instance().hal();
        sram_noc_base_ = hal.get_dev_noc_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::BASE);
        // Above the Ethernet-link slot and router regions, which sit inside IDLE_ETH's unreserved region.
        base_ = (MEM_IERISC_ROUTER_UNRESERVED_BASE + 63) & ~63u;
        TT_FATAL(
            base_ >= hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED) &&
                base_ + region_bytes <= MEM_IERISC_L1_SIZE,
            "Test region 0x{:x} + 0x{:x} does not fit the IDLE_ETH unreserved region",
            base_,
            region_bytes);
        const auto& chiplet = mimir_contract::soc_desc_of(device_).get_arch_descriptor().get_keraunos();
        TT_FATAL(chiplet.tile_stride.has_value(), "mmk descriptor has no keraunos tile_stride");
        tile_stride_ = *chiplet.tile_stride;
    }

    // Flat SRAM offset of the test region: 64-byte aligned, the same on every core.
    uint32_t base() const { return base_; }
    // Bytes between consecutive HSIO tiles in the chiplet address map.
    uint64_t tile_stride() const { return tile_stride_; }

    // CCE0 of HSIO tile @p hsio, or nullopt when the descriptor does not list it.
    std::optional<KeraunosCore> core(uint32_t hsio) const {
        const auto& soc = mimir_contract::soc_desc_of(device_);
        for (const auto& c : soc.get_arch_descriptor().get_keraunos().cores) {
            if (c.hsio == hsio && c.cce == 0) {
                const CoreCoord logical = soc.get_logical_ethernet_core_from_physical({c.coord.x, c.coord.y});
                if (!device_->ethernet_cores().contains(logical)) {
                    return std::nullopt;
                }
                return KeraunosCore{logical, hsio};
            }
        }
        return std::nullopt;
    }

    // Host access to flat SRAM @p offset of @p core; blocking.
    void write(const KeraunosCore& core, uint32_t offset, const std::vector<uint32_t>& words) {
        mimir_contract::write_eth_sram(
            device_, core.logical, sram_noc_base_ + offset, words.data(), words.size() * sizeof(uint32_t));
    }
    std::vector<uint32_t> read(const KeraunosCore& core, uint32_t offset, uint32_t num_words) {
        std::vector<uint32_t> words(num_words, 0);
        mimir_contract::read_eth_sram(
            device_, core.logical, sram_noc_base_ + offset, words.data(), num_words * sizeof(uint32_t));
        return words;
    }
    template <typename T>
    T read_struct(const KeraunosCore& core, uint32_t offset) {
        T value{};
        mimir_contract::read_eth_sram(device_, core.logical, sram_noc_base_ + offset, &value, sizeof(value));
        return value;
    }

    // eth_cce_send.h's EthLink: the link peer's HSIO tile, then the valid word.
    void write_eth_link(const KeraunosCore& core, const KeraunosCore& peer) {
        write(core, MEM_IERISC_FABRIC_ETH_LINK_BASE, {peer.hsio, MEM_IERISC_FABRIC_ETH_LINK_VALID, 0, 0});
    }

    // Launches @p program without blocking, polls each watched word in turn while it reads 0 or @p running_status,
    // then waits for the program. When @p timeout passes first the kernels are hung: @p dump prints their state and
    // the process exits with status 3, because nothing after it could reach the device.
    void run(
        Program& program,
        const std::vector<StatusWord>& watched,
        uint32_t running_status,
        std::chrono::seconds timeout,
        const std::function<void()>& dump) {
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/false);

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (const StatusWord& w : watched) {
            for (uint32_t status = read(w.core, w.offset, 1)[0]; status == running_status || status == 0;
                 status = read(w.core, w.offset, 1)[0]) {
                if (std::chrono::steady_clock::now() > deadline) {
                    fmt::print(stderr, "Keraunos: host timeout; device state follows\n");
                    dump();
                    fmt::print(stderr, "Exiting: the kernels are stuck. Reset the emulator job before the next run.\n");
                    std::fflush(stderr);
                    std::_Exit(3);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        distributed::Finish(mesh_->mesh_command_queue());
    }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_;
    IDevice* device_ = nullptr;
    uint64_t sram_noc_base_ = 0;
    uint32_t base_ = 0;
    uint64_t tile_stride_ = 0;
};

}  // namespace tt::tt_metal::keraunos_session
