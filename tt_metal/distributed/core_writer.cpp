// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/core_writer.hpp"

#include <stdexcept>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/io_window/io_window.hpp>

#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

CoreWriter::CoreWriter(
    const tt::Cluster& cluster,
    ChipId chip,
    const tt_metal::CoreCoord& virtual_core,
    uint64_t end,
    uint64_t l1_offset,
    std::string_view owner) :
    cluster_(cluster), chip_(chip), virtual_core_(virtual_core), l1_offset_(l1_offset) {
    // A window of our own only on Blackhole silicon: on Wormhole B0 the device address space isn't fully mapped this
    // way, and mock and emulated devices have no device to map. The window holds one of the device's TLBs until the
    // writer is destroyed; when none is free, writes go through cluster.write_core, which holds none.
    //
    // Posted, on a static write VC: a Strict window would wait for each write's ack, one write per round trip (the
    // strict MMIO rate into Tensix L1, #64622). Posted writes are ordered only by the VC: a packet on a dynamic VC
    // may switch VCs at each router (and a DRAM core gives each VC its own AXI ID), so a count could overtake the
    // data it counts; on one static VC, writes to the core arrive in issue order.
    if (cluster.is_mock_or_emulated() || cluster.arch() != tt::ARCH::BLACKHOLE) {
        return;
    }
    std::unique_ptr<tt::umd::IoWindow> window;
    try {
        window = cluster.get_driver()->create_io_window(
            chip,
            cluster.get_soc_desc(chip).get_coord_at(virtual_core, tt::CoordSystem::TRANSLATED),
            /*addr=*/l1_offset,
            /*host=*/{},
            tt::umd::IoOrdering::Posted,
            /*core_end=*/std::nullopt,
            tt::umd::WindowFlags::UnicastWrite);
    } catch (const std::runtime_error& e) {
        log_warning(
            LogMetal,
            "{}: no TLB window for core {} on device {}; its writes go through the driver, which is slower: {}",
            owner,
            virtual_core.str(),
            chip,
            e.what());
    }
    // The window is addressed from its anchor (the start of the core's L1), so a local address is its offset into it.
    if (window != nullptr && end <= window->get_size()) {
        window_ = std::move(window);
    }
}

CoreWriter::~CoreWriter() = default;

void CoreWriter::write(const void* data, uint32_t num_bytes, uint64_t addr) const {
    if (window_ != nullptr) {
        window_->write_block(addr, data, num_bytes);
    } else {
        cluster_.write_core(data, num_bytes, tt_cxy_pair(chip_, virtual_core_), addr + l1_offset_);
    }
}

}  // namespace tt::tt_metal::distributed
