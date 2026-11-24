// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/host_rdma/host_rdma.hpp"

#include <algorithm>
#include <chrono>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <hostdevcommon/fabric_common.h>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/core_writer.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"

namespace tt::tt_fabric::host_rdma {

namespace {

constexpr std::chrono::seconds kServiceTimeout{10};

namespace sabi = tt::tt_fabric::host_rdma::service_abi;

void launch_service_kernel(
    tt_metal::distributed::MeshDevice& mesh_device, tt_metal::CoreCoord logical_core, uint32_t layout_addr) {
    tt_metal::Program program = tt_metal::CreateProgram();
    const auto kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/host_rdma_service.cpp",
        logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    tt_metal::SetRuntimeArgs(program, kernel, logical_core, {layout_addr});
    tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
}

std::unique_ptr<tt_metal::Program> launch_service_kernel_on_dram(
    tt_metal::distributed::MeshDevice& mesh_device,
    tt_metal::CoreCoord logical_core,
    tt_metal::NOC noc,
    uint32_t layout_addr) {
    auto program = std::make_unique<tt_metal::Program>(tt_metal::CreateProgram(mesh_device));
    // A DRAM-core kernel is not given the fabric defines (the packet header's routing mode) by its build.
    const auto kernel = tt_metal::CreateKernel(
        *program,
        "tt_metal/fabric/impl/kernels/host_rdma_service.cpp",
        logical_core,
        tt_metal::DramConfig{
            .noc = noc,
            .defines = mesh_device.impl().metal_context().get_control_plane().get_fabric_kernel_defines(),
            .opt_level = tt_metal::KernelBuildOptLevel::O2});
    tt_metal::SetRuntimeArgs(*program, kernel, logical_core, {layout_addr});
    auto* device = mesh_device.get_devices().front();
    program->impl().compile(device, /*force_slow_dispatch=*/true);
    tt_metal::slow_dispatch::WriteRuntimeArgsToDevice(*device, *program, /*force_slow_dispatch=*/true);
    tt_metal::slow_dispatch::LaunchProgramAsync(*device, *program, /*force_slow_dispatch=*/true);
    return program;
}

sabi::Channel inactive_channel_entry(const fabric_connection_info_t& router, uint32_t max_payload) {
    sabi::Channel entry{};
    entry.max_payload = tt::round_down(max_payload, sabi::NOC_ALIGNMENT);
    entry.router_x = router.edm_noc_x;
    entry.router_y = router.edm_noc_y;
    entry.slots_base = router.edm_buffer_base_addr;
    entry.num_slots = router.num_buffers_per_channel;
    entry.slot_bytes = router.buffer_size_bytes;
    entry.free_slots_stream = router.worker_free_slots_stream_id;
    entry.connection_semaphore_addr = router.edm_connection_handshake_addr;
    entry.connection_info_addr = router.edm_worker_location_info_addr;
    entry.producer_cursor_addr = router.buffer_index_semaphore_id;
    return entry;
}

}  // namespace

std::shared_ptr<HostRdmaService> HostRdmaService::create(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    tt_metal::CoreCoord logical_core,
    ServiceCore core) {
    TT_FATAL(mesh_device != nullptr, "The host_rdma service needs a mesh device");
    return std::shared_ptr<HostRdmaService>(new HostRdmaService(std::move(mesh_device), logical_core, core));
}

HostRdmaService::HostRdmaService(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    tt_metal::CoreCoord logical_core,
    ServiceCore core) :
    mesh_device_(std::move(mesh_device)),
    context_(mesh_device_->impl().metal_context()),
    cluster_(context_.get_cluster()),
    core_(core) {
    const auto& hal = context_.hal();
    const auto& fabric_context = context_.get_control_plane().get_fabric_context();
    TT_FATAL(mesh_device_->get_devices().size() == 1, "The host_rdma service runs on a unit mesh (one chip)");
    chip_ = mesh_device_->get_devices().front()->id();
    node_ = context_.get_control_plane().get_fabric_node_id_from_physical_chip_id(chip_);
    const bool on_dram = core_ == ServiceCore::Dram;
    const auto hal_core =
        on_dram ? tt::tt_metal::HalProgrammableCoreType::DRAM : tt::tt_metal::HalProgrammableCoreType::TENSIX;
    const auto unreserved =
        on_dram ? tt::tt_metal::HalL1MemAddrType::UNRESERVED : tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED;
    tt_metal::NOC dram_noc = tt_metal::NOC::NOC_0;
    if (on_dram) {
        TT_FATAL(
            hal.has_programmable_core_type(hal_core),
            "The host_rdma service on a DRAM core needs DRAM programmable cores");
        virtual_core_ = mesh_device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
        // The router credits the service at a plain L1 address on its own NoC, which reaches DRAM-core L1 only
        // through a NIU in stream mode.
        const auto& router_config = fabric_context.get_builder_context().get_fabric_router_config();
        dram_noc = router_config.requires_forced_assignment_to_noc1() ? tt_metal::NOC::NOC_1 : tt_metal::NOC::NOC_0;
        const uint8_t noc2axi_nius = cluster_.get_soc_desc(chip_).get_dram_endpoint_noc_mask(virtual_core_);
        TT_FATAL(
            ((noc2axi_nius >> static_cast<uint8_t>(dram_noc)) & 1u) == 0,
            "DRAM core {} of chip {} serves DRAM on NOC{}, which the fabric routers credit the host_rdma service on",
            logical_core.str(),
            chip_,
            static_cast<uint32_t>(dram_noc));
    } else {
        virtual_core_ = mesh_device_->worker_core_from_logical_core(logical_core);
    }
    l1_noc_offset_ = hal.get_l1_noc_offset(hal_core);

    const uint32_t l1_base = static_cast<uint32_t>(hal.get_dev_addr(hal_core, unreserved));
    const uint32_t l1_end = l1_base + static_cast<uint32_t>(hal.get_dev_size(hal_core, unreserved));
    layout_addr_ = tt::align(l1_base, sabi::LAYOUT_ALIGNMENT);
    const uint32_t routing_table_addr = tt::align(layout_addr_ + sizeof(sabi::Layout), sabi::LAYOUT_ALIGNMENT);
    const uint32_t routing_table_bytes = on_dram ? static_cast<uint32_t>(sizeof(routing_l1_info_t)) : 0;
    const uint32_t staging_addr = tt::align(routing_table_addr + routing_table_bytes, sabi::LAYOUT_ALIGNMENT);
    const uint32_t staging_stride =
        tt::align(static_cast<uint32_t>(fabric_context.get_fabric_channel_buffer_size_bytes()), sabi::LAYOUT_ALIGNMENT);
    TT_FATAL(
        staging_addr + sabi::WINDOW * staging_stride <= l1_end,
        "The host_rdma service core's L1 cannot hold {} staging slots of {} B",
        sabi::WINDOW,
        staging_stride);

    auto layout = std::make_unique<sabi::Layout>();
    layout->header.staging_addr = staging_addr;
    layout->header.staging_stride = staging_stride;
    if (on_dram) {
        const auto routing_table_source = mesh_device_->worker_core_from_logical_core(tt_metal::CoreCoord{0, 0});
        layout->header.routing_table_addr = routing_table_addr;
        layout->header.routing_table_src_x = static_cast<uint32_t>(routing_table_source.x);
        layout->header.routing_table_src_y = static_cast<uint32_t>(routing_table_source.y);
    }
    writer_ = std::make_unique<tt_metal::distributed::CoreWriter>(
        cluster_, chip_, virtual_core_, staging_addr, l1_noc_offset_, "host_rdma service");
    writer_->write(layout.get(), sizeof(sabi::Layout), layout_addr_);
    tt_driver_atomics::sfence();
    barrier_service_core();

    if (on_dram) {
        dram_program_ = launch_service_kernel_on_dram(*mesh_device_, logical_core, dram_noc, layout_addr_);
    } else {
        launch_service_kernel(*mesh_device_, logical_core, layout_addr_);
    }
}

void HostRdmaService::barrier_service_core() const {
    if (core_ == ServiceCore::Dram) {
        cluster_.l1_barrier(chip_);
    } else {
        cluster_.l1_barrier(chip_, {virtual_core_});
    }
}

HostRdmaService::~HostRdmaService() noexcept {
    try {
        publish_word(layout_addr_ + offsetof(sabi::Header, terminate), 1);
        if (dram_program_ != nullptr) {
            tt_metal::slow_dispatch::WaitProgramDone(*mesh_device_->get_devices().front(), *dram_program_);
        } else {
            tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "host_rdma: stopping the service on chip {} failed: {}", chip_, e.what());
    } catch (...) {
        log_warning(
            tt::LogFabric, "host_rdma: stopping the service on chip {} failed with an unknown exception", chip_);
    }
}

uint32_t HostRdmaService::channel_addr(uint32_t entry) const {
    return layout_addr_ + offsetof(sabi::Layout, channels) + entry * static_cast<uint32_t>(sizeof(sabi::Channel));
}

uint32_t HostRdmaService::read_word(uint32_t addr) const {
    uint32_t value = 0;
    cluster_.read_core(&value, sizeof(value), tt_cxy_pair(chip_, virtual_core_), addr + l1_noc_offset_);
    return value;
}

uint32_t HostRdmaService::read_consumed(Channel& ch) const {
    ch.consumed_as_last_read = read_word(channel_addr(ch.entry) + offsetof(sabi::Channel, consumed));
    return ch.consumed_as_last_read;
}

void HostRdmaService::publish_word(uint32_t addr, uint32_t value) {
    tt_driver_atomics::sfence();
    writer_->write(&value, sizeof(value), addr);
    tt_driver_atomics::sfence();
}

size_t HostRdmaService::channel_toward(FabricNodeId dst) {
    if (const auto it = channel_index_by_destination_.find(dst); it != channel_index_by_destination_.end()) {
        return it->second;
    }
    TT_FATAL(dst != node_, "A host_rdma write goes to another chip (the destination is the source, {})", node_);
    const auto eth_chans = context_.get_control_plane().get_forwarding_eth_chans_to_chip(node_, dst);
    TT_FATAL(!eth_chans.empty(), "No router on {} forwards toward {}", node_, dst);
    const chan_id_t eth_chan = eth_chans.front();
    const size_t index = channel_on(eth_chan);
    channel_index_by_destination_.emplace(dst, index);
    return index;
}

size_t HostRdmaService::channel_on(chan_id_t eth_chan) {
    const auto& control_plane = context_.get_control_plane();
    for (size_t i = 0; i < channels_.size(); i++) {
        if (channels_[i].eth_chan == eth_chan) {
            return i;
        }
    }
    TT_FATAL(
        channels_.size() < sabi::MAX_CHANNELS,
        "The host_rdma service on chip {} serves at most {} routers",
        chip_,
        sabi::MAX_CHANNELS);
    const auto& fabric_context = control_plane.get_fabric_context();
    Channel ch;
    ch.entry = static_cast<uint32_t>(channels_.size());
    ch.eth_chan = eth_chan;
    ch.router = control_plane.get_host_rdma_connection_info(node_, eth_chan);
    const uint32_t max_payload = static_cast<uint32_t>(std::min<size_t>(
        fabric_context.get_fabric_max_payload_size_bytes(),
        ch.router.buffer_size_bytes - fabric_context.get_fabric_packet_header_size_bytes()));

    const sabi::Channel entry = inactive_channel_entry(ch.router, max_payload);
    cluster_.write_core(
        &entry,
        offsetof(sabi::Channel, queue),
        tt_cxy_pair(chip_, virtual_core_),
        channel_addr(ch.entry) + l1_noc_offset_);
    barrier_service_core();
    publish_word(channel_addr(ch.entry) + offsetof(sabi::Channel, active), 1);
    log_debug(
        tt::LogFabric,
        "host_rdma: service on chip {} serves {} eth chan {} as entry {} ({} slots, {} B payload per packet)",
        chip_,
        node_,
        eth_chan,
        ch.entry,
        entry.num_slots,
        entry.max_payload);
    channels_.push_back(ch);
    return channels_.size() - 1;
}

sabi::Request HostRdmaService::encode(const WriteRequest& request) const {
    TT_FATAL(request.size > 0, "A host_rdma write needs data");
    TT_FATAL(request.dst != node_, "A host_rdma write goes to another chip (the destination is the source, {})", node_);
    TT_FATAL(
        request.src.addr % sabi::NOC_ALIGNMENT == 0 && request.dst_at.addr % sabi::NOC_ALIGNMENT == 0 &&
            request.size % sabi::NOC_ALIGNMENT == 0,
        "host_rdma write addresses and sizes are 16 B aligned (src 0x{:x}, dst 0x{:x}, {} B)",
        request.src.addr,
        request.dst_at.addr,
        request.size);
    sabi::Request r{};
    r.src = {
        request.src.addr,
        static_cast<uint16_t>(request.src.virtual_core.x),
        static_cast<uint16_t>(request.src.virtual_core.y)};
    r.dst = {
        request.dst_at.addr,
        static_cast<uint16_t>(request.dst_at.virtual_core.x),
        static_cast<uint16_t>(request.dst_at.virtual_core.y)};
    r.size = request.size;
    r.dst_dev_id = static_cast<uint16_t>(request.dst.chip_id);
    r.dst_mesh_id = static_cast<uint16_t>(*request.dst.mesh_id);
    if (request.completion.has_value()) {
        r.completion = {
            static_cast<uint32_t>(request.completion->addr),
            static_cast<uint16_t>(request.completion->virtual_core.x),
            static_cast<uint16_t>(request.completion->virtual_core.y)};
        r.completion_inc = request.completion->increment;
    }
    return r;
}

void HostRdmaService::post(Channel& ch, std::span<const sabi::Request> requests) {
    size_t done = 0;
    while (done < requests.size()) {
        const auto room_as_last_read = [&] { return sabi::QUEUE_DEPTH - (ch.produced - ch.consumed_as_last_read); };
        if (room_as_last_read() == 0) {
            const auto deadline = std::chrono::steady_clock::now() + kServiceTimeout;
            while (read_consumed(ch) == ch.produced - sabi::QUEUE_DEPTH) {
                TT_FATAL(
                    std::chrono::steady_clock::now() < deadline,
                    "host_rdma: the service on chip {} took no request for {} s: {}",
                    chip_,
                    std::chrono::duration_cast<std::chrono::seconds>(kServiceTimeout).count(),
                    describe());
            }
        }
        const uint32_t slot = ch.produced % sabi::QUEUE_DEPTH;
        const uint32_t count = static_cast<uint32_t>(
            std::min<size_t>({requests.size() - done, room_as_last_read(), sabi::QUEUE_DEPTH - slot}));
        writer_->write(
            requests.data() + done,
            count * sizeof(sabi::Request),
            channel_addr(ch.entry) + offsetof(sabi::Channel, queue) +
                slot * static_cast<uint32_t>(sizeof(sabi::Request)));
        ch.produced += count;
        publish_word(channel_addr(ch.entry) + offsetof(sabi::Channel, produced), ch.produced);
        done += count;
    }
}

void HostRdmaService::write(const WriteRequest& request) { write_batch(std::span<const WriteRequest>(&request, 1)); }

void HostRdmaService::write_batch(std::span<const WriteRequest> requests) {
    std::vector<std::pair<size_t, std::vector<sabi::Request>>> by_channel;
    for (const auto& request : requests) {
        const size_t index = channel_toward(request.dst);
        auto it = std::find_if(by_channel.begin(), by_channel.end(), [&](const auto& e) { return e.first == index; });
        if (it == by_channel.end()) {
            by_channel.emplace_back(index, std::vector<sabi::Request>{});
            it = std::prev(by_channel.end());
        }
        it->second.push_back(encode(request));
    }
    for (auto& [index, encoded] : by_channel) {
        post(channels_[index], encoded);
    }
}

void HostRdmaService::drain() {
    const auto deadline = std::chrono::steady_clock::now() + kServiceTimeout;
    for (auto& ch : channels_) {
        while (read_consumed(ch) != ch.produced) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "host_rdma: the service on chip {} did not drain in {} s: {}",
                chip_,
                std::chrono::duration_cast<std::chrono::seconds>(kServiceTimeout).count(),
                describe());
        }
    }
}

std::string HostRdmaService::describe() const {
    std::string out = fmt::format(
        "service on chip {} core {}: heartbeat {}",
        chip_,
        virtual_core_.str(),
        read_word(layout_addr_ + offsetof(sabi::Header, heartbeat)));
    for (const auto& ch : channels_) {
        sabi::Channel entry{};
        cluster_.read_core(
            &entry,
            offsetof(sabi::Channel, queue),
            tt_cxy_pair(chip_, virtual_core_),
            channel_addr(ch.entry) + l1_noc_offset_);
        out += fmt::format(
            "; eth chan {}: produced {} (host {}), consumed {}, router credits {} ({} slots)",
            ch.eth_chan,
            entry.produced,
            ch.produced,
            entry.consumed,
            entry.credits,
            entry.num_slots);
    }
    return out;
}

}  // namespace tt::tt_fabric::host_rdma
