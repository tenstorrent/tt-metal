// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_cce_sender.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/buffers/tensor_prefetcher_sender_backend.hpp"
#include "impl/buffers/tensor_prefetcher_transport.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

namespace {

// How long the host worker waits on a TL1 request ring that takes no page when TT_METAL_OPERATION_TIMEOUT_SECONDS is
// unset.
constexpr std::chrono::minutes kDefaultRequestRingNoProgressLimit{5};
// Alignment of a CCE sender's stage ring size: its thirds then start on 64 B lines.
constexpr uint32_t kCceStageRingAlignBytes = 6 * kCceSenderLineBytes;

// How long the host worker pauses when no TL1 request ring has room.
constexpr std::chrono::milliseconds kRequestRingRetryInterval{1};

uint32_t read_bytes_acked(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    uint32_t bytes_acked = 0;
    read_dram_core_l1(
        mesh_device,
        device,
        sender_logical,
        control_base + kCceRingMetaOffset + offsetof(RingMeta, bytes_acked),
        std::as_writable_bytes(std::span(&bytes_acked, 1)));
    return bytes_acked;
}

// The GO_MSG signal of the DRAM core `sender_logical` on `device`.
uint8_t read_go_message_signal(const MetalContext& metal, const IDevice* device, const CoreCoord& sender_logical) {
    const Hal& hal = metal.hal();
    dev_msgs::go_msg_t go_message =
        hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::go_msg_t>();
    const uint64_t go_message_address =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG) & ~uint64_t{0x3};
    const CoreCoord sender_virtual = device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    metal.get_cluster().read_core(
        go_message.data(), go_message.size(), tt_cxy_pair(device->id(), sender_virtual), go_message_address);
    return go_message.view().signal();
}

// A CCE sender reads its own DRAM view and checks no address itself: local receiver r of mapping entry s reads slab
// recv_index_base_per_sender[s] + r of the tensor's shards on that sender's bank. So every slab one request reads must
// hold this tensor's data on that bank and end inside the view.
void require_receiver_slabs_in_tensor_and_view(
    const MeshTensor& tensor,
    const TensorPrefetcherTensorLayout& layout,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping,
    const std::vector<uint32_t>& recv_index_base_per_sender,
    uint64_t dram_view_bytes,
    uint32_t tensor_idx) {
    const Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
    const std::optional<BufferDistributionSpec>& distribution = buffer->buffer_distribution_spec();
    TT_FATAL(
        distribution.has_value(),
        "Tensor prefetcher: input tensor {} has no buffer distribution spec, so its receiver slabs cannot be placed",
        tensor_idx);
    const std::vector<CoreCoord>& shard_cores = distribution->cores();
    const uint64_t tensor_base = tensor.mesh_buffer().address();
    for (size_t sender_slot = 0; sender_slot < mapping.size(); ++sender_slot) {
        const uint32_t bank = static_cast<uint32_t>(mapping[sender_slot].first.x);
        const std::vector<CoreCoord>::const_iterator bank_core =
            std::find(shard_cores.begin(), shard_cores.end(), CoreCoord{bank, 0});
        const uint64_t shards_on_bank =
            bank_core == shard_cores.end()
                ? 0
                : distribution->num_shards_per_core(static_cast<size_t>(std::distance(shard_cores.begin(), bank_core)));
        const uint64_t slabs_read =
            static_cast<uint64_t>(recv_index_base_per_sender[sender_slot]) + mapping[sender_slot].second.num_cores();
        const uint64_t bytes_read = slabs_read * layout.recv_stride_bytes;
        TT_FATAL(
            slabs_read <= shards_on_bank && bytes_read <= buffer->aligned_size_per_bank(),
            "Tensor prefetcher: input tensor {} holds {} receiver slabs ({} B) on DRAM bank {}, but that bank's sender "
            "reads slabs [0, {}) of {} B each, so the request runs past the tensor",
            tensor_idx,
            shards_on_bank,
            buffer->aligned_size_per_bank(),
            bank,
            slabs_read,
            layout.recv_stride_bytes);
        TT_FATAL(
            tensor_base + bytes_read <= dram_view_bytes,
            "Tensor prefetcher: input tensor {} on DRAM bank {} reads [0x{:x}, 0x{:x}), past the {} B DRAM view",
            tensor_idx,
            bank,
            tensor_base,
            tensor_base + bytes_read,
            dram_view_bytes);
    }
}

// The request ring in each CCE sender's TL1 control block at `control_base`. The host writes a page, then
// the running byte count, and re-reads the kernel's count only when the ring looks full.
class Tl1RingTransport final : public RequestTransport {
public:
    // `no_progress_limit` is how long the worker waits on a ring that takes no page before it gives up on that
    // sender. `give_up_on_running_senders` is whether it also gives up then on a sender that still runs; otherwise
    // only on a sender that never started.
    Tl1RingTransport(
        const MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        uint32_t control_base,
        std::chrono::steady_clock::duration no_progress_limit,
        bool give_up_on_running_senders);

    bool try_write(uint32_t endpoint, std::vector<uint8_t>& page) override;
    bool target_exited(uint32_t endpoint) override;
    bool gave_up_on(uint32_t endpoint) const override;
    bool give_up_on_stalled(
        uint32_t endpoint, uint8_t command, std::chrono::steady_clock::duration without_progress) override;
    void pause_after_stalled_pass() const override;
    std::optional<std::chrono::steady_clock::duration> gave_up_after() const override;
    void log_sender_statuses() const override;
    std::optional<std::string> failure_summary(uint32_t dropped_request_pages) const override;
    uint32_t page_bytes() const override;

private:
    struct Endpoint {
        IDevice* device = nullptr;
        CoreCoord sender_logical;
        uint32_t bytes_sent = 0;
        // The kernel's count as last read.
        uint32_t bytes_acked = 0;
        // Whether a failed ring write or status read has been logged; each is logged once per endpoint.
        bool write_error_logged = false;
        bool status_error_logged = false;
        // Whether the worker gave up on this sender.
        bool given_up = false;
        // Whether the worker has logged keeping this sender's pages past the no-progress limit.
        bool retention_logged = false;
    };

    // One line per sender that ended Failed or whose status cannot be read, after log_sender_statuses when there is
    // any. Never throws.
    std::vector<std::string> failed_senders() const;

    // Pages sent to `endpoint` that its sender never consumed. Only meaningful once the sender has exited and no
    // more pages are being sent; throws if the device cannot be read.
    uint32_t unconsumed_pages(const Endpoint& endpoint) const;

    // Whether the sender's status says it is still in its request loop, so a page may yet reach it.
    bool sender_running(const Endpoint& endpoint) const;

    const MeshDevice& mesh_device_;
    uint32_t control_base_;
    std::chrono::steady_clock::duration no_progress_limit_;
    bool give_up_on_running_senders_;
    // Set when the worker gives up on a sender, so stop() reports the requests it dropped.
    bool gave_up_on_any_sender_ = false;
    std::vector<Endpoint> endpoints_;
};

Tl1RingTransport::Tl1RingTransport(
    const MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const std::vector<CoreCoord>& sender_logical_cores,
    uint32_t control_base,
    std::chrono::steady_clock::duration no_progress_limit,
    bool give_up_on_running_senders) :
    mesh_device_(mesh_device),
    control_base_(control_base),
    no_progress_limit_(no_progress_limit),
    give_up_on_running_senders_(give_up_on_running_senders) {
    endpoints_.reserve(devices.size() * sender_logical_cores.size());
    for (IDevice* device : devices) {
        for (const CoreCoord& sender_logical : sender_logical_cores) {
            endpoints_.push_back(Endpoint{.device = device, .sender_logical = sender_logical});
        }
    }
}

// Runs on the worker thread, where a throw would terminate: the manager builds every page page_bytes() long and
// only passes endpoints it provisioned. A failed device access leaves the page unsent; the first is logged.
bool Tl1RingTransport::try_write(uint32_t endpoint_index, std::vector<uint8_t>& page) {
    Endpoint& endpoint = endpoints_[endpoint_index];
    try {
        if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
            endpoint.bytes_acked =
                read_bytes_acked(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
                return false;
            }
        }
        // Strict ordering: each write completes before the next is issued, so a ring page lands before the count
        // that covers it. A DRAM core's writes are otherwise relaxed.
        const uint32_t slot = (endpoint.bytes_sent / kRequestPageBytes) % kCceRequestRingSlots;
        write_dram_core_l1(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            control_base_ + kCceRequestRingOffset + (slot * kRequestPageBytes),
            std::as_bytes(std::span(page.data(), kRequestPageBytes)),
            tt::umd::IoOrdering::Strict);
        // A separate write after the page's, so the kernel never sees a count that covers a page still in flight.
        const uint32_t bytes_sent = endpoint.bytes_sent + kRequestPageBytes;
        write_dram_core_l1(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            control_base_ + kCceRingMetaOffset + offsetof(RingMeta, bytes_sent),
            std::as_bytes(std::span(&bytes_sent, 1)),
            tt::umd::IoOrdering::Strict);
        endpoint.bytes_sent = bytes_sent;
    } catch (const std::exception& error) {
        if (!endpoint.write_error_logged) {
            endpoint.write_error_logged = true;
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: writing a request page to sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
        return false;
    }
    return true;
}

// A status that cannot be read is unknown, not exited: the worker keeps the sender's pages until its no-progress limit.
bool Tl1RingTransport::target_exited(uint32_t endpoint_index) {
    Endpoint& endpoint = endpoints_[endpoint_index];
    try {
        const CceSenderStatus status =
            read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
        return status.state == static_cast<uint32_t>(CceSenderState::Stopped) ||
               status.state == static_cast<uint32_t>(CceSenderState::Failed);
    } catch (const std::exception& error) {
        if (!endpoint.status_error_logged) {
            endpoint.status_error_logged = true;
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: reading the status of sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
        return false;
    }
}

bool Tl1RingTransport::gave_up_on(uint32_t endpoint_index) const { return endpoints_[endpoint_index].given_up; }

// A sender frees a ring slot only once it has handled the command in it, so a ring also takes no page while its sender
// sits in a legitimate credit, CQ or drain wait, which wait_budget_cycles = 0 never bounds. Without an operation
// timeout, every page stays while the sender may still run: dropped, a page loses its data, and a dropped STOP leaves
// the sender running for good. With one, the page is dropped, and stop()'s WaitProgramDone times out on that sender.
bool Tl1RingTransport::give_up_on_stalled(
    uint32_t endpoint_index, uint8_t command, std::chrono::steady_clock::duration without_progress) {
    if (without_progress < no_progress_limit_) {
        return false;
    }
    Endpoint& endpoint = endpoints_[endpoint_index];
    const std::string sender_name =
        fmt::format("sender {} on device {}", endpoint.sender_logical.str(), endpoint.device->id());
    if (!give_up_on_running_senders_ && sender_running(endpoint)) {
        if (!endpoint.retention_logged) {
            endpoint.retention_logged = true;
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; keeping its command {} page and "
                "later pages while it runs, because TT_METAL_OPERATION_TIMEOUT_SECONDS is unset",
                sender_name,
                std::chrono::duration<float>(without_progress).count(),
                command);
        }
        return false;
    }
    log_error(
        tt::LogMetal,
        "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; dropping its command {} page and every "
        "later page to it but STOP",
        sender_name,
        std::chrono::duration<float>(without_progress).count(),
        command);
    endpoint.given_up = true;
    gave_up_on_any_sender_ = true;
    return true;
}

// Each ring pass reads sender TL1 over the device link, so the worker pauses rather than only yielding.
void Tl1RingTransport::pause_after_stalled_pass() const { std::this_thread::sleep_for(kRequestRingRetryInterval); }

std::optional<std::chrono::steady_clock::duration> Tl1RingTransport::gave_up_after() const {
    if (!gave_up_on_any_sender_) {
        return std::nullopt;
    }
    return no_progress_limit_;
}

void Tl1RingTransport::log_sender_statuses() const {
    for (const Endpoint& endpoint : endpoints_) {
        try {
            const CceSenderStatus status =
                read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: sender {} on device {}: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                format_cce_sender_status(status));
        } catch (const std::exception& error) {
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: reading the status of sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
    }
}

// When the worker gave up on a sender, stop() has already logged every status, so the senders are not read again.
std::optional<std::string> Tl1RingTransport::failure_summary(uint32_t dropped_request_pages) const {
    const std::optional<std::chrono::steady_clock::duration> worker_gave_up_after = gave_up_after();
    if (worker_gave_up_after.has_value()) {
        return fmt::format(
            "Tensor prefetcher: a sender's request ring took no page for {:.0f} s, so the host worker dropped that "
            "sender's later requests; the log names the sender and its status",
            std::chrono::duration<float>(*worker_gave_up_after).count());
    }
    const std::vector<std::string> failed_sender_lines = failed_senders();
    if (failed_sender_lines.empty()) {
        return std::nullopt;
    }
    return fmt::format(
        "Tensor prefetcher: {} sender(s) ended Failed, so their requests were not all delivered ({} more request "
        "pages never fit in an exited sender's ring and were dropped): {}",
        failed_sender_lines.size(),
        dropped_request_pages,
        fmt::join(failed_sender_lines, "; "));
}

std::vector<std::string> Tl1RingTransport::failed_senders() const {
    std::vector<std::string> failed_senders;
    for (const Endpoint& endpoint : endpoints_) {
        try {
            const CceSenderStatus status =
                read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            if (status.state == static_cast<uint32_t>(CceSenderState::Failed)) {
                std::string unread_pages;
                try {
                    unread_pages = std::to_string(unconsumed_pages(endpoint));
                } catch (const std::exception&) {
                    unread_pages = "an unknown number of";
                }
                failed_senders.push_back(fmt::format(
                    "sender {} on device {}: {}; {} request page(s) left unread in its ring, counting stop's DRAIN and "
                    "STOP",
                    endpoint.sender_logical.str(),
                    endpoint.device->id(),
                    format_cce_sender_status(status),
                    unread_pages));
            }
        } catch (const std::exception& error) {
            failed_senders.push_back(fmt::format(
                "sender {} on device {}: status unreadable: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what()));
        }
    }
    if (!failed_senders.empty()) {
        log_sender_statuses();
    }
    return failed_senders;
}

uint32_t Tl1RingTransport::page_bytes() const { return kRequestPageBytes; }

// The kernel acks a page only once it has handled it, Failed included, so the device's count, not the cached one,
// marks where the sender stopped reading.
uint32_t Tl1RingTransport::unconsumed_pages(const Endpoint& endpoint) const {
    const uint32_t bytes_acked =
        read_bytes_acked(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
    return (endpoint.bytes_sent - bytes_acked) / kRequestPageBytes;
}

// A status that cannot be read is unknown and counts as running. Called on the host worker thread, so a failed read is
// not thrown; target_exited, which reads the same status earlier in the pass, logs it.
bool Tl1RingTransport::sender_running(const Endpoint& endpoint) const {
    try {
        const uint32_t state =
            read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_).state;
        return state == static_cast<uint32_t>(CceSenderState::Running) ||
               state == static_cast<uint32_t>(CceSenderState::WaitingCq) ||
               state == static_cast<uint32_t>(CceSenderState::Draining);
    } catch (const std::exception&) {
        return true;
    }
}

// TL1 offset of every CCE sender's control block, unchecked: cce_sender_control_base adds the checks.
uint32_t sender_control_block_base(const MeshDevice& mesh_device) {
    return tt::align(
        static_cast<uint32_t>(mesh_device.impl().drisc_l1_arena().kernel_working_region_base()), kCceSenderLineBytes);
}

// CCE DRAM cores: one CCE sender per provisioned bank, fed by a request ring in its own TL1. The CCE sender implements
// only the batched receiver-contiguous round; the Blackhole K-row-major, streaming and PrefetcherPipe paths are not
// ported to it.
class CceSenderBackend final : public PrefetcherSenderBackend {
public:
    TensorPrefetcherPreconditions preconditions(const MeshDevice& mesh_device) const override {
        const MetalContext& metal = MetalContext::instance(mesh_device.impl().get_context_id());
        const Cluster& cluster = metal.get_cluster();
        const std::vector<IDevice*> devices = mesh_device.impl().get_devices();
        for (const IDevice* device : devices) {
            const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
            const uint32_t num_views = static_cast<uint32_t>(soc_desc.get_num_dram_views());
            std::unordered_map<CoreCoord, uint32_t> view_by_sender_core;
            for (uint32_t view = 0; view < num_views; ++view) {
                if (!soc_desc.is_dram_view_programmable(static_cast<int>(view))) {
                    return {
                        .ok = false,
                        .reason = fmt::format("DRAM view {} on device {} has no programmable CCE", view, device->id())};
                }
                for (const CoreCoord& sender_logical : mesh_device.impl().dram_sender_logical_cores(device, view)) {
                    const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(sender_logical);
                    if (view_by_sender_core.contains(sender_translated)) {
                        return {
                            .ok = false,
                            .reason = fmt::format(
                                "DRAM views {} and {} on device {} share the sender CCE at translated {}",
                                view_by_sender_core.at(sender_translated),
                                view,
                                device->id(),
                                sender_translated.str())};
                    }
                    view_by_sender_core.emplace(sender_translated, view);
                }
            }
        }

        if (metal.rtoptions().get_fast_dispatch()) {
            return {
                .ok = false, .reason = "the CCE prefetcher requires slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1)"};
        }

        const KernelRegionCarve carve = CceSenderBackend().carve(mesh_device);
        if (!carve.failure_reason.empty()) {
            return {.ok = false, .reason = carve.failure_reason};
        }

        // Device open writes RUN_MSG_INIT and waits for firmware to flip it to RUN_MSG_DONE, so INIT here
        // means the CCE was reset or re-inited since. RUN_MSG_GO is a running Program, which is fine: this
        // checks that firmware is present, not that the CCE is idle.
        for (const IDevice* device : devices) {
            const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
            const uint32_t num_views = static_cast<uint32_t>(soc_desc.get_num_dram_views());
            for (uint32_t view = 0; view < num_views; ++view) {
                for (const CoreCoord& sender_logical : mesh_device.impl().dram_sender_logical_cores(device, view)) {
                    const uint8_t signal = read_go_message_signal(metal, device, sender_logical);
                    if (signal == dev_msgs::RUN_MSG_DONE || signal == dev_msgs::RUN_MSG_GO) {
                        continue;
                    }
                    const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(sender_logical);
                    const uint32_t cce_index = package_dram_core_index(soc_desc, sender_logical);
                    return {
                        .ok = false,
                        .reason = fmt::format(
                            "CCE {} (device {}, DRAM view {}, translated {}) holds GO_MSG signal 0x{:02x}: {}",
                            cce_index,
                            device->id(),
                            view,
                            sender_translated.str(),
                            signal,
                            signal == dev_msgs::RUN_MSG_INIT ? "its firmware has not finished init"
                                                             : "its mailbox is corrupt")};
                }
            }
        }
        return {.ok = true, .reason = ""};
    }

    void resolve_config(
        const MeshDevice& mesh_device,
        const experimental::TensorPrefetcherConfig& config,
        const std::optional<QuasarTensorPrefetcherConfig>& quasar_config) override {
        TT_FATAL(
            !config.blackhole.has_value(),
            "Blackhole tensor prefetcher configuration cannot be used on architecture {}",
            mesh_device.arch());
        config_ = quasar_config.value_or(QuasarTensorPrefetcherConfig{});
    }

    std::vector<uint32_t> sender_banks() const override { return config_.dram_sender_banks; }

    uint32_t senders_per_bank() const override { return 1; }

    KernelRegionCarve carve(const MeshDevice& mesh_device) const override {
        return carve_kernel_region(
            MetalContext::instance(mesh_device.impl().get_context_id()).hal(),
            mesh_device.impl().drisc_l1_arena(),
            "request ring",
            config_.stage_ring_bytes,
            kCceStageRingAlignBytes,
            [control_base = sender_control_block_base(mesh_device)](
                KernelRegionCarve& carve, uint32_t /*kernel_region_base*/) {
                // The CQ signal slots are the control block's first lines, so WAIT_CQ fencing addresses them the
                // same way on both architectures.
                carve.cq_signal_l1_addr = control_base + kCceCqSignalSlotsOffset;
                carve.cq_signal_slot_stride = kCceCqSignalSlotStrideBytes;
                carve.stage_ring_base = control_base + kCceSenderControlBytes;
            });
    }

    PreparedSenders prepare_senders(
        MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        const KernelRegionCarve& carve) const override {
        const MetalContext& metal = MetalContext::instance(mesh_device.impl().get_context_id());
        const size_t num_cces = metal.get_cluster()
                                    .get_soc_desc(devices.front()->id())
                                    .get_metal_dram_cores(tt::CoordSystem::TRANSLATED)
                                    .size();
        // A warning, not a failure: a session with no receivers, such as one that only fences, is still valid.
        if (sender_logical_cores.size() >= num_cces) {
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: all {} CCEs are senders, so none is left for receiver Programs. Leave some "
                "banks out of the prefetcher's sender banks to keep CCEs free for receivers.",
                num_cces);
        }
        // Before anything is written: a busy sender CCE must fail Start with its TL1 untouched.
        require_idle_sender_cces(metal, devices, sender_logical_cores);
        const float timeout_seconds = metal.rtoptions().get_timeout_duration_for_operations().count();
        if (config_.wait_budget_cycles != 0 && timeout_seconds == 0.0f) {
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: senders give up a wait after {} rdcycle cycles, but "
                "TT_METAL_OPERATION_TIMEOUT_SECONDS is unset, so StopTensorPrefetcher waits for them without a limit. "
                "The budget is CCE cycles, not wall time: on the emulator the chip clock advances only with host "
                "traffic (about 85,000 cycles/s while the host polls, 90 cycles/s while it is idle).",
                config_.wait_budget_cycles);
        }
        // How long the worker waits on a request ring that takes no page before it gives up on that sender. It also
        // gives up then on a sender that still runs only when TT_METAL_OPERATION_TIMEOUT_SECONDS is set, so stop()
        // reaches WaitProgramDone and that times out. Otherwise only a sender that never started is given up on.
        const std::chrono::steady_clock::duration no_progress_limit =
            timeout_seconds > 0.0f
                ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<float>(timeout_seconds))
                : std::chrono::duration_cast<std::chrono::steady_clock::duration>(kDefaultRequestRingNoProgressLimit);
        const bool give_up_on_running_senders = timeout_seconds > 0.0f;
        const uint32_t control_base = sender_control_block_base(mesh_device);
        std::vector<std::unique_ptr<Program>> programs =
            build_programs(metal, devices, sender_logical_cores, control_base, carve.stage_ring_size);
        zero_control_blocks(mesh_device, devices, sender_logical_cores, control_base);
        return PreparedSenders{
            .programs = std::move(programs),
            .transport = std::make_unique<Tl1RingTransport>(
                mesh_device,
                devices,
                sender_logical_cores,
                control_base,
                no_progress_limit,
                give_up_on_running_senders)};
    }

    bool benchmark_enabled() const override { return false; }

    void require_prefetcher_pipe_delivery(const MeshDevice& mesh_device) const override {
        TT_THROW(
            "QueueTensorPrefetcherRequest with PrefetcherPipes is not supported on {}; queue a GlobalCircularBuffer",
            mesh_device.arch());
    }

    // A CCE runs one launch at a time, so a receiver on a provisioned sender CCE could never run.
    void require_receivers_off_senders(
        const MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) const override {
        const Cluster& cluster = MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster();
        // Logical-to-CCE resolution follows each device's DRAM harvesting, so every device is checked.
        for (const IDevice* device : devices) {
            const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
            std::unordered_set<uint32_t> sender_cces;
            for (const CoreCoord& sender_logical : sender_logical_cores) {
                sender_cces.insert(package_dram_core_index(soc_desc, sender_logical));
            }
            for (const std::pair<CoreCoord, CoreRangeSet>& sender_and_receivers : mapping) {
                for (const CoreCoord& receiver : corerange_to_cores(sender_and_receivers.second)) {
                    const uint32_t receiver_cce = package_dram_core_index(soc_desc, receiver);
                    TT_FATAL(
                        !sender_cces.contains(receiver_cce),
                        "QueueTensorPrefetcherRequest: receiver CCE {} (logical {}) on device {} is one of this "
                        "prefetcher's provisioned sender CCEs, so its receiver Program cannot run while the "
                        "prefetcher does. Leave its bank out of the prefetcher's sender banks.",
                        receiver_cce,
                        receiver.str(),
                        device->id());
                }
            }
        }
    }

    // Ahead of the streaming topology guard, so a streaming tensor gets this message.
    void validate_request_tensors(
        const MeshDevice& mesh_device, const std::vector<experimental::TensorPrefetcherInput>& tensors) const override {
        for (size_t tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
            TT_FATAL(
                tensors[tensor_idx].rotation.empty(),
                "Tensor prefetcher on {} delivers batched tensors only, but input tensor {} has a streaming rotation",
                mesh_device.arch(),
                tensor_idx);
        }
    }

    // Ahead of compute_tensor_layout, whose K-row-major sizer would reject such a tensor for its own reasons.
    void validate_tensor_before_layout(
        const MeshDevice& mesh_device,
        const experimental::TensorPrefetcherInput& input,
        uint32_t total_receivers,
        size_t tensor_idx) const override {
        TT_FATAL(
            detect_layout_mode(
                input.tensor.get(),
                *input.tensor.get().mesh_buffer().get_reference_buffer(),
                total_receivers,
                LayoutTieBreak::PreferReceiverContiguous) == TENSOR_PREFETCHER_LAYOUT_RECEIVER_CONTIGUOUS,
            "Tensor prefetcher on {} supports receiver-contiguous tensors only, but input tensor {} resolves to "
            "the K-row-major layout. Shard it so each receiver owns a disjoint contiguous shard.",
            mesh_device.arch(),
            tensor_idx);
    }

    void validate_tensor_after_layout(
        const MeshDevice& mesh_device,
        const IDevice* reference_device,
        const MeshTensor& tensor,
        const TensorPrefetcherTensorLayout& layout,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping,
        const std::vector<uint32_t>& recv_index_base_per_sender,
        size_t tensor_idx) const override {
        require_receiver_slabs_in_tensor_and_view(
            tensor,
            layout,
            mapping,
            recv_index_base_per_sender,
            MetalContext::instance(mesh_device.impl().get_context_id())
                .get_cluster()
                .get_soc_desc(reference_device->id())
                .dram_view_size,
            static_cast<uint32_t>(tensor_idx));
    }

    LayoutTieBreak global_cb_layout_tie_break() const override { return LayoutTieBreak::PreferReceiverContiguous; }

private:
    static constexpr const char* kKernelPath = "tt_metal/impl/buffers/kernels/tensor_prefetcher_cce.cpp";

    // A launch onto a CCE still running a kernel is lost, so every provisioned sender CCE must be idle.
    static void require_idle_sender_cces(
        const MetalContext& metal,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores) {
        for (const IDevice* device : devices) {
            const metal_SocDescriptor& soc_desc = metal.get_cluster().get_soc_desc(device->id());
            for (const CoreCoord& sender_logical : sender_logical_cores) {
                const uint8_t signal = read_go_message_signal(metal, device, sender_logical);
                TT_FATAL(
                    signal == dev_msgs::RUN_MSG_DONE,
                    "Tensor prefetcher: sender CCE {} (device {}, logical {}) holds GO_MSG signal 0x{:02x}, not "
                    "RUN_MSG_DONE (0x{:02x}). It is still running a Program, and a launch onto it would be lost; leave "
                    "its bank out of the prefetcher's sender banks or wait for that Program first.",
                    package_dram_core_index(soc_desc, sender_logical),
                    device->id(),
                    sender_logical.str(),
                    signal,
                    dev_msgs::RUN_MSG_DONE);
            }
        }
    }

    // Zeroes each sender's CQ signal slots, ring meta and status, which the CCE kernel never does.
    static void zero_control_blocks(
        const MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        uint32_t control_base) {
        const std::vector<std::byte> zeros(kCceSenderZeroedBytes, std::byte{0});
        for (IDevice* device : devices) {
            for (const CoreCoord& sender_logical : sender_logical_cores) {
                write_dram_core_l1(mesh_device, device, sender_logical, control_base, zeros);
            }
        }
    }

    std::vector<std::unique_ptr<Program>> build_programs(
        const MetalContext& metal,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        uint32_t control_base,
        uint32_t stage_ring_size) const {
        std::vector<uint32_t> runtime_args(kCceSenderArgs, 0);
        runtime_args[kCceSenderArgControlBase] = control_base;
        runtime_args[kCceSenderArgBudgetLo] = static_cast<uint32_t>(config_.wait_budget_cycles);
        runtime_args[kCceSenderArgBudgetHi] = static_cast<uint32_t>(config_.wait_budget_cycles >> 32);
        runtime_args[kCceSenderArgStageRingBytes] = stage_ring_size;
        const Cluster& cluster = metal.get_cluster();
        std::vector<std::unique_ptr<Program>> programs;
        for (uint32_t d = 0; d < devices.size(); ++d) {
            const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(devices[d]->id());
            const uint32_t cces_per_mimir = static_cast<uint32_t>(soc_desc.get_grid_size(tt::CoreType::DRAM).y);
            std::unique_ptr<Program> program = std::make_unique<Program>();
            for (const CoreCoord& sender_logical : sender_logical_cores) {
                // A sender reads only its own view, which must sit on its own Mimir: the GDDR route it reads through
                // is the local one.
                const uint32_t bank = static_cast<uint32_t>(sender_logical.x);
                const uint32_t view_mimir =
                    static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(bank)));
                const uint32_t sender_cce = package_dram_core_index(soc_desc, sender_logical);
                TT_FATAL(
                    sender_cce / cces_per_mimir == view_mimir,
                    "Tensor prefetcher: sender CCE {} of DRAM view {} is on Mimir {}, but the view is on Mimir {}",
                    sender_cce,
                    bank,
                    sender_cce / cces_per_mimir,
                    view_mimir);
                const uint64_t view_offset = soc_desc.get_address_offset(static_cast<int>(bank));
                runtime_args[kCceSenderArgMimir] = view_mimir;
                runtime_args[kCceSenderArgViewOffsetLo] = static_cast<uint32_t>(view_offset);
                runtime_args[kCceSenderArgViewOffsetHi] = static_cast<uint32_t>(view_offset >> 32);
                const KernelHandle kernel_id = CreateKernel(*program, kKernelPath, sender_logical, DramConfig{});
                SetRuntimeArgs(*program, kernel_id, sender_logical, runtime_args);
            }
            programs.push_back(std::move(program));
        }
        return programs;
    }

    QuasarTensorPrefetcherConfig config_;
};

}  // namespace

std::unique_ptr<PrefetcherSenderBackend> make_cce_sender_backend() { return std::make_unique<CceSenderBackend>(); }

uint32_t cce_sender_control_base(const MeshDevice& mesh_device) {
    TT_FATAL(
        MetalContext::instance(mesh_device.impl().get_context_id()).hal().has_cce_dram_cores(),
        "CCE prefetcher senders exist only on devices with CCE DRAM cores, not {}",
        mesh_device.arch());
    const KernelRegionCarve carve = CceSenderBackend().carve(mesh_device);
    TT_FATAL(carve.failure_reason.empty(), "{}", carve.failure_reason);
    return sender_control_block_base(mesh_device);
}

CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    CceSenderStatus status{};
    read_dram_core_l1(
        mesh_device,
        device,
        sender_logical,
        control_base + kCceSenderStatusOffset,
        std::as_writable_bytes(std::span(&status, 1)));
    return status;
}

std::string format_cce_sender_status(const CceSenderStatus& status) {
    return fmt::format(
        "state {} code {} request {} target 0x{:x} receiver {} sent {} acked {}",
        status.state,
        status.code,
        status.request_index,
        status.target_state_addr,
        status.receiver,
        status.sent,
        status.acked);
}

}  // namespace tt::tt_metal::distributed
