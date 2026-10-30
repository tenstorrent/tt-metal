// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How the Tensor prefetcher's host worker delivers request pages to its senders, when it gives up on
// a sender, and what it reports of their status. Blackhole DRISC senders read an H2D socket each;
// Quasar CCE senders read a ring in their own TL1 (tensor_prefetcher_cce_abi.hpp), which the host
// fills with plain core writes.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "impl/buffers/tensor_prefetcher_cce_abi.hpp"

namespace tt::tt_metal {

class IDevice;

namespace distributed {

class H2DSocket;
class MeshDevice;

// Endpoint e is sender e % num_senders on device e / num_senders, in the manager's device and sender
// order.
class RequestTransport {
public:
    virtual ~RequestTransport() = default;

    // Sends `page`, page_bytes() long, to `endpoint` when it has room. Returns false at once otherwise.
    virtual bool try_write(uint32_t endpoint, std::vector<uint8_t>& page) = 0;

    // Whether the sender at `endpoint` has exited, so it will take no more pages.
    virtual bool target_exited(uint32_t endpoint) = 0;

    // Whether the worker has given up on the sender at `endpoint`, so it drops every page to it but STOP.
    virtual bool gave_up_on(uint32_t endpoint) const = 0;

    // For a page to `endpoint` still unsent after a pass in which no endpoint took a page, `without_progress` after the
    // worker took its request: whether to give up on that sender and drop the page.
    virtual bool give_up_on_stalled(
        uint32_t endpoint, uint8_t command, std::chrono::steady_clock::duration without_progress) = 0;

    // Pauses the worker after a pass in which no endpoint took a page.
    virtual void pause_after_stalled_pass() const = 0;

    // The no-progress limit at which the worker gave up on a sender, if it gave up on any. Read once the worker has
    // stopped.
    virtual std::optional<std::chrono::steady_clock::duration> gave_up_after() const = 0;

    // Logs every sender's status. Never throws.
    virtual void log_sender_statuses() const = 0;

    // One line per sender that ended Failed or whose status cannot be read, after log_sender_statuses when there is
    // any. Never throws.
    virtual std::vector<std::string> failed_senders() const = 0;

    // Bytes of one request page as it is sent.
    virtual uint32_t page_bytes() const = 0;
};

// One H2D socket per (device, sender), at the same DRISC L1 offsets on every sender core.
class H2DSocketTransport final : public RequestTransport {
public:
    H2DSocketTransport(
        MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        uint32_t socket_config_l1_addr,
        uint32_t socket_data_l1_addr);
    ~H2DSocketTransport() override;

    bool try_write(uint32_t endpoint, std::vector<uint8_t>& page) override;
    bool target_exited(uint32_t endpoint) override;
    bool gave_up_on(uint32_t endpoint) const override;
    bool give_up_on_stalled(
        uint32_t endpoint, uint8_t command, std::chrono::steady_clock::duration without_progress) override;
    void pause_after_stalled_pass() const override;
    std::optional<std::chrono::steady_clock::duration> gave_up_after() const override;
    void log_sender_statuses() const override;
    std::vector<std::string> failed_senders() const override;
    uint32_t page_bytes() const override;

    // The socket config address the sender kernel at `endpoint` reads.
    uint32_t config_buffer_address(uint32_t endpoint) const;

private:
    uint32_t page_bytes_;
    std::vector<std::unique_ptr<H2DSocket>> sockets_;
};

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
    std::vector<std::string> failed_senders() const override;
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

// The status of the CCE sender on `sender_logical` whose control block is at `control_base`.
CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base);

std::string format_cce_sender_status(const CceSenderStatus& status);

}  // namespace distributed
}  // namespace tt::tt_metal
