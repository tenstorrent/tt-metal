// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How the Tensor prefetcher's host worker delivers request pages to its senders, when it gives up on
// a sender, and what it reports of their status. Blackhole DRISC senders read an H2D socket each;
// other senders' transports implement RequestTransport elsewhere.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

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

    // Why stop() throws once the session is released, if it must; `dropped_request_pages` is how many pages the
    // worker dropped because their sender had exited. Read once the worker has stopped and the senders have exited.
    virtual std::optional<std::string> failure_summary(uint32_t dropped_request_pages) const = 0;

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
    std::optional<std::string> failure_summary(uint32_t dropped_request_pages) const override;
    uint32_t page_bytes() const override;

    // The socket config address the sender kernel at `endpoint` reads.
    uint32_t config_buffer_address(uint32_t endpoint) const;

private:
    uint32_t page_bytes_;
    std::vector<std::unique_ptr<H2DSocket>> sockets_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
