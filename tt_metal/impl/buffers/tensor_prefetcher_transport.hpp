// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How the Tensor prefetcher's host worker delivers request pages to its senders. Blackhole DRISC
// senders read an H2D socket each; Quasar CCE senders read a ring in their own TL1
// (tensor_prefetcher_cce_abi.hpp), which the host fills with plain core writes.

#pragma once

#include <cstdint>
#include <memory>
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
    Tl1RingTransport(
        const MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        uint32_t control_base);

    bool try_write(uint32_t endpoint, std::vector<uint8_t>& page) override;
    bool target_exited(uint32_t endpoint) override;
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
    };

    const MeshDevice& mesh_device_;
    uint32_t control_base_;
    std::vector<Endpoint> endpoints_;
};

// The status of the CCE sender on `sender_logical` whose control block is at `control_base`.
CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base);

std::string format_cce_sender_status(const CceSenderStatus& status);

}  // namespace distributed
}  // namespace tt::tt_metal
