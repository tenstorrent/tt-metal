// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <span>
#include <stop_token>
#include <vector>

#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/sync/clock_solver.hpp"
#include "impl/streaming_profiler/external_connection.hpp"

namespace tt::umd {
class IoWindow;
class TTDevice;
class SysmemBuffer;
class SiliconSysmemManager;
}  // namespace tt::umd

namespace tt::tt_metal::streaming_profiler {

// The external process's receiver of one Metal process's capture, which it learns from the capture descriptor. It
// pins a ring per endpoint, points the devices at them itself through UMD, and repacks the routed records that land in
// the record rings into frames of a stream per device, which the consumers decode as the Metal process's consumers
// decode its own streams. A solver of this process's own builds its clock map from the clock sync rings and its own
// reads of the refclk, so the Metal process sends it no clock map.
class ExternalReceiver final : public Receiver {
public:
    explicit ExternalReceiver(std::unique_ptr<external::ClientConnection> connection);
    ~ExternalReceiver() override;

    std::span<const ReceiverStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return capture_; }
    uint64_t live_head(uint32_t stream) const override;
    const ClockMap& clock_map() const override;
    const experimental::device_observability::detail::CaptureLanes& lanes() const override { return *lanes_; }

    // Reads the rings until `stop`, or until the Metal process closes the connection. Unless it ended its capture,
    // which stops the relays, their writes are then turned off. Either way, the rings are read a last time and the
    // clock map finished.
    void run(const std::stop_token& stop);
    // Logs `failure` and ends the connection, which has the Metal process turn its relays' writes off.
    void drop(const std::exception& failure);

private:
    struct Ring;
    struct Stream;
    struct Clock;
    bool read_ring(Ring& ring);
    // The words of the entry that starts at `entry` in a ring, or 0 when it can't be one.
    uint32_t entry_words(const Ring& ring, const uint32_t* entry) const;
    void repack(const uint32_t* entry, uint32_t dev);
    void on_clock_sync(const uint32_t* frame, uint32_t dev);
    // The core index of the core a frame names.
    uint32_t core_of(const uint32_t* frame, uint32_t dev) const;
    // Finishes the clock map once no sync record can follow.
    void end_capture();
    // Points each endpoint's device at its ring, or stops its writes, and waits up to a second for each to follow. A
    // relay that has stopped doesn't.
    void point_devices(bool enable);

    void write(Stream& stream);

    std::unique_ptr<external::ClientConnection> connection_;
    const external::proto::CaptureDescriptor& descriptor_;  // which connection_ holds
    const external::proto::FrameLayout& frames_;
    const external::proto::PacketFormat& packets_;
    uint32_t header_words_ = 0;  // a record frame's pages that hold its prefix and control block, in words

    CaptureContext capture_;
    LanesReference lanes_;  // naming the Metal process's call sites, which the connection holds

    // Each PCIe device's handle and pinned memory, which outlive the rings in it and the refclk window.
    std::map<uint32_t, std::pair<std::unique_ptr<tt::umd::TTDevice>, std::unique_ptr<tt::umd::SiliconSysmemManager>>>
        pcie_devices_;

    std::unique_ptr<Clock> clock_;
    bool solver_pending_ = false;
    bool ended_ = false;

    // The relays write nothing more to the rings: the capture ended, or this process turned their writes off.
    bool relays_stopped_ = false;
    std::vector<Ring> rings_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;

    std::vector<uint32_t> words_, frame_, packed_;
    uint64_t lost_bytes_ = 0;
};

// Makes this process the external process: it listens on the socket name and receives the records sent with
// kSendToExternalProcess by every Metal process that connects, each one's until another connects on its chips. The
// first call starts listening, and throws when another process holds the name; later calls do nothing.
void serve_metal_processes();

}  // namespace tt::tt_metal::streaming_profiler
