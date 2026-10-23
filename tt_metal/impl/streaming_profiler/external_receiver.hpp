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
#include <optional>
#include <span>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/sync/clock_solver.hpp"
#include "impl/streaming_profiler/external_link.hpp"

namespace tt::umd {
class IoWindow;
class TTDevice;
class SysmemBuffer;
class SiliconSysmemManager;
}  // namespace tt::umd

namespace tt::tt_metal::streaming_profiler {

// The external process's receiver of one profiled process's capture. It pins a ring per relay, points the relays at
// them itself through UMD, and repacks the copies that land there into frames of a stream per device, which the
// consumers decode as the profiled process's consumers decode its own streams. The eth relays copy every clock sync
// record here too, and a solver of this process's own builds its clock map from them and from its own reads of the
// root chip's refclk, so the map goes on when the profiled process dies or hangs.
class ExternalReceiver final : public Receiver {
public:
    explicit ExternalReceiver(std::unique_ptr<external::ExternalEnd> link);
    ~ExternalReceiver() override;

    std::span<const ReceiverStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return capture_; }
    uint64_t live_head(uint32_t stream) const override;
    const ClockMap& clock_map() const override;
    bool external() const override { return true; }

    // Reads the rings until `stop`, or until the profiled process ends its capture, which stops the relays. The relays
    // keep copying after the profiled process dies, so then reading goes on past the link's end, until the next
    // profiled process takes the devices.
    void run(const std::stop_token& stop);
    // Logs `failure` and ends the link, which has the profiled process turn its relays' copies off.
    void drop(const std::exception& failure);

private:
    struct Ring;
    struct Stream;
    struct Clock;

    bool read_ring(Ring& ring);
    void repack(const uint32_t* entry, uint32_t dev);
    void on_sync(const uint32_t* frame, uint32_t dev);
    // Whether every relay has finished, as a profiled process that ends its capture makes them before it closes the
    // link.
    bool relays_finished();
    // Finishes the clock map once no sync record can follow.
    void end_capture();
    // Points each relay's external socket at its ring, with `attach` a fresh generation, or turns its copies off with
    // 0, and waits up to a second for the relays to follow. A relay that has stopped doesn't.
    void point_relays(uint32_t attach);

    void write(Stream& stream);
    std::optional<uint32_t> local_id(uint32_t id);

    std::unique_ptr<external::ExternalEnd> link_;
    CaptureContext capture_;
    ClockBases bases_;
    tt_xy_pair root_pcie_;
    uint32_t root_pcie_device_ = 0;
    // Each PCIe device's handle and pinned memory, which outlive the rings in it and the refclk window.
    std::map<uint32_t, std::pair<std::unique_ptr<tt::umd::TTDevice>, std::unique_ptr<tt::umd::SiliconSysmemManager>>>
        pcie_devices_;
    std::unique_ptr<Clock> clock_;
    bool solver_pending_ = false;
    bool ended_ = false;
    bool relays_done_ = false;  // the relays finished the capture
    std::vector<external::Hello::Relay> relays_;
    uint32_t l1_alignment_ = 0;
    // A window onto each relay's L1, which holds its control block and its external socket's config. Each keeps its
    // own mapping, so turning the copies off at exit needs nothing of UMD's, whose statics may be gone by then.
    std::vector<std::unique_ptr<tt::umd::IoWindow>> relay_l1_;
    std::vector<Ring> rings_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;
    // The profiled process's call-site ids, to this process's.
    std::unordered_map<uint32_t, std::optional<uint32_t>> sites_;
    std::vector<uint32_t> words_, frame_, packed_;
    uint64_t lost_bytes_ = 0;
};

// Has this process receive the records sent with kSendToExternalProcess by every profiled process that connects to its
// socket name, each one's until the next one connects. The first call starts listening, and later calls do nothing.
void serve_profiled_processes();
// Has this process, which is creating a MetalContext, stop listening and drop any profiled process it serves.
void stop_serving_profiled_processes();

}  // namespace tt::tt_metal::streaming_profiler
