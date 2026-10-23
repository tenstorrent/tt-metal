// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/external_link.hpp"

namespace tt::umd {
class TTDevice;
class SysmemBuffer;
class SiliconSysmemManager;
}  // namespace tt::umd

namespace tt::tt_metal::streaming_profiler {

// The external process's receiver of one profiled process's capture. It pins a ring per relay, has the profiled
// process point the relays at them, and repacks the copies that land there into frames of a stream per device, which
// the consumers decode as the profiled process's consumers decode its own streams. Its clock map is a copy of the
// profiled process's, kept in step by the changes the link carries.
class ExternalReceiver final : public Receiver {
public:
    explicit ExternalReceiver(std::unique_ptr<external::ExternalEnd> link);
    ~ExternalReceiver() override;

    std::span<const ReceiverStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return capture_; }
    uint64_t live_head(uint32_t stream) const override;
    const ClockMap& clock_map() const override { return *clock_map_; }

    // Reads the rings until the profiled process has closed the link and every copy that landed is read.
    void run();

private:
    struct Ring;
    struct Stream;

    bool read_ring(Ring& ring);
    void repack(const uint32_t* entry, uint32_t dev);
    void write(Stream& stream);
    std::optional<uint32_t> local_id(uint32_t id);

    std::unique_ptr<external::ExternalEnd> link_;
    CaptureContext capture_;
    std::unique_ptr<ClockMap> clock_map_;
    // Each PCIe device's handle and pinned memory, which outlive the rings in it.
    std::map<uint32_t, std::pair<std::unique_ptr<tt::umd::TTDevice>, std::unique_ptr<tt::umd::SiliconSysmemManager>>>
        pcie_devices_;
    std::vector<Ring> rings_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;
    // The profiled process's call-site ids, to this process's.
    std::unordered_map<uint32_t, std::optional<uint32_t>> sites_;
    std::vector<uint32_t> words_, frame_, packed_;
    uint64_t lost_bytes_ = 0;
};

// Has this process receive the records sent with kSendToExternalProcess by every profiled process that connects to its
// socket name, one at a time. The first call starts listening, and later calls do nothing.
void serve_profiled_processes();
// Has this process, which is creating a MetalContext, stop listening.
void stop_serving_profiled_processes();

}  // namespace tt::tt_metal::streaming_profiler
