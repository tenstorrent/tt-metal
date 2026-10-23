// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The link between a profiled process and the external process: a Unix socket under one abstract name per user, which
// the external process listens on, and the only thing the two processes share. Only the profiled process sends, and it
// never waits for the external process. It connects when its capture starts, and again while it runs whenever none is
// connected. It first sends a hello: its relays, by the PCIe device each sits on, its capture's devices as the decoder
// reads them, and the bases its clock map counts from, which the external process's clock map shares. The external
// process pins one buffer per relay in its own memory and points each relay's external socket at its buffer itself; the
// relays then copy there the records sent with kSendToExternalProcess, and the eth relays every clock sync record. The
// profiled process sends each call site whose records go there as its image loads, before any of its records exists,
// so the link carries every call site ahead of its records.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"

namespace tt::tt_metal::streaming_profiler::external {

// What a profiled process sends first: its relays, the capture's devices and links, and the bases its clock map counts
// from.
struct Hello {
    // A relay: its chip's PCIe device number and capture device, its core (translated), the NoC addresses of its
    // control block and its external socket's config, and its chip's PCIe tile's NoC encoding, which the config holds.
    struct Relay {
        uint32_t pcie_device = 0;
        uint32_t dev = 0;
        uint32_t core_x = 0, core_y = 0;
        uint64_t ctrl = 0;
        uint64_t config = 0;
        uint32_t pcie_encoding = 0;
    };
    std::vector<Relay> relays;
    std::vector<CaptureContext::Device> devices;
    std::vector<CaptureContext::Link> links;
    ClockBases bases;
    // The root chip's PCIe tile (translated), whose refclk timer the external process reads for its own clock map.
    uint32_t root_pcie_x = 0, root_pcie_y = 0;
    uint32_t l1_alignment = 0;  // which the socket config's layout follows
};

// The profiled process's end. `detach` turns a relay's copies off and returns once its last one has landed, which this
// end does for every relay when an external process goes, so none writes to its freed buffers.
class ProfiledEnd {
public:
    using Detach = std::function<void(uint32_t relay)>;
    ProfiledEnd(const Hello& hello, Detach detach);
    // Closes the link.
    ~ProfiledEnd();
    // Connects to the external process listening on the socket name, and keeps trying while none is connected. Called
    // once the relays run. Records published before the external process attaches the relays stay in this process.
    void start();
    ProfiledEnd(const ProfiledEnd&) = delete;
    ProfiledEnd& operator=(const ProfiledEnd&) = delete;

private:
    void run();
    void serve(int fd);
    void wake();
    // Sends the external process the call sites loaded since the last push, under send_mu_; false when the link
    // failed.
    bool push_sites();

    const std::vector<uint8_t> hello_;
    const uint32_t relay_count_;
    const Detach detach_;
    int wake_[2] = {-1, -1};

    std::mutex mu_;
    // Every write to the link, from its thread or from a thread loading an image.
    std::mutex send_mu_;
    int fd_ = -1;  // the connected external process's socket, -1 while none is
    uint32_t sites_pushed_ = 0;
    bool stop_ = false;

    std::thread thread_;
};

// The external process's socket, which profiled processes connect to.
class ExternalListener {
public:
    ExternalListener();
    ~ExternalListener();
    ExternalListener(const ExternalListener&) = delete;
    ExternalListener& operator=(const ExternalListener&) = delete;

    // Listens on the socket name; false while another process holds it.
    bool listen();
    // Waits for the next profiled process and returns its connection, or -1 once stopped.
    int accept() const;
    // Ends a wait in accept(), and every later one.
    void stop();

private:
    int fd_ = -1;
};

// The external process's end of one profiled process's link. Once started, a thread of its own reads what the
// profiled process sends until it closes the link.
class ExternalEnd {
public:
    // Reads the hello from `fd`, which it then owns. hello() is nullopt when the profiled process closed first.
    explicit ExternalEnd(int fd);
    ~ExternalEnd();
    ExternalEnd(const ExternalEnd&) = delete;
    ExternalEnd& operator=(const ExternalEnd&) = delete;

    const std::optional<Hello>& hello() const { return hello_; }
    // The profiled process's pid, which messages name it by.
    int pid() const { return pid_; }
    // Starts the link's thread, which reads the call sites the profiled process pushes.
    void start();
    // Whether the profiled process has closed the link. Callable from any thread.
    bool closed() const { return closed_.load(std::memory_order_acquire); }

    // Ends the link, which has the profiled process turn its relays' copies off.
    void drop();
    // The id here of the call site the profiled process knows by `id`, which it pushed before any of the site's records
    // existed: waits for the link's thread to read it, and is nullopt when the link ended without it.
    std::optional<uint32_t> local_id(uint32_t id);

private:
    void read_thread();

    const int fd_;
    int pid_ = 0;
    std::optional<Hello> hello_;
    std::atomic<bool> closed_{false};
    std::mutex sites_mu_;
    std::condition_variable sites_cv_;
    std::unordered_map<uint32_t, uint32_t> sites_;  // the profiled process's call-site ids, to this process's
    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler::external
