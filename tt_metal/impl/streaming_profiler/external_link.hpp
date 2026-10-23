// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The link between a profiled process and the external process: a Unix socket under one abstract name per user, which
// the external process listens on, and the only thing the two processes share. A profiled process connects when its
// capture starts, and again while it runs whenever none is connected. It first sends a hello: its relays, by the PCIe
// device each sits on, its capture's devices as the decoder reads them, and the base of its clock map. From then on it
// sends every change to its clock map. The external process pins one buffer per relay in its own memory and sends their
// device addresses, and the profiled process points each relay's external socket at its buffer; the relay then copies
// there the records sent with kSendToExternalProcess. The external process asks for each call site it meets by id, and
// the profiled process answers from the images it has loaded. A call site's id is assigned when its image loads, before
// any of its records exists, so every id the external process meets has an answer.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"

namespace tt::tt_metal::streaming_profiler::external {

// An external process's buffer for one relay: a ring of `bytes` at device address `noc_addr`, with the relay's
// bytes_sent in the word after it.
struct RelayBuffer {
    uint32_t relay = 0;
    uint64_t noc_addr = 0;
    uint32_t bytes = 0;
};

// What a profiled process sends first: each relay's PCIe device and the capture device it serves, the capture's
// devices, and the bases its clock map counts from.
struct Hello {
    struct Relay {
        uint32_t pcie_device = 0;
        uint32_t dev = 0;
    };
    std::vector<Relay> relays;
    std::vector<CaptureContext::Device> devices;
    ClockBases bases;
};

// The profiled process's end. `attach` points a relay at a buffer and returns whether the relay follows; `detach`
// turns a relay's copies off and returns once its last one has landed.
class ProfiledEnd {
public:
    using Attach = std::function<bool(const RelayBuffer&)>;
    using Detach = std::function<void(uint32_t relay)>;
    ProfiledEnd(const Hello& hello, Attach attach, Detach detach);
    // Sends the clock map changes made before it, then closes the link.
    ~ProfiledEnd();
    // Connects to the external process listening on the socket name, and keeps trying while none is connected. Called
    // once the relays run. When an external process listens now, returns once it has attached, so a kernel launched
    // after this returns has its records copied there.
    void start();
    ProfiledEnd(const ProfiledEnd&) = delete;
    ProfiledEnd& operator=(const ProfiledEnd&) = delete;

    // Sends a change to the capture's clock map. Called on the thread that makes it, from before start() on.
    void send(const ClockUpdate& update);

private:
    void run();
    void serve(int fd);
    void wake();

    const std::vector<uint8_t> hello_;
    const Attach attach_;
    const Detach detach_;
    int wake_[2] = {-1, -1};

    std::mutex mu_;
    std::condition_variable started_cv_;
    enum class Start { Trying, Done } start_ = Start::Trying;
    bool connected_ = false;
    bool stop_ = false;
    std::vector<ClockUpdate> queued_;
    // The newest change to each series, which an external process that connects later starts from.
    std::vector<std::optional<ClockUpdate>> newest_chip_;
    std::optional<ClockUpdate> newest_host_;
    std::vector<bool> finished_;

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
    using OnClock = std::function<void(std::span<const ClockUpdate>)>;
    // Reads the hello from `fd`, which it then owns. hello() is nullopt when the profiled process closed first.
    explicit ExternalEnd(int fd);
    ~ExternalEnd();
    ExternalEnd(const ExternalEnd&) = delete;
    ExternalEnd& operator=(const ExternalEnd&) = delete;

    const std::optional<Hello>& hello() const { return hello_; }
    // Starts handing the clock map's changes to `on_clock`, on the link's thread. Requests need it started.
    void start(OnClock on_clock);
    // Whether the profiled process has closed the link. Callable from any thread.
    bool closed() const { return closed_.load(std::memory_order_acquire); }

    // Points the relays at these buffers; false when the profiled process can't or has gone.
    bool attach(std::span<const RelayBuffer> buffers);
    // Turns the relays' copies off and returns once nothing more lands in the buffers, so they can be freed.
    void detach();
    // Adds the call site the profiled process knows by `id` to this process's call sites and returns its id here, or
    // nullopt when the profiled process has none or has gone.
    std::optional<uint32_t> add_site(uint32_t id);

private:
    std::optional<std::vector<uint8_t>> request(const std::vector<uint8_t>& message);
    void read_thread();

    const int fd_;
    std::optional<Hello> hello_;
    OnClock on_clock_;
    std::atomic<bool> closed_{false};
    std::mutex request_mu_;  // one request is outstanding at a time
    std::mutex reply_mu_;
    std::condition_variable reply_cv_;
    std::optional<std::vector<uint8_t>> reply_;
    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler::external
