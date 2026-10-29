// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The connection between a Metal process and the external process: a Unix socket at a fixed path,
// /tmp/tt_metal_device_observability/external_process.sock, which the external process listens on and any user's Metal
// process on the host connects to, and the only thing the two processes share. A container reaches it by mounting that
// directory. A Metal process connects when its capture starts, and again while it runs whenever none is connected. It
// first sends a hello: its relays, by the PCIe device each sits on, its capture's devices as the decoder reads them,
// and the base of its clock map. From then on it sends every change to its clock map. The external process pins one
// buffer per relay in its own memory and sends their device addresses, and the Metal process points each relay's
// external socket at its buffer; the relay then copies there the records sent with kSendToExternalProcess. The external
// process asks for each call site it meets by id, and the Metal process answers from the images it has loaded. A call
// site's id is assigned when its image loads, before any of its records exists, so every id the external process meets
// has an answer.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"

namespace tt::llrt {
struct CallSite;
}
namespace tt::tt_metal::experimental::device_observability::detail {
class CallSites;
}

namespace tt::tt_metal::streaming_profiler::external {

// An external process's buffer for one relay: a ring of `bytes` at device address `noc_addr`, with the relay's
// bytes_sent in the word after it.
struct RelayBuffer {
    uint32_t relay = 0;
    uint64_t noc_addr = 0;
    uint32_t bytes = 0;
};

// What a Metal process sends first: each relay's PCIe device and the capture device it serves, the capture's
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

// The Metal process's connection to the external process, which it opens as the client. `attach` points a relay at a
// buffer and returns whether the relay follows; `detach` turns a relay's copies off and returns once its last one has
// landed.
class ExternalProcessClient {
public:
    using Attach = std::function<bool(const RelayBuffer&)>;
    using Detach = std::function<void(uint32_t relay)>;
    ExternalProcessClient(const Hello& hello, Attach attach, Detach detach);
    // Sends the clock map changes made before it, then closes the connection.
    ~ExternalProcessClient();
    // Connects to the external process listening on the socket name, and keeps trying while none is connected. Called
    // once the relays run. When an external process listens now, returns once it has attached, so a kernel launched
    // after this returns has its records copied there.
    void start();
    ExternalProcessClient(const ExternalProcessClient&) = delete;
    ExternalProcessClient& operator=(const ExternalProcessClient&) = delete;

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

// The external process's socket, which Metal processes connect to.
class ExternalListener {
public:
    ExternalListener();
    ~ExternalListener();
    ExternalListener(const ExternalListener&) = delete;
    ExternalListener& operator=(const ExternalListener&) = delete;

    // Listens on the socket; false while another process does. It holds a lock next to the socket while it listens, so
    // one process listens at a time, and the next takes over the socket file one leaves when it dies.
    bool listen();
    // Waits for the next Metal process and returns its connection, or -1 once stopped.
    int accept() const;
    // Ends a wait in accept(), and every later one.
    void stop();

private:
    int fd_ = -1;
    int lock_fd_ = -1;
};

// One Metal process's connection, as the external process accepted it. Once started, a thread of its own reads what the
// Metal process sends until it closes the connection.
class ClientConnection {
public:
    using OnClock = std::function<void(std::span<const ClockUpdate>)>;
    // Reads the hello from `fd`, which it then owns. hello() is nullopt when the Metal process closed first.
    explicit ClientConnection(int fd);
    ~ClientConnection();
    ClientConnection(const ClientConnection&) = delete;
    ClientConnection& operator=(const ClientConnection&) = delete;

    const std::optional<Hello>& hello() const { return hello_; }
    // Starts handing the clock map's changes to `on_clock`, on the connection's thread. Requests need it started.
    void start(OnClock on_clock);
    // Whether the Metal process has closed the connection. Callable from any thread.
    bool closed() const { return closed_.load(std::memory_order_acquire); }

    // Points the relays at these buffers; false when the Metal process can't or has gone.
    bool attach(std::span<const RelayBuffer> buffers);
    // Turns the relays' copies off and returns once nothing more lands in the buffers, so they can be freed.
    void detach();
    // The Metal process's call sites, by its ids, which add_site() adds to.
    std::shared_ptr<const experimental::device_observability::detail::CallSites> call_sites() const {
        return call_sites_;
    }
    // Adds the call site the Metal process knows by `id` to its call sites here and returns it, or null when the Metal
    // process has none or has gone.
    const tt::llrt::CallSite* add_site(uint32_t id);

private:
    std::optional<std::vector<uint8_t>> request(const std::vector<uint8_t>& message);
    void read_thread();

    const int fd_;
    std::optional<Hello> hello_;
    OnClock on_clock_;
    std::atomic<bool> closed_{false};
    std::shared_ptr<experimental::device_observability::detail::CallSites> call_sites_;
    std::mutex request_mu_;  // one request is outstanding at a time
    std::mutex reply_mu_;
    std::condition_variable reply_cv_;
    std::optional<std::vector<uint8_t>> reply_;
    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler::external
