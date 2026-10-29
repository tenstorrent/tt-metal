// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The connection between a Metal process and the external process: a Unix socket at a fixed path,
// /tmp/tt_metal_device_observability/external_process.sock, which the external process listens on and any user's Metal
// process on the host connects to, and the only thing the two processes share. A container reaches it by mounting that
// directory. Only the Metal process sends, and it never waits for the external process. It connects when its capture
// starts, and while none is connected, as soon as an external process starts listening, which an inotify watch on the
// directory reports. It sends the messages of protobuf/external_process.proto, one Envelope per length-prefixed frame:
// its CaptureDescriptor with every call site loaded so far, then the call sites of each image that loads later, before
// the image can run, then CaptureEnded once its relays have stopped. A record reaches the external process's rings only
// after its call site went out: an image whose call sites fail to go out runs with the relays' writes there off.
// The external process points the devices at rings in its own memory itself, through the endpoints the descriptor
// names, and the relays write the routed records and clock sync measurements there.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

// protobuf/external_process.pb.h, which only the sources that build or read the messages include.
namespace tt::streaming_profiler::proto {
class CaptureDescriptor;
class FrameLayout;
class PacketFormat;
}  // namespace tt::streaming_profiler::proto

namespace tt::llrt {
struct CallSite;
}
namespace tt::tt_metal::experimental::device_observability::detail {
class CallSites;
}

namespace tt::tt_metal::streaming_profiler::external {

namespace proto = ::tt::streaming_profiler::proto;

// The protocol version this build speaks: an external process refuses a Metal process with another.
inline constexpr uint32_t kProtocolVersion = 2;

// This build's frame and packet formats, as a capture descriptor gives them.
proto::FrameLayout this_build_frame_layout();
proto::PacketFormat this_build_packet_format();

// The Metal process's connection to the external process, which it opens as the client. `detach` turns the relays'
// writes to the external process's rings off and returns once their last ones have landed, which this client does when
// an external process goes, so none writes to its freed rings.
class ExternalProcessClient {
public:
    using Detach = std::function<void()>;
    ExternalProcessClient(const proto::CaptureDescriptor& descriptor, Detach detach);
    // Closes the connection.
    ~ExternalProcessClient();
    // Connects to the external process now if one listens, and otherwise once one starts listening, and again after a
    // connection ends. Called once the relays run. Records published before the external process points the relays at
    // its rings stay in this process.
    void start();
    // Tells the external process the capture has ended, once every relay has stopped and its last write has landed.
    void end_capture();
    ExternalProcessClient(const ExternalProcessClient&) = delete;
    ExternalProcessClient& operator=(const ExternalProcessClient&) = delete;

private:
    void run();
    void serve(int fd);
    // Sends the external process the call sites for it loaded since the last push, under send_mu_; false when the
    // connection failed.
    bool push_call_sites();

    // The capture's descriptor, to which each connection's first message adds the call sites loaded so far.
    const std::unique_ptr<const proto::CaptureDescriptor> descriptor_;
    const Detach detach_;

    std::mutex mu_;
    bool stop_ = false;
    int wake_fd_ = -1;  // an eventfd the destructor writes to wake the connecting thread

    // Every write to the connection, from its thread or from a thread loading an image.
    std::mutex send_mu_;
    int fd_ = -1;  // the connected external process's socket, -1 while none is
    size_t call_sites_pushed_ = 0;
    bool capture_ended_ = false;

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
    // Waits for the next Metal process and returns its connection, or -1 when the socket fails.
    int accept() const;

private:
    int fd_ = -1;
    int lock_fd_ = -1;
};

// One Metal process's connection, as the external process accepted it. Once started, a thread of its own reads what the
// Metal process sends until it closes the connection.
class ClientConnection {
public:
    // Reads the CaptureDescriptor from `fd`, which it then owns. descriptor() is null when the Metal process
    // closed first, sent none in time, or speaks another protocol version.
    explicit ClientConnection(int fd);
    ~ClientConnection();
    ClientConnection(const ClientConnection&) = delete;
    ClientConnection& operator=(const ClientConnection&) = delete;

    const proto::CaptureDescriptor* descriptor() const { return descriptor_.get(); }
    // The Metal process's pid, which messages name it by.
    int pid() const { return pid_; }
    // Starts the connection's thread, which reads the call sites the Metal process pushes.
    void start();
    // Whether the Metal process has closed the connection. Callable from any thread.
    bool closed() const { return closed_.load(std::memory_order_acquire); }
    // Whether the Metal process ended its capture before it closed the connection, which a process that died doesn't.
    // Set before closed().
    bool capture_ended() const { return capture_ended_.load(std::memory_order_acquire); }

    // Ends the connection, which has the Metal process turn its relays' writes off.
    void drop();
    // Whether this process ended the connection, rather than the Metal process. Set before closed().
    bool dropped() const { return dropped_.load(std::memory_order_acquire); }
    // The Metal process's call sites, by its ids, which the connection's thread adds as they arrive.
    std::shared_ptr<const experimental::device_observability::detail::CallSites> call_sites() const {
        return call_sites_;
    }
    // The Metal process's call site `id`. It is sent before any of the site's records can reach this process's rings,
    // but the connection's thread may not have read it yet, so this waits until it has. Throws when the connection ends
    // without it, which only a corrupt record can cause.
    const tt::llrt::CallSite& wait_for_call_site(uint32_t id);

private:
    void read_thread();

    const int fd_;
    int pid_ = 0;
    std::unique_ptr<proto::CaptureDescriptor> descriptor_;

    std::atomic<bool> closed_{false};
    std::atomic<bool> capture_ended_{false};
    std::atomic<bool> dropped_{false};

    std::shared_ptr<experimental::device_observability::detail::CallSites> call_sites_;
    std::mutex sites_mu_;  // with sites_cv_, wakes wait_for_call_site() as call sites arrive
    std::condition_variable sites_cv_;

    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler::external
