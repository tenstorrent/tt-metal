// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/coordination/tcp_transport.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::scaleout_tools::fabric_manager {

namespace {

// --- Blocking, EOF-aware socket helpers ---------------------------------------

// Returns true iff all `n` bytes were written.
bool send_all(int fd, const void* data, std::size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    std::size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        sent += static_cast<std::size_t>(r);
    }
    return true;
}

// Returns true iff all `n` bytes were read; false on clean EOF or error.
bool recv_all(int fd, void* data, std::size_t n) {
    auto* p = static_cast<uint8_t*>(data);
    std::size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, p + got, n - got, 0);
        if (r == 0) {
            return false;  // peer closed
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        got += static_cast<std::size_t>(r);
    }
    return true;
}

template <typename T>
bool send_pod(int fd, const T& v) {
    return send_all(fd, &v, sizeof(T));
}

template <typename T>
bool recv_pod(int fd, T& v) {
    return recv_all(fd, &v, sizeof(T));
}

bool send_blob(int fd, const Bytes& b) {
    uint32_t len = static_cast<uint32_t>(b.size());
    if (!send_pod(fd, len)) {
        return false;
    }
    return len == 0 || send_all(fd, b.data(), b.size());
}

bool recv_blob(int fd, Bytes& b) {
    uint32_t len = 0;
    if (!recv_pod(fd, len)) {
        return false;
    }
    b.resize(len);
    return len == 0 || recv_all(fd, b.data(), b.size());
}

}  // namespace

// ============================== Client ========================================

TcpTransport::TcpTransport(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    const std::string port_str = std::to_string(port);
    int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    TT_FATAL(rc == 0, "TcpTransport: getaddrinfo({}:{}) failed: {}", host, port, gai_strerror(rc));

    // Retry to tolerate a controller that is still starting up (bounded to ~30s).
    int fd = -1;
    constexpr int kMaxAttempts = 300;
    for (int attempt = 0; attempt < kMaxAttempts && fd < 0; ++attempt) {
        for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
            fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) {
                continue;
            }
            if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                break;
            }
            ::close(fd);
            fd = -1;
        }
        if (fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    ::freeaddrinfo(res);
    TT_FATAL(fd >= 0, "TcpTransport: could not connect to controller at {}:{} after retries", host, port);

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fd_ = fd;
}

TcpTransport::~TcpTransport() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

std::vector<Bytes> TcpTransport::exchange(
    const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload) {
    // Request: [has_mesh u8][mesh_id u32][epoch u64][index i32][count i32][blob]
    const uint8_t has_mesh = scope.mesh_id.has_value() ? 1 : 0;
    const uint32_t mesh_id = scope.mesh_id.value_or(0);
    bool ok = send_pod(fd_, has_mesh) && send_pod(fd_, mesh_id) && send_pod(fd_, epoch) &&
              send_pod(fd_, static_cast<int32_t>(index)) && send_pod(fd_, static_cast<int32_t>(count)) &&
              send_blob(fd_, payload);
    TT_FATAL(ok, "TcpTransport: failed to send exchange request (epoch {}, index {})", epoch, index);

    // Response: [n u32] then n blobs.
    uint32_t n = 0;
    TT_FATAL(recv_pod(fd_, n), "TcpTransport: failed to read response count (epoch {})", epoch);
    std::vector<Bytes> result(n);
    for (uint32_t i = 0; i < n; ++i) {
        TT_FATAL(recv_blob(fd_, result[i]), "TcpTransport: failed to read response blob {} (epoch {})", i, epoch);
    }
    return result;
}

// ============================== Server ========================================

TcpControllerServer::TcpControllerServer(uint16_t port, std::shared_ptr<Controller> controller) :
    controller_(std::move(controller)) {
    TT_FATAL(controller_ != nullptr, "TcpControllerServer requires a non-null Controller");

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TT_FATAL(listen_fd_ >= 0, "TcpControllerServer: socket() failed: {}", std::strerror(errno));

    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    TT_FATAL(
        ::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
        "TcpControllerServer: bind(port={}) failed: {}",
        port,
        std::strerror(errno));

    TT_FATAL(::listen(listen_fd_, 64) == 0, "TcpControllerServer: listen() failed: {}", std::strerror(errno));

    sockaddr_in bound{};
    socklen_t bound_len = sizeof(bound);
    TT_FATAL(
        ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0,
        "TcpControllerServer: getsockname() failed: {}",
        std::strerror(errno));
    port_ = ntohs(bound.sin_port);
}

TcpControllerServer::~TcpControllerServer() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
    }
}

void TcpControllerServer::handle_connection(int conn_fd) {
    // Serve exchange requests from a single agent until it disconnects.
    for (;;) {
        uint8_t has_mesh = 0;
        if (!recv_pod(conn_fd, has_mesh)) {
            break;  // clean disconnect (agent finished) or error
        }
        uint32_t mesh_id = 0;
        uint64_t epoch = 0;
        int32_t index = 0;
        int32_t count = 0;
        Bytes payload;
        bool ok = recv_pod(conn_fd, mesh_id) && recv_pod(conn_fd, epoch) && recv_pod(conn_fd, index) &&
                  recv_pod(conn_fd, count) && recv_blob(conn_fd, payload);
        if (!ok) {
            break;
        }

        ScopeKey scope{has_mesh ? std::optional<uint32_t>(mesh_id) : std::nullopt};
        auto gathered = controller_->exchange(scope, epoch, index, count, payload);

        uint32_t n = static_cast<uint32_t>(gathered.size());
        if (!send_pod(conn_fd, n)) {
            break;
        }
        bool sent = true;
        for (const auto& blob : gathered) {
            if (!send_blob(conn_fd, blob)) {
                sent = false;
                break;
            }
        }
        if (!sent) {
            break;
        }
    }
    ::close(conn_fd);
}

void TcpControllerServer::serve(int world_size) {
    TT_FATAL(world_size >= 1, "TcpControllerServer::serve requires world_size >= 1 (got {})", world_size);
    std::vector<std::thread> handlers;
    handlers.reserve(static_cast<std::size_t>(world_size));

    for (int accepted = 0; accepted < world_size; ++accepted) {
        int conn_fd = -1;
        for (;;) {
            conn_fd = ::accept(listen_fd_, nullptr, nullptr);
            if (conn_fd >= 0) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            TT_FATAL(false, "TcpControllerServer: accept() failed: {}", std::strerror(errno));
        }
        int one = 1;
        ::setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        handlers.emplace_back([this, conn_fd] { this->handle_connection(conn_fd); });
    }

    for (auto& t : handlers) {
        t.join();
    }
}

}  // namespace tt::scaleout_tools::fabric_manager
