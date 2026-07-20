// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// TCP reference transport for the fabric-manager coordinator (Option (a)).
//
// TcpTransport (agent side) implements ControllerTransport by shipping each
// exchange() over a persistent TCP connection to the controller. TcpControllerServer
// (controller side) accepts one connection per agent, handles each on its own thread,
// and drives a shared Controller. This proves cross-process / cross-host, no-MPI
// coordination. A future gRPC transport is a drop-in replacement for this pair; the
// wire framing here is a minimal length-prefixed encoding (native-endian; homogeneous
// x86 cluster assumption for the PoC).
//

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tools/scaleout/fabric_manager/coordination/controller.hpp"
#include "tools/scaleout/fabric_manager/coordination/transport.hpp"

namespace tt::scaleout_tools::fabric_manager {

// Agent-side client. Connects on construction; one connection is used serially by a
// single agent thread (the control-plane bring-up is single-threaded per agent).
class TcpTransport final : public ControllerTransport {
public:
    TcpTransport(const std::string& host, uint16_t port);
    ~TcpTransport() override;

    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;

    [[nodiscard]] std::vector<Bytes> exchange(
        const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload) override;

private:
    int fd_ = -1;
};

// Controller-side server. Binds on construction (port 0 => OS-assigned; query port()).
class TcpControllerServer {
public:
    TcpControllerServer(uint16_t port, std::shared_ptr<Controller> controller);
    ~TcpControllerServer();

    TcpControllerServer(const TcpControllerServer&) = delete;
    TcpControllerServer& operator=(const TcpControllerServer&) = delete;

    [[nodiscard]] uint16_t port() const { return port_; }

    // Accept exactly `world_size` agent connections, serve each on its own thread, and
    // return once every connection has closed (i.e. all agents finished bring-up).
    void serve(int world_size);

private:
    void handle_connection(int conn_fd);

    int listen_fd_ = -1;
    uint16_t port_ = 0;
    std::shared_ptr<Controller> controller_;
};

}  // namespace tt::scaleout_tools::fabric_manager
