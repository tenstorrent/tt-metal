// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// InProcessTransport: a ControllerTransport that calls a shared Controller
// directly. Used by the self-test (N agent threads in one process sharing one
// Controller) to exercise the full ServiceCoordinator + rendezvous logic without
// hardware or sockets.
//

#include <memory>
#include <utility>

#include "tools/scaleout/fabric_manager/coordination/controller.hpp"
#include "tools/scaleout/fabric_manager/coordination/transport.hpp"

namespace tt::scaleout_tools::fabric_manager {

class InProcessTransport final : public ControllerTransport {
public:
    explicit InProcessTransport(std::shared_ptr<Controller> controller) : controller_(std::move(controller)) {}

    [[nodiscard]] std::vector<Bytes> exchange(
        const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload) override {
        return controller_->exchange(scope, epoch, index, count, payload);
    }

private:
    std::shared_ptr<Controller> controller_;
};

}  // namespace tt::scaleout_tools::fabric_manager
