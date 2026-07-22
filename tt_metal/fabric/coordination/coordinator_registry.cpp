// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Process-wide registry for the external SystemCoordinator factory (Option (a)).
//
// The factory is set by the fabric-manager tool/service (which owns the concrete
// transport backend) and read by MetalContext when it builds the ControlPlane.
// Keeping the storage here, in tt_metal, means there is a single shared instance
// across the tool binary and libtt_metal.so, and no transport dependency enters
// the core library.

#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

#include <utility>

namespace tt::tt_fabric::coordination {

namespace {
SystemCoordinatorFactory& registry() {
    static SystemCoordinatorFactory instance;
    return instance;
}
}  // namespace

void set_system_coordinator_factory(SystemCoordinatorFactory factory) { registry() = std::move(factory); }

const SystemCoordinatorFactory& get_system_coordinator_factory() { return registry(); }

}  // namespace tt::tt_fabric::coordination
