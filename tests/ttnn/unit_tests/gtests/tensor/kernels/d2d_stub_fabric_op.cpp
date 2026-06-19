// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stub competing fabric op for the cross-process D2D chain test. Stands in for a
// model-graph fabric op (e.g. a row CCL) that contends with the D2D sender service
// for the EDM sender channel toward the downstream mesh.
//
// It opens a WorkerToFabricEdmSender on the SAME link the D2D sender service uses
// (the host builds the connection RT args from the same sender→downstream
// FabricNodeId pair + link index) and immediately closes it — becoming the single
// connected client on that channel for the duration. No payload is sent; connection
// ownership is the contention point ("the EDM allows only one connected client per
// sender channel").
//
// The host launches this ONLY between wait_for_fabric_links() (D2D sender confirmed
// off the link) and release_fabric_links() (grant it back). So it exercises the
// lease: if the lease were broken (the D2D sender still connected), this open would
// contend with it. And because it sits after the producing op, the op's overwrite of
// the outbound backing tensor is protected solely by the d2d_sync gate.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    fabric_connection.open();
    fabric_connection.close();
}
