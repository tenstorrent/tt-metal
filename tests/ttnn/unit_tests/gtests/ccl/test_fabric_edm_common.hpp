// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

struct WriteThroughputStabilityTestWithPersistentFabricParams {
    size_t line_size = 4;
    size_t num_devices_with_workers = 0;
    bool line_sync = true;
};

void RunWriteThroughputStabilityTestWithPersistentFabric(
    size_t num_mcasts,
    size_t num_unicasts,
    size_t num_links,
    size_t num_op_invocations,
    const WriteThroughputStabilityTestWithPersistentFabricParams& params,
    size_t packet_payload_size_bytes);
