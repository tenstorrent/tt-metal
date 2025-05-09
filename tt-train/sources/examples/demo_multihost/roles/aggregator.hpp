// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

namespace roles {
class Aggregator {
public:
    void aggregate_gradients(std::span<int> workers_group);
    void send_gradients(int optimizer_rank);
    void receive_weights(int optimizer_rank);
    void broadcast_weights(std::span<int> workers_group);
};
}  // namespace roles
