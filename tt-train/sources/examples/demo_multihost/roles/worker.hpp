// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <span>

namespace roles {
class Worker {
public:
    void forward_pass();
    void backward_pass();

    void send_gradients(int aggregator_rank);
    void receive_weights(int aggregator_rank);

private:
};
}  // namespace roles
