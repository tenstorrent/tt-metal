// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace roles {
class Optimizer {
public:
    void optimization_step();
    void send_weights(int aggregator_rank);
};

}  // namespace roles
