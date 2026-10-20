// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_peer_kernel.hpp"

void kernel_main() { p2p_latency::run_metal_peer_kernel(); }
