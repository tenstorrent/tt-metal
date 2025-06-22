// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

enum LiteFabricInitState : uint8_t {
    MMIO_ETH_INIT_NEIGHBOUR = 0,
    LOCAL_HANDSHAKE = 1,
    NON_MMIO_ETH_INIT_LOCAL_ETHS = 2,
    NEIGHBOUR_HANDSHAKE = 3,
    DONE = 4,
};
