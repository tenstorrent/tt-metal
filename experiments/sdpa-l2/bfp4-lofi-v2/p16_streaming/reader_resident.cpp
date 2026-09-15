// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Initialize Q/K/V slots once; only FIFO handshakes occur in the steady state.
#include "../resident/reader.cpp"
