// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/pause.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.h"

void kernel_main() {
    uint32_t lite_fabric_config_addr = get_arg_val<uint32_t>(0);

    tunneling::do_init_and_handshake_sequence(lite_fabric_config_addr);
}
