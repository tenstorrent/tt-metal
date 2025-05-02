// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/hw/inc/socket_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t socket_config_addr = get_compile_time_arg_val(0);
    volatile tt_l1_ptr receiver_socket_md* receiver_md =
        reinterpret_cast<volatile tt_l1_ptr receiver_socket_md*>(socket_config_addr);
    DPRINT << "Receiver socket config addr: " << socket_config_addr << ENDL();
    DPRINT << "Receiver socket config bytes_sent:" << receiver_md->bytes_sent << ENDL();
    DPRINT << "Receiver socket config bytes_acked:" << receiver_md->bytes_acked << ENDL();
    DPRINT << "Receiver socket config read_ptr:" << receiver_md->read_ptr << ENDL();
    DPRINT << "Receiver socket config fifo_addr:" << receiver_md->fifo_addr << ENDL();
    DPRINT << "Receiver socket config fifo_size:" << receiver_md->fifo_total_size << ENDL();
    DPRINT << "Receiver socket config upstream_mesh_id:" << receiver_md->upstream_mesh_id << ENDL();
    DPRINT << "Receiver socket config upstream_chip_id:" << receiver_md->upstream_chip_id << ENDL();
    DPRINT << "Receiver socket config upstream_noc_y:" << receiver_md->upstream_noc_y << ENDL();
    DPRINT << "Receiver socket config upstream_noc_x:" << receiver_md->upstream_noc_x << ENDL();
    DPRINT << "Receiver socket config upstream_bytes_acked_addr:" << receiver_md->upstream_bytes_acked_addr << ENDL();
    DPRINT << "Receiver socket config is_sender:" << receiver_md->is_sender << ENDL();
}
