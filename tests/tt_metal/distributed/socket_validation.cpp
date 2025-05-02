// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/hw/inc/socket_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t socket_config_addr = get_compile_time_arg_val(0);
    volatile tt_l1_ptr sender_socket_md* sender_md =
        reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(socket_config_addr);
    DPRINT << "Sender socket config addr: " << socket_config_addr << ENDL();
    DPRINT << "Sender socket config bytes_acked: " << sender_md->bytes_acked << ENDL();
    DPRINT << "Sender socket config write_ptr: " << sender_md->write_ptr << ENDL();
    DPRINT << "Sender socket config bytes_sent: " << sender_md->bytes_sent << ENDL();
    DPRINT << "Sender socket config downstream_mesh_id: " << sender_md->downstream_mesh_id << ENDL();
    DPRINT << "Sender socket config downstream_chip_id: " << sender_md->downstream_chip_id << ENDL();
    DPRINT << "Sender socket config downstream_noc_x: " << sender_md->downstream_noc_x << ENDL();
    DPRINT << "Sender socket config downstream_noc_y: " << sender_md->downstream_noc_y << ENDL();
    DPRINT << "Sender socket config downstream_bytes_sent_addr: " << sender_md->downstream_bytes_sent_addr << ENDL();
    DPRINT << "Sender socket config downstream_fifo_addr: " << sender_md->downstream_fifo_addr << ENDL();
    DPRINT << "Sender socket config downstream_fifo_total_size: " << sender_md->downstream_fifo_total_size << ENDL();
    DPRINT << "Sender socket config is_sender: " << sender_md->is_sender << ENDL();
}
