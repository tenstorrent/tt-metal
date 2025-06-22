// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/pause.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.h"

// enum SocketState : uint8_t {
//     IDLE = 0,
//     OPENING = 1,
//     ACTIVE = 2,
//     CLOSING = 3,
// };

void kernel_main() {
    uint32_t initial_state = get_arg_val<uint32_t>(0);
    uint32_t binary_address = get_arg_val<uint32_t>(1);     // only used by eths that set up kernels
    uint32_t binary_size_bytes = get_arg_val<uint32_t>(2);  // only used by eths that set up kernels
    bool multi_eth_cores_setup = get_arg_val<uint32_t>(3) == 1;
    uint32_t primary_eth_core_x = get_arg_val<uint32_t>(4);
    uint32_t primary_eth_core_y = get_arg_val<uint32_t>(5);
    uint32_t eth_chans_mask = get_arg_val<uint32_t>(6);
    uint32_t num_local_eths = get_arg_val<uint32_t>(7);
    // 8, 9 used

    DPRINT << "multi_eth_cores_setup " << (uint32_t)multi_eth_cores_setup << ENDL();

    constexpr uint32_t local_handshake_stream_id = 0;
    constexpr uint32_t remote_handshake_stream_id = 1;  // neighbour eth core will write here
    /*

        NEW - simple case where we only init one eth device
        I-am-mmio-eth-setting-up-remote: 0
        I-am-doing-local-handshake: 1
        I-am-remote-eth-setting-up-local-eths: 2
        I-am-wating-on-neighbour-handshake: 3
        I-am-done-with-handshake: 4
    */

    uint32_t launch_msg_addr = (uint32_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->launch);
    constexpr uint32_t launch_and_go_msg_size_bytes =
        ((sizeof(launch_msg_t) * launch_msg_buffer_num_entries) + sizeof(go_msg_t) + 15) & ~0xF;
    static_assert(launch_and_go_msg_size_bytes % 16 == 0, "Launch and go msg size must be multiple of 16 bytes");
    uint32_t go_msg_addr = launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries);

    constexpr uint32_t total_num_eths = sizeof(eth_chan_to_noc_xy[0]) / sizeof(eth_chan_to_noc_xy[0][0]);

    uint32_t exclude_eth_chan = get_absolute_logical_y();

    uint32_t rt_arg_base_addr = get_arg_addr(0);  // using rt arg space as scratch area for kernel metadata

    // local_neighbour_handshake_addr is where this eth core will check that its neighbour has completed its handshake
    // remote_neighbour_handshake_addr is where this eth core will write value before sending it over link to
    // local_neighbour_handshake_addr
    uint32_t local_neighbour_handshake_addr = get_arg_addr(12);
    uint32_t remote_neighbour_handshake_addr = get_arg_addr(16);

    DPRINT << "local_neighbour_handshake_addr: " << HEX() << local_neighbour_handshake_addr
           << " remote_neighbour_handshake_addr: " << HEX() << remote_neighbour_handshake_addr << DEC() << ENDL();

    volatile tt_l1_ptr uint32_t* local_neighbour_handshake_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_neighbour_handshake_addr);
    volatile tt_l1_ptr uint32_t* remote_neighbour_handshake_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_neighbour_handshake_addr);
    *remote_neighbour_handshake_ptr = 0xFEEDE145;

    uint32_t state = initial_state;
    while (state != 4) {
        invalidate_l1_cache();
        switch (state) {
            case 0: {
                // clobber the initial_state arg with the state that remote eth core should come up in
                // first send the rt args to the remote core
                // set additional rt args in the kernel that are the x-y of this core
                rta_l1_base[0] = multi_eth_cores_setup ? 2 : 3;
                rta_l1_base[10] = 0;  // mmio eth cores are the handshake initializers
                internal_::eth_send_packet<false>(0, rt_arg_base_addr >> 4, rt_arg_base_addr >> 4, 1024 >> 4); // just send all rt args for now

                DPRINT << "Sent runtime args to " << HEX() << rt_arg_base_addr << DEC() << ENDL();

                // send the kernel binary
                internal_::eth_send_packet<false>(0, binary_address >> 4, binary_address >> 4, binary_size_bytes >> 4);

                DPRINT << "Sent binary to " << HEX() << binary_address << DEC() << ENDL();

                // send launch, don't send go msg here because + go msg ... same as what was written on this core
                internal_::eth_send_packet<false>(
                    0, launch_msg_addr >> 4, launch_msg_addr >> 4, launch_and_go_msg_size_bytes >> 4);

                DPRINT << "Sent launch/go msg to 0x" << HEX() << launch_msg_addr << DEC() << " of size "
                       << launch_and_go_msg_size_bytes << " go msg addr " << HEX()
                       << (launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries)) << DEC() << ENDL();

                uint32_t next_state = multi_eth_cores_setup ? 1 : 3;
                DPRINT << "going to next state: " << next_state << ENDL();
                state = next_state;
                break;
            }
            case 1: {
                // go over the eth chan header and do case 0 for all cores using noc writes
                // set additional rt args in the kernel that are the x-y of this core

                uint32_t primary_local_handshake_addr = get_arg_addr(8);  // for mmio devices make host clear these
                uint32_t subordinate_local_handshake_addr = get_arg_addr(9);

                DPRINT << "primary_local_handshake_addr: " << HEX() << primary_local_handshake_addr
                       << " subordinate_local_handshake_addr: " << HEX() << subordinate_local_handshake_addr << DEC()
                       << ENDL();

                volatile tt_l1_ptr uint32_t* primary_handshake_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(primary_local_handshake_addr);
                volatile tt_l1_ptr uint32_t* subordinate_handshake_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(subordinate_local_handshake_addr);

                if (initial_state == 0 or initial_state == 2) {
                    uint32_t remaining_cores = eth_chans_mask;
                    for (uint32_t i = 0; i < total_num_eths; i++) {
                        if (remaining_cores == 0) {
                            break;
                        }
                        if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                            uint64_t dest_handshake_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], subordinate_local_handshake_addr);
                            noc_semaphore_inc(dest_handshake_addr, 1);
                            remaining_cores &= ~(0x1 << i);
                        }
                    }

                    while (*primary_handshake_ptr != num_local_eths - 1) {
                        // wait for the subordinate eth cores to send us a handshake signal
                        invalidate_l1_cache();
                    }

                } else {
                    noc_semaphore_inc(get_noc_addr(primary_eth_core_x, primary_eth_core_y, primary_local_handshake_addr), 1);
                    while (*subordinate_handshake_ptr != 1) {
                        // wait for the primary eth core
                        invalidate_l1_cache();
                    }
                }

                state = 3;
                break;
            }
            case 2: {
                rta_l1_base[0] = 1;
                rta_l1_base[8] = 0;  // clear where subordinate eths will signal
                rta_l1_base[9] = 0;  // clear where primary eth will signal to subordinates

                uint32_t remaining_cores = eth_chans_mask;
                for (uint32_t i = 0; i < total_num_eths; i++) {
                    if (remaining_cores == 0) {
                        break;
                    }
                    if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                        uint64_t dest_rt_args_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], rt_arg_base_addr);
                        uint64_t dest_binary_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], binary_address);
                        uint64_t dest_launch_and_go_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], launch_msg_addr);
                        noc_async_write(rt_arg_base_addr, dest_rt_args_addr, 1024);
                        noc_async_write(binary_address, dest_binary_addr, binary_size_bytes);
                        noc_async_write(launch_msg_addr, dest_launch_and_go_addr, launch_and_go_msg_size_bytes);
                        remaining_cores &= ~(0x1 << i);
                    }
                }
                noc_async_write_barrier();

                state = 1;
                break;
            }
            case 3: {
                // we can only come into this state if our local handshakes are done or we are initializing one core

                internal_::eth_send_packet<false>(
                    0, remote_neighbour_handshake_addr >> 4, local_neighbour_handshake_addr >> 4, 16 >> 4);

                if (*local_neighbour_handshake_ptr == 0xFEEDE145) {
                    DPRINT << "done with handshaking" << ENDL();
                    state = 4;
                }

                break;
            }
            default: ASSERT(false);
        }
    }

    DPRINT << "done init" << ENDL();
}
