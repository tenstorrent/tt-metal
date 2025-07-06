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

    volatile tunneling::lite_fabric_config_t* lite_fabric_config =
        reinterpret_cast<volatile tunneling::lite_fabric_config_t*>(lite_fabric_config_addr);

    DPRINT << "multi_eth_cores_setup " << (uint32_t)lite_fabric_config->multi_eth_cores_setup << ENDL();

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
    uint32_t local_neighbour_handshake_addr =
        lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, local_neighbour_handshake);
    uint32_t remote_neighbour_handshake_addr =
        lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, remote_neighbour_handshake);

    DPRINT << "local_neighbour_handshake_addr: " << HEX() << local_neighbour_handshake_addr
           << " remote_neighbour_handshake_addr: " << HEX() << remote_neighbour_handshake_addr << DEC() << ENDL();

    volatile tt_l1_ptr uint32_t* local_neighbour_handshake_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_neighbour_handshake_addr);
    lite_fabric_config->remote_neighbour_handshake = 0xFEEDE145;

    // capture the initial state because it will be changed when sending it to neighbouring eths/subordinate eths
    tunneling::LiteFabricInitState initial_state = lite_fabric_config->init_state;
    tunneling::LiteFabricInitState state = initial_state;
    while (state != tunneling::LiteFabricInitState::DONE) {
        invalidate_l1_cache();
        switch (state) {
            case tunneling::LiteFabricInitState::MMIO_ETH_INIT_NEIGHBOUR: {
                // first send the rt args and config to the remote core
                // clobber the initial state arg with the state that remote eth core should come up in
                lite_fabric_config->init_state = lite_fabric_config->multi_eth_cores_setup
                                                     ? tunneling::LiteFabricInitState::NON_MMIO_ETH_INIT_LOCAL_ETHS
                                                     : tunneling::LiteFabricInitState::NEIGHBOUR_HANDSHAKE;
                internal_::eth_send_packet<false>(
                    0,
                    lite_fabric_config_addr >> 4,
                    lite_fabric_config_addr >> 4,
                    sizeof(tunneling::lite_fabric_config_t) >> 4);
                DPRINT << "Sent lite_fabric_config to " << HEX() << lite_fabric_config_addr << DEC() << ENDL();

                internal_::eth_send_packet<false>(
                    0, rt_arg_base_addr >> 4, rt_arg_base_addr >> 4, 1024 >> 4);  // just send all rt args
                DPRINT << "Sent runtime args to " << HEX() << rt_arg_base_addr << DEC() << ENDL();

                // send the kernel binary
                internal_::eth_send_packet<false>(
                    0,
                    lite_fabric_config->binary_address >> 4,
                    lite_fabric_config->binary_address >> 4,
                    lite_fabric_config->binary_size_bytes >> 4);
                DPRINT << "Sent binary to " << HEX() << lite_fabric_config->binary_address << DEC() << " size is "
                       << lite_fabric_config->binary_size_bytes << ENDL();

                // send launch and go message
                internal_::eth_send_packet<false>(
                    0, launch_msg_addr >> 4, launch_msg_addr >> 4, launch_and_go_msg_size_bytes >> 4);
                DPRINT << "Sent launch/go msg to 0x" << HEX() << launch_msg_addr << DEC() << " of size "
                       << launch_and_go_msg_size_bytes << " go msg addr " << HEX()
                       << (launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries)) << DEC() << ENDL();

                state = lite_fabric_config->multi_eth_cores_setup ? tunneling::LiteFabricInitState::LOCAL_HANDSHAKE
                                                                  : tunneling::LiteFabricInitState::NEIGHBOUR_HANDSHAKE;
                ;
                DPRINT << "going to next state: " << (uint32_t)state << ENDL();
                break;
            }
            case tunneling::LiteFabricInitState::LOCAL_HANDSHAKE: {
                // go over the eth chan header and do case 0 for all cores using noc writes
                // set additional rt args in the kernel that are the x-y of this core

                uint32_t primary_local_handshake_addr =
                    lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, primary_local_handshake);
                uint32_t subordinate_local_handshake_addr =
                    lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, subordinate_local_handshake);

                DPRINT << "primary_local_handshake_addr: " << HEX() << primary_local_handshake_addr
                       << " subordinate_local_handshake_addr: " << HEX() << subordinate_local_handshake_addr << DEC()
                       << ENDL();

                if (initial_state == tunneling::LiteFabricInitState::MMIO_ETH_INIT_NEIGHBOUR or
                    initial_state == tunneling::LiteFabricInitState::NON_MMIO_ETH_INIT_LOCAL_ETHS) {
                    uint32_t remaining_cores = lite_fabric_config->eth_chans_mask;
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

                    while (lite_fabric_config->primary_local_handshake != lite_fabric_config->num_local_eths - 1) {
                        // wait for the subordinate eth cores to send us a handshake signal
                        invalidate_l1_cache();
                    }

                } else {
                    noc_semaphore_inc(
                        get_noc_addr(
                            lite_fabric_config->primary_eth_core_x,
                            lite_fabric_config->primary_eth_core_y,
                            primary_local_handshake_addr),
                        1);
                    while (lite_fabric_config->subordinate_local_handshake != 1) {
                        // wait for the primary eth core
                        invalidate_l1_cache();
                    }
                }

                state = tunneling::LiteFabricInitState::NEIGHBOUR_HANDSHAKE;
                break;
            }
            case tunneling::LiteFabricInitState::NON_MMIO_ETH_INIT_LOCAL_ETHS: {
                // update primary_eth_core_x and primary_eth_core_y with this core's virtual x and y
                lite_fabric_config->init_state = tunneling::LiteFabricInitState::LOCAL_HANDSHAKE;
                lite_fabric_config->primary_local_handshake = 0;
                lite_fabric_config->subordinate_local_handshake = 0;

                if (lite_fabric_config->multi_eth_cores_setup) {
                    uint32_t remaining_cores = lite_fabric_config->eth_chans_mask;
                    for (uint32_t i = 0; i < total_num_eths; i++) {
                        if (remaining_cores == 0) {
                            break;
                        }
                        if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                            uint64_t dest_config_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], lite_fabric_config_addr);
                            uint64_t dest_rt_args_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], rt_arg_base_addr);
                            uint64_t dest_binary_addr = get_noc_addr_helper(
                                eth_chan_to_noc_xy[noc_index][i], lite_fabric_config->binary_address);
                            uint64_t dest_launch_and_go_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], launch_msg_addr);
                            noc_async_write(
                                lite_fabric_config_addr, dest_config_addr, sizeof(tunneling::lite_fabric_config_t));
                            noc_async_write(rt_arg_base_addr, dest_rt_args_addr, 16);
                            noc_async_write(
                                lite_fabric_config->binary_address,
                                dest_binary_addr,
                                lite_fabric_config->binary_size_bytes);
                            noc_async_write(launch_msg_addr, dest_launch_and_go_addr, launch_and_go_msg_size_bytes);
                            remaining_cores &= ~(0x1 << i);
                        }
                    }
                    noc_async_write_barrier();
                }

                state = tunneling::LiteFabricInitState::LOCAL_HANDSHAKE;
                break;
            }
            case tunneling::LiteFabricInitState::NEIGHBOUR_HANDSHAKE: {
                // we can only come into this state if our local handshakes are done or we are initializing one core

                internal_::eth_send_packet<false>(
                    0, remote_neighbour_handshake_addr >> 4, local_neighbour_handshake_addr >> 4, 16 >> 4);

                if (lite_fabric_config->local_neighbour_handshake == 0xFEEDE145) {
                    DPRINT << "done with handshaking" << ENDL();
                    state = tunneling::LiteFabricInitState::DONE;
                }

                break;
            }
            default: ASSERT(false);
        }
    }

    DPRINT << "done init" << ENDL();
}
