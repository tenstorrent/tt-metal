// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>
#include <memory>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>
#include "core_coord.hpp"

namespace tt::tt_metal {

// FabricTensixDatamoverConfig has the global view of a device
class FabricTensixDatamoverConfig {
    // constructor
    // first calculate these fields:
    // num_configs_per_core - based on the size of logical_fabric_mux_cores_,
    //                              and the max number of active ethernet channels from control plane, get the num
    //                              configs = ceil(max_chan / size)
    // num_buffers_per_channel - based on HAL unserved space info, and num_configs_per_core and
    // get_max_payload_size_bytes from fabric context, evenly divide the space
    //
    // buffer_size_bytes_full_size_channel
    // base_l1_address[risc type] - based on risc type (BRISC/NCRISC), can get it from tensix builder,
    //                         brisc start at top unreserved space, ncrisc start at brisc address +
    //                         num_full_size_channels * num_buffers_full_size_channel
    // based on the above, then:
    // create mux configs[risc type]
    // populate the following with correct caculations (might need to get addresses from mux):
    // noc_x/noc_y[eth chan id] - from eth chan id, index into the [eth chan] -> [core, risc_type] mapping, get the
    // core, then get the physical noc address xy.
    //
    // channel_base_address[risc type][tensix channel id]
    // channel_num_buffers[risc type][tensix channel id]
    // channel_local_flow_control_semaphore_address[risc type][tensix channel id]
    // channel_connection_semaphore_address[risc type][tensix channel id]
    // channel_worker_conn_info_base_address[risc type][tensix channel id]
    // channels_worker_conn_info_base_address[risc type][tensix channel id]
    // channel_buffer_slot_size_bytes
    // channels_buffer_index_semaphore_address[risc type][tensix channel id]
    //

    // function member to get the channel related info
    // get_num_configs_per_core()
    // get_num_buffers_per_channel()
    // get_buffer_size_bytes_full_size_channel
    //
    // get_base_l1_address(),

    // get_noc_xy(chan id),

    // get_channels_base_address(tensix channel id) - note this channel id is the mux channel id inside of a mux.

    // more getters for control plane side

private:
    // logical_fabric_tensix_cores_ - this is from core descriptor, simply copied here for ease of use.
    //

    // [eth chan] -> [core, risc_type] mapping, below is breakdown of how to map [eth chan] -> [core] and [eth chan] ->
    // [risc type] [eth chan] -> [core] mapping, based on ethernet channel id, map to core, core is stored in
    // logical_fabric_mux_cores_, for now lets just do a simple round-robin mapping, fabric channels will ordered from
    // small x to large x coord, logical_fabric_mux_cores_ should also be ascending order. then round robin mapping is
    // fine. [eth chan] -> [risc type] mapping, depends on when we map the cores, if there are not enough cores in the
    // mapping, two channel can map to one core, we change from brisc (default core) to ncrisc (new channel mapped to
    // the same core).
    //

}

// FabricTensixDatamoverBuilder only has a local view of the mux kernel it will be building.
/**
 * FabricTensixDatamoverBuilder
 * - Builds mux kernels on fabric tensix cores for worker → mux → fabric router routing.
 * - Build the connections between Fabric routers, fabric router -> mux -> downstream fabric router.
 */
class FabricTensixDatamoverBuilder {
public:
    // constructor FabricTensixDatamoverBuilder
    // args:
    // my_core_logical,
    // local_fabric_node_id
    // remote_fabric_node_id
    // fabric tensix config (contains channel and address info)

    // build, a static member, called from topology to construct a tensix builder.
    // args:
    // device,
    // program
    // local_fabric_node_id
    // remote_fabric_node_id
    // ethernet channel id - used to find the link idx used by tensix config. control_plane.get_routing_plane_id?
    // in function:
    // get the fabric tensix datamover config here (currently ony support mux) from fabric context.
    // Need to pass in RISC type, so need a mapping [eth chan] -> [risc type], it can be done in the tensix datamover
    // builder. from eth chan id, get risc type, noc_x/noc_y based on the provided info, initilize the local fabric mux
    // config

    // class function members.
    // create_and_compile, need to use provate member like get_compile_time_args/get_runtime_args
    // called from create_and_compile_tt_fabric_program in topology.cpp
    // similar to how it handles the fabric router compile, instead we create and compile inside the class.
    // kernel would be tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp
    //

private:
    // class vairables
    // my_core_logical,
    // local_fabric_node_id
    // remote_fabric_node_id
    // link_idx
    // ethernet channel id
    // risc type (brisc/ncrisc)  - need to figure out if we have something similar to RiscType already in metal.
    // noc_x/noc_y
    // fabric mux config

    // class function members.
    // get_compile_time_args - inside will call underlying mux config get_compile_time_args

    // get_runtime_args - inside will call underlying mux config get_runtime_time_args
};

}  // namespace tt::tt_metal
