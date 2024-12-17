
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Routing table structs used by FabricRouter
// Packed and written to device by ControlPlane
// Consumed in router FW

#pragma once
#include <stdint.h>

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/hw/inc/risc_attribs.h"
#else
#define tt_l1_ptr
#define tt_reg_ptr
#define FORCE_INLINE inline
#endif

namespace tt::tt_fabric {

using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t MAX_MESH_SIZE = 1024;
static constexpr std::uint32_t MAX_NUM_MESHES = 1024;

static constexpr std::uint32_t NUM_CHANNELS_PER_UINT32 = sizeof(std::uint32_t) / sizeof(chan_id_t);
static constexpr std::uint32_t LOG_BASE_2_NUM_CHANNELS_PER_UINT32 = 2;
static constexpr std::uint32_t MODULO_LOG_BASE_2 = (1 << LOG_BASE_2_NUM_CHANNELS_PER_UINT32) - 1;
static constexpr std::uint32_t NUM_TABLE_ENTRIES = MAX_MESH_SIZE >> LOG_BASE_2_NUM_CHANNELS_PER_UINT32;

static_assert(MAX_MESH_SIZE == MAX_NUM_MESHES, "MAX_MESH_SIZE must be equal to MAX_NUM_MESHES");
static_assert((sizeof(std::uint32_t) / sizeof(chan_id_t)) == NUM_CHANNELS_PER_UINT32, "LOG_BASE_2_NUM_CHANNELS_PER_UINT32 must be equal to log2(sizeof(std::uint32_t) / sizeof(chan_id_t))");

enum eth_chan_magic_values {
    INVALID_DIRECTION = 0xDD,
    INVALID_ROUTING_TABLE_ENTRY = 0xFF,
};

struct routing_table_t {
  chan_id_t dest_entry[MAX_MESH_SIZE];
};

struct port_direction_t {
  chan_id_t north;
  chan_id_t south;
  chan_id_t east;
  chan_id_t west;
};

struct fabric_router_l1_config_t {
    routing_table_t intra_mesh_table;
    routing_table_t inter_mesh_table;
    port_direction_t port_direction;
    std::uint16_t my_mesh_id;  // Do we need this if we tag routing tables with magic values for outbound eth channels
                               // and route to local NOC?
    std::uint16_t my_device_id;
} __attribute__((packed));

/*
FORCE_INLINE std::uint8_t get_eth_chan_from_table(std::uint32_t base_addr, std::uint32_t target_id) {
    //  i.e. intra mesh table:
    //  0xffffffee  to get to chip 3,2,1,0
    //  0xffffffff  to get to chip 7,6,5,4
    //  0x5050505   to get to chip 11,10,9,8
    //
    //  0xff denotes outgoing channel, 0xee denotes route to itself
    return ((*(std::uint32_t* tt_l1_ptr)(base_addr + target_id >> LOG_BASE_2_NUM_CHANNELS_PER_UINT32)) >> (target_id &
MODULO_LOG_BASE_2)) & 0xFF;
};*/
/*
void pack_routing_table(routing_table_t* table, const chan_id_t* data, uint32_t len) {
  // Loop over data and pack into routing_table_t of Vdchan_id_t
  for (uint32_t i = 0; i < len; i++) {
      table->dest_entry[i] |= (data[i] << ((i & MODULO_LOG_BASE_2) * sizeof(chan_id_t) * 8));
  }
}*/

}  // namespace tt::tt_fabric
