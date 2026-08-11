<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# WIP: fabric-level integration

**This does not build.** It is kept because it maps out where the boundary
actually falls, one layer above the working demo in the parent directory.

`hello_fabric_x280.cpp` tries to exercise real tt-fabric device code under
freedom-metal — `LowLatencyPacketHeader` layout, `routing_encoding::encode_1d_*`,
and `ChannelBufferPointer` credit arithmetic — with `port_shim/` supplying
bare-metal replacements for three host-only or sfpi-only headers
(`risc_attribs.h`, `tt_stl/assert.hpp`, `api/debug/assert.h`).

## Why it stops

`fabric_edm_packet_header.hpp` includes `edm_fabric_utils.hpp`, which includes
`dataflow_api.h`. That header expects the full environment tt-metal's JIT
generates per kernel, not just headers:

- **Compile-time constants injected per launch:** `NOC_INDEX`, `NOC_MODE`,
  `NUM_DRAM_BANKS`, `NUM_L1_BANKS`, `LOG_BASE_2_OF_NUM_DRAM_BANKS`,
  `LOG_BASE_2_OF_NUM_L1_BANKS`, `PCIE_NOC_X`, `PCIE_NOC_Y`,
  `PROGRAMMABLE_CORE_TYPE`. Supplying plausible values is easy; supplying
  *correct* ones means reproducing the HAL's per-device computation.
- **Globals defined in firmware, not headers:** `rta_l1_base`, `crta_l1_base`,
  `sem_l1_base`, `dram_bank_to_noc_xy`, `bank_to_dram_offset`,
  `l1_bank_to_noc_xy`, `bank_to_l1_offset`. These live in the DM firmware `.cc`
  files and are populated by the dispatcher at launch.

So the packet-header and routing-encoder code being tested is portable; the
`#include` graph reaching it is not. Making this work means either providing a
launch-environment shim (the JIT-generated defines plus stub globals), or teasing
the layout/encoding headers apart from `dataflow_api.h` upstream.

`hello_fabric_x280.cpp` also has one real bug independent of the above:
`LowLatencyPacketHeader hdr{}` at line 152 uses a deleted default constructor.

## Relationship to the working demo

The parent demo deliberately stops at the cache layer, which is exactly the layer
`risc_common.h` documents as X280-derived, and which has no NOC dependency. That
is why it builds and this does not.
