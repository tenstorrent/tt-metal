<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# WIP: fabric-level integration

**Does not build.** Kept to document where the boundary falls above the parent demos.

`hello_fabric_x280.cpp` tries freedom-metal + tt-fabric device code
(`LowLatencyPacketHeader`, `routing_encoding::encode_1d_*`, `ChannelBufferPointer`)
with `port_shim/` replacing three host/sfpi-only headers.

## Why it stops

`fabric_edm_packet_header.hpp` → `edm_fabric_utils.hpp` → `dataflow_api.h`, which
expects the full JIT launch environment:

- Per-launch defines: `NOC_INDEX`, `NOC_MODE`, bank counts/logs, `PCIE_NOC_*`,
  `PROGRAMMABLE_CORE_TYPE`.
- Firmware globals: `rta_l1_base`, `crta_l1_base`, `sem_l1_base`, bank→NOC maps.

Packet-header/routing code is portable; the include graph is not. Needs a
launch-environment shim or upstream split of layout headers from `dataflow_api.h`.

Also: `LowLatencyPacketHeader hdr{}` uses a deleted default constructor.

## vs working demos

Parent demos stop at the cache layer (`risc_common.h`), which has no NOC dependency.
