# Mesh (2D) Fabric API — `mesh/api.h`

Entry points for 2D mesh topology. Namespace: `tt::tt_fabric::mesh::experimental`.
Requires `FABRIC_2D` compile flag. Routes via `(dst_dev_id, dst_mesh_id)`.

## Unicast Entry Points

| Entry Point                                          | Auto-Packetize | AddrGen | Conn Mgr | Variants                    | Silicon Test (2D)                                        | AddrGen Test (2D)                                     |
|------------------------------------------------------|----------------|---------|----------|-----------------------------|----------------------------------------------------------|-------------------------------------------------------|
| `fabric_unicast_noc_unicast_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationUnicastWriteSilicon*`                  | `*AddrgenComprehensiveTest*UnicastWrite*`              |
| `fabric_unicast_noc_scatter_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationUnicastScatterSilicon*`                | `*AddrgenComprehensiveTest*ScatterWrite*`              |
| `fabric_unicast_noc_fused_unicast_with_atomic_inc`   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationUnicastFusedAtomicIncSilicon*`         | `*AddrgenComprehensiveTest*FusedAtomicIncWrite*`       |
| `fabric_unicast_noc_fused_scatter_write_atomic_inc`  | yes            | no      | yes      | `_with_state`, `_set_state` | `*AutoPacketizationUnicastFusedScatterAtomicIncSilicon*`  | —                                                     |
| `fabric_unicast_noc_unicast_atomic_inc`              | N/A (no data)  | N/A     | yes      | `_with_state`, `_set_state` | —                                                        | —                                                     |
| `fabric_unicast_noc_unicast_inline_write`            | N/A (inline)   | N/A     | yes      | `_with_state`, `_set_state` | —                                                        | —                                                     |

## Multicast Entry Points

| Entry Point                                            | Auto-Packetize | AddrGen | Conn Mgr | Variants                    | Silicon Test (2D)                                            | AddrGen Test (2D)                                          |
|--------------------------------------------------------|----------------|---------|----------|-----------------------------|--------------------------------------------------------------|------------------------------------------------------------|
| `fabric_multicast_noc_unicast_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationMulticastWriteSilicon*`                    | `*AddrgenComprehensiveTest*MulticastWrite*`                 |
| `fabric_multicast_noc_scatter_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationMulticastScatterSilicon*`                  | `*AddrgenComprehensiveTest*MulticastScatterWrite*`          |
| `fabric_multicast_noc_fused_unicast_with_atomic_inc`   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationMulticastFusedAtomicIncSilicon*`           | `*AddrgenComprehensiveTest*MulticastFusedAtomicIncWrite*`   |
| `fabric_multicast_noc_fused_scatter_write_atomic_inc`  | yes            | yes     | yes      | `_with_state`, `_set_state` | `*AutoPacketizationMulticastFusedScatterAtomicIncSilicon*`    | —                                                          |
| `fabric_multicast_noc_unicast_atomic_inc`              | N/A (no data)  | N/A     | yes      | `_with_state`, `_set_state` | —                                                            | —                                                          |
| `fabric_multicast_noc_unicast_inline_write`            | N/A (inline)   | N/A     | yes      | `_with_state`, `_set_state` | —                                                            | —                                                          |

## Notes

- **Auto-Packetize**: Payloads > `FABRIC_MAX_PACKET_SIZE` are automatically chunked. `_single_packet` variants skip this.
- **AddrGen**:  Overloads accepting `AddrGenType` template parameter (supports `InterleavedAddrGen`, `InterleavedAddrGenFast`, `ShardedAddrGen`, `TensorAccessor`). Defined via `mesh/addrgen_api.h` (which re-exports `linear/addrgen_api.h`).
- **Conn Mgr**: Overloads accepting `RoutingPlaneConnectionManager&` for multi-route parallel dispatch.
- **Variants**: `_with_state` reuses previously set route/header state. `_set_state` sets state without sending. Both available in raw + conn_mgr + addrgen overloads.
- **N/A (no data)**: `atomic_inc` has no data payload — auto-packetization and addrgen are not applicable.
- **N/A (inline)**: `inline_write` carries small payload in header — auto-packetization and addrgen are not applicable.

## Test Binaries

| Test Suite                           | Binary                       | Filter                                                    |
|--------------------------------------|------------------------------|-----------------------------------------------------------|
| Auto-packetization silicon (2D)      | `fabric_unit_tests`          | `--gtest_filter="*AutoPacketization*Silicon*"`             |
| Auto-packetization compile-only (2D) | `fabric_unit_tests`          | `--gtest_filter="*CompileOnlyAutoPacketization2D*"`        |
| AddrGen comprehensive (2D)           | `fabric_addrgen_write_tests` | `--gtest_filter="*AddrgenComprehensiveTest*"`              |
| ConnMgr AddrGen (2D)                 | `fabric_addrgen_write_tests` | `--gtest_filter="*AddrgenComprehensiveTest*ConnMgr*"`      |
