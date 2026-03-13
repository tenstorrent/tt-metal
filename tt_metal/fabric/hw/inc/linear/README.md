# Linear (1D) Fabric API — `linear/api.h`

Entry points for 1D linear topology. Namespace: `tt::tt_fabric::linear::experimental`.
Routes via `num_hops` (unicast) or `start_distance`/`range` (multicast). Supports `FABRIC_2D` compile flag for 2D portability.

## Unicast Entry Points

| Entry Point                                          | Auto-Packetize | AddrGen | Conn Mgr | Variants                    | 1D Silicon Test                                   | 1D->2D Portability                                                      | AddrGen Test (1D)                                    |
|------------------------------------------------------|----------------|---------|----------|-----------------------------|---------------------------------------------------|-------------------------------------------------------------------------|------------------------------------------------------|
| `fabric_unicast_noc_unicast_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*LinearUnicastWriteSilicon*`                     | `*AutoPacketizationUnicastWriteSilicon*` (2D fixture)                    | `*AddrgenLinear1DTest*LinearUnicastWrite*`            |
| `fabric_unicast_noc_scatter_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*LinearUnicastScatterSilicon*`                   | `*AutoPacketizationUnicastScatterSilicon*` (2D fixture)                  | `*AddrgenLinear1DTest*LinearScatterWrite*`            |
| `fabric_unicast_noc_fused_unicast_with_atomic_inc`   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*LinearUnicastFusedAtomicIncSilicon*`            | `*AutoPacketizationUnicastFusedAtomicIncSilicon*` (2D fixture)           | `*AddrgenLinear1DTest*LinearFusedAtomicIncWrite*`     |
| `fabric_unicast_noc_fused_scatter_write_atomic_inc`  | yes            | no      | yes      | `_with_state`, `_set_state` | `*LinearUnicastFusedScatterAtomicIncSilicon*`     | `*AutoPacketizationUnicastFusedScatterAtomicIncSilicon*` (2D fixture)    | —                                                    |
| `fabric_unicast_noc_unicast_atomic_inc`              | N/A (no data)  | N/A     | yes      | `_with_state`, `_set_state` | `*TestLinearFabricUnicastNocAtomicInc*`           | —                                                                       | —                                                    |
| `fabric_unicast_noc_unicast_inline_write`            | N/A (inline)   | N/A     | yes      | `_with_state`, `_set_state` | `*TestLinearFabricUnicastNocInlineWrite*`         | —                                                                       | —                                                    |

## Multicast Entry Points

| Entry Point                                            | Auto-Packetize | AddrGen | Conn Mgr | Variants                    | 1D Silicon Test                                     | 1D->2D Portability                                                        | AddrGen Test (1D)                                          |
|--------------------------------------------------------|----------------|---------|----------|-----------------------------|-----------------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------|
| `fabric_multicast_noc_unicast_write`                   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*LinearMulticastWriteSilicon*`                     | `*AutoPacketizationMulticastWriteSilicon*` (2D fixture)                    | `*AddrgenLinear1DTest*LinearMulticastWrite*`                |
| `fabric_multicast_noc_scatter_write`                   | yes            | no      | yes      | `_with_state`, `_set_state` | `*LinearMulticastScatterSilicon*`                   | `*AutoPacketizationMulticastScatterSilicon*` (2D fixture)                  | `*AddrgenLinear1DTest*LinearMulticastScatterWrite*`         |
| `fabric_multicast_noc_fused_unicast_with_atomic_inc`   | yes            | yes     | yes      | `_with_state`, `_set_state` | `*LinearMulticastFusedAtomicIncSilicon*`            | `*AutoPacketizationMulticastFusedAtomicIncSilicon*` (2D fixture)           | `*AddrgenLinear1DTest*LinearMulticastFusedAtomicIncWrite*`  |
| `fabric_multicast_noc_fused_scatter_write_atomic_inc`  | yes            | no      | yes      | `_with_state`, `_set_state` | `*LinearMulticastFusedScatterAtomicIncSilicon*`     | `*AutoPacketizationMulticastFusedScatterAtomicIncSilicon*` (2D fixture)    | —                                                          |
| `fabric_multicast_noc_unicast_atomic_inc`              | N/A (no data)  | N/A     | yes      | `_with_state`, `_set_state` | `*TestLinearFabricMulticastNocAtomicInc*`           | —                                                                         | —                                                          |
| `fabric_multicast_noc_unicast_inline_write`            | N/A (inline)   | N/A     | yes      | `_with_state`, `_set_state` | `*TestLinearFabricMulticastNocInlineWrite*`         | —                                                                         | —                                                          |

## Sparse Multicast Entry Points (1D only)

| Entry Point                                    | Auto-Packetize | AddrGen | Conn Mgr | Variants | 1D Silicon Test                                                     | 1D->2D Portability | AddrGen Test (1D) |
|------------------------------------------------|----------------|---------|----------|----------|---------------------------------------------------------------------|--------------------|--------------------|
| `fabric_sparse_multicast_noc_unicast_write`    | yes            | no      | yes      | —        | `*AutoPacketizationSparseMulticastSilicon*` (SKIPPED: issue #36581) | N/A (1D only)      | —                  |

## Notes

- **Auto-Packetize**: Payloads > `FABRIC_MAX_PACKET_SIZE` are automatically chunked. `_single_packet` variants skip this.
- **AddrGen**:  Overloads accepting `AddrGenType` template parameter (supports `InterleavedAddrGen`, `InterleavedAddrGenFast`, `ShardedAddrGen`, `TensorAccessor`). Defined in `linear/addrgen_api.h`.
- **Conn Mgr**: Overloads accepting `RoutingPlaneConnectionManager&` for multi-route parallel dispatch.
- **Variants**: `_with_state` reuses previously set route/header state. `_set_state` sets state without sending. Both available in raw + conn_mgr + addrgen overloads.
- **1D->2D Portability**: The linear API's `fabric_set_unicast_route` and `fabric_set_mcast_route` contain `#if defined(FABRIC_2D)` branches that handle 2D mesh routing when compiled with `FABRIC_2D`. The 2D fixture tests validate this portability path using the same kernel code compiled with the `FABRIC_2D` flag.
- **SparseMulticast**: Silicon test GTEST_SKIP'd due to firmware limitation (Ethernet core lockup, issue #36581). Test infrastructure is complete; blocked on firmware fix.

## Test Binaries

| Test Suite                           | Binary                       | Filter                                                     |
|--------------------------------------|------------------------------|------------------------------------------------------------|
| Auto-packetization silicon (1D)      | `fabric_unit_tests`          | `--gtest_filter="*AutoPacketizationLinear*Silicon*"`       |
| Auto-packetization compile-only (1D) | `fabric_unit_tests`          | `--gtest_filter="*CompileOnlyAutoPacketization1D*"`        |
| 1D basic API (conn mgr)              | `fabric_unit_tests`          | `--gtest_filter="*TestLinearFabric*"`                      |
| 1D sparse multicast                  | `fabric_unit_tests`          | `--gtest_filter="*TestLinearFabricSparseMulticast*"`       |
| AddrGen 1D                           | `fabric_unit_tests` | `--gtest_filter="*AddrgenLinear1DTest*"`                   |
| AddrGen 1D conn mgr                  | `fabric_unit_tests` | `--gtest_filter="*AddrgenLinear1DTest*ConnMgr*"`           |
