# v4 express / Z: port note

Status (2026-10-29): validated, no router change (Galaxy Slurm job 162896; RESULTS §98 "Express / Z on v4"): the
express sanity groups pass on v4tel as on main, both descriptors, two ERISCs and one (120/120 and 72/72 each). v4trm's
express images are identical to v4tel's.
Sources: main's router and builder at the merge (6d8a103; express routing is #57785), V4_FEATURE_SPEC.md §4,
`tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_express_routing_sanity_common.yaml`.

## 1. What express is to a router

Express links are physical Y-axis chords inside a mesh (on one BH Galaxy: rows 2 <-> 5 of each column; the cluster
descriptor has 68 chip pairs = the 64 of the 8x4 torus + these 4). The host models them as a fifth direction, Z.
Nothing selects express in the kernel; it shows only in the shape of a router's config:
- every router of an express mesh has 5 VC0 sender channels (worker + up to 4 forwarding; channel 4 is the Z ingress
  on cardinal routers);
- the Z-facing routers have `MY_DIRECTION` = 4 and up to 4 downstreams (E, W, N, S);
- the host turns off the speedy path, trim bypass and channel remapping on express meshes, refuses mux, and
  `FABRIC_EXPRESS_ENABLED` + UDM is a static_assert in the packet header.

Main's router has no other express code: decode, downstream tables and credit-stream mapping cover Z by their general
rules (V4_FEATURE_SPEC.md §4).

## 2. v4 against main's rules (offline review)

| Rule (main) | v4 |
| --- | --- |
| Decode: E/W-facing routers read the column's byte; N/S/Z-facing read the row's byte, the column's when it is 0 | `Routing2D::action` (router_routing.hpp): `column_only` for E/W; every other facing, Z included, reads Y then X |
| Action bit d = direction d (Z = 0x10) | static_assert in router_routing.hpp |
| Downstream compact index: directions E, W, N, S, Z without its own | `downstream_direction_table` / `downstream_compact_index` (router_common.hpp), row 4 = {E, W, N, S} |
| Live downstreams: E/W routers inside a mesh drop N and S, keep Z | `live_downstreams` (router_downstream.hpp) |
| Credit stream on the downstream: channel `my_direction` if it is above the downstream's direction, else `my_direction + 1` | `downstream_sender_free_slots_stream` (router_common.hpp) |
| Channel 4 visited every pass, its register masked like the others | v4 reads every serviced channel's register in one group (`channels_with_packets<vc0_channels>`), masked by `stream_count_mask`; no per-channel idle skip |

No change needed. `glx_express` (8x4 torus XY on the express descriptor, 56 configs) built on v4 before this note.

## 3. Shapes and tests on one Galaxy

| Descriptor | Topologies it admits | Sanity groups |
| --- | --- | --- |
| `express_links_8x4_mesh_graph_descriptor.textproto` (8x4 RING x RING, ex4 chords [2,5]) | Mesh, Torus XY (Torus X refused: dim 0 has express links but no end wrap) | `express/glx_express_ring.yaml`: MeshAllToAll, MeshAllToAllIntra, MeshMulticastAllToAllIntra, TorusXYAllToAll, TorusXYAllToAllIntra |
| `fabric_exp/express/express_links_8x4_line_ring_mesh_graph_descriptor.textproto` (one mesh of CI's quad 8x4: dim 0 LINE with the ex4 and row 0 <-> 7 chords, dim 1 RING) | Mesh, Torus X (Torus XY refused) | `express/glx_express_line.yaml`: TorusXAllToAll, TorusXAllToAllIntra, MeshMulticastAllToAllIntra (CI's quad 8x4 groups) |

Each group (24 tests: 4 send types x 1-2 links x 3 sizes, 100 packets, flow control, data-checked) gets past setup on
the mock Galaxy with main's router on two ERISCs; on one ERISC, one group per topology (Mesh, Torus XY, Torus X) does
(`express/probe`). VC1 with express needs several meshes
(CI's quad 8x4 on 4 Galaxies); not run here.

## 4. Validation plan

1. Offline: v4tel builds every config of `glx_express`, `glx_express_mesh`, `glx_express_line` and their one-ERISC
   cases (coverage.py, mock captures with `variants/mainv2cap`).
2. Galaxy (frs_glx_torus.sbatch, suites `xr`, `xl`, `xr1`, `xl1`): main's router first, then v4tel, both
   descriptors, two ERISCs then one.
