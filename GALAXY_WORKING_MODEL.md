# Galaxy Working Model

Physical and current-software baseline for the selected Galaxy routing work. This document focuses
on the homogeneous Blackhole (BH) four-Galaxy `[Y,X]=[32,4]` deployment and its consecutive
`[8,4]`, `[16,4]`, and `[24,4]` carve-outs. It is not a catalog of every topology that can be cabled
from four Galaxy systems. Wormhole (WH) Galaxy remains supported by the broader software stack but is
not the primary physical fixture described here.

Authority boundary:

- This document owns the physical context, descriptor/discovery flow, and compact current-code
  baseline.
- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` owns the target routing assumptions, command
  meanings, route oracle, proof, generation policy, and cross-layer invariants.
- The builder, codec, and kernel contracts own their target component details.

Target documents describe intended express-routing support unless they explicitly identify current
behavior. This working model is baseline context, not a product specification.

---

## Chassis

One Galaxy is one host:

| Item | Value |
|---|---|
| Form factor | 6U server |
| Motherboard | `S7T-MB` |
| Trays | 4 UBB trays (tray IDs 1–4) |
| ASICs per tray | 8 (`ASIC_LOCATION_1` … `ASIC_LOCATION_8`) |
| Chips per Galaxy | **32** |

### Board types

| Variant | Board type | Notes |
|---|---|---|
| Wormhole Galaxy | `UBB_WORMHOLE` | Supported, but not the primary new scale-out path |
| Blackhole Galaxy Rev A/B | `UBB_BLACKHOLE` | Tray and port pairing differs from Rev C |
| Blackhole Galaxy Rev C | `UBB_BLACKHOLE` | Tray order differs slightly from Rev A/B; includes PCIe upgrades |

Software detects Rev C from the `board_id` revision bits (`revision >= 3`).

QuietBox (`SIENAD8-2L2T` + `P150`) and similar non-UBB systems are outside this model.

---

## Physical links

These link classes carry Ethernet lanes. Descriptor channel counts are lane counts on a TRACE or
QSFP edge.

| Port or link class | Physical role |
|---|---|
| **TRACE** | On-tray ASIC-to-ASIC connections within one UBB |
| **LINKING_BOARD_1 / 2 / 3** | Connections between trays within one Galaxy chassis |
| **QSFP_DD** | External-facing links used for intra-chassis wraps and inter-Galaxy/inter-pod cabling |

Optional intra-Galaxy torus configurations use the software node types `MESH`, `X_TORUS`,
`Y_TORUS`, and `XY_TORUS`. Available QSFP ports may close X, Y, or both axes inside the chassis.

Channel counts are topology-specific. In-tree descriptors commonly use four lanes per WH edge and
two lanes per BH edge; the selected descriptor remains authoritative.

---

## Scale hierarchy

```text
ASIC
  → Tray (8 ASICs, TRACE)
    → Galaxy / Host (4 trays; linking boards and optional QSFP wraps)
      → Selected pod (4 consecutive Galaxies = 128 chips; inter-Galaxy QSFP)
        → SuperPod (multiple pods)
          → Larger deployment
```

- Pod shape follows the cabling and host layout; it is not another chassis SKU. This document uses
  only the selected four-Galaxy 4×32 cabling.
- “Scale-up” and “scale-out” are deployment terminology, not topology definitions.
- “Exabox” is a broader deployment name and is not required for runtime topology reasoning.

### Selected fixture family

The routing documents use `[Y,X]` notation while this physical section describes the same complete
system as 4×32:

| Fixture `[Y,X]` | Physical extent | Galaxy grouping |
|---|---:|---|
| `[8,4]` | 4×8 | One Galaxy |
| `[16,4]` | 4×16 | Two consecutive Galaxies |
| `[24,4]` | 4×24 | Three consecutive Galaxies |
| `[32,4]` | 4×32 | Four consecutive Galaxies |

The deployment/physical-grouping layer guarantees that partial fixtures use consecutive Galaxies.
The routing stack does not define how those carve-outs are selected.

### Current four-Galaxy-mesh intermesh envelope

In the current multi-mesh setup, a pair of four-Galaxy meshes is connected by two intermesh
links/cables. Current intermesh traffic patterns and workloads are predominantly linear, and VC1 does
not currently preserve a bubble. This is the existing VC1 operating envelope; it is not a claim that
arbitrary multi-mesh traffic is deadlock-free. The central assessment owns the decision and proof
requirements for enabling VC1 BFC when arbitrary cross-mesh traffic patterns are required.

---

## Quad Galaxy 4×32 physical topology

This section records the physical working model for the four-Galaxy 4×32 configuration. It does not
assign logical dimensions, routing commands, protected domains, or route-selection policy.

### Composition and orientation

- `4 Galaxies × 32 chips/Galaxy = 128 chips`.
- Each Galaxy contributes a physical 4×8 segment to the complete 4×32 system.
- Diagrams and descriptors may transpose the dimensions (`4×32` versus `32×4`). Here, 4×32 means
  four chips on the short axis and 32 chips on the four-Galaxy axis.
- Chip labels are local to each Galaxy and should be qualified, for example `G0:12`.

### Hardware guarantees used by the routing design

For the selected fixture family, the physical/deployment setup guarantees:

- the same Y/ring structure is repeated in every X column;
- the same four-chip X ring is repeated at every Y coordinate;
- participating intramesh cardinal, express, and X-ring edges expose one uniform ordered
  routing-plane set;
- software can preserve plane index around each ring and across each turn;
- `[16,4]` and `[24,4]` are consecutive-Galaxy subsets of the complete `[32,4]` deployment.

These are inputs to the routing design, not topology shapes or plane-homogeneity properties that
route generation must rediscover or prove.

### Staggered 4×4 subtori

The 4×4 regions are staggered along the long axis rather than forming two self-contained halves in
each Galaxy:

- one complete 4×4 region lies in the middle of each Galaxy;
- half-regions lie at both Galaxy edges;
- facing halves in adjacent Galaxies form a complete boundary-crossing 4×4 region;
- the two halves at the ends of the complete 4×32 system join through the cluster wrap.

The physical system therefore contains 4×4 subtori within Galaxies and across Galaxy boundaries.

### Physical edge classes

- **X direct:** single-hop neighbors on the short axis.
- **X wrap:** links closing the four-chip X ring.
- **Y direct:** single-hop neighbors on the long axis, including Galaxy-boundary links.
- **4×4 Y wrap:** links closing one staggered 4×4 region.
- **4×8 Y wrap:** links closing a Galaxy-sized 4×8 region.
- **4×32 Y wrap:** links closing the complete four-Galaxy system.

The 4×4 and 4×8 wrap links remain physical links in the full 4×32 cabling. Their logical use is
defined by the selected routing design, not by this physical classification.

---

## Descriptor and discovery stack

```text
Cabling Descriptor          Deployment Descriptor
(topology hierarchy)        (hostnames and rack locations; order matters)
            \                      /
             \                    /
              → Cabling generator
                       ├─ FSD (.textproto)
                       └─ Cabling guide CSV (cutsheet)
                              ↓
                     Cluster descriptor YAML

Live system:
  UMD cluster descriptor
    → physical system discovery
    → PSD (flat ASIC and physical-link graph)

Runtime fabric:
  MGD — logical meshes and intermesh graph
  PGD — permitted physical groupings/carve-outs
  PSD — discovered physical ASIC and link graph
    → TopologyMapper
    → logical-to-physical bindings
    → ControlPlane
```

The expected FSD and discovered PSD provide the physical-conformance boundary. MGD supplies the
logical graph, while PGD constrains how logical resources may be placed on the physical system.

### Primary repository locations

| Artifact | Location |
|---|---|
| Physical discovery | `tt_metal/fabric/physical_system_discovery.cpp` |
| MGD schema and implementation | `tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`, `tt_metal/fabric/mesh_graph_descriptor.cpp` |
| In-tree MGDs | `tt_metal/fabric/mesh_graph_descriptors/` |
| PGD examples | `tests/tt_metal/tt_fabric/physical_groupings/` |
| Board and node definitions | `tools/scaleout/board/`, `tools/scaleout/node/` |
| Cabling generator and FSD tooling | `tools/scaleout/` |
| BH Exabox bringup material | `tools/scaleout/exabox/` |

---

## Current fabric software baseline

This section summarizes implemented behavior needed to compare the current stack with a target
routing design. Component contracts contain the detailed current-code inventories and target
changes.

### Initialization and ownership order

The current setup is split across existing initialization phases:

1. A `ControlPlane` constructor runs either the explicit-MGD or auto-discovery path.
   - With an explicit MGD, it constructs `MeshGraph`, runs physical discovery, applies
     `TopologyMapper`, constructs `RoutingTableGenerator`, and refreshes intermesh connectivity.
   - With auto-discovery, physical discovery precedes construction of the generated `MeshGraph`;
     mapping and routing-table generation then follow.
2. The constructor calls `initialize_fabric_context()` after the control-plane initialization path
   returns. `FabricContext` selects topology behavior and packet-resource sizing.
3. `MetalEnvImpl::initialize_fabric_config()` later calls
   `configure_routing_tables_for_fabric_ethernet_channels()`. This host-side pass maps logical
   directions to physical Ethernet channels and builds per-source-channel routing tables; it does
   not write device L1.
4. During fabric firmware initialization, `write_routing_tables_to_all_chips()` writes routing data
   to Tensix and Ethernet-core L1.
5. Firmware compilation then reaches `create_and_compile_tt_fabric_program()`, which constructs and
   runs `FabricBuilder`.

The canonical entry points are `tt_metal/fabric/control_plane.cpp`,
`tt_metal/impl/context/metal_env.cpp`,
`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`, and
`tt_metal/fabric/fabric_init.cpp`.

### Logical graph and routing policy

- MGD contains an `express_connections` field, and the descriptor parser can populate it.
- The current runtime constructs `MeshGraphDescriptor` with `backwards_compatible=true`; legacy
  validation rejects nonempty express connections.
- `MeshGraph::initialize_from_mgd()` builds intramesh connectivity from regular 2D `LINE`/`RING`
  geometry and does not merge descriptor express connections into the runtime graph.
- `RouterEdge` carries direction, connected-chip/port entries, and weight under the current
  one-neighbor-per-direction assumption.
- `RoutingTableGenerator` produces
  `[mesh][source][destination] → RoutingDirection`. Intramesh routing resolves coordinate 0
  (N/S) before coordinate 1 (E/W); an exact distance tie follows the first candidate direction
  passed to the helper.
- Intermesh route generation and exit selection are separate from the intramesh table.

The primary files are `tt_metal/fabric/mesh_graph_descriptor.cpp`,
`tt_metal/fabric/mesh_graph.cpp`, `tt_metal/fabric/routing_table_generator.cpp`, and their public
headers under `tt_metal/api/tt-metalium/experimental/fabric/`.

### ControlPlane and device routing artifacts

`ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()` builds
`router_port_directions_to_physical_eth_chan_map_` and converts logical directions into per-channel
next-hop tables. Channels in one direction are parallel routing planes; turns preserve the plane
index when selecting a channel in the next direction.

`get_fabric_route()` follows those channel tables to reconstruct channel-accurate paths, while
`get_forwarding_direction()` returns the logical direction selected by `RoutingTableGenerator`.
Later L1 setup provides direction tables and the compressed 2D path table in `routing_l1_info_t`.

The current 2D path representation is `compressed_route_2d_t`:

```text
N/S hop count      7 bits
E/W hop count      7 bits
N/S direction      1 bit
E/W direction      1 bit
turn point         7 bits
                  -------
used              23 bits

7 + 7 + 1 + 1 + 7 = 23
storage             = 32 bits = 4 bytes
```

It represents one monotonic segment per axis. It does not identify an edge, next chip, jump length,
physical channel, or routing plane. Device setup expands it into directional commands in the packet
`route_buffer`.

### Current 4×32 sizing

`FabricContext` sizes the current 2D route buffer from Manhattan geometry:

```text
max_hops = (rows - 1) + (columns - 1)

4×32:
(4 - 1) + (32 - 1)
= 3 + 31
= 34 hops
```

The 34-hop result selects a 35-byte route-buffer tier. The resulting packet header is:

```text
61-byte non-route portion + 35-byte route buffer = 96 bytes
```

The current checks also require no more than 256 chips per mesh and no axis above 32:

```text
4 × 32 = 128 chips
128 < 256
```

These calculations establish current storage compatibility for a regular 4×32 route. They do not
establish express-link support.

### Builder and ERISC realization

- `FabricBuilder::discover_channels()` rejects multiple neighbor meshes or multiple neighbor chips
  in one direction.
- It caches one `FabricNodeId` neighbor per `RoutingDirection` and treats active Ethernet channels in
  that direction as parallel links/planes to the same neighbor.
- `FabricBuilder::create_routers()` creates one router for each active Ethernet channel.
- `RouterConnectionMapping` maps receiver VC/channels onto downstream sender channels and
  directions; it does not choose between different destination chips sharing one direction.
- In current 2D mode, the packet hop command is a four-bit E/W/N/S mask.
- The ERISC router uses the command and router facing to select forwarding or local delivery.
  At the final hop, the encoder uses its opposite-direction write convention.
- Edge routers may recompute from L1 direction/path tables, which carry the same direction-based
  representation.

The primary files are `tt_metal/fabric/fabric_builder.cpp`,
`tt_metal/fabric/builder/router_connection_mapping.cpp`,
`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`, and
`tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp`.

### End-to-end current limitation

The implemented path is direction-centric:

1. runtime `MeshGraph` does not enable descriptor express connections;
2. `RoutingTableGenerator` returns only `RoutingDirection`;
3. `ControlPlane` maps each direction to parallel physical channels;
4. `FabricBuilder` requires one neighbor chip per direction;
5. `compressed_route_2d_t` records axis directions and hop counts;
6. the kernel consumes a directional command and maps it to local sender channels.

Therefore, the current end-to-end representation cannot distinguish two different neighbor chips
that are assigned the same logical direction. This is a current implementation constraint; the
assessment owns the target command and edge-role solution.

### Current Z and intermesh behavior

Z is a logical `RoutingDirection` in the current fabric stack and is treated specially by existing
intermesh and Z-router paths. Intermesh is not physically restricted to Z: the implementation also
supports XY-intermesh cases and tracks intermesh-facing Ethernet channels separately from their
direction. Target intramesh-Z and capability semantics are defined only by the assessment and
component contracts.

---

## Target routing boundary

The physical nested wraps motivate the target routing work, but this document does not define how
they are represented or used logically:

- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` owns the target topology interpretation, command
  contract, canonical route relation, generation policy, safety proof, cross-layer dependency
  ledger, and validation oracles.
- `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` owns detailed MGD/MeshGraph express
  materialization, ring synthesis, RoutingTableGenerator changes, ControlPlane route/domain state,
  setup checks, and pre-builder handoffs.
- `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md` owns the target ControlPlane-to-builder surface, local
  effects, wiring/allocation, and BFC compile-time realization.
- `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` owns target L1/device artifacts, packet/header ABI,
  encode/load/decode behavior, multicast encoding, and source fanout/reroot.
- `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` owns target ERISC decode/admit/forward behavior,
  intermesh execution, BFC consumption, and controlled same-link return.

None of those target requirements should be read back into the current baseline unless the code
implements them.

---

## Related references

- `tools/scaleout/README.md` — cabling-generator inputs and outputs
- `tools/scaleout/exabox/BRINGUP.md` — BH Galaxy pod topologies
- `tools/scaleout/exabox/README.md` — bringup and validation workflow
- `tt_metal/fabric/MGD_README.md` — Mesh Graph Descriptor
- `tt_metal/fabric/PHYSICAL_GROUPING_DESCRIPTOR_README.md` — Physical Grouping Descriptor
- `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` — current fabric architecture
