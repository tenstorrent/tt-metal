# Fabric reliability mode

This lets mapping finish when a factory cable is down. A `RELAXED` mesh graph places on the live cables, and the missing factory cables are recorded on the control plane.

## Turning it on

Pass the factory descriptor to tt-run, and use a mesh graph whose channels are `RELAXED`:

```bash
tt-run \
  -m mesh_graph.textproto \
  --factory-system-descriptor factory_system_descriptor.textproto \
  --hosts host0,host1,host2,host3 \
  ./my_app
```

```
mesh_descriptors {
  channels { count: 2 policy: RELAXED }
}
connections {
  channels { count: 4 policy: RELAXED }
}
```

Intra `channels` sit on the mesh. Intermesh `channels` sit on each connection. Intra policy is per mesh. Intermesh policy is one value for the whole graph.

`--factory-system-descriptor` is exported to every rank as `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH`. Setting that variable yourself is the same thing.

In the process, set relaxed system health before fabric init:

```cpp
#include <tt-metalium/experimental/fabric/fabric.hpp>

tt::tt_fabric::SetFabricConfig(
    tt::tt_fabric::FabricConfig::FABRIC_2D,
    tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
```

With no factory descriptor, `get_downed_links()` and `get_unused_downed_links()` are empty and `is_link_healthy` returns true.

After init:

```cpp
const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
```

## Rules

A missing cable is one the factory system descriptor (FSD) lists and the live physical system descriptor (PSD) does not. Each cable is stored twice, once from each end. The mesh graph descriptor (MGD) decides which of those cables fabric still routes.

| Name | What it is |
|------|------------|
| FSD | Cables that should exist |
| PSD | Cables that are live |
| MGD | Cables fabric routes |

### RELAXED intramesh

A routing plane is one parallel ethernet path in a direction. Planes are `min(MGD, FSD)`. The mesh uses the minimum across a row or column. `max(0, MGD − FSD)` is a downgrade and is not a link record.

`get_downed_links()` holds `max(0, min(MGD, FSD) − PSD)` missing cables, the ones the graph still needs. `get_unused_downed_links()` holds the rest of the FSD−PSD mismatch. Counts are per direction.

| MGD | FSD | PSD | Planes | `get_downed_links()` | `get_unused_downed_links()` |
|-----|-----|-----|--------|----------------------|-----------------------------|
| 2 | 2 | 1 | 2 | 1 | 0 |
| 2 | 4 | 3 | 2 | 0 | 1 |
| 4 | 2 | 1 | 2 | 1 | 0 |

On the 4-host pod, intra `channels { count: 2 policy: RELAXED }`, init returns. Then:

| Cable | Row | `get_downed_links()` | `get_unused_downed_links()` | `get_num_usable_routing_planes` |
|-------|-----|----------------------|-----------------------------|---------------------------------|
| chip 0 chan 6 ↔ chip 4 chan 0 (mesh) | 1 | both directions | absent | 2 |
| chip 15 chan 4 ↔ chip 23 chan 4 (torus) | 2 | absent | both directions | 2 |
| chip 0 chan 0 ↔ chip 8 chan 0 (subtorus) | not routed | absent | both directions, direction `NONE` | — |

### RELAXED intermesh

No routing planes. Registered is `min(MGD, PSD)`, the live cables pairing binds. Downed and unused use the same split as intramesh. `max(0, MGD − FSD)` channels are not registered and are not records.

| MGD | FSD | PSD | Registered | Downed | Unused | Not registered |
|-----|-----|-----|------------|--------|--------|----------------|
| 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| 2 | 4 | 3 | 2 | 0 | 1 | 0 |
| 4 | 2 | 1 | 1 | 1 | 0 | 2 |
| 8 | 4 | 1 | 1 | 3 | 0 | 4 |

Pod cable chip 5 chan 9 ↔ chip 29 chan 9, FSD 16, PSD 15:

| MGD count | After init |
|-----------|------------|
| 4 | The cable is in `get_unused_downed_links()`, both directions. Pairing registered 4 live channels. |
| 16 | The cable is in `get_downed_links()`, both directions. Pairing registered 15 live channels. |

## Control plane

`tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp`

### `get_fabric_reliability_mode()`

The mode passed to `SetFabricConfig`.

```cpp
cp.get_fabric_reliability_mode() == FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
```

### `has_factory_descriptor()`

True after `--factory-system-descriptor` is ingested. False when it is omitted, and the calls below then report a healthy system.

### `get_downed_links()`

Missing FSD cables the mesh graph still uses. Empty with no FSD. On the pod with relaxed intra count 2, this contains the mesh cable and not the torus or subtorus cable.

```cpp
for (const auto& link : cp.get_downed_links()) {
    // link.src_node, link.src_chan, link.dst_node, link.dst_chan
    // link.is_intramesh() for chip 0 chan 6 → chip 4 chan 0
    // link.src_direction is the mesh-graph direction
}
```

Each cable appears twice, once from each end. `logical_resolved` is true when both ends were placed. `scope` is `IntraMesh`, `InterMesh`, or `Unknown`.

### `get_unused_downed_links()`

Missing FSD cables the mesh graph does not use. Empty with no FSD. On the same pod run this contains the torus cable (one per direction) and the subtorus cable (direction `NONE`). With intermesh count 4 on chip 5 chan 9 ↔ chip 29 chan 9, that cable is here too. With intermesh count 16 it is in `get_downed_links()` instead.

### `fsd_rerouting_active()`

True when `get_downed_links()` is not empty. The relaxed count-2 pod run returns true because of the mesh cable. A run whose only holes are the torus, the subtorus, or an intermesh count the live cables already cover returns false.

### `is_link_healthy(node, chan)`

True when that FSD endpoint is in the PSD. On the mesh cable, `is_link_healthy(chip 0, chan 6)` is false and a live channel on that chip is true. With no FSD every call returns true. Throws if the FSD never declared that endpoint.

### `get_num_usable_routing_planes(node, direction)`

The plane count from the intramesh table, on the rank that owns the mesh. For chip 0 chan 6 ↔ chip 4 chan 0 this is 2 at intra count 2 and at intra count 4.

### `get_locally_unhealthy_links()`

The records in `get_downed_links()` on this host that ethernet also reports down. A cable inferred only from the descriptors is in `get_downed_links()` and absent here.

### `refresh_connectivity_diff()`

Recomputes both sets against the current PSD. No-op with no FSD. References returned by the getters above are invalid after it returns.

### `get_link_health()`

The `LinkHealth` object, or null when no FSD was loaded. The calls above cover the sets. Use this when you want one node, one direction, or one mesh pair: `find_downed_link`, `get_downed_links(node)`, `get_downed_eth_chans_in_direction`, `get_num_downed_routing_planes_in_direction`, `get_downed_intermesh_links(src, dst)`, `get_downed_links_for_host`. The rest of that surface is in `tt_metal/api/tt-metalium/experimental/fabric/link_health.hpp`.
