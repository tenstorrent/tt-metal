# Task plan — topology manifest emitter (v1)

Scope: emit a machine-readable description of **this fabric instance's topology** at fabric init, so the
visualizer and the ttexalens capture layer can both work without a live tt-metal process.

Parent plan: `visualizer_plan.md` (this is task 0+1, topology only).

**Explicitly deferred** (future revisions of the same file, not this task):

- static L1 addresses (`ROUTING_TABLE_BASE`, routing path base, exit node table, …)
- full per-router ABI from EDM builder `named_args` (stream IDs, VC counts, credit transport, CT flags)

Those are additive: v1 defines `manifest_version` and a per-router object that later grows `l1` and `abi`
keys. Nothing in v1 should need to change shape to accommodate them.

---

## 1. What v1 must answer

The manifest alone (no devices, no host process) must be enough to:

1. **Draw** the cluster: meshes, chips, chip coordinates, and fabric links including torus wrap and inter-mesh.
2. **Enumerate the peek targets**: which ethernet cores are fabric routers, on which physical chip, at which
   core coordinate. This is the vertex set the capture layer iterates — *not* `get_block_locations("eth")`
   and *not* ttexalens `idle_eth_blocks`.
3. **Identify the run**: arch, fabric config, host rank, so a snapshot can be matched to a manifest.

Non-goal for v1: any value that changes during the run. The manifest is frozen at init; snapshots carry
the live numbers.

---

## 2. Where the data comes from

All from **`public:` class members of `ControlPlane`**, so this task adds no new accessors.

Terminology, because the header layout is easy to misread:

| Location | Build treatment | Meaning |
|---|---|---|
| `tt_metal/api/tt-metalium/**` | `api` FILE_SET on `tt_metal`, installed (`metalium-dev`) | shipped to external consumers |
| `tt_metal/api/tt-metalium/experimental/**` | same file set, same install | shipped, **no stability guarantee** |
| `tt_metal/fabric/*.hpp` | `target_include_directories(fabric PRIVATE .)`, not installed | internal implementation headers |

`ControlPlane` is declared **only** in `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp`; there is no
internal `tt_metal/fabric/control_plane.hpp` parallel. So "public" here means *public class members we are allowed to
call*, not *stable supported API*. The header is explicitly experimental.

Consequences for this task, all favourable:

- `fabric_host_utils.cpp` already includes that header as its first include, so taking `const ControlPlane&` adds no
  new dependency.
- Our serializer is declared in `fabric_host_utils.hpp`, which is **internal and not installed** — we are not adding
  to any shipped API surface.
- Because `ControlPlane` itself is unstable, **the manifest schema is the stability boundary** for everything
  downstream. The Python capture layer and the viewer depend on `manifest_version` + the JSON shape, never on
  `ControlPlane` signatures. ControlPlane churn is then a one-file fix in the emitter.

| Manifest field | Source |
|---|---|
| `fabric_config` (1D / ring / 2D / torus X/Y/XY) | `get_fabric_config()` |
| `reliability_mode`, `tensix_config`, `udm_mode` | `get_fabric_reliability_mode()`, `get_fabric_tensix_config()`, `get_fabric_udm_mode()` |
| mesh ids | `get_mesh_graph().get_all_mesh_ids()` |
| mesh shape | `get_mesh_graph().get_mesh_shape(mesh_id)` |
| chips in mesh | `get_mesh_graph().get_chip_ids(mesh_id)` |
| chip mesh coordinate | `get_mesh_graph().chip_to_coordinate(mesh_id, chip_id)` |
| physical (UMD) chip id | `try_get_physical_chip_id_from_fabric_node_id()` — **try_** variant, see §6 |
| ASIC id | `get_asic_id_from_fabric_node_id()` |
| **fabric router channels + direction** | `get_active_fabric_eth_channels(node)` → `set<pair<chan_id_t, eth_chan_directions>>` |
| link peer (node + channel) | `try_get_connected_mesh_chip_chan_ids(node, chan)` |
| routing plane id | `get_routing_plane_id(node, chan)` |
| intermesh vs intramesh facing | `get_intermesh_facing_eth_chans()` / `get_intramesh_facing_eth_chans()` |
| cross-host link | `is_cross_host_eth_link(physical_chip_id, chan)` |
| host rank / binding | `get_local_host_rank_id_binding()`, `get_local_mesh_id_bindings()` |
| eth core coordinate for a channel | `soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL)` (and translated/virtual, see §4) |
| arch | `cluster.arch()` |

`get_active_fabric_eth_channels` is the tight router set discussed earlier: it is derived from the port map,
which only holds channels that passed `EthRouterMode::FABRIC_ROUTER` + link-up in
`Cluster::get_fabric_ethernet_channels`, then ordering/trimming.

---

## 3. Hook point (important correction)

The existing serializers (`serialize_mesh_coordinates_to_file`, asic mapping, intermesh ports) run inside
`init_control_plane` / `init_control_plane_auto_discovery`. **We cannot hook there.** At that point
`router_port_directions_to_physical_eth_chan_map_` is empty, so there are no router channels or directions yet.

The port map is built later, in `configure_routing_tables_for_fabric_ethernet_channels()`
(called from `MetalEnvImpl::initialize_fabric_config()`), and is only final after ordering, trimming, and the
multi-host merge:

```1230:1246:tt_metal/fabric/control_plane.cpp
    this->initialize_dynamic_routing_plane_counts(
        intra_mesh_connectivity, this->fabric_config_, this->fabric_reliability_mode_);

    // Order the ethernet channels so that when we use them for deciding connections, indexing into ports per direction
    // is consistent for each each neighbouring chip.
    this->order_ethernet_channels();

    // Trim the ethernet channels that don't map to live fabric routing planes.
    // NOTE: This MUST be called after ordering ethernet channels
    this->trim_ethernet_channels_not_mapped_to_live_routing_planes();

    this->collect_and_merge_router_port_directions_from_all_hosts();

    this->convert_fabric_routing_table_to_chip_routing_table();
    // After this, router_port_directions_to_physical_eth_chan_map_, intra_mesh_routing_tables_,
    // inter_mesh_routing_tables_ should be populated for all hosts in BigMesh
}
```

**Decision: emit at the end of `configure_routing_tables_for_fabric_ethernet_channels()`**, after
`convert_fabric_routing_table_to_chip_routing_table()`.

Conventions to copy from the existing exports:

- Path: `<rtoptions.get_logs_dir()>/generated/fabric/fabric_debug_manifest_rank_<r+1>_of_<n>.json`
- Always written at init (no env gate) — a hang must find it already on disk.
- Wrapped in `try`/`catch` with `log_warning`, never fatal. A debug artifact must not break fabric bring-up.

---

## 4. Router node identity: which coordinate to store

Capture needs to turn a manifest entry into a ttexalens peek. ttexalens takes an `OnChipCoordinate` and a
device id. To keep v1 self-sufficient, each router entry stores:

- `physical_chip_id` (UMD chip id — matches ttexalens `device.id`)
- `eth_chan` (channel id, the fabric-native identifier)
- `logical_core` `{x, y}` — what `fabric_erisc_dumper.py` prints and filters on today
- `translated_core` or `virtual_core` `{x, y}` — needed because the dumper resolves eth locations by
  channel index into `get_block_locations("eth")`; storing the coordinate avoids depending on that ordering

This is **node identity, not ABI**: it says *where the router lives*, not what its registers mean. Including it
is what lets the capture layer be written before the ABI task lands.

Open item: confirm which coordinate system ttexalens' `get_block_locations("eth")` ordering corresponds to,
and store whichever makes the Python side a direct lookup. Verify on hardware in §7.

---

## 5. Schema v1 (draft)

JSON, emitted with `nlohmann::json` (already a `tt_metal` dependency — used in `tt_metal/llrt`, `tt_metal/common`).

```jsonc
{
  "manifest_version": 1,
  "kind": "fabric_debug_manifest",
  "run": {
    "arch": "wormhole_b0",
    "fabric_config": "FABRIC_2D_TORUS_XY",
    "reliability_mode": "...",
    "tensix_config": "DISABLED",
    "udm_mode": "DISABLED",
    "host_rank": 0,
    "world_size": 1,
    "local_mesh_ids": [0]
  },
  "meshes": [
    {
      "mesh_id": 0,
      "shape": [2, 2],
      "torus": { "x": true, "y": true },
      "chips": [
        {
          "fabric_chip_id": 0,
          "mesh_coord": [0, 0],
          "physical_chip_id": 0,        // null if non-local (multi-host)
          "asic_id": "0x...",
          "is_local": true,
          "routers": [
            {
              "eth_chan": 5,
              "direction": "W",          // E/W/N/S/Z
              "routing_plane": 0,
              "logical_core": [0, 5],
              "translated_core": [17, 25],
              "link_class": "intramesh"  // intramesh | intermesh
            }
          ]
        }
      ]
    }
  ],
  "links": [
    {
      "src": { "mesh_id": 0, "chip_id": 0, "eth_chan": 5 },
      "dst": { "mesh_id": 0, "chip_id": 1, "eth_chan": 13 },
      "direction": "W",
      "routing_plane": 0,
      "class": "intramesh",   // intramesh | intermesh
      "wrap": false,           // torus wrap edge
      "cross_host": false
    }
  ]
}
```

Notes on the shape:

- **`routers` under chips** is what capture iterates. **`links`** is what the viewer draws. Both reference the
  same `(mesh_id, chip_id, eth_chan)` triple, which is the stable node/edge id for snapshots.
- Links are emitted **once per directed source channel** (a cable appears twice, once from each end). Simpler to
  generate, and each direction has its own credit state to color. The viewer can pair them.
- `wrap` is computed in the emitter, not guessed in the frontend: an intra-mesh link whose coordinate delta along
  the axis is greater than 1 (and where the dimension genuinely closes — see `is_genuine_torus_dim`, which
  excludes size-1 and size-2 dims). Frontend draws arcs from this flag; it never re-derives topology.
- Enums serialized as **strings**, not raw ints, so the artifact is readable and stable if enum values shift.
- Output must be **deterministic** (sorted by mesh, chip, channel) so it can be used as a golden in tests.

---

## 6. Multi-host handling

After `collect_and_merge_router_port_directions_from_all_hosts()`, the port map contains entries for nodes owned
by other hosts. Those nodes:

- have **no** local physical chip id → use `try_get_physical_chip_id_from_fabric_node_id()` and emit `null`
  (`get_physical_chip_id_from_fabric_node_id` is fatal for unmapped nodes)
- must be marked `is_local: false`; capture must skip them (this rank cannot peek them)
- still belong in `links`, so the viewer can draw the full mesh and show the cross-host boundary

Each rank writes its own manifest file. Merging N rank files into a cluster-wide view is a later concern for the
viewer/capture layer, not the emitter.

---

## 7. Validation

Hardware available: 4 Wormhole devices (`/dev/tenstorrent/0..3`), active build dir `build_Release` (symlinked as
`build/`), so incremental builds are viable.

1. **Compile** incrementally (fabric lib + relink).
2. **1D**: run an existing 1D fabric test (e.g. `tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_1d_fabric.cpp`),
   inspect the manifest: expect line topology, no wrap edges, directions E/W only.
3. **2D**: run a 2D fabric test; expect a grid and N/S/E/W.
4. **Torus**: if reachable on a 4-chip system via `FabricConfig`, confirm `wrap: true` edges appear and that
   size-2 dimensions do **not** produce spurious wrap edges (the `is_genuine_torus_dim` case).
5. **Cross-check** the router set against `get_fabric_ethernet_channels` expectations and against
   `fabric_erisc_dumper.py --fabric-streams` output on the same system: every core the dumper shows as an active
   fabric router should appear in the manifest, and the manifest should not contain unused eth.
6. **Golden test**: add a case near the existing serializer test in
   `tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp` (which already exercises
   `serialize_mesh_coordinates_to_file`) asserting the file is created, parses, and is deterministic.

---

## 8. Files touched

All C++ changes land in **internal** fabric files; nothing is added to the installed `api/` file set.

| File | Change |
|---|---|
| `tt_metal/fabric/fabric_host_utils.hpp` | declare `serialize_fabric_debug_manifest_to_file(...)` (internal header, not installed) |
| `tt_metal/fabric/fabric_host_utils.cpp` | implement it (nlohmann json; enum→string helpers; deterministic ordering) |
| `tt_metal/fabric/control_plane.cpp` | call it at end of `configure_routing_tables_for_fabric_ethernet_channels()`, try/catch + `log_warning` |
| `tt_metal/fabric/CMakeLists.txt` | add `nlohmann_json` link if the fabric target does not already have it |
| `tt_metal/fabric/debug/visualizer/schema/fabric_debug_manifest_schema.json` | checked-in schema for the viewer/capture contract |
| `tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp` | golden/smoke test |

Signature sketch (takes `ControlPlane` since every field it needs is a public class member):

```cpp
void serialize_fabric_debug_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path);
```

Prefer this free function, matching the existing serializer pattern. Fall back to a private `ControlPlane` method
only if some needed field turns out not to be publicly reachable (§9 flags `get_active_fabric_eth_channels` on
remote nodes as the likeliest case).

---

## 9. Risks / open questions

Resolved during implementation (evidence from a live T3K, 8 WH chips, single host):

- **Coordinate system mismatch** — *resolved*. tt-metal's **virtual** coordinate is the one ttexalens calls
  **`translated`**, not `noc0`. Verified across all 40 routers: `virtual_core == loc.to('translated')` and
  `logical_core == loc.to('logical')[0]` with zero mismatches. Note `loc.to('logical')` returns
  `((x, y), 'eth')`, so the core type must be stripped. On Wormhole `logical_core == [0, eth_chan]`, and
  ttexalens indexes `get_block_locations('eth')` **by channel**, so `eth_chan` alone is a sufficient key.
- **`get_active_fabric_eth_channels` for remote nodes** — *sidestepped*. `routers` is populated only for chips
  where `try_get_physical_chip_id_from_fabric_node_id` resolves. A chip this rank cannot map is a chip it
  cannot peek, so resolvability is the definition of `is_local`. Remote chips still appear with
  `routers: []` so the viewer can draw the full mesh and the host boundary.
- **Torus wrap detection** — *resolved, and simpler than planned*. `fabric_host_utils.hpp` already exports
  `has_genuine_torus_axis(fabric_type, mesh_shape, axis)` with exactly the intended semantics. Caveat: it
  `TT_FATAL`s unless `mesh_shape.dims() == 2`, so the emitter guards on that and omits `torus` for non-2D
  meshes. Wrap is then `has_genuine_torus_axis && |Δcoord| == extent - 1`.
- **Determinism** — *confirmed*. Separate FABRIC_1D and FABRIC_2D runs produced byte-identical manifests apart
  from the `fabric_config` string. On T3K the physical router set is the same for 1D and 2D; only routing
  behaviour differs, which is what `fabric_config` records.

Still open:

- **Galaxy size**: 29 KB for 8 chips / 40 routers, so a 32-chip galaxy should land in the low hundreds of KB.
  Fine, but unmeasured.
- **Mock / emulated clusters**: the emitter is wrapped in try/catch + `log_warning` at the hook, so a mock run
  degrades to "no manifest" rather than failing init. Not yet exercised on a mock cluster.
- **Multi-host**: `is_local`, `cross_host` and null `dst` paths are implemented but untested — this box is
  single-rank, so every link came back `cross_host: false` with a resolved peer.
- **Blackhole**: untested; nothing in the emitter is Wormhole-specific, but the coordinate equivalence above
  was only verified on WH.
- **Two init paths** (`init_control_plane` and `init_control_plane_auto_discovery`) both converge on
  `configure_routing_tables_for_fabric_ethernet_channels()`, so one hook covers both. Only the standard path
  has been exercised.

---

## 10. Step order

All steps below are implemented and validated on T3K.

1. ~~Draft `fabric_debug_manifest_schema.json` (v1, topology only).~~ Done; validates under Draft 2020-12.
2. ~~Implement `serialize_fabric_debug_manifest_to_file`.~~ Done, using `enchantum::to_string` with a numeric
   fallback for unnamed bitmask combinations, and `nlohmann::ordered_json` so key order is readable as well as
   deterministic.
3. ~~Hook at end of `configure_routing_tables_for_fabric_ethernet_channels()`.~~ Done, try/catch + warning.
4. ~~Build; fix compile/link.~~ `nlohmann_json` is PRIVATE on `llrt`, so it does not propagate to `fabric`;
   added to both the `fabric` and `fabric_unit_tests` targets.
5. ~~Run 1D and 2D fabric on hardware; inspect artifacts.~~ Both emit
   `generated/fabric/fabric_debug_manifest_rank_1_of_1.json`. Correct 2×4 mesh: corner chips 4 routers,
   interior 6, two routing planes per direction, 40 links.
6. ~~Cross-check router set; resolve coordinate question.~~ ttexalens sees **128** eth cores; the manifest names
   **40** as fabric routers — the ~3× narrowing that motivated the manifest. Chip→coordinate mapping matches
   the independently generated `physical_chip_mesh_coordinate_mapping_1_of_1.yaml` exactly.
7. ~~Add the test.~~ `ControlPlaneFixture.TestT3kFabricDebugManifest`. Chose invariants over a byte-golden: a
   golden needs one file per cluster type and churns on unrelated topology changes, whereas the invariants
   (link symmetry, opposite directions, routing-plane agreement, direction-vs-geometry, unique router keys)
   hold on any cluster. Regression-checked against a stashed baseline: the same 8 Galaxy/Custom2x2 tests fail
   before and after, all from MGDs that do not fit T3K hardware.
