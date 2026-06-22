# Sub-torus Routing — Working Doc

**Branch:** `agupta/subtorus-routing`  ·  **Status:** in progress

**Two tracks, different destinations:**
- **Capture instrumentation** (dump/markers/parser) — **local-only, throwaway.** Used for scoping
  the work and spot-checking correctness of upcoming changes. Drop/stash before any feature PR.
- **Skip-link feature** (proto `skip_links` + stages 0–5) — **production work, meant for main.**
  Must meet merge standards: clean proto design, validation, unit tests, firmware considerations,
  CODEOWNERS/review.

Goal: capture the real fabric routing state from a live Blackhole-galaxy `4x32` Torus-XY run,
parse it into analyzable tables, and use it to design + validate **skip-link ("sub-torus")
routing** — extra intra-mesh links that let packets jump `stride` hops along an axis. Long
term: develop routing-table generation against captured topology on a CPU-only box.

---

## 1. Status — DONE

### Capture instrumentation (local-only, do NOT upstream)
- **`tt_metal/fabric/control_plane.cpp`** — `print_routing_tables()` / `print_ethernet_channels()`
  bumped `log_debug` → `log_info` so they survive a Release build (`log_debug` is compiled out
  unless `TT_METAL_ENABLE_LOGGING=ON`, which defaults OFF for Release). Markers:
  - `FABRIC_RT_DUMP`  — IntraMesh routing table (InterMesh kept at `log_debug` → compiled out, acts as a filter)
  - `FABRIC_ETH_DUMP` — physical eth channels per direction
- **`tests/.../routing/tt_fabric_test_context.cpp`** — dump gated on `TT_FABRIC_DUMP_DIR`:
  - `FABRIC_RANK_HOST rank=.. host=..` — **all ranks** (supplies rank↔host join)
  - `FABRIC_NODE_MAP host=.. tray=.. asic_loc=.. node=M<m>D<c>` — **all ranks** (see finding below)
  - PSD (`psd.textproto`/`psd.yaml`) + routing tables — **rank 0 only**
- **`tools/scaleout/exabox/run_fabric_tests.sh`** — `--dump-fabric-state` flag (sets env, copies
  mesh_graph_descriptor, prints artifact summary).
- **`tools/scaleout/exabox/parse_fabric_node_map.py`** — parses the markers into CSVs, defaulting
  output to the run's `fabric_state_<ts>/` dir:
  - `node_map.csv` (rank, host, tray, asic_loc, node)
  - `directional_channels.csv` (rank, node, direction, channels)
  - `intramesh_routing.csv` (rank, node, eth_chan, dest_chip, egress_chan — long form)

### Findings
- **PSD is identical across ranks** (modulo ordering) → dump from rank 0 only.
- **Routing tables are full-mesh on every rank** (`control_plane` iterates the *global* chip-id
  container) and deterministic from mesh-graph + PSD → identical across ranks → dump rank 0 only.
- **Node map (asic_id→FabricNodeId) is rank-local** — `get_fabric_node_id_from_asic_id` resolves
  only local asics (throws for remote), so `FABRIC_NODE_MAP` is emitted on **all ranks**; the union
  covers the full mesh.
- Latest good capture `fabric_state_20260617_200614/`: 128 nodes, even 32/rank across 4 hosts,
  full rank attribution, directional + intramesh routing complete.

### Understanding captured (MGD lowering pipeline)
`textproto → proto::MeshGraphDescriptor → MeshGraphDescriptor (instances/connections) →
MeshGraph (IntraMeshConnectivity adjacency + coords + host-rank split) → RoutingTableGenerator
(per-dest RoutingDirection LUTs) → ControlPlane (per-eth-channel egress tables, plane-preserving)`.
- A "plane" = the *index* of a channel within its direction's channel list (N[0]↔S[0]↔E[0]…).
- Physical channels > requested `channels.count`: surplus is **trimmed** (warning) and unused;
  plane count per direction = min(requested/golden, physical, row/col-min). `< requested` = fatal.

---

## 2. Plan — skip-link feature (NEXT) — *intended for main*

> This track ships. Hold it to merge standards (proto review, validation, unit tests, firmware
> impact) — unlike the throwaway instrumentation in §1.

New proto field on `MeshDescriptor` to declare skip links by **pattern**, not enumerated pairs:

```protobuf
skip_links { axis: ROW  pattern { start: 2  step: 4 } }   // names TBD
```

**Geometry rule (LOCKED):**
- Tile the chosen `axis` into consecutive `step`-wide blocks starting at `start`.
- Each skip link connects a block's **endpoints**: `b ↔ b + step − 1` (interior nodes get none).
- RING wrap: last block `[30,31,0,1]` → edge **30-1** (uniform stride; covers nodes before `start`).
- Replicate the pattern across the **orthogonal axis** (×4 for a 32×4 mesh — one per line), mapping
  `(row,col) → linearized chip id (row*4+col)`.
- `start=2, step=4`, axis=ROW → 8 blocks × 4 lines = **32 skip edges**.

**Implementation surface:**
| Stage | File | Work | CPU-only? |
|---|---|---|---|
| 0–3 | proto + `mesh_graph_descriptor.cpp` + `mesh_graph.cpp` | add `skip_links` message/field; parse + expand pattern; fold into `intra_mesh_connectivity_` as `RouterEdge`s | **yes** (`MeshGraph(ClusterType, path)`) |
| validation | `mesh_graph_descriptor.cpp` | lift the MGD-1.0-compat rejection of express/skip connections | yes |
| 4 | `routing_table_generator.cpp` | route *over* skip links — algorithm: when to take skip vs base ring | yes |
| 5 | `control_plane.cpp` | plane/egress lowering for skip links | needs HW or replay |

**Dev loop:** `MeshGraphValidation` test in `tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp`
(target `fabric_unit_tests`) — build a mesh from a skip descriptor, assert the 32 endpoint edges in
`get_intra_mesh_connectivity()`. Red → implement stages 0–3 → green, then move to stage 4.

**Still to pin (not blocking stages 0–3):** `channels` (lanes per skip link).

---

## 3. Identity fork — DECIDED: skip links use the Z direction

**How does a skip link get a `RoutingDirection`?** Everything in stages 4–5 and the device firmware
keys off `RoutingDirection` (`N=0, E=1, S=2, W=3, Z=4, C=5, NONE=6`).

**Decision: assign skip links `RoutingDirection::Z`.** Z is an existing, first-class direction
(already used for the Blackhole 8/9 inter-mesh-on-galaxy path via `assign_z_direction`), so:
- Skip channels get their **own bucket** `router_port_directions_to_physical_eth_chan_map_[node][Z]`,
  cleanly separate from the in-plane N/S/E/W grid links (the disambiguation Option 1 lacked).
- **No new enum value** → avoids the `*_SKIP` firmware blast radius of Option 2.
- The N/S/E/W routing-plane **trimmer** (`trim_ethernet_channels_not_mapped_to_live_routing_planes`,
  control_plane.cpp:1014–1015) only iterates N/S/E/W, so Z skip channels are **not trimmed away**.

Implemented in `mesh_graph.cpp` (both ends of each bidirectional skip edge → `Z`). The `[8,4]` test
asserts `port_direction == Z`.

**Still TBD (stage 4+):** make `RoutingTableGenerator` actually route over Z for **intra-mesh** skip
links (today Z is exercised for inter-mesh), and confirm Z plane-count handling in the control-plane
lowering for intra skip channels. Also: multiple skip families (ROW + COLUMN) would currently share
the single Z bucket — revisit if that needs separating.

---

## 4. Reference

**Capture run (single Torus-XY case):**
```bash
./tools/scaleout/exabox/run_fabric_tests.sh \
  --hosts $HOST --image $IMG --config 4x32 \
  --test-config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_subtorus_routing_capture.yaml \
  --dump-fabric-state
```
**Parse:**
```bash
tools/scaleout/exabox/parse_fabric_node_map.py fabric_test_logs/fabric_tests_<ts>.log
# → CSVs in fabric_test_logs/fabric_state_<ts>/
```
**CPU-only stage 0–3 test:**
```bash
cmake --build build_Release --target fabric_unit_tests
./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='MeshGraphValidation.*'
```

**Notes / gotchas:**
- `express_connections` (existing proto field) is a stub for this purpose: bare src/dst, rejected by
  the 1.0-compat validator, never folded into `intra_mesh_connectivity_`, gets `routing_direction = C`.
  Pattern-based `skip_links` is the chosen replacement.
- Custom descriptors live in `tests/tt_metal/tt_fabric/custom_mesh_descriptors/` (incl. `fabric_cpu_only_*`).
- This instrumentation is local-only; expect to drop/stash it when prepping the real feature PR.
