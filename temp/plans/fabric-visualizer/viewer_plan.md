# Task 5 — Static viewer (HTML + JS)

Parent: `visualizer_plan.md` §"5. Viewer". Design reference: `fabric_debug_infrastructure_design.md` §3.8
("Visualizer"). Inspiration, not a copy: `temp/deadlock_snapshot.html` (map + summary split, occupancy
bars, dark theme).

Goal: open one `decoded.json` and see which hop looks stalled, then click through to named credits and
the region tree. The page does not talk to a device, ttexalens, or MPI. It does not re-derive topology,
credit polarity, or occupancy. Decode already did that.

Task 6 (backpressure walk, two-capture diff, observations list) is **out**. v1 colours `stall_score` and
shows the numbers that produced it.

---

## 1. What we learned from decode output and the other artifacts (drives every decision below)

### 1.1 The file the viewer actually reads

`kind: fabric_debug_decoded`, `decoded_version: 1`. Required top-level keys: `run`, `fabric_context`,
`enums`, `coverage`, `topology`, `routers`, `inputs`. Schema:
`visualizer/schema/fabric_debug_decoded_schema.json`. `additionalProperties: false` on the router
object, so the UI may only depend on fields decode already emits.

Idle T3K decode (`/tmp/decoded_slice_e_live.json`, 2026-09-15): **4.2 MB**, 40 routers, 40 directed
links, one mesh `shape [2, 4]`. Do **not** commit that file. Galaxy with `--slots headers` is the
parent plan's size concern (~8×); the viewer must still open a 4 MB file via a file picker without a
bundler.

### 1.2 Topology is already an edge list

`topology.meshes[].chips[]` has `fabric_chip_id`, `mesh_coord` `[y, x]` or `null`, and `routers[]` as
endpoints only. Full router rows live in `routers[]`, keyed by `(mesh_id, chip_id, eth_chan)`.

`topology.links[]` copies the manifest link (`src`, `dst`, `direction`, `routing_plane`, `link_class`,
`wrap`, `cross_host`) plus decode's `stall_score` and `status`. The manifest schema already forbids the
viewer from inferring torus wrap: `wrap` is computed by the C++ emitter when the axis delta spans the
mesh. Draw `wrap: true` as an outside arc. `dst` may be `null` (unresolved peer) — draw a stub, do not
drop the edge. `link_class: intermesh` is the Z / other-mesh case; style dashed, do not hide.

`run.topology` / `fabric_context.topology` is a name (`Mesh`, `Linear`, …). Layout engines key off
**coordinates + the edge list**, not off that string. A torus looks like a mesh grid plus wrap arcs.
A 1D line is chips with `mesh_coord [0, i]` in a row.

### 1.3 What is honest to draw on a router

Decode's contract, which the viewer must not walk back:

| Want | Have | Must not |
|---|---|---|
| Slot occupied vs free | `rings[].occupied_count` + `occupancy_status` | Paint individual slots occupied. `slot_state` is always `unknown` |
| "What packet is in slot i" | `slots[].header` = last header written there; `plausible` is a dimming hint | Treat plausible zeros as in-flight |
| Stall colour | `router.stall_score` and `link.stall_score` in `[0,1]` or `null` | Diagnose idle vs hung: they look the same at occupancy 0 |
| Credits | `channels.senders/receivers/downstream` | Read stream ids or `FABRIC_STREAM_GROUPS` |
| Alive? | `liveness.classification`, `lifecycle.exit_state` | Treat a moving WH `0x1F80` word as fabric progress (shared with base FW) |
| Coverage | `capture.status`, top-level `coverage` | Grey "idle eth" from ttexalens |

`connection_sem` on idle T3K was 72 `open` / 88 `unused`. The summary must show that; it is not a stall.

### 1.4 Deadlock HTML — take the layout idea, not the occupancy lie

`temp/deadlock_snapshot.html` is a **simulator** snapshot: per-node cards, topology SVG, and occupied
slot rectangles because that model *has* indices. Ours does not. Take:

- dark page, monospace, map on the left, summary on the right
- occupancy as a count / depth bar (`3/14`), not a slot strip painted occupied/free
- a coverage / meta block at the top

Leave the cycle-table, "DEADLOCK at cycle N", and per-slot rectangles behind. Task 6 can add a
blocking-chain highlight once we know we want it.

### 1.5 Design-doc viewer vs our split

Doc §3.8: the viewer is generic; it does not hardcode stream IDs or enum integers; enums come from
tables; unknown raw stays raw; expert raw is explicit; no websocket; static file load is V0.

Our parent plan splits two halves: **map (ours)** and **generic region panel (theirs)**. Decode added a
third thing the map cannot live without: the normalized `channels` block. Decision: the page may know
**decoded schema keys** (`channels`, `stall_score`, `liveness`, `lifecycle.exit_state`, `rings[].occupied_count`).
It must not know builder names (`SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID`, CT flags, overlay indices).
New regions still appear in the tree with no UI change.

### 1.6 `file://` will not fetch

Chrome blocks `fetch("decoded.json")` from `file://`. Hang-debug UX is: **file picker / drag-and-drop**
of a `fabric_debug_decoded` JSON. Optional loopback HTTP (`python3 -m http.server` or a 20-line
`serve.py`) only so committed fixtures can be chosen from a list when served. The viewer never starts
decode or capture.

Parent plan's "can read a topology-only manifest" is **rejected for v1**. Empty graphs are decoded
fixtures with `not_captured` routers. Two schemas in the browser is how fabric knowledge leaks back
into the UI.

### 1.7 Existing code to reuse

- `decode/output.py::build_decoded` + `decode/tests/fixtures.py::write_input` produce schema-valid
  decoded objects in a tmp dir. Viewer fixtures are those objects, trimmed, committed.
- `jsonschema` already in decode tests. Fixture tests validate against
  `fabric_debug_decoded_schema.json`.
- No npm, no React, no bundler. The repo does not have a frontend toolchain for this folder, and a hang
  on a lab box should open with whatever browser is there.

---

## 2. Decisions

1. **Vanilla HTML + JS + CSS**, ES modules, no build. `index.html` + `js/*.js` + `css/viewer.css`.
2. **One input:** `decoded.json`. Kind/version checked in the page; anything else is an error banner.
3. **Load path:** `<input type="file">` and drag-drop always. When served over HTTP, a small fixture
   list (`fixtures/index.json`) may `fetch` relative URLs. No default fetch of a 4 MB T3K dump.
4. **SVG map**, not canvas: hit-testing and wrap arcs are cheaper as DOM. Zoom/pan = `viewBox` + pointer
   drag + wheel. No WebGL.
5. **Layout from `mesh_coord` + `links[]`.** `wrap` / `link_class` / `cross_host` are attributes on the
   already-emitted edges. `fabric_config` / `run.topology` are labels in the header, not switch cases
   that invent wraparound.
6. **Three zoom levels:** mesh (chips), chip (ports), selected router (panel). Back stack. Multi-mesh:
   one grid per mesh, laid out left-to-right with a gap; intermesh edges cross the gap.
7. **Colour function is the only stall interpretation in the UI:**
   - edge stroke: `null` / `not_captured` / `unknown` / `unreadable` → grey; `ok` → interpolate
     stall_score `0 → 1` (cool → hot); `torn` → same colour plus hatch; `reset` → extra outline on the
     **source** node, not a different polarity.
   - node fill: `capture.status` first (grey / hatch / outline), then a thin stall ring if `ok`.
   Idle T3K will be uniformly cool. That is correct. A stalled-link screenshot comes from a **fixture**,
   not from hoping the lab is hung.
8. **Selection summary is allowed to render `channels`.** Occupancy bars + connection names + downstream
   free slots. Region tree underneath is generic (`parent` grouping, last id segment as label, `value`
   pretty-printed, `status` badge). `raw_hex` only if the decoded file has it (`--expert-raw`) **and**
   the page's Expert toggle is on. Payload is never requested; decode never put it in.
9. **Rings in the panel:** depth, `occupied_count`, `occupancy_status`. Expandable slot list shows
   header fields and dims `plausible: false`. No occupied/free slot glyphs.
10. **No observations, no rule pack, no two-file diff, no live poll.** Task 6.

---

## 3. Package layout

```
tt_metal/fabric/debug/visualizer/
  viewer/
    index.html          # shell: coverage strip, file picker, map, panel
    css/viewer.css
    js/app.js           # boot, selection, drill stack
    js/load.js          # file / fetch / kind+version check
    js/model.js         # index routers by endpoint; join links → router
    js/layout.js        # chip grid, port positions, wrap arcs (pure)
    js/map.js           # SVG render + pointer
    js/panel.js         # summary + region tree + rings
    js/color.js         # status / stall_score → stroke/fill/hatch
    serve.py            # argparse directory server, Cache-Control: no-store
    fixtures/
      index.json        # [{id, title, file}]
      line_1d.json
      mesh_2d.json
      torus_xy.json
      two_mesh.json
      stalled_link.json # stall_score 1.0 on one edge (success criterion)
      coverage_holes.json
    tests/
      __init__.py
      test_fixtures.py  # every fixture validates; index files exist
      test_layout.py    # chipPosition formula vs mesh_coord (Python mirror of the documented math)
  README.md             # add Viewer section (this task, slice D)
```

`serve.py` is optional convenience (`python3 viewer/serve.py` → `http://127.0.0.1:8765`). Document
`python3 -m http.server` as the fallback. Neither is required when using the file picker.

Constants in `layout.js` (also copied into `test_layout.py` comments / module-level numbers):

```
CHIP_W, CHIP_H = 72, 72
CHIP_GAP = 28
PORT = 10
MESH_GAP = 96          # between meshes
WRAP_BOW = 36          # how far wrap arcs sit outside the grid
```

Chip origin (mesh local): `x = coord[1] * (CHIP_W + CHIP_GAP)`, `y = coord[0] * (CHIP_H + CHIP_GAP)`.
`mesh_coord` null → park the chip in an "unplaced" row under that mesh and list it as a coverage
warning, do not invent a grid slot.

Port side from `router.direction`: N top, S bottom, E right, W left, Z/C/NONE inside the chip.
Several routers on one side (routing planes): pack along the edge with a 4 px stride.

Wrap edge: cubic/quadratic SVG path from src port to dst port that bows outward on the wrap axis
(E/W wrap bows horizontally past the grid min/max x; N/S analogously). Straight `line` for
`wrap: false`.

---

## 4. Load and model (`load.js`, `model.js`)

`load.js`:

- `File.text()` → `JSON.parse`. Catch and show the parse error; do not dump a stack into the map.
- Require `kind === "fabric_debug_decoded"` and `decoded_version === 1`.
- Do not schema-validate in the browser (no `jsonschema` there). Invalid-but-kind-correct files may
  render partially; missing `routers` / `topology` is a hard fail.

`model.js` builds:

```
byEndpoint: Map "mesh:chip:chan" → router
chipOf: endpoint → chip row
linksNormalized: each link + srcRouter + dstRouter (dstRouter null if dst null or not_captured)
```

Coverage strip binds `coverage.*` plus `run.arch`, `run.fabric_config`, `fabric_context.packet_header_type`,
`generated_at`, `inputs[].snapshot.provenance.hostname` (hosts are provenance, not graph vertices).

Filter box (identity only, design-doc "identity first"): substring over
`mesh_id, chip_id, eth_chan, direction, capture.status, exit_state`. Filters the clickable router list
and dims non-matching nodes. Not a named-field query over 94 regions × 40 routers.

---

## 5. Map (`layout.js`, `color.js`, `map.js`)

Render order: wrap arcs (under), straight links, chips, ports, selection halo.

Click chip → drill to chip view (same SVG, scale around that chip, ports become the hit targets).
Click port / router → select, fill the panel, do not lose the mesh context (highlight the edge).
Click an edge → select its **src** router (stall_score lives on src) and highlight the edge.
Back button pops the stack.

Keyboard: `Esc` pops; `+`/`-` or wheel zooms.

`cross_host: true` gets a small mark on the edge (tick or label), because those hops are the ones
ttexalens would have called idle.

Colour is CSS variables so the hatch pattern is one `<pattern id="torn">` in the SVG defs.

Success wiring for the stalled fixture: one link `stall_score === 1`, src `channels.downstream[].free_slots === 0`
or a sender `occupied === depth`; that stroke is visibly the hot end of the scale next to cool
neighbors.

---

## 6. Panel (`panel.js`)

Selected router, three stacked blocks:

**A. Summary (decoded keys).** Endpoint, `direction`, `link_class`, `routing_plane`, `capture.status`,
`identity.matches`, `lifecycle.exit_state` + named EDM/termination/go, `liveness.classification` with
sample count/format, `stall_score`. Then a table:

- senders: index, vc, role, `occupied/depth`, free_slots, acked/completed or `counters`, connection name
- receivers: `pkts_pending/depth`
- downstream: vc, edge, free_slots (depth shown as "—" because decode leaves it null)

Occupancy bar width = `occupied/depth` when both are numbers and `occupancy_status !== inconsistent`;
inconsistent shows the numbers plus a warning, not a clamped bar.

**B. Rings.** One row per `rings[]`: id, occupied_count, depth, occupancy_status. Expand → slot index,
`plausible`, dest mesh/chip from the header if present. `slot_state` is displayed as `unknown` so we
do not teach the wrong lesson.

**C. Region tree.** Group `regions[]` by `parent` (manifest order, no sort). Folders for `backing === "group"`.
Leaf: id tail, status badge, formatted `value` (enums already have `{raw, name}` from decode — show
`name` with `raw` in a title/tooltip; unknown name shows `raw`). Disabled regions stay visible but muted
(the tree is the ABI; hiding them reintroduces "why is stream 23 garbage"). Unallocated: badge only.

Expert toggle (page-level): if `raw_hex` is present, show it; if not, the toggle still exists and says
the file was decoded without `--expert-raw`. Never `fetch` the `.bin`.

---

## 7. Fixtures

Built with `decode.output.build_decoded` + a small generator script invoked from tests or by hand,
then **checked in** (they are the UI's hardware stand-in). Each file must validate.

| File | Shape | Why |
|---|---|---|
| `line_1d.json` | 4 chips × 1 router, coords `[0,i]`, 3 forward links | 1D line engine |
| `mesh_2d.json` | 2×2, 4 chips, E/W/N/S ports, no wrap | T3K-like grid, small |
| `torus_xy.json` | 2×2 with `wrap: true` on the axis-spanning edges | wrap arcs; must not look like a mesh chord across the interior |
| `two_mesh.json` | two `mesh_id`s, one intermesh link | gap + dashed edge |
| `stalled_link.json` | `mesh_2d` with one src stall_score 1.0, downstream free_slots 0, one sender occupied==depth | colour success criterion |
| `coverage_holes.json` | one `not_captured`, one `reset`, one `torn`, one identity mismatch | grey / outline / hatch |

Do not check in T3K 4 MB. README tells you to decode `generated/fabric/slice_e_*.json` and pick the
result. Idle T3K expected: all cool, 40 chips-worth of ports, coverage `ok: 40`.

Generator lives at `viewer/tests/make_fixtures.py` so regenerating is `python3 -m ...` not a one-off
notebook. It may import `decode.tests.fixtures.write_input` and then **patch** `stall_score` / statuses
on the built object for `stalled_link` / `coverage_holes` (those states are easier to inject than to
fake in streams). Patched files still have to pass schema (`stall_score` 0–1, status enums).

---

## 8. Tests (no hardware, no browser driver)

1. `test_fixtures.py`: every `fixtures/*.json` except `index.json` validates; `index.json` names files
   that exist; stalled fixture has max link stall_score 1; torus fixture has at least one `wrap: true`;
   coverage fixture has each of `not_captured|reset|torn`.
2. `test_layout.py`: for `mesh_2d` / `line_1d`, `chipPosition` using the documented constants matches a
   golden table (chip 0 at origin, chip with `mesh_coord [1,1]` at known x,y). Port side mapping
   N/E/S/W. Wrap path bounding-box is **outside** the chip bounding-box (the arc is not a chord).
   Implemented in Python against the same formulas `layout.js` comments as the source of truth; a
   one-line comment in `layout.js` points at the test. Drift is a review item, not a second JS runtime
   in CI (node is not a tt-metal test dependency today).
3. No Playwright in v1. Manual: serve + click through each fixture; then file-picker the T3K decode.

Run:

```
python3 -m pytest tt_metal/fabric/debug/visualizer/viewer/tests -q --noconftest
```

Same `--noconftest` reason as capture/decode.

---

## 9. Manual T3K pass (README, not CI)

```
python3 tt_metal/fabric/debug/visualizer/decode/cli.py \
  generated/fabric/slice_e_live.json generated/fabric/fabric_debug_manifest_rank_1_of_1.json \
  -o /tmp/decoded_slice_e_live.json
python3 tt_metal/fabric/debug/visualizer/viewer/serve.py
# open http://127.0.0.1:8765 and pick /tmp/decoded_slice_e_live.json
```

Expected:

1. Coverage `40/40 ok`, header `WORMHOLE_B0` / `FABRIC_2D` / `HybridMeshPacketHeaderT<36>`.
2. One 2×4 chip grid; 40 ports; 40 edges all cool (`stall_score` 0).
3. Click a port: senders occupied 0 (worker depth 14, upstream 4), receivers pending 0, some
   `connection` `open`. Region tree shows `sender.0.free_slots` without the UI knowing stream 22.
4. Identity matches; `exit_state` `running_or_host_gone`; liveness `advancing` or `insufficient`, not a
   red "dead" colour — insufficient is not reset.
5. Killed decode: same map; `owner_alive` null in the meta line (provenance on `inputs[].snapshot`).

Optional: a real all-gather capture to see non-zero occupancy. Not a gate (same as decode).

---

## 10. Slices (each builds, each reviewable)

| # | Slice | Files | Done when |
|---|---|---|---|
| A | Shell + load + coverage + model index | `index.html`, `css/`, `js/{app,load,model}.js`, `serve.py` | file picker shows coverage for a fixture JSON; wrong `kind` errors; no map yet |
| B | Map + layout + colour + drill | `js/{layout,color,map}.js`, `test_layout.py` | 2D grid, wrap arcs on torus fixture, stalled fixture edge is hot, reset/torn/not_captured look different, chip drill works |
| C | Panel: summary, rings, region tree, expert toggle | `js/panel.js` | click shows channels table + grouped regions; no per-slot occupancy paint; expert hex only when present |
| D | Fixtures + README + T3K picker pass | `fixtures/*`, `make_fixtures.py`, README viewer section | all fixtures validate; T3K 40 cool routers recorded |

Stop for review after **A**, after **B**, and after **D**.

---

## 11. Risks

- **JS/Python layout drift.** Keep the formula to a few constants and one function; comment the pairing.
  If it hurts, a later slice can add a node test; do not add node as a required CI image in v1.
- **4 MB JSON parse on a laptop is fine; 30 MB Galaxy + headers may hitch.** `--slots none` decode is
  already the size valve. Viewer does not lazy-load per router in v1 (parent open question stays open).
- **`mesh_coord` convention.** Decode T3K matched `routing_l1_info_t.my_mesh_coord` as `{y, x}` to
  `chip.mesh_coord` `[y, x]`. If a 1D fixture uses `[0, i]`, the line engine is just a grid with height 1.
- **Null `dst` links** on partial multi-host files: stub edge + `not_captured` on the missing side.
- **Idle T3K cannot demonstrate a hot edge.** That is why `stalled_link.json` exists. Do not weaken the
  colour scale so that 0 vs 0.02 is visible on idle; that would lie about stalls.

---

## 12. Explicitly not this task

- Task 6 hang UX (walk, diff, observations).
- Reading manifests, snapshots, or `.bin` in the browser.
- Bundlers, React, npm, websockets, live poll.
- Painting ring slots occupied/free.
- SSH gather, BH-specific chrome (2-ERISC shows up as region `writer` in the tree automatically).
- Search over arbitrary region ids (identity filter only).
- Committing T3K decoded JSON.
