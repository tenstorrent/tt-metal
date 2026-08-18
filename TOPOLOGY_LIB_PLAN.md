# Plan: `scaleout_topology` — a shared, runtime-free topology library

**Goal:** put the topology **solver + PSD** (and the descriptors they need) in one lean library that both
**tt-metal** (tt-run) and **Fabric Manager** can use **without linking `libtt_metal`**.
**Constraint:** `mesh_graph` stays in `fabric`.

---

## 1. The two libraries

```
TT::ScaleoutTools  (OBJECT, exists today)          TT::ScaleoutTopology  (STATIC, new)
  board → connector → node → cabling_gen             PSD · MGD · PGD  (proto + class)
  → FSD utils / query                                topology solver (SAT + CSP)
  owns: FSD proto                                    topology mapper utils
  Cabling ──► FSD ─────────── FSD proto ───────────► FSD → PSD → solve / map
  (no runtime)                                       (no runtime; links ScaleoutTools)
                                                                 │
                                                                 ▼
                                                     fabric → libtt_metal
                                                       mesh_graph (STAYS) · TopologyMapper class
                                                       · live discovery · links ScaleoutTopology
```

They meet at the **FSD proto**: `CablingGenerator` produces it; `build_psd_from_fsd()` consumes it. No cycles.

---

## 2. What goes where

| Piece | Home | Moves? |
|---|---|---|
| FSD proto, FSD `utils`/`query` | `scaleout_tools` | no |
| **MGD, PSD, PGD** (proto + class) | `scaleout_topology` | **yes** |
| topology **solver** + **mapper utils** | `scaleout_topology` | **yes** |
| **FSD → PSD** builder (`fsd_to_psd`) | `scaleout_topology` | new |
| **mesh_graph** (`MeshGraph` class) | `fabric` | **no — stays** |
| runtime protos (router/port/intermesh) | `fabric` | no |

---

## 3. The only real code change: relocate 3 functions

`mesh_graph` stays in `fabric`, so the **3 functions that touch `MeshGraph`** move out of the solver/mapper-utils
into a new fabric-side TU `tt_metal/fabric/topology_logical_adapters.cpp`:

- `build_adjacency_graph_logical(const MeshGraph&)`  ← `topology_solver.cpp`
- `build_adjacency_map_logical(const MeshGraph&)`  ← `topology_mapper_utils.cpp`
- `build_logical_multi_mesh_adjacency_graph(const MeshGraph&)`  ← `topology_mapper_utils.cpp`

Everything else in the solver/mapper-utils is `MeshGraph`-free (the runtime-free path uses the **MGD** overload,
which `generate_rank_bindings` already uses). No constructor/signature changes — just relocating 3 definitions.

> Keeping `mesh_graph` in fabric means `mesh_graph.cpp` and `fabric_host_utils.*` stay **untouched** — simpler
> than moving it.

---

## 4. Consumer linking (the one cost, build-verified)

`scaleout_topology` uses **hidden visibility** (protobuf convention, same as `scaleout_tools`), so `libtt_metal`
doesn't re-export its symbols. Every target that uses the moved types must **explicitly** add
`PRIVATE TT::ScaleoutTopology` — confirmed by `generate_rank_bindings` failing to link until it does.

≈ a dozen targets need this one-line add: `generate_rank_bindings`, `generate_mgd`, `run_cluster_validation`, and
the `tt_fabric` router / discovery tests. Mechanical; mirrors what `scaleout_tools` already requires.

---

## 5. How to use it

### In tt-metal (`generate_rank_bindings` — the FSD-mapping feature)

```cmake
target_link_libraries(generate_rank_bindings PRIVATE tt_metal TT::ScaleoutTopology scaleout_tools ...)
```
```cpp
#include "fsd_to_psd/fsd_to_psd.hpp"

PhysicalSystemDescriptor psd = [&]() -> PhysicalSystemDescriptor {
    const auto& fsd_path = MetalContext::instance().rtoptions().get_factory_system_descriptor_path();
    if (!fsd_path.empty()) {
        auto fsd       = tt::scaleout_tools::load_fsd_textproto(fsd_path);   // parse FSD
        auto psd_proto = tt::scaleout_tools::build_psd_from_fsd(fsd);        // offline FSD → PSD proto
        return PhysicalSystemDescriptor(psd_proto);                         // proto → C++ PSD
    }
    return run_psd_discovery();                                            // live fallback
}();
// unchanged downstream:
auto phys = experimental::tt_fabric::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
auto res  = experimental::tt_fabric::map_multi_mesh_to_physical(logical_adj, phys, config, ...);
```

### In Fabric Manager (as a static library, no `libtt_metal`)

**CMake** — FM builds tt-metal from its vendored submodule:
```cmake
target_link_libraries(tt-fabric-manager-controller PRIVATE TT::ScaleoutTopology protobuf::libprotobuf)
```
Or FM links a prebuilt tt-metalium (requires the lib installed):
```cmake
# tt-metal side, one-time:
install(TARGETS scaleout_topology EXPORT Metalium ARCHIVE COMPONENT metalium-dev)
install(FILES tools/scaleout/fsd_to_psd/fsd_to_psd.hpp DESTINATION include/tt-metalium/fsd_to_psd)
# FM side:
find_package(Metalium REQUIRED)
target_link_libraries(tt-fabric-manager-controller PRIVATE TT::ScaleoutTopology protobuf::libprotobuf)
```
**Code** — FM drops its FSD→PSD fork and calls the shared builder:
```cpp
#include "fsd_to_psd/fsd_to_psd.hpp"

auto psd_proto = tt::scaleout_tools::build_psd_from_fsd(fsd);   // was FM's own copy of this
// FM's existing mapping pipeline is unchanged:
auto psd  = tt::tt_metal::PhysicalSystemDescriptor(psd_proto);
auto phys = tt::tt_metal::experimental::tt_fabric::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
auto map  = tt::tt_metal::experimental::tt_fabric::map_multi_mesh_to_physical(logical_adj, phys, config, ...);
```
FM's controller links **only `TT::ScaleoutTopology`** — no device / HAL / dispatch.

**Two FM gotchas:** (1) it must name `TT::ScaleoutTopology` explicitly (hidden visibility); (2) it must use the
protos this lib provides — not also compile its own copies — or protobuf aborts on duplicate descriptors.

---

## 6. API surface

```cpp
// tools/scaleout/fsd_to_psd/fsd_to_psd.hpp   (namespace tt::scaleout_tools)
FactorySystemDescriptor load_fsd_textproto(const std::string& path);
tt::fabric::proto::PhysicalSystemDescriptor build_psd_from_fsd(const FactorySystemDescriptor&);
std::vector<tt::fabric::proto::PhysicalSystemDescriptor> build_psds_from_fsd(const FactorySystemDescriptor&);
// + the topology types now in this lib: PhysicalSystemDescriptor(proto ctor), MeshGraphDescriptor,
//   PhysicalGroupingDescriptor, build_physical_multi_mesh_adjacency_graph, build_logical_..._graph(MGD),
//   map_multi_mesh_to_physical
```

---

## 7. Status

- ✅ Feasibility proven (superset branch that also moved `mesh_graph`): `scaleout_topology` + `test_fsd_to_psd`
  build clean; the test links with **no `libtt_metal`**; 5/5 FSD→PSD gtests pass.
- ✅ `fabric` compiles with the sources removed; `generate_rank_bindings` needs the explicit
  `TT::ScaleoutTopology` link (confirmed).
- ☐ Rework the branch to this "mesh_graph stays" layout (less code), add the ~dozen consumer links, rebuild.

## 8. Steps

1. New `tt_metal/fabric/topology_logical_adapters.cpp` — the 3 relocated `MeshGraph` functions.
2. `sources.cmake` / `CMakeLists.txt`: define `TOPOLOGY_SOURCES` + `scaleout_topology`; split protos; `fabric`
   links `TT::ScaleoutTopology`.
3. `fsd_to_psd.{hpp,cpp}` + `test_fsd_to_psd.cpp` (done) compiled into the lib.
4. Leave `mesh_graph.cpp` / `fabric_host_utils.*` untouched.
5. Add `PRIVATE TT::ScaleoutTopology` to the ~dozen consumers.
