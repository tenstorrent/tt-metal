# Plan: `scaleout_topology` — a shared, runtime-free topology library

## Problem

The topology **solver + PSD + descriptors (MGD/PGD) + FSD→PSD conversion** are pure host-side graph work, but they
live in tt-metal's `fabric` → compiled into **`libtt_metal.so`** (the full Metalium runtime). There is no lean way
to reuse just this logic. Fabric Manager shows both symptoms today:

**1. It imports the whole runtime** — `tt-fabric-manager/controller/CMakeLists.txt`:
```cmake
target_link_libraries(
    fabric_manager_controller_lib
    PUBLIC
        fabric_manager_common
        TT::Metalium   # the entire libtt_metal: device mgmt · HAL · dispatch · JIT · LLRT · UMD driver
        ...
)
```

**2. And it forks the FSD→PSD conversion** to avoid re-implementing it — a hand-maintained copy of tt-metal's
builder in `tt-fabric-manager/controller/physical_system_descriptor_builder.cpp`:
```cpp
::tt::fabric::proto::PhysicalSystemDescriptor build_physical_system_descriptor(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) { /* ~200 lines, drifts from tt-metal */ }
```

FM uses none of the device runtime it drags in, and the fork doubles the maintenance surface.

---

## Solution

**Goal:** extract the runtime-free topology code — **solver + PSD + descriptors + FSD→PSD builder** — into one
lean library, **`TT::ScaleoutTopology`**, that both tt-metal tooling (tt-run / `generate_rank_bindings`) and Fabric
Manager can consume **without** dragging in the Metalium runtime.
**Constraint:** `mesh_graph` stays in `fabric`.

### The two libraries

```
TT::ScaleoutTools  (OBJECT, exists today)          TT::ScaleoutTopology  (new)
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

They meet at the **FSD proto**: `CablingGenerator` produces it; `build_physical_descriptor()` consumes it. No cycles.

**`TT::ScaleoutTools`** — `tools/scaleout/` (exists today):
```
tools/scaleout/
  board/board.hpp                              board types
  connector/connector.hpp   node/node.hpp
  cabling_generator/cabling_generator.hpp
  factory_system_descriptor/{utils,query}.hpp
  protobuf/factory_system_descriptor.proto     ← the FSD proto
```

**`TT::ScaleoutTopology`** — headers in the tt-metalium API tree, `.cpp` compiled from `tt_metal/fabric/`:
```
tt_metal/api/tt-metalium/experimental/fabric/   (headers, unchanged location)
  physical_system_descriptor.hpp     PSD (proto + C++ class)
  mesh_graph_descriptor.hpp          MGD
  physical_grouping_descriptor.hpp   PGD
  topology_solver.hpp                solver (SAT + CSP)
  topology_mapper_utils.hpp          mapper utils
  physical_descriptor_builder.hpp    FSD → PSD
  (mesh_graph.hpp, topology_mapper.hpp stay with fabric — the runtime side)

tt_metal/fabric/   (.cpp compiled into libscaleout_topology)
  physical_system_descriptor.cpp   mesh_graph_descriptor.cpp
  physical_grouping_descriptor_{core,graph_building,matching}.cpp
  topology_solver{,_sat,_sat_solver}.cpp   topology_mapper_utils.cpp
  physical_descriptor_builder.cpp
```

### The only real code change: relocate 3 functions

`mesh_graph` stays in `fabric`, so the **3 functions that touch `MeshGraph`** are handled as follows (no new file):

- `build_adjacency_map_logical(const MeshGraph&)` — had no callers → **deleted**.
- `build_adjacency_graph_logical(const MeshGraph&)` (was `topology_solver.cpp`) → **moved into `topology_mapper.cpp`**.
- `build_logical_multi_mesh_adjacency_graph(const MeshGraph&)` (was `topology_mapper_utils.cpp`) → **moved into
  `topology_mapper.cpp`**, forwarding to an exposed "parts" overload that stays in `topology_mapper_utils.cpp`.

Everything else in the solver/mapper-utils is `MeshGraph`-free (the runtime-free path uses the **MGD** overload,
which `generate_rank_bindings` already uses).

> Keeping `mesh_graph` in fabric means `mesh_graph.cpp` and `fabric_host_utils.*` stay **untouched** — simpler
> than moving it.


### Header locations / includes

- **PSD / MGD / PGD / solver / mapper-utils headers don't move** — they're already tt-metalium API headers under
  `tt_metal/api/tt-metalium/experimental/fabric/`, included as `<tt-metalium/experimental/fabric/*.hpp>`. Only the
  `.cpp` *compilation* moves to `scaleout_topology`; the headers and their include paths are unchanged.
- **`physical_descriptor_builder.hpp` is promoted** into that same API tree →
  `tt_metal/api/tt-metalium/experimental/fabric/physical_descriptor_builder.hpp`, included as
  `<tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>` (consistent with its siblings). It
  forward-declares the FSD/PSD proto types (the generated `.pb.h` are private build artifacts, not part of the
  public include surface), so the public header stays proto-free.

### How to use it — tt-metal (`generate_rank_bindings`, the FSD-mapping feature)

```cpp
#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>

PhysicalSystemDescriptor psd = [&]() -> PhysicalSystemDescriptor {
    const auto& fsd_path = MetalContext::instance().rtoptions().get_factory_system_descriptor_path();
    if (!fsd_path.empty()) {
        return tt::scaleout_tools::build_physical_descriptor_from_file(fsd_path);   // path in → C++ PSD out (no protos)
    }
    return run_psd_discovery();                                        // live fallback
}();
// unchanged downstream:
auto phys = experimental::tt_fabric::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
auto res  = experimental::tt_fabric::map_multi_mesh_to_physical(logical_adj, phys, config, ...);
```

### API surface

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/physical_descriptor_builder.hpp   (namespace tt::scaleout_tools)

// Proto-FREE entry point (recommended for path-based consumers like tt-run): file path in, C++ PSD out.
tt::tt_metal::PhysicalSystemDescriptor build_physical_descriptor_from_file(const std::string& fsd_path);

// Proto in / proto out (for consumers that already hold the FSD proto, e.g. Fabric Manager / gRPC):
FactorySystemDescriptor load_factory_descriptor(const std::string& path);
tt::fabric::proto::PhysicalSystemDescriptor build_physical_descriptor(const FactorySystemDescriptor&);
std::vector<tt::fabric::proto::PhysicalSystemDescriptor> build_physical_descriptors(const FactorySystemDescriptor&);

// + the topology types now in this lib: PhysicalSystemDescriptor(proto ctor), MeshGraphDescriptor,
//   PhysicalGroupingDescriptor, build_physical_multi_mesh_adjacency_graph, build_logical_..._graph(MGD),
//   map_multi_mesh_to_physical
```

- `build_physical_descriptor_from_file(path)` → returns the C++ `PhysicalSystemDescriptor`; caller needs **no**
  protobuf headers. Best for tt-run / `generate_rank_bindings` (they only have a path). **Implemented + verified.**
- `build_physical_descriptor(FactorySystemDescriptor)` → proto in/out, for FM which already holds (and re-serializes)
  the proto over gRPC.

### The Fabric Manager payoff (follow-up)

The end state that closes the problem above: FM drops its FSD→PSD fork and links **only the topology lib** — no
device / HAL / dispatch / JIT / LLRT.

```cpp
#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>

auto psd_proto = tt::scaleout_tools::build_physical_descriptor(fsd);   // was FM's own copy of this
auto psd  = tt::tt_metal::PhysicalSystemDescriptor(psd_proto);         // FM's existing pipeline, unchanged
auto phys = tt::tt_metal::experimental::tt_fabric::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
auto map  = tt::tt_metal::experimental::tt_fabric::map_multi_mesh_to_physical(logical_adj, phys, config, ...);
```

For FM to link the lib **without** `libtt_metal`, the lib must be self-contained at link time. Today it still
references a few symbols defined in the runtime (e.g. `mesh_coord`'s `MeshShape`/`MeshCoordinateRange` in
`tt_metal/common`), so in-tree it is consumed via `libtt_metal.so` (default visibility) rather than linked
standalone. Making it a true standalone archive — carve those small `tt_metal/common` pieces into a shared
lightweight lib, restore hidden visibility, and add a runtime-free link test — is the remaining follow-up that
delivers the FM lightweight-linking win.

---

## Status — implemented on this branch

- ✅ `TT::ScaleoutTopology` created (PSD/MGD/PGD + solver + mapper-utils + `physical_descriptor_builder`);
  `mesh_graph` + `fabric_types` + the runtime `TopologyMapper` class stay in `fabric`; `fabric` links the new lib.
- ✅ **Default visibility** — symbols exported from `libtt_metal.so` as before the split; in-tree consumers need
  no explicit link changes.
- ✅ Builds + links clean (incl. the full ASan build): `scaleout_topology`, `fabric`, `libtt_metal.so`,
  `generate_rank_bindings`, `fabric_unit_tests`, `run_fabric_manager`, and the scaleout / distributed tools+tests.
- ✅ CPU tests pass: `fabric_unit_tests` topology suites (MeshGraphDescriptor / PhysicalGroupingDescriptor /
  TopologySolver / TopologySatEncoder / TopologyMapperUtils / **PhysicalDescriptorBuilder**); no protobuf
  double-registration. The only failing suites need real hardware (galaxy / live GSD).

## Follow-ups (not in this PR)

1. Wire `build_physical_descriptor_from_file` into `generate_rank_bindings` behind the FSD RTOption from #53451.
2. Make the lib a true standalone archive (carve out `mesh_coord`/other `tt_metal/common` deps, restore hidden
   visibility, add a runtime-free link test), then migrate Fabric Manager to link **only** `TT::ScaleoutTopology`
   and delete its FSD→PSD fork.
