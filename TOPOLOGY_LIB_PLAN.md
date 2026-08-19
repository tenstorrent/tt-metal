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

They meet at the **FSD proto**: `CablingGenerator` produces it; `build_physical_descriptor()` consumes it. No cycles.

---

## 2. What goes where

| Piece | Home | Moves? |
|---|---|---|
| FSD proto, FSD `utils`/`query` | `scaleout_tools` | no |
| **MGD, PSD, PGD** (proto + class) | `scaleout_topology` | **yes** |
| topology **solver** + **mapper utils** | `scaleout_topology` | **yes** |
| **FSD → PSD** builder (`physical_descriptor_builder`) | `scaleout_topology` | new |
| **mesh_graph** (`MeshGraph` class) | `fabric` | **no — stays** |
| runtime protos (router/port/intermesh) | `fabric` | no |

---

## 3. The only real code change: relocate 3 functions

`mesh_graph` stays in `fabric`, so the **3 functions that touch `MeshGraph`** are handled as follows (no new file):

- `build_adjacency_map_logical(const MeshGraph&)` — had no callers → **deleted**.
- `build_adjacency_graph_logical(const MeshGraph&)` (was `topology_solver.cpp`) → **moved into `topology_mapper.cpp`**.
- `build_logical_multi_mesh_adjacency_graph(const MeshGraph&)` (was `topology_mapper_utils.cpp`) → **moved into
  `topology_mapper.cpp`**, forwarding to an exposed "parts" overload that stays in `topology_mapper_utils.cpp`.

Everything else in the solver/mapper-utils is `MeshGraph`-free (the runtime-free path uses the **MGD** overload,
which `generate_rank_bindings` already uses).

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

## 4b. Header locations / includes

- **PSD / MGD / PGD / solver / mapper-utils headers don't move** — they're already tt-metalium API headers under
  `tt_metal/api/tt-metalium/experimental/fabric/`, included as `<tt-metalium/experimental/fabric/*.hpp>`. Only the
  `.cpp` *compilation* moves to `scaleout_topology`; the headers and their include paths are unchanged.
- **`physical_descriptor_builder.hpp` is promoted** into that same API tree →
  `tt_metal/api/tt-metalium/experimental/fabric/physical_descriptor_builder.hpp`, included as
  `<tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>` (consistent with its siblings, installable for FM). The `.cpp`
  stays where it is — no source files move, only this one header is promoted.
- `.pb.h` proto headers are internal build artifacts (resolved from the generated dir), not part of the public
  include surface.

## 5. How to use it

### In tt-metal (`generate_rank_bindings` — the FSD-mapping feature)

```cmake
target_link_libraries(generate_rank_bindings PRIVATE tt_metal TT::ScaleoutTopology scaleout_tools ...)
```
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

### In Fabric Manager (as a static library, no `libtt_metal`)

**CMake** — FM builds tt-metal from its vendored submodule:
```cmake
target_link_libraries(tt-fabric-manager-controller PRIVATE TT::ScaleoutTopology protobuf::libprotobuf)
```
Or FM links a prebuilt tt-metalium (requires the lib installed):
```cmake
# tt-metal side, one-time:
install(TARGETS scaleout_topology EXPORT Metalium ARCHIVE COMPONENT metalium-dev)
# physical_descriptor_builder.hpp already lives in the tt-metalium API tree, installed with the other tt-metalium headers
# FM side:
find_package(Metalium REQUIRED)
target_link_libraries(tt-fabric-manager-controller PRIVATE TT::ScaleoutTopology protobuf::libprotobuf)
```
**Code** — FM drops its FSD→PSD fork and calls the shared builder:
```cpp
#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>

auto psd_proto = tt::scaleout_tools::build_physical_descriptor(fsd);   // was FM's own copy of this
// FM's existing mapping pipeline is unchanged:
auto psd  = tt::tt_metal::PhysicalSystemDescriptor(psd_proto);
auto phys = tt::tt_metal::experimental::tt_fabric::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd);
auto map  = tt::tt_metal::experimental::tt_fabric::map_multi_mesh_to_physical(logical_adj, phys, config, ...);
```
FM's controller links **only `TT::ScaleoutTopology`** — no device / HAL / dispatch.

**Two FM gotchas:** (1) it must name `TT::ScaleoutTopology` explicitly (hidden visibility); (2) it must use the
protos this lib provides — not also compile its own copies — or protobuf aborts on duplicate descriptors.

---

## 5b. Why this keeps Fabric Manager lightweight (no `libtt_metal`)

The whole point of a *runtime-free* `TT::ScaleoutTopology` is that FM links **only that one static library** (plus a
few small support libs) and **never links the Metalium runtime**. FM does **not** import the whole `libtt_metal`.

**What FM's link closure actually is:**
```
tt-fabric-manager-controller
└─ TT::ScaleoutTopology              (libscaleout_topology.a — PSD/MGD/PGD, solver, mapper-utils, FSD→PSD)
   ├─ TT::ScaleoutTools              (FSD proto + board lib)
   ├─ cadical                        (SAT solver)
   ├─ protobuf · umd · fmt · tt-logger · TT::STL · TT::Metalium::HostDevCommon   (small headers/defs libs)
   └─ ✗ libtt_metal                  NOT linked
```

**What FM therefore does NOT pull in** (the entire Metalium runtime, i.e. `libtt_metal`):
device management · HAL · command-queue / dispatch · JIT kernel build · LLRT · kernel infrastructure ·
the UMD device *driver* runtime. None of it is in `TT::ScaleoutTopology`'s link closure, because the topology code
is pure data-structure / graph work that touches no hardware.

**How the lib achieves this:**
- It links only the lean set above. It reads tt-metalium **headers** through include-dirs
  (`$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>`) — a **compile-time include path only**, *not* a link
  dependency on `libtt_metal`.
- The runtime-free path uses the **MGD-based** logical builder + PSD/PGD, none of which touch `MeshGraph`. Because
  `mesh_graph` stays in `fabric`/`libtt_metal`, staying off that one type is exactly what keeps the runtime out of
  the closure. (Reaching `MeshGraph` would drag `libtt_metal` back in.)

**FM CMake — the lib, not the runtime:**
```cmake
# No `tt_metal` / `TT::Metalium` on this line — only the lean topology lib + protobuf.
target_link_libraries(tt-fabric-manager-controller PRIVATE TT::ScaleoutTopology protobuf::libprotobuf)
```

**Verification:** `TT::ScaleoutTopology`'s link interface pulls no runtime target (only `TT::ScaleoutTools` +
protobuf/cadical/umd/fmt/tt-logger/STL/HostDevCommon). Note the current FSD→PSD test runs inside `fabric_unit_tests`,
which *does* link `tt_metal`, so that binary is **not** a proof of runtime-free linkage. A dedicated executable that
links **only** `TT::ScaleoutTopology` (and exercises MGD + FSD→PSD) is a follow-up to lock this down (see review).

**Only compile-time requirement:** the tt-metalium API *headers* must be available (installed with the package, or
via FM's vendored submodule) and FM must use the same protobuf. Those are headers/ABI — not the runtime library.

---

## 6. API surface

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

**Two entry points, by consumer need** (the FSD proto is inherent to the *input*, so a fully proto-free API is the
path-based one):
- `build_physical_descriptor_from_file(path)` → returns the C++ `PhysicalSystemDescriptor`; caller needs **no** protobuf headers.
  Best for tt-run / `generate_rank_bindings` (they only have a path). **Implemented + build-verified.**
- `build_physical_descriptor(FactorySystemDescriptor)` → proto in/out, for FM which already holds (and re-serializes) the
  proto over gRPC.

---

## 7. Status — implemented on this branch

- ✅ `TT::ScaleoutTopology` created (PSD/MGD/PGD + solver + mapper-utils + `physical_descriptor_builder`), hidden
  visibility; `mesh_graph` + the runtime `TopologyMapper` class stay in `fabric`; `fabric` links the new lib.
- ✅ Builds clean: `scaleout_topology`, `fabric`, `libtt_metal.so`, `generate_rank_bindings`, `fabric_unit_tests`,
  `test_physical_discovery`, and the scaleout tools/tests — with the explicit `TT::ScaleoutTopology` link.
- ✅ CPU tests pass: `fabric_unit_tests` topology suites (MeshGraphDescriptor / PhysicalGroupingDescriptor /
  TopologySolver / TopologySatEncoder / TopologyMapperUtils / PhysicalDescriptorBuilder) **280 passed, 0 failed**;
  `test_cabling_descriptor_mgd_generation` 6/6; `test_physical_discovery` 5 pass/1 skip. The only failing suites
  need real hardware (galaxy / live GSD).

## 8. What was done

1. The 3 `MeshGraph` functions: one deleted (unused), two moved into `topology_mapper.cpp` (§3).
2. `sources.cmake` / `CMakeLists.txt`: `TOPOLOGY_SOURCES` + `scaleout_topology` (hidden vis); protos split; `fabric`
   links `TT::ScaleoutTopology`.
3. `physical_descriptor_builder.{hpp,cpp}` (header in the tt-metalium API tree, `.cpp` in `tt_metal/fabric/`)
   compiled into the lib; test in `tests/tt_metal/tt_fabric/fabric_router/` (part of `fabric_unit_tests`).
4. `mesh_graph.cpp` / `fabric_host_utils.*` left untouched.
5. `PRIVATE TT::ScaleoutTopology` added to the consumers that reference the moved types.
