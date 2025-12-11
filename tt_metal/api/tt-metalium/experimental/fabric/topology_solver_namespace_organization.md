# Topology Solver Namespace Organization

## Overview

This document shows the namespace organization for the topology solver, clearly separating public API from internal implementation.

## Namespace Structure

```
tt::tt_fabric                    (Public API - users interact with this)
│
├── AdjacencyGraph<T>            (Public graph representation)
├── MappingConstraints<T, G>    (Public constraint system)
├── MappingResult<T, G>         (Public result type)
├── solve_topology_mapping()     (Public solver function)
├── build_adjacency_map_logical() (Public helper)
└── build_adjacency_map_physical() (Public helper)
│
└── detail                       (Internal implementation - hidden from users)
    │
    ├── GraphIndexData<T, G>    (Graph preprocessing)
    ├── ConstraintIndexData<T, G> (Constraint processing)
    ├── NodeSelector<T, G>      (Heuristic selection)
    ├── CandidateGenerator<T, G> (Candidate generation)
    ├── ConsistencyChecker<T, G> (Consistency validation)
    ├── PathGraphDetector<T, G> (Fast path optimization)
    ├── DFSSearchEngine<T, G>   (Core search engine)
    └── MappingValidator<T, G>  (Result validation)
```

## File Organization

### Public API Files

**`topology_solver.hpp`** (Public declarations)
```cpp
namespace tt::tt_fabric {
    // Public types and functions only
    template <typename NodeId> class AdjacencyGraph;
    template <typename T, typename G> class MappingConstraints;
    template <typename T, typename G> struct MappingResult;
    template <typename T, typename G> MappingResult<T, G> solve_topology_mapping(...);
    std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(...);
    std::map<MeshId, AdjacencyGraph<AsicID>> build_adjacency_map_physical(...);
}
```

**`topology_solver.tpp`** (Public template implementations)
```cpp
namespace tt::tt_fabric {
    // Template method implementations for public types
    // Includes solve_topology_mapping() implementation
    // which uses detail:: types internally
}
```

**`topology_solver.cpp`** (Public non-template implementations)
```cpp
namespace tt::tt_fabric {
    // Non-template implementations for build_adjacency_map_* functions
    // These stay in tt::tt_fabric namespace
}
```

### Internal Implementation Files

**`tt_metal/fabric/topology_solver_internal.hpp`** (Internal declarations - NOT part of public API)
```cpp
namespace tt::tt_fabric::detail {
    // All internal types declared here
    template <typename T, typename G> struct GraphIndexData;
    template <typename T, typename G> struct ConstraintIndexData;
    template <typename T, typename G> struct NodeSelector;
    template <typename T, typename G> struct CandidateGenerator;
    template <typename T, typename G> struct ConsistencyChecker;
    template <typename T, typename G> struct PathGraphDetector;
    template <typename T, typename G> class DFSSearchEngine;
    template <typename T, typename G> struct MappingValidator;
}
```

**`tt_metal/fabric/topology_solver.cpp`** (Internal implementations - non-template)
```cpp
namespace tt::tt_fabric::detail {
    // Non-template function implementations
    // Template specializations if needed
}
```

**Note**: The internal header is in the implementation directory (`tt_metal/fabric/`), not the public API directory, so it's not part of the external interface.

## Code Examples

### Public API Usage (User Code)

```cpp
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

using namespace tt::tt_fabric;

// User code only sees public API
AdjacencyGraph<FabricNodeId> target_graph = ...;
AdjacencyGraph<AsicID> global_graph = ...;
MappingConstraints<FabricNodeId, AsicID> constraints = ...;

auto result = solve_topology_mapping(target_graph, global_graph, constraints);
// detail:: types are NOT accessible
```

### Internal Implementation (Solver Code)

```cpp
// In topology_solver.tpp
namespace tt::tt_fabric {

template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(...) {
    // Use detail namespace internally
    using namespace detail;

    // Internal types are accessible here
    auto graph_data = build_graph_index_data(target_graph, global_graph);
    auto constraint_data = build_constraint_index_data(constraints, graph_data);

    DFSSearchEngine<TargetNode, GlobalNode> engine;
    // ... use internal types ...
}

} // namespace tt::tt_fabric
```

## Visibility Rules

### Public (`tt::tt_fabric`)
- ✅ Visible to users
- ✅ Documented in public API
- ✅ Stable interface (changes require deprecation)
- ✅ Types: `AdjacencyGraph`, `MappingConstraints`, `MappingResult`
- ✅ Functions: `solve_topology_mapping()`, `build_adjacency_map_*()`

### Internal (`tt::tt_fabric::detail`)
- ❌ NOT visible to users
- ❌ Can change without notice
- ❌ Implementation details only
- ❌ Types: `GraphIndexData`, `ConstraintIndexData`, `NodeSelector`, etc.
- ❌ Functions: All helper functions

## Include Dependencies

```
User Code
  ↓
tt_metal/api/tt-metalium/experimental/fabric/topology_solver.hpp (public API)
  ↓
tt_metal/api/tt-metalium/experimental/fabric/topology_solver.tpp (includes internal.hpp)
  ↓
tt_metal/fabric/topology_solver_internal.hpp (internal types - NOT public API)
  ↓
tt_metal/fabric/topology_solver.cpp (internal implementations)
```

### Include Guards

**`topology_solver.hpp`**:
```cpp
#pragma once
// Public API only
```

**`topology_solver.tpp`**:
```cpp
#ifndef TOPOLOGY_SOLVER_TPP
#define TOPOLOGY_SOLVER_TPP
#include "tt_metal/fabric/topology_solver_internal.hpp"  // Include internal types from implementation dir
// Template implementations
#endif
```

**`tt_metal/fabric/topology_solver_internal.hpp`**:
```cpp
#pragma once
// Internal types only - NOT part of public API
// Located in implementation directory
```

## Migration from Current Code

### Current State
```cpp
namespace tt::tt_fabric {
    // Everything is here (public and internal mixed)
    class AdjacencyGraph { ... };
    class MappingConstraints { ... };
    // solve_topology_mapping() - TODO
}
```

### Target State
```cpp
namespace tt::tt_fabric {
    // Public API only
    class AdjacencyGraph { ... };
    class MappingConstraints { ... };
    MappingResult solve_topology_mapping(...) {
        using namespace detail;
        // Uses internal types
    }
}

namespace tt::tt_fabric::detail {
    // All internal implementation
    struct GraphIndexData { ... };
    struct ConstraintIndexData { ... };
    // ... etc
}
```

## Benefits of This Organization

1. **Clear Separation**: Public API vs. implementation details
2. **Namespace Safety**: Users can't accidentally use internal types
3. **Easier Maintenance**: Internal code can change without affecting users
4. **Better Documentation**: Only public API needs extensive docs
5. **Compile-Time Safety**: Compiler enforces namespace boundaries

## Testing Considerations

### Unit Tests
- Test internal types in `detail` namespace
- Test public API through public interface
- Use friend classes/functions if needed for testing

### Integration Tests
- Test through public API only
- Verify internal behavior indirectly
- Compare with existing implementation

## Example: Complete File Structure

```
tt_metal/api/tt-metalium/experimental/fabric/topology_solver.hpp:
  namespace tt::tt_fabric {
    // Public declarations
    template<typename T> class AdjacencyGraph;
    template<typename T, typename G> class MappingConstraints;
    template<typename T, typename G> MappingResult<T, G> solve_topology_mapping(...);
  }
  #include "topology_solver.tpp"

tt_metal/api/tt-metalium/experimental/fabric/topology_solver.tpp:
  #include "tt_metal/fabric/topology_solver_internal.hpp"  // Include from implementation dir
  namespace tt::tt_fabric {
    // Template implementations
    template<typename T, typename G>
    MappingResult<T, G> solve_topology_mapping(...) {
      using namespace detail;
      // Use internal types
    }
  }

tt_metal/fabric/topology_solver_internal.hpp:  // NOT part of public API
  namespace tt::tt_fabric::detail {
    // Internal type declarations
    template<typename T, typename G> struct GraphIndexData;
    // ... etc
  }

tt_metal/fabric/topology_solver.cpp:
  namespace tt::tt_fabric {
    // Public non-template implementations
    std::map<...> build_adjacency_map_logical(...) { ... }
  }

  namespace tt::tt_fabric::detail {
    // Internal non-template implementations
    // Template specializations if needed
  }
```

## Summary

- **Public API**: `tt::tt_fabric` namespace only
- **Internal Implementation**: `tt::tt_fabric::detail` namespace
- **Users**: Only interact with `tt::tt_fabric` namespace
- **Implementation**: Uses `detail` namespace internally
- **Testing**: Can test both namespaces, but prefer testing through public API
