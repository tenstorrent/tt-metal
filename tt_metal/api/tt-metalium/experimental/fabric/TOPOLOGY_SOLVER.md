# Topology Solver Documentation

## Overview

The topology solver implements a **constraint satisfaction problem (CSP)** solver for **graph isomorphism/subgraph matching**. It finds a mapping from a target graph (smaller) to a global graph (larger) while satisfying constraints.

**Key Features**:
- Generic template-based design (works with any node types)
- Stateless API (thread-safe, reentrant)
- Three connection validation modes (STRICT, RELAXED, NONE)
- Fast path optimization for path graphs (O(n) instead of exponential)
- Comprehensive constraint system (required/preferred, trait-based, explicit pairs)

## Public API

### Namespace

All public types and functions are in `tt::tt_fabric` namespace:

```cpp
namespace tt::tt_fabric {
    template <typename NodeId> class AdjacencyGraph;
    template <typename TargetNode, typename GlobalNode> class MappingConstraints;
    template <typename TargetNode, typename GlobalNode> struct MappingResult;
    enum class ConnectionValidationMode;
    template <typename TargetNode, typename GlobalNode>
    MappingResult<TargetNode, GlobalNode> solve_topology_mapping(...);
    std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(...);
    std::map<MeshId, AdjacencyGraph<AsicID>> build_adjacency_map_physical(...);
}
```

### Core Types

#### AdjacencyGraph

Generic graph representation that works with any node type.

```cpp
template <typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    AdjacencyGraph() = default;
    explicit AdjacencyGraph(const AdjacencyMap& adjacency_map);

    const std::vector<NodeId>& get_nodes() const;
    const std::vector<NodeId>& get_neighbors(NodeId node) const;
    void print_adjacency_map(const std::string& graph_name = "Graph") const;
};
```

#### MappingConstraints

Unified constraint system for specifying required and preferred mappings.

```cpp
template <typename TargetNode, typename GlobalNode>
class MappingConstraints {
public:
    // Trait-based constraints (one-to-many)
    template <typename TraitType>
    void add_required_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits,
        const std::map<GlobalNode, TraitType>& global_traits);

    template <typename TraitType>
    void add_preferred_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits,
        const std::map<GlobalNode, TraitType>& global_traits);

    // Explicit pair constraints (one-to-one)
    void add_required_constraint(TargetNode target_node, GlobalNode global_node);
    void add_preferred_constraint(TargetNode target_node, GlobalNode global_node);

    // Query Interface
    const std::set<GlobalNode>& get_valid_mappings(TargetNode target) const;
    const std::set<GlobalNode>& get_preferred_mappings(TargetNode target) const;
    bool is_valid_mapping(TargetNode target, GlobalNode global) const;
};
```

**Constraint Types**:
- **Required**: MUST be satisfied - solver fails if any cannot be met
- **Preferred**: SHOULD be satisfied - solver optimizes for them but may skip if needed
- **Trait constraints**: Constrain nodes with same trait value to map together
- **Explicit pairs**: Pin specific target nodes to specific global nodes
- All constraints are intersected - a target node must satisfy ALL required constraints simultaneously

#### ConnectionValidationMode

Enum controlling how connection counts (multi-edge channel counts) are validated:

```cpp
enum class ConnectionValidationMode {
    STRICT,   ///< Require exact channel counts, fail if not met
    RELAXED,  ///< Prefer correct channel counts, allow mismatches with warnings (default)
    NONE      ///< Only check edge existence, ignore channel counts
};
```

**Mode Behavior**:
- **STRICT**: Fails if any edge doesn't have sufficient channels
- **RELAXED**: Allows insufficient channels but adds warnings to `result.warnings`
- **NONE**: Completely ignores channel counts, only validates edge existence

#### MappingResult

Result structure containing mapping outcome, success status, error messages, warnings, and statistics.

```cpp
template <typename TargetNode, typename GlobalNode>
struct MappingResult {
    bool success = false;
    std::string error_message;
    std::vector<std::string> warnings;

    std::map<TargetNode, GlobalNode> target_to_global;
    std::map<GlobalNode, TargetNode> global_to_target;

    struct {
        size_t required_satisfied = 0;
        size_t preferred_satisfied = 0;
        size_t preferred_total = 0;
    } constraint_stats;

    struct {
        size_t dfs_calls = 0;
        size_t backtrack_count = 0;
        std::chrono::milliseconds elapsed_time{};
    } stats;

    void print(const AdjacencyGraph<TargetNode>& target_graph) const;
};
```

**Important**: Even if `success = false`, `target_to_global` and `global_to_target` contain the best partial mapping found, allowing users to inspect what progress was made.

### Main Solver Function

```cpp
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED);
```

## Usage Examples

### Basic Usage

```cpp
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

using namespace tt::tt_fabric;

// Build graphs
AdjacencyGraph<FabricNodeId> target_graph(mesh_graph, MeshId{0});
AdjacencyGraph<AsicID> global_graph(psd, MeshId{0}, asic_to_rank);

// Build constraints
MappingConstraints<FabricNodeId, AsicID> constraints;
constraints.add_required_constraint(target_node, global_node);
constraints.add_preferred_constraint(target_node, global_node);

// Solve
auto result = solve_topology_mapping(target_graph, global_graph, constraints);

if (result.success) {
    for (const auto& [target, global] : result.target_to_global) {
        // Process mapping
    }
} else {
    std::cerr << "Mapping failed: " << result.error_message << std::endl;
    // Partial mapping still available in result.target_to_global
}
```

### Connection Validation Modes

```cpp
// Strict mode - fails on channel count mismatches
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::STRICT);

// Relaxed mode (default) - allows mismatches with warnings
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::RELAXED);
if (result.success && !result.warnings.empty()) {
    // Check warnings for channel count mismatches
}

// None mode - ignores channel counts
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::NONE);
```

### Helper Functions

```cpp
// Build logical graphs from MeshGraph
std::map<MeshId, AdjacencyGraph<FabricNodeId>> logical_graphs =
    build_adjacency_map_logical(mesh_graph);

// Build physical graphs from PhysicalSystemDescriptor
std::map<MeshId, AdjacencyGraph<AsicID>> physical_graphs =
    build_adjacency_map_physical(psd, asic_id_to_mesh_rank);
```

## Architecture

### File Structure

```
tt_metal/api/tt-metalium/experimental/fabric/
├── topology_solver.hpp              # Public API (tt::tt_fabric)
└── topology_solver.tpp              # Public template implementations

tt_metal/fabric/
├── topology_solver_internal.hpp     # Internal types (tt::tt_fabric::detail)
├── topology_solver_internal.tpp    # Internal template implementations
└── topology_solver.cpp              # Implementation file
```

**Important**: Internal implementation (`tt::tt_fabric::detail`) is in `tt_metal/fabric/` directory, not part of public API.

### Internal Modules

All implementation details are in `tt::tt_fabric::detail` namespace:

1. **GraphIndexData**: Converts `AdjacencyGraph` to indexed representation for O(1) lookups
2. **ConstraintIndexData**: Converts `MappingConstraints` to index-based representation
3. **SearchHeuristic**: Unified node selection and candidate generation with integer cost-based priority
4. **ConsistencyChecker**: Local and forward consistency validation during DFS
5. **PathGraphDetector**: Fast path optimization for path graphs (O(n) instead of exponential)
6. **DFSSearchEngine**: Core backtracking search with memoization
7. **MappingValidator**: Final mapping validation and result building

### Algorithm Flow

```
solve_topology_mapping(target_graph, global_graph, constraints)
│
├─► Preprocessing: Build GraphIndexData and ConstraintIndexData
│
├─► Fast Path Detection: Check if target is path graph, try O(n) algorithm
│
├─► General DFS Search:
│   ├─ SearchHeuristic::select_and_generate_candidates() - select node, generate ordered candidates
│   ├─ ConsistencyChecker::check_local_consistency() - validate with mapped neighbors
│   ├─ ConsistencyChecker::check_forward_consistency() - ensure future nodes have options
│   └─ Backtracking with memoization
│
└─► Validation: MappingValidator validates mapping and builds result
```

### Key Implementation Details

#### SearchHeuristic Cost System

**Important**: Hard constraints are **absolute requirements** - candidates that violate hard constraints are **filtered out entirely** before cost computation. They are not given a high cost; they are excluded from consideration.

**Hard Constraints** (must be satisfied, filtered before cost computation):
- Required constraints (pinning, trait-based restrictions)
- Graph isomorphism (edges exist to all mapped neighbors)
- Degree sufficient (global_deg >= target_deg)
- Channel counts sufficient (in STRICT mode)

**Node Selection Cost** (lower = more constrained = selected first):
```cpp
cost = (candidate_count * HARD_WEIGHT)
     - (preferred_count * SOFT_WEIGHT)
     - (mapped_neighbors * RUNTIME_WEIGHT)
```

**What these values mean**:
- **`candidate_count`**: Number of unused global nodes that satisfy all hard constraints for this target node. Only candidates passing hard constraint filtering are counted.

  Lower `candidate_count` = more constrained = selected first (MRV heuristic - Minimum Remaining Values).

- **`preferred_count`**: Number of candidates (from `candidate_count`) that are also in the preferred constraints set. This is a subset of `candidate_count`.

  Higher `preferred_count` = more preferred options = lower cost = selected earlier.

- **`mapped_neighbors`**: Number of neighbors of this target node that are already mapped. More mapped neighbors = more constraints from existing assignments = lower cost = selected earlier.

**Candidate Ordering** (lower = better = tried first):
```cpp
// Only computed for candidates that passed hard constraint filtering
cost = -is_preferred * SOFT_WEIGHT
     - channel_match_score * (varies)
     + degree_gap * RUNTIME_WEIGHT
```

**What these values mean**:
- **`is_preferred`**: 1 if this candidate is in the preferred constraints set, 0 otherwise
- **`channel_match_score`**: In RELAXED mode, rewards candidates with channel counts closer to required (exact match = best, then closest above, then closest below)
- **`degree_gap`**: Difference between global and target degree (smaller gap = better fit)

**Cost Weights**:
- `HARD_WEIGHT = 1000000` - Used for node selection priority (candidate count is primary factor for MRV)
- `SOFT_WEIGHT = 1000` - Soft constraints secondary (preferred constraints, channel matching)
- `RUNTIME_WEIGHT = 1` - Runtime optimization minor (degree gap, mapped neighbors)

**Note**: `HARD_WEIGHT` is used to prioritize nodes with fewer valid candidates (MRV heuristic), not to penalize hard constraint violations. Hard constraint violations result in candidate exclusion, not high cost.

#### Consistency Checking

- **Local Consistency**: Verifies mapped neighbors are connected in global graph
- **Forward Consistency**: Ensures future neighbors have viable candidates
- **Channel Counts**: Validated according to `ConnectionValidationMode`

#### Memoization

- Uses FNV-1a hash function to cache failed states
- Avoids revisiting previously failed partial assignments
- Reduces redundant work in search tree

#### Fast Path for Path Graphs

- Detects path graphs: 2 endpoints (degree 1), all others degree ≤ 2
- Uses O(n) path-extension algorithm instead of exponential search
- Significantly faster for linear chains

### Logging

The solver uses `tt-logger` for all messages:

```cpp
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

log_info(tt::LogFabric, "Message: {}", arg);      // Informational
log_error(tt::LogFabric, "Error: {}", error_msg);  // Errors
```

**Logging Points**:
- Mapping start: Degree histograms for target and global graphs
- Validation failures: Detailed error messages explaining problems
- Success: Statistics (DFS calls, backtracks, constraint satisfaction)

### Print Functions

For debugging, use the print functions:

```cpp
// Print adjacency map for a graph
target_graph.print_adjacency_map("Target Graph");
global_graph.print_adjacency_map("Global Graph");

// Print mapping result
result.print(target_graph);
```

## Performance Considerations

1. **Graph Size**: Worst case exponential, but pruning makes it much better in practice
2. **Constraints**: More constraints generally improve performance (smaller search space)
3. **Fast Path**: Path graphs solved in O(n) time
4. **Memoization**: Failed states cached to avoid redundant work
5. **Statistics**: Use `result.stats` to understand solver performance

## Thread Safety

The solver is **stateless** - all state is passed as parameters:
- ✅ Thread-safe: Multiple threads can call `solve_topology_mapping()` concurrently
- ✅ Reentrant: Can be called recursively
- ✅ No global state: Each call is independent

## Template Requirements

Both `TargetNode` and `GlobalNode` must be:
- Comparable (for use in `std::map` and `std::set`)
- Copyable or movable
- Default constructible (if used in certain contexts)

Common types: `FabricNodeId`, `AsicID`, integers, strings, or custom types with comparison operators.
