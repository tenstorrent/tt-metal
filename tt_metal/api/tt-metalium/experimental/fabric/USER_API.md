# Topology Solver User API

## Overview

The topology solver provides a generic, stateless API for finding isomorphic embeddings of a target graph within a global graph while satisfying constraints.

## Core Components

### AdjacencyGraph

Generic graph representation that works with any node type. Provides minimal interface for querying graph structure.

```cpp
template <typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    // Construction
    AdjacencyGraph() = default;
    explicit AdjacencyGraph(const AdjacencyMap& adjacency_map);

    // Query Interface
    const std::vector<NodeId>& get_nodes() const;
    const std::vector<NodeId>& get_neighbors(NodeId node) const;
};
```

**Key Points**:
- Works with any node type
- Minimal interface: `get_nodes()`, `get_neighbors()`
- Can construct from adjacency maps

### MappingConstraints

Unified constraint system for specifying required and preferred mappings. Represents all constraints internally as trait maps. Both trait-based constraints (one-to-many) and explicit pair constraints (one-to-one) are unified into a single intersection-based representation.

```cpp
template <typename TargetNode, typename GlobalNode>
class MappingConstraints {
public:
    // Construction
    MappingConstraints() = default;
    MappingConstraints(
        const std::set<std::pair<TargetNode, GlobalNode>>& required_constraints,
        const std::set<std::pair<TargetNode, GlobalNode>>& preferred_constraints = {});

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

**About Constraints**:
- **Required constraints**: MUST be satisfied - solver fails if any cannot be met
- **Preferred constraints**: SHOULD be satisfied - solver optimizes for them but may skip if needed
- **Trait constraints**: Constrain nodes with same trait value to map together (e.g., host rank matching)
- **Explicit pairs**: Pin specific target nodes to specific global nodes
- All constraints are intersected - a target node must satisfy ALL required constraints simultaneously

### ConnectionValidationMode

Enum controlling how connection counts (multi-edge channel counts) are validated:

```cpp
enum class ConnectionValidationMode {
    STRICT,   ///< Strict mode: require exact channel counts, fail if not met
    RELAXED,  ///< Relaxed mode: prefer correct channel counts, but allow mismatches with warnings
    NONE      ///< No validation: only check edge existence, ignore channel counts
};
```

### MappingResult

Result structure containing the mapping outcome, success status, error messages, warnings, and statistics.

```cpp
template <typename TargetNode, typename GlobalNode>
struct MappingResult {
    /// Whether the mapping was successful
    bool success = false;

    /// Error message if mapping failed
    std::string error_message;

    /// Warning messages (e.g., relaxed mode connection count mismatches)
    std::vector<std::string> warnings;

    /// Mapping from target nodes to global nodes
    std::map<TargetNode, GlobalNode> target_to_global;

    /// Reverse mapping from global nodes to target nodes
    std::map<GlobalNode, TargetNode> global_to_target;

    /// Statistics about constraint satisfaction
    struct {
        size_t required_satisfied = 0;   ///< Number of required constraints satisfied
        size_t preferred_satisfied = 0;  ///< Number of preferred constraints satisfied
        size_t preferred_total = 0;      ///< Total number of preferred constraints
    } constraint_stats;

    /// Statistics about the solving process
    struct {
        size_t dfs_calls = 0;                      ///< Number of DFS calls made
        size_t backtrack_count = 0;                ///< Number of backtracks performed
        std::chrono::milliseconds elapsed_time{};  ///< Time taken to solve
    } stats;
};
```

### solve_topology_mapping

Main solver function that performs constraint satisfaction search to find a valid mapping.

```cpp
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED);
```

**Parameters**:
- `target_graph`: The target graph (subgraph pattern to find)
- `global_graph`: The global graph (larger host graph that contains the target)
- `constraints`: The mapping constraints to satisfy
- `connection_validation_mode`: How to validate connection counts (default: RELAXED)

**Returns**: `MappingResult` containing success status, bidirectional mappings, warnings, and statistics

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
    // Use result.target_to_global mapping
    for (const auto& [target, global] : result.target_to_global) {
        // Process mapping
    }
} else {
    // Handle error
    std::cerr << "Mapping failed: " << result.error_message << std::endl;
}
```

### Using Connection Validation Modes

#### Strict Mode

```cpp
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::STRICT);

// Fails if any edge doesn't have sufficient channels
if (!result.success) {
    // Error: channel count mismatch
}
```

#### Relaxed Mode (Default)

```cpp
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::RELAXED);

if (result.success) {
    // Check warnings for channel count mismatches
    for (const auto& warning : result.warnings) {
        std::cout << "Warning: " << warning << std::endl;
    }
}
```

#### None Mode

```cpp
auto result = solve_topology_mapping(
    target_graph, global_graph, constraints,
    ConnectionValidationMode::NONE);

// Only checks edge existence, ignores channel counts
```

### Constraint Examples

#### Required Constraints

```cpp
MappingConstraints<FabricNodeId, AsicID> constraints;

// Pin specific target node to specific global node
constraints.add_required_constraint(target_node_0, asic_5);

// Add trait constraint: nodes on same host rank must map to ASICs on same host rank
constraints.add_required_trait_constraint(node_to_host_rank, asic_to_host_rank);

// All constraints are intersected - target_node_0 must map to asic_5 AND satisfy trait constraints
```

#### Preferred Constraints

```cpp
MappingConstraints<FabricNodeId, AsicID> constraints;

// Suggest a mapping (doesn't restrict valid mappings)
constraints.add_preferred_constraint(target_node_1, asic_6);

// Add preferred trait constraint
constraints.add_preferred_trait_constraint(node_to_rack, asic_to_rack);

// Solver optimizes for these but can choose other mappings if needed
```

#### Combined Constraints

```cpp
MappingConstraints<FabricNodeId, AsicID> constraints;

// Required: pin node 0 to asic 5
constraints.add_required_constraint(target_node_0, asic_5);

// Required: host rank matching
constraints.add_required_trait_constraint(node_to_host_rank, asic_to_host_rank);

// Preferred: rack matching (optimize for, but not required)
constraints.add_preferred_trait_constraint(node_to_rack, asic_to_rack);

// Preferred: suggest node 1 → asic 6
constraints.add_preferred_constraint(target_node_1, asic_6);

// Solve - enforces required constraints, optimizes for preferred
auto result = solve_topology_mapping(target_graph, global_graph, constraints);
```

### Helper Functions

#### Building Graphs from MeshGraph

```cpp
// Build logical graphs from MeshGraph
std::map<MeshId, AdjacencyGraph<FabricNodeId>> logical_graphs =
    build_adjacency_map_logical(mesh_graph);

// Build physical graphs from PhysicalSystemDescriptor
std::map<MeshId, AdjacencyGraph<AsicID>> physical_graphs =
    build_adjacency_map_physical(psd, asic_id_to_mesh_rank);
```

## Error Handling

### Checking Results

```cpp
auto result = solve_topology_mapping(target_graph, global_graph, constraints);

if (result.success) {
    // Success - use mapping
    for (const auto& [target, global] : result.target_to_global) {
        // Process mapping
    }
    
    // Check constraint satisfaction
    std::cout << "Required constraints satisfied: " 
              << result.constraint_stats.required_satisfied << std::endl;
    std::cout << "Preferred constraints satisfied: " 
              << result.constraint_stats.preferred_satisfied 
              << " / " << result.constraint_stats.preferred_total << std::endl;
    
    // Check warnings (relaxed mode)
    if (!result.warnings.empty()) {
        std::cout << "Warnings:" << std::endl;
        for (const auto& warning : result.warnings) {
            std::cout << "  - " << warning << std::endl;
        }
    }
    
    // Check statistics
    std::cout << "DFS calls: " << result.stats.dfs_calls << std::endl;
    std::cout << "Backtracks: " << result.stats.backtrack_count << std::endl;
    std::cout << "Elapsed time: " << result.stats.elapsed_time.count() << " ms" << std::endl;
} else {
    // Failure - check error message
    std::cerr << "Mapping failed: " << result.error_message << std::endl;
}
```

### Common Error Scenarios

1. **No Valid Mapping**: Target graph cannot be embedded in global graph
   - Error message: "No valid mapping found"
   - Check constraints and graph connectivity

2. **Unsatisfiable Constraints**: Required constraints conflict
   - Error message: "No valid mappings for target node X"
   - Check constraint intersections

3. **Strict Mode Channel Count Mismatch**: Physical edge doesn't have enough channels
   - Error message: "Channel count mismatch: ..."
   - Use RELAXED mode to allow with warnings

## Best Practices

1. **Validate Constraints**: Check `constraints.validate()` before solving if you're building constraints dynamically

2. **Use Relaxed Mode**: Default RELAXED mode provides flexibility while still optimizing for correct channel counts

3. **Check Warnings**: Always check `result.warnings` in relaxed mode to see if channel counts don't match

4. **Use Preferred Constraints**: Guide the solver with preferred constraints rather than making everything required

5. **Handle Errors**: Always check `result.success` before using the mapping

6. **Check Statistics**: Use `result.stats` to understand solver performance and tune constraints if needed

## Template Parameters

Both `TargetNode` and `GlobalNode` must be:
- Comparable (for use in `std::map` and `std::set`)
- Copyable or movable
- Default constructible (if used in certain contexts)

Common types:
- `FabricNodeId` for target nodes
- `AsicID` for global nodes
- Any integer or string type
- Custom types with proper comparison operators

## Thread Safety

The solver is **stateless** - all state is passed as parameters. This means:
- ✅ Thread-safe: Multiple threads can call `solve_topology_mapping()` concurrently
- ✅ Reentrant: Can be called recursively (though not recommended)
- ✅ No global state: Each call is independent

## Performance Considerations

1. **Graph Size**: Performance degrades exponentially with graph size in worst case, but pruning makes it much better in practice

2. **Constraints**: More constraints generally improve performance (smaller search space)

3. **Fast Path**: Path graphs (linear chains) are solved in O(n) time

4. **Memoization**: Failed states are cached to avoid redundant work

5. **Statistics**: Use `result.stats` to understand solver performance for your use case
