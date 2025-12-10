# Topology Solver Design

## Overview

Generic, stateless topology mapping solver that finds an isomorphic embedding of a **target graph** (subgraph pattern) within a **global graph** (larger host graph).

## Core Components

The solver consists of three core components:

1. **AdjacencyGraph**: Generic graph representation with minimal query interface
2. **MappingConstraints**: Unified constraint system for specifying required and preferred mappings
3. **TopologySolver**: Stateless function that performs the mapping

## Usage Examples

### Basic Usage

```cpp
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
}
```

## Core Component: AdjacencyGraph

Generic graph representation that works with any node type. Provides minimal interface for querying graph structure.

```cpp
template<typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    // Construction
    explicit AdjacencyGraph(const AdjacencyMap& adj_map);
    AdjacencyGraph(const ::tt::tt_fabric::MeshGraph& mesh_graph, MeshId mesh_id);
    AdjacencyGraph(
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        MeshId mesh_id,
        const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_id_to_mesh_rank);

    // Query Interface
    const std::vector<NodeId>& get_nodes() const;
    const std::vector<NodeId>& get_neighbors(NodeId node) const;

private:
    AdjacencyMap adj_map_;
    std::vector<NodeId> nodes_cache_;
};
```

**Key Points:**
- Works with any node type
- Minimal interface: `get_nodes()`, `get_neighbors()`
- Can construct from `MeshGraph`, `PhysicalSystemDescriptor`, or adjacency maps

## Core Component: MappingConstraints

Unified constraint system that represents all constraints internally as trait maps. Both trait-based constraints (one-to-many) and explicit pair constraints (one-to-one) are unified into a single intersection-based representation.

**About Constraints:**
- **Required constraints**: MUST be satisfied - solver fails if any cannot be met
- **Preferred constraints**: SHOULD be satisfied - solver optimizes for them but may skip if needed
- **Trait constraints**: Constrain nodes with same trait value to map together (e.g., host rank matching)
- **Explicit pairs**: Pin specific target nodes to specific global nodes
- All constraints are intersected - a target node must satisfy ALL required constraints simultaneously

```cpp
template<typename TargetNode, typename GlobalNode>
class MappingConstraints {
public:
    MappingConstraints() = default;

    /**
     * Constructor from sets of constraint pairs.
     * Converts pairs into the internal mapping representation.
     */
    MappingConstraints(
        const std::set<std::pair<TargetNode, GlobalNode>>& required_constraints,
        const std::set<std::pair<TargetNode, GlobalNode>>& preferred_constraints = {});

    // Trait-based constraints (one-to-many)
    // Constrains target nodes with trait value T to only map to global nodes with same trait value T
    template<typename TraitType>
    void add_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits,
        const std::map<GlobalNode, TraitType>& global_traits,
        bool is_required = true);  // true for required, false for preferred

    // Explicit pair constraints (one-to-one)
    void add_required_constraint(TargetNode target_node, GlobalNode global_node);
    void add_preferred_constraint(TargetNode target_node, GlobalNode global_node);

    // Validation and query
    bool validate() const;  // Returns false if any target has no valid mappings
    const std::set<GlobalNode>& get_valid_mappings(TargetNode target) const;
    const std::set<GlobalNode>& get_preferred_mappings(TargetNode target) const;
    bool is_valid_mapping(TargetNode target, GlobalNode global) const;

    // Accessors for solver
    const std::map<TargetNode, std::set<GlobalNode>>& get_valid_mappings() const;
    const std::map<TargetNode, std::set<GlobalNode>>& get_preferred_mappings() const;

private:
    // Internal representation: intersection of all constraints
    std::map<TargetNode, std::set<GlobalNode>> valid_mappings_;      // Required constraints
    std::map<TargetNode, std::set<GlobalNode>> preferred_mappings_;  // Preferred constraints

    void rebuild_valid_mappings();
    void rebuild_preferred_mappings();
};
```

**Implementation:**

1. **Internal Representation**:
   - `valid_mappings_`: `std::map<TargetNode, std::set<GlobalNode>>` - Stores intersection of all **required** constraints
   - `preferred_mappings_`: `std::map<TargetNode, std::set<GlobalNode>>` - Stores intersection of all **preferred** constraints
   - Both use the same structure: for each target node, the set of global nodes it can/should map to

2. **Constructor**: Converts sets of pairs into mapping format:
   - Required pairs: `(target, global)` → `valid_mappings_[target].insert(global)`
   - Preferred pairs: `(target, global)` → `preferred_mappings_[target].insert(global)`

3. **Trait Constraints**: When `add_trait_constraint()` is called:
   - For each target node, find its trait value
   - Find all global nodes with the same trait value
   - Intersect with `valid_mappings_[target]` (if required) or `preferred_mappings_[target]` (if preferred)

4. **Required Constraints**: When `add_required_constraint()` is called:
   - Intersect `valid_mappings_[target]` with `{global}` (restrict to single node)
   - Intersects with existing constraints - if incompatible, `validate()` will return false

5. **Preferred Constraints**: When `add_preferred_constraint()` is called:
   - Add `global` to `preferred_mappings_[target]` set
   - Intersects with existing preferred constraints
   - Doesn't restrict `valid_mappings_` - solver can still choose other nodes if needed

6. **Validation**: `validate()` checks if any target node has an empty `valid_mappings_` set, indicating unsatisfiable required constraints

**Key Points:**
- Unified representation: both required and preferred constraints use the same `std::map<TargetNode, std::set<GlobalNode>>` structure
- Trait constraints and explicit pairs both modify the same internal mappings
- Intersection semantics: all required constraints must be satisfied simultaneously
- Preferred mappings guide solver but don't restrict valid mappings
- Validation: check `validate()` before solving to detect unsatisfiable constraints early

### Constraint Usage Examples

#### From Sets

```cpp
std::set<std::pair<FabricNodeId, AsicID>> required_set = {
    {FabricNodeId(MeshId{0}, 0), AsicID{5}}
};
std::set<std::pair<FabricNodeId, AsicID>> preferred_set = {
    {FabricNodeId(MeshId{0}, 1), AsicID{6}}
};
MappingConstraints<FabricNodeId, AsicID> constraints(required_set, preferred_set);
// Internally stored as:
// valid_mappings_[FabricNodeId(MeshId{0}, 0)] = {AsicID{5}}
// preferred_mappings_[FabricNodeId(MeshId{0}, 1)] = {AsicID{6}}
```

#### Trait Constraints

```cpp
// Add required trait constraint: nodes on same host rank can only map to ASICs on same host rank
MappingConstraints<FabricNodeId, AsicID> constraints;
constraints.add_trait_constraint(node_to_host_rank, asic_to_host_rank, true);  // required

// Add another required trait constraint: nodes on same rack can only map to ASICs on same rack
constraints.add_trait_constraint(node_to_rack, asic_to_rack, true);  // required
// Intersection: target can only map to global nodes with BOTH same host rank AND same rack

// Add explicit required constraint (pinning)
constraints.add_required_constraint(target_node_0, asic_5);
// Intersection: target_node_0 can ONLY map to asic_5 (if asic_5 satisfies trait constraints)

// Add preferred trait constraint (optional guidance)
constraints.add_trait_constraint(node_to_rack, asic_to_rack, false);  // preferred

// Add explicit preferred constraint
constraints.add_preferred_constraint(target_node_1, asic_6);

// Validate before solving
if (!constraints.validate()) {
    // Error: some target node has no valid mappings
    return;
}
```

## Core Component: TopologySolver

Stateless function that performs constraint satisfaction search to find a valid mapping from target graph to global graph.

```cpp
template<typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints);
```

**Key Points:**
- Stateless function - pure mapping from inputs to result
- `target_graph`: Subgraph pattern to find (smaller graph)
- `global_graph`: Host graph that contains the target (larger graph)
- Returns `MappingResult` with success status and bidirectional mappings
- Enforces required constraints first, then optimizes for preferred constraints

## MappingResult

```cpp
template<typename TargetNode, typename GlobalNode>
struct MappingResult {
    bool success = false;
    std::string error_message;

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
        std::chrono::milliseconds elapsed_time;
    } stats;
};
```

```
