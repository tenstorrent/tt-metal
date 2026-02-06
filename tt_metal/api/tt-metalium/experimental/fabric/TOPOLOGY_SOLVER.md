# Topology Solver Documentation

## Overview

The topology solver implements a **constraint satisfaction problem (CSP)** solver for **graph isomorphism/subgraph matching**. It finds a mapping from a target graph (smaller) to a global graph (larger) while satisfying constraints.

**Key Features**:
- Generic template-based design (works with any node types)
- Stateless API (thread-safe, reentrant)
- Three connection validation modes (STRICT, RELAXED, NONE)
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

**Key Methods**:
- **`add_required_trait_constraint`**: Restricts target nodes with a specific trait value to map ONLY to global nodes with the same trait value.
- **`add_preferred_trait_constraint`**: Prioritizes mapping target nodes to global nodes with the same trait value.
- **`add_required_constraint`**: Pins a specific target node to a specific global node.
- **`add_preferred_constraint`**: Suggests a mapping for a specific target node.
- **`get_valid_mappings`**: Query allowed global nodes for a target node.

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

**Parameters**:
- **`target_graph`**: The subgraph pattern to find (logical mesh).
- **`global_graph`**: The larger host graph to search in (physical machine/cluster).
- **`constraints`**: The set of required and preferred constraints.
- **`connection_validation_mode`**: (Optional) Validation strictness for channel counts.

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
    build_adjacency_map_physical(cluster_type, psd, asic_id_to_mesh_rank);
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

### Key Implementation Details

All internal components reside in `tt::tt_fabric::detail` namespace.

#### 1. GraphIndexData

Converts the generic `AdjacencyGraph` (node-ID based) into an efficient index-based representation (0 to N-1 integers) for O(1) lookups during the search.

- **Index Generation**:
  1. **Node Lists**: All unique node IDs from the input graph are collected into a vector (`target_nodes` and `global_nodes`).
  2. **Mapping**: A hash map (`target_to_idx` / `global_to_idx`) is built mapping each Node ID to its position (index) in the vector (0 to N-1).
  3. **Adjacency Lists**: The original adjacency map is converted to a vector of vectors (`target_adj_idx`). For each node `i`, `target_adj_idx[i]` contains the *indices* of its neighbors. These neighbor lists are **sorted** to allow for fast `std::binary_search` lookups.

- **`GraphIndexData(target_graph, global_graph)`**: Constructor. Maps node IDs to indices, builds deduplicated adjacency lists, and pre-calculates connection counts (multi-edges).
- **`target_to_idx` / `global_to_idx`**: Maps to convert public Node IDs to internal `size_t` indices.
- **`target_conn_count`**: Stores required channel counts for each edge (for STRICT validation).

#### 2. ConstraintIndexData

Converts high-level `MappingConstraints` into index-based lookups.

- **Index Generation**:
  - Uses the `target_to_idx` and `global_to_idx` maps from `GraphIndexData`.
  - Iterates through the high-level `MappingConstraints` (which use Node IDs).
  - For each constraint, converts the Target Node ID to `target_idx` and the Global Node ID to `global_idx`.
  - Stores the results in `restricted_global_indices` (vector of vectors). `restricted_global_indices[target_idx]` holds the sorted list of allowed `global_idx` values.

- **`ConstraintIndexData(constraints, graph_data)`**: Constructor. Intersects all required and preferred constraints to build per-node allowed/preferred lists.
- **`is_valid_mapping(target_idx, global_idx)`**: Fast check if a specific mapping is allowed by required constraints.
- **`get_candidates(target_idx)`**: Returns the specific list of global node indices allowed for a target node (if restricted).

#### 3. SearchHeuristic

Implements the intelligence of the solver: which node to map next (Variable Ordering) and which global node to try (Value Ordering).

- **`select_and_generate_candidates(...)`**: The main entry point. Selects the "most constrained" target node and returns it along with a sorted list of global candidates.
- **`check_hard_constraints(...)`**: Filters candidates based on:
  - Required constraints (pinnings).
  - Graph isomorphism (edges to already-mapped neighbors must exist).
  - Degree constraints (global degree >= target degree).
  - Channel capacity (in STRICT mode).
- **`compute_node_cost(...)`**: Scores unmapped target nodes. Lower cost = picked earlier.
  - **Logic**: Prioritizes nodes with fewer valid candidates (Minimum Remaining Values - MRV) and more mapped neighbors.
- **`compute_candidate_cost(...)`**: Scores valid global candidates. Lower cost = tried earlier.
  - **Logic**: Prioritizes preferred nodes, better channel count matches (in RELAXED mode), and tighter degree fits.

### Difference Between Node Cost and Candidate Cost

The solver uses two distinct cost calculations for two different decisions in the search process:

1.  **Node Cost (Variable Ordering)**: Decides **"Which target node should I map next?"**
    *   **Goal**: Pick the "hardest" or "most constrained" node to map (Fail-Fast).
    *   **Heuristic**: Minimum Remaining Values (MRV).
    *   **Key Input**: Number of valid global candidates available (`candidate_count`).
    *   **Formula**: `(candidate_count * HARD_WEIGHT) ...`
    *   **Result**: The solver picks the unmapped target node with the *lowest* cost (fewest options).

2.  **Candidate Cost (Value Ordering)**: Decides **"Which global node should I try assigning to this target node first?"**
    *   **Goal**: Pick the "best fit" or "most likely to succeed" global node.
    *   **Heuristic**: Least Constraining Value / Preferred Value.
    *   **Key Input**: Soft constraints like preferences and channel count matches.
    *   **Formula**: `-is_preferred * SOFT_WEIGHT ...`
    *   **Result**: The solver sorts candidates by cost (lowest first) and tries them in that order.

- **Cost System Details**:

- **Hard Constraints vs Hard Weight**:
  - **Hard Constraints** are boolean checks that *filter out* invalid global nodes entirely. They define `candidate_count`.
  - **`HARD_WEIGHT`** is a multiplier for `candidate_count`. It ensures the solver prioritizes the **Minimum Remaining Values (MRV)** heuristic above all else.
  - **Logic**: A node with fewer valid candidates (lower `candidate_count`) yields a lower cost, making it the highest priority to solve next. The large weight ensures that a node with 2 options is *always* picked before a node with 3 options, regardless of soft constraints.
  - **Candidate Count vs Mapped Neighbors**:
    - **Candidate Count (`candidate_count`)**: How many global nodes can *potentially* host this target node right now. Fewer is better (more critical).
    - **Mapped Neighbors (`mapped_neighbors`)**: How many of this target node's neighbors have *already* been assigned. More is better (higher constraint density).
    - **Relation**: A target node with many mapped neighbors usually has a low candidate count because those neighbors restrict the valid options (must connect to them). Both metrics guide the solver to the most "difficult" parts of the graph first.
  - **Why MRV? (Fail-Fast Principle)**:
    - **Scenario A (Without MRV)**: You map a node with 100 options first. You pick option #1. Later, you try to map a "bottleneck" node that had only 1 valid option, but it conflicts with your first choice. You must backtrack and try option #2 for the first node. You might repeat this 100 times before finding a combo that works for the bottleneck.
    - **Scenario B (With MRV)**: You map the "bottleneck" node (1 option) *first*. It succeeds or fails immediately. If it succeeds, the node with 100 options is then filtered to only those compatible with the bottleneck. This drastically reduces the search tree size by pushing the branching factor (multiplication) to the bottom of the tree rather than the top.
- **Node Cost Formula**: `(candidate_count * HARD_WEIGHT) - (preferred_count * SOFT_WEIGHT) - (mapped_neighbors * RUNTIME_WEIGHT)`
- **Candidate Cost Formula**: `-is_preferred * SOFT_WEIGHT - channel_match_score + degree_gap * RUNTIME_WEIGHT`
  - **`channel_match_score` Logic (RELAXED mode)**:
    - **Sufficient Channels (>= required)**: Grants a **bonus** (reduces cost). Exact match gets max bonus (`SOFT_WEIGHT`). Excess channels reduce the bonus slightly (preferring tighter fits).
    - **Insufficient Channels (< required)**: Applies a **penalty** (increases cost).
    - **Result**: The solver strongly prefers candidates with *enough* connections over those without. Among those with enough, it prefers exact matches to save larger pipes for other needs.
- **Weights**: `HARD_WEIGHT` (1M) >> `SOFT_WEIGHT` (1K) >> `RUNTIME_WEIGHT` (1).

#### 4. ConsistencyChecker

Validates partial mappings to prune the search tree early.

- **`check_local_consistency(target, global, ...)`**:
  - **Goal**: Ensure the current assignment `target -> global` is compatible with *past* decisions (already mapped neighbors).
  - **Algorithm**:
    1. Iterate over all neighbors of `target`.
    2. If a neighbor `N_T` is already mapped to `N_G`:
       - Check if an edge exists between `global` and `N_G` in the global graph.
       - If `STRICT` mode: Check if the edge capacity between `global` and `N_G` is >= required capacity.
    3. If any check fails, return `false`.

- **`check_forward_consistency(target, global, ...)`**:
  - **Goal**: Ensure the current assignment doesn't break *future* decisions (unmapped neighbors).
  - **Algorithm**:
    1. Iterate over all *unmapped* neighbors of `target`.
    2. For each unmapped neighbor `N_T`:
       - Scan all *unused* neighbors of `global` (`Candidate_G`).
       - Check if `Candidate_G` is a valid match for `N_T` (satisfies hard constraints and local consistency).
       - If **zero** valid candidates are found for `N_T`, then `target -> global` is a dead end. Return `false` immediately.
  - **Impact**: Detects failures one step earlier, pruning large branches of the search tree.

#### 5. DFSSearchEngine

The core backtracking search algorithm.

- **`search(...)`**: Initializes the search. Performs pre-assignment of pinned nodes (optimization) and starts the recursive process.
- **`dfs_recursive(...)`**: The recursive step.
  1. Checks memoization cache (failed states).
  2. Calls `SearchHeuristic` to pick the next node and candidates.
  3. Iterates candidates, running `ConsistencyChecker`.
  4. Recurses.
  5. Backtracks if needed.
- **`hash_state(...)`**: Computes FNV-1a hash of the current partial mapping for memoization.

**State Management**:

The solver maintains a comprehensive state object (`SearchState`) that evolves during the recursion.

```cpp
struct SearchState {
    std::vector<int> mapping;                    // Current partial mapping
    std::vector<bool> used;                      // Usage tracking O(1)
    std::unordered_set<uint64_t> failed_states;  // Memoization Cache (FNV-1a hashes)
    size_t dfs_calls = 0;
    size_t backtrack_count = 0;
    std::string error_message;
};
```

**Search Process Visualization**:

```mermaid
flowchart TD
    Start([Start Search]) --> Init[Initialize SearchState]
    Init --> PreAssign[Apply Required Constraints<br/>(Pinnings)]
    PreAssign --> CheckCount{Enough Global Nodes?}
    CheckCount -- No --> Fail([Return Failure])
    CheckCount -- Yes --> DFS[Start DFS Recursive]

    subgraph DFS Logic
        DFS --> MemoCheck{In Failed States?}
        MemoCheck -- Yes --> ReturnFalse([Return False])
        MemoCheck -- No --> Select[Select Target Node<br/>(Heuristic: Most Constrained)]

        Select --> GenCand[Generate Candidates<br/>(Heuristic: Cost Ordered)]
        GenCand --> CandLoop{Next Candidate?}

        CandLoop -- No More Candidates --> MarkFail[Hash & Cache State<br/>(Memoization)]
        MarkFail --> ReturnFalse

        CandLoop -- Yes --> CheckLocal{Local Consistency?<br/>(Edges exist?)}
        CheckLocal -- No --> CandLoop

        CheckLocal -- Yes --> CheckFwd{Forward Consistency?<br/>(Future viable?)}
        CheckFwd -- No --> CandLoop

        CheckFwd -- Yes --> Apply[Apply Assignment<br/>mapping[t] = g<br/>used[g] = true]
        Apply --> Recurse[Recurse DFS]

        Recurse -- Success --> ReturnTrue([Return True])
        Recurse -- Failure --> Backtrack[Backtrack<br/>mapping[t] = -1<br/>used[g] = false]
        Backtrack --> CandLoop
    end
```

**Memoization Details**:
- **Hash Function (FNV-1a)**:
  - The hash function generates a unique 64-bit signature for the current partial mapping state.
  - **Input**: The `mapping` vector (array of size `N_target`).
  - **Logic**:
    1. Initialize `hash` with FNV offset basis.
    2. Iterate through `mapping`. If `mapping[i]` is assigned (not -1):
       - XOR `hash` with a combined value of `target_index` and `global_index`.
       - Multiply `hash` by FNV prime.
    3. Result is a signature that uniquely identifies *which* target nodes are mapped to *which* global nodes.
- **Cache**: `failed_states` (unordered set) stores hashes of states that have been fully explored and proven to have **no solution**.
- **Pruning**:
  - At the start of each `dfs_recursive` call, we compute the hash of the current state.
  - If the hash is found in `failed_states`, we know this specific partial assignment leads to a dead end (already explored via a different path).
  - We return `false` immediately, skipping redundant work.

#### 6. MappingValidator

Final verification layer.

- **`build_result(...)`**: Assembles the final `MappingResult`. Even if the search failed, it preserves the "best effort" partial mapping for debugging.
- **`validate_mapping(...)`**: Double-checks the final mapping for correctness (all edges exist).
- **`validate_connection_counts(...)`**: Verifies bandwidth requirements.
  - **STRICT**: Fails if physical channels < logical channels.
  - **RELAXED**: Emits warnings but allows the mapping.

### Logging

The solver uses `tt-logger` for all messages:

```cpp
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

log_info(tt::LogFabric, "Message: {}", arg);      // Informational
log_error(tt::LogFabric, "Error: {}", error_msg);  // Errors
```

**Logging Points**:
- Mapping start: Degree histograms for target and global graphs (e.g., `target_degree_histogram={2:4}, global_degree_histogram={2:4}`)
- Pre-assignment conflicts: Early detection of conflicting required constraints
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
3. **Memoization**: Failed states cached to avoid redundant work
4. **Statistics**: Use `result.stats` to understand solver performance

## Thread Safety

The public API (`solve_topology_mapping()`) is **stateless** - each call creates its own internal state:
- ✅ Thread-safe: Multiple threads can call `solve_topology_mapping()` concurrently
- ✅ Reentrant: Can be called recursively
- ✅ No global state: Each call creates a new `DFSSearchEngine` instance internally

## Template Requirements

Both `TargetNode` and `GlobalNode` must be:
- Comparable (for use in `std::map` and `std::set`)
- Copyable or movable
- Default constructible (if used in certain contexts)

Common types: `FabricNodeId`, `AsicID`, integers, strings, or custom types with comparison operators.
