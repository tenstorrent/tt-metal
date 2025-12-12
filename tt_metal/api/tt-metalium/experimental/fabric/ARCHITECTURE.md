# Topology Solver Architecture

## Overview

The topology solver implements a **constraint satisfaction problem (CSP)** solver for **graph isomorphism/subgraph matching**. It finds a mapping from a target graph (smaller) to a global graph (larger) while satisfying constraints.

## Namespace Organization

### Public API (`tt::tt_fabric`)
**Location**: `topology_solver.hpp` and `topology_solver.tpp`

These are the only types/functions exposed to users:

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

### Internal Implementation (`tt::tt_fabric::detail`)
**Location**: `tt_metal/fabric/topology_solver_internal.hpp` and `tt_metal/fabric/topology_solver_internal.tpp`

All implementation details are hidden in the `detail` namespace and located in the implementation directory (not part of public API):

```cpp
namespace tt::tt_fabric::detail {
    template <typename TargetNode, typename GlobalNode> struct GraphIndexData;
    template <typename TargetNode, typename GlobalNode> struct ConstraintIndexData;
    template <typename TargetNode, typename GlobalNode> class SearchHeuristic;
    template <typename TargetNode, typename GlobalNode> struct ConsistencyChecker;
    template <typename TargetNode, typename GlobalNode> struct PathGraphDetector;
    template <typename TargetNode, typename GlobalNode> class DFSSearchEngine;
    template <typename TargetNode, typename GlobalNode> struct MappingValidator;
}
```

## File Structure

```
tt_metal/api/tt-metalium/experimental/fabric/
├── topology_solver.hpp              # Public API only (tt::tt_fabric)
└── topology_solver.tpp              # Public template implementations (includes internal.hpp)

tt_metal/fabric/
├── topology_solver_internal.hpp     # Internal types (tt::tt_fabric::detail) - NOT part of public API
├── topology_solver_internal.tpp    # Internal template implementations
└── topology_solver.cpp              # Implementation file (build_adjacency_map_* + internal implementations)
```

**Note**: `topology_solver_internal.hpp` is in the implementation directory (`tt_metal/fabric/`) rather than the public API directory, so it's not part of the external interface.

## Architecture Decisions

### Unified SearchHeuristic Design

Instead of separate `NodeSelector` and `CandidateGenerator` classes, we use a single `SearchHeuristic` class that handles both operations with a coherent priority system.

**Rationale**:
1. **Coherent Heuristic**: Both operations use the same constraint information and priority system
2. **Smaller Search Tree**: Unified approach allows better coordination between selection and generation
3. **Simpler Interface**: Single method `select_and_generate_candidates()` instead of two separate calls
4. **Better Optimization**: Can optimize node selection based on candidate availability

### Integer Cost-Based Priority System

Simple integer cost scoring (lower cost = better, selected first).

**Cost Weights**:
```cpp
static constexpr int HARD_WEIGHT = 1000000;    // Hard constraints dominate
static constexpr int SOFT_WEIGHT = 1000;        // Soft constraints secondary  
static constexpr int RUNTIME_WEIGHT = 1;        // Runtime optimization minor
```

**Priority Hierarchy**:
1. **Hard Constraints** (Must Satisfy): Required constraints, graph isomorphism, degree requirements, channel counts (strict mode)
2. **Soft Constraints** (Optimize For): Preferred constraints, channel count matching (relaxed mode)
3. **Runtime Optimization** (Minimize Search Tree): MRV (fewest candidates), most mapped neighbors, degree gap

**Benefits**:
- Simple integer comparison (no complex structs)
- Clear priority hierarchy (weights ensure hard >> soft >> runtime)
- Easy to tune (adjust weights)
- Efficient (single integer comparison)

### ConstraintIndexData Status

`ConstraintIndexData` is implemented but may be removed. `SearchHeuristic` can query `MappingConstraints` directly using `GraphIndexData` node vectors.

**Trade-off**: Slightly slower lookups vs. simpler code
**Decision**: Keep for now, can remove later if performance is acceptable

## Module Breakdown

### Phase 1: Foundation (Data Structures)
1. **GraphIndexData**: Graph preprocessing - converts `AdjacencyGraph` to indexed representation
2. **ConstraintIndexData**: Constraint processing - converts `MappingConstraints` to index-based representation

### Phase 2: Heuristics & Pruning
3. **SearchHeuristic**: Unified node selection and candidate generation with integer cost-based priority
4. **ConsistencyChecker**: Local and forward consistency validation

### Phase 3: Search Engine
5. **PathGraphDetector**: Fast path optimization for path graphs (O(n) instead of exponential)
6. **DFSSearchEngine**: Core backtracking search with memoization

### Phase 4: Integration
7. **MappingValidator**: Final mapping validation and result building
8. **solve_topology_mapping()**: Main solver function integrating all modules

## Algorithm Flow

### High-Level Flow

```
solve_topology_mapping(target_graph, global_graph, constraints)
│
├─► Stage 1: Preprocessing
│   ├─ Build GraphIndexData
│   └─ Build ConstraintIndexData (optional)
│
├─► Stage 2: Fast Path Detection
│   ├─ Check if target is path graph
│   ├─ If yes: try path-extension DFS
│   └─ If succeeds: return result
│
├─► Stage 3: General DFS Search
│   ├─ Initialize search state
│   ├─ Handle pinned nodes
│   ├─ Try multiple starting points
│   └─ Run DFS with backtracking
│       ├─ SearchHeuristic::select_and_generate_candidates()
│       ├─ ConsistencyChecker validation
│       └─ Memoization
│
└─► Stage 4: Validation
    ├─ Validate mapping
    ├─ Connection validation mode checks
    ├─ Collect warnings (relaxed mode)
    └─ Build MappingResult
```

### DFS Search Flow

```
DFS(pos, mapping, used, ...)
│
├─► Base Case: pos == n_target
│   └─► return true (success)
│
├─► Check memoization cache
│   └─► if state already failed: return false
│
├─► SearchHeuristic::select_and_generate_candidates()
│   ├─ Compute node costs for all unassigned nodes
│   ├─ Select node with LOWEST cost (most constrained)
│   ├─ Generate candidates for selected node
│   └─ Order candidates by cost (LOWEST first)
│
├─► For each candidate (in cost order):
│   │
│   ├─► ConsistencyChecker::check_local_consistency()
│   │   ├─ Mapped neighbors must be connected
│   │   └─ Channel counts (strict mode)
│   │
│   ├─► ConsistencyChecker::check_forward_consistency()
│   │   ├─ Count unused neighbors
│   │   ├─ Verify future neighbors have candidates
│   │   └─ Check reachability (for path graphs)
│   │
│   ├─► Assign: mapping[target] = candidate, used[candidate] = true
│   │
│   ├─► Recursive call: DFS(pos + 1, ...)
│   │   └─► if succeeds: return true
│   │
│   └─► Backtrack: mapping[target] = -1, used[candidate] = false
│
└─► Mark state as failed in memoization cache
    └─► return false
```

## SearchHeuristic Cost Formulas

### Node Selection Cost

**Goal**: Select most constrained node first (fewest valid candidates).

```cpp
cost = (candidate_count * HARD_WEIGHT) 
     - (preferred_count * SOFT_WEIGHT)
     - (mapped_neighbors * RUNTIME_WEIGHT)
```

- **candidate_count**: Number of valid candidates (fewer = more constrained = lower cost)
- **preferred_count**: Number of preferred candidates (more = lower cost)
- **mapped_neighbors**: Number of already-mapped neighbors (more = lower cost)

**Select node with LOWEST cost** (most constrained first).

### Candidate Filtering and Ordering

**Goal**: Filter out invalid candidates, then order valid ones by preference.

**Step 1: Filter by Hard Constraints**
- Remove candidates with any hard constraint violations
- Hard constraints checked:
  - Required constraints (pinning, trait-based)
  - Graph isomorphism (edges must exist to mapped neighbors)
  - Degree sufficient (global_deg >= target_deg)
  - Channel counts sufficient (strict mode)

**Step 2: Order Valid Candidates by Cost**
```cpp
cost = -is_preferred * SOFT_WEIGHT
     - channel_match_score * (varies by closeness to required)
     + degree_gap * RUNTIME_WEIGHT
```

- **is_preferred**: 1 if preferred, 0 otherwise
- **channel_match_score** (relaxed mode): 
  - Exact match (actual == required): +SOFT_WEIGHT
  - Close above (actual > required): +SOFT_WEIGHT/(1+gap), decreases with gap
  - Below required (actual < required): -SOFT_WEIGHT/(10+gap), small penalty that decreases with gap
  - Prefers connections closer to required count
- **degree_gap**: Difference between global and target degree

**Result**: Only valid candidates remain, ordered by cost (LOWEST first = best first).

## ConsistencyChecker

The `ConsistencyChecker` validates partial mappings during DFS to prune invalid branches early. It ensures that assignments are consistent with already-assigned neighbors and that future neighbors will have viable options.

**Note**: `ConsistencyChecker` is a non-templated struct with templated methods. Template types are deduced from `GraphIndexData` arguments, so no explicit template parameters are needed when calling methods: `ConsistencyChecker::check_local_consistency(...)`.

### Purpose

**Difference from SearchHeuristic**:
- **SearchHeuristic**: Selects which target node to work on and generates/orders candidates
- **ConsistencyChecker**: Validates that a specific candidate assignment is valid before committing

**Analogy**:
- SearchHeuristic = "Which task should I do next, and in what order should I try approaches?"
- ConsistencyChecker = "Is this specific approach valid, and will it leave me with options later?"

### Functions

#### 1. `check_local_consistency()` - Local Consistency Check

**Purpose**: Verify the current assignment is consistent with already-assigned neighbors.

**What it checks**:
- If target node A is mapped to global node X
- And target node B (neighbor of A) is already mapped to global node Y
- Then: global nodes X and Y **must be connected** (edge must exist)
- In STRICT mode: X→Y must have **sufficient channel count**

**Example**:
```
Target graph: A -- B
Global graph:  X -- Y    Z

If we've mapped: A → X, B → Y
Local consistency: ✓ (X and Y are connected)

If we've mapped: A → X, B → Z  
Local consistency: ✗ (X and Z are NOT connected, but A and B are neighbors)
```

**Why it matters**: Prevents assignments that break graph isomorphism.

#### 2. `check_forward_consistency()` - Forward Checking

**Purpose**: Ensure the current assignment leaves viable options for future neighbors.

**What it checks**:
- Count unassigned neighbors of the current target node
- Count unused neighbors of the current global node
- Verify each unassigned neighbor has at least one viable candidate

**Example**:
```
Target graph: A -- B -- C
Global graph:  X -- Y

If we've mapped: A → X
Unassigned neighbors of A: [B]
Unused neighbors of X: [Y]

Forward check: Can B map to Y?
  - Check degree: ✓
  - Check constraints: ✓
  - Check local consistency: ✓
Result: ✓ (B has at least one viable candidate: Y)
```

**Why it matters**: Prunes branches that would leave future nodes with no options.

#### 3. `count_reachable_unused()` - Reachability Counting

**Purpose**: Count unused global nodes reachable from a starting point (used for path graphs).

**What it does**:
- Starting from a global node, count how many unused nodes are reachable
- Used to verify path graphs have enough unused nodes for remaining target nodes

**Why it matters**: Fast path optimization for linear chains.

### Usage in DFS

The ConsistencyChecker is used during DFS to validate candidates before committing:

```cpp
for (each candidate from SearchHeuristic) {
    // 1. Local consistency check
    if (!ConsistencyChecker::check_local_consistency(...)) {
        continue;  // Skip this candidate
    }
    
    // 2. Forward checking
    if (!ConsistencyChecker::check_forward_consistency(...)) {
        continue;  // Skip this candidate
    }
    
    // 3. Try assignment
    mapping[target] = candidate;
    if (dfs_recursive(...)) {
        return true;  // Success!
    }
    // Backtrack...
}
```

**Why both SearchHeuristic and ConsistencyChecker are needed**:
- SearchHeuristic guides the search (which node, which candidates)
- ConsistencyChecker prunes invalid branches (validates assignments)

## Connection Validation Modes

The solver supports three modes for connection count validation:

1. **STRICT Mode**: Requires that physical edges have at least as many channels as logical edges require. Fails if any edge doesn't meet the requirement.

2. **RELAXED Mode** (Default): Prefers mappings with correct channel counts, but allows mappings with insufficient channels. Adds warnings to `MappingResult::warnings` for any mismatches.

3. **NONE Mode**: Only validates that edges exist, completely ignores channel counts.

**Relaxed Mode Behavior**:
- During candidate generation and consistency checking, prefer candidates with sufficient channel counts
- If no perfect match exists, allow candidates with insufficient channels
- After finding a mapping, validate all edges and add warnings for any channel count mismatches
- Warnings format: `"Relaxed mode: logical edge from node {} to {} requires {} channels, but physical edge from {} to {} only has {} channels"`

## Optimization Techniques

### 1. Memoization
- Hash partial assignments to avoid revisiting failed states
- Uses FNV-1a hash function
- Stores hash of (target_idx, global_idx) pairs

### 2. Forward Checking
- Prune candidates that would leave future nodes without options
- Count unused neighbors vs. unassigned neighbors
- Verify each future neighbor has at least one viable candidate

### 3. Candidate Ordering
- Order by cost: prefer candidates with lower cost (preferred, matching channels, tighter degree fit)
- Reduces search space by trying promising candidates first

### 4. Fast Path Optimization
- Specialized O(n) algorithm for path graphs (linear chains)
- Detects path graphs: 2 endpoints (degree 1), all others degree ≤ 2
- Uses linear path-extension DFS instead of exponential search

### 5. Early Termination
- Fast path for path graphs: O(n) instead of exponential
- Memoization: avoid redundant work
- Forward checking: prune early

## Complexity Analysis

### Time Complexity
- **Preprocessing**: O(n + m + E_target + E_global)
  - n = target nodes, m = global nodes
  - E_target = target edges, E_global = global edges

- **Fast Path**: O(n * m) worst case, typically O(n)
  - Linear path extension with neighbor checks

- **General DFS**: O(m^n) worst case, but heavily pruned
  - Exponential in worst case
  - Typically much better due to pruning
  - Memoization reduces redundant work

### Space Complexity
- **GraphIndexData**: O(n + m + E_target + E_global)
- **ConstraintIndexData**: O(n * m) worst case
- **Search State**: O(n + m + failed_states)
- **Total**: O(n + m + E + failed_states)

## Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Genericity**: Works with any node types via templates
3. **Efficiency**: Indexed representations for O(1) lookups
4. **Flexibility**: Connection validation modes for different use cases
5. **Maintainability**: Clear namespace separation, well-documented

## Benefits of Architecture

1. **Clear Separation**: Public API vs. implementation details
2. **Namespace Safety**: Users can't accidentally use internal types
3. **Easier Maintenance**: Internal code can change without affecting users
4. **Better Documentation**: Only public API needs extensive docs
5. **Compile-Time Safety**: Compiler enforces namespace boundaries
6. **Smaller Search Tree**: Constraint-based pruning happens early
7. **Simple Heuristic**: Integer cost system is easy to understand and tune
