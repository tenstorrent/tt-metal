# Topology Solver Implementation and Testing Plan

## Overview

This document outlines the complete implementation and testing plan for the topology solver, including namespace reorganization, module implementation order, and comprehensive testing strategy.

## Namespace Organization

### Public API (`tt::tt_fabric`)
**Location**: `topology_solver.hpp` and `topology_solver.tpp`

These are the only types/functions exposed to users:

```cpp
namespace tt::tt_fabric {

// Public graph representation
template <typename NodeId>
class AdjacencyGraph { ... };

// Public constraint system
template <typename TargetNode, typename GlobalNode>
class MappingConstraints { ... };

// Public result type
template <typename TargetNode, typename GlobalNode>
struct MappingResult { ... };

// Public solver function
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    ...,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED);

// Connection validation mode enum
enum class ConnectionValidationMode {
    STRICT,      ///< Strict mode: require exact channel counts, fail if not met
    RELAXED,     ///< Relaxed mode: prefer correct channel counts, but allow mismatches with warnings
    NONE         ///< No validation: only check edge existence, ignore channel counts
};

// Public helper functions
std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(...);
std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_map_physical(...);

} // namespace tt::tt_fabric
```

### Internal Implementation (`tt::tt_fabric::detail`)
**Location**: `tt_metal/fabric/topology_solver_internal.hpp` and `tt_metal/fabric/topology_solver.cpp`

All implementation details are hidden in the `detail` namespace and located in the implementation directory (not part of public API):

```cpp
namespace tt::tt_fabric::detail {

// Graph preprocessing
template <typename TargetNode, typename GlobalNode>
struct GraphIndexData { ... };

// Constraint processing
template <typename TargetNode, typename GlobalNode>
struct ConstraintIndexData { ... };

// Node selection heuristics
template <typename TargetNode, typename GlobalNode>
struct NodeSelector { ... };

// Candidate generation
template <typename TargetNode, typename GlobalNode>
struct CandidateGenerator { ... };

// Consistency checking
template <typename TargetNode, typename GlobalNode>
struct ConsistencyChecker { ... };

// Path graph detection and fast path
template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector { ... };

// DFS search engine
template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine { ... };

// Mapping validation
template <typename TargetNode, typename GlobalNode>
struct MappingValidator { ... };

} // namespace tt::tt_fabric::detail
```

## File Structure

```
tt_metal/api/tt-metalium/experimental/fabric/
├── topology_solver.hpp              # Public API only (tt::tt_fabric)
└── topology_solver.tpp              # Public template implementations (includes internal.hpp)

tt_metal/fabric/
├── topology_solver_internal.hpp     # Internal types (tt::tt_fabric::detail) - NOT part of public API
└── topology_solver.cpp              # Implementation file (build_adjacency_map_* + internal implementations)
```

**Note**: `topology_solver_internal.hpp` is in the implementation directory (`tt_metal/fabric/`) rather than the public API directory, so it's not part of the external interface.

## Implementation Guidelines

### Connection Validation Modes

The solver supports three modes for connection count validation:

1. **STRICT Mode**: Requires that physical edges have at least as many channels as logical edges require. Fails if any edge doesn't meet the requirement.

2. **RELAXED Mode** (Default): Prefers mappings with correct channel counts, but allows mappings with insufficient channels. Adds warnings to `MappingResult::warnings` for any mismatches.

3. **NONE Mode**: Only validates that edges exist, completely ignores channel counts.

**Relaxed Mode Behavior**:
- During candidate generation and consistency checking, prefer candidates with sufficient channel counts
- If no perfect match exists, allow candidates with insufficient channels
- After finding a mapping, validate all edges and add warnings for any channel count mismatches
- Warnings format: `"Relaxed mode: logical edge from node {} to {} requires {} channels, but physical edge from {} to {} only has {} channels"`

**Implementation Notes**:
- Use `ConnectionValidationMode` enum parameter in `solve_topology_mapping()`
- Pass mode through to `ConsistencyChecker`, `MappingValidator`, etc.
- In relaxed mode, candidate ordering should prefer sufficient channel counts
- Collect warnings during validation phase, not during search

### Logging Utilities

**Important**: For all error messages and informational messages, use the same logging utilities as `topology_mapper_utils.cpp`:

```cpp
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

// For informational messages (progress, status updates)
log_info(tt::LogFabric, "Message: {}", arg);

// For debug messages (detailed debugging information)
log_debug(tt::LogFabric, "Debug: {}", arg);

// For error messages (use fmt::format for complex formatting)
log_error(tt::LogFabric, "Error: {}", fmt::format("Details: {}", details));
```

**Usage Examples**:
- Progress logging during DFS search
- Fast path success/failure messages
- Constraint validation messages
- Error messages in `MappingResult::error_message` (use `fmt::format`)

**Reference**: See `tt_metal/fabric/topology_mapper_utils.cpp` for examples of logging patterns used in the existing topology mapper.

## Implementation Phases

### Phase 1: Foundation - Data Structures (Week 1)

#### 1.1 Create Internal Header File
**File**: `tt_metal/fabric/topology_solver_internal.hpp`

**Tasks**:
- [ ] Create file in `tt_metal/fabric/` directory (not public API directory)
- [ ] Create namespace `tt::tt_fabric::detail`
- [ ] Forward declare all internal types
- [ ] Include necessary headers (`<tt-logger/tt-logger.hpp>`, `<fmt/format.h>`)
- [ ] Document namespace purpose

**Dependencies**: None
**Testing**: Compile-time check only

**Note**: This file is in the implementation directory, not part of the public API.

---

#### 1.2 Implement GraphIndexData
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct GraphIndexData {
    // Node vectors
    std::vector<TargetNode> target_nodes;
    std::vector<GlobalNode> global_nodes;

    // Index mappings
    std::map<TargetNode, size_t> target_to_idx;
    std::map<GlobalNode, size_t> global_to_idx;

    // Adjacency index vectors (deduplicated, sorted)
    std::vector<std::vector<size_t>> target_adj_idx;
    std::vector<std::vector<size_t>> global_adj_idx;

    // Connection count maps (for strict mode / multi-edge support)
    std::vector<std::map<size_t, size_t>> target_conn_count;
    std::vector<std::map<size_t, size_t>> global_conn_count;

    // Degree vectors
    std::vector<size_t> target_deg;
    std::vector<size_t> global_deg;

    size_t n_target;
    size_t n_global;
};

// Factory function
template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode> build_graph_index_data(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph);

} // namespace detail
```

**Functions to Implement**:
- `build_graph_index_data()` - Main factory function
- `build_adjacency_indices()` - Build deduplicated adjacency lists
- `build_connection_counts()` - Build multi-edge connection counts
- `compute_degrees()` - Compute node degrees

**Tasks**:
- [ ] Declare GraphIndexData struct
- [ ] Implement build_graph_index_data()
- [ ] Implement helper functions
- [ ] Add unit tests

**Dependencies**: None
**Testing**: Unit tests with simple graphs

---

#### 1.3 Implement ConstraintIndexData
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct ConstraintIndexData {
    // Restricted mappings: target_idx -> vector of valid global_indices
    std::vector<std::vector<size_t>> restricted_global_indices;

    // Preferred mappings: target_idx -> vector of preferred global_indices
    std::vector<std::vector<size_t>> preferred_global_indices;

    // Helper: check if mapping is valid
    bool is_valid_mapping(size_t target_idx, size_t global_idx) const;

    // Helper: get candidates for target node (returns all if no restrictions)
    const std::vector<size_t>& get_candidates(size_t target_idx, size_t n_global) const;
};

// Factory function
template <typename TargetNode, typename GlobalNode>
ConstraintIndexData<TargetNode, GlobalNode> build_constraint_index_data(
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data);

} // namespace detail
```

**Functions to Implement**:
- `build_constraint_index_data()` - Main factory function
- `is_valid_mapping()` - Check if mapping satisfies constraints
- `get_candidates()` - Get valid candidates for target node

**Tasks**:
- [ ] Declare ConstraintIndexData struct
- [ ] Implement build_constraint_index_data()
- [ ] Implement helper methods
- [ ] Add unit tests

**Dependencies**: GraphIndexData
**Testing**: Unit tests with various constraint configurations

---

### Phase 2: Heuristics & Pruning (Week 1-2)

#### 2.1 Implement NodeSelector
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct NodeSelector {
    // Select next unassigned target node using MRV heuristic
    static size_t select_next_target(
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used);

    // Count candidates for a target node
    static size_t count_candidates(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used);
};

} // namespace detail
```

**Heuristic Logic**:
1. Prefer nodes with most mapped neighbors
2. Among those, prefer nodes with fewest candidates
3. Among those, prefer nodes with lowest degree

**Tasks**:
- [ ] Declare NodeSelector struct
- [ ] Implement select_next_target()
- [ ] Implement count_candidates()
- [ ] Add unit tests

**Dependencies**: GraphIndexData, ConstraintIndexData
**Testing**: Unit tests with various graph structures

---

#### 2.2 Implement CandidateGenerator
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct CandidateGenerator {
    // Generate candidates for a target node
    static std::vector<size_t> generate_candidates(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used);

    // Order candidates by degree gap (prefer tighter fits)
    static void order_candidates(
        std::vector<size_t>& candidates,
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data);
};

} // namespace detail
```

**Filtering Criteria**:
1. Not already used
2. Degree sufficient: `global_deg[j] >= target_deg[i]`
3. Constraint satisfaction: `is_valid_mapping(i, j)`
4. Local consistency: mapped neighbors are connected

**Tasks**:
- [ ] Declare CandidateGenerator struct
- [ ] Implement generate_candidates()
- [ ] Implement order_candidates()
- [ ] Add unit tests

**Dependencies**: GraphIndexData, ConstraintIndexData
**Testing**: Unit tests with various constraint and graph configurations

---

#### 2.3 Implement ConsistencyChecker
**File**: `topology_solver_internal.hpp` (declaration), `topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct ConsistencyChecker {
    // Check local consistency (mapped neighbors must be connected)
    static bool check_local_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

    // Forward checking: ensure future neighbors have viable candidates
    static bool check_forward_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

    // Check reachability (for path graphs)
    static size_t count_reachable_unused(
        size_t start_global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const std::vector<bool>& used);
};

} // namespace detail
```

**Tasks**:
- [ ] Declare ConsistencyChecker struct
- [ ] Implement check_local_consistency() with ConnectionValidationMode parameter
- [ ] Implement check_forward_consistency() with ConnectionValidationMode parameter
- [ ] Implement count_reachable_unused()
- [ ] In relaxed mode, prefer candidates with sufficient channel counts but allow insufficient ones
- [ ] Add unit tests for all three validation modes

**Dependencies**: GraphIndexData, ConstraintIndexData
**Testing**: Unit tests with various consistency scenarios

**Relaxed Mode**: When `validation_mode == RELAXED`, prefer candidates with sufficient channel counts during consistency checking, but don't reject candidates with insufficient counts. This allows the solver to find a mapping even when perfect channel counts aren't available.

---

### Phase 3: Search Engine (Week 2)

#### 3.1 Implement PathGraphDetector
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector {
    // Detect if graph is a path (2 endpoints, all others degree <= 2)
    static bool is_path_graph(const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    // Build ordered path from endpoints
    static std::vector<size_t> build_path_order(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    // Path-extension DFS
    static bool try_path_mapping(
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        std::vector<int>& mapping,
        std::vector<bool>& used);
};

} // namespace detail
```

**Tasks**:
- [ ] Declare PathGraphDetector struct
- [ ] Implement is_path_graph()
- [ ] Implement build_path_order()
- [ ] Implement try_path_mapping()
- [ ] Add logging: `log_info(tt::LogFabric, ...)` on fast path success, `log_debug(tt::LogFabric, ...)` on failure
- [ ] Add unit tests

**Dependencies**: GraphIndexData, ConstraintIndexData, ConsistencyChecker
**Testing**: Unit tests with path graphs of various sizes

**Logging**: Use `log_info(tt::LogFabric, "Fast-path path-graph mapping succeeded...")` on success and `log_debug(tt::LogFabric, "Fast-path path-graph mapping failed...")` on failure, similar to `topology_mapper_utils.cpp` lines 403-405.

---

#### 3.2 Implement DFSSearchEngine
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine {
public:
    struct SearchState {
        std::vector<int> mapping;           // mapping[target_idx] = global_idx or -1
        std::vector<bool> used;             // used[global_idx] = true if assigned
        std::unordered_set<uint64_t> failed_states;  // Memoization cache
        size_t dfs_calls = 0;
        size_t backtrack_count = 0;
    };

    // Main search function
    bool search(
        size_t assigned_count,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        SearchState& state,
        ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

    // Note: Use log_info(tt::LogFabric, ...) for progress logging
    // Use log_debug(tt::LogFabric, ...) for detailed debugging

private:
    // Hash function for state memoization
    uint64_t hash_state(const std::vector<int>& mapping) const;

    // Recursive DFS helper
    bool dfs_recursive(
        size_t pos,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        SearchState& state,
        bool strict_mode);
};

} // namespace detail
```

**Tasks**:
- [ ] Declare DFSSearchEngine class
- [ ] Implement hash_state()
- [ ] Implement dfs_recursive()
- [ ] Implement search()
- [ ] Add progress logging using `log_info(tt::LogFabric, ...)` (similar to topology_mapper_utils.cpp)
- [ ] Add unit tests

**Dependencies**: All previous modules
**Testing**: Unit tests with small graphs, integration tests

**Logging**: Use `log_info(tt::LogFabric, ...)` for periodic progress updates (e.g., every 2^18 DFS calls), similar to the pattern in `topology_mapper_utils.cpp` lines 557-571.

---

### Phase 4: Validation & Integration (Week 2-3)

#### 4.1 Implement MappingValidator
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver.cpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct MappingValidator {
    // Validate complete mapping
    static bool validate_mapping(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED,
        std::vector<std::string>* warnings = nullptr);

    // Validate connection counts and collect warnings (for relaxed mode)
    static void validate_connection_counts(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        ConnectionValidationMode validation_mode,
        std::vector<std::string>* warnings);

    // Build result from mapping
    static MappingResult<TargetNode, GlobalNode> build_result(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const DFSSearchEngine<TargetNode, GlobalNode>::SearchState& state,
        const MappingConstraints<TargetNode, GlobalNode>& constraints);
};

} // namespace detail
```

**Tasks**:
- [ ] Declare MappingValidator struct
- [ ] Implement validate_mapping() with ConnectionValidationMode parameter
- [ ] Implement validate_connection_counts() to check channel counts and collect warnings
- [ ] Implement build_result() with warnings collection
- [ ] Use `fmt::format` for error messages in `MappingResult::error_message`
- [ ] Use `fmt::format` for warning messages in `MappingResult::warnings`
- [ ] In STRICT mode: fail if channel counts insufficient
- [ ] In RELAXED mode: collect warnings for insufficient channel counts but don't fail
- [ ] In NONE mode: skip channel count validation entirely
- [ ] Add unit tests for all three validation modes

**Dependencies**: GraphIndexData, DFSSearchEngine
**Testing**: Unit tests with various mapping scenarios

**Error Messages**: Use `fmt::format` for formatting error messages that go into `MappingResult::error_message`, similar to error message patterns in `topology_mapper_utils.cpp`.

**Warning Messages**: In RELAXED mode, collect warnings for channel count mismatches:
```cpp
warnings->push_back(fmt::format(
    "Relaxed mode: logical edge from node {} to {} requires {} channels, "
    "but physical edge from {} to {} only has {} channels",
    target_node, neighbor_node, required_count,
    global_node, neighbor_global, actual_count));
```

---

#### 4.2 Implement solve_topology_mapping()
**File**: `topology_solver.tpp`

**Structure**:
```cpp
namespace tt::tt_fabric {

template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode) {

    using namespace detail;

    // 1. Preprocessing
    auto graph_data = build_graph_index_data(target_graph, global_graph);
    auto constraint_data = build_constraint_index_data(constraints, graph_data);

    // 2. Fast path for path graphs
    if (PathGraphDetector::is_path_graph(graph_data)) {
        // Try fast path with validation_mode...
    }

    // 3. General DFS
    DFSSearchEngine<TargetNode, GlobalNode> engine;
    // ... run search with validation_mode ...

    // 4. Validation and result building
    // Pass validation_mode and collect warnings
    return MappingValidator::build_result(..., connection_validation_mode);
}

} // namespace tt::tt_fabric
```

**Tasks**:
- [ ] Implement solve_topology_mapping() with ConnectionValidationMode parameter
- [ ] Integrate all modules
- [ ] Handle preferred constraints (optimization)
- [ ] Collect statistics
- [ ] Add error handling with `fmt::format` for error messages
- [ ] Add warning collection for relaxed mode channel count mismatches
- [ ] Pass validation_mode through to all consistency checks and validation functions
- [ ] Add initial logging: `log_info(tt::LogFabric, ...)` for mapping start (similar to topology_mapper_utils.cpp line 164-171)
- [ ] Log warnings when relaxed mode finds mismatches: `log_info(tt::LogFabric, "Relaxed mode: {} warnings collected", warnings.size())`

**Dependencies**: All modules
**Testing**: Integration tests, end-to-end tests

**Logging**:
- Use `log_info(tt::LogFabric, ...)` for initial mapping start message with graph statistics
- Use `fmt::format` for all error messages in `MappingResult::error_message`
- Use `fmt::format` for all warning messages in `MappingResult::warnings`
- Log summary of warnings when relaxed mode is used
- Follow logging patterns from `topology_mapper_utils.cpp`

**Relaxed Mode Implementation**:
- During candidate generation: prefer candidates with sufficient channel counts, but allow insufficient ones
- During consistency checking: prefer sufficient channel counts, but don't reject insufficient ones
- After mapping found: validate all edges, collect warnings for mismatches, but don't fail
- Warnings are added to `result.warnings` vector for user inspection

---

### Phase 5: Testing & Optimization (Week 3-4)

#### 5.1 Unit Tests
**File**: `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp`

**Test Categories**:

1. **GraphIndexData Tests**
   - [ ] Empty graphs
   - [ ] Single node graphs
   - [ ] Path graphs
   - [ ] Complete graphs
   - [ ] Disconnected graphs
   - [ ] Multi-edge graphs

2. **ConstraintIndexData Tests**
   - [ ] Empty constraints
   - [ ] Required constraints only
   - [ ] Preferred constraints only
   - [ ] Mixed constraints
   - [ ] Over-constrained scenarios
   - [ ] Trait constraints

3. **NodeSelector Tests**
   - [ ] MRV heuristic correctness
   - [ ] Various graph structures
   - [ ] Constraint interactions

4. **CandidateGenerator Tests**
   - [ ] Filtering correctness
   - [ ] Ordering correctness
   - [ ] Constraint interactions

5. **ConsistencyChecker Tests**
   - [ ] Local consistency
   - [ ] Forward consistency
   - [ ] Reachability counting
   - [ ] Strict mode

6. **PathGraphDetector Tests**
   - [ ] Path detection
   - [ ] Path mapping
   - [ ] Non-path graphs

7. **DFSSearchEngine Tests**
   - [ ] Simple mappings
   - [ ] Complex mappings
   - [ ] Memoization
   - [ ] Backtracking

8. **MappingValidator Tests**
   - [ ] Validation correctness
   - [ ] STRICT mode validation (fails on insufficient channels)
   - [ ] RELAXED mode validation (warnings on insufficient channels)
   - [ ] NONE mode validation (no channel count checks)
   - [ ] Warning message formatting
   - [ ] Result building with warnings

**Tasks**:
- [ ] Create test fixtures
- [ ] Implement unit tests for each module
- [ ] Run tests and fix bugs
- [ ] Achieve >90% code coverage

---

#### 5.2 Integration Tests
**File**: `tests/tt_metal/tt_fabric/fabric_router/test_topology_solver.cpp`

**Test Scenarios**:

1. **Basic Functionality**
   - [ ] Simple 2-node mapping
   - [ ] Simple 3-node path mapping
   - [ ] Simple 4-node cycle mapping

2. **Constraint Scenarios**
   - [ ] Required constraints
   - [ ] Preferred constraints
   - [ ] Trait constraints
   - [ ] Mixed constraints

3. **Edge Cases**
   - [ ] Empty graphs
   - [ ] Single node
   - [ ] Disconnected graphs
   - [ ] Over-constrained (no solution)
   - [ ] Under-constrained (many solutions)

4. **Connection Validation Mode Tests**
   - [ ] STRICT mode: fails when channel counts insufficient
   - [ ] RELAXED mode: succeeds with warnings when channel counts insufficient
   - [ ] RELAXED mode: prefers correct channel counts when available
   - [ ] RELAXED mode: warning messages are correctly formatted
   - [ ] NONE mode: ignores channel counts completely
   - [ ] Mode switching: same graph with different modes produces expected results

5. **Real-World Scenarios**
   - [ ] Compare with `map_mesh_to_physical` results
   - [ ] Use real mesh graphs
   - [ ] Use real physical topologies
   - [ ] Test relaxed mode with real topologies that have channel count mismatches

**Tasks**:
- [ ] Create integration test suite
- [ ] Compare with existing implementation
- [ ] Verify equivalence
- [ ] Performance benchmarks

---

#### 5.3 Performance Optimization
**Tasks**:
- [ ] Profile performance bottlenecks
- [ ] Optimize hot paths
- [ ] Memory usage optimization
- [ ] Cache optimization
- [ ] Parallel search (future work)

---

## Testing Strategy

### Unit Testing Approach

**For Each Module**:
1. **Test Fixtures**: Create reusable test fixtures
2. **Edge Cases**: Test boundary conditions
3. **Error Cases**: Test error handling
4. **Correctness**: Verify expected behavior
5. **Performance**: Benchmark critical paths

### Integration Testing Approach

1. **Equivalence Testing**: Compare with `map_mesh_to_physical`
2. **Regression Testing**: Use existing test cases
3. **Real-World Testing**: Use actual mesh graphs and topologies
4. **Performance Testing**: Benchmark against original

### Test Data

**Simple Graphs**:
- Empty graph
- Single node
- 2-node path
- 3-node path
- 4-node cycle
- Star graph
- Complete graph

**Complex Graphs**:
- Real mesh graphs from test files
- Real physical topologies from test files
- Generated graphs with various properties

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Create `tt_metal/fabric/topology_solver_internal.hpp`
- [ ] Implement `GraphIndexData`
- [ ] Implement `ConstraintIndexData`
- [ ] Unit tests for Phase 1

### Week 2: Heuristics & Search
- [ ] Implement `NodeSelector`
- [ ] Implement `CandidateGenerator`
- [ ] Implement `ConsistencyChecker`
- [ ] Implement `PathGraphDetector`
- [ ] Implement `DFSSearchEngine`
- [ ] Unit tests for Phase 2-3

### Week 3: Integration
- [ ] Implement `MappingValidator`
- [ ] Implement `solve_topology_mapping()`
- [ ] Integration tests
- [ ] Compare with existing implementation

### Week 4: Testing & Polish
- [ ] Complete unit test suite
- [ ] Complete integration tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] Code review

---

## Success Criteria

### Functionality
- ✅ Solves same problems as `map_mesh_to_physical`
- ✅ Handles all constraint types
- ✅ Fast path works for path graphs
- ✅ STRICT mode validation works (fails on insufficient channels)
- ✅ RELAXED mode works (prefers correct channels, allows mismatches with warnings)
- ✅ NONE mode works (ignores channel counts)
- ✅ Warning messages are correctly formatted and collected
- ✅ Preferred constraints optimization works

### Code Quality
- ✅ Modular, testable design
- ✅ Clear separation of concerns
- ✅ Well-documented
- ✅ Follows existing code style
- ✅ No namespace pollution (only public API in `tt::tt_fabric`)

### Performance
- ✅ Comparable or better than original
- ✅ Fast path significantly faster for paths
- ✅ Memory usage acceptable

### Testing
- ✅ >90% code coverage
- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Performance benchmarks meet targets

---

## Risk Mitigation

### Risk 1: Algorithm Complexity
**Mitigation**: Start with simple cases, incrementally add complexity

### Risk 2: Performance Regression
**Mitigation**: Benchmark early and often, optimize hot paths

### Risk 3: Integration Issues
**Mitigation**: Test integration points early, compare with existing implementation

### Risk 4: Namespace Conflicts
**Mitigation**: Use `detail` namespace for all internal code, only expose public API

---

## Notes

- All internal implementation goes in `tt::tt_fabric::detail` namespace
- Only public API exposed in `tt::tt_fabric` namespace
- Template implementations in `.tpp` file
- Non-template implementations in `.cpp` file
- Comprehensive testing at each phase
- Incremental implementation with testing
