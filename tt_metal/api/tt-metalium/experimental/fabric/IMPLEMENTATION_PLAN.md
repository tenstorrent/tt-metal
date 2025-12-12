# Topology Solver Implementation Plan

## Overview

This document outlines the complete implementation and testing plan for the topology solver, including module implementation order, detailed code structure, and comprehensive testing strategy.

## Implementation Guidelines

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
- Pass mode through to `SearchHeuristic`, `ConsistencyChecker`, `MappingValidator`, etc.
- In relaxed mode, candidate ordering should prefer sufficient channel counts
- Collect warnings during validation phase, not during search

## Implementation Phases

### Phase 1: Foundation - Data Structures

#### 1.1 Create Internal Header File
**File**: `tt_metal/fabric/topology_solver_internal.hpp`

**Tasks**:
- [x] Create file in `tt_metal/fabric/` directory (not public API directory)
- [x] Create namespace `tt::tt_fabric::detail`
- [x] Forward declare all internal types
- [x] Include necessary headers (`<tt-logger/tt-logger.hpp>`, `<fmt/format.h>`)
- [x] Document namespace purpose

**Dependencies**: None
**Testing**: Compile-time check only

**Note**: This file is in the implementation directory, not part of the public API.

---

#### 1.2 Implement GraphIndexData
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

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

    // Degree vectors (number of unique neighbors, excluding self-connections)
    std::vector<size_t> target_deg;
    std::vector<size_t> global_deg;

    size_t n_target = 0;
    size_t n_global = 0;
};

template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode> build_graph_index_data(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph);

} // namespace detail
```

**Key Implementation Details**:
- Self-connections must be ignored during graph processing
- Degree calculation: count of unique neighbors (not counting multi-edges)
- Adjacency lists must be deduplicated and sorted for efficient binary search
- Connection counts stored as maps for multi-edge support

**Tasks**:
- [x] Declare GraphIndexData struct
- [x] Implement build_graph_index_data()
- [x] Add unit tests

**Dependencies**: None
**Testing**: Unit tests with simple graphs, empty graphs, self-connections

---

#### 1.3 Implement ConstraintIndexData
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

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

    // Helper: get candidates for target node
    const std::vector<size_t>& get_candidates(size_t target_idx) const;
};

template <typename TargetNode, typename GlobalNode>
ConstraintIndexData<TargetNode, GlobalNode> build_constraint_index_data(
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data);

} // namespace detail
```

**Tasks**:
- [x] Declare ConstraintIndexData struct
- [x] Implement build_constraint_index_data()
- [x] Implement is_valid_mapping() and get_candidates() helper methods
- [x] Add unit tests

**Dependencies**: GraphIndexData
**Testing**: Unit tests with various constraint configurations

**Note**: ConstraintIndexData may be removed - SearchHeuristic queries MappingConstraints directly using GraphIndexData node vectors.

---

### Phase 2: Heuristics & Pruning

#### 2.1 Implement SearchHeuristic (Unified)
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
class SearchHeuristic {
public:
    struct SelectionResult {
        size_t target_idx;                    // Selected target node index
        std::vector<size_t> candidates;       // Valid candidates (ordered by cost, lower = better)
    };
    
    static SelectionResult select_and_generate_candidates(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);
    
private:
    static int compute_node_cost(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);
    
    static bool check_hard_constraints(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);
    
    static int compute_candidate_cost(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);
    
    static std::vector<size_t> generate_ordered_candidates(...);
    
    static constexpr int HARD_WEIGHT = 1000000;
    static constexpr int SOFT_WEIGHT = 1000;
    static constexpr int RUNTIME_WEIGHT = 1;
};

} // namespace detail
```

**Cost Formulas**:

**Node Cost** (lower = more constrained = selected first):
```
cost = (candidate_count * HARD_WEIGHT) 
     - (preferred_count * SOFT_WEIGHT)
     - (mapped_neighbors * RUNTIME_WEIGHT)
```

**Candidate Filtering and Ordering**:
```
Step 1: Filter by hard constraints
  - Check hard constraints for each candidate
  - Remove candidates with any hard violations (don't include in list)
  
Step 2: Order valid candidates by cost
  cost = -is_preferred * SOFT_WEIGHT
       - channel_match_count * SOFT_WEIGHT
       + degree_gap * RUNTIME_WEIGHT
```

**Tasks**:
- [x] Declare SearchHeuristic class
- [x] Implement select_and_generate_candidates()
- [x] Implement compute_node_cost()
- [x] Implement check_hard_constraints() - returns true if candidate satisfies all hard constraints
- [x] Implement compute_candidate_cost() - only called for valid candidates
- [x] Implement generate_ordered_candidates() - filters invalid candidates first, then orders by cost
- [x] Add unit tests

**Dependencies**: GraphIndexData, MappingConstraints
**Testing**: Unit tests with various constraint configurations and graph structures

---

#### 2.2 Implement ConsistencyChecker
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

/**
 * Non-templated struct with templated methods - template types are deduced from GraphIndexData arguments.
 * Usage: ConsistencyChecker::check_local_consistency(...) (no template parameters needed)
 */
struct ConsistencyChecker {
    template <typename TargetNode, typename GlobalNode>
    static bool check_local_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    template <typename TargetNode, typename GlobalNode>
    static bool check_forward_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    template <typename TargetNode, typename GlobalNode>
    static size_t count_reachable_unused(
        size_t start_global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<bool>& used);
};

} // namespace detail
```

**Tasks**:
- [x] Declare ConsistencyChecker struct
- [x] Implement check_local_consistency() with ConnectionValidationMode parameter
- [x] Implement check_forward_consistency() with ConnectionValidationMode parameter
- [x] Implement count_reachable_unused()
- [x] In relaxed mode, prefer candidates with sufficient channel counts but allow insufficient ones
- [x] Add unit tests for all three validation modes

**Purpose**: The ConsistencyChecker validates partial mappings during DFS to prune invalid branches early. It ensures that assignments are consistent with already-assigned neighbors and that future neighbors will have viable options.

**Difference from SearchHeuristic**:
- **SearchHeuristic**: Selects which target node to work on and generates/orders candidates
- **ConsistencyChecker**: Validates that a specific candidate assignment is valid before committing

**Functions**:

1. **`check_local_consistency()`**: Verifies the current assignment is consistent with already-assigned neighbors
   - Checks that if target node A is mapped to global X, and target node B (neighbor of A) is mapped to global Y, then X and Y must be connected
   - In STRICT mode: also checks channel counts are sufficient
   - Prevents assignments that break graph isomorphism

2. **`check_forward_consistency()`**: Ensures the current assignment leaves viable options for future neighbors
   - Counts unassigned neighbors of current target node
   - Counts unused neighbors of current global node
   - Verifies each unassigned neighbor has at least one viable candidate
   - Prunes branches that would leave future nodes with no options

3. **`count_reachable_unused()`**: Counts unused global nodes reachable from a starting point
   - Used for path graph fast path optimization
   - Verifies path graphs have enough unused nodes for remaining target nodes

**Usage in DFS**:
```cpp
for (each candidate from SearchHeuristic) {
    if (!ConsistencyChecker::check_local_consistency(...)) {
        continue;  // Skip invalid candidate
    }
    if (!ConsistencyChecker::check_forward_consistency(...)) {
        continue;  // Skip candidate that leaves no options
    }
    // Try assignment...
}
```

**Dependencies**: GraphIndexData, MappingConstraints
**Testing**: Unit tests with various consistency scenarios

---

### Phase 3: Search Engine

#### 3.1 Implement PathGraphDetector
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector {
    static bool is_path_graph(const GraphIndexData<TargetNode, GlobalNode>& graph_data);
    
    static std::vector<size_t> build_path_order(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data);
    
    static bool try_path_mapping(
        const GraphIndexData<TargetNode, GlobalNode>& target_data,
        const GraphIndexData<TargetNode, GlobalNode>& global_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        std::vector<int>& mapping,
        std::vector<bool>& used,
        ConnectionValidationMode validation_mode);
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

**Dependencies**: GraphIndexData, SearchHeuristic, ConsistencyChecker
**Testing**: Unit tests with path graphs of various sizes

---

#### 3.2 Implement DFSSearchEngine
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine {
public:
    struct SearchState {
        std::vector<int> mapping;
        std::vector<bool> used;
        std::unordered_set<uint64_t> failed_states;
        size_t dfs_calls = 0;
        size_t backtrack_count = 0;
    };

    bool search(
        size_t assigned_count,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        SearchState& state,
        ConnectionValidationMode validation_mode);

private:
    uint64_t hash_state(const std::vector<int>& mapping) const;
    
    bool dfs_recursive(
        size_t pos,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        SearchState& state,
        ConnectionValidationMode validation_mode);
};

} // namespace detail
```

**Tasks**:
- [ ] Declare DFSSearchEngine class
- [ ] Implement hash_state()
- [ ] Implement dfs_recursive()
- [ ] Implement search()
- [ ] Use SearchHeuristic::select_and_generate_candidates() for node selection and candidate generation
- [ ] Add progress logging using `log_info(tt::LogFabric, ...)` (similar to topology_mapper_utils.cpp)
- [ ] Add unit tests

**Dependencies**: All previous modules
**Testing**: Unit tests with small graphs, integration tests

---

### Phase 4: Validation & Integration

#### 4.1 Implement MappingValidator
**File**: `tt_metal/fabric/topology_solver_internal.hpp` (declaration), `tt_metal/fabric/topology_solver_internal.tpp` (implementation)

**Structure**:
```cpp
namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
struct MappingValidator {
    static bool validate_mapping(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        ConnectionValidationMode validation_mode,
        std::vector<std::string>* warnings = nullptr);

    static void validate_connection_counts(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        ConnectionValidationMode validation_mode,
        std::vector<std::string>* warnings);

    static MappingResult<TargetNode, GlobalNode> build_result(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const DFSSearchEngine<TargetNode, GlobalNode>::SearchState& state,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        ConnectionValidationMode validation_mode);
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

**Warning Message Format**:
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
    GraphIndexData<TargetNode, GlobalNode> graph_data(target_graph, global_graph);
    ConstraintIndexData<TargetNode, GlobalNode> constraint_data(constraints, graph_data);
    
    // 2. Fast path for path graphs
    if (PathGraphDetector::is_path_graph(graph_data)) {
        // Try fast path with validation_mode...
    }
    
    // 3. General DFS
    DFSSearchEngine<TargetNode, GlobalNode> engine;
    // Uses SearchHeuristic::select_and_generate_candidates() for node selection and candidate generation
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
- [ ] Pass validation_mode through to SearchHeuristic and all validation functions
- [ ] Use SearchHeuristic for node selection and candidate generation in DFS
- [ ] Add initial logging: `log_info(tt::LogFabric, ...)` for mapping start
- [ ] Log warnings when relaxed mode finds mismatches

**Dependencies**: All modules
**Testing**: Integration tests, end-to-end tests

---

## Testing Strategy

### Unit Testing Approach

**For Each Module**:
1. **Test Fixtures**: Create reusable test fixtures
2. **Edge Cases**: Test boundary conditions (empty, single node, etc.)
3. **Error Cases**: Test error handling
4. **Correctness**: Verify expected behavior
5. **Performance**: Benchmark critical paths

### Integration Testing Approach

1. **Equivalence Testing**: Compare with `map_mesh_to_physical`
2. **Regression Testing**: Use existing test cases
3. **Real-World Testing**: Use actual mesh graphs and topologies
4. **Performance Testing**: Benchmark against original

### Test Categories

1. **GraphIndexData Tests**: Empty graphs, single node, path graphs, multi-edge graphs, self-connections
2. **ConstraintIndexData Tests**: Empty constraints, required/preferred/mixed constraints, trait constraints
3. **SearchHeuristic Tests**: Node selection cost, candidate ordering, priority hierarchy
4. **ConsistencyChecker Tests**: Local/forward consistency, all validation modes
5. **PathGraphDetector Tests**: Path detection, path mapping, non-path graphs
6. **DFSSearchEngine Tests**: Simple/complex mappings, memoization, backtracking
7. **MappingValidator Tests**: Validation correctness, all validation modes, warning formatting
8. **Integration Tests**: End-to-end scenarios, connection validation modes, real-world graphs

### Build and Test Commands

```bash
# Build
./build_metal.sh --build-metal-tests --debug

# Run tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
```

## Implementation Checklist

### ✅ Phase 1: Foundation
- [x] Create `tt_metal/fabric/topology_solver_internal.hpp`
- [x] Implement `GraphIndexData`
- [x] Implement `ConstraintIndexData`
- [x] Unit tests for Phase 1

### ✅ Phase 2: Heuristics & Pruning
- [x] Implement `SearchHeuristic` (unified)
- [x] Implement `ConsistencyChecker`
- [x] Unit tests for Phase 2

### ⏳ Phase 3: Search Engine
- [ ] Implement `PathGraphDetector`
- [ ] Implement `DFSSearchEngine`
- [ ] Unit tests for Phase 3

### ⏳ Phase 4: Integration
- [ ] Implement `MappingValidator`
- [ ] Implement `solve_topology_mapping()`
- [ ] Integration tests
- [ ] Compare with existing implementation

### ⏳ Phase 5: Testing & Polish
- [ ] Complete unit test suite
- [ ] Complete integration tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] Code review

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
