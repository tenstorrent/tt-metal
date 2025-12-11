# Topology Solver Implementation Summary

## Quick Reference

### Algorithm Overview
The topology solver implements a **constraint satisfaction problem (CSP)** solver for **graph isomorphism/subgraph matching**. It finds a mapping from a target graph (smaller) to a global graph (larger) while satisfying constraints.

### Core Strategy
1. **Preprocessing**: Convert graphs to efficient indexed representation
2. **Fast Path**: Specialized O(n) algorithm for path graphs
3. **General DFS**: Backtracking search with heuristics and pruning
4. **Validation**: Verify mapping correctness

## Module Breakdown

### ✅ Stage 1: Graph Preprocessing (`GraphIndexData`)
**Purpose**: Convert `AdjacencyGraph` to indexed representation for efficient lookups

**Key Structures**:
- Node vectors and index mappings
- Deduplicated, sorted adjacency lists
- Connection count maps (for multi-edge/strict mode)
- Degree vectors

**Reusability**: 100% - works with any graph types

---

### ✅ Stage 2: Constraint Processing (`ConstraintIndexData`)
**Purpose**: Convert `MappingConstraints` to index-based representation

**Key Structures**:
- Restricted mappings: `target_idx → set of valid global_indices`
- Preferred mappings: `target_idx → set of preferred global_indices`

**Reusability**: 100% - works with any constraint system

---

### ✅ Stage 3: Fast Path (`PathGraphDetector`)
**Purpose**: Optimize for common case of path graphs (linear chains)

**Detection**: 2 endpoints (degree 1), all others degree ≤ 2

**Algorithm**: Linear path-extension DFS

**Reusability**: 100% - generic path detection

---

### ✅ Stage 4: Node Selection (`NodeSelector`)
**Purpose**: Dynamic node selection using MRV (Minimum Remaining Values) heuristic

**Heuristic**:
1. Prefer nodes with most mapped neighbors
2. Among those, prefer nodes with fewest candidates
3. Among those, prefer nodes with lowest degree

**Reusability**: 100% - generic heuristic (can be swapped)

---

### ✅ Stage 5: Candidate Generation (`CandidateGenerator`)
**Purpose**: Generate and filter valid candidates

**Filtering Criteria**:
1. Not already used
2. Degree sufficient
3. Constraint satisfaction
4. Local consistency (mapped neighbors connected)
5. Forward checking (future neighbors viable)

**Ordering**: By degree gap (tighter fits first)

**Reusability**: 100% - generic candidate generation

---

### ✅ Stage 6: Consistency Checking (`ConsistencyChecker`)
**Purpose**: Validate local and forward consistency

**Checks**:
- **Local**: Mapped neighbors must be connected
- **Forward**: Future neighbors must have viable candidates
- **Strict Mode**: Channel counts must be sufficient

**Reusability**: 100% - generic consistency checks

---

### ✅ Stage 7: DFS Search Engine (`DFSSearchEngine`)
**Purpose**: Core backtracking search with memoization

**Features**:
- Memoization of failed states
- Progress logging
- Backtrack counting
- Configurable strict mode

**Reusability**: 100% - generic DFS engine

---

### ✅ Stage 8: Validation (`MappingValidator`)
**Purpose**: Validate final mapping

**Checks**:
- All edges preserved
- Strict mode channel counts
- Constraint satisfaction

**Reusability**: 100% - generic validation

## Implementation Order

### Phase 1: Foundation (Core Data Structures)
1. ✅ `GraphIndexData` - Graph preprocessing
2. ✅ `ConstraintIndexData` - Constraint processing

**Dependencies**: None
**Testability**: High - can test independently with mock data

---

### Phase 2: Heuristics & Pruning (Search Components)
3. ✅ `NodeSelector` - Node selection heuristic
4. ✅ `CandidateGenerator` - Candidate generation
5. ✅ `ConsistencyChecker` - Consistency validation

**Dependencies**: Phase 1
**Testability**: High - can test with simple graphs

---

### Phase 3: Search Engine (Core Algorithm)
6. ✅ `DFSSearchEngine` - Backtracking search
7. ✅ `PathGraphDetector` - Fast path optimization

**Dependencies**: Phases 1-2
**Testability**: Medium - requires integration testing

---

### Phase 4: Integration (Top-Level Solver)
8. ✅ `MappingValidator` - Result validation
9. ✅ `solve_topology_mapping()` - Main solver function

**Dependencies**: Phases 1-3
**Testability**: High - end-to-end tests

---

### Phase 5: Optimization & Testing
10. ✅ Unit tests for each module
11. ✅ Integration tests
12. ✅ Performance profiling
13. ✅ Memory optimization

## Key Design Patterns

### 1. Index-Based Representation
**Why**: Efficient lookups, cache-friendly
**Trade-off**: Extra memory for index mappings
**Benefit**: O(1) adjacency lookups

### 2. Two-Stage Constraint Processing
**Stage 1**: Convert constraints to index-based representation
**Stage 2**: Apply constraints during candidate generation
**Benefit**: Separation of concerns, easier to test

### 3. Modular Heuristics
**Why**: Easy to swap/experiment
**Current**: MRV (Minimum Remaining Values)
**Future**: Could add degree-based, constraint-based heuristics

### 4. Memoization Strategy
**Why**: Avoid revisiting failed states
**Implementation**: Hash-based state caching
**Trade-off**: Memory vs. time

### 5. Fast Path Optimization
**Why**: Path graphs are common and easy to solve
**Benefit**: O(n) instead of exponential for paths

## Mapping from Old to New Code

| Old Code (map_mesh_to_physical) | New Code (solve_topology_mapping) |
|----------------------------------|-----------------------------------|
| `LogicalAdjacencyMap` | `AdjacencyGraph<TargetNode>` |
| `PhysicalAdjacencyMap` | `AdjacencyGraph<GlobalNode>` |
| `FabricNodeId` | `TargetNode` (template) |
| `AsicID` | `GlobalNode` (template) |
| `TopologyMappingConfig` | `MappingConstraints` + `strict_mode` param |
| `node_to_host_rank` | Trait constraint in `MappingConstraints` |
| `asic_to_host_rank` | Trait constraint in `MappingConstraints` |
| `config.pinnings` | `MappingConstraints::add_required_constraint()` |
| `log_adj_idx`, `phys_adj_idx` | `GraphIndexData` |
| `restricted_phys_indices_for_logical` | `ConstraintIndexData` |
| `try_fast_path_for_logical_chain()` | `PathGraphDetector` |
| `select_next_logical()` | `NodeSelector` |
| Candidate generation loop | `CandidateGenerator` |
| Consistency checks | `ConsistencyChecker` |
| `dfs()` function | `DFSSearchEngine` |
| Final validation | `MappingValidator` |

## Testing Strategy

### Unit Tests (Per Module)
- **GraphIndexData**: Test with various graph structures
- **ConstraintIndexData**: Test constraint intersection, validation
- **NodeSelector**: Test MRV heuristic on different graphs
- **CandidateGenerator**: Test filtering and ordering
- **ConsistencyChecker**: Test local and forward checks
- **PathGraphDetector**: Test path detection and mapping
- **DFSSearchEngine**: Test backtracking, memoization
- **MappingValidator**: Test validation logic

### Integration Tests
- End-to-end solver tests
- Compare with original `map_mesh_to_physical` results
- Test with real mesh graphs and physical topologies

### Edge Cases
- Empty graphs
- Disconnected graphs
- Over-constrained (no solution)
- Under-constrained (many solutions)
- Path graphs (fast path)
- Complete graphs
- Star graphs

## Performance Targets

### Time Complexity
- **Preprocessing**: O(n + m + E) - acceptable
- **Fast Path**: O(n * m) worst case, typically O(n) - excellent
- **General DFS**: Exponential worst case, but heavily pruned - acceptable for small-medium graphs

### Space Complexity
- **GraphIndexData**: O(n + m + E) - acceptable
- **ConstraintIndexData**: O(n * m) worst case - acceptable
- **Search State**: O(n + m + failed_states) - bounded by search space

### Optimization Opportunities
- Early termination on fast path success ✅
- Aggressive pruning with forward checking ✅
- Candidate ordering (try promising first) ✅
- Memoization (avoid redundant work) ✅
- Parallel search (future work)

## Migration Path

1. **Keep existing code**: `map_mesh_to_physical` continues to work
2. **Implement new solver**: Build modules incrementally
3. **Test equivalence**: Verify same results on test cases
4. **Switch over**: Update callers to use new solver
5. **Deprecate old code**: Remove `map_mesh_to_physical` after migration

## Next Steps (Immediate Actions)

### Step 1: Create Internal Header Files
Create header files for internal modules:
- `topology_solver_internal.hpp` - Forward declarations
- `topology_solver_graph_index.hpp` - GraphIndexData
- `topology_solver_constraints.hpp` - ConstraintIndexData

### Step 2: Implement GraphIndexData
Start with `GraphIndexData` - it's the foundation:
- `build_graph_index_data()` function
- Test with simple graphs

### Step 3: Implement ConstraintIndexData
Next, implement constraint processing:
- `build_constraint_index_data()` function
- Test constraint intersection

### Step 4: Implement Heuristics
Build the search components:
- `NodeSelector::select_next_target()`
- `CandidateGenerator::generate_candidates()`
- `ConsistencyChecker` methods

### Step 5: Implement Search Engine
Build the core DFS:
- `DFSSearchEngine` class
- `PathGraphDetector` for fast path

### Step 6: Integrate
Put it all together:
- `solve_topology_mapping()` implementation
- `MappingValidator` for result validation

### Step 7: Test & Optimize
- Unit tests for each module
- Integration tests
- Performance profiling

## Questions to Resolve

1. **Strict Mode**: How to handle multi-edge/channel counts in generic solver?
   - **Answer**: Add `strict_mode` parameter, use connection count maps

2. **Preferred Constraints**: How to optimize for preferred constraints?
   - **Answer**: Score solutions by preferred constraint satisfaction, try preferred candidates first

3. **Error Messages**: How to provide helpful error messages?
   - **Answer**: Track which constraints failed, provide diagnostic info

4. **Statistics**: What statistics to collect?
   - **Answer**: DFS calls, backtrack count, elapsed time, constraint satisfaction

5. **Memory Limits**: How to handle very large graphs?
   - **Answer**: Add configurable limits, early termination, iterative deepening (future)

## Success Criteria

✅ **Functionality**:
- Solves same problems as `map_mesh_to_physical`
- Handles all constraint types
- Fast path works for path graphs
- Strict mode validation works

✅ **Code Quality**:
- Modular, testable design
- Clear separation of concerns
- Well-documented
- Follows existing code style

✅ **Performance**:
- Comparable or better than original
- Fast path significantly faster for paths
- Memory usage acceptable

✅ **Maintainability**:
- Easy to understand and modify
- Easy to add new heuristics
- Easy to add new constraint types
- Good test coverage
