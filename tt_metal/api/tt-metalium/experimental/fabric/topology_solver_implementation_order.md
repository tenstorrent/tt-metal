# Topology Solver Implementation Order

## Implementation Steps (Piece by Piece)

### ✅ Phase 1.1: Create Internal Header File
**Status**: COMPLETE
- ✅ Create `tt_metal/fabric/topology_solver_internal.hpp`
- ✅ Set up namespace and forward declarations
- ✅ Include necessary headers

### ✅ Phase 1.2: Implement GraphIndexData
**Status**: COMPLETE - Ready for Review
- ✅ Declare GraphIndexData struct
- ✅ Implement build_graph_index_data()
- ✅ Basic unit tests (GraphIndexDataBasic, GraphIndexDataEmpty)

### ⏳ Phase 1.3: Implement ConstraintIndexData
**Status**: Pending Phase 1.2
- Declare ConstraintIndexData struct
- Implement build_constraint_index_data()
- Basic unit tests

### ⏳ Phase 2.1: Implement NodeSelector
**Status**: Pending Phase 1.3
- Declare NodeSelector struct
- Implement select_next_target()
- Basic unit tests

### ⏳ Phase 2.2: Implement CandidateGenerator
**Status**: Pending Phase 2.1
- Declare CandidateGenerator struct
- Implement generate_candidates()
- Basic unit tests

### ⏳ Phase 2.3: Implement ConsistencyChecker
**Status**: Pending Phase 2.2
- Declare ConsistencyChecker struct
- Implement consistency checking functions
- Basic unit tests

### ⏳ Phase 3.1: Implement PathGraphDetector
**Status**: Pending Phase 2.3
- Declare PathGraphDetector struct
- Implement path detection and fast path
- Basic unit tests

### ⏳ Phase 3.2: Implement DFSSearchEngine
**Status**: Pending Phase 3.1
- Declare DFSSearchEngine class
- Implement search functions
- Basic unit tests

### ⏳ Phase 4.1: Implement MappingValidator
**Status**: Pending Phase 3.2
- Declare MappingValidator struct
- Implement validation functions
- Basic unit tests

### ⏳ Phase 4.2: Implement solve_topology_mapping()
**Status**: Pending Phase 4.1
- Integrate all modules
- Implement main solver function
- Integration tests

## Testing Strategy

For each module:
1. Create minimal unit tests covering:
   - Basic functionality
   - Edge cases (empty, single node)
   - Error conditions
2. Run tests before moving to next module
3. Fix any issues before proceeding

## Build and Test Commands

```bash
# Build
./build_metal.sh --build-metal-tests --debug

# Run tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
```
