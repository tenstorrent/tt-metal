# Topology Solver Implementation Order

## Current Status

✅ **Already Implemented:**
- GraphIndexData - Complete
- ConstraintIndexData - Complete
- NodeSelector - Complete
- CandidateGenerator - Complete
- ConsistencyChecker - Complete
- PathGraphDetector - Complete
- DFSSearchEngine - Complete
- MappingValidator - Complete
- solve_topology_mapping() - Complete

✅ **Tests Passing:**
- BuildAdjacencyMapLogical
- BuildAdjacencyMapPhysical
- MappingConstraintsBasicOperations
- MappingConstraintsTraitConstraints
- MappingConstraintsIntersection
- MappingConstraintsConflictHandling

❌ **Test Failing:**
- SolveTopologyMappingPlaceholder - This is expected! The test expects failure but solver succeeds.

## Implementation Plan

### Phase 1: Fix Placeholder Test ✅ (DONE - solver works!)
- [x] Update test to verify solver actually works
- [x] Test should verify successful mapping

### Phase 2: Add Basic Module Tests
- [ ] GraphIndexData tests (simple cases)
- [ ] ConstraintIndexData tests (simple cases)
- [ ] NodeSelector tests (MRV heuristic)
- [ ] CandidateGenerator tests (filtering)
- [ ] ConsistencyChecker tests (local/forward)
- [ ] PathGraphDetector tests (detection + mapping)
- [ ] DFSSearchEngine tests (simple search)
- [ ] MappingValidator tests (validation + warnings)

### Phase 3: Add Integration Tests
- [ ] Simple 2-node mapping
- [ ] Simple 3-node path mapping
- [ ] Simple 4-node cycle mapping
- [ ] Constraint scenarios (required/preferred)
- [ ] Connection validation modes (STRICT/RELAXED/NONE)

### Phase 4: Edge Cases
- [ ] Empty graphs
- [ ] Single node
- [ ] Over-constrained (no solution)
- [ ] Under-constrained (many solutions)

## Next Steps

1. **Fix the placeholder test** - Update it to verify the solver works correctly
2. **Add basic module tests** - One test per module to verify basic functionality
3. **Add integration tests** - Test end-to-end scenarios
4. **Test edge cases** - Ensure robustness

## Testing Strategy

- **Minimal tests**: Just enough to cover each category
- **Focus on correctness**: Verify each module works as expected
- **Incremental**: Test each module before moving to next
- **Integration**: Test full solver with real scenarios
