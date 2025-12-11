# Topology Solver Implementation Plan - Summary

## Quick Reference

This document provides a quick reference to all planning documents and key decisions.

## Planning Documents

1. **`topology_solver_modularization_plan.md`** - Detailed algorithm breakdown and module design
2. **`topology_solver_algorithm_flow.md`** - Visual flow diagrams and algorithm details
3. **`topology_solver_implementation_summary.md`** - Quick reference for modules and design decisions
4. **`topology_solver_implementation_plan.md`** - Complete implementation and testing plan (THIS IS THE MAIN PLAN)
5. **`topology_solver_namespace_organization.md`** - Namespace structure and organization

## Key Decisions

### Namespace Organization
- **Public API**: `tt::tt_fabric` namespace only
- **Internal Implementation**: `tt::tt_fabric::detail` namespace
- **Users**: Only interact with public API
- **Implementation**: Uses `detail` namespace internally

### File Structure
```
tt_metal/api/tt-metalium/experimental/fabric/
├── topology_solver.hpp              # Public API declarations
└── topology_solver.tpp              # Public template implementations (includes internal.hpp)

tt_metal/fabric/
├── topology_solver_internal.hpp     # Internal type declarations (NOT part of public API)
└── topology_solver.cpp              # Non-template implementations (public + internal)
```

**Note**: `topology_solver_internal.hpp` is in the implementation directory (`tt_metal/fabric/`), not the public API directory, so it's not part of the external interface.

### Module Breakdown
1. **GraphIndexData** - Graph preprocessing
2. **ConstraintIndexData** - Constraint processing
3. **NodeSelector** - MRV heuristic
4. **CandidateGenerator** - Candidate generation
5. **ConsistencyChecker** - Consistency validation (with ConnectionValidationMode)
6. **PathGraphDetector** - Fast path optimization
7. **DFSSearchEngine** - Core search engine (with ConnectionValidationMode)
8. **MappingValidator** - Result validation (with warnings collection for relaxed mode)

### Key Features
- **ConnectionValidationMode**: STRICT (fail on mismatch), RELAXED (warn on mismatch), NONE (ignore counts)
- **Relaxed Mode**: Prefers correct channel counts during search, but allows mismatches with warnings
- **Warning Collection**: `MappingResult::warnings` vector contains formatted warning messages

## Implementation Phases

### Phase 1: Foundation (Week 1)
- Create `tt_metal/fabric/topology_solver_internal.hpp` (internal header file)
- Implement GraphIndexData
- Implement ConstraintIndexData
- Unit tests

### Phase 2: Heuristics & Pruning (Week 1-2)
- Implement NodeSelector
- Implement CandidateGenerator
- Implement ConsistencyChecker
- Unit tests

### Phase 3: Search Engine (Week 2)
- Implement PathGraphDetector
- Implement DFSSearchEngine
- Unit tests

### Phase 4: Integration (Week 2-3)
- Implement MappingValidator
- Implement solve_topology_mapping()
- Integration tests

### Phase 5: Testing & Optimization (Week 3-4)
- Complete unit test suite
- Complete integration tests
- Performance optimization
- Documentation

## Testing Strategy

### Unit Tests
- Each module tested independently
- >90% code coverage target
- Edge cases and error scenarios

### Integration Tests
- End-to-end solver tests
- Compare with `map_mesh_to_physical` results
- Real-world scenarios

## Success Criteria

### Functionality
- ✅ Solves same problems as `map_mesh_to_physical`
- ✅ Handles all constraint types
- ✅ Fast path works for path graphs
- ✅ Connection validation modes: STRICT, RELAXED (with warnings), NONE
- ✅ Relaxed mode prefers correct channel counts but allows mismatches with warnings

### Code Quality
- ✅ Modular, testable design
- ✅ Clear namespace separation
- ✅ Well-documented
- ✅ Follows existing code style
- ✅ Uses same logging utilities as topology_mapper

### Performance
- ✅ Comparable or better than original
- ✅ Fast path significantly faster for paths
- ✅ Memory usage acceptable

## Next Steps

1. **Review Plans**: Review all planning documents
2. **Start Phase 1**: Create internal header and implement GraphIndexData
3. **Incremental Development**: Implement modules one at a time with tests
4. **Integration**: Integrate modules into solve_topology_mapping()
5. **Testing**: Complete test suite and verify equivalence
6. **Optimization**: Profile and optimize performance

## Important Notes

- **DO NOT** start implementing until plans are reviewed
- **DO** implement incrementally with tests at each step
- **DO** keep internal code in `detail` namespace
- **DO** only expose public API in `tt::tt_fabric` namespace
- **DO** place internal header in `tt_metal/fabric/` (not public API directory)
- **DO** use logging utilities from `topology_mapper_utils.cpp`:
  - `log_info(tt::LogFabric, ...)` for informational messages
  - `log_debug(tt::LogFabric, ...)` for debug messages
  - `fmt::format` for error messages in `MappingResult::error_message`
- **DO** test thoroughly before moving to next phase

## Questions?

Refer to the detailed planning documents:
- Implementation details: `topology_solver_implementation_plan.md`
- Algorithm details: `topology_solver_algorithm_flow.md`
- Namespace details: `topology_solver_namespace_organization.md`
