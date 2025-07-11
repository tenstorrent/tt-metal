# Program Factory Refactoring - Summary

## Overview

We are undertaking a massive refactoring effort to convert all Program Factories from the old approach (creating `Program` directly) to the new approach (creating `ProgramDescriptor`). This affects **100+ operations** across the codebase.

## What We've Established

### 1. Tracking System ✅
- **Main Tracker**: `PROGRAM_FACTORY_REFACTORING_TRACKER.md` - Comprehensive tracking document
- **Discovery Script**: `scripts/discover_program_factories.py` - Automated discovery tool
- **Example Guide**: `REFACTORING_EXAMPLE.md` - Step-by-step refactoring example

### 2. Scope Identified ✅
Based on codebase analysis, we've identified operations in these categories:

#### Basic Operations (High Priority)
- **Copy Operations**: typecast, clone
- **Eltwise Operations**: unary, binary, tanh_accurate
- **Creation Operations**: full, full_like, uniform, bernoulli, rand
- **Reduction Operations**: cumsum, cumprod
- **Normalization Operations**: batch_norm
- **Experimental Operations**: dropout, gather, scatter, sort, gelu_backward

#### Moreh Operations (Medium Priority)
- **Optimizers**: moreh_adam, moreh_sgd
- **Normalization**: moreh_norm, moreh_layer_norm
- **Utilities**: moreh_arange, moreh_fold, moreh_matmul
- **Gradients**: moreh_clip_grad_norm, moreh_linear_backward, moreh_sum_backward
- **Statistics**: moreh_mean, moreh_mean_backward

#### TT-Train Operations (Low Priority)
- **Training Ops**: rmsnorm_fw, softmax, cross_entropy_fw/bw, profiler_no_op

### 3. Refactoring Pattern ✅

#### Old Way (Current)
```cpp
Program program{};  // Direct creation
CreateCircularBuffer(program, cores, config);
CreateKernel(program, path, cores, config);
return {std::move(program), shared_vars};
```

#### New Way (Target)
```cpp
ProgramDescriptor descriptor{};  // Create descriptor
descriptor.cbs.push_back(CBDescriptor{...});
descriptor.kernels.push_back(KernelDescriptor{...});
Program program = Program(descriptor);  // Convert to program
return {std::move(program), shared_vars};
```

### 4. Key Benefits ✅
1. **Separation of Concerns**: Program description vs. creation
2. **Better Testing**: Test descriptions independently
3. **Serialization**: Can serialize/deserialize program descriptions
4. **Validation**: Validate before creation
5. **Optimization**: Optimize descriptions before creating programs
6. **Debugging**: Easier to inspect program structure

## Next Steps

### Immediate Actions
1. **Run Discovery Script**: Execute `python3 scripts/discover_program_factories.py` to get exact counts
2. **Start with Simple Operation**: Begin with typecast or unary operation
3. **Establish Pattern**: Create first refactored operation as template
4. **Update Tracker**: Mark completed operations in tracker

### Short-term Goals
1. **Phase 1**: Complete all basic operations (copy, eltwise, creation)
2. **Phase 2**: Complete Moreh operations (optimizers, normalization)
3. **Phase 3**: Complete experimental and TT-train operations

### Medium-term Goals
1. **Automation**: Create tools to help with conversion
2. **Testing**: Comprehensive test suite for refactored operations
3. **Documentation**: Update all operation documentation

## Risk Mitigation

### High Risk Areas
- **Complex Operations**: Operations with many kernels/circular buffers
- **Performance**: Ensure no regression from descriptor overhead
- **Functionality**: Maintain exact same behavior

### Mitigation Strategies
- **Start Simple**: Begin with straightforward operations
- **Extensive Testing**: Test each refactored operation thoroughly
- **Performance Monitoring**: Benchmark before/after
- **Incremental Approach**: Refactor one operation at a time

## Success Metrics

- [ ] All 100+ operations successfully refactored
- [ ] Zero functional regressions
- [ ] Performance maintained or improved
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code maintainability improved

## Resources Available

1. **Tracker**: `PROGRAM_FACTORY_REFACTORING_TRACKER.md`
2. **Discovery Script**: `scripts/discover_program_factories.py`
3. **Example**: `REFACTORING_EXAMPLE.md`
4. **API Reference**: `tt_metal/api/tt-metalium/program_descriptors.hpp`

## Getting Started

1. **Choose First Operation**: Pick a simple operation (e.g., typecast)
2. **Follow Example**: Use `REFACTORING_EXAMPLE.md` as guide
3. **Test Thoroughly**: Ensure functionality is preserved
4. **Update Tracker**: Mark as completed
5. **Repeat**: Move to next operation

## Questions to Resolve

1. **Exact Count**: Run discovery script to get precise number of operations
2. **Priority Order**: Confirm priority order of operation categories
3. **Testing Strategy**: Define comprehensive testing approach
4. **Performance Baseline**: Establish performance benchmarks
5. **Rollback Plan**: Plan for reverting changes if needed

---

**Status**: Ready to begin implementation
**Estimated Timeline**: 2-3 months for complete refactoring
**Team Size**: 1-2 developers recommended
**Risk Level**: Medium (manageable with proper testing)
