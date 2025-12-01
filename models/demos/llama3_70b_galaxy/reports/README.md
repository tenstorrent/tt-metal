# Llama3-70B Galaxy Implementation: Complete Documentation

This directory contains **comprehensive, code-level documentation** for the Llama3-70B Galaxy implementation, optimized for 32-chip Wormhole (Galaxy) hardware.

## 🎯 Quick Navigation

### I'm new to this codebase, where do I start?

**→ Start with [COMPLETE_TECHNICAL_REFERENCE.md](COMPLETE_TECHNICAL_REFERENCE.md)**

This provides:
- High-level architecture overview
- Hardware and model configuration
- Complete data flow diagrams
- Tensor shapes reference
- Quick reference tables

### I want to understand specific operations in detail

**→ Read [DETAILED_CODE_WALKTHROUGH.md](DETAILED_CODE_WALKTHROUGH.md)**

This provides:
- Line-by-line code analysis
- Actual tensor shapes at each step
- Memory and program configurations
- Complete decode attention and MLP flows

### I need to understand collective communication (CCL)

**→ Read [CCL_OPERATIONS_COMPLETE.md](CCL_OPERATIONS_COMPLETE.md)**

This provides:
- Complete CCL class architecture
- Buffer management (all types)
- All-reduce, all-gather, reduce-scatter operations
- Ring topology details

### I want to understand the prefetcher system

**→ Read [PREFETCHER_SYSTEM_COMPLETE.md](PREFETCHER_SYSTEM_COMPLETE.md)**

This provides:
- Prefetcher architecture and design
- Sub-device management
- Global circular buffer details
- Performance analysis (2-3x speedup)

### I'm comparing TT-Transformers vs Galaxy implementation

**→ Read [COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md](COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md)**

This provides:
- Side-by-side comparison
- When to use each implementation
- Migration guide
- Performance differences

---

## 📚 Document Index

### Core Documentation (New, Detailed)

| Document | Purpose | Lines | Completeness |
|----------|---------|-------|--------------|
| **[COMPLETE_TECHNICAL_REFERENCE.md](COMPLETE_TECHNICAL_REFERENCE.md)** | High-level overview, quick reference | ~1,000 | ✅ Complete |
| **[DETAILED_CODE_WALKTHROUGH.md](DETAILED_CODE_WALKTHROUGH.md)** | Line-by-line code analysis | ~1,000+ | 🔄 Expanding |
| **[CCL_OPERATIONS_COMPLETE.md](CCL_OPERATIONS_COMPLETE.md)** | Complete CCL reference | ~1,200+ | 🔄 Expanding |
| **[PREFETCHER_SYSTEM_COMPLETE.md](PREFETCHER_SYSTEM_COMPLETE.md)** | Complete prefetcher reference | ~900 | ✅ Complete |
| **[COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md](COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md)** | Comparison guide | ~1,100 | ✅ Complete |

### Original Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| **[OPERATIONS_INDEX.md](OPERATIONS_INDEX.md)** | Navigation index | ~325 |
| **[OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md)** | Decode operations | ~960 |
| **[PREFILL_OPERATIONS_DETAILED.md](PREFILL_OPERATIONS_DETAILED.md)** | Prefill operations | ~630 |
| **[CCL_AND_MEMORY_DETAILED.md](CCL_AND_MEMORY_DETAILED.md)** | CCL and memory | ~535 |

---

## 🔍 What's Covered

### Complete Coverage

✅ **Decode Attention** - Every step with code references
✅ **Decode MLP** - Every step with code references
✅ **Prefetcher System** - Complete architecture and performance
✅ **CCL Operations** - Buffer management and all-reduce
✅ **Memory Management** - All memory configurations
✅ **Comparison** - TT-Transformers vs Galaxy
✅ **Performance** - Throughput, latency, utilization

### Partial Coverage (Expanding)

🔄 **Prefill Attention** - Basic flow covered
🔄 **Prefill MLP** - Basic flow covered
🔄 **All-Gather Operations** - Basic coverage
🔄 **Reduce-Scatter Operations** - Basic coverage

---

## 📊 Key Specifications

### Hardware
- **Devices**: 32 (8 rows × 4 columns mesh)
- **Topology**: Ring (both row and column)
- **Links**: 3 per direction (~900 GB/s total)
- **Cores per Device**: 70 (7×10 grid)

### Model (Llama3-70B)
- **Hidden Dimension**: 8192 (1024 per device)
- **Intermediate Size**: 28672 (3584 per device)
- **Num Heads**: 64 (8 per device)
- **Num KV Heads**: 8 (1 per device, GQA)
- **Num Layers**: 80
- **Vocab Size**: 128256

### Performance (Decode, Batch=32)
- **Throughput**: ~2,667 tokens/second
- **Latency**: ~12 ms per token
- **Core Utilization**: ~93%
- **Speedup vs No-Prefetcher**: 2.3x

---

## 💡 Key Optimizations

### 1. Prefetcher System (Decode)
- **Dedicated sub-device** with 12 cores for weight prefetching
- **Global circular buffer** (~1.6 GB in L1)
- **Overlaps** weight loading with computation
- **Speedup**: 2-3x throughput improvement

### 2. Ring Topology CCL
- **Optimal bandwidth** utilization
- **Persistent buffers** for all CCL operations
- **Double buffering** for continuous operation
- **Low latency**: ~1-2 µs per hop

### 3. Fused Operations
- **Double MatMul** (W1/W3 together)
- **RS Create Heads** (create + reduce-scatter)
- **All-Gather Concat** (gather + concatenate)
- **SiLU + Multiply** (activation + element-wise)

### 4. Mode-Specific Paths
- **Decode**: Prefetcher, ring all-reduce, fused ops
- **Prefill**: Minimal matmul (≥4096 seq), ring SDPA (>1024 seq)

---

## 🎓 Learning Path

### Beginner
1. Read **COMPLETE_TECHNICAL_REFERENCE.md** - Get overview
2. Look at architecture diagrams
3. Review tensor shape tables
4. Understand data flow (decode and prefill)

### Intermediate
1. Read **DETAILED_CODE_WALKTHROUGH.md** - Decode attention
2. Read **DETAILED_CODE_WALKTHROUGH.md** - Decode MLP
3. Read **PREFETCHER_SYSTEM_COMPLETE.md** - Prefetcher system
4. Review performance characteristics

### Advanced
1. Read **CCL_OPERATIONS_COMPLETE.md** - CCL internals
2. Study buffer management
3. Understand ring topology implementation
4. Review **COMPARISON** doc for alternatives

### Expert
1. Read all detailed code walkthroughs
2. Trace execution through actual code
3. Modify and experiment
4. Contribute to documentation

---

## 🛠️ How to Use This Documentation

### For Development
- Use **COMPLETE_TECHNICAL_REFERENCE** for quick lookups
- Use **DETAILED_CODE_WALKTHROUGH** for implementation details
- Use **CCL_OPERATIONS_COMPLETE** for CCL modifications
- Use **PREFETCHER_SYSTEM_COMPLETE** for prefetcher changes

### For Debugging
- Check **DETAILED_CODE_WALKTHROUGH** for operation flows
- Check **tensor shapes reference** for shape mismatches
- Check **memory configurations** for memory issues
- Check **CCL operations** for communication problems

### For Optimization
- Review **PREFETCHER_SYSTEM_COMPLETE** for decode optimization
- Review **CCL_OPERATIONS_COMPLETE** for communication optimization
- Review **performance characteristics** for bottlenecks
- Review **COMPARISON** for alternative implementations

### For Integration
- Review **COMPLETE_TECHNICAL_REFERENCE** for interfaces
- Review **data flow diagrams** for integration points
- Review **configuration files** for settings
- Review **COMPARISON** for compatibility

---

## 📖 Code References

All documentation includes **actual code references** with file names and line numbers:

**Example**:
```
File: llama_attention.py
Method: forward_decode()
Lines: 389-398
Operation: QKV Projection
```

This makes it easy to:
- Navigate to the actual code
- Verify documentation accuracy
- Understand implementation details
- Debug issues

---

## 🔗 Related Documentation

### Source Code
- **Implementation**: `models/demos/llama3_70b_galaxy/tt/`
- **Tests**: `models/demos/llama3_70b_galaxy/tests/`
- **Demos**: `models/demos/llama3_70b_galaxy/demo/`

### Other Documentation
- **README.md**: `models/demos/llama3_70b_galaxy/README.md`
- **PERF.md**: `models/demos/llama3_70b_galaxy/PERF.md`

---

## 🤝 Contributing

This documentation is actively maintained. To contribute:

1. **Add missing sections** to DETAILED_CODE_WALKTHROUGH.md
2. **Expand CCL operations** in CCL_OPERATIONS_COMPLETE.md
3. **Add examples** and use cases
4. **Update code references** if code changes
5. **Add diagrams** for complex operations

---

## 📝 Changelog

### 2025-11 - Major Documentation Update
- ✅ Created COMPLETE_TECHNICAL_REFERENCE.md (comprehensive reference)
- ✅ Created DETAILED_CODE_WALKTHROUGH.md (line-by-line analysis)
- ✅ Created CCL_OPERATIONS_COMPLETE.md (complete CCL reference)
- ✅ Created PREFETCHER_SYSTEM_COMPLETE.md (complete prefetcher reference)
- ✅ Updated OPERATIONS_INDEX.md (improved navigation)
- ✅ Created this README.md (documentation index)
- 🔄 Expanding code walkthroughs (ongoing)
- 🔄 Expanding CCL operations (ongoing)

### Previous
- Original OPERATION_DETAILED_REPORTS.md
- Original PREFILL_OPERATIONS_DETAILED.md
- Original CCL_AND_MEMORY_DETAILED.md
- Original COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md

---

## 📬 Feedback

Have questions or suggestions? Found errors or omissions?

- **File an issue** in the repository
- **Submit a pull request** with corrections
- **Add comments** directly in the code

---

## Summary

This documentation suite provides **comprehensive, code-level** coverage of the Llama3-70B Galaxy implementation:

- **8 major documents** (~7,000+ lines total)
- **Complete code walkthroughs** with line numbers
- **Detailed tensor shapes** at every step
- **Performance analysis** and optimization details
- **Comparison** with alternative implementations

**Start with [COMPLETE_TECHNICAL_REFERENCE.md](COMPLETE_TECHNICAL_REFERENCE.md) and follow the learning path above!**
