# Llama3-70B Galaxy: Complete Documentation Index

## Overview

This directory contains **comprehensive, code-level documentation** for the Llama3-70B Galaxy implementation optimized for 32-chip Wormhole hardware. The documentation includes detailed code walkthroughs, tensor shapes, memory configurations, and performance analysis.

## 🎯 Quick Start Guide

**New to the codebase?** Start here:
1. **[COMPLETE_TECHNICAL_REFERENCE.md](COMPLETE_TECHNICAL_REFERENCE.md)** - High-level overview, architecture, and quick reference
2. **[DETAILED_CODE_WALKTHROUGH.md](DETAILED_CODE_WALKTHROUGH.md)** - Step-by-step code walkthrough with actual line numbers
3. **[COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md](COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md)** - Compare with TT-Transformers

**Need specific information?**
- **CCL operations**: [CCL_OPERATIONS_COMPLETE.md](CCL_OPERATIONS_COMPLETE.md)
- **Prefetcher system**: [PREFETCHER_SYSTEM_COMPLETE.md](PREFETCHER_SYSTEM_COMPLETE.md)
- **Original detailed reports**: [OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md)

---

## 📚 Complete Documentation Suite

### 1. [COMPLETE_TECHNICAL_REFERENCE.md](COMPLETE_TECHNICAL_REFERENCE.md)
**★ START HERE - Comprehensive high-level reference**

**Contents**:
- Architecture overview with diagrams
- Hardware configuration (8x4 mesh, 32 devices)
- Model configuration (dimensions, weights, dtypes)
- Complete data flow for decode and prefill
- Tensor shapes reference tables
- Memory configuration reference
- Program configuration reference
- Performance characteristics
- Quick reference tables

**Best for**:
- Understanding the overall system
- Quick reference for shapes and configs
- Performance characteristics
- Getting oriented in the codebase

**Length**: ~1,000 lines, comprehensive coverage

---

### 2. [DETAILED_CODE_WALKTHROUGH.md](DETAILED_CODE_WALKTHROUGH.md)
**★ Line-by-line code analysis with actual implementation details**

**Contents**:
1. **Decode Attention Forward Pass** (complete)
   - Step 1: QKV Projection (lines 389-398)
   - Step 2: Create QKV Heads with RS (lines 405-416)
   - Step 3: QK Normalization (optional, lines 418-469)
   - Step 4: RoPE Application (lines 474-479)
   - Step 5: KV Cache Update (lines 484-500)
   - Step 6: SDPA (lines 506-530)
   - Step 7: All-Gather Concat (lines 533-542)
   - Step 8: Output Projection (lines 545-554)
   - Step 9: All-Reduce (lines 556-563)

2. **Decode MLP Forward Pass** (complete)
   - Step 1: Double MatMul with RS (lines 120-139)
   - Step 2: Reduce-Scatter W3 (lines 143-150)
   - Step 3: SiLU + Multiply (lines 152-158)
   - Step 4: All-Gather (lines 163-171)
   - Step 5: W2 MatMul (lines 175-185)
   - Step 6: All-Reduce (lines 186-193)

3. **Prefill Attention Forward Pass** (partial)
   - Input reshaping, QKV projection, all-reduce
   - Create heads, RoPE, fill KV cache
   - Ring distributed SDPA (for long sequences)

4. **Additional Sections** (to be completed)
   - Prefill MLP Forward Pass
   - CCL Operations Deep Dive
   - Memory Management Deep Dive

**Best for**:
- Understanding exactly what the code does
- Debugging specific operations
- Learning the implementation details
- Seeing actual tensor shapes at each step

**Length**: ~1,000+ lines (expanding)

---

### 3. [CCL_OPERATIONS_COMPLETE.md](CCL_OPERATIONS_COMPLETE.md)
**★ Complete reference for all collective communication operations**

**Contents**:
1. **CCL Class Architecture**
   - Initialization and setup
   - Core range sets and sub-devices
   - Semaphore management (double-buffered)
   - Index tracking and cycling

2. **Buffer Management**
   - Persistent buffers (decode): Cluster axis 0 & 1
   - All-gather buffers: SDPA, Layernorm, Sampling, Binary Mul
   - Reduce-scatter buffers: Intermediate buffers for RS
   - RS create heads buffers: QKV head creation
   - Prefill buffers: For all supported sequence lengths (128, 1024, 2048, 4096)

3. **All-Reduce Operations**
   - Decode mode: Ring all-reduce with persistent buffers
   - Prefill mode: RS + AG decomposition
   - Line-by-line code analysis
   - Usage examples (QKV, WO, W2)

4. **All-Gather Operations** (to be completed)
5. **Reduce-Scatter Operations** (to be completed)
6. **Specialized Operations** (to be completed)
7. **Ring Topology Details** (to be completed)

**Best for**:
- Understanding distributed communication
- Buffer allocation and management
- Ring topology implementation
- CCL operation internals

**Length**: ~1,200+ lines (expanding)

---

### 4. [PREFETCHER_SYSTEM_COMPLETE.md](PREFETCHER_SYSTEM_COMPLETE.md)
**★ Complete reference for the prefetcher optimization system**

**Contents**:
1. **Prefetcher Architecture**
   - System overview with diagrams
   - Timeline visualization
   - Key components

2. **Sub-Device Management**
   - Core allocation (12 prefetcher, 58 worker)
   - Core range sets
   - Sub-device creation (prefill vs decode)
   - Stall groups

3. **Global Circular Buffer**
   - Purpose and benefits
   - Size calculation (728 × 1088 tiles ≈ 1.6 GB)
   - Creation and sender-receiver mapping
   - Usage in operations

4. **Weight Management**
   - Tensor insertion (per layer)
   - Tensor count (5 per layer, 400 total)
   - Tensor order across layers

5. **Tensor Address Tracking**
   - Address collection
   - Address tensor creation (sharded across prefetcher cores)
   - Address caching

6. **Integration with Operations**
   - MatMul integration (all decode matmuls)
   - Conditional usage (USE_PREFETCHER flag)

7. **Performance Analysis**
   - Latency breakdown (with/without prefetcher)
   - Memory bandwidth savings
   - Throughput analysis (2.3x improvement)
   - Core utilization (93% vs 50%)
   - Power efficiency (~30% better)

**Best for**:
- Understanding decode optimization
- Prefetcher system internals
- Performance characteristics
- Sub-device management

**Length**: ~900 lines, complete

---

### 5. [COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md](COMPARISON_TT_TRANSFORMERS_VS_GALAXY.md)
**Side-by-side comparison with TT-Transformers generic implementation**

**Contents**:
1. **Executive Summary** - Key differences at a glance
2. **Architecture Comparison** - Overall structure
3. **Code Structure Comparison** - File organization
4. **Attention Operations Comparison** - QKV, create heads, output
5. **MLP Operations Comparison** - W1/W3/W2 projections
6. **CCL Operations Comparison** - All-reduce, all-gather, RS
7. **Memory Management Comparison** - Configs and buffers
8. **Prefetcher System** - Galaxy-only feature
9. **Use Cases and Recommendations** - When to use each
10. **Migration Guide** - How to switch between implementations
11. **Code Examples Comparison** - Side-by-side code
12. **Summary Table** - Complete feature comparison

**Best for**:
- Deciding which implementation to use
- Understanding Galaxy-specific optimizations
- Migrating between implementations
- Learning the differences

**Length**: ~1,100 lines

---

### 6. [OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md)
**Original detailed operation reports (pre-code-analysis)**

**Contents**:
1. **Overview** - System architecture
2. **Attention Operations** (decode)
3. **MLP Operations** (decode)
4. **Distributed Norm Operations**
5. **CCL Operations** (overview)
6. **Prefetcher System** (overview)

**Best for**:
- High-level operation understanding
- Original reference documentation
- Quick operation summaries

**Length**: ~960 lines

---

### 7. [PREFILL_OPERATIONS_DETAILED.md](PREFILL_OPERATIONS_DETAILED.md)
**Detailed guide for prefill-specific operations**

**Contents**:
1. **Overview** - Key differences from decode
2. **Attention Prefill Operations**:
   - QKV Projection (prefill)
   - All-Reduce for QKV
   - Create QKV Heads
   - RoPE Application
   - KV Cache Fill
   - Scaled Dot-Product Attention
3. **MLP Prefill Operations**:
   - W1/W3 Projection (with minimal matmul)
   - Element-wise Multiply with SiLU
   - W2 Projection (with minimal matmul)

**Best for**: Understanding prefill operations and optimizations

**Length**: ~630 lines

---

### 8. [CCL_AND_MEMORY_DETAILED.md](CCL_AND_MEMORY_DETAILED.md)
**Deep dive into CCL operations and memory management (original)**

**Contents**:
1. **CCL Operations**:
   - Ring Topology Overview
   - Line Reduce-Scatter
   - Line All-Gather
   - Line All-Reduce
   - Double MatMul Line Reduce-Scatter
   - Llama RS Create Heads
   - All-Gather Concat
2. **Memory Management**:
   - Memory Configurations
   - Persistent Buffers
   - Buffer Management
   - Memory Layouts
3. **Prefetcher System**:
   - Architecture
   - Benefits
   - Code Examples

**Best for**: Understanding distributed communication and memory optimization

**Length**: ~535 lines

---

## Quick Navigation

### "I want to understand..."

#### **...how attention works in decode**
→ Read [OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md) section 2
- QKV Projection
- Create QKV Heads
- RoPE Application
- KV Cache Update
- SDPA
- Output Projection

#### **...how MLP works in decode**
→ Read [OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md) section 3
- Double MatMul
- Element-wise Multiply
- All-Gather
- W2 MatMul

#### **...how prefill differs from decode**
→ Read [PREFILL_OPERATIONS_DETAILED.md](PREFILL_OPERATIONS_DETAILED.md)
- All prefill operations
- Minimal matmul usage
- Sequence length handling

#### **...how CCL operations work**
→ Read [CCL_AND_MEMORY_DETAILED.md](CCL_AND_MEMORY_DETAILED.md) section 1
- Ring topology
- Reduce-scatter
- All-gather
- All-reduce

#### **...how memory is managed**
→ Read [CCL_AND_MEMORY_DETAILED.md](CCL_AND_MEMORY_DETAILED.md) section 2
- Memory configurations
- Persistent buffers
- Buffer management
- Memory layouts

#### **...how the prefetcher works**
→ Read [CCL_AND_MEMORY_DETAILED.md](CCL_AND_MEMORY_DETAILED.md) section 3
- Prefetcher architecture
- Circular buffer
- Overlap benefits

---

## Key Concepts

### Ring Topology
- 32 devices arranged in a ring
- Efficient communication pattern
- Uses all links simultaneously
- O(num_devices) latency

### Reduce-Scatter
- Reduces partial results across devices
- Scatters to appropriate devices
- Used for column-wise sharded operations

### All-Gather
- Gathers distributed results from all devices
- Used when full tensor needed
- Concatenates partial results

### All-Reduce
- Reduces results across all devices (sum)
- Same result on all devices
- Used for row-wise sharded operations

### Prefetcher
- Dedicated sub-device for weight prefetching
- Overlaps computation and memory access
- Uses circular buffer in L1

### Minimal MatMul
- Optimized matmul for long sequences
- Used for sequences >= 4096
- More efficient than standard matmul

---

## Operation Flow Summary

### Decode Flow

```
Input Token
  ↓
Embedding
  ↓
For each layer:
  ├─ Attention Norm
  ├─ QKV Projection → Reduce-Scatter → Create Heads
  ├─ RoPE Application
  ├─ KV Cache Update
  ├─ SDPA
  ├─ All-Gather Concat
  ├─ Output Projection → All-Reduce
  ├─ Residual Add
  ├─ FFN Norm
  ├─ W1/W3 Double MatMul → Reduce-Scatter
  ├─ SiLU + Multiply
  ├─ All-Gather
  ├─ W2 MatMul → All-Reduce
  └─ Residual Add
  ↓
Final Norm
  ↓
LM Head
  ↓
Logits
```

### Prefill Flow

```
Input Tokens (full sequence)
  ↓
Embedding
  ↓
For each layer:
  ├─ Attention Norm
  ├─ QKV Projection → All-Reduce
  ├─ Create Heads
  ├─ RoPE Application
  ├─ KV Cache Fill (full sequence)
  ├─ SDPA (full sequence)
  ├─ Output Projection → All-Reduce
  ├─ Residual Add
  ├─ FFN Norm
  ├─ W1 MatMul → Reduce-Scatter
  ├─ W3 MatMul → Reduce-Scatter
  ├─ SiLU + Multiply
  ├─ All-Gather
  ├─ W2 MatMul → All-Reduce
  └─ Residual Add
  ↓
Final Norm
  ↓
LM Head
  ↓
Logits (last token)
```

---

## Code References

### Key Files

| File | Purpose |
|------|---------|
| `llama_attention.py` | Attention operations (prefill and decode) |
| `llama_mlp.py` | MLP operations (prefill and decode) |
| `llama_decoder.py` | Transformer block orchestration |
| `llama_ccl.py` | CCL operations and buffers |
| `prefetcher_common.py` | Prefetcher setup and management |
| `llama_model.py` | Main model class |

### Key Classes

| Class | Purpose |
|------|---------|
| `TtLlamaAttention` | Attention module |
| `TtLlamaMLP` | MLP module |
| `TtTransformerBlock` | Single transformer layer |
| `TT_CCL` | Collective communication library |
| `TtLlamaPrefetcherSetup` | Prefetcher management |

---

## Performance Considerations

### Decode Optimizations
- **Fused Operations**: Double matmul, fused CCL ops
- **Prefetcher**: Overlaps computation and memory access
- **Persistent Buffers**: Pre-allocated for efficiency
- **Ring Topology**: Optimal communication pattern

### Prefill Optimizations
- **Minimal MatMul**: For sequences >= 4096
- **Chunking**: Long sequences processed in chunks
- **DRAM Memory**: For large sequences
- **Batch Processing**: Multiple users simultaneously

### Memory Optimizations
- **Sharding**: Distributes tensors across devices
- **Buffer Reuse**: Persistent buffers for CCL ops
- **Memory Layouts**: Optimized for compute operations
- **Prefetcher**: Reduces memory stalls

---

## Debugging Tips

### Understanding Operation Flow
1. **Trace Execution**: Use ttnn tracer to see operation sequence
2. **Check Shapes**: Verify tensor shapes at each step
3. **Memory Configs**: Ensure correct memory configs are used
4. **CCL Operations**: Verify reduce-scatter/all-gather correctness

### Common Issues
1. **Shape Mismatches**: Check sharding dimensions
2. **Memory Config Errors**: Verify memory config compatibility
3. **CCL Errors**: Check cluster_axis and num_links
4. **Prefetcher Issues**: Verify buffer allocation

### Performance Debugging
1. **Profile Operations**: Use benchmarking tools
2. **Check Buffer Usage**: Verify persistent buffers are used
3. **Memory Bandwidth**: Check DRAM vs L1 usage
4. **CCL Overhead**: Measure communication time

---

## Additional Resources

### Related Documentation
- Main README: [README.md](README.md)
- Performance Guide: [PERF.md](PERF.md)
- Demo Scripts: `demo/` directory
- Test Suite: `tests/` directory

### Key Configuration Files
- `model_config.py`: Model configuration
- `qwen_model_config.py`: Qwen-specific configuration
- `model_params/`: Per-model parameters

---

## Summary

These documents provide comprehensive coverage of:

1. **All Operations**: Decode and prefill operations
2. **CCL Details**: Ring topology communication
3. **Memory Management**: Buffers, layouts, configurations
4. **Optimizations**: Performance optimizations explained
5. **Code References**: Direct code references for each operation

**Start with**: [OPERATION_DETAILED_REPORTS.md](OPERATION_DETAILED_REPORTS.md) for decode operations, then explore prefill and CCL details as needed.
