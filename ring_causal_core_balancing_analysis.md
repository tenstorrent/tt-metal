# Ring Joint Causal Attention - Core Balancing Analysis

## Problem Statement

Implement core-level balancing for ring joint causal attention to optimize workload distribution within devices. Currently, only ring-level (cross-device) balancing exists.

## Current Implementation Analysis

### Ring-Level Balancing (Already Implemented)
- **Input reordering**: Sequence chunks distributed so each device gets chunks from both start and end
- **Example with 4 devices, 8 chunks**: Device 0 → chunks [0,7], Device 1 → chunks [1,6], Device 2 → chunks [2,5], Device 3 → chunks [3,4]
- **Benefit**: Balances total compute per device since early chunks attend to more KV pairs than later chunks

### Current Core Assignment (Lines 763-805)
- **Flat chunk distribution**: `total_q_chunks = B * NH * num_q_chunks`
- **Sequential core assignment**: Each core gets `base_chunks_per_core + (extra_chunks ? 1 : 0)` chunks
- **Information to kernels**: `global_q_start`, `global_q_count` → kernels extract (batch, head, q_chunk)

### Standard SDPA Balancing Pattern
- Uses `BALANCED_Q_PARALLEL` for first/last chunk distribution within cores
- **First half of chunks**: from beginning of sequence (high causal compute)
- **Second half of chunks**: from end of sequence (low causal compute)

## Target Test Case: test_mla_sdpa_bh_galaxy

### Configuration
- **Shapes**: B=1, NH_Q=128, NH_K=1, NH_V=128, seq_len=128K, head_dim=576/128
- **Chunks**: q_chunk_size=256, k_chunk_size=128
- **Device mesh**: 4x8 with rp_factor=8, up_factor=4
- **Worker cores**: ~112 cores per device
- **Per device**: 32 heads (128 heads ÷ 4 devices)

### Core-Head Math
- 112 cores ÷ 32 heads = 3.5 cores per head
- **Solution**: Group heads in pairs → 2 heads = 7 cores
- **Groups**: 16 groups of 7 cores, each handling 2 heads

## Proposed Core-Level Balancing Algorithm

### Head Pairing Strategy
- **Groups**: 16 groups of 7 cores each
- **Head pairs**: (0,1), (2,3), (4,5), ..., (30,31)
- **Assignment**: Core group 0 handles head pair (0,1), group 1 handles pair (2,3), etc.

### Logical Concatenation
- **Conceptual**: Treat 2 heads as one long sequence for balancing purposes
- **Implementation**: No actual data concatenation, just affects chunk assignment logic
- **Pattern**: First/last distribution across the logical concatenated sequence space

### Balanced Workload Distribution
Within each 7-core group:
- **Cores 0-3**: Early chunks from head 1 + Late chunks from head 2 (balanced compute)
- **Cores 4-6**: Late chunks from head 1 + Early chunks from head 2 (balanced compute)

## Implementation Approach

### Data Structure Extension
```cpp
struct CoreWork {
    CoreCoord logical_core;
    CoreCoord physical_core;
    uint32_t global_q_start = 0;      // First range
    uint32_t global_q_count = 0;      // First range
    uint32_t global_q_start_2 = 0;    // NEW: Second range
    uint32_t global_q_count_2 = 0;    // NEW: Second range
    std::vector<CoreHeadWork> head_work;
};
```

### Kernel Argument Extension
- **Current**: `global_q_start`, `global_q_end`
- **New**: `global_q_start`, `global_q_end`, `global_q_start_2`, `global_q_end_2`

### Kernel Modifications (Minimal)
```cpp
// First range processing (existing loop)
for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
    // existing logic unchanged
}

// Second range processing (new loop)
for (uint32_t global_q_chunk = global_q_start_2; global_q_chunk < global_q_end_2; ++global_q_chunk) {
    // same logic as first loop
}
```

### When Balancing is Disabled
- Set `global_q_start_2 = global_q_end_2` to skip second loop
- Fallback to current sequential assignment

## Key Design Decisions

1. **No new flags**: Always apply core balancing (no separate `core_balanced` flag)
2. **Inline implementation**: Modify existing loop rather than separate function (initially)
3. **Assert constraint**: Require even number of heads
4. **Chain optimization**: Leveraged existing skip when `args.is_balanced = true`
5. **Handle remainder**: Distribute remainder chunks across first few cores in group

## Files to Modify

### Program Factory
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:763-805`

### Kernel Files (3 kernels)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`

### Runtime Argument Passing
- Extend argument lists in program factory where kernels are invoked
- Update compile-time argument counts if needed

## Expected Benefits

- **Balanced workload**: Each core gets mix of high and low compute chunks
- **Performance improvement**: Better utilization of 112 worker cores
- **Proof of concept**: Foundation for general core balancing solution
