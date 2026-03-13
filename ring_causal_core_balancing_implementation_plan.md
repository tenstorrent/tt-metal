# Ring Joint Causal Attention - Core-Level Balancing Implementation Plan

## Context

This change addresses performance optimization for ring joint causal attention by implementing core-level balancing within devices. Currently, only ring-level (cross-device) balancing exists, leading to workload imbalance across the ~112 worker cores per device in causal attention scenarios.

The problem stems from causal attention's computational pattern where early sequence chunks attend to fewer KV pairs (less work) while later chunks attend to more KV pairs (more work). The current sequential chunk assignment causes cores processing early chunks to be underutilized while cores processing later chunks are overloaded.

This implementation adds core-level balancing that groups heads in pairs and applies a first/last chunk distribution pattern to balance the workload, similar to the existing `BALANCED_Q_PARALLEL` in standard SDPA but adapted for ring joint operations.

## Target Configuration

- **112 cores** → 16 groups of 7 cores each
- **32 heads** (128 ÷ 4 devices) → 16 head pairs: (0,1), (2,3), ..., (30,31)
- **Each group** handles one head pair using first/last balancing pattern
- **Test case**: `test_mla_sdpa_bh_galaxy` with shapes B=1, NH=128, seq_len=128K

## Implementation Strategy

### 1. Data Structure Extension

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
**Location**: Lines 717-723 (CoreWork struct)

Add second range fields to support dual-range chunk assignment:
```cpp
struct CoreWork {
    CoreCoord logical_core;
    CoreCoord physical_core;
    uint32_t global_q_start = 0;     // First range
    uint32_t global_q_count = 0;     // First range
    uint32_t global_q_start_2 = 0;   // NEW: Second range
    uint32_t global_q_count_2 = 0;   // NEW: Second range
    std::vector<CoreHeadWork> head_work;
};
```

### 2. Program Factory Logic Replacement

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
**Location**: Lines 763-805 (main chunk distribution loop)

#### Add Assertion
Insert after line 752:
```cpp
if (args.is_balanced && NH % 2 != 0) {
    TT_FATAL(false, "Balanced core distribution requires even number of heads, got {}", NH);
}
```

#### Replace Sequential with Balanced Distribution
Replace the existing loop with balanced head-pairing logic:

**Key Algorithm**:
- Group assignment: `group_id = core_id / 7`, `core_within_group = core_id % 7`
- Head pairing: Group handles heads `(2*group_id, 2*group_id + 1)`
- First/last distribution: Early chunks from head 1 + Late chunks from head 2 → balanced workload

**Mathematical Pattern**:
```cpp
if (args.is_balanced) {
    const uint32_t cores_per_group = 7;
    const uint32_t num_groups = 16;

    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t group_id = i / cores_per_group;
        const uint32_t core_within_group = i % cores_per_group;

        // Calculate balanced chunk assignment for this core
        // Split work into two ranges: early chunks + late chunks
        // Use first/last pattern across logical head pair sequence
    }
} else {
    // Existing sequential logic unchanged
    // Set global_q_count_2 = 0 for all cores
}
```

### 3. Runtime Arguments Extension

Extend all three kernels to accept second range arguments:

#### Reader Kernel
**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`
**Location**: Lines 878-889, argument parsing at lines 55-56

Extend argument list:
```cpp
std::vector<uint32_t> reader_args = {
    q_addr, k_addr, v_addr, gathered_k_addr, gathered_v_addr,
    joint_q_addr, joint_k_addr, joint_v_addr,
    global_q_start,   // Index 8
    global_q_end,     // Index 9
    global_q_start_2, // Index 10 - NEW
    global_q_end_2,   // Index 11 - NEW
};

// Update argument parsing:
const uint32_t global_q_start_2 = get_arg_val<uint32_t>(10);
const uint32_t global_q_end_2 = get_arg_val<uint32_t>(11);
```

#### Writer Kernel
**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
**Location**: Lines 931-937, argument parsing at lines 108-109

Extend argument list (indices 5-6):
```cpp
std::vector<uint32_t> writer_args = {
    out_addr, joint_out_addr, lse_addr,
    global_q_start,   // Index 3
    global_q_end,     // Index 4
    global_q_start_2, // Index 5 - NEW
    global_q_end_2,   // Index 6 - NEW
};
```

#### Compute Kernel
**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`
**Location**: Lines 942-945, argument parsing at lines 73-74

Extend argument list (indices 2-3):
```cpp
std::vector<uint32_t> compute_args = {
    global_q_start,   // Index 0
    global_q_end,     // Index 1
    global_q_start_2, // Index 2 - NEW
    global_q_end_2,   // Index 3 - NEW
};
```

### 4. Kernel Processing Loop Extensions

Add second processing loop to all three kernels:

#### Pattern for All Kernels
```cpp
// First processing loop (existing code unchanged)
for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
    // Extract batch, head, q_chunk from global_q_chunk using existing decode logic
    // ... existing processing logic
}

// Second processing loop (NEW)
if (global_q_end_2 > global_q_start_2) {  // Only run if second range is non-empty
    for (uint32_t global_q_chunk = global_q_start_2; global_q_chunk < global_q_end_2; ++global_q_chunk) {
        // Same processing logic as first loop
        // Reuse existing decode_flat_chunk extraction
        // ... duplicate the main processing code
    }
}
```

**Implementation Locations**:
- **Reader**: After main loop around line 350
- **Writer**: After main loop around line 260
- **Compute**: After main sdpa_ring function call around line 210

### 5. Critical Files to Modify

- **`ring_joint_sdpa_program_factory.cpp`**: Core balancing logic and argument construction
- **`ring_joint_reader.cpp`**: Reader kernel argument parsing and second loop
- **`ring_joint_writer.cpp`**: Writer kernel argument parsing and second loop
- **`ring_joint_sdpa.cpp`**: Compute kernel argument parsing and second loop
- **`test_ring_joint_mla.py`**: Validation testing

### 6. Implementation Order (Minimize Breaking Changes)

1. **Phase 1**: Data structure extension (add unused fields)
2. **Phase 2**: Assertion and fallback logic (verify no regression)
3. **Phase 3**: Balanced distribution algorithm (behind `is_balanced` flag)
4. **Phase 4**: Runtime argument extension (backward compatible)
5. **Phase 5**: Kernel loop extensions (conditional on new arguments)
6. **Phase 6**: Integration testing and performance validation

### 7. Verification Strategy

#### Functional Testing
1. **Assertion Testing**: Verify even head count requirement
2. **Fallback Testing**: Ensure `is_balanced=false` maintains original behavior
3. **Correctness Testing**: Compare outputs with reference implementation
4. **Boundary Testing**: Test edge cases and various configurations

#### Performance Testing
1. **Balance Verification**: Confirm improved workload distribution across cores
2. **Performance Measurement**: Quantify speedup in `test_mla_sdpa_bh_galaxy`
3. **Regression Testing**: Ensure no performance degradation in other cases

#### Test Execution
Run the target test case:
```bash
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa_bh_galaxy -v
```

### 8. Key Design Decisions

- **No new flags**: Always apply core balancing when `is_balanced=true` (existing ring-level flag)
- **Minimal kernel changes**: Reuse existing logic with dual-range processing
- **Backward compatibility**: Fallback to sequential when `is_balanced=false`
- **Head pairing constraint**: Assert even number of heads for proof of concept
- **Remainder handling**: Distribute extra chunks to first cores in each group

This implementation provides a foundation for core-level balancing optimization while maintaining compatibility with existing ring-level balancing and chain optimization features.
