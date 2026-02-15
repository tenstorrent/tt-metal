# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Focus: TT-MoE Infrastructure Debugging

### Latest Update (2026-02-15 15:35)
✅ **SHAREDEXPERT FIX COMPLETE - ALL TESTS PASSING**

**Problem Solved:**
- Fixed SharedExpert numerical explosion (std was 1.85×10¹¹, now normalized)
- Root cause: Float8 weight dequantization failure in SharedExpert module
- Solution: Refactored to match reference architecture with proper 3D tensors and weight replication

**Fix Applied:**
1. Refactored `models/tt-moe/components/experts/shared_expert.py` to match reference
2. Added missing `weight_scale_inv` keys in `moe_block.py`
3. Uses `ReplicateTensorToMesh` for SharedExpert (weights replicated, not sharded)
4. Properly handles Float8 → Float32 → BFloat16 conversion flow

**Results:**
- **PCC: 0.989** (exceeds required 0.98) ✅
- All tests passing
- Debug code removed and cleaned up

**Commits:**
- `7265229fa2` - Debugging status tracking
- `527c33dd21` - Initial TT-MoE infrastructure refactoring
- `ff375843e1` - Removed intermediate tensor debugging code after fix

### Performance Analysis & Optimization Completed (2026-02-15 Evening)

**Work Completed:**
1. **Comprehensive Memory & Implementation Analysis:**
   - Memory placement (L1 vs DRAM) verified consistent between implementations
   - Batch processing mode already configurable via `replicated_batch_mode` flag
   - Decode path differences identified and minor fixes applied

2. **CCL Hyperparameter Sweep Infrastructure Created:**
   - `test_deepseek_ag_hyperparameter_sweep_perf.py` - All-gather performance testing
   - `test_deepseek_rs_hyperparameter_sweep_perf.py` - Reduce-scatter performance testing
   - `sweep_deepseek_ag_hyperparameters.py` - All-gather configuration sweep
   - `sweep_deepseek_rs_hyperparameters.py` - Reduce-scatter configuration sweep
   - Tests DeepSeek-V3 specific shapes for decode and prefill modes
   - Sweeps num_links (1,2,4), topology (Linear/Ring), chunks_per_sync, workers_per_link

3. **Fixes Applied:**
   - Fixed weight repeat dimensions default in `moe_block.py`
   - Verified score correction bias configuration (already correct)
   - Verified cluster axis configuration (already correct)
   - Verified memory configurations (already optimal)

4. **Key Findings:**
   - Infrastructure supports both batch modes (replicated vs distributed)
   - Memory placement strategy is already optimal
   - CCL operations ready for hyperparameter optimization
   - No major architectural issues remaining

**Current Status:**
- ✅ All tests passing with PCC 0.989 (exceeds 0.98 requirement)
- ✅ Infrastructure stable and matches reference implementation
- ✅ Performance sweep tests ready for optimization analysis

### Current Task: GPT-OSS Integration Planning (2026-02-15 Evening)

**Objective:** Extend TT-MoE infrastructure to support GPT-OSS models through JSON configuration

**Plan Summary:**
1. Create `gpt_oss.json` configuration file for GPT-OSS parameters
2. Implement TopKRouter support (GPT-OSS uses different router than DeepSeek)
3. Adapt existing infrastructure to handle GPT-OSS differences:
   - No shared experts (unlike DeepSeek-V3)
   - Different dimensions (128 experts, 4 per token, 2048 hidden, 360 intermediate)
   - Ring topology for all-to-all operations
4. Create comprehensive tests to validate against GPT-OSS reference

**Key Files to Create:**
- `models/tt-moe/configs/gpt_oss.json` - Configuration
- `models/tt-moe/components/routers/topk_router.py` - Router implementation
- `models/tt-moe/tests/test_gpt_oss_moe_block.py` - Tests

**Status:** Planning phase complete, ready for implementation

### GPT-OSS Integration Status (2026-02-15 19:00)

**Partial Implementation Complete:**
- ✅ Created `gpt_oss.json` configuration file
- ✅ Implemented `TopKRouter` component for GPT-OSS routing (verified correct)
- ✅ Updated `moe_block.py` to support multiple router types
- ✅ Created test suite in `test_gpt_oss_moe_block.py`
- ✅ Configuration loading test passes
- ✅ GPT-OSS-120B weights organized at `/data/MLPerf/huggingface/hub/models--openai--gpt-oss-120b/`

**🔴 CRITICAL ISSUE FOUND:**
- ❌ **Missing ThroughputExperts Implementation** - GPT-OSS requires fundamentally different expert architecture
- Current implementation uses DeepSeek's `DistributedExpert` which lacks:
  - Sparse matmul operations (`ttnn.sparse_matmul`)
  - MoE expert token remap (`ttnn.moe_expert_token_remap`)
  - Proper SwiGLU with clamping: `(up + 1) * (gate * sigmoid(gate * alpha))`
  - All-reduce operation after combine
- Without ThroughputExperts, GPT-OSS **cannot function correctly**

**Files Created/Modified:**
- `models/tt-moe/configs/gpt_oss.json` - GPT-OSS configuration
- `models/tt-moe/components/routers/topk_router.py` - TopK router implementation
- `models/tt-moe/tests/test_gpt_oss_moe_block.py` - Test suite
- `models/tt-moe/moe_block.py` - Updated to support TopKRouter
- `models/tt-moe/GPT_OSS_IMPLEMENTATION.md` - Documentation

**Key Features:**
- Supports 128 experts (4 per device on 32-device Galaxy)
- TopK routing with 4 experts per token
- Ring topology for all-to-all operations
- No shared experts (GPT-OSS specific)
- Throughput experts mode for performance

**Testing:**
```bash
# Test configuration loading (works without hardware)
pytest models/tt-moe/tests/test_gpt_oss_moe_block.py::test_gpt_oss_config_loading -xvs

# Run full test suite (requires hardware)
bash /tmp/test_gpt_oss_integration.sh
```

### Next Steps: Performance Optimization & Multi-Model Support

1. **Validate GPT-OSS with Real Weights**:
   - Test with actual GPT-OSS model weights
   - Ensure PCC >= 0.98 against reference implementation
   - Profile performance vs native GPT-OSS

2. **Run CCL Hyperparameter Sweeps** (for both DeepSeek and GPT-OSS):
   ```bash
   pytest models/tt-moe/tests/test_deepseek_ag_hyperparameter_sweep_perf.py -xvs
   pytest models/tt-moe/tests/test_deepseek_rs_hyperparameter_sweep_perf.py -xvs
   ```
   - Analyze CSV results for optimal parameters
   - Update configurations with best settings

3. **Integration & Deployment:**
   - Support multiple models through JSON configurations
   - Create unified test suite for all supported architectures
   - Document configuration format for new models

### Quick Start - Test Commands
```bash
# Setup environment first
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Run working reference test (PASSES)
pytest models/demos/deepseek_v3/tests/test_moe_experts.py::test_forward_pass[decode-128-random-model.layers.3.mlp.experts.0-255] -xvvs

# Run distributed expert test (PASSES with PCC: 0.998415)
pytest models/tt-moe/tests/test_moe_components.py::test_08_distributed_expert_with_reference_comparison -xvvs

# Run infrastructure test (NOW PASSES with PCC: 0.989)
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
```

### Performance Test Commands
```bash
# Performance Analysis Commands
# Setup environment first (see Quick Start above)

# Run CCL hyperparameter sweeps
pytest models/tt-moe/tests/test_deepseek_ag_hyperparameter_sweep_perf.py -xvs
pytest models/tt-moe/tests/test_deepseek_rs_hyperparameter_sweep_perf.py -xvs

# Test specific configurations
pytest models/tt-moe/tests/sweep_deepseek_ag_hyperparameters.py -k "decode" -xvs  # Decode mode only
pytest models/tt-moe/tests/sweep_deepseek_rs_hyperparameters.py -k "4L" -xvs     # 4 links only

# Test distributed batch mode (memory efficient)
# Set "replicated_batch_mode": false in deepseek_v3.json, then run:
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
```

### Overview
The TT-MoE infrastructure is a generalized implementation for creating Mixture of Experts (MoE) blocks that can be configured via JSON for different architectures (DeepSeek-V3, GPT-OSS, etc.).

### Current Status (2026-02-10)
**Fixes Applied:**
1. ✅ **Fabric Initialization**: Added `ttnn.set_fabric_config()` before opening mesh device to enable CCL operations
2. ✅ **Compute Config**: Set `packer_l1_acc=True` in WormholeComputeKernelConfig for LoFi math
3. ✅ **Expert Distribution**: Correctly using 8 experts per device (256 total experts / 32 devices)
4. ✅ **Tensor Shapes**: Input [1, 8, 32, 7168] now matches weights [1, 8, 7168, 2048]
5. ✅ **Weight Loading**: Replaced all `ttnn.as_tensor` calls with `ttnn.from_torch` using `ReplicateTensorToMesh`

**Latest Fix Attempt (2026-02-10 18:35):**
- **Root Cause Hypothesis**: `ttnn.as_tensor` doesn't properly initialize tensor metadata needed by matmul kernel
- **Solution Applied**: Replaced all `ttnn.as_tensor` with `ttnn.from_torch` + `ReplicateTensorToMesh` throughout:
  - `distributed_expert.py`: Fixed `_shard_and_convert_weights` method
  - `shared_expert.py`: Fixed weight loading in `load_weights`
  - `moe_gate.py`: Fixed all 5 tensor creations
- **Result**: ❌ STILL FAILING with same error "incompatible values 5 and 4"

**Current Status (2026-02-12):**

### Major Breakthrough (2026-02-13):
- ✅ **CRITICAL FIX APPLIED**: Resolved reshape volume mismatch that was causing infrastructure to crash
  - **Bug**: After all_to_all_dispatch, was using `batch_size_per_device` instead of full `batch_size`
  - **Fix Location**: `moe_block.py` lines 849-856 in `_forward_moe_with_intermediates` method
  - **Fix Details**: Changed reshape from `(1, 1, batch_size_per_device * seq_len, hidden_size)` to `(1, 1, batch_size * seq_len, hidden_size)`
- ✅ **Infrastructure now runs successfully** without crashes through full pipeline
- ⚠️ **PCC Gap Still Exists**: Current PCC is **0.829** vs required **0.98** (gap of 0.151)
- ✅ Successfully capturing and analyzing intermediate tensors for debugging

### Latest Debugging Status (2026-02-13):
- Created comprehensive intermediate tensor comparison script (`compare_reference_vs_infrastructure.py`)
- Successfully captured 21 intermediate tensors from infrastructure implementation
- Next step: Fix comparison script imports to identify exact divergence point between implementations

### Component Tests Status
- ✅ **SUCCESS**: test_08_distributed_expert_with_reference_comparison **PASSES with PCC: 0.998415**
- **Root Cause Fixed**: Incorrect `weight_block_size` was causing dequantization failure
  - Was using: `[1, 1]` (incorrect for DeepSeek-V3)
  - Now using: `[128, 128]` (matches actual model configuration)

### Full Infrastructure Test Status (Latest Run: 2026-02-15 14:52)
- ✅ **TEST PASSES**: test_deepseek_moe_block achieves **0.989 PCC** (exceeds required 0.98)
- **Test Execution**: Full test completes successfully
- **Weight Loading**: Confirmed all weights loaded including Float8 quantization scales
- **SharedExpert Fix**: Resolved numerical explosion by matching reference architecture
- **Graph Capture**: Successfully captured to `/tmp/ttnn_infra_test_graph_rqrkm2bc/infra_test_captured_operations.json`

### Investigation Findings:
1. ✅ **Weight Loading**: Both implementations load identical weights (1544 for layer 3)
2. ✅ **Weight Quantization**: Confirmed Float8_e4m3fn format with weight_block_size [128, 128]
3. ✅ **Module Flow**: Correct execution sequence: all_gather → router → MoE → SharedExpert → add → reduce_scatter
4. ✅ **Tensor Shapes**: All intermediate shapes match expected values
5. ✅ **Shared Expert Config**: Using correct moe_intermediate_size (2048)

### Debug Instrumentation Added:
- **MoEBlock**: Added comprehensive logging at 11 key stages
- **Reference Models**: Instrumented MoE and MoEDecoderBlock2D
- **Debug Logger**: Created `/tmp/moe_debug_logger.py` for tensor comparison
- **Analysis Scripts**: Multiple comparison scripts in `/tmp/`

### Potential Root Causes for PCC Gap (Still Under Investigation):
1. **Numerical Precision in Operations**:
   - Possible differences in how ttnn.linear handles Float8 dequantized weights
   - BFloat16 precision loss during collective operations

2. **Router Score Normalization**:
   - Check if softmax implementation differs
   - Verify score_correction_bias application

3. **Expert Weight Preparation**:
   - `_prepare_expert_weights` permutation logic
   - Repeat operations for weight broadcasting

4. **All-to-All Operations**:
   - Numerical precision during dispatch/combine
   - Potential reordering of operations affecting accumulation

5. **Activation Functions**:
   - SwiGLU (silu activation) implementation differences
   - Order of gate_proj * up_proj operations

- **Major Rework Completed**: Rewrote distributed_expert.py as direct port of reference implementation
- **Key Changes Made**:
  1. **Complete Rework of DistributedExpert**:
     - Created new self-contained implementation in `models/tt-moe/components/experts/distributed_expert.py`
     - Direct port of `models/demos/deepseek_v3/tt/experts.py` (working reference)
     - All utilities copied directly into file (no external dependencies)
     - Changed to use @classmethod for all methods (matching TTExperts interface exactly)
     - Uses exact same weight loading and sharding logic as reference
     - Proper sharding with `ShardTensorToMesh(dim=1)` to distribute 256 experts across 32 devices (8 per device)
     - Added `convert_weights` method for proper weight loading
     - Weights passed via config dict, not stored as instance variables
     - Fixed dequantization to handle Float8 weights correctly

  2. **Test Status**:
     - `test_08_distributed_expert_with_reference_comparison`: ✅ **PASSES** (PCC: 0.998415)
     - Weights properly dequantized: Float8 → Float32 → BFloat16
     - Output matches reference model with high accuracy

**Key Implementation Differences (Old vs New DistributedExpert)**:
| Aspect | Old Implementation | New Implementation (2026-02-11) |
|--------|-------------------|----------------------------------|
| **Base Design** | Inherits from BaseExpert | Self-contained, direct port of reference |
| **Weight Loading** | Complex with multiple paths | Simple, matches reference exactly |
| **Weight Sharding** | Various attempts at sharding | `ShardTensorToMesh(dim=1)` only |
| **Forward Pass** | Extra parameters (indices, weights) | Clean interface like reference |
| **Memory Config** | Complex inheritance | Simple mode-based (L1/DRAM) |
| **Utilities** | External dependencies | All utilities copied locally |
| **Testing** | Part of larger test | Dedicated unit tests |

### Work Completed (2026-02-12):

**Session 1 (Earlier):**

1. **Major Refactoring**:
   - Complete rewrite of DistributedExpert as class-based API
   - Complete rewrite of SharedExpert matching reference
   - Updated MoEBlock to use class-based components
   - Fixed Float8 dequantization (weight_block_size [128,128])

2. **Test Infrastructure**:
   - Created comprehensive test suite in test_moe_components.py
   - test_08 passes with PCC 0.998415 for DistributedExpert alone
   - Full integration test runs but PCC is 0.8158 (needs improvement)

3. **Debug Infrastructure**:
   - Added comprehensive instrumentation to both implementations
   - Created multiple debug and comparison scripts
   - Captured operation graphs for analysis

4. **Configuration**:
   - Fixed deepseek_v3.json configuration values
   - Created CLAUDE.md documentation for tracking progress

### Key Fix Applied - Reshape Bug (2026-02-13)

**Critical Bug Fixed**: The infrastructure was crashing due to incorrect tensor volume after all_to_all_dispatch.

**Root Cause**:
- After all_to_all_dispatch, each device gets ALL tokens for its subset of experts
- Infrastructure was incorrectly using `batch_size_per_device` in reshape instead of full `batch_size`
- This caused a volume mismatch: trying to reshape 32×1×7168 elements to 8×1×7168 shape

**Fix Applied**:
```python
# BEFORE (incorrect):
dispatch_chunk = ttnn.reshape(dispatch_output, shape=(1, 1, batch_size_per_device * seq_len, hidden_size))

# AFTER (correct):
dispatch_chunk = ttnn.reshape(dispatch_output, shape=(1, 1, batch_size * seq_len, hidden_size))
```

**Result**: Infrastructure now runs successfully without crashes!

### Systematic Debugging Plan for PCC Issue (2026-02-12) - ✅ IMPLEMENTED

#### Approach: Side-by-Side Intermediate Tensor Comparison

**Status: COMPLETE** - Test created and ready to run

1. **✅ Single Test File with Two Functions**:
   - Created: `models/tt-moe/tests/test_intermediate_comparison.py`
   - Function 1: `run_our_implementation()` - uses our `moe_block.py`
   - Function 2: `run_reference_implementation()` - uses reference `moe_decoder_block_2d.py`
   - Both functions return intermediate tensors at key points

2. **✅ Intermediate Tensors Captured**:
   - **All-gather output** (if tensor parallel is enabled)
   - **MoE gate/router output** (weights and indices)
   - **Dispatch output** (after all_to_all_dispatch)
   - **Routed expert output** (after distributed experts forward)
   - **Shared expert output** (parallel computation)
   - **Final combined output** (after adding MoE + shared)
   - **Reduce-scatter output** (if tensor parallel is enabled)

3. **✅ Modifications Completed**:
   - **moe_block.py**: Added `forward_with_intermediates()` method
   - **modified_reference_decoder.py**: Created wrapper for reference to capture intermediates
   - Both return tensors at exact same logical points

4. **✅ Test Implementation**:
   - Pytest-based test with parametrization
   - Loads real DeepSeek weights (layer 3)
   - Runs both functions with identical input
   - Compares each intermediate tensor:
     * Shape verification
     * PCC calculation for each stage
     * Identifies first point of divergence

5. **Run the Test**:
   ```bash
   # Quick run script
   bash /tmp/run_intermediate_comparison_test.sh

   # Or run directly
   cd /home/ntarafdar/tt-moe/tt-metal
   source python_env/bin/activate
   pytest models/tt-moe/tests/test_intermediate_comparison.py::test_intermediate_comparison -xvs
   ```

6. **Expected Outcomes**:
   - Identify exact operation where numerical divergence begins
   - Quantify PCC at each stage to understand degradation pattern
   - Pinpoint specific tensor operations causing accuracy loss
   - Test will show where PCC drops below 0.98 threshold

#### Implementation Files:
- **Main Test**: `/home/ntarafdar/tt-moe/tt-metal/models/tt-moe/tests/test_intermediate_comparison.py`
- **Helper Module**: `/home/ntarafdar/tt-moe/tt-metal/models/tt-moe/tests/modified_reference_decoder.py`
- **Modified moe_block.py**: Added `forward_with_intermediates()` method to capture intermediates
- **Run Script**: `/tmp/run_intermediate_comparison_test.sh` - Quick script to run the test

#### Success Criteria:
- Find stage where PCC drops below 0.98
- Identify specific operation causing numerical difference
- Have actionable fix to improve accuracy

### Current Debugging Approach (2026-02-14) - UPDATED

#### ✅ COMPLETED: Standardized Intermediate Tensor Capture

**Objective**: Identify exact point of numerical divergence between infrastructure and reference

**Status**: Infrastructure runs successfully but with PCC gap (0.829 vs required 0.98)

### Intermediate Tensor Comparison Plan

**Goal**: Capture and compare exact same intermediate tensors from both implementations to find divergence point.

#### Tensors to Capture (exactly these 6):
1. **Input Tensor** - The TP=8 input before all-gather
2. **All-Gather Output** - Replicated tensor across row (8 devices)
3. **MoE Gate Input** - Tensor being fed into moe_gate
4. **MoE Module Output** - After combine operation
5. **Shared Expert Output** - Parallel computation result
6. **Reduce Scatter Output** - Final output tensor

#### Implementation Files:
- **Reference**: `models/tt-moe/tests/modified_reference_decoder.py`
  - Already modified to store intermediates
  - Uses combined moe_gate + moe_module in one module

- **Infrastructure**: `models/tt-moe/moe_block.py`
  - Has `forward_with_intermediates()` method
  - Separate moe_gate and moe_module components

#### Data Flow:
```
TP=8 Input → All-Gather → Replicated Tensor → [MoE Gate → MoE Module] + [Shared Expert] → Add → Reduce Scatter → Output
```

#### Action Items:
1. ✅ Verify both implementations save exactly the 6 tensors above - COMPLETED
   - Updated `moe_block.py` to save exactly 6 tensors with consistent names
   - Updated `modified_reference_decoder.py` to save same 6 tensors with same names
2. ✅ Remove any extra intermediate tensors being saved - COMPLETED
   - Removed redundant `router_input`, `combined_output` from infrastructure
   - Removed `mlp_input`, `mlp_output` from reference, renamed to match
3. ✅ Create test that dumps intermediates to files - COMPLETED
   - Created `/home/ntarafdar/tt-moe/tt-metal/save_and_compare_intermediates.py`
   - Saves to `/tmp/our_intermediates/` and `/tmp/reference_intermediates/`
4. ✅ Run both implementations and save intermediates - READY TO RUN
5. ✅ Compare tensor values at each stage to find divergence - READY TO RUN

#### Run the Comparison Script:
```bash
# Setup environment
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52

# Run the script to save and compare intermediates
python save_and_compare_intermediates.py

# Check saved tensors manually if needed
ls -la /tmp/our_intermediates/
ls -la /tmp/reference_intermediates/

# Compare specific tensors with numpy
python -c "
import numpy as np
our = np.load('/tmp/our_intermediates/moe_gate_input.npy')
ref = np.load('/tmp/reference_intermediates/moe_gate_input.npy')
print(f'Shape match: {our.shape == ref.shape}')
print(f'Our shape: {our.shape}')
print(f'Ref shape: {ref.shape}')
"
```

**Key Scripts Created**:
- `/home/ntarafdar/tt-moe/tt-metal/analyze_intermediate_tensors.py` - Analyzes infrastructure tensor flow
- `/home/ntarafdar/tt-moe/tt-metal/compare_reference_vs_infrastructure.py` - Direct comparison tool

**Captured Intermediates** (21 total - TO BE REDUCED TO 6):
- Input/output stages: all_gather, reduce_scatter
- Router outputs: indices, weights, dispatch metadata
- Expert outputs: MoE experts, shared expert
- Combination: combined output after adding MoE + shared

### Systematic Debugging Plan (2026-02-11) - COMPLETED

**Objective**: Compare operation configurations between infrastructure and reference implementations using captured graph traces

#### Phase 1: Operation-by-Operation Configuration Comparison
1. **All-gather operation**
   - Find all_gather operations in both graph traces
   - Compare configurations: cluster_axis, memory_config, topology settings
   - Verify runtime argument population matches

2. **MoE Gate operations**
   - TopK operations - compare k values, dimensions
   - Gather/Scatter operations - check indices handling
   - Score normalization - verify computation steps match

3. **Weight preparation**
   - Repeat operations - compare repeat_dims configurations
   - Permute operations - verify dimension ordering
   - Layout conversions (ROW_MAJOR ↔ TILE)

4. **All-to-all dispatch (where hang occurs)**
   - Compare cluster_axis settings
   - Check expert_mapping_tensors configuration
   - Verify memory_config settings match

#### Phase 2: Tensor Configuration Analysis
For each operation, compare:
- Input tensor shapes
- Memory configurations (L1 vs DRAM)
- Layout (TILE vs ROW_MAJOR)
- Data types (bfloat16 consistency)
- Sharding configurations

#### Phase 3: CCL Configuration Validation
- All-gather runtime arguments
- Reduce-scatter configurations
- Semaphore handling
- Topology settings (Linear vs Ring)

#### Implementation Strategy:
1. Create a script to parse both JSON graph traces
2. Extract operation configurations by node type
3. Build a comparison table for each operation type
4. Identify configuration mismatches
5. Test fixes for identified differences

#### Key Areas to Focus:
- Memory configuration transitions
- Tensor sharding vs replication strategies
- CCL collective operation parameters
- Program configs for matmul operations

### Key Findings (2026-02-10 18:00)

1. **Working test (test_moe_experts.py) passes successfully:**
   - Uses 128 tokens in decode mode
   - Input shape: [1, 8, 128, 7168]
   - Uses `ttnn.ReplicateTensorToMesh` to replicate input across devices
   - No block configuration error

2. **Our test fails with various batch sizes:**
   - Tried 32, 64, 128 tokens - all fail with same block config error
   - Using `ttnn.ShardTensor2dMesh` for input distribution
   - Same tensor shapes and memory configs as working test

3. **Key difference identified:**
   - Working: Uses `ttnn.ReplicateTensorToMesh` (replicates tensor to all devices)
   - Ours: Uses `ttnn.ShardTensor2dMesh` (shards tensor across devices)
   - When we try ReplicateTensorToMesh, we get OOM in L1 after all_to_all_dispatch

### Investigation Results (2026-02-10 18:25)

**Working test_decoder_block.py with MoE:**
- Initial batch: 32 (USERS_PER_ROW)
- After all_to_all: 128 tokens
- Expert input: [1, 8, 128, 7168] in L1
- Weights: [1, 8, 7168, 2048] in DRAM
- ✅ Passes without block config error

**Our infrastructure test with identical shapes:**
- Same shapes and memory configs as working test
- ❌ Still fails with "incompatible values 5 and 4" error

**Isolated tests that WORK:**
1. Simple ttnn.linear with [1, 8, 32, 7168] input - ✅ Works
2. With ttnn.repeat from [1, 1, 32, 7168] to [1, 8, 32, 7168] - ✅ Works
3. With 128 tokens - ✅ Works

**Conclusion:**
- The matmul operation itself works with these dimensions
- The issue was specific to how tensors are created in our infrastructure
- Root cause: `ttnn.as_tensor` doesn't properly initialize tensor metadata
- Working test uses `ttnn.from_torch` with proper mesh_mapper configuration

### Fix Applied (2026-02-10 18:35 - 19:30)
- ✅ **Weight Tensor Creation**: Replaced all `ttnn.as_tensor` calls with `ttnn.from_torch` + `ReplicateTensorToMesh`
  - Files modified: `distributed_expert.py`, `shared_expert.py`, `moe_gate.py`
  - This matches exactly how the working test creates tensors
- ✅ **Linear Operations**: Ensured all `ttnn.linear` calls use:
  - Keyword argument `input_tensor_b` for weights
  - Explicit `program_config=None`
  - Proper memory_config and compute_kernel_config
- ✅ **Compute Config**: Using LoFi with `packer_l1_acc=True`
- ✅ **Fabric Init**: Set before opening mesh device
- ⏳ Test needs to be run to confirm all fixes work together

### PCC Gap Root Cause Analysis and Solution (0.829 → 0.98)

#### Root Cause Identified (2026-02-13)

**Configuration Mismatch Found**:
1. ✅ **FIXED**: `num_experts_per_tok` was 6, should be 8 (like reference)
   - Updated in `/home/ntarafdar/tt-moe/tt-metal/models/tt-moe/configs/deepseek_v3.json`

2. **Batch Parallelization Strategy Difference** (Primary cause of remaining PCC gap):

   **Our Implementation (Distributed Batch)**:
   - Input: `torch_input.permute(1, 0, 2).unsqueeze(0)` → `(1, 1, 32, 7168)`
   - Uses `ShardTensor2dMesh(dims=(-2, -1))` to distribute batch across EP devices
   - Each EP device processes 8 tokens (32 ÷ 4 EP devices)
   - **More efficient**: Better memory utilization, true parallelization

   **Reference Implementation (Replicated Batch)**:
   - Input: Also `torch_input.permute(1, 0, 2).unsqueeze(0)` → `(1, 1, 32, 7168)` (same shape!)
   - Also uses `ShardTensor2dMesh(dims=(-2, -1))` but processes full batch on each device
   - Each device processes ALL 32 tokens (batch replicated, not distributed)
   - **Less efficient**: Higher memory usage, but simpler logic

   **Impact on Router**:
   - Router outputs are COMPLETELY DIFFERENT due to different token sets
   - Router weights PCC: ~0.4-0.6 (should be 1.0 for identical implementations)
   - Router indices PCC: ~0.5-0.7 (should be 1.0)
   - This cascades through the entire MoE pipeline causing final PCC of 0.829

#### Option A: Match Reference Implementation (Requires Architectural Change)

**Why Current Approach Doesn't Match**:
- Both implementations use same shape `(1, 1, 32, 7168)` and same sharding dims `(-2, -1)`
- BUT: Reference processes **full batch on each device** (replicated), we distribute it
- This fundamental difference causes router outputs to be completely different
- Router sees different token sets → different expert assignments → cascading differences

**Real Solution Would Require**:
1. **Change all-to-all dispatch behavior**:
   - Need to process full batch on each EP device (not distribute it)
   - Would require changes to how `batch_size_per_device` is calculated
   - Major architectural change to the MoE block internals

2. **Memory implications**:
   - 4x memory usage (each device processes 32 tokens instead of 8)
   - Less efficient but would match reference exactly

3. **Current Status**:
   - Configuration fixed (num_experts_per_tok: 6 → 8) ✅
   - PCC improved slightly (0.829 → 0.824)
   - Remaining gap due to architectural difference

#### Option B: Keep Efficient Design (Future Work)

**Alternative**: Keep our efficient distributed batch design but acknowledge it's architecturally different from the reference. This would require:
- Adjusting PCC comparison methodology
- Creating separate reference tests for distributed batch mode
- Documenting that our implementation is an optimization over the reference

**Status**: User chose Option A to match reference first. Option B can be explored later as an optimization.

### What Was Actually Implemented (2026-02-13)

#### Configuration Fix Applied
✅ **Fixed `num_experts_per_tok` from 6 to 8** in `/home/ntarafdar/tt-moe/tt-metal/models/tt-moe/configs/deepseek_v3.json`

#### Attempted Shape Changes (Reverted)
- Initially tried changing input shape from `(1, 1, 32, 7168)` to `(1, 32, 1, 7168)`
- Discovered reference actually uses same shape as us: `(1, 1, 32, 7168)`
- Reverted shape detection logic in MoEBlock

#### Current Results
- PCC: **0.824** (required: 0.98)
- Test runs successfully without crashes
- Configuration now matches reference (8 experts per token)
- Remaining gap due to batch distribution architectural difference

### Next Steps for Full Fix (Option A - Match Reference)

#### Step 1: Update Test Input Preparation
In `models/tt-moe/tests/test_deepseek_moe_block.py`:

```python
# Line ~75-85: Update the input tensor preparation
tt_input_tensor = ttnn.from_torch(
    torch_input.unsqueeze(0),  # Change from permute(1,0,2).unsqueeze(0)
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3)),  # Change from dims=(-2,-1)
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    layout=ttnn.TILE_LAYOUT,
)
```

#### Step 2: Update MoEBlock Forward Method
In `models/tt-moe/moe_block.py`:

1. **Remove batch permutation logic** (lines ~800-810):
   ```python
   # Remove or comment out:
   # if self.ep_size > 1:
   #     batch_size_per_device = batch_size // self.ep_size
   #     x = ttnn.reshape(x, shape=(1, batch_size_per_device, seq_len, hidden_size))
   ```

2. **Update all-gather input handling** to work with replicated batch format

3. **Adjust router input shape expectations**

#### Step 3: Verify Router Configuration
Ensure `num_experts_per_tok = 8` in the config (already fixed).

#### Step 4: Run Tests
```bash
# Test should now achieve PCC >= 0.98
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
```

### Next Steps to Complete Fix
1. **Run the main test** to see if our fixes work:
   ```bash
   pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
   ```

2. **If still failing**, run the debug test scripts to isolate the issue:
   ```bash
   # These scripts test different aspects of the pipeline
   python /tmp/test_all_to_all_simulation.py  # Most comprehensive
   ```

3. **If tests reveal new issues**, potential fixes:
   - Check memory configuration transitions
   - Verify tensor properties after all_to_all_dispatch
   - Compare with working test's exact tensor flow

### Attempted Fixes That Didn't Work

1. **Removing compute_kernel_config**: Still get same "5 and 4" error even with defaults

### Next Steps to Try

1. **Specify explicit program config for linear ops**:
   ```python
   # In distributed_expert.py forward()
   program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
       compute_with_storage_grid_size=(8, 8),  # Use 8x8 instead of 8x9
       in0_block_w=4,
       out_subblock_h=1,
       out_subblock_w=4,
       per_core_M=1,
       per_core_N=4,
       fused_activation=None,
   )
   w1_out = ttnn.linear(x, self.w1_experts, program_config=program_config, ...)
   ```

2. **Use a different matmul variant**:
   - Try `ttnn.matmul` instead of `ttnn.linear`
   - Or adjust tensor padding to make dimensions more compatible

3. **Check if working version has specific configs**:
   - Look for any MatmulProgramConfig usage in deepseek_v3
   - Check if there are dimension-specific workarounds

### Test Configuration Details

#### Working Reference Test Parameters
- **Mode**: `decode` (single token generation)
- **Batch Size**: 32 (USERS_PER_ROW)
- **Tokens**: 128 (after all_to_all expansion)
- **Hidden Size**: 7168
- **Intermediate Size**: 2048
- **Experts**: 256 total, 8 per device, 6 selected per token
- **Weight Type**: `random` or `real` (from HF model)

#### Infrastructure Test Parameters
- **Mode**: `decode` (matching reference)
- **Batch Size**: 32 (matching reference)
- **Config File**: `models/tt-moe/configs/deepseek_v3.json`
- **PCC Required**: 0.98 (Pearson Correlation Coefficient threshold)

### Summary of Today's Work (2026-02-11 to 2026-02-12)

**What Was Accomplished**:
1. ✅ Analyzed the working reference implementation (`test_decoder_block.py` and `test_moe_experts.py`)
2. ✅ Deep dive into `models/demos/deepseek_v3/tt/experts.py` to understand the exact implementation
3. ✅ Completely rewrote `distributed_expert.py` as an exact port of the reference `Experts` class
4. ✅ Changed from instance-based to class-method based implementation (matching reference exactly)
5. ✅ Created comprehensive unit tests that match the reference test pattern exactly
6. ✅ Fixed all interface mismatches between our implementation and reference

**Key Insights Gained**:
- The reference uses all class methods (@classmethod), not instance methods
- Weights are passed through the config dict, not stored as instance variables
- The exact sharding pattern is `ShardTensorToMesh(dim=1)` for expert distribution
- Output concatenation uses `ConcatMesh2dToTensor(dims=(0,1))` not `(-2,-1)`
- The test pattern requires specific fixtures and exact parameter matching

### Files Modified (2026-02-11/12 Update)

**Completely Rewritten**:
- `models/tt-moe/components/experts/distributed_expert.py` - Now a direct port of reference implementation

**Tests Added**:
- `models/tt-moe/tests/test_moe_components.py` - Added tests 07 and 08 for distributed expert

### Key Files and Components

#### Working Reference Implementation (Hand-written)
- `models/demos/deepseek_v3/tests/test_decoder_block.py` - Working hand-written version
- `models/demos/deepseek_v3/tt/decoder_block/moe_decoder_block_2d.py` - Hand-written MoE implementation

#### TT-MoE Infrastructure (Has Crash)
- `models/tt-moe/tests/test_deepseek_moe_block.py` - Test using infrastructure (CRASHES)
- `models/tt-moe/moe_block.py` - Main MoE block implementation
- `models/tt-moe/configs/deepseek_v3.json` - Configuration for DeepSeek-V3
- `models/tt-moe/components/` - Modular components:
  - `routers/moe_gate.py` - MoE gate router implementation
  - `experts/distributed_expert.py` - Distributed expert implementation
  - `experts/shared_expert.py` - Shared expert implementation
- `models/tt-moe/utils/ccl.py` - CCL utilities for collective operations

### Environment Setup
```bash
# Activate Python environment (REQUIRED)
source python_env/bin/activate

# Set environment variables
export MESH_DEVICE=TG  # TensorGrid configuration (4x8 = 32 devices)
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD

# Set HuggingFace model path (required for tests)
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52

# Optional: Set cache directory for weight caching (speeds up repeated test runs)
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Optional: Enable debug output for expert operations
export DEEPSEEK_V3_DEBUG_EXPERTS=1  # Shows tensor stats during forward pass
```

### Restarting Galaxy
Galaxy should be reset if there is a hang. To reset after a hang:
```bash
source python_env/bin/activate
tt-smi -glx_reset
```

### Common Test Outputs

#### Successful Test (What We Want)
```
Testing DeepSeek MoE Block Against Reference
============================================================
Input shape: torch.Size([32, 1, 7168])
Reference output shape: torch.Size([32, 1, 7168])
Running MoEBlock forward pass...
TT output shape: torch.Size([32, 1, 7168])
✅ Test passed with PCC >= 0.98
```

#### Block Configuration Error (Fixed)
```
TT_FATAL @ matmul.cpp:123: program_config.out_block_w % program_config.out_subblock_w == 0
info: actual values for 'program_config.out_block_w' is '5' and for 'program_config.out_subblock_w' is '4'
```
**Fix**: Replace `ttnn.as_tensor` with `ttnn.from_torch` + `ReplicateTensorToMesh`

#### Fabric Not Initialized Error (Fixed)
```
TT_FATAL @ ccl_runtime.hpp:165: Trying to get un-initialized fabric context
```
**Fix**: Add `ttnn.set_fabric_config()` before opening mesh device

### Running Tests

#### Working Reference Implementations (Hand-written)
These tests pass successfully and serve as the reference for correct behavior:

```bash
# 1. Test MoE experts directly (working reference)
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache  # For weight caching
pytest models/demos/deepseek_v3/tests/test_moe_experts.py -xvvs

# Run specific mode (decode with 128 tokens - the working config)
pytest models/demos/deepseek_v3/tests/test_moe_experts.py::test_forward_pass[decode-128-random-model.layers.3.mlp.experts.0-255] -xvvs

# 2. Test full decoder block with MoE (working reference)
pytest models/demos/deepseek_v3/tests/test_decoder_block.py -xvvs

# Run decode mode specifically
pytest models/demos/deepseek_v3/tests/test_decoder_block.py -k "decode" -xvvs
```

#### TT-MoE Infrastructure Test (Being Fixed)
This uses the generalized infrastructure and should match the reference implementation:

```bash
# Run the infrastructure test
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs

# With debug output for experts (shows tensor stats)
export DEEPSEEK_V3_DEBUG_EXPERTS=1
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
```

## Test Comparison

### Working Reference Tests vs Infrastructure Test

| Aspect | Working Reference | TT-MoE Infrastructure |
|--------|------------------|----------------------|
| **Location** | `models/demos/deepseek_v3/tests/` | `models/tt-moe/tests/` |
| **Implementation** | Hand-written, DeepSeek-specific | Generalized, JSON-configurable |
| **Weight Loading** | Uses cached weights with `get_test_weight_config` | Direct weight loading from state dict |
| **Tensor Creation** | `ttnn.from_torch` with `ReplicateTensorToMesh` | Fixed: now uses same as working |
| **Test Focus** | Individual components (experts, decoder) | Full MoE block with PCC validation |
| **Passing Status** | ✅ All tests pass | ⏳ Testing after fixes |

### Quick Test Commands

```bash
# Compare both implementations side-by-side
# Terminal 1: Run working reference
pytest models/demos/deepseek_v3/tests/test_moe_experts.py::test_forward_pass[decode-128-random-model.layers.3.mlp.experts.0-255] -xvvs

# Terminal 2: Run infrastructure test
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs
```

## Debug Logging Added (2026-02-10 19:30 - 19:45)

### Complete Operation Coverage
All critical operations are now logged in both implementations including:

### Operation Tracking
Added comprehensive logging to track every TTNN operation in both implementations:

**Files with logging added:**
- Reference:
  - `moe.py` - all_to_all_dispatch, reshape, repeat, to_layout
  - `experts.py` - ttnn.linear operations
  - `moe_decoder_block_2d.py` - all_gather, reduce_scatter (prefix "REF:")
- Infrastructure:
  - `moe_block.py` - all_gather, all_to_all_dispatch, reshape, repeat, to_layout, reduce_scatter
  - `distributed_expert.py` - ttnn.linear operations (prefix "INFRA:")

**What's logged:**
- Operation name (ttnn.linear, ttnn.reshape, ttnn.repeat, etc.)
- Input tensor properties (shape, memory config, layout, dtype)
- Operation config (memory_config, compute_kernel_config)
- Output tensor properties

### Run Comparison Test
```bash
# Run both tests with logging
bash /tmp/run_comparison_tests.sh

# Or run individually:
# Reference test
pytest 'models/demos/deepseek_v3/tests/test_decoder_block.py::test_forward_pass[mode_decode_seq_1_batch_32_pos_random-MoEDecoderBlock2D-model.layers.3-3-run_test_forward_pass_decoder2d-device_params0]' -xvs 2>&1 | tee /tmp/ref.log

# Infrastructure test
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs 2>&1 | tee /tmp/infra.log
```

### Analyze Differences
```bash
# Extract operation sequences
grep "OP_START" /tmp/ref.log > /tmp/ref_ops.txt
grep "OP_START" /tmp/infra.log > /tmp/infra_ops.txt

# Compare
diff /tmp/ref_ops.txt /tmp/infra_ops.txt

# Look for tensor property differences
grep "TENSOR" /tmp/ref.log > /tmp/ref_tensors.txt
grep "TENSOR" /tmp/infra.log > /tmp/infra_tensors.txt
```

## Debugging Strategy for Current Issue

### Isolating the Failure Point
To determine exactly where the failure occurs, run the test with these debug outputs:

```python
# In moe_block.py forward():
logger.info(f"DEBUG: Before all_to_all_dispatch")
dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(...)
logger.info(f"DEBUG: After all_to_all_dispatch, output shape: {dispatch_output.shape}")

# In distributed_expert.py forward():
logger.info(f"DEBUG: Before w1 linear, input shape: {x.shape}")
w1_out = ttnn.linear(x, ...)
logger.info(f"DEBUG: After w1 linear")
```

### Test Scripts for Verification
Run these scripts in order to isolate the exact failure point:

```bash
# 1. Test simple linear operation (baseline - should PASS)
python /tmp/test_linear_simple.py

# 2. Test with repeat operation (should PASS)
python /tmp/test_linear_with_repeat.py

# 3. Test weight creation with ttnn.from_torch (our fix - should PASS)
python /tmp/test_weight_creation.py

# 4. Test repeat-then-linear pattern (like our infrastructure)
python /tmp/test_repeat_then_linear.py

# 5. Full all_to_all simulation (most realistic test)
python /tmp/test_all_to_all_simulation.py
```

**What each test validates:**
- **test_linear_simple.py**: Basic matmul with the problem dimensions works
- **test_linear_with_repeat.py**: Repeat operation doesn't break matmul
- **test_weight_creation.py**: Our ttnn.from_torch fix works
- **test_repeat_then_linear.py**: The exact pattern our infrastructure uses
- **test_all_to_all_simulation.py**: Full pipeline with all_to_all_dispatch

### Potential Remaining Issues
1. **All-to-all output tensor properties**: The tensor from all_to_all_dispatch might have different properties than a directly created tensor
2. **Memory configuration transitions**: Moving tensors between L1 and DRAM at the wrong points
3. **Tensor layout issues**: ROW_MAJOR vs TILE_LAYOUT conversions

## Debugging Plan

### Phase 1: Identify the Crash
1. **Run the test and capture full error output**
   - Check for tensor dimension mismatches
   - Look for memory allocation issues
   - Verify device placement errors

2. **Compare with working version**
   - Trace the execution flow in both implementations
   - Identify divergence points
   - Check tensor shapes at each step

### Phase 2: Potential Issue Areas to Investigate

#### 1. Tensor Shape and Layout Issues
- **Input reshaping**: Check if input tensors are correctly reshaped for all-to-all operations
- **Batch dimension handling**: Verify batch_size_per_device calculations
- **Memory layouts**: Ensure correct conversion between TILE_LAYOUT and ROW_MAJOR_LAYOUT

#### 2. All-to-All Operations
- **Expert mapping tensors**: Verify `expert_mapping_tensors` creation
- **Dispatch/Combine metadata**: Check reshape operations for dispatch_metadata
- **Cluster axis configuration**: Validate ep_axis and tp_axis settings

#### 3. Weight Loading and Preparation
- **State dict mapping**: Ensure weights are correctly mapped from HuggingFace format
- **Weight preparation**: Check `_prepare_expert_weights` permutation logic
- **Quantization handling**: Verify weight dequantization if using quantized weights

#### 4. Memory Configuration
- **L1 vs DRAM**: Check memory_config settings for decode mode
- **Chunking parameters**: Verify chunk_size calculations don't cause OOM

#### 5. CCL and Collective Operations
- **CCL initialization**: Ensure CCL is properly initialized with mesh_device
- **All-gather operations**: Check if tensor parallel operations are correctly configured
- **Semaphore handling**: Verify CCL runtime arguments are properly populated

### Phase 3: Debugging Strategies

1. **Add detailed logging**
   ```python
   from loguru import logger
   logger.debug(f"Tensor shape at step X: {tensor.shape}")
   ```

2. **Insert shape assertions**
   ```python
   assert x.shape == expected_shape, f"Shape mismatch: {x.shape} != {expected_shape}"
   ```

3. **Compare intermediate outputs**
   - Save intermediate tensors from working version
   - Compare with infrastructure version at each step

4. **Simplify test case**
   - Start with smaller batch sizes
   - Disable shared expert initially
   - Test without tensor parallelism

### Phase 4: Common Fixes

1. **Shape mismatches**
   - Adjust reshape operations to match expected dimensions
   - Fix permute operations in weight preparation

2. **Memory issues**
   - Switch problematic operations from L1 to DRAM
   - Reduce chunk sizes for prefill mode

3. **Device placement**
   - Ensure all tensors are on correct device
   - Check mesh_mapper configurations

4. **Weight loading**
   - Verify state_dict key mappings
   - Check weight tensor dimensions after loading

### Testing Approach

1. **Unit test individual components**
   ```bash
   # Test router separately
   pytest models/tt-moe/components/routers/test_moe_gate.py -xvvs

   # Test experts separately
   pytest models/tt-moe/components/experts/test_distributed_expert.py -xvvs
   ```

2. **Use smaller configurations**
   - Reduce num_experts_per_tok
   - Use smaller batch sizes
   - Simplify the model configuration

3. **Compare with reference**
   - Use PCC (Pearson Correlation Coefficient) checks
   - Required threshold: 0.98

### Success Criteria
- Test passes without crashes
- PCC >= 0.98 compared to reference implementation
- Performance comparable to hand-written version

### Notes on Architecture
- **MESH_DEVICE=TG**: TensorGrid configuration with shape (4, 8)
- **Tensor Parallel**: Enabled on axis 1
- **Expert Parallel**: Distributed on axis 0
- **DeepSeek-V3 specifics**:
  - 256 total experts
  - 6 experts per token
  - 8 experts per device
  - Shared expert runs in parallel with MoE

### Quick Reference Commands
```bash
# Build if needed
./build_metal.sh

# Run with debugging
PYTHONPATH=$PWD python -m pdb models/tt-moe/tests/test_deepseek_moe_block.py

# Run with specific logging
LOGURU_LEVEL=DEBUG pytest models/tt-moe/tests/test_deepseek_moe_block.py -xvvs

# Check tensor operations
python -c "import ttnn; print(ttnn.__version__)"

# Run isolated tensor creation tests (for debugging)
python /tmp/test_linear_simple.py    # Test simple linear operation
python /tmp/test_linear_with_repeat.py  # Test with repeat operation

# List available tests
pytest models/tt-moe/tests/ --collect-only
pytest models/demos/deepseek_v3/tests/ --collect-only

# Run with captured output (useful for debugging crashes)
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs 2>&1 | tee test_output.log
```
