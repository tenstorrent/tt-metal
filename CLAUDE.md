# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TT-MoE Infrastructure Documentation

### Current Status (2026-02-16)
- ✅ **DeepSeek-V3**: Fully working (PCC: 0.989, exceeds 0.98 requirement)
  - All tests passing with simplified implementation
  - SharedExpert fix complete - resolved numerical explosion
  - CCL hyperparameter sweep tests ready for optimization
- ✅ **GPT-OSS**: Basic implementation WORKING (2026-02-16)
  - ✅ TopKRouter: Working
  - ✅ All-to-All: Working with Linear topology (Ring has routing issues)
  - ✅ ThroughputExperts: Sparse matmul implementation PASSING unit tests
    - ✅ Correct dimensions (hidden=2880, intermediate=2880)
    - ✅ Program configs from reference (cores=(5,9), in0_block_w=10)
    - ✅ Sparse matmul operations working
    - ✅ Mesh tensor conversion handled properly
  - ⚠️ Next: Full model test with PCC validation needed
  - ✅ Reference implementation at `/models/demos/gpt_oss/tt/experts_throughput/decode.py`
  - ✅ Real weights available at `/data/MLPerf/huggingface/hub/models--openai--gpt-oss-120b/`
- ✅ **Infrastructure**: Simplified and consolidated
  - Single moe_block.py implementation (replaced complex version)
  - Clean JSON configuration format
  - Supports both replicated and distributed batch modes
- ✅ **Topology**: Both models using Linear (Ring topology causes fabric routing errors)

### Latest Implementation (2026-02-15 Evening)

**Simplified Infrastructure Complete:**
- Consolidated to single simplified moe_block.py
- Replaced verbose configs with minimal JSON
- Maintained DeepSeek-V3 PCC: 0.9885 (exceeds 0.98 requirement)
- Added GPT-OSS TopKRouter support
- All tests passing with simplified code

**Key Files:**
- `models/tt-moe/moe_block.py` - Single simplified implementation
- `models/tt-moe/configs/deepseek_v3.json` - Clean minimal configuration
- `models/tt-moe/configs/gpt_oss.json` - GPT-OSS configuration (needs ThroughputExperts)
- `models/tt-moe/components/routers/topk_router.py` - TopK router for GPT-OSS
- `models/tt-moe/components/experts/throughput_expert.py` - Needs implementation
- `models/tt-moe/tests/test_deepseek_moe_block.py` - DeepSeek tests
- `models/tt-moe/tests/test_gpt_oss_moe_block.py` - GPT-OSS tests
- `models/tt-moe/tests/test_moe_components.py` - Component unit tests

**GPT-OSS Integration Status:**
- Configuration supports 128 experts (4 per device on 32-device Galaxy)
- TopK routing with 4 experts per token
- Linear topology for all-to-all operations (Ring causes errors)
- No shared experts (GPT-OSS specific)
- **CRITICAL**: GPT-OSS uses intermediate_size=2880 (same as hidden_size), NOT 360
- Sparse matmul configs from reference:
  - Core grids: (5, 9) for both gate_up and down projections
  - in0_block_w=10 (divides K tiles = 2880/32 = 90 evenly)
  - Sparsity block size: 32
- Current implementation status:
  - Sparse matmul operations implemented
  - Buffer/sharding issues with mesh device being debugged
- Component tests: ThroughputExperts test in progress
- Full model test: Not yet implemented (pending ThroughputExperts fix)

### Quick Start

```bash
# Setup environment
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache
export GPT_OSS_HF_MODEL=/data/MLPerf/huggingface/hub/models--openai--gpt-oss-120b/snapshots/dc61ed29c478a29c51039f82fa4dcdf4f85e3ad2
export GPT_OSS_CACHE=/tmp/gpt_oss_cache

# Run working reference test (PASSES)
pytest models/demos/deepseek_v3/tests/test_moe_experts.py::test_forward_pass[decode-128-random-model.layers.3.mlp.experts.0-255] -xvvs

# Run distributed expert test (PASSES with PCC: 0.998415)
pytest models/tt-moe/tests/test_moe_components.py::test_06_distributed_expert_with_reference_comparison -xvvs

# Run infrastructure test (PASSES with PCC: 0.9885)
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs

# Test GPT-OSS configuration loading (works without hardware)
pytest models/tt-moe/tests/test_gpt_oss_moe_block.py::test_gpt_oss_config_loading -xvs

# ===== CRITICAL: GPT-OSS Reference Tests (Use as Ground Truth!) =====
# Test just the throughput experts (reference implementation that WORKS)
pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder -k "4x8 and decode_128 and layer_0" --test-modules experts

# Test entire MoE block (router + experts) - reference implementation
pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder --test-modules experts,mlp

# Run full GPT-OSS test suite (requires hardware)
bash /tmp/test_gpt_oss_integration.sh
```

### Performance Testing

```bash
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

### Key Learnings from Sparse Matmul Deep Dive (2026-02-16)

**Critical Discoveries and Solutions:**
1. **GPT-OSS Dimensions**: The model uses intermediate_size=2880 (same as hidden_size), NOT 360
   - This was causing all our initial configuration errors

2. **Exact Program Configs from Reference**:
   - Core grids: (5, 9) = 45 cores for both projections
   - in0_block_w=10 because K tiles = 2880/32 = 90, and 10 divides 90
   - per_core_M=1, per_core_N=2 (90 tiles / 45 cores)

3. **Topology Must Be Linear**:
   - Ring topology causes fabric routing errors on Galaxy
   - All CCL operations (all_to_all, all_reduce) must use Linear topology

4. **Mesh Tensor Conversion**:
   - all_reduce returns a sharded tensor even after reduction
   - Direct ttnn.to_torch() fails with "buffers.size() == 1" error
   - Solution: Use ConcatMeshToTensor for proper conversion:
     ```python
     output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
     ```

5. **Implementation Status**:
   - ✅ ThroughputExpert unit test passing
   - ✅ Sparse matmul operations working correctly
   - ⚠️ Bias addition temporarily disabled (can be re-enabled)
   - Next: Full model test with real weights and PCC validation

### Current Plan: GPT-OSS Full Model Testing (2026-02-16)

**Goal**: Complete GPT-OSS implementation by copying EXACT reference implementation that is PROVEN to work

**CRITICAL INSIGHT:** The reference implementation at `/models/demos/gpt_oss/tt/experts_throughput/` is WORKING and passes tests!
- Reference test command: `pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder -k "4x8 and decode_128 and layer_0" --test-modules experts`
- The reference USES sparse matmul and IT WORKS - we should copy it exactly!

**Key Files to Copy From (Reference Implementation):**
1. `/models/demos/gpt_oss/tt/experts_throughput/decode.py` - The working decode_forward() with sparse matmul
2. `/models/demos/gpt_oss/tt/experts_throughput/config.py` - ThroughputProgramConfig with working core grids
3. `/models/demos/gpt_oss/tt/experts_throughput/weights.py` - ThroughputExpertWeights loading pattern
4. `/models/demos/gpt_oss/tests/unit/test_modules.py` - run_throughput_experts_component() test pattern

**Implementation Plan (Copy Reference Exactly):**

1. **Fix ThroughputExpert to Match Reference Structure**
   - Our current file: `/models/tt-moe/components/experts/throughput_expert.py`
   - Copy the exact sparse matmul usage from reference `decode_forward()`
   - Use the EXACT program configs from reference (gate_up_cores=(5,9), in0_block_w=10)
   - Copy the exact weight loading/sharding pattern
   - Key differences to fix:
     * Reference returns ThroughputExpertWeights dataclass from convert_weights
     * Reference uses specific reshape/permute patterns for sparse matmul
     * Reference has working program configs that avoid the core grid errors

2. **Fix Weight Loading to Match Reference**
   - Reference loads weights into a ThroughputExpertWeights dataclass
   - Reference shards weights with ShardTensorToMesh on expert dimension
   - Reference handles fused gate_up_proj weights properly
   - Our convert_weights should return the same structure

3. **Fix Test to Match Reference Pattern**
   - Update `test_08_throughput_expert_basic` to match `run_throughput_experts_component()`
   - Exact input shapes from reference:
     * hidden_states: `torch.randn(batch, seq, hidden)` → `.unsqueeze(1)` → ttnn
     * routing_weights: dense format `[batch*seq, num_experts_per_tok]` → `.unsqueeze(1).unsqueeze(1)`
     * router_indices: `[batch*seq, num_experts_per_tok]` → `.unsqueeze(1).unsqueeze(1)` → uint16
   - Use mesh_mapper for row sharding when needed

4. **Verify Against Reference Test**
   - Run our test: `pytest models/tt-moe/tests/test_moe_components.py::test_08_throughput_expert_basic -xvs`
   - Run reference: `pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder -k "4x8 and decode_128 and layer_0" --test-modules experts`
   - Both should pass with similar PCC values

**Execution Steps:**

```bash
# Step 1: Fix ThroughputExpert implementation
# Edit: /models/tt-moe/components/experts/throughput_expert.py
# - Remove the expert loop (lines ~420-550)
# - Copy exact sparse matmul pattern from reference decode.py
# - Fix program configs to match reference (in0_block_w=10, cores=(5,9))
# - Fix convert_weights to return ThroughputExpertWeights dataclass

# Step 2: Run reference test to verify it works
pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder -k "4x8 and decode_128 and layer_0" --test-modules experts -xvs

# Step 3: Run our updated test (should now work!)
pytest models/tt-moe/tests/test_moe_components.py::test_08_throughput_expert_basic -xvs

# Step 4: Full GPT-OSS model test (after unit test passes)
pytest models/tt-moe/tests/test_gpt_oss_moe_block.py::test_gpt_oss_moe_against_reference -xvs

# Step 5: Verify DeepSeek still works
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs
```

**Files to Modify:**
1. `/models/tt-moe/components/experts/throughput_expert.py` - Main implementation
2. `/models/tt-moe/tests/test_moe_components.py` - Test to match reference pattern

**Reference Files (Read-Only, Copy From):**
1. `/models/demos/gpt_oss/tt/experts_throughput/decode.py` - Working sparse matmul
2. `/models/demos/gpt_oss/tt/experts_throughput/config.py` - Working program configs
3. `/models/demos/gpt_oss/tt/experts_throughput/weights.py` - Weight loading pattern
4. `/models/demos/gpt_oss/tests/unit/test_modules.py` - Test pattern

### Key Technical Details

**ThroughputExpert Implementation - Copy Reference Exactly:**

**Reference Working Implementation:**
- Location: `/models/demos/gpt_oss/tt/experts_throughput/decode.py`
- Uses sparse matmul SUCCESSFULLY - no need for regular matmul fallback
- Program configs that work:
  ```python
  # From reference config.py
  gate_up_cores: tuple[int, int] = (5, 9)
  down_cores: tuple[int, int] = (5, 9)
  in0_block_w: int = 10  # NOT 2, this is important!
  out_subblock_h: int = 1
  out_subblock_w: int = 1
  per_core_M: int = 1
  ```

**Key Implementation Details from Reference:**
1. **Weight Structure**: Returns `ThroughputExpertWeights` dataclass with w1, w2, w3, w1_bias, w2_bias, w3_bias
2. **Reshape Pattern for Sparse**:
   - After dispatch: `[1, 1, total_tokens, hidden]` → `[1, num_blocks, block_size, hidden]`
   - num_blocks = total_tokens // 32
3. **Sparse Matmul Call**:
   ```python
   ttnn.sparse_matmul(
       expert_input,
       weights.w1,
       sparsity=sparsity,
       memory_config=memory_config,
       program_config=program_config.get_gate_up_config(intermediate_size),
       is_input_a_sparse=False,
       is_input_b_sparse=True,
       output_tile=ttnn.Tile([32, ttnn.TILE_SIZE])  # block_size=32
   )
   ```
4. **SwiGLU Details**:
   - Gate: clamped max=7.0 only (no min)
   - Up: clamped min=-7.0, max=7.0
   - Formula: `(up + 1) * (gate * sigmoid(gate * 1.702))`

**Differences to Fix in Our Implementation:**
- We're trying to loop through experts - reference doesn't when using sparse
- We're using wrong program configs (our in0_block_w=2, should be 10)
- We're not returning ThroughputExpertWeights from convert_weights
- We're not using the exact reshape patterns

**GPT-OSS Model Configuration:**
- 128 total experts, 4 experts per device (32 devices)
- 4 experts selected per token (top-k routing)
- Hidden size: 2880, Intermediate size: 360
- Sparsity block size: 32
- No shared experts (unlike DeepSeek)

**Known Issues:**
- Ring topology causes fabric routing errors - use Linear topology
- ThroughputExpert currently has stub implementation with fallback
- No reference model for GPT-OSS (will need to create or mock)

### Reference

#### Environment Setup

```bash
# Required environment variables
export MESH_DEVICE=TG  # TensorGrid configuration (4x8 = 32 devices)
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD

# Model paths
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Optional debug flags
export DEEPSEEK_V3_DEBUG_EXPERTS=1  # Shows tensor stats during forward pass

# Reset Galaxy after a hang
source python_env/bin/activate && tt-smi -glx_reset
```

#### Key Test Commands

```bash
# Component tests
pytest models/tt-moe/tests/test_moe_components.py -xvvs  # All component tests

# Full model tests
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvvs  # DeepSeek-V3
pytest models/tt-moe/tests/test_gpt_oss_moe_block.py -xvs  # GPT-OSS (partial)

# Performance sweeps
pytest models/tt-moe/tests/test_deepseek_ag_hyperparameter_sweep_perf.py -xvs  # All-gather
pytest models/tt-moe/tests/test_deepseek_rs_hyperparameter_sweep_perf.py -xvs  # Reduce-scatter

# Reference tests (hand-written implementations)
pytest models/demos/deepseek_v3/tests/test_moe_experts.py -xvvs  # Reference experts
pytest models/demos/deepseek_v3/tests/test_decoder_block.py -k "decode" -xvvs  # Reference decoder
```

#### Architecture Overview

**TT-MoE Infrastructure:**
- Generalized MoE implementation configurable via JSON
- Supports multiple architectures (DeepSeek-V3, GPT-OSS, etc.)
- Modular components: routers, experts, CCL utilities

**DeepSeek-V3 Configuration:**
- 256 total experts, 8 experts per device (32 devices)
- 8 experts selected per token (was 6, fixed to match reference)
- Shared expert runs in parallel with MoE
- Float8 quantized weights with proper dequantization

**GPT-OSS Configuration:**
- 128 total experts, 4 experts per device (32 devices)
- 4 experts selected per token
- No shared experts
- Requires ThroughputExperts (not yet implemented)

**Mesh Configuration:**
- MESH_DEVICE=TG: TensorGrid with shape (4, 8)
- Tensor Parallel: axis 1
- Expert Parallel: axis 0
- CCL operations: All-gather, All-to-all, Reduce-scatter

#### Troubleshooting

**Common Errors and Fixes:**

1. **Block Configuration Error:**
   ```
   TT_FATAL @ matmul.cpp:123: program_config.out_block_w % program_config.out_subblock_w == 0
   ```
   Fix: Use `ttnn.from_torch` with `ReplicateTensorToMesh` instead of `ttnn.as_tensor`

2. **Fabric Not Initialized:**
   ```
   TT_FATAL @ ccl_runtime.hpp:165: Trying to get un-initialized fabric context
   ```
   Fix: Add `ttnn.set_fabric_config()` before opening mesh device

3. **Ring Topology Routing Error:**
   ```
   Fabric routing error with Ring topology
   ```
   Fix: Use Linear topology instead of Ring in configuration

4. **SharedExpert Numerical Explosion:**
   Fix: Refactored to match reference architecture with proper Float8 dequantization

5. **PCC Below Threshold:**
   Fix: Ensure `num_experts_per_tok` matches reference (8 for DeepSeek-V3)

#### Configuration Format

**Example JSON Configuration:**
```json
{
  "model_params": {
    "model_name": "DeepSeek-V3",
    "hidden_size": 7168,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "tp_size": 8,
    "ep_size": 4
  },
  "mode": "auto",
  "topology": "Linear",
  "num_ccl_links": 1,
  "replicated_batch_mode": true,
  "weight_block_size": [128, 128]
}
```

### Success Criteria

- **DeepSeek-V3**: PCC >= 0.98 (currently 0.989 ✅)
- **GPT-OSS**: PCC >= 0.98 (pending ThroughputExperts implementation)
- **Performance**: Comparable to hand-written implementations
- **Stability**: All tests pass without crashes or hangs
