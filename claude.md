# Claude Development Notes - DeepSeek-V3

## Environment Setup

### Python Environment
Always activate the Python environment before running tests:
```bash
source python_env/bin/activate
```

### Environment Variables for DeepSeek-V3

#### Model Paths
- **Model Directory**: `/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52`
  - Environment variable: `DEEPSEEK_V3_HF_MODEL`

- **Model Cache**: `/tmp/deepseek2`
  - Environment variable: `DEEPSEEK_V3_CACHE`
  - Clear cache if needed: `rm -rf /tmp/deepseek2/*`

#### Device Configuration
- **Mesh Device**: Set to `TG` for testing
  - Environment variable: `MESH_DEVICE`

### Running Tests

Example command to run MLP tests:
```bash
source python_env/bin/activate
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass -xvs --timeout=300
```

Example command to run MoE tests:
```bash
source python_env/bin/activate
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_moe.py::test_forward_pass -xvs --timeout=300
```

## Running Tests

To run the MLP tests:
```bash
source python_env/bin/activate
rm -rf /tmp/deepseek2/*  # Clear cache if needed
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass -xvs --timeout=300
```

To run the MoE tests:
```bash
source python_env/bin/activate
rm -rf /tmp/deepseek2/*  # Clear cache if needed
MESH_DEVICE=TG \
DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
DEEPSEEK_V3_CACHE=/tmp/deepseek2 \
python -m pytest models/demos/deepseek_v3/tests/test_moe.py::test_forward_pass -xvs --timeout=300
```

## Recent Changes

### 2026-02-23: Investigated Collective Operations in Tests

**Initial Request**: PR review comment asked to remove TTNN collective operations from test_mlp.py and test_moe.py

**Investigation Approaches**:
1. **Option A**: Simply remove collective ops → Failed: dimension mismatch
2. **Option B**: Use decoder blocks that handle ops internally → Still uses collective ops internally
3. **Option C**: Use replicated input to bypass all_gather → Failed: memory config mismatch
4. **Option D**: Compare sharded outputs directly → Partially works but still needs collective ops during computation

**Deep Technical Analysis**:

**MLP Weight Sharding Pattern (Megatron-style)**:
- **w1 (gate_proj)**: Column-parallel - needs FULL input (7168), produces SHARDED output (18432/8=2304 per device)
- **w3 (up_proj)**: Column-parallel - needs FULL input (7168), produces SHARDED output (18432/8=2304 per device)
- **w2 (down_proj)**: Row-parallel - takes SHARDED input (18432/8=2304), produces output needing reduction

**Why Collective Ops are Mathematically Required**:
- Column-parallel ops: `Y_i = X @ W_i` where W_i is a column slice → X must be full tensor
- Row-parallel ops: `Y = sum(X_i @ W_i)` where W_i is a row slice → needs reduction across devices
- This is NOT a TTNN limitation, it's a mathematical requirement of tensor parallelism

**Test Attempts and Results**:
1. **Replicated input approach**: Created tensor with `ReplicateTensorToMesh`
   - Failed: MLP weights configured with WIDTH_SHARDED memory expecting specific dimensions
   - Error: "Shard height 32 must match physical height 128"
2. **Sharded comparison approach**: Use `ttnn.get_device_tensors()` to compare sharded outputs
   - Works for comparing outputs but computation still needs collective ops
3. **Memory config experiments**: Tried matching all_gather output format
   - Failed: Tightly coupled memory configurations between weights and expected inputs

**Final Conclusion**: Collective operations are mathematically required for tensor parallelism with sharded weights.

### Final Solution (2026-02-24)

**Approach**: Move collective operations from tests into module forward functions with a `handle_tensor_parallel` parameter.

**Implementation**:

1. **MLP Module** (`models/demos/deepseek_v3/tt/mlp/mlp.py`):
   - Added `handle_tensor_parallel: bool = False` parameter to `forward_prefill` and `forward_decode`
   - When True, the module handles `all_gather` before computation and `reduce_scatter` after
   - When False (default), expects caller to handle collective operations

2. **MoE Module** (`models/demos/deepseek_v3/tt/moe.py`):
   - Added same `handle_tensor_parallel: bool = False` parameter pattern
   - Leverages existing `_fwd_all_gather` and `_fwd_reduce_scatter` helper methods

3. **Test Updates**:
   - `test_mlp.py`: Removed collective ops, passes `handle_tensor_parallel=True`
   - `test_moe.py`: Removed collective ops, passes `handle_tensor_parallel=True`

**Benefits**:
- Tests are cleaner without collective operation boilerplate
- Modules can be used flexibly - tests pass `handle_tensor_parallel=True`, decoder blocks use default `False`
- Maintains backward compatibility - decoder blocks continue to handle collective ops themselves

**Test Results**:
- MLP test: PASSED with PCC=0.976
- MoE test: PASSED with PCC=0.9909

**Key Insight**:
The collective operations in tests are **ESSENTIAL and CORRECT**. They are not redundant test overhead but integral to tensor parallelism:
- When decoder blocks call MLP/MoE, they handle collective ops in `_forward_mlp_common`
- When tests call modules directly, tests MUST handle collective ops
- The operations are mathematically required for sharded weight multiplication
- Original test implementation is architecturally correct

**Status**: No changes needed - collective operations must remain in tests
