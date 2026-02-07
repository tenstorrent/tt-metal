# TTTv2 Design Proposal

## Problem Analysis

The core issue with TTTv1 (lives in models/tt_transformers/ directory) is the **N×M×P explosion**:
- N models × M platforms × P tests = exponential complexity
- Every change requires validating all combinations
- Adding model #11 requires testing against models 1-10

This problem gets worse with each newly added model
Our motivations for TTTv2 is to address this scaling problem and enalbe scaling to 100+ models

## Goals

Single developer can add a new LLM model without tribal knowledge

TT-Transformers (TTT) code as a library
- Modular
- Composable
- Readable
- Reproducible
- Maintainable and releasable
- Verifiable

TTTv1 is good, first achievement of the goals for 10+ models; now TTTv2 must do better to get to 100+ models. TTTv2 should be a collection of building blocks that models consume, not a framework that controls models.

## Proposed Architecture

### Zen of TTTv2 Architecture

- Library, not Framework
- If-else on static conditions in forward() is bad
- Lazy and Transparent is better than proactive and opaque
- Unit tests are better than end-to-end tests
- Balance between code and codegen

# Instruction for AI agent to refactor tt_transformers modules to TTTv2 style

NOTE: models/common/modules/mlp contains the MLP module, which is the example to follow. `models/common/tests/modules/mlp/` contains the tests for the MLP module.

## Phase 1: Analysis

1. Read the original module (e.g., `models/tt_transformers/tt/attention.py`)
2. Identify all static branching axes in `forward()`:
   - Hardware topology: `is_galaxy` / `TG`
   - Mode: `decode` / `prefill`
   - Model dimensions: `dim`, `n_heads`, etc.
   - Other compile-time-known conditions
3. Draw an execution path graph showing all unique paths through the forward function
4. Identify runtime-dependent logic that MUST stay in forward (e.g., `seq_len`-dependent reshapes)


## Phase 2: Split by Hardware Topology

1. Create 1D (corresponds to 1x1, 1x2, 1x8 mesh_shapes -- non-TG) and 2D (corresponds to 4x8, 8x4 mesh_shapes -- TG) modules in the original file
2. Move the corresponding `if is_galaxy` branch into the 2D module
3. Update `forward()` to be a simple dispatcher

## Phase 3: Extract to Separate Files

1. Create `attention_1d.py` and `attention_2d.py` with:
   - Tightened config classes (remove TG-specific fields)
   - `Attention1D` class with the forward logic
   - Helper functions copied locally (don't import from original)

2. Config class structure:
3. **Test**: Add `test_attention_1d_class_vs_reference` and `test_attention_2d_class_vs_reference` comparing outputs

## Phase 4: Move Static Configs to `__init__`

1. Make separate forward methods
2. Pre-compute all program configs in `__init__`:
3. `Attention1D` and `Attention2D` classes should have `decode_forward` and `prefill_forward` methods
4. Simplify forward methods to use pre-computed configs

## Phase 5: Flatten Forward Functions

1. Create `decode_forward(self, x)` with NO if-else
2. Create `prefill_forward(self, x)` with only runtime logic
3. Inline utility functions (e.g., `tt_all_gather`, `tt_all_reduce`) as mode-specific methods

## Phase 6: Factory Method for Backward Compatibility

Look at `MLP1D.from_model_args` for an example

## Testing Strategy

1. **Unit tests** (no device): Config creation, helper functions
2. **Integration tests** (with device):
   - `test_attention_1d_vs_reference`: Compare `Attention1D` to HuggingFace reference
   - `test_attention_2d_vs_reference`: Compare `Attention2D` to HuggingFace reference
3. **Rejection tests**: Ensure 2D class rejects non-TG devices

## File Structure Convention

```
models/common/modules/
├── attention/
│   ├── attention_1d.py
│   └── attention_2d.py
├── mlp/
│   ├── mlp_1d.py
│   └── mlp_2d.py
models/common/tests/modules/attention/
├── test_attention_1d.py
├── test_attention_2d.py
models/common/tests/modules/mlp/
├── test_mlp_1d.py
├── test_mlp_2d.py
```

# Key Principles to Remember

1. **Never guess configs** - always read the original code to understand what program configs are used
2. **Keep runtime logic minimal** - only `seq_len` checks, input shape checks belong in forward
3. **Use method overriding** - config classes use methods so subclasses can override
4. **Test incrementally** - test after each phase before proceeding
5. **Preserve backward compatibility** - `from_model_args()` factory method bridges old and new
6. **Single device check** - always handle `[1, 1]` mesh as special case (no CCL ops)

# More specific instructions for AI agent to refactor other modules to TTTv2 style

## Key files to reference:
- **Example module**: `models/common/modules/mlp/mlp_1d.py`
- **Example tests**: `models/common/tests/modules/mlp/test_mlp_1d.py`
- **TTTv1 source**: `models/tt_transformers/tt/<module>.py`
- **TTTv1 config**: `models/tt_transformers/tt/model_config.py`
- **CCL functions**: `models/tt_transformers/tt/ccl.py`

## Step 0: Analyze dependencies with trace_dependencies.py
Before starting any refactoring work, run the dependency tracer to understand the module's parameter dependencies:
```bash
python_env/bin/python models/common/modules/trace_dependencies.py --matmul-helper \
    models/tt_transformers/tt/<module>.py \
    models/tt_transformers/tt/model_config.py
```
This will output a hierarchical dependency graph showing what parameters affect the module's behavior.

### How to run trace_dependencies.py:

```bash
# Default: analyze TTTv1 MLP module
python_env/bin/python models/common/modules/trace_dependencies.py

# Analyze a specific module (provide paths to module and its config):
python_env/bin/python models/common/modules/trace_dependencies.py <module_path> <config_path>

# Example: analyze attention module
python_env/bin/python models/common/modules/trace_dependencies.py \
    models/tt_transformers/tt/attention.py \
    models/tt_transformers/tt/model_config.py

# Include detailed matmul helper analysis:
python_env/bin/python models/common/modules/trace_dependencies.py --matmul-helpers

# Output as JSON:
python_env/bin/python models/common/modules/trace_dependencies.py --json

# Write JSON to file:
python_env/bin/python models/common/modules/trace_dependencies.py --json-file deps.json
```

### What the tool outputs:
1. **Config Accesses** - All `model_config["KEY"]` accesses in the module
2. **Attributes Used in Conditions** - `self.*` attributes that control branching
3. **Config Key Dependencies** - What each config key depends on
4. **Root Parameters** - The minimal set of parameters affecting behavior
5. **CCL Function Analysis** - Dependencies in collective communication functions
6. **Matmul Helper Methods** - Analysis of config helper functions (with `--matmul-helpers`)
7. **Complete Parameter Hierarchy** - 6-level dependency graph from hardware to terminal ops

## Step 1: Copy the TTTv1 module to TTTv2 location
```bash
# Create the module directory (do NOT add __init__.py)
mkdir -p models/common/modules/<module_name>/

# Copy the original module
cp models/tt_transformers/tt/<module>.py models/common/modules/<module_name>/<module>.py
```

### NOTES:
- Do NOT create `__init__.py`, `conftest.py`, or other boilerplate files unless explicitly requested. This project has specific file structure conventions.

## Step 2: Identify and bring in required dependencies
From the trace_dependencies.py output:
- **Terminal params**: `dim`, `hidden_dim`, `n_heads`, `cluster_shape`, etc. (from ModelArgs)
- **Config keys**: `DECODE_*_PRG_CONFIG`, `PREFILL_*_PRG_CONFIG`, `*_MEMCFG` (from model_config)
- **Helper functions**: `matmul_config`, `dram_matmul_config`, `create_sharded_memory_config`, etc.
- **CCL functions**: `tt_all_reduce`, `tt_all_gather` (if distributed)

Copy only the required helper functions into the new module file - don't import from original.

## Step 3: Remove model_config and args from __init__ signature
The goal is to make the module self-contained with explicit parameters:
```python
# BEFORE (TTTv1 style):
def __init__(self, mesh_device, args, model_config, layer_num, ...):
    self.model_config = model_config
    self.args = args

# AFTER (TTTv2 style):
# Happy path:
def __init__(self, w1: LazyWeight, w2: LazyWeight, w3: LazyWeight):
    # All config values come from the explicit ModuleConfig dataclass

# Power-user path:
@classmethod
def from_config(cls, config: <module>Config):
    # bypass the __init__ method of the base class for power users who want to customize the config
```
See `MLP1D.__init__` and `MLP1D.from_config` in `mlp_1d.py` for real examples.

## Step 4: Create config dataclass hierarchy
Follow the MLP pattern with sub-config classes, read `models/common/modules/mlp/mlp_1d.py` for reference.

Include `decode_input_memcfg` and `prefill_input_memcfg` fields in the config, and add a `_load_input_device_tensor()` helper function to resolve `LazyWeight` inputs. See `MLP1DConfig` and `_load_input_device_tensor()` in `mlp_1d.py`, or `RMSNorm1DConfig` in `rmsnorm_1d.py` for distributed sharding support.

If a module contains sub-modules, compose the sub-modules into the main module's config dataclass. See `Attention1DConfig` in `attention_1d.py` for an example of how to compose the sub-modules (i.e., `RMSNorm1DConfig`).

## Step 5: Split into 1D and 2D variants
- **1D**: For non-TG topologies (1x1, 1x2, 1x8 mesh shapes)
- **2D**: For TG topologies (4x8, 8x4 mesh shapes)

Create separate files: `<module>_1d.py` and `<module>_2d.py`

## Step 6: Flatten forward() - eliminate static branching
```python
# BEFORE (TTTv1 - branching on static conditions):
def forward(self, x, mode):
    if self.args.is_galaxy:
        # TG path
    else:
        # non-TG path
    if mode == "decode":
        # decode path
    else:
        # prefill path

# AFTER (TTTv2 - separate methods, no branching):
def decode_forward(self, x):
    # Only decode logic, no if-else on mode

def prefill_forward(self, x, seq_len: int):
    # Only prefill logic, seq_len-dependent reshapes are OK
```

Each `*_forward` method should accept `ttnn.Tensor | LazyWeight` and call `_load_input_device_tensor()` at the start to resolve the input. See `decode_forward()` in `mlp_1d.py` or `_decode_local_sharded()` in `rmsnorm_1d.py`.

## Step 7: Implement from_model_args factory method
For backward compatibility with TTTv1 models, see `MLP1D.from_model_args` in `models/common/modules/mlp/mlp_1d.py` for an example implementation.

## Step 8: Create tests
```
models/common/tests/modules/<module_name>/
├── test_<module>_1d.py    # Tests for 1D variant
├── test_<module>_2d.py    # Tests for 2D variant (if applicable)
```

### Test categories:
1. **Unit tests** (no device): Config creation, helper function correctness
2. **Integration tests** (with device): Compare output against HuggingFace reference
3. **Rejection tests**: Ensure 2D class rejects non-TG devices

### Requirements for unit tests:
- Since `forward()` accepts `LazyWeight`, tests can wrap torch input in `LazyWeight` and pass directly to `forward()` - no manual `ttnn.from_torch` conversion needed. This enables optional input caching for faster repeated tests. See `test_mlp_1d.py` or `test_rmsnorm_1d.py` for examples.
- There is pytest fixture -- `ttnn_mesh_device` in `models/common/tests/conftest.py`; see `test_mlp_1d.py` for examples on how to use it.
- two important test cases to include (see `test_mlp_1d.py` for examples):
1) `test_<module>_vs_reference`, where the test cases are collected and parameters are hardcoded as pytest mark parameters; this test focus on testing the `__init__`, `from_config` factory methods, and the `forward` method; the forward method must be run and the output tensor compared to the HuggingFace reference model's output tensor.
2) `test_<module>_vs_reference_from_model_args`, where we just test a single model with small number of parameters to prove backward compatibility; this test focus on testing the `from_model_args` factory method; the forward method must be run and the output tensor compared to the HuggingFace reference model's output tensor.
- Use the `ttnn_mesh_device` pytest fixture from `models/common/tests/conftest.py`.
- When adding collected test cases, hardcode the parameters as pytest mark parameters; do not rely on the csv files (they will be removed); see the below section "NOTES about collecting test cases" for more details.

### Requirements for running tests on hardware:

- Use `python_env/bin/python`
- Use `HF_MODEL` to specify a HF model name for testing `from_model_args` factory method. For example, `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- Use `git submodule update --init --recursive` to update the submodules before running tests.
- Use `./build_metal.sh -c --development && ./create_venv.sh` to build and create the virtual environment after pulling the submodules  and before running tests.
- If wondering "Would you like me to wait for the build to complete and then run the device tests, or would you prefer to run them manually later?", the answer is "wait and then run the device tests".
- As `models/common/tests/setup.cfg` dictates >80% coverage, however, the unit tests aims to achieve >90% coverage.
- to gather coverage metrics, e.g., when running test_mlp_1d.py with pytest, use:
```
"--cov=models.common.modules.mlp.mlp_1d",
"--cov-report=term-missing",
"--cov-config=models/common/tests/setup.cfg"
```
- Use `tt-smi -r` to reset all devices if tests failed due to bad device states; it is a good idea to first `source python_env/bin/activate` to activate the virtual environment before running `tt-smi -r`.
- do NOT skip tests!!! --> unless there is a mismatch in device shape as the example module's tests do.

### Requirements for running tests on simulator:
- Use `/localdev/gwang/scripts/setup_ttsim.sh` to setup the simulator environment. Let me know if you cannot find the script.
- Follow the instructions in the script to use `TT_METAL_SIMULATOR_HOME`, `TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE` to specify the simulator environment.
- Run the tests as you would on hardware, and the simulator should be used automatically -- look for log output like the following to confirm:
```bash
| info     |             UMD | Creating Simulation device (cluster.cpp:222)
```

### Collecting test cases:

#### mlp as an example
in the `models/tt_transformers/tt/mlp.py` file, do at the module level:

```python
collected = set()
if os.path.exists("mlp_1d_performance.csv"):
    with open("mlp_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                collected.add(",".join(row[1:]))

```

inside the `forward()` of `models/common/modules/mlp/mlp.py` as the first thing, do:
```python
        # "layer_num,cluster_shape,dtype,batch_size,dim,hidden_dim,hf_model_name,seq_len,mode",
        file_exists = os.path.exists("mlp_1d_performance.csv")
        with open("mlp_1d_performance.csv", "a") as f:
            if not file_exists:
                f.write(
                    "layer_num,cluster_shape_x,cluster_shape_y,x_dtype,w1_dtype,w2_dtype,w3_dtype,batch_size,filler,seq_len,dim,hidden_dim,hf_model_name,mode\n"
                )
            entry = f"{self.args.cluster_shape[0]},{self.args.cluster_shape[1]},{x.dtype},{self.w1.dtype},{self.w2.dtype},{self.w3.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},{self.args.hidden_dim},{self.args.model_name},{mode}"
            if entry not in collected:
                collected.add(entry)
                f.write(f"{self.layer_num},{entry}\n")
```

### script `run_tttv1_simple_demos.sh` that can run all the simple demos in TTTv1

With the changes done to `models/common/modules/mlp/mlp.py`, the script `run_tttv1_simple_demos.sh` can be updated to run the simple demos in TTTv1.

## Step 9: Run the tests
- When debugging failing tests, prioritize code comparison and analysis BEFORE repeatedly running tests. Compare implementation against working reference code to identify discrepancies.
- Always run tests after implementing changes - don't wait for user to remind you. If a plan says 'run test', execute it before reporting completion.
- After code change, make sure to run linter before reporting completion.

Two strategies for running the tests are as the following.
### When in active development
Run all the tests on the simulator to quickly iterate on the changes and to decouple from hardware. This should also allow parallel development of multiple modules, multiple ideas, and etc. The main benefit is that the development is not bottlenecked by hardware availability.

### When in verification mode
Run all the tests on hardware to ensure the TTTv2 module is working as expected. If there are any tests that fail, fix them.

## Step 10: Audit the changes
After the tests are created and run through all the collected test cases, do an audit of the changes that were needed to fix all the tests and compare those changes to TTTv1 module implementation in models/tt_transformers/tt/<module>.py. The goal is to check if the TTTv2 module is doing the same thing as the TTTv1 module. If there are any discrepancies, fix them, go to step 9 to run the tests again, go to step 10 to audit the changes again, and repeat until all the tests pass and the audit is successful.

## Step 11: Double-check dependencies
`modules/` is considered the core of the TTTv2 library, so we need to make sure that the dependencies are up to the design goals of TTTv2 -- no explicit dependencies except for TTNN and Python standard library. For example, one key thing to double-check is no explicit dependencies on e.g., `torch`.

One situation where it is fine to use torch is when the module uses non-core modules that imports torch. For example, `models/common/tensor_utils.py` imports `torch` and TTTv2 modules can uses tensor_utils.py to construct default configurations. This is not considered an explicit dependency on torch, because the users of the modules could easily override the default configurations with whatever libraries (e.g., other than torch) they want to use.

## Step 12: Double-check the code for more subtle issues
- Make sure there is no dead code
- Make sure there is no magic constant values
- Make sure there is no dependency on TTTv1 code within TTTv2 modules and tests except for the code in `from_model_args` and its tests.
- Double-check the test cases against the collected test cases to make sure they match
- Double-check the default configs against TTTv1 code to make sure there is no fabricated and dangerous defaults
- Comb through the prefill_forward and decode_forward to find if-else that are switching on static conditions
- Audit the comments and docstrings to make sure they are updated -- matching the code
