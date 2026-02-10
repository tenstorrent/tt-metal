# Layer Norm RM — Operation Prompt

Build a TTNN normalization operation called `layer_norm_rm` (row-major layer normalization). Use the **Generic Op Workflow** (Python-based, `ttnn.generic_op()`), **single-core** execution.

## Mathematical Definition

Given input tensor `X` of shape `[..., H, W]` (arbitrary rank, row-major, interleaved):

1. **Row-wise mean**: `μ_i = (1/W) * Σ_j X[i, j]` for each row `i`
2. **Centering**: `X̂[i, j] = X[i, j] - μ_i`
3. **Variance**: `σ²_i = (1/W) * Σ_j X̂[i, j]²`
4. **Inverse standard deviation**: `rstd_i = 1 / sqrt(σ²_i + ε)`
5. **Standardization**: `Y[i, j] = X̂[i, j] * rstd_i`
6. **Affine transform**: `Z[i, j] = γ_j * Y[i, j] + β_j`

Where `ε` (epsilon) is a scalar parameter of the operation (default: `1e-5`).

## Tensor Specifications

| Tensor   | Shape            | Layout    | Memory             | Data Type          |
|----------|------------------|-----------|--------------------|--------------------|
| Input X  | `[..., H, W]`   | ROW_MAJOR | DRAM, interleaved  | bfloat16 or float32 |
| Gamma γ  | `[1, ..., 1, W]` | ROW_MAJOR | DRAM, interleaved  | bfloat16 or float32 |
| Beta β   | `[1, ..., 1, W]` | ROW_MAJOR | DRAM, interleaved  | bfloat16 or float32 |
| Output Z | `[..., H, W]` (same as input) | ROW_MAJOR | DRAM, interleaved | bfloat16 or float32 |

Both **bfloat16** and **float32** data types must be supported (but you should write the code for a "general datatype"). All tensors in a single invocation share the same dtype.

## Dimension Constraints

- **W** (last dimension) must be a multiple of 32 (tile-aligned).
- **H** (second-to-last dimension) must be a multiple of 32 (tile-aligned).
- Higher dimensions are unrestricted.

## Key Implementation Constraints

1. **Gamma/Beta handling**: Read gamma and beta **once** from DRAM, **tilize them inside the compute kernel**, and reuse the tilized versions across all rows. Tilize and untilize are NOT in-place operations — they require separate source and destination CBs.
2. **Input tilization**: The row-major input must be tilized before compute operations.
3. **Output untilization**: The result must be untilized back to row-major inside the compute kernel. After untilize, the data is already in row-major stick order — the **writer should write sticks directly to DRAM** (based on the work unit) with no additional untilization step.
4. **Epsilon**: Must be a configurable operation parameter, not hardcoded. Default: `1e-5`.
5. **Single-core**: All computation runs on a single Tensix core.
6. **Reduce scalar CB**: The circular buffer used for the reduce scaler must be in **bfloat16 format regardless of the input tensor's data type**. The reduce hardware expects bfloat16 scalers even when the operation runs in float32 mode.

## Kernel Writer: Incremental Implementation Strategy

The kernel writer **must NOT** implement the full layer norm computation in one shot. Instead, it must follow an **incremental, test-driven approach**, building up the operation step by step. At each step, run the test to verify correctness (or at minimum, no hangs) before moving on. Debug and fix any hangs or issues at the current step before adding more complexity.

The recommended progression is:

1. **Step 1 — Passthrough (tilize → untilize)**: Read input from DRAM, tilize it in the compute kernel, immediately untilize it back, write to output. Verify the output matches the input exactly. This validates the full reader → compute → writer data path and CB synchronization.
2. **Step 2 — Row-wise mean + subtraction**: Add the row reduction to compute the mean, and subtract it from each element. Verify against `X - mean(X, dim=-1, keepdim=True)`.
3. **Step 3 — Variance + inverse std**: Add the variance computation and `rstd = 1/sqrt(var + eps)`. Verify the standardized output `(X - μ) * rstd` against PyTorch.
4. **Step 4 — Gamma/Beta affine transform**: Add gamma and beta reading, tilization, and the final affine transform. Verify against full `torch.nn.functional.layer_norm`.

At **each step**, run the test on at least the minimal shape (`[1, 1, 32, 32]`). Fix any hangs, CB mismatches, or correctness issues before proceeding to the next step. This incremental approach makes it far easier to isolate which stage caused a problem.

## Test Shapes

Test with a wide range of shapes covering both wide and tall tensors. Examples:

| Description       | Shape              |
|-------------------|--------------------|
| Square            | `[1, 1, 32, 32]`  |
| Square, larger    | `[1, 1, 128, 128]` |
| Wide              | `[1, 1, 32, 1024]` |
| Tall              | `[1, 1, 1024, 32]` |
| Very tall         | `[1, 1, 4096, 32]` |
| Tall and wide     | `[1, 1, 512, 512]` |
| Batched           | `[2, 3, 64, 128]`  |
| 3D                | `[1, 64, 128]`     |
| Minimal           | `[1, 1, 32, 32]`   |

Test each shape with **both bfloat16 and float32**. Do correctness checks against PyTorch `torch.nn.functional.layer_norm`.

## Execution Mode

Run in **fully automated mode** — introduce reasonable assumptions and do NOT ask for confirmation or clarifications at any stage of the pipeline.

## Breadcrumbs & Reporting

- Enable **full breadcrumbs logging** — ensure the breadcrumbs signal file exists before launching agents.
- Each agent should report: pain points encountered, decisions made, and any deviations from this spec.
- At the end, produce a well-structured **Markdown report** summarizing the entire operation creation process, including:
  - A summary section for each agent in the pipeline
  - Key decisions and assumptions made by each agent
  - Any deviations from this spec, with rationale
  - References/links to all log and breadcrumbs files produced
