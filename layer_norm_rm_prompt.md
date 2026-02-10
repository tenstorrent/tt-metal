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
3. **Output untilization**: The result must be untilized back to row-major before writing to DRAM.
4. **Epsilon**: Must be a configurable operation parameter, not hardcoded. Default: `1e-5`.
5. **Single-core**: All computation runs on a single Tensix core.

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
