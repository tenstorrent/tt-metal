---
name: golden-tests
description: Create golden test suite, API contract, and eval prompt for a TTNN operation. Ensures pipeline-generated ops are testable against ground truth. Use before /create-op to establish what "correct" means.
argument-hint: "<op_name> [description]"
---

# Golden Test Suite Generator

Creates the complete evaluation harness for a TTNN operation: golden tests, API contract, and eval prompt. Everything generated is consistent — the prompt tells the pipeline exactly what signature to produce, the contract validates it, and the golden tests verify correctness.

## What Gets Created

```
eval/golden_tests/{op_name}/
├── __init__.py
├── conftest.py                  # Pytest markers
├── helpers.py                   # PyTorch reference + conversion helpers
├── api_contract.md              # Machine-readable API contract
├── test_golden_shapes.py        # ~70 shape parametrizations
├── test_golden_modes.py         # Optional param variations + data distributions
└── test_golden_validation.py    # Input validation (Python-side, no device)

eval/prompts/{op_name}.txt       # Eval prompt with # golden: tag + explicit API
```

## Workflow

### Step 1: Parse Arguments

Extract `op_name` from the first argument. If a description follows, use it as initial context.

Check if files already exist:
- If `eval/golden_tests/{op_name}/` exists → warn and ask to overwrite or abort
- If `eval/prompts/{op_name}.txt` exists → warn and ask to overwrite or update

### Step 2: Gather Operation API (Interactive)

Use `AskUserQuestion` to collect all information needed. Ask up to 4 questions at a time for efficiency. The goal is to fully define the function signature and test expectations.

#### Question Set A: Core Definition

Ask these together:

1. **Math definition**: "What does `{op_name}` compute? Provide the mathematical formula or algorithm steps."
2. **PyTorch reference**: "What PyTorch expression computes the expected output? (e.g., `torch.nn.functional.layer_norm(x, [W], weight=gamma, bias=beta, eps=epsilon)`)"

#### Question Set B: Function Signature

Based on the math definition, ask:

3. **Input tensor**: "Describe the primary input tensor:"
   - Options: "bfloat16 ROW_MAJOR", "bfloat16 TILE", "float32 TILE", Other

4. **Additional tensor parameters**: "Does the operation take extra tensor parameters (like gamma, beta, weight)? If yes, describe each: name, shape pattern, whether optional (default=None)."
   - Options: "None (single input only)", "Has optional tensor params", "Has required tensor params", Other

#### Question Set C: Parameters and Output

5. **Scalar parameters**: "Does the operation take scalar parameters (like epsilon, dim)? For each, give: name, type, default value, and whether keyword-only."
   - Options: "None", "Has scalar params (specify in Other)", Other

6. **Output**: "What is the output?"
   - Options: "Same shape/dtype/layout as input", "Same shape, different layout", "Different shape (reduction)", Other

#### Question Set D: Validation and Tolerances

7. **Validations**: "What invalid inputs should raise ValueError/RuntimeError?"
   - Options with multiSelect: "Wrong dtype", "Wrong layout", "Wrong param shapes", "Wrong rank"

8. **Tolerances**: "What numerical tolerances should golden tests use?"
   - Options: "Tight (rtol=0.02, atol=0.1) — for reductions/normalization", "Medium (rtol=0.01, atol=0.05) — for simple compute", "Loose (rtol=0.05, atol=0.2) — for multi-step with approximations", Other

### Step 3: Derive Full API

From the gathered information, construct:

1. **Function signature** — determine parameter order, which are optional (default=None), which are keyword-only (after `*`)
2. **Valid call patterns** — enumerate all valid ways to call the function (with/without optional params)
3. **Shape categories** — generate the comprehensive shape list (see Shape Generation below)
4. **Mode test matrix** — determine what mode variations to test based on optional params

### Step 4: Generate Files

Generate all files using the templates below. Write them all, then verify with a quick syntax check.

### Step 5: Verify and Report

1. Run `python3 -c "import ast; ast.parse(open('path').read())"` on each generated .py file to verify syntax
2. Report all created files with a summary table
3. Suggest: "Run `/create-op` with `eval/prompts/{op_name}.txt` or use it as input to `run_eval.sh`"

---

## File Templates

### `__init__.py`

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
```

### `conftest.py`

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for {op_name} golden tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: fast tests for smoke checking (<5s each)")
    config.addinivalue_line("markers", "standard: standard shape coverage tests")
    config.addinivalue_line("markers", "large: large tensor tests (may be slow)")
    config.addinivalue_line("markers", "stress: stress tests with extreme values")
    config.addinivalue_line("markers", "validation: input validation tests")
```

### `helpers.py`

Structure:

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for {op_name} golden tests."""

import torch
import ttnn


def pytorch_{op_name}({params_matching_op_signature}):
    """
    Reference implementation using PyTorch, computed in float32 for precision.
    Returns result in bfloat16 to match expected output dtype.
    """
    # Convert to float32 for precision
    x = input_tensor.to(torch.float32)
    # {user-provided PyTorch reference logic}
    return result.to(torch.bfloat16)


def to_ttnn(tensor, device, dtype=ttnn.bfloat16):
    """Helper to convert a torch tensor to a ttnn tensor on device."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.{LAYOUT},  # ROW_MAJOR_LAYOUT or TILE_LAYOUT
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_output(ttnn_output, expected, shape, rtol, atol):
    """Validate shape, dtype, layout, and numerical correctness."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == ttnn.bfloat16, (
        f"Dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"
    )

    assert ttnn_output.layout == ttnn.{LAYOUT}, (
        f"Layout mismatch: got {ttnn_output.layout}, expected {LAYOUT}"
    )

    actual = ttnn.to_torch(ttnn_output).float()
    expected_f = expected.float()

    abs_diff = (actual - expected_f).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    passing = torch.allclose(actual, expected_f, rtol=rtol, atol=atol)
    assert passing, (
        f"Numerical mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
        f"rtol={rtol}, atol={atol}"
    )
```

**CRITICAL**: The `pytorch_{op_name}` function MUST accept the same parameters as the operation's function signature (same names, same defaults, same optional/keyword-only structure). This ensures golden tests can call the reference with the same arguments they pass to the TTNN op.

### `api_contract.md`

```markdown
# API Contract: {op_name}

Golden tests import and call the operation as documented below.
Any pipeline-generated implementation MUST match this contract for golden tests to pass.

## Import Path

\```python
from ttnn.operations.{op_name} import {op_name}
\```

## Function Signature

\```python
def {op_name}(
    {full signature with types and defaults, annotated with comments}
) -> ttnn.Tensor
\```

## Valid Call Patterns

| Pattern | Example | Test File |
|---------|---------|-----------|
{one row per valid call pattern}

## Input Requirements

| Property | Requirement |
|----------|-------------|
{rows for dtype, layout, memory, shape constraints}

## Output Requirements

| Property | Requirement |
|----------|-------------|
{rows for shape, dtype, layout}

## Validation (Python-side, pre-device)

The operation MUST raise `ValueError` or `RuntimeError` for:
{list of invalid input conditions}

## Numerical Tolerances

| Test category | rtol | atol |
|---------------|------|------|
{rows per tolerance tier}
```

### `test_golden_shapes.py`

Generate ~70 parametrized test cases across these categories:

| Category | Count | Strategy |
|----------|-------|----------|
| Minimal | 1 | Single tile: (1, 1, 32, 32) |
| Width scaling | 11 | H=32, W from 64 to 4096 (powers of 2 + some non-powers) |
| Height scaling | 11 | W=32, H from 64 to 4096 |
| Square | 5 | 64x64 through 1024x1024 |
| Wide rectangles | 7 | W >> H |
| Tall rectangles | 7 | H >> W |
| Multi-batch | 7 | N > 1 with various spatial sizes |
| Multi-channel | 5 | C > 1 |
| Batch + channel | 6 | Both N > 1 and C > 1 |
| Large | 6 | Stress memory and multi-core |

Each test calls the operation with ALL required + optional params provided (the "full" invocation).

**CRITICAL**: All shapes MUST be tile-aligned (H and W divisible by 32). Adapt this if the operation requires different alignment.

### `test_golden_modes.py`

Generate tests for each variation:

1. **Optional tensor params omitted** — one test per optional tensor param set to None
   - If gamma and beta are both optional: test with neither, test with gamma only (if that makes sense), test with both
2. **Identity affine** — if the op has scale/shift params, test with identity values (ones/zeros)
3. **Scalar parameter variations** — for each scalar param, test multiple values spanning its valid range
   - Example for epsilon: [1e-5, 1e-6, 1e-3, 1e-2]
   - Example for dim: test each valid dimension
4. **Data distribution variations** — ALWAYS include these regardless of operation type:
   - Uniform input [0, 1]
   - Small magnitude (near zero, ×0.01)
   - Large magnitude (×10.0)
   - Positive-only (rand + 0.5)
   - Negative-only (-(rand + 0.5))

Use a representative subset of shapes (5 shapes covering minimal, medium, non-square, multi-batch, large).

Slightly relaxed tolerances for near-zero inputs (1.5x-2.5x the standard tolerance).

### `test_golden_validation.py`

Generate one test per validation rule:

```python
def test_rejects_{condition}(device):
    """Must reject {description}."""
    t = _make_tensor(device, ...)
    with pytest.raises((ValueError, RuntimeError)):
        {op_name}(t)
```

Common validations:
- Wrong dtype (e.g., float32 when bfloat16 required)
- Wrong layout (e.g., TILE_LAYOUT when ROW_MAJOR required, or vice versa)
- Wrong parameter shape (e.g., gamma width != input width)

### `eval/prompts/{op_name}.txt`

```
# golden: {op_name}
/create-op
{math description from user, naturally phrased}

{Explicit API section:}
The function signature must be:

    {op_name}({full_signature})

This supports these call patterns:
{list each valid call pattern with a comment}

The operation must validate inputs on the Python side and raise ValueError or
RuntimeError for: {list validation conditions}.

{Input requirements: dtype, layout, shape alignment, etc.}

Run in FULLY AUTOMATED mode. Introduce reasonable assumptions and DO NOT ask
for confirmation or clarifications.

Run with FULL BREADCRUMBS LOGGING ENABLED.

## Reporting

In the end, produce a well-structured markdown report summarizing the whole
process, including:
- Summary for each agent (analyzer, planner, builder, designer, writer)
- References to log/breadcrumb files
- TDD pipeline results: which stages passed, how many attempts each took,
  any failure classifications encountered
- Decisions made, deviations from specs, pain points
- Infrastructure issues: any device access errors, device hangs, build
  failures or delays, venv problems, or other environment issues encountered
- Suggestions for improving the agent pipeline
```

**CRITICAL**: The prompt MUST include the exact function signature and all valid call patterns. This is what prevents the mismatch between what the pipeline generates and what the golden tests expect.

---

## Shape Generation Rules

All shapes are 4D: (N, C, H, W).

**Tile alignment**: H and W must be divisible by 32 for tile-aligned ops. For row-major-only ops that don't need tile alignment, shapes can have arbitrary H/W (but still keep them reasonable).

**Standard shape set** (reuse across operations):

```python
MINIMAL_SHAPES = [(1, 1, 32, 32)]

WIDTH_SCALING_SHAPES = [
    (1, 1, 32, w) for w in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
]

HEIGHT_SCALING_SHAPES = [
    (1, 1, h, 32) for h in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
]

SQUARE_SHAPES = [(1, 1, s, s) for s in [64, 128, 256, 512, 1024]]

WIDE_SHAPES = [
    (1, 1, 32, 512), (1, 1, 32, 1024), (1, 1, 64, 512),
    (1, 1, 64, 1024), (1, 1, 64, 2048), (1, 1, 128, 1024), (1, 1, 128, 2048),
]

TALL_SHAPES = [
    (1, 1, 512, 32), (1, 1, 1024, 32), (1, 1, 512, 64),
    (1, 1, 1024, 64), (1, 1, 2048, 64), (1, 1, 1024, 128), (1, 1, 2048, 128),
]

BATCH_SHAPES = [
    (2, 1, 32, 32), (4, 1, 32, 32), (8, 1, 32, 32),
    (2, 1, 64, 128), (4, 1, 128, 256), (2, 1, 256, 512), (8, 1, 64, 64),
]

CHANNEL_SHAPES = [
    (1, 2, 32, 32), (1, 4, 32, 32), (1, 8, 64, 64),
    (1, 3, 128, 128), (1, 16, 32, 64),
]

BATCH_CHANNEL_SHAPES = [
    (2, 3, 32, 32), (4, 2, 64, 64), (2, 4, 128, 128),
    (3, 3, 64, 128), (2, 2, 256, 256), (8, 4, 32, 64),
]

LARGE_SHAPES = [
    (1, 1, 2048, 2048), (1, 1, 4096, 128), (1, 1, 128, 4096),
    (2, 1, 1024, 1024), (1, 1, 4096, 256), (4, 1, 512, 512),
]
```

For operations with **reduced output shape** (e.g., reduce on dim W), adapt the expected output shape in the test accordingly.

---

## Cross-Consistency Checks

Before writing files, verify internal consistency:

1. **Signature match**: `helpers.py:pytorch_{op_name}()` signature mirrors the TTNN op signature in `api_contract.md`
2. **Call patterns match**: Every call pattern in `api_contract.md` has corresponding tests in `test_golden_modes.py`
3. **Validation match**: Every validation in `api_contract.md` has a test in `test_golden_validation.py`
4. **Prompt match**: The prompt's explicit signature section matches `api_contract.md`
5. **Import path**: All test files import from `ttnn.operations.{op_name}` (NOT `ttnn.ttnn.operations`)

---

## Example Invocation

```
/golden-tests layer_norm_rm
```

Interactive flow asks about the operation, then generates all files.

For a quick invocation with inline description:

```
/golden-tests my_op "Element-wise sigmoid: output = 1 / (1 + exp(-input))"
```
