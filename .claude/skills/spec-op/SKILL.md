---
name: spec-op
description: Create a structured operation specification for a new TTNN operation. Produces op_spec.md consumed by create-op pipeline. Use 'auto' keyword for simplest defaults.
argument-hint: "[auto] <operation requirements>"
---

# TTNN Operation Specification Skill

Produces a structured `op_spec.md` that feeds directly into the `create-op` pipeline as Phase -1. The spec captures what the operation does, what it needs, and how it should behave on hardware.

## Modes

| Mode | Invocation | Behavior |
|------|-----------|----------|
| **Interactive** (default) | `/spec-op <requirements>` | Progressive disclosure: mandatory questions first, then optional hardware steering |
| **Auto** | `/spec-op auto <requirements>` | Fill mandatory sections from args, use safest defaults for everything else |

---

## Input Parsing

Extract from the user's arguments:
1. **Mode**: Check if first word is `auto`
2. **Operation name**: snake_case identifier (e.g., `reduce_row_mean`)
3. **Requirements**: Everything else — math description, tensor info, constraints

---

## Constraint Reference

Before starting, read `.claude/references/ttnn-op-constraints.md` for validation rules. Use this throughout the interactive flow to eagerly validate user choices.

---

## Auto Mode Flow

When the user invokes `/spec-op auto <requirements>`:

1. **Parse** the operation name and math definition from args
2. **Infer** input/output tensors from the math definition where possible:
   - Unary op (one input) → single input tensor, output same shape
   - Binary op (two inputs) → two input tensors, output follows broadcasting rules
   - Reduction → output shape reduced on specified dim(s)
3. **Fill mandatory sections** from what's provided. For anything not specified:
   - Dtype: BFLOAT16
   - Layout: TILE_LAYOUT
   - Shape pattern: [N, C, H, W] (generic 4D)
   - Test shapes: (1, 1, 32, 64), (1, 1, 64, 128), (4, 8, 32, 256)
   - Tolerances: rtol=0.01, atol=0.01
4. **Fill all optional sections** with safe defaults from the constraints reference (Section 9: Common Auto-Mode Defaults)
5. **Create directory** `ttnn/ttnn/operations/{op_name}/` if it doesn't exist
6. **Write** `op_spec.md` using the output template below
7. **Report** the path and a brief summary

---

## Interactive Mode Flow

Progressive disclosure — mandatory sections first, then a hardware steering gate.

### Phase 1: Mandatory Sections

Ask these questions using `AskUserQuestion`. After each answer, validate against the constraint reference and warn immediately if conflicts are detected.

#### Question 1: Math Definition

Ask: "What does this operation compute?"

Capture:
- Mathematical formula (e.g., `output[i] = input[i] - mean(input, dim=W)`)
- Algorithm steps if multi-step
- Edge cases (division by zero, empty tensors, etc.)

#### Question 2: Input Tensors

Ask: "Describe the input tensor(s)."

For each input, capture:
| Field | Example values |
|-------|---------------|
| Role | input / weight / mask / bias / index |
| Shape pattern | [N, C, H, W] / [1, 1, H, W] / scalar |
| Supported dtypes | BFLOAT16, FLOAT32 / etc. |
| Layout | TILE_LAYOUT / ROW_MAJOR_LAYOUT |
| Rank | 2 / 4 / "any" |

**Eager validation** after this answer:
- BFLOAT8_B or BFLOAT4_B selected? → Verify layout is TILE_LAYOUT
- ROW_MAJOR selected with compute? → Warn that tilize/untilize stages will be needed

#### Question 3: Output Tensor

Ask: "What is the output?"

Capture:
| Field | Example values |
|-------|---------------|
| Shape relationship | same as input / reduced on dim X / expanded / custom formula |
| Shape formula | [N, C, 1, W] / [N, C, H, W] |
| Dtype | same as input / always FLOAT32 / etc. |
| Layout | same as input / TILE_LAYOUT |

#### Question 4: Parameters

Ask: "Does this operation take extra parameters (beyond tensors)?"

Capture:
| Field | Example values |
|-------|---------------|
| Name | dim / epsilon / keepdim / scale |
| Type | int / float / bool |
| Default | -1 / 1e-5 / True |
| Range/constraints | 0 <= dim < rank / epsilon > 0 |

If no parameters, write "None" for this section.

#### Question 5: Test Criteria

Ask: "How should we test this? Provide a PyTorch reference expression and test shapes."

Capture:
| Field | Example values |
|-------|---------------|
| PyTorch reference | `torch.mean(input, dim=2, keepdim=True)` |
| Test shapes | (1,1,32,64), (1,32,64,128), (4,8,32,256) |
| Tolerances | rtol=0.01, atol=0.01 |

If the user doesn't specify shapes, suggest tile-aligned defaults: (1, 1, 32, 64), (1, 1, 64, 128), (4, 8, 32, 256).

If the user doesn't specify tolerances, use defaults: rtol=0.01, atol=0.01 for BFLOAT16, rtol=0.001, atol=0.001 for FLOAT32.

### Phase 2: Hardware Steering Gate

Ask: "Do you want to steer hardware decisions (memory layout, buffering, compute config)? If not, I'll use the simplest safe defaults."

Options:
- **Yes, steer** → proceed to Phase 2 questions
- **No, use defaults** → fill sections 6-9 with auto defaults, skip to writing

### Phase 2 Questions (only if steering)

#### Question 6: Memory Layout

Ask: "What memory layout?"

Options:
- **INTERLEAVED only** (simplest, default) — single program factory, works everywhere
- **SHARDED only** — optimized for specific use case, needs shard spec decisions
- **BOTH** — two program factories, maximum flexibility

If SHARDED or BOTH:
- Ask sharding strategy: HEIGHT / WIDTH / BLOCK

That's it — no further hardware steering questions. Compute config is always exposed as an optional parameter with defaults inferred from dtypes (e.g., FP32 accumulation enabled when input is FLOAT32, disabled for BFLOAT16).

---

## Output Template

Write the following to `ttnn/ttnn/operations/{op_name}/op_spec.md`:

```markdown
# Operation Specification: {op_name}

> Generated by spec-op skill. Consumed by create-op pipeline.
> Mode: {interactive|auto}

## 1. Math Definition

**Formula**: {formula}

**Algorithm**:
{step-by-step if multi-step, or "Single-step element-wise/reduction/etc."}

**Edge cases**: {division by zero handling, empty tensor behavior, etc. or "None identified"}

## 2. Input Tensors

| Property | Tensor 0 | Tensor 1 |
|----------|----------|----------|
| Role | {input} | {weight/mask/none} |
| Shape pattern | {[N, C, H, W]} | {[N, C, H, W] or N/A} |
| Supported dtypes | {BFLOAT16, FLOAT32} | {same or N/A} |
| Layout | {TILE_LAYOUT} | {same or N/A} |
| Rank | {4} | {4 or N/A} |

## 3. Output Tensor

| Property | Value |
|----------|-------|
| Shape relationship | {same as input / reduced on dim X / etc.} |
| Shape formula | {[N, C, 1, W]} |
| Dtype | {same as input / FLOAT32 / etc.} |
| Layout | {same as input / TILE_LAYOUT} |

## 4. Parameters

| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| {dim} | {int} | {-1} | {0 <= dim < rank} | {Dimension to reduce} |

## 5. Test Criteria

| Property | Value |
|----------|-------|
| Reference (PyTorch) | `{torch expression}` |
| Test shapes | {(1,1,32,64), (1,1,64,128), (4,8,32,256)} |
| Tolerances | rtol={0.01}, atol={0.01} |

## 6. Memory Layout

| Property | Value |
|----------|-------|
| Preference | {INTERLEAVED} |
| Sharding strategy | {N/A} |

## 7. L1 Budget

| Property | Value |
|----------|-------|
| Expected CB count | {AUTO} |
| Buffering strategy | {SINGLE} |
| Large tensor concern | {NO} |

## 8. Work Distribution

| Property | Value |
|----------|-------|
| Work unit | {tile} |

## 9. Compute Config

| Property | Value |
|----------|-------|
| Exposed as parameter | YES (optional) |
| FP32 accumulation default | {Inferred: YES if input FLOAT32, NO if BFLOAT16} |
| Math fidelity default | {HiFi4} |
```

---

## Post-Write Actions

After writing `op_spec.md`:

1. **Confirm** the file path to the user
2. **Summarize** key decisions (especially any constraints flagged during validation)
3. **Suggest next step**: "Run `/create-op {op_name}` to start the implementation pipeline. It will auto-detect your spec."

---

## Validation Rules Quick Reference

These are checked eagerly in interactive mode. Full rules in `.claude/references/ttnn-op-constraints.md`.

| Check | Condition | Warning |
|-------|-----------|---------|
| Block format + ROW_MAJOR | BFLOAT8_B or BFLOAT4_B with ROW_MAJOR | "Block formats require TILE_LAYOUT" |
| ROW_MAJOR + compute | ROW_MAJOR input with any math | "Tilize/untilize stages will be needed" |
| Non-tile-aligned shape | Shape dims not multiples of 32 with TILE_LAYOUT | "Padding will be applied to reach tile boundaries" |
