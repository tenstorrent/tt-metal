# Nuke Skills Comparison: `/nuke-op` vs `/sfpu-unary-nuke-op`

## Overview

Both skills remove TTNN operations from the codebase so that agent systems can be evaluated on recreating them from scratch. They differ fundamentally in **what kind of operation** they target and therefore **how they remove it**.

| | `/nuke-op` | `/sfpu-unary-nuke-op` |
|---|---|---|
| **Target** | Any full TTNN operation (has its own directory) | A single SFPU unary eltwise operation (shares infrastructure) |
| **Deletion model** | Directory-level deletion + reference cleanup | Surgical line-level edits across 12 shared files |
| **Automation** | Shell script (`scripts/nuke_op.sh`) | Manual Edit tool, layer by layer |
| **Scope discovery** | Auto-discovers related variants (moreh, backward, attention) | Analyzes operation families to avoid breaking siblings |
| **Args** | `<category> <operation>` (e.g., `normalization softmax`) | `<UnaryOpType enum value>` (e.g., `ELU`, `CBRT`) |

---

## Structural Difference: Why Two Skills Exist

TTNN operations fall into two categories:

1. **Directory-based operations** (softmax, concat, groupnorm, etc.) — each has its own directory under `ttnn/cpp/ttnn/operations/{category}/{operation}/` containing program factories, device code, kernels, headers, and pybinds. `/nuke-op` handles these by deleting the directory and cleaning references from parent CMake/nanobind/Python files.

2. **SFPU unary operations** (ELU, CBRT, SIGMOID, etc.) — these do **not** have their own directories. They all share the `eltwise/unary` infrastructure: a common program factory, common dataflow kernels, and a common compute kernel dispatch (`eltwise_sfpu.cpp`). Each operation is just an enum value + switch cases + a few dedicated header files sprinkled across ~12 abstraction layers. `/sfpu-unary-nuke-op` handles these by surgically removing the operation's footprint from each shared file.

Using `/nuke-op` on an SFPU unary operation would delete the entire `eltwise/unary` directory and break every other unary op. Using `/sfpu-unary-nuke-op` on a directory-based operation would miss the vast majority of the code.

---

## `/nuke-op` — Directory-Level Nuke

### What It Removes

| Content | Action |
|---------|--------|
| `ttnn/cpp/ttnn/operations/{category}/{operation}/` (entire directory) | **DELETE** |
| All auto-discovered related op directories (moreh, backward, attention variants) | **DELETE** |
| Test `.py` files with op name in filename | **DELETE** |
| CMakeLists.txt references (per affected category) | **EDIT** |
| Nanobind registration (per affected category) | **EDIT** |
| Python golden/config entries (per affected category) | **EDIT** |

### Execution Flow

```
1. Pre-flight checks (op exists, clean git, detect name variants)
2. Dry run (--dry-run flag shows all discovered targets)
3. Run nuke_op.sh (possibly multiple times for name variants like groupnorm/group_norm)
4. LLM review of modified files (CMake, nanobind, Python per category)
5. Build verification (background agent)
6. Smoke test a surviving operation (background agent)
7. Summary report
```

### Key Features

- **Auto-discovery**: Given `normalization softmax`, automatically finds and nukes `moreh/moreh_softmax`, `moreh/moreh_softmax_backward`, `transformer/attention_softmax`, etc.
- **Script-driven**: Uses `scripts/nuke_op.sh` which handles backup, deletion, and reference stripping.
- **Backup**: Everything backed up to `/tmp/nuked_ops/{category}/{variant}/`.
- **Name variants**: Handles inconsistent naming (e.g., `groupnorm` vs `group_norm`) by running the script once per variant.
- **Multi-category cleanup**: Automatically cleans CMake/nanobind/Python in every category that had a target nuked.

### Limitations

- Keeps YAML sweep configs and `sweep_framework/` test files.
- Mixed test files (op name only in content, not filename) are kept but may break at runtime.
- Cannot handle SFPU unary operations (would delete the entire shared infrastructure).

---

## `/sfpu-unary-nuke-op` — Surgical Layer-by-Layer Nuke

### The 12 Abstraction Layers

Each SFPU unary operation has a footprint across these layers:

| Layer | File(s) | Action |
|-------|---------|--------|
| 1. UnaryOpType Enum | `unary_op_types.hpp` | EDIT: remove enum value |
| 2. Macro Definition Dispatch | `unary_op_utils.cpp` / `get_macro_definition()` | EDIT: remove case line |
| 3. Init & Function Dispatch | `unary_op_utils.cpp` / `get_op_init_and_func_*()` | EDIT: remove case block |
| 4. is_parametrized_type | `unary_op_utils.hpp` | EDIT: remove case (if applicable) |
| 5. C++ API Registration | `unary.hpp` | EDIT: remove registration macro/decl |
| 6. C++ API Implementation | `unary.cpp` | EDIT: remove function (if custom) |
| 7. Python Bindings | `unary_nanobind.cpp` | EDIT: remove binding call |
| 8. SFPU Split Includes | `sfpu_split_includes.h` | EDIT/SKIP: remove `#if` block (unless family shared) |
| 9. Compute API Header | `eltwise_unary/{op}.h` | DELETE or EDIT (if family shared) |
| 10. Architecture Kernels | `ckernel_sfpu_{op}.h` + `llk_math_eltwise_unary_sfpu_{op}.h` (WH + BH) | DELETE or EDIT (if family shared) |
| 11. SfpuType Enum | `llk_sfpu_types.h` (WH + BH) | EDIT/SKIP: remove entry (unless shared) |
| 12. Specialized Compute Kernel | `{op}_kernel.cpp` (if exists) | DELETE + remove dispatch case |

Plus test file deletion.

### Execution Flow

```
1. Pre-flight checks & family analysis (detect families, shared SfpuType, specialized kernels)
2. Dry run (list all files to modify/delete per layer)
3. Execute nuke across all 12 layers using Edit/Delete
4. Review modified shared files for syntax validity
5. Build verification (background agent)
6. Smoke test a surviving unary op (background agent)
7. Summary report
```

### Key Features

- **Family awareness**: Detects operation families (e.g., RELU family = RELU, RELU6, RELU_MAX, RELU_MIN, LEAKY_RELU). When nuking a family member, shared headers are edited surgically rather than deleted.
- **SfpuType analysis**: Checks whether the operation's `SfpuType` enum entry is shared by other operations before removing it.
- **Parameterized detection**: Handles both parameterized ops (in `get_op_init_and_func_parameterized()`) and non-parameterized ops (in `get_op_init_and_func_default()`).
- **Dual architecture**: Handles files in both `wormhole_b0` and `blackhole` architecture directories.
- **No script**: All edits done manually via the Edit tool — no shell script equivalent.

### Known Operation Families

| Macro | Family Members |
|-------|---------------|
| `SFPU_OP_RELU_FAMILY_INCLUDE` | RELU, RELU6, RELU_MAX, RELU_MIN, LEAKY_RELU |
| `SFPU_OP_TRIG_FAMILY_INCLUDE` | SIN, COS, TAN, SINH, COSH, ASIN, ACOSH, ASINH, ATANH |
| `SFPU_OP_ROUND_FAMILY_INCLUDE` | FLOOR, CEIL, TRUNC, FRAC, ROUND |
| `SFPU_OP_ISINF_ISNAN_INCLUDE` | ISINF, ISNAN, ISPOSINF, ISNEGINF, ISFINITE |
| `SFPU_OP_BINOP_WITH_SCALAR_INCLUDE` | ADD_UNARY_SFPU, SUB_UNARY_SFPU, MUL_UNARY_SFPU, DIV_UNARY_SFPU |
| `SFPU_OP_UNARY_COMP_INCLUDE` | UNARY_NE, UNARY_EQ, UNARY_GT, UNARY_LT, UNARY_GE, UNARY_LE, GTZ, LTZ, EQZ, LEZ, GEZ, NEZ |
| `SFPU_OP_ACTIVATIONS_INCLUDE` | SOFTSHRINK, SOFTSIGN, HARDSIGMOID, CELU |
| `SFPU_OP_ERF_ERFC_INCLUDE` | ERF, ERFC |

### Limitations

- Manual process (no automation script) — more error-prone than `/nuke-op`.
- Requires deep understanding of family relationships to avoid breaking sibling operations.
- Does NOT remove the Python golden function (`unary.py`) — this is intentional because it serves as the test oracle for re-implementation evaluation.

---

## Side-by-Side Comparison

| Dimension | `/nuke-op` | `/sfpu-unary-nuke-op` |
|-----------|-----------|----------------------|
| **Input** | `<category> <operation>` | `<UnaryOpType enum value>` |
| **Example** | `normalization softmax` | `ELU` |
| **Deletion granularity** | Whole directory tree(s) | Individual lines/blocks across shared files |
| **Script automation** | `nuke_op.sh` with `--dry-run` | No script; manual Edit tool |
| **Variant discovery** | Auto-discovers moreh/backward/attention dirs | Detects operation families via macro analysis |
| **Files typically deleted** | 10s–100s (entire op directories + tests) | 4–8 dedicated files (headers, kernels, tests) |
| **Files typically edited** | 3 per affected category (CMake, nanobind, Python) | 7–10 shared infrastructure files |
| **Shared file risk** | Low (each op has its own directory) | High (must preserve sibling operations in families) |
| **Backup** | `/tmp/nuked_ops/` | None (use `git checkout -- .` to restore) |
| **Golden function** | Removed (part of the deleted directory) | Kept (serves as test oracle) |
| **Build verification** | Background agent | Background agent |
| **Smoke test** | Surviving op from same category | Surviving unary op (e.g., abs, neg) |
| **Restore** | `git checkout -- .` | `git checkout -- .` |

---

## When to Use Which

| Scenario | Skill |
|----------|-------|
| Removing `softmax`, `groupnorm`, `concat`, `reduce` | `/nuke-op` |
| Removing `ELU`, `SIGMOID`, `CBRT`, `SELU` | `/sfpu-unary-nuke-op` |
| Removing an operation that has its own directory under `ttnn/cpp/ttnn/operations/` | `/nuke-op` |
| Removing an operation listed in the `UnaryOpType` enum that uses shared `eltwise/unary` infra | `/sfpu-unary-nuke-op` |
| Not sure which one | Check if the operation has a directory: `ls ttnn/cpp/ttnn/operations/*/{op_name}`. If yes, `/nuke-op`. If no, check if it's in `UnaryOpType` enum — if yes, `/sfpu-unary-nuke-op`. |

---

## Common Post-Nuke Steps (Both Skills)

1. **Build verification** — `./build_metal.sh` to catch any dangling references
2. **Smoke test** — run a test for a surviving operation to verify infrastructure integrity
3. **Fix residual references** — grep for the operation name across `ttnn/cpp/` and `tt_metal/hw/`
4. **Restore** — `git checkout -- .` to undo everything if needed
