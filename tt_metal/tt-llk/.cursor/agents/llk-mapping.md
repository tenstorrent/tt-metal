---
name: llk-mapping
model: gpt-5.2-codex
description: Trace a Python TTNN model to its LLK API calls and produce a GitHub-issue-style mapping table (one row per unique LLK API, with TTNN ops, configs, data formats, math fidelity, defines, etc.). Use when asked to map LLK APIs, analyze TTNN ops, trace compute kernels, or generate an LLK mapping for a model. Requires two inputs from the caller: `MODEL_PATH` (path to the Python model under `models/`) and `DEVICE` (`blackhole` | `wormhole_b0` | `quasar`).
readonly: true
---

# Static LLK Mapping Plan: Python Model → LLK API Table

This is the complete repeatable methodology used to trace a model's Python TTNN calls
all the way down to the LLK (Low-Level Kernel) API calls compiled onto the Tensix core.

## Required inputs

Before doing any analysis, confirm both values with the user (or extract them from the user's message):

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the Python model/test directory (relative to repo root) | `models/demos/yolov8x/tt` |
| `DEVICE` | Target hardware architecture (check `tt_metal/hw/ckernels/` for valid options) | `blackhole`, `wormhole_b0`, `quasar` |

If either value is missing or ambiguous, ask the user before proceeding.

Once confirmed, set these derived variables — every command below uses them:

```bash
LLK_ROOT="tt_metal/hw/ckernels/${DEVICE}/metal/llk_api"
LLK_SFPU_MACROS="${LLK_ROOT}/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
API_BASE="tt_metal/hw/inc/api/compute"
```

> All `grep` commands in Phases 1–5 reference `$MODEL_PATH`, `$LLK_ROOT`,
> `$LLK_SFPU_MACROS`, and `$API_BASE`. Set them once; run the plan for any model on any device.

> All paths in this file are relative to the **tt-metal repository root** (the directory
> that contains `ttnn/`, `tt_metal/`, `models/`, etc. — i.e. the top-level `tt-metal/`
> checkout, **not** the `tt_metal/tt-llk/` subdirectory where this agent lives, and **not**
> the `tt_metal/` subdirectory). Always `cd` to the tt-metal repo root before running any
> command in this file.

---

## Phase 1 — Enumerate All TTNN Ops from the Python Model

**Goal:** Produce a flat list of every `ttnn.*` call with its configuration.

### Steps

1. Identify all Python source files under `$MODEL_PATH/`
2. Read each file and record every `ttnn.*` call
3. For each call capture:
   - Op name (e.g. `ttnn.conv2d`, `ttnn.softmax`)
   - Input/output dtypes (`ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.float32`, `ttnn.int32`)
   - Memory config (`ttnn.L1_MEMORY_CONFIG`, `ttnn.DRAM_MEMORY_CONFIG`, sharded configs)
   - `compute_kernel_config` kwargs: `math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`, `math_approx_mode`
   - Activation fusion flags (e.g. `activation="silu"`, `activation="gelu"`)
   - Any scalar arguments (e.g. `ttnn.div(x, 2)`)

### Grep commands

```bash
# From a specific git commit:
git show <commit>:${MODEL_PATH}/<file>.py | grep -n "ttnn\."

# In working tree:
grep -rn "ttnn\." ${MODEL_PATH}/
```

### Important: $MODEL_PATH may be a test/demo harness, not the model module

If `$MODEL_PATH` points to a `demo/`, `tests/`, or test-entry-point directory (e.g.
`models/demos/<name>/wormhole/`, `…/<name>/demo/`, `…/<name>/tests/`), the grep above
will mostly return pytest decorators, parametrize fixtures, dtype/fidelity sweeps, and
imports — **not** the real `ttnn.*` call sites. The actual model implementation usually
lives in a sibling/parent directory under a name like `tt/`, `ttnn_<model>/tt/`, or
`<model>/tt/`.

**Workflow when the grep is dominated by harness code:**

1. Note the test-harness path — record from it the dtype/fidelity/batch-size/sharding
   sweeps the model is exercised under (these become the *Config Summary* values).
2. Follow the import chain in those test files to locate the model module:
   ```bash
   grep -rn "^from models\.\|^import models\." ${MODEL_PATH}/
   # Then read the imported file(s); their directory is the model implementation.
   ```
3. Re-run the Phase 1 grep on the implementation directory:
   ```bash
   IMPL_PATH="<resolved-import-path>"   # e.g. models/demos/<name>/ttnn_<name>/tt
   grep -rn "ttnn\." ${IMPL_PATH}/
   ```
4. Record **both** paths in the analysis output (test harness for config; implementation
   for op call sites). Treat the union as the effective model source for Phases 2–5.

**Heuristic for "is this a harness?":** If `grep -c "@pytest\|@run_for_\|parametrize\|run_for_blackhole\|run_for_wormhole_b0\|InfraType\|test_" $MODEL_PATH/**/*.py` is comparable to or larger than the count of unique `ttnn.*` op calls, treat `$MODEL_PATH` as a harness and follow the imports.

### Expected output (generic template — fill in per model)

One row per unique `ttnn.*` call found in the model. The values in "Config Summary"
come directly from reading the Python source — do not guess; record only what is
explicitly passed in the call site.

| TTNN Op | Config Summary |
|---------|---------------|
| `ttnn.<op_name>` | dtype, math_fidelity, activation, memory_config, scalar args, etc. |
| ... | ... |

**Example populated rows (from YOLOv8 and Qwen3-VL for illustration only):**

| TTNN Op | Config Summary |
|---------|---------------|
| `ttnn.conv2d` | `input_dtype=bfloat8_b`, `weight_dtype=bfloat8_b`, `math_fidelity=LoFi`, `activation="silu"` |
| `ttnn.add` | `bfloat16` inputs, residual |
| `ttnn.div` | `bfloat16` ÷ scalar `2` |
| `ttnn.softmax` | `dim=-1`, `bfloat16`, rank-4 tensor |
| `ttnn.linear` | `bfloat16`, `math_fidelity=HiFi2`, `activation="gelu"` |
| `ttnn.layer_norm` | `bfloat16`, `do_gamma=True`, `do_beta=True` |
| `ttnn.rms_norm` | `bfloat16`, `do_gamma=True`, `do_beta=False` (no bias) |

---

## Phase 2 — Map Each TTNN Op to Its C++ DeviceOperation and ProgramFactory

**Goal:** Find which `ProgramFactory` class is selected at runtime for each TTNN op.

### Steps

1. For each op, locate its C++ device operation file:
```bash
find ttnn/cpp/ttnn/operations -name "*device_operation*" | grep -i <op_name>
find ttnn/cpp/ttnn/operations -name "*op.cpp" | grep -i <op_name>
```

2. Read the `select_program_factory()` or `create_program()` method
3. Trace the branching logic against the Python config (dtype, shard strategy, tensor rank, dim, etc.)
4. Record the selected `ProgramFactory` class name

### Key branching logic to look for

These patterns appear in many device operation files — always check them for the op you are analysing:

- **Shard config present?** → routes to a multi-core sharded factory vs single-core factory
- **Tensor rank and dim** → some ops (softmax, reduce) select different factories based on which axis is targeted
- **`is_binary_sfpu_op()` or equivalent flag** → controls whether an FPU or SFPU compute kernel is compiled; often gated by `math_approx_mode` or the specific op type
- **`program_config` type** → ops like softmax and matmul use `std::variant` program configs; the active type selects the factory via `std::visit`
- **Data type of inputs/outputs** → integer vs float types can route to entirely different kernel files

### Expected output (generic template — fill in per model)

One row per unique TTNN op found in Phase 1. The factory name and selection condition
come from reading `select_program_factory()` in the op's device operation `.cpp`.

| TTNN Op | DeviceOperation Class | ProgramFactory Selected | Selection Condition |
|---------|-----------------------|-------------------------|---------------------|
| `ttnn.<op_name>` | `<Op>DeviceOperation` | `<Op>ProgramFactory` | e.g. dtype, rank, dim, shard config |
| ... | ... | ... | ... |

**Examples of factory selection patterns (from common TTNN ops — not exhaustive):**

These illustrate what the branching logic looks like. For any op not listed here, apply the
same reading process to its own device operation file.

| TTNN Op | DeviceOperation File | Key Branching Logic |
|---------|---------------------|---------------------|
| `ttnn.conv2d` | `conv/conv2d/device/conv2d_device_operation.cpp` | Sharded config present → `Conv2dMultiCoreShardedProgramFactory`; else single-core |
| `ttnn.softmax` | `normalization/softmax/device/softmax_device_operation.cpp` | rank=4, dim=rank-1, no sharded config → `SoftmaxProgramFactoryAttentionOptimized`; general dim → `SoftmaxProgramFactoryGeneralW/H` |
| `ttnn.add` / `.multiply` / `.sub` | `eltwise/binary_ng/device/binary_ng_device_operation.cpp` | `is_binary_sfpu_op()=false` → FPU path; `true` → SFPU path |
| `ttnn.sigmoid` / unary SFPU | `eltwise/unary/device/unary_device_operation.cpp` | SFPU op type → `UnaryProgramFactory` |
| `ttnn.linear` / `ttnn.matmul` | `matmul/device/matmul_device_operation.cpp` | Output sharding, 1D vs 2D → various `Matmul*ProgramFactory` variants |
| `ttnn.layer_norm` / `ttnn.rms_norm` | `normalization/layernorm/device/layernorm_device_operation.cpp` | Sharded → `LayerNormShardedMultiCoreProgramFactory` |
| `ttnn.concat` | `data_movement/concat/device/concat_device_operation.cpp` | Shard strategy → different factory per layout |

---

## Phase 3 — Extract Compute Kernel `.cpp` Paths and Compiler Defines

**Goal:** For each ProgramFactory, find the exact compute kernel `.cpp` file(s) it compiles
and all `#define`s / compile-time args injected.

### Steps

1. Read the ProgramFactory's `create_program()` function
2. Search for `CreateKernel(` calls with `ComputeConfig{...}`
3. Record the kernel file path string and the `defines` map
4. Note any `constexpr` compile-time args (CTAs) that gate SFPU branches

### Grep commands

The key insight: **never grep for specific define name strings** (like `SFPU_OP`, `ACTIVATION`,
`RELU`, `BCAST`). Those are model-specific and will always be incomplete. Instead grep for the
**C++ patterns that assign defines** — these are consistent across all program factories.

```bash
# Step 1: Find the compute kernel path — look for CreateKernel with a .cpp path string
grep -n "CreateKernel\|\.cpp\"\|compute_kernel_path\|compute_kernel_file" \
  ttnn/cpp/ttnn/operations/<op>/device/<factory>.cpp

# Step 2: Find ALL preprocessor defines injected into the compute kernel.
# Defines are always assigned by indexing a map with a string key, or via an initializer list.
# This pattern captures everything regardless of the define name.
grep -n 'defines\["\|compute_defines\["\|kernel_defines\["\|defines\.emplace\|defines\.insert\|"defines"\s*=' \
  ttnn/cpp/ttnn/operations/<op>/device/<factory>.cpp

# Step 3: Find ALL compile-time args (CTAs) passed to the compute kernel.
# CTAs control constexpr branches inside the kernel (e.g. do_gamma, do_beta, pack_relu).
grep -n "compile_time_args\|ComputeConfig{\|\.compile_time_args\s*=\|compute_args\b" \
  ttnn/cpp/ttnn/operations/<op>/device/<factory>.cpp

# Step 4 (optional): If defines are built conditionally, look at the full block around
# each CreateKernel call to capture all conditional define assignments:
grep -n -A 30 "CreateKernel" \
  ttnn/cpp/ttnn/operations/<op>/device/<factory>.cpp | grep -E '\.cpp|defines\[|compile_time_args'
```

> **Why not grep for define names?**
> Every model and op injects different define names — `SFPU_OP_INIT_ACTIVATION`, `GELU_ACTIVATION`,
> `FUSE_BIAS`, `PACK_RELU`, `SRC_BCAST`, `do_gamma`, `ELTWISE_OP_TYPE`, etc.
> Hardcoding any subset will silently miss the rest.
> The patterns above (`defines["`, `compile_time_args`, `ComputeConfig{`) are the
> **stable C++ idioms** used by every program factory regardless of what they inject.

### Expected output (generic template — fill in per model)

One row per unique TTNN op found in Phase 1. The kernel path and defines come from
reading `CreateKernel(...)` inside the selected ProgramFactory's `create_program()`.

| TTNN Op | Compute Kernel Path | Key Defines / CTAs |
|---------|--------------------|--------------------|
| `ttnn.<op_name>` | `<relative path from ttnn/cpp/ttnn/operations/>.cpp` | `DEFINE_NAME=value`, CTA index N = value |
| ... | ... | ... |

**Common kernel paths and their key defines (model-independent reference):**

| Op Category | Compute Kernel Path | Typical Defines |
|-------------|--------------------|-----------------|
| Conv2d | `conv/conv2d/device/kernels/conv_bmm_tilize.cpp` | `SFPU_OP_INIT_ACTIVATION`, `SFPU_OP_FUNC_ACTIVATION`, `FUSE_BIAS`, `PACKER_L1_ACC`, `SPLIT_READER` |
| Matmul / Linear | `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | `GELU_ACTIVATION`, `SILU_ACTIVATION`, `FUSE_BIAS`, `PACKER_L1_ACC` |
| Softmax (attention, dim=-1, rank-4) | `normalization/softmax/device/kernels/attention/compute/softmax.cpp` | `REDUCE_OP=MAX/SUM`, `EXP_APPROX`, `ENABLE_FP32_DEST_ACC` |
| Softmax (general dim) | `normalization/softmax/device/kernels/compute/softmax_general_*.cpp` | varies |
| Eltwise binary FPU (add/mul/sub) | `eltwise/binary_ng/device/kernels/compute/eltwise_binary.cpp` | `ELTWISE_OP`, `ELTWISE_OP_TYPE=ELWADD/ELWMUL/ELWSUB` |
| Eltwise binary SFPU scalar (div) | `eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` | `BINARY_SFPU_INIT`, `BINARY_SFPU_OP` |
| Unary SFPU (sigmoid, typecast, etc.) | `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `SFPU_OP_CHAIN_0=<init>; <op>(i);` |
| Copy / clone | `eltwise/unary/device/kernels/compute/eltwise_copy.cpp` | — |
| LayerNorm / RMSNorm | `normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp` | `do_gamma` (CTA), `do_beta` (CTA) |
| Max pool / reduce | `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | `REDUCE_OP=MAX`, `REDUCE_DIM=REDUCE_ROW` |
| Height-sharded concat | `data_movement/concat/device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` | — |

---

## Phase 3.5 — Build the kernel ↔ CTA-index ↔ value table

**Goal:** Resolve every `if constexpr (cta_N)` and `#ifdef DEFINE` guard in the kernel
to a concrete *value for this model's config*. Without this step Phase 4's guard
evaluation is error-prone (kernels routinely declare 5–10+ CTAs whose order matters).

### Why this matters

Compile-time args (CTAs) reach the kernel as a *positional* sequence. The program
factory pushes them in some order (`compile_time_args.push_back(value)` or via
`std::vector<uint32_t>{...}`), and the kernel reads them by index
(`constexpr X = get_compile_time_arg_val(N)`). The names on the two sides do **not**
have to match — only the order does. A misaligned CTA index silently changes which
`if constexpr` branches compile in, which silently changes which LLKs end up in the
final table.

### Steps

For each compute kernel identified in Phase 3:

1. **In the program factory `.cpp`:** list every CTA push in order. Search for the
   patterns that build the CTA vector:
   ```bash
   grep -n "compile_time_args\.push_back\|ComputeConfig{[^}]*\.compile_time_args\|std::vector<uint32_t>\s*{[^}]*}" \
     ttnn/cpp/ttnn/operations/<op>/device/<factory>.cpp
   ```
   Read the surrounding context to see the full ordered list. Note: many factories
   build the vector across multiple lines — read 30–50 lines around each match.
2. **In the kernel `.cpp`:** list every CTA read in order:
   ```bash
   grep -n "get_compile_time_arg_val" <kernel>.cpp
   ```
   Each match looks like `constexpr <type> <name> = get_compile_time_arg_val(<N>);`.
   Record `(N, name)`.
3. **Match by index.** For each CTA index `N`, you now have:
   - the kernel-side name (from step 2),
   - the program-factory expression that produced its value (from step 1).
4. **Evaluate** the factory expression against the Phase 1 model config (dtype,
   shard config, activation, fp32_dest_acc_en, etc.) and record the concrete
   value the model sends.

### Output table (one per kernel)

| CTA Index | Kernel Name | Factory Expression | Value for this Model | Guard(s) it Controls |
|-----------|-------------|--------------------|----------------------|----------------------|
| 0 | `<name from kernel>` | `<expression from factory>` | `<true/false/N>` | `if constexpr (<name>)` at line `<L>`; `#ifdef <macro>` at line `<L>` |
| 1 | … | … | … | … |

Also build a parallel table for `#ifdef`/`#define` guards (these come from the
`defines` map collected in Phase 3, not from CTAs):

| Define Name | Set By Factory? | Value for this Model | Guard(s) it Controls |
|-------------|-----------------|----------------------|----------------------|
| `<define name>` | `defines["<name>"] = …` (file:line) | `<value or "unset">` | `#ifdef <name>` / `#if defined(<name>)` at line `<L>` |

### Why two tables

CTAs and defines are different mechanisms with different guard syntax:
- **CTAs** → `if constexpr (cta_name)` / `if constexpr (cta_name == VALUE)`
- **Defines** → `#ifdef NAME` / `#if defined(NAME)` / `#if NAME == VALUE`

Some kernels use both heavily (e.g. `conv_bmm_tilize.cpp` has 8+ CTAs *and* defines
like `SFPU_OP_INIT_ACTIVATION`, `SPLIT_READER`, `FUSE_BIAS`, `PACK_RELU`). The two
tables together let Phase 4 mechanically tick off every guard.

### Once both tables are built

Phase 4's guard evaluation becomes a lookup: for every `if constexpr (...)` or
`#ifdef ...` block in the kernel, find the controlling identifier in the
appropriate table above and read the concrete value. Only LLKs in branches whose
guard evaluates to *true for this model* enter the Phase 5 final table.

---

## Phase 4 — Read Each Compute Kernel and Map to LLK API Calls

**Goal:** Find every LLK function invoked inside each compute kernel (directly or via
`compute_kernel_api.h` / `moreh_common.hpp` wrappers).

### Steps

1. Read the compute kernel `.cpp` directly
2. For every high-level call (e.g. `exp_tile()`, `sub_bcast_cols_init_short()`), trace it to
   the corresponding API header under:
   - `tt_metal/hw/inc/api/compute/` — top-level compute API wrappers
   - `tt_metal/hw/inc/api/compute/eltwise_unary/` — per-SFPU-op headers
3. In each header, find the `llk_*` function name inside `MATH(...)`, `PACK(...)`, or `UNPACK(...)` macros
4. Record the exact LLK function name, template parameters, and BroadcastType if applicable

### Grep commands

**Why Option 2 (extract API names first, then grep kernel) is more efficient:**

- The `api/compute/` headers contain ~500 named public API functions — this set is **fixed** and
  extracted once per analysis session, not per kernel.
- Option 1 (per-identifier reverse lookup) would require: for each of the ~N identifiers used in a
  kernel, search all 100+ header files to see if it is a public API function — O(N × headers).
- Option 2 is: build a single grep alternation pattern from all headers once — O(headers) — then
  run one grep pass over any kernel file — O(kernel_lines). Far fewer operations total.

```bash
# ── Step 1a: Build the complete set of public compute API function names ──────────
# Run ONCE per analysis session and reuse for every kernel.
#
# NOTE: APIs use several ALWI return types — not just void:
#   ALWI void   — the vast majority (tile ops, init calls, pack/unpack wrappers)
#   ALWI bool   — e.g. verify_calls (sentinel/testing infra — not called from kernels)
#   ALWI uint32_t — e.g. get_tile_address, get_compute_special_value_flags
#   ALWI <Class>& — sentinel class method chaining (exclude — contains '::')
# We include void + uint32_t + bool but exclude class-method declarations (contain '::').
API_PATTERN=$(grep -rh "^ALWI" "${API_BASE}" \
  | grep -v "::" \
  | sed 's/^ALWI [A-Za-z0-9_]* //' \
  | sed 's/(.*//' \
  | grep -E "^[a-z_][a-z0-9_]*$" \
  | sort -u \
  | tr '\n' '|' \
  | sed 's/|$//')
# Verify the count (expect ~500):
echo "$API_PATTERN" | tr '|' '\n' | wc -l

# ── Step 1b: Grep the kernel .cpp exhaustively ────────────────────────────────────
# Any call to a public compute API function will be matched.
grep -nE "\b($API_PATTERN)\b" <kernel>.cpp | grep -v "^[[:space:]]*//.*$"

# ── Step 1c: Also catch SFPU calls injected via #define (not present as call sites) ─
# Some kernels use macro placeholders that are filled by the program factory defines map.
# These are NOT literal function calls in the kernel source — they expand to API calls
# at compile time. Find all such placeholders by looking for define-site usages:
grep -n "#ifdef\|#ifndef\|defined(" <kernel>.cpp | grep -v "^[[:space:]]*//"
# Then cross-reference with the defines collected in Phase 3 to see what value was injected.

# Step 2: For each API function found in Step 1b, look up its LLK mapping.
# Use the Phase 4 Reference section below to grep the specific header directly:
grep -n "^ALWI" tt_metal/hw/inc/api/compute/<header>.h | grep "<function_name>"

# Then find the LLK inside that function body — look for MATH/PACK/UNPACK macro calls
# and any of the three LLK dispatch patterns:
#   (a) Direct:        MATH((llk_math_eltwise_unary_sfpu_sigmoid(idst)))
#   (b) Macro-wrapped: MATH(SFPU_UNARY_KERNEL_INIT(negative, APPROX))
#   (c) Macro-wrapped: MATH(SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX))
grep -n "llk_\|SFPU_UNARY_KERNEL_INIT\|SFPU_INIT_KERNEL_CALL\|SFPU_TWO_PARAM_KERNEL\|SFPU_THREE_TEMPLATE_PARAM_INIT\|SFPU_UNARY_ONE_PARAM_KERNEL\|SFPU_UNARY_NO_PARAM_KERNEL\|MATH(\|PACK(\|UNPACK(" \
  tt_metal/hw/inc/api/compute/<header>.h

# Step 3: Resolve SFPU macro calls to their final LLK symbol.
#
# When Step 2 finds a body like:
#   MATH(SFPU_UNARY_KERNEL_INIT(negative, APPROX))
# the macro expands to:
#   llk_math_eltwise_unary_sfpu_init<SfpuType::negative, APPROX>()
# So the LLK is always: llk_math_eltwise_unary_sfpu_init<SfpuType::<name>>
#
# The canonical list of all SFPU op names available on this arch is the SfpuType enum.
# Extract it directly from the LLK headers — this is more reliable than reading the macro file:
grep -rh "SfpuType::" "${LLK_ROOT}/" \
  | grep -oP "SfpuType::\w+" \
  | sort -u
# Each value maps to: llk_math_eltwise_unary_sfpu_init<SfpuType::<value>, APPROX>()
# For non-SFPU_UNARY_KERNEL_INIT macros (e.g. SFPU_INIT_KERNEL_CALL, SFPU_TWO_PARAM_KERNEL),
# resolve by reading the macro definition directly:
grep -n "^#define SFPU_" "${LLK_SFPU_MACROS}"
```

> **Why the API-name-first approach is exhaustive and the old grep was not:**
>
> The old approach grepped for a fixed list of call-site name fragments (`exp_tile`, `recip_tile`,
> `sub_bcast`, ...) which silently missed any API function not in that list.
>
> The new approach derives its search pattern **directly from the header files themselves** —
> any function defined in `api/compute/` will be in `$API_PATTERN` regardless of its name.
> There is no way for a call to slip through unless it is not a public compute API at all.
>
> The only calls `$API_PATTERN` does not catch are:
> - Calls injected via `#define` macro expansion (e.g. `SFPU_OP_FUNC_ACTIVATION`, `BINARY_SFPU_OP`)
>   — these are not call sites in the kernel source, they are textual substitutions set by defines.
>   To find these, grep the kernel for `SFPU_OP_FUNC_ACTIVATION\|BINARY_SFPU_OP\|SFPU_OP_CHAIN_0`
>   and then look up what value was injected via the defines map in Phase 3.
> - Direct `llk_*` calls that bypass `api/compute/` entirely (rare, kernel-specific).
>   Catch these with: `grep -n "llk_" <kernel>.cpp`

### Important guard patterns to respect

Kernels frequently contain conditional blocks that compile certain LLK calls only when a
specific define or compile-time arg is set. **Always check these guards before adding a row
to the final table** — only include LLKs that are actually compiled for the config your
model passes.

```bash
# Find all conditional compilation guards in a kernel:
grep -n "#ifdef\|#ifndef\|#if defined\|if constexpr" <kernel>.cpp | grep -v "^[[:space:]]*//"
```

For each guard found, cross-reference with the defines and CTAs collected in Phase 3 to
determine whether that branch is active for your model's configuration.

---

## Phase 4 Reference: How to Look Up Any API → LLK Mapping
> Use the commands below instead — they are always up to date with the actual source.

### Find what LLK a given compute API function dispatches to

```bash
# 1. Find which header declares the function:
grep -rn "^ALWI.*\b<function_name>\b" "${API_BASE}"

# 2. Read the function body to find MATH/PACK/UNPACK dispatch:
#    Look for one of three patterns:
#    (a) Direct:  MATH((llk_math_eltwise_unary_sfpu_sigmoid(idst)))
#    (b) Macro:   MATH(SFPU_UNARY_KERNEL_INIT(sigmoid, APPROX))
#                 → expands to llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROX>()
#    (c) Macro:   MATH(SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX))
#                 → resolve via the macro definition below
grep -A 10 "^ALWI.*\b<function_name>\b" "${API_BASE}"/**/*.h

# 3. If pattern (b) or (c), resolve the macro:
grep -n "^#define SFPU_" "${LLK_SFPU_MACROS}"

# 4. To see every SfpuType value the hardware supports (all SFPU op names):
grep -rh "SfpuType::" "${LLK_ROOT}/" \
  | grep -oP "SfpuType::\w+" | sort -u
```

### List every header in `api/compute/`

```bash
find tt_metal/hw/inc/api/compute -name "*.h" | sort
```

---

## Phase 5 — Merge Into Final LLK Issue Table

**Goal:** One row per **unique LLK API** (template parameters stripped to the bare
function name), with the TTNN ops that invoke it consolidated into the row. This
follows the layout used in [tt-llk#1363](https://github.com/tenstorrent/tt-llk/issues/1363),
where heavy LLK reuse across ops would otherwise produce a very repetitive per-op table.

### Required columns (12-column layout)

| Column | Description |
|--------|-------------|
| **LLK API** | Bare `llk_*` function name (template parameters stripped — e.g. `llk_math_eltwise_binary`, not `llk_math_eltwise_binary<ELWADD,NONE,…>`). |
| **Configs** | Number of textually-distinct template instantiations of this LLK seen across all kernels. (Static-analysis equivalent of "Total Invocations" — runtime call counts require Tracy and aren't available from static analysis.) |
| **TTNN Ops** | Comma-separated list of every `ttnn.*` op whose compute kernel invokes this LLK. |
| **Op Args** | Distinct template-parameter combinations seen for this LLK, separated by `;`. E.g. for `llk_math_eltwise_binary`: `<ELWADD,NONE,…,LoFi,NONE>; <ELWMUL,COL,…,HiFi4,NONE>; <ELWSUB,COL,…,LoFi,NONE>`. |
| **Input Data Formats** | Union of all input dtypes seen across kernels invoking this LLK (`Bfloat16, Bfp8_b, Bfp4_b, Float32, Int32, UInt16, …`). |
| **Output Data Formats** | Union of output dtypes (same notation). |
| **Tile Dims** | Always `32×32` on Tenstorrent hardware. |
| **Math Fidelity** | Union of fidelities seen (`LoFi, HiFi2, HiFi3, HiFi4`). Use `n/a` for non-fidelity-consuming LLKs (data-movement, sync helpers). |
| **Math Approx** | `True` / `False` / `True, False` if mixed across kernels. Mark `False (default)` when no kernel sets `math_approx_mode` or an `EXP_APPROX_MODE`-style define explicitly. |
| **FP32 Dest Accum** | `True` / `False` / `True, False` if mixed. (Equivalent to `DST_ACCUM_MODE` / `fp32_dest_acc_en`.) |
| **Dst Sync Mode** | `SyncHalf` (`dst_full_sync_en=false`, the default) or `SyncFull` (`dst_full_sync_en=true`). |
| **Kernel Defines** | Union of macro names from the `Key Defines / CTAs` columns of every kernel invoking this LLK. List names only for compactness; CTA values can be cross-referenced via the Phase 3.5 tables. |

### Folding rules for SFPU helper LLKs

Some LLKs are internal helpers shared by many SFPU ops:

- `_llk_math_eltwise_unary_sfpu_params_` — used by `exp`, `recip`, `fill`, `gelu`, etc.
- `_llk_math_eltwise_binary_sfpu_params_` — used by `swiglu`, `binary_add_sfpu`, etc.

Do **not** give these their own row. Instead fold them into rows for the named SFPU op
they implement (e.g. `llk_math_eltwise_unary_sfpu_exponential`,
`llk_math_eltwise_binary_sfpu_swiglu`). The "Op Args" cell of the named SFPU row should
record both the `_init<…>` template instantiation and the `_params_(calculate_<op>,…)`
template instantiation it dispatched to.

### Sort order

Group rows by Tensix thread, then alphabetical within group:

1. UNPACK LLKs (`llk_unpack_*`) — alphabetical
2. MATH LLKs (`llk_math_*`, `llk_init_packer_dest_offset_registers`) — alphabetical
3. PACK LLKs (`llk_pack_*`) — alphabetical

### Validation checklist before finalizing

- [ ] Every TTNN op from Phase 1 has ≥1 LLK row (or is recorded as **dataflow-only**)
- [ ] Every SFPU op has both `_init` and execution call accounted for (either as
      separate rows or folded into one named SFPU row per the rules above)
- [ ] All broadcast LLKs specify `BroadcastType` in the **Op Args** column
      (`ROW` / `COL` / `SCALAR` / `NONE`)
- [ ] Every conditional guard found in Phase 4 has been evaluated against the Phase 3.5
      CTA-index and define tables — only LLKs in branches whose guard evaluates to *true
      for this model* are present in the final table
- [ ] For any op that selects between FPU and SFPU paths (e.g. binary ops with
      `math_approx_mode`, or `is_binary_sfpu_op`), the chosen path is locked in and only
      its LLKs appear (not both)
- [ ] For any normalization op with optional affine parameters (gamma/beta), the
      broadcast LLKs corresponding to the *active* `do_gamma`/`do_beta` CTAs are present
- [ ] **Dataflow-only ops** (no compute LLK — the `ProgramFactory` creates only
      reader/writer kernels, no `ComputeConfig`) are listed in a separate section
      below the main table, not as empty rows in it
- [ ] **Configs** count matches the number of distinct entries in **Op Args** for the
      same row (sanity check that the two columns agree)
- [ ] Final row count is the number of unique bare-LLK names seen across all kernels —
      verify by counting `llk_` matches across the kernels (after stripping templates)

---

## Phase 6 — Deliver: ask the user where to put the output

Once Phases 1–5 are complete and validated, **always** ask the user how they want
the result delivered before printing or writing anything large.

### Required final question

End your run by asking exactly one question:

> **Where should I put the LLK mapping result?**
> - **(a)** Write it to a new Markdown file at `models/demos/<model>_llk_mapping.md`
>   (replace `<model>` with a short slug derived from `$MODEL_PATH`, e.g.
>   `yolov8x`, `qwen3_vl`, `gpt_oss`, `resnet50`).
> - **(b)** Print the full table inline in the chat output.
> - **(c)** Both — write the file *and* echo the Phase 5 table inline.

Wait for the user's choice before doing anything else.

### Behavior per choice

- **(a) File only.** Write **all** of: the model header (test entry point, resolved
  configuration, device, caveats), Phase 1 table, Phase 2 table, Phase 3 table,
  per-kernel Phase 4 traces, the Phase 5 12-column pivoted table, the dataflow-only
  list, and the "Not exercised" list (if any). Use the same structure as
  `models/demos/gpt_oss_llk_mapping.md` and `models/demos/resnet50_llk_mapping.md`
  as references. After writing, print only the file path and a 2–3 line summary
  (number of unique LLKs, number of compute kernels, most-shared LLKs).
- **(b) Inline only.** Print the same content directly in the chat response. Do not
  create a file.
- **(c) Both.** Do (a), then immediately echo the Phase 5 table (only Phase 5 — not
  the per-kernel Phase 4 traces, which are too long for inline) so the user can see
  it without opening the file.

### Filename convention

For choice (a) or (c), pick the slug deterministically:

1. If `$MODEL_PATH` matches `models/demos/<slug>/...`, use `<slug>`.
2. Otherwise use the *deepest* path component that uniquely identifies the model
   (e.g. `models/demos/vision/classification/resnet50/wormhole` → `resnet50`).
3. Append `_llk_mapping.md` and place the file under `models/demos/`.

If the chosen filename already exists, ask the user whether to overwrite or pick a
different slug — do not silently overwrite an existing mapping.

### Notes

- Do **not** ask this question before Phase 5 is complete. The whole point is to
  give the user the option to keep the (long) result on disk vs. inline. Asking
  before there's anything to deliver wastes a turn.
- If the user has already specified an output mode in the original request
  (e.g. "save the mapping to `<path>`" or "just print the table"), skip the
  question and honor their preference directly.
