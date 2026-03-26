---
name: LLK Per-Group Discovery + Locate Agent Prompt
description: Stage 1b per-group agent. Receives a pre-specified op list from the Stage 1a Catalog, deep-reads each op's wrapper header to understand its implementation, then locates all source files. One instance per functional group, all run in parallel.
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{GROUP_NAME}}` — functional group name (e.g. Activations, Trigonometry)
- `{{LLK_CATEGORY}}` — the operation category (e.g. elementwise unary)
- `{{OPS_LIST}}` — comma-separated ops assigned to this group by the Stage 1a Catalog
- `{{COMPUTE_API_DIR}}` — e.g. `tt_metal/hw/inc/api/compute/eltwise_unary`
- `{{LLK_DIR}}` — LLK/ckernel directory to search, e.g. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu` or `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api` for broader categories
- `{{CODEGEN_FILE}}` — e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `{{PROGRAM_FACTORY_DIR}}` — e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/device`

## Prompt Template

```
Investigate the {{GROUP_NAME}} group of {{LLK_CATEGORY}} operations: {{OPS_LIST}}.

BREADCRUMB LOGGING:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
GROUP_SLUG = "{{GROUP_NAME}}" lowercased, spaces → underscores.
BCRUMB="agent_logs/${CATEGORY_SLUG}_${GROUP_SLUG}_discovery_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"discovery","group":"{{GROUP_NAME}}","ops":"{{OPS_LIST}}"}' >> $BCRUMB

PHASE 1: DEEP READ — understand each op's wrapper

For each op in {{OPS_LIST}}, find and read its header in `/localdev/astancov/tt-metal/{{COMPUTE_API_DIR}}/`.
Extract:
- `{op}_tile_init()` signature (template params, what it calls internally)
- `{op}_tile()` signature (template params, runtime params, what it calls internally)
- What LLK/ckernel function or SFPU macro the wrapper calls
- Any key template params that control behavior (approx mode, int32 variants, etc.)

IMPORTANT: Many ops use SFPU macros directly (no named LLK function). For these, the wrapper header IS the primary definition — record the macro it calls.

After reading each wrapper:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","phase":"1","op":"OP","file":"wrapper_path","init_calls":"e.g. llk_math_eltwise_unary_sfpu_init<SfpuType::EXP>","compute_calls":"e.g. SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN","template_params":["approx:bool=false"],"runtime_params":["dst_index:uint32_t"]}' >> $BCRUMB

PHASE 2: VERIFY GROUP MEMBERSHIP

For each op, based on what Phase 1 revealed about what it computes:
- Confirm it belongs in {{GROUP_NAME}} — or flag if it seems misclassified
- Note any op that is borderline / could also belong to another group

Flag a misclassification if: the op's computation is clearly different from the group's theme (e.g., a bitwise op landed in Activations).

  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","group_confirmed":true/false,"reason":"WHY — what the op actually does","suggested_group":"other_group_or_NONE"}' >> $BCRUMB

PHASE 3: LOCATE — find source files for each op

For each op in {{OPS_LIST}}:
1. WRAPPER FILE: already found in Phase 1
2. LLK / CKERNEL FILE: grep for `{op}` in `/localdev/astancov/tt-metal/{{LLK_DIR}}/` and its subdirectories
3. CODEGEN: grep for op name or its enum value in `/localdev/astancov/tt-metal/{{CODEGEN_FILE}}`
4. PROGRAM FACTORY: grep for op name in `/localdev/astancov/tt-metal/{{PROGRAM_FACTORY_DIR}}/` (*.cpp)
5. CUSTOM KERNEL: check `/localdev/astancov/tt-metal/{{PROGRAM_FACTORY_DIR}}/kernels/compute/` for a file matching the op name

RULES:
- Do NOT read file contents in Phase 3 — only grep for paths and line numbers
- NONE for any field not found, with a brief reason
- For ambiguous matches, pick most specific, note the choice

After locating each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","phase":"3_locate","op":"OP","wrapper":"path:line","llk":"path_or_NONE","codegen":"path:line_or_NONE","factory":"path_or_NONE","custom_kernel":"path_or_NONE"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","group":"{{GROUP_NAME}}","ops_read":N,"misclassified":M,"ops_located":N}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_${GROUP_SLUG}_discovery_execution_log.md: per-op summary of what was found, misclassification flags, any NONE locate fields with reasons.

OUTPUT FORMAT — structured tables only:

## Group: {{GROUP_NAME}}

### Op Implementations

| Op | Init Calls | Compute Calls | Template Params | Runtime Params | Implementation Mechanism |
|----|-----------|---------------|-----------------|----------------|-------------------------|

(Implementation Mechanism = "named LLK fn" or "SFPU macro: MACRO_NAME" or "ckernel direct")

### Misclassifications

| Op | Assigned Group | Suggested Group | Reason |
|----|---------------|-----------------|--------|

(Empty table if none.)

### Locator Results

| Op | Wrapper File | Wrapper Line | LLK/Ckernel File | Codegen File:Line | Program Factory | Custom Kernel |
|----|-------------|-------------|-------------------|-------------------|-----------------|---------------|

One row per op. Absolute paths. NONE where not applicable.
```

## Known operation categories

| Category | LLK Prefix(es) | Compute API Directory | Notes |
|---|---|---|---|
| Elementwise unary | `llk_math_eltwise_unary_sfpu`, `llk_math_eltwise_unary_datacopy` | `tt_metal/hw/inc/api/compute/eltwise_unary/` | Many ops use SFPU macros directly without named LLK function. |
| Binary eltwise | `llk_math_eltwise_binary` | `tt_metal/hw/inc/api/compute/eltwise_binary/` | |
| Ternary SFPU | `llk_math_eltwise_ternary_sfpu` | (some live under `eltwise_unary/`: addcdiv, addcmul, lerp, where) | |
| Matmul | `llk_math_matmul` | `tt_metal/hw/inc/api/compute/matmul/` | |
| Reduce (FPU) | `llk_math_reduce` | `tt_metal/hw/inc/api/compute/reduce/` | |
| Pack | `llk_pack` | `tt_metal/hw/inc/api/compute/pack/` | |
| Unpack | `llk_unpack` | `tt_metal/hw/inc/api/compute/unpack/` | |
| Tilize | `llk_math_fast_tilize`, `llk_math_eltwise_unary_datacopy` | `tt_metal/hw/inc/api/compute/tilize/` | |
