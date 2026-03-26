# Breadcrumb Logging - ttnn-unary-sfpu-operation-implementor

**This document is MANDATORY reading. Breadcrumbs are always enabled.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-unary-sfpu-operation-implementor`
- **Predecessor**: `ttnn-unary-sfpu-operation-generator`
- **Session type**: Implementation only (12 abstraction layers)

---

## Why Logging Matters for This Agent

You implement SFPU operations across 12 abstraction layers. Without breadcrumbs:
- The self-reflection agent cannot analyze what happened
- The tester agent cannot correlate failures to specific layers
- Patterns across operations cannot be identified (e.g., "layer 7 always has registration bugs")

**Every layer implementation MUST be logged.**

---

## Mandatory Events

### 1. references_parsed — After reading all reference analyses

Log once at session start after reading all reference analysis files:
```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-implementor" \
  '{"event":"references_parsed","op_name":"{op_name}","references_count":5,"references":["sigmoid","exp","relu","gelu","silu"],"math_definition":"alpha * (exp(x) - 1) for x < 0, x for x >= 0"}'
```

### 2. layer_implemented — After implementing each layer

Log after every Write/Edit that implements a layer:
```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-implementor" \
  '{"event":"layer_implemented","layer":1,"layer_name":"SFPU Kernel","files_created":["tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h","tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h"],"approach":"used exp primitive from reference, v_if for x < 0 branch"}'

.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-implementor" \
  '{"event":"layer_implemented","layer":7,"layer_name":"Op Utils Registration","files_modified":["ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"],"functions_updated":["get_macro_definition","get_op_init_and_func_parameterized","get_op_approx_mode"]}'
```

### 3. implementation_complete — After all 12 layers are done

```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-implementor" \
  '{"event":"implementation_complete","op_name":"{op_name}","layers_completed":12,"new_files_count":5,"modified_files_count":7,"has_parameter":true,"parameter_name":"alpha"}'
```

### 4. complete — At session end

```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-implementor" \
  '{"event":"complete","final_status":"IMPLEMENTATION_DONE","layers_completed":12,"new_files_count":5,"modified_files_count":7}'
```

---

## Example: Full Breadcrumb Trail

```jsonl
{"event":"references_parsed","op_name":"elu","references_count":5,"references":["sigmoid","exp","relu","gelu","silu"],"math_definition":"alpha * (exp(x) - 1) for x < 0, x for x >= 0"}
{"event":"layer_implemented","layer":1,"layer_name":"SFPU Kernel","files_created":["...ckernel_sfpu_elu.h (wh)","...ckernel_sfpu_elu.h (bh)"]}
{"event":"layer_implemented","layer":2,"layer_name":"LLK Dispatch","files_created":["...llk_math_eltwise_unary_sfpu_elu.h (wh)","...llk_math_eltwise_unary_sfpu_elu.h (bh)"]}
{"event":"layer_implemented","layer":3,"layer_name":"Compute API Header","files_created":["...elu.h"]}
{"event":"layer_implemented","layer":4,"layer_name":"SFPU Include Guard","files_modified":["...sfpu_split_includes.h"]}
{"event":"layer_implemented","layer":5,"layer_name":"SFPU Type Enum","files_modified":["...llk_sfpu_types.h (wh)","...llk_sfpu_types.h (bh)"]}
{"event":"layer_implemented","layer":6,"layer_name":"UnaryOpType Enum","files_modified":["...unary_op_types.hpp"]}
{"event":"layer_implemented","layer":7,"layer_name":"Op Utils Registration","files_modified":["...unary_op_utils.cpp"]}
{"event":"layer_implemented","layer":8,"layer_name":"Op Utils Header","files_modified":["...unary_op_utils.hpp"]}
{"event":"layer_implemented","layer":9,"layer_name":"C++ API Registration","files_modified":["...unary.hpp"]}
{"event":"layer_implemented","layer":10,"layer_name":"Python Nanobind","files_modified":["...unary_nanobind.cpp"]}
{"event":"layer_implemented","layer":11,"layer_name":"Python Golden Function","files_modified":["...unary.py"]}
{"event":"implementation_complete","op_name":"elu","layers_completed":12,"new_files_count":5,"modified_files_count":7}
{"event":"complete","final_status":"IMPLEMENTATION_DONE","layers_completed":12}
```

**Minimum breadcrumbs: 15** (1 references_parsed + 12 layer_implemented + 1 implementation_complete + 1 complete).

---

## Logging Frequency Rule

| Action | Breadcrumb Required? |
|--------|---------------------|
| Read all reference analyses | YES — `references_parsed` (once) |
| Implement a layer | YES — `layer_implemented` |
| All layers done | YES — `implementation_complete` |
| Session complete | YES — `complete` |

---

## Checklist Before Completing

- [ ] Breadcrumbs initialized at start
- [ ] `references_parsed` logged after reading analyses
- [ ] Every layer has `layer_implemented`
- [ ] `implementation_complete` logged after all layers
- [ ] `complete` logged at session end
