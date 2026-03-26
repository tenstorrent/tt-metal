# Breadcrumb Logging - ttnn-unary-sfpu-operation-tester

**This document is MANDATORY reading. Breadcrumbs are always enabled.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-unary-sfpu-operation-tester`
- **Predecessor**: `ttnn-unary-sfpu-operation-implementor`
- **Session type**: Testing and debugging (create test → run → diagnose → fix → iterate)

---

## Why Logging Matters for This Agent

You test SFPU operations and iterate through failure-fix cycles. Without breadcrumbs:
- The self-reflection agent cannot analyze what happened during debugging
- Failed runs cannot be debugged post-mortem (what was the error? what fix was tried? did it help?)
- Patterns across operations cannot be identified (e.g., "hang always caused by missing SfpuType")

**Every test run, every hypothesis, every fix MUST be logged.**

---

## Mandatory Events

### 1. notes_parsed — After reading implementation notes

Log once at session start after reading the implementor's notes:
```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"notes_parsed","op_name":"{op_name}","new_files_count":5,"modified_files_count":7,"has_parameter":true,"known_limitations":"precision loss for large negative values"}'
```

### 2. test_created — After creating the test file

```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_created","test_file":"tests/ttnn/unit_tests/operations/eltwise/test_elu.py","test_type":"exhaustive_bfloat16","ulp_threshold":2,"param_values":[0.1,0.5,1.0,2.0]}'
```

### 3. test_run — After EVERY test run (pass, fail, or hang)

```bash
# Pass
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_run","status":"pass","test_file":"test_elu.py","max_ulp":1.5,"allclose_pass":true,"attempt":1}'

# Fail — build error
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_run","status":"fail","test_file":"test_elu.py","failure_type":"build_error","details":"undefined reference to elu_tile_init in eltwise_sfpu.cpp","attempt":1}'

# Fail — numerical
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_run","status":"fail","test_file":"test_elu.py","failure_type":"numerical_error","details":"max ULP 15.0 exceeds threshold 2, worst at x=-0.5","attempt":2}'

# Fail — runtime
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_run","status":"fail","test_file":"test_elu.py","failure_type":"runtime_error","details":"RuntimeError: Unknown unary op type ELU","attempt":1}'

# Hang
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"test_run","status":"hang","test_file":"test_elu.py","suspected_cause":"missing SfpuType entry in llk_sfpu_types.h","attempt":1}'
```

### 4. hypothesis — Before making any fix for a failure

```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"hypothesis","id":"H1","description":"get_macro_definition returns SFPU_OP_ELU but sfpu_split_includes.h uses SFPU_OP_ELU_INCLUDE","confidence":"HIGH","evidence":"build error: undefined reference to elu_tile_init"}'
```

### 5. fix_applied — After each code change to fix a failure

```bash
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"fix_applied","hypothesis_id":"H1","files_modified":["ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"],"change":"fixed get_macro_definition to return SFPU_OP_ELU_INCLUDE instead of SFPU_OP_ELU"}'
```

### 6. complete — At session end

```bash
# Tests passed
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"complete","final_status":"TESTS_PASSED","total_test_runs":3,"total_fixes":2,"max_ulp":1.5,"allclose_pass":true}'

# Budget exhausted
.claude/scripts/logging/append_breadcrumb.sh "{output_folder}" "ttnn-unary-sfpu-operation-tester" \
  '{"event":"complete","final_status":"BUDGET_EXHAUSTED","total_test_runs":5,"total_fixes":4,"last_error":"numerical_error: max ULP 8.0 exceeds threshold 2"}'
```

---

## Example: Full Breadcrumb Trail (with one failure)

```jsonl
{"event":"notes_parsed","op_name":"elu","new_files_count":5,"modified_files_count":7,"has_parameter":true}
{"event":"test_created","test_file":"tests/ttnn/unit_tests/operations/eltwise/test_elu.py","test_type":"exhaustive_bfloat16"}
{"event":"test_run","status":"fail","failure_type":"build_error","details":"undefined reference to elu_tile_init","attempt":1}
{"event":"hypothesis","id":"H1","description":"wrong include guard macro name in get_macro_definition","confidence":"HIGH"}
{"event":"fix_applied","hypothesis_id":"H1","files_modified":["unary_op_utils.cpp"],"change":"SFPU_OP_ELU -> SFPU_OP_ELU_INCLUDE"}
{"event":"test_run","status":"pass","max_ulp":1.5,"allclose_pass":true,"attempt":2}
{"event":"complete","final_status":"TESTS_PASSED","total_test_runs":2,"total_fixes":1,"max_ulp":1.5}
```

**Minimum breadcrumbs (clean pass): 4** (notes_parsed + test_created + test_run + complete).
**With failures**: Add hypothesis + fix_applied + extra test_run per retry.

---

## Logging Frequency Rule

| Action | Breadcrumb Required? |
|--------|---------------------|
| Read implementation notes | YES — `notes_parsed` (once) |
| Create test file | YES — `test_created` |
| After running a test | YES — `test_run` |
| Before making a fix | YES — `hypothesis` |
| After making a fix | YES — `fix_applied` |
| Session complete | YES — `complete` |

---

## Checklist Before Completing

- [ ] Breadcrumbs initialized at start
- [ ] `notes_parsed` logged after reading implementation notes
- [ ] `test_created` logged after creating test file
- [ ] Every test run has `test_run` (pass, fail, or hang)
- [ ] Every failure has `hypothesis` before fix and `fix_applied` after
- [ ] `complete` logged at session end
