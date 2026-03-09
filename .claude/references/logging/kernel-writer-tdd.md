# Breadcrumb Logging - ttnn-kernel-writer-tdd

**This document is MANDATORY reading. Breadcrumbs are always enabled.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-kernel-writer-tdd`
- **Predecessor**: `ttnn-generic-op-builder`
- **Session type**: Long-running, multi-stage (all TDD stages in one session)

---

## Why Logging Matters More for This Agent

You are the longest-running agent in the pipeline. You implement multiple TDD stages, debug failures, fix upstream code, and make design judgment calls — all in one session. Without breadcrumbs:
- The self-reflection agent cannot analyze what happened
- Failed runs cannot be debugged post-mortem
- Patterns across operations cannot be identified

**Every stage, every file change, every test run, every debugging step MUST be logged.**

---

## Mandatory Events (MUST Log ALL of These)

### 1. design_parsed — After reading op_design.md

Log once at session start after reading the design:
```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"design_parsed","stages_count":4,"helpers_required":["tilize","reduce","untilize"],"raw_phases":["reader","writer"]}'
```

### 2. stage_start — Before implementing each stage

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"stage_start","stage":"data_pipeline","description":"tilize + untilize identity passthrough"}'
```

### 3. kernel_implemented — After writing/modifying each kernel file

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"kernel_implemented","stage":"data_pipeline","kernel":"reader","approach":"TensorAccessor for RM sticks, push Wt pages per block"}'
```

### 4. upstream_fix — After modifying program descriptor, entry point, or __init__.py

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"upstream_fix","stage":"data_pipeline","file":"layer_norm_rm_program_descriptor.py","change":"fixed CB0 page_size from 2048 to stick_size_bytes","reason":"reader pushes stick-sized pages not tile-sized"}'
```

### 5. cb_sync_check — Before running each test

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"cb_sync_check","stage":"subtract_mean","balanced":true,"cb_summary":"c0:Wt/Wt c8:1/1 c16:Wt/Wt c24:1/1"}'
```

### 6. test_run — After EVERY test run (pass, fail, or hang)

```bash
# Pass
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"test_run","status":"pass","stage":"data_pipeline","test_file":"test_stage_data_pipeline.py","shapes":["1x1x32x32","1x1x64x128"],"hard_attempts":0}'

# Fail
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"test_run","status":"fail","stage":"subtract_mean","failure_type":"numerical","details":"expected -0.5, got 0.0 — subtraction not applied"}'

# Hang
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"test_run","status":"hang","stage":"normalize","triage":"TRISC0 stuck at cb_wait_front(c24)","suspected_cause":"variance CB never pushed"}'
```

### 7. hypothesis — Before making any fix for a failure

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"hypothesis","id":"H1","stage":"normalize","description":"epsilon CB c9 popped by reduce but never repushed for next block","confidence":"HIGH","evidence":"hang at cb_wait_front(c9) on second block"}'
```

### 8. fix_applied — After each code change to fix a failure

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"fix_applied","hypothesis_id":"H1","stage":"normalize","files_modified":["compute.cpp"],"change":"changed c9 policy from WaitAndPop to WaitUpfrontNoPop to preserve epsilon across blocks"}'
```

### 9. stage_complete — After advancing each stage

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"stage_complete","stage":"data_pipeline","attempts":1,"upstream_fixes":["fixed CB0 page_size"],"design_deviations":[]}'
```

### 10. complete — At session end

```bash
.claude/scripts/logging/append_breadcrumb.sh "{op_path}" "ttnn-kernel-writer-tdd" \
  '{"event":"complete","final_status":"ALL_PASSED","stages_completed":["data_pipeline","subtract_mean","normalize","affine"],"total_test_runs":6,"total_upstream_fixes":3}'
```

---

## Example: Full Breadcrumb Trail for One Stage

This shows what a COMPLETE trail looks like for a single stage with one failure:

```jsonl
{"event":"stage_start","stage":"normalize","description":"variance + rsqrt + normalize"}
{"event":"kernel_implemented","stage":"normalize","kernel":"compute","approach":"added mul_tiles for square, reduce for var, add_eps+rsqrt, mul_inv_std COL bcast"}
{"event":"upstream_fix","stage":"normalize","file":"program_descriptor.py","change":"added c9 epsilon CB with WaitUpfrontNoPop policy"}
{"event":"cb_sync_check","stage":"normalize","balanced":true,"cb_summary":"c0:Wt/Wt c8:1/persist c9:1/persist c16:Wt/Wt c24:Wt/Wt c25:1/1"}
{"event":"test_run","status":"hang","stage":"normalize","triage":"TRISC0 stuck at cb_wait_front(c25)","suspected_cause":"variance output CB never pushed"}
{"event":"hypothesis","id":"H1","stage":"normalize","description":"reduce for variance writes to c25 but reduce template output CB is wrong","confidence":"HIGH","evidence":"triage shows compute waiting on c25"}
{"event":"fix_applied","hypothesis_id":"H1","stage":"normalize","files_modified":["compute.cpp"],"change":"changed reduce output CB from c24 to c25 for variance phase"}
{"event":"test_run","status":"pass","stage":"normalize","shapes":["1x1x32x32","1x1x64x128","1x1x32x256","4x2x64x64"],"hard_attempts":1}
{"event":"stage_complete","stage":"normalize","attempts":2,"upstream_fixes":["added c9 epsilon CB"],"design_deviations":[]}
```

**Minimum breadcrumbs per stage (clean pass):** 5 entries (stage_start, kernel_implemented, cb_sync_check, test_run, stage_complete)

**With failures:** Add hypothesis + fix_applied + extra test_run per retry.

---

## Logging Frequency Rule

| Action | Breadcrumb Required? |
|--------|---------------------|
| Read op_design.md | YES — `design_parsed` (once) |
| Start implementing a stage | YES — `stage_start` |
| Write or Edit a kernel file | YES — `kernel_implemented` |
| Write or Edit a .py file | YES — `upstream_fix` |
| Before running a test | YES — `cb_sync_check` |
| After running a test | YES — `test_run` |
| Before making a fix | YES — `hypothesis` |
| After making a fix | YES — `fix_applied` |
| After advancing a stage | YES — `stage_complete` |
| Session complete | YES — `complete` |

---

## Checklist Before Completing

- [ ] `design_parsed` logged at start
- [ ] Every stage has `stage_start` and `stage_complete`
- [ ] Every kernel modification has `kernel_implemented`
- [ ] Every upstream fix has `upstream_fix`
- [ ] Every test run has `test_run` (pass, fail, or hang)
- [ ] Every failure has `hypothesis` before fix and `fix_applied` after
- [ ] `complete` logged at session end
