# Breadcrumb Logging - ttnn-operation-architect

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-operation-architect`
- **Predecessor**: `ttnn-operation-analyzer` (or orchestrator)
- **Produces**: Operation Design Document (`op_design.md`) + `.tdd_state.json`

---

## Agent-Specific Events

### reference_read
After reading a reference analysis:
```json
{"event":"reference_read","path":"softmax_analysis.md","role":"input_stage","key_findings":"tilize pattern, sharded input"}
```

### mode_detection
Mode determination:
```json
{"event":"mode_detection","mode":"Hybrid","references":["tilize_analysis.md","reduce_analysis.md"],"roles":["input_stage","compute_core"]}
```

### component_mapping
Component source mapping (Hybrid mode):
```json
{"event":"component_mapping","component":"reader_kernel","source_ref":"tilize_analysis.md","modifications":"none"}
```

### helper_analysis
After analyzing a helper header:
```json
{"event":"helper_analysis","file":"tilize_helpers.hpp","applicable":true,"functions":["tilize()"]}
```

### design_decision
Key design choices:
```json
{"event":"design_decision","phase":"reduce","choice":"USE HELPER","helper":"reduce<AVG, REDUCE_ROW>","rationale":"Helper handles CB ops internally"}
```

### architecture_revision
When Pass 2 revises a Pass 1 decision:
```json
{"event":"architecture_revision","what":"CB c_0 page count","from":2,"to":4,"reason":"reduce helper expects 4 pages per wait"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Reference Analysis Extraction

| Reference | Role | Key Information Extracted |
|-----------|------|---------------------------|
| {path} | {role} | {summary of extracted info} |

### Helper Library Analysis

| Helper File | Read? | Applicable? | Functions Identified |
|-------------|-------|-------------|---------------------|
| tilize_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| untilize_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| reduce_helpers_compute.hpp | YES/NO | YES/NO | {list or "N/A"} |
| binary_op_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| dest_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |

### Architecture Revisions (Pass 2 corrections)

| What Changed | Original (Pass 1) | Revised | Reason |
|--------------|-------------------|---------|--------|
| {e.g., CB page count} | {original value} | {new value} | {helper requirement} |

### Design Decisions

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| {topic} | {options} | {choice} | {why} |
