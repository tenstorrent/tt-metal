# Breadcrumb Logging - ttnn-operation-planner

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-operation-planner`
- **Predecessor**: `ttnn-operation-analyzer` (or orchestrator)
- **Produces**: Functional Specification (`*_spec.md`)

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

### interface_check
Interface compatibility (Hybrid mode):
```json
{"event":"interface_check","from":"ref1.reader","to":"ref2.compute","compatible":true}
```

### deepwiki_query
DeepWiki consultation:
```json
{"event":"deepwiki_query","question":"Which operations convert ROW_MAJOR to TILE?","findings":"tilize, tilize_with_val_padding"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Reference Analysis Extraction

| Reference | Role | Key Information Extracted |
|-----------|------|---------------------------|
| {path} | {role} | {summary of extracted info} |

### Component Mapping (Hybrid Mode)

| Component | Source Reference | Modifications Needed |
|-----------|-----------------|---------------------|
| Reader kernel | {ref} | {mods or "None"} |
| Compute phase | {ref} | {mods} |
| Writer kernel | {ref} | {mods or "None"} |

### Interface Compatibility (Hybrid Mode)

| Interface | From | To | Compatible? | Notes |
|-----------|------|-----|-------------|-------|
| Reader→Compute | {comp} | {comp} | YES/NO | {details} |
| Compute→Writer | {comp} | {comp} | YES/NO | {details} |

### DeepWiki Queries

| Query | Findings | How Used |
|-------|----------|----------|
| {question} | {summary} | {impact on design} |

### Design Decisions

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| {topic} | {options} | {choice} | {why} |
