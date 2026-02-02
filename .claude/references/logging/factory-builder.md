# Breadcrumb Logging - ttnn-factory-builder

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-factory-builder`
- **Predecessor**: `ttnn-operation-scaffolder`
- **Stages owned**: 4, 5, 6

---

## Agent-Specific Events

### cb_config
CB configuration decisions:
```json
{"event":"cb_config","cb_id":"c_0","page_size":2048,"num_pages":2,"purpose":"input"}
```

### work_distribution
Work split calculations:
```json
{"event":"work_distribution","grid":"8x8","total_tiles":256,"tiles_per_core":4}
```

### tdd_cycle
TDD phase tracking:
```json
{"event":"tdd_cycle","stage":4,"phase":"RED","result":"FAIL","expected":true}
{"event":"tdd_cycle","stage":6,"phase":"GREEN","result":"PASS"}
```

### cb_audit
CB sync verification:
```json
{"event":"cb_audit","cb_id":"c_0","producer":"reader","push_count":"N","consumer":"compute","pop_count":"N","balanced":true}
```

### cb_sync_summary
Final CB balance check (**CRITICAL - log before completing Stage 6**):
```json
{"event":"cb_sync_summary","total_cbs":2,"all_balanced":true}
```

### hang_debug
When debugging hangs:
```json
{"event":"hang_debug","symptom":"timeout","diagnosis":"CB sync mismatch"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Circular Buffer Configuration

| CB ID | Index | Page Size | Num Pages | Data Type | Purpose | Source |
|-------|-------|-----------|-----------|-----------|---------|--------|
| cb_in | c_0 | {bytes} | {count} | {dtype} | {purpose} | Spec/Inferred |

### CB Synchronization Verification (CRITICAL)

| CB | Producer | Push Operation | Consumer | Pop Operation | Balanced? |
|----|----------|----------------|----------|---------------|-----------|
| c_0 | Reader | cb_push_back x N | Compute | cb_pop_front x N | YES/NO |
| c_16 | Compute | cb_push_back x M | Writer | cb_pop_front x M | YES/NO |

**Total tiles through pipeline**:
- Reader pushes: {total} tiles
- Compute pops/pushes: {total} / {total}
- Writer pops: {total} tiles

### Work Distribution

| Parameter | Value | Source |
|-----------|-------|--------|
| Core grid | {rows x cols} | Spec/Calculated |
| Total work units | {count} | |
| Work per core | {formula} | |

### Stub Kernel Summary

| Kernel | File | CB In | CB Out |
|--------|------|-------|--------|
| Reader | {path} | N/A | {cb_ids} |
| Compute | {path} | {cb_ids} | {cb_ids} |
| Writer | {path} | {cb_ids} | N/A |
