# Performance Budget Tracker

**Purpose:** Track latency budget and performance metrics throughout Phase 1  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part E

---

## Budget Overview

### Per-Step Latency Budget

| Component | Budget | Notes |
|-----------|--------|-------|
| UNet forward pass | ~90ms | Baseline from full-loop / steps |
| IPC overhead | < 10ms | 10% of per-step budget |
| Format conversion | < 2ms | Should be minimal |
| Session overhead | < 1ms | Lookup and update |
| ControlNet injection | < 2ms | If applicable |
| **Total per-step** | **< 105ms** | **With 10% headroom** |

### Calculation Method

```python
# Budget Calculation
full_loop_time = measure_denoise_only(steps=20)  # e.g., 2000ms
per_step_baseline = full_loop_time / 20          # e.g., 100ms

# Overhead budget
overhead_budget_percent = 10
overhead_budget_ms = per_step_baseline * (overhead_budget_percent / 100)  # e.g., 10ms

# Maximum acceptable per-step time
max_per_step = per_step_baseline + overhead_budget_ms  # e.g., 110ms
```

---

## Baseline Measurements

### Phase 0 Day 1 Baseline

| Measurement | Value | Date | Notes |
|-------------|-------|------|-------|
| Full-loop (20 steps) | ___ms | | |
| Per-step baseline | ___ms | | |
| IPC round-trip | ___ms | | |
| IPC P99 | ___ms | | |
| Overhead budget | ___ms | | |
| Headroom | ___% | | |

**Environment:**
- Hardware: _______________
- TT-Metal version: _______________
- Python version: _______________

---

## Weekly Performance Measurements

### Week 1 Measurements

| Day | Metric | Value | Budget | Status |
|-----|--------|-------|--------|--------|
| Mon | Per-step latency | | | |
| Tue | Per-step latency | | | |
| Wed | Per-step latency | | | |
| Thu | Per-step latency | | | |
| Fri | Per-step latency | | | |

**Week 1 Summary:**
- Average per-step: ___ms
- Overhead vs baseline: ___% 
- Status: [ ] Within budget [ ] At risk [ ] Over budget

### Week 2 Measurements

| Day | Metric | Value | Budget | Status |
|-----|--------|-------|--------|--------|
| Mon | Per-step + session | | | |
| Tue | Per-step + session | | | |
| Wed | Per-step + session | | | |
| Thu | Per-step + session | | | |
| Fri | Per-step + session | | | |

**Week 2 Summary:**
- Average per-step: ___ms
- Session overhead: ___ms
- Overhead vs baseline: ___%
- Status: [ ] Within budget [ ] At risk [ ] Over budget

### Week 3 Measurements

| Day | Metric | Value | Budget | Status |
|-----|--------|-------|--------|--------|
| Mon | Per-step + ControlNet | | | |
| Tue | Per-step + ControlNet | | | |
| Wed | Per-step + ControlNet | | | |
| Thu | Per-step + ControlNet | | | |
| Fri | Per-step + ControlNet | | | |

**Week 3 Summary:**
- Average per-step: ___ms
- ControlNet overhead: ___ms
- Overhead vs baseline: ___%
- Status: [ ] Within budget [ ] At risk [ ] Over budget

### Week 4 Measurements

| Test | Per-step Avg | Overhead % | Status |
|------|--------------|------------|--------|
| 20 steps | | | |
| 30 steps | | | |
| 50 steps | | | |
| With ControlNet | | | |

**Week 4 Summary:**
- Final per-step: ___ms
- Final overhead: ___%
- Status: [ ] PASS [ ] FAIL

### Week 5 Final Measurements

| Test | Latency | Overhead | Target | Status |
|------|---------|----------|--------|--------|
| 1000 gen stress test | | | | |
| Average per-step | | | < 10% | |
| P99 per-step | | | < 15% | |

---

## Headroom Tracking

### Headroom Over Time

| Week | Baseline | Actual | Overhead % | Headroom % |
|------|----------|--------|------------|------------|
| Phase 0 | ___ms | N/A | N/A | N/A |
| Week 1 | | | | |
| Week 2 | | | | |
| Week 3 | | | | |
| Week 4 | | | | |
| Week 5 | | | | |

### Headroom Status

```
Target overhead: 10%
Warning threshold: 8%
Critical threshold: 9%

Current overhead: ___%
Headroom remaining: ___%

Status: [ ] Green (< 8%)  [ ] Yellow (8-9%)  [ ] Red (>= 9%)
```

---

## Component Breakdown

### Latency by Component

| Component | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 |
|-----------|--------|--------|--------|--------|--------|
| UNet forward | | | | | |
| IPC transfer | | | | | |
| Format conversion | | | | | |
| Session management | | | | | |
| ControlNet injection | N/A | N/A | | | |
| **Total** | | | | | |

### Contribution Analysis

```
Week ___ Component Breakdown:

UNet Forward:       [####################] ___%
IPC Transfer:       [####                ] ___%
Format Conversion:  [##                  ] ___%
Session Mgmt:       [#                   ] ___%
ControlNet:         [###                 ] ___%
```

---

## Optimization Opportunities

### Identified Optimizations

| Optimization | Expected Savings | Effort | Priority | Status |
|--------------|-----------------|--------|----------|--------|
| | | | | |

### Applied Optimizations

| Optimization | Before | After | Savings | Date |
|--------------|--------|-------|---------|------|
| | | | | |

---

## Memory Budget

### Memory Tracking

| Week | Initial (MB) | Final (MB) | Growth (MB) | Status |
|------|--------------|------------|-------------|--------|
| Week 1 | | | | |
| Week 2 | | | | |
| Week 3 | | | | |
| Week 4 | | | | |
| Week 5 | | | | |

**Stress Test (1000 gen):**
- Initial: ___MB
- Final: ___MB
- Growth: ___MB
- Growth per gen: ___KB
- Status: [ ] PASS (< 100MB) [ ] FAIL

---

## Benchmark Commands

### Daily Benchmark

```bash
# Quick per-step benchmark
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py --quick

# Output format:
# Per-step avg: X.XXms
# Overhead: X.X%
# Status: PASS/FAIL
```

### Full Benchmark

```bash
# Complete benchmark suite
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py --full

# Includes:
# - Multiple step counts
# - Multiple iterations
# - Component breakdown
# - Memory tracking
```

### Memory Benchmark

```bash
# Memory stability test
python /home/tt-admin/tt-metal/comfyui_bridge/tests/stress_test.py --memory

# Output format:
# Initial: XXX MB
# Final: XXX MB
# Growth: XX MB
# Status: PASS/FAIL
```

---

## Alert Thresholds

### Latency Alerts

| Level | Threshold | Action |
|-------|-----------|--------|
| Green | < 8% overhead | Continue as planned |
| Yellow | 8-9% overhead | Monitor closely, identify optimizations |
| Red | >= 9% overhead | Immediate optimization, consider target adjustment |

### Memory Alerts

| Level | Threshold | Action |
|-------|-----------|--------|
| Green | < 50MB growth / 1000 gen | Continue as planned |
| Yellow | 50-100MB growth / 1000 gen | Investigate, plan fix |
| Red | > 100MB growth / 1000 gen | Immediate fix required |

---

## Sign-Off

### Final Performance Sign-Off

**Date:** _______________

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Per-step overhead | < 10% | ___% | [ ] PASS [ ] FAIL |
| Memory growth (1000 gen) | < 100MB | ___MB | [ ] PASS [ ] FAIL |
| P99 latency | < 15% overhead | ___% | [ ] PASS [ ] FAIL |

**Sign-Off:** [ ] APPROVED  [ ] CONDITIONAL  [ ] REJECTED

**Signed:** _______________ Date: ______

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
