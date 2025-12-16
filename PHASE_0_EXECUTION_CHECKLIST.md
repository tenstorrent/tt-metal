# Phase 0 Execution Checklist

**Purpose:** Pre-implementation validation to reduce mid-project pivot risk from 40% to < 5%  
**Duration:** 5 days  
**Owner:** Senior Engineer  
**Reference:** `/home/tt-admin/tt-metal/PHASE_1_BRIDGE_EXTENSION_PROMPT.md` Part B

---

## Day 1: IPC Performance Baseline + Scheduler Design Start

### Morning (4 hours)

- [ ] **Task 0.2.1:** Set up performance measurement environment
  - Location: `/home/tt-admin/tt-metal/comfyui_bridge/`
  - Deliverable: Test script created
  - Estimated: 1 hour

- [ ] **Task 0.2.2:** Measure full-loop latency baseline
  ```python
  full_loop_latency = measure_denoise_only(steps=20)
  ```
  - Document: Baseline in ms
  - Estimated: 1 hour

- [ ] **Task 0.2.3:** Calculate per-step budget
  ```python
  per_step_budget = full_loop_latency / 20
  ipc_budget = per_step_budget * 0.10
  ```
  - Document: Budget calculations
  - Estimated: 30 minutes

- [ ] **Task 0.2.4:** Measure IPC round-trip latency
  - Tool: Time roundtrip message
  - Expected: 1-5ms
  - Target: < 10ms
  - Estimated: 1.5 hours

### Afternoon (4 hours)

- [ ] **Task 0.1.1:** Review existing scheduler implementations
  - Files to examine:
    - `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` (lines 500-700)
    - ComfyUI scheduler implementations
  - Output: Notes on current approach
  - Estimated: 2 hours

- [ ] **Task 0.1.2:** Draft scheduler sync design document outline
  - Location: `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md`
  - Content: Initial structure and decisions
  - Estimated: 2 hours

### Day 1 Deliverables

- [ ] IPC latency measured and documented
- [ ] Per-step budget calculated
- [ ] Scheduler design outline created

### Day 1 Go/No-Go Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| IPC latency | < 10ms | ___ms | [ ] PASS / [ ] FAIL |

**Day 1 Status:** [ ] PROCEED / [ ] ESCALATE

---

## Day 2: Scheduler Design Completion + Prototype

### Morning (4 hours)

- [ ] **Task 0.1.3:** Complete scheduler sync design document
  - Decision: Option A (Stateless Bridge) or Option B (Stateful)
  - Document rationale
  - Include error handling strategy
  - Estimated: 2.5 hours

- [ ] **Task 0.1.4:** Design review with second engineer
  - Reviewer: _______________
  - Approval: [ ] Approved / [ ] Revisions needed
  - Estimated: 1.5 hours

### Afternoon (4 hours)

- [ ] **Task 0.1.5:** Implement scheduler sync prototype
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`
  - Function: `handle_denoise_step_single` (skeleton)
  - Estimated: 2 hours

- [ ] **Task 0.1.6:** Write basic scheduler sync tests
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_scheduler_sync.py`
  - Tests: 5 basic scenarios
  - Estimated: 2 hours

### Day 2 Deliverables

- [ ] Scheduler sync design document complete
- [ ] Design reviewed and approved
- [ ] Prototype implementation started
- [ ] Basic tests written

### Day 2 Go/No-Go Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Design approved | Yes | [ ] Yes / [ ] No | [ ] PASS / [ ] FAIL |
| Prototype compiles | Yes | [ ] Yes / [ ] No | [ ] PASS / [ ] FAIL |

**Day 2 Status:** [ ] PROCEED / [ ] ESCALATE

---

## Day 3: ControlNet Feasibility - Analysis

### Morning (4 hours)

- [ ] **Task 0.3.1:** Analyze ComfyUI ControlNet implementation
  - File: ComfyUI `comfy/controlnet.py`
  - Question: Where does ControlNet output conditioning?
  - Output: Data flow notes
  - Estimated: 2.5 hours

- [ ] **Task 0.3.2:** Document ControlNet data flow
  - Create: Data flow diagram
  - Include: CPU vs GPU operations
  - Estimated: 1.5 hours

### Afternoon (4 hours)

- [ ] **Task 0.3.3:** Examine TT UNet for control injection point
  - Files to check:
    - `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/`
  - Question: Does TT UNet accept control_hint parameter?
  - Output: Injection point analysis
  - Estimated: 2.5 hours

- [ ] **Task 0.3.4:** Document UNet modification requirements (if any)
  - Output: List of required changes
  - Estimated: 1.5 hours

### Day 3 Deliverables

- [ ] ControlNet data flow documented
- [ ] TT UNet analysis complete
- [ ] Required modifications identified

### Day 3 Status Check

| Question | Answer | Impact |
|----------|--------|--------|
| ControlNet CPU-side works? | [ ] Yes / [ ] No | GO / DEFER |
| TT UNet accepts control_hint? | [ ] Yes / [ ] No / [ ] Needs mod | GO / Modify |
| Estimated modification effort | ___hours | |

**Day 3 Status:** [ ] PROCEED / [ ] ESCALATE

---

## Day 4: ControlNet Feasibility - Prototype

### Morning (4 hours)

- [ ] **Task 0.3.5:** Implement conditioning transfer prototype
  ```python
  def handle_denoise_step_single(params):
      control_hint = params.get("control_hint")
      # Test transfer and injection
  ```
  - Estimated: 3 hours

- [ ] **Task 0.3.6:** Write transfer validation test
  - Test: Data integrity across IPC
  - Test: Shape preservation
  - Estimated: 1 hour

### Afternoon (4 hours)

- [ ] **Task 0.3.7:** Run prototype with simple ControlNet case
  - Input: Canny edge example
  - Output: Verify conditioning reaches UNet
  - Estimated: 2.5 hours

- [ ] **Task 0.3.8:** Document prototype results
  - Success: What worked
  - Issues: What needs addressing
  - Estimated: 1.5 hours

### Day 4 Deliverables

- [ ] Conditioning transfer prototype working
- [ ] Prototype test results documented
- [ ] Issues list created

### Day 4 Go/No-Go Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Control_hint transfers correctly | Yes | [ ] Yes / [ ] No | [ ] PASS / [ ] FAIL |
| Data integrity maintained | Yes | [ ] Yes / [ ] No | [ ] PASS / [ ] FAIL |

**Day 4 Status:** [ ] PROCEED / [ ] ESCALATE

---

## Day 5: Final Validation + Go/No-Go Decision

### Morning (4 hours)

- [ ] **Task 0.4.1:** Run complete scheduler sync test suite
  - Target: 100% pass rate
  - Tests: 20+ step scenarios
  - Estimated: 1.5 hours

- [ ] **Task 0.4.2:** Run ControlNet prototype end-to-end
  - Workflow: LoadImage -> ControlNet -> TT_Sampler
  - Validation: Visual output check
  - Estimated: 1.5 hours

- [ ] **Task 0.4.3:** Compile performance measurements
  - IPC latency summary
  - Headroom analysis
  - Estimated: 1 hour

### Afternoon (4 hours)

- [ ] **Task 0.4.4:** Write Phase 0 Feasibility Report
  - File: `/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md`
  - Sections:
    - Scheduler sync design summary
    - IPC baseline measurements
    - ControlNet feasibility assessment
    - Risk assessment update
  - Estimated: 2 hours

- [ ] **Task 0.4.5:** Prepare Go/No-Go recommendation
  - Decision: GO / CONDITIONAL GO / NO-GO
  - Rationale documented
  - Estimated: 1 hour

- [ ] **Task 0.4.6:** Team review meeting
  - Present findings
  - Make final decision
  - Document in decision log
  - Estimated: 1 hour

### Day 5 Deliverables

- [ ] Feasibility report complete
- [ ] Go/No-Go recommendation documented
- [ ] Risk register updated
- [ ] Decision made and communicated

---

## Phase 0 Go/No-Go Decision Framework

### GO Criteria (ALL must be met)

- [ ] IPC latency < 10ms
- [ ] Scheduler sync design approved
- [ ] ControlNet conditioning transfer validated
- [ ] No blocking issues identified

### CONDITIONAL GO Criteria

- [ ] Minor issues that can be addressed in Week 1
- [ ] ControlNet feasible but needs minor modifications
- [ ] Document constraints for Phase 1

### NO-GO Criteria (ANY triggers)

- [ ] IPC latency > 15ms (significant overhead)
- [ ] ControlNet requires TT-side implementation (deferred)
- [ ] Architecture incompatibility discovered

---

## Final Phase 0 Summary

**Date Completed:** ____________

**Decision:** [ ] GO / [ ] CONDITIONAL GO / [ ] NO-GO

**Approvers:**
- Lead Engineer: ______________ Date: ______
- Technical Lead: _____________ Date: ______

### Key Findings

| Area | Finding | Impact |
|------|---------|--------|
| IPC Performance | | |
| Scheduler Sync | | |
| ControlNet | | |

### Next Steps

If GO:
- [ ] Proceed to Week 1 tasks
- [ ] Team briefing scheduled for: ___________

If CONDITIONAL GO:
- [ ] Constraints documented
- [ ] Adjusted timeline: ___________

If NO-GO:
- [ ] Pivot strategy: ___________
- [ ] Escalation to: ___________

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
