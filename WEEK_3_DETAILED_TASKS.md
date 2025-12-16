# Week 3: ControlNet Implementation - Detailed Tasks

**Goal:** Enable ControlNet workflows through bridge extension  
**Duration:** 5 working days  
**Prerequisite:** Week 2 complete, Phase 0 ControlNet GO decision  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part C, Week 3

---

## Monday (Day 1): ControlNet Integration Design

### Morning (4 hours)

- [ ] **Task 3.1.1:** Review Phase 0 ControlNet findings (1h)
  - Document: `/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md`
  - Confirm: CPU-side approach validated
  - Note: Any constraints identified
  - Deliverable: Findings summary

- [ ] **Task 3.1.2:** Design control_hint parameter spec (1.5h)
  - Format: Expected tensor shape
  - Type: Data type requirements
  - Transfer: SHM handle specification
  - Deliverable: Parameter specification

- [ ] **Task 3.1.3:** Design conditioning injection point (1.5h)
  - Location: In handle_denoise_step_single
  - Method: How control_hint reaches UNet
  - Deliverable: Injection design doc

### Afternoon (4 hours)

- [ ] **Task 3.1.4:** Document data flow diagram (1.5h)
  - Flow: ComfyUI -> ControlNet -> Bridge -> UNet
  - Include: SHM transfer points
  - Deliverable: Data flow diagram

- [ ] **Task 3.1.5:** Design review meeting (1h)
  - Reviewer: _______________
  - Focus: Control injection approach
  - Status: [ ] Approved / [ ] Changes needed
  - Deliverable: Review notes

- [ ] **Task 3.1.6:** Create test plan for ControlNet (1.5h)
  - Tests: Canny, Depth, OpenPose
  - Validation: SSIM targets
  - Human validation: Process
  - Deliverable: Test plan document

**Monday Success Criteria:**
- [ ] Design approved
- [ ] Data flow documented
- [ ] Test plan created

---

## Tuesday (Day 2): Control Hint Injection Implementation

### Morning (4 hours)

- [ ] **Task 3.1.7:** Implement control_hint parameter handling (2h)
  - Parse: control_hint_shm from params
  - Validate: Tensor shape and type
  - Convert: To TT format if needed
  - Deliverable: Parameter handling code

- [ ] **Task 3.1.8:** Implement _prepare_control_hint helper (1.5h)
  - Function: Convert torch tensor to TT format
  - Handle: Shape adjustments if needed
  - Deliverable: Helper function

- [ ] **Task 3.1.9:** Add logging for ControlNet operations (0.5h)
  - Log: Control hint received
  - Log: Shape and step info
  - Deliverable: Logging added

### Afternoon (4 hours)

- [ ] **Task 3.1.10:** Integrate control_hint into UNet call (2h)
  - Pass: control_hint to runner.denoise_step
  - Handle: None case (no ControlNet)
  - Deliverable: Integration complete

- [ ] **Task 3.1.11:** Write control hint transfer test (1h)
  - Test: Data integrity across IPC
  - Test: Shape preservation
  - Deliverable: Transfer test passing

- [ ] **Task 3.1.12:** Test basic injection (1h)
  - Test: Control hint reaches UNet
  - Verify: Logging shows correct flow
  - Deliverable: Basic injection verified

**Tuesday Success Criteria:**
- [ ] Control hint parsing implemented
- [ ] Integration with UNet complete
- [ ] Transfer test passing

---

## Wednesday (Day 3): ComfyUI ControlNet Wrapper Node

### Morning (4 hours)

- [ ] **Task 3.2.1:** Design TT_ApplyControlNet node (1.5h)
  - Inputs: conditioning, control_net, image, strength
  - Output: Modified conditioning
  - Deliverable: Node design doc

- [ ] **Task 3.2.2:** Implement TT_ApplyControlNet node (2h)
  - File: `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`
  - Function: apply_controlnet method
  - Deliverable: Node implementation

- [ ] **Task 3.2.3:** Implement control_hint extraction (0.5h)
  - Extract: From ComfyUI ControlNet output
  - Attach: To conditioning for TT_KSampler
  - Deliverable: Extraction logic

### Afternoon (4 hours)

- [ ] **Task 3.2.4:** Update TT_KSampler for ControlNet (1.5h)
  - Add: control_hint optional input
  - Pass: To bridge denoise_step_single
  - Deliverable: KSampler updated

- [ ] **Task 3.2.5:** Implement strength parameter handling (1h)
  - Parameter: control_strength
  - Apply: Scale control_hint by strength
  - Deliverable: Strength handling

- [ ] **Task 3.2.6:** Test node registration (0.5h)
  - Verify: Node appears in ComfyUI
  - Verify: Correct input/output types
  - Deliverable: Registration verified

- [ ] **Task 3.2.7:** Document node usage (1h)
  - Document: How to use TT_ApplyControlNet
  - Document: Workflow examples
  - Deliverable: Usage documentation

**Wednesday Success Criteria:**
- [ ] TT_ApplyControlNet node implemented
- [ ] TT_KSampler updated
- [ ] Nodes visible in ComfyUI

---

## Thursday (Day 4): ControlNet Integration Testing

### Morning (4 hours)

- [ ] **Task 3.3.1:** Create Canny edge test workflow (1.5h)
  - Workflow: LoadImage -> CannyEdge -> TT_ApplyControlNet -> TT_KSampler
  - Test image: Standard Canny test image
  - Deliverable: Canny workflow file

- [ ] **Task 3.3.2:** Run Canny test and measure SSIM (1.5h)
  - Run: TT pipeline with Canny ControlNet
  - Compare: To CPU reference
  - Target: SSIM >= 0.90
  - Deliverable: Canny test results

- [ ] **Task 3.3.3:** Create Depth ControlNet test (1h)
  - Workflow: Depth estimation -> TT_ApplyControlNet -> TT_KSampler
  - Deliverable: Depth workflow file

### Afternoon (4 hours)

- [ ] **Task 3.3.4:** Run Depth test and measure SSIM (1h)
  - Target: SSIM >= 0.90
  - Deliverable: Depth test results

- [ ] **Task 3.3.5:** Create OpenPose test workflow (1h)
  - Workflow: Pose detection -> TT_ApplyControlNet -> TT_KSampler
  - Deliverable: OpenPose workflow file

- [ ] **Task 3.3.6:** Run OpenPose test and measure SSIM (1h)
  - Target: SSIM >= 0.90
  - Deliverable: OpenPose test results

- [ ] **Task 3.3.7:** Document all test results (1h)
  - Create: Test results table
  - Include: SSIM scores for each type
  - Deliverable: Test results document

**Thursday Success Criteria:**
- [ ] All 3 ControlNet types tested
- [ ] SSIM >= 0.90 for all types
- [ ] Results documented

---

## Friday (Day 5): Human Validation + Multi-ControlNet

### Morning (4 hours)

- [ ] **Task 3.3.8:** Prepare human validation samples (1h)
  - Generate: 5 samples per ControlNet type
  - Create: Reference images (CPU)
  - Create: TT images
  - Deliverable: Validation image set

- [ ] **Task 3.3.9:** Conduct human validation review (1.5h)
  - Raters: 5 people
  - Question: "Does control influence appear correct?"
  - Target: 5/5 confirm correct
  - Deliverable: Human validation results

- [ ] **Task 3.3.10:** Document validation results (0.5h)
  - Document: Rater feedback
  - Document: Pass/fail status
  - Deliverable: Validation report

- [ ] **Task 3.4.1:** Design multi-ControlNet support (1h)
  - Design: Multiple control_hint handling
  - Design: Aggregation strategy
  - Deliverable: Multi-ControlNet design

### Afternoon (4 hours)

- [ ] **Task 3.4.2:** Implement multi-ControlNet (if time permits) (2h)
  - Support: Up to 10 ControlNets
  - Implement: Control hint combination
  - Deliverable: Multi-ControlNet support (optional)

- [ ] **Task 3.4.3:** Run full test suite (1h)
  - Run: All Week 1-3 tests
  - Fix: Any regressions
  - Deliverable: All tests passing

- [ ] **Task 3.4.4:** Week 3 review and planning (1h)
  - Review: ControlNet implementation status
  - Document: Issues or gaps
  - Plan: Week 4 focus areas
  - Deliverable: Week 3 summary

**Friday Success Criteria:**
- [ ] Human validation passed (5/5)
- [ ] All tests passing
- [ ] Week 3 complete

---

## Week 3 Summary

### Deliverables Checklist

- [ ] ControlNet conditioning injection working
- [ ] TT_ApplyControlNet node implemented
- [ ] TT_KSampler updated with control support
- [ ] 3 ControlNet types tested (Canny, Depth, OpenPose)
- [ ] SSIM >= 0.90 vs CPU reference for all types
- [ ] Human validation passed (5/5 raters confirm correct)
- [ ] Multi-ControlNet support (optional)

### Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Canny SSIM | >= 0.90 | | |
| Depth SSIM | >= 0.90 | | |
| OpenPose SSIM | >= 0.90 | | |
| Human validation | 5/5 correct | | |

### Hours Summary

| Day | Estimated | Actual |
|-----|-----------|--------|
| Monday | 8h | |
| Tuesday | 8h | |
| Wednesday | 8h | |
| Thursday | 8h | |
| Friday | 8h | |
| **Total** | **40h** | |

### ControlNet Test Results

| ControlNet Type | Test Images | SSIM Score | Human Validation | Status |
|-----------------|-------------|------------|------------------|--------|
| Canny | 10 | | | |
| Depth | 10 | | | |
| OpenPose | 10 | | | |

### Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| Per-step API | Complete | |
| Session management | Complete | |
| ControlNet injection | | |
| ComfyUI nodes | | |

### Week 4 Readiness

- [ ] Week 3 tasks complete
- [ ] ControlNet validation passed
- [ ] No blockers for Week 4
- [ ] Test suite comprehensive

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
