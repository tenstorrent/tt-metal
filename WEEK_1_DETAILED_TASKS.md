# Week 1: Per-Timestep API Foundation - Detailed Tasks

**Goal:** Implement `handle_denoise_step_single` with model-agnostic design  
**Duration:** 5 working days  
**Prerequisite:** Phase 0 GO decision  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part C, Week 1

---

## Monday (Day 1): Per-Step API Design

### Morning (4 hours)

- [ ] **Task 1.1.1:** Review existing `handle_denoise_only` implementation (1.5h)
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`
  - Lines: ~500-700
  - Document: Current flow and extraction points
  - Deliverable: Notes on refactoring approach

- [ ] **Task 1.1.2:** Design `handle_denoise_step_single` signature (1.5h)
  - Input parameters specification
  - Output format specification
  - Error handling approach
  - Deliverable: Function signature document

- [ ] **Task 1.1.3:** Identify shared memory interfaces needed (1h)
  - Latent SHM handles
  - Conditioning SHM handles
  - Control hint SHM handles (optional)
  - Deliverable: SHM interface specification

### Afternoon (4 hours)

- [ ] **Task 1.1.4:** Write API design document (2h)
  - File: `/home/tt-admin/tt-metal/docs/api/per_step_api_design.md`
  - Include: Full parameter spec with types
  - Include: Example request/response
  - Deliverable: Design document draft

- [ ] **Task 1.1.5:** Design review meeting (1h)
  - Reviewer: _______________
  - Status: [ ] Approved / [ ] Changes needed
  - Deliverable: Review notes

- [ ] **Task 1.1.6:** Begin skeleton implementation (1h)
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`
  - Function: Empty `handle_denoise_step_single`
  - Deliverable: Skeleton committed

**Monday Success Criteria:**
- [ ] API design documented
- [ ] Design reviewed
- [ ] Skeleton function created

---

## Tuesday (Day 2): Per-Step API Implementation

### Morning (4 hours)

- [ ] **Task 1.1.7:** Extract single-step logic from `handle_denoise_only` (3h)
  - Identify: Loop body
  - Extract: Single iteration logic
  - Preserve: Original function (no breakage)
  - Deliverable: Extracted logic in new function

- [ ] **Task 1.1.8:** Implement parameter extraction (1h)
  - Parse: session_id, timestep, sigma, timestep_index
  - Parse: latent_shm, conditioning_shm
  - Parse: Optional control_hint_shm
  - Deliverable: Parameter parsing code

### Afternoon (4 hours)

- [ ] **Task 1.1.9:** Implement timestep/sigma handling (2h)
  - Receive: External scheduler state
  - Pass: To UNet forward call
  - No internal scheduling
  - Deliverable: Scheduler state handling

- [ ] **Task 1.1.10:** Implement output format standardization (1.5h)
  - Output: Standard [B, C, H, W] format
  - Include: Step metadata (timing, diagnostics)
  - Deliverable: Output formatting code

- [ ] **Task 1.1.11:** Write basic unit test (0.5h)
  - Test: `test_denoise_step_single_basic`
  - Verify: Function executes without error
  - Deliverable: Test file created

**Tuesday Success Criteria:**
- [ ] Single-step logic extracted
- [ ] Parameters parsed correctly
- [ ] Basic test passing

---

## Wednesday (Day 3): Model-Agnostic Refactoring

### Morning (4 hours)

- [ ] **Task 1.2.1:** Create model configuration module (2h)
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py`
  - Content: MODEL_CONFIGS dictionary
  - Models: sdxl, sd35, sd14 (initially)
  - Deliverable: Config module created

- [ ] **Task 1.2.2:** Update channel validation to use config (1h)
  - Remove: Hardcoded `C != 4` checks
  - Add: Config-based channel lookup
  - Deliverable: Handlers updated

- [ ] **Task 1.2.3:** Update format conversion helpers (1h)
  - Function: `_detect_and_convert_tt_to_standard_format`
  - Change: Make expected_channels required parameter
  - Add: model_type parameter for logging
  - Deliverable: Helper function updated

### Afternoon (4 hours)

- [ ] **Task 1.2.4:** Audit all call sites for format conversion (1.5h)
  - Check: handle_denoise_only
  - Check: handle_denoise_step_single
  - Check: handle_vae_decode
  - Deliverable: All call sites updated

- [ ] **Task 1.2.5:** Write model-agnostic tests (1.5h)
  - Test: Config lookup for all model types
  - Test: Channel validation with different models
  - Deliverable: Tests passing

- [ ] **Task 1.2.6:** Integration test setup (1h)
  - Test: Per-step with SDXL config
  - Verify: Correct channels used
  - Deliverable: Integration test framework

**Wednesday Success Criteria:**
- [ ] Model config module created
- [ ] All hardcoded values removed
- [ ] Config-based tests passing

---

## Thursday (Day 4): Scheduler State Implementation

### Morning (4 hours)

- [ ] **Task 1.3.1:** Implement stateless bridge pattern (2.5h)
  - Receive: timestep from ComfyUI
  - Receive: sigma from ComfyUI
  - Receive: timestep_index from ComfyUI
  - No: Internal scheduler state
  - Deliverable: Stateless implementation

- [ ] **Task 1.3.2:** Add validation for scheduler parameters (1h)
  - Validate: timestep in valid range
  - Validate: sigma positive
  - Validate: timestep_index >= 0
  - Deliverable: Validation code

- [ ] **Task 1.3.3:** Implement error handling for invalid state (0.5h)
  - Handle: Missing parameters
  - Handle: Invalid values
  - Return: Clear error messages
  - Deliverable: Error handling code

### Afternoon (4 hours)

- [ ] **Task 1.3.4:** Write scheduler sync tests (2h)
  - Test: 20-step sequence with explicit timesteps
  - Test: Various sigma values
  - Test: Error cases
  - Deliverable: Scheduler sync test suite

- [ ] **Task 1.3.5:** Integration test: per-step vs full-loop (1.5h)
  - Run: 20 individual steps
  - Run: Single denoise_only call
  - Compare: Output SSIM
  - Target: SSIM >= 0.99
  - Deliverable: Comparison test

- [ ] **Task 1.3.6:** Document scheduler synchronization (0.5h)
  - Update: Scheduler sync design doc
  - Document: Implementation decisions
  - Deliverable: Documentation updated

**Thursday Success Criteria:**
- [ ] Stateless bridge implemented
- [ ] Scheduler sync tests passing
- [ ] Per-step vs full-loop SSIM >= 0.99

---

## Friday (Day 5): Session Framework + Week Wrap-up

### Morning (4 hours)

- [ ] **Task 1.5.1:** Create session data class (1h)
  - File: `/home/tt-admin/tt-metal/comfyui_bridge/session_manager.py`
  - Class: DenoiseSession dataclass
  - Fields: session_id, model_id, timestamps, step tracking
  - Deliverable: Session dataclass created

- [ ] **Task 1.5.2:** Create session manager class (1.5h)
  - Class: SessionManager
  - Methods: create_session, get_session, update_session
  - Deliverable: Session manager skeleton

- [ ] **Task 1.5.3:** Implement session creation in handler (1h)
  - Add: Session creation on first step
  - Add: Session lookup on subsequent steps
  - Deliverable: Session integration

- [ ] **Task 1.5.4:** Write session tests (0.5h)
  - Test: Session creation
  - Test: Session retrieval
  - Test: Session update
  - Deliverable: Session tests

### Afternoon (4 hours)

- [ ] **Task 1.4.1:** Complete format conversion audit (1h)
  - Verify: All paths use config-based channels
  - Verify: No hardcoded values remain
  - Deliverable: Audit checklist complete

- [ ] **Task 1.4.2:** Run full test suite (1h)
  - Run: All unit tests
  - Run: All integration tests
  - Fix: Any failures
  - Deliverable: All tests green

- [ ] **Task 1.4.3:** Code cleanup and documentation (1h)
  - Add: Reusability comments to new code
  - Add: Docstrings to all new functions
  - Deliverable: Code documented

- [ ] **Task 1.4.4:** Week 1 review and planning (1h)
  - Review: All tasks complete
  - Document: Blockers or delays
  - Plan: Week 2 adjustments if needed
  - Deliverable: Week 1 summary

**Friday Success Criteria:**
- [ ] Session framework created
- [ ] All tests passing
- [ ] Code documented
- [ ] Week 1 complete

---

## Week 1 Summary

### Deliverables Checklist

- [ ] `handle_denoise_step_single` implemented and tested
- [ ] Model-agnostic config system working
- [ ] Scheduler state synchronization implemented
- [ ] Session framework skeleton created
- [ ] SSIM >= 0.99 vs full-loop baseline achieved
- [ ] All unit tests passing

### Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Per-step vs full-loop SSIM | >= 0.99 | | |
| Unit tests passing | 100% | | |
| Code coverage | >= 80% | | |

### Hours Summary

| Day | Estimated | Actual |
|-----|-----------|--------|
| Monday | 8h | |
| Tuesday | 8h | |
| Wednesday | 8h | |
| Thursday | 8h | |
| Friday | 8h | |
| **Total** | **40h** | |

### Issues/Blockers Identified

| Issue | Severity | Resolution | Status |
|-------|----------|------------|--------|
| | | | |

### Risks Updated

| Risk | Change | New Likelihood | Impact |
|------|--------|----------------|--------|
| | | | |

### Week 2 Readiness

- [ ] Week 1 tasks complete
- [ ] Session framework ready for expansion
- [ ] No blockers for Week 2
- [ ] Team briefed on Week 2 plan

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
