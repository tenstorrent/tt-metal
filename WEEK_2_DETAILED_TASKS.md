# Week 2: Session Management and Robustness - Detailed Tasks

**Goal:** Production-ready session lifecycle with error handling  
**Duration:** 5 working days  
**Prerequisite:** Week 1 complete, per-step API functional  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part C, Week 2

---

## Monday (Day 1): Session Lifecycle - Create Operation

### Morning (4 hours)

- [ ] **Task 2.1.1:** Design session state machine (1.5h)
  - States: CREATED, IN_PROGRESS, COMPLETED
  - Transitions: Documented with triggers
  - Deliverable: State diagram document

- [ ] **Task 2.1.2:** Implement `handle_session_create` (2h)
  - Input: model_id, total_steps
  - Output: session_id, session_info
  - Initialize: Session state
  - Deliverable: Create handler implemented

- [ ] **Task 2.1.3:** Write create session tests (0.5h)
  - Test: Valid creation
  - Test: Multiple sessions
  - Deliverable: Tests passing

### Afternoon (4 hours)

- [ ] **Task 2.1.4:** Implement session info retrieval (1h)
  - Function: `handle_session_info`
  - Return: Session metadata
  - Deliverable: Info handler implemented

- [ ] **Task 2.1.5:** Integrate create with step operation (2h)
  - Option A: Explicit create required
  - Option B: Auto-create on first step
  - Decision: _____________
  - Deliverable: Integration complete

- [ ] **Task 2.1.6:** Test session-step integration (1h)
  - Test: Create then step
  - Test: Step without create (if auto-create)
  - Deliverable: Integration tests passing

**Monday Success Criteria:**
- [ ] Session create implemented
- [ ] Integration with step operation working
- [ ] Tests passing

---

## Tuesday (Day 2): Session Lifecycle - Step and Complete

### Morning (4 hours)

- [ ] **Task 2.1.7:** Implement `handle_session_step` wrapper (2h)
  - Wrapper: Around denoise_step_single
  - Add: Session tracking and validation
  - Add: Step counting and progress
  - Deliverable: Step wrapper implemented

- [ ] **Task 2.1.8:** Add step validation (1h)
  - Validate: Session exists
  - Validate: Session not completed
  - Validate: Step sequence (warn if out-of-order)
  - Deliverable: Validation logic

- [ ] **Task 2.1.9:** Implement session progress tracking (1h)
  - Track: Current step
  - Track: Total steps
  - Calculate: Progress percentage
  - Deliverable: Progress tracking

### Afternoon (4 hours)

- [ ] **Task 2.1.10:** Implement `handle_session_complete` (2h)
  - Mark: Session as completed
  - Return: Final latents (optional)
  - Return: Session statistics
  - Deliverable: Complete handler implemented

- [ ] **Task 2.1.11:** Implement session statistics collection (1h)
  - Collect: Total time
  - Collect: Per-step times
  - Collect: Step count
  - Deliverable: Statistics collection

- [ ] **Task 2.1.12:** Write full lifecycle test (1h)
  - Test: Create -> Step x20 -> Complete
  - Verify: Statistics correct
  - Deliverable: Lifecycle test passing

**Tuesday Success Criteria:**
- [ ] Step wrapper implemented
- [ ] Complete handler implemented
- [ ] Full lifecycle test passing

---

## Wednesday (Day 3): Session Timeout and Cleanup

### Morning (4 hours)

- [ ] **Task 2.2.1:** Implement background cleanup thread (2h)
  - Thread: Daemon thread checking every 60s
  - Check: Session last_activity vs timeout
  - Default: 30-minute timeout
  - Deliverable: Cleanup thread implemented

- [ ] **Task 2.2.2:** Implement resource cleanup on expiry (1h)
  - Release: Cached tensors
  - Release: SHM handles (if any)
  - Log: Expiry warning
  - Deliverable: Resource cleanup

- [ ] **Task 2.2.3:** Add configurable timeout (1h)
  - Config: timeout_seconds parameter
  - Default: 1800 (30 minutes)
  - Deliverable: Configurable timeout

### Afternoon (4 hours)

- [ ] **Task 2.2.4:** Write timeout tests (2h)
  - Test: Session expires after inactivity
  - Test: Active session not expired
  - Test: Resource cleanup on expiry
  - Deliverable: Timeout tests passing

- [ ] **Task 2.2.5:** Implement graceful degradation (1h)
  - Handle: Operations on expired session
  - Return: Clear error with suggestion
  - Deliverable: Error handling

- [ ] **Task 2.2.6:** Manual cleanup operation (1h)
  - Function: `handle_session_cleanup`
  - Use: Force cleanup of specific session
  - Deliverable: Manual cleanup implemented

**Wednesday Success Criteria:**
- [ ] Background cleanup working
- [ ] Resources properly released
- [ ] Timeout tests passing

---

## Thursday (Day 4): Error Handling and Recovery

### Morning (4 hours)

- [ ] **Task 2.3.1:** Define error categories (1h)
  - Category: Session Not Found
  - Category: Model Mismatch
  - Category: Step Out of Order
  - Category: Format Error
  - Category: Device Error
  - Deliverable: Error category document

- [ ] **Task 2.3.2:** Implement error response format (1h)
  - Fields: success, error (type, message, context)
  - Fields: recoverable, suggestion
  - Deliverable: Error response structure

- [ ] **Task 2.3.3:** Implement recovery strategies (2h)
  - Strategy: Session not found -> suggest create
  - Strategy: Model mismatch -> suggest new session
  - Strategy: Step out of order -> warn and proceed
  - Deliverable: Recovery logic

### Afternoon (4 hours)

- [ ] **Task 2.3.4:** Write error handling tests (2h)
  - Test: Each error category
  - Test: Recovery suggestions correct
  - Test: Graceful failures
  - Deliverable: Error tests passing

- [ ] **Task 2.3.5:** Implement logging for errors (1h)
  - Log: Error type and context
  - Log: Recovery attempt
  - Log: Resolution status
  - Deliverable: Error logging

- [ ] **Task 2.3.6:** Document error handling (1h)
  - Document: All error types
  - Document: Recovery procedures
  - Deliverable: Error documentation

**Thursday Success Criteria:**
- [ ] Error categories defined
- [ ] Recovery strategies implemented
- [ ] Error tests passing

---

## Friday (Day 5): Performance Testing + Week Wrap-up

### Morning (4 hours)

- [ ] **Task 2.5.1:** Implement performance benchmark (1.5h)
  - Measure: Full-loop baseline
  - Measure: Per-step times
  - Calculate: Overhead percentage
  - Deliverable: Benchmark script

- [ ] **Task 2.5.2:** Run performance comparison (1.5h)
  - Compare: 20-step per-step vs full-loop
  - Target: < 10% overhead
  - Document: Results
  - Deliverable: Performance report

- [ ] **Task 2.5.3:** Memory stability test (1h)
  - Run: 100 consecutive generations
  - Monitor: Memory usage
  - Check: No leaks
  - Deliverable: Memory report

### Afternoon (4 hours)

- [ ] **Task 2.4.1:** Create TT_KSampler node placeholder (1.5h)
  - File: `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`
  - Class: TT_KSampler (skeleton)
  - Deliverable: Node placeholder

- [ ] **Task 2.4.2:** Run full test suite (1h)
  - Run: All Week 1 and Week 2 tests
  - Fix: Any regressions
  - Deliverable: All tests passing

- [ ] **Task 2.4.3:** Code review preparation (1h)
  - Review: All new code
  - Add: Missing documentation
  - Deliverable: Code ready for review

- [ ] **Task 2.4.4:** Week 2 review and planning (0.5h)
  - Review: All tasks complete
  - Document: Issues or delays
  - Plan: Week 3 adjustments
  - Deliverable: Week 2 summary

**Friday Success Criteria:**
- [ ] Performance within budget
- [ ] Memory stable
- [ ] All tests passing

---

## Week 2 Summary

### Deliverables Checklist

- [ ] Session lifecycle complete (create, step, complete)
- [ ] Timeout mechanism working (30-min default)
- [ ] Error handling graceful and documented
- [ ] Performance within budget (< 10% overhead)
- [ ] 100 consecutive generations stable
- [ ] TT_KSampler node placeholder created

### Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Session lifecycle tests | 100% pass | | |
| Per-step overhead | < 10% | | |
| Memory leak | 0 | | |
| Concurrent sessions | 5+ supported | | |

### Hours Summary

| Day | Estimated | Actual |
|-----|-----------|--------|
| Monday | 8h | |
| Tuesday | 8h | |
| Wednesday | 8h | |
| Thursday | 8h | |
| Friday | 8h | |
| **Total** | **40h** | |

### Integration Points

| Component | Week 1 Status | Week 2 Integration | Status |
|-----------|--------------|-------------------|--------|
| Per-step API | Complete | Session wrapper | |
| Model config | Complete | Model ID tracking | |
| Scheduler sync | Complete | Step validation | |

### Week 3 Readiness

- [ ] Week 2 tasks complete
- [ ] ControlNet feasibility (Phase 0) confirmed GO
- [ ] No blockers for Week 3
- [ ] Team briefed on ControlNet integration

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
