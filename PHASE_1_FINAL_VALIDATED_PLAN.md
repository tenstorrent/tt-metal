# Phase 1: Bridge Extension - Final Validated Implementation Plan

**Document Version:** 1.0
**Approval Status:** APPROVED
**Effective Date:** December 15, 2025
**Total Timeline:** 5-6 weeks (Phase 0: 5 days + Phase 1: 4 weeks + Buffer: 3-5 days)

---

## Section 1: Executive Summary

### Overview

Phase 1: Bridge Extension adds per-timestep denoising capability to the ComfyUI-TT bridge, enabling ControlNet, IP-Adapter, and custom sampler integration. This implementation path was selected because it validates per-timestep patterns before committing to full native integration (12-17 weeks), delivers high-value features (ControlNet) in 5-6 weeks, and provides 100% reusability of work for subsequent phases.

The implementation follows a two-phase approach: Phase 0 (5 days) validates critical assumptions before commitment, then Phase 1 (4 weeks) implements the per-timestep API, session management, ControlNet integration, and comprehensive documentation. All user decisions have been incorporated: IP-Adapter is deferred to Phase 1.5, timeline extended to 5-6 weeks with flexibility, Option A (Stateless Bridge) selected for scheduler management, and documentation deliverables (ADRs, architecture docs, reusability comments) included.

### Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 0 | 5 days | Feasibility validation, Go/No-Go decision |
| Week 1 | 5 days | Per-timestep API, model-agnostic config |
| Week 2 | 5 days | Session management, error handling |
| Week 3 | 5 days | ControlNet implementation |
| Week 4 | 5 days | Testing, performance, documentation (Part 1) |
| Week 5 | 5 days | Final validation, documentation (Part 2) |
| Buffer | 3-5 days | Contingency |
| **Total** | **5-6 weeks** | **Full bridge extension with ControlNet** |

### Success Criteria

**Phase 0 Gate:**
- Scheduler sync design approved
- ControlNet architecture validated (or pivot defined)
- IPC baseline < 10ms measured

**Phase 1 Gate:**
- Per-step SSIM >= 0.99 vs full-loop (100 test prompts)
- ControlNet SSIM >= 0.90 vs CPU reference (30 images)
- Human validation: 5/5 correct
- Performance: < 10% overhead per-step
- Robustness: 1000 generations, 0 crashes
- Documentation complete

### Risk Assessment

**Confidence Level:** 85% (with all mitigations applied)

**Critical Risks:**
1. ControlNet TT-side requirement (25% likelihood) - Mitigated by Phase 0 validation
2. Scheduler state desync (30% likelihood) - Mitigated by Option A stateless design
3. IPC overhead > 15% (40% likelihood) - 60% headroom expected based on 1-5ms baseline

---

## Section 2: Phase 0 - Pre-Implementation Validation (Days 1-5)

### Purpose

Phase 0 is the highest-ROI investment in Phase 1, reducing mid-implementation pivot risk from 40% to < 5% by validating critical assumptions before committing to full implementation.

### Task 0.1: Scheduler State Sync Design (Days 1-2)

**Objective:** Design and validate stateless bridge approach (Option A).

**Selected Approach: Option A - Stateless Bridge**
- ComfyUI owns scheduler state (timesteps[], sigmas[], current_step)
- Bridge receives timestep/sigma per call
- No state synchronization between processes

**Deliverables:**
1. Design document: `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md`
2. Prototype in handlers.py demonstrating stateless operation
3. Test suite validating 20+ step sequences
4. Error handling strategy document

**Acceptance Criteria:**
- [ ] Design document reviewed and approved
- [ ] Prototype passes scheduler sync tests
- [ ] Edge cases documented with recovery strategies

### Task 0.2: IPC Performance Baseline (Day 1, Parallel)

**Objective:** Measure baseline latency to establish per-step budget.

**Measurements:**
```
Full-loop latency (20 steps): _____ ms
Per-step baseline: full-loop / 20 = _____ ms
IPC budget (10%): per-step * 0.10 = _____ ms
Actual IPC roundtrip: _____ ms (expected: 1-5ms)
Headroom: IPC budget - actual = _____ ms
```

**Deliverables:**
1. Performance baseline document
2. Test script for reproducible measurement
3. Headroom analysis with recommendations

**Acceptance Criteria:**
- [ ] IPC latency < 10ms confirmed
- [ ] Baseline measurements documented
- [ ] Per-step budget calculated

### Task 0.3: ControlNet Architecture Feasibility Study (Days 3-5)

**Objective:** Validate CPU-side ControlNet execution works with bridge.

**Investigation:**
1. Analyze ComfyUI ControlNet data flow
2. Prototype control_hint transfer via IPC
3. Validate TT UNet can accept control_hint parameter

**Go/No-Go Decision Matrix:**

| Result | Decision | Next Step |
|--------|----------|-----------|
| CPU-side works, hint transfers | GO | Proceed to Week 1 |
| Needs TT-side execution | DEFER | Move ControlNet to Phase 2 |
| Architecture incompatible | PIVOT | Focus on per-step API only |

**Deliverables:**
1. Feasibility report with data flow diagrams
2. Prototype demonstrating conditioning transfer
3. Go/No-Go recommendation with rationale

**Acceptance Criteria:**
- [ ] ControlNet architecture fully documented
- [ ] Prototype validates or invalidates approach
- [ ] Clear recommendation with reasoning

### Phase 0 Output

**Deliverable:** `/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md`

Contents:
1. Scheduler sync design summary and approval
2. IPC baseline measurements
3. ControlNet feasibility assessment
4. Updated risk register
5. Go/No-Go recommendation

**Decision Point:** Proceed to Phase 1 ONLY if Phase 0 criteria met.

---

## Section 3: Week 1 - Per-Timestep API Foundation

### Goal

Implement `handle_denoise_step_single` operation with model-agnostic design.

### Task 1.1: handle_denoise_step_single Operation

**Duration:** 2 days

**File:** `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`

**Operation Specification:**
```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute SINGLE denoising step. ComfyUI calls N times for N steps.

    Params:
        session_id: str
        latent_shm: dict [B, C, H, W]
        timestep: float
        timestep_index: int
        sigma: float
        conditioning_shm: dict
        negative_conditioning_shm: dict (optional)
        control_hint_shm: dict (optional)
        guidance_scale: float (default 7.5)

    Returns:
        latent_shm: dict [B, C, H, W]
        step_metadata: dict
    """
```

**Implementation Notes:**
- Extract single-step logic from existing `handle_denoise_only` (lines 440-625)
- Accept timestep/sigma from ComfyUI (stateless bridge)
- Return latents in standard format [B, C, H, W]
- Include timing information

### Task 1.2: Model-Agnostic Refactoring

**Duration:** 1 day

**Create:** `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py`

```python
MODEL_CONFIGS = {
    "sdxl": {
        "latent_channels": 4,
        "clip_dim": 2048,
        "clip_dim_l": 768,
        "clip_dim_g": 1280,
        "vae_scale_factor": 8,
    },
    "sd35": {
        "latent_channels": 16,
        "clip_dim": 4096,
        "vae_scale_factor": 8,
    },
    "sd14": {
        "latent_channels": 4,
        "clip_dim": 768,
        "vae_scale_factor": 8,
    },
}
```

**Changes to handlers.py:**
- Replace hardcoded `C != 4` checks with config lookup
- Parametrize format conversion calls

### Task 1.3: Scheduler State Sync Implementation

**Duration:** 1 day

**Implementation:** Stateless bridge pattern from Phase 0 design.

**ComfyUI side sends:**
```python
{
    "timestep": scheduler.timesteps[step],
    "sigma": scheduler.sigmas[step],
    "timestep_index": step,
}
```

**Bridge side receives and passes through (no state management).**

### Task 1.4: Format Conversion Audit

**Duration:** 0.5 days

**Update:** `_detect_and_convert_tt_to_standard_format`
- Make `expected_channels` required parameter
- Add `model_type` parameter for logging
- Update all call sites

### Task 1.5: Session State Framework

**Duration:** 0.5 days

**Create:** `/home/tt-admin/tt-metal/comfyui_bridge/session_manager.py`

Basic framework for Week 2 implementation:
- `DenoiseSession` dataclass
- `SessionManager` class skeleton
- Timeout configuration

### Week 1 Testing

| Test | Description | Target |
|------|-------------|--------|
| `test_denoise_step_single_basic` | Single step produces valid output | Pass |
| `test_denoise_step_single_format` | Output is [B, C, H, W] | Pass |
| `test_model_agnostic_config` | Config lookup works all models | Pass |
| `test_per_step_matches_full_loop` | 20 steps == denoise_only | SSIM >= 0.99 |

### Week 1 Definition of Done

- [ ] `handle_denoise_step_single` implemented
- [ ] Model-agnostic config system working
- [ ] Format conversion parametrized
- [ ] SSIM >= 0.99 vs full-loop baseline
- [ ] All Week 1 tests passing

---

## Section 4: Week 2 - Session Management & Robustness

### Goal

Production-ready session lifecycle with error handling.

### Task 2.1: Session Lifecycle Management

**Duration:** 2 days

**Operations:**
1. `handle_session_create` - Create new session, return session_id
2. `handle_session_step` - Execute step with session tracking
3. `handle_session_complete` - Cleanup and return stats

**Session States:** CREATED -> IN_PROGRESS -> COMPLETED -> removed

### Task 2.2: Session Timeout and Cleanup

**Duration:** 1 day

**Implementation:**
- Background cleanup thread (60-second check interval)
- 30-minute default timeout
- Resource cleanup on expiration
- Warning logs for expired sessions

### Task 2.3: Error Handling and Recovery

**Duration:** 1 day

**Error Categories:**
| Error | Recovery |
|-------|----------|
| Session Not Found | Suggest create_session |
| Model Mismatch | Suggest new session |
| Step Out of Order | Warning + proceed |
| Format Error | Detailed message |
| Device Error | Attempt recovery |

### Task 2.4: ComfyUI Node Infrastructure

**Duration:** 1 day

**File:** `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`

**Add:** `TT_KSampler` placeholder node for per-step sampling.

### Task 2.5: IPC Performance Testing

**Duration:** 0.5 days

**Validate:** Per-step overhead < 10% budget

### Week 2 Testing

| Test | Description | Target |
|------|-------------|--------|
| `test_session_lifecycle` | Create -> 20 steps -> complete | Pass |
| `test_session_timeout` | Expires after 30 min | Pass |
| `test_concurrent_sessions` | Multiple sessions work | Pass |
| `test_error_recovery` | Graceful failure | Pass |
| `test_memory_stability` | 100 generations | 0 leaks |

### Week 2 Definition of Done

- [ ] Session lifecycle complete
- [ ] Timeout mechanism working
- [ ] Error handling graceful
- [ ] Performance < 10% overhead
- [ ] 100 consecutive generations stable

---

## Section 5: Week 3 - ControlNet Implementation

### Goal

Enable ControlNet workflows through bridge extension.

**Prerequisite:** Phase 0 Go decision on ControlNet.

### Task 3.1: ControlNet Conditioning Injection

**Duration:** 2 days

**Add to `handle_denoise_step_single`:**
```python
# ControlNet conditioning (from ComfyUI CPU/GPU side)
control_hint = None
if "control_hint_shm" in params:
    control_hint_torch = self._get_tensor_from_shm(params["control_hint_shm"])
    control_hint = self._prepare_control_hint(control_hint_torch)

# Pass to UNet
output = self.runner.denoise_step(..., control_hint=control_hint)
```

### Task 3.2: ComfyUI TT_ControlNet Wrapper

**Duration:** 1.5 days

**Add:** `TT_ApplyControlNet` node that:
1. Runs ControlNet on CPU/GPU
2. Attaches control_hint to conditioning
3. Passes through to TT_KSampler

### Task 3.3: Integration Testing

**Duration:** 1 day

**Test ControlNet Types:**
1. Canny edge following
2. Depth-aware generation
3. OpenPose pose guidance

### Task 3.4: Multi-ControlNet Support (Optional)

**Duration:** 0.5 days

Support multiple control_hints if time permits.

### Week 3 Testing

| Test | Description | Target |
|------|-------------|--------|
| `test_controlnet_canny` | Canny edges followed | SSIM >= 0.90 |
| `test_controlnet_depth` | Depth respected | SSIM >= 0.90 |
| `test_controlnet_openpose` | Pose followed | SSIM >= 0.90 |
| Human validation | Visual correctness | 5/5 correct |

### Week 3 Definition of Done

- [ ] ControlNet conditioning injection working
- [ ] 3 ControlNet types tested
- [ ] SSIM >= 0.90 vs CPU reference
- [ ] Human validation passed

---

## Section 6: Week 4 - Validation & Performance & Documentation (Part 1)

### Goal

Comprehensive testing, performance validation, and ADR documentation.

### Task 4.1: Comprehensive Test Suite Execution

**Duration:** 2 days

**Test Categories:**
- Unit tests: 30+ tests (format, session, error, config)
- Integration tests: 20+ tests (per-step, ControlNet, timeout)
- Performance tests: 10+ tests (latency, memory, throughput)
- Regression tests: 10+ tests (existing functionality)

**Execution:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/ -v --tb=short
```

### Task 4.2: Performance Optimization

**Duration:** 1 day

**Profile and optimize if needed:**
- Tensor transfer batching
- Conditioning caching
- Format conversion
- IPC compression

**Target:** Per-step latency < 10% overhead

### Task 4.3: Documentation (Part 1) - ADRs

**Duration:** 2 days

**Create ADRs:**

1. **ADR-001: Per-Timestep API Design**
   - File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-001-per-timestep-api.md`
   - Content: Why per-step, alternatives considered, consequences

2. **ADR-002: Scheduler State Synchronization**
   - File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-002-scheduler-sync.md`
   - Content: Option A rationale, implementation details

3. **ADR-003: ControlNet Integration**
   - File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-003-controlnet-integration.md`
   - Content: CPU-side approach, data flow, extensibility

### Week 4 Definition of Done

- [ ] All test suites passing (95%+)
- [ ] Performance targets met
- [ ] 3 ADRs written and approved
- [ ] Code annotated with reusability comments

---

## Section 7: Week 5 - Final Validation & Documentation (Part 2)

### Goal

Release preparation and documentation completion.

### Task 5.1: Final End-to-End Validation

**Duration:** 1.5 days

**Validation Matrix:**

| Category | Tests | Target |
|----------|-------|--------|
| Per-step API | 100 prompts | SSIM >= 0.99 |
| ControlNet | 30 images (3 types x 10) | SSIM >= 0.90 |
| Human | 5 raters | 5/5 correct |
| Performance | 1000 generations | 0 crashes |
| Memory | Profiling | 0 leaks |
| Regression | All existing | 100% pass |

### Task 5.2: Documentation (Part 2)

**Duration:** 2 days

**Create:**

1. **Architecture Documentation**
   - File: `/home/tt-admin/tt-metal/docs/architecture/bridge_extension.md`

2. **API Reference**
   - File: `/home/tt-admin/tt-metal/docs/api/bridge_extension_api.md`

3. **User Guide**
   - File: `/home/tt-admin/tt-metal/docs/guides/controlnet_guide.md`

4. **Native Integration Handoff**
   - File: `/home/tt-admin/tt-metal/docs/architecture/native_integration_handoff.md`

### Task 5.3: Release Preparation

**Duration:** 1.5 days

**Checklist:**
- [ ] Code review completed
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes written
- [ ] Phase 1.5 plan drafted

**Phase 1.5 Planning:**
- File: `/home/tt-admin/tt-metal/docs/roadmap/PHASE_1_5_IP_ADAPTER.md`
- Timeline: 2 weeks
- Dependencies: Phase 1 complete

### Week 5 Definition of Done

- [ ] All validation criteria met
- [ ] Documentation complete (5+ documents)
- [ ] Release notes written
- [ ] Phase 1.5 plan ready
- [ ] Code review approved

---

## Section 8: Success Criteria & Metrics

### Phase 0 Go/No-Go Gate

| Criterion | Target | Status |
|-----------|--------|--------|
| Scheduler sync design | Approved | [ ] |
| ControlNet feasible | GO or pivot defined | [ ] |
| IPC baseline | < 10ms | [ ] |

**Decision Rule:** Proceed ONLY if ALL criteria met.

### Phase 1 Completion Gate

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Per-step SSIM | >= 0.99 | vs full-loop, 100 prompts |
| ControlNet SSIM | >= 0.90 | vs CPU ref, 30 images |
| Human validation | 5/5 | Visual inspection |
| Per-step latency | < 10% overhead | Benchmark |
| Robustness | 0 crashes | 1000 generations |
| Memory | 0 leaks | Profiling |
| Regression | 100% pass | Existing tests |
| Compatibility | Backward compat | API validation |
| Documentation | Complete | Review checklist |
| Reusability | < 2 hour sketch | Engineer exercise |

### Quantitative Thresholds

| Score | Result | Action |
|-------|--------|--------|
| 95%+ criteria | SUCCESS | Release Phase 1 |
| 80-94% criteria | CONDITIONAL | Defer remainder to Phase 1.5 |
| < 80% criteria | REVISIT | Extend timeline or pivot |

---

## Section 9: Risk Mitigation & Contingency

### Critical Assumptions

| Assumption | Validation | Fallback |
|------------|------------|----------|
| ControlNet CPU-side works | Phase 0 Day 3-5 | Defer to Phase 2 |
| Scheduler sync works | Phase 0 Day 1-2 | Option B (stateful bridge) |
| IPC overhead < 10% | Phase 0 Day 1 | Optimize or adjust targets |
| ComfyUI schedulers compatible | Week 1 | Document incompatible |
| Session management scales | Week 2 | Single-model sessions |

### Technical Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scheduler desync | 30% | Critical | Extensive testing, stateless design |
| ControlNet TT-side needed | 25% | High | Phase 0 validation |
| IPC overhead > 15% | 40% | Medium | Early optimization window |
| Session cleanup failures | 20% | Medium | Timeout mechanism |
| Format conversion bugs | 15% | High | Audit + round-trip tests |

### Schedule Buffers

| Week | Buffer | Reason |
|------|--------|--------|
| Week 1 | +2 days | Scheduler complexity |
| Week 2 | +1 day | Session edge cases |
| Week 3 | +2 days | ControlNet integration |
| Week 4 | +1 day | Test debugging |
| Week 5 | +2 days | Documentation, fixes |
| **Total** | **8 days** | **1.6 weeks contingency** |

### Timeline Scenarios

| Scenario | Duration | Probability |
|----------|----------|-------------|
| Optimistic | 4 weeks | 20% |
| Realistic | 5-6 weeks | 65% |
| Conservative | 7 weeks | 15% |

---

## Section 10: Reusability & Long-Term Value

### SD3.5 Support (Phase 2, 1-2 weeks)

**Already Done in Phase 1:**
- Per-step API (model-agnostic)
- Config-based channels (SD3.5 entry exists)
- Format conversion (parametrized)

**Remaining for SD3.5:**
- SD3.5 runner implementation
- Model-specific validation
- Testing

**Savings:** 1-2 weeks vs 3-4 weeks from scratch

### Native Integration (Phase 3, 12-17 weeks)

**Patterns Validated by Phase 1:**
- Per-timestep calling pattern
- Scheduler state management
- ControlNet integration approach
- Format conversions

**Code Reusable:**
- Format conversion helpers: 100%
- Model config system: 100%
- Session patterns: 80%
- Scheduler patterns: 90%
- ControlNet approach: 100%

**Savings:** 2-4 weeks of discovery work

### Documentation for Reusability

**Code Annotations:**
```python
"""
Reusability Note for Native Integration:
    This component is [X]% reusable.
    - What to port: [elements]
    - What changes: [IPC specifics]
"""
```

**ADRs Document:**
- Why decisions were made
- Alternatives considered
- Implications for future work

---

## Section 11: File Modifications & Deliverables

### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` | Add per-step operations, model config | +300 |
| `/home/tt-admin/tt-metal/comfyui_bridge/server.py` | Add operation routing | +50 |
| `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` | Add TT_KSampler, TT_ApplyControlNet | +200 |

### New Files to Create

| File | Purpose | Lines |
|------|---------|-------|
| `comfyui_bridge/model_config.py` | Centralized model configs | ~100 |
| `comfyui_bridge/session_manager.py` | Session lifecycle management | ~200 |
| `comfyui_bridge/tests/test_per_step.py` | Per-step unit tests | ~200 |
| `comfyui_bridge/tests/test_controlnet.py` | ControlNet integration tests | ~150 |
| `docs/architecture/adr/ADR-001-per-timestep-api.md` | Design decision record | ~100 |
| `docs/architecture/adr/ADR-002-scheduler-sync.md` | Design decision record | ~80 |
| `docs/architecture/adr/ADR-003-controlnet-integration.md` | Design decision record | ~100 |
| `docs/architecture/bridge_extension.md` | Architecture documentation | ~300 |
| `docs/api/bridge_extension_api.md` | API reference | ~200 |
| `docs/guides/controlnet_guide.md` | User guide | ~150 |

### Documentation Deliverables

1. **Phase 0 Report** - Feasibility findings
2. **3 ADRs** - Design decisions
3. **Architecture Docs** - System design
4. **API Reference** - Operation specifications
5. **User Guide** - ControlNet workflows
6. **Native Integration Handoff** - Reusability documentation
7. **Phase 1.5 Plan** - IP-Adapter roadmap

---

## Section 12: Team & Resource Requirements

### Recommended Team

| Role | Responsibilities | Allocation |
|------|------------------|------------|
| Senior Engineer | API design, ControlNet, architecture, code review | 100% |
| Mid-level Engineer | Session management, error handling, model-agnostic | 100% |
| QA Engineer | Test strategy, implementation, performance | 50% |
| Technical Writer | Documentation (concurrent) | 25% |

### Constrained Team (2 Engineers)

| Engineer | Weeks 1-2 | Weeks 3-4 | Week 5 |
|----------|-----------|-----------|--------|
| Senior | Phase 0 + Per-step API | ControlNet | Final validation |
| Mid-level | Model-agnostic + Sessions | Testing | Documentation |

### Bottleneck Analysis

**Critical Path:** Senior engineer
- Phase 0 validation
- Per-step API design
- ControlNet integration

**Cannot Parallelize:** API design and ControlNet (both senior work)

---

## Section 13: File References & Code Locations

### Reference Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Parity Status Correction | `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md` | Current state (v2.0) |
| Strategic Analysis | `/home/tt-admin/tt-metal/STRATEGIC_PATH_ANALYSIS.md` | Path selection rationale |
| Native Integration Analysis | `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md` | Future integration guide |

### Code Locations to Review

| Location | Lines | Purpose |
|----------|-------|---------|
| `comfyui_bridge/handlers.py` | 570-625 | Current denoise_only implementation |
| `comfyui_bridge/handlers.py` | 32-93 | Format conversion helper |
| `tenstorrent_nodes/nodes.py` | 1-100 | Existing node patterns |

### Model Config Reference

| File | Content |
|------|---------|
| `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/utils.py` | Existing MODEL_CONFIGS |

---

## Section 14: Decision Points & Next Steps

### Pre-Phase 0 Approvals (Complete)

- [x] Timeline approved (5-6 weeks)
- [x] Team composition confirmed
- [x] IP-Adapter deferred to Phase 1.5
- [x] Option A (stateless bridge) selected
- [x] Phase 0 resource allocation confirmed

### Phase 0 Decision Point

After Day 5, review:
- [ ] Feasibility report
- [ ] Go/No-Go recommendation
- [ ] Risk register updates

**Decision:** Proceed to Phase 1 OR define pivot strategy

### Phase 1 Completion Decision Point

After Week 5, review:
- [ ] All success criteria
- [ ] Documentation completeness
- [ ] Phase 1.5 plan

**Decisions:**
- Proceed to Phase 1.5 (IP-Adapter)?
- Begin Phase 2 planning (native integration)?

---

## Section 15: Success Stories & Validation

### Week 4 Review Checkpoint

Expected state:
- [x] Per-step denoising working (SSIM >= 0.99)
- [x] ControlNet workflows functional
- [x] 100+ tests passing
- [x] Performance benchmarks documented
- [x] ADRs written

### Week 5 Completion State

Expected state:
- [x] Phase 1 released and documented
- [x] Phase 1.5 plan ready (IP-Adapter)
- [x] Native integration handoff prepared
- [x] Team ready for Phase 2

### Long-term Validation

Phase 1 success validated by:
- Phase 1.5 (IP-Adapter) completed in 2 weeks (confirms reusability)
- Phase 2 (SD3.5) completed in 1-2 weeks (confirms model-agnosticism)
- Phase 3 (native) saves 2-4 weeks (confirms architectural learning)

---

## Final Notes

### Implementation Philosophy

This plan prioritizes:
1. **Validation before commitment** (Phase 0)
2. **Incremental delivery** (weekly milestones)
3. **Reusability** (documented patterns)
4. **Quality** (comprehensive testing)
5. **Transparency** (ADRs and documentation)

### Timeline Confidence

The 5-6 week timeline is **conservative but realistic**. It includes:
- Proper validation (Phase 0)
- Adequate testing (Weeks 1-4)
- Complete documentation (Weeks 4-5)
- Buffer for unexpected issues

### Executive Sign-off

This plan is approved for execution. Phase 0 begins immediately.

---

**END OF FINAL VALIDATED PLAN**

---

# APPENDIX: Summary Comparison - Old vs New Plan

## What Changed Based on User Decisions

| Aspect | Original Proposal | Final Plan (with User Decisions) |
|--------|-------------------|----------------------------------|
| **IP-Adapter** | Included in Phase 1 (Week 3-4) | **Deferred to Phase 1.5** (separate 2-week effort) |
| **Timeline** | 4 weeks fixed | **5-6 weeks with flexibility** (Phase 0: 5 days + Phase 1: 4 weeks + Buffer) |
| **Scheduler Management** | Two options presented | **Option A (Stateless Bridge) selected** - ComfyUI owns scheduler |
| **Pre-Implementation** | Validation implicit | **Phase 0 added** (5-day explicit validation period) |
| **Documentation** | Minimal | **Comprehensive** - ADRs, architecture docs, reusability comments |

## Timeline Comparison

### Original Timeline (Implicit)

```
Week 1: Per-step API
Week 2: Session Management
Week 3: ControlNet + IP-Adapter
Week 4: Testing + Release
Total: 4 weeks (no buffer, high risk)
```

### New Timeline (With User Decisions)

```
Phase 0 (Days 1-5): Pre-implementation validation
  - Scheduler sync design
  - IPC performance baseline
  - ControlNet feasibility study
  - Go/No-Go decision

Week 1: Per-step API + Model-agnostic
Week 2: Session Management + Error Handling
Week 3: ControlNet Implementation
Week 4: Validation + Performance + Documentation (Part 1)
Week 5: Final Validation + Documentation (Part 2)
Buffer: 3-5 days contingency

Total: 5-6 weeks (validated, documented, robust)
```

## Risk Profile Comparison

| Risk Factor | Original | New (with Decisions) |
|-------------|----------|----------------------|
| Mid-implementation pivot risk | 40% | **< 5%** (Phase 0 validation) |
| Scheduler sync issues | Unmitigated | **Mitigated** (Option A stateless design) |
| ControlNet feasibility | Assumed | **Validated in Phase 0** |
| IPC overhead concerns | Assumed acceptable | **Measured and budgeted** |
| Documentation gaps | High | **Eliminated** (3 ADRs + architecture docs) |
| Reusability uncertain | Implicit | **Explicit guarantees documented** |

## Deliverables Comparison

### Original Deliverables

1. Per-step API
2. Session management
3. ControlNet support
4. IP-Adapter support
5. Basic tests

### New Deliverables (Expanded)

**Technical:**
1. Per-step API (handle_denoise_step_single)
2. Session management with timeout
3. ControlNet support (3 types)
4. Model-agnostic config system
5. Comprehensive test suite (70+ tests)

**Documentation:**
6. Phase 0 Feasibility Report
7. ADR-001: Per-Timestep API Design
8. ADR-002: Scheduler State Synchronization
9. ADR-003: ControlNet Integration
10. Architecture Documentation
11. API Reference
12. User Guide (ControlNet)
13. Native Integration Handoff Document
14. Phase 1.5 Planning Document (IP-Adapter)

## Confidence Comparison

| Metric | Original | New |
|--------|----------|-----|
| Success probability | ~60% | **85%** |
| Reusability confidence | Unknown | **100% documented** |
| Native integration readiness | Uncertain | **Handoff document prepared** |
| SD3.5 enablement | Not considered | **1-2 weeks with Phase 1 foundation** |

## Key Improvements

1. **Phase 0 Validation**: The 5-day pre-implementation phase is the single highest-ROI investment, reducing pivot risk from 40% to < 5%.

2. **Option A Selection**: The stateless bridge approach eliminates an entire class of scheduler synchronization bugs.

3. **IP-Adapter Deferral**: Moving IP-Adapter to Phase 1.5 reduces Phase 1 scope by ~25%, increasing focus and reducing risk.

4. **Documentation Strategy**: The three-tier documentation (ADRs + code comments + architecture docs) ensures institutional knowledge capture and enables reusability.

5. **Explicit Reusability Guarantees**: Every component now has documented reusability percentages for native integration.

## What Success Looks Like

**Original Success:** "ControlNet works"

**New Success (Measurable):**
- Per-step SSIM >= 0.99 vs full-loop (100 test prompts)
- ControlNet SSIM >= 0.90 vs CPU reference (30 images)
- Human validation: 5/5 raters confirm "correct"
- Performance: < 10% overhead per-step
- Robustness: 1000 generations, 0 crashes, 0 memory leaks
- Documentation: All ADRs approved, architecture docs complete
- Reusability: Senior engineer can sketch native integration in < 2 hours

---

**END OF COMPARISON**
