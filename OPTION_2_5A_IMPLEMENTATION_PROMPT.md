# Option 2.5A Implementation Plan - Agent Prompt

**Objective**: Create a detailed, actionable implementation plan for Option 2.5A (Bridge-First with Native Hooks) ComfyUI-Tenstorrent integration, focusing on immediate validation steps and phased rollout.

---

## Context & Background

You are planning the implementation of **Option 2.5A: Bridge-First with Native Hooks**, which scored **6.85/10** in the strategic analysis (highest overall score).

### What is Option 2.5A?

**Architecture**: Hybrid approach that combines proven bridge pattern with strategic native hooks
- **Core**: Keep full bridge pattern for model execution (proven with TT_FullDenoise)
- **Enhancement**: Add native ComfyUI hooks for ControlNet/IP-Adapter support
- **Philosophy**: Start with stability, enhance selectively

**Strategic Advantages**:
- Builds on proven TT_FullDenoise (SSIM 0.998+)
- Minimal risk (proven technology)
- Incremental enhancement path
- 8-11 week timeline (reasonable)
- Can migrate to fuller native integration later

### Available Context Documents

**MUST READ** before planning:
1. `/home/tt-admin/tt-metal/COMFYUI_INTEGRATION_ROOT_CAUSE_ANALYSIS.md`
   - Precision boundary issues and resolutions
   - Scheduler state conflict analysis
   - TT_FullDenoise success pattern
   - **CRITICAL**: Timestep format bug (sigma vs timestep_index) - potential blocker
   - **CRITICAL**: CFG batching concerns - must verify

2. `/home/tt-admin/tt-metal/COMFYUI_ARCHITECTURE_KNOWLEDGE_BASE.md`
   - `model_function_wrapper` hook documentation
   - ComfyUI extension points catalog
   - Backend integration patterns (AMD, Intel precedents)
   - Custom node best practices

3. `/home/tt-admin/COMFYUI_INTEGRATION_AGENT_PROMPT.md`
   - Original requirements and success criteria
   - Feature requirements matrix
   - Constraint analysis

4. `/home/tt-admin/tt-metal/PICKUP_COMFYUI.md`
   - Previous integration attempt summary
   - What worked vs what failed

5. `/home/tt-admin/tt-metal/CURSOR_TOME.md`
   - Standalone server debugging journey
   - Known working patterns

### Existing Codebase

**Working Components** (leverage these):
- `/home/tt-admin/tt-metal/tt_sdxl_pipeline.py` - Standalone SDXL server
- `/home/tt-admin/tt-metal/sdxl_runner.py` - Runner implementation
- `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py` - TT_FullDenoise node (PROVEN)
- `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` - Bridge client
- `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/tt_samplers.py` - Custom samplers

**Clean Base** (for new work):
- `/home/tt-admin/ComfyUI-tt_standalone/` - Clean ComfyUI installation

---

## Your Mission

Create a comprehensive, phased implementation plan that:

1. **Validates critical assumptions FIRST** (Immediate steps this week)
2. **Stabilizes the bridge pattern** (Short-term: 4 weeks)
3. **Adds native hooks strategically** (Medium-term: 8-11 weeks)
4. **Provides clear success criteria** at each phase
5. **Identifies parallelizable work** to accelerate delivery
6. **Includes rollback/pivot plans** if assumptions prove wrong

---

## Phase 0: Critical Validation (This Week - Days 1-7)

### BLOCKING ISSUE 1: Timestep Format Bug 🔴 CRITICAL

**Background** (from Root Cause Analysis):
- ComfyUI passes continuous sigma values (e.g., 14.6146)
- TT scheduler expects discrete timestep indices (e.g., 999)
- Current code (`wrappers.py` line 350) may be passing sigma directly
- **If this is wrong, it causes completely incorrect denoising**

**Your Task**: Create detailed investigation plan

**Questions to Answer**:
1. What exactly does `wrappers.py` line 350 pass to the bridge?
2. What does the bridge server's `apply_unet()` function expect?
3. What does TT's `tt_euler_discrete_scheduler.py` require as input?
4. Is there a mismatch? If yes, how severe?

**Investigation Steps** (be specific):
- [ ] Log timestep values at bridge boundary (before sending)
- [ ] Log values received in bridge server
- [ ] Compare with TT scheduler expectations
- [ ] Test: Run same prompt with known-good standalone server vs bridge
- [ ] Measure: SSIM score should be > 0.95 if correct

**Expected Outcomes**:
- **Scenario A**: No mismatch - Continue with plan
- **Scenario B**: Minor mismatch - Add conversion wrapper (1-2 days)
- **Scenario C**: Major mismatch - Requires scheduler redesign (2-3 weeks risk)

**Deliverable**: Markdown report with:
- Evidence (logs, code traces)
- Severity assessment (blocker / fixable / non-issue)
- Fix strategy if needed
- Timeline impact

**Assigned Files to Inspect**:
- `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (lines 340-360)
- `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` (bridge client)
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py`
- Bridge server code (if accessible)

**Success Criteria**:
- Definitive answer on mismatch severity
- If mismatch exists: Fix implemented and tested
- SSIM >= 0.95 vs standalone server

---

### BLOCKING ISSUE 2: CFG Batching Verification 🔴 HIGH

**Background** (from Root Cause Analysis):
- ComfyUI batches conditional/unconditional as `[uncond, cond]`
- TT UNet must process this correctly
- TT_FullDenoise works, but per-step integration untested
- **If batch order wrong, all outputs corrupted**

**Your Task**: Create verification plan

**Questions to Answer**:
1. How does TT UNet handle batched inputs?
2. Is the batch order `[uncond, cond]` or something else?
3. Does CFG computation in bridge match ComfyUI expectations?

**Verification Steps** (be specific):
- [ ] Add detailed batch logging in `wrappers.py` apply_model()
- [ ] Log shape, order, and values of batched tensors
- [ ] Compare single-pass batch CFG vs two-pass separate CFG
- [ ] Test with known prompt and seed
- [ ] Measure output quality difference

**Expected Outcomes**:
- **Scenario A**: Batching works correctly - No action needed
- **Scenario B**: Batch order needs reordering - Add reorder logic (1 day)
- **Scenario C**: Batching fundamentally broken - Major issue (1-2 weeks)

**Deliverable**: Test report with:
- Batch order confirmation
- CFG computation validation
- Fix if needed
- Quality comparison (SSIM/LPIPS)

**Assigned Files to Inspect**:
- `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (lines 310-335)
- `/home/tt-admin/ComfyUI-tt_standalone/comfy/samplers.py` (lines 298-326)
- TT UNet forward pass implementation

**Success Criteria**:
- Batch order confirmed correct
- CFG computation validated
- No quality degradation vs separate passes

---

### TASK 3: TT_FullDenoise Stability Audit 🟡 MEDIUM

**Background**:
- TT_FullDenoise is the proven success pattern
- We're building Option 2.5A on top of this
- Must ensure it's production-ready

**Your Task**: Audit for production readiness

**Checklist**:
- [ ] Error handling: All edge cases covered?
- [ ] Memory leaks: Shared memory cleanup reliable?
- [ ] Resource cleanup: Models unloaded properly?
- [ ] Input validation: Malformed inputs handled?
- [ ] Concurrent requests: Thread-safe?
- [ ] Fallback paths: What if TT hardware unavailable?

**Testing Plan**:
- Run 100 consecutive generations (memory leak test)
- Test with invalid inputs (error handling)
- Test concurrent requests (thread safety)
- Test hardware disconnect/reconnect (recovery)

**Deliverable**:
- Production readiness report
- List of fixes needed (if any)
- Test coverage report
- Updated documentation

**Assigned Files**:
- `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py` (TT_FullDenoise)
- `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` (cleanup logic)

**Success Criteria**:
- Zero crashes in 100 consecutive runs
- Clean memory profile (no leaks)
- Graceful degradation if hardware unavailable

---

## Phase 1: Bridge Stabilization (Weeks 2-5)

### Architecture: Proven Bridge Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Node Graph                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │ Load Model   │─────▶│TT_FullDenoise│─────▶│VAE Decode│ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │ Unix Socket / Shared Memory
                               ▼
                    ┌────────────────────────┐
                    │   Bridge Server        │
                    │  (TT Hardware Access)  │
                    │                        │
                    │  ┌──────────────────┐  │
                    │  │ Text Encoder     │  │
                    │  │ (bfloat16)       │  │
                    │  └──────────────────┘  │
                    │  ┌──────────────────┐  │
                    │  │ UNet Loop        │  │
                    │  │ (20 steps)       │  │
                    │  │ (bfloat16)       │  │
                    │  └──────────────────┘  │
                    │  ┌──────────────────┐  │
                    │  │ VAE Encoder      │  │
                    │  │ (bfloat16)       │  │
                    │  └──────────────────┘  │
                    └────────────────────────┘
```

**Key Properties**:
- Single precision boundary (input/output only)
- Bridge owns denoising loop
- State managed on bridge side
- Proven with SSIM 0.998+

### Work Items

**Week 2: Fix Critical Issues**
- [ ] Implement timestep conversion fix (if needed from Phase 0)
- [ ] Implement CFG batch reordering (if needed from Phase 0)
- [ ] Apply production hardening fixes (from TT_FullDenoise audit)
- [ ] Add comprehensive error logging
- [ ] Create bridge health monitoring

**Week 3: Enhanced Bridge Features**
- [ ] Add bridge-level caching (model weights, compiled ops)
- [ ] Implement smart batching (multiple requests → single TT batch)
- [ ] Add bridge performance metrics (latency tracking)
- [ ] Optimize shared memory transfers (pinned memory, async)

**Week 4: Testing & Documentation**
- [ ] Comprehensive test suite (unit + integration)
- [ ] Performance benchmarking vs standalone server
- [ ] User documentation (when to use TT_FullDenoise)
- [ ] Developer documentation (bridge architecture)

**Week 5: Release v1.0**
- [ ] Package as ComfyUI custom node
- [ ] Installation guide
- [ ] Example workflows
- [ ] Troubleshooting guide

**Deliverables**:
- Production-ready TT_FullDenoise node
- Bridge server with monitoring
- Comprehensive documentation
- Benchmark results (vs CUDA)

**Success Criteria**:
- SSIM >= 0.95 vs standalone server
- < 1% failure rate over 1000 runs
- Performance within 15% of standalone
- Clear user documentation

---

## Phase 2: Native Hook Integration (Weeks 6-11)

### Architecture: Bridge + Native Hooks

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Node Graph                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐         │
│  │ Load Model   │─▶│ ControlNet   │─▶│TT_KSampler│─▶...   │
│  └──────────────┘  └──────────────┘  └──────────┘         │
│                           │                  │               │
│                           │  ┌───────────────┘               │
│                           ▼  ▼                               │
│                    ┌──────────────┐                         │
│                    │TT Model      │◄── Native Hook         │
│                    │Wrapper       │    (model_function_     │
│                    │              │     wrapper)            │
│                    └──────────────┘                         │
└──────────────────────────────────┼──────────────────────────┘
                                   │ Unix Socket / Shared Memory
                                   ▼
                        ┌────────────────────────┐
                        │   Bridge Server        │
                        │  (per-step calls)      │
                        │                        │
                        │  ┌──────────────────┐  │
                        │  │ UNet Forward     │  │
                        │  │ (single step)    │  │
                        │  │ (bfloat16)       │  │
                        │  └──────────────────┘  │
                        └────────────────────────┘
```

**Key Properties**:
- Native ControlNet/IP-Adapter hooks in ComfyUI
- Bridge for UNet execution only
- Per-step calling pattern (for extension support)
- State management in wrapper

### Extension Strategy

**ControlNet Support** (Week 6-7):
```python
# In custom_nodes/tt_model_wrapper/controlnet_integration.py

class TTModelWithControlNet:
    def __init__(self, tt_bridge, controlnet):
        self.bridge = tt_bridge
        self.controlnet = controlnet

    def apply_model(self, x, timestep, c):
        # 1. Apply ControlNet conditioning (native ComfyUI)
        if self.controlnet:
            control_features = self.controlnet(x, timestep, c)
            c = merge_conditioning(c, control_features)

        # 2. Send to TT bridge for UNet forward
        epsilon = self.bridge.forward(x, timestep, c)

        return epsilon
```

**Implementation Tasks**:
- [ ] Week 6: Design `model_function_wrapper` hook
- [ ] Week 6: Implement TTModelWrapper class
- [ ] Week 6: Add timestep conversion logic (if needed)
- [ ] Week 7: Integrate ControlNet conditioning
- [ ] Week 7: Test with standard ControlNet nodes
- [ ] Week 7: Benchmark performance impact

**IP-Adapter Support** (Week 8-9):
- [ ] Week 8: Research IP-Adapter hook points
- [ ] Week 8: Design IP-Adapter integration
- [ ] Week 9: Implement IP-Adapter wrapper
- [ ] Week 9: Test with IP-Adapter nodes

**Custom Sampler Support** (Week 10):
- [ ] Design TT-compatible sampler interface
- [ ] Port `sample_euler_tt` to use native hooks
- [ ] Test with ComfyUI's scheduler system

**Week 11: Integration Testing & Release v2.0**
- [ ] Test complex workflows (ControlNet + IP-Adapter)
- [ ] Performance optimization (reduce bridge overhead)
- [ ] Documentation updates
- [ ] Release v2.0 with extension support

**Deliverables**:
- TTModelWrapper with ControlNet support
- IP-Adapter integration
- Custom sampler compatibility
- Performance benchmarks

**Success Criteria**:
- ControlNet works with standard nodes
- IP-Adapter works with standard nodes
- Performance degradation < 20% vs bridge loop
- User workflows "just work"

---

## Phase 3: Optimization & Production Hardening (Weeks 12-14)

### Performance Targets

| Workflow | Target | Baseline (Bridge Loop) | Stretch Goal |
|----------|--------|------------------------|--------------|
| Basic txt2img | < 3.0s/img | 2.8s/img | 2.5s/img |
| txt2img + ControlNet | < 4.5s/img | 4.5s/img | 4.0s/img |
| txt2img + IP-Adapter | < 5.0s/img | 5.0s/img | 4.5s/img |
| Complex (CN + IPA) | < 6.5s/img | 7.2s/img | 6.0s/img |

### Optimization Tasks

**Week 12: Profiling & Bottleneck Analysis**
- [ ] Profile end-to-end workflow
- [ ] Identify bottlenecks (transfer vs compute)
- [ ] Measure bridge overhead per call
- [ ] Analyze memory usage patterns

**Week 13: Targeted Optimizations**
- [ ] Implement tensor transfer optimizations (pinned memory, async)
- [ ] Add operation fusion where possible
- [ ] Optimize CFG computation (fused kernel?)
- [ ] Implement compilation caching

**Week 14: Production Hardening**
- [ ] Comprehensive error handling
- [ ] Fallback paths for all operations
- [ ] Resource leak prevention
- [ ] Load testing (sustained throughput)
- [ ] Stress testing (edge cases)

**Deliverables**:
- Performance optimization report
- Production-ready release
- Load testing results
- Final documentation

**Success Criteria**:
- Meet or exceed performance targets
- Zero memory leaks in 24-hour stress test
- Graceful degradation under load
- < 0.1% error rate in production

---

## Parallel Work Streams

### Stream 1: Critical Validation (Week 1)
**Owner**: Senior Engineer (bridge architecture expert)
- Timestep bug investigation
- CFG batching verification
- TT_FullDenoise audit

### Stream 2: Bridge Stabilization (Weeks 2-5)
**Owner**: Mid-level Engineer (Python/ComfyUI experience)
- Production hardening
- Performance monitoring
- Documentation

### Stream 3: Native Hook Design (Weeks 4-6)
**Owner**: Senior Engineer (ComfyUI internals expert)
- `model_function_wrapper` design
- TTModelWrapper architecture
- Can start in Week 4 (parallel with Stream 2)

### Stream 4: Extension Integration (Weeks 7-11)
**Owner**: Mid-level Engineer (extension expertise)
- ControlNet integration
- IP-Adapter integration
- Custom samplers

### Stream 5: Testing & Documentation (Ongoing)
**Owner**: QA Engineer + Tech Writer
- Test suite development
- Documentation
- User guides

---

## Risk Management

### High-Risk Items

**Risk 1: Timestep Format Bug is Blocker**
- **Likelihood**: 40%
- **Impact**: 2-3 week delay
- **Mitigation**: Investigate in Week 1 (Phase 0)
- **Fallback**: Stay with bridge loop only (Option 3), defer native hooks

**Risk 2: Per-step Bridge Overhead Too High**
- **Likelihood**: 30%
- **Impact**: Performance < targets
- **Mitigation**: Profile early (Week 12), optimize transfers
- **Fallback**: Use bridge loop for fast path, per-step for extensions only

**Risk 3: ControlNet Integration Harder Than Expected**
- **Likelihood**: 25%
- **Impact**: 1-2 week delay
- **Mitigation**: Research upfront (Week 6), prototype early
- **Fallback**: Document limitation, provide workaround

**Risk 4: Scheduler State Desync in Per-step**
- **Likelihood**: 35%
- **Impact**: Incorrect outputs
- **Mitigation**: Comprehensive state tracking in wrapper
- **Fallback**: Use ComfyUI's scheduler only, don't call TT scheduler directly

---

## Decision Points & Pivots

### Week 1 Decision: Continue or Pivot?

**After Phase 0 validation**:

```
IF timestep bug is blocker AND fix > 3 weeks:
  → PIVOT to pure Option 3 (bridge loop only)
  → Defer native hooks to future release

IF CFG batching is broken AND unfixable:
  → STOP integration
  → Investigate fundamental TT UNet issue

IF both validations pass:
  → CONTINUE with Option 2.5A as planned
```

### Week 7 Decision: Extension Support Viable?

**After ControlNet integration attempt**:

```
IF ControlNet works with < 20% performance hit:
  → CONTINUE with IP-Adapter and other extensions

IF ControlNet performance hit > 40%:
  → RECONSIDER: Maybe keep bridge loop as primary
  → Extensions as secondary feature

IF ControlNet fundamentally broken:
  → PIVOT back to Option 3 (bridge loop only)
```

### Week 11 Decision: Ship v2.0 or Optimize More?

**After integration complete**:

```
IF performance meets targets:
  → SHIP v2.0

IF performance within 30% of targets:
  → Allocate 2 more weeks for optimization
  → SHIP v2.1 with optimizations

IF performance not acceptable:
  → Defer v2.0
  → Prioritize optimization (Phase 3)
  → SHIP when targets met
```

---

## Success Metrics

### Phase 0 (Week 1)
- ✅ Timestep bug investigated (blocker assessment)
- ✅ CFG batching verified (working correctly)
- ✅ TT_FullDenoise production-ready

### Phase 1 (Week 5)
- ✅ v1.0 released (bridge loop)
- ✅ SSIM >= 0.95 vs standalone
- ✅ < 1% failure rate
- ✅ Documentation complete

### Phase 2 (Week 11)
- ✅ v2.0 released (with extensions)
- ✅ ControlNet working
- ✅ IP-Adapter working
- ✅ Performance within 20% of bridge loop

### Phase 3 (Week 14)
- ✅ Performance targets met
- ✅ Production hardening complete
- ✅ Load testing passed
- ✅ Final documentation

---

## Questions for Planning Agent

When creating the detailed implementation plan, please address:

1. **Granular Task Breakdown**: Break each work item into 1-day tasks with clear deliverables

2. **Dependency Mapping**: Create critical path diagram showing which tasks block others

3. **Resource Allocation**: Specify skill requirements for each task (senior/mid/junior engineer)

4. **Testing Strategy**: Define unit, integration, and E2E tests for each component

5. **Rollback Plans**: For each phase, what's the rollback strategy if issues arise?

6. **Validation Checkpoints**: Where are the go/no-go decision points?

7. **Parallel Optimization**: What tasks can run in parallel to compress timeline?

8. **Code Reuse Strategy**: Which existing code can be reused vs needs rewriting?

9. **Integration Points**: Precisely where does new code hook into existing ComfyUI/TT code?

10. **Documentation Plan**: What docs are needed at each phase?

---

## Deliverable Format

Create a detailed implementation plan document that includes:

1. **Executive Summary** (1 page)
   - Option 2.5A overview
   - Timeline summary
   - Key milestones
   - Success criteria

2. **Phase 0: Critical Validation** (3-5 pages)
   - Detailed investigation procedures
   - Expected outcomes
   - Decision criteria
   - Risk assessment

3. **Phase 1: Bridge Stabilization** (5-7 pages)
   - Week-by-week task breakdown
   - Code changes needed
   - Testing strategy
   - Documentation requirements

4. **Phase 2: Native Hook Integration** (7-10 pages)
   - Architecture diagrams
   - Component designs
   - Integration procedures
   - Extension support matrix

5. **Phase 3: Optimization & Hardening** (3-5 pages)
   - Performance optimization strategy
   - Production readiness checklist
   - Load testing plan
   - Final release criteria

6. **Appendices**
   - A: Dependency graph (critical path)
   - B: Resource allocation matrix
   - C: Risk registry
   - D: Decision tree flowcharts
   - E: Code file mapping (what changes where)

---

## Success Criteria for This Planning Exercise

The plan is complete when it provides:

✅ **Actionable**: Every task has clear inputs, outputs, and success criteria
✅ **Sequenced**: Dependencies identified, parallel work maximized
✅ **Risk-aware**: Risks identified with mitigation strategies
✅ **Testable**: Clear validation at each phase
✅ **Realistic**: Timeline accounts for complexity and unknowns
✅ **Flexible**: Pivot points identified if assumptions prove wrong
✅ **Comprehensive**: Covers code, tests, docs, and deployment
✅ **Evidence-based**: References analysis documents and proven patterns

---

**Now, create the detailed implementation plan for Option 2.5A using the planner-agent with all available context.**
