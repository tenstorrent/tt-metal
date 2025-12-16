# ControlNet Feasibility Scenarios

**Purpose:** Decision paths based on Phase 0 ControlNet validation results  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part B, Task 0.3

---

## Scenario Overview

Phase 0 Day 3-5 validates ControlNet feasibility. Three scenarios are possible:

| Scenario | Condition | Decision |
|----------|-----------|----------|
| Scenario 1 | CPU-side works | GO |
| Scenario 2 | Needs TT-side | DEFER |
| Scenario 3 | Architecture incompatible | PIVOT |

---

## Scenario 1: CPU-Side Works (GO)

### Condition

ControlNet runs successfully on ComfyUI side (CPU/GPU), and control_hint tensor can be transferred to bridge and used in TT UNet.

### Validation Signals

- [ ] ControlNet model loads and runs on ComfyUI side
- [ ] control_hint tensor generated successfully
- [ ] Tensor transfers via IPC without corruption
- [ ] TT UNet accepts control_hint (or minor modification only)
- [ ] Visual output shows control influence

### Evidence Required

```
1. Prototype code showing:
   - ControlNet forward pass on CPU
   - control_hint extraction
   - IPC transfer
   - Injection into UNet call

2. Test results showing:
   - Data integrity (tensor comparison)
   - Shape preservation
   - Visual validation (control visible in output)

3. Performance data showing:
   - ControlNet CPU time: <Xms
   - Transfer time: <Yms
   - Acceptable overhead
```

### Decision: GO

**Action:** Proceed to Phase 1 with ControlNet in scope for Week 3

**Week 3 Plan:**
- Day 1: Finalize control_hint handling
- Day 2: Implement TT_ApplyControlNet node
- Day 3-4: Integration testing
- Day 5: Human validation

---

## Scenario 2: Needs TT-Side Implementation (DEFER)

### Condition

ControlNet cannot run on CPU side effectively, or TT UNet requires significant modification to support control_hint injection.

### Validation Signals

- [ ] ControlNet runs on CPU but performance unacceptable
- [ ] TT UNet requires > 8 hours modification
- [ ] Control_hint format incompatible with TT expectations
- [ ] Integration complexity exceeds Week 3 allocation

### Evidence Required

```
1. Analysis showing:
   - Why CPU-side is problematic
   - Specific UNet changes needed
   - Effort estimate (hours)

2. Performance data showing:
   - CPU-side ControlNet latency
   - Impact on total generation time
   - Whether overhead is acceptable

3. Technical assessment:
   - Required TT-side implementation scope
   - Dependencies on TT infrastructure
   - Estimated timeline for TT-side
```

### Decision: DEFER

**Action:** Remove ControlNet from Phase 1 scope, plan for Phase 2

**Phase 1 Adjusted Scope:**
- Week 1: Per-step API (unchanged)
- Week 2: Session management (unchanged)
- Week 3: Extended testing + optimization (no ControlNet)
- Week 4: Testing + ADRs (unchanged)
- Week 5: Documentation + release (unchanged)

**Phase 2 Plan:**
- TT-side ControlNet implementation
- Timeline: 3-4 weeks additional
- Depends on: TT infrastructure readiness

### Actions for Deferral

1. **Document findings:**
   ```markdown
   # ControlNet Deferral Decision
   
   ## Reason for Deferral
   [Specific technical reasons]
   
   ## Requirements for Phase 2
   - [Requirement 1]
   - [Requirement 2]
   
   ## Estimated Phase 2 Effort
   [X weeks]
   
   ## Impact on Phase 1
   Week 3 repurposed for:
   - Extended per-step testing
   - Performance optimization
   - Additional scheduler validation
   ```

2. **Update success criteria:**
   - Remove ControlNet SSIM from Phase 1
   - Remove human validation for ControlNet from Phase 1
   - Add replacement metrics for Week 3

3. **Communicate to stakeholders:**
   - ControlNet deferred to Phase 2
   - Phase 1 still delivers per-step API
   - Phase 1 enables future ControlNet

---

## Scenario 3: Architecture Incompatible (PIVOT)

### Condition

Fundamental incompatibility prevents ControlNet integration through either CPU-side or TT-side approach.

### Validation Signals

- [ ] ControlNet conditioning fundamentally different from TT UNet expectations
- [ ] Required modifications would break existing functionality
- [ ] No clear path to integration without major redesign
- [ ] Performance would be severely degraded (> 50% overhead)

### Evidence Required

```
1. Technical analysis showing:
   - Specific incompatibility
   - Why neither approach works
   - Architectural constraints

2. Options assessment:
   - Alternative approaches considered
   - Why each was rejected
   - Cost/benefit analysis

3. Impact analysis:
   - Effect on Phase 1 goals
   - Effect on strategic roadmap
   - Stakeholder implications
```

### Decision: PIVOT

**Action:** Escalate to leadership, explore alternatives

### Pivot Options

#### Option A: Focus on Per-Step API Only

**Scope:**
- Complete per-step API (Phase 1 core)
- Defer all conditioning extensions
- Deliver value through custom samplers, basic per-step control

**Impact:**
- Reduced Phase 1 scope
- ControlNet/IP-Adapter in future phase
- Still validates per-step pattern

#### Option B: Alternative Conditioning Approach

**Scope:**
- Investigate alternative conditioning mechanisms
- May require different ControlNet implementation
- Custom TT-specific approach

**Impact:**
- Additional research time
- May not be compatible with standard ComfyUI workflows
- Novel implementation required

#### Option C: Native Integration Priority

**Scope:**
- Accelerate native integration instead
- ControlNet via native path
- Skip bridge extension for advanced features

**Impact:**
- Longer timeline to ControlNet (12-17 weeks)
- More comprehensive solution
- Higher risk but higher reward

### Actions for Pivot

1. **Immediate escalation:**
   - Meeting with Technical Lead and PM
   - Present findings and options
   - Get decision authority guidance

2. **Documentation:**
   ```markdown
   # ControlNet Pivot Decision
   
   ## Incompatibility Discovered
   [Detailed technical explanation]
   
   ## Options Considered
   1. Option A: [description, pros, cons]
   2. Option B: [description, pros, cons]
   3. Option C: [description, pros, cons]
   
   ## Recommended Path
   [Selected option with rationale]
   
   ## Timeline Impact
   [Updated timeline]
   ```

3. **Stakeholder communication:**
   - Clear explanation of constraint
   - Options presented
   - Recommendation made
   - Decision required from leadership

---

## Decision Tree

```
Phase 0 Day 3-5: ControlNet Validation
                |
                v
        [Run prototype]
                |
                v
    [CPU-side ControlNet works?]
           /            \
         Yes             No
          |               |
          v               v
[control_hint      [Can implement
 transfers OK?]     on TT side?]
    /      \           /      \
  Yes       No       Yes       No
   |         |        |         |
   v         v        v         v
[TT UNet   [Debug   [DEFER   [PIVOT
 accepts?]  issue]   to P2]   options]
  /    \
Yes    No
 |      |
 v      v
[GO]  [Minor mod
       needed?]
        /    \
      Yes     No
       |       |
       v       v
     [GO]   [DEFER
     +mod    to P2]
```

---

## Summary Matrix

| Scenario | Signals | Decision | Phase 1 Impact | Next Steps |
|----------|---------|----------|----------------|------------|
| 1: Works | All green | GO | Full scope | Week 3 as planned |
| 2: Needs TT | CPU slow or UNet needs major mod | DEFER | No ControlNet | Repurpose Week 3 |
| 3: Incompatible | No clear path | PIVOT | Reduced/changed | Escalate, choose option |

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part B
