# Architecture Context Document

**Purpose:** Explain the WHY behind key architectural decisions  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part D

---

## Overview

This document explains the architectural decisions for Phase 1: Bridge Extension, providing context for why specific approaches were chosen over alternatives.

---

## Decision 1: Why Per-Timestep Pattern

### Context

ComfyUI's ecosystem includes powerful features that require per-timestep control:
- **ControlNet:** Injects conditioning at each denoising step
- **IP-Adapter:** Modifies generation based on reference images
- **Custom Samplers:** Implement advanced sampling strategies

### The Problem

The current `denoise_only` operation runs the entire denoising loop internally:

```
ComfyUI                         Bridge
   |                               |
   |---- denoise(steps=20) ------->|
   |                               | [runs 20 steps internally]
   |<------ final_latents ---------|
```

This blocks external control because:
1. ComfyUI cannot inject conditioning between steps
2. Custom samplers cannot modify intermediate results
3. ControlNet cannot apply per-step guidance

### The Solution

Implement `handle_denoise_step_single` for external loop control:

```
ComfyUI                         Bridge
   |                               |
   | for step in range(20):        |
   |---- denoise_step(step) ------>|
   |<------ latents ----------------|
   |   [apply ControlNet]          |
   |---- denoise_step(step+1) ---->|
   |                               |
```

### Why This Over Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Callbacks in full-loop | IPC overhead per callback too high |
| Native integration first | 12-17 weeks vs 4-6 weeks |
| Batch steps (2-5 at time) | Deferred - can optimize later |

### Data Flow Diagram

```
+------------------+         +-------------------+         +-----------------+
|    ComfyUI       |         |      Bridge       |         |   TT Hardware   |
|                  |         |                   |         |                 |
| +-------------+  |         | +--------------+  |         | +------------+  |
| | Scheduler   |  |  IPC    | | Step Handler |  |  TT API | |    UNet    |  |
| | (Euler,etc) |--|-------->| | (stateless)  |--|-------->| |  Forward   |  |
| +-------------+  |         | +--------------+  |         | +------------+  |
|       |         |         |        |          |         |       |        |
|       v         |         |        v          |         |       v        |
| +-------------+  |         | +--------------+  |         | +------------+  |
| | ControlNet  |  |         | |Format Convert|  |         | |  Result    |  |
| | (CPU/GPU)   |  |  hint   | |   Helpers    |  |         | +------------+  |
| +-------------+--|-------->| +--------------+  |         |                 |
+------------------+         +-------------------+         +-----------------+
```

---

## Decision 2: Why Stateless Bridge (ComfyUI Owns Scheduler)

### Context

Schedulers (Euler, DPM++, etc.) maintain state across timesteps:
- `timesteps[]` - sequence of timestep values
- `sigmas[]` - corresponding sigma values
- `current_step` - progress tracking

### The Options

**Option A: Stateless Bridge (Selected)**
```
ComfyUI owns scheduler state, sends timestep/sigma per call
Bridge receives values, passes to UNet, returns result
```

**Option B: Stateful Bridge (Fallback)**
```
Bridge owns scheduler state, initializes from config
ComfyUI sends step index only
Bridge computes timestep/sigma internally
```

### Why Stateless (Option A)

| Factor | Stateless (A) | Stateful (B) |
|--------|---------------|--------------|
| Implementation complexity | Lower | Higher |
| Synchronization bugs | None possible | Risk of desync |
| Custom scheduler support | Immediate | Requires bridge update |
| Bridge maintenance | Simpler | More complex |
| ComfyUI compatibility | Native | Requires adaptation |

### Comparison Diagram

**Option A (Stateless):**
```
+----------------+                    +----------------+
|    ComfyUI     |                    |     Bridge     |
| +-----------+  |     timestep=0.5   | +------------+ |
| | Scheduler |--|------------------->| | No state   | |
| | timesteps |  |     sigma=1.2      | |            | |
| | sigmas    |  |                    | | Passes to  | |
| +-----------+  |<-------------------| | UNet       | |
+----------------+     result         +----------------+
```

**Option B (Stateful):**
```
+----------------+                    +----------------+
|    ComfyUI     |                    |     Bridge     |
|                |     step_index=5   | +-----------+  |
| (no scheduler) |------------------->| | Scheduler |  |
|                |                    | | timesteps |  |
|                |                    | | sigmas    |  |
|                |<-------------------| +-----------+  |
+----------------+     result         +----------------+

Risk: Schedulers can diverge!
```

### When to Switch to Option B

Signals that Option B may be needed:
- ComfyUI scheduler state cannot be serialized
- IPC overhead for state transfer is prohibitive
- Custom schedulers have non-standard state

See: `/home/tt-admin/tt-metal/SCHEDULER_STATE_OPTIONS.md` for full fallback procedure.

---

## Decision 3: Why Phase 0 Validation

### Context

Major implementation work carries risk of mid-project pivots. Historical data shows:
- 40% of projects require significant pivots
- Average pivot cost: 2-4 weeks of rework
- Early validation reduces pivot rate to < 5%

### Risk Analysis

| Risk | Without Phase 0 | With Phase 0 |
|------|-----------------|--------------|
| Scheduler sync issues | Discovered in Week 2-3 | Discovered Day 1-2 |
| ControlNet infeasible | Discovered in Week 3 | Discovered Day 3-5 |
| IPC performance issue | Discovered in Week 2 | Discovered Day 1 |
| Total pivot risk | ~40% | < 5% |
| Cost of pivot | 2-4 weeks | 0-3 days |

### Phase 0 ROI Calculation

```
Investment:  5 days of validation
Return:      35% reduction in pivot probability
             If pivot happens without Phase 0: 3 weeks lost
             Expected value of Phase 0: 0.35 * 3 weeks = ~1 week saved
             
Net ROI: 1 week saved - 1 week invested = break-even minimum
         With reduced stress and better planning: highly positive
```

### Validation vs Implementation

```
Time
  |
  |  Phase 0         Phase 1
  |  [=====]   [========================]
  |  Validate        Build
  |     |               |
  v     v               v
     If GO:          Full implementation
     PROCEED         with confidence
     
     If NO-GO:       Pivot early
     PIVOT           minimal loss
```

---

## Decision 4: Why Bridge Extension Before Native Integration

### Context

Two paths to full ComfyUI support:

**Path A: Bridge Extension (Selected)**
- Timeline: 5-6 weeks
- Output: Per-step API via IPC
- Validates: Per-timestep patterns

**Path B: Native Integration**
- Timeline: 12-17 weeks
- Output: Direct ComfyUI integration
- Validates: Everything at once

### Comparison

| Factor | Bridge Extension | Native Integration |
|--------|-----------------|-------------------|
| Time to ControlNet | 5-6 weeks | 12-17 weeks |
| Investment before validation | 5-6 weeks | 12-17 weeks |
| Risk of major pivot | Medium | High |
| Reusability of work | 100% | 100% |
| Performance (final) | Good (~10% overhead) | Optimal |

### Strategic Reasoning

```
                    Week
            0   5   10   15   20
Bridge:     [===]
                 ControlNet works!
                 |
                 v Pattern validated
                 
Native:     [================]
                              ControlNet works!
                              (if no pivots needed)
```

**Key insight:** Bridge extension validates per-timestep patterns in 5-6 weeks. Native integration builds on validated patterns, reducing risk.

### How Bridge Connects to Native

```
Phase 1: Bridge Extension      Phase 2: SD3.5        Phase 3: Native
[Per-step API via IPC]    --> [Same API, new model] --> [Direct calls]
[Session management]      --> [Same patterns]       --> [Adapted]
[Format conversions]      --> [100% reuse]          --> [100% port]
[ControlNet approach]     --> [Same approach]       --> [Same approach]
```

---

## Decision 5: Why CPU-Side ControlNet

### Context

ControlNet can theoretically run in two locations:

**Option 1: CPU/GPU Side (ComfyUI)**
- ControlNet model runs on standard hardware
- Only control_hint tensor sent to TT

**Option 2: TT Side**
- ControlNet model runs on TT hardware
- Requires TT-side implementation

### Analysis

```
CPU-Side ControlNet:

ComfyUI (CPU/GPU)              Bridge (TT)
+-------------------+          +------------------+
| 1. Load image     |          |                  |
| 2. Preprocess     |          |                  |
| 3. Run ControlNet |          |                  |
|    (standard PyTorch)        |                  |
| 4. Get control_hint|         |                  |
|         |         |   IPC    |                  |
|         +---------|--------->| 5. Inject hint   |
|                   |          | 6. UNet forward  |
|                   |<---------|--- result        |
+-------------------+          +------------------+

Advantages:
- Uses existing ComfyUI ControlNet
- No TT-side development needed
- Immediate compatibility

TT-Side ControlNet:

ComfyUI                        Bridge (TT)
+-------------------+          +------------------+
| 1. Load image     |   IPC    | 2. Preprocess    |
|         +---------|--------->| 3. Run ControlNet|
|                   |          |    (TT-optimized)|
|                   |          | 4. UNet forward  |
|                   |<---------|--- result        |
+-------------------+          +------------------+

Advantages:
- Potentially faster
- Single device execution

Disadvantages:
- Requires TT ControlNet implementation
- Timeline extends significantly
```

### Decision

CPU-Side ControlNet selected because:
1. Immediate compatibility with ComfyUI ecosystem
2. No additional TT-side development
3. Performance is acceptable (control_hint is small tensor)
4. Validated in Phase 0 Day 3-5

---

## Data Flow Diagrams

### Complete Per-Step Flow (ASCII)

```
+============================================================================+
|                              ComfyUI Process                                |
+============================================================================+
|                                                                             |
|  +------------------+     +------------------+     +-------------------+    |
|  | Load Checkpoint  |---->| Load ControlNet  |---->| Prepare Latents   |    |
|  +------------------+     +------------------+     +-------------------+    |
|                                   |                        |               |
|                                   v                        v               |
|                           +------------------+     +-------------------+    |
|                           | Preprocess Image |     | Scheduler Setup   |    |
|                           | (Canny/Depth)    |     | (timesteps, sigmas)|   |
|                           +------------------+     +-------------------+    |
|                                   |                        |               |
|                                   v                        |               |
|                           +------------------+             |               |
|                           | Run ControlNet   |             |               |
|                           | (CPU/GPU)        |             |               |
|                           +------------------+             |               |
|                                   |                        |               |
|                                   v                        v               |
|                           +----------------------------------------+       |
|                           |          Denoising Loop                |       |
|                           |  for step in range(steps):             |       |
|                           |    timestep = scheduler.timesteps[step]|       |
|                           |    sigma = scheduler.sigmas[step]      |       |
|                           |    control = controlnet.hint(step)     |       |
|                           |                                        |       |
|                           |    +--------------------------------+  |       |
|                           |    |        IPC CALL                |  |       |
|                           |    | denoise_step_single(           |  |       |
|                           |    |   latents, timestep, sigma,    |  |       |
|                           |    |   conditioning, control_hint)  |  |       |
|                           |    +--------------------------------+  |       |
|                           |              |                         |       |
+============================================================================+
                                           | IPC (Socket/SHM)
                                           v
+============================================================================+
|                              Bridge Process                                 |
+============================================================================+
|                                                                             |
|  +------------------+     +------------------+     +-------------------+    |
|  | Receive Request  |---->| Session Lookup   |---->| Parse Parameters  |    |
|  +------------------+     +------------------+     +-------------------+    |
|                                                            |               |
|                                                            v               |
|  +------------------+     +------------------+     +-------------------+    |
|  | Get Latents SHM  |---->| Format Convert   |---->| Get Control Hint  |    |
|  +------------------+     +------------------+     +-------------------+    |
|                                                            |               |
|                                                            v               |
|                                                    +-------------------+    |
|                                                    |  TT UNet Forward  |    |
|                                                    |  (with control)   |    |
|                                                    +-------------------+    |
|                                                            |               |
|                                                            v               |
|  +------------------+     +------------------+     +-------------------+    |
|  | Format Convert   |---->| Write Latents SHM|---->| Return Response   |    |
|  +------------------+     +------------------+     +-------------------+    |
|                                                                             |
+============================================================================+
```

---

## Summary

| Decision | Choice | Key Reason |
|----------|--------|------------|
| Per-timestep pattern | Yes | Required for ControlNet ecosystem |
| Stateless bridge | ComfyUI owns scheduler | Simpler, no sync bugs |
| Phase 0 validation | 5 days | Reduces pivot risk from 40% to <5% |
| Bridge before native | Bridge first | Validates patterns in 5-6 weeks |
| CPU-side ControlNet | CPU/GPU | Immediate compatibility |

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part D
