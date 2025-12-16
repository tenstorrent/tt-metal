# Risk Mitigation Playbook

**Purpose:** Detailed response procedures for all identified risks  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part F

---

## Risk Register Overview

| Risk ID | Risk | Likelihood | Impact | Priority |
|---------|------|------------|--------|----------|
| R1 | Scheduler desync | 30% | Critical | HIGH |
| R2 | ControlNet TT-side needed | 25% | High | HIGH |
| R3 | IPC overhead > 15% | 40% | Medium | MEDIUM |
| R4 | Session cleanup failures | 20% | Medium | MEDIUM |
| R5 | Format conversion bugs | 15% | High | MEDIUM |
| R6 | Memory leaks | 25% | High | MEDIUM |
| R7 | ComfyUI scheduler incompatible | 20% | Medium | LOW |
| R8 | Team availability | 15% | High | LOW |

---

## Risk R1: Scheduler Desync

**Risk:** Scheduler state becomes desynchronized between ComfyUI and bridge

**Likelihood:** 30%  
**Impact:** Critical - Results in incorrect outputs

### Detection Signals

- [ ] Output images show visual artifacts (banding, noise)
- [ ] SSIM scores drop significantly mid-generation
- [ ] Different outputs from same seed/prompt
- [ ] Error messages about timestep mismatch

### When to Escalate

**Escalate immediately if:**
- SSIM drops below 0.95 on previously working prompts
- Visual artifacts appear in > 10% of generations
- Error logs show timestep validation failures

**Proceed normally if:**
- Isolated issue with specific scheduler type
- Can reproduce and isolate to specific steps

### Mitigation Steps

**Level 1: Diagnostic (1-2 hours)**
```python
# Add detailed logging
def handle_denoise_step_single(self, params):
    logger.debug(f"Received timestep={params['timestep']}, "
                 f"sigma={params['sigma']}, "
                 f"step_index={params['timestep_index']}")
    
    # Validate against expected sequence
    if self.session:
        expected_timestep = self._get_expected_timestep(params['timestep_index'])
        if abs(params['timestep'] - expected_timestep) > 0.001:
            logger.warning(f"Timestep mismatch: expected {expected_timestep}, "
                          f"got {params['timestep']}")
```

**Level 2: Validation Layer (4-8 hours)**
- Add timestep validation in bridge
- Add sigma validation against timestep
- Add step index sequence checking
- Return clear error on mismatch

**Level 3: Fallback to Option B (1-2 days)**
- Switch to stateful bridge pattern
- Bridge maintains scheduler internally
- ComfyUI sends only initial config
- Requires more extensive refactoring

### Decision Tree

```
Scheduler desync detected
    |
    v
[Add logging] --> [Can reproduce?]
    |                   |
    | No                | Yes
    v                   v
[Monitor]        [Analyze logs]
                       |
                       v
               [Root cause identified?]
                   |           |
                   | No        | Yes
                   v           v
           [Fallback B]   [Fix root cause]
```

### Contacts

- Primary: Lead Engineer
- Escalation: Technical Architect
- External: ComfyUI community (if scheduler bug)

---

## Risk R2: ControlNet TT-side Needed

**Risk:** ControlNet cannot run on CPU/GPU side, requires TT hardware implementation

**Likelihood:** 25%  
**Impact:** High - Requires significant additional work

### Detection Signals

- [ ] Phase 0 prototype fails to produce correct conditioning
- [ ] ControlNet output dimensions incompatible with bridge
- [ ] UNet cannot accept control hints in expected format
- [ ] Performance unacceptable with CPU-side ControlNet

### When to Escalate

**Escalate immediately if:**
- Phase 0 Day 4 prototype shows fundamental incompatibility
- Required UNet modifications exceed 40 hours
- ControlNet conditioning format not transferable

**Proceed normally if:**
- Minor format conversions needed
- Performance acceptable with CPU-side

### Mitigation Steps

**Level 1: Adapt Format (4-8 hours)**
- Add format conversion for ControlNet output
- Adjust tensor shapes as needed
- Document required conversions

**Level 2: Defer ControlNet (0 hours, planning)**
- Remove ControlNet from Phase 1 scope
- Document current findings
- Plan TT-side implementation for Phase 2
- Focus Phase 1 on per-step API only

**Level 3: Pivot to Alternative (1-2 weeks)**
- Investigate alternative conditioning approaches
- Consider IP-Adapter first (simpler)
- Re-evaluate after Phase 1 core complete

### Decision Tree

```
Phase 0 Day 3-4 findings
    |
    v
[Control hint transfer works?]
    |           |
    | Yes       | No
    v           v
[GO]       [Format issue?]
              |       |
              | Yes   | No
              v       v
         [Fix format] [Defer to Phase 2]
```

### Contacts

- Primary: Lead Engineer
- Domain Expert: ControlNet maintainer
- Escalation: Project Manager (scope change)

---

## Risk R3: IPC Overhead > 15%

**Risk:** Inter-process communication adds more than acceptable overhead

**Likelihood:** 40%  
**Impact:** Medium - May require optimization or target adjustment

### Detection Signals

- [ ] Phase 0 Day 1 IPC measurement > 10ms
- [ ] Per-step total time significantly higher than full-loop/steps
- [ ] Network latency spikes during generation
- [ ] CPU utilization high during IPC

### When to Escalate

**Escalate immediately if:**
- IPC P99 latency > 15ms
- Total per-step overhead > 20%
- No obvious optimization path

**Proceed normally if:**
- IPC latency 10-15ms (borderline)
- Clear optimization opportunities exist

### Mitigation Steps

**Level 1: Quick Optimizations (2-4 hours)**
```python
# Batch tensor transfers
def _batch_transfer(self, tensors):
    # Transfer multiple tensors in single IPC call
    combined = torch.cat([t.flatten() for t in tensors])
    transferred = self._transfer_single(combined)
    return self._unbatch(transferred, [t.shape for t in tensors])

# Use shared memory for large tensors
def _use_shm_for_large(self, tensor, threshold_mb=1):
    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    if size_mb > threshold_mb:
        return self._transfer_via_shm(tensor)
    return self._transfer_direct(tensor)
```

**Level 2: Architecture Optimization (1-2 days)**
- Switch to Unix domain sockets (if using TCP)
- Implement tensor pooling to reduce allocations
- Add compression for large tensors
- Implement async IPC where possible

**Level 3: Adjust Targets (0 hours, agreement)**
- Document actual performance achieved
- Propose adjusted target (e.g., 15%)
- Get stakeholder agreement
- Update success criteria

### Decision Tree

```
IPC overhead measured
    |
    v
[< 10%?] --> Yes --> [PASS]
    |
    | No
    v
[10-15%?] --> [Apply Level 1 optimizations]
    |                    |
    | > 15%              v
    v               [Re-measure]
[Apply Level 2]          |
    |                    v
    v               [< 10%?] --> [PASS]
[Re-measure]             |
    |                    | No
    v                    v
[< 15%?]            [Adjust targets]
```

### Contacts

- Primary: Lead Engineer
- Performance Expert: Systems Engineer
- Escalation: Technical Lead

---

## Risk R4: Session Cleanup Failures

**Risk:** Session resources not properly cleaned up, causing resource leaks

**Likelihood:** 20%  
**Impact:** Medium - Degrades stability over time

### Detection Signals

- [ ] Memory usage grows over time
- [ ] Session count grows without bound
- [ ] GPU memory not released
- [ ] "Too many open files" errors

### When to Escalate

**Escalate immediately if:**
- Server crashes after extended operation
- Memory growth > 10MB per generation
- GPU memory exhaustion

**Proceed normally if:**
- Isolated cleanup failures
- Can identify specific session causing issue

### Mitigation Steps

**Level 1: Add Monitoring (2-4 hours)**
```python
class SessionManager:
    def __init__(self):
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Track session metrics."""
        self._metrics = {
            "created": 0,
            "completed": 0,
            "expired": 0,
            "current": 0,
        }
    
    def log_metrics(self):
        logger.info(f"Sessions: {self._metrics}")
```

**Level 2: Aggressive Cleanup (4-8 hours)**
- Reduce timeout to 10 minutes
- Add explicit resource release on complete
- Add periodic forced cleanup
- Add memory limit per session

**Level 3: Restart Strategy (0 hours, operational)**
- Schedule periodic server restarts
- Implement graceful restart (drain sessions)
- Add health check endpoint

### Decision Tree

```
Session cleanup issue detected
    |
    v
[Add monitoring] --> [Pattern identified?]
    |                      |
    | Isolated             | Systematic
    v                      v
[Fix specific]      [Level 2: Aggressive cleanup]
                           |
                           v
                    [Issue resolved?]
                       |       |
                       | No    | Yes
                       v       v
                   [Restart strategy] [Document fix]
```

### Contacts

- Primary: Lead Engineer
- Operations: DevOps Engineer
- Escalation: Technical Lead

---

## Risk R5: Format Conversion Bugs

**Risk:** Tensor format conversion produces incorrect results

**Likelihood:** 15%  
**Impact:** High - Silent corruption of results

### Detection Signals

- [ ] Output images have wrong colors
- [ ] Tensor shapes unexpected after conversion
- [ ] NaN or Inf values in output
- [ ] SSIM scores very low (< 0.5)

### When to Escalate

**Escalate immediately if:**
- Cannot reproduce correct output
- Format conversion produces NaN/Inf
- Multiple conversion paths broken

**Proceed normally if:**
- Single model type affected
- Can identify specific conversion issue

### Mitigation Steps

**Level 1: Round-Trip Validation (2-4 hours)**
```python
def validate_format_conversion(original, converted):
    """Validate conversion preserves data."""
    # Convert back
    reconverted = convert_back(converted)
    
    # Check shape
    assert original.shape == reconverted.shape, "Shape mismatch"
    
    # Check values
    diff = (original - reconverted).abs().max()
    assert diff < 1e-5, f"Value diff {diff} exceeds tolerance"
```

**Level 2: Add Extensive Tests (4-8 hours)**
- Test all model types
- Test edge cases (min/max values)
- Test different batch sizes
- Test different resolutions

**Level 3: Fallback to Known-Good (0 hours)**
- Revert to previous working conversion
- Document limitation
- Plan proper fix for next release

### Contacts

- Primary: Lead Engineer
- Format Expert: ML Engineer
- Escalation: Technical Architect

---

## Risk R6: Memory Leaks

**Risk:** Memory not properly released during operation

**Likelihood:** 25%  
**Impact:** High - Server instability

### Detection Signals

- [ ] Memory usage grows during stress test
- [ ] GPU memory not released after generation
- [ ] Python garbage collector not reclaiming objects
- [ ] Server OOM after extended operation

### When to Escalate

**Escalate immediately if:**
- Memory growth > 50MB per 100 generations
- GPU memory exhausted within 100 generations
- Server crashes from OOM

**Proceed normally if:**
- Small consistent growth (< 10MB per 100 gen)
- Growth stabilizes after warmup

### Mitigation Steps

**Level 1: Memory Profiling (2-4 hours)**
```python
import tracemalloc
import gc

def profile_memory(func, *args, **kwargs):
    tracemalloc.start()
    gc.collect()
    
    snapshot1 = tracemalloc.take_snapshot()
    result = func(*args, **kwargs)
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:10]:
        print(stat)
    
    return result
```

**Level 2: Explicit Cleanup (4-8 hours)**
- Add `del` statements for large tensors
- Add explicit `gc.collect()` after generations
- Clear caches periodically
- Use context managers for resources

**Level 3: Reduce Memory Footprint (1-2 days)**
- Reduce cache sizes
- Implement tensor pooling
- Use memory-mapped tensors where possible
- Consider model offloading

### Contacts

- Primary: Lead Engineer
- Memory Expert: Systems Engineer
- Escalation: Technical Lead

---

## Escalation Paths

### Level 1: Team Lead
- Contact: [Team Lead Name]
- Method: Slack/Teams message
- Response time: 2-4 hours
- Use for: Technical blockers, design decisions

### Level 2: Technical Architect
- Contact: [Architect Name]
- Method: Email + meeting request
- Response time: 1 day
- Use for: Architecture changes, scope changes

### Level 3: Project Manager
- Contact: [PM Name]
- Method: Formal escalation email
- Response time: 4 hours
- Use for: Timeline impact, resource needs

### Level 4: Executive
- Contact: [Director Name]
- Method: Through PM
- Response time: 1-2 days
- Use for: Major pivots, budget requests

---

## Risk Review Schedule

| Frequency | Activity | Participants |
|-----------|----------|--------------|
| Daily | Risk check in standup | Team |
| Weekly | Risk review in weekly | Team + Lead |
| Phase Gate | Comprehensive risk review | Full team + PM |

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
