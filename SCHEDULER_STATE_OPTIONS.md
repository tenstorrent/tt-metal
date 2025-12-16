# Scheduler State Options

**Purpose:** Document both scheduler sync approaches with full specifications  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part D, Decision 1

---

## Overview

Two options exist for handling scheduler state synchronization between ComfyUI and the bridge:

- **Option A (Selected):** Stateless Bridge - ComfyUI owns scheduler
- **Option B (Fallback):** Stateful Bridge - Bridge owns scheduler

---

## Option A: Stateless Bridge (SELECTED)

### Description

ComfyUI maintains all scheduler state. The bridge receives timestep and sigma values with each step call and performs no internal scheduling.

### Architecture

```
ComfyUI Process                    Bridge Server Process
+------------------------+         +------------------------+
| Scheduler State        |         | No scheduler state     |
| - timesteps[20]        |   -->   |                        |
| - sigmas[20]           |   IPC   | Receives per call:     |
| - current_step         |         | - timestep             |
+------------------------+         | - sigma                |
        |                          | - timestep_index       |
        v                          +------------------------+
  for step in range(20):                    |
    timestep = timesteps[step]              |
    sigma = sigmas[step]                    |
    response = bridge.step(                 v
      timestep=timestep,           +------------------------+
      sigma=sigma,                 | UNet Forward           |
      ...                          | (uses provided values) |
    )                              +------------------------+
```

### API Specification

**Request:**
```python
{
    "operation": "denoise_step_single",
    "params": {
        "session_id": "uuid-string",
        "latent_shm": {"name": "shm_xxx", "shape": [1, 4, 128, 128], "dtype": "float32"},
        "timestep": 999.0,           # From scheduler
        "timestep_index": 0,         # Step number (0 to N-1)
        "sigma": 14.6146,            # From scheduler
        "total_steps": 20,           # Total steps in sequence
        "conditioning_shm": {...},
        "guidance_scale": 7.5,
    }
}
```

**Response:**
```python
{
    "success": True,
    "latent_shm": {"name": "shm_yyy", "shape": [1, 4, 128, 128], "dtype": "float32"},
    "step_metadata": {
        "timestep": 999.0,
        "timestep_index": 0,
        "sigma": 14.6146,
        "execution_time_ms": 95.3,
    }
}
```

### Implementation

```python
# handlers.py

def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute single step with externally provided scheduler state."""
    
    # Extract scheduler state from params (NOT maintained internally)
    timestep = params["timestep"]
    timestep_index = params["timestep_index"]
    sigma = params["sigma"]
    
    # Validate
    if timestep < 0 or timestep > 1000:
        return self._error("InvalidTimestep", f"timestep {timestep} out of range")
    
    if sigma < 0:
        return self._error("InvalidSigma", f"sigma must be positive, got {sigma}")
    
    # Get latents
    latents = self._get_tensor_from_shm(params["latent_shm"])
    
    # Run UNet with provided scheduler values
    output = self.runner.denoise_step(
        latents=latents,
        timestep=timestep,
        sigma=sigma,
        ...
    )
    
    return {
        "success": True,
        "latent_shm": self._put_tensor_to_shm(output),
        "step_metadata": {"timestep": timestep, "timestep_index": timestep_index, ...}
    }
```

### Advantages

- Simple bridge implementation
- No state synchronization bugs
- Supports any scheduler ComfyUI implements
- Custom schedulers work immediately
- Easier to debug (state visible in ComfyUI)

### Disadvantages

- More data per IPC call
- ComfyUI must know scheduler details
- Cannot optimize based on upcoming steps

---

## Option B: Stateful Bridge (FALLBACK)

### Description

Bridge maintains scheduler state internally. ComfyUI sends configuration at session start, and bridge computes timestep/sigma values internally.

### Architecture

```
ComfyUI Process                    Bridge Server Process
+------------------------+         +------------------------+
| No scheduler state     |         | Scheduler State        |
|                        |   -->   | - timesteps[20]        |
| Sends at session start:|   IPC   | - sigmas[20]           |
| - scheduler_type       |         | - current_step         |
| - total_steps          |         +------------------------+
+------------------------+                   |
        |                                    v
        v                          +------------------------+
  for step in range(20):           | Compute internally:    |
    response = bridge.step(        | timestep = timesteps[i]|
      step_index=step,             | sigma = sigmas[i]      |
    )                              +------------------------+
                                            |
                                            v
                                   +------------------------+
                                   | UNet Forward           |
                                   +------------------------+
```

### API Specification

**Session Create Request:**
```python
{
    "operation": "session_create",
    "params": {
        "model_id": "sdxl",
        "scheduler_type": "euler",
        "total_steps": 20,
        "scheduler_config": {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
        }
    }
}
```

**Step Request:**
```python
{
    "operation": "denoise_step_single",
    "params": {
        "session_id": "uuid-string",
        "latent_shm": {...},
        "step_index": 0,  # Only index needed, not timestep/sigma
        "conditioning_shm": {...},
    }
}
```

### Implementation

```python
# handlers.py

def handle_session_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create session with internal scheduler."""
    
    scheduler_type = params["scheduler_type"]
    total_steps = params["total_steps"]
    scheduler_config = params.get("scheduler_config", {})
    
    # Initialize scheduler internally
    scheduler = self._create_scheduler(scheduler_type, scheduler_config)
    scheduler.set_timesteps(total_steps)
    
    # Store in session
    session = DenoiseSession(
        session_id=str(uuid.uuid4()),
        scheduler=scheduler,
        timesteps=scheduler.timesteps.tolist(),
        sigmas=scheduler.sigmas.tolist(),
        current_step=0,
        total_steps=total_steps,
    )
    
    self.sessions[session.session_id] = session
    
    return {"success": True, "session_id": session.session_id}


def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute single step with internally maintained scheduler."""
    
    session = self.sessions[params["session_id"]]
    step_index = params["step_index"]
    
    # Get scheduler state from internal storage
    timestep = session.timesteps[step_index]
    sigma = session.sigmas[step_index]
    
    # Run UNet
    output = self.runner.denoise_step(
        latents=latents,
        timestep=timestep,
        sigma=sigma,
        ...
    )
    
    return {...}
```

### Advantages

- Less data per IPC call
- Bridge has full scheduler knowledge
- Can optimize based on sequence
- Cleaner ComfyUI integration

### Disadvantages

- More complex bridge implementation
- Risk of scheduler state divergence
- Must implement/port all scheduler types
- Custom schedulers require bridge updates

---

## Comparison Table

| Factor | Option A (Stateless) | Option B (Stateful) |
|--------|---------------------|---------------------|
| Bridge complexity | Low | High |
| Sync bug risk | None | Medium |
| IPC data size | Larger | Smaller |
| Custom scheduler support | Immediate | Requires implementation |
| Debug visibility | High | Lower |
| Maintenance burden | Lower | Higher |
| Performance potential | Good | Slightly better |
| Implementation time | 1-2 days | 3-5 days |

---

## When to Switch to Option B

### Signals That Option B May Be Needed

1. **Scheduler State Too Complex**
   - ComfyUI scheduler has state that cannot be serialized
   - Internal state affects computation non-trivially
   - State includes GPU tensors

2. **IPC Overhead Unacceptable**
   - Extra timestep/sigma data causes measurable slowdown
   - > 15% overhead attributed to data transfer
   - Bandwidth limited scenarios

3. **Accuracy Issues**
   - Results differ between ComfyUI-computed and bridge-received values
   - Floating point precision loss in transfer
   - Scheduler-specific edge cases

### Decision Criteria for Switching

```
Should switch to Option B if ANY:
- [ ] IPC overhead from state transfer > 5% (significant portion of budget)
- [ ] Scheduler state cannot be transferred accurately
- [ ] Custom scheduler support blocked by Option A limitations
- [ ] Debug evidence shows state synchronization as root cause
```

### Rollback Procedure

If Option A fails during Phase 0:

**Day 1-2 (if needed):**
1. Document specific failure mode
2. Assess Option B feasibility
3. Estimate implementation effort

**Day 3-5 (if switching):**
1. Implement Option B scheduler factory
2. Port Euler and DPM++ schedulers
3. Update tests for internal scheduling
4. Re-run validation suite

**Impact on Timeline:**
- Add 2-3 days to Phase 0
- Week 1 schedule unchanged (uses final approach)

---

## Hybrid Approach (If Needed)

If neither pure option works, consider hybrid:

```python
# Hybrid: ComfyUI provides timesteps, bridge verifies

def handle_denoise_step_single(self, params):
    # Receive from ComfyUI
    provided_timestep = params["timestep"]
    provided_sigma = params["sigma"]
    step_index = params["timestep_index"]
    
    # Bridge also has scheduler for verification
    expected_timestep = self.session.scheduler.timesteps[step_index]
    expected_sigma = self.session.scheduler.sigmas[step_index]
    
    # Verify (warn but continue)
    if abs(provided_timestep - expected_timestep) > 0.001:
        logger.warning(f"Timestep mismatch: got {provided_timestep}, expected {expected_timestep}")
    
    # Use provided values (trust ComfyUI)
    ...
```

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part D
