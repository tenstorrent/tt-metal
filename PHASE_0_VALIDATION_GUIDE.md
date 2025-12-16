# Phase 0 Validation Guide

**Purpose:** Exact steps to execute each Phase 0 validation task  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part B

---

## Task 0.1: Scheduler State Synchronization Design

### Objective

Design and prototype the stateless bridge approach for scheduler synchronization.

### Day 1: Analysis and Initial Design

#### Step 1.1.1: Examine Current Implementation (2 hours)

**Files to examine:**

1. `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`
   - Lines ~500-700: `handle_denoise_only` implementation
   - Focus on: How scheduler state is currently managed
   - Note: Where timesteps/sigmas come from

2. ComfyUI scheduler implementations
   - Path: `comfy/k_diffusion/sampling.py` (or similar)
   - Focus on: State variables maintained
   - Note: How timesteps/sigmas are accessed

**Output checklist:**
- [ ] Current scheduler usage documented
- [ ] State variables identified (timesteps, sigmas, etc.)
- [ ] Data flow between ComfyUI and bridge understood

#### Step 1.1.2: Design Stateless Interface (2 hours)

**Design document location:** `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md`

**Required sections:**

```markdown
# Scheduler Synchronization Design

## Current State
[Document how scheduler currently works]

## Proposed Design (Option A: Stateless)
[Document stateless approach]

## Interface Specification
[Define exact parameters passed]

## Error Handling
[Define error cases and responses]

## Alternatives Considered
[Document Option B for reference]
```

**Interface to document:**

```python
# Per-step call parameters
{
    "timestep": float,        # Current timestep value
    "timestep_index": int,    # Step index (0 to N-1)
    "sigma": float,           # Current sigma value
    "total_steps": int,       # Total steps in sequence
}
```

### Day 2: Prototype and Testing

#### Step 1.1.3: Implement Prototype (3 hours)

**File:** `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`

**Code to add:**

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a SINGLE denoising step.
    
    PROTOTYPE - Phase 0 validation only
    """
    # Extract scheduler state from params (stateless approach)
    timestep = params["timestep"]
    timestep_index = params["timestep_index"]
    sigma = params["sigma"]
    
    # Log for validation
    logger.info(f"Step {timestep_index}: timestep={timestep}, sigma={sigma}")
    
    # Extract other params
    latent_shm = params["latent_shm"]
    conditioning_shm = params["conditioning_shm"]
    
    # Get latents from shared memory
    latents = self._get_tensor_from_shm(latent_shm)
    
    # TODO: Full implementation in Week 1
    # For now, validate data flow
    
    # Run single step (placeholder)
    output = self._run_single_step(latents, timestep, conditioning)
    
    return {
        "success": True,
        "latent_shm": self._put_tensor_to_shm(output),
        "step_metadata": {
            "timestep": timestep,
            "timestep_index": timestep_index,
            "sigma": sigma,
        }
    }
```

#### Step 1.1.4: Write Validation Tests (2 hours)

**File:** `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_scheduler_sync.py`

```python
import pytest

class TestSchedulerSync:
    """Validate scheduler synchronization."""
    
    def test_timestep_passed_correctly(self):
        """Verify timestep value arrives correctly."""
        params = {
            "timestep": 999.0,
            "timestep_index": 0,
            "sigma": 14.6,
            "total_steps": 20,
            # ... other params
        }
        
        response = handler.handle_denoise_step_single(params)
        
        assert response["success"]
        assert response["step_metadata"]["timestep"] == 999.0
    
    def test_sigma_passed_correctly(self):
        """Verify sigma value arrives correctly."""
        params = {
            "timestep": 999.0,
            "timestep_index": 0,
            "sigma": 14.6,
            # ...
        }
        
        response = handler.handle_denoise_step_single(params)
        
        assert response["step_metadata"]["sigma"] == 14.6
    
    def test_full_sequence(self):
        """Verify 20-step sequence processes correctly."""
        # Simulate Euler scheduler timesteps
        timesteps = [999, 950, 900, ...]  # Full sequence
        sigmas = [14.6, 13.8, 13.0, ...]
        
        for i, (t, s) in enumerate(zip(timesteps, sigmas)):
            params = {
                "timestep": t,
                "timestep_index": i,
                "sigma": s,
                "total_steps": 20,
                # ...
            }
            
            response = handler.handle_denoise_step_single(params)
            assert response["success"], f"Step {i} failed"
    
    def test_error_on_missing_timestep(self):
        """Verify error when timestep missing."""
        params = {
            # "timestep" missing
            "timestep_index": 0,
            "sigma": 14.6,
        }
        
        response = handler.handle_denoise_step_single(params)
        
        assert not response["success"]
        assert "timestep" in response["error"]["message"]
```

**Run tests:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_scheduler_sync.py -v
```

### Acceptance Criteria

- [ ] Design document exists at specified path
- [ ] Design reviewed and signed off
- [ ] Prototype passes all scheduler sync tests
- [ ] Error handling covers missing/invalid parameters

---

## Task 0.2: IPC Performance Baseline

### Objective

Measure current full-loop latency to establish per-step budget.

### Day 1: Setup and Measurement (Parallel with Task 0.1)

#### Step 0.2.1: Create Benchmark Script (1 hour)

**File:** `/home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_ipc.py`

```python
#!/usr/bin/env python3
"""IPC Performance Baseline Measurement"""

import time
import statistics
import json
from comfyui_bridge.client import BridgeClient

def measure_full_loop_latency(client, steps=20, iterations=10):
    """Measure full denoise_only latency."""
    latencies = []
    
    # Warm-up
    client.denoise_only(steps=steps)
    
    for i in range(iterations):
        start = time.perf_counter()
        client.denoise_only(steps=steps)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        print(f"Iteration {i+1}: {latency_ms:.1f}ms")
    
    return {
        "iterations": iterations,
        "steps": steps,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }

def measure_ipc_roundtrip(client, iterations=100):
    """Measure raw IPC round-trip latency."""
    latencies = []
    
    for i in range(iterations):
        start = time.perf_counter_ns()
        client.ping()  # Minimal operation
        end = time.perf_counter_ns()
        
        latency_ms = (end - start) / 1_000_000
        latencies.append(latency_ms)
    
    return {
        "iterations": iterations,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "max_ms": max(latencies),
    }

def calculate_budget(full_loop_results, ipc_results):
    """Calculate per-step budget."""
    steps = full_loop_results["steps"]
    full_loop_avg = full_loop_results["mean_ms"]
    
    per_step_baseline = full_loop_avg / steps
    ipc_overhead = ipc_results["mean_ms"]
    overhead_percent = (ipc_overhead / per_step_baseline) * 100
    
    return {
        "full_loop_avg_ms": full_loop_avg,
        "steps": steps,
        "per_step_baseline_ms": per_step_baseline,
        "ipc_overhead_ms": ipc_overhead,
        "overhead_percent": overhead_percent,
        "budget_ok": overhead_percent < 10,
        "headroom_percent": 10 - overhead_percent,
    }

if __name__ == "__main__":
    client = BridgeClient()
    
    print("=" * 60)
    print("IPC Performance Baseline Measurement")
    print("=" * 60)
    
    print("\n1. Measuring full-loop latency...")
    full_loop = measure_full_loop_latency(client, steps=20, iterations=10)
    print(f"   Mean: {full_loop['mean_ms']:.1f}ms")
    print(f"   Median: {full_loop['median_ms']:.1f}ms")
    
    print("\n2. Measuring IPC round-trip...")
    ipc = measure_ipc_roundtrip(client, iterations=100)
    print(f"   Mean: {ipc['mean_ms']:.2f}ms")
    print(f"   P99: {ipc['p99_ms']:.2f}ms")
    
    print("\n3. Budget calculation...")
    budget = calculate_budget(full_loop, ipc)
    print(f"   Per-step baseline: {budget['per_step_baseline_ms']:.1f}ms")
    print(f"   IPC overhead: {budget['ipc_overhead_ms']:.2f}ms")
    print(f"   Overhead: {budget['overhead_percent']:.1f}%")
    print(f"   Budget OK: {budget['budget_ok']}")
    print(f"   Headroom: {budget['headroom_percent']:.1f}%")
    
    # Save results
    results = {
        "full_loop": full_loop,
        "ipc": ipc,
        "budget": budget,
    }
    
    with open("/home/tt-admin/tt-metal/docs/ipc_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n4. Results saved to docs/ipc_baseline_results.json")
    
    # Exit code based on budget check
    exit(0 if budget["budget_ok"] else 1)
```

**Run benchmark:**
```bash
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_ipc.py
```

### Expected Results

| Metric | Expected Value |
|--------|----------------|
| Full-loop (20 steps) | ~2000-3000ms |
| Per-step baseline | ~100-150ms |
| IPC round-trip | 1-5ms |
| Overhead | < 10% |
| Headroom | > 60% |

### Acceptance Criteria

- [ ] IPC latency < 10ms (P99)
- [ ] Full-loop baseline documented
- [ ] Per-step budget calculated
- [ ] Results saved to JSON file

---

## Task 0.3: ControlNet Architecture Feasibility Study

### Objective

Validate that ControlNet conditioning can be injected via bridge.

### Day 3: Analysis

#### Step 0.3.1: Analyze ComfyUI ControlNet (2.5 hours)

**Files to examine:**

1. ComfyUI `comfy/controlnet.py`
   - Where ControlNet model is loaded
   - Where conditioning is computed
   - What output format is used

2. ComfyUI `comfy/sample.py` or sampling modules
   - Where ControlNet is applied
   - How control_hint is used in denoising

**Questions to answer:**
- [ ] What is the shape of control_hint? `[B, C, H, W]`
- [ ] What dtype is control_hint? `float32/float16`
- [ ] Where is ControlNet applied? `Before/during UNet forward`
- [ ] Can control_hint be computed independently? `Yes/No`

**Output document:** Data flow notes

#### Step 0.3.2: Document Data Flow (1.5 hours)

**Create:** `/home/tt-admin/tt-metal/docs/controlnet_data_flow.md`

```markdown
# ControlNet Data Flow Analysis

## ComfyUI Side

1. Image loading and preprocessing
   - Input: Control image (e.g., Canny edges)
   - Output: Preprocessed tensor [B, 3, H, W]

2. ControlNet forward
   - Input: Preprocessed image, timestep
   - Output: control_hint tensors (multiple scales)

3. Conditioning preparation
   - Attach control_hints to conditioning dict

## Bridge Side (Proposed)

1. Receive control_hint via IPC
   - Format: [B, C, H, W] per scale
   - Transfer: Shared memory

2. Inject into UNet forward
   - Method: Pass as additional argument
   - Injection point: [specific location]

## Validation Points

- [ ] control_hint can be serialized
- [ ] control_hint transfer is fast enough
- [ ] TT UNet can accept control_hint
```

### Day 4: Prototype

#### Step 0.3.3: Examine TT UNet (2.5 hours)

**Files to examine:**

1. `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/`
   - UNet implementation files
   - Forward function signature
   - Current conditioning handling

**Questions to answer:**
- [ ] Does TT UNet have control_hint parameter? `Yes/No`
- [ ] If no, where can it be added? `[location]`
- [ ] What format does TT UNet expect? `[format]`
- [ ] Estimated effort to add support? `[hours]`

#### Step 0.3.4: Prototype Transfer (3 hours)

**Test code:**

```python
# Test control_hint transfer via bridge

import torch
import numpy as np

def test_control_hint_transfer():
    """Prototype control_hint IPC transfer."""
    
    # Simulate control_hint from ControlNet
    control_hint = torch.randn(1, 320, 64, 64, dtype=torch.float32)
    
    # Convert to numpy for SHM
    hint_np = control_hint.numpy()
    
    # Simulate SHM write (placeholder)
    shm_handle = write_to_shm(hint_np)
    
    # Simulate SHM read on bridge side
    hint_received = read_from_shm(shm_handle)
    
    # Verify integrity
    assert hint_np.shape == hint_received.shape
    assert np.allclose(hint_np, hint_received, rtol=1e-5)
    
    # Convert back to tensor
    hint_tensor = torch.from_numpy(hint_received)
    
    print(f"Transfer successful: {hint_tensor.shape}")
    return True

# Run test
test_control_hint_transfer()
```

#### Step 0.3.5: Visual Validation (1.5 hours)

**Test workflow:**

1. Load a Canny edge image
2. Run through ComfyUI ControlNet (CPU)
3. Get control_hint tensor
4. Transfer to bridge prototype
5. Verify tensor arrives correctly
6. (Optional) Run UNet with control_hint if supported

### Day 5: Report and Decision

#### Step 0.3.6: Write Feasibility Report (2 hours)

**File:** `/home/tt-admin/tt-metal/docs/controlnet_feasibility_report.md`

```markdown
# ControlNet Feasibility Report

## Executive Summary
[1 paragraph: GO / CONDITIONAL GO / NO-GO]

## Findings

### CPU-Side ControlNet
- Works: [Yes/No]
- Evidence: [description]

### Control_hint Transfer
- Works: [Yes/No]
- Latency: [Xms per transfer]
- Size: [X MB typical]

### TT UNet Integration
- Current support: [Yes/No/Partial]
- Required changes: [description]
- Effort estimate: [hours]

## Recommendation

**Decision:** [GO / CONDITIONAL GO / NO-GO]

**Rationale:**
[Detailed explanation]

**Constraints (if CONDITIONAL GO):**
[List any constraints]

## Next Steps
[Actions based on decision]
```

### Go/No-Go Decision Framework

| Scenario | Validation Result | Decision |
|----------|-------------------|----------|
| CPU-side works, transfer works, UNet ready | All green | **GO** |
| CPU-side works, transfer works, UNet needs minor mod | Minor work | **GO** |
| CPU-side works, transfer works, UNet needs major mod | Significant work | **CONDITIONAL GO** (defer ControlNet) |
| Transfer fails or unacceptable latency | Blocking issue | **NO-GO** for ControlNet |
| Architecture fundamentally incompatible | Major issue | **PIVOT** |

### Acceptance Criteria

- [ ] ControlNet architecture documented
- [ ] Prototype validates data transfer
- [ ] TT UNet analysis complete
- [ ] Clear Go/No-Go decision made
- [ ] Feasibility report written

---

## Phase 0 Final Output

### Day 5 Afternoon: Compile Results

**Deliverable 1:** Feasibility Report
- Location: `/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md`
- Content: All three task findings combined

**Deliverable 2:** Go/No-Go Recommendation
- Format: Meeting + written decision
- Participants: Lead Engineer, Technical Lead
- Output: Decision recorded in decision log

**Deliverable 3:** Updated Risk Register
- Update: Risk likelihoods based on findings
- Add: Any new risks discovered
- Update: Mitigation strategies

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part B
