# Success Criteria Validation Framework

**Purpose:** Define precise measurement methods for all Phase 0 and Phase 1 success criteria  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part E

---

## Phase 0 Gate Criteria

### Criterion 0.1: Scheduler Sync Design Approved

**Target:** Design document approved by technical lead

**Measurement Method:**
1. Design document at: `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md`
2. Review meeting conducted with 2+ senior engineers
3. Sign-off obtained with signatures and date

**Acceptance Test:**
```
[ ] Design document exists and is complete
[ ] Review meeting minutes documented
[ ] At least 2 approvers signed off
[ ] No open blocking comments
```

**Evidence Required:**
- Signed design document
- Review meeting notes
- Approval email or sign-off form

---

### Criterion 0.2: IPC Baseline < 10ms

**Target:** IPC round-trip latency < 10ms

**Measurement Method:**
```python
# File: /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_ipc.py

import time
import statistics

def measure_ipc_latency(iterations=100):
    """Measure IPC round-trip latency."""
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter_ns()
        # Send minimal message and receive response
        response = client.send_receive({"op": "ping"})
        end = time.perf_counter_ns()
        
        latency_ms = (end - start) / 1_000_000
        latencies.append(latency_ms)
    
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "max_ms": max(latencies),
    }

# Run benchmark
results = measure_ipc_latency()
assert results["p99_ms"] < 10, f"IPC P99 latency {results['p99_ms']:.2f}ms exceeds 10ms target"
```

**Acceptance Test:**
```
[ ] 100 round-trip measurements completed
[ ] Mean latency < 10ms
[ ] P99 latency < 10ms
[ ] Results documented in baseline report
```

**Evidence Required:**
- Benchmark script output
- Results table with all measurements
- Hardware/environment details

---

### Criterion 0.3: ControlNet Feasibility GO

**Target:** ControlNet conditioning transfer validated

**Measurement Method:**
1. Data flow analysis document complete
2. Prototype demonstrates control_hint transfer
3. Visual verification of conditioning effect

**Acceptance Test:**
```
[ ] Data flow diagram exists
[ ] Prototype code transfers control_hint tensor
[ ] Tensor shape preserved across IPC
[ ] Visual output shows control influence
```

**Evidence Required:**
- Data flow diagram
- Prototype code
- Test images showing control effect

---

## Phase 1 Gate Criteria

### Criterion 1.1: Per-Step SSIM >= 0.99

**Target:** Per-step denoising achieves SSIM >= 0.99 vs full-loop baseline

**Measurement Method:**
```python
# File: /home/tt-admin/tt-metal/comfyui_bridge/tests/validate_ssim.py

import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compare_per_step_vs_full_loop(prompt, steps=20, seed=42):
    """Compare per-step vs full-loop output."""
    
    # Full loop baseline
    full_loop_output = run_denoise_only(prompt, steps=steps, seed=seed)
    
    # Per-step execution
    per_step_output = run_per_step_sequence(prompt, steps=steps, seed=seed)
    
    # Convert to numpy for SSIM
    full_np = full_loop_output.cpu().numpy().squeeze().transpose(1, 2, 0)
    per_step_np = per_step_output.cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Calculate SSIM (multichannel for RGB)
    ssim_score = ssim(full_np, per_step_np, multichannel=True, data_range=1.0)
    
    return ssim_score

# Run validation with 100 diverse prompts
def validate_ssim_criteria():
    prompts = load_test_prompts("test_prompts_100.txt")
    results = []
    
    for prompt in prompts:
        score = compare_per_step_vs_full_loop(prompt)
        results.append({"prompt": prompt, "ssim": score, "pass": score >= 0.99})
    
    pass_rate = sum(r["pass"] for r in results) / len(results)
    assert pass_rate >= 0.95, f"Only {pass_rate:.1%} prompts achieved SSIM >= 0.99"
    
    return results
```

**Acceptance Test:**
```
[ ] 100 test prompts executed
[ ] SSIM calculated for each prompt
[ ] >= 95% of prompts achieve SSIM >= 0.99
[ ] Any failures investigated and documented
```

**Evidence Required:**
- Test results spreadsheet (100 rows)
- Summary statistics
- Failure analysis (if any)

---

### Criterion 1.2: ControlNet SSIM >= 0.90

**Target:** ControlNet workflows achieve SSIM >= 0.90 vs CPU reference

**Measurement Method:**
```python
# For each ControlNet type (Canny, Depth, OpenPose)
def validate_controlnet_ssim(control_type, test_images):
    """Validate ControlNet output vs CPU reference."""
    results = []
    
    for image_path in test_images:
        # Generate CPU reference
        cpu_output = run_controlnet_cpu(image_path, control_type)
        
        # Generate TT output
        tt_output = run_controlnet_tt(image_path, control_type)
        
        # Calculate SSIM
        ssim_score = calculate_ssim(cpu_output, tt_output)
        results.append({"image": image_path, "ssim": ssim_score})
    
    avg_ssim = sum(r["ssim"] for r in results) / len(results)
    assert avg_ssim >= 0.90, f"{control_type} average SSIM {avg_ssim:.3f} below 0.90"
    
    return results
```

**Acceptance Test:**
```
[ ] 10 Canny edge images tested, average SSIM >= 0.90
[ ] 10 Depth images tested, average SSIM >= 0.90
[ ] 10 OpenPose images tested, average SSIM >= 0.90
[ ] All individual images SSIM >= 0.85 (no severe outliers)
```

**Evidence Required:**
- Test images used
- SSIM scores per image per type
- CPU reference images
- TT output images

---

### Criterion 1.3: Human Validation 5/5

**Target:** 5/5 human raters confirm "correct" visual output

**Measurement Method:**

**Process:**
1. Select 15 test images (5 per ControlNet type)
2. Recruit 5 independent raters (not involved in development)
3. Present each image pair (control input + generated output)
4. Ask: "Does the generated image correctly follow the control guidance?"
5. Rating scale: Correct / Partially Correct / Incorrect

**Evaluation Form:**
```
Rater ID: ___________
Date: ___________

For each image pair, rate whether the generated image correctly follows the control:

Image 1 (Canny - edges): [ ] Correct  [ ] Partially  [ ] Incorrect
Image 2 (Canny - scene): [ ] Correct  [ ] Partially  [ ] Incorrect
...
Image 15 (OpenPose - 5): [ ] Correct  [ ] Partially  [ ] Incorrect

Comments: ___________
```

**Acceptance Test:**
```
[ ] 5 independent raters recruited
[ ] All 15 images rated by all 5 raters
[ ] >= 90% of ratings are "Correct"
[ ] No image receives majority "Incorrect"
```

**Evidence Required:**
- Rater evaluation forms (5 forms)
- Summary statistics
- Any rater comments

---

### Criterion 1.4: Per-Step Latency < 10% Overhead

**Target:** Per-step approach adds < 10% overhead vs full-loop / steps

**Measurement Method:**
```python
# File: /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py

def benchmark_latency_overhead(steps=20, iterations=10):
    """Compare per-step vs full-loop latency."""
    
    # Measure full-loop
    full_loop_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_denoise_only(steps=steps)
        end = time.perf_counter()
        full_loop_times.append(end - start)
    
    full_loop_avg = statistics.mean(full_loop_times)
    per_step_baseline = full_loop_avg / steps
    
    # Measure per-step
    per_step_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for step in range(steps):
            run_denoise_step_single(step)
        end = time.perf_counter()
        per_step_times.append(end - start)
    
    per_step_avg = statistics.mean(per_step_times)
    per_step_per_iteration = per_step_avg / steps
    
    overhead = (per_step_per_iteration - per_step_baseline) / per_step_baseline
    
    return {
        "full_loop_avg_ms": full_loop_avg * 1000,
        "per_step_avg_ms": per_step_avg * 1000,
        "overhead_percent": overhead * 100,
    }

# Validation
results = benchmark_latency_overhead()
assert results["overhead_percent"] < 10, f"Overhead {results['overhead_percent']:.1f}% exceeds 10%"
```

**Acceptance Test:**
```
[ ] 10 iterations of full-loop measured
[ ] 10 iterations of per-step measured
[ ] Overhead calculated correctly
[ ] Overhead < 10%
```

**Evidence Required:**
- Benchmark script output
- Timing measurements table
- Hardware/environment details

---

### Criterion 1.5: Robustness (1000 gen, 0 crashes)

**Target:** 1000 consecutive generations with 0 crashes, 0 memory leaks

**Measurement Method:**
```python
# File: /home/tt-admin/tt-metal/comfyui_bridge/tests/stress_test.py

import psutil
import os

def stress_test(generations=1000):
    """Run stress test for robustness."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    crash_count = 0
    memory_readings = []
    
    for i in range(generations):
        try:
            run_per_step_generation(steps=20)
            
            if i % 100 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_readings.append(current_memory)
                print(f"Generation {i}: Memory {current_memory:.1f} MB")
                
        except Exception as e:
            crash_count += 1
            print(f"Crash at generation {i}: {e}")
    
    final_memory = process.memory_info().rss / (1024 * 1024)
    memory_growth = final_memory - initial_memory
    
    return {
        "generations": generations,
        "crashes": crash_count,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_growth_mb": memory_growth,
    }

# Validation
results = stress_test()
assert results["crashes"] == 0, f"{results['crashes']} crashes occurred"
assert results["memory_growth_mb"] < 100, f"Memory grew by {results['memory_growth_mb']:.1f} MB"
```

**Acceptance Test:**
```
[ ] 1000 generations completed
[ ] 0 crashes
[ ] Memory growth < 100 MB (reasonable overhead)
[ ] No hung processes
```

**Evidence Required:**
- Stress test log
- Memory readings over time
- Final results summary

---

### Criterion 1.6: Documentation Complete

**Target:** ADRs + architecture docs + code annotations complete

**Measurement Method:**

**Documentation Checklist:**
```
ADRs:
[ ] ADR-001-per-timestep-api.md exists and approved
[ ] ADR-002-scheduler-sync.md exists and approved
[ ] ADR-003-controlnet-integration.md exists and approved

Architecture Docs:
[ ] bridge_extension.md complete
[ ] Contains: Overview, Components, Data Flow, Performance

API Reference:
[ ] bridge_extension_api.md complete
[ ] All operations documented with params and returns

User Guide:
[ ] controlnet_guide.md complete
[ ] Contains: Usage instructions, examples, troubleshooting

Native Integration Handoff:
[ ] native_integration_handoff.md complete
[ ] Contains: Reusable components, modification needs

Code Annotations:
[ ] All new functions have docstrings
[ ] Reusability comments present on key components
[ ] No undocumented public functions
```

**Acceptance Test:**
```
[ ] 3 ADRs approved
[ ] 4+ documentation files complete
[ ] Code documentation coverage > 90%
```

**Evidence Required:**
- List of documentation files
- ADR approval records
- Code coverage report for docstrings

---

## Dashboard Template

### Phase 1 Progress Dashboard

**Week: ___  |  Date: ___________**

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| Per-step SSIM | >= 0.99 | | [ ] Pass / [ ] Fail / [ ] In Progress |
| ControlNet SSIM | >= 0.90 | | [ ] Pass / [ ] Fail / [ ] In Progress |
| Human Validation | 5/5 | | [ ] Pass / [ ] Fail / [ ] In Progress |
| Latency Overhead | < 10% | | [ ] Pass / [ ] Fail / [ ] In Progress |
| Robustness | 0 crashes | | [ ] Pass / [ ] Fail / [ ] In Progress |
| Memory Leaks | 0 | | [ ] Pass / [ ] Fail / [ ] In Progress |
| Documentation | Complete | | [ ] Pass / [ ] Fail / [ ] In Progress |

**Overall Progress:** ___/7 criteria met

**Blockers:**
-

**Risks:**
-

**Next Actions:**
-

---

## Success Determination

### Phase 0 Decision Matrix

| IPC < 10ms | Scheduler Approved | ControlNet GO | Decision |
|------------|-------------------|---------------|----------|
| Yes | Yes | Yes | GO |
| Yes | Yes | No | CONDITIONAL GO (defer ControlNet) |
| Yes | No | Any | NO-GO (revisit scheduler design) |
| No | Any | Any | NO-GO (investigate IPC) |

### Phase 1 Success Matrix

| Criteria Met | Status | Action |
|--------------|--------|--------|
| 7/7 (100%) | SUCCESS | Proceed to Phase 1.5 |
| 6/7 (86%) | CONDITIONAL SUCCESS | Document gap, plan fix |
| 5/7 (71%) | CONDITIONAL SUCCESS | Evaluate severity, may extend |
| < 5/7 (< 71%) | REVISIT | Root cause analysis, pivot |

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
