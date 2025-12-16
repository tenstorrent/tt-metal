# Phase 1: Bridge Extension - Comprehensive Implementation Prompt

**Date Created:** December 15, 2025
**Version:** 1.0
**Status:** APPROVED FOR IMPLEMENTATION
**Timeline:** 5-6 weeks (Phase 0: 5 days + Phase 1: 4 weeks + Buffer: 3-5 days)

---

## PART A: STRATEGIC CONTEXT

### Why This Path Was Chosen

This implementation follows the strategic analysis documented in:
- `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md` - Clarifies that ControlNet/IP-Adapter CAN work with bridge extension
- `/home/tt-admin/tt-metal/STRATEGIC_PATH_ANALYSIS.md` - Analyzes refactoring risk and reusability across paths
- `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md` - Details native integration pathway

**Key Strategic Decisions:**

1. **Bridge Extension Before Native Integration**
   - Validates per-timestep patterns before 12-17 week investment
   - Delivers ControlNet/IP-Adapter in 5-6 weeks (vs 12-17 weeks for native)
   - Work is 100% reusable for native integration

2. **Per-Timestep Validation Approach**
   - ComfyUI ecosystem requires per-step control (ControlNet, IP-Adapter, custom samplers)
   - Current `denoise_only` runs full loop internally - blocks extension integration
   - New `handle_denoise_step_single` enables external loop control

3. **ControlNet Before Native Integration**
   - ControlNet operates via conditioning injection (works at inference time)
   - Does NOT require weight patching (unlike LoRA)
   - CPU-side execution validated in Phase 0 enables bridge integration

### Reusability Guarantees

**100% of Phase 1 Work Reusable:**

| Component | SD3.5 (Phase 2) | Native Integration (Phase 3) |
|-----------|-----------------|------------------------------|
| Per-step API pattern | Identical usage | Direct port |
| Session management | Concepts reuse | 80% code reuse |
| Format conversions | Already parametrized | 100% code port |
| Scheduler patterns | Same approach | 90% concept reuse |
| ControlNet approach | Same integration | 100% pattern reuse |
| Model-agnostic config | Enables SD3.5 | Enables any model |

**Estimated Savings:**
- SD3.5 support: 1-2 weeks (vs 3-4 weeks from scratch)
- Native integration: 2-4 weeks saved (patterns already validated)

### Success Criteria Summary

Phase 1 is SUCCESSFUL when:
1. Per-step denoising achieves SSIM >= 0.99 vs full-loop baseline
2. ControlNet workflows produce SSIM >= 0.90 vs CPU reference
3. 5/5 human raters confirm "correct" visual output
4. Per-step latency < 10% overhead vs full-loop / steps
5. 1000 consecutive generations with 0 crashes, 0 memory leaks
6. All documentation complete (ADRs, architecture docs, code annotations)

---

## PART B: PHASE 0 PRE-IMPLEMENTATION VALIDATION (Days 1-5)

### Purpose

Phase 0 validates critical assumptions BEFORE committing to 4-week implementation. This is the highest-ROI investment in the entire project, reducing mid-implementation pivot risk from 40% to < 5%.

### Task 0.1: Scheduler State Synchronization Design (Days 1-2)

**Objective:** Design and prototype the stateless bridge approach for scheduler synchronization.

**Background:**
ComfyUI uses various schedulers (Euler, DPM++, etc.) that maintain internal state across timesteps. The bridge must synchronize this state correctly.

**Design Decision: Option A - Stateless Bridge (ComfyUI Owns Scheduler)**

```
ComfyUI Process                    Bridge Server Process
+------------------------+         +------------------------+
| Scheduler State        |         | No scheduler state     |
| - timesteps[]          |   -->   | - Receives timestep    |
| - sigmas[]             |   IPC   | - Receives sigma       |
| - current_step         |         | - Runs UNet forward    |
+------------------------+         +------------------------+
```

**Rationale:**
- ComfyUI already has scheduler implementations
- Bridge becomes simpler (stateless)
- Enables custom scheduler support immediately
- No state synchronization bugs possible

**Deliverables:**
1. Design document: `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md`
2. Prototype implementation in handlers.py
3. Test suite validating scheduler state across 20+ steps
4. Error handling strategy for state mismatches

**Acceptance Criteria:**
- [ ] Design document approved
- [ ] Prototype passes 100% of scheduler sync tests
- [ ] Error handling gracefully recovers from edge cases

### Task 0.2: IPC Performance Baseline (Day 1, Parallel)

**Objective:** Measure current full-loop latency to establish per-step budget.

**Measurements Required:**

```python
# Current full-loop timing (20 steps example)
full_loop_latency = measure_denoise_only(steps=20)  # e.g., 2000ms

# Per-step budget calculation
per_step_budget = full_loop_latency / 20  # e.g., 100ms per step

# IPC overhead budget (10% of per-step)
ipc_budget = per_step_budget * 0.10  # e.g., 10ms per step

# Actual IPC measurement
ipc_latency = measure_roundtrip()  # e.g., 1-5ms
```

**Deliverables:**
1. Baseline measurements document
2. Performance test script
3. Headroom analysis (expect 60% headroom based on 1-5ms IPC)

**Acceptance Criteria:**
- [ ] IPC latency < 10ms (within budget)
- [ ] Full-loop baseline documented
- [ ] Per-step budget calculated

### Task 0.3: ControlNet Architecture Feasibility Study (Days 3-5)

**Objective:** Validate that ControlNet conditioning can be injected via bridge.

**Key Question:** Does ControlNet run on CPU/GPU (ComfyUI side) or must it run on TT hardware?

**Investigation Steps:**

1. **Analyze ComfyUI ControlNet Implementation**
   - Location: ComfyUI's `comfy/controlnet.py`
   - Determine: Where does ControlNet output conditioning?
   - Output: Data flow diagram

2. **Prototype Data Transfer**
   ```python
   # Pseudo-code for conditioning injection
   def handle_denoise_step_single(params):
       latent = params["latent"]
       timestep = params["timestep"]
       conditioning = params["conditioning"]

       # NEW: ControlNet conditioning hint (if provided)
       control_hint = params.get("control_hint")  # From CPU-side ControlNet

       # Inject into UNet forward pass
       output = self.unet.forward(latent, timestep, conditioning, control_hint)
       return output
   ```

3. **Validate TT UNet Compatibility**
   - Check: Does TT UNet accept control_hint parameter?
   - If not: Design injection point
   - Document: Required UNet modifications (if any)

**Go/No-Go Decision Framework:**

| Scenario | Validation Result | Decision |
|----------|-------------------|----------|
| ControlNet CPU-side works | control_hint transfers correctly | GO - Proceed to Phase 1 |
| ControlNet needs TT-side | UNet must integrate ControlNet | DEFER - Move to Phase 2 |
| Neither works cleanly | Architecture incompatibility | PIVOT - Explore alternatives |

**Deliverables:**
1. Feasibility report with data flow diagrams
2. Prototype demonstrating conditioning transfer
3. Go/No-Go recommendation with rationale

**Acceptance Criteria:**
- [ ] ControlNet architecture documented
- [ ] Prototype validates data transfer
- [ ] Clear Go/No-Go decision made

### Phase 0 Output

At the end of Day 5, deliver:

1. **Feasibility Report** (`/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md`)
   - Scheduler sync design summary
   - IPC baseline measurements
   - ControlNet feasibility assessment
   - Risk assessment update

2. **Go/No-Go Recommendation**
   - GO: Proceed to Phase 1 Week 1
   - CONDITIONAL GO: Proceed with documented constraints
   - NO-GO: Pivot strategy (defer ControlNet, focus on per-step API only)

3. **Updated Risk Register**
   - Risks validated or invalidated
   - Updated likelihood and impact scores
   - Mitigation strategies confirmed

---

## PART C: WEEK-BY-WEEK IMPLEMENTATION PLAN

### Week 1: Per-Timestep API Foundation

**Goal:** Implement `handle_denoise_step_single` with model-agnostic design.

#### Task 1.1: `handle_denoise_step_single` Operation Design (2 days)

**File:** `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`

**New Operation Signature:**

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a SINGLE denoising step.

    ComfyUI calls this N times for N-step denoising.
    Enables per-step control for ControlNet, IP-Adapter, custom samplers.

    Args (via params):
        session_id: str - Session identifier for state tracking
        latent_shm: dict - Shared memory handle for input latents [B, C, H, W]
        timestep: float - Current timestep value
        timestep_index: int - Current step index (0 to N-1)
        sigma: float - Current sigma value (for scheduler)
        conditioning_shm: dict - Shared memory handle for CLIP embeddings
        negative_conditioning_shm: dict - Negative prompt embeddings (optional)
        control_hint_shm: dict - ControlNet conditioning (optional)
        guidance_scale: float - CFG scale (default 7.5)

    Returns:
        latent_shm: dict - Shared memory handle for output latents [B, C, H, W]
        step_metadata: dict - Step timing and diagnostics

    Reusability Note for Native Integration:
        This operation implements the per-timestep pattern required by ComfyUI.
        The same pattern will be used in native integration - only IPC layer changes.
    """
```

**Implementation Requirements:**
1. Extract single-step logic from existing `handle_denoise_only`
2. Accept timestep and sigma from ComfyUI (stateless bridge)
3. Return latents in standard format [B, C, H, W]
4. Include timing information for performance analysis

#### Task 1.2: Model-Agnostic Refactoring (1 day)

**Objective:** Enable per-step API to work with SDXL, SD3.5, and future models.

**Current State (Hardcoded):**
```python
# handlers.py line ~587
if C != 4:
    raise ValueError(f"Expected 4 channels for SDXL, got {C}")
```

**Required Change (Config-Based):**
```python
# New config system
MODEL_CONFIGS = {
    "sdxl": {"latent_channels": 4, "clip_dim": 2048},
    "sd35": {"latent_channels": 16, "clip_dim": 4096},
    "sd14": {"latent_channels": 4, "clip_dim": 768},
}

def handle_denoise_step_single(self, params):
    model_type = self.model_type  # Set during init_model
    config = MODEL_CONFIGS[model_type]
    expected_channels = config["latent_channels"]

    if C != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels for {model_type}, got {C}")
```

**Files to Modify:**
- `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` - Add config lookup
- Create `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py` - Centralized configs

#### Task 1.3: Scheduler State Synchronization Implementation (1 day)

**Based on Phase 0 Design:** Implement stateless bridge pattern.

**ComfyUI Side (Owns Scheduler):**
```python
# In TT_KSampler node (new node to create in Week 2)
for step in range(num_steps):
    timestep = scheduler.timesteps[step]
    sigma = scheduler.sigmas[step]

    # Send to bridge with scheduler state
    response = backend.denoise_step_single(
        latents=latents,
        timestep=timestep,
        timestep_index=step,
        sigma=sigma,
        conditioning=cond,
    )

    # Get result for next step
    latents = response["latents"]
```

**Bridge Side (Receives Scheduler State):**
```python
def handle_denoise_step_single(self, params):
    # Receive scheduler state (don't maintain internally)
    timestep = params["timestep"]
    timestep_index = params["timestep_index"]
    sigma = params["sigma"]

    # Pass to UNet (no scheduler state management in bridge)
    output = self.runner.denoise_step(
        latents=tt_latents,
        timestep=timestep,
        timestep_index=timestep_index,
        ...
    )
    return output
```

#### Task 1.4: Format Conversion Audit and Update (0.5 days)

**Objective:** Ensure format conversion works for all model types.

**Current Helper (handlers.py:32-93):**
```python
def _detect_and_convert_tt_to_standard_format(
    tensor: torch.Tensor,
    expected_channels: int = 4  # Hardcoded default
) -> torch.Tensor:
```

**Required Update:**
```python
def _detect_and_convert_tt_to_standard_format(
    tensor: torch.Tensor,
    expected_channels: int,  # Required parameter (no default)
    model_type: str = "sdxl"  # For logging
) -> torch.Tensor:
    """
    Reusability Note for Native Integration:
        This helper is 100% reusable. Port directly to native code.
    """
```

**Also Update Call Sites:**
- `handle_denoise_only` - Pass model config channels
- `handle_denoise_step_single` - Pass model config channels
- `handle_vae_decode` - Validate channels from config

#### Task 1.5: Session State Management Framework (0.5 days)

**Objective:** Design session tracking for multi-step workflows.

**Session Structure:**
```python
@dataclass
class DenoiseSession:
    session_id: str
    model_id: str
    created_at: float
    last_activity: float
    current_step: int
    total_steps: int

    # Cached tensors (optional, for optimization)
    cached_conditioning: Optional[Any] = None

class SessionManager:
    def __init__(self, timeout_seconds: int = 1800):  # 30 min default
        self.sessions: Dict[str, DenoiseSession] = {}
        self.timeout = timeout_seconds

    def create_session(self, model_id: str, total_steps: int) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = DenoiseSession(
            session_id=session_id,
            model_id=model_id,
            created_at=time.time(),
            last_activity=time.time(),
            current_step=0,
            total_steps=total_steps,
        )
        return session_id

    def cleanup_expired(self):
        """Remove sessions older than timeout."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_activity > self.timeout
        ]
        for sid in expired:
            del self.sessions[sid]
```

#### Week 1 Testing

**Unit Tests:**
- `test_denoise_step_single_basic` - Single step produces valid output
- `test_denoise_step_single_format` - Output format is standard [B, C, H, W]
- `test_denoise_step_single_scheduler_state` - Timestep/sigma passed correctly
- `test_model_agnostic_config` - Config lookup works for all model types

**Integration Test:**
- `test_per_step_matches_full_loop` - 20 individual steps match single `denoise_only` call

**Definition of Done:**
- [ ] `handle_denoise_step_single` implemented and tested
- [ ] Model-agnostic config system working
- [ ] SSIM >= 0.99 vs full-loop baseline
- [ ] All unit tests passing

---

### Week 2: Session Management and Robustness

**Goal:** Production-ready session lifecycle with error handling.

#### Task 2.1: Session Lifecycle Management (2 days)

**Operations to Implement:**

```python
def handle_session_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new denoising session.

    Returns:
        session_id: str - Unique session identifier
        session_info: dict - Session metadata
    """

def handle_session_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a denoising step within a session.
    Alias for handle_denoise_step_single with session tracking.
    """

def handle_session_complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark session complete and cleanup resources.

    Returns:
        final_latents_shm: dict - Final latents if requested
        session_stats: dict - Session statistics (timing, steps, etc.)
    """
```

**Session State Machine:**
```
            [create]
               |
               v
    +----------+----------+
    |       CREATED       |
    +----------+----------+
               |
          [step 0]
               |
               v
    +----------+----------+
    |      IN_PROGRESS    |<----+
    +----------+----------+     |
               |                |
          [step N]        [step 1..N-1]
               |                |
               +----------------+
               |
          [complete]
               |
               v
    +----------+----------+
    |      COMPLETED      |
    +----------+----------+
               |
          [cleanup]
               |
               v
           [removed]
```

#### Task 2.2: Session Timeout and Cleanup (1 day)

**Implementation:**

```python
class SessionManager:
    def __init__(self, timeout_seconds: int = 1800):
        self.sessions = {}
        self.timeout = timeout_seconds
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Background thread for session cleanup."""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                self.cleanup_expired()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def cleanup_expired(self):
        now = time.time()
        expired = [
            (sid, session) for sid, session in self.sessions.items()
            if now - session.last_activity > self.timeout
        ]

        for sid, session in expired:
            logger.warning(
                f"Session {sid} expired after {self.timeout}s of inactivity. "
                f"Steps completed: {session.current_step}/{session.total_steps}"
            )
            # Release any cached resources
            self._cleanup_session_resources(session)
            del self.sessions[sid]
```

#### Task 2.3: Error Handling and Recovery (1 day)

**Error Categories:**

| Error Type | Example | Recovery Strategy |
|------------|---------|-------------------|
| Session Not Found | Invalid session_id | Return error, suggest create_session |
| Model Mismatch | session.model_id != current model | Return error, suggest new session |
| Step Out of Order | step_index != expected | Warning + proceed (flexible) |
| Format Error | Wrong tensor shape | Detailed error message |
| Device Error | TT hardware failure | Attempt recovery, fail gracefully |

**Error Response Format:**
```python
def handle_error(self, error: Exception, context: Dict) -> Dict[str, Any]:
    return {
        "success": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "recoverable": self._is_recoverable(error),
            "suggestion": self._get_recovery_suggestion(error),
        }
    }
```

#### Task 2.4: ComfyUI Node Infrastructure (1 day)

**File:** `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`

**New Node: TT_KSampler (Placeholder)**

```python
class TT_KSampler:
    """
    Per-timestep sampler with ControlNet support.

    This node controls the denoising loop, calling bridge per-step.
    Enables ControlNet, IP-Adapter, and custom sampler integration.

    Reusability Note for Native Integration:
        The per-step pattern is identical for native integration.
        Only the IPC layer changes (direct calls vs socket).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "scheduler": (["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral"], {"default": "euler"}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "control_hint": ("IMAGE",),  # ControlNet input
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Tenstorrent/sampling"
```

#### Task 2.5: IPC Performance Testing (0.5 days)

**Performance Test Suite:**

```python
def test_per_step_performance():
    """Validate per-step overhead is within budget."""

    # Measure full-loop baseline
    full_loop_time = measure_denoise_only(steps=20)
    per_step_baseline = full_loop_time / 20

    # Measure per-step API
    per_step_times = []
    for step in range(20):
        step_time = measure_denoise_step_single(step)
        per_step_times.append(step_time)

    per_step_avg = sum(per_step_times) / len(per_step_times)
    overhead = (per_step_avg - per_step_baseline) / per_step_baseline

    # Assert overhead < 10%
    assert overhead < 0.10, f"Per-step overhead {overhead:.1%} exceeds 10% budget"
```

#### Week 2 Testing

**Integration Tests:**
- `test_session_lifecycle` - Create, step x20, complete
- `test_session_timeout` - Session expires after inactivity
- `test_session_cleanup` - Resources released properly
- `test_concurrent_sessions` - Multiple sessions work correctly
- `test_error_recovery` - Graceful failure and recovery

**Performance Tests:**
- `test_per_step_latency` - Overhead < 10%
- `test_memory_stability` - 100 generations, no leaks

**Definition of Done:**
- [ ] Session lifecycle complete (create, step, complete)
- [ ] Timeout mechanism working (30-min default)
- [ ] Error handling graceful
- [ ] Performance within budget (< 10% overhead)
- [ ] 100 consecutive generations stable

---

### Week 3: ControlNet Implementation

**Goal:** Enable ControlNet workflows through bridge extension.

**Prerequisite:** Phase 0 Go decision on ControlNet feasibility.

#### Task 3.1: ControlNet Conditioning Injection (2 days)

**Based on Phase 0 Validation:**

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute single denoising step with optional ControlNet."""

    # ... existing parameter extraction ...

    # NEW: ControlNet conditioning (validated in Phase 0)
    control_hint = None
    if "control_hint_shm" in params:
        control_hint_torch = self._get_tensor_from_shm(params["control_hint_shm"])

        # Convert to TT format if needed
        control_hint = self._prepare_control_hint(control_hint_torch)

        logger.info(
            f"ControlNet conditioning applied: "
            f"shape={control_hint_torch.shape}, step={timestep_index}"
        )

    # UNet forward pass with ControlNet
    output = self.runner.denoise_step(
        latents=tt_latents,
        timestep=timestep,
        conditioning=conditioning,
        control_hint=control_hint,  # NEW: ControlNet injection
    )
```

**ControlNet Data Flow:**
```
ComfyUI Process                    Bridge Server Process
+------------------------+         +------------------------+
| 1. Load ControlNet     |         |                        |
| 2. Preprocess image    |         |                        |
| 3. Run ControlNet      |         |                        |
|    (CPU/GPU side)      |         |                        |
| 4. Get control_hint    |         |                        |
|         |              |         |                        |
|         v              |         |                        |
| 5. Send via IPC -------|-------->| 6. Receive control_hint|
|                        |         | 7. Inject into UNet    |
|                        |<--------|----8. Return latents   |
| 9. Continue loop       |         |                        |
+------------------------+         +------------------------+
```

#### Task 3.2: ComfyUI TT_ControlNet Wrapper Node (1.5 days)

**File:** `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`

```python
class TT_ApplyControlNet:
    """
    Apply ControlNet conditioning to TT sampling.

    This node wraps ComfyUI's ControlNet output and passes it to the
    TT_KSampler for injection during denoising.

    Note: ControlNet model runs on CPU/GPU (ComfyUI side).
    Only the conditioning hint is passed to TT hardware.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),  # Standard ComfyUI ControlNet
                "image": ("IMAGE",),  # Preprocessed control image (e.g., Canny edges)
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Tenstorrent/conditioning"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        """
        Run ControlNet on ComfyUI side and prepare conditioning for TT.
        """
        # Run ControlNet (CPU/GPU)
        control_hint = control_net.get_control_hint(image, strength)

        # Attach to conditioning for TT_KSampler
        modified_cond = [(
            cond[0],
            {
                **cond[1],
                "tt_control_hint": control_hint,  # Passed to bridge
                "tt_control_strength": strength,
            }
        ) for cond in conditioning]

        return (modified_cond,)
```

#### Task 3.3: Per-Step ControlNet Integration Testing (1 day)

**Test Workflow:**
```
LoadImage -> CannyEdgeDetector -> TT_ApplyControlNet
                                        |
                                        v
TT_CheckpointLoader -> TT_KSampler (with control) -> TT_VAEDecode -> SaveImage
```

**Test Cases:**
1. Canny edge ControlNet - Line art following
2. Depth ControlNet - Depth-aware generation
3. OpenPose ControlNet - Pose-guided generation

**Validation:**
- SSIM >= 0.90 vs CPU reference (ControlNet output on GPU)
- Human validation: Control image influence visible

#### Task 3.4: Multi-ControlNet Support (0.5 days, if time permits)

**Extension for Multiple ControlNets:**

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    # Support multiple control hints
    control_hints = []

    for i in range(10):  # Support up to 10 ControlNets
        key = f"control_hint_{i}_shm"
        if key in params:
            hint = self._get_tensor_from_shm(params[key])
            strength = params.get(f"control_strength_{i}", 1.0)
            control_hints.append((hint, strength))

    # Aggregate control hints (if any)
    if control_hints:
        combined_hint = self._combine_control_hints(control_hints)
    else:
        combined_hint = None
```

#### Week 3 Testing

**End-to-End Tests:**
- `test_controlnet_canny` - Canny edge following
- `test_controlnet_depth` - Depth-aware generation
- `test_controlnet_openpose` - Pose-guided generation
- `test_multi_controlnet` - Two ControlNets combined (if implemented)

**Quality Validation:**
- SSIM >= 0.90 vs CPU reference
- Human validation: 5/5 raters confirm "correct"

**Definition of Done:**
- [ ] ControlNet conditioning injection working
- [ ] 3 ControlNet types tested (Canny, Depth, OpenPose)
- [ ] SSIM >= 0.90 vs CPU reference
- [ ] Human validation passed (5/5)

---

### Week 4: Validation, Performance, and Documentation (Part 1)

**Goal:** Comprehensive testing and ADR documentation.

#### Task 4.1: Comprehensive Test Suite Execution (2 days)

**Test Categories:**

1. **Unit Tests** (30+ tests)
   - Format conversion correctness
   - Session lifecycle states
   - Error handling paths
   - Config lookup

2. **Integration Tests** (20+ tests)
   - Per-step vs full-loop equivalence
   - ControlNet workflows
   - Session timeout behavior
   - Multi-model support

3. **Performance Tests** (10+ tests)
   - Per-step latency overhead
   - Memory usage over time
   - IPC throughput
   - Concurrent session handling

4. **Regression Tests** (10+ tests)
   - Existing txt2img still works
   - Existing img2img still works
   - Quality metrics maintained

**Test Execution:**
```bash
# Run all tests
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/ -v --tb=short

# Performance benchmarks
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py

# Quality validation
python /home/tt-admin/tt-metal/comfyui_bridge/tests/validate_quality.py
```

#### Task 4.2: Performance Optimization (1 day)

**Profiling:**
```python
# Profile per-step execution
import cProfile
cProfile.run('run_20_steps()', 'step_profile.stats')

# Analyze hotspots
import pstats
p = pstats.Stats('step_profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

**Optimization Targets (if needed):**
1. Tensor transfer batching
2. Conditioning caching within session
3. Format conversion optimization
4. IPC message compression

**Validation:**
- Per-step latency < budget (10% overhead)
- Memory stable over 1000 generations

#### Task 4.3: Documentation (Part 1) - ADRs (2 days)

**ADR-001: Per-Timestep API Design**

File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-001-per-timestep-api.md`

```markdown
# ADR-001: Per-Timestep API Design

## Status
ACCEPTED

## Context
ComfyUI ecosystem features (ControlNet, IP-Adapter, custom samplers) require
per-timestep control over the denoising loop. The existing denoise_only
operation runs the full loop internally, blocking these integrations.

## Decision
Implement handle_denoise_step_single operation that:
1. Executes a single denoising step
2. Accepts scheduler state from ComfyUI (stateless bridge)
3. Returns latents in standard format for next step
4. Supports optional ControlNet conditioning injection

## Consequences
### Positive
- Enables ControlNet, IP-Adapter, custom samplers
- Pattern is 100% reusable for native integration
- Validates per-step approach before larger investment

### Negative
- ~10% IPC overhead vs full-loop
- More complex ComfyUI node implementation
- Session management required for state tracking

## Alternatives Considered
1. **Full loop with callbacks**: Rejected - IPC overhead per callback too high
2. **Native integration first**: Rejected - 12-17 weeks vs 4-6 weeks
3. **Batch steps**: Deferred - Can optimize later if needed
```

**ADR-002: Scheduler State Synchronization**

File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-002-scheduler-sync.md`

**ADR-003: ControlNet Integration**

File: `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-003-controlnet-integration.md`

#### Week 4 Deliverables

- [ ] All test suites passing (95%+ pass rate)
- [ ] Performance targets met (< 10% overhead)
- [ ] 3 ADRs written and approved
- [ ] Code annotated with reusability comments

---

### Week 5: Final Validation and Documentation (Part 2)

**Goal:** Release preparation and documentation completion.

#### Task 5.1: Final End-to-End Validation (1.5 days)

**Validation Checklist:**

```
Per-Step API Validation:
[ ] SSIM >= 0.99 vs full-loop (100 test prompts)
[ ] 20 diverse prompts tested
[ ] 3 different step counts (10, 20, 50)
[ ] Multiple seeds validated

ControlNet Validation:
[ ] Canny: 10 test images, SSIM >= 0.90
[ ] Depth: 10 test images, SSIM >= 0.90
[ ] OpenPose: 10 test images, SSIM >= 0.90
[ ] Human validation: 5/5 raters confirm correct

Performance Validation:
[ ] 1000 consecutive generations
[ ] 0 crashes
[ ] 0 memory leaks
[ ] Latency within budget

Regression Validation:
[ ] txt2img unchanged
[ ] img2img unchanged
[ ] VAE decode unchanged
[ ] All existing tests pass
```

#### Task 5.2: Documentation (Part 2) (2 days)

**Architecture Documentation:**

File: `/home/tt-admin/tt-metal/docs/architecture/bridge_extension.md`

```markdown
# Bridge Extension Architecture

## Overview
The bridge extension adds per-timestep denoising capability to the ComfyUI
bridge, enabling ControlNet, IP-Adapter, and custom sampler integration.

## Architecture Diagram
[Include diagram]

## Components
### Per-Step API
### Session Management
### ControlNet Integration
### Format Conversion

## Data Flow
[Include sequence diagram]

## Performance Characteristics
[Include benchmark results]

## Reusability for Native Integration
[Document what ports directly]
```

**API Reference:**

File: `/home/tt-admin/tt-metal/docs/api/bridge_extension_api.md`

**User Guide:**

File: `/home/tt-admin/tt-metal/docs/guides/controlnet_guide.md`

**Native Integration Handoff:**

File: `/home/tt-admin/tt-metal/docs/architecture/native_integration_handoff.md`

#### Task 5.3: Release Preparation (1.5 days)

**Release Checklist:**

```
Code Quality:
[ ] Code review completed
[ ] All tests passing
[ ] Performance benchmarks documented
[ ] No critical TODOs remaining

Documentation:
[ ] ADRs complete (3)
[ ] Architecture docs complete
[ ] API reference complete
[ ] User guide complete

Release Artifacts:
[ ] Version bumped
[ ] Changelog updated
[ ] Release notes written
[ ] Phase 1.5 (IP-Adapter) plan drafted
```

**Phase 1.5 Planning Document:**

File: `/home/tt-admin/tt-metal/docs/roadmap/PHASE_1_5_IP_ADAPTER.md`

```markdown
# Phase 1.5: IP-Adapter Integration

## Overview
IP-Adapter enables image-prompted generation using the per-step API
established in Phase 1.

## Estimated Timeline
2 weeks (leveraging Phase 1 infrastructure)

## Dependencies
- Phase 1 per-step API (complete)
- Phase 1 session management (complete)
- Phase 1 format conversions (complete)

## Implementation Approach
[Details based on Phase 1 learnings]
```

#### Week 5 Deliverables

- [ ] Final validation complete (all criteria met)
- [ ] Documentation complete (5+ documents)
- [ ] Release notes written
- [ ] Phase 1.5 plan drafted
- [ ] Code review approved

---

## PART D: CRITICAL DESIGN DECISIONS

### Decision 1: Scheduler State - Stateless Bridge

**Decision:** ComfyUI owns scheduler, bridge receives timestep/sigma per call.

**Rationale:**
- ComfyUI has scheduler implementations (Euler, DPM++, etc.)
- Enables any scheduler without bridge changes
- Simpler bridge code (no state management)
- No synchronization bugs possible

**Implementation:**
```python
# ComfyUI sends per-step
params = {
    "timestep": scheduler.timesteps[step],
    "sigma": scheduler.sigmas[step],
    "timestep_index": step,
}
```

### Decision 2: Session Management - Dict-Based with Timeout

**Decision:** Simple dict-based session tracking with background cleanup.

**Rationale:**
- Patterns well-established
- No external dependencies (Redis, etc.)
- 30-minute timeout handles abandoned sessions
- Scales to expected load (< 10 concurrent sessions)

### Decision 3: ControlNet Integration - CPU-Side Conditioning

**Decision:** ControlNet runs on ComfyUI side, conditioning passed to bridge.

**Rationale:**
- ControlNet is CPU/GPU operation (preprocessing + conditioning)
- Only control_hint tensor needs IPC transfer
- No TT-side ControlNet implementation needed
- Uses standard ComfyUI ControlNet nodes

**Validated in:** Phase 0 Task 0.3

### Decision 4: Model-Agnosticism - Config-Based Channel Lookups

**Decision:** Centralized model config with runtime lookup.

**Rationale:**
- Enables SD3.5 without code changes
- Future models add config entry only
- Single source of truth for model parameters

```python
MODEL_CONFIGS = {
    "sdxl": {"latent_channels": 4, "clip_dim": 2048, ...},
    "sd35": {"latent_channels": 16, "clip_dim": 4096, ...},
}
```

### Decision 5: Error Handling - Graceful Failures with Logging

**Decision:** All errors logged, non-critical errors allow recovery.

**Rationale:**
- Production stability
- Debugging visibility
- User-friendly error messages

### Decision 6: Documentation - ADRs + Code Comments + Architecture Docs

**Decision:** Three-tier documentation strategy.

**Rationale:**
- ADRs: Capture WHY decisions were made
- Code comments: Enable code reuse
- Architecture docs: Onboard new developers

---

## PART E: SUCCESS CRITERIA (PRECISE)

### Phase 0 Go/No-Go Gate

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Scheduler sync design | Approved | Design review sign-off |
| ControlNet feasible | GO or defined pivot | Feasibility report |
| IPC baseline | < 10ms | Measured latency |

**Decision:** Proceed to Phase 1 only if ALL criteria met.

### Phase 1 Completion Gate

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Per-step SSIM | >= 0.99 vs full-loop | 100 test prompts |
| ControlNet SSIM | >= 0.90 vs CPU ref | 30 test images (3 types) |
| Human validation | 5/5 correct | 5 raters, visual inspection |
| Per-step latency | < 10% overhead | Benchmark vs full-loop |
| Robustness | 1000 gen, 0 crashes | Stress test |
| Memory stability | 0 leaks | Memory profiling |
| Regression | 100% existing tests | Test suite |
| Compatibility | API backward compat | API validation |
| Documentation | ADRs + docs complete | Review checklist |
| Reusability | < 2 hours design sketch | Senior engineer exercise |

**Quantitative Threshold:**
- 95%+ criteria met = PHASE 1 SUCCESS
- 80-94% = CONDITIONAL SUCCESS (defer remainder)
- < 80% = REVISIT (pivot or extend)

---

## PART F: RISK MITIGATION

### Critical Assumptions Requiring Validation

| Assumption | Validation | Fallback |
|------------|------------|----------|
| ControlNet CPU-side works | Phase 0 Day 3-5 | Defer to Phase 2 |
| Scheduler sync works | Phase 0 Day 1-2 | Option B (stateful) |
| IPC overhead < 10% | Phase 0 Day 1 | Optimize or adjust targets |
| ComfyUI schedulers compatible | Week 1 | Document incompatible |
| Session management scales | Week 2 | Single-model sessions |

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scheduler desync | 30% | Critical | Extensive testing |
| ControlNet TT-side needed | 25% | High | Phase 0 validation |
| IPC overhead > 15% | 40% | Medium | Early optimization |
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
| **Total** | **8 days** | **1.6 weeks** |

### Timeline Summary

- **Optimistic:** 4 weeks (no issues)
- **Realistic:** 5-6 weeks (typical issues)
- **Conservative:** 7 weeks (contingency)

---

## PART G: REUSABILITY GUARANTEES

### How Phase 1 Enables SD3.5 (1-2 Weeks Add-on)

**Work Already Done:**
- Per-step API (model-agnostic)
- Config-based channels (SD3.5 entry exists)
- Format conversion (parametrized)

**Work Remaining for SD3.5:**
- SD3.5 runner implementation
- SD3.5 specific format validation
- Testing

**Estimated Time:** 1-2 weeks (vs 3-4 weeks from scratch)

### How Phase 1 Informs Native Integration (2-4 Weeks Savings)

**Patterns Validated:**
- Per-timestep calling (required for ComfyUI)
- Scheduler state management
- ControlNet integration approach
- Format conversions

**Code Portable:**
- Format conversion helpers (100%)
- Model config system (100%)
- Session patterns (80% concepts)

**Estimated Savings:** 2-4 weeks of discovery work

### Documentation for Reusability

Each code section includes:
```python
"""
Reusability Note for Native Integration:
    This component is [X]% reusable for native integration.
    - What to port: [specific elements]
    - What changes: [IPC layer specifics]
    - Rationale: [why this design]
"""
```

---

## FILE REFERENCES

### Key Reference Documents

- `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md`
- `/home/tt-admin/tt-metal/STRATEGIC_PATH_ANALYSIS.md`
- `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md`

### Code Locations

- `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` (modify)
- `/home/tt-admin/tt-metal/comfyui_bridge/server.py` (modify)
- `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` (modify)

### New Files to Create

- `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py`
- `/home/tt-admin/tt-metal/comfyui_bridge/session_manager.py`
- `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_per_step.py`
- `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_controlnet.py`
- `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-001-per-timestep-api.md`
- `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-002-scheduler-sync.md`
- `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-003-controlnet-integration.md`

---

## EXECUTION INSTRUCTIONS

### For Senior Engineer

1. **Read this document completely** before starting
2. **Complete Phase 0** (5 days) before committing to Phase 1
3. **Make Go/No-Go decision** based on Phase 0 findings
4. **Follow week-by-week plan** with daily progress tracking
5. **Document decisions** in ADRs as you make them
6. **Add reusability comments** to all new code

### Daily Standup Format

```
Yesterday: [completed tasks]
Today: [planned tasks]
Blockers: [any blockers]
Risks: [any new risks identified]
```

### Weekly Review Checklist

- [ ] Week tasks complete
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Risks updated
- [ ] Next week prepared

---

**END OF COMPREHENSIVE PROMPT**

This document is the complete specification for Phase 1: Bridge Extension.
A senior engineer should be able to execute this plan without asking clarifying questions.
