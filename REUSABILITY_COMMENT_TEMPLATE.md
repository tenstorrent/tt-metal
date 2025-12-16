# Reusability Comment Template

**Purpose:** Standard format for documenting reusability in code  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part G

---

## Standard Format

### Function/Method Reusability Note

```python
def function_name(self, params):
    """
    Brief description of what this function does.
    
    Args:
        params: Description of parameters
    
    Returns:
        Description of return value
    
    Reusability Note for Native Integration:
        This component is [X]% reusable for native integration.
        
        Portable (direct port):
            - [specific element 1]
            - [specific element 2]
        
        Requires modification:
            - [element needing change]: [why/how]
        
        Rationale:
            [Why this design enables reusability]
    """
```

### Class Reusability Note

```python
class ClassName:
    """
    Brief description of what this class does.
    
    Reusability Note for Native Integration:
        Reusability: [X]% (code) / [Y]% (concepts)
        
        Portable classes/methods:
            - [method1]: Direct port
            - [method2]: Direct port
        
        Concepts to reuse:
            - [concept1]: [how it applies]
            - [concept2]: [how it applies]
        
        IPC-specific (will change):
            - [method3]: Replace socket with direct call
            - [method4]: Remove serialization
        
        Design decisions preserved:
            - [decision1]
            - [decision2]
    """
```

---

## Examples by Component Type

### Example 1: Per-Step API Handler

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a SINGLE denoising step.
    
    ComfyUI calls this N times for N-step denoising.
    Enables per-step control for ControlNet, IP-Adapter, custom samplers.
    
    Args:
        params: Dictionary containing:
            - session_id: str - Session identifier
            - latent_shm: dict - Shared memory handle for input latents
            - timestep: float - Current timestep value
            - timestep_index: int - Current step index (0 to N-1)
            - sigma: float - Current sigma value
            - conditioning_shm: dict - Shared memory handle for embeddings
            - control_hint_shm: dict - ControlNet conditioning (optional)
            - guidance_scale: float - CFG scale
    
    Returns:
        Dictionary containing:
            - latent_shm: dict - Output latents handle
            - step_metadata: dict - Timing and diagnostics
    
    Reusability Note for Native Integration:
        This operation is 100% reusable as a pattern, 80% as code.
        
        Portable (direct port):
            - Parameter validation logic
            - Format conversion calls
            - Session tracking logic
            - Error response format
            - Control hint injection pattern
        
        Requires modification:
            - _get_tensor_from_shm: Replace with direct tensor access
            - _put_tensor_to_shm: Replace with direct tensor return
            - IPC response format: Simplify to direct returns
        
        Rationale:
            Per-timestep pattern is required for ComfyUI ecosystem features.
            Native integration will use identical calling pattern, only the
            transport layer changes (socket -> direct Python call).
    """
```

### Example 2: Model Configuration

```python
MODEL_CONFIGS = {
    "sdxl": {
        "latent_channels": 4,
        "clip_dim": 2048,
        "vae_scale_factor": 8,
        "default_resolution": (1024, 1024),
    },
    "sd35": {
        "latent_channels": 16,
        "clip_dim": 4096,
        "vae_scale_factor": 8,
        "default_resolution": (1024, 1024),
    },
}
"""
Model configuration dictionary.

Reusability Note for Native Integration:
    This configuration is 100% portable.
    
    Portable:
        - Entire MODEL_CONFIGS dictionary
        - All model type definitions
        - All parameter values
    
    Requires modification:
        - None
    
    To add new models:
        1. Add entry to MODEL_CONFIGS
        2. No code changes required
    
    Rationale:
        Centralizing model configuration enables model-agnostic code
        and simplifies adding new model support. This pattern should
        be preserved exactly in native integration.
"""
```

### Example 3: Session Manager

```python
class SessionManager:
    """
    Manages denoising sessions for multi-step workflows.
    
    Provides session lifecycle management (create, update, complete),
    timeout handling, and resource cleanup.
    
    Reusability Note for Native Integration:
        Reusability: 80% (code) / 100% (concepts)
        
        Portable classes/methods:
            - DenoiseSession dataclass: 100% portable
            - create_session: Direct port
            - get_session: Direct port
            - update_session: Direct port
            - cleanup_expired: Direct port
        
        Concepts to reuse:
            - Session state tracking
            - Timeout mechanism
            - Background cleanup thread
            - Session statistics collection
        
        IPC-specific (will change):
            - Session ID format may simplify
            - No SHM handle tracking needed
        
        Design decisions preserved:
            - Dict-based storage (no external DB)
            - 30-minute default timeout
            - Background cleanup thread
            - Graceful expiry with logging
    """
```

### Example 4: Format Conversion Helper

```python
def _detect_and_convert_tt_to_standard_format(
    tensor: torch.Tensor,
    expected_channels: int,
    model_type: str = "sdxl"
) -> torch.Tensor:
    """
    Convert TT tensor format to standard ComfyUI format.
    
    TT hardware may return tensors in various formats depending on
    optimization. This helper normalizes to standard [B, C, H, W].
    
    Args:
        tensor: Input tensor from TT hardware
        expected_channels: Expected channel count from model config
        model_type: Model type for logging
    
    Returns:
        Tensor in standard [B, C, H, W] format
    
    Reusability Note for Native Integration:
        This helper is 100% reusable - port directly to native code.
        
        Portable:
            - Entire function implementation
            - Shape detection logic
            - Format conversion logic
            - Error handling
        
        Requires modification:
            - None
        
        Why it's needed in native:
            TT hardware output format is independent of bridge vs native.
            Same conversion will be needed in both contexts.
    """
```

### Example 5: ComfyUI Node

```python
class TT_KSampler:
    """
    Per-timestep sampler with ControlNet support for TT hardware.
    
    This node controls the denoising loop, calling the TT backend
    per-step. Enables ControlNet, IP-Adapter, and custom sampler
    integration.
    
    Reusability Note for Native Integration:
        The per-step calling pattern is identical for native integration.
        Only the backend call mechanism changes.
        
        Portable:
            - INPUT_TYPES definition
            - Loop structure in sample()
            - Scheduler integration
            - ControlNet handling pattern
        
        Requires modification:
            - Backend call: socket -> direct function call
            - Session management: may be simplified
        
        Rationale:
            ComfyUI nodes define the user-facing interface. The same
            node structure works for both bridge and native backends,
            only the internal implementation differs.
    """
```

---

## Model-Agnostic vs Model-Specific Marking

### Model-Agnostic Code (Preferred)

```python
def validate_latent_channels(tensor, model_type):
    """
    Validate latent channels match model configuration.
    
    Model-Agnostic: Yes
        Works with any model type defined in MODEL_CONFIGS.
        No hardcoded values. Add new models via config only.
    """
    config = MODEL_CONFIGS[model_type]
    expected = config["latent_channels"]
    actual = tensor.shape[1]
    
    if actual != expected:
        raise ValueError(
            f"Expected {expected} channels for {model_type}, got {actual}"
        )
```

### Model-Specific Code (Document Limitation)

```python
def apply_sdxl_specific_processing(tensor):
    """
    Apply SDXL-specific post-processing.
    
    Model-Specific: Yes (SDXL only)
        This function is specific to SDXL architecture.
        For other models, create equivalent functions:
        - apply_sd35_specific_processing for SD3.5
        - apply_sd14_specific_processing for SD1.4
        
    Future work:
        Consider refactoring to model-agnostic if pattern emerges.
    """
    # SDXL-specific logic here
    pass
```

---

## What to Document in ADRs

### ADR References in Code

```python
def handle_denoise_step_single(self, params):
    """
    Per-step denoising operation.
    
    Design Decision: Per-timestep API pattern
        See: ADR-001 (docs/architecture/adr/ADR-001-per-timestep-api.md)
        
    Design Decision: Stateless bridge (scheduler ownership)
        See: ADR-002 (docs/architecture/adr/ADR-002-scheduler-sync.md)
    """
```

### When Code Should Reference ADR

- [ ] Major architectural decisions
- [ ] Choice between alternatives
- [ ] Trade-offs accepted
- [ ] Design patterns selected

### ADR Template for Code-Related Decisions

```markdown
# ADR-00X: [Decision Title]

## Status
[PROPOSED | ACCEPTED | DEPRECATED | SUPERSEDED]

## Context
[What problem we're solving]

## Decision
[What we decided]

## Code Locations
- Primary implementation: [file:line]
- Tests: [file]
- Related code: [files]

## Consequences
### Positive
### Negative

## Alternatives Considered
[Why other options were rejected]
```

---

## Handoff Documentation Structure

### For Native Integration Handoff

Create: `/home/tt-admin/tt-metal/docs/architecture/native_integration_handoff.md`

```markdown
# Native Integration Handoff

## Components Ready for Port

### 1. Per-Step API (100% pattern reuse)
- Source: handlers.py::handle_denoise_step_single
- Pattern: Identical calling convention
- Changes: Replace IPC with direct calls

### 2. Model Configuration (100% code reuse)
- Source: model_config.py::MODEL_CONFIGS
- Changes: None required

### 3. Session Management (80% code reuse)
- Source: session_manager.py
- Portable: DenoiseSession, create/get/update/cleanup
- Changes: Simplify ID generation, remove SHM handles

### 4. Format Conversion (100% code reuse)
- Source: handlers.py::_detect_and_convert_*
- Changes: None required

## Estimated Effort Savings
- Pattern validation: Already done
- Format conversion: Ready to use
- Session concepts: Validated
- Total savings: 2-4 weeks
```

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part G
