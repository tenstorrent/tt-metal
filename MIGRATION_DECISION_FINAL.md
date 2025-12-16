# Final Migration Decision: ComfyUI-tt → ComfyUI-tt_standalone

**Date**: 2025-12-12
**Decision**: **START FRESH** with Full Inference Bridge architecture
**Phase 0 Fixes Needed**: **NONE**

---

## Executive Summary

After comprehensive investigation using multiple agents (Explore, Problem-Investigator, Local-File-Searcher), the decision is clear:

### ✅ **ABANDON ComfyUI-tt - START FRESH IN ComfyUI-tt_standalone**

**Rationale**:
1. ComfyUI-tt_standalone is a **clean** ComfyUI v0.3.68 installation (zero TT code)
2. **Full Inference Bridge** architecture eliminates need for Phase 0 fixes
3. Proven pattern (TT_FullDenoise) achieves SSIM 0.998+
4. Simpler, cleaner, more maintainable than per-step integration

**Work to Migrate**: Specific code patterns only (not wholesale copy)

---

## Investigation Findings Summary

### Finding 1: ComfyUI-tt_standalone is Clean
**Agent**: Explore
**Result**: Zero Tenstorrent modifications, perfect base for new integration

### Finding 2: Phase 0 Fixes Not Needed
**Agent**: Problem-Investigator
**Result**:
- **Timestep fix**: NOT NEEDED (bridge handles internally)
- **CFG fix**: NOT NEEDED (separate parameters, not batched)

### Finding 3: TT_FullDenoise Pattern is Proven
**Agent**: Local-File-Searcher
**Result**: Full inference bridge achieves SSIM 0.998+, eliminates precision boundaries

---

## What to Migrate (and What to Skip)

### ✅ MIGRATE THESE PATTERNS

#### 1. TensorBridge Class (Direct Copy)
**Source**: `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` (lines 25-160)

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tenstorrent_backend.py`

**Why**: Proven shared memory protocol, reusable as-is

**Code**:
```python
class TensorBridge:
    """Zero-copy tensor sharing via shared memory"""

    def tensor_to_shm(self, tensor: torch.Tensor) -> Dict[str, Any]:
        # Creates shared memory segment
        # Returns handle: {"shm_name": "...", "shape": [...], "dtype": "..."}

    def tensor_from_shm(self, handle: Dict[str, Any]) -> torch.Tensor:
        # Reconstructs tensor from shared memory
        # Copies data to avoid lifetime issues
```

---

#### 2. Bridge Communication Protocol (Direct Copy)
**Source**: `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` (lines 193-248)

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tenstorrent_backend.py`

**Why**: Proven msgpack + length-prefix framing

**Protocol**:
```
Request format:
{
    "operation": "full_denoise",
    "data": {...},
    "request_id": "optional-uuid"
}

Wire format:
[4-byte length (big-endian)][msgpack binary data]

Response format:
{
    "status": "success" | "error",
    "error": "message if error",
    "data": {...}
}
```

---

#### 3. TT_FullDenoise Pattern (Reference, Adapt)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py` (lines 251-433)

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`

**Why**: Proven architecture (SSIM 0.998+)

**Key Patterns to Replicate**:

```python
class TT_FullDenoise:
    def denoise(self, model, positive, negative, latent_image, seed, steps, cfg, scheduler):
        # PATTERN 1: Extract conditioning from ComfyUI format
        positive_cond = positive[0][0]  # [B, seq_len, dim]
        negative_cond = negative[0][0]

        # PATTERN 2: Extract SDXL metadata
        positive_meta = positive[0][1] if len(positive[0]) > 1 else {}
        positive_pooled = positive_meta.get("pooled_output")
        time_ids = positive_meta.get("time_ids")

        # PATTERN 3: Default time_ids if missing
        if time_ids is None:
            time_ids = torch.tensor([[1024, 1024, 0, 0, 6.0, 2.5]])

        # PATTERN 4: Bridge call with shared memory
        data = {
            "model_id": model.model_id,
            "latent": backend.tensor_bridge.tensor_to_shm(latent_samples),
            "positive_conditioning": backend.tensor_bridge.tensor_to_shm(positive_cond),
            "negative_conditioning": backend.tensor_bridge.tensor_to_shm(negative_cond),
            "positive_text_embeds": backend.tensor_bridge.tensor_to_shm(positive_pooled),
            "negative_text_embeds": backend.tensor_bridge.tensor_to_shm(negative_pooled),
            "time_ids": time_ids.tolist(),
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "seed": seed,
            "scheduler": scheduler,
        }

        response = backend._send_receive("full_denoise", data)

        # PATTERN 5: Return in ComfyUI latent format
        denoised_latent = backend.tensor_bridge.tensor_from_shm(response["denoised_latent"])
        result = latent_image.copy()
        result["samples"] = denoised_latent
        return (result,)
```

**CRITICAL**:
- Conditioning extraction pattern (positive[0][0], positive[0][1])
- Default time_ids (1024, 1024, 0, 0, 6.0, 2.5)
- Separate positive/negative parameters (NOT batched)

---

#### 4. Utility Functions (Direct Copy)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/utils.py`

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/utils.py`

**Functions**:
- `get_model_config(model_type)` - Model configuration lookups
- `validate_latent_shape(latent, model_type)` - Input validation
- `format_bytes(bytes_val)` - Human-readable byte formatting

---

#### 5. Model Management Modifications (Minimal)
**Source**: `/home/tt-admin/ComfyUI-tt/comfy/model_management.py` (lines 142-209)

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/comfy/model_management.py`

**Changes** (+33 lines):
```python
# Add to CPUState enum
class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2
    TENSTORRENT = 3  # NEW

# Add Tenstorrent detection
tt_available = False
try:
    socket_path = os.getenv("TT_COMFY_SOCKET", "/tmp/tt-comfy.sock")
    if os.path.exists(socket_path):
        tt_available = True
except:
    tt_available = False

# Add helper function
def is_tenstorrent():
    return tt_available and cpu_state == CPUState.TENSTORRENT
```

---

#### 6. CLI Arguments (Minimal)
**Source**: `/home/tt-admin/ComfyUI-tt/comfy/cli_args.py` (lines 94-97)

**Destination**: `/home/tt-admin/ComfyUI-tt_standalone/comfy/cli_args.py`

**Changes** (+3 lines):
```python
parser.add_argument("--tenstorrent", action="store_true", help="Use Tenstorrent accelerator")
parser.add_argument("--tt-socket", type=str, default="/tmp/tt-comfy.sock", help="Bridge socket path")
parser.add_argument("--tt-device", type=int, default=0, help="Tenstorrent device ID")
```

---

### ❌ DO NOT MIGRATE THESE

#### 1. TTModelWrapper (Per-Step Integration)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (lines 29-1081)

**Why Skip**:
- Full inference bridge eliminates per-step calls
- Complex, unnecessary with full denoise approach
- Source of Phase 0 issues (timestep, CFG)

---

#### 2. Phase 0 Fixes (Timestep Conversion)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (line 363)

**Why Skip**:
- Full inference bridge never receives sigma from ComfyUI
- Bridge handles timesteps internally via its scheduler
- No precision boundary to cross

---

#### 3. Phase 0 Fixes (CFG Unbatching)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (lines 264-336)

**Why Skip**:
- Full inference sends positive/negative as separate parameters
- Bridge handles CFG internally
- No batched input to unbatch

---

#### 4. Custom Samplers (tt_samplers.py)
**Source**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/tt_samplers.py`

**Why Skip**:
- Designed for per-step integration
- Full inference uses bridge's internal scheduler
- ComfyUI standard samplers can be used if needed

---

## Implementation Plan for ComfyUI-tt_standalone

### Phase 1: Core Infrastructure (2-3 hours)

**Step 1.1**: Create backend structure
```bash
mkdir -p /home/tt-admin/ComfyUI-tt_standalone/comfy/backends
touch /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/__init__.py
```

**Step 1.2**: Copy TensorBridge class
- Source: ComfyUI-tt/comfy/backends/tenstorrent_backend.py (lines 25-160)
- Destination: ComfyUI-tt_standalone/comfy/backends/tenstorrent_backend.py
- Action: Direct copy, no modifications

**Step 1.3**: Copy bridge communication protocol
- Source: ComfyUI-tt/comfy/backends/tenstorrent_backend.py (lines 193-248)
- Destination: Same file as Step 1.2
- Action: Direct copy

**Step 1.4**: Update model_management.py
- Add CPUState.TENSTORRENT enum value
- Add bridge detection logic
- Add is_tenstorrent() function
- Total: +33 lines

**Step 1.5**: Update cli_args.py
- Add --tenstorrent, --tt-socket, --tt-device arguments
- Total: +3 lines

---

### Phase 2: Custom Nodes (3-4 hours)

**Step 2.1**: Create node structure
```bash
mkdir -p /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes
touch /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/__init__.py
```

**Step 2.2**: Copy utility functions
- Source: ComfyUI-tt/custom_nodes/tenstorrent_nodes/utils.py
- Destination: ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/utils.py
- Functions: get_model_config, validate_latent_shape, format_bytes

**Step 2.3**: Implement TT_FullDenoise node
- Reference: ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py (lines 251-433)
- Destination: ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py
- Action: Adapt pattern, ensure correct conditioning extraction

**Step 2.4**: Implement TT_CheckpointLoader node
- Reference: ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py (lines 58-153)
- Destination: Same as Step 2.3
- Action: Simplified version that just returns model_id

**Step 2.5**: Register nodes
- Update __init__.py to export NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

---

### Phase 3: Bridge Server (4-6 hours)

**Step 3.1**: Create bridge server structure
```bash
mkdir -p /home/tt-admin/tt-metal/comfyui_bridge
cd /home/tt-admin/tt-metal/comfyui_bridge
touch server.py
```

**Step 3.2**: Implement bridge server
- Listen on Unix socket /tmp/tt-comfy.sock
- Implement msgpack + length-prefix protocol (match client)
- Handle "full_denoise" operation
- Forward to SDXLRunner

**Pseudocode**:
```python
# server.py
import socket
import msgpack
import struct
from pathlib import Path

def handle_full_denoise(data):
    # Reconstruct tensors from shared memory
    latent = tensor_bridge.tensor_from_shm(data["latent"])
    positive_cond = tensor_bridge.tensor_from_shm(data["positive_conditioning"])
    # ...

    # Call SDXLRunner
    result = sdxl_runner.run_inference(
        prompt=None,  # Already encoded
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        # ...
    )

    # Return via shared memory
    return {"denoised_latent": tensor_bridge.tensor_to_shm(result)}

def main():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind("/tmp/tt-comfy.sock")
    sock.listen(1)

    while True:
        conn, _ = sock.accept()
        # Handle request/response with same protocol as client
        # ...
```

**Step 3.3**: Integrate with standalone server
- Import SDXLRunner from sdxl_runner.py
- Initialize TT device
- Handle model loading

---

### Phase 4: Testing (2-3 hours)

**Step 4.1**: Unit test TensorBridge
```python
# Test shared memory transfer
tensor = torch.randn(1, 4, 128, 128)
handle = tensor_bridge.tensor_to_shm(tensor)
reconstructed = tensor_bridge.tensor_from_shm(handle)
assert torch.allclose(tensor, reconstructed)
```

**Step 4.2**: Test bridge communication
```python
# Test msgpack protocol
backend = TenstorrentBackend()
response = backend._send_receive("ping", {})
assert response["status"] == "success"
```

**Step 4.3**: Integration test
```python
# Run full workflow
# 1. Load model
# 2. Encode prompts
# 3. TT_FullDenoise
# 4. Decode with VAE
# 5. Save image
# 6. Calculate SSIM vs reference
```

**Success Criteria**:
- SSIM >= 0.90 vs PyTorch reference
- No crashes in 10 consecutive runs
- Latency < 30s for 20 steps @ 1024x1024

---

### Phase 5: Documentation (1-2 hours)

**Step 5.1**: Create README
- Installation instructions
- Bridge server setup
- Example workflow
- Troubleshooting

**Step 5.2**: Document architecture
- Diagram showing ComfyUI ↔ Bridge ↔ TT hardware
- Message flow diagram
- API reference

---

## Timeline Estimate

| Phase | Time | Dependencies |
|-------|------|--------------|
| Phase 1: Core Infrastructure | 2-3 hours | None |
| Phase 2: Custom Nodes | 3-4 hours | Phase 1 complete |
| Phase 3: Bridge Server | 4-6 hours | Phase 1 complete |
| Phase 4: Testing | 2-3 hours | Phase 2 & 3 complete |
| Phase 5: Documentation | 1-2 hours | Phase 4 complete |
| **Total** | **12-18 hours** | Sequential implementation |

**Parallelization Opportunity**:
- Phase 2 and Phase 3 can be done in parallel
- Reduces total time to **10-14 hours**

---

## Architecture Comparison

### Old Approach (ComfyUI-tt with Phase 0 Fixes)
```
ComfyUI → TTModelWrapper → Bridge → TT UNet (per-step)
           ↓ Timestep fix
           ↓ CFG unbatch fix
           ↓ Per-step calls (40-100 calls per image)
```

**Issues**:
- Precision boundaries at every step
- CFG batching incompatibility
- Complex timestep conversion
- 2x bridge calls per step with CFG

---

### New Approach (ComfyUI-tt_standalone Full Inference)
```
ComfyUI → TT_FullDenoise → Bridge → SDXLRunner (full loop)
           ↓ Extract conditioning
           ↓ Single call per image
           ↓ Bridge owns denoising loop
```

**Advantages**:
- No precision boundaries
- No CFG batching issues
- No timestep conversion needed
- Single bridge call per image
- **Phase 0 fixes NOT NEEDED**

---

## Files to Create/Modify Summary

### NEW FILES (Create from scratch):
1. `/home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tenstorrent_backend.py` (~400 lines)
2. `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/__init__.py` (~50 lines)
3. `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` (~300 lines)
4. `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/utils.py` (~150 lines)
5. `/home/tt-admin/tt-metal/comfyui_bridge/server.py` (~400 lines)

### MODIFY EXISTING FILES:
1. `/home/tt-admin/ComfyUI-tt_standalone/comfy/model_management.py` (+33 lines)
2. `/home/tt-admin/ComfyUI-tt_standalone/comfy/cli_args.py` (+3 lines)

**Total New Code**: ~1,300 lines (vs ~2,600 in ComfyUI-tt, but simpler architecture)

---

## Risk Assessment

### Low Risk Items ✅
- TensorBridge implementation (proven, copy as-is)
- Bridge protocol (proven, copy as-is)
- Full denoise pattern (SSIM 0.998+ proven)

### Medium Risk Items ⚠️
- Bridge server integration with SDXLRunner (new code)
- Conditioning extraction (must match exactly)
- time_ids defaults (must be correct for SDXL)

### High Risk Items 🔴
- None identified

**Overall Risk**: **LOW** - Architecture is proven, just need careful implementation

---

## Success Criteria

### Phase 1 Complete:
- [ ] Backend communicates with bridge
- [ ] TensorBridge transfers tensors correctly
- [ ] CLI arguments recognized

### Phase 2 Complete:
- [ ] Nodes show up in ComfyUI menu
- [ ] TT_FullDenoise node receives correct parameters
- [ ] Conditioning extraction works

### Phase 3 Complete:
- [ ] Bridge server starts and listens
- [ ] Receives and parses requests
- [ ] SDXLRunner generates images

### Phase 4 Complete:
- [ ] SSIM >= 0.90 vs reference
- [ ] 10 consecutive successful generations
- [ ] No memory leaks
- [ ] Latency acceptable

---

## Final Recommendation

### ✅ **PROCEED WITH MIGRATION PLAN**

**Actions**:
1. **Abandon** `/home/tt-admin/ComfyUI-tt/` for new work
2. **Use** `/home/tt-admin/ComfyUI-tt_standalone/` as integration base
3. **Reference** ComfyUI-tt code for patterns (don't copy wholesale)
4. **Implement** Full Inference Bridge architecture
5. **Skip** Phase 0 fixes entirely (not needed)

**Expected Outcome**:
- Simpler codebase (~50% less code)
- No precision boundary issues
- No CFG batching issues
- No timestep conversion issues
- Better maintainability
- Same or better quality (SSIM 0.998+)

**Time Investment**: 12-18 hours for complete implementation

---

## Next Steps

1. ✅ Review and approve this migration plan
2. ▶️ Begin Phase 1: Core Infrastructure implementation
3. → Build bridge server (Phase 3 in parallel)
4. → Implement nodes (Phase 2)
5. → Test integration (Phase 4)
6. → Document and deploy (Phase 5)

---

**Decision Made**: START FRESH, REFERENCE PATTERNS, USE FULL INFERENCE BRIDGE

**Phase 0 Work Status**: NOT WASTED - Taught us what NOT to do and validated the full inference approach

**Confidence**: HIGH - Architecture is proven, migration is straightforward
