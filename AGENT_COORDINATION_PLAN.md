# Agent Coordination Plan: ComfyUI-tt_standalone Integration

**Date**: 2025-12-12
**Objective**: Coordinate specialized agents to implement Full Inference Bridge architecture in ComfyUI-tt_standalone

---

## Executive Summary

This plan orchestrates 5 specialized agents to implement the migration in 5 phases:
1. **code-writer** → Core Infrastructure (backend system)
2. **code-writer** → Custom Nodes (TT_CheckpointLoader, TT_FullDenoise)
3. **code-writer** → Bridge Server (comfyui_bridge/)
4. **critical-reviewer** → Quality validation and testing
5. **integration-orchestrator** → Final integration and validation

**Total Estimated Time**: 12-18 hours (10-14 with parallelization)

---

## Phase 1: Core Infrastructure (Backend System)

### Agent: `code-writer`
### Timeline: 2-3 hours
### Dependencies: None (can start immediately)

**Prompt**:
```
Implement the Tenstorrent backend infrastructure for ComfyUI-tt_standalone.

**Context**:
We're building a Full Inference Bridge integration that connects ComfyUI-tt_standalone
to the standalone SDXL server. This phase creates the communication layer.

**Source References**:
- /home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py (copy TensorBridge and protocol)
- /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/utils.py (copy utilities)

**Target Location**: /home/tt-admin/ComfyUI-tt_standalone/

**Tasks**:

1. **Create Backend Directory Structure**
   ```
   /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/
   ├── __init__.py
   └── tenstorrent_backend.py
   ```

2. **Implement tenstorrent_backend.py** (~400 lines)
   - Copy TensorBridge class from ComfyUI-tt (lines 25-160)
   - Copy TenstorrentBackend class (lines 162-361)
   - Modify operations to support "full_denoise" (not per-step)
   - Keep: tensor_to_shm(), tensor_from_shm(), _send_receive()
   - Remove: apply_unet() (not needed for full inference)
   - Add: full_denoise() operation handler

3. **Modify comfy/model_management.py** (+33 lines)
   Location: /home/tt-admin/ComfyUI-tt_standalone/comfy/model_management.py

   Add to CPUState enum (around line 30):
   ```python
   TENSTORRENT = 6  # Tenstorrent hardware acceleration
   ```

   Add detection function (around line 200):
   ```python
   def is_tenstorrent_device(device):
       """Check if device is Tenstorrent hardware."""
       if isinstance(device, str):
           return device.lower() == 'tenstorrent' or device.lower() == 'tt'
       return False

   def get_tenstorrent_backend():
       """Get Tenstorrent backend singleton."""
       try:
           from comfy.backends.tenstorrent_backend import get_backend
           return get_backend()
       except ImportError:
           return None
   ```

4. **Modify comfy/cli_args.py** (+3 lines)
   Location: /home/tt-admin/ComfyUI-tt_standalone/comfy/cli_args.py

   Add arguments (around line 50):
   ```python
   parser.add_argument("--tenstorrent", action="store_true", help="Enable Tenstorrent hardware acceleration")
   parser.add_argument("--tt-socket", type=str, default="/tmp/tt-comfy.sock", help="Path to Tenstorrent bridge Unix socket")
   parser.add_argument("--tt-device", type=int, default=0, help="Tenstorrent device ID (0-31)")
   ```

5. **Create Utility Module**
   ```
   /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tt_utils.py
   ```

   Copy from /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/utils.py:
   - get_model_config() (lines 78-126)
   - validate_latent_shape() (lines 147-168)
   - format_bytes() (lines 185-199)
   - estimate_tensor_memory() (lines 170-183)

**Validation**:
After implementation, verify:
1. `python3 -c "from comfy.backends.tenstorrent_backend import TensorBridge; print('✓ Import OK')"`
2. `python3 -c "from comfy.backends.tt_utils import get_model_config; print(get_model_config('sdxl'))"`
3. Check that cli_args.py has --tenstorrent flag

**Deliverable**:
- Backend infrastructure complete
- All imports successful
- CLI args registered
- Ready for custom nodes (Phase 2)
```

---

## Phase 2: Custom Nodes (TT Integration Nodes)

### Agent: `code-writer`
### Timeline: 3-4 hours
### Dependencies: Phase 1 complete

**Prompt**:
```
Implement Tenstorrent custom nodes for ComfyUI-tt_standalone.

**Context**:
Create custom nodes that provide the ComfyUI interface to Tenstorrent hardware.
These nodes use the backend from Phase 1 to communicate with the bridge server.

**Source References**:
- /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py (pattern reference)
- TT_FullDenoise (lines 251-433) - proven SSIM 0.998+ pattern
- TT_CheckpointLoader (lines 36-126) - initialization pattern

**Target Location**: /home/tt-admin/ComfyUI-tt_standalone/

**Tasks**:

1. **Create Custom Nodes Directory**
   ```
   /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/
   ├── __init__.py
   ├── nodes.py
   └── README.md
   ```

2. **Implement __init__.py** (~30 lines)
   ```python
   """
   Tenstorrent ComfyUI Nodes

   Provides integration with Tenstorrent hardware via bridge server.
   """

   from .nodes import (
       TT_CheckpointLoader,
       TT_FullDenoise,
       TT_ModelInfo,
       TT_UnloadModel
   )

   NODE_CLASS_MAPPINGS = {
       "TT_CheckpointLoader": TT_CheckpointLoader,
       "TT_FullDenoise": TT_FullDenoise,
       "TT_ModelInfo": TT_ModelInfo,
       "TT_UnloadModel": TT_UnloadModel,
   }

   NODE_DISPLAY_NAME_MAPPINGS = {
       "TT_CheckpointLoader": "TT Checkpoint Loader",
       "TT_FullDenoise": "TT Full Denoise",
       "TT_ModelInfo": "TT Model Info",
       "TT_UnloadModel": "TT Unload Model",
   }

   __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
   ```

3. **Implement nodes.py** (~500 lines)

   **TT_CheckpointLoader** (adapt from ComfyUI-tt pattern):
   - INPUT_TYPES: model_type (sdxl/sd35/sd14), device_id (0-31)
   - RETURN_TYPES: ("MODEL", "CLIP", "VAE")
   - FUNCTION: load_checkpoint()
   - Logic:
     * Get backend singleton
     * Initialize model on bridge: backend.init_model(model_type, config)
     * Return lightweight wrappers (just store model_id, no actual model weights)

   **TT_FullDenoise** (adapt from proven pattern):
   - INPUT_TYPES: model, positive, negative, latent_image, seed, steps, cfg, scheduler
   - RETURN_TYPES: ("LATENT",)
   - FUNCTION: denoise()
   - Logic:
     * Extract conditioning tensors from ComfyUI format: [[tensor, {metadata}]]
     * Extract pooled_output and time_ids from metadata
     * Serialize all tensors to shared memory via backend.tensor_bridge.tensor_to_shm()
     * Call backend._send_receive("full_denoise", data)
     * Deserialize result from shared memory
     * Return in ComfyUI latent format: {"samples": denoised_latent}

   **CRITICAL**: Use exact conditioning extraction pattern from TT_FullDenoise lines 350-389:
   ```python
   positive_cond = positive[0][0]  # [B, seq_len, dim]
   positive_meta = positive[0][1] if len(positive[0]) > 1 else {}
   positive_pooled = positive_meta.get("pooled_output")
   time_ids = positive_meta.get("time_ids", default_time_ids)
   ```

   **TT_ModelInfo** (copy pattern):
   - Display model information (model_id, type, config)
   - Useful for debugging

   **TT_UnloadModel** (copy pattern):
   - Explicitly unload model from bridge
   - Frees device memory

4. **Create README.md** (~100 lines)
   - Installation instructions
   - Usage examples
   - Workflow example (text → CLIP → FullDenoise → VAE decode → save)
   - Troubleshooting (bridge not running, socket errors)

**Validation**:
After implementation, verify:
1. Nodes load without errors: `python3 -c "import sys; sys.path.insert(0, '/home/tt-admin/ComfyUI-tt_standalone'); from custom_nodes.tenstorrent_nodes import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"`
2. Check node has correct INPUT_TYPES
3. Verify conditioning extraction logic matches proven pattern

**Deliverable**:
- 4 custom nodes implemented
- Node registration complete
- README documentation
- Ready for bridge server (Phase 3)
```

---

## Phase 3: Bridge Server Implementation

### Agent: `code-writer`
### Timeline: 4-6 hours
### Dependencies: Phase 1 complete (can run parallel with Phase 2)

**Prompt**:
```
Implement the Tenstorrent ComfyUI Bridge Server.

**Context**:
Create a Unix socket server that receives requests from ComfyUI custom nodes
and executes them using the standalone SDXL server/runner.

**Architecture**:
```
ComfyUI (frontend)
    ↓ Unix socket (/tmp/tt-comfy.sock)
Bridge Server (this component)
    ↓ Direct Python API
SDXLRunner (tt-metal backend)
    ↓
Tenstorrent Hardware
```

**Source References**:
- /home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py (protocol reference)
- /home/tt-admin/tt-metal/sdxl_runner.py (integration target)
- /home/tt-admin/tt-metal/sdxl_server.py (HTTP server pattern)

**Target Location**: /home/tt-admin/tt-metal/

**Tasks**:

1. **Create Bridge Directory**
   ```
   /home/tt-admin/tt-metal/comfyui_bridge/
   ├── __init__.py
   ├── server.py           (main server)
   ├── handlers.py         (operation handlers)
   ├── protocol.py         (msgpack protocol)
   └── README.md
   ```

2. **Implement protocol.py** (~150 lines)
   - Function: receive_message(sock) → dict
     * Receive 4-byte length prefix (big-endian)
     * Receive message bytes
     * Deserialize with msgpack.unpackb()

   - Function: send_message(sock, data) → None
     * Serialize with msgpack.packb()
     * Send 4-byte length prefix
     * Send message bytes

   - Function: send_error(sock, error_msg)
     * Package error response
     * Use send_message()

   - Function: send_success(sock, data)
     * Package success response
     * Use send_message()

3. **Implement handlers.py** (~300 lines)
   - Class: OperationHandler
     * __init__(sdxl_runner: SDXLRunner)
     * handle_init_model(data) → model_id
     * handle_full_denoise(data) → denoised_latent
     * handle_ping(data) → status
     * handle_unload_model(data) → None

   **CRITICAL: handle_full_denoise() implementation**:
   ```python
   def handle_full_denoise(self, data):
       """
       Run complete denoising loop.

       Input data:
           - model_id: str
           - latent: shm_handle (initial noise)
           - positive_conditioning: shm_handle
           - negative_conditioning: shm_handle
           - positive_text_embeds: shm_handle (pooled)
           - negative_text_embeds: shm_handle (pooled)
           - time_ids: list
           - num_inference_steps: int
           - guidance_scale: float
           - seed: int
           - scheduler: str

       Returns:
           - denoised_latent: shm_handle
       """
       # 1. Deserialize tensors from shared memory
       latent = self.tensor_bridge.tensor_from_shm(data["latent"])
       positive_cond = self.tensor_bridge.tensor_from_shm(data["positive_conditioning"])
       # ... etc

       # 2. Call SDXLRunner.generate_image()
       result_latent = self.sdxl_runner.generate_image(
           latent=latent,
           prompt_embeds=positive_cond,
           negative_prompt_embeds=negative_cond,
           pooled_prompt_embeds=positive_pooled,
           negative_pooled_prompt_embeds=negative_pooled,
           num_inference_steps=data["num_inference_steps"],
           guidance_scale=data["guidance_scale"],
           generator=torch.Generator().manual_seed(data["seed"])
       )

       # 3. Serialize result to shared memory
       result_handle = self.tensor_bridge.tensor_to_shm(result_latent)

       return {"denoised_latent": result_handle}
   ```

4. **Implement server.py** (~200 lines)
   ```python
   class ComfyUIBridgeServer:
       def __init__(self, socket_path, device_id=0):
           self.socket_path = socket_path
           self.device_id = device_id
           self.sdxl_runner = None
           self.handler = None
           self.sock = None

       def start(self):
           # Initialize SDXLRunner
           self.sdxl_runner = SDXLRunner(device_id=self.device_id)
           self.handler = OperationHandler(self.sdxl_runner)

           # Create Unix socket
           if os.path.exists(self.socket_path):
               os.unlink(self.socket_path)

           self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
           self.sock.bind(self.socket_path)
           self.sock.listen(5)

           print(f"Bridge server listening on {self.socket_path}")

           # Accept connections
           while True:
               client_sock, _ = self.sock.accept()
               self.handle_client(client_sock)

       def handle_client(self, client_sock):
           try:
               # Receive request
               request = receive_message(client_sock)
               operation = request["operation"]
               data = request["data"]

               # Dispatch to handler
               if operation == "init_model":
                   result = self.handler.handle_init_model(data)
               elif operation == "full_denoise":
                   result = self.handler.handle_full_denoise(data)
               # ... etc

               # Send response
               send_success(client_sock, result)

           except Exception as e:
               send_error(client_sock, str(e))
           finally:
               client_sock.close()
   ```

5. **Create Launch Script**
   ```
   /home/tt-admin/tt-metal/launch_comfyui_bridge.sh
   ```

   ```bash
   #!/bin/bash

   # Launch ComfyUI Bridge Server

   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

   # Activate tt-metal environment
   source "${SCRIPT_DIR}/python_env/bin/activate"

   # Default values
   SOCKET_PATH="${SOCKET_PATH:-/tmp/tt-comfy.sock}"
   DEVICE_ID="${DEVICE_ID:-0}"

   echo "Starting ComfyUI Bridge Server..."
   echo "Socket: ${SOCKET_PATH}"
   echo "Device: ${DEVICE_ID}"

   python3 -m comfyui_bridge.server \
       --socket-path "${SOCKET_PATH}" \
       --device-id "${DEVICE_ID}"
   ```

6. **Create README.md**
   - Architecture diagram
   - Installation steps
   - Usage examples
   - API reference (operations and data formats)

**Validation**:
After implementation, verify:
1. Import works: `python3 -c "from comfyui_bridge.server import ComfyUIBridgeServer; print('✓ OK')"`
2. Protocol test: Create simple client that sends ping operation
3. Check launch script is executable: `chmod +x launch_comfyui_bridge.sh`

**Deliverable**:
- Bridge server implementation complete
- Protocol handlers implemented
- Launch script ready
- Ready for integration testing (Phase 4)
```

---

## Phase 4: Testing and Validation

### Agent: `critical-reviewer`
### Timeline: 2-3 hours
### Dependencies: Phases 1, 2, 3 complete

**Prompt**:
```
Validate the ComfyUI-tt_standalone integration implementation.

**Context**:
Three components have been implemented:
1. Backend infrastructure (Phase 1)
2. Custom nodes (Phase 2)
3. Bridge server (Phase 3)

Your task is to review, test, and validate the complete integration.

**Review Scope**:

1. **Code Quality Review**
   - Check all imports are correct
   - Verify error handling is comprehensive
   - Validate tensor shape handling
   - Check shared memory cleanup (memory leaks?)
   - Verify socket connection management
   - Review logging levels and messages

2. **Architecture Validation**
   - Verify backend follows ComfyUI patterns
   - Check custom nodes match ComfyUI interface requirements
   - Validate bridge protocol matches both sides
   - Verify tensor serialization is consistent

3. **Integration Points**
   - Backend ↔ Custom Nodes: Check get_backend() singleton
   - Custom Nodes ↔ Bridge: Check protocol format matches
   - Bridge ↔ SDXLRunner: Check API compatibility

4. **Unit Tests** (create if missing)
   Create test files:
   ```
   /home/tt-admin/tt-metal/comfyui_bridge/tests/
   ├── test_protocol.py
   ├── test_handlers.py
   └── test_integration.py
   ```

   **test_protocol.py**:
   - Test message framing (length prefix)
   - Test msgpack serialization
   - Test error responses

   **test_handlers.py**:
   - Test TensorBridge.tensor_to_shm() / tensor_from_shm()
   - Test shared memory cleanup
   - Mock full_denoise handler

   **test_integration.py**:
   - Test complete flow: ComfyUI node → Bridge → (mock) SDXLRunner
   - Test with actual tensor data
   - Validate result format

5. **Manual Testing Checklist**
   Create: /home/tt-admin/tt-metal/INTEGRATION_TEST_PLAN.md

   Steps:
   1. Start bridge server: `./launch_comfyui_bridge.sh`
   2. Start ComfyUI: `cd /home/tt-admin/ComfyUI-tt_standalone && python main.py --tenstorrent`
   3. Load workflow with TT nodes
   4. Run simple generation (20 steps)
   5. Check logs for errors
   6. Validate output image exists
   7. Calculate SSIM vs reference (target >= 0.90)

6. **Performance Validation**
   - Check latency per operation
   - Monitor memory usage (no leaks)
   - Verify shared memory segments are cleaned up
   - Test multiple sequential generations

7. **Error Handling Tests**
   - Bridge server not running → clear error message
   - Invalid socket path → graceful failure
   - Malformed request → error response, no crash
   - Device not available → proper error

**Deliverables**:
1. Code review report with issues found
2. Unit tests implemented
3. Integration test plan document
4. List of fixes needed (if any)
5. Go/No-Go recommendation for Phase 5

**Success Criteria**:
- All unit tests pass
- No memory leaks detected
- Clean error handling
- Integration test successful
- SSIM >= 0.90 on test generation
```

---

## Phase 5: Final Integration and Documentation

### Agent: `integration-orchestrator`
### Timeline: 1-2 hours
### Dependencies: Phase 4 complete and approved

**Prompt**:
```
Coordinate final integration, documentation, and deployment of ComfyUI-tt_standalone.

**Context**:
All components are implemented and tested. Final coordination needed for:
1. Apply any fixes from Phase 4 review
2. Create comprehensive documentation
3. Validate end-to-end workflow
4. Prepare for production use

**Tasks**:

1. **Apply Critical Fixes**
   - Review Phase 4 critical-reviewer findings
   - Coordinate code-writer agent to fix any issues
   - Re-run validation tests

2. **Create User Documentation**
   Location: /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/

   **USER_GUIDE.md**:
   - Installation steps
   - Starting the bridge server
   - Starting ComfyUI with --tenstorrent flag
   - Creating workflows with TT nodes
   - Example workflow JSON
   - Troubleshooting common issues

   **ARCHITECTURE.md**:
   - System architecture diagram
   - Data flow explanation
   - Component responsibilities
   - Protocol specification
   - Extension points for future work

3. **Create Developer Documentation**
   Location: /home/tt-admin/tt-metal/comfyui_bridge/

   **DEVELOPER_GUIDE.md**:
   - Code structure overview
   - Adding new operations
   - Debugging techniques
   - Performance profiling
   - Testing guidelines

4. **Integration Validation**
   Run complete end-to-end test:
   ```bash
   # Terminal 1: Start bridge
   cd /home/tt-admin/tt-metal
   ./launch_comfyui_bridge.sh

   # Terminal 2: Start ComfyUI
   cd /home/tt-admin/ComfyUI-tt_standalone
   python main.py --tenstorrent --listen 0.0.0.0 --port 8188

   # Terminal 3: Run test workflow
   python test_workflow.py
   ```

   Expected results:
   - Bridge connects successfully
   - Model loads on device
   - Generation completes in ~30-40s (20 steps)
   - Output image matches quality expectations (SSIM >= 0.90)
   - No errors or warnings in logs

5. **Create Deployment Checklist**
   Location: /home/tt-admin/ComfyUI-tt_standalone/DEPLOYMENT.md

   - Prerequisites (tt-metal installed, device available)
   - Environment setup
   - Configuration options
   - Starting services
   - Health checks
   - Monitoring and logging
   - Backup and recovery

6. **Performance Baseline**
   Document performance metrics:
   - Model load time
   - First inference time (cold start)
   - Subsequent inference time (warm)
   - Memory usage (bridge + ComfyUI)
   - Quality metrics (SSIM vs reference)

7. **Known Issues and Roadmap**
   Create: /home/tt-admin/tt-metal/ROADMAP.md

   **Known Limitations**:
   - Only supports SDXL currently (sd35, sd14 placeholders)
   - Full inference only (no per-step sampling)
   - Single device support

   **Future Enhancements**:
   - Multi-device support
   - Additional schedulers
   - ControlNet integration
   - LoRA support
   - Per-step sampling (if needed)

**Deliverables**:
1. All documentation complete and reviewed
2. End-to-end test passing
3. Deployment checklist validated
4. Performance baseline documented
5. System ready for production use

**Final Sign-Off Criteria**:
- ✅ All Phase 4 fixes applied
- ✅ Documentation complete and accurate
- ✅ End-to-end test successful
- ✅ SSIM >= 0.90 achieved
- ✅ No critical issues remaining
- ✅ Deployment checklist validated
```

---

## Orchestration Flow

### Sequential Dependencies

```
Phase 1 (Backend)
    ↓
    ├─→ Phase 2 (Custom Nodes) [depends on Phase 1]
    │
    └─→ Phase 3 (Bridge Server) [independent, can run parallel with Phase 2]
         ↓
         Phase 2 + Phase 3 complete
              ↓
         Phase 4 (Testing & Validation)
              ↓
         Phase 5 (Final Integration)
```

### Parallel Execution Opportunities

**Parallel Group 1** (after Phase 1):
- Phase 2: Custom Nodes implementation
- Phase 3: Bridge Server implementation

**Reason**: These components don't directly depend on each other, only on Phase 1.

---

## Integration Points to Monitor

### Critical Integration Point 1: Backend ↔ Custom Nodes
**Interface**: `get_backend()` → `TenstorrentBackend` instance
**Validator**: Check singleton pattern works
**Test**: Import from node and verify backend.tensor_bridge exists

### Critical Integration Point 2: Custom Nodes ↔ Bridge Server
**Interface**: Unix socket + msgpack protocol
**Validator**: Check message format matches on both ends
**Test**: Send ping operation, verify response structure

### Critical Integration Point 3: Bridge ↔ SDXLRunner
**Interface**: SDXLRunner.generate_image() API
**Validator**: Check parameter names and types match
**Test**: Call generate_image with mock data, verify output shape

### Critical Integration Point 4: Shared Memory
**Interface**: TensorBridge shared memory segments
**Validator**: Check cleanup happens on both sides
**Test**: Monitor /dev/shm for leaked segments after operations

---

## Risk Mitigation

### Risk 1: Protocol Mismatch
**Mitigation**: Create protocol specification document before Phase 3
**Validation**: Unit test protocol.py send/receive roundtrip

### Risk 2: Memory Leaks (Shared Memory)
**Mitigation**: Comprehensive cleanup in finally blocks
**Validation**: Monitor /dev/shm during stress testing

### Risk 3: SDXLRunner API Changes
**Mitigation**: Document exact API used, version pin if needed
**Validation**: Check sdxl_runner.py for parameter compatibility

### Risk 4: ComfyUI Interface Assumptions
**Mitigation**: Reference official ComfyUI node examples
**Validation**: Test nodes with standard ComfyUI workflows first

### Risk 5: Socket Permission Issues
**Mitigation**: Document socket path requirements, provide fallbacks
**Validation**: Test with different socket paths and permissions

---

## Rollback Plan

If critical issues found during Phase 4:

1. **Backend Issues**:
   - Rollback Phase 1 changes to comfy/model_management.py and cli_args.py
   - Keep backend in separate file for debugging
   - Re-implement with fixes

2. **Custom Node Issues**:
   - Custom nodes are isolated, can be disabled
   - Fix and reload without restarting ComfyUI

3. **Bridge Server Issues**:
   - Bridge is separate process, can be restarted
   - Fix and relaunch without affecting ComfyUI

4. **Complete Rollback**:
   - Remove /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tenstorrent*
   - Remove custom_nodes/tenstorrent_nodes/
   - Revert cli_args.py and model_management.py
   - System returns to clean state

---

## Success Metrics

### Phase 1 Success:
- ✅ Backend imports successfully
- ✅ CLI args registered
- ✅ No syntax errors

### Phase 2 Success:
- ✅ Nodes appear in ComfyUI UI
- ✅ Node validation passes
- ✅ Correct INPUT/OUTPUT types

### Phase 3 Success:
- ✅ Bridge server starts without errors
- ✅ Socket created and listening
- ✅ Protocol test successful

### Phase 4 Success:
- ✅ All unit tests pass
- ✅ Integration test passes
- ✅ SSIM >= 0.90
- ✅ No memory leaks
- ✅ Clean error handling

### Phase 5 Success:
- ✅ Documentation complete
- ✅ End-to-end workflow successful
- ✅ System ready for production
- ✅ Performance baseline met

---

## Coordination Commands

### Start Phase 1:
```
Use Task tool with code-writer agent, prompt from "Phase 1: Core Infrastructure"
```

### Start Phase 2 (after Phase 1):
```
Use Task tool with code-writer agent, prompt from "Phase 2: Custom Nodes"
```

### Start Phase 3 (after Phase 1, can run parallel with Phase 2):
```
Use Task tool with code-writer agent, prompt from "Phase 3: Bridge Server"
```

### Start Phase 4 (after Phases 2 and 3):
```
Use Task tool with critical-reviewer agent, prompt from "Phase 4: Testing and Validation"
```

### Start Phase 5 (after Phase 4 approved):
```
Use Task tool with integration-orchestrator agent, prompt from "Phase 5: Final Integration"
```

---

## Agent Communication Protocol

Each agent will:
1. **Report status** at start, milestones, and completion
2. **Document artifacts** created (file paths, line counts)
3. **Flag blockers** immediately if dependencies missing
4. **Validate outputs** against success criteria before marking complete
5. **Provide handoff summary** for next phase agent

---

## Estimated Timeline

**Optimistic** (parallel execution, no issues): 10-12 hours
- Phase 1: 2 hours
- Phase 2+3 (parallel): 4 hours
- Phase 4: 2 hours
- Phase 5: 1 hour
- Buffer: 1-2 hours

**Realistic** (some serial execution, minor fixes): 14-16 hours
- Phase 1: 3 hours
- Phase 2: 4 hours
- Phase 3: 5 hours
- Phase 4: 3 hours
- Phase 5: 2 hours

**Conservative** (full serial, issues found): 18-22 hours
- Includes iteration cycles for fixes
- Additional testing and validation
- Documentation refinement

---

## Ready to Execute?

This plan is ready for orchestration. Next steps:

1. **Review and approve** this coordination plan
2. **Launch Phase 1** using code-writer agent
3. **Monitor progress** through agent status updates
4. **Proceed sequentially** through phases with validation gates
5. **Celebrate success** when Phase 5 completes!

---

**Questions before execution?**
- Resource allocation (can we run Phase 2+3 in parallel)?
- Priority adjustments (fast-track any phase)?
- Risk tolerance (stop on first issue vs. continue)?
- Testing depth (minimal vs. comprehensive)?
