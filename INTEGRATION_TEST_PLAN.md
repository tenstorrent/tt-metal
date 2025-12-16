# ComfyUI-tt_standalone Integration Test Plan

**Date**: 2025-12-12  
**Scope**: Complete validation of ComfyUI-Tenstorrent integration  
**Components**: Backend, Custom Nodes, Bridge Server

---

## Overview

This document outlines the comprehensive testing strategy for validating the ComfyUI-tt_standalone integration with Tenstorrent hardware acceleration.

## Test Environment

### Prerequisites

- **Hardware**: Tenstorrent Wormhole or T3K
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **Dependencies**: 
  - torch
  - ttnn
  - msgpack
  - ComfyUI-tt_standalone
  - tt-metal

### Environment Setup

```bash
# 1. Activate tt-metal environment
cd /home/tt-admin/tt-metal
source python_env/bin/activate

# 2. Verify device availability
python3 -c "import ttnn; print(f'Devices available: {ttnn.get_num_devices()}')"

# 3. Check ComfyUI installation
cd /home/tt-admin/ComfyUI-tt_standalone
python3 -c "import comfy; print('ComfyUI OK')"
```

---

## Phase 1: Unit Tests

### 1.1 Protocol Tests

**Location**: `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_protocol.py`

**Run**:
```bash
cd /home/tt-admin/tt-metal
python3 -m pytest comfyui_bridge/tests/test_protocol.py -v
```

**Expected Results**:
- All message framing tests pass
- Length-prefixed encoding/decoding works correctly
- Error handling for malformed messages
- Large message support (> 4KB)

**Success Criteria**: 100% pass rate

---

### 1.2 Handlers Tests

**Location**: `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_handlers.py`

**Run**:
```bash
python3 -m pytest comfyui_bridge/tests/test_handlers.py -v
```

**Test Coverage**:
- TensorBridge shared memory operations
- Tensor serialization/deserialization
- Multiple dtypes (float32, float16, int64)
- Memory cleanup verification
- Large tensor support (> 100MB)

**Success Criteria**: 100% pass rate, no memory leaks

---

### 1.3 Integration Tests

**Location**: `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_integration.py`

**Run**:
```bash
python3 -m pytest comfyui_bridge/tests/test_integration.py -v
```

**Test Coverage**:
- Full protocol flow (client -> server)
- TensorBridge compatibility between backend and bridge
- Error propagation
- Memory leak detection

**Success Criteria**: 100% pass rate

---

## Phase 2: Component Integration Tests

### 2.1 Backend Standalone Test

**Objective**: Verify backend can connect to bridge server

**Steps**:

1. Start bridge server:
```bash
cd /home/tt-admin/tt-metal
./launch_comfyui_bridge.sh --dev
```

2. In separate terminal, test backend connection:
```bash
cd /home/tt-admin/ComfyUI-tt_standalone
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'comfy')
from backends.tenstorrent_backend import get_backend

try:
    backend = get_backend()
    print("✓ Backend connected")
    
    # Test ping
    result = backend.ping()
    print(f"✓ Ping successful: {result}")
    
    backend.close()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Error: {e}")
    raise
PYEOF
```

**Expected Output**:
```
✓ Backend connected
✓ Ping successful: {'status': 'ok', 'model_loaded': False, 'model_id': None}
✓ Cleanup successful
```

**Success Criteria**: No connection errors, ping returns valid response

---

### 2.2 Custom Nodes Import Test

**Objective**: Verify custom nodes can be loaded by ComfyUI

**Steps**:

```bash
cd /home/tt-admin/ComfyUI-tt_standalone
python3 << 'PYEOF'
import sys
import os

# Add ComfyUI to path
sys.path.insert(0, os.getcwd())

# Test custom nodes import
from custom_nodes.tenstorrent_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

print(f"✓ Found {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_CLASS_MAPPINGS:
    print(f"  - {node_name}: {NODE_DISPLAY_NAME_MAPPINGS[node_name]}")

# Test node instantiation
TT_CheckpointLoader = NODE_CLASS_MAPPINGS["TT_CheckpointLoader"]
print(f"✓ Can instantiate TT_CheckpointLoader")

TT_FullDenoise = NODE_CLASS_MAPPINGS["TT_FullDenoise"]
print(f"✓ Can instantiate TT_FullDenoise")
PYEOF
```

**Expected Output**:
```
✓ Found 4 nodes:
  - TT_CheckpointLoader: TT Checkpoint Loader
  - TT_FullDenoise: TT Full Denoise
  - TT_ModelInfo: TT Model Info
  - TT_UnloadModel: TT Unload Model
✓ Can instantiate TT_CheckpointLoader
✓ Can instantiate TT_FullDenoise
```

**Success Criteria**: All nodes load without import errors

---

### 2.3 Bridge Server Initialization Test

**Objective**: Verify bridge server can initialize and load model

**Steps**:

1. Start bridge server with dev mode:
```bash
cd /home/tt-admin/tt-metal
./launch_comfyui_bridge.sh --dev
```

2. In separate terminal, test model initialization:
```bash
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/tt-admin/ComfyUI-tt_standalone/comfy')
from backends.tenstorrent_backend import get_backend

backend = get_backend()

print("Initializing SDXL model...")
model_id = backend.init_model(model_type="sdxl", config={"device_id": "0"})
print(f"✓ Model initialized: {model_id}")

# Verify model is loaded
status = backend.ping()
print(f"✓ Model status: loaded={status['model_loaded']}, id={status['model_id']}")

# Cleanup
backend.unload_model(model_id)
print("✓ Model unloaded")

backend.close()
PYEOF
```

**Expected Output**:
```
Initializing SDXL model...
✓ Model initialized: sdxl_abc12345
✓ Model status: loaded=True, id=sdxl_abc12345
✓ Model unloaded
```

**Success Criteria**: Model initializes successfully, takes < 5 min in dev mode

---

## Phase 3: End-to-End Tests

### 3.1 Full ComfyUI Workflow Test

**Objective**: Run complete generation through ComfyUI interface

**Steps**:

1. Start bridge server:
```bash
cd /home/tt-admin/tt-metal
./launch_comfyui_bridge.sh --dev
```

2. Start ComfyUI:
```bash
cd /home/tt-admin/ComfyUI-tt_standalone
python3 main.py --listen 0.0.0.0 --port 8188
```

3. Open browser: `http://localhost:8188`

4. Create workflow:
   - Add node: "TT Checkpoint Loader"
     - Set model_type: "sdxl"
     - Set device_id: 0
   - Add node: "TT Full Denoise"
     - Connect to checkpoint loader
     - Set positive prompt: "a beautiful mountain landscape"
     - Set negative prompt: "blurry, low quality"
     - Set steps: 12 (dev mode)
     - Set CFG: 7.0
     - Set size: 1024x1024
     - Set seed: 42
   - Add node: "Save Image"
     - Connect to TT Full Denoise output

5. Click "Queue Prompt"

**Expected Behavior**:
- Model loads successfully (< 3 min in dev mode)
- Generation completes without errors
- Image appears in output
- Generation time: < 30 seconds for 12 steps

**Success Criteria**:
- No Python errors in logs
- Image generated successfully
- Image quality visually acceptable

---

### 3.2 Multiple Generation Test

**Objective**: Test stability across multiple generations

**Steps**:

1. Use same workflow from 3.1
2. Change seed to different values (43, 44, 45)
3. Queue 3 prompts

**Expected Behavior**:
- All 3 generations complete
- No memory leaks (check with `nvidia-smi` or `ps aux`)
- Consistent generation time

**Success Criteria**: All generations succeed with similar timing

---

### 3.3 Error Handling Test

**Objective**: Verify graceful error handling

**Test Cases**:

#### 3.3.1 Bridge Server Not Running

1. Stop bridge server
2. Try to load checkpoint in ComfyUI

**Expected**: Clear error message "Cannot connect to Tenstorrent bridge server"

#### 3.3.2 Invalid Socket Path

1. Set wrong socket path:
```bash
export TT_COMFY_SOCKET=/tmp/nonexistent.sock
```
2. Start ComfyUI, try to load checkpoint

**Expected**: Connection error with socket path in message

#### 3.3.3 Malformed Request

Create test script:
```python
import socket
socket_path = "/tmp/tt-comfy.sock"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(socket_path)
sock.sendall(b"invalid data")
sock.close()
```

**Expected**: Bridge server logs error but doesn't crash

#### 3.3.4 Device Not Available

1. Use invalid device ID:
```bash
./launch_comfyui_bridge.sh --device-id 99
```

**Expected**: Clear error message about device availability

**Success Criteria**: All errors handled gracefully, no crashes

---

## Phase 4: Performance Validation

### 4.1 Latency Measurement

**Test**: Measure end-to-end latency for single generation

**Script**:
```python
import time
import sys
sys.path.insert(0, '/home/tt-admin/ComfyUI-tt_standalone/comfy')
from backends.tenstorrent_backend import get_backend

backend = get_backend()

# Initialize model (one-time)
print("Initializing model...")
start = time.time()
model_id = backend.init_model("sdxl")
init_time = time.time() - start
print(f"Init time: {init_time:.2f}s")

# Run generation
print("Running generation...")
start = time.time()
result = backend.full_denoise(
    model_id=model_id,
    prompt="a beautiful landscape",
    negative_prompt="blurry",
    steps=12,
    guidance_scale=7.0,
    width=1024,
    height=1024,
    seed=42
)
gen_time = time.time() - start
print(f"Generation time: {gen_time:.2f}s")

backend.unload_model(model_id)
backend.close()
```

**Target Performance** (dev mode, 12 steps):
- Init time: < 3 min
- Generation time: < 30 sec

**Success Criteria**: Meets or exceeds target performance

---

### 4.2 Memory Leak Test

**Test**: Monitor memory usage over multiple generations

**Script**:
```bash
# Terminal 1: Start bridge with memory monitoring
watch -n 1 'ps aux | grep comfyui_bridge'

# Terminal 2: Run repeated generations
for i in {1..10}; do
  echo "Generation $i..."
  # Use script from 4.1
done
```

**Success Criteria**: Memory usage stable after warmup, no continuous growth

---

### 4.3 Shared Memory Cleanup Test

**Test**: Verify shared memory segments are cleaned up

**Script**:
```bash
# Before test
ls /dev/shm | grep tt_ | wc -l

# Run 10 generations
# (use script from 4.1)

# After test
ls /dev/shm | grep tt_ | wc -l
```

**Success Criteria**: Count before == count after (no leaks)

---

## Phase 5: Quality Validation

### 5.1 SSIM Comparison Test

**Objective**: Compare output quality with reference implementation

**Prerequisites**:
- Reference image from standalone SDXL runner
- Same seed, prompt, and parameters

**Script**:
```python
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

# Load images
img_comfyui = np.array(Image.open("comfyui_output.png"))
img_reference = np.array(Image.open("reference_output.png"))

# Calculate SSIM
score = ssim(img_comfyui, img_reference, channel_axis=2, data_range=255)
print(f"SSIM Score: {score:.4f}")

if score >= 0.90:
    print("✓ PASS: Quality target met")
else:
    print("✗ FAIL: Quality below target")
```

**Success Criteria**: SSIM >= 0.90

---

### 5.2 Visual Quality Test

**Objective**: Subjective quality assessment

**Test Prompts**:
1. "a beautiful mountain landscape, highly detailed, 8k"
2. "portrait of a cat, professional photography"
3. "futuristic city at night, neon lights"

**Evaluation**:
- Correct subject matter
- No obvious artifacts
- Proper composition
- Appropriate detail level

**Success Criteria**: All images visually acceptable

---

## Phase 6: Stress Tests

### 6.1 Rapid Fire Test

**Test**: Queue 20 generations as fast as possible

**Expected**: All complete successfully, no queue failures

### 6.2 Long Running Test

**Test**: Run bridge server for 1 hour with periodic generations

**Monitor**:
- Memory usage
- CPU usage
- Socket stability
- Error rate

**Success Criteria**: No crashes, stable performance

---

## Test Execution Checklist

- [ ] Phase 1: Unit Tests (all pass)
- [ ] Phase 2: Component Integration (all pass)
- [ ] Phase 3: End-to-End (workflow succeeds)
- [ ] Phase 4: Performance (meets targets)
- [ ] Phase 5: Quality (SSIM >= 0.90)
- [ ] Phase 6: Stress (stable under load)

---

## Issue Tracking Template

When issues are found, document using:

```markdown
### Issue: [Short Description]

**Severity**: Critical | Major | Minor
**Component**: Backend | Custom Nodes | Bridge
**Reproducible**: Yes | No | Sometimes

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**:

**Actual Behavior**:

**Error Messages**:
```

---

## Sign-off

**Phase 1 (Unit Tests)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Phase 2 (Integration)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Phase 3 (E2E)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Phase 4 (Performance)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Phase 5 (Quality)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Phase 6 (Stress)**:
- Date: __________
- Tester: __________
- Result: PASS / FAIL

**Overall Go/No-Go**: ___________
