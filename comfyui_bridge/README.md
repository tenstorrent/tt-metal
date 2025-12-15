# ComfyUI Bridge Server

Unix socket bridge between ComfyUI frontend and Tenstorrent tt-metal backend.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ComfyUI Frontend                        │
│  (Custom Nodes: TT SDXL Sampler, TT Text Encoder, etc.)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Unix Socket (/tmp/tt-comfy.sock)
                      │ Protocol: msgpack over length-prefixed binary
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              ComfyUI Bridge Server                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Protocol Layer (protocol.py)                        │  │
│  │  - receive_message() / send_message()                │  │
│  │  - Length-prefixed msgpack encoding                  │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Operation Handlers (handlers.py)                    │  │
│  │  - handle_init_model()                               │  │
│  │  - handle_full_denoise()                             │  │
│  │  - handle_ping() / handle_unload_model()             │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Tensor Bridge (shared memory)                       │  │
│  │  - Zero-copy tensor transfer via shm                 │  │
│  │  - tensor_to_shm() / tensor_from_shm()               │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  SDXL Runner (sdxl_runner.py)                        │  │
│  │  - Device initialization                             │  │
│  │  - Model loading and warmup                          │  │
│  │  - Inference execution                               │  │
│  └──────────────────────┬───────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          │ ttnn API
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              Tenstorrent Hardware (WH/T3K)                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Launch Bridge Server

```bash
# Standard mode
./launch_comfyui_bridge.sh

# Dev mode (fast startup, single worker)
./launch_comfyui_bridge.sh --dev

# Custom socket path
./launch_comfyui_bridge.sh --socket-path /tmp/my-bridge.sock

# Specific device
./launch_comfyui_bridge.sh --device-id 1
```

### 2. Launch ComfyUI

In a separate terminal:

```bash
cd /home/tt-admin/ComfyUI-tt
python main.py --listen 0.0.0.0 --port 8188
```

### 3. Use Tenstorrent Nodes in ComfyUI

In the ComfyUI web interface:
1. Add node: "TT SDXL Sampler"
2. Connect to standard CLIP Text Encode nodes
3. Generate image

## Components

### protocol.py

Implements length-prefixed msgpack protocol for Unix socket communication.

**Functions:**
- `receive_message(sock)` - Receive and deserialize msgpack message
- `send_message(sock, data)` - Serialize and send msgpack message
- `send_error(sock, error_msg)` - Send error response
- `send_success(sock, data)` - Send success response

**Protocol Format:**
```
[4 bytes: length (big-endian uint32)]
[N bytes: msgpack-encoded data]
```

### handlers.py

Operation handlers that interface with SDXLRunner.

**Classes:**
- `TensorBridge` - Shared memory tensor transfer
  - `tensor_to_shm(tensor)` → handle (dict)
  - `tensor_from_shm(handle)` → tensor
  - `cleanup_segment(shm_name)`
  - `cleanup_all()`

- `OperationHandler` - Main operation dispatcher
  - `handle_init_model(data)` → {model_id, status}
  - `handle_full_denoise(data)` → {denoised_latent}
  - `handle_ping(data)` → {status, model_loaded}
  - `handle_unload_model(data)` → {status}

### server.py

Unix socket server that accepts connections and dispatches operations.

**Classes:**
- `ComfyUIBridgeServer` - Main server class
  - `start()` - Start server and accept connections
  - `_handle_client(sock)` - Handle single client connection
  - `_dispatch_operation(op, data)` - Route operation to handler

**CLI:**
```bash
python -m comfyui_bridge.server \
    --socket-path /tmp/tt-comfy.sock \
    --device-id 0 \
    [--dev]
```

## API Reference

### Operations

#### ping

Health check operation.

**Request:**
```python
{
    "operation": "ping",
    "data": {}
}
```

**Response:**
```python
{
    "status": "success",
    "data": {
        "status": "ok",
        "model_loaded": bool,
        "model_id": str | None
    }
}
```

#### init_model

Initialize SDXL model on Tenstorrent hardware.

**Request:**
```python
{
    "operation": "init_model",
    "data": {
        "model_type": "sdxl",
        "config": {},  # Optional config overrides
        "device_id": "0"
    }
}
```

**Response:**
```python
{
    "status": "success",
    "data": {
        "model_id": "sdxl_a3f7b9c1",
        "status": "ready"
    }
}
```

#### full_denoise

Run complete denoising loop with pre-computed embeddings.

**Request:**
```python
{
    "operation": "full_denoise",
    "data": {
        "model_id": "sdxl_a3f7b9c1",

        # Tensors (via shared memory)
        "latent": {
            "shm_name": "tt_comfy_abc123",
            "shape": [1, 4, 128, 128],
            "dtype": "torch.float32",
            "size_bytes": 262144
        },
        "positive_conditioning": {
            "shm_name": "tt_comfy_def456",
            "shape": [1, 77, 2048],
            "dtype": "torch.float32",
            "size_bytes": 629760
        },
        "negative_conditioning": {...},
        "positive_text_embeds": {...},  # Pooled embeddings
        "negative_text_embeds": {...},

        # Inference parameters
        "time_ids": [1024, 1024, 0, 0, 1024, 1024],
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "guidance_rescale": 0.0,
        "seed": 42
    }
}
```

**Response:**
```python
{
    "status": "success",
    "data": {
        "denoised_latent": {
            "shm_name": "tt_bridge_xyz789",
            "shape": [1, 4, 128, 128],
            "dtype": "torch.float32",
            "size_bytes": 262144
        }
    }
}
```

#### unload_model

Unload model and free resources.

**Request:**
```python
{
    "operation": "unload_model",
    "data": {
        "model_id": "sdxl_a3f7b9c1"
    }
}
```

**Response:**
```python
{
    "status": "success",
    "data": {
        "status": "unloaded"
    }
}
```

## Shared Memory Protocol

### Overview

Tensors are transferred via POSIX shared memory for zero-copy performance.

**Workflow:**
1. Client creates shared memory segment
2. Client writes tensor data to segment
3. Client sends metadata (shm_name, shape, dtype) via msgpack
4. Server attaches to segment and reads tensor
5. Server unlinks segment after reading

### Shared Memory Handle Format

```python
{
    "shm_name": "tt_comfy_abc123",  # Unique segment name
    "shape": [1, 4, 128, 128],      # Tensor shape
    "dtype": "torch.float32",        # Tensor dtype (as string)
    "size_bytes": 262144             # Total size in bytes
}
```

### Example: Client-side tensor_to_shm

```python
import torch
import numpy as np
from multiprocessing import shared_memory
import uuid

def tensor_to_shm(tensor):
    # Ensure CPU and contiguous
    tensor = tensor.cpu().contiguous()
    np_array = tensor.numpy()

    # Create shared memory
    shm_name = f"tt_comfy_{uuid.uuid4().hex[:16]}"
    shm = shared_memory.SharedMemory(
        create=True,
        size=np_array.nbytes,
        name=shm_name
    )

    # Write data
    shm_array = np.ndarray(
        shape=np_array.shape,
        dtype=np_array.dtype,
        buffer=shm.buf
    )
    shm_array[:] = np_array[:]

    # Return handle
    return {
        "shm_name": shm_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "size_bytes": np_array.nbytes
    }
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TT_COMFY_SOCKET` | Unix socket path | `/tmp/tt-comfy.sock` |
| `SDXL_DEV_MODE` | Enable dev mode | `false` |
| `TT_VISIBLE_DEVICES` | Device IDs (comma-separated) | `0` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |

### SDXL Configuration

The bridge server uses `SDXLConfig` from `/home/tt-admin/tt-metal/sdxl_config.py`.

Key settings:
- `num_inference_steps` - Default: 50
- `guidance_scale` - Default: 5.0
- `device_mesh_shape` - Default: (1, 1) for WH, (1, 4) for T3K
- `encoders_on_device` - Run text encoders on device
- `capture_trace` - Enable trace mode for performance

## Troubleshooting

### "Connection refused" error

**Cause:** Bridge server not running or wrong socket path.

**Fix:**
```bash
# Check if server is running
ps aux | grep comfyui_bridge

# Launch server
./launch_comfyui_bridge.sh

# Or specify socket path
./launch_comfyui_bridge.sh --socket-path /tmp/my-bridge.sock
```

### "Failed to initialize device" error

**Cause:** Device already in use or unavailable.

**Fix:**
```bash
# Check device availability
python3 -c "import ttnn; print(ttnn.get_num_devices())"

# Use different device
./launch_comfyui_bridge.sh --device-id 1
```

### "Shared memory read failed" error

**Cause:** Client unlinked shared memory before server read it.

**Fix:** Ensure client keeps shared memory alive until server responds.

### Performance issues

**Dev mode for testing:**
```bash
# Fast startup (12 inference steps for warmup)
./launch_comfyui_bridge.sh --dev
```

**Trace mode for production:**
Set in `sdxl_config.py`:
```python
capture_trace = True  # Enable trace mode
```

## Development

### Running Tests

```bash
# Import test
python3 -c "from comfyui_bridge.server import ComfyUIBridgeServer; print('✓ OK')"

# Protocol test
python3 -c "from comfyui_bridge.protocol import receive_message, send_message; print('✓ OK')"

# Handler test
python3 -c "from comfyui_bridge.handlers import OperationHandler; print('✓ OK')"
```

### Adding New Operations

1. Add handler method to `OperationHandler` in `handlers.py`:
```python
def handle_my_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # Implementation
    return {"result": "success"}
```

2. Add dispatcher case in `server.py`:
```python
elif operation == "my_operation":
    return self.handler.handle_my_operation(data)
```

3. Document in README API Reference.

### Logging

Logs are written to stdout with the following levels:
- `INFO` - Server lifecycle, operations
- `DEBUG` - Protocol details, tensor transfers
- `WARNING` - Non-fatal issues
- `ERROR` - Operation failures

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with ComfyUI

### Client Reference Implementation

See `/home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py` for client implementation.

**Example client usage:**
```python
from comfy.backends.tenstorrent_backend import TenstorrentBackend

# Connect to bridge
backend = TenstorrentBackend(socket_path="/tmp/tt-comfy.sock")

# Initialize model
model_id = backend.init_model(model_type="sdxl")

# Ping server
status = backend.ping()
print(f"Server status: {status}")

# Cleanup
backend.unload_model(model_id)
backend.close()
```

### Custom Node Integration

ComfyUI custom nodes use the backend via:
```python
from comfy.backends.tenstorrent_backend import get_backend

backend = get_backend()
model_id = backend.init_model("sdxl")

# Use model...
```

## Performance

### Startup Time

| Mode | Workers | Warmup Steps | Startup Time |
|------|---------|--------------|--------------|
| Dev | 1 | 12 | ~2-3 min |
| Standard | 1 | 50 | ~5-6 min |

### Inference Throughput

| Steps | Guidance | Time per Image |
|-------|----------|----------------|
| 12 | 5.0 | ~2-3 sec |
| 25 | 5.0 | ~4-5 sec |
| 50 | 5.0 | ~8-10 sec |

Note: Timings on Wormhole (single device). T3K provides higher throughput with parallel workers.

## License

Apache-2.0

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
