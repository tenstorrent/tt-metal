# ComfyUI Bridge Developer Guide

**Version:** 1.0.0
**Date:** 2025-12-12

---

## Table of Contents

1. [Code Structure](#code-structure)
2. [Development Setup](#development-setup)
3. [Adding New Operations](#adding-new-operations)
4. [Debugging Techniques](#debugging-techniques)
5. [Performance Profiling](#performance-profiling)
6. [Testing Guidelines](#testing-guidelines)
7. [Common Patterns](#common-patterns)
8. [Contributing](#contributing)

---

## Code Structure

```
comfyui_bridge/
├── __init__.py          # Package initialization
├── server.py            # Unix socket server implementation
├── handlers.py          # Operation handlers and TensorBridge
├── protocol.py          # Message protocol (msgpack + framing)
├── README.md            # Bridge component documentation
├── DEVELOPER_GUIDE.md   # This file
└── tests/
    ├── __init__.py
    ├── test_protocol.py     # Protocol unit tests
    ├── test_handlers.py     # Handler unit tests
    └── test_integration.py  # Integration tests
```

### Key Files

| File | Purpose |
|------|---------|
| `server.py` | Main entry point, socket handling, request routing |
| `handlers.py` | Business logic for each operation |
| `protocol.py` | Message serialization/deserialization |

---

## Development Setup

### Prerequisites

```bash
# Install development dependencies
cd /home/tt-admin/tt-metal
source python_env/bin/activate
pip install pytest pytest-cov black isort mypy
```

### Running the Bridge in Dev Mode

```bash
# Fast startup (12 warmup steps instead of 50)
./launch_comfyui_bridge.sh --dev

# With custom socket for testing
./launch_comfyui_bridge.sh --socket-path /tmp/tt-comfy-test.sock
```

### Running Tests

```bash
# Run all tests
python3 -m pytest comfyui_bridge/tests/ -v

# Run with coverage
python3 -m pytest comfyui_bridge/tests/ -v --cov=comfyui_bridge

# Run specific test file
python3 -m pytest comfyui_bridge/tests/test_protocol.py -v
```

### Code Formatting

```bash
# Format code
black comfyui_bridge/
isort comfyui_bridge/

# Type checking
mypy comfyui_bridge/
```

---

## Adding New Operations

### Step 1: Define Handler Method

In `handlers.py`, add a new method to `OperationHandler`:

```python
def handle_my_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle my custom operation.

    Input data:
        - param1: str - Description
        - param2: int - Description

    Returns:
        - result: str - Description
    """
    # Validate inputs
    param1 = data.get("param1")
    if not param1:
        raise ValueError("param1 is required")

    param2 = data.get("param2", 10)  # Default value

    # Perform operation
    logger.info(f"Executing my_operation with param1={param1}, param2={param2}")

    try:
        result = self._do_something(param1, param2)
        return {"result": result}
    except Exception as e:
        logger.error(f"my_operation failed: {e}", exc_info=True)
        raise RuntimeError(f"my_operation failed: {e}")
```

### Step 2: Register in Server

In `server.py`, add dispatch case in `_dispatch_operation`:

```python
def _dispatch_operation(self, operation: str, data: dict) -> dict:
    # ... existing operations ...

    elif operation == "my_operation":
        return self.handler.handle_my_operation(data)

    else:
        raise ValueError(f"Unknown operation: {operation}")
```

### Step 3: Add Client Method

In `tenstorrent_backend.py` (ComfyUI side):

```python
def my_operation(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Execute my custom operation.

    Args:
        param1: Description
        param2: Description (default: 10)

    Returns:
        Result dictionary
    """
    logger.info(f"Calling my_operation...")

    return self._send_receive(
        operation="my_operation",
        data={
            "param1": param1,
            "param2": param2
        }
    )
```

### Step 4: Create Custom Node (Optional)

In `nodes.py` (custom_nodes):

```python
class TT_MyOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "param1": ("STRING", {"default": ""}),
                "param2": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Tenstorrent"

    def execute(self, model, param1, param2):
        backend = model.backend
        result = backend.my_operation(param1, param2)
        return (result.get("result"),)
```

### Step 5: Add Tests

In `tests/test_handlers.py`:

```python
def test_handle_my_operation():
    """Test my_operation handler."""
    handler = OperationHandler(config=mock_config)

    result = handler.handle_my_operation({
        "param1": "test",
        "param2": 20
    })

    assert "result" in result
    assert isinstance(result["result"], str)

def test_handle_my_operation_missing_param():
    """Test my_operation with missing required param."""
    handler = OperationHandler(config=mock_config)

    with pytest.raises(ValueError, match="param1 is required"):
        handler.handle_my_operation({"param2": 10})
```

---

## Debugging Techniques

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger("comfyui_bridge").setLevel(logging.DEBUG)
```

### Log Tensor Shapes

```python
def debug_tensor(name: str, tensor: torch.Tensor):
    logger.debug(
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
        f"range=[{tensor.min():.3f}, {tensor.max():.3f}]"
    )
```

### Socket Debugging

```python
# In server.py, add request logging
def _handle_client(self, client_sock):
    request = receive_message(client_sock)
    logger.debug(f"Raw request: {request}")
    logger.debug(f"Request keys: {request.keys()}")
    logger.debug(f"Data keys: {request.get('data', {}).keys()}")
```

### Shared Memory Debugging

```python
# List all shared memory segments
import subprocess
result = subprocess.run(['ls', '-la', '/dev/shm/'], capture_output=True, text=True)
print(result.stdout)

# Check specific segment
import os
shm_path = '/dev/shm/tt_comfy_abc123'
if os.path.exists(shm_path):
    stat = os.stat(shm_path)
    print(f"Size: {stat.st_size}, Mode: {oct(stat.st_mode)}")
```

### Interactive Testing

```python
# Test client connection
from comfy.backends.tenstorrent_backend import get_backend

backend = get_backend()
print(f"Connected: {backend.is_connected()}")

# Ping server
result = backend.ping()
print(f"Ping result: {result}")

# Check model status
if result.get("model_loaded"):
    print(f"Model ID: {result.get('model_id')}")
```

---

## Performance Profiling

### Timing Operations

```python
import time

class TimedOperation:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        logger.info(f"{self.name} took {elapsed*1000:.2f}ms")

# Usage
with TimedOperation("full_denoise"):
    result = handler.handle_full_denoise(data)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# ... operations ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
tracemalloc.stop()
```

### Shared Memory Stats

```python
def get_shm_stats(tensor_bridge):
    """Get shared memory statistics."""
    total_size = 0
    for name, shm in tensor_bridge._active_segments.items():
        total_size += shm.size
    return {
        "num_segments": len(tensor_bridge._active_segments),
        "total_size_mb": total_size / 1024 / 1024
    }
```

### Inference Timing

```python
# Add to handlers.py
def handle_full_denoise(self, data):
    timings = {}

    t0 = time.perf_counter()
    # ... setup ...
    timings["setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    images = self.sdxl_runner.run_inference([request])
    timings["inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # ... postprocess ...
    timings["postprocess"] = time.perf_counter() - t0

    logger.info(f"Timings: {timings}")

    return {"images_shm": ..., "timings": timings}
```

---

## Testing Guidelines

### Unit Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestOperationHandler:
    """Tests for OperationHandler class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock SDXLConfig."""
        return Mock()

    @pytest.fixture
    def handler(self, mock_config):
        """Create handler with mock config."""
        with patch('comfyui_bridge.handlers.SDXLRunner'):
            return OperationHandler(config=mock_config)

    def test_handle_ping(self, handler):
        """Test ping operation."""
        result = handler.handle_ping({})

        assert result["status"] == "ok"
        assert "model_loaded" in result

    def test_handle_init_model_invalid_type(self, handler):
        """Test init_model with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            handler.handle_init_model({"model_type": "invalid"})
```

### Integration Test Structure

```python
import socket
import threading
from comfyui_bridge.server import ComfyUIBridgeServer
from comfyui_bridge.protocol import send_message, receive_message

class TestIntegration:
    """Integration tests for bridge server."""

    @pytest.fixture
    def server(self, tmp_path):
        """Start bridge server in background."""
        socket_path = str(tmp_path / "test.sock")
        server = ComfyUIBridgeServer(socket_path=socket_path)

        thread = threading.Thread(target=server.start)
        thread.daemon = True
        thread.start()

        # Wait for server to start
        time.sleep(0.5)

        yield socket_path

        server.running = False

    def test_ping_operation(self, server):
        """Test ping via socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(server)

        send_message(sock, {
            "operation": "ping",
            "data": {}
        })

        response = receive_message(sock)

        assert response["status"] == "success"
        assert response["data"]["status"] == "ok"

        sock.close()
```

### Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| protocol.py | 95%+ |
| handlers.py | 85%+ |
| server.py | 75%+ |

---

## Common Patterns

### Error Handling Pattern

```python
def handle_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Standard error handling pattern."""
    try:
        # Validate inputs
        required_param = data.get("required_param")
        if required_param is None:
            raise ValueError("required_param is required")

        # Perform operation
        result = self._do_operation(required_param)

        return {"result": result}

    except ValueError as e:
        # Client error - don't log stack trace
        logger.warning(f"Invalid input: {e}")
        raise
    except Exception as e:
        # Server error - log full stack trace
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise RuntimeError(f"Operation failed: {e}")
```

### Cleanup Pattern

```python
class ResourceHandler:
    """Pattern for resource management."""

    def __init__(self):
        self.resources = []

    def acquire_resource(self):
        resource = create_resource()
        self.resources.append(resource)
        return resource

    def cleanup(self):
        """Clean up all resources."""
        for resource in self.resources:
            try:
                resource.close()
            except Exception as e:
                logger.warning(f"Failed to close resource: {e}")
        self.resources.clear()

    def __del__(self):
        self.cleanup()
```

### Thread-Safe Singleton Pattern

```python
import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

---

## Contributing

### Code Style

- Follow PEP 8
- Use type hints for all public methods
- Write docstrings for all classes and methods
- Keep functions under 50 lines

### Commit Messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(handlers): add encode_prompt operation

Adds new operation for text prompt encoding without full inference.
Useful for debugging and prompt engineering workflows.

Closes #123
```

### Pull Request Checklist

- [ ] Tests pass (`pytest comfyui_bridge/tests/`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] README updated (if needed)
- [ ] DEVELOPER_GUIDE updated (if needed)

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-12
**Maintainer:** Tenstorrent AI ULC
