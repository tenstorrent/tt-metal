# TTNN Dispatchers

This directory contains different dispatcher implementations for handling PyTorch ATen operations with TTNN.

## Overview

The dispatcher system allows you to switch between different TTNN operation implementations at runtime.

## Usage

### Basic Usage

```python
from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

# Use default dispatcher (automatically selected)
if can_dispatch_to_ttnn("aten::mul.Tensor", args, kwargs):
    result = dispatch_to_ttnn("aten::mul.Tensor", args, kwargs)
```

### Changing Dispatcher Programmatically

```python
from models.experimental.tt_symbiote.core.dispatcher import set_dispatcher

# Set at application startup
set_dispatcher("default")  # or "optimized", "debug", etc.

# List available dispatchers
from models.experimental.tt_symbiote.core.dispatcher import list_available_dispatchers
print(list_available_dispatchers())
```

### Environment Variable

Set the dispatcher via environment variable (takes precedence over programmatic setting):

```bash
export TT_SYMBIOTE_DISPATCHER=default
pytest tests/test_vit.py

# Or inline
TT_SYMBIOTE_DISPATCHER=default pytest tests/test_vit.py
```

## Available Dispatchers

### default
Standard TTNN dispatcher with all supported operations. This includes:
- Binary operations (add, mul, sub, div, etc.) with helper functions for reduced code duplication
- Matrix operations (bmm, matmul, addmm)
- Activation functions (relu, gelu, silu, sigmoid, softmax)
- Tensor manipulation (view, transpose, slice, cat, stack, split)
- Comparison operations (ge, gt, eq, lt)
- And more...

## Creating a New Dispatcher

There are two ways to create a custom dispatcher:

### Option 1: Manual Registration (Recommended for External Dispatchers)

Create a dispatcher module anywhere in your project and register it manually:

1. **Create your dispatcher module** (e.g., `my_project/custom_dispatcher.py`):

```python
"""My custom TTNN dispatcher."""

def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        True if dispatchable, False otherwise
    """
    # Your custom dispatch logic here
    pass


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation to TTNN handler.

    Args:
        func_name: ATen operation name
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Result of the TTNN operation
    """
    # Your custom handler logic here
    pass
```

2. **Register it** in your application code:

```python
from models.experimental.tt_symbiote.core.dispatcher import register_dispatcher
import my_project.custom_dispatcher

# Register your dispatcher
register_dispatcher("custom", my_project.custom_dispatcher)

# Use it
from models.experimental.tt_symbiote.core.dispatcher import set_dispatcher
set_dispatcher("custom")
```

### Option 2: Auto-Registration (For Built-in Dispatchers)

To create a custom dispatcher that auto-registers with the framework:

1. **Create a new file** in this directory (e.g., `optimized_dispatcher.py`):

```python
"""Optimized TTNN dispatcher with performance enhancements."""

from typing import Any

def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        True if dispatchable, False otherwise
    """
    # Your custom dispatch logic here
    pass


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation to TTNN handler.

    Args:
        func_name: ATen operation name
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Result of the TTNN operation
    """
    # Your custom handler logic here
    pass
```

2. **Register it** in `dispatcher_config.py`:

```python
def _auto_register_dispatchers():
    """Automatically register available dispatcher implementations."""
    try:
        from models.experimental.tt_symbiote.core.dispatchers import default_dispatcher
        register_dispatcher("default", default_dispatcher)
    except ImportError:
        pass

    # Add your new dispatcher
    try:
        from models.experimental.tt_symbiote.core.dispatchers import optimized_dispatcher
        register_dispatcher("optimized", optimized_dispatcher)
    except ImportError:
        pass
```

3. **Use it**:

```python
from models.experimental.tt_symbiote.core.dispatcher import set_dispatcher

set_dispatcher("optimized")
# or via environment: export TT_SYMBIOTE_DISPATCHER=optimized
```

## Dispatcher Requirements

Each dispatcher module must implement:

- `can_dispatch_to_ttnn(func_name: str, args, kwargs) -> bool`
  - Determines if an operation can be handled by TTNN
  - Should check tensor types, dtypes, and operation support

- `dispatch_to_ttnn(func_name: str, args, kwargs) -> Any`
  - Executes the TTNN operation
  - Returns a `TorchTTNNTensor` or appropriate result

## Architecture

```
core/
├── dispatcher.py              # Public API (thin wrapper)
├── dispatchers/
│   ├── __init__.py           # Exports config functions
│   ├── dispatcher_config.py  # Registry and configuration
│   ├── default_dispatcher.py # Standard implementation
│   └── README.md             # This file
```
