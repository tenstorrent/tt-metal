# TTNN Dispatchers

This directory contains different dispatcher implementations for handling PyTorch ATen operations with TTNN.

## Overview

The dispatcher system allows you to switch between different TTNN operation implementations at runtime, similar to how `run_config.py` handles different execution modes.

## Usage

### Basic Usage

```python
from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

# Use default dispatcher (automatically selected)
if can_dispatch_to_ttnn("aten::mul.Tensor", args, kwargs):
    result = dispatch_to_ttnn("aten::mul.Tensor", args, kwargs)
```

### Changing Dispatcher Programmatically

```python
from models.tt_symbiote.core.dispatcher import set_dispatcher

# Set at application startup
set_dispatcher("default")  # or "optimized", "debug", etc.

# List available dispatchers
from models.tt_symbiote.core.dispatcher import list_available_dispatchers
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
from models.tt_symbiote.core.dispatcher import register_dispatcher
import my_project.custom_dispatcher

# Register your dispatcher
register_dispatcher("custom", my_project.custom_dispatcher)

# Use it
from models.tt_symbiote.core.dispatcher import set_dispatcher
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
        from models.tt_symbiote.core.dispatchers import default_dispatcher
        register_dispatcher("default", default_dispatcher)
    except ImportError:
        pass

    # Add your new dispatcher
    try:
        from models.tt_symbiote.core.dispatchers import optimized_dispatcher
        register_dispatcher("optimized", optimized_dispatcher)
    except ImportError:
        pass
```

3. **Use it**:

```python
from models.tt_symbiote.core.dispatcher import set_dispatcher

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

## Best Practices

1. **Keep dispatchers focused**: Each dispatcher should have a clear purpose (e.g., performance, debugging, compatibility)
2. **Reuse helpers**: Import common helper functions from `default_dispatcher` to avoid duplication
3. **Document differences**: Clearly document how your dispatcher differs from the default
4. **Test thoroughly**: Add tests for custom dispatchers in the tests directory
5. **Handle errors gracefully**: Provide clear error messages when operations aren't supported

## Examples

### Performance-Optimized Dispatcher

```python
# optimized_dispatcher.py
from models.tt_symbiote.core.dispatchers.default_dispatcher import (
    _prepare_binary_inputs,
    _ensure_tile_layout,
    _cleanup_tensors,
)

def dispatch_to_ttnn(func_name: str, args, kwargs):
    # Use cached compute configs, fused operations, etc.
    pass
```

### Debug Dispatcher

```python
# debug_dispatcher.py
import logging

logger = logging.getLogger(__name__)

def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    logger.debug(f"Checking dispatch for {func_name}")
    # Delegate to default with logging
    from models.tt_symbiote.core.dispatchers.default_dispatcher import can_dispatch_to_ttnn as default_check
    result = default_check(func_name, args, kwargs)
    logger.debug(f"Dispatch result for {func_name}: {result}")
    return result
```

## Troubleshooting

### Dispatcher not found
```
ValueError: Unknown dispatcher 'my_dispatcher'
```
- Ensure your dispatcher is registered in `dispatcher_config.py`
- Check for import errors in your dispatcher module

### Import errors
```
ImportError: cannot import name 'my_dispatcher'
```
- Verify the file exists in `core/dispatchers/`
- Check for syntax errors in your dispatcher
- Ensure all required functions are implemented

## See Also

- `run_config.py` - Runtime configuration for execution modes
- `tensor.py` - TorchTTNNTensor wrapper implementation
- `module.py` - TTNNModule base class
