"""Debug dispatcher with verbose logging for TTNN operations.

This dispatcher wraps the default dispatcher and adds detailed logging
for debugging purposes. Useful for understanding dispatch behavior.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        True if the operation can be dispatched to TTNN, False otherwise
    """
    from models.tt_symbiote.core.dispatchers.default_dispatcher import can_dispatch_to_ttnn as default_can_dispatch

    logger.debug(f"Checking dispatch for operation: {func_name}")

    if args:
        logger.debug(f"  Args types: {[type(arg).__name__ for arg in args]}")
    if kwargs:
        logger.debug(f"  Kwargs: {list(kwargs.keys())}")

    result = default_can_dispatch(func_name, args, kwargs)

    logger.debug(f"  Dispatch decision for {func_name}: {result}")

    return result


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        Result of the TTNN operation (typically a TorchTTNNTensor)
    """
    from models.tt_symbiote.core.dispatchers.default_dispatcher import dispatch_to_ttnn as default_dispatch

    logger.debug(f"Dispatching operation: {func_name}")

    try:
        result = default_dispatch(func_name, args, kwargs)
        logger.debug(f"  Successfully dispatched {func_name}")
        if hasattr(result, "shape"):
            logger.debug(f"  Result shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"  Failed to dispatch {func_name}: {e}")
        raise
