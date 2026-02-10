"""
Debug logger for tracking TTNN operations and tensor properties
"""
from loguru import logger


def log_tensor_props(name, tensor, prefix="INFRA:"):
    """Log detailed tensor properties"""
    if tensor is None:
        logger.info(f"{prefix}[TENSOR] {name}: None")
        return

    props = {}
    if hasattr(tensor, "shape"):
        props["shape"] = str(tensor.shape)
    if hasattr(tensor, "memory_config"):
        props["memory"] = str(tensor.memory_config())
    if hasattr(tensor, "layout"):
        props["layout"] = str(tensor.layout)
    if hasattr(tensor, "dtype"):
        props["dtype"] = str(tensor.dtype)

    logger.info(f"{prefix}[TENSOR] {name}: {props}")
    return props


def log_op(op_name, inputs=None, config=None, output=None, prefix="INFRA:"):
    """Log a TTNN operation with inputs, config, and output"""
    logger.info(f"{prefix}[OP_START] {op_name}")

    if inputs:
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                log_tensor_props(f"  input.{key}", val, prefix)
        elif isinstance(inputs, (list, tuple)):
            for i, val in enumerate(inputs):
                log_tensor_props(f"  input[{i}]", val, prefix)
        else:
            log_tensor_props("  input", inputs, prefix)

    if config:
        logger.info(f"{prefix}  config: {config}")

    if output is not None:
        log_tensor_props("  output", output, prefix)

    logger.info(f"{prefix}[OP_END] {op_name}")
    return output
