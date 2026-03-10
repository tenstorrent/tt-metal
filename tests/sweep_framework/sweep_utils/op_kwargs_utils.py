# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for extracting and parsing op kwargs from sweep test vectors.

The V2 master JSON cleanly separates:
  - Positional tensor args (arg0, arg1, ...) → become input_a_*, input_b_*, etc.
  - Named kwargs → kept with original names (memory_config, program_config, dim, etc.)
  - Named tensor kwargs → expanded to {name}_shape, {name}_dtype, etc.

When the sweep framework calls run(**test_vector, device=device), ALL of these
land in the run() function. This module helps sweep tests extract only the actual
op kwargs and parse dict values into ttnn objects, so they can do:

    op_kwargs = build_op_kwargs(kwargs)
    output = ttnn.some_op(input_tensor, **op_kwargs)
"""

from typing import Dict, Any, Optional, Set
from tests.sweep_framework.master_config_loader_v2 import (
    dict_to_memory_config,
    dict_to_compute_kernel_config,
    dict_to_program_config,
    dict_to_core_grid,
    parse_dtype,
    parse_layout,
)

# Keys added by the sweep infrastructure — never pass to ops
_INFRA_KEYS = frozenset(
    {
        "traced_source",
        "traced_machine_info",
        "config_hash",
        "suite_name",
        "validity",
        "invalid_reason",
        "status",
        "sweep_name",
        "storage_type",
        "mesh_coords",
    }
)

# Keys for positional tensor inputs — handled by sweep test tensor creation code
_TENSOR_SUFFIXES = ("_shape", "_dtype", "_layout", "_memory_config", "_tensor_placement", "_activations")
_TENSOR_PREFIXES = (
    "input_a",
    "input_b",
    "input_c",
    "input_d",
    "input_e",
    "input_f",
    # Alternative naming convention from traced JSON (input_tensor, input_tensor_a, etc.)
    "input_tensor",
    "input_tensor_a",
    "input_tensor_b",
    "input_tensor_c",
    "input_tensor_d",
    # Named tensor kwargs from specific ops
    "input_tensor_q",
    "input_tensor_k",
    "input_tensor_v",
    "page_table_tensor",
)


def _is_infrastructure_key(key: str) -> bool:
    """Check if a key is infrastructure/metadata that should not be passed to ops."""
    if key in _INFRA_KEYS:
        return True
    # Positional tensor params: input_a_shape, input_b_dtype, etc.
    for prefix in _TENSOR_PREFIXES:
        if key.startswith(prefix) and any(key.endswith(suffix) for suffix in _TENSOR_SUFFIXES):
            return True
    # Positional arg keys (arg0, arg1, ...) from V2 JSON — these are positional
    # parameters already handled by the sweep test's tensor creation code
    if key.startswith("arg") and key[3:].isdigit():
        return True
    # output_memory_config is handled separately by most sweep tests
    if key == "output_memory_config":
        return True
    # Any key ending with _tensor_placement is tensor placement metadata
    if key.endswith("_tensor_placement"):
        return True
    return False


def _is_named_tensor_kwarg(key: str, all_keys: Set[str]) -> bool:
    """Check if a key is part of a named tensor kwarg (e.g., weight_shape, bias_dtype).

    Named tensor kwargs come from JSON entries like:
        "weight": {type: "ttnn.Tensor", ...}
    which the loader expands to weight_shape, weight_dtype, weight_layout,
    weight_memory_config, weight_tensor_placement.

    These need special handling (create tensor from components), not direct op passing.
    """
    for suffix in _TENSOR_SUFFIXES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            # It's a named tensor kwarg if any other suffix with same base exists
            if base and any(f"{base}{s}" in all_keys for s in _TENSOR_SUFFIXES if s != suffix):
                return True
    return False


def _is_memory_config_dict(value: Any) -> bool:
    """Check if a value looks like a memory config dict."""
    return isinstance(value, dict) and ("memory_layout" in value or "buffer_type" in value)


def _is_compute_kernel_config_dict(value: Any) -> bool:
    """Check if a value looks like a compute kernel config dict."""
    return isinstance(value, dict) and "math_fidelity" in value


def _is_program_config_dict(value: Any) -> bool:
    """Check if a value looks like a program config dict."""
    if not isinstance(value, dict):
        return False
    type_str = str(value.get("type", ""))
    if "Config" in type_str and type_str != "CoreGrid":
        return True
    # Matmul program config without explicit type (legacy)
    if {"in0_block_w", "per_core_M", "per_core_N"}.issubset(value.keys()):
        return True
    return False


def _is_core_grid_dict(value: Any) -> bool:
    """Check if a value looks like a core_grid dict."""
    if not isinstance(value, dict):
        return False
    return value.get("type") in ("CoreGrid", "CoreCoord")


def _is_dtype_dict(value: Any) -> bool:
    """Check if a value looks like a dtype dict {type: 'DataType', repr: 'DataType.BFLOAT16'}."""
    return isinstance(value, dict) and value.get("type") == "DataType"


def _is_layout_dict(value: Any) -> bool:
    """Check if a value looks like a layout dict {type: 'Layout', repr: 'Layout.TILE'}."""
    return isinstance(value, dict) and value.get("type") == "Layout"


def parse_dict_value(key: str, value: Any) -> Any:
    """Parse a dict value into the appropriate ttnn object based on its structure.

    Returns the parsed value, or None if parsing fails.
    """
    if not isinstance(value, dict):
        return value

    try:
        if _is_memory_config_dict(value):
            return dict_to_memory_config(value)
        elif _is_compute_kernel_config_dict(value):
            return dict_to_compute_kernel_config(value)
        elif _is_program_config_dict(value):
            return dict_to_program_config(value)
        elif _is_core_grid_dict(value):
            return dict_to_core_grid(value)
        elif _is_dtype_dict(value):
            return parse_dtype(value.get("repr", ""))
        elif _is_layout_dict(value):
            return parse_layout(value.get("repr", ""))
    except (ValueError, TypeError, KeyError):
        pass

    # Return as-is if we can't parse it (sweep test can handle it)
    return value


def build_op_kwargs(
    kwargs: Dict[str, Any],
    *,
    exclude: Optional[Set[str]] = None,
    include_only: Optional[Set[str]] = None,
    output_memory_config: Any = None,
) -> Dict[str, Any]:
    """Extract actual op kwargs from the full test vector kwargs.

    Filters out infrastructure keys, positional tensor params, named tensor kwargs,
    output_memory_config, and ``__ABSENT__`` sentinel values.  Parses dict values
    into ttnn objects.

    The ``output_memory_config`` parameter (typically from the run() function
    signature) is used as a **fallback** for ``memory_config`` in the returned
    dict.  If the traced config already provides ``memory_config`` via kwargs it
    takes precedence; otherwise ``output_memory_config`` is used.

    Args:
        kwargs: The **kwargs from the run() function (everything not in named params)
        exclude: Additional keys to exclude (op-specific, e.g., keys already handled)
        include_only: If set, only include these specific keys (overrides default filtering)
        output_memory_config: Fallback value for ``memory_config`` when not present
            in kwargs.  Pass the ``output_memory_config`` parameter from run().

    Returns:
        Dict of parsed op kwargs ready to pass as **op_kwargs to the ttnn op.
        None values and ``__ABSENT__`` sentinels are excluded.
    """
    all_keys = set(kwargs.keys())
    exclude = exclude or set()
    op_kwargs = {}

    for key, value in kwargs.items():
        # Skip None values
        if value is None:
            continue

        # Skip __ABSENT__ sentinel values (parameter not present in traced config)
        if value == "__ABSENT__":
            continue

        # If include_only is specified, only include those keys
        if include_only is not None:
            if key not in include_only:
                continue
        else:
            # Default filtering
            if _is_infrastructure_key(key):
                continue
            if _is_named_tensor_kwarg(key, all_keys):
                continue
            if key in exclude:
                continue

        # Parse dict values into ttnn objects
        parsed = parse_dict_value(key, value)
        if parsed is not None:
            op_kwargs[key] = parsed

    # Use output_memory_config as fallback for memory_config
    if "memory_config" not in op_kwargs and output_memory_config is not None:
        parsed_mem = parse_dict_value("memory_config", output_memory_config)
        if parsed_mem is not None:
            op_kwargs["memory_config"] = parsed_mem

    return op_kwargs


def extract_named_tensor_kwargs(kwargs: Dict[str, Any], tensor_name: str) -> Optional[Dict[str, Any]]:
    """Extract a named tensor kwarg's components from kwargs.

    For a tensor named "weight", extracts:
        weight_shape, weight_dtype, weight_layout, weight_memory_config, weight_tensor_placement

    Args:
        kwargs: The **kwargs dict
        tensor_name: Base name of the tensor (e.g., "weight", "bias", "mask")

    Returns:
        Dict with keys: shape, dtype, layout, memory_config, tensor_placement
        or None if the tensor is not present.
    """
    shape = kwargs.get(f"{tensor_name}_shape")
    if shape is None:
        return None

    return {
        "shape": shape,
        "dtype": kwargs.get(f"{tensor_name}_dtype"),
        "layout": kwargs.get(f"{tensor_name}_layout"),
        "memory_config": kwargs.get(f"{tensor_name}_memory_config"),
        "tensor_placement": kwargs.get(f"{tensor_name}_tensor_placement"),
    }
