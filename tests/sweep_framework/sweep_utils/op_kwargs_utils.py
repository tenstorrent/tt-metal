# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        "timestamp",
        "input_hash",
        "tag",
        "__absent_keys__",
    }
)

# Keys for positional tensor inputs — handled by sweep test tensor creation code
_TENSOR_SUFFIXES = ("_shape", "_dtype", "_layout", "_memory_config", "_tensor_placement")
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
    "cur_pos_tensor",
    "attn_mask",
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
    # memory_config from traced kwargs should not leak into op_kwargs.
    # It is handled via the output_memory_config parameter in sweep module run() functions.
    # Passing it through causes "incompatible function arguments" for ops that don't accept it.
    if key == "memory_config":
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
    """Check if a value looks like a memory config dict.

    Handles the flat form ({"memory_layout": ..., "buffer_type": ...}) and the
    serialized wrapper form ({"type": "...MemoryConfig", "data": {"memory_layout":
    ...}}). Without the wrapper case, a sharded input_*_memory_config serialized
    in wrapper form falls through parse_dict_value and is dropped to None, then
    silently degrades to DRAM-interleaved (memory_config mismatch vs master)."""
    if not isinstance(value, dict):
        return False
    if "memory_layout" in value or "buffer_type" in value:
        return True
    if "MemoryConfig" in str(value.get("type", "")):
        return True
    data = value.get("data")
    return isinstance(data, dict) and ("memory_layout" in data or "buffer_type" in data)


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


def _is_core_range_set_dict(value: Any) -> bool:
    """Check if a value looks like a CoreRangeSet dict."""
    return isinstance(value, dict) and value.get("type") == "CoreRangeSet"


def _parse_core_range_set(value: Any) -> Any:
    """Parse a CoreRangeSet dict into a ttnn.CoreRangeSet.

    Handles C++ repr format: {"type": "CoreRangeSet", "value": "{[0-0 - 7-7]}"}
    where each CoreRange is rendered as [x1-y1 - x2-y2] by CoreRange::str().
    Also handles structured data format with "data" key.
    """
    import re
    import json as _json
    import ttnn  # required: this helper builds ttnn.CoreRange/CoreCoord/CoreRangeSet

    data = value.get("data")
    if data is not None:
        try:
            if isinstance(data, str):
                data = _json.loads(data)
            if isinstance(data, list):
                core_ranges = set()
                for rd in data:
                    start = rd["start"]
                    end = rd["end"]
                    core_ranges.add(
                        ttnn.CoreRange(
                            ttnn.CoreCoord(start["x"], start["y"]),
                            ttnn.CoreCoord(end["x"], end["y"]),
                        )
                    )
                if core_ranges:
                    return ttnn.CoreRangeSet(core_ranges)
        except (KeyError, TypeError, ValueError):
            # Malformed/partial structured "data" — fall through to the
            # regex-based repr parser below rather than failing the vector.
            pass

    repr_str = str(value.get("value", value.get("repr", "")))
    core_ranges = set()
    for m in re.finditer(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]", repr_str):
        x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        core_ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2)))
    if core_ranges:
        return ttnn.CoreRangeSet(core_ranges)
    return None


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


def _is_unary_op_dict(value: Any) -> bool:
    """Check if a value looks like a UnaryOpType dict {type: 'UnaryOpType', repr: 'UnaryOpType.SILU'}."""
    return isinstance(value, dict) and value.get("type") == "UnaryOpType"


def _parse_unary_op(value: Any) -> Any:
    """Parse a UnaryOpType dict to a ttnn.UnaryOpType enum value."""
    if not _is_unary_op_dict(value):
        return value
    repr_str = str(value.get("repr", ""))
    # Format: "UnaryOpType.SILU" → SILU
    name = repr_str.split(".", 1)[-1] if "." in repr_str else repr_str
    try:
        import ttnn as _ttnn

        return getattr(_ttnn.UnaryOpType, name)
    except (ImportError, AttributeError):
        return value


def _is_unary_with_param_dict(value: Any) -> bool:
    """Check for a UnaryWithParam dict {type: 'UnaryWithParam', value: 'UnaryWithParam(op_type=UnaryOpType::SILU, ...)'}."""
    return isinstance(value, dict) and value.get("type") == "UnaryWithParam"


def _parse_unary_with_param(value: Any) -> Any:
    """Parse a UnaryWithParam dict to a ttnn.UnaryWithParam (the form eltwise
    ops' ``activations`` kwarg expects). Format:
    ``UnaryWithParam(op_type=UnaryOpType::SILU)`` or with ``param=<float>``."""
    if not _is_unary_with_param_dict(value):
        return value
    import re as _re

    s = str(value.get("value", ""))
    m = _re.search(r"UnaryOpType::(\w+)", s)
    if not m:
        return value
    try:
        import ttnn as _ttnn

        op = getattr(_ttnn.UnaryOpType, m.group(1))
        pm = _re.search(r"param\s*[=:]\s*(-?\d+\.?\d*(?:[eE]-?\d+)?)", s)
        if pm:
            return _ttnn.UnaryWithParam(op, float(pm.group(1)))
        return _ttnn.UnaryWithParam(op)
    except (ImportError, AttributeError, ValueError):
        return value


def _maybe_parse_unary_list(value: Any) -> Any:
    """If value is a list of UnaryOpType or UnaryWithParam dicts, convert each."""
    if isinstance(value, list) and value and all(_is_unary_op_dict(v) for v in value):
        parsed = [_parse_unary_op(v) for v in value]
        if all(not isinstance(p, dict) for p in parsed):
            return parsed
    # Fused-activation lists carry UnaryWithParam dicts (e.g. add/mul activations).
    if isinstance(value, list) and value and all(_is_unary_with_param_dict(v) for v in value):
        parsed = [_parse_unary_with_param(v) for v in value]
        if all(not isinstance(p, dict) for p in parsed):
            return parsed
    return value


def _create_output_tensor(descriptor: dict, device, input_shape=None) -> Any:
    """Create a preallocated output tensor from a traced output_tensor descriptor.

    The descriptor has: original_shape, original_dtype, layout, memory_config.
    We create an empty tensor on device with matching specs so the op trace
    records the same output_tensor argument as the model.
    """
    import ttnn
    import torch

    shape = descriptor.get("original_shape")
    if not shape:
        shape = input_shape
    if not shape:
        return None

    if isinstance(shape, dict) and "value" in shape:
        import re

        m = re.match(r"Shape\(\[([0-9,\s]+)\]\)", str(shape["value"]))
        if m:
            shape = [int(x.strip()) for x in m.group(1).split(",")]
    if isinstance(shape, str):
        import ast

        try:
            shape = ast.literal_eval(shape)
        except Exception:
            return None

    dtype_str = descriptor.get("original_dtype", "DataType.BFLOAT16")
    dtype = parse_dtype(dtype_str) if isinstance(dtype_str, str) else None
    if dtype is None:
        dtype = ttnn.bfloat16

    layout_str = descriptor.get("layout", "Layout.TILE")
    layout = parse_layout(layout_str) if isinstance(layout_str, str) else None
    if layout is None:
        layout = ttnn.TILE_LAYOUT

    mc_dict = descriptor.get("memory_config")
    memory_config = dict_to_memory_config(mc_dict) if isinstance(mc_dict, dict) else ttnn.DRAM_MEMORY_CONFIG

    try:
        # Preallocated output buffer is overwritten by the op; use empty() to
        # skip a host-side memset of potentially large tensors.
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    except Exception:
        return None


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
        elif _is_core_range_set_dict(value):
            return _parse_core_range_set(value)
        elif value.get("type") == "Shape":
            import re as _shape_re

            m = _shape_re.search(r"Shape\(\[(.*?)\]\)", str(value.get("value", "")))
            if m:
                return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except (ValueError, TypeError, KeyError):
        # malformed structured Shape data — fall through to regex repr parsing below
        pass

    # Dicts with a "type" key that we couldn't parse into a ttnn object
    # must NOT be passed to C++ bindings — they'll cause "incompatible
    # function arguments". Return None so build_op_kwargs drops them.
    if isinstance(value, dict) and "type" in value:
        return None

    return value


def extract_positional_args(kwargs: Dict[str, Any]) -> Dict[int, Any]:
    """Extract non-tensor positional args (arg0, arg1, …) from the run() kwargs.

    The V2 config loader stores non-tensor positional arguments under the keys
    ``arg0``, ``arg1``, ``arg2``, etc.  ``build_op_kwargs`` intentionally filters
    these out (they are positional, not keyword arguments to the ttnn op).

    Sweep tests should call this helper **before** the op invocation to retrieve
    positional args such as ``scale``, ``cache_idx``, ``output_dtype``, etc.

    Example::

        pos_args = extract_positional_args(kwargs)
        scale = float(pos_args.get(1, 1.0))        # arg1 = scale
        output_dtype = pos_args.get(2, ttnn.float32)  # arg2 = dtype

    Args:
        kwargs: The ``**kwargs`` dict from the ``run()`` function.

    Returns:
        Dict mapping integer index → parsed value.  Only indices whose
        ``argN`` key is present and not ``__ABSENT__`` are included.
        Dict values (enums, memory configs, etc.) are auto-parsed into ttnn objects.
    """
    positional = {}
    for key, value in kwargs.items():
        if key.startswith("arg") and key[3:].isdigit():
            if value is None or value == "__ABSENT__":
                continue
            parsed = parse_dict_value(key, value)
            positional[int(key[3:])] = parsed
    return positional


def build_op_kwargs(
    kwargs: Dict[str, Any],
    *,
    exclude: Optional[Set[str]] = None,
    include_only: Optional[Set[str]] = None,
    output_memory_config: Any = None,
    device: Any = None,
) -> Dict[str, Any]:
    """Extract actual op kwargs from the full test vector kwargs.

    Filters out infrastructure keys, positional tensor params, named tensor kwargs,
    output_memory_config, memory_config, and ``__ABSENT__`` sentinel values.
    Parses dict values into ttnn objects.

    Note: ``memory_config`` is filtered out by default because most ops either
    don't accept it or handle it via separate positional parameters.  Sweep
    modules that need ``memory_config`` in op kwargs should add it explicitly
    after calling this function.

    Note: Positional args (``arg0``, ``arg1``, …) are filtered out because they
    are positional parameters, not keyword arguments.  Use
    :func:`extract_positional_args` to retrieve them.

    Args:
        kwargs: The **kwargs from the run() function (everything not in named params)
        exclude: Additional keys to exclude (op-specific, e.g., keys already handled)
        include_only: If set, only include these specific keys (overrides default filtering)
        output_memory_config: Unused, kept for backward compatibility.

    Returns:
        Dict of parsed op kwargs ready to pass as **op_kwargs to the ttnn op.
        None values and ``__ABSENT__`` sentinels are excluded.
    """
    all_keys = set(kwargs.keys())
    exclude = exclude or set()
    op_kwargs = {}

    # V2 vectors carry __absent_keys__: parameter names that were *absent* in
    # the master config (vs explicitly None). A None value whose key is not in
    # this set was explicitly None in master and must be preserved so the
    # sweep trace records the same kwarg. Without this, ops drop kwargs like
    # sub_core_grids=None and produce a hash divergence vs master.
    absent_keys = kwargs.get("__absent_keys__") or set()
    if not isinstance(absent_keys, (set, frozenset, list, tuple)):
        absent_keys = set()
    else:
        absent_keys = set(absent_keys)

    for key, value in kwargs.items():
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

        # None handling: keep explicit None when V2 says the key was present
        # in master (i.e. not in __absent_keys__). Drop None when absent.
        if value is None:
            if key in absent_keys:
                continue
            op_kwargs[key] = None
            continue

        # Parse list-of-UnaryOpType-dicts (e.g. input_tensor_a_activations)
        list_parsed = _maybe_parse_unary_list(value)
        if list_parsed is not value:
            op_kwargs[key] = list_parsed
            continue
        # Create preallocated output tensor from descriptor
        if key == "output_tensor" and isinstance(value, dict) and device is not None:
            input_shape = kwargs.get("input_a_shape")
            tensor = _create_output_tensor(value, device, input_shape=input_shape)
            if tensor is not None:
                op_kwargs[key] = tensor
            continue
        # Parse dict values into ttnn objects
        parsed = parse_dict_value(key, value)
        if parsed is not None:
            op_kwargs[key] = parsed

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


def comp_pcc_chunked(golden, calculated, pcc_threshold=0.99, chunk=8 * 1024 * 1024):
    """Exact Pearson PCC via single-pass streaming float64 sums — O(chunk) memory.

    The standard comp_pcc clones both tensors, builds several full-size NaN/Inf
    boolean masks, and runs a float64 corrcoef; for ~1e9-element tensors that
    exhausts host RAM (the device op itself is fine). This accumulates the same
    correlation statistics in fixed-size chunks without materializing full-size
    intermediates, so it validates arbitrarily large outputs.
    """
    import math

    import torch

    g = golden.reshape(-1)
    c = calculated.reshape(-1)
    if c.dtype != g.dtype:
        try:
            c = c.to(g.dtype)
        except Exception:
            pass
    n = int(g.numel())
    Sg = Sc = Sgg = Scc = Sgc = 0.0
    cnt = 0
    for i in range(0, n, chunk):
        gg = g[i : i + chunk].to(torch.float64)
        cc = c[i : i + chunk].to(torch.float64)
        m = torch.isfinite(gg) & torch.isfinite(cc)
        if not bool(m.all()):
            gg = gg[m]
            cc = cc[m]
        Sg += float(gg.sum())
        Sc += float(cc.sum())
        Sgg += float((gg * gg).sum())
        Scc += float((cc * cc).sum())
        Sgc += float((gg * cc).sum())
        cnt += int(gg.numel())
    if cnt == 0:
        return True, 1.0
    cov = cnt * Sgc - Sg * Sc
    vg = cnt * Sgg - Sg * Sg
    vc = cnt * Scc - Sc * Sc
    if vg <= 0.0 or vc <= 0.0:
        both_const = vg <= 0.0 and vc <= 0.0
        return both_const, (1.0 if both_const else 0.0)
    p = cov / math.sqrt(vg * vc)
    p = max(-1.0, min(1.0, p))
    return (p >= pcc_threshold), p


def check_with_pcc_safe(expected, actual, pcc=0.99, large_numel=100_000_000):
    """Drop-in for check_with_pcc that avoids host OOM on very large tensors.

    Below `large_numel` elements it defers to the standard check_with_pcc; above
    it, it uses the streaming comp_pcc_chunked (which the ~1e9-element linear /
    multiply outputs need — the full-tensor PCC OOMs/crashes the host).
    """
    from tests.ttnn.utils_for_testing import check_with_pcc

    if tuple(expected.shape) != tuple(actual.shape):
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected.shape)} vs "
            f"list(actual_pytorch_result.shape)={list(actual.shape)}",
        )
    if expected.numel() <= large_numel:
        return check_with_pcc(expected, actual, pcc)
    ok, p = comp_pcc_chunked(expected, actual, pcc)
    return (ok, f"PCC: {p}")
