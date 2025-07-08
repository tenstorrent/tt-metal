# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from types import NoneType
from typing import Any, Callable, overload

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub, OpConfigBase

MESH_DEVICE_STATE_DICT_KEY = "mesh_device"

WeightConfig = dict[str, "WeightConfig | str"]

_PRIMITIVE_COPYABLE_TYPES = bool | int | float | complex | str | bytes | None | Enum
# In general, we require ModelConfig to be deepcopyable
ModelPrefillConfig = dict[str, "ModelPrefillConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
ModelDecodeConfig = dict[str, "ModelDecodeConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase

ModelState = Any  # Type of the persistent model state

RunPrefillConfig = dict[str, "RunPrefillConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
RunDecodeConfig = dict[str, "RunDecodeConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase


@overload
def run_config(
    model_config: ModelPrefillConfig, weight_config: WeightConfig, model_state: ModelState
) -> RunPrefillConfig:
    ...


@overload
def run_config(  # type: ignore
    model_config: ModelDecodeConfig, weight_config: WeightConfig, model_state: ModelState
) -> RunDecodeConfig:
    ...


def create_run_config(model_config, weight_config, model_state):
    # The model config and state config are merged first to determine the mesh devices to load the configs on.
    model_state_config = _merge_config_containers(
        model_config,
        model_state,
        merge_config_specific_items=_merge_model_config_state_items,
        search_for_mesh_device=True,
        mb_mesh_device=None,
    )

    run_config = _merge_config_containers(
        model_state_config,
        weight_config,
        merge_config_specific_items=_merge_run_config,
        search_for_mesh_device=False,
        mb_mesh_device=None,
    )

    print(f"run config: {_convert_run_config_to_pretty_print(run_config)}")

    return run_config


def _merge_model_config_state_items(model_config_item: Any, state_item: Any, mb_mesh_device: ttnn.Device | None) -> Any:
    if state_item is None and isinstance(model_config_item, FromWeightConfig):
        return FromWeightConfig(
            mesh_device=_merge_model_config_state_items(model_config_item.mesh_device, state_item, mb_mesh_device)
        )

    if state_item is None and isinstance(model_config_item, MeshDeviceStub):
        if mb_mesh_device is None:
            raise ValueError(
                f"MeshDevice expected but not found under the '{MESH_DEVICE_STATE_DICT_KEY}' key in the model state"
            )
        if not model_config_item.mesh_shape == tuple(mb_mesh_device.shape):
            raise ValueError("MeshDevice has the wrong shape for a given FromWeightConfig")
        return mb_mesh_device

    if state_item is None:
        return model_config_item

    if model_config_item is None:
        return state_item

    raise ValueError(f"Unsupported model_weight and state config items to merge: {model_config_item} and {state_item}")


def _merge_run_config(model_state_config_item: Any, weight_config_item: Any, _: ttnn.Device | None) -> Any:
    if isinstance(model_state_config_item, FromWeightConfig) and isinstance(weight_config_item, str):
        return ttnn.load_tensor(weight_config_item, device=model_state_config_item.mesh_device)

    if weight_config_item is None:
        assert not isinstance(
            model_state_config_item, MeshDeviceStub
        ), "MeshDeviceStub should have been replaced by a real MeshDevice from the model state"
        return model_state_config_item

    raise ValueError(
        f"Unsupported model and weight config items to merge: {model_state_config_item} and {weight_config_item}"
    )


def _merge_config_containers(
    cfg_a: Any,
    cfg_b: Any,
    merge_config_specific_items: Callable[[Any, Any, ttnn.MeshDevice | None], Any],
    search_for_mesh_device: bool,
    mb_mesh_device: ttnn.MeshDevice | None = None,
) -> Any:
    """Helper function to merge two configs, where the first one may partially consist of OpConfigs."""
    if cfg_a is None and cfg_b is None:
        return None

    if is_op_config(cfg_a):
        op_config_dict = {f.name: getattr(cfg_a, f.name) for f in fields(cfg_a)}  # type: ignore
        return cfg_a.__class__(**_merge_config_containers(op_config_dict, cfg_b, merge_config_specific_items, search_for_mesh_device, mb_mesh_device))  # type: ignore

    # If both configs are lists/tuples of the same length or one of them is None, merge them as a list/tuple.
    if isinstance(cfg_a, (list, tuple, NoneType)) and isinstance(cfg_b, (list, tuple, NoneType)):
        if cfg_a is None or cfg_b is None or (len(cfg_a) == len(cfg_b) and type(cfg_a) == type(cfg_b)):
            container = type(cfg_a) if cfg_a is not None else type(cfg_b)
            cfg_a = cfg_a or (container([None]) * len(cfg_b))
            cfg_b = cfg_b or (container([None]) * len(cfg_a))
            return container(
                _merge_config_containers(a, b, merge_config_specific_items, search_for_mesh_device, mb_mesh_device)
                for a, b in zip(cfg_a, cfg_b, strict=True)
            )

    if isinstance(cfg_a, (dict, NoneType)) and isinstance(cfg_b, (dict, NoneType)):
        cfg_a = cfg_a or {}
        cfg_b = cfg_b or {}
        if (
            search_for_mesh_device and MESH_DEVICE_STATE_DICT_KEY in cfg_b
        ):  # If we are searching for a mesh device, we need to find it in cfg_b (model state)
            mb_mesh_device = cfg_b[MESH_DEVICE_STATE_DICT_KEY]
            # Remove mesh device key from cfg_b so it does not clash with possible mesh device stubs deeper in the config
            cfg_b = {k: v for k, v in cfg_b.items() if k != MESH_DEVICE_STATE_DICT_KEY}
        return {
            k: _merge_config_containers(
                cfg_a.get(k, None),
                cfg_b.get(k, None),
                merge_config_specific_items,
                search_for_mesh_device,
                mb_mesh_device,
            )
            for k in set(cfg_a.keys()) | set(cfg_b.keys())
        }

    return merge_config_specific_items(cfg_a, cfg_b, mb_mesh_device)


def _convert_run_config_to_pretty_print(run_config_item: Any, indent: int = 0) -> str:
    """Convert run config to a pretty-printed string with proper indentation."""
    indent_str = "  " * indent
    next_indent_str = "  " * (indent + 1)

    if isinstance(run_config_item, dict):
        if not run_config_item:
            return "{}"

        lines = ["{"]
        for k, v in run_config_item.items():
            value_str = _convert_run_config_to_pretty_print(v, indent + 1)
            lines.append(f"{next_indent_str}{k!r}: {value_str},")
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    elif is_op_config(run_config_item):
        class_name = run_config_item.__class__.__name__

        config_fields = []
        for f in fields(run_config_item):
            field_value = getattr(run_config_item, f.name)
            value_str = _convert_run_config_to_pretty_print(field_value, indent + 1)

            # If the value string contains newlines, format it properly indented
            if "\n" in value_str:
                config_fields.append(f"{next_indent_str}{f.name}=")
                # Indent the multi-line value
                indented_lines = [
                    f"{next_indent_str}  {line}" if line.strip() else line for line in value_str.split("\n")
                ]
                config_fields.append("\n".join(indented_lines))
            else:
                config_fields.append(f"{next_indent_str}{f.name}={value_str}")

        if not config_fields:
            return f"{class_name}()"

        return f"{class_name}(\n" + ",\n".join(config_fields) + f"\n{indent_str})"

    elif isinstance(run_config_item, ttnn.Tensor):
        shape = run_config_item.shape
        dtype = run_config_item.dtype

        # Get memory config if available - simplified display
        try:
            memory_config = run_config_item.memory_config()
            layout = str(memory_config.memory_layout).split(".")[-1]
            buffer_type = str(memory_config.buffer_type).split(".")[-1]
            if memory_config.shard_spec is not None:
                shard_info = f", sharded{memory_config.shard_spec.shape}"
            else:
                shard_info = ""
            mem_str = f"{layout}_{buffer_type}{shard_info}"
        except:
            mem_str = "Unknown"

        return f"ttnn.Tensor(shape={shape}, dtype={dtype}, memory={mem_str})"

    elif isinstance(run_config_item, (list, tuple)):
        if not run_config_item:
            return "[]" if isinstance(run_config_item, list) else "()"

        container_type = "[" if isinstance(run_config_item, list) else "("
        container_end = "]" if isinstance(run_config_item, list) else ")"

        if len(run_config_item) == 1 and isinstance(run_config_item, tuple):
            # Special case for single-element tuples
            value_str = _convert_run_config_to_pretty_print(run_config_item[0], indent)
            return f"({value_str},)"

        # Short list/tuple on one line if all elements are simple
        if all(
            isinstance(item, (str, int, float, bool)) or (hasattr(item, "__name__") and not isinstance(item, dict))
            for item in run_config_item
        ):
            items_str = ", ".join(str(item) for item in run_config_item)
            if len(items_str) < 50:  # Keep short lists on one line
                return f"{container_type}{items_str}{container_end}"

        # Multi-line for complex items or long lists
        lines = [container_type]
        for item in run_config_item:
            value_str = _convert_run_config_to_pretty_print(item, indent + 1)
            lines.append(f"{next_indent_str}{value_str},")
        lines.append(f"{indent_str}{container_end}")
        return "\n".join(lines)

    else:
        # Special handling for specific object types
        if hasattr(run_config_item, "__class__"):
            class_name = run_config_item.__class__.__name__

            # Special case for ComputeKernelConfig
            if "ComputeKernelConfig" in class_name:
                try:
                    # Try to extract meaningful info
                    if hasattr(run_config_item, "math_fidelity"):
                        fidelity = str(run_config_item.math_fidelity).split(".")[-1]
                        return f"ComputeKernelConfig(math_fidelity={fidelity})"
                    else:
                        return f"ComputeKernelConfig(...)"
                except:
                    return f"ComputeKernelConfig(...)"

            # Special case for MemoryConfig - simplified display
            elif "MemoryConfig" in class_name:
                try:
                    layout = str(run_config_item.memory_layout).split(".")[-1]
                    buffer_type = str(run_config_item.buffer_type).split(".")[-1]
                    return f"MemoryConfig(layout={layout}, buffer={buffer_type})"
                except:
                    return f"MemoryConfig(...)"

            # Special case for program configs
            elif "ProgramConfig" in class_name:
                try:
                    if hasattr(run_config_item, "in0_block_w"):
                        return f"{class_name}(in0_block_w={run_config_item.in0_block_w}, ...)"
                    else:
                        return f"{class_name}(...)"
                except:
                    return f"{class_name}(...)"

        # Handle simple types and fallback objects
        return repr(run_config_item)


def is_op_config(obj: Any) -> bool:
    """Check if the object is an op config instance."""
    return issubclass(type(obj), OpConfigBase) and is_dataclass(obj)
