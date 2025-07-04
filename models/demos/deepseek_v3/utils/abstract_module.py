# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import (
    FromWeightConfig,
    MeshDeviceStub,
    ModelDecodeConfig,
    ModelPrefillConfig,
    OpConfigBase,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
    is_op_config,
    merge_config_containers,
)


class AbstractModule(ABC):
    """Abstract base class for Deepseek submodules.

    This class defines the common interface for submodules. The modules are not instantiated directly, but rather
    used as a namespace for the methods that define the model's behavior in prefill and decode. This is to make it easy
    to separate the stateful and stateless parts of the model, and allow for easy re-use of the methods.

    All subclasses must implement the following methods:
    - `forward_prefill` - defines the prefill-variant forward pass for the module.
    - `forward_decode` - defines the decode-variant forward pass for the module.
    - `prefill_model_config` - generates the model configuration for prefill mode.
    - `decode_model_config` - generates the model configuration for decode mode.
    - `convert_weights` - converts PyTorch weights to TTNN format and saves them to the specified path.
    - `_new_state` (optional) - creates a new state for the module, which is used to store persistent model state.
    - `run_config` - creates a `RunPrefillConfig` and a `RunDecodeConfig` for the module, which combine
      the model prefill/decode configuration, weight configuration, and state configuration.

    Typical usage by a caller would be:
    1. (one-off) use `convert_weights` to convert PyTorch weights to TTNN format and save them to disk.
       This returns a `WeightConfig` that contains the paths to the saved weights.
    2. (one-off/runtime) call `prefill_model_config` and `decode_model_config` to generate static model configs.
    3. (runtime) create `RunPrefillConfig` for prefill and `RunDecodeConfig` for decode using the `run_config` method.
    4. (runtime) call either `forward_prefill` or `forward_decode` with the input tensor and `RunPrefillConfig` or
       `RunDecodeConfig` (respectively) to run the model.

    `ModelPrefillConfig`, `ModelDecodeConfig` and `WeightConfig` are static configurations that define
    the configurations of operators used in the module. They are typically meant to be hierarchies of string-keyed
    dictionaries, where the keys are operator names. In `ModelPrefillConfig` and `ModelDecodeConfig`, the
    operator-specific configurations are dataclasses that inherit from `OpConfigBase`. They behave exactly like
    dictionaries, in that they can be string-addressed, but with the added benefit of restricting the keys to only
    the ones allowed by the operators.

    The `OpConfigBase` dataclasses are meant to provide a clear interface for the operator arguments. They are designed
    to be kwargs-destructured into the operator calls, e.g. `ttnn.linear(x, **cfg["w1"])`. This allows for a clean and
    readable forward pass.

    `RunPrefillConfig` and `RunDecodeConfig` are meant to combine the model configs, `WeightConfig`, and the model
    state from `_new_state` into a single dictionary that can be used during the forward pass. The default
    implementation of `run_config` merges the model configs, `WeightConfig` and state config recursively using
    the field keys (or the dataclass field names). If there is no corresponding key in any of the configurations,
    or if one of the values is `None`, the value from the other configurations is used.

    A special case during the creation of the run config is when loading the weights. If a `FromWeightConfig`
    is found in a model config, along with a corresponding tensor path in the `WeightConfig`, said weight tensor
    is loaded to a `ttnn.Tensor` on the device specified by the `mesh_device` argument to the `run_config` method.

    Another special case is when a `MeshDeviceStub` is found in a model config. In this case, it is replaced with the
    `mesh_device` argument passed to the `run_config` method. An additional check is performed to ensure that the shape
    of the `mesh_device` matches the one specified in the `MeshDeviceStub`. It is generally meant to be used in
    op configs directly.
    """

    ModelState = Any  # Type of the persistent model state
    StatelessRunPrefillConfig = dict[str, "StatelessRunPrefillConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
    StatelessRunDecodeConfig = dict[str, "StatelessRunDecodeConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase

    @final
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        raise NotImplementedError("Model state should be created with the _new_state method, not the constructor.")

    @final
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Model state should be created with the _new_state method, not the constructor.")

    @classmethod
    @abstractmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass for prefill mode.
        Subclasses must reimplement this method to handle the prefill logic.

        Args:
            x: Input tensor
            cfg: RunPrefillConfig containing weights and op configurations for prefill

        Returns:
            Output tensor after prefill computation
        """
        raise NotImplementedError(f"Subclasses of {AbstractModule.__name__} must implement the forward_prefill method")

    @classmethod
    @abstractmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        """Forward pass for decode mode.
        Subclasses must reimplement this method to handle the decode logic.

        Args:
            x: Input tensor
            cfg: RunDecodeConfig containing weights and op configurations for decode

        Returns:
            Output tensor after decode computation
        """
        raise NotImplementedError(f"Subclasses of {AbstractModule.__name__} must implement the forward_decode method")

    @classmethod
    @abstractmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.
        Subclasses must implement this method to generate the model configuration for prefill mode. This configuration
        typically includes operator configurations that define how the module should behave during prefill.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        raise NotImplementedError(
            f"Subclasses of {AbstractModule.__name__} must implement the prefill_model_config method"
        )

    @classmethod
    @abstractmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelDecodeConfig:
        """Generate decode configuration for this module.
        Subclasses must implement this method to generate the model configuration for decode mode. This configuration
        typically includes operator configurations that define how the module should behave during decode.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """
        raise NotImplementedError(
            f"Subclasses of {AbstractModule.__name__} must implement the decode_model_config method"
        )

    @classmethod
    @abstractmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.
        Subclasses must implement this method to convert the PyTorch state dict to a TTNN-compatible format and
        return a (nested) dictionary of paths created from the `ttnn.Tensor`s saved using `save_and_get_path`.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation keyword tensor arguments to their save paths
        """
        raise NotImplementedError(f"Subclasses of {AbstractModule.__name__} must implement the convert_weights method")

    @classmethod
    def _new_state(
        cls,
        stateless_run_prefill_config: StatelessRunPrefillConfig,
        stateless_run_decode_config: StatelessRunDecodeConfig,
        mesh_device: ttnn.Device,
    ) -> Any:
        """Create a new state for the module.
        Subclasses may override this method to initialize the state of the module, which is typically used to
        store persistent model state that is not part of the model configuration or weights. This is merged
        with the `ModelConfig` and `WeightConfig` to create a `RunConfig` that can be used during the forward pass.

        Args:
            stateless_run_prefill_config: StatelessRunPrefillConfig containing the merged model prefill configuration and weights
            stateless_run_decode_config: StatelessRunDecodeConfig containing the merged model decode configuration and weights
            mesh_device: TTNN mesh device

        Returns:
            A new object initializing the state of the module
        """
        return None

    @classmethod
    def run_config(
        cls,
        model_prefill_config: ModelPrefillConfig,
        model_decode_config: ModelDecodeConfig,
        weight_config: WeightConfig,
        *args,
        **kwargs,
    ) -> tuple[RunPrefillConfig, RunDecodeConfig]:
        """Create a RunPrefillConfig and RunDecodeConfig for the module.
        Subclasses may reimplement this method to handle multi-mesh-device modules.

        Args:
            model_prefill_config: Model prefill configuration for the module
            model_decode_config: Model decode configuration for the module
            weight_config: Weights configuration for the module
            mesh_device: TTNN mesh device

        Returns:
            RunConfig containing all necessary configurations for running the module
        """
        return AbstractModule._run_config_impl(
            model_prefill_config, model_decode_config, weight_config, cls._new_state, *args, **kwargs
        )

    @staticmethod
    def _run_config_impl(
        model_prefill_config: ModelPrefillConfig,
        model_decode_config: ModelDecodeConfig,
        weight_config: WeightConfig,
        state_factory: Callable[[StatelessRunPrefillConfig, StatelessRunDecodeConfig, ttnn.MeshDevice], Any],
        mesh_device: ttnn.Device,
    ) -> tuple[RunPrefillConfig, RunDecodeConfig]:
        stateless_run_prefill_config = merge_config_containers(
            model_prefill_config, weight_config, AbstractModule._merge_model_weight_config_items, mesh_device
        )
        stateless_run_decode_config = merge_config_containers(
            model_decode_config, weight_config, AbstractModule._merge_model_weight_config_items, mesh_device
        )

        state = state_factory(stateless_run_prefill_config, stateless_run_decode_config, mesh_device)

        run_prefill_config = merge_config_containers(
            stateless_run_prefill_config,
            state,
            AbstractModule._merge_run_config_items,
            mesh_device,
        )
        run_decode_config = merge_config_containers(
            stateless_run_decode_config,
            state,
            AbstractModule._merge_run_config_items,
            mesh_device,
        )

        print(f"RunPrefillConfig: {AbstractModule._convert_run_config_to_pretty_print(run_prefill_config)}")
        print(f"RunDecodeConfig: {AbstractModule._convert_run_config_to_pretty_print(run_decode_config)}")

        return run_prefill_config, run_decode_config

    @staticmethod
    def _merge_model_weight_config_items(
        model_config_item: Any, weight_config_item: Any, mesh_device: ttnn.Device
    ) -> Any:
        if isinstance(model_config_item, FromWeightConfig) and isinstance(weight_config_item, str):
            return ttnn.load_tensor(weight_config_item, device=mesh_device)

        if weight_config_item is None:
            if isinstance(model_config_item, MeshDeviceStub):
                assert model_config_item.mesh_shape == tuple(mesh_device.shape)
                return mesh_device
            return model_config_item

        raise ValueError(
            f"Unsupported model and weight config items to merge: {model_config_item} and {weight_config_item}"
        )

    @staticmethod
    def _merge_run_config_items(model_weight_config_item: Any, state_item: Any, mesh_device: ttnn.Device) -> Any:
        if state_item is None:
            return model_weight_config_item
        if model_weight_config_item is None:
            return state_item
        raise ValueError(
            f"Unsupported model_weight and state config items to merge: {model_weight_config_item} and {state_item}"
        )

    @staticmethod
    def _convert_run_config_to_pretty_print(run_config_item: Any, indent: int = 0) -> str:
        """Convert run config to a pretty-printed string with proper indentation."""
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)

        if isinstance(run_config_item, dict):
            if not run_config_item:
                return "{}"

            lines = ["{"]
            for k, v in run_config_item.items():
                value_str = AbstractModule._convert_run_config_to_pretty_print(v, indent + 1)
                lines.append(f"{next_indent_str}{k!r}: {value_str},")
            lines.append(f"{indent_str}}}")
            return "\n".join(lines)

        elif is_op_config(run_config_item):
            assert is_dataclass(run_config_item), "OpConfigs must be dataclasses"
            class_name = run_config_item.__class__.__name__

            config_fields = []
            for f in fields(run_config_item):
                field_value = getattr(run_config_item, f.name)
                value_str = AbstractModule._convert_run_config_to_pretty_print(field_value, indent + 1)

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
                value_str = AbstractModule._convert_run_config_to_pretty_print(run_config_item[0], indent)
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
                value_str = AbstractModule._convert_run_config_to_pretty_print(item, indent + 1)
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
