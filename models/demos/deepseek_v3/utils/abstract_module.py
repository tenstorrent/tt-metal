# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import (
    MeshDeviceStub,
    ModelConfig,
    OpConfigBase,
    RunConfig,
    TensorStub,
    WeightConfig,
    WeightStub,
    is_op_config,
    merge_config_containers,
)

# TODO:
# - Update README.md

# Use-cases to describe in docs:, implementing custom op configs


class InferenceMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class AbstractModule(ABC):
    """Abstract base class for Deepseek submodules.

    This class defines the common interface for submodules. The modules are not instantiated directly, but rather
    used as a namespace for the methods that define the model's behavior in prefill and decode. This is to make it easy
    to separate the stateful and stateless parts of the model, and allow for easy re-use of the methods.

    All subclasses must implement the following methods:
    - either `forward` or both `_forward_prefill` and `_forward_decode` - these define the forward pass for the module.
    - `prefill_model_config` - generates the model configuration for prefill mode.
    - `decode_model_config` - generates the model configuration for decode mode.
    - `convert_weights` - converts PyTorch weights to TTNN format and saves them to the specified path.
    - `_new_state` (optional) - creates a new state for the module, which is used to store persistent model state.

    Typical usage by a caller would be:
    1. (one-off) use `convert_weights` to convert PyTorch weights to TTNN format and save them to disk.
       This returns a `WeightConfig` that contains the paths to the saved weights.
    2. (one-off/runtime) call `prefill_model_config` and `decode_model_config` to generate static `ModelConfig`s.
    3. (runtime) create `RunConfig`s for prefill and decode using the `run_config` method.
    4. (runtime) call `forward` with the input tensor and the appropriate `RunConfig` to run the model.

    Both `ModelConfig` and `WeightConfig` are static configurations that define the configurations of operators
    used in the module. They are typically meant to be hierarchies of string-keyed dictionaries, where the keys
    are operator names. In the `ModelConfig`, the operator-specific configurations are dataclasses that inherit from
    `OpConfigBase`. They behave exactly like dictionaries, in that they can be string-addressed, but with the added
    benefit of restricting the keys to only the ones allowed by the operators.

    The `OpConfigBase` dataclasses are meant to provide a clear interface for the operator arguments. They are designed
    to be kwargs-destructured into the operator calls, e.g. `ttnn.linear(x, **cfg["w1"])`. This allows for a clean and
    readable forward pass.

    The `RunConfig` is meant to combine the `ModelConfig`, `WeightConfig`, and the model state from `_new_state` into a
    single dictionary that can be used during the forward pass. The default implementation of `run_config` merges the
    `ModelConfig`, `WeightConfig` and state config recursively using the field keys (or the dataclass field names).
    If there is no corresponding key in any of the configurations, or if one of the values is `None`, the value from
    the other configurations is used.

    A special case during the creation of the `RunConfig` is when loading the weights. If a `TensorStub` is found in the
    `ModelConfig`, along with a corresponding `WeightStub` in the `WeightConfig`, the path to the weight tensor in the
    `WeightStub` is loaded to a `ttnn.Tensor` on the device specified by the `mesh_device` argument to the `run_config`
    method.

    Another special case is when a `MeshDeviceStub` is found in the `ModelConfig`. In this case, it is replaced with the
    `mesh_device` argument passed to the `run_config` method. An additional check is performed to ensure that the shape
    of the `mesh_device` matches the one specified in the `MeshDeviceStub`. It is generally meant to be used in
    op configs directly.
    """

    FORWARD_REIMPL_ERR = NotImplementedError(
        f"Subclasses of {__name__} must either reimplement the forward method or both _forward_prefill and _forward_decode methods."
    )
    ModelState = Any  # Type of the persistent model state
    StatelessRunConfig = dict[str, "StatelessRunConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Model state should be created with the _new_state method, not the constructor.")

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass of the module.
        Subclasses may reimplement this method to handle the forward pass for both prefill and decode modes, or
        they may implement the `_forward_prefill` and `_forward_decode` methods separately. In the latter case,
        the prefill/decode mode will be dispatched based on the `RunConfig`'s `mode` attribute
        (taken from the `ModelConfig`) and must be an instance of `InferenceMode`.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after module computation
        """
        assert isinstance(cfg, dict), "Expected the RunConfig to be a dict"
        assert "mode" in cfg and isinstance(
            cfg["mode"], InferenceMode
        ), "RunConfig must contain a valid 'mode' key of type InferenceMode"

        if cfg["mode"] == InferenceMode.PREFILL:
            return cls._forward_prefill(x, cfg)
        else:
            return cls._forward_decode(x, cfg)

    @classmethod
    def _forward_prefill(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass for prefill mode.
        If the `forward` method is not reimplemented, this method must be implemented by subclasses
        to handle the prefill logic.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for prefill

        Returns:
            Output tensor after prefill computation
        """
        raise AbstractModule.FORWARD_REIMPL_ERR

    @classmethod
    def _forward_decode(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass for decode mode.
        If the `forward` method is not reimplemented, this method must be implemented by subclasses
        to handle the decode logic.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for decode
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after prefill computation
        """
        raise AbstractModule.FORWARD_REIMPL_ERR

    @classmethod
    @abstractmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelConfig:
        """Generate prefill configuration for this module.
        Subclasses must implement this method to generate the model configuration for prefill mode. This configuration
        typically includes operator configurations that define how the module should behave during prefill.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            ModelConfig containing operator configurations for prefill mode
        """
        raise NotImplementedError(
            f"Subclasses of {AbstractModule.__name__} must implement the prefill_model_config method"
        )

    @classmethod
    @abstractmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelConfig:
        """Generate decode configuration for this module.
        Subclasses must implement this method to generate the model configuration for decode mode. This configuration
        typically includes operator configurations that define how the module should behave during decode.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            ModelConfig containing operator configurations for decode mode
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
        return a (nested) dictionary of `WeightStub`s created from the converted `ttnn.Tensor`s. The `WeightStub`s
        also save the `ttnn.Tensor`s to the specified output path.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to keyword tensor arguments and their tensor stubs
        """
        raise NotImplementedError(f"Subclasses of {AbstractModule.__name__} must implement the convert_weights method")

    @classmethod
    def _new_state(cls, stateless_run_config: StatelessRunConfig, mesh_device: ttnn.Device) -> Any:
        """Create a new state for the module.
        Subclasses may override this method to initialize the state of the module, which is typically used to
        store persistent model state that is not part of the model configuration or weights. This is merged
        with the `ModelConfig` and `WeightConfig` to create a `RunConfig` that can be used during the forward pass.

        Args:
            stateless_run_config: StatelessRunConfig containing the merged model configuration and weights
            mesh_device: TTNN mesh device

        Returns:
            A new object initializing the state of the module
        """
        return None

    @classmethod
    def run_config(cls, model_config: ModelConfig, weights_config: WeightConfig, *args, **kwargs) -> RunConfig:
        """Create a RunConfig for the module.
        Subclasses may reimplement this method to handle multi-mesh-device modules.

        Args:
            model_config: Model configuration for the module
            weights_config: Weights configuration for the module
            mesh_device: TTNN mesh device

        Returns:
            RunConfig containing all necessary configurations for running the module
        """
        return AbstractModule._run_config_impl(model_config, weights_config, cls._new_state, *args, **kwargs)

    @staticmethod
    def _run_config_impl(
        model_config: ModelConfig,
        weight_config: WeightConfig,
        state_factory: Callable[[StatelessRunConfig, ttnn.MeshDevice], Any],
        mesh_device: ttnn.Device,
    ) -> RunConfig:
        stateless_run_config = merge_config_containers(
            model_config, weight_config, AbstractModule._merge_model_weight_config_items, mesh_device
        )
        state = state_factory(stateless_run_config, mesh_device)
        run_config = merge_config_containers(
            stateless_run_config,
            state,
            AbstractModule._merge_run_config_items,
            mesh_device,
        )
        print(AbstractModule._convert_run_config_to_pretty_print(run_config))
        return run_config

    @staticmethod
    def _merge_model_weight_config_items(
        model_config_item: Any, weight_config_item: Any, mesh_device: ttnn.Device
    ) -> Any:
        if isinstance(model_config_item, TensorStub) and isinstance(weight_config_item, WeightStub):
            return weight_config_item.to_weight(mesh_device)

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
    def _convert_run_config_to_pretty_print(run_config_item: Any) -> str:
        if isinstance(run_config_item, dict):
            return str({k: AbstractModule._convert_run_config_to_pretty_print(v) for k, v in run_config_item.items()})
        elif is_op_config(run_config_item):
            assert is_dataclass(run_config_item), "OpConfigs must be dataclasses"
            op_config_fields_str = ", ".join(
                f"{f.name}={AbstractModule._convert_run_config_to_pretty_print(getattr(run_config_item, f.name))}"
                for f in fields(run_config_item)
            )
            return f"{run_config_item.__class__.__name__}({op_config_fields_str})"  # type: ignore
        elif isinstance(run_config_item, ttnn.Tensor):
            return f"ttnn.Tensor(shape={run_config_item.shape}, \
                dtype={run_config_item.dtype}, \
                memory_config={run_config_item.memory_config() if hasattr(run_config_item, 'memory_config') else 'None'})"
        else:
            return str(run_config_item)
