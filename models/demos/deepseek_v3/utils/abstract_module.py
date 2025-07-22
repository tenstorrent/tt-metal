# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class AbstractModule(ABC):  # TODO: update the doc
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
    - `create_state` (optional) - creates a new state for the module, which is used to store persistent model state.

    Typical usage by a caller would be:
    1. (one-off) use `convert_weights` to convert PyTorch weights to TTNN format and save them to disk.
       This returns a `WeightConfig` that contains the paths to the saved weights.
    2. (one-off/runtime) call `prefill_model_config` and `decode_model_config` to generate static model configs.
    3. (runtime) call `create_state` to create a new state for the module.
    4. (runtime) create `RunPrefillConfig` for prefill and `RunDecodeConfig` for decode using the `run_config` method.
    5. (runtime) call either `forward_prefill` or `forward_decode` with the input tensor and `RunPrefillConfig` or
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
    def create_state(cls, hf_config: PretrainedConfig, *args, **kwargs) -> ModelState:
        """Create a new state for the module.
        Subclasses may override this method to initialize the state of the module, which is typically used to
        store persistent model state that is not part of the model configuration or weights.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device (default implementation only): TTNN mesh device on which to load the weights and instantiate the `MeshDeviceStub`s.

        Returns:
            A new object initializing the state of the module
        """
        return cls._create_state_impl(*args, **kwargs)

    @final
    @classmethod
    def _create_state_impl(cls, mesh_device: ttnn.Device) -> ModelState:
        """Default implementation of creating a new state for a module."""
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device}
