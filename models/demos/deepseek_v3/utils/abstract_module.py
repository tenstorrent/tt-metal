# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TypedDict

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule


class InferenceMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ModelConfig(TypedDict, total=False):
    mode: InferenceMode


WeightsConfig = dict[str, dict[str, str]]


class RunConfig(TypedDict, total=False):
    mode: InferenceMode


class AbstractModule(LightweightModule, ABC):
    """Abstract base class for Deepseek submodules.

    This class defines the common interface for submodules.

    All subclasses must implement the static methods for weight conversion and model configuration,
    as well as the forward pass.

    Typical usage by a caller would be split between converting torch weights to ttnn weights and running those weights.

    Weight conversion one-off:
    - Use ModuleClass.convert_weights to convert PyTorch weights to TTNN format and save to disk

    At run-time:
    1. Call ModuleClass.prefill_model_config and ModuleClass.decode_model_config to generate static model configs
    2. Create prefill and decode RunConfigs with the model configs and the path to the weights to load into it
    3. Call ModuleClass.forward to run the model with each RunConfig as needed

    A RunConfig is a dict with everything each ttnn op needs to run except the input tensor, e.g.
    you can run ttnn.linear(x, **cfg["w1"]) and it will expand with the weights and program configs etc.
    This keeps the forward pass clean and readable.

    Both convert_weights and the model configs are static methods and can be called without instantiating the class.
    This functional design makes it easy to re-use them in other models if we want to, without having to subclass or
    instantiate it; the class is essentially a namespace for them.

    Keep the constructor as empty as you can. A good use of it is to set up ttnn tensors that are not weights,
    e.g. kv_cache, or as in this example dynamic program configs for prefill.
    """

    @staticmethod
    @abstractmethod
    def convert_weights(
        hf_config: PretrainedConfig, state_dict: dict[str, torch.Tensor], output_path: Path, mesh_device: ttnn.Device
    ) -> WeightsConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to keyword tensor arguments and their TTNN weight file paths
        """

    @staticmethod
    @abstractmethod
    def prefill_model_config(hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelConfig:
        """Prefill model config for a module with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """

    @staticmethod
    @abstractmethod
    def decode_model_config(hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelConfig:
        """Generate decode operator configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """

    def __init__(self, hf_config: PretrainedConfig, mesh_device: ttnn.Device):
        """Initialize the module with the given HuggingFace config and mesh device.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
        """
        super().__init__()

    def forward(self, x: ttnn.Tensor, cfg: RunConfig, mesh_device: ttnn.Device) -> ttnn.Tensor:
        """Forward pass of the module.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after module computation
        """
        assert "mode" in cfg and isinstance(
            cfg["mode"], InferenceMode
        ), "RunConfig must contain a valid 'mode' key of type InferenceMode"

        if cfg["mode"] == InferenceMode.PREFILL:
            return self._forward_prefill(x, cfg, mesh_device)
        else:
            return self._forward_decode(x, cfg, mesh_device)

    def _forward_prefill(self, x: ttnn.Tensor, cfg: RunConfig, mesh_device: ttnn.Device) -> ttnn.Tensor:
        """Forward pass for prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for prefill
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after prefill computation
        """
        raise FORWARD_REIMPL_ERR

    def _forward_decode(self, x: ttnn.Tensor, cfg: RunConfig, mesh_device: ttnn.Device) -> ttnn.Tensor:
        """Forward pass for prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for prefill
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after prefill computation
        """
        raise FORWARD_REIMPL_ERR


FORWARD_REIMPL_ERR = NotImplementedError(
    f"Subclasses of {AbstractModule.__name__} must either reimplement the forward method or both _forward_prefill and _forward_decode methods."
)
