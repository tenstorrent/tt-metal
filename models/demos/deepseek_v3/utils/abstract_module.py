# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType
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
    WeightsConfig,
    WeightStub,
)


class InferenceMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class AbstractModule(ABC):
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

    def __new__(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Initialize the module with the given HuggingFace config and mesh device.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
        """
        return cls.forward(x, cfg)

    @classmethod
    @abstractmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightsConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to keyword tensor arguments and their tensor stubs
        """

    @classmethod
    @abstractmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelConfig:
        """Prefill model config for a module with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
            kwargs: Additional arguments for specifying the model configuration

        Returns:
            Dict containing operator configurations for prefill mode
        """

    @classmethod
    @abstractmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, **kwargs) -> ModelConfig:
        """Generate decode operator configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
            kwargs: Additional arguments for specifying the model configuration

        Returns:
            Dict containing operator configurations for decode mode
        """

    @classmethod
    def run_config(cls, model_config: ModelConfig, weights_config: WeightsConfig, *args, **kwargs) -> "RunConfig":
        """Create a RunConfig for the module.

        Args:
            model_config: Model configuration for the module
            weights_config: Weights configuration for the module
            mesh_device: TTNN mesh device

        Returns:
            RunConfig containing all necessary configurations for running the module

        NOTE: Subclasses may reimplement this method to handle multi-mesh-device modules.
        """
        mesh_device = kwargs.get("mesh_device", next(iter(args), None))
        if mesh_device is None:
            raise ValueError("RunConfig requires a 'mesh_device' argument")
        run_config = AbstractModule._merge_config_items(model_config, weights_config, mesh_device)
        print(AbstractModule._convert_run_config_to_pretty_print(run_config))
        return run_config

    @staticmethod
    def _merge_config_items(
        model_config_item: Any, weight_config_item: Any, mesh_device: ttnn.MeshDevice
    ) -> RunConfig | OpConfigBase | ttnn.Tensor | None:
        """Merge model and weights configurations into a single RunConfig item.

        Args:
            model_config_item: Model configuration item
            weights_config_item: Weights configuration item
        """
        if issubclass(type(model_config_item), OpConfigBase):
            assert is_dataclass(model_config_item), "OpConfigs must be dataclasses"
            op_config_dict = {f.name: getattr(model_config_item, f.name) for f in fields(model_config_item)}
            return model_config_item.__class__(
                **AbstractModule._merge_config_items(op_config_dict, weight_config_item, mesh_device)
            )

        if isinstance(model_config_item, TensorStub) and isinstance(weight_config_item, WeightStub):
            return weight_config_item.to_weight(mesh_device)

        if isinstance(model_config_item, (dict, NoneType)) and isinstance(weight_config_item, (dict, NoneType)):
            model_config_item = model_config_item or {}
            weight_config_item = weight_config_item or {}
            return {
                k: AbstractModule._merge_config_items(
                    model_config_item.get(k, None), weight_config_item.get(k, None), mesh_device
                )
                for k in itertools.chain(model_config_item.keys(), weight_config_item.keys())
            }

        if weight_config_item is None:
            if isinstance(model_config_item, MeshDeviceStub):
                assert model_config_item.mesh_shape == tuple(mesh_device.shape)
                return mesh_device
            return model_config_item

        raise ValueError(f"Unsupported config items to merge: {model_config_item} and {weight_config_item}")

    @staticmethod
    def _convert_run_config_to_pretty_print(run_config_item: Any) -> str:
        if isinstance(run_config_item, dict):
            return str({k: AbstractModule._convert_run_config_to_pretty_print(v) for k, v in run_config_item.items()})
        elif issubclass(type(run_config_item), OpConfigBase):
            assert is_dataclass(run_config_item), "OpConfigs must be dataclasses"
            op_config_fields_str = ", ".join(
                f"{f.name}={AbstractModule._convert_run_config_to_pretty_print(getattr(run_config_item, f.name))}"
                for f in fields(run_config_item)
            )
            return f"{run_config_item.__class__.__name__}({op_config_fields_str})"
        elif isinstance(run_config_item, ttnn.Tensor):
            return f"ttnn.Tensor(shape={run_config_item.shape}, \
                dtype={run_config_item.dtype}, \
                memory_config={run_config_item.memory_config() if hasattr(run_config_item, 'memory_config') else 'None'})"
        else:
            return str(run_config_item)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
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
            return cls._forward_prefill(x, cfg)
        else:
            return cls._forward_decode(x, cfg)

    def _forward_prefill(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass for prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for prefill
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after prefill computation
        """
        raise FORWARD_REIMPL_ERR

    def _forward_decode(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass for decode mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations for decode
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after prefill computation
        """
        raise FORWARD_REIMPL_ERR


FORWARD_REIMPL_ERR = NotImplementedError(
    f"Subclasses of {AbstractModule.__name__} must either reimplement the forward method or both _forward_prefill and _forward_decode methods."
)
