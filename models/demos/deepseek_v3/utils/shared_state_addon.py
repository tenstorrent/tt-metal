# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from typing import final

from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.run_config import MESH_DEVICE_STATE_DICT_KEY, ModelState


class SharedStateAddOn:
    f"""This class adds the shared state functionality to the `AbstractModule`.
    Please put it before any other inherited classes when adding a shared state.
    """

    @classmethod
    def create_shared_state(cls, hf_config: PretrainedConfig, *args, **kwargs) -> ModelState:
        """Create a new shared state for the module.
        This is shared among all or some instances of the module in a model, depending on the model.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device (default implementation only): TTNN mesh device on which to load the weights and instantiate the `MeshDeviceStub`s.

        Returns:
            A new object initializing the state of the module
        """
        return cls._create_state_impl(*args, **kwargs)

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, *args, **kwargs) -> ModelState:
        """Create a new state for the module.
        Subclasses may override this method to initialize the state of the module, which is typically used to
        store persistent model state that is not part of the model configuration or weights.

        Args:
            hf_config: HuggingFace model configuration object

        Returns:
            A new object initializing the state of the module
        """
        return {}

    @final
    @classmethod
    def _create_state_impl(cls, mesh_device: ttnn.Device) -> ModelState:
        """Default implementation of creating a new state for a module."""
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device}
