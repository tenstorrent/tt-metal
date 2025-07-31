# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class DecoderLayer(AbstractModule):
    """
    x += attn(attn_norm(x))
    x += ffn(ffn_norm(x))
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
        mesh_row: ttnn.MeshDevice,
    ) -> WeightConfig:
        cls.validate_mesh_devices(mesh_device, mesh_row)
        return {
            "mla_norm": DistributedRMSNorm.convert_weights(
                hf_config, sub_state_dict(state_dict, "input_layernorm."), output_path / "mla_norm", mesh_row
            ),
            "mla": MLA1D.convert_weights(
                hf_config, sub_state_dict(state_dict, "self_attn."), output_path / "mla", mesh_row
            ),
            "mlp_norm": DistributedRMSNorm.convert_weights(
                hf_config, sub_state_dict(state_dict, "post_attention_layernorm."), output_path / "mlp_norm", mesh_row
            ),
            "mlp": NonExpert.convert_weights(
                hf_config, sub_state_dict(state_dict, "mlp."), output_path / "mlp", mesh_row
            ),
        }

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, mesh_row: ttnn.MeshDevice
    ) -> ModelPrefillConfig:
        cls.validate_mesh_devices(mesh_device, mesh_row)
        return {
            "mla_norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_row),
            "mla": MLA1D.prefill_model_config(hf_config, mesh_row),
            "mlp_norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_row),
            "mlp": NonExpert.prefill_model_config(hf_config, mesh_row),
        }

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, mesh_row: ttnn.MeshDevice
    ) -> ModeldecodeConfig:
        cls.validate_mesh_devices(mesh_device, mesh_row)
        return {
            "mla_norm": DistributedRMSNorm.decode_model_config(hf_config, mesh_row),
            "mla": MLA1D.decode_model_config(hf_config, mesh_row),
            "mlp_norm": DistributedRMSNorm.decode_model_config(hf_config, mesh_row),
            "mlp": NonExpert.decode_model_config(hf_config, mesh_row),
        }

    @classmethod
    def create_state(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, mesh_row: ttnn.MeshDevice
    ) -> ModelState:
        cls.validate_mesh_devices(mesh_device, mesh_row)
        return {
            "mla_norm": DistributedRMSNorm.create_state(hf_config, mesh_row),
            "mla": MLA1D.create_state(hf_config, mesh_row),
            "mlp_norm": DistributedRMSNorm.create_state(hf_config, mesh_row),
            "mlp": NonExpert.create_state(hf_config, mesh_row),
        }

    @classmethod
    def validate_mesh_devices(cls, mesh_device: ttnn.MeshDevice, mesh_row: ttnn.MeshDevice) -> None:
        assert tuple(mesh_device.shape) == (4, 8), "decoder layer runs on a full galaxy"
        assert tuple(mesh_row.shape) == (1, 8), "the mesh row that the norms and MLP are supposed to run on must be 1x8"
        assert set(mesh_row.get_device_ids()).issubset(
            mesh_device.get_device_ids()
        ), "the mesh row must be a subdevice of the mesh device"

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        raise NotImplementedError("TODO")

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        raise NotImplementedError("TODO")
