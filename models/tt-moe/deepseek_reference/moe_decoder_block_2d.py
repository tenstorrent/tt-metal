# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)

from .ccl import CCL
from .decoder_block_2d_base import DecoderBlock2DBase
from .moe import MoE
from .shared_expert import SharedExpert


class MoEDecoderBlock2D(DecoderBlock2DBase):
    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return {
            "shared_expert": SharedExpert.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "shared_experts."),) * mesh_device.shape[0],
                output_path / "shared_experts",
                mesh_device,
            ),
            "moe": MoE.convert_weights(hf_config, (state_dict,), output_path / "moe", mesh_device),
        }

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return {
            "shared_expert": SharedExpert.prefill_model_config(hf_config, mesh_device),
            "moe": MoE.prefill_model_config(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        return {
            "shared_expert": SharedExpert.decode_model_config(hf_config, mesh_device),
            "moe": MoE.decode_model_config(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        return {
            "shared_expert": SharedExpert.create_state(hf_config, mesh_device, ccl),
            "moe": MoE.create_state(hf_config, mesh_device, ccl),
        }

    @classmethod
    @abstractmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return {
            "shared_expert": {},
            "moe": MoE.create_shared_state(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        mlp_out = MoE.forward_prefill(x, cfg["moe"])
        mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"])
        return mlp_out

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        mlp_out = MoE.forward_decode(x, cfg["moe"])
        mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"])

        # INSTRUMENTATION: Save MoE+SharedExpert output for comparison
        import os

        if os.environ.get("SAVE_MOE_DECODER_OUTPUT", "0") == "1":
            from pathlib import Path

            import numpy as np

            save_dir = Path("/tmp/moe_decoder_copied_output")
            save_dir.mkdir(parents=True, exist_ok=True)

            # Convert to numpy and save - need mesh composer for distributed tensor
            import ttnn

            mesh_device = mlp_out.device() if hasattr(mlp_out, "device") else None
            if mesh_device and hasattr(mesh_device, "shape"):
                output_torch = ttnn.to_torch(
                    mlp_out,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )
            else:
                output_torch = ttnn.to_torch(mlp_out)
            output_np = output_torch.cpu().float().numpy()
            np.save(save_dir / "moe_decoder_output.npy", output_np)

            # Also save hash for quick comparison
            import hashlib

            output_bytes = output_np.tobytes()
            output_hash = hashlib.md5(output_bytes).hexdigest()
            with open(save_dir / "moe_decoder_hash.txt", "w") as f:
                f.write(output_hash)

            from loguru import logger

            logger.info(f"[Copied MoEDecoderBlock2D] Saved output to {save_dir}")
            logger.info(f"  Output shape: {output_torch.shape}, dtype: {output_torch.dtype}")
            logger.info(f"  MD5 hash: {output_hash}")

        return mlp_out
