# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.wormhole.patchtst.reference.hf_reference import ReferenceArtifacts
from models.demos.wormhole.patchtst.tt.common import MEMORY_CONFIG_BY_TIER, PatchTSTRuntimePolicy


def build_sincos_position_encoding(num_positions: int, d_model: int, device: torch.device) -> torch.Tensor:
    if num_positions <= 0:
        raise ValueError(f"num_positions must be positive, got {num_positions}")
    position = torch.arange(num_positions, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros((num_positions, d_model), dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@dataclass
class PatchTSTPositionalEncoding:
    position_enc: torch.Tensor
    cls_token: torch.Tensor | None
    use_cls_token: bool
    positional_encoding_type: str
    num_input_channels: int

    def __call__(self, embedding: ttnn.Tensor, device, runtime: PatchTSTRuntimePolicy) -> ttnn.Tensor:
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        num_patch_tokens = int(embedding.shape[2])
        expected_tokens = num_patch_tokens + (1 if self.use_cls_token else 0)
        pos = self.position_enc
        if int(pos.shape[0]) != expected_tokens:
            if self.positional_encoding_type != "sincos":
                raise ValueError(
                    "Runtime patch count does not match checkpoint positional table and checkpoint is not sincos. "
                    f"runtime_tokens={expected_tokens}, checkpoint_tokens={pos.shape[0]}"
                )
            pos = build_sincos_position_encoding(expected_tokens, int(pos.shape[-1]), pos.device)

        if self.use_cls_token:
            if self.cls_token is None:
                raise KeyError("use_cls_token=True but cls_token is missing from checkpoint.")
            patch_pos_tt = ttnn.from_torch(
                pos[1:].reshape(1, 1, num_patch_tokens, -1).expand(embedding.shape[0], self.num_input_channels, -1, -1),
                dtype=embedding.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mem_cfg,
            )
            patch_part = ttnn.add(embedding, patch_pos_tt, memory_config=mem_cfg)
            ttnn.deallocate(patch_pos_tt)
            cls_tt = ttnn.from_torch(
                (self.cls_token + pos[:1])
                .reshape(1, 1, 1, -1)
                .expand(embedding.shape[0], self.num_input_channels, -1, -1),
                dtype=embedding.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mem_cfg,
            )
            encoded = ttnn.concat([cls_tt, patch_part], dim=2)
            ttnn.deallocate(cls_tt)
            ttnn.deallocate(patch_part)
            return encoded

        pos_tt = ttnn.from_torch(
            pos.reshape(1, 1, num_patch_tokens, -1).expand(embedding.shape[0], self.num_input_channels, -1, -1),
            dtype=embedding.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_cfg,
        )
        encoded = ttnn.add(embedding, pos_tt, memory_config=mem_cfg)
        ttnn.deallocate(pos_tt)
        return encoded


def build_positional_encoding(reference: ReferenceArtifacts) -> PatchTSTPositionalEncoding:
    state = reference.model.state_dict()
    return PatchTSTPositionalEncoding(
        position_enc=state["model.encoder.positional_encoder.position_enc"].detach().to(torch.float32),
        cls_token=(
            state["model.encoder.positional_encoder.cls_token"].detach().to(torch.float32)
            if "model.encoder.positional_encoder.cls_token" in state
            else None
        ),
        use_cls_token=bool(reference.config.use_cls_token),
        positional_encoding_type=str(reference.config.positional_encoding_type),
        num_input_channels=int(reference.model.model.encoder.positional_encoder.num_input_channels),
    )
