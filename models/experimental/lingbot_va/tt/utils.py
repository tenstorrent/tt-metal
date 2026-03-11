# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading TT-implemented Lingbot-VA models.

Provides load_transformer and load_transformer_from_state_dict to instantiate the
TTNN WanTransformer3DModel and load weights from the reference (PyTorch) checkpoint
or from an existing state dict.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from .transformer_wan import (
    CROSS_ATTN_NORM,
    DIM,
    EPS,
    FFN_DIM,
    FREQ_DIM,
    IN_CHANNELS,
    NUM_HEADS,
    NUM_LAYERS,
    OUT_CHANNELS,
    PATCH_SIZE,
    ROPE_MAX_SEQ_LEN,
    TEXT_DIM,
    WanTransformer3DModel,
)

if TYPE_CHECKING:
    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager


# Lazy import to avoid loading reference/diffusers when only load_transformer_from_state_dict is used
def _get_torch_wan_transformer():
    from models.experimental.lingbot_va.reference.transformer_wan import (
        WanTransformer3DModel as TorchWanTransformer3DModel,
    )

    return TorchWanTransformer3DModel


def _local_path(p: str | os.PathLike) -> str:
    """Resolve to absolute path for from_pretrained so it is treated as local, not a HF repo id."""
    return str(os.path.abspath(os.path.expanduser(p)))


def load_transformer(
    transformer_path: str | os.PathLike,
    mesh_device: "ttnn.MeshDevice",
    parallel_config: "DiTParallelConfig",
    *,
    ccl_manager: "CCLManager | None" = None,
    num_layers: int | None = None,
    is_fsdp: bool = False,
    torch_dtype: torch.dtype = torch.float32,
) -> WanTransformer3DModel:
    """
    Build TTNN WanTransformer3DModel and load weights from the reference checkpoint.

    Loads the reference (PyTorch) WanTransformer3DModel from transformer_path to obtain
    the state dict, then builds the TT model and loads the state (with key/weight mapping
    applied via _prepare_torch_state). Use this when you have a diffusers-format
    transformer checkpoint on disk.

    Args:
        transformer_path: Path to the reference transformer checkpoint (diffusers format).
        mesh_device: TTNN mesh device for the model.
        parallel_config: DiT parallel config (tensor/sequence parallel).
        ccl_manager: Optional CCL manager for multi-device. Can be None for (1,1).
        num_layers: Number of transformer layers. Defaults to NUM_LAYERS from transformer_wan.
        is_fsdp: Whether FSDP is used (affects layer norm placement).
        torch_dtype: Dtype used to load the reference model (state dict is then applied to TT).

    Returns:
        WanTransformer3DModel (TT) with weights loaded.
    """
    TorchWanTransformer3DModel = _get_torch_wan_transformer()
    torch_model = TorchWanTransformer3DModel.from_pretrained(
        _local_path(transformer_path),
        torch_dtype=torch_dtype,
        attn_mode="torch",
        local_files_only=True,
    )
    state_dict = {k: v.cpu() for k, v in torch_model.state_dict().items()}
    del torch_model
    return load_transformer_from_state_dict(
        state_dict,
        mesh_device,
        parallel_config,
        ccl_manager=ccl_manager,
        num_layers=num_layers,
        is_fsdp=is_fsdp,
    )


def load_transformer_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    mesh_device: "ttnn.MeshDevice",
    parallel_config: "DiTParallelConfig",
    *,
    ccl_manager: "CCLManager | None" = None,
    num_layers: int | None = None,
    is_fsdp: bool = False,
) -> WanTransformer3DModel:
    """
    Build TTNN WanTransformer3DModel and load weights from a PyTorch state dict.

    The state dict must match the reference WanTransformer3DModel (e.g. from
    TorchWanTransformer3DModel.state_dict()). Key/weight mapping for the TT model
    (e.g. patch_embedding_mlp -> patch_embedding) is applied via _prepare_torch_state.

    Args:
        state_dict: PyTorch state dict from the reference WanTransformer3DModel.
        mesh_device: TTNN mesh device for the model.
        parallel_config: DiT parallel config (tensor/sequence parallel).
        ccl_manager: Optional CCL manager for multi-device. Can be None for (1,1).
        num_layers: Number of transformer layers. Defaults to NUM_LAYERS from transformer_wan.
        is_fsdp: Whether FSDP is used (affects layer norm placement).

    Returns:
        WanTransformer3DModel (TT) with weights loaded.
    """
    import ttnn  # avoid top-level ttnn for environments that only need torch

    n_layers = num_layers if num_layers is not None else NUM_LAYERS
    # Copy so we can mutate for _prepare_torch_state
    state = dict(state_dict)

    tt_model = WanTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        action_dim=30,  # ACTION_DIM
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=n_layers,
        cross_attn_norm=CROSS_ATTN_NORM,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model._prepare_torch_state(state)
    tt_model.load_torch_state_dict(state)
    return tt_model


__all__ = [
    "load_transformer",
    "load_transformer_from_state_dict",
    "_local_path",
]
