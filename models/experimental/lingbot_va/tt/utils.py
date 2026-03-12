# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading TT-implemented Lingbot-VA models.

Provides load_transformer, load_transformer_from_state_dict, and load_text_encoder
to instantiate the TTNN WanTransformer3DModel and UMT5Encoder and load weights
from the reference (PyTorch) checkpoints.
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
    from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig
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


def load_text_encoder(
    text_encoder_path: str | os.PathLike,
    mesh_device: "ttnn.MeshDevice",
    *,
    ccl_manager: "CCLManager | None" = None,
    parallel_config: "EncoderParallelConfig | None" = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    max_prompt_length: int = 512,
):
    """
    Build TTNN UMT5Encoder (from models.tt_dit.encoders.umt5.model_umt5) and load weights
    from the HuggingFace UMT5EncoderModel checkpoint at text_encoder_path.

    Same pattern as test_encoder_wan.py: load HF model to get config and state dict,
    build UMT5Config and TTUMT5Encoder, then load_torch_state_dict.

    Args:
        text_encoder_path: Path to the HuggingFace UMT5EncoderModel checkpoint.
        mesh_device: TTNN mesh device for the model.
        ccl_manager: Optional CCL manager. If None, created for (1,1) mesh.
        parallel_config: Optional encoder parallel config. If None, tensor_parallel factor=1.
        torch_dtype: Dtype for loading the HF model (state dict is then applied to TT).
        max_prompt_length: max_prompt_length for UMT5Config.

    Returns:
        TT UMT5Encoder with weights loaded.
    """
    import ttnn

    from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    from transformers import UMT5EncoderModel

    hf_encoder = UMT5EncoderModel.from_pretrained(
        _local_path(text_encoder_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device="cpu")
    hf_encoder.eval()
    text_weights = {k: v.cpu() for k, v in hf_encoder.state_dict().items()}
    cfg = hf_encoder.config
    del hf_encoder

    umt5_config = UMT5Config(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.d_model,
        ff_dim=cfg.d_ff,
        kv_dim=cfg.d_kv,
        num_heads=cfg.num_heads,
        num_hidden_layers=cfg.num_layers,
        max_prompt_length=max_prompt_length,
        layer_norm_eps=cfg.layer_norm_epsilon,
        relative_attention_num_buckets=cfg.relative_attention_num_buckets,
        relative_attention_max_distance=cfg.relative_attention_max_distance,
    )

    if parallel_config is None:
        parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[1], mesh_axis=1)
        )
    if ccl_manager is None:
        ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    tt_encoder = TTUMT5Encoder(
        config=umt5_config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_encoder.load_torch_state_dict(text_weights)
    return tt_encoder


__all__ = [
    "load_transformer",
    "load_transformer_from_state_dict",
    "load_text_encoder",
    "_local_path",
]
