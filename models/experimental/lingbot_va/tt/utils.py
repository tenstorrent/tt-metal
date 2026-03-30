# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading TT-implemented Lingbot-VA models.

Provides load_transformer, load_transformer_from_state_dict, and load_text_encoder
to instantiate the TTNN WanTransformer3DModel and UMT5Encoder and load weights
from the reference (PyTorch) checkpoints.
"""

from __future__ import annotations

import gc
import os

import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
)
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.lingbot_va.tt.vae_encoder import WanVAEEncoder, patch_wan_causal_conv_wormhole_bf16_parity
from models.experimental.lingbot_va.tt.vae_decoder import WanVAEDecoder
from models.tt_dit.models.vae.vae_wan2_1 import WanCausalConv3d
from models.tt_dit.utils.conv3d import conv_pad_in_channels, conv_pad_height, conv_unpad_height

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


def _safe_deallocate_tensor(tensor, label: str = "") -> None:
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception as e:
        logger.warning("Failed to deallocate{}: {}", f" {label}" if label else "", e)


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
    parallel_config: DiTParallelConfig,
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
    parallel_config: DiTParallelConfig,
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
    parallel_config: EncoderParallelConfig | None = None,
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
    from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
    from transformers import UMT5EncoderModel

    hf_encoder = UMT5EncoderModel.from_pretrained(
        _local_path(text_encoder_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).cpu()
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


def load_vae_encoder(
    van_encoder,
    config,
    mesh_device: "ttnn.MeshDevice",
    torch_dtype: torch.dtype = torch.bfloat16,
    ccl_manager: "CCLManager | None" = None,
    parallel_config: "VaeHWParallelConfig | None" = None,
):
    _ = torch_dtype
    encoder_weights = {k: v.cpu() for k, v in van_encoder.state_dict().items()}
    cfg = config
    del van_encoder

    if parallel_config is None:
        parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )
    if ccl_manager is None:
        ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    vae_encoder = WanVAEEncoder(
        in_channels=cfg.in_channels,
        dim=cfg.base_dim,
        z_dim=cfg.z_dim * 2,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_scales=list(cfg.attn_scales),
        temperal_downsample=list(cfg.temperal_downsample),
        is_residual=cfg.is_residual,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=ttnn.bfloat16,
    )
    vae_encoder.load_torch_state_dict(encoder_weights)
    return vae_encoder


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )
    return x


class WanVAEStreamingWrapper:
    def __init__(
        self,
        vae_model,
        mesh_device: "ttnn.MeshDevice",
        ccl_manager: "CCLManager | None" = None,
        parallel_config: "VaeHWParallelConfig | None" = None,
    ):
        self.vae = vae_model
        self.encoder = load_vae_encoder(
            vae_model.encoder,
            vae_model.config,
            mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.quant_conv = WanCausalConv3d(
            vae_model.config.z_dim * 2,
            vae_model.config.z_dim * 2,
            kernel_size=1,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat16,
        )
        quant_conv_weights = {k: v.cpu() for k, v in vae_model.quant_conv.state_dict().items()}
        self.quant_conv.load_torch_state_dict(quant_conv_weights)
        patch_wan_causal_conv_wormhole_bf16_parity(self.quant_conv, mesh_device)
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        self.enc_conv_num = self.encoder.num_causal_conv_slots
        if hasattr(self.vae, "_cached_conv_counts"):
            ref_n = self.vae._cached_conv_counts.get("encoder")
            if ref_n is not None and ref_n != self.enc_conv_num:
                import warnings

                warnings.warn(
                    f"TT encoder num_causal_conv_slots={self.enc_conv_num} != "
                    f"vae._cached_conv_counts['encoder']={ref_n}; streaming feat_cache may be misaligned.",
                    stacklevel=2,
                )
            if ref_n is not None:
                self.enc_conv_num = max(self.enc_conv_num, ref_n)

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def cleanup_all(self) -> None:
        """Synchronize device, reset streaming feat cache, and encourage GC (does not unload weights)."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception as e:
            logger.warning("WanVAEStreamingWrapper.cleanup_all: synchronize_device failed: {}", e)
        self.clear_cache()
        gc.collect()

    def deallocate_weights(self) -> None:
        """Unload TTNN parameters from device (call before closing the mesh)."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception as e:
            logger.warning("WanVAEStreamingWrapper.deallocate_weights: synchronize_device failed: {}", e)
        try:
            self.encoder.deallocate_weights()
        except Exception as e:
            logger.warning("WanVAEStreamingWrapper.deallocate_weights: encoder failed: {}", e)
        try:
            self.quant_conv.deallocate_weights()
        except Exception as e:
            logger.warning("WanVAEStreamingWrapper.deallocate_weights: quant_conv failed: {}", e)
        gc.collect()

    def __enter__(self) -> WanVAEStreamingWrapper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()

    def encode_chunk(self, x_chunk):
        # Full chunk is self-contained; reset causal feat_cache each call.
        self.clear_cache()
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        video_BTHWC = x_chunk.permute(0, 2, 3, 4, 1)
        video_BTHWC = conv_pad_in_channels(video_BTHWC)
        video_BTHWC, logical_h = conv_pad_height(video_BTHWC, self.parallel_config.height_parallel.factor)
        video_BTHWC = ttnn.from_torch(
            video_BTHWC,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )
        enc_tt = None
        out_tt = None
        try:
            out_tt, logical_h = self.encoder(video_BTHWC, logical_h, feat_cache=self.feat_cache, feat_idx=feat_idx)
            enc_tt = self.quant_conv(out_tt, logical_h)
            enc_h = enc_tt.shape[2]
            enc_tt = conv_unpad_height(enc_tt, min(logical_h, enc_h))
            ttnn.synchronize_device(self.mesh_device)
            enc = ttnn.to_torch(enc_tt)
            enc = enc.permute(0, 4, 1, 2, 3)
            return enc
        finally:
            _safe_deallocate_tensor(video_BTHWC, "encode_chunk video_BTHWC")
            _safe_deallocate_tensor(out_tt, "encode_chunk encoder out")
            _safe_deallocate_tensor(enc_tt, "encode_chunk enc")
            gc.collect()


def load_vae_decoder(
    vae_model,
    mesh_device: "ttnn.MeshDevice",
    ccl_manager: "CCLManager | None" = None,
    parallel_config: "VaeHWParallelConfig | None" = None,
):
    """Build TTNN WanVAEDecoder and load weights from the PyTorch VAE decoder."""
    state = {k: v.cpu() for k, v in vae_model.state_dict().items()}
    cfg = vae_model.config
    if parallel_config is None:
        parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )
    if ccl_manager is None:
        ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_decoder = WanVAEDecoder(
        base_dim=cfg.base_dim,
        decoder_base_dim=getattr(cfg, "decoder_base_dim", None),
        z_dim=cfg.z_dim,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_scales=list(cfg.attn_scales),
        temperal_downsample=list(cfg.temperal_downsample),
        out_channels=cfg.out_channels,
        patch_size=getattr(cfg, "patch_size", 1) or 1,
        latents_mean=list(cfg.latents_mean),
        latents_std=list(cfg.latents_std),
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_decoder.load_torch_state_dict(state)
    return tt_decoder


class WanVAEDecoderWrapper:
    """Wrapper around TTNN WanVAEDecoder that exposes decode(latents) -> video tensor for inference."""

    def __init__(
        self,
        vae_model,
        mesh_device: "ttnn.MeshDevice",
        ccl_manager: "CCLManager | None" = None,
        parallel_config: "VaeHWParallelConfig | None" = None,
    ):
        self.vae = vae_model
        self.parallel_config = parallel_config or VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager or CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        self.decoder = load_vae_decoder(
            vae_model, mesh_device, ccl_manager=self.ccl_manager, parallel_config=self.parallel_config
        )

    def cleanup_all(self) -> None:
        """Synchronize device and encourage GC (does not unload weights)."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception as e:
            logger.warning("WanVAEDecoderWrapper.cleanup_all: synchronize_device failed: {}", e)
        gc.collect()

    def deallocate_weights(self) -> None:
        """Unload TTNN decoder parameters from device (call before closing the mesh)."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception as e:
            logger.warning("WanVAEDecoderWrapper.deallocate_weights: synchronize_device failed: {}", e)
        try:
            self.decoder.deallocate_weights()
        except Exception as e:
            logger.warning("WanVAEDecoderWrapper.deallocate_weights: decoder failed: {}", e)
        gc.collect()

    def __enter__(self) -> WanVAEDecoderWrapper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents (B, C, T, H, W) to video (B, C, T, H, W). Latents are normalized with mean/std before calling."""
        latents = latents.to(self.vae.dtype)
        tt_latents_BTHWC = latents.permute(0, 2, 3, 4, 1)
        tt_latents_BTHWC = conv_pad_in_channels(tt_latents_BTHWC)
        tt_latents_BTHWC, logical_h = conv_pad_height(tt_latents_BTHWC, self.parallel_config.height_parallel.factor)
        tt_latents_BTHWC = ttnn.from_torch(
            tt_latents_BTHWC,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )
        tt_video_BCTHW = None
        try:
            tt_video_BCTHW, new_logical_h = self.decoder(tt_latents_BTHWC, logical_h)
            ttnn.synchronize_device(self.mesh_device)
            video_torch = ttnn.to_torch(tt_video_BCTHW)
            video_torch = video_torch[:, :, :, :new_logical_h, :]
            ps = getattr(self.vae.config, "patch_size", None)
            if ps and ps > 1:
                B, Cp, T_out, H_out, W_out = video_torch.shape
                channels = Cp // (ps * ps)
                video_torch = video_torch.view(B, channels, ps, ps, T_out, H_out, W_out)
                video_torch = video_torch.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
                video_torch = video_torch.view(B, channels, T_out, H_out * ps, W_out * ps)
            video_torch = video_torch.clamp(-1.0, 1.0)
            return video_torch
        finally:
            _safe_deallocate_tensor(tt_latents_BTHWC, "decode latents")
            _safe_deallocate_tensor(tt_video_BCTHW, "decode video")
            gc.collect()


__all__ = [
    "load_transformer",
    "load_transformer_from_state_dict",
    "load_text_encoder",
    "load_vae_encoder",
    "load_vae_decoder",
    "patchify",
    "WanVAEStreamingWrapper",
    "WanVAEDecoderWrapper",
]
