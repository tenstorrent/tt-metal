# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Swin Transformer backbone for MaskFormer.

This module will wrap the TT-NN kernels required to execute the Swin-Base
backbone that feeds MaskFormer.  The engineering plan targets the following
feature set:

* Patch embedding implemented via ``ttnn.Conv2d`` (with activation fusion
  toggles mirroring the YOLOv4 TT-CNN builder patterns).
* Window partition / reverse utilities that operate on TT-NN tensors without
  incurring unsupported 6-D permutes.
* Windowed and shifted-window attention blocks implemented with
  ``ttnn.transformer.scaled_dot_product_attention`` including relative position
  bias injections.
* Patch merging downsample stages that keep layouts aligned with Swin's channel
  ordering.

Only the high‑level API is stubbed today to unblock scaffolding of dependent
modules. For bounty #30876 the backbone executes via the Hugging Face CPU
fallback; TT kernels for Swin stages are tracked as follow‑up work and are not
part of the bounty acceptance. Any "Future" notes below indicate optional
enhancements rather than missing requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Any, List
import os
import numpy as np

import math
import torch
import torch.nn.functional as F
import warnings


try:
    from models.common.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm, is_blackhole
except ModuleNotFoundError:  # pragma: no cover - optional if running outside repo context.
    tt_to_torch_tensor = None
    torch_to_tt_tensor_rm = None

    def is_blackhole() -> bool:
        return False


from .weights import extract_backbone_state, extract_patch_embed_state
from .ttnn_compat import ttnn, get_default_dtype, require_ttnn

try:
    from models.tt_cnn.tt.builder import (
        Conv2dConfiguration,
        HeightShardedStrategyConfiguration,
        HeightSliceStrategyConfiguration,
        L1FullSliceStrategyConfiguration,
        to_conv2d_config,
        to_compute_config,
        to_slice_config,
    )
except ModuleNotFoundError:  # pragma: no cover - TT-CNN helpers unavailable without TTNN
    Conv2dConfiguration = None  # type: ignore[assignment]
    HeightShardedStrategyConfiguration = None  # type: ignore[assignment]
    HeightSliceStrategyConfiguration = None  # type: ignore[assignment]
    L1FullSliceStrategyConfiguration = None  # type: ignore[assignment]
    to_conv2d_config = None  # type: ignore[assignment]
    to_compute_config = None  # type: ignore[assignment]
    to_slice_config = None  # type: ignore[assignment]

DEFAULT_TT_DTYPE = get_default_dtype()
TILE_WIDTH = 32


def _make_compute_kernel_config(
    *,
    math_fidelity,
    math_approx_mode: bool,
    fp32_dest_acc_en: bool,
    packer_l1_acc: bool,
):
    """Select an appropriate compute kernel config for the current arch."""

    if ttnn is None:
        raise RuntimeError("TT-NN runtime is required to construct Swin compute kernel configs.")
    ComputeConfigClass = ttnn.WormholeComputeKernelConfig
    try:
        if is_blackhole() and hasattr(ttnn, "types") and hasattr(ttnn.types, "BlackholeComputeKernelConfig"):
            ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig  # type: ignore[assignment]
    except Exception:
        # Conservatively fall back to Wormhole config on detection errors.
        pass
    return ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


@dataclass
class SwinBackboneConfig:
    """Configuration holder for the Swin-Base backbone."""

    image_size: Tuple[int, int] = (384, 384)
    patch_size: int = 4
    embed_dim: int = 128
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (4, 8, 16, 32)
    window_size: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    use_checkpoint: bool = False
    layer_norm_eps: float = 1e-5
    num_channels: int = 3
    out_features: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    out_indices: Tuple[int, ...] = (1, 2, 3, 4)

    @classmethod
    def from_hf_dict(cls, payload: Dict[str, object]) -> "SwinBackboneConfig":
        """Instantiate from a HuggingFace `backbone_config` dictionary."""

        image_size_value = payload.get("image_size", 384)
        if isinstance(image_size_value, (list, tuple)):
            if len(image_size_value) >= 2:
                image_size = (int(image_size_value[0]), int(image_size_value[1]))
            else:
                image_size = (int(image_size_value[0]), int(image_size_value[0]))
        else:
            image_size = (int(image_size_value), int(image_size_value))

        return cls(
            image_size=image_size,
            patch_size=int(payload.get("patch_size", 4)),
            embed_dim=int(payload.get("embed_dim", 128)),
            depths=tuple(int(x) for x in payload.get("depths", (2, 2, 18, 2))),
            num_heads=tuple(int(x) for x in payload.get("num_heads", (4, 8, 16, 32))),
            window_size=int(payload.get("window_size", 12)),
            mlp_ratio=float(payload.get("mlp_ratio", 4.0)),
            qkv_bias=bool(payload.get("qkv_bias", True)),
            drop_path_rate=float(payload.get("drop_path_rate", 0.0)),
            use_checkpoint=bool(payload.get("use_checkpoint", False)),
            layer_norm_eps=float(payload.get("layer_norm_eps", 1e-5)),
            num_channels=int(payload.get("num_channels", payload.get("in_channels", 3))),
            out_features=tuple(payload.get("out_features", ("stage1", "stage2", "stage3", "stage4"))),
            out_indices=tuple(payload.get("out_indices", (1, 2, 3, 4))),
        )

    def to_hf_dict(self) -> Dict[str, object]:
        """Convert to a dictionary consumable by `MaskFormerSwinConfig`."""

        return {
            "image_size": list(self.image_size),
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": list(self.depths),
            "num_heads": list(self.num_heads),
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "drop_path_rate": self.drop_path_rate,
            "use_checkpoint": self.use_checkpoint,
            "layer_norm_eps": self.layer_norm_eps,
            "num_channels": self.num_channels,
            "out_features": list(self.out_features),
            "out_indices": list(self.out_indices),
        }


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass
class SwinAttentionWeights:
    query: LinearWeights
    key: LinearWeights
    value: LinearWeights
    proj: LinearWeights
    relative_position_bias_table: torch.Tensor


@dataclass
class SwinMLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass
class SwinBlockWeights:
    norm_before: LayerNormWeights
    attention: SwinAttentionWeights
    norm_after: LayerNormWeights
    mlp: SwinMLPWeights


class MaskFormerSwinBackbone:
    """
    TT-NN Swin backbone entry-point.

    Notes
    -----
    * Methods prefixed with ``_init_`` follow the TT-NN paradigm of allocating
      weight tensors up-front.  Placeholder implementations raise ``NotImplementedError``
      until hooked up.
    * ``forward`` returns the multi-scale feature maps expected by the pixel
      decoder (typically ``C2``–``C5`` features).
    * ``from_huggingface`` will ingest HuggingFace weights via the helper
      functions exposed in ``weights.py`` once implemented.
    """

    def __init__(
        self,
        config: SwinBackboneConfig,
        device: Optional[object],
        *,
        dtype: Optional[object] = DEFAULT_TT_DTYPE,
        enable_profiling: bool = False,
    ) -> None:
        if device is not None and ttnn is None:
            require_ttnn("allocate the Swin backbone on a TT device")
        self.config = config
        self.device = device
        self.dtype = dtype
        self.enable_profiling = enable_profiling
        self._torch_device = torch.device("cpu")
        # Stage metadata mirrors the reference HuggingFace implementation and is
        # intentionally kept explicit to simplify parity against
        # ``models.experimental.swin`` modules (see ``models/experimental/swin/tt``).
        self.stage_depths = config.depths
        self.stage_dims = [config.embed_dim * (2**i) for i in range(len(self.stage_depths))]
        self.stage_heads = config.num_heads
        self.window_size = config.window_size

        # Future: instantiate TT‑NN kernels by adapting:
        # * ``models.experimental.swin.tt.swin_patch_embedding.TtSwinPatchEmbedding``
        # * ``models.experimental.swin.tt.swin_stage.TtSwinStage``
        # The newer implementation should avoid nn.Module usage and instead rely on
        # TT‑NN tensor ops for lighter‑weight graph construction. Not required for
        # bounty #30876, which exercises the TT decoder + heads.

        self._weights: Dict[str, Any] = {}
        self._patch_embed_state: Dict[str, Any] = {}
        self._patch_embed_spec: Optional[PatchEmbeddingSpec] = None
        self._patch_embed_kernel: Optional[Dict[str, Any]] = None
        self._stage_weights: Dict[str, Dict[str, Any]] = {}
        self._torch_backbone_state: Dict[str, Any] = {}
        self._hf_backbone_model = None
        self._hf_config_dict: Dict[str, object] = {}
        self._stage_block_params: Dict[int, List[_StageBlockParams]] = {}
        self._patch_merge_params: Dict[int, Optional[_PatchMergingParams]] = {}
        self._attn_mask_cache: Dict[Tuple[int, int, Tuple[int, int], int], Any] = {}
        self._hf_stage_plans: List[SwinStagePlan] = self._build_stage_plans()
        self._init_patch_embedding()
        self._init_stages()

    def _debug(self, message: str) -> None:
        if self.enable_profiling:
            print(f"[maskformer][tt-debug] {message}", flush=True)

    @classmethod
    def from_huggingface(
        cls,
        weights: Dict[str, object],
        device: Optional[object],
        *,
        config_dict: Optional[Dict[str, object]] = None,
        config: Optional[SwinBackboneConfig] = None,
    ) -> "MaskFormerSwinBackbone":
        """Factory consuming converted HuggingFace weights (see ``weights.py``)."""

        if config is None:
            if config_dict is None:
                config = SwinBackboneConfig()
            else:
                config = SwinBackboneConfig.from_hf_dict(config_dict)
        backbone = cls(config=config, device=device)
        merged_config = config.to_hf_dict()
        if config_dict:
            merged_config.update({k: v for k, v in config_dict.items() if k not in merged_config})
        backbone._hf_config_dict = merged_config
        backbone.load_weights(weights)
        return backbone

    def load_weights(self, weights: Dict[str, object]) -> None:
        """
        Load TT-compatible weights into the backbone.

        Parameters
        ----------
        weights:
            Mapping from canonical Swin parameter names to TT-NN tensors or host
            arrays.  The ``weights`` module will expose utilities to generate
            this mapping from HuggingFace checkpoints.
        """

        backbone_prefix = "model.pixel_level_module.encoder.model."
        filtered = {name: tensor for name, tensor in weights.items() if name.startswith(backbone_prefix)}

        if not filtered:
            raise ValueError("Provided weight dictionary does not contain Swin backbone parameters.")

        self._weights = filtered
        self._patch_embed_state = extract_patch_embed_state(weights)
        self._patch_embed_spec = self._derive_patch_embed_spec()
        self._prepare_patch_embed_kernels()
        self._stage_weights = self._partition_stage_weights(filtered)
        self._torch_backbone_state = extract_backbone_state(weights)
        self._build_torch_backbone()

    def forward(self, images: Any) -> Iterable[torch.Tensor]:
        """
        Run the Swin backbone and return FPN-ready feature maps.

        Returns
        -------
        tuple(list(torch.Tensor), list(torch.Tensor))
            Feature tensors ordered by scale, plus intermediate encoder states.
        """

        if not self._weights:
            raise RuntimeError("Backbone weights not loaded. Call `load_weights` first.")

        if self._hf_backbone_model is None:
            raise NotImplementedError("TT-NN execution pending; PyTorch fallback unavailable (transformers missing).")

        torch_images = self._ensure_torch_tensor(images).to(self._torch_device)
        with torch.no_grad():
            outputs = self._hf_backbone_model(torch_images)
        feature_maps = list(outputs.feature_maps)

        hidden_states = getattr(outputs, "hidden_states", None)
        encoder_hidden = (
            [self._ensure_torch_tensor(t) for t in hidden_states]
            if isinstance(hidden_states, (list, tuple))
            else feature_maps
        )
        return feature_maps, encoder_hidden

    # ------------------------------------------------------------------
    # Patch embedding execution helpers
    # ------------------------------------------------------------------

    def _patch_embed_forward_tt_raw(self, images: torch.Tensor):
        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT patch embedding unavailable (device or TT runtime missing).")

        padded_images, _ = self._pad_images_for_patch(images)
        tt_input = self._convert_images_to_tt(padded_images)
        spec = self.get_patch_embed_spec()
        if spec is None:
            raise RuntimeError("Patch embedding spec missing.")
        self._debug(f"patch_embed: padded_images={tuple(padded_images.shape)} tt_input={tuple(tt_input.shape)}")

        conv_kwargs, compute_config = self._build_patch_conv_params(tt_input, spec)
        self._debug(
            f"patch_embed: input={tuple(tt_input.shape)} layout={tt_input.get_layout()} "
            f"conv_out={spec.out_channels} stride={spec.stride}"
        )
        output_tensor, (out_height, out_width) = ttnn.conv2d(
            input_tensor=tt_input,
            weight_tensor=self._patch_embed_kernel["weight"],
            bias_tensor=self._patch_embed_kernel["bias"],
            return_output_dim=True,
            return_weights_and_bias=False,
            compute_config=compute_config,
            dtype=self.dtype or get_default_dtype(),
            **conv_kwargs,
        )
        self._debug(
            f"patch_embed: conv_done shape={tuple(output_tensor.shape)} layout={output_tensor.get_layout()} "
            f"dims=({out_height},{out_width})"
        )
        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            self._debug("patch_embed: converting conv output to ROW_MAJOR")
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        self._debug(f"patch_embed: output ready shape={tuple(output_tensor.shape)} layout={output_tensor.get_layout()}")
        return output_tensor, (out_height, out_width)

    def run_patch_embedding_tt(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Execute the patch embedding layer on a TT device and return torch tensors."""

        output_tensor, (out_height, out_width) = self._patch_embed_forward_tt_raw(images)
        # Reshape to sequence and apply LayerNorm to mirror HF embeddings
        sequence = self._reshape_patch_embeddings(output_tensor, out_height, out_width)
        if (
            self._patch_embed_kernel
            and "norm_weight" in self._patch_embed_kernel
            and "norm_bias" in self._patch_embed_kernel
        ):
            sequence = self._layer_norm(
                sequence,
                self._patch_embed_kernel["norm_weight"],
                self._patch_embed_kernel["norm_bias"],
                self.config.layer_norm_eps,
            )
        torch_embeddings = self._tt_tensor_to_torch(sequence)
        return torch_embeddings.cpu(), (out_height, out_width)

    # ------------------------------------------------------------------
    # Stage execution helpers
    # ------------------------------------------------------------------

    def run_stage1_tt(self, images: torch.Tensor) -> torch.Tensor:
        """Execute patch embedding + Stage 1 blocks on TT hardware and return stage outputs (B, C, H, W)."""

        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT stage execution unavailable (device or TT runtime missing).")
        use_fallback = os.environ.get("MASKFORMER_PATCH_EMBED_FALLBACK") == "1"
        if use_fallback:
            # Use HF embeddings for parity isolation
            hf_embeds, dims = self.run_patch_embedding_hf(images)
            height, width = dims
            seq_tt = ttnn.from_torch(
                hf_embeds.to(self._torch_device),
                dtype=self.dtype or get_default_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            stage_tt, stage_torch, _, _ = self._run_stage_tt(
                stage_index=0, hidden_states=seq_tt, height=height, width=width
            )
            _ = stage_tt
            return stage_torch
        else:
            output_tensor, (height, width) = self._patch_embed_forward_tt_raw(images)
            stage_input = self._reshape_patch_embeddings(output_tensor, height, width)
            # Apply LayerNorm after patch embedding to match HF embeddings behavior
            if (
                self._patch_embed_kernel
                and "norm_weight" in self._patch_embed_kernel
                and "norm_bias" in self._patch_embed_kernel
            ):
                stage_input = self._layer_norm(
                    stage_input,
                    self._patch_embed_kernel["norm_weight"],
                    self._patch_embed_kernel["norm_bias"],
                    self.config.layer_norm_eps,
                )
            stage_tt, stage_torch, _, _ = self._run_stage_tt(
                stage_index=0, hidden_states=stage_input, height=height, width=width
            )
            _ = stage_tt  # reserved for future chaining when Stage 2 is implemented
            return stage_torch

    def _reshape_patch_embeddings(self, tensor, height: int, width: int):
        sequence = self._to_row_major(tensor)
        batch = sequence.shape[0]
        channels = sequence.shape[-1]
        sequence = self._reshape_tensor(sequence, (batch, height * width, channels), tag="patch_embed_seq")
        # Keep large activations in DRAM to avoid L1 OOM.
        return self._to_tile_dram(sequence)

    def _run_stage_tt(
        self, stage_index: int, hidden_states, height: int, width: int, taps_out: Optional[List[torch.Tensor]] = None
    ):
        block_params = self._prepare_stage_blocks(stage_index)
        state = hidden_states
        for block_idx, params in enumerate(block_params):
            state = self._run_swin_block_tt(stage_index, block_idx, params, state, height, width)
            if taps_out is not None:
                taps_out.append(self._stage_output_to_torch(state, height, width))

        stage_feature_torch = self._stage_output_to_torch(state, height, width)
        patch_merge = self._prepare_patch_merging(stage_index)
        merged = None
        next_shape = (height, width)
        if patch_merge is not None:
            merged, next_shape = self._run_patch_merging_tt(patch_merge, state, height, width)
        return state, stage_feature_torch, merged, next_shape

    def _run_swin_block_tt(
        self, stage_index: int, block_index: int, params: _StageBlockParams, hidden_states, height: int, width: int
    ):
        batch = hidden_states.shape[0]
        dim = params.dim
        window_size = min(params.window_size, min(height, width))
        shift_size = params.shift_size if min(height, width) > params.window_size else 0
        norm_eps = self.config.layer_norm_eps

        x = self._layer_norm(hidden_states, params.norm_before_weight, params.norm_before_bias, norm_eps)
        spatial = self._sequence_to_spatial(x, height, width)
        spatial, pad_h, pad_w = self._pad_for_windows(spatial, window_size)
        padded_height = height + pad_h
        padded_width = width + pad_w
        if shift_size > 0:
            spatial = ttnn.roll(spatial, shifts=(-shift_size, -shift_size), dim=(1, 2))

        windows, num_windows = self._window_partition(spatial, window_size)
        # Build full attention mask once; streaming will slice per chunk.
        attn_mask = None
        if shift_size > 0:
            attn_mask = self._build_attention_mask(
                params,
                stage_index,
                block_index,
                padded_height,
                padded_width,
                batch,
                window_size,
                shift_size,
                num_windows,
            )
        # Stream attention block over window chunks to keep L1 usage bounded.
        attn_output = self._run_window_attention_streaming(
            windows,
            attn_mask,
            params,
            window_size,
            shift_size,
            stage_index,
            block_index,
            padded_height,
            padded_width,
            batch,
        )
        attn_spatial = self._window_reverse(attn_output, batch, padded_height, padded_width, window_size)
        if shift_size > 0:
            attn_spatial = ttnn.roll(attn_spatial, shifts=(shift_size, shift_size), dim=(1, 2))
        if pad_h or pad_w:
            attn_spatial = self._crop_spatial(attn_spatial, height, width)
        attn_sequence = self._spatial_to_sequence(attn_spatial)
        hidden_states = ttnn.add(hidden_states, attn_sequence)

        # Stream MLP over sequence to avoid large L1 allocations.
        mlp_input = self._layer_norm(hidden_states, params.norm_after_weight, params.norm_after_bias, norm_eps)
        mlp_hidden = self._run_mlp_streaming(mlp_input, params)
        hidden_states = ttnn.add(hidden_states, mlp_hidden)
        return hidden_states

    def _sequence_to_spatial(self, tensor, height: int, width: int):
        tensor = self._to_row_major(tensor)
        return self._reshape_tensor(tensor, (tensor.shape[0], height, width, tensor.shape[-1]), tag="seq_to_spatial")

    def _spatial_to_sequence(self, tensor):
        tensor = self._to_row_major(tensor)
        batch = tensor.shape[0]
        height = tensor.shape[1]
        width = tensor.shape[2]
        channels = tensor.shape[3]
        sequence = self._reshape_tensor(tensor, (batch, height * width, channels), tag="spatial_to_seq")
        return self._to_tile_dram(sequence)

    def _pad_for_windows(self, tensor, window_size: int):
        height = tensor.shape[1]
        width = tensor.shape[2]
        pad_h = (window_size - (height % window_size)) % window_size
        pad_w = (window_size - (width % window_size)) % window_size
        if pad_h == 0 and pad_w == 0:
            tensor = self._to_row_major(tensor)
            tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
            return tensor, 0, 0

        row_major = self._to_row_major(tensor)
        padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        row_major = ttnn.pad(row_major, padding=padding, value=0.0)
        row_major = ttnn.to_memory_config(row_major, ttnn.DRAM_MEMORY_CONFIG)
        return row_major, pad_h, pad_w

    def _window_partition(self, tensor, window_size: int):
        batch = tensor.shape[0]
        height = tensor.shape[1]
        width = tensor.shape[2]
        channels = tensor.shape[3]
        num_windows_h = height // window_size
        num_windows_w = width // window_size
        tensor = self._to_row_major(tensor)
        reshaped = self._reshape_tensor(
            tensor,
            (batch, num_windows_h, window_size, num_windows_w, window_size, channels),
            tag="window_partition_blocks",
        )
        permuted = ttnn.permute(reshaped, (0, 1, 3, 2, 4, 5))
        windows = self._reshape_tensor(
            permuted,
            (batch * num_windows_h * num_windows_w, window_size * window_size, channels),
            tag="window_partition_merge",
        )
        # Keep windows in DRAM to allow chunked attention downstream.
        windows = self._to_tile_dram(windows)
        return windows, num_windows_h * num_windows_w

    def _window_reverse(self, windows, batch: int, height: int, width: int, window_size: int):
        channels = windows.shape[-1]
        num_windows_h = height // window_size
        num_windows_w = width // window_size
        windows = self._to_row_major(windows)
        reshaped = self._reshape_tensor(
            windows,
            (batch, num_windows_h, num_windows_w, window_size, window_size, channels),
            tag="window_reverse_blocks",
        )
        permuted = ttnn.permute(reshaped, (0, 1, 3, 2, 4, 5))
        return self._reshape_tensor(permuted, (batch, height, width, channels), tag="window_reverse_merge")

    def _crop_spatial(self, tensor, height: int, width: int):
        current_h = tensor.shape[1]
        current_w = tensor.shape[2]
        if current_h == height and current_w == width:
            return tensor
        return ttnn.slice(tensor, (0, 0, 0, 0), (tensor.shape[0], height, width, tensor.shape[-1]))

    def _split_qkv(self, query, key, value, num_heads: int):
        return (
            self._split_heads(query, num_heads),
            self._split_heads(key, num_heads),
            self._split_heads(value, num_heads),
        )

    def _split_heads(self, tensor, num_heads: int):
        batch = tensor.shape[0]
        seq_len = tensor.shape[1]
        head_dim = tensor.shape[2] // num_heads
        tensor = self._to_row_major(tensor)
        reshaped = self._reshape_tensor(tensor, (batch, seq_len, num_heads, head_dim), tag="split_heads")
        permuted = ttnn.permute(reshaped, (0, 2, 1, 3))
        # Use DRAM for large sequences; streaming will move chunks to L1 as needed.
        return self._to_tile_dram(permuted)

    def _merge_heads(self, tensor, num_heads: int, dim: int):
        batch = tensor.shape[0]
        seq_len = tensor.shape[2]
        tensor = self._to_row_major(tensor)
        merged = ttnn.permute(tensor, (0, 2, 1, 3))
        reshaped = self._reshape_tensor(merged, (batch, seq_len, dim), tag="merge_heads")
        return self._to_tile_dram(reshaped)

    def _layer_norm(self, tensor, weight, bias, epsilon: float):
        original_layout = tensor.get_layout()
        tile_tensor = tensor if original_layout == ttnn.TILE_LAYOUT else ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        normalized = ttnn.layer_norm(tile_tensor, weight=weight, bias=bias, epsilon=epsilon)
        # Keep LN outputs in DRAM to avoid allocating massive L1 buffers.
        return ttnn.to_memory_config(
            normalized if normalized.get_layout() == ttnn.TILE_LAYOUT else ttnn.to_layout(normalized, ttnn.TILE_LAYOUT),
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def _linear(self, activation, weight, bias, compute_kernel_config=None):
        out = ttnn.matmul(activation, weight, transpose_b=True, compute_kernel_config=compute_kernel_config)
        if bias is not None:
            out = ttnn.add(out, bias)
        return out

    def _run_patch_merging_tt(self, params: _PatchMergingParams, hidden_states, height: int, width: int):
        spatial = self._sequence_to_spatial(hidden_states, height, width)
        spatial, pad_h, pad_w = self._pad_for_patch_merging(spatial)
        padded_height = height + pad_h
        padded_width = width + pad_w
        half_h = padded_height // 2
        half_w = padded_width // 2
        reshaped = self._reshape_tensor(
            spatial,
            (spatial.shape[0], half_h, 2, half_w, 2, spatial.shape[-1]),
            tag="patch_merge_blocks",
        )
        # Order groups to match HF: [x0(0,0), x1(1,0), x2(0,1), x3(1,1)] along channels
        # Bring blocks to (B, half_h, half_w, 2, 2, C)
        permuted = ttnn.permute(reshaped, (0, 1, 3, 2, 4, 5))
        groups = []
        for hi, wi in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            g = ttnn.slice(
                permuted, (0, 0, 0, hi, wi, 0), (permuted.shape[0], half_h, half_w, hi + 1, wi + 1, permuted.shape[-1])
            )
            g = self._reshape_tensor(g, (permuted.shape[0], half_h, half_w, spatial.shape[-1]), tag="patch_merge_group")
            groups.append(g)
        merged = ttnn.concat(groups, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        merged = self._reshape_tensor(
            merged, (spatial.shape[0], half_h * half_w, spatial.shape[-1] * 4), tag="patch_merge_flat"
        )
        merged = self._layer_norm(merged, params.norm_weight, params.norm_bias, self.config.layer_norm_eps)
        merged = self._linear(
            self._to_tile_dram(merged),
            params.reduction_weight,
            None,
            compute_kernel_config=_make_compute_kernel_config(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        return merged, (half_h, half_w)

    def _pad_for_patch_merging(self, tensor):
        height = tensor.shape[1]
        width = tensor.shape[2]
        pad_h = height % 2
        pad_w = width % 2
        if pad_h == 0 and pad_w == 0:
            tensor = self._to_row_major(tensor)
            tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
            return tensor, 0, 0
        row_major = self._to_row_major(tensor)
        padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        row_major = ttnn.pad(row_major, padding=padding, value=0.0)
        row_major = ttnn.to_memory_config(row_major, ttnn.DRAM_MEMORY_CONFIG)
        return row_major, pad_h, pad_w

    def _stage_output_to_torch(self, tensor, height: int, width: int) -> torch.Tensor:
        spatial = self._sequence_to_spatial(tensor, height, width)
        spatial = self._to_row_major(spatial)
        torch_tensor = self._tt_tensor_to_torch(spatial)
        return torch_tensor.permute(0, 3, 1, 2).contiguous().cpu().float()

    def _prepare_stage_blocks(self, stage_index: int) -> List[_StageBlockParams]:
        if stage_index in self._stage_block_params:
            return self._stage_block_params[stage_index]

        blocks = self.get_stage_block_weights(stage_index)
        prepared: List[_StageBlockParams] = []
        window_size = self.window_size
        for block_idx, weights in enumerate(blocks):
            shift_size = 0 if (block_idx % 2 == 0) else window_size // 2
            prepared.append(
                _StageBlockParams(
                    shift_size=shift_size,
                    window_size=window_size,
                    num_heads=self.stage_heads[stage_index],
                    dim=self.stage_dims[stage_index],
                    norm_before_weight=self._prepare_layernorm_tensor(weights.norm_before.weight),
                    norm_before_bias=self._prepare_layernorm_tensor(weights.norm_before.bias),
                    norm_after_weight=self._prepare_layernorm_tensor(weights.norm_after.weight),
                    norm_after_bias=self._prepare_layernorm_tensor(weights.norm_after.bias),
                    query_weight=self._prepare_linear_weight(weights.attention.query.weight),
                    query_bias=self._prepare_bias(weights.attention.query.bias),
                    key_weight=self._prepare_linear_weight(weights.attention.key.weight),
                    key_bias=self._prepare_bias(weights.attention.key.bias),
                    value_weight=self._prepare_linear_weight(weights.attention.value.weight),
                    value_bias=self._prepare_bias(weights.attention.value.bias),
                    proj_weight=self._prepare_linear_weight(weights.attention.proj.weight),
                    proj_bias=self._prepare_bias(weights.attention.proj.bias),
                    mlp_fc1_weight=self._prepare_linear_weight(weights.mlp.fc1.weight),
                    mlp_fc1_bias=self._prepare_bias(weights.mlp.fc1.bias),
                    mlp_fc2_weight=self._prepare_linear_weight(weights.mlp.fc2.weight),
                    mlp_fc2_bias=self._prepare_bias(weights.mlp.fc2.bias),
                    relative_position_bias=self._compute_relative_position_bias(
                        weights.attention.relative_position_bias_table,
                        window_size,
                        self.stage_heads[stage_index],
                    ),
                )
            )

        self._stage_block_params[stage_index] = prepared
        return prepared

    def _prepare_patch_merging(self, stage_index: int) -> Optional[_PatchMergingParams]:
        if stage_index in self._patch_merge_params:
            return self._patch_merge_params[stage_index]

        stage_name = f"stage{stage_index + 1}"
        stage_state = self._stage_weights.get(stage_name, {})
        reduction_key = "downsample.reduction.weight"
        norm_weight_key = "downsample.norm.weight"
        norm_bias_key = "downsample.norm.bias"
        if reduction_key not in stage_state or norm_weight_key not in stage_state:
            self._patch_merge_params[stage_index] = None
            return None

        params = _PatchMergingParams(
            norm_weight=self._prepare_layernorm_tensor(self._ensure_torch_tensor(stage_state[norm_weight_key])),
            norm_bias=self._prepare_layernorm_tensor(self._ensure_torch_tensor(stage_state[norm_bias_key])),
            reduction_weight=self._prepare_linear_weight(self._ensure_torch_tensor(stage_state[reduction_key])),
            input_dim=self.stage_dims[stage_index],
        )
        self._patch_merge_params[stage_index] = params
        return params

    def _prepare_layernorm_tensor(self, tensor: torch.Tensor):
        dtype = self.dtype or get_default_dtype()
        dim = tensor.numel()
        padded = tensor.detach().contiguous()
        remainder = dim % TILE_WIDTH
        if remainder != 0:
            pad = TILE_WIDTH - remainder
            padded = F.pad(padded, (0, pad))
            dim = padded.numel()
        reshaped = padded.view(1, 1, dim)
        tensor_tt = ttnn.from_torch(
            reshaped,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tensor_tt = ttnn.to_layout(tensor_tt, ttnn.TILE_LAYOUT)
        self._debug(f"layernorm tensor shape={tensor_tt.shape} layout={tensor_tt.get_layout()}")
        return tensor_tt

    def _prepare_linear_weight(self, tensor: torch.Tensor):
        dtype = self.dtype or get_default_dtype()
        prepared = ttnn.from_torch(
            tensor.detach().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return prepared

    def _prepare_bias(self, tensor: torch.Tensor):
        if tensor is None:
            return None
        dtype = self.dtype or get_default_dtype()
        reshaped = tensor.detach().contiguous().view(1, 1, -1)
        return ttnn.from_torch(
            reshaped,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _compute_relative_position_bias(self, table: torch.Tensor, window_size: int, num_heads: int) -> torch.Tensor:
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        try:
            coord_mesh = torch.meshgrid(coords_h, coords_w, indexing="ij")
        except TypeError:
            coord_mesh = torch.meshgrid(coords_h, coords_w)
        coords = torch.stack(coord_mesh)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        bias = table[relative_position_index.view(-1)]
        bias = bias.view(window_size * window_size, window_size * window_size, num_heads)
        return bias.permute(2, 0, 1).contiguous()

    def _build_attention_mask(
        self,
        params: _StageBlockParams,
        stage_index: int,
        block_index: int,
        padded_height: int,
        padded_width: int,
        batch: int,
        window_size: int,
        shift_size: int,
        num_windows: int,
    ):
        seq = window_size * window_size
        rel_bias = params.relative_position_bias
        # TTNN SDPA expects mask heads=1; collapse per-head bias via mean.
        # Shape: [num_windows, 1, seq, seq]
        bias_per_head = rel_bias.mean(dim=0, keepdim=True)
        bias = bias_per_head.unsqueeze(0).expand(num_windows, -1, -1, -1)
        if shift_size > 0:
            shift_mask = self._get_shift_mask(
                stage_index, block_index, padded_height, padded_width, window_size, shift_size
            )
            shift_mask = shift_mask.unsqueeze(1)
            mask = bias + shift_mask
        else:
            mask = bias
        if batch > 1:
            mask = mask.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            mask = mask.view(batch * num_windows, 1, seq, seq)
        else:
            mask = mask.view(num_windows, 1, seq, seq)
        dtype = self.dtype or get_default_dtype()
        return ttnn.from_torch(
            mask.to(dtype=torch.float32),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            # Keep large attention masks in DRAM; streaming slices per chunk.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _get_shift_mask(
        self, stage_index: int, block_index: int, height: int, width: int, window_size: int, shift_size: int
    ):
        cache_key = (stage_index, block_index, (height, width), shift_size)
        if cache_key in self._attn_mask_cache:
            return self._attn_mask_cache[cache_key]

        img_mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
        shift = shift_size or window_size // 2
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift),
            slice(-shift, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift),
            slice(-shift, None),
        )
        idx = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = idx
                idx += 1
        mask_windows = self._window_partition_torch(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def _window_partition_torch(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        batch, height, width, channels = tensor.shape
        try:
            tensor = tensor.view(
                batch,
                height // window_size,
                window_size,
                width // window_size,
                window_size,
                channels,
            )
        except RuntimeError:
            raise ValueError("Window partition received incompatible tensor dimensions.")
        return tensor.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, channels)

    def _build_sdpa_configs(self, seq_len: int):
        grid = self.device.compute_with_storage_grid_size()
        # Choose largest power-of-two chunk size that divides seq_len and is >= TILE_WIDTH
        pow2 = 1 << (seq_len - 1).bit_length()  # next power-of-two >= seq_len
        candidate = pow2 // 2 if pow2 > seq_len else pow2
        while candidate >= TILE_WIDTH and (seq_len % candidate) != 0:
            candidate //= 2
        target = max(TILE_WIDTH, candidate)
        program_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=target,
            k_chunk_size=target,
            exp_approx_mode=True,
        )
        compute_cfg = _make_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        return program_cfg, compute_cfg

    def run_patch_embedding_hf(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self._hf_backbone_model is None:
            raise RuntimeError("HuggingFace backbone not available for patch embedding reference.")
        hf_obj = getattr(self._hf_backbone_model, "model", self._hf_backbone_model)
        emb_module = getattr(hf_obj, "embeddings", None)
        if emb_module is None:
            raise RuntimeError("Backbone embeddings module missing.")
        padded_images, _ = self._pad_images_for_patch(images)
        with torch.no_grad():
            embeddings, dims = emb_module(padded_images.to(self._torch_device), interpolate_pos_encoding=False)
        return embeddings.cpu(), tuple(int(x) for x in dims)

    def run_stage1_tt_with_taps(self, images: torch.Tensor):
        """Execute Stage 1 and return stage output plus per-block taps as torch tensors."""
        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT stage execution unavailable (device or TT runtime missing).")
        use_fallback = os.environ.get("MASKFORMER_PATCH_EMBED_FALLBACK") == "1"
        if use_fallback:
            hf_embeds, dims = self.run_patch_embedding_hf(images)
            height, width = dims
            seq = ttnn.from_torch(
                hf_embeds.to(self._torch_device),
                dtype=self.dtype or get_default_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            output_tensor, (height, width) = self._patch_embed_forward_tt_raw(images)
            seq = self._reshape_patch_embeddings(output_tensor, height, width)
            # Apply LayerNorm after patch embedding to match HF embeddings
            if (
                self._patch_embed_kernel
                and "norm_weight" in self._patch_embed_kernel
                and "norm_bias" in self._patch_embed_kernel
            ):
                seq = self._layer_norm(
                    seq,
                    self._patch_embed_kernel["norm_weight"],
                    self._patch_embed_kernel["norm_bias"],
                    self.config.layer_norm_eps,
                )
        taps: List[torch.Tensor] = []
        _, stage_feature_torch, _, _ = self._run_stage_tt(
            stage_index=0, hidden_states=seq, height=height, width=width, taps_out=taps
        )
        return stage_feature_torch, taps, (height, width)

    def run_stage1_tt_outputs(self, images: torch.Tensor):
        """Return both pre-merge and post-merge Stage 1 features as torch tensors (if merge available)."""
        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT stage execution unavailable (device or TT runtime missing).")
        use_fallback = os.environ.get("MASKFORMER_PATCH_EMBED_FALLBACK") == "1"
        if use_fallback:
            hf_embeds, dims = self.run_patch_embedding_hf(images)
            height, width = dims
            seq_tt = ttnn.from_torch(
                hf_embeds.to(self._torch_device),
                dtype=self.dtype or get_default_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            state, stage_feature_torch, merged, next_shape = self._run_stage_tt(
                stage_index=0, hidden_states=seq_tt, height=height, width=width
            )
        else:
            output_tensor, (height, width) = self._patch_embed_forward_tt_raw(images)
            stage_input = self._reshape_patch_embeddings(output_tensor, height, width)
            # Apply LayerNorm after patch embedding to align with HF embeddings
            if (
                self._patch_embed_kernel
                and "norm_weight" in self._patch_embed_kernel
                and "norm_bias" in self._patch_embed_kernel
            ):
                stage_input = self._layer_norm(
                    stage_input,
                    self._patch_embed_kernel["norm_weight"],
                    self._patch_embed_kernel["norm_bias"],
                    self.config.layer_norm_eps,
                )
            state, stage_feature_torch, merged, next_shape = self._run_stage_tt(
                stage_index=0, hidden_states=stage_input, height=height, width=width
            )
        merged_feature_torch = None
        if merged is not None:
            merged_feature_torch = self._stage_output_to_torch(merged, next_shape[0], next_shape[1])
        return stage_feature_torch, merged_feature_torch

    def run_stage2_tt_with_taps(self, images: torch.Tensor):
        """Execute Stage 2 and return stage output plus per-block taps as torch tensors."""
        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT stage execution unavailable (device or TT runtime missing).")
        # Build Stage 1 first to obtain Stage 2 inputs (post-merge sequence + dims)
        use_fallback = os.environ.get("MASKFORMER_PATCH_EMBED_FALLBACK") == "1"
        if use_fallback:
            hf_embeds, dims1 = self.run_patch_embedding_hf(images)
            h1, w1 = dims1
            seq1 = ttnn.from_torch(
                hf_embeds.to(self._torch_device),
                dtype=self.dtype or get_default_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            output_tensor, (h1, w1) = self._patch_embed_forward_tt_raw(images)
            seq1 = self._reshape_patch_embeddings(output_tensor, h1, w1)
            if (
                self._patch_embed_kernel
                and "norm_weight" in self._patch_embed_kernel
                and "norm_bias" in self._patch_embed_kernel
            ):
                seq1 = self._layer_norm(
                    seq1,
                    self._patch_embed_kernel["norm_weight"],
                    self._patch_embed_kernel["norm_bias"],
                    self.config.layer_norm_eps,
                )
        # Run Stage 1 fully to obtain merged sequence and next dims
        state1, stage1_pre_torch, merged1, (h2, w2) = self._run_stage_tt(
            stage_index=0, hidden_states=seq1, height=h1, width=w1
        )
        if merged1 is None:
            raise RuntimeError("Stage 1 did not produce merged output required for Stage 2.")
        taps2: List[torch.Tensor] = []
        _, stage2_pre_torch, _, _ = self._run_stage_tt(
            stage_index=1, hidden_states=merged1, height=h2, width=w2, taps_out=taps2
        )
        return stage2_pre_torch, taps2, (h2, w2)

    def run_stage2_tt_outputs(self, images: torch.Tensor):
        """Return pre-merge Stage 2 features as torch tensors (post-merge optional)."""
        if not self._can_run_patch_embed_tt():
            raise RuntimeError("TT stage execution unavailable (device or TT runtime missing).")
        # Stage 1 to get Stage 2 inputs
        use_fallback = os.environ.get("MASKFORMER_PATCH_EMBED_FALLBACK") == "1"
        if use_fallback:
            hf_embeds, dims1 = self.run_patch_embedding_hf(images)
            h1, w1 = dims1
            seq1 = ttnn.from_torch(
                hf_embeds.to(self._torch_device),
                dtype=self.dtype or get_default_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            output_tensor, (h1, w1) = self._patch_embed_forward_tt_raw(images)
            seq1 = self._reshape_patch_embeddings(output_tensor, h1, w1)
            if (
                self._patch_embed_kernel
                and "norm_weight" in self._patch_embed_kernel
                and "norm_bias" in self._patch_embed_kernel
            ):
                seq1 = self._layer_norm(
                    seq1,
                    self._patch_embed_kernel["norm_weight"],
                    self._patch_embed_kernel["norm_bias"],
                    self.config.layer_norm_eps,
                )
        _, stage1_pre_torch, merged1, (h2, w2) = self._run_stage_tt(
            stage_index=0, hidden_states=seq1, height=h1, width=w1
        )
        if merged1 is None:
            return None, None
        _, stage2_pre_torch, merged2, (h3, w3) = self._run_stage_tt(
            stage_index=1, hidden_states=merged1, height=h2, width=w2
        )
        stage2_post_torch = None
        if merged2 is not None:
            stage2_post_torch = self._stage_output_to_torch(merged2, h3, w3)
        return stage2_pre_torch, stage2_post_torch

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize internal TT weights for checkpointing."""

        if not self._weights:
            raise RuntimeError("Backbone weights not loaded.")
        return dict(self._weights)

    # ---------------------------------------------------------------------
    # Internal helpers (skeletons)
    # ---------------------------------------------------------------------

    def _init_patch_embedding(self) -> None:
        """Allocate patch embedding convolution and norm layers."""

        self.patch_embed = None
        if ttnn is None or self.device is None:
            return

        if self._patch_embed_spec is None:
            self._patch_embed_spec = self._derive_patch_embed_spec()

        if self._patch_embed_spec is None:
            warnings.warn("Patch embedding spec unavailable; TT kernels will be deferred.")
            return

        spec = self._patch_embed_spec
        require_ttnn("prepare patch embedding convolution")
        self.patch_embed = {
            "weight_key": spec.weight_key,
            "bias_key": spec.bias_key,
            "conv_kwargs": {
                "in_channels": spec.in_channels,
                "out_channels": spec.out_channels,
                "kernel_size": spec.kernel_size,
                "stride": spec.stride,
                "padding": spec.padding,
                "dilation": (1, 1),
                "groups": 1,
            },
        }

    def _init_stages(self) -> None:
        """Instantiate Swin stages with window attention + MLP blocks."""

        # Future: hold stage objects or lightweight callables per depth level.
        self.stages: list[object] = []
        if ttnn is None or self.device is None:
            return

        require_ttnn("construct Swin stages on device")
        for plan in self._hf_stage_plans:
            self.stages.append(
                {
                    "name": plan.name,
                    "tasks": plan.tasks,
                    "num_heads": plan.num_heads,
                    "depth": plan.depth,
                }
            )

    def _forward_stage(
        self,
        stage_index: int,
        hidden_states: Any,
        attention_mask: Optional[Any],
    ) -> Any:
        """Execute a single Swin stage (placeholder)."""

        # Future: orchestrate window partitioning, attention, and patch merging.
        # Not required for bounty #30876 (backbone runs on HF fallback).
        raise NotImplementedError

    # ------------------------------------------------------------------
    # PyTorch fallback helpers
    # ------------------------------------------------------------------

    def _build_torch_backbone(self) -> None:
        """Instantiate HuggingFace backbone for CPU fallback execution."""

        try:
            from transformers import MaskFormerSwinConfig as HFConfig, MaskFormerSwinBackbone as HFBackbone
        except ModuleNotFoundError:
            self._hf_backbone_model = None
            warnings.warn(
                "transformers not installed; MaskFormer Swin fallback will be unavailable until TT implementation lands.",
                RuntimeWarning,
            )
            return

        config_dict = dict(self._hf_config_dict) if self._hf_config_dict else self.config.to_hf_dict()
        hf_config = HFConfig(**config_dict)
        backbone = HFBackbone(hf_config)
        if self._torch_backbone_state:
            torch_state = {k: self._ensure_torch_tensor(v) for k, v in self._torch_backbone_state.items()}
            missing, unexpected = backbone.load_state_dict(torch_state, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"Backbone weight load mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}",
                    RuntimeWarning,
                )
        backbone.eval()
        self._hf_backbone_model = backbone.to(self._torch_device)

    def _ensure_torch_tensor(self, tensor: Any) -> torch.Tensor:
        """Convert TT-NN tensors to torch tensors for fallback execution."""

        if isinstance(tensor, torch.Tensor):
            return tensor
        if tt_to_torch_tensor is not None:
            try:
                return tt_to_torch_tensor(tensor)
            except Exception:
                pass
        if hasattr(tensor, "to_torch"):
            return tensor.to_torch()
        if hasattr(tensor, "cpu"):
            return torch.tensor(tensor.cpu().numpy())
        if isinstance(tensor, (list, tuple)):
            return torch.tensor(tensor)
        raise TypeError(f"Unsupported tensor type for conversion to torch: {type(tensor)!r}")

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def _derive_patch_embed_spec(self) -> Optional[PatchEmbeddingSpec]:
        if not self._patch_embed_state:
            return None

        projection_weight = None
        projection_bias = None
        for name, tensor in self._patch_embed_state.items():
            if "projection.weight" in name:
                projection_weight = name
            elif "projection.bias" in name:
                projection_bias = name

        if projection_weight is None or projection_bias is None:
            return None

        patch = (self.config.patch_size, self.config.patch_size)
        return PatchEmbeddingSpec(
            in_channels=self.config.num_channels,
            out_channels=self.config.embed_dim,
            kernel_size=patch,
            stride=patch,
            padding=(0, 0),
            weight_key=projection_weight,
            bias_key=projection_bias,
        )

    def get_patch_embed_spec(self) -> Optional[PatchEmbeddingSpec]:
        if self._patch_embed_spec is None:
            self._patch_embed_spec = self._derive_patch_embed_spec()
        return self._patch_embed_spec

    def _prepare_patch_embed_kernels(self) -> None:
        if self.device is None or ttnn is None:
            return
        if torch_to_tt_tensor_rm is None:
            return
        if not self._patch_embed_state or self._patch_embed_spec is None:
            return
        if self._patch_embed_kernel is not None:
            return

        spec = self._patch_embed_spec
        weight = self._patch_embed_state.get(spec.weight_key)
        bias = self._patch_embed_state.get(spec.bias_key)
        if weight is None or bias is None:
            return

        dtype = self.dtype or get_default_dtype()
        if dtype is None:
            raise RuntimeError("TT dtype unavailable while preparing patch embedding kernels.")

        weight_torch = self._ensure_torch_tensor(weight).detach().contiguous()
        bias_torch = self._ensure_torch_tensor(bias).detach().view(1, 1, 1, -1).contiguous()

        weight_tt = ttnn.from_torch(weight_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        bias_tt = ttnn.from_torch(bias_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        self._patch_embed_kernel = {
            "weight": weight_tt,
            "bias": bias_tt,
        }

        # Prepare LayerNorm weights for patch embedding to match HF embeddings
        norm_weight = self._patch_embed_state.get("norm.weight")
        norm_bias = self._patch_embed_state.get("norm.bias")
        if norm_weight is not None and norm_bias is not None:
            try:
                norm_weight_torch = self._ensure_torch_tensor(norm_weight).detach().contiguous()
                norm_bias_torch = self._ensure_torch_tensor(norm_bias).detach().contiguous()
                self._patch_embed_kernel["norm_weight"] = self._prepare_layernorm_tensor(norm_weight_torch)
                self._patch_embed_kernel["norm_bias"] = self._prepare_layernorm_tensor(norm_bias_torch)
            except Exception:
                # Leave LN unset if preparation fails; downstream will skip LN.
                pass

    def _build_patch_conv_params(self, tt_input, spec: PatchEmbeddingSpec):
        if self._patch_embed_kernel is None:
            raise RuntimeError("Patch embedding kernels not initialized.")
        if Conv2dConfiguration is None or to_conv2d_config is None or to_compute_config is None:
            raise RuntimeError("TT-CNN Conv2d helpers unavailable; install TTNN to run patch embedding on device.")

        dtype = self.dtype or get_default_dtype()
        if dtype is None:
            raise RuntimeError("Unable to determine default TT dtype for patch embedding.")

        if HeightSliceStrategyConfiguration is not None:
            slice_strategy = HeightSliceStrategyConfiguration(num_slices=4)
        elif L1FullSliceStrategyConfiguration is not None:
            slice_strategy = L1FullSliceStrategyConfiguration()
        else:
            slice_strategy = None

        configuration = Conv2dConfiguration(
            input_height=tt_input.shape[1],
            input_width=tt_input.shape[2],
            in_channels=spec.in_channels,
            out_channels=spec.out_channels,
            batch_size=tt_input.shape[0],
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=(1, 1),
            groups=1,
            weight=self._patch_embed_kernel["weight"],
            bias=self._patch_embed_kernel["bias"],
            activation_dtype=dtype,
            weights_dtype=dtype,
            output_dtype=dtype,
            output_layout=ttnn.TILE_LAYOUT,
            sharding_strategy=HeightShardedStrategyConfiguration(
                reshard_if_not_optimal=True,
                act_block_h_override=32,
            ),
            slice_strategy=slice_strategy,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            deallocate_activation=False,
            reallocate_halo_output=True,
            config_tensors_in_dram=True,
        )
        self._debug(
            "patch_embed: conv_config "
            f"in_hw={configuration.input_height}x{configuration.input_width} stride={configuration.stride} "
            f"kernel={configuration.kernel_size} output_layout={configuration.output_layout}"
        )

        conv_config = to_conv2d_config(configuration)
        compute_config = to_compute_config(configuration, self.device)
        slice_config = ttnn.Conv2dL1FullSliceConfig
        if to_slice_config is not None and slice_strategy is not None:
            derived_slice = to_slice_config(slice_strategy)
            if derived_slice is not None:
                slice_config = derived_slice

        conv_kwargs = {
            "in_channels": configuration.in_channels,
            "out_channels": configuration.out_channels,
            "batch_size": configuration.batch_size,
            "input_height": configuration.input_height,
            "input_width": configuration.input_width,
            "kernel_size": configuration.kernel_size,
            "stride": configuration.stride,
            "padding": configuration.padding,
            "dilation": configuration.dilation,
            "groups": configuration.groups,
            "device": self.device,
            "conv_config": conv_config,
            "slice_config": slice_config,
        }
        self._debug(
            "patch_embed: conv_kwargs "
            f"batch={configuration.batch_size} in_ch={configuration.in_channels} out_ch={configuration.out_channels}"
        )
        return conv_kwargs, compute_config

    def _partition_stage_weights(self, weights: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        stage_map: Dict[str, Dict[str, Any]] = {}
        for idx in range(len(self.stage_depths)):
            prefix = f"model.pixel_level_module.encoder.model.encoder.layers.{idx}."
            stage_name = f"stage{idx + 1}"
            stage_map[stage_name] = {
                name[len(prefix) :]: tensor for name, tensor in weights.items() if name.startswith(prefix)
            }
        return stage_map

    def _get_stage_state(self, stage_index: int) -> Dict[str, Any]:
        stage_name = f"stage{stage_index + 1}"
        if stage_name not in self._stage_weights:
            raise ValueError(f"Stage '{stage_name}' weights not available.")
        return self._stage_weights[stage_name]

    def _block_exists(self, stage_state: Dict[str, Any], block_index: int) -> bool:
        prefix = f"blocks.{block_index}."
        return any(key.startswith(prefix) for key in stage_state.keys())

    def _extract_block_weights(self, stage_state: Dict[str, Any], block_index: int) -> SwinBlockWeights:
        base = f"blocks.{block_index}."
        norm_before = LayerNormWeights(
            weight=self._stage_tensor(stage_state, f"{base}layernorm_before.weight"),
            bias=self._stage_tensor(stage_state, f"{base}layernorm_before.bias"),
        )
        attention = SwinAttentionWeights(
            query=self._linear_from_stage(stage_state, f"{base}attention.self.query"),
            key=self._linear_from_stage(stage_state, f"{base}attention.self.key"),
            value=self._linear_from_stage(stage_state, f"{base}attention.self.value"),
            proj=self._linear_from_stage(stage_state, f"{base}attention.output.dense"),
            relative_position_bias_table=self._stage_tensor(
                stage_state, f"{base}attention.self.relative_position_bias_table"
            ),
        )
        norm_after = LayerNormWeights(
            weight=self._stage_tensor(stage_state, f"{base}layernorm_after.weight"),
            bias=self._stage_tensor(stage_state, f"{base}layernorm_after.bias"),
        )
        mlp = SwinMLPWeights(
            fc1=self._linear_from_stage(stage_state, f"{base}intermediate.dense"),
            fc2=self._linear_from_stage(stage_state, f"{base}output.dense"),
        )
        return SwinBlockWeights(
            norm_before=norm_before,
            attention=attention,
            norm_after=norm_after,
            mlp=mlp,
        )

    def _stage_tensor(self, stage_state: Dict[str, Any], key: str) -> torch.Tensor:
        if key not in stage_state:
            raise KeyError(f"Missing stage tensor: {key}")
        return self._ensure_torch_tensor(stage_state[key])

    def _linear_from_stage(self, stage_state: Dict[str, Any], base: str) -> LinearWeights:
        return LinearWeights(
            weight=self._stage_tensor(stage_state, f"{base}.weight"),
            bias=self._stage_tensor(stage_state, f"{base}.bias"),
        )

    def _build_stage_plans(self) -> List[SwinStagePlan]:
        height = self.config.image_size[0] // self.config.patch_size
        width = self.config.image_size[1] // self.config.patch_size
        plans: List[SwinStagePlan] = []
        current_dim = self.config.embed_dim
        for idx, depth in enumerate(self.stage_depths):
            name = f"stage{idx + 1}"
            num_heads = self.stage_heads[idx]
            output_dim = current_dim * (2 if idx < len(self.stage_depths) - 1 else 1)
            tasks: Tuple[str, ...]
            if idx < len(self.stage_depths) - 1:
                tasks = (
                    "window_partition",
                    "shifted_window_attention",
                    "mlp",
                    "patch_merging",
                )
            else:
                tasks = (
                    "window_partition",
                    "shifted_window_attention",
                    "mlp",
                )
            plans.append(
                SwinStagePlan(
                    name=name,
                    depth=depth,
                    num_heads=num_heads,
                    input_dim=current_dim,
                    output_dim=output_dim,
                    window_size=self.window_size,
                    tasks=tasks,
                )
            )
            current_dim = output_dim
            if idx < len(self.stage_depths) - 1:
                height //= 2
                width //= 2
        return plans

    def describe_stage_plan(self) -> List[SwinStagePlan]:
        """Return the cached Swin stage plan (useful for logging or debugging)."""

        return list(self._hf_stage_plans)

    def get_stage_block_weights(self, stage_index: int) -> List[SwinBlockWeights]:
        stage_state = self._get_stage_state(stage_index)
        blocks: List[SwinBlockWeights] = []
        block_idx = 0
        while self._block_exists(stage_state, block_idx):
            blocks.append(self._extract_block_weights(stage_state, block_idx))
            block_idx += 1
        if not blocks:
            raise ValueError(f"No blocks found for stage index {stage_index}.")
        return blocks

    def _can_run_patch_embed_tt(self) -> bool:
        return self.device is not None and ttnn is not None and tt_to_torch_tensor is not None

    def _pad_images_for_patch(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        patch = self.config.patch_size
        _, _, height, width = images.shape
        pad_h = (patch - (height % patch)) % patch
        pad_w = (patch - (width % patch)) % patch
        if pad_h == 0 and pad_w == 0:
            return images, (height, width)
        padded = F.pad(images, (0, pad_w, 0, pad_h))
        return padded, (height + pad_h, width + pad_w)

    def _convert_images_to_tt(self, images: torch.Tensor):
        nhwc = images.to(self._torch_device).permute(0, 2, 3, 1).contiguous()
        return ttnn.from_torch(
            nhwc,
            dtype=self.dtype or get_default_dtype(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _tt_tensor_to_torch(self, tensor) -> torch.Tensor:
        if tt_to_torch_tensor is None:
            raise RuntimeError("tt_to_torch_tensor unavailable.")
        if hasattr(tensor, "get_layout") and tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        return tt_to_torch_tensor(tensor)

    def _to_row_major(self, tensor):
        if tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        return tensor

    def _to_tile_l1(self, tensor):
        tile = tensor
        if tile.get_layout() != ttnn.TILE_LAYOUT:
            tile = ttnn.to_layout(tile, ttnn.TILE_LAYOUT)
        return ttnn.to_memory_config(tile, ttnn.L1_MEMORY_CONFIG)

    def _to_tile_dram(self, tensor):
        tile = tensor
        if tile.get_layout() != ttnn.TILE_LAYOUT:
            tile = ttnn.to_layout(tile, ttnn.TILE_LAYOUT)
        return ttnn.to_memory_config(tile, ttnn.DRAM_MEMORY_CONFIG)

    def _run_window_attention_streaming(
        self,
        windows,
        attn_mask,
        params: _StageBlockParams,
        window_size: int,
        shift_size: int,
        stage_index: int,
        block_index: int,
        padded_height: int,
        padded_width: int,
        batch: int,
    ):
        """Compute attention over window partitions in DRAM-friendly chunks.

        This avoids allocating the full [num_windows, seq, dim] buffers in L1 by
        slicing along the window axis and concatenating results in DRAM.
        """
        total_windows = int(windows.shape[0])
        seq = int(windows.shape[1])
        dim = int(windows.shape[2])
        # Heuristic chunk size to keep L1 under budget.
        chunk = 64
        outputs: List[Any] = []
        # Pad sequence length to TILE multiple for SDPA constraints
        tile_size = TILE_WIDTH
        pad_tokens = (tile_size - (seq % tile_size)) % tile_size
        padded_seq = seq + pad_tokens
        program_cfg, compute_cfg = self._build_sdpa_configs(padded_seq)
        matmul_compute_cfg = _make_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        softmax_compute_cfg = _make_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        head_dim = dim // params.num_heads
        scale_value = 1.0 / math.sqrt(head_dim)
        scale_tt = ttnn.from_torch(
            torch.tensor([[[scale_value]]], dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Optional debug targeting for a single window/head/row
        debug_enabled = os.environ.get("MASKFORMER_DEBUG_WINDOW") == "1"
        try:
            debug_seq_idx = int(os.environ.get("MASKFORMER_DEBUG_SEQ_IDX", "93"))
        except Exception:
            debug_seq_idx = 93
        for start in range(0, total_windows, chunk):
            end = min(start + chunk, total_windows)
            # Slice window chunk
            win_chunk = ttnn.slice(windows, (start, 0, 0), (end, seq, dim))
            if pad_tokens:
                # Pad sequence tokens with zeros to reach padded_seq
                rm = self._to_row_major(win_chunk)
                rm = ttnn.pad(rm, padding=((0, 0), (0, pad_tokens), (0, 0)), value=0.0)
                win_chunk = self._to_tile_dram(rm)
            # QKV projections on window chunk
            q_chunk = self._linear(
                win_chunk, params.query_weight, params.query_bias, compute_kernel_config=matmul_compute_cfg
            )
            k_chunk = self._linear(
                win_chunk, params.key_weight, params.key_bias, compute_kernel_config=matmul_compute_cfg
            )
            v_chunk = self._linear(
                win_chunk, params.value_weight, params.value_bias, compute_kernel_config=matmul_compute_cfg
            )
            # Split heads: [chunk, NH, S, DH]
            q_heads = self._split_heads(q_chunk, params.num_heads)
            k_heads = self._split_heads(k_chunk, params.num_heads)
            v_heads = self._split_heads(v_chunk, params.num_heads)
            shift_mask_chunk = None
            if shift_size > 0:
                shift_mask_chunk = self._get_shift_mask(
                    stage_index,
                    block_index,
                    padded_height,
                    padded_width,
                    window_size,
                    shift_size,
                )[start:end, :, :]
            # Compute attention per head manually (QK^T + bias + softmax) to avoid SDPA mask constraints.
            head_outputs_rm: List[Any] = []
            for h in range(params.num_heads):
                q_h = ttnn.slice(q_heads, (0, h, 0, 0), (end - start, h + 1, padded_seq, dim // params.num_heads))
                k_h = ttnn.slice(k_heads, (0, h, 0, 0), (end - start, h + 1, padded_seq, dim // params.num_heads))
                v_h = ttnn.slice(v_heads, (0, h, 0, 0), (end - start, h + 1, padded_seq, dim // params.num_heads))
                # Remove head dim and tilize for matmul
                q_rm = ttnn.reshape(q_h, (end - start, padded_seq, dim // params.num_heads))
                k_rm = ttnn.reshape(k_h, (end - start, padded_seq, dim // params.num_heads))
                v_rm = ttnn.reshape(v_h, (end - start, padded_seq, dim // params.num_heads))
                q_t = self._to_tile_dram(q_rm)
                k_t = self._to_tile_dram(k_rm)
                v_t = self._to_tile_dram(v_rm)
                # QK^T logits
                logits = ttnn.matmul(
                    q_t,
                    k_t,
                    transpose_b=True,
                    compute_kernel_config=matmul_compute_cfg,
                )
                logits = self._to_tile_dram(logits)
                logits = ttnn.mul(logits, scale_tt)
                # Per-head relative position bias and optional shift mask
                rel_bias = params.relative_position_bias[h]
                bias = rel_bias.unsqueeze(0).expand(end - start, -1, -1)
                if shift_mask_chunk is not None:
                    bias = bias + shift_mask_chunk
                if pad_tokens:
                    bias = F.pad(bias, (0, pad_tokens, 0, pad_tokens), value=float(-100.0))
                bias_tt = ttnn.from_torch(
                    bias.to(dtype=torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if debug_enabled and start == 0 and h == 0 and shift_mask_chunk is not None:
                    try:
                        mask_row = shift_mask_chunk[0, debug_seq_idx, :10].detach().cpu().numpy()
                        print(
                            f"[attn_debug] shift_mask row={debug_seq_idx} first10={np.array2string(mask_row, precision=6)}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[attn_debug] failed to print shift mask: {e}", flush=True)
                logits = ttnn.add(logits, bias_tt)
                # Softmax over last dim
                attn_weights = ttnn.softmax(
                    logits, dim=-1, numeric_stable=True, compute_kernel_config=softmax_compute_cfg
                )
                attn_weights = self._to_tile_dram(attn_weights)
                if debug_enabled and start == 0 and h == 0:
                    try:
                        bias_row = bias[0, debug_seq_idx, :10].detach().cpu().numpy()
                        print(
                            f"[attn_debug] start={start} head={h} row={debug_seq_idx} "
                            f"scale={scale_value:.8f} bias[:10]={np.array2string(bias_row, precision=6)}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[attn_debug] failed to print bias row: {e}", flush=True)
                # Output = attn_weights @ V
                out_t = ttnn.matmul(
                    attn_weights,
                    v_t,
                    compute_kernel_config=matmul_compute_cfg,
                )
                out_t = self._to_tile_dram(out_t)
                head_outputs_rm.append(self._to_row_major(out_t))
            # Concatenate heads along channel dim
            out_concat_rm = ttnn.concat(head_outputs_rm, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Convert to TILE and project
            out_chunk = self._to_tile_dram(out_concat_rm)
            fp32_proj_bias = os.environ.get("MASKFORMER_FP32_PROJ_BIAS") == "1"
            if fp32_proj_bias:
                # Compute projection matmul on device, add bias on host in FP32, then return to device.
                out_no_bias = ttnn.matmul(
                    out_chunk,
                    params.proj_weight,
                    transpose_b=True,
                    compute_kernel_config=matmul_compute_cfg,
                )
                # Host-side FP32 bias add for accuracy
                rm = self._to_row_major(out_no_bias)
                torch_out = self._tt_tensor_to_torch(rm).to(dtype=torch.float32)
                torch_bias = ttnn.to_torch(params.proj_bias).view(-1).to(dtype=torch.float32)
                torch_out = torch_out + torch_bias
                out_chunk = ttnn.from_torch(
                    torch_out,
                    dtype=self.dtype or get_default_dtype(),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                out_chunk = self._linear(
                    out_chunk, params.proj_weight, params.proj_bias, compute_kernel_config=matmul_compute_cfg
                )
            # Slice off padded tokens before projection merge back
            if pad_tokens:
                out_chunk = ttnn.slice(out_chunk, (0, 0, 0), (out_chunk.shape[0], seq, dim))
            outputs.append(out_chunk)
        # Concatenate all chunks along window axis into DRAM
        attn_windows = ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return attn_windows

    def _run_mlp_streaming(self, activation, params: _StageBlockParams):
        """Stream MLP over the sequence axis to bound L1 usage."""
        batch = int(activation.shape[0])
        seq = int(activation.shape[1])
        dim = int(activation.shape[2])
        chunk = 2048  # 2k tokens * 128 dim is ~0.5MB per tensor
        outputs: List[Any] = []
        for start in range(0, seq, chunk):
            end = min(start + chunk, seq)
            act_chunk = ttnn.slice(activation, (0, start, 0), (batch, end, dim))
            hidden = self._linear(
                act_chunk,
                params.mlp_fc1_weight,
                params.mlp_fc1_bias,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
            hidden = ttnn.gelu(hidden)
            hidden = self._linear(
                hidden,
                params.mlp_fc2_weight,
                params.mlp_fc2_bias,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
            outputs.append(hidden)
        return ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _reshape_tensor(self, tensor, target_shape: Tuple[int, ...], tag: str):
        input_shape = tuple(int(dim) for dim in tensor.shape)
        if math.prod(input_shape) != math.prod(target_shape):
            raise RuntimeError(f"Reshape mismatch ({tag}): {input_shape} -> {target_shape}")
        return ttnn.reshape(tensor, target_shape)


@dataclass
class PatchEmbeddingSpec:
    """Metadata describing the patch embedding convolution."""

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    weight_key: str
    bias_key: str


@dataclass
class SwinStagePlan:
    """High-level to-do list for a Swin stage implementation."""

    name: str
    depth: int
    num_heads: int
    input_dim: int
    output_dim: int
    window_size: int
    tasks: Tuple[str, ...]


@dataclass
class _StageBlockParams:
    """Prepared TT-NN tensors and metadata for a single Swin block."""

    shift_size: int
    window_size: int
    num_heads: int
    dim: int
    norm_before_weight: Any
    norm_before_bias: Any
    norm_after_weight: Any
    norm_after_bias: Any
    query_weight: Any
    query_bias: Any
    key_weight: Any
    key_bias: Any
    value_weight: Any
    value_bias: Any
    proj_weight: Any
    proj_bias: Any
    mlp_fc1_weight: Any
    mlp_fc1_bias: Any
    mlp_fc2_weight: Any
    mlp_fc2_bias: Any
    relative_position_bias: torch.Tensor


@dataclass
class _PatchMergingParams:
    """Prepared tensors for Swin patch merging."""

    norm_weight: Any
    norm_bias: Any
    reduction_weight: Any
    input_dim: int
