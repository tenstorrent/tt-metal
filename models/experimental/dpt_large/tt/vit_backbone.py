# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import math
import logging
from typing import Dict, List, Optional

import torch

from .tt_modules import build_attn_padding_mask_4d, pad_tokens_3d, unpad_tokens_3d
from .config import DPTLargeConfig

LOG = logging.getLogger(__name__)

try:
    from transformers import DPTConfig, DPTForDepthEstimation
except Exception:  # pragma: no cover
    DPTConfig = None
    DPTForDepthEstimation = None


@dataclass
class ViTBackboneOutputs:
    features: Dict[int, torch.Tensor]  # layer index -> B x C x H x W
    # Optional token-level representations (including CLS) for each layer,
    # keyed by the same 1-based layer index as `features`.
    tokens: Optional[Dict[int, torch.Tensor]] = None


class DPTViTBackboneTTNN(torch.nn.Module):
    """
    A ViT-L backbone wrapper.

    For now, this uses the HF DPT encoder executed on the host. The class
    mirrors the interface we expect from a TT-backed implementation so we can
    swap implementations later without touching the pipeline.
    """

    def __init__(
        self,
        config: DPTLargeConfig | None = None,
        hf_model: Optional[torch.nn.Module] = None,
        pretrained: bool = True,
        device: str = "cpu",
        tt_layer_cfg=None,
    ):
        super().__init__()
        self.config = config if config is not None else DPTLargeConfig()
        # Host device for HF model (usually 'cpu'); TT device comes from config.device.
        self.device_str = device
        self.tt_layer_cfg = tt_layer_cfg

        if hf_model is not None:
            self.model = hf_model
        else:
            if DPTForDepthEstimation is None:
                raise ImportError("transformers is required to build the ViT backbone.")
            if pretrained:
                self.model = DPTForDepthEstimation.from_pretrained(self.config.model_name, output_hidden_states=True)
            else:
                hf_cfg = DPTConfig(**self.config.to_hf_kwargs())
                self.model = DPTForDepthEstimation(hf_cfg)

        self.model.to(self.device_str)
        self.model.eval()
        # Align our config depth with the attached HF model (important for tiny-config tests).
        try:
            hf_cfg = getattr(self.model, "config", None)
            if hf_cfg is not None and hasattr(hf_cfg, "num_hidden_layers"):
                self.config.num_hidden_layers = int(hf_cfg.num_hidden_layers)
        except Exception:
            pass

        # TT-specific members
        self.tt_device = None
        self.TTTransformerBlock = None
        self.TTPatchEmbedding = None
        self.tt_prog_cfg = None
        self.tt_patch = None
        self.tt_blocks = []
        self._attn_mask_cache = {}
        self.used_tt_encoder_last_forward: bool = False
        # TTNN defaults to l1_small_size=0 in some runtimes, which can force
        # kernels down slow fallback paths or fail allocation in conv/halo ops.
        self._tt_l1_small_size = 24576
        try:
            import ttnn  # noqa: F401
            from .tt_modules import TTTransformerBlock, TTPatchEmbedding

            # Use config.device for TT accelerator selection so the host can remain on CPU.
            if self.config.enable_tt_device and self.config.device != "cpu":
                # `fallback_ops` conversion wrappers expect MeshDevice-based tensors.
                if hasattr(ttnn, "open_mesh_device") and hasattr(ttnn, "MeshShape"):
                    try:
                        self.tt_device = ttnn.open_mesh_device(
                            mesh_shape=ttnn.MeshShape(1, 1),
                            physical_device_ids=[0],
                            l1_small_size=self._tt_l1_small_size,
                        )
                    except Exception:
                        self.tt_device = ttnn.open_mesh_device(
                            mesh_shape=ttnn.MeshShape(1, 1),
                            l1_small_size=self._tt_l1_small_size,
                        )
                else:
                    self.tt_device = ttnn.open_device(device_id=0, l1_small_size=self._tt_l1_small_size)
                try:
                    ttnn.SetDefaultDevice(self.tt_device)
                except Exception:
                    pass
                self.TTTransformerBlock = TTTransformerBlock
                self.TTPatchEmbedding = TTPatchEmbedding
                # Pass through TT layer config only for perf encoder path
                self.tt_prog_cfg = self.tt_layer_cfg if getattr(self.config, "tt_perf_encoder", False) else None
        except Exception:
            self.tt_device = None

        if self.tt_device:
            sd = self.model.state_dict()
            memcfg = (
                self.tt_layer_cfg.memcfg()
                if (self.tt_layer_cfg and getattr(self.config, "tt_perf_encoder", False))
                else None
            )
            # patch embedding conv
            self.tt_patch = self.TTPatchEmbedding(
                sd["dpt.embeddings.patch_embeddings.projection.weight"],
                sd["dpt.embeddings.patch_embeddings.projection.bias"],
                device=self.tt_device,
                stride=self.config.patch_size,
                padding=0,
                output_mem=memcfg,
            )
            self.tt_blocks = [
                self.TTTransformerBlock(
                    sd,
                    f"dpt.encoder.layer.{i}",
                    num_heads=self.config.num_attention_heads,
                    head_dim=self.config.hidden_size // self.config.num_attention_heads,
                    eps=self.config.layer_norm_eps,
                    device=self.tt_device,
                    output_mem=memcfg,
                    program_config=self.tt_prog_cfg,
                )
                for i in range(self.config.num_hidden_layers)
            ]
            self.tt_pos_embed = sd["dpt.embeddings.position_embeddings"]
            self.tt_cls_token = sd["dpt.embeddings.cls_token"]

    # ------------------------------------------------------------------ backbone implementations
    def _pos_embed_for_hw(self, h_patches: int, w_patches: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Resize checkpoint positional embeddings to the runtime patch grid."""
        pos_embed = self.tt_pos_embed.to(dtype=dtype, device=device)
        target_patches = int(h_patches) * int(w_patches)
        target_seq = target_patches + 1
        if pos_embed.shape[1] == target_seq:
            return pos_embed

        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        src_patches = int(patch_pos.shape[1])
        src_h = math.isqrt(src_patches)

        if src_h * src_h == src_patches:
            patch_pos_2d = patch_pos.reshape(1, src_h, src_h, -1).permute(0, 3, 1, 2)
            patch_pos_resized = torch.nn.functional.interpolate(
                patch_pos_2d,
                size=(h_patches, w_patches),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos_resized = patch_pos_resized.permute(0, 2, 3, 1).reshape(1, target_patches, -1)
        else:
            patch_pos_1d = patch_pos.transpose(1, 2)
            patch_pos_resized = torch.nn.functional.interpolate(
                patch_pos_1d,
                size=target_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return torch.cat([cls_pos, patch_pos_resized], dim=1)

    def _forward_cpu_backbone(self, pixel_values: torch.Tensor, return_tt: bool = False) -> ViTBackboneOutputs:
        """Reference CPU backbone using HF DPT encoder."""
        patch = self.config.patch_size
        h_out = self.config.image_size // patch

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states: List[torch.Tensor] = outputs.hidden_states

        feats: Dict[int, torch.Tensor] = {}
        token_maps: Dict[int, torch.Tensor] = {}

        max_idx = max(0, int(self.config.num_hidden_layers) - 1)
        safe_out = [min(max(int(i), 0), max_idx) for i in self.config.output_layers]
        for idx in safe_out:
            tokens = hidden_states[idx + 1]  # includes CLS at position 0
            token_maps[idx + 1] = tokens
            patch_tokens = tokens[:, 1:, :]  # drop CLS for spatial features
            B, N, C = patch_tokens.shape
            feat = patch_tokens.transpose(1, 2).reshape(B, C, h_out, h_out)
            feats[idx + 1] = feat

        return ViTBackboneOutputs(features=feats, tokens=token_maps)

    def _use_tt_encoder(self) -> bool:
        """Return True when we should run the TT encoder path (any supported config)."""
        return self.tt_device is not None and self.config.enable_tt_device

    def _forward_tt_encoder(self, pixel_values: torch.Tensor, return_tt: bool = False) -> ViTBackboneOutputs:
        """General TT encoder path using TTTransformerBlock stack for small and large configs."""
        import ttnn  # type: ignore

        if self.tt_device is None or self.tt_patch is None or not self.tt_blocks:
            # Fallback to CPU backbone if TT is unavailable or misconfigured.
            return self._forward_cpu_backbone(pixel_values, return_tt=return_tt)

        patch = self.config.patch_size
        h_out = self.config.image_size // patch

        # Pixel input: torch [B, C, H, W] -> TT tensor.
        tt_in = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
        )
        # Patch embedding via TT conv.
        x = self.tt_patch(tt_in)  # TT tensor [B, C, H_out, W_out]

        # Convert to host to form tokens and add CLS + positional embeddings.
        x_host = x.cpu()
        if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
            x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
        x_torch = x_host.to_torch()  # [B, C, H_out, W_out]
        B, C, H, W = x_torch.shape
        patch_tokens = x_torch.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        cls = self.tt_cls_token.expand(B, -1, -1)  # [B, 1, C]
        tokens_torch = torch.cat([cls, patch_tokens], dim=1)  # [B, N+1, C]
        pos_embed = self._pos_embed_for_hw(H, W, dtype=tokens_torch.dtype, device=tokens_torch.device)
        tokens_torch = tokens_torch + pos_embed

        # Perf path only: pad sequence to tile multiple for sharded program configs.
        orig_len = tokens_torch.shape[1]
        pad_seq = bool(getattr(self.config, "tt_perf_encoder", False))
        if pad_seq:
            tokens_torch, orig_len = pad_tokens_3d(tokens_torch, pad_multiple=32)

        tokens_tt = ttnn.from_torch(
            tokens_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
        )

        mm_opts = self.tt_layer_cfg.matmul_opts(seq_len=tokens_torch.shape[1]) if self.tt_layer_cfg else {}
        if pad_seq and orig_len < tokens_torch.shape[1]:
            # Provide an attention mask so padded tokens do not participate in attention.
            mm_opts["valid_seq_len"] = int(orig_len)
            cache_key = (int(tokens_torch.shape[1]), int(orig_len))
            attn_mask_tt = self._attn_mask_cache.get(cache_key)
            if attn_mask_tt is None:
                mask_torch = build_attn_padding_mask_4d(tokens_torch.shape[1], orig_len, dtype=torch.float32)
                attn_mask_tt = ttnn.from_torch(
                    mask_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.tt_device,
                )
                self._attn_mask_cache[cache_key] = attn_mask_tt
            mm_opts["attn_mask"] = attn_mask_tt
        hidden_states_tt: List[ttnn.Tensor] = [tokens_tt]
        for blk in self.tt_blocks[: self.config.num_hidden_layers]:
            tokens_tt = blk(tokens_tt, **mm_opts)
            hidden_states_tt.append(tokens_tt)

        feats: Dict[int, torch.Tensor] = {}
        token_maps: Dict[int, torch.Tensor] = {}

        max_idx = max(0, int(self.config.num_hidden_layers) - 1)
        safe_layers = [min(max(int(i), 0), max_idx) for i in self.config.output_layers]
        for idx in safe_layers:
            # +1 offset because hidden_states_tt[0] holds embeddings.
            h_tt = hidden_states_tt[idx + 1]
            h_host = h_tt.cpu()
            if hasattr(h_host, "layout") and h_host.layout == ttnn.TILE_LAYOUT:
                h_host = h_host.to(ttnn.ROW_MAJOR_LAYOUT)
            tokens_layer = h_host.to_torch()  # [B, N+1, C]
            if pad_seq:
                tokens_layer = unpad_tokens_3d(tokens_layer, orig_len)
            token_maps[idx + 1] = tokens_layer

            patch_tokens = tokens_layer[:, 1:, :]  # drop CLS
            B, N, C = patch_tokens.shape
            feat_torch = patch_tokens.transpose(1, 2).reshape(B, C, h_out, h_out)

            if return_tt:
                tt_feat = ttnn.from_torch(
                    feat_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.tt_device,
                )
                feats[idx + 1] = tt_feat
            else:
                feats[idx + 1] = feat_torch

        return ViTBackboneOutputs(features=feats, tokens=token_maps)

    def forward(self, pixel_values: torch.Tensor, return_tt: bool = False) -> ViTBackboneOutputs:
        """Unified backbone forward that can use either HF CPU encoder or a TT encoder."""
        use_tt_encoder = self._use_tt_encoder()
        self.used_tt_encoder_last_forward = use_tt_encoder
        if use_tt_encoder:
            return self._forward_tt_encoder(pixel_values, return_tt=return_tt)
        return self._forward_cpu_backbone(pixel_values, return_tt=return_tt)

    def close(self):
        try:
            if self.tt_device is None:
                return
            import ttnn  # type: ignore

            try:
                if hasattr(ttnn, "MeshDevice") and isinstance(self.tt_device, ttnn.MeshDevice):
                    ttnn.close_mesh_device(self.tt_device)
                else:
                    ttnn.close_device(self.tt_device)
            finally:
                self.tt_device = None
        except Exception:
            # Best-effort cleanup; avoid raising during interpreter shutdown.
            self.tt_device = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
