# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import math
import logging
from typing import Any, Dict, List, Optional

import torch

from .tt_modules import build_attn_padding_mask_4d, pad_tokens_3d, unpad_tokens_3d
from .config import DPTLargeConfig
from .perf_counters import inc_vit_backbone_fallback

LOG = logging.getLogger(__name__)

try:
    from transformers import DPTConfig, DPTForDepthEstimation
except Exception:  # pragma: no cover
    DPTConfig = None
    DPTForDepthEstimation = None


@dataclass
class ViTBackboneOutputs:
    # Values may be torch.Tensor (CPU path) or ttnn.Tensor (TT encoder + return_tt=True).
    features: Dict[int, Any]  # layer index -> [B, C, H, W]
    # Optional token-level representations (including CLS) for each layer,
    # keyed by the same 1-based layer index as `features`.
    tokens: Optional[Dict[int, Any]] = None


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
        self._pos_embed_cache = {}
        self._tt_pos_embed_cache = {}
        self._tt_cls_cache = {}
        self.used_tt_encoder_last_forward: bool = False
        # TTNN defaults to l1_small_size=0 in some runtimes, which can force
        # kernels down slow fallback paths or fail allocation in conv/halo ops.
        self._tt_l1_small_size = 24576
        self._tt_trace_region_size = 8 * 1024 * 1024
        req_exec_mode = str(getattr(self.config, "tt_execution_mode", "eager")).lower()
        self._tt_num_command_queues = 2 if req_exec_mode == "trace_2cq" else 1
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
                            trace_region_size=self._tt_trace_region_size,
                            num_command_queues=self._tt_num_command_queues,
                        )
                    except Exception:
                        try:
                            self.tt_device = ttnn.open_mesh_device(
                                mesh_shape=ttnn.MeshShape(1, 1),
                                l1_small_size=self._tt_l1_small_size,
                                trace_region_size=self._tt_trace_region_size,
                                num_command_queues=self._tt_num_command_queues,
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
            # Avoid first-iteration host work and host->device copies on the hot path.
            try:
                import ttnn  # type: ignore

                h_out = int(self.config.image_size // int(self.config.patch_size))
                w_out = int(self.config.image_size // int(self.config.patch_size))
                _ = self._pos_embed_tt_for_hw(h_out, w_out, ttnn=ttnn)
                _ = self._cls_token_tt(batch=1, ttnn=ttnn)
            except Exception:
                pass

    # ------------------------------------------------------------------ backbone implementations
    def _pos_embed_for_hw(self, h_patches: int, w_patches: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Resize checkpoint positional embeddings to the runtime patch grid."""
        cache_key = (int(h_patches), int(w_patches), dtype, str(device))
        cached = self._pos_embed_cache.get(cache_key)
        if cached is not None:
            return cached

        pos_embed = self.tt_pos_embed.to(dtype=dtype, device=device)
        target_patches = int(h_patches) * int(w_patches)
        target_seq = target_patches + 1
        if pos_embed.shape[1] == target_seq:
            self._pos_embed_cache[cache_key] = pos_embed
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

        resized = torch.cat([cls_pos, patch_pos_resized], dim=1)
        self._pos_embed_cache[cache_key] = resized
        return resized

    def _pos_embed_tt_for_hw(self, h_patches: int, w_patches: int, *, ttnn):
        """
        Cached TT positional embeddings for the runtime patch grid.

        Returned tensor is on device and row-major layout: [1, N+1, C].
        """
        cache_key = (int(h_patches), int(w_patches))
        cached = self._tt_pos_embed_cache.get(cache_key)
        if cached is not None:
            return cached

        pos_torch = self._pos_embed_for_hw(
            int(h_patches),
            int(w_patches),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pos_tt = ttnn.from_torch(
            pos_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
        )
        self._tt_pos_embed_cache[cache_key] = pos_tt
        return pos_tt

    def _cls_token_tt(self, *, batch: int, ttnn):
        """Cached TT cls token expanded to the given batch: [B, 1, C] row-major."""
        b = int(batch)
        cached = self._tt_cls_cache.get(b)
        if cached is not None:
            return cached
        cls_torch = self.tt_cls_token.expand(b, -1, -1).contiguous()
        cls_tt = ttnn.from_torch(
            cls_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
        )
        self._tt_cls_cache[b] = cls_tt
        return cls_tt

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

    def _forward_tt_encoder_from_tt_input(self, tt_in, return_tt: bool = False) -> ViTBackboneOutputs:
        """
        TT encoder path that assumes the input is already a TT tensor on device.

        Expected input:
            tt_in: ttnn.Tensor [B, C, H, W] on `self.tt_device`.
        """
        import ttnn  # type: ignore

        if self.tt_device is None or self.tt_patch is None or not self.tt_blocks:
            raise RuntimeError("TT encoder is not initialized")

        # Patch embedding via TT conv.
        x = self.tt_patch(tt_in)  # TT tensor [B, C, H_out, W_out]

        patch = int(self.config.patch_size)
        h_cfg = int(self.config.image_size // patch)
        w_cfg = int(self.config.image_size // patch)

        # Form tokens and add CLS + positional embeddings fully on device.
        try:
            B = int(x.shape[0])
            C = int(x.shape[1])
            h_out = int(x.shape[2])
            w_out = int(x.shape[3])

            # Sanity: for bring-up we expect square grids matching config.
            if (h_out, w_out) != (h_cfg, w_cfg):
                h_out, w_out = h_cfg, w_cfg

            patch_nhwc = ttnn.permute(x, (0, 2, 3, 1))  # [B,H,W,C]
            patch_nhwc = ttnn.to_layout(patch_nhwc, ttnn.ROW_MAJOR_LAYOUT)
            patch_tokens = ttnn.reshape(patch_nhwc, (int(B), int(h_out) * int(w_out), int(C)))  # [B,N,C]
            cls_tt = self._cls_token_tt(batch=int(B), ttnn=ttnn)  # [B,1,C]
            tokens_tt3 = ttnn.concat([cls_tt, patch_tokens], dim=1)  # [B,N+1,C]
            pos_tt = self._pos_embed_tt_for_hw(int(h_out), int(w_out), ttnn=ttnn)  # [1,N+1,C]
            tokens_tile = ttnn.to_layout(tokens_tt3, ttnn.TILE_LAYOUT)
            pos_tile = ttnn.to_layout(pos_tt, ttnn.TILE_LAYOUT)
            tokens_tile = ttnn.add(tokens_tile, pos_tile, dtype=ttnn.bfloat16)
            tokens_tt3 = ttnn.to_layout(tokens_tile, ttnn.ROW_MAJOR_LAYOUT)
        except Exception:
            inc_vit_backbone_fallback()
            if not bool(getattr(self.config, "allow_cpu_fallback", True)):
                raise
            # Conservative fallback: materialize patch embeddings on host, build the
            # token embedding sequence on host, then move back to device.
            x_host = x.cpu()
            if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
                x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
            x_torch = x_host.to_torch()  # [B, C, H_out, W_out]
            B, C, H, W = x_torch.shape
            h_out, w_out = int(H), int(W)
            patch_tokens_torch = x_torch.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
            cls = self.tt_cls_token.expand(B, -1, -1)  # [B, 1, C]
            tokens_torch = torch.cat([cls, patch_tokens_torch], dim=1)  # [B, N+1, C]
            pos_embed = self._pos_embed_for_hw(H, W, dtype=tokens_torch.dtype, device=tokens_torch.device)
            tokens_torch = tokens_torch + pos_embed
            tokens_tt3 = ttnn.from_torch(
                tokens_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
            )

        # Perf path only: pad sequence to tile multiple for sharded program configs.
        orig_len = int(tokens_tt3.shape[1])
        pad_seq = bool(getattr(self.config, "tt_perf_encoder", False))
        seq_len = int(orig_len)
        if pad_seq:
            padded_len = ((seq_len + 31) // 32) * 32
            if padded_len != seq_len:
                tokens_tt3 = ttnn.pad(tokens_tt3, [(0, 0), (0, int(padded_len - seq_len)), (0, 0)], value=0.0)
                seq_len = int(padded_len)

        tokens_tt = ttnn.to_layout(tokens_tt3, ttnn.TILE_LAYOUT)
        # Keep [B,1,N,C] through encoder blocks to reduce per-layer reshapes.
        tokens_tt = ttnn.reshape(tokens_tt, (int(B), 1, int(seq_len), int(C)))

        mm_opts = self.tt_layer_cfg.matmul_opts(seq_len=int(seq_len)) if self.tt_layer_cfg else {}
        if pad_seq and orig_len < seq_len:
            mm_opts["valid_seq_len"] = int(orig_len)
            cache_key = (int(seq_len), int(orig_len))
            attn_mask_tt = self._attn_mask_cache.get(cache_key)
            if attn_mask_tt is None:
                mask_torch = build_attn_padding_mask_4d(int(seq_len), int(orig_len), dtype=torch.float32)
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

        feats: Dict[int, Any] = {}
        token_maps: Dict[int, Any] = {}

        max_idx = max(0, int(self.config.num_hidden_layers) - 1)
        safe_layers = [min(max(int(i), 0), max_idx) for i in self.config.output_layers]
        for idx in safe_layers:
            # +1 offset because hidden_states_tt[0] holds embeddings.
            h_tt = hidden_states_tt[idx + 1]
            if return_tt:
                try:
                    tokens_tt4 = h_tt
                    shape = tuple(int(v) for v in tuple(getattr(tokens_tt4, "shape", ())))
                    if len(shape) == 3:
                        b, n, c = shape
                        tokens_tt4 = ttnn.reshape(tokens_tt4, (int(b), 1, int(n), int(c)))
                        shape = (int(b), 1, int(n), int(c))
                    if len(shape) != 4 or int(shape[1]) != 1:
                        raise RuntimeError(f"Unexpected TT token shape: {shape}")

                    b, _one, n_total, c = shape
                    if pad_seq and int(orig_len) < int(n_total):
                        tokens_tt4 = ttnn.slice(
                            tokens_tt4,
                            (0, 0, 0, 0),
                            (int(b), 1, int(orig_len), int(c)),
                        )
                        n_total = int(orig_len)
                    token_maps[idx + 1] = tokens_tt4

                    # Drop CLS and reshape patch tokens -> NCHW feature map on device.
                    if n_total <= 1:
                        raise RuntimeError(f"Unexpected token sequence length: {n_total}")
                    patch_tt4 = ttnn.slice(
                        tokens_tt4,
                        (0, 0, 1, 0),
                        (int(b), 1, int(n_total), int(c)),
                    )  # [B,1,H*W,C]
                    n_patches = int(n_total) - 1
                    if n_patches != int(h_out) * int(w_out):
                        raise RuntimeError(
                            f"Unexpected patch count: got {n_patches}, expected {int(h_out) * int(w_out)}"
                        )
                    nhwc = ttnn.reshape(patch_tt4, (int(b), int(h_out), int(w_out), int(c)))
                    feats[idx + 1] = ttnn.permute(nhwc, (0, 3, 1, 2))
                except Exception:
                    inc_vit_backbone_fallback()
                    if not bool(getattr(self.config, "allow_cpu_fallback", True)):
                        raise
                    # Conservative fallback: materialize tokens/features on host to
                    # keep correctness if runtime reshape/slice semantics change.
                    h_host = h_tt.cpu()
                    if hasattr(h_host, "layout") and h_host.layout == ttnn.TILE_LAYOUT:
                        h_host = h_host.to(ttnn.ROW_MAJOR_LAYOUT)
                    tokens_layer = h_host.to_torch()
                    if tokens_layer.dim() == 4:
                        if tokens_layer.shape[1] != 1:
                            raise RuntimeError(f"Unexpected TT token shape: {tuple(tokens_layer.shape)}")
                        tokens_layer = tokens_layer[:, 0, :, :]
                    elif tokens_layer.dim() != 3:
                        raise RuntimeError(f"Unexpected token rank from TT backbone: {tuple(tokens_layer.shape)}")
                    if pad_seq:
                        tokens_layer = unpad_tokens_3d(tokens_layer, orig_len)
                    token_maps[idx + 1] = tokens_layer

                    patch_tokens = tokens_layer[:, 1:, :]  # drop CLS
                    b_t, n_t, c_t = patch_tokens.shape
                    if n_t != int(h_out) * int(w_out):
                        raise RuntimeError(f"Unexpected patch token length: {n_t} vs {int(h_out) * int(w_out)}")
                    feats[idx + 1] = patch_tokens.transpose(1, 2).reshape(b_t, c_t, int(h_out), int(w_out))
            else:
                # Non-perf mode isn't expected to be used with TT tracing. Keep the
                # host path for compatibility.
                h_host = h_tt.cpu()
                if hasattr(h_host, "layout") and h_host.layout == ttnn.TILE_LAYOUT:
                    h_host = h_host.to(ttnn.ROW_MAJOR_LAYOUT)
                tokens_layer = h_host.to_torch()
                if tokens_layer.dim() == 4:
                    if tokens_layer.shape[1] != 1:
                        raise RuntimeError(f"Unexpected TT token shape: {tuple(tokens_layer.shape)}")
                    tokens_layer = tokens_layer[:, 0, :, :]
                elif tokens_layer.dim() != 3:
                    raise RuntimeError(f"Unexpected token rank from TT backbone: {tuple(tokens_layer.shape)}")
                if pad_seq:
                    tokens_layer = unpad_tokens_3d(tokens_layer, orig_len)
                token_maps[idx + 1] = tokens_layer
                patch_tokens = tokens_layer[:, 1:, :]
                b_t, n_t, c_t = patch_tokens.shape
                if n_t != int(h_out) * int(w_out):
                    raise RuntimeError(f"Unexpected patch token length: {n_t} vs {int(h_out) * int(w_out)}")
                feats[idx + 1] = patch_tokens.transpose(1, 2).reshape(b_t, c_t, int(h_out), int(w_out))

        return ViTBackboneOutputs(features=feats, tokens=token_maps)

    def _forward_tt_encoder(self, pixel_values: torch.Tensor, return_tt: bool = False) -> ViTBackboneOutputs:
        """General TT encoder path using TTTransformerBlock stack for small and large configs."""
        import ttnn  # type: ignore

        if self.tt_device is None or self.tt_patch is None or not self.tt_blocks:
            # Fallback to CPU backbone if TT is unavailable or misconfigured.
            return self._forward_cpu_backbone(pixel_values, return_tt=return_tt)

        # Pixel input: torch [B, C, H, W] -> TT tensor.
        tt_in = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
        )
        return self._forward_tt_encoder_from_tt_input(tt_in, return_tt=return_tt)

    def forward_tt_input(self, tt_pixel_values, return_tt: bool = False) -> ViTBackboneOutputs:
        """Forward assuming inputs are already a device-side TT tensor."""
        if not self._use_tt_encoder():
            raise RuntimeError("TT device is not available but forward_tt_input was requested")
        return self._forward_tt_encoder_from_tt_input(tt_pixel_values, return_tt=return_tt)

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
