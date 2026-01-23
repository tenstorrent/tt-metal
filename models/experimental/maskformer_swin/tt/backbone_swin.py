# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MaskFormer Swin-Base backbone implemented with TTNN ops.

The reference checkpoint is `facebook/maskformer-swin-base-coco` which uses a
Swin backbone with `window_size=12` (segmentation/detection variant).

This module implements:
  - Patch embedding (Conv2d stride=patch_size) + LayerNorm
  - 4 Swin stages with shifted window attention + MLP
  - Patch merging downsample between stages
  - Per-stage output LayerNorms (`hidden_states_norms.*`) used by MaskFormer FPN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .ttnn_compat import ttnn, require_ttnn, get_default_dtype

DEFAULT_TT_DTYPE = get_default_dtype()


@dataclass(frozen=True)
class SwinBackboneConfig:
    embed_dim: int = 128
    depths: Tuple[int, int, int, int] = (2, 2, 18, 2)
    num_heads: Tuple[int, int, int, int] = (4, 8, 16, 32)
    window_size: int = 12
    patch_size: int = 4
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-5


def _parse_backbone_config(config_dict: Dict[str, object]) -> SwinBackboneConfig:
    def _as_int_tuple(value, *, length: int, default: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(value, (list, tuple)) and len(value) == length:
            return tuple(int(v) for v in value)
        return default

    return SwinBackboneConfig(
        embed_dim=int(config_dict.get("embed_dim", 128)),
        depths=_as_int_tuple(config_dict.get("depths"), length=4, default=(2, 2, 18, 2)),
        num_heads=_as_int_tuple(config_dict.get("num_heads"), length=4, default=(4, 8, 16, 32)),
        window_size=int(config_dict.get("window_size", 12)),
        patch_size=int(config_dict.get("patch_size", 4)),
        mlp_ratio=float(config_dict.get("mlp_ratio", 4.0)),
        qkv_bias=bool(config_dict.get("qkv_bias", True)),
        layer_norm_eps=float(config_dict.get("layer_norm_eps", 1e-5)),
    )


def _roll(x, shifts: Tuple[int, int], dims: Tuple[int, int]):
    """TTNN roll for NHWC tensors (dim indices are for H/W)."""

    if ttnn is None:
        raise RuntimeError("TTNN runtime is required for Swin roll().")
    assert len(shifts) == len(dims) == 2
    out = x
    shape = out.shape
    num_dims = len(shape)
    for shift, dim in zip(shifts, dims):
        shift %= int(shape[dim])
        if shift == 0:
            continue
        start_left, end_left = [0] * num_dims, list(shape)
        start_right, end_right = [0] * num_dims, list(shape)
        start_left[dim] = int(shape[dim]) - shift
        start_right[dim] = 0
        end_right[dim] = int(shape[dim]) - shift

        left_part = ttnn.slice(
            out,
            slice_start=start_left,
            slice_end=end_left,
            slice_step=[1] * num_dims,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        right_part = ttnn.slice(
            out,
            slice_start=start_right,
            slice_end=end_right,
            slice_step=[1] * num_dims,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.concat([left_part, right_part], dim, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(left_part)
        ttnn.deallocate(right_part)
    return out


class MaskFormerSwinBackbone:
    """Swin backbone that returns the 4 stage feature maps (NHWC TT tensors)."""

    def __init__(
        self,
        *,
        config_dict: Dict[str, object],
        device: Optional[object],
        dtype: Optional[object] = DEFAULT_TT_DTYPE,
    ) -> None:
        self.config = _parse_backbone_config(config_dict)
        self.device = device
        self.dtype = dtype
        self._attn_mask_cache: Dict[Tuple[int, int, int], Any] = {}

        # Patch embed weights
        self._patch_w = None
        self._patch_b = None
        self._patch_norm_w = None
        self._patch_norm_b = None

        # Stage blocks + downsample weights
        self._stages: List[Dict[str, Any]] = []

        # Output norms (hidden_states_norms.{0..3})
        self._out_norms: List[Dict[str, Any]] = []

    @classmethod
    def from_huggingface(
        cls,
        weights: Dict[str, object],
        device: Optional[object],
        *,
        config_dict: Optional[Dict[str, object]] = None,
        dtype: Optional[object] = DEFAULT_TT_DTYPE,
    ) -> "MaskFormerSwinBackbone":
        backbone = cls(config_dict=dict(config_dict or {}), device=device, dtype=dtype)
        backbone.load_weights(weights)
        return backbone

    def load_weights(self, weights: Dict[str, object]) -> None:
        if self.device is None or ttnn is None:
            require_ttnn("load MaskFormer Swin backbone weights on device")
        if torch is None:
            raise RuntimeError("torch is required to load MaskFormer Swin backbone weights.")

        cfg = self.config
        dtype = self.dtype or DEFAULT_TT_DTYPE
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        def _to_tt_linear(w: torch.Tensor, b: Optional[torch.Tensor]):
            wt = ttnn.from_torch(
                w.detach().contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
            bt = None
            if b is not None:
                bt = ttnn.from_torch(
                    b.detach().contiguous().view(1, 1, -1),
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=mem_cfg,
                )
                bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT)
            return wt, bt

        def _to_tt_norm_param(param: torch.Tensor):
            tt_param = ttnn.from_torch(
                param.detach().contiguous().view(1, 1, -1),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
            return ttnn.to_layout(tt_param, ttnn.TILE_LAYOUT)

        def _to_tt_conv_weight_bias(w: torch.Tensor, b: Optional[torch.Tensor]):
            wt = ttnn.from_torch(
                w.detach().contiguous(),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
            bt = None
            if b is not None:
                bt = ttnn.from_torch(
                    b.detach().contiguous().view(1, 1, 1, -1),
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=mem_cfg,
                )
            return wt, bt

        # Patch embed conv + norm
        patch_prefix = "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection"
        self._patch_w, self._patch_b = _to_tt_conv_weight_bias(
            weights[f"{patch_prefix}.weight"], weights.get(f"{patch_prefix}.bias")
        )
        norm_prefix = "model.pixel_level_module.encoder.model.embeddings.norm"
        self._patch_norm_w = _to_tt_norm_param(weights[f"{norm_prefix}.weight"])
        self._patch_norm_b = _to_tt_norm_param(weights[f"{norm_prefix}.bias"])

        # Output norms
        self._out_norms = []
        for i in range(4):
            out_norm_prefix = f"model.pixel_level_module.encoder.hidden_states_norms.{i}"
            self._out_norms.append(
                {
                    "w": _to_tt_norm_param(weights[f"{out_norm_prefix}.weight"]),
                    "b": _to_tt_norm_param(weights[f"{out_norm_prefix}.bias"]),
                }
            )

        # Stage blocks
        self._stages = []
        window_size = int(cfg.window_size)
        for stage_idx, depth in enumerate(cfg.depths):
            dim = int(cfg.embed_dim * (2**stage_idx))
            num_heads = int(cfg.num_heads[stage_idx])
            stage_prefix = f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}"
            blocks: List[Dict[str, Any]] = []
            for block_idx in range(int(depth)):
                block_prefix = f"{stage_prefix}.blocks.{block_idx}"
                # Norms
                ln1_w = _to_tt_norm_param(weights[f"{block_prefix}.layernorm_before.weight"])
                ln1_b = _to_tt_norm_param(weights[f"{block_prefix}.layernorm_before.bias"])
                ln2_w = _to_tt_norm_param(weights[f"{block_prefix}.layernorm_after.weight"])
                ln2_b = _to_tt_norm_param(weights[f"{block_prefix}.layernorm_after.bias"])

                # Attention projections
                q_w, q_b = _to_tt_linear(
                    weights[f"{block_prefix}.attention.self.query.weight"],
                    weights.get(f"{block_prefix}.attention.self.query.bias"),
                )
                k_w, k_b = _to_tt_linear(
                    weights[f"{block_prefix}.attention.self.key.weight"],
                    weights.get(f"{block_prefix}.attention.self.key.bias"),
                )
                v_w, v_b = _to_tt_linear(
                    weights[f"{block_prefix}.attention.self.value.weight"],
                    weights.get(f"{block_prefix}.attention.self.value.bias"),
                )
                attn_out_w, attn_out_b = _to_tt_linear(
                    weights[f"{block_prefix}.attention.output.dense.weight"],
                    weights.get(f"{block_prefix}.attention.output.dense.bias"),
                )

                # MLP
                fc1_w, fc1_b = _to_tt_linear(
                    weights[f"{block_prefix}.intermediate.dense.weight"],
                    weights.get(f"{block_prefix}.intermediate.dense.bias"),
                )
                fc2_w, fc2_b = _to_tt_linear(
                    weights[f"{block_prefix}.output.dense.weight"],
                    weights.get(f"{block_prefix}.output.dense.bias"),
                )

                # Relative position bias (precompute on host, keep as TT tensor)
                table = weights[f"{block_prefix}.attention.self.relative_position_bias_table"].detach().contiguous()
                index = weights[f"{block_prefix}.attention.self.relative_position_index"].detach().contiguous()
                if index.dtype != torch.long:
                    index = index.to(torch.long)
                n = window_size * window_size
                rel_bias = table[index.view(-1)].view(n, n, num_heads).permute(2, 0, 1).contiguous()
                rel_bias = rel_bias.unsqueeze(0)  # [1, H, N, N]
                rel_bias_tt = ttnn.from_torch(
                    rel_bias,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=mem_cfg,
                )

                blocks.append(
                    {
                        "dim": dim,
                        "num_heads": num_heads,
                        "ln1_w": ln1_w,
                        "ln1_b": ln1_b,
                        "ln2_w": ln2_w,
                        "ln2_b": ln2_b,
                        "q_w": q_w,
                        "q_b": q_b,
                        "k_w": k_w,
                        "k_b": k_b,
                        "v_w": v_w,
                        "v_b": v_b,
                        "attn_out_w": attn_out_w,
                        "attn_out_b": attn_out_b,
                        "fc1_w": fc1_w,
                        "fc1_b": fc1_b,
                        "fc2_w": fc2_w,
                        "fc2_b": fc2_b,
                        "rel_bias": rel_bias_tt,
                    }
                )

            downsample = None
            if stage_idx < 3:
                down_prefix = f"{stage_prefix}.downsample"
                down_norm_w = _to_tt_norm_param(weights[f"{down_prefix}.norm.weight"])
                down_norm_b = _to_tt_norm_param(weights[f"{down_prefix}.norm.bias"])
                red_w, _ = _to_tt_linear(weights[f"{down_prefix}.reduction.weight"], None)
                downsample = {"norm_w": down_norm_w, "norm_b": down_norm_b, "reduction_w": red_w}

            self._stages.append({"blocks": blocks, "downsample": downsample})

    def _get_attn_mask(self, *, stage_idx: int, height: int, width: int) -> Any:
        """Compute and cache the shifted-window attention mask for (stage_idx, H, W)."""

        if self.device is None or ttnn is None:
            require_ttnn("build Swin attention masks")
        if torch is None:
            raise RuntimeError("torch is required to build Swin attention masks.")

        key = (stage_idx, height, width)
        cached = self._attn_mask_cache.get(key)
        if cached is not None:
            return cached

        window = int(self.config.window_size)
        shift = window // 2

        hp = int(math.ceil(height / window) * window)
        wp = int(math.ceil(width / window) * window)

        img_mask = torch.zeros((1, hp, wp, 1), dtype=torch.int64)
        h_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
        w_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # window partition (torch)
        mask_windows = img_mask.view(1, hp // window, window, wp // window, window, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, window * window)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # [num_windows, N, N]

        dtype = self.dtype or DEFAULT_TT_DTYPE
        tt_mask = ttnn.from_torch(
            attn_mask.to(torch.float32),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._attn_mask_cache[key] = tt_mask
        return tt_mask

    def forward(self, images: Any) -> Tuple[List[Any], List[Any]]:
        """Run the backbone on device and return feature maps (NHWC TT tensors).

        Returns
        -------
        feature_maps, hidden_states
            Both lists contain 4 tensors, one per Swin stage, with channels-last layout.
        """

        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer Swin backbone")
        if self._patch_w is None:
            raise RuntimeError("Backbone weights are not loaded.")

        cfg = self.config
        dtype = self.dtype or DEFAULT_TT_DTYPE
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG

        # Convert input to TT tensor (NHWC)
        if torch is not None and isinstance(images, torch.Tensor):
            nhwc = images.detach().contiguous().permute(0, 2, 3, 1)
            x = ttnn.from_torch(
                nhwc, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=mem_cfg
            )
        else:
            x = images
            if getattr(x, "get_layout", None) is not None and x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        B = int(x.shape[0])
        H = int(x.shape[1])
        W = int(x.shape[2])

        # Patch embedding conv (stride=patch_size)
        patch = int(cfg.patch_size)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._patch_w,
            bias_tensor=self._patch_b,
            in_channels=int(x.shape[-1]),
            out_channels=int(cfg.embed_dim),
            batch_size=B,
            input_height=H,
            input_width=W,
            kernel_size=(patch, patch),
            stride=(patch, patch),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            memory_config=mem_cfg,
        )
        # Patch norm
        x = ttnn.layer_norm(x, weight=self._patch_norm_w, bias=self._patch_norm_b, memory_config=ttnn.L1_MEMORY_CONFIG)

        feature_maps: List[Any] = []
        hidden_states: List[Any] = []
        window = int(cfg.window_size)
        head_scale = None  # computed per block

        # Stage loop
        for stage_idx, stage in enumerate(self._stages):
            # Blocks
            for block_idx, block in enumerate(stage["blocks"]):
                shift = 0 if (block_idx % 2 == 0) else window // 2

                # LN before attention
                xn = ttnn.layer_norm(x, weight=block["ln1_w"], bias=block["ln1_b"], memory_config=ttnn.L1_MEMORY_CONFIG)
                attn_out = self._window_attention(
                    xn,
                    block=block,
                    window=window,
                    shift=shift,
                    stage_idx=stage_idx,
                )
                ttnn.deallocate(xn)
                x = ttnn.add(x, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(attn_out)

                # LN + MLP
                xn2 = ttnn.layer_norm(
                    x, weight=block["ln2_w"], bias=block["ln2_b"], memory_config=ttnn.L1_MEMORY_CONFIG
                )
                mlp_out = self._mlp(
                    xn2, fc1_w=block["fc1_w"], fc1_b=block["fc1_b"], fc2_w=block["fc2_w"], fc2_b=block["fc2_b"]
                )
                ttnn.deallocate(xn2)
                x = ttnn.add(x, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(mlp_out)

            # Per-stage output norm (used by FPN and transformer module)
            out_norm = self._out_norms[stage_idx]
            x_out = ttnn.layer_norm(x, weight=out_norm["w"], bias=out_norm["b"], memory_config=ttnn.L1_MEMORY_CONFIG)
            feature_maps.append(x_out)
            hidden_states.append(x_out)

            # Downsample for next stage
            down = stage.get("downsample")
            if down is not None:
                x = self._patch_merging(
                    x, norm_w=down["norm_w"], norm_b=down["norm_b"], reduction_w=down["reduction_w"]
                )

        return feature_maps, hidden_states

    def _mlp(self, x, *, fc1_w, fc1_b, fc2_w, fc2_b):
        """Token MLP over NHWC feature map."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin MLP.")
        B, H, W, C = x.shape
        seq = ttnn.reshape(x, (int(B), int(H) * int(W), int(C)))
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.linear(
            seq,
            fc1_w,
            fc1_b,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        # GELU (fallback to relu if gelu isn't available at runtime)
        if hasattr(ttnn, "gelu"):
            try:
                y = ttnn.gelu(y)
            except Exception:
                y = ttnn.relu(y)
        else:
            y = ttnn.relu(y)
        y = ttnn.linear(
            y,
            fc2_w,
            fc2_b,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.reshape(y, (int(B), int(H), int(W), int(C)))
        return y

    def _patch_merging(self, x, *, norm_w, norm_b, reduction_w):
        """Patch merging downsample (H,W)/2 and channels*2."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin patch merging.")
        B, H, W, C = x.shape
        H = int(H)
        W = int(W)
        if (H % 2) != 0 or (W % 2) != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = ttnn.pad(x, (int(B), H + pad_h, W + pad_w, int(C)), [0, 0, 0, 0], 0)
            H = H + pad_h
            W = W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        y = ttnn.concat([x0, x1, x2, x3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)

        y = ttnn.layer_norm(y, weight=norm_w, bias=norm_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        B2, H2, W2, C4 = y.shape
        seq = ttnn.reshape(y, (int(B2), int(H2) * int(W2), int(C4)))
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.linear(
            seq,
            reduction_w,
            None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, (int(B2), int(H2), int(W2), int(out.shape[-1])))
        ttnn.deallocate(y)
        return out

    def _window_attention(self, x, *, block: Dict[str, Any], window: int, shift: int, stage_idx: int):
        """Shifted window attention over NHWC feature map."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin window attention.")

        B, H, W, C = x.shape
        B = int(B)
        H = int(H)
        W = int(W)
        C = int(C)
        num_heads = int(block["num_heads"])
        head_dim = C // num_heads
        assert head_dim * num_heads == C

        pad_h = (window - (H % window)) % window
        pad_w = (window - (W % window)) % window
        if pad_h or pad_w:
            x = ttnn.pad(x, (B, H + pad_h, W + pad_w, C), [0, 0, 0, 0], 0)
        Hp = int(x.shape[1])
        Wp = int(x.shape[2])

        if shift:
            x = _roll(x, shifts=(-shift, -shift), dims=(1, 2))

        num_windows_h = Hp // window
        num_windows_w = Wp // window
        num_windows = num_windows_h * num_windows_w
        xw = ttnn.reshape(
            x,
            (B, num_windows_h, window, num_windows_w, window, C),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        xw = ttnn.permute(xw, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)
        xw = ttnn.reshape(xw, (B * num_windows, window * window, C), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Q, K, V projections
        xw = ttnn.to_layout(xw, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.linear(
            xw,
            block["q_w"],
            block["q_b"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        k = ttnn.linear(
            xw,
            block["k_w"],
            block["k_b"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        v = ttnn.linear(
            xw,
            block["v_w"],
            block["v_b"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        ttnn.deallocate(xw)

        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # [B*nW, N, C] -> [B*nW, H, N, Hd]
        N = window * window
        q = ttnn.reshape(q, (B * num_windows, N, num_heads, head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.reshape(k, (B * num_windows, N, num_heads, head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.reshape(v, (B * num_windows, N, num_heads, head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.permute(k, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.permute(v, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        q = q * (head_dim**-0.5)
        k_t = ttnn.permute(k, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        ttnn.deallocate(k)

        # Add relative position bias (broadcast on batch/windows)
        attn = ttnn.add(attn, block["rel_bias"], memory_config=ttnn.L1_MEMORY_CONFIG)

        # Add attention mask for shifted blocks
        if shift:
            mask = self._get_attn_mask(stage_idx=stage_idx, height=Hp, width=Wp)  # [nW, N, N] row-major
            attn_rm = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn_rm = ttnn.reshape(attn_rm, (B, num_windows, num_heads, N, N), memory_config=ttnn.L1_MEMORY_CONFIG)
            mask_rm = ttnn.reshape(mask, (1, num_windows, 1, N, N), memory_config=ttnn.L1_MEMORY_CONFIG)
            attn_rm = ttnn.add(attn_rm, mask_rm, memory_config=ttnn.L1_MEMORY_CONFIG, use_legacy=False)
            attn = ttnn.reshape(attn_rm, (B * num_windows, num_heads, N, N), memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_rm)
            ttnn.deallocate(mask_rm)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ctx = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        ctx = ttnn.permute(ctx, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        ctx = ttnn.reshape(ctx, (B * num_windows, N, C), memory_config=ttnn.L1_MEMORY_CONFIG)
        ctx = ttnn.to_layout(ctx, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.linear(
            ctx,
            block["attn_out_w"],
            block["attn_out_b"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi),
        )
        ttnn.deallocate(ctx)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(
            out, (B, num_windows_h, num_windows_w, window, window, C), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.permute(out, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, (B, Hp, Wp, C), memory_config=ttnn.L1_MEMORY_CONFIG)

        if shift:
            out = _roll(out, shifts=(shift, shift), dims=(1, 2))
        if pad_h or pad_w:
            out = out[:, :H, :W, :]
        return out
