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
import inspect
import math
import os
import time

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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _backbone_compute_kernel_config():
    """Compute kernel config for Swin backbone matmul/linear ops.

    Defaults are set for Stage 1 correctness; override via env vars:
      - MASKFORMER_TT_BACKBONE_MATH_FIDELITY: lofi|hifi2|hifi4 (default: hifi2)
      - MASKFORMER_TT_BACKBONE_FP32_DEST_ACC: 0|1 (default: 1)
      - MASKFORMER_TT_BACKBONE_MATH_APPROX: 0|1 (default: 0)
    """

    if ttnn is None:
        raise RuntimeError("TTNN runtime is required to construct backbone compute kernel configs.")

    fidelity_name = os.environ.get("MASKFORMER_TT_BACKBONE_MATH_FIDELITY", "hifi2").strip().lower()
    fidelity = ttnn.MathFidelity.HiFi2
    if fidelity_name in {"lofi", "low"}:
        fidelity = ttnn.MathFidelity.LoFi
    elif fidelity_name in {"hifi4", "high"} and hasattr(ttnn.MathFidelity, "HiFi4"):
        fidelity = ttnn.MathFidelity.HiFi4

    fp32_dest_acc = os.environ.get("MASKFORMER_TT_BACKBONE_FP32_DEST_ACC", "1").strip() == "1"
    math_approx = os.environ.get("MASKFORMER_TT_BACKBONE_MATH_APPROX", "0").strip() == "1"

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=math_approx,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=False,
    )


def _grid_tuple() -> Tuple[int, int]:
    # Wormhole N300 has an 8x8 grid; reserve one row for dispatch/safety margin.
    return (8, 7)


def _init_with_signature(cls, overrides: Dict[str, Any]) -> Optional[Any]:
    """Best-effort instantiate a TTNN config class with partial overrides."""

    try:
        sig = inspect.signature(cls)
    except Exception:
        return None
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in overrides:
            kwargs[name] = overrides[name]
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        elif param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        else:
            kwargs[name] = overrides.get(name, 1)
    try:
        return cls(**kwargs)
    except Exception:
        return None


def _maybe_matmul_program_config(grid: Tuple[int, int]) -> Optional[Any]:
    if ttnn is None or not hasattr(ttnn, "MatmulProgramConfig"):
        return None
    return _init_with_signature(
        ttnn.MatmulProgramConfig,
        {
            "compute_with_storage_grid_size": grid,
            "in0_block_w": 1,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
            "transpose_m0": False,
            "transpose_m1": False,
        },
    )


def _roll(x, shifts: Tuple[int, int], dims: Tuple[int, int], *, memory_config: Optional[Any] = None):
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
            memory_config=memory_config,
        )
        right_part = ttnn.slice(
            out,
            slice_start=start_right,
            slice_end=end_right,
            slice_step=[1] * num_dims,
            memory_config=memory_config,
        )
        out = ttnn.concat([left_part, right_part], dim, memory_config=memory_config)
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
        self._prefer_l1_activations = _env_flag("MASKFORMER_TT_BACKBONE_L1_ACT", False)
        self._attn_mask_cache: Dict[Tuple[int, int, int], Any] = {}
        self._attn_mask_tile_cache: Dict[Tuple[int, int, int], Any] = {}
        self._core_grid = None
        self._matmul_pc = None
        self._program_cfg_initialized = False

        # Patch embed weights
        self._patch_w = None
        self._patch_b = None
        self._patch_conv_prepared = False
        self._patch_norm_w = None
        self._patch_norm_b = None

        # Stage blocks + downsample weights
        self._stages: List[Dict[str, Any]] = []

        # Output norms (hidden_states_norms.{0..3})
        self._out_norms: List[Dict[str, Any]] = []

    def _activation_memory_config(self):
        if ttnn is None:
            return None
        if self._prefer_l1_activations:
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def _maybe_init_program_configs(self) -> None:
        if self._program_cfg_initialized:
            return
        self._program_cfg_initialized = True
        if ttnn is None:
            return
        grid = _grid_tuple()
        disable_core_grid = _env_flag("MASKFORMER_TT_DISABLE_CORE_GRID", False)
        disable_matmul_pc = _env_flag("MASKFORMER_TT_DISABLE_MATMUL_PC", False)

        if (not disable_core_grid) and hasattr(ttnn, "CoreGrid"):
            try:
                self._core_grid = ttnn.CoreGrid(y=grid[1], x=grid[0])  # type: ignore[attr-defined]
            except Exception:
                self._core_grid = None
        if not disable_matmul_pc:
            self._matmul_pc = _maybe_matmul_program_config(grid)

    def _linear_matmul_kwargs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._maybe_init_program_configs()
        linear_kwargs: Dict[str, Any] = {}
        matmul_kwargs: Dict[str, Any] = {}
        if self._core_grid is not None:
            linear_kwargs["core_grid"] = self._core_grid
            matmul_kwargs["core_grid"] = self._core_grid
        if self._matmul_pc is not None:
            linear_kwargs["program_config"] = self._matmul_pc
            matmul_kwargs["program_config"] = self._matmul_pc
        return linear_kwargs, matmul_kwargs

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
        # Keep persistent model parameters in DRAM to avoid L1/CB allocation clashes
        # during large model initialization on N300.
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG

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
                bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT, memory_config=mem_cfg)
            return wt, bt

        def _to_tt_norm_param(param: torch.Tensor):
            tt_param = ttnn.from_torch(
                param.detach().contiguous().view(1, 1, -1),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
            return ttnn.to_layout(tt_param, ttnn.TILE_LAYOUT, memory_config=mem_cfg)

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
                q_w_torch = weights[f"{block_prefix}.attention.self.query.weight"]
                q_b_torch = weights.get(f"{block_prefix}.attention.self.query.bias")
                k_w_torch = weights[f"{block_prefix}.attention.self.key.weight"]
                k_b_torch = weights.get(f"{block_prefix}.attention.self.key.bias")
                v_w_torch = weights[f"{block_prefix}.attention.self.value.weight"]
                v_b_torch = weights.get(f"{block_prefix}.attention.self.value.bias")

                q_w, q_b = _to_tt_linear(q_w_torch, q_b_torch)
                k_w, k_b = _to_tt_linear(k_w_torch, k_b_torch)
                v_w, v_b = _to_tt_linear(v_w_torch, v_b_torch)

                qkv_w = None
                qkv_b = None
                enable_fused_qkv_global = os.environ.get("MASKFORMER_TT_ENABLE_FUSED_QKV", "0").strip() == "1"
                enable_fused_qkv_backbone = os.environ.get("MASKFORMER_TT_ENABLE_FUSED_QKV_BACKBONE")
                if enable_fused_qkv_backbone is None:
                    enable_fused_qkv_backbone = "1" if enable_fused_qkv_global else "0"
                if enable_fused_qkv_backbone.strip() == "1":
                    try:
                        # The sharded split op expects QKV interleaved per head: (q0,k0,v0,q1,k1,v1,...).
                        head_dim = int(dim // num_heads)
                        qkv_w_chunks = []
                        for h in range(int(num_heads)):
                            hs = h * head_dim
                            he = hs + head_dim
                            qkv_w_chunks.append(q_w_torch[hs:he].detach().contiguous())
                            qkv_w_chunks.append(k_w_torch[hs:he].detach().contiguous())
                            qkv_w_chunks.append(v_w_torch[hs:he].detach().contiguous())
                        qkv_w_torch = torch.cat(qkv_w_chunks, dim=0)
                        qkv_b_torch = None
                        if q_b_torch is not None and k_b_torch is not None and v_b_torch is not None:
                            qkv_b_chunks = []
                            for h in range(int(num_heads)):
                                hs = h * head_dim
                                he = hs + head_dim
                                qkv_b_chunks.append(q_b_torch[hs:he].detach().contiguous())
                                qkv_b_chunks.append(k_b_torch[hs:he].detach().contiguous())
                                qkv_b_chunks.append(v_b_torch[hs:he].detach().contiguous())
                            qkv_b_torch = torch.cat(qkv_b_chunks, dim=0)
                        qkv_w, qkv_b = _to_tt_linear(qkv_w_torch, qkv_b_torch)
                    except Exception:
                        qkv_w, qkv_b = (None, None)
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
                        "qkv_w": qkv_w,
                        "qkv_b": qkv_b,
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

    def _get_attn_mask(self, *, stage_idx: int, height: int, width: int, clone: bool = True) -> Any:
        """Compute and cache the shifted-window attention mask for (stage_idx, H, W).

        The cached mask is stored in broadcast-ready shape: [1, nW, 1, N, N] (row-major),
        where nW is the number of windows and N = window_size^2.

        Notes
        -----
        By default we return a cloned TT tensor so the caller can safely deallocate the
        per-use mask without risking the cached buffer. Callers that will not deallocate
        the returned tensor can request `clone=False` to reuse the cached buffer and
        reduce per-forward allocations.
        """

        if self.device is None or ttnn is None:
            require_ttnn("build Swin attention masks")
        if torch is None:
            raise RuntimeError("torch is required to build Swin attention masks.")

        key = (stage_idx, height, width)
        cache_enabled = os.environ.get("MASKFORMER_TT_DISABLE_ATTN_MASK_CACHE", "0").strip() != "1"
        cached = self._attn_mask_cache.get(key) if cache_enabled else None
        if cached is not None:
            if os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_MASK", "0").strip() == "1":
                print(
                    f"[maskformer][backbone] attn_mask cache hit stage={stage_idx} H={height} W={width}",
                    flush=True,
                )
            if clone:
                return ttnn.clone(cached, memory_config=self._activation_memory_config())
            return cached
        if os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_MASK", "0").strip() == "1":
            state = "miss" if cache_enabled else "disabled"
            print(
                f"[maskformer][backbone] attn_mask cache {state} stage={stage_idx} H={height} W={width}",
                flush=True,
            )

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
        # [num_windows, N, N] (torch)

        dtype = self.dtype or DEFAULT_TT_DTYPE
        act_mem_cfg = self._activation_memory_config()
        num_windows = int(attn_mask.shape[0])
        N = int(window * window)

        tt_mask_rm = ttnn.from_torch(
            attn_mask.to(torch.float32),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=act_mem_cfg,
        )
        # Cache in broadcast-ready shape to avoid per-forward reshapes and accidental deallocation of the cached buffer.
        tt_mask_rm = ttnn.reshape(tt_mask_rm, (1, num_windows, 1, N, N), memory_config=act_mem_cfg)

        tt_mask_tile = None
        try:
            # For fused attention softmax, we need a 4D TILE mask with batch==num_windows (B==1 path).
            tt_mask_tile = ttnn.from_torch(
                attn_mask.to(torch.float32).unsqueeze(1),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=act_mem_cfg,
            )
        except Exception:
            tt_mask_tile = None

        if cache_enabled:
            self._attn_mask_cache[key] = tt_mask_rm
            if tt_mask_tile is not None:
                self._attn_mask_tile_cache[key] = tt_mask_tile
            if clone:
                return ttnn.clone(tt_mask_rm, memory_config=act_mem_cfg)
            return tt_mask_rm
        return tt_mask_rm

    def _get_attn_mask_tile(self, *, stage_idx: int, height: int, width: int, clone: bool = True) -> Any:
        """Return shifted-window attention mask in TILE layout for fused attention softmax paths.

        The mask is cached in shape [num_windows, 1, N, N] (TILE layout) which matches the
        attention score tensor shape for B==1 window attention: [num_windows, num_heads, N, N].
        """

        if self.device is None or ttnn is None:
            require_ttnn("build Swin attention masks (tile)")

        key = (stage_idx, height, width)
        cache_enabled = os.environ.get("MASKFORMER_TT_DISABLE_ATTN_MASK_CACHE", "0").strip() != "1"
        cached = self._attn_mask_tile_cache.get(key) if cache_enabled else None
        if cached is not None:
            if clone:
                return ttnn.clone(cached, memory_config=self._activation_memory_config())
            return cached

        # Build caches (row-major + tile) via the canonical builder.
        _ = self._get_attn_mask(stage_idx=stage_idx, height=height, width=width)
        cached = self._attn_mask_tile_cache.get(key)
        if cached is None:
            raise RuntimeError("Tile attention mask cache miss (tile conversion unavailable).")
        if clone:
            return ttnn.clone(cached, memory_config=self._activation_memory_config())
        return cached

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
        act_mem_cfg = self._activation_memory_config()
        debug = os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE", "0").strip() == "1"
        sync_per_block = os.environ.get("MASKFORMER_TT_SYNC_BACKBONE_PER_BLOCK", "0").strip() == "1"
        sync_stage_boundaries = os.environ.get("MASKFORMER_TT_SYNC_BACKBONE_STAGE_BOUNDARIES", "0").strip() == "1"
        tt = require_ttnn("sync Swin backbone per-block") if sync_per_block else None
        tt_stage = require_ttnn("sync Swin backbone stage boundaries") if sync_stage_boundaries else None
        ts0 = time.perf_counter()
        if debug:
            print("[maskformer][backbone] forward begin", flush=True)

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
        if debug:
            print(f"[maskformer][backbone] input BHW=({B},{H},{W}) C={int(x.shape[3])}", flush=True)

        # Patch embedding conv (stride=patch_size)
        patch = int(cfg.patch_size)
        if self._patch_conv_prepared:
            [x, [out_h, out_w]] = ttnn.conv2d(
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
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            x, [out_h, out_w], [self._patch_w, self._patch_b] = ttnn.conv2d(
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
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            self._patch_conv_prepared = True
        x = ttnn.reshape(x, (B, out_h, out_w, int(cfg.embed_dim)))
        # Patch norm
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        x = ttnn.layer_norm(x, weight=self._patch_norm_w, bias=self._patch_norm_b, memory_config=act_mem_cfg)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)

        feature_maps: List[Any] = []
        hidden_states: List[Any] = []
        window = int(cfg.window_size)
        head_scale = None  # computed per block

        # Stage loop
        for stage_idx, stage in enumerate(self._stages):
            if debug:
                print(f"[maskformer][backbone] stage {stage_idx} begin x_shape={tuple(int(s) for s in x.shape)}", flush=True)
            # Blocks
            for block_idx, block in enumerate(stage["blocks"]):
                shift = 0 if (block_idx % 2 == 0) else window // 2
                if debug:
                    print(f"[maskformer][backbone] stage{stage_idx} block{block_idx} shift={shift} begin", flush=True)
                debug_steps = (
                    debug
                    and os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_STEPS", "0").strip() == "1"
                    and tt is not None
                    and hasattr(tt, "synchronize_device")
                )
                if debug_steps:
                    try:
                        stage_filter = int(os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_STEPS_STAGE", "2"))
                        block_filter = int(os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_STEPS_BLOCK", "0"))
                    except ValueError:
                        stage_filter = 2
                        block_filter = 0
                    debug_steps = stage_idx == stage_filter and block_idx == block_filter

                # LN before attention
                x_ln = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
                xn = ttnn.layer_norm(x_ln, weight=block["ln1_w"], bias=block["ln1_b"], memory_config=act_mem_cfg)
                ttnn.deallocate(x_ln)
                xn = ttnn.to_layout(xn, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} ln1 done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )
                attn_out = self._window_attention(
                    xn,
                    block=block,
                    window=window,
                    shift=shift,
                    stage_idx=stage_idx,
                )
                ttnn.deallocate(xn)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} attn done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )
                x = ttnn.add(x, attn_out, memory_config=act_mem_cfg)
                ttnn.deallocate(attn_out)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} attn+residual done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )

                # LN + MLP
                x_ln2 = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
                xn2 = ttnn.layer_norm(x_ln2, weight=block["ln2_w"], bias=block["ln2_b"], memory_config=act_mem_cfg)
                ttnn.deallocate(x_ln2)
                xn2 = ttnn.to_layout(xn2, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} ln2 done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )
                mlp_out = self._mlp(
                    xn2, fc1_w=block["fc1_w"], fc1_b=block["fc1_b"], fc2_w=block["fc2_w"], fc2_b=block["fc2_b"]
                )
                ttnn.deallocate(xn2)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} mlp done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )
                x = ttnn.add(x, mlp_out, memory_config=act_mem_cfg)
                ttnn.deallocate(mlp_out)
                if debug_steps:
                    tt.synchronize_device(self.device)
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} mlp+residual done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )
                if sync_per_block and tt is not None and hasattr(tt, "synchronize_device"):
                    tt.synchronize_device(self.device)
                if debug and sync_per_block:
                    print(
                        f"[maskformer][backbone] stage{stage_idx} block{block_idx} done t={time.perf_counter() - ts0:.2f}s",
                        flush=True,
                    )

            # Per-stage output norm (used by FPN and transformer module)
            out_norm = self._out_norms[stage_idx]
            x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
            x_out = ttnn.layer_norm(x_tile, weight=out_norm["w"], bias=out_norm["b"], memory_config=act_mem_cfg)
            ttnn.deallocate(x_tile)
            x_out = ttnn.to_layout(x_out, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
            feature_maps.append(x_out)
            hidden_states.append(x_out)

            # Downsample for next stage
            down = stage.get("downsample")
            if down is not None:
                x = self._patch_merging(
                    x, norm_w=down["norm_w"], norm_b=down["norm_b"], reduction_w=down["reduction_w"]
                )
                if sync_stage_boundaries and tt_stage is not None and hasattr(tt_stage, "synchronize_device"):
                    tt_stage.synchronize_device(self.device)
                    if debug:
                        print(
                            f"[maskformer][backbone] stage {stage_idx} downsample sync t={time.perf_counter() - ts0:.2f}s",
                            flush=True,
                        )

        return feature_maps, hidden_states

    def _mlp(self, x, *, fc1_w, fc1_b, fc2_w, fc2_b):
        """Token MLP over NHWC feature map."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin MLP.")
        act_mem_cfg = self._activation_memory_config()
        compute_cfg = _backbone_compute_kernel_config()
        linear_kwargs, _ = self._linear_matmul_kwargs()
        fuse_linear_act = os.environ.get("MASKFORMER_TT_FUSE_LINEAR_ACT", "0").strip() == "1"
        B, H, W, C = x.shape
        seq = ttnn.reshape(x, (int(B), int(H) * int(W), int(C)))
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        y = None
        if fuse_linear_act:
            try:
                y = ttnn.linear(
                    seq,
                    fc1_w,
                    bias=fc1_b,
                    transpose_b=True,
                    activation="gelu",
                    compute_kernel_config=compute_cfg,
                    memory_config=act_mem_cfg,
                    **linear_kwargs,
                )
            except Exception:
                y = None
        if y is None:
            y = ttnn.linear(
                seq,
                fc1_w,
                bias=fc1_b,
                transpose_b=True,
                compute_kernel_config=compute_cfg,
                memory_config=act_mem_cfg,
                **linear_kwargs,
            )
            # Swin MLP uses GELU; fail fast instead of silently changing activation.
            if not hasattr(ttnn, "gelu"):
                raise RuntimeError("Swin MLP requires ttnn.gelu, but it is unavailable in this runtime.")
            y = ttnn.gelu(y)
        y = ttnn.linear(
            y,
            fc2_w,
            bias=fc2_b,
            transpose_b=True,
            compute_kernel_config=compute_cfg,
            memory_config=act_mem_cfg,
            **linear_kwargs,
        )
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
        y = ttnn.reshape(y, (int(B), int(H), int(W), int(C)))
        return y

    def _patch_merging(self, x, *, norm_w, norm_b, reduction_w):
        """Patch merging downsample (H,W)/2 and channels*2."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin patch merging.")
        act_mem_cfg = self._activation_memory_config()
        compute_cfg = _backbone_compute_kernel_config()
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
        y = ttnn.concat([x0, x1, x2, x3], dim=-1, memory_config=act_mem_cfg)
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)

        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        y = ttnn.layer_norm(y, weight=norm_w, bias=norm_b, memory_config=act_mem_cfg)
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
        B2, H2, W2, C4 = y.shape
        seq = ttnn.reshape(y, (int(B2), int(H2) * int(W2), int(C4)))
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        out = ttnn.linear(
            seq,
            reduction_w,
            bias=None,
            transpose_b=True,
            compute_kernel_config=compute_cfg,
        )
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
        out = ttnn.reshape(out, (int(B2), int(H2), int(W2), int(out.shape[-1])))
        ttnn.deallocate(y)
        return out

    def _window_attention(self, x, *, block: Dict[str, Any], window: int, shift: int, stage_idx: int):
        """Shifted window attention over NHWC feature map."""

        if ttnn is None:
            raise RuntimeError("TTNN runtime is required for Swin window attention.")
        act_mem_cfg = self._activation_memory_config()
        compute_cfg = _backbone_compute_kernel_config()
        linear_kwargs, matmul_kwargs = self._linear_matmul_kwargs()
        debug_attn = os.environ.get("MASKFORMER_TT_DEBUG_BACKBONE_ATTN", "0").strip() == "1"
        sync_attn = debug_attn and os.environ.get("MASKFORMER_TT_SYNC_BACKBONE_ATTN_STEPS", "0").strip() == "1"
        tt_dbg = require_ttnn("sync Swin window attention debug") if sync_attn else None
        enable_inplace = os.environ.get("MASKFORMER_TT_BACKBONE_INPLACE_ADDS", "0").strip() == "1"
        reuse_mask_cache = os.environ.get("MASKFORMER_TT_BACKBONE_REUSE_ATTN_MASK_CACHE", "0").strip() == "1"
        cache_enabled = os.environ.get("MASKFORMER_TT_DISABLE_ATTN_MASK_CACHE", "0").strip() != "1"
        reuse_mask_cache = reuse_mask_cache and cache_enabled
        attn_ts = time.perf_counter()

        def _attn_mark(msg: str) -> None:
            if debug_attn:
                print(f"[maskformer][backbone_attn] {msg} t={time.perf_counter() - attn_ts:.2f}s", flush=True)
            if sync_attn and tt_dbg is not None and hasattr(tt_dbg, "synchronize_device"):
                tt_dbg.synchronize_device(self.device)

        B, H, W, C = x.shape
        B = int(B)
        H = int(H)
        W = int(W)
        C = int(C)
        num_heads = int(block["num_heads"])
        head_dim = C // num_heads
        assert head_dim * num_heads == C

        if debug_attn:
            print(f"[maskformer][backbone_attn] begin BHW=({B},{H},{W}) C={C} window={window}", flush=True)

        pad_h = (window - (H % window)) % window
        pad_w = (window - (W % window)) % window
        if pad_h or pad_w:
            if debug_attn:
                print("[maskformer][backbone_attn] pad begin", flush=True)
            x = ttnn.pad(x, (B, H + pad_h, W + pad_w, C), [0, 0, 0, 0], 0)
            _attn_mark("pad end")
        Hp = int(x.shape[1])
        Wp = int(x.shape[2])

        if shift:
            x = _roll(x, shifts=(-shift, -shift), dims=(1, 2), memory_config=act_mem_cfg)

        num_windows_h = Hp // window
        num_windows_w = Wp // window
        num_windows = num_windows_h * num_windows_w
        safety_sync_windows1 = (
            num_windows == 1 and os.environ.get("MASKFORMER_TT_SYNC_WINDOW_ATTN_WINDOWS1", "1").strip() == "1"
        )
        if debug_attn:
            print("[maskformer][backbone_attn] window partition begin", flush=True)
        xw = ttnn.reshape(
            x,
            (B, num_windows_h, window, num_windows_w, window, C),
            memory_config=act_mem_cfg,
        )
        xw = ttnn.permute(xw, (0, 1, 3, 2, 4, 5), memory_config=act_mem_cfg)
        xw = ttnn.reshape(xw, (B * num_windows, window * window, C), memory_config=act_mem_cfg)
        _attn_mark("window partition end")

        # Q, K, V projections
        if debug_attn:
            print("[maskformer][backbone_attn] qkv begin", flush=True)
        xw = ttnn.to_layout(xw, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        enable_fused_qkv_global = os.environ.get("MASKFORMER_TT_ENABLE_FUSED_QKV", "0").strip() == "1"
        enable_fused_qkv_backbone = os.environ.get("MASKFORMER_TT_ENABLE_FUSED_QKV_BACKBONE")
        if enable_fused_qkv_backbone is None:
            enable_fused_qkv_backbone = "1" if enable_fused_qkv_global else "0"
        enable_fused_qkv = enable_fused_qkv_backbone.strip() == "1"
        q = None
        k = None
        v = None
        key_is_transposed = False
        if (
            enable_fused_qkv
            and block.get("qkv_w") is not None
            and hasattr(ttnn, "transformer")
            and hasattr(ttnn.transformer, "split_query_key_value_and_split_heads")
        ):
            qkv = ttnn.linear(
                xw,
                block["qkv_w"],
                bias=block.get("qkv_b"),
                transpose_b=True,
                compute_kernel_config=compute_cfg,
                memory_config=act_mem_cfg,
                **linear_kwargs,
            )
            try:
                # The sharded implementation expects interleaved QKV-by-head input.
                # If the output isn't sharded, fall back to the known-correct separate Q/K/V path.
                if hasattr(qkv, "is_sharded") and not qkv.is_sharded():
                    raise RuntimeError("Fused QKV output is not sharded; skipping sharded split op.")
                # Avoid transpose inside the split op; explicit permute on K is typically faster for Swin window sizes.
                q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv, num_heads=num_heads, transpose_key=False
                )
                key_is_transposed = False
            except Exception:
                q = None
                k = None
                v = None
            ttnn.deallocate(qkv)

        if q is None or k is None or v is None:
            q = ttnn.linear(
                xw,
                block["q_w"],
                bias=block["q_b"],
                transpose_b=True,
                compute_kernel_config=compute_cfg,
                memory_config=act_mem_cfg,
                **linear_kwargs,
            )
            k = ttnn.linear(
                xw,
                block["k_w"],
                bias=block["k_b"],
                transpose_b=True,
                compute_kernel_config=compute_cfg,
                memory_config=act_mem_cfg,
                **linear_kwargs,
            )
            v = ttnn.linear(
                xw,
                block["v_w"],
                bias=block["v_b"],
                transpose_b=True,
                compute_kernel_config=compute_cfg,
                memory_config=act_mem_cfg,
                **linear_kwargs,
            )
        ttnn.deallocate(xw)

        # [B*nW, N, C] -> [B*nW, H, N, Hd]
        N = window * window
        if q is not None and (len(q.shape) == 3):
            q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
            k = ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
            v = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
            q = ttnn.reshape(q, (B * num_windows, N, num_heads, head_dim), memory_config=act_mem_cfg)
            k = ttnn.reshape(k, (B * num_windows, N, num_heads, head_dim), memory_config=act_mem_cfg)
            v = ttnn.reshape(v, (B * num_windows, N, num_heads, head_dim), memory_config=act_mem_cfg)
            q = ttnn.permute(q, (0, 2, 1, 3), memory_config=act_mem_cfg)
            k = ttnn.permute(k, (0, 2, 1, 3), memory_config=act_mem_cfg)
            v = ttnn.permute(v, (0, 2, 1, 3), memory_config=act_mem_cfg)
        _attn_mark("qkv end")

        if debug_attn:
            print("[maskformer][backbone_attn] scores begin", flush=True)
        scale = head_dim**-0.5
        if enable_inplace and hasattr(ttnn, "multiply_"):
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
            try:
                q = ttnn.multiply_(q, scale, use_legacy=False)
            except TypeError:
                q = ttnn.multiply_(q, scale)
            except Exception:
                q = q * scale
        else:
            # Keep scale on the pre-TILE layout (typically row-major) for the baseline path.
            q = q * scale
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        k_t = k if key_is_transposed else ttnn.permute(k, (0, 1, 3, 2), memory_config=act_mem_cfg)

        k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        attn = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=compute_cfg,
            memory_config=act_mem_cfg,
            **matmul_kwargs,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        if not key_is_transposed:
            ttnn.deallocate(k)

        # Add relative position bias (broadcast on batch/windows)
        if enable_inplace and hasattr(ttnn, "add_"):
            try:
                attn = ttnn.add_(attn, block["rel_bias"], use_legacy=False)
            except TypeError:
                attn = ttnn.add_(attn, block["rel_bias"])
            except Exception:
                attn = ttnn.add(attn, block["rel_bias"], memory_config=act_mem_cfg)
        else:
            attn = ttnn.add(attn, block["rel_bias"], memory_config=act_mem_cfg)
        _attn_mark("scores end")

        masked_softmax_done = False
        # Add attention mask for shifted blocks
        if shift:
            use_fused_mask_softmax = os.environ.get("MASKFORMER_TT_BACKBONE_FUSED_MASK_SOFTMAX", "0").strip() == "1"
            use_tile_mask_add = os.environ.get("MASKFORMER_TT_BACKBONE_TILE_MASK_ADD", "0").strip() == "1"
            if use_fused_mask_softmax and B == 1 and hasattr(ttnn, "scale_mask_softmax_in_place"):
                try:
                    mask_tile = self._get_attn_mask_tile(stage_idx=stage_idx, height=Hp, width=Wp)
                    attn = ttnn.scale_mask_softmax_in_place(
                        attn,
                        scale=None,
                        mask=mask_tile,
                        is_causal_mask=False,
                        compute_kernel_config=compute_cfg,
                        numeric_stable=True,
                    )
                    ttnn.deallocate(mask_tile)
                    masked_softmax_done = True
                except Exception:
                    masked_softmax_done = False

            if not masked_softmax_done:
                tile_mask_added = False
                if use_tile_mask_add and B == 1:
                    try:
                        mask_tile = self._get_attn_mask_tile(
                            stage_idx=stage_idx, height=Hp, width=Wp, clone=not reuse_mask_cache
                        )
                        if enable_inplace and hasattr(ttnn, "add_"):
                            try:
                                attn = ttnn.add_(attn, mask_tile, use_legacy=False)
                            except TypeError:
                                attn = ttnn.add_(attn, mask_tile)
                        else:
                            attn_added = ttnn.add(attn, mask_tile, memory_config=act_mem_cfg, use_legacy=False)
                            ttnn.deallocate(attn)
                            attn = attn_added
                        if not reuse_mask_cache:
                            ttnn.deallocate(mask_tile)
                        tile_mask_added = True
                    except Exception:
                        tile_mask_added = False

                if not tile_mask_added:
                    mask_rm = self._get_attn_mask(
                        stage_idx=stage_idx, height=Hp, width=Wp, clone=not reuse_mask_cache
                    )  # [1, nW, 1, N, N] row-major
                    attn_rm = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
                    attn_rm = ttnn.reshape(attn_rm, (B, num_windows, num_heads, N, N), memory_config=act_mem_cfg)
                    # Note: TTNN inplace elementwise ops do not currently support row-major layout outputs.
                    attn_rm = ttnn.add(attn_rm, mask_rm, memory_config=act_mem_cfg, use_legacy=False)
                    attn = ttnn.reshape(attn_rm, (B * num_windows, num_heads, N, N), memory_config=act_mem_cfg)
                    attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
                    ttnn.deallocate(attn_rm)
                    if not reuse_mask_cache:
                        ttnn.deallocate(mask_rm)

        if debug_attn:
            print("[maskformer][backbone_attn] softmax begin", flush=True)
        if not masked_softmax_done:
            attn = ttnn.softmax(attn, dim=-1, memory_config=act_mem_cfg)
        if safety_sync_windows1:
            tt_sync = require_ttnn("synchronize window attention (windows=1)")
            if hasattr(tt_sync, "synchronize_device"):
                tt_sync.synchronize_device(self.device)
        _attn_mark("softmax end")
        if debug_attn:
            print("[maskformer][backbone_attn] ctx begin", flush=True)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        ctx = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=compute_cfg,
            memory_config=act_mem_cfg,
            **matmul_kwargs,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)
        if safety_sync_windows1:
            tt_sync = require_ttnn("synchronize window attention (windows=1)")
            if hasattr(tt_sync, "synchronize_device"):
                tt_sync.synchronize_device(self.device)
        _attn_mark("ctx end")

        if debug_attn:
            print("[maskformer][backbone_attn] out proj begin", flush=True)
        ctx_seq = None
        if hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "concatenate_heads"):
            try:
                ctx_seq = ttnn.transformer.concatenate_heads(ctx)
            except Exception:
                ctx_seq = None
        if ctx_seq is None:
            ctx_seq = ttnn.permute(ctx, (0, 2, 1, 3), memory_config=act_mem_cfg)
            ctx_seq = ttnn.reshape(ctx_seq, (B * num_windows, N, C), memory_config=act_mem_cfg)
        ttnn.deallocate(ctx)
        ctx_seq = ttnn.to_layout(ctx_seq, ttnn.TILE_LAYOUT, memory_config=act_mem_cfg)
        out = ttnn.linear(
            ctx_seq,
            block["attn_out_w"],
            bias=block.get("attn_out_b"),
            transpose_b=True,
            compute_kernel_config=compute_cfg,
            memory_config=act_mem_cfg,
            **linear_kwargs,
        )
        ttnn.deallocate(ctx_seq)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=act_mem_cfg)
        out = ttnn.reshape(
            out, (B, num_windows_h, num_windows_w, window, window, C), memory_config=act_mem_cfg
        )
        out = ttnn.permute(out, (0, 1, 3, 2, 4, 5), memory_config=act_mem_cfg)
        out = ttnn.reshape(out, (B, Hp, Wp, C), memory_config=act_mem_cfg)
        _attn_mark("out proj end")

        if shift:
            out = _roll(out, shifts=(shift, shift), dims=(1, 2), memory_config=act_mem_cfg)
            _attn_mark("roll back end")
        if pad_h or pad_w:
            if debug_attn:
                print("[maskformer][backbone_attn] crop begin", flush=True)
            out_flat = ttnn.reshape(out, (B, Hp, Wp * C), memory_config=act_mem_cfg)
            out_flat = ttnn.slice(
                out_flat,
                slice_start=[0, 0, 0],
                slice_end=[B, H, W * C],
                slice_step=[1, 1, 1],
                memory_config=act_mem_cfg,
            )
            out = ttnn.reshape(out_flat, (B, H, W, C), memory_config=act_mem_cfg)
            if safety_sync_windows1:
                tt_sync = require_ttnn("synchronize window attention (windows=1)")
                if hasattr(tt_sync, "synchronize_device"):
                    tt_sync.synchronize_device(self.device)
            _attn_mark("crop end")
        _attn_mark("return")
        return out
