# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Native TTNNModule vision transformer for dots.ocr.

Full pipeline: PatchEmbed -> 42 VisionBlocks -> post-trunk RMSNorm -> PatchMerger.
All sub-modules are native TTNNModules with proper lifecycle (preprocess_weights,
move_weights_to_device, forward).

Includes: 2D RoPE, vision RMSNorm, SwiGLU MLP, patch embedding, vision attention,
vision block, patch merger, and the top-level vision tower.
"""

from __future__ import annotations


import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from ttnn.operations.transformer import SDPAProgramConfig


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate-half helper for RoPE: [-x2, x1] from [x1, x2]."""
    last = x.shape[-1]
    half = last // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (x.shape[0], x.shape[1], x.shape[2], last))
    neg_x2 = ttnn.mul(x2, -1, use_legacy=False)
    return ttnn.concat([neg_x2, x1], dim=-1)


def apply_rotary_tt(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    out_dtype=None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply rotary embedding to Q and K tensors in fp32 then cast back."""
    if out_dtype is None:
        out_dtype = ttnn.bfloat16

    f32 = getattr(ttnn, "float32", None)

    if f32 is not None:
        qf = ttnn.typecast(q, dtype=f32)
        kf = ttnn.typecast(k, dtype=f32)
        cos_f = ttnn.typecast(cos, dtype=f32)
        sin_f = ttnn.typecast(sin, dtype=f32)
    else:
        qf, kf, cos_f, sin_f = q, k, cos, sin

    q_embed = ttnn.add(
        ttnn.mul(qf, cos_f, use_legacy=False),
        ttnn.mul(_rotate_half(qf), sin_f, use_legacy=False),
    )
    k_embed = ttnn.add(
        ttnn.mul(kf, cos_f, use_legacy=False),
        ttnn.mul(_rotate_half(kf), sin_f, use_legacy=False),
    )

    if f32 is not None and out_dtype is not None:
        q_embed = ttnn.typecast(q_embed, dtype=out_dtype)
        k_embed = ttnn.typecast(k_embed, dtype=out_dtype)

    return q_embed, k_embed


class TTNNDotsVision2DRoPE:
    """2D factored RoPE for Dots vision attention.

    Not a TTNNModule -- no learnable weights. Produces cos/sin tensors
    and cu_seqlens given grid_thw and a device reference.
    """

    def __init__(
        self,
        *,
        device,
        head_dim: int = 128,
        spatial_merge_size: int = 2,
        theta: float = 10000.0,
    ):
        self.device = device
        self.head_dim = head_dim
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.rotary_dim = head_dim // 2
        if self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")

        self._inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))

    def build(
        self,
        grid_thw: torch.Tensor,
        seq_len: int,
    ) -> tuple[tuple[ttnn.Tensor, ttnn.Tensor], ttnn.Tensor]:
        """Build 2D RoPE cos/sin tables and cu_seqlens for vision attention."""
        g = grid_thw.detach().cpu() if getattr(grid_thw, "is_cuda", False) else grid_thw
        if g.dim() != 2 or g.shape[1] != 3:
            raise ValueError(f"grid_thw must be [N,3], got {g.shape}")

        token_counts = [int(t) * int(h) * int(w) for t, h, w in g.tolist()]
        expected = sum(token_counts)
        if seq_len != expected:
            raise ValueError(f"seq_len={seq_len} != grid_thw total={expected}")

        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        sms = self.spatial_merge_size

        rope_dtype = getattr(ttnn, "float32", None) or ttnn.bfloat16

        inv = ttnn.from_torch(
            self._inv_freq.to(torch.float32).reshape(1, 1, 1, self.rotary_dim // 2),
            device=self.device,
            dtype=rope_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        hpos_segments = []
        wpos_segments = []
        cu = [0]
        running = 0

        for t, h, w in g.tolist():
            t, h, w = int(t), int(h), int(w)
            if h % sms != 0 or w % sms != 0:
                raise ValueError(f"grid {h}x{w} not divisible by spatial_merge_size={sms}")

            h_ids = ttnn.arange(0, h, dtype=rope_dtype, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
            h_ids = ttnn.reshape(h_ids, (1, 1, h, 1))
            ones_w = ttnn.ones(
                (1, 1, 1, w), dtype=rope_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mem
            )
            h_grid = ttnn.matmul(h_ids, ones_w, memory_config=mem)
            h_grid = ttnn.reshape(h_grid, (h, w))
            h_grid = ttnn.reshape(h_grid, (h // sms, sms, w // sms, sms))
            h_grid = ttnn.permute(h_grid, (0, 2, 1, 3))
            hpos = ttnn.reshape(h_grid, (1, 1, h * w, 1))

            ones_h = ttnn.ones(
                (1, 1, h, 1), dtype=rope_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mem
            )
            w_ids = ttnn.arange(0, w, dtype=rope_dtype, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
            w_ids = ttnn.reshape(w_ids, (1, 1, 1, w))
            w_grid = ttnn.matmul(ones_h, w_ids, memory_config=mem)
            w_grid = ttnn.reshape(w_grid, (h, w))
            w_grid = ttnn.reshape(w_grid, (h // sms, sms, w // sms, sms))
            w_grid = ttnn.permute(w_grid, (0, 2, 1, 3))
            wpos = ttnn.reshape(w_grid, (1, 1, h * w, 1))

            if t != 1:
                hpos = ttnn.concat([hpos] * t, dim=2)
                wpos = ttnn.concat([wpos] * t, dim=2)

            hpos_segments.append(hpos)
            wpos_segments.append(wpos)
            running += t * h * w
            cu.append(running)

        hpos_all = ttnn.concat(hpos_segments, dim=2) if len(hpos_segments) > 1 else hpos_segments[0]
        wpos_all = ttnn.concat(wpos_segments, dim=2) if len(wpos_segments) > 1 else wpos_segments[0]

        hpos_rm = ttnn.to_layout(ttnn.typecast(hpos_all, dtype=rope_dtype), ttnn.TILE_LAYOUT)
        wpos_rm = ttnn.to_layout(ttnn.typecast(wpos_all, dtype=rope_dtype), ttnn.TILE_LAYOUT)
        freqs_h = ttnn.matmul(hpos_rm, inv, memory_config=mem)
        freqs_w = ttnn.matmul(wpos_rm, inv, memory_config=mem)

        cos_h = ttnn.cos(freqs_h, memory_config=mem)
        sin_h = ttnn.sin(freqs_h, memory_config=mem)
        cos_w = ttnn.cos(freqs_w, memory_config=mem)
        sin_w = ttnn.sin(freqs_w, memory_config=mem)

        cos_half = ttnn.concat([cos_h, cos_w], dim=-1)
        sin_half = ttnn.concat([sin_h, sin_w], dim=-1)

        cos_full = ttnn.concat([cos_half, cos_half], dim=-1)
        sin_full = ttnn.concat([sin_half, sin_half], dim=-1)

        cos = ttnn.typecast(cos_full, dtype=ttnn.bfloat16)
        sin = ttnn.typecast(sin_full, dtype=ttnn.bfloat16)
        rot_mats = (
            ttnn.to_layout(cos, ttnn.TILE_LAYOUT),
            ttnn.to_layout(sin, ttnn.TILE_LAYOUT),
        )

        cu_t = ttnn.from_torch(
            torch.tensor(cu, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        return rot_mats, cu_t


# ---------------------------------------------------------------------------
# Vision RMSNorm
# ---------------------------------------------------------------------------


class TTNNDotsVisionRMSNorm(TTNNModule):
    """RMSNorm for Dots vision blocks.

    When the HF checkpoint contains bias for the norm layer, this falls back
    to LayerNorm (matching HF behavior). Otherwise uses RMSNorm.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self._use_layer_norm = False
        self._weight_torch = None
        self._bias_torch = None
        self.tt_weight = None
        self.tt_bias = None

    @classmethod
    def from_torch(cls, hf_norm):
        new_norm = cls()
        new_norm._fallback_torch_layer = hf_norm
        new_norm.eps = getattr(hf_norm, "variance_epsilon", getattr(hf_norm, "eps", 1e-5))

        if hasattr(hf_norm, "weight") and hf_norm.weight is not None:
            new_norm._weight_torch = hf_norm.weight.data.clone()
        else:
            raise ValueError("Vision RMSNorm requires a weight parameter")

        if hasattr(hf_norm, "bias") and hf_norm.bias is not None:
            new_norm._bias_torch = hf_norm.bias.data.clone()
            new_norm._use_layer_norm = True

        return new_norm

    def preprocess_weights_impl(self):
        if self._use_layer_norm:
            self.tt_weight = ttnn.from_torch(
                self._weight_torch.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            if self._bias_torch is not None:
                self.tt_bias = ttnn.from_torch(
                    self._bias_torch.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
        else:
            dim = self._weight_torch.numel()
            tile = 32
            w = self._weight_torch.to(torch.bfloat16)
            w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
            self.tt_weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        if self._use_layer_norm:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.tt_bias is not None:
                self.tt_bias = ttnn.to_device(self.tt_bias, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._use_layer_norm:
            return ttnn.layer_norm(
                x,
                weight=self.tt_weight,
                bias=self.tt_bias,
                epsilon=self.eps,
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            return ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                compute_kernel_config=self.compute_kernel_config,
            )


# ---------------------------------------------------------------------------
# Vision SwiGLU MLP
# ---------------------------------------------------------------------------


class TTNNDotsVisionMLP(TTNNModule):
    """SwiGLU MLP for Dots vision blocks: y = fc2(silu(fc1(x)) * fc3(x))."""

    def __init__(self):
        super().__init__()
        self._fc1_weight = None
        self._fc1_bias = None
        self._fc2_weight = None
        self._fc2_bias = None
        self._fc3_weight = None
        self._fc3_bias = None
        self.tt_fc1_weight = None
        self.tt_fc1_bias = None
        self.tt_fc2_weight = None
        self.tt_fc2_bias = None
        self.tt_fc3_weight = None
        self.tt_fc3_bias = None

    @classmethod
    def from_torch(cls, hf_mlp):
        new_mlp = cls()
        new_mlp._fallback_torch_layer = hf_mlp

        if hasattr(hf_mlp, "fc1"):
            new_mlp._fc1_weight = hf_mlp.fc1.weight.data.clone()
            if hf_mlp.fc1.bias is not None:
                new_mlp._fc1_bias = hf_mlp.fc1.bias.data.clone()
        elif hasattr(hf_mlp, "gate_proj"):
            new_mlp._fc1_weight = hf_mlp.gate_proj.weight.data.clone()
            if hf_mlp.gate_proj.bias is not None:
                new_mlp._fc1_bias = hf_mlp.gate_proj.bias.data.clone()

        if hasattr(hf_mlp, "fc2"):
            new_mlp._fc2_weight = hf_mlp.fc2.weight.data.clone()
            if hf_mlp.fc2.bias is not None:
                new_mlp._fc2_bias = hf_mlp.fc2.bias.data.clone()
        elif hasattr(hf_mlp, "down_proj"):
            new_mlp._fc2_weight = hf_mlp.down_proj.weight.data.clone()
            if hf_mlp.down_proj.bias is not None:
                new_mlp._fc2_bias = hf_mlp.down_proj.bias.data.clone()

        if hasattr(hf_mlp, "fc3"):
            new_mlp._fc3_weight = hf_mlp.fc3.weight.data.clone()
            if hf_mlp.fc3.bias is not None:
                new_mlp._fc3_bias = hf_mlp.fc3.bias.data.clone()
        elif hasattr(hf_mlp, "up_proj"):
            new_mlp._fc3_weight = hf_mlp.up_proj.weight.data.clone()
            if hf_mlp.up_proj.bias is not None:
                new_mlp._fc3_bias = hf_mlp.up_proj.bias.data.clone()

        return new_mlp

    def _prepare_weight(self, w):
        if w is None:
            return None
        return torch.transpose(w, -2, -1).contiguous()

    def _prepare_bias(self, b):
        if b is None:
            return None
        return b.reshape(1, 1, 1, -1)

    def preprocess_weights_impl(self):
        def _to_host(w, layout=ttnn.TILE_LAYOUT):
            if w is None:
                return None
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout)

        self.tt_fc1_weight = _to_host(self._prepare_weight(self._fc1_weight))
        self.tt_fc1_bias = _to_host(self._prepare_bias(self._fc1_bias))
        self.tt_fc2_weight = _to_host(self._prepare_weight(self._fc2_weight))
        self.tt_fc2_bias = _to_host(self._prepare_bias(self._fc2_bias))
        self.tt_fc3_weight = _to_host(self._prepare_weight(self._fc3_weight))
        self.tt_fc3_bias = _to_host(self._prepare_bias(self._fc3_bias))

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.tt_fc1_weight = _to_dev(self.tt_fc1_weight)
        self.tt_fc1_bias = _to_dev(self.tt_fc1_bias)
        self.tt_fc2_weight = _to_dev(self.tt_fc2_weight)
        self.tt_fc2_bias = _to_dev(self.tt_fc2_bias)
        self.tt_fc3_weight = _to_dev(self.tt_fc3_weight)
        self.tt_fc3_bias = _to_dev(self.tt_fc3_bias)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG

        gate = ttnn.linear(
            hidden_states,
            self.tt_fc1_weight,
            bias=self.tt_fc1_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate = ttnn.silu(gate, memory_config=mem)

        up = ttnn.linear(
            hidden_states,
            self.tt_fc3_weight,
            bias=self.tt_fc3_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )

        gate_up = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        output = ttnn.linear(
            gate_up,
            self.tt_fc2_weight,
            bias=self.tt_fc2_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(gate_up)

        return output


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class TTNNDotsVisionPatchEmbed(TTNNModule):
    """Patch embedding for Dots vision (14x14 patches, no CLS token, no pos embed)."""

    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.in_channels = 3
        self.embed_dim = 1536
        self._proj_weight = None
        self._proj_bias = None
        self._norm_weight = None
        self.tt_proj_weight = None
        self.tt_proj_bias = None
        self.tt_norm_weight = None
        self.compute_kernel_config = None
        self._bypass_tensor_wrapping = True

    @classmethod
    def from_torch(cls, hf_patch_embed, patch_size=14, in_channels=3, embed_dim=1536):
        new_pe = cls()
        new_pe._fallback_torch_layer = hf_patch_embed
        new_pe.patch_size = patch_size
        new_pe.in_channels = in_channels
        new_pe.embed_dim = embed_dim

        proj = None
        if hasattr(hf_patch_embed, "proj"):
            proj = hf_patch_embed.proj
        elif hasattr(hf_patch_embed, "patchifier") and hasattr(hf_patch_embed.patchifier, "proj"):
            proj = hf_patch_embed.patchifier.proj

        if proj is not None:
            w = proj.weight.data.clone()
            if w.dim() == 4:
                w = w.reshape(w.shape[0], -1)
            new_pe._proj_weight = w
            if proj.bias is not None:
                new_pe._proj_bias = proj.bias.data.clone()

        norm = None
        if hasattr(hf_patch_embed, "norm"):
            norm = hf_patch_embed.norm
        elif hasattr(hf_patch_embed, "patchifier") and hasattr(hf_patch_embed.patchifier, "norm"):
            norm = hf_patch_embed.patchifier.norm

        if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
            new_pe._norm_weight = norm.weight.data.clone()

        return new_pe

    def preprocess_weights_impl(self):
        if self._proj_weight is not None:
            self.tt_proj_weight = ttnn.from_torch(
                self._proj_weight.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        if self._proj_bias is not None:
            self.tt_proj_bias = ttnn.from_torch(
                self._proj_bias.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        if self._norm_weight is not None:
            dim = self._norm_weight.numel()
            tile = 32
            w = self._norm_weight.to(torch.bfloat16)
            w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
            self.tt_norm_weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if self.tt_proj_weight is not None:
            self.tt_proj_weight = ttnn.to_device(self.tt_proj_weight, self.device, memory_config=mem)
        if self.tt_proj_bias is not None:
            self.tt_proj_bias = ttnn.to_device(self.tt_proj_bias, self.device, memory_config=mem)
        if self.tt_norm_weight is not None:
            self.tt_norm_weight = ttnn.to_device(self.tt_norm_weight, self.device, memory_config=mem)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None) -> ttnn.Tensor:
        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if pixel_values.dim() == 2:
            x = pixel_values.to(torch.bfloat16).unsqueeze(0)
            x_tt = ttnn.from_torch(
                x,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )
            if len(x_tt.shape) == 3:
                x_tt = ttnn.reshape(x_tt, (1, 1, x_tt.shape[1], x_tt.shape[2]))

            out = ttnn.linear(
                x_tt,
                self.tt_proj_weight,
                bias=self.tt_proj_bias,
                transpose_b=True,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel_config,
            )

            if self.tt_norm_weight is not None:
                out = ttnn.rms_norm(
                    out,
                    weight=self.tt_norm_weight,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )

            return out

        B, C, H, W = pixel_values.shape

        if grid_thw is not None:
            g = grid_thw.detach().cpu() if hasattr(grid_thw, "is_cuda") and grid_thw.is_cuda else grid_thw
            if g.dim() == 1:
                g = g.unsqueeze(0)
            temporal = int(g[0, 0].item())
            height_patches = int(g[0, 1].item())
            width_patches = int(g[0, 2].item())
        else:
            temporal = 1
            height_patches = H // self.patch_size
            width_patches = W // self.patch_size

        num_patches = temporal * height_patches * width_patches

        temporal_patch_size = temporal
        x = pixel_values.view(-1, C, temporal_patch_size, self.patch_size, self.patch_size)
        x = x[:, :, 0]
        x = x.reshape(1, num_patches, C * self.patch_size * self.patch_size)

        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        if len(x_tt.shape) == 3:
            x_tt = ttnn.reshape(x_tt, (1, 1, x_tt.shape[1], x_tt.shape[2]))

        out = ttnn.linear(
            x_tt,
            self.tt_proj_weight,
            bias=self.tt_proj_bias,
            transpose_b=True,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )

        if self.tt_norm_weight is not None:
            out = ttnn.rms_norm(
                out,
                weight=self.tt_norm_weight,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )

        return out


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------


class TTNNDotsVisionAttention(TTNNModule):
    """Vision attention for Dots OCR with per-segment SDPA and 2D RoPE."""

    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.num_heads = 12
        self.head_dim = 128
        self.num_kv_heads = 12
        self._qkv_weight = None
        self._qkv_bias = None
        self._o_proj_weight = None
        self._o_proj_bias = None

        self.tt_qkv_weight = None
        self.tt_qkv_bias = None
        self.tt_o_proj_weight = None
        self.tt_o_proj_bias = None

    @classmethod
    def from_torch(cls, hf_attn, hidden_size=1536, num_heads=12):
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn
        new_attn.hidden_size = hidden_size
        new_attn.num_heads = num_heads
        new_attn.head_dim = hidden_size // num_heads
        new_attn.num_kv_heads = num_heads

        if hasattr(hf_attn, "qkv"):
            new_attn._qkv_weight = hf_attn.qkv.weight.data.clone()
            if hf_attn.qkv.bias is not None:
                new_attn._qkv_bias = hf_attn.qkv.bias.data.clone()
        elif hasattr(hf_attn, "qkv_proj"):
            new_attn._qkv_weight = hf_attn.qkv_proj.weight.data.clone()
            if hf_attn.qkv_proj.bias is not None:
                new_attn._qkv_bias = hf_attn.qkv_proj.bias.data.clone()
        else:
            q_weight = getattr(hf_attn, "q_proj", getattr(hf_attn, "wq", None))
            k_weight = getattr(hf_attn, "k_proj", getattr(hf_attn, "wk", None))
            v_weight = getattr(hf_attn, "v_proj", getattr(hf_attn, "wv", None))
            if q_weight is not None and k_weight is not None and v_weight is not None:
                new_attn._qkv_weight = torch.cat(
                    [q_weight.weight.data, k_weight.weight.data, v_weight.weight.data],
                    dim=0,
                )
                if q_weight.bias is not None and k_weight.bias is not None and v_weight.bias is not None:
                    new_attn._qkv_bias = torch.cat(
                        [q_weight.bias.data, k_weight.bias.data, v_weight.bias.data],
                        dim=0,
                    )

        o_proj = getattr(hf_attn, "proj", getattr(hf_attn, "o_proj", getattr(hf_attn, "out_proj", None)))
        if o_proj is not None:
            new_attn._o_proj_weight = o_proj.weight.data.clone()
            if o_proj.bias is not None:
                new_attn._o_proj_bias = o_proj.bias.data.clone()

        return new_attn

    def _transpose_for_linear(self, w):
        if w is None:
            return None
        return torch.transpose(w, -2, -1).contiguous()

    def preprocess_weights_impl(self):
        def _to_host(w, layout=ttnn.TILE_LAYOUT):
            if w is None:
                return None
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout)

        self.tt_qkv_weight = _to_host(self._transpose_for_linear(self._qkv_weight))
        if self._qkv_bias is not None:
            self.tt_qkv_bias = _to_host(self._qkv_bias.reshape(1, 1, 1, -1))
        self.tt_o_proj_weight = _to_host(self._transpose_for_linear(self._o_proj_weight))
        if self._o_proj_bias is not None:
            self.tt_o_proj_bias = _to_host(self._o_proj_bias.reshape(1, 1, 1, -1))

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.tt_qkv_weight = _to_dev(self.tt_qkv_weight)
        self.tt_qkv_bias = _to_dev(self.tt_qkv_bias)
        self.tt_o_proj_weight = _to_dev(self.tt_o_proj_weight)
        self.tt_o_proj_bias = _to_dev(self.tt_o_proj_bias)

    def _get_sdpa_program_config(self, seq_len: int):
        if seq_len <= 2048:
            return None
        chunk_size = min(256, seq_len)
        chunk_size = (chunk_size // 32) * 32
        if chunk_size < 32:
            chunk_size = 32
        return SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        cu_seqlens: ttnn.Tensor | torch.Tensor | list | None = None,
    ) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG
        s = int(hidden_states.shape[2])
        h = self.num_heads
        hd = self.head_dim

        qkv = ttnn.linear(
            hidden_states,
            self.tt_qkv_weight,
            bias=self.tt_qkv_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )

        qkv = ttnn.reshape(qkv, (1, 1, s, 3, h, hd))
        q = ttnn.slice(qkv, (0, 0, 0, 0, 0, 0), (1, 1, s, 1, h, hd))
        k = ttnn.slice(qkv, (0, 0, 0, 1, 0, 0), (1, 1, s, 2, h, hd))
        v = ttnn.slice(qkv, (0, 0, 0, 2, 0, 0), (1, 1, s, 3, h, hd))

        q = ttnn.permute(ttnn.reshape(q, (1, s, h, hd)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (1, s, h, hd)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (1, s, h, hd)), (0, 2, 1, 3))

        if rot_mats is not None and len(rot_mats) == 2:
            cos, sin = rot_mats
            q, k = apply_rotary_tt(q, k, cos, sin, out_dtype=ttnn.bfloat16)

        if cu_seqlens is None:
            program_config = self._get_sdpa_program_config(s)
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=program_config,
            )
            ctx = ttnn.reshape(ttnn.permute(ctx, (0, 2, 1, 3)), (1, 1, s, h * hd))
            return ttnn.linear(
                ctx,
                self.tt_o_proj_weight,
                bias=self.tt_o_proj_bias,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel_config,
            )

        cu_host = self._cu_seqlens_to_list(cu_seqlens, s)

        ctx_segments = []

        for seg_start, seg_end in zip(cu_host[:-1], cu_host[1:]):
            seg_start, seg_end = int(seg_start), int(seg_end)
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue

            q_seg = ttnn.slice(q, (0, 0, seg_start, 0), (1, h, seg_end, hd))
            k_seg = ttnn.slice(k, (0, 0, seg_start, 0), (1, h, seg_end, hd))
            v_seg = ttnn.slice(v, (0, 0, seg_start, 0), (1, h, seg_end, hd))

            program_config = self._get_sdpa_program_config(seg_len)
            ctx_seg = ttnn.transformer.scaled_dot_product_attention(
                q_seg,
                k_seg,
                v_seg,
                is_causal=False,
                program_config=program_config,
            )
            ctx_segments.append(ctx_seg)

        ctx = ttnn.concat(ctx_segments, dim=2) if len(ctx_segments) > 1 else ctx_segments[0]

        ctx = ttnn.reshape(ttnn.permute(ctx, (0, 2, 1, 3)), (1, 1, s, h * hd))
        return ttnn.linear(
            ctx,
            self.tt_o_proj_weight,
            bias=self.tt_o_proj_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _cu_seqlens_to_list(self, cu_seqlens, expected_total: int) -> list[int]:
        if isinstance(cu_seqlens, list):
            cu_host = cu_seqlens
        elif isinstance(cu_seqlens, torch.Tensor):
            cu_host = cu_seqlens.flatten().to(torch.int64).tolist()
        elif isinstance(cu_seqlens, ttnn.Tensor):
            composer = None
            if self.device is not None and self.device.get_num_devices() > 1:
                composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
            out = ttnn.to_torch(cu_seqlens, mesh_composer=composer) if composer else ttnn.to_torch(cu_seqlens)
            try:
                num_dev = self.device.get_num_devices() if self.device is not None else 1
                if num_dev > 1 and out.shape[0] % num_dev == 0:
                    per = out.shape[0] // num_dev
                    out = out[:per]
            except Exception:
                pass
            cu_host = out.flatten().to(torch.int64).tolist()
        else:
            cu_host = list(cu_seqlens)

        if len(cu_host) < 2 or cu_host[0] != 0 or cu_host[-1] != expected_total:
            raise ValueError(f"Invalid cu_seqlens={cu_host} for S={expected_total}")

        return cu_host


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------


class TTNNDotsVisionBlock(TTNNModule):
    """Single vision transformer block with post-norm architecture."""

    def __init__(self):
        super().__init__()
        self.norm1 = None
        self.norm2 = None
        self.attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, hf_block, hidden_size=1536, num_heads=12):
        new_block = cls()
        new_block._fallback_torch_layer = hf_block

        new_block.norm1 = TTNNDotsVisionRMSNorm.from_torch(hf_block.norm1)
        new_block.norm2 = TTNNDotsVisionRMSNorm.from_torch(hf_block.norm2)

        attn_module = getattr(hf_block, "attn", getattr(hf_block, "attention", getattr(hf_block, "self_attn", None)))
        if attn_module is None:
            raise ValueError("Could not find attention sub-module in HF block")
        new_block.attn = TTNNDotsVisionAttention.from_torch(attn_module, hidden_size=hidden_size, num_heads=num_heads)

        mlp_module = getattr(hf_block, "mlp", getattr(hf_block, "feed_forward", None))
        if mlp_module is None:
            raise ValueError("Could not find MLP sub-module in HF block")
        new_block.mlp = TTNNDotsVisionMLP.from_torch(mlp_module)

        return new_block

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        cu_seqlens=None,
    ) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, rot_mats=rot_mats, cu_seqlens=cu_seqlens)
        hidden_states = ttnn.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class TTNNDotsPatchMerger(TTNNModule):
    """Patch merger for Dots vision: spatial merge + LayerNorm/RMSNorm + MLP(GELU)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.out_hidden_size = 1536
        self.spatial_merge_size = 2
        self.mlp_size = None

        self._use_layer_norm = False
        self._ln_weight = None
        self._ln_bias = None
        self._w1_weight = None
        self._w2_weight = None
        self._w1_bias = None
        self._w2_bias = None

        self.tt_ln_weight = None
        self.tt_ln_bias = None
        self.tt_w1 = None
        self.tt_w2 = None
        self.tt_w1_bias = None
        self.tt_w2_bias = None

    @classmethod
    def from_torch(cls, hf_merger, hidden_size=1536, out_hidden_size=1536, spatial_merge_size=2):
        new_merger = cls()
        new_merger._fallback_torch_layer = hf_merger
        new_merger.hidden_size = hidden_size
        new_merger.out_hidden_size = out_hidden_size
        new_merger.spatial_merge_size = spatial_merge_size
        new_merger.mlp_size = hidden_size * (spatial_merge_size**2)

        ln_q = getattr(hf_merger, "ln_q", getattr(hf_merger, "norm", None))
        if ln_q is not None:
            if hasattr(ln_q, "weight") and ln_q.weight is not None:
                new_merger._ln_weight = ln_q.weight.data.clone()
            if hasattr(ln_q, "bias") and ln_q.bias is not None:
                new_merger._ln_bias = ln_q.bias.data.clone()
                new_merger._use_layer_norm = True

        mlp = getattr(hf_merger, "mlp", getattr(hf_merger, "feed_forward", None))
        if mlp is not None:
            if hasattr(mlp, "0") or (hasattr(mlp, "__getitem__") and len(list(mlp.children())) >= 3):
                try:
                    children = list(mlp.children())
                    fc1 = children[0]
                    fc2 = children[2] if len(children) > 2 else children[1]
                    new_merger._w1_weight = torch.transpose(fc1.weight.data, -2, -1).contiguous()
                    new_merger._w2_weight = torch.transpose(fc2.weight.data, -2, -1).contiguous()
                    if hasattr(fc1, "bias") and fc1.bias is not None:
                        new_merger._w1_bias = fc1.bias.data.clone().reshape(1, 1, 1, -1)
                    if hasattr(fc2, "bias") and fc2.bias is not None:
                        new_merger._w2_bias = fc2.bias.data.clone().reshape(1, 1, 1, -1)
                except (IndexError, AttributeError):
                    pass
            if new_merger._w1_weight is None:
                for name in ("0", "fc1", "linear1"):
                    sub = getattr(mlp, name, None)
                    if sub is not None and hasattr(sub, "weight"):
                        new_merger._w1_weight = torch.transpose(sub.weight.data, -2, -1).contiguous()
                        if hasattr(sub, "bias") and sub.bias is not None:
                            new_merger._w1_bias = sub.bias.data.clone().reshape(1, 1, 1, -1)
                        break
                for name in ("2", "fc2", "linear2"):
                    sub = getattr(mlp, name, None)
                    if sub is not None and hasattr(sub, "weight"):
                        new_merger._w2_weight = torch.transpose(sub.weight.data, -2, -1).contiguous()
                        if hasattr(sub, "bias") and sub.bias is not None:
                            new_merger._w2_bias = sub.bias.data.clone().reshape(1, 1, 1, -1)
                        break

        return new_merger

    def preprocess_weights_impl(self):
        def _to_host(w, layout=ttnn.TILE_LAYOUT):
            if w is None:
                return None
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout)

        if self._use_layer_norm:
            self.tt_ln_weight = _to_host(self._ln_weight.unsqueeze(0))
            if self._ln_bias is not None:
                self.tt_ln_bias = _to_host(self._ln_bias.unsqueeze(0))
        else:
            if self._ln_weight is not None:
                dim = self._ln_weight.numel()
                tile = 32
                w = self._ln_weight.to(torch.bfloat16)
                w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
                self.tt_ln_weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.tt_w1 = _to_host(self._w1_weight)
        self.tt_w2 = _to_host(self._w2_weight)
        self.tt_w1_bias = _to_host(self._w1_bias)
        self.tt_w2_bias = _to_host(self._w2_bias)

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.tt_ln_weight = _to_dev(self.tt_ln_weight)
        self.tt_ln_bias = _to_dev(self.tt_ln_bias)
        self.tt_w1 = _to_dev(self.tt_w1)
        self.tt_w2 = _to_dev(self.tt_w2)
        self.tt_w1_bias = _to_dev(self.tt_w1_bias)
        self.tt_w2_bias = _to_dev(self.tt_w2_bias)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG

        if self._use_layer_norm:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.tt_ln_weight,
                bias=self.tt_ln_bias,
                epsilon=1e-6,
            )
        elif self.tt_ln_weight is not None:
            hidden_states = ttnn.rms_norm(
                hidden_states,
                weight=self.tt_ln_weight,
                epsilon=1e-6,
            )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1], -1, self.mlp_size),
        )
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w1,
            bias=self.tt_w1_bias,
            memory_config=mem,
        )
        hidden_states = ttnn.gelu(hidden_states)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w2,
            bias=self.tt_w2_bias,
            memory_config=mem,
        )

        return hidden_states


# ---------------------------------------------------------------------------
# Vision Tower (top-level)
# ---------------------------------------------------------------------------


class TTNNDotsOCRVisionTower(TTNNModule):
    """Native TTNNModule vision tower for dots.ocr.

    Full pipeline: PatchEmbed -> 42 VisionBlocks -> post-trunk RMSNorm -> PatchMerger.
    """

    def __init__(self):
        super().__init__()
        self._hf_config = None
        self.patch_embed = None
        self.blocks = []
        self.post_trunk_norm = None
        self.patch_merger = None
        self.rope = None
        self.num_layers = 42
        self.hidden_size = 1536
        self.num_heads = 12
        self.head_dim = 128
        self.spatial_merge_size = 2
        self._bypass_tensor_wrapping = True

    @classmethod
    def from_torch(cls, hf_vision_tower, hf_config=None):
        new_tower = cls()
        new_tower._fallback_torch_layer = hf_vision_tower
        new_tower._hf_config = hf_config or getattr(hf_vision_tower, "config", None)

        vc = None
        if new_tower._hf_config is not None:
            vc = getattr(new_tower._hf_config, "vision_config", new_tower._hf_config)

        if vc is not None:
            new_tower.hidden_size = getattr(vc, "hidden_size", 1536)
            new_tower.num_heads = getattr(vc, "num_attention_heads", 12)
            new_tower.head_dim = new_tower.hidden_size // new_tower.num_heads
            new_tower.num_layers = getattr(vc, "num_hidden_layers", 42)
            new_tower.spatial_merge_size = getattr(vc, "spatial_merge_size", 2)

        patch_embed_module = getattr(hf_vision_tower, "patch_embed", None)
        if patch_embed_module is not None:
            patch_size = getattr(vc, "patch_size", 14) if vc else 14
            in_channels = getattr(vc, "num_channels", 3) if vc else 3
            new_tower.patch_embed = TTNNDotsVisionPatchEmbed.from_torch(
                patch_embed_module,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=new_tower.hidden_size,
            )

        blocks_attr = getattr(hf_vision_tower, "blocks", getattr(hf_vision_tower, "layers", None))
        if blocks_attr is not None:
            new_tower.blocks = []
            for hf_block in blocks_attr:
                block = TTNNDotsVisionBlock.from_torch(
                    hf_block,
                    hidden_size=new_tower.hidden_size,
                    num_heads=new_tower.num_heads,
                )
                new_tower.blocks.append(block)
            new_tower.num_layers = len(new_tower.blocks)

        post_trunk = getattr(
            hf_vision_tower,
            "post_trunk_norm",
            getattr(hf_vision_tower, "norm", None),
        )
        if post_trunk is not None:
            new_tower.post_trunk_norm = TTNNDotsVisionRMSNorm.from_torch(post_trunk)

        merger = getattr(
            hf_vision_tower,
            "merger",
            getattr(hf_vision_tower, "patch_merger", None),
        )
        if merger is not None:
            out_hidden = new_tower.hidden_size
            new_tower.patch_merger = TTNNDotsPatchMerger.from_torch(
                merger,
                hidden_size=new_tower.hidden_size,
                out_hidden_size=out_hidden,
                spatial_merge_size=new_tower.spatial_merge_size,
            )

        return new_tower

    def preprocess_weights_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.preprocess_weights()
        for block in self.blocks:
            block.preprocess_weights()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.preprocess_weights()
        if self.patch_merger is not None:
            self.patch_merger.preprocess_weights()

    def move_weights_to_device_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.move_weights_to_device()
        for block in self.blocks:
            block.move_weights_to_device()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.move_weights_to_device()
        if self.patch_merger is not None:
            self.patch_merger.move_weights_to_device()

        self.rope = TTNNDotsVision2DRoPE(
            device=self.device,
            head_dim=self.head_dim,
            spatial_merge_size=self.spatial_merge_size,
        )

    def to_device(self, device):
        super().to_device(device)
        if self.patch_embed is not None:
            self.patch_embed.to_device(device)
        for block in self.blocks:
            block.to_device(device)
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.to_device(device)
        if self.patch_merger is not None:
            self.patch_merger.to_device(device)
        return self

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision")

        x = self.patch_embed(pixel_values, grid_thw)

        if isinstance(x, torch.Tensor):
            mem = ttnn.DRAM_MEMORY_CONFIG
            mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            x = x.unsqueeze(1) if x.dim() == 3 else x
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))

        seq_len = int(x.shape[2])

        rot_mats, cu_seqlens = self.rope.build(grid_thw, seq_len)

        for block in self.blocks:
            x = block(x, rot_mats=rot_mats, cu_seqlens=cu_seqlens)

        if self.post_trunk_norm is not None:
            x = self.post_trunk_norm(x)

        if self.patch_merger is not None:
            x = self.patch_merger(x)

        composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        result = ttnn.to_torch(x, mesh_composer=composer).to(torch.bfloat16)

        try:
            num_devices = self.device.get_num_devices()
            if num_devices > 1 and result.dim() >= 1 and result.shape[0] % num_devices == 0:
                per = result.shape[0] // num_devices
                result = result[:per]
        except Exception:
            pass

        if result.dim() == 4:
            result = result.squeeze(0).squeeze(0)

        return result
