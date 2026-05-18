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


import os

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule, TTNNLayerStack
from ttnn.operations.transformer import SDPAProgramConfig

# Tracy (perf): vision Matmul/SDPA show HiFi4; use lower fidelity for ViT matmul/SDPA only.
# Norms (RMS/LayerNorm) keep HiFi4 for stability.
# QKV/O attn matmul lowered to LoFi to match the MLP path -- weights are already
# BFP8 so the LoFi multiplication delta lands inside the existing BFP8 output
# quantization noise. Saves ~50% on the per-layer attn matmul time at the
# 60.9 / 130 TFLOPs peak HiFi2 ceiling we were hitting in tracy.
VISION_MATMUL_MATH_FIDELITY = ttnn.MathFidelity.LoFi
VISION_SDPA_MATH_FIDELITY = ttnn.MathFidelity.LoFi
# RMSNorm/LayerNorm at HiFi2: the variance reduce is a sum of squares of BF16
# inputs, and HiFi2 already gives full BF16 multiplication precision in the
# single-pass kernel -- HiFi4's extra phases are wasted work for our dtype mix.
VISION_NORM_MATH_FIDELITY = ttnn.MathFidelity.HiFi2


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


# Cache program_config objects per (grid, m, k, n) so trace capture sees the
# same Python object across calls. Recreating the config in the forward pass
# every iteration breaks ttnn trace dedup and silently drops the entire
# decode/prefill back to eager-mode dispatch (per-op kernel launches), which
# in TP2 makes decode 4x slower than the trace-replay path.
_VISION_MATMUL_PC_CACHE: dict = {}


def _vision_matmul_program_config(device, m_dim: int, k_dim: int, n_dim: int):
    """2D-mcast prefill matmul config for vision-tower DRAM-interleaved matmuls.

    Tracy reports every vision matmul as ``SLOW`` because none of the vision
    ttnn.linear calls passed an explicit program_config -- the auto-config
    falls back to ``in0_block_w=1`` and small subblocks, which is why
    QKV/MLP land at 26-29% TFLOPs and 12-13% DRAM. Setting a 2D-mcast config
    with a larger ``in0_block_w`` reuses the L1-resident weight block across
    more output tiles.

    The 2D-mcast factory hard-asserts
    ``num_blocks_y == m_tiles / per_core_M <= grid_y``, so for vision shapes
    (M=12288 -> m_tiles=384, grid_y=8) we must run with ``per_core_M=48``.
    With BFP8 output the natural ``per_core_M*per_core_N`` output CB is
    ~3.4 MB which doesn't fit in 1.5 MB L1; we use ``out_block_h`` to chunk
    the in-flight output and bound the actual L1 footprint to
    ``out_block_h * per_core_N * tile_bytes`` instead of the full per-core
    output. The kernel iterates ``per_core_M / out_block_h`` blocks per
    core (matmul_multicore_reuse_mcast_2d_program_factory.cpp:111), so the
    total work is unchanged but the L1 fits.

    Returns ``None`` for shapes that don't tile cleanly so callers fall
    back to the auto-config.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    cache_key = (grid_x, grid_y, m_dim, k_dim, n_dim)
    cached = _VISION_MATMUL_PC_CACHE.get(cache_key)
    if cached is not None or cache_key in _VISION_MATMUL_PC_CACHE:
        return cached

    if m_dim % tile != 0 or k_dim % tile != 0 or n_dim % tile != 0:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    if m_tiles % grid_y != 0:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    # ``num_blocks_x = ceil(n_tiles / per_core_n)`` in the 2D mcast factory
    # (matmul_multicore_reuse_mcast_2d_program_factory.cpp:1669), so n_tiles
    # doesn't need to divide grid_x cleanly -- the kernel pads the trailing
    # output tiles internally. This unlocks the MLP gate/up matmuls
    # (12288 x 1536 x 4224, n_tiles=132) which were falling back to the
    # auto-config because 132 % 8 != 0. With per_core_n=ceil(132/8)=17 the
    # 8x8 grid covers 136 tiles (4 padded) and we get the same in0_block_w
    # tuning as the divisible matmuls.
    per_core_m = m_tiles // grid_y
    per_core_n = (n_tiles + grid_x - 1) // grid_x

    if per_core_n > 24 or per_core_m > 64:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    # Pick the largest in0_block_w (cap at 8) that divides K_tiles cleanly.
    # Larger in0_block_w => fewer K iterations and better DRAM-to-L1
    # weight-block reuse. For vision QKV (K=48 tiles) this picks 8 (vs the
    # auto-config's 1) -> 6 K iterations instead of 48.
    in0_block_w = _largest_divisor_le(k_tiles, 8)

    # In-flight output block sized to keep total L1 (act + weight + output)
    # comfortably under 1.5 MB per core. ``out_block_h`` chunks the per-core
    # output along M; the kernel iterates ``per_core_M / out_block_h`` blocks
    # per core. Larger ``out_block_h`` -> fewer outer-block iterations.
    #
    # Tracy ablation across the 4 vision matmul shapes (TP=2):
    #   * QKV/gate/up (K=48 -> 6 inner-K iters/outer) and O proj (K=48):
    #     out_block_h=16 either ties or *regresses* (O proj 40.8% -> 35.5%
    #     FLOPs) -- the (16, 1) output-subblock shape fights the matmul
    #     kernel's tile layout when N is small.
    #   * Down (K=132 -> 22 inner-K iters/outer): out_block_h=16 wins
    #     (28.8% -> 48.5% FLOPs) because the 33% saving in outer-M
    #     iterations dominates each iteration's heavy K loop.
    # Discriminate on inner-K iters per outer-M block: bigger K loop ->
    # bigger out_block_h gains. For QKV (per_core_n=18, BFP8 out,
    # in0_block_w=8) out_block_h=12 lands at ~384 KB act + 324 KB weight +
    # 486 KB out (all double-buffered) = ~1.2 MB << 1.5 MB.
    inner_k_iters = max(1, k_tiles // max(in0_block_w, 1))
    target_out_block_h_tiles = 16 if inner_k_iters >= 12 else 12
    out_block_h = min(per_core_m, target_out_block_h_tiles)
    while out_block_h > 1 and per_core_m % out_block_h != 0:
        out_block_h -= 1
    out_block_w = per_core_n

    # Pick (out_subblock_h, out_subblock_w) to fill the DST register budget
    # (8 tiles when fp32_dest_acc_en=False, which is the vision matmul setup).
    # Prefer a wide subblock (keeps the M loop tight); when out_block_w is
    # prime (e.g. 17 for the MLP gate/up at n_tiles=132/8) the wide axis
    # collapses to 1, so fall back to growing out_subblock_h to keep DST
    # full. Empty 1x1 subblocks waste 7/8 of the DST register file and
    # compound across the 384-tile M dimension of the vision matmuls.
    dst_tiles_budget = 8
    out_subblock_h = 1
    out_subblock_w = 1
    best_area = 0
    for h in range(min(out_block_h, dst_tiles_budget), 0, -1):
        if out_block_h % h != 0:
            continue
        for w in range(min(out_block_w, dst_tiles_budget // h), 0, -1):
            if out_block_w % w != 0:
                continue
            area = h * w
            if area > best_area:
                best_area = area
                out_subblock_h = h
                out_subblock_w = w
                break

    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    _VISION_MATMUL_PC_CACHE[cache_key] = pc
    return pc


def _vision_matmul_compute_config(device, *, math_fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    """Compute config for vision linear/matmul ops.

    ``packer_l1_acc=True`` keeps partial output accumulators in L1 across the
    K-loop iterations rather than spilling to DRAM each pass, which directly
    improves throughput on long-K matmuls (K=1536, K=4224 in the dots vision
    MLP). ``math_approx_mode=True`` enables the fast polynomial path for any
    fused transcendental activation (e.g. SILU on the gate matmul).
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _vision_sdpa_compute_config(device, *, math_fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    """Compute config for vision SDPA and norm ops.

    ``packer_l1_acc=True`` keeps the chunked-SDPA partial accumulators in L1
    instead of round-tripping through DRAM, which directly speeds up the
    inner softmax-reduce loop. ``math_approx_mode=True`` uses the fast
    polynomial path for transcendentals (matches the decode-mode SDPA).
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


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
        self._padded_cache_key = None
        self._padded_cache_rot_mats = None
        self._padded_cache_cu_seqlens = None

    def build(
        self,
        grid_thw: torch.Tensor,
        seq_len: int,
    ) -> tuple[tuple[ttnn.Tensor, ttnn.Tensor], list[int]]:
        """Build 2D RoPE cos/sin tables and cu_seqlens for vision attention.

        Returns cu_seqlens as a Python list to avoid repeated device-to-host syncs.
        """
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

        inv_freq = self._inv_freq  # shape: [rotary_dim // 2]

        hpos_segments = []
        wpos_segments = []
        cu = [0]
        running = 0

        for t, h, w in g.tolist():
            t, h, w = int(t), int(h), int(w)
            if h % sms != 0 or w % sms != 0:
                raise ValueError(f"grid {h}x{w} not divisible by spatial_merge_size={sms}")

            h_ids = torch.arange(h, dtype=torch.float32)
            w_ids = torch.arange(w, dtype=torch.float32)

            # Build 2D grids: h_grid[i,j] = i, w_grid[i,j] = j
            h_grid = h_ids.unsqueeze(1).expand(h, w)  # [h, w]
            w_grid = w_ids.unsqueeze(0).expand(h, w)  # [h, w]

            # Spatial merge reshuffling
            h_grid = h_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
            w_grid = w_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)

            # Repeat for temporal dimension
            if t > 1:
                h_grid = h_grid.repeat(t)
                w_grid = w_grid.repeat(t)

            hpos_segments.append(h_grid)
            wpos_segments.append(w_grid)
            running += t * h * w
            cu.append(running)

        hpos_all = torch.cat(hpos_segments) if len(hpos_segments) > 1 else hpos_segments[0]
        wpos_all = torch.cat(wpos_segments) if len(wpos_segments) > 1 else wpos_segments[0]

        # Compute frequencies on CPU: [S] x [rotary_dim//2] -> [S, rotary_dim//2]
        freqs_h = hpos_all.unsqueeze(1) * inv_freq.unsqueeze(0)
        freqs_w = wpos_all.unsqueeze(1) * inv_freq.unsqueeze(0)

        cos_h = torch.cos(freqs_h)
        sin_h = torch.sin(freqs_h)
        cos_w = torch.cos(freqs_w)
        sin_w = torch.sin(freqs_w)

        # Concat h and w: [S, rotary_dim]
        cos_half = torch.cat([cos_h, cos_w], dim=-1)
        sin_half = torch.cat([sin_h, sin_w], dim=-1)

        # Repeat to full head_dim: [S, head_dim]
        cos_full = torch.cat([cos_half, cos_half], dim=-1)
        sin_full = torch.cat([sin_half, sin_half], dim=-1)

        # Reshape to [1, 1, S, head_dim]. cos/sin stay in the native HF
        # half-half head_dim layout (each half repeats), which is exactly
        # the format ``ttnn.experimental.rotary_embedding`` (non-llama)
        # expects -- no meta-style interleave conversion needed.
        cos_full = cos_full.unsqueeze(0).unsqueeze(0)
        sin_full = sin_full.unsqueeze(0).unsqueeze(0)

        cos_full = cos_full.to(torch.bfloat16)
        sin_full = sin_full.to(torch.bfloat16)

        cos_tt = ttnn.from_torch(
            cos_full,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        sin_tt = ttnn.from_torch(
            sin_full,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        rot_mats = (cos_tt, sin_tt)

        return rot_mats, cu

    def build_padded(
        self,
        grid_thw: torch.Tensor,
        actual_seq_len: int,
        bucket_size: int,
    ) -> tuple[tuple, list[int]]:
        """Build 2D RoPE cos/sin padded to bucket_size for trace compatibility."""
        g = grid_thw.detach().cpu() if getattr(grid_thw, "is_cuda", False) else grid_thw
        cache_key = (tuple(int(x) for x in g.reshape(-1).tolist()), int(actual_seq_len), int(bucket_size))
        if cache_key == self._padded_cache_key and self._padded_cache_rot_mats is not None:
            return self._padded_cache_rot_mats, self._padded_cache_cu_seqlens

        rot_mats, cu_seqlens = self.build(grid_thw, actual_seq_len)
        cos_tt, sin_tt = rot_mats

        if actual_seq_len < bucket_size:
            pad_len = bucket_size - actual_seq_len
            cos_tt = ttnn.pad(cos_tt, padding=((0, 0), (0, 0), (0, pad_len), (0, 0)), value=0.0)
            sin_tt = ttnn.pad(sin_tt, padding=((0, 0), (0, 0), (0, pad_len), (0, 0)), value=0.0)

        self._padded_cache_key = cache_key
        self._padded_cache_rot_mats = (cos_tt, sin_tt)
        self._padded_cache_cu_seqlens = cu_seqlens
        return self._padded_cache_rot_mats, cu_seqlens


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

        self.compute_kernel_config = _vision_sdpa_compute_config(self.device, math_fidelity=VISION_NORM_MATH_FIDELITY)

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

    def preprocess_weights_impl(self):
        """Linear weights in low precision (PyTorch Linear [out,in]; preprocessing applies TT layout)."""

        def pw(w):
            if w is None:
                return None
            return preprocess_linear_weight(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)

        def pb(b):
            if b is None:
                return None
            return preprocess_linear_bias(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        # Unfused gate/up projections (two linears). Vision MLP uses DRAM linears without
        # decoder-style CCL, so fusion only saved one matmul launch while forcing two
        # SliceDeviceOperation per block on the fused [gate|up] output; unfused removes those slices.
        self._intermediate_size = None
        self.tt_fused_gate_up_weight = None
        self.tt_fused_gate_up_bias = None
        self.tt_fc1_weight = pw(self._fc1_weight)
        self.tt_fc1_bias = pb(self._fc1_bias)
        self.tt_fc3_weight = pw(self._fc3_weight)
        self.tt_fc3_bias = pb(self._fc3_bias)

        self.tt_fc2_weight = pw(self._fc2_weight)
        self.tt_fc2_bias = pb(self._fc2_bias)

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = _vision_matmul_compute_config(self.device, math_fidelity=ttnn.MathFidelity.LoFi)

        self.tt_fused_gate_up_weight = _to_dev(getattr(self, "tt_fused_gate_up_weight", None))
        self.tt_fused_gate_up_bias = _to_dev(getattr(self, "tt_fused_gate_up_bias", None))
        self.tt_fc1_weight = _to_dev(getattr(self, "tt_fc1_weight", None))
        self.tt_fc1_bias = _to_dev(getattr(self, "tt_fc1_bias", None))
        self.tt_fc2_weight = _to_dev(self.tt_fc2_weight)
        self.tt_fc2_bias = _to_dev(self.tt_fc2_bias)
        self.tt_fc3_weight = _to_dev(getattr(self, "tt_fc3_weight", None))
        self.tt_fc3_bias = _to_dev(getattr(self, "tt_fc3_bias", None))

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG

        # Explicit 2D-mcast program_config: tracy showed gate/up running at
        # 27.7% LoFi TFLOPs / 13% DRAM with the auto-config (in0_block_w=1).
        # Setting a tuned in0_block_w + out_subblock_w increases L1 weight
        # reuse and matches the text-decoder linears' ``SLOW``->faster path.
        m_dim = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) * int(hidden_states.shape[2])
        k_dim = int(self.tt_fc1_weight.shape[-2])
        n_dim = int(self.tt_fc1_weight.shape[-1])
        gate_up_pc = _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim)
        # Gate/up outputs are BFP8: halves the matmul writeback (38 MB BF16
        # -> 21 MB BFP8) and the matching read in the SILU multiply, with
        # no quality impact since the silu+mul output is already BFP8 in
        # this path. BFP4 weight x BF16 activation -> BFP8 output is a
        # supported matmul dtype combo on Wormhole.
        gate = ttnn.linear(
            hidden_states,
            self.tt_fc1_weight,
            bias=self.tt_fc1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        up = ttnn.linear(
            hidden_states,
            self.tt_fc3_weight,
            bias=self.tt_fc3_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        # SILU dominates this op (the elementwise mul is cheap; the per-tile
        # exp+sigmoid is the bottleneck). ``fast_and_approximate_mode=True``
        # routes the fused SILU through the polynomial exp/sigmoid path,
        # cutting the ~2.2 ms BinaryNg time per vision layer. Output goes to
        # BFP8 so the down-projection matmul reads half the bandwidth (BFP4
        # weight x BFP8 activation instead of BFP4 x BF16).
        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # ``dtype=bfloat8_b`` halves the down-projection writeback (38 MB BF16
        # -> 21 MB BFP8 per vision layer) and the matching read in the
        # following residual add. The downstream RMSNorm/attention path
        # already operates on BFP8 cleanly, so this stays inside the existing
        # quantization envelope.
        down_m = int(gate_up_mul.shape[0]) * int(gate_up_mul.shape[1]) * int(gate_up_mul.shape[2])
        down_k = int(self.tt_fc2_weight.shape[-2])
        down_n = int(self.tt_fc2_weight.shape[-1])
        down_pc = _vision_matmul_program_config(self.device, down_m, down_k, down_n)
        output = ttnn.linear(
            gate_up_mul,
            self.tt_fc2_weight,
            bias=self.tt_fc2_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=down_pc,
        )
        ttnn.deallocate(gate_up_mul)

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
        self.vision_matmul_compute_kernel_config = None
        self.vision_norm_compute_kernel_config = None
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
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
        if self._proj_bias is not None:
            self.tt_proj_bias = ttnn.from_torch(
                self._proj_bias.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
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

        self.vision_matmul_compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )
        self.vision_norm_compute_kernel_config = _vision_sdpa_compute_config(
            self.device, math_fidelity=VISION_NORM_MATH_FIDELITY
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
                compute_kernel_config=self.vision_matmul_compute_kernel_config,
            )

            if self.tt_norm_weight is not None:
                out = ttnn.rms_norm(
                    out,
                    weight=self.tt_norm_weight,
                    epsilon=1e-5,
                    compute_kernel_config=self.vision_norm_compute_kernel_config,
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
            compute_kernel_config=self.vision_matmul_compute_kernel_config,
        )

        if self.tt_norm_weight is not None:
            out = ttnn.rms_norm(
                out,
                weight=self.tt_norm_weight,
                epsilon=1e-5,
                compute_kernel_config=self.vision_norm_compute_kernel_config,
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

    def preprocess_weights_impl(self):
        # QKV weights/bias are kept in the native HF "half-half" head_dim
        # layout. Combined with cos/sin built in the same half-half layout
        # (see TTNNDotsVision2DRoPE.build), this lets us use the cheaper
        # ``ttnn.experimental.rotary_embedding`` (non-llama) kernel that
        # preserves input dtype -- so Q/K can stay BFP8 the whole way from
        # the QKV matmul into SDPA, eliminating the 4 typecasts per layer
        # the llama kernel forced (~1.3 ms x 42 layers in vision prefill).
        self.tt_qkv_weight = preprocess_linear_weight(self._qkv_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        if self._qkv_bias is not None:
            self.tt_qkv_bias = preprocess_linear_bias(self._qkv_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_o_proj_weight = preprocess_linear_weight(
            self._o_proj_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        if self._o_proj_bias is not None:
            self.tt_o_proj_bias = preprocess_linear_bias(
                self._o_proj_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )

        self.tt_qkv_weight = _to_dev(self.tt_qkv_weight)
        self.tt_qkv_bias = _to_dev(self.tt_qkv_bias)
        self.tt_o_proj_weight = _to_dev(self.tt_o_proj_weight)
        self.tt_o_proj_bias = _to_dev(self.tt_o_proj_bias)

        self.sdpa_compute_kernel_config = _vision_sdpa_compute_config(
            self.device, math_fidelity=VISION_SDPA_MATH_FIDELITY
        )

    def _concat_heads(self, ctx: ttnn.Tensor) -> ttnn.Tensor:
        """Gather head slices back into a single sequence-major tensor."""
        if ctx.memory_config().buffer_type == ttnn.BufferType.L1:
            return ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ctx_l1 = ttnn.typecast(ctx, ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        ctx_gathered = ttnn.experimental.nlp_concat_heads(ctx_l1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ctx_l1)
        return ttnn.typecast(ctx_gathered, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _get_sdpa_program_config(self, seq_len: int):
        """Chunked SDPA program config for the vision tower.

        ``exp_approx_mode=True`` uses the fast polynomial exp in the softmax
        kernel — vision SDPA already runs at LoFi math fidelity, so the small
        precision delta from the approximate exp is in the noise.

        Chunk sizes: the per-core attention-scores circular buffer is
        ``q_chunk × k_chunk × sizeof(fp32_accumulator)``. With Wormhole's
        ~1.5 MB usable L1, this hard-caps the product at ~256K elements
        (e.g. 256 × 512 = 128K, leaves headroom for Q/K/V chunk buffers and
        the partial-output accumulator). Going beyond (e.g. 512 × 1024 =
        512K) overflows L1 with a "Statically allocated CBs grow to ... B"
        error.

        Tracy ablation (TP=2, S=12288):

            q=256, k=512 -> 12,246 us / SDPA call (baseline -- kept)
            q=512, k=256 -> 13,464 us / SDPA call (+10%, regressed)

        The naive analysis says swapping should halve DRAM K/V re-reads
        (24 outer Q chunks vs 48), but the kernel doesn't actually win:
        the larger Q-chunk's softmax + DEST-register packing for the
        16x8 QK^T tile shape eats the saved bandwidth and then some.
        Stay at q=256/k=512.
        """
        grid = self.device.compute_with_storage_grid_size()
        grid_size = ttnn.CoreCoord(grid.x, grid.y)

        if seq_len <= 256:
            q_chunk = k_chunk = max(32, ((seq_len + 31) // 32) * 32)
        elif seq_len <= 1024:
            q_chunk = k_chunk = 128
        else:
            q_chunk = 256
            k_chunk = 512

        return SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=True,
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

        # Output the fused QKV in BFP8 directly. At S=12288 this halves bandwidth on:
        #   1. the qkv matmul writeback ([1,1,S,4608]: 113 MB BF16 -> 56 MB BFP8)
        #   2. nlp_create_qkv_heads, which becomes "BFP8 => BFP8" (~221 MB IO -> ~110 MB)
        # Q/K/V all stay BFP8 from the QKV matmul straight into SDPA -- the
        # non-llama ``ttnn.experimental.rotary_embedding`` preserves dtype,
        # so we don't need any typecasts around the rotary like the llama
        # kernel forced before.
        qkv_m = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) * s
        qkv_k = int(self.tt_qkv_weight.shape[-2])
        qkv_n = int(self.tt_qkv_weight.shape[-1])
        qkv_pc = _vision_matmul_program_config(self.device, qkv_m, qkv_k, qkv_n)
        qkv = ttnn.linear(
            hidden_states,
            self.tt_qkv_weight,
            bias=self.tt_qkv_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=qkv_pc,
        )

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        if rot_mats is not None and len(rot_mats) == 2:
            cos, sin = rot_mats

            # ``ttnn.experimental.rotary_embedding`` (non-llama) preserves
            # input dtype, so Q/K stay BFP8 the whole way: no typecasts
            # around the rotary, and SDPA reads BFP8 Q/K/V directly.
            q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # V is BFP8 from the QKV matmul; cast to BFP4 to halve the V DRAM
        # stream in the chunked SDPA kernel. Per-core SDPA reads V tiles
        # ``q_per_core`` times during prefill (10 outer-Q iterations at
        # q_chunk=256, S=12288), so V dominates DRAM bandwidth in non-causal
        # attention. BFP4 V costs one typecast per layer (~0.1 ms reading
        # 21 MB BFP8 + writing 10.5 MB BFP4) and saves ~2-3 ms per layer
        # of SDPA bandwidth -- accuracy stays good because V only feeds
        # the post-softmax weighted sum (no further matmul accumulation).
        # Q/K stay BFP8: K participates in the full QK^T matmul where
        # BFP4 would lose precision in the early-attention scores.
        # SDPA validates BFP4 V in sdpa_device_operation.cpp:40 ("Data
        # type ... must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B"), and the
        # output dtype matches Q's dtype (BFP8) regardless of V dtype.
        v = ttnn.typecast(v, dtype=ttnn.bfloat4_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SDPA still requires interleaved Q/K/V (sdpa_device_operation.cpp:44 forbids
        # sharded inputs) and at S=12288 the BFP8 Q+K (~36 MB) + BFP4 V (~10 MB)
        # plus SDPA's static circular buffers exceed the per-core L1 budget --
        # keep these tensors on DRAM.

        if cu_seqlens is None:
            program_config = self._get_sdpa_program_config(s)
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ctx = self._concat_heads(ctx)
            o_m = int(ctx.shape[0]) * int(ctx.shape[1]) * int(ctx.shape[2])
            o_k = int(self.tt_o_proj_weight.shape[-2])
            o_n = int(self.tt_o_proj_weight.shape[-1])
            o_pc = _vision_matmul_program_config(self.device, o_m, o_k, o_n)
            return ttnn.linear(
                ctx,
                self.tt_o_proj_weight,
                bias=self.tt_o_proj_bias,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel_config,
                program_config=o_pc,
            )

        cu_host = self._cu_seqlens_to_list(cu_seqlens, s)

        if len(cu_host) == 2:
            # Single segment — skip slicing and concat
            program_config = self._get_sdpa_program_config(s)
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
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
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ctx_segments.append(ctx_seg)

            ctx = ttnn.concat(ctx_segments, dim=2) if len(ctx_segments) > 1 else ctx_segments[0]

        ctx = self._concat_heads(ctx)
        o_m = int(ctx.shape[0]) * int(ctx.shape[1]) * int(ctx.shape[2])
        o_k = int(self.tt_o_proj_weight.shape[-2])
        o_n = int(self.tt_o_proj_weight.shape[-1])
        o_pc = _vision_matmul_program_config(self.device, o_m, o_k, o_n)
        return ttnn.linear(
            ctx,
            self.tt_o_proj_weight,
            bias=self.tt_o_proj_bias,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=o_pc,
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

        self.tt_w1 = (
            ttnn.from_torch(self._w1_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w1_weight is not None
            else None
        )
        self.tt_w2 = (
            ttnn.from_torch(self._w2_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w2_weight is not None
            else None
        )
        self.tt_w1_bias = (
            ttnn.from_torch(self._w1_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w1_bias is not None
            else None
        )
        self.tt_w2_bias = (
            ttnn.from_torch(self._w2_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w2_bias is not None
            else None
        )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.tt_ln_weight = _to_dev(self.tt_ln_weight)
        self.tt_ln_bias = _to_dev(self.tt_ln_bias)
        self.tt_w1 = _to_dev(self.tt_w1)
        self.tt_w1_bias = _to_dev(self.tt_w1_bias)

        # Col-shard w2 across devices so the patch merger natively produces
        # col-sharded output matching text embeddings.  Weight shape is
        # [intermediate, H] (already transposed); sharding dim=-1 gives each
        # device [intermediate, H/num_devices].
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        if num_devices > 1:
            col_shard_mapper = ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, -1),
                mesh_shape=list(self.device.shape),
            )
            # Re-create w2 on device with col-shard mapper
            self.tt_w2 = ttnn.from_torch(
                self._w2_weight.to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mem,
                mesh_mapper=col_shard_mapper,
            )
            if self._w2_bias is not None:
                self.tt_w2_bias = ttnn.from_torch(
                    self._w2_bias.to(torch.bfloat16),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=mem,
                    mesh_mapper=col_shard_mapper,
                )
            else:
                self.tt_w2_bias = None
        else:
            self.tt_w2 = _to_dev(self.tt_w2)
            self.tt_w2_bias = _to_dev(self.tt_w2_bias)

        self.compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )

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

        compute_kc = getattr(self, "compute_kernel_config", None)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w1,
            bias=self.tt_w1_bias,
            activation="gelu",
            memory_config=mem,
            compute_kernel_config=compute_kc,
        )
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w2,
            bias=self.tt_w2_bias,
            memory_config=mem,
            compute_kernel_config=compute_kc,
        )

        return hidden_states


class TTNNDotsVisionBlockStack(TTNNLayerStack):
    """Trace-enabled stack of vision blocks + post-norm + merger.

    Captures the entire vision encoder core as a single trace.
    Different sequence length buckets get different traces automatically
    via the cache key mechanism in TracedRun.
    """

    SEQ_LEN_BUCKETS = [
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        8192,
        12288,
        16384,
        20480,
        24576,
    ]

    def __init__(self, blocks, *, post_trunk_norm=None, patch_merger=None):
        super().__init__(blocks)
        self.post_trunk_norm = post_trunk_norm
        self.patch_merger = patch_merger
        self._bypass_tensor_wrapping = True

    @classmethod
    def nearest_bucket(cls, seq_len: int) -> int:
        for bucket in cls.SEQ_LEN_BUCKETS:
            if seq_len <= bucket:
                return bucket
        return -1

    def preprocess_weights_impl(self):
        super().preprocess_weights_impl()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.preprocess_weights()
        if self.patch_merger is not None:
            self.patch_merger.preprocess_weights()

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.move_weights_to_device()
        if self.patch_merger is not None:
            self.patch_merger.move_weights_to_device()

    def to_device(self, device):
        super().to_device(device)
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.to_device(device)
        if self.patch_merger is not None:
            self.patch_merger.to_device(device)
        return self

    def forward(self, hidden_states, **kwargs):
        rot_mats = kwargs.get("rot_mats")
        cu_seqlens = kwargs.get("cu_seqlens")

        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states,
                rot_mats=rot_mats,
                cu_seqlens=cu_seqlens,
            )

        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm.forward(hidden_states)

        if self.patch_merger is not None:
            hidden_states = self.patch_merger.forward(hidden_states)

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
        self.block_stack = None
        self._trace_enabled = True
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

        new_tower.block_stack = TTNNDotsVisionBlockStack(
            new_tower.blocks,
            post_trunk_norm=new_tower.post_trunk_norm,
            patch_merger=new_tower.patch_merger,
        )

        return new_tower

    def preprocess_weights_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.preprocess_weights()
        if self.block_stack is not None:
            self.block_stack.preprocess_weights()
        else:
            for block in self.blocks:
                block.preprocess_weights()
            if self.post_trunk_norm is not None:
                self.post_trunk_norm.preprocess_weights()
            if self.patch_merger is not None:
                self.patch_merger.preprocess_weights()

    def move_weights_to_device_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.move_weights_to_device()
        if self.block_stack is not None:
            self.block_stack.move_weights_to_device()
        else:
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
        if self.block_stack is not None:
            self.block_stack.to_device(device)
        else:
            for block in self.blocks:
                block.to_device(device)
            if self.post_trunk_norm is not None:
                self.post_trunk_norm.to_device(device)
            if self.patch_merger is not None:
                self.patch_merger.to_device(device)
        return self

    def merged_vision_sequence_length(self, grid_thw: torch.Tensor, pixel_values: torch.Tensor | None = None) -> int:
        """Merged vision token count on dim=2 after ``patch_merger`` (matches ``forward``).

        Used for scatter gather sizing without running the vision trunk.
        """
        if grid_thw is None:
            raise ValueError("grid_thw is required")
        g = grid_thw.detach().cpu() if hasattr(grid_thw, "is_cuda") and grid_thw.is_cuda else grid_thw
        if g.dim() == 1:
            g = g.unsqueeze(0)
        temporal = int(g[0, 0].item())
        height_patches = int(g[0, 1].item())
        width_patches = int(g[0, 2].item())
        num_patches = temporal * height_patches * width_patches
        if pixel_values is not None and self.patch_embed is not None and pixel_values.dim() == 4:
            patch_size = int(getattr(self.patch_embed, "patch_size", 14))
            _b, _c, h, w = pixel_values.shape
            alt = temporal * (h // patch_size) * (w // patch_size)
            if alt > 0:
                num_patches = min(num_patches, alt)
        # dim != 4 (e.g. HF ``pixel_values`` [num_patches, C*patch*patch]): use grid-derived count only.
        if self.patch_merger is not None:
            return num_patches // (self.spatial_merge_size**2)
        return num_patches

    def forward_post_patch_embed(self, x: ttnn.Tensor, grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Vision blocks + post-trunk norm + patch merger (``patch_embed`` already applied)."""
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision")
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
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

        actual_seq_len = int(x.shape[2])

        bucket = (
            TTNNDotsVisionBlockStack.nearest_bucket(actual_seq_len)
            if self._trace_enabled and self.block_stack is not None
            else -1
        )
        if os.environ.get("DOTS_OCR_PROFILE_SYNC", "").lower() in {"1", "true", "yes", "on"}:
            print(f"[DOTS_OCR_PROFILE_SYNC] vision.actual_seq_len={actual_seq_len} bucket={bucket}")

        if bucket == -1:
            rot_mats, cu_seqlens = self.rope.build(grid_thw, actual_seq_len)

            for block in self.blocks:
                x = block(x, rot_mats=rot_mats, cu_seqlens=cu_seqlens)

            if self.post_trunk_norm is not None:
                x = self.post_trunk_norm(x)

            if self.patch_merger is not None:
                x = self.patch_merger(x)
        else:
            if actual_seq_len < bucket:
                pad_len = bucket - actual_seq_len
                x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_len), (0, 0)), value=0.0)

            rot_mats, _ = self.rope.build_padded(grid_thw, actual_seq_len, bucket)

            x = self.block_stack(x, rot_mats=rot_mats, cu_seqlens=None)

            merged_seq_len = actual_seq_len // (self.spatial_merge_size**2)
            if int(x.shape[2]) > merged_seq_len:
                x = ttnn.slice(x, (0, 0, 0, 0), (int(x.shape[0]), int(x.shape[1]), merged_seq_len, int(x.shape[3])))

        return x

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run the full vision pipeline and return the result as a ttnn.Tensor on device.

        Returns:
            ttnn.Tensor: [1, 1, N_vision, H/num_devices] in TILE_LAYOUT,
                col-sharded across devices (on multi-device), or
                [1, 1, N_vision, H] replicated (on single device).
        """
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision")

        x = self.patch_embed(pixel_values, grid_thw)
        return self.forward_post_patch_embed(x, grid_thw)
