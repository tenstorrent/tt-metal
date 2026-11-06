# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision tower (HF ``Glm5NextVisionModel``) on one chip.

Pipeline (HF names in parentheses), all per image token unless noted:
    patches [N, 3*2*14*14] -> patch_embed (Conv3d, kernel == stride, i.e. a matmul over the already
    unfolded patches) -> 24 x [RMSNorm -> qkv (+bias) -> q/k RMSNorm per head -> 2D axial RoPE ->
    non-causal SDPA within each image -> proj (+bias) -> residual -> RMSNorm -> clamped SwiGLU
    (gate/up/down with bias, limit 10) -> residual] -> post_layernorm (RMSNorm) -> downsample
    (Conv2d 2x2 stride 2 over each 2x2 merge block == matmul over 4 consecutive rows) -> merger
    (proj -> LayerNorm -> GELU -> clamped SwiGLU 10240 -> down) -> [N/4, 4096].

Token order: the HF image processor emits patches block-major over 2x2 merge blocks, so a merge group
is 4 consecutive rows and the RoPE position ids follow the same order (``vision_position_ids``).

Precision: the tower is numerically touchy (24 pre-norm blocks whose residual grows to |x| ~ 200
with std ~3; on smooth images the fp32 HF model itself is chaotic under 1e-3 pixel noise, and even
rounding the *pixels* to bf16 costs 1e-3 of PCC on textured images). Hence: pixel patches and all
activations in fp32 (``act_dtype``; matmul outputs, norms, rope and the SwiGLU products), only the
flash SDPA runs on bf16 q/k/v (the kernel accepts nothing else; a 1024-key chunk makes a 448x448
image a single softmax pass). Weights are bf16 by default (1.13 GB, the design's budget) with
bfp8_b optional (-0.002..0.004 PCC). ``act_dtype=bfloat16`` is the all-bf16 variant (same accuracy
as HF run in bf16). Weights are uploaded once; no torch inside ``forward``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.reference.checkpoint import (
    FP8_BLOCK,
    default_snapshot_dir,
    dequantize_weight_tensor,
    load_tensors,
    shard_headers,
)

VISUAL_PREFIX = "model.visual."
TILE = 32


# ---------------------------------------------------------------------------------------------
# config + weights
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GLM53VisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5  # torch.nn.LayerNorm default (merger.post_projection_norm)
    swiglu_limit: float = 10.0
    rope_theta: float = 10000.0
    image_token_id: int = 154854
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads  # 64

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size  # 1176

    @property
    def patch_dim_padded(self) -> int:
        return -(-self.patch_dim // TILE) * TILE  # 1184

    @property
    def merge_group(self) -> int:
        return self.spatial_merge_size**2  # 4 patches -> 1 token

    @property
    def merged_dim(self) -> int:
        return self.hidden_size * self.merge_group  # 4096

    @classmethod
    def from_hf_config_json(cls, path: str | Path) -> "GLM53VisionConfig":
        cfg = json.loads(Path(path).read_text())
        v = cfg["vision_config"]
        rope = v.get("rope_parameters") or {}
        return cls(
            depth=v["depth"],
            hidden_size=v["hidden_size"],
            num_heads=v["num_heads"],
            intermediate_size=v["intermediate_size"],
            out_hidden_size=v["out_hidden_size"],
            projection_intermediate_size=v["projection_intermediate_size"],
            patch_size=v["patch_size"],
            temporal_patch_size=v["temporal_patch_size"],
            in_channels=v["in_channels"],
            spatial_merge_size=v["spatial_merge_size"],
            rms_norm_eps=v["rms_norm_eps"],
            swiglu_limit=v["swiglu_limit"],
            rope_theta=rope.get("rope_theta", 10000.0),
            image_token_id=cfg["image_token_id"],
            image_start_token_id=cfg["image_start_token_id"],
            image_end_token_id=cfg["image_end_token_id"],
        )

    @classmethod
    def from_snapshot(cls, snapshot: str | Path | None = None) -> "GLM53VisionConfig":
        snapshot = default_snapshot_dir() if snapshot is None else Path(snapshot)
        return cls.from_hf_config_json(Path(snapshot) / "config.json")

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0 and self.head_dim % TILE == 0, "head_dim must be tile aligned"
        assert self.spatial_merge_size == 2, "downsample is a 2x2 conv with stride 2"


def load_vision_weights(snapshot: Path | None = None, headers=None) -> dict[str, torch.Tensor]:
    """Raw ``model.visual.*`` checkpoint tensors (prefix stripped); fp8 block-scaled groups, if any, are
    dequantized to bf16 with the same helper as the text layers (the shipped checkpoint keeps the whole
    tower in bf16: 347 tensors, 0.564 B parameters)."""
    snapshot = default_snapshot_dir() if snapshot is None else Path(snapshot)
    headers = headers or shard_headers(snapshot)
    keys = [k for k in headers if k.startswith(VISUAL_PREFIX)]
    raw = load_tensors(snapshot, keys, headers)
    out: dict[str, torch.Tensor] = {}
    for k, t in raw.items():
        name = k[len(VISUAL_PREFIX) :]
        if name.endswith(".weight_scale_inv"):
            continue
        if t.dtype == torch.float8_e4m3fn:
            t = dequantize_weight_tensor(t, raw[k.replace(".weight", ".weight_scale_inv")], FP8_BLOCK)
        out[name] = t
    return out


# ---------------------------------------------------------------------------------------------
# host-side geometry (position ids, rope tables, windows); torch at the boundary only
# ---------------------------------------------------------------------------------------------
def vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """``[N, 2]`` (h, w) patch positions in the processor's block-major merge order (HF
    ``get_vision_position_ids``)."""
    out = []
    m = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_rope_cos_sin(grid_thw: torch.Tensor, cfg: GLM53VisionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` ``[N, head_dim]`` fp32 in the HF layout (``Glm5NextVisionRotaryEmbedding``):
    16 inverse frequencies over ``head_dim // 2``, ``freq_hw = cat(h * f, w * f)`` (32) and the table
    ``cat(freq_hw, freq_hw)`` (64) for the split-half rotation ``x * cos + rotate_half(x) * sin``."""
    pos = vision_position_ids(grid_thw, cfg.spatial_merge_size).float()  # [N, 2]
    spatial = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))
    freq_hw = (pos.unsqueeze(-1) * inv_freq).flatten(1)  # [N, 32] = [h freqs | w freqs]
    emb = torch.cat([freq_hw, freq_hw], dim=-1)  # [N, 64]
    return emb.cos(), emb.sin()


def rotate_half_matrix(head_dim: int) -> torch.Tensor:
    """``x @ R == cat(-x[half:], x[:half])`` (HF ``rotate_half`` as a matmul; exact in fp32)."""
    half = head_dim // 2
    R = torch.zeros(head_dim, head_dim)
    R[half:, :half] = -torch.eye(half)
    R[:half, half:] = torch.eye(half)
    return R


def vision_seq_lens(grid_thw: torch.Tensor) -> list[int]:
    """Attention window lengths: one per image, one per frame for videos (HF ``cu_seqlens``)."""
    out = []
    for t, h, w in grid_thw.tolist():
        out.extend([h * w] * t)
    return out


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _row(device, t: torch.Tensor, dtype):
    """``[D]`` -> ``[1, 1, 1, D]`` (norm weights broadcast over rows)."""
    return _upload(device, t.float().reshape(1, 1, 1, -1), dtype)


def _bias(device, t: torch.Tensor, dtype):
    """``[N]`` -> ``[1, N]`` for ``ttnn.linear(bias=...)``."""
    return _upload(device, t.float().reshape(1, -1), dtype)


def _cast(x: ttnn.Tensor, dtype) -> ttnn.Tensor:
    """typecast (consuming ``x``) unless already ``dtype``."""
    if x.dtype == dtype:
        return x
    y = ttnn.typecast(x, dtype)
    ttnn.deallocate(x)
    return y


def _fold_rows(x: ttnn.Tensor, width: int) -> ttnn.Tensor:
    """``[1, 1, S, D]`` -> ``[1, 1, S * D / width, width]`` by folding consecutive rows (row-major view
    between two layout conversions; ``reshape`` on a tiled tensor leaves a per-shape residue)."""
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    view = ttnn.reshape(rm, (1, 1, -1, width))
    out = ttnn.to_layout(view, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    if view.is_allocated():
        ttnn.deallocate(view)
    return out


# ---------------------------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------------------------
class TtClampedSwiGLUBias(LightweightModule):
    """``down(silu(clamp(gate, max=L)) * clamp(up, -L, L))`` with optional biases (vision MLP, merger)."""

    def __init__(
        self, device, gate_w, up_w, down_w, *, gate_b=None, up_b=None, down_b=None, limit, weight_dtype, act_dtype
    ):
        self.limit = float(limit)
        self.act_dtype = act_dtype
        self.intermediate = gate_w.shape[0]
        self.w13 = _upload(device, torch.cat([gate_w.t(), up_w.t()], dim=-1).float(), weight_dtype)
        self.b13 = _bias(device, torch.cat([gate_b, up_b]), act_dtype) if gate_b is not None else None
        self.w2 = _upload(device, down_w.t().float(), weight_dtype)
        self.b2 = _bias(device, down_b, act_dtype) if down_b is not None else None
        self.compute = _compute(device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        S = x.shape[-2]
        gu = ttnn.linear(x, self.w13, bias=self.b13, dtype=self.act_dtype, compute_kernel_config=self.compute)
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, S, self.intermediate])
        up = ttnn.slice(gu, [0, 0, 0, self.intermediate], [1, 1, S, 2 * self.intermediate])
        ttnn.deallocate(gu)
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        act = ttnn.silu(gate)
        ttnn.deallocate(gate)
        h = ttnn.multiply(act, up, output_tensor=act)
        ttnn.deallocate(up)
        y = ttnn.linear(h, self.w2, bias=self.b2, dtype=self.act_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(h)
        return y


class TtVisionAttention(LightweightModule):
    """HF ``Glm5NextVisionAttention``: qkv (+bias) -> per-head q/k RMSNorm -> 2D rope -> SDPA within each
    window (bf16) -> proj (+bias). Everything but the SDPA runs in ``act_dtype``."""

    def __init__(self, device, cfg: GLM53VisionConfig, w: dict, prefix: str, *, weight_dtype, act_dtype, rot_mat):
        self.cfg = cfg
        self.act_dtype = act_dtype
        self.wqkv = _upload(device, w[f"{prefix}qkv.weight"].t().float(), weight_dtype)  # [hidden, 3*H*D]
        self.bqkv = _bias(device, w[f"{prefix}qkv.bias"], act_dtype)
        self.q_norm_w = _row(device, w[f"{prefix}q_norm.weight"], act_dtype)
        self.k_norm_w = _row(device, w[f"{prefix}k_norm.weight"], act_dtype)
        self.wo = _upload(device, w[f"{prefix}proj.weight"].t().float(), weight_dtype)
        self.bo = _bias(device, w[f"{prefix}proj.bias"], act_dtype)
        self.rot_mat = rot_mat  # [D, D] rotate_half as a matmul
        self.scale = cfg.head_dim**-0.5
        self.compute = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        grid = device.compute_with_storage_grid_size()
        # one K chunk per 1024-token window: no online-softmax rescaling of bf16 partials for a 448x448 image
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=64, k_chunk_size=1024, exp_approx_mode=False
        )

    def _norm_rope(self, x: ttnn.Tensor, w: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """per-head RMSNorm then ``x * cos + rotate_half(x) * sin`` ([1, H, S, D], cos/sin [1, 1, S, D])."""
        n = ttnn.rms_norm(x, epsilon=self.cfg.rms_norm_eps, weight=w, compute_kernel_config=self.hifi4)
        ttnn.deallocate(x)
        rh = ttnn.matmul(n, self.rot_mat, compute_kernel_config=self.hifi4)
        n = ttnn.multiply(n, cos, output_tensor=n)
        rh = ttnn.multiply(rh, sin, output_tensor=rh)
        n = ttnn.add(n, rh, output_tensor=n)
        ttnn.deallocate(rh)
        return n

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, cu_window: ttnn.Tensor | None):
        H = self.cfg.num_heads
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, dtype=self.act_dtype, compute_kernel_config=self.compute)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = _cast(self._norm_rope(q, self.q_norm_w, cos, sin), ttnn.bfloat16)
        k = _cast(self._norm_rope(k, self.k_norm_w, cos, sin), ttnn.bfloat16)
        v = _cast(v, ttnn.bfloat16)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            cu_window_seqlens=cu_window,
            program_config=self.sdpa_cfg,
            compute_kernel_config=self.hifi4,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,H*D] bf16
        ttnn.deallocate(attn)
        out = ttnn.linear(cat, self.wo, bias=self.bo, dtype=self.act_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(cat)
        return out


class TtVisionBlock(LightweightModule):
    def __init__(self, device, cfg: GLM53VisionConfig, w: dict, idx: int, *, weight_dtype, act_dtype, rot_mat):
        self.cfg = cfg
        self.act_dtype = act_dtype
        p = f"blocks.{idx}."
        self.norm1_w = _row(device, w[f"{p}norm1.weight"], act_dtype)
        self.norm2_w = _row(device, w[f"{p}norm2.weight"], act_dtype)
        self.attn = TtVisionAttention(
            device, cfg, w, f"{p}attn.", weight_dtype=weight_dtype, act_dtype=act_dtype, rot_mat=rot_mat
        )
        self.mlp = TtClampedSwiGLUBias(
            device,
            w[f"{p}mlp.gate_proj.weight"],
            w[f"{p}mlp.up_proj.weight"],
            w[f"{p}mlp.down_proj.weight"],
            gate_b=w[f"{p}mlp.gate_proj.bias"],
            up_b=w[f"{p}mlp.up_proj.bias"],
            down_b=w[f"{p}mlp.down_proj.bias"],
            limit=cfg.swiglu_limit,
            weight_dtype=weight_dtype,
            act_dtype=act_dtype,
        )
        self.norm_compute = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

    def forward(self, x: ttnn.Tensor, cos, sin, cu_window) -> ttnn.Tensor:
        """``x`` [1, 1, S, hidden] in ``act_dtype`` -> same shape/dtype; ``x`` is consumed (in-place adds)."""
        eps = self.cfg.rms_norm_eps
        h = ttnn.rms_norm(x, epsilon=eps, weight=self.norm1_w, compute_kernel_config=self.norm_compute)
        a = self.attn(h, cos, sin, cu_window)
        ttnn.deallocate(h)
        x = ttnn.add(x, a, output_tensor=x)
        ttnn.deallocate(a)
        h = ttnn.rms_norm(x, epsilon=eps, weight=self.norm2_w, compute_kernel_config=self.norm_compute)
        m = self.mlp(h)
        ttnn.deallocate(h)
        x = ttnn.add(x, m, output_tensor=x)
        ttnn.deallocate(m)
        return x


@dataclass
class VisionInputs:
    """Device inputs of one ``TtVisionTower.forward`` call (built by ``prepare_inputs``)."""

    patches: ttnn.Tensor  # [1, 1, S_pad, patch_dim_padded] act_dtype (fp32: bf16 pixels already cost 1e-3 PCC)
    cos: ttnn.Tensor  # [1, 1, S_pad, head_dim] act_dtype
    sin: ttnn.Tensor
    cu_window: ttnn.Tensor | None  # [n_windows + 1] uint32 ROW_MAJOR (always set by prepare_inputs)
    num_patches: int  # real rows of ``patches``
    split_sizes: list[int]  # merged tokens per image (HF ``get_image_features`` split)

    @property
    def num_tokens(self) -> int:
        return sum(self.split_sizes)


class TtVisionTower(LightweightModule):
    """Pixel patches -> projected image embeddings ``[1, 1, pad(N/4), out_hidden]`` in ``out_dtype``
    (rows past ``inputs.num_tokens`` are padding)."""

    def __init__(
        self,
        device,
        cfg: GLM53VisionConfig,
        weights: dict,
        *,
        weight_dtype=ttnn.bfloat16,
        act_dtype=ttnn.float32,
        out_dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.cfg = cfg
        self.act_dtype = act_dtype
        self.out_dtype = out_dtype
        w = weights
        # patch embed: Conv3d(3 -> hidden, kernel == stride [2, 14, 14]) over unfolded patches == matmul
        pe = w["patch_embed.proj.weight"].float().reshape(cfg.hidden_size, cfg.patch_dim)
        pe = torch.nn.functional.pad(pe, (0, cfg.patch_dim_padded - cfg.patch_dim))
        self.w_patch = _upload(device, pe.t(), act_dtype)  # [1184, hidden] (bf16 checkpoint values, exact)
        self.b_patch = _bias(device, w["patch_embed.proj.bias"], act_dtype)
        self.rot_mat = _upload(device, rotate_half_matrix(cfg.head_dim), act_dtype)
        self.blocks = [
            TtVisionBlock(device, cfg, w, i, weight_dtype=weight_dtype, act_dtype=act_dtype, rot_mat=self.rot_mat)
            for i in range(cfg.depth)
        ]
        self.post_norm_w = _row(device, w["post_layernorm.weight"], act_dtype)
        # downsample: Conv2d(hidden -> out, 2x2, stride 2) over a [2, 2, hidden] merge block whose 4
        # patches are consecutive rows (i, j) -> row 2i + j, so W[o, c, i, j] -> W2d[o, (2i + j) * hidden + c]
        ds = w["downsample.weight"].float().permute(0, 2, 3, 1).reshape(cfg.out_hidden_size, cfg.merged_dim)
        self.w_down = _upload(device, ds.t(), weight_dtype)  # [4096, 4096]
        self.b_down = _bias(device, w["downsample.bias"], act_dtype)
        self.w_mproj = _upload(device, w["merger.proj.weight"].t().float(), weight_dtype)
        self.ln_w = _row(device, w["merger.post_projection_norm.weight"], act_dtype)
        self.ln_b = _row(device, w["merger.post_projection_norm.bias"], act_dtype)
        self.merger_mlp = TtClampedSwiGLUBias(
            device,
            w["merger.gate_proj.weight"],
            w["merger.up_proj.weight"],
            w["merger.down_proj.weight"],
            limit=cfg.swiglu_limit,
            weight_dtype=weight_dtype,
            act_dtype=act_dtype,
        )
        self.compute = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

    # ---- host boundary -------------------------------------------------------------------
    def prepare_inputs(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, pad_to: int | None = None
    ) -> VisionInputs:
        """HF processor output (``pixel_values`` [N, patch_dim], ``image_grid_thw`` [n, 3]) -> device
        tensors. Pads the token axis to ``pad_to`` rows (a multiple of 32 holding N; default: N's own multiple
        of 32); the pad rows form their own attention window. The serving path pads every tower call to a patch
        bucket (``multimodal.vision_pad_patches``) so the tower's programs exist at a few shapes, all built at
        boot: a 2-group clip (2048 patches) after a 1024-patch warm-up compiled its patch embed at request time
        under the sealed program cache (2026-10-28 8x4). The window bounds are always a tensor (a single
        unpadded image included) so the attention runs one program family."""
        cfg = self.cfg
        N = pixel_values.shape[0]
        assert pixel_values.shape[1] == cfg.patch_dim, pixel_values.shape
        lens = vision_seq_lens(grid_thw)
        assert sum(lens) == N, (lens, N)
        S = -(-N // TILE) * TILE if pad_to is None else int(pad_to)
        if S < N or S % TILE:
            raise ValueError(f"pad_to {pad_to} must hold the {N} patches and be a multiple of {TILE}")
        patches = torch.zeros(S, cfg.patch_dim_padded, dtype=torch.float32)
        patches[:N, : cfg.patch_dim] = pixel_values.float()
        cos, sin = vision_rope_cos_sin(grid_thw, cfg)
        cos_p = torch.ones(S, cfg.head_dim)
        sin_p = torch.zeros(S, cfg.head_dim)
        cos_p[:N], sin_p[:N] = cos, sin
        windows = list(lens) + ([S - N] if S > N else [])
        bounds = torch.tensor([0] + list(torch.tensor(windows).cumsum(0).tolist()), dtype=torch.int32)
        cu = ttnn.from_torch(bounds, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        return VisionInputs(
            patches=_upload(self.device, patches.reshape(1, 1, S, -1), self.act_dtype),
            cos=_upload(self.device, cos_p.reshape(1, 1, S, -1), self.act_dtype),
            sin=_upload(self.device, sin_p.reshape(1, 1, S, -1), self.act_dtype),
            cu_window=cu,
            num_patches=N,
            split_sizes=[int(t * h * w) // cfg.merge_group for t, h, w in grid_thw.tolist()],
        )

    # ---- device ------------------------------------------------------------------------------
    def embed(self, inputs: VisionInputs) -> ttnn.Tensor:
        """Patch embed -> residual stream ``[1, 1, S, hidden]`` in ``act_dtype``."""
        return ttnn.linear(
            inputs.patches, self.w_patch, bias=self.b_patch, dtype=self.act_dtype, compute_kernel_config=self.hifi4
        )

    def run_blocks(self, x: ttnn.Tensor, inputs: VisionInputs, start: int = 0, stop: int | None = None):
        for blk in self.blocks[start:stop]:
            x = blk(x, inputs.cos, inputs.sin, inputs.cu_window)
        return x

    def merge(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """post_layernorm -> downsample -> merger; ``x`` [1, 1, S, hidden] (consumed) -> [1, 1, S/4, out]."""
        cfg = self.cfg
        h = ttnn.rms_norm(x, epsilon=cfg.rms_norm_eps, weight=self.post_norm_w, compute_kernel_config=self.hifi4)
        ttnn.deallocate(x)
        g = _fold_rows(h, cfg.merged_dim)  # [1, 1, S/4, 4 * hidden]
        ttnn.deallocate(h)
        y = ttnn.linear(g, self.w_down, bias=self.b_down, dtype=self.act_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(g)
        p = ttnn.linear(y, self.w_mproj, dtype=self.act_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(y)
        n = ttnn.layer_norm(
            p, epsilon=cfg.layer_norm_eps, weight=self.ln_w, bias=self.ln_b, compute_kernel_config=self.hifi4
        )
        ttnn.deallocate(p)
        a = ttnn.gelu(n)  # exact erf form, as nn.GELU()
        ttnn.deallocate(n)
        out = self.merger_mlp(a)
        ttnn.deallocate(a)
        return _cast(out, self.out_dtype)

    def forward(self, inputs: VisionInputs) -> ttnn.Tensor:
        x = self.embed(inputs)
        x = self.run_blocks(x, inputs)
        return self.merge(x)

    def image_embeddings(self, inputs: VisionInputs, out: ttnn.Tensor) -> list[torch.Tensor]:
        """Host boundary: split the padded device output into per-image ``[256, 4096]`` tensors."""
        t = ttnn.to_torch(out).reshape(-1, self.cfg.out_hidden_size)[: inputs.num_tokens]
        return list(torch.split(t, inputs.split_sizes))


__all__ = [
    "GLM53VisionConfig",
    "TtVisionTower",
    "TtVisionBlock",
    "VisionInputs",
    "load_vision_weights",
    "vision_position_ids",
    "vision_rope_cos_sin",
    "vision_seq_lens",
    "rotate_half_matrix",
]
