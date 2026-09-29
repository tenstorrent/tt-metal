# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 vision encoder: ViT x32 + aligner (bead 10.1, graph nodes V1-V3).

Semantics are ``inference/vision.py`` (``ViT``, ``Aligner``) for one image of ``n_h x n_w`` patches:

* V1 patch embed: host patchify (``image_processor.load_image``), ``Linear(588 -> 1024, bias)``;
* V2 block: pre-RMSNorm (eps 1e-6), ``wqkv`` (+bias) chunked into head-major q | k | v, 2D RoPE, bidirectional
  SDPA over the image (scale 64^-0.5), ``wo`` (+bias), pre-RMSNorm SwiGLU ``w2(silu(gate) * up)`` with
  ``gate, up = w1(x).chunk(2)`` and no biases; final RMSNorm;
* 2D RoPE is rotate-half over the full head_dim 64: ``x[:32]`` pairs with ``x[32:]`` and the 32 angles of a
  patch at row h, column w are ``[h*f0..h*f15, w*f0..w*f15]``, ``f_i = theta^(-2i/32)``;
* V3 aligner: grid ``[n_h, n_w, 1024]``, zero-pad right/bottom to multiples of 3, 3x3 stride-3 unfold with feature
  order ``c*9 + kh*3 + kw`` and tokens in row-major order, ``Linear(9216 -> 5120, bias)``, exact GELU,
  ``Linear(5120 -> 5120, bias)``.

Device mapping. The patch sequence is padded to a multiple of the SDPA chunk; ``cu_window_seqlens = [0, N, S]``
keeps the padded rows out of the image's attention (block-diagonal windowed SDPA, the Qwen2.5-VL precedent), and
rotate-half RoPE is ``rotary_embedding_hf`` with ``cos/sin = [c, c]`` over the 64 dims. The aligner never runs an
unfold: the grid is reshaped in row-major so that each 3-column group becomes one ``3*1024`` row (``kw*C + c``), the
three ``kh`` rows are sliced and concatenated (``kh*3C + kw*C + c``), and ``w1``'s input columns are permuted on
the host to that order, which is exact.

Placement: the whole encoder (~0.9 GB bf16) is replicated on every device of the mesh and runs without
collectives; the output ``[1, 1, T, 5120]`` is replicated, which is what the image-span merge (10.2) slices from.
Qwen-VL's vision modules are not reused as classes because they are bound to ``tt_transformers`` ``ModelArgs``
(decode/paged/prefetch configuration); their SDPA windowing and QKV head split are reused as mechanisms.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tt.v41.weights import V41Checkpoint

SDPA_CHUNK = 128  # q and k chunk; the patch sequence is padded to a multiple of it


def vision_weight_names(config=DeepSeekV41FlashConfig) -> list[str]:
    """Checkpoint names of the ViT and aligner tensors (the image span delimiters belong to 10.2)."""
    names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight"]
    for i in range(config.VISION_N_LAYERS):
        p = f"vision.blocks.{i}"
        names += [f"{p}.{s}" for s in ("norm1.weight", "norm2.weight", "mlp.w1.weight", "mlp.w2.weight")]
        names += [f"{p}.attn.{s}.{t}" for s in ("wqkv", "wo") for t in ("weight", "bias")]
    names += [f"aligner.{s}.{t}" for s in ("w1", "w2") for t in ("weight", "bias")]
    return names


def load_vision_weights(ckpt: V41Checkpoint, config=DeepSeekV41FlashConfig) -> dict[str, torch.Tensor]:
    """Stored ViT/aligner tensors by checkpoint name, ``[out, in]`` orientation (the vision tower is not quantized)."""
    raw = ckpt.read(vision_weight_names(config))
    for name, t in raw.items():
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name}: stored dtype {t.dtype}, expected bf16/fp32")
    return raw


def vision_cos_sin(n_h: int, n_w: int, rows: int, config=DeepSeekV41FlashConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin ``[1, 1, rows, head_dim]`` fp32 in rotate-half layout for the ``n_h x n_w`` grid in row-major order.

    Rows past ``n_h * n_w`` (sequence padding) get the identity rotation."""
    head_dim = config.VISION_DIM // config.VISION_N_HEADS
    half = head_dim // 2
    inv_freq = 1.0 / (config.VISION_ROPE_THETA ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    h = torch.arange(n_h, dtype=torch.float32).repeat_interleave(n_w)
    w = torch.arange(n_w, dtype=torch.float32).repeat(n_h)
    angles = torch.cat([torch.outer(h, inv_freq), torch.outer(w, inv_freq)], dim=-1)  # [N, half]
    angles = torch.cat([angles, angles], dim=-1)  # x[:half] and x[half:] share angle i
    cos = torch.ones(rows, head_dim)
    sin = torch.zeros(rows, head_dim)
    cos[: n_h * n_w], sin[: n_h * n_w] = angles.cos(), angles.sin()
    return cos[None, None], sin[None, None]


class TtV41Vision(LightweightModule):
    """ViT + aligner for one image; ``weights`` by checkpoint name (``load_vision_weights``)."""

    def __init__(self, mesh_device, weights: dict[str, torch.Tensor], config=DeepSeekV41FlashConfig):
        self.mesh_device = mesh_device
        self.config = config
        self.dim = config.VISION_DIM
        self.n_heads = config.VISION_N_HEADS
        self.head_dim = self.dim // self.n_heads
        self.inter = config.VISION_INTER_DIM
        self.ratio = config.VISION_DOWNSAMPLE_RATIO
        self.eps = config.VISION_NORM_EPS
        self.patch_in = 3 * config.VISION_PATCH_SIZE**2
        self.patch_in_padded = math.ceil(self.patch_in / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=SDPA_CHUNK,
            k_chunk_size=SDPA_CHUNK,
            exp_approx_mode=False,
        )
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def upload(t: torch.Tensor):
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        def linear_weight(w: torch.Tensor):  # [out, in] -> [1, 1, in, out]
            return upload(w.transpose(-2, -1).contiguous()[None, None])

        def row(v: torch.Tensor):  # bias / norm weight -> [1, 1, 1, n]
            return upload(v.reshape(1, 1, 1, -1))

        pe = weights["vision.patch_embed.proj.weight"].reshape(self.dim, self.patch_in)
        pe = torch.nn.functional.pad(pe, (0, self.patch_in_padded - self.patch_in))
        self.patch_w = linear_weight(pe)
        self.patch_b = row(weights["vision.patch_embed.proj.bias"])

        self.blocks = []
        for i in range(config.VISION_N_LAYERS):
            p = f"vision.blocks.{i}"
            w1 = weights[f"{p}.mlp.w1.weight"]
            self.blocks.append(
                {
                    "norm1": row(weights[f"{p}.norm1.weight"]),
                    "wqkv": linear_weight(weights[f"{p}.attn.wqkv.weight"]),
                    "wqkv_b": row(weights[f"{p}.attn.wqkv.bias"]),
                    "wo": linear_weight(weights[f"{p}.attn.wo.weight"]),
                    "wo_b": row(weights[f"{p}.attn.wo.bias"]),
                    "norm2": row(weights[f"{p}.norm2.weight"]),
                    "w_gate": linear_weight(w1[: self.inter]),
                    "w_up": linear_weight(w1[self.inter :]),
                    "w2": linear_weight(weights[f"{p}.mlp.w2.weight"]),
                }
            )
        self.norm = row(weights["vision.norm.weight"])

        # w1 columns c*r*r + kh*r + kw -> kh*r*C + kw*C + c, the order the device gather produces
        r = self.ratio
        a1 = weights["aligner.w1.weight"]
        a1 = a1.reshape(a1.shape[0], self.dim, r, r).permute(0, 2, 3, 1).reshape(a1.shape[0], -1)
        self.align_w1 = linear_weight(a1)
        self.align_b1 = row(weights["aligner.w1.bias"])
        self.align_w2 = linear_weight(weights["aligner.w2.weight"])
        self.align_b2 = row(weights["aligner.w2.bias"])

    def _linear(self, x, w, bias=None):
        return ttnn.linear(
            x,
            w,
            bias=bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _block(self, x, blk, cos, sin, cu_seqlens):
        h = ttnn.rms_norm(x, weight=blk["norm1"], epsilon=self.eps, compute_kernel_config=self.compute_kernel_config)
        qkv = self._linear(h, blk["wqkv"], blk["wqkv_b"])
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=self.head_dim**-0.5,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            cu_window_seqlens=cu_seqlens,
        )
        for t in (q_rot, k_rot, v):
            ttnn.deallocate(t)
        o_cat = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        attn = self._linear(o_cat, blk["wo"], blk["wo_b"])
        ttnn.deallocate(o_cat)
        x1 = ttnn.add(x, attn)
        ttnn.deallocate(attn)

        h = ttnn.rms_norm(x1, weight=blk["norm2"], epsilon=self.eps, compute_kernel_config=self.compute_kernel_config)
        gate = self._linear(h, blk["w_gate"])
        up = self._linear(h, blk["w_up"])
        ttnn.deallocate(h)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mlp = self._linear(act, blk["w2"])
        ttnn.deallocate(act)
        out = ttnn.add(x1, mlp)
        ttnn.deallocate(x1)
        ttnn.deallocate(mlp)
        return out

    def vit(self, patches: torch.Tensor, n_h: int, n_w: int):
        """``patches`` ``[n_h*n_w, 3, p, p]`` (host, row-major grid) -> ViT output ``[1, 1, S, 1024]`` bf16 on device,
        replicated; rows ``>= n_h*n_w`` are sequence padding."""
        n = n_h * n_w
        if patches.shape[0] != n:
            raise ValueError(f"{patches.shape[0]} patches for a {n_h}x{n_w} grid")
        rows = math.ceil(n / SDPA_CHUNK) * SDPA_CHUNK
        x = torch.zeros(1, 1, rows, self.patch_in_padded, dtype=torch.bfloat16)
        x[0, 0, :n, : self.patch_in] = patches.reshape(n, -1).to(torch.bfloat16)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tt_x = ttnn.from_torch(
            x, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate
        )
        cos, sin = (
            ttnn.from_torch(
                t, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate
            )
            for t in vision_cos_sin(n_h, n_w, rows, self.config)
        )
        bounds = [0, n] + ([rows] if rows > n else [])
        cu_seqlens = ttnn.from_torch(
            torch.tensor(bounds, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate,
        )

        h = self._linear(tt_x, self.patch_w, self.patch_b)
        ttnn.deallocate(tt_x)
        for blk in self.blocks:
            h = self._block(h, blk, cos, sin, cu_seqlens)
        out = ttnn.rms_norm(h, weight=self.norm, epsilon=self.eps, compute_kernel_config=self.compute_kernel_config)
        for t in (h, cos, sin, cu_seqlens):
            ttnn.deallocate(t)
        return out

    def aligner(self, x, n_h: int, n_w: int):
        """ViT output ``[1, 1, S >= n_h*n_w, 1024]`` -> ``[1, 1, T, 5120]`` with ``T = ceil(n_h/3) * ceil(n_w/3)``."""
        r, c = self.ratio, self.dim
        hr, wr = math.ceil(n_h / r), math.ceil(n_w / r)
        tokens = hr * wr
        grid = ttnn.slice(x, [0, 0, 0, 0], [1, 1, n_h * n_w, c])
        grid = ttnn.reshape(ttnn.to_layout(grid, ttnn.ROW_MAJOR_LAYOUT), [1, n_h, n_w, c])
        if hr * r != n_h or wr * r != n_w:
            grid = ttnn.pad(grid, [(0, 0), (0, hr * r - n_h), (0, wr * r - n_w), (0, 0)], 0.0)
        # row (r*i + kh) * (r*wr) + r*j + kw  ->  [i, kh, j, kw*C + c]
        grid = ttnn.reshape(grid, [hr, r, wr, r * c])
        parts = [
            ttnn.reshape(ttnn.slice(grid, [0, kh, 0, 0], [hr, kh + 1, wr, r * c]), [1, 1, tokens, r * c])
            for kh in range(r)
        ]
        feats = ttnn.to_layout(ttnn.concat(parts, dim=-1), ttnn.TILE_LAYOUT)  # [.., kh*r*C + kw*C + c]
        for t in parts + [grid]:
            ttnn.deallocate(t)
        h = self._linear(feats, self.align_w1, self.align_b1)
        ttnn.deallocate(feats)
        act = ttnn.gelu(h, fast_and_approximate_mode=False)
        ttnn.deallocate(h)
        out = self._linear(act, self.align_w2, self.align_b2)
        ttnn.deallocate(act)
        return out

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int):
        """One image's patches ``[n_h*n_w, 3, p, p]`` -> aligner rows ``[1, 1, T, 5120]`` bf16, replicated."""
        vit_out = self.vit(patches, n_h, n_w)
        out = self.aligner(vit_out, n_h, n_w)
        ttnn.deallocate(vit_out)
        return out
