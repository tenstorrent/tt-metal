# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hunyuan3D-2.1 ShapeVAE decoder on one device: post_kl + 16-layer transformer over the 4096 latents, then the
geometry decoder (Fourier query embedding -> one cross-attention block to the latents -> LayerNorm -> SDF logit),
evaluated over chunks of grid points.

Upstream splits q|k|v (k|v) per head out of one fused projection, so head h's q, k, v are interleaved slices of it;
`_split_interleaved` reorders the rows so a standard head split matches. q/k norms are LayerNorm(head_dim) here. The
per-latent k and v of the geometry decoder do not depend on the queries, so they are computed once per mesh.
"""

from __future__ import annotations


import torch

import ttnn
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor

WIDTH = 1024
HEADS = 16
HEAD_DIM = WIDTH // HEADS
LAYERS = 16
LATENTS = 4096
EMBED_DIM = 64
NUM_FREQS = 8
QUERY_DIM = 3 * (2 * NUM_FREQS + 1)  # 51: x, sin, cos per axis
QUERY_PAD = 64
EPS = 1e-6
SCALE_FACTOR = 1.0039506158752403


def _hifi4(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _bf16(x):
    return ttnn.typecast(x, ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x


def _split_interleaved(w: torch.Tensor, parts: int) -> list[torch.Tensor]:
    """Rows of a fused [parts*WIDTH, in] projection, head h owning rows [h*parts*d, (h+1)*parts*d) -> per-part
    [WIDTH, in] weights in the standard (head-major) layout."""
    w = w.reshape(HEADS, parts, HEAD_DIM, *w.shape[1:])
    return [w[:, j].reshape(WIDTH, *w.shape[3:]).contiguous() for j in range(parts)]


def fourier_embed(points: torch.Tensor) -> torch.Tensor:
    """[N, 3] -> [N, 51], FourierEmbedder(num_freqs=8, logspace, include_input, no pi): x, sin(2^k x), cos(2^k x)."""
    f = 2.0 ** torch.arange(NUM_FREQS, dtype=torch.float32)
    e = (points[..., None] * f).reshape(*points.shape[:-1], -1)
    return torch.cat([points, e.sin(), e.cos()], dim=-1)


def dense_grid(resolution: int, bound: float = 1.01) -> torch.Tensor:
    """generate_dense_grid_points with indexing='ij': (res+1)^3 points over [-bound, bound]^3, x slowest."""
    lin = torch.linspace(-bound, bound, resolution + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([x, y, z], -1).reshape(-1, 3)


class HeadLayerNorm(Module):
    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        self.weight = Parameter(total_shape=[1, 1, 1, HEAD_DIM], device=mesh_device, dtype=ttnn.float32)
        self.bias = Parameter(total_shape=[1, 1, 1, HEAD_DIM], device=mesh_device, dtype=ttnn.float32)
        self.cfg = _hifi4(mesh_device)

    def _prepare_torch_state(self, state):
        for p in ("weight", "bias"):
            if p in state:
                state[p] = state[p].reshape(1, 1, 1, -1)

    def forward(self, x):
        return ttnn.layer_norm(
            x, epsilon=EPS, weight=self.weight.data, bias=self.bias.data, compute_kernel_config=self.cfg
        )


def _heads(x, n=HEADS):
    out, _, _ = ttnn.experimental.nlp_create_qkv_heads(x, num_heads=n, num_kv_heads=0, transpose_k_heads=False)
    return out


class _AttnCore(Module):
    def __init__(self, *, mesh_device, k_chunk: int) -> None:
        super().__init__()
        self.q_norm = HeadLayerNorm(mesh_device=mesh_device)
        self.k_norm = HeadLayerNorm(mesh_device=mesh_device)
        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=k_chunk, exp_approx_mode=False
        )
        self.sdpa_kernel = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def norm_q(self, q):
        return _bf16(self.q_norm(ttnn.typecast(_heads(q), ttnn.float32)))

    def norm_k(self, k):
        return _bf16(self.k_norm(ttnn.typecast(_heads(k), ttnn.float32)))

    def forward(self, q, k, v):
        return self.attend(q, k, v)

    def attend(self, q, k, v):
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, program_config=self.sdpa_cfg, compute_kernel_config=self.sdpa_kernel
        )
        o = ttnn.transformer.concatenate_heads(out)  # [B, S, D] -> [B, 1, S, D]
        return ttnn.reshape(o, (o.shape[0], 1, o.shape[-2], o.shape[-1]))


class MLP(Module):
    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        mm = _hifi4(mesh_device)
        self.c_fc = Linear(WIDTH, 4 * WIDTH, activation_fn="gelu", mesh_device=mesh_device, compute_kernel_config=mm)
        self.c_proj = Linear(4 * WIDTH, WIDTH, mesh_device=mesh_device, compute_kernel_config=mm)

    def forward(self, x):
        return self.c_proj(self.c_fc(x))


class SelfBlock(Module):
    """ResidualAttentionBlock: x + attn(ln_1 x), x + mlp(ln_2 x); qkv without bias."""

    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        mm = _hifi4(mesh_device)
        self.ln_1 = LayerNorm(WIDTH, norm_eps=EPS, mesh_device=mesh_device)
        self.ln_2 = LayerNorm(WIDTH, norm_eps=EPS, mesh_device=mesh_device)
        self.to_q = Linear(WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_k = Linear(WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_v = Linear(WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.c_proj = Linear(WIDTH, WIDTH, mesh_device=mesh_device, compute_kernel_config=mm)
        self.core = _AttnCore(mesh_device=mesh_device, k_chunk=512)
        self.mlp = MLP(mesh_device=mesh_device)

    def _prepare_torch_state(self, state):
        if "attn.c_qkv.weight" in state:
            q, k, v = _split_interleaved(state.pop("attn.c_qkv.weight"), 3)
            state["to_q.weight"], state["to_k.weight"], state["to_v.weight"] = q, k, v
            state["c_proj.weight"], state["c_proj.bias"] = state.pop("attn.c_proj.weight"), state.pop(
                "attn.c_proj.bias"
            )
            for n in ("q_norm", "k_norm"):
                for p in ("weight", "bias"):
                    state[f"core.{n}.{p}"] = state.pop(f"attn.attention.{n}.{p}")

    def forward(self, x):
        h = _bf16(self.ln_1(x))
        a = self.core.attend(self.core.norm_q(self.to_q(h)), self.core.norm_k(self.to_k(h)), _heads(self.to_v(h)))
        x = ttnn.add(x, ttnn.typecast(self.c_proj(a), ttnn.float32))
        return ttnn.add(x, ttnn.typecast(self.mlp(_bf16(self.ln_2(x))), ttnn.float32))


class GeoDecoder(Module):
    """CrossAttentionDecoder: query_proj(fourier(p)); x + cross_attn(ln_1 x, ln_2 latents); x + mlp(ln_3 x);
    ln_post; output_proj -> 1 logit per point."""

    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        mm = _hifi4(mesh_device)
        self.query_proj = Linear(QUERY_PAD, WIDTH, mesh_device=mesh_device, compute_kernel_config=mm)
        self.ln_1 = LayerNorm(WIDTH, norm_eps=EPS, mesh_device=mesh_device)
        self.ln_2 = LayerNorm(WIDTH, norm_eps=EPS, mesh_device=mesh_device)
        self.ln_3 = LayerNorm(WIDTH, norm_eps=EPS, mesh_device=mesh_device)
        self.c_q = Linear(
            WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm
        )  # qkv_bias: false
        self.to_k = Linear(WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_v = Linear(WIDTH, WIDTH, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.c_proj = Linear(WIDTH, WIDTH, mesh_device=mesh_device, compute_kernel_config=mm)
        self.core = _AttnCore(mesh_device=mesh_device, k_chunk=512)
        self.mlp = MLP(mesh_device=mesh_device)
        self.ln_post = LayerNorm(WIDTH, norm_eps=1e-5, mesh_device=mesh_device)  # nn.LayerNorm default eps
        self.output_proj = Linear(WIDTH, 32, mesh_device=mesh_device, compute_kernel_config=mm)  # 1 logit, padded

    def _prepare_torch_state(self, state):
        if "query_proj.weight" in state:
            w = state["query_proj.weight"]
            state["query_proj.weight"] = torch.nn.functional.pad(w, (0, QUERY_PAD - w.shape[1]))
        p = "cross_attn_decoder."
        if p + "attn.c_kv.weight" in state:
            for n in ("ln_1", "ln_2", "ln_3"):
                for q in ("weight", "bias"):
                    state[f"{n}.{q}"] = state.pop(f"{p}{n}.{q}")
            state["c_q.weight"] = state.pop(f"{p}attn.c_q.weight")
            for q in ("weight", "bias"):
                state[f"c_proj.{q}"] = state.pop(f"{p}attn.c_proj.{q}")
            state["to_k.weight"], state["to_v.weight"] = _split_interleaved(state.pop(p + "attn.c_kv.weight"), 2)
            for n in ("q_norm", "k_norm"):
                for q in ("weight", "bias"):
                    state[f"core.{n}.{q}"] = state.pop(f"{p}attn.attention.{n}.{q}")
            for q in ("c_fc", "c_proj"):
                for r in ("weight", "bias"):
                    state[f"mlp.{q}.{r}"] = state.pop(f"{p}mlp.{q}.{r}")
        if "output_proj.weight" in state:
            w, b = state["output_proj.weight"], state["output_proj.bias"]
            state["output_proj.weight"] = torch.nn.functional.pad(w, (0, 0, 0, 32 - w.shape[0]))
            state["output_proj.bias"] = torch.nn.functional.pad(b, (0, 32 - b.shape[0]))

    def prepare_kv(self, latents):
        d = _bf16(self.ln_2(latents))
        return self.core.norm_k(self.to_k(d)), _heads(self.to_v(d))

    def forward(self, query_emb, kv):
        x = self.query_proj(query_emb, dtype=ttnn.float32)
        q = self.core.norm_q(self.c_q(_bf16(self.ln_1(x))))
        x = ttnn.add(x, ttnn.typecast(self.c_proj(self.core.attend(q, *kv)), ttnn.float32))
        x = ttnn.add(x, ttnn.typecast(self.mlp(_bf16(self.ln_3(x))), ttnn.float32))
        return self.output_proj(_bf16(self.ln_post(x)), dtype=ttnn.float32)


class TtShapeVAEDecoder(Module):
    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.post_kl = Linear(EMBED_DIM * 2, WIDTH, mesh_device=mesh_device, compute_kernel_config=_hifi4(mesh_device))
        self.blocks = ModuleList(SelfBlock(mesh_device=mesh_device) for _ in range(LAYERS))
        self.geo = GeoDecoder(mesh_device=mesh_device)

    def _prepare_torch_state(self, state):
        if "post_kl.weight" in state:  # latents enter as bf16 hi + lo against [W; W]
            w = state["post_kl.weight"]
            state["post_kl.weight"] = torch.cat([w, w], dim=1)
        for k in list(state):
            if k.startswith("transformer.resblocks."):
                state["blocks." + k[len("transformer.resblocks.") :]] = state.pop(k)
            elif k.startswith("geo_decoder."):
                state["geo." + k[len("geo_decoder.") :]] = state.pop(k)
            elif k.startswith(("encoder.", "pre_kl.")):
                del state[k]

    def forward(self, latents: torch.Tensor) -> ttnn.Tensor:
        return self.decode_latents(latents)

    def decode_latents(self, latents: torch.Tensor) -> ttnn.Tensor:
        """Sampler output [1, 4096, 64] (before 1/scale_factor) -> transformer output on device [1, 1, 4096, 1024] fp32."""
        z = latents.float() / SCALE_FACTOR
        hi = z.bfloat16().float()
        x = self.post_kl(
            bf16_tensor(torch.cat([hi, z - hi], -1).unsqueeze(1), device=self.mesh_device), dtype=ttnn.float32
        )
        for blk in self.blocks:
            x = blk(x)
        return x

    def grid_logits(
        self, latents_dev: ttnn.Tensor, resolution: int, chunk: int = 32768, bound: float = 1.01
    ) -> torch.Tensor:
        """Dense grid SDF logits [(res+1)^3] for one mesh (VanillaVolumeDecoder), chunk queries per call."""
        kv = self.geo.prepare_kv(latents_dev)
        pts = dense_grid(resolution, bound)
        out = []
        for s in range(0, pts.shape[0], chunk):
            p = pts[s : s + chunk]
            n = p.shape[0]
            emb = torch.zeros(1, 1, chunk, QUERY_PAD)
            emb[0, 0, :n, :QUERY_DIM] = fourier_embed(p)
            o = self.geo(bf16_tensor(emb, device=self.mesh_device), kv)
            out.append(ttnn.to_torch(o)[0, 0, :n, 0].float())
        return torch.cat(out).reshape(resolution + 1, resolution + 1, resolution + 1)
