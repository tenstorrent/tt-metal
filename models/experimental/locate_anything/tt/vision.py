# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of NVIDIA LocateAnything-3B vision tower (MoonViT-SO-400M) + mlp1 projector.

Single Blackhole p150a, batch=1, single image. PRECISION FIRST: bf16 activations /
weights with HiFi4 math fidelity (fp32 dest accumulate) on every matmul + SDPA.

Architecture (mirrors ~/.cache/.../modeling_vit.py, authoritative):
  patch_embed:  Conv2d(3,1152,k=14,s=14) == matmul([L,588] @ [588,1152]) + bias,
                then + bicubic-interpolated Learnable2DInterpPosEmb (host one-time const).
  encoder:      27 x MoonVitEncoderLayer (LayerNorm eps=1e-5, attn_bias=True):
                  x = x + wo(attn(norm0(x))) ;  x = x + mlp(norm1(x))
                attn: fused wqkv(1152->3456) -> 16 heads x head_dim 72 (pad 96),
                      2D-RoPE (interleaved complex convention) on q,k,
                      full bidirectional SDPA (one window, cu_seqlens=[0,L]),
                      wo(1152->1152)+bias.
                mlp:  fc0(1152->4304) -> GELU(tanh) -> fc1(4304->1152).
                final_layernorm after the 27 blocks.
  patch_merger: 2x2 spatial merge -> [L/4, 4608].
  mlp1:         LayerNorm(4608) -> Linear(4608,2048) -> GELU -> Linear(2048,2048).

RoPE gotcha (validated against torch apply_rope at PCC 1.0, and on-device at 0.99999):
  MoonViT uses the *interleaved complex* convention (view_as_complex over adjacent
  pairs), which is EXACTLY what ttnn.experimental.rotary_embedding_llama implements
  given cos/sin built as repeat_interleave of Re/Im(freqs_cis). head_dim 72 is padded
  to 96 with cos=1, sin=0 so the padded lanes are an identity rotation.
"""

import glob
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn

HIDDEN = 1152
N_LAYERS = 27
N_HEADS = 16
HEAD_DIM = 72
PAD_HEAD_DIM = 96  # tile-aligned (multiple of 32)
INTERMEDIATE = 4304
PATCH = 14
MERGE = (2, 2)
LN_EPS = 1e-5
THETA_BASE = 10000.0
POS_EMB_HW = 64
MLP1_IN = HIDDEN * MERGE[0] * MERGE[1]  # 4608
PROJ_OUT = 2048


def _load_vision_state_dict(model_path):
    """Load only vision_model.* and mlp1.* tensors from the HF safetensors snapshot."""
    sd = {}
    for st in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(st, "pt") as f:
            for k in f.keys():
                if k.startswith("vision_model.") or k.startswith("mlp1."):
                    sd[k] = f.get_tensor(k)
    assert sd, f"No vision/mlp1 weights found under {model_path}"
    return sd


def _precompute_freqs_cis(head_dim, max_h, max_w, theta_base=THETA_BASE):
    """Exact port of Rope2DPosEmb._precompute_freqs_cis (returns [max_h, max_w, head_dim/2] complex)."""
    N = max_h * max_w
    flat_pos = torch.arange(0, N).float()
    x_pos = flat_pos % max_w
    y_pos = flat_pos // max_w
    dim_range = torch.arange(0, head_dim, 4)[: (head_dim // 4)].float()  # C/4
    freqs = 1.0 / (theta_base ** (dim_range / head_dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(max_h, max_w, -1)  # [max_h, max_w, head_dim/2]


def build_rope_cos_sin(grid_hw, head_dim=HEAD_DIM, pad_head_dim=PAD_HEAD_DIM):
    """Host cos/sin for ttnn.experimental.rotary_embedding_llama (interleaved convention).

    Returns cos,sin torch tensors of shape [1, 1, L, pad_head_dim], padded lanes = identity.
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    fc = _precompute_freqs_cis(head_dim, max(h, POS_EMB_HW), max(w, POS_EMB_HW))
    fc = fc[:h, :w].reshape(-1, head_dim // 2)  # [L, head_dim/2] complex
    cos = torch.repeat_interleave(fc.real, 2, dim=-1)  # [L, head_dim]
    sin = torch.repeat_interleave(fc.imag, 2, dim=-1)
    cos_p = F.pad(cos, (0, pad_head_dim - head_dim), value=0.0)
    cos_p[:, head_dim:] = 1.0  # identity rotation on padded lanes (cos=1)
    sin_p = F.pad(sin, (0, pad_head_dim - head_dim), value=0.0)  # sin=0
    return cos_p.unsqueeze(0).unsqueeze(0), sin_p.unsqueeze(0).unsqueeze(0)


def build_patch_embed_const(state_dict, grid_hw):
    """Host: conv weight (flattened to matmul) + per-position interpolated pos_emb.

    Returns (proj_w [588,1152], proj_b [1152], pos_emb [L,1152]).
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    conv_w = state_dict["vision_model.patch_embed.proj.weight"].float()  # [1152,3,14,14]
    conv_b = state_dict["vision_model.patch_embed.proj.bias"].float()  # [1152]
    proj_w = conv_w.reshape(conv_w.shape[0], -1).t().contiguous()  # [588,1152]
    pos = state_dict["vision_model.patch_embed.pos_emb.weight"].float()  # [64,64,1152]
    if (h, w) == (POS_EMB_HW, POS_EMB_HW):
        pos_emb = pos.reshape(-1, HIDDEN)
    else:
        pos_emb = (
            F.interpolate(pos.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode="bicubic")
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, HIDDEN)
        )
    return proj_w, conv_b, pos_emb


def _pad_per_head(t_2d_or_1d, n_heads, head_dim, pad_head_dim):
    """Pad a packed-per-head weight/bias tensor's head_dim from head_dim->pad_head_dim with zeros.

    For a 2D weight the LAST dim is the packed (n_heads*head_dim) output; for 1D it's the only dim.
    """
    if t_2d_or_1d.dim() == 2:
        in_dim = t_2d_or_1d.shape[0]
        t = t_2d_or_1d.reshape(in_dim, n_heads, head_dim)
        t = F.pad(t, (0, pad_head_dim - head_dim))
        return t.reshape(in_dim, n_heads * pad_head_dim)
    else:
        t = t_2d_or_1d.reshape(n_heads, head_dim)
        t = F.pad(t, (0, pad_head_dim - head_dim))
        return t.reshape(-1)


class MoonViT:
    """TT-NN MoonViT vision tower + mlp1 projector for a single image on one device."""

    def __init__(self, device, model_path, grid_hw, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.L = self.grid_hw[0] * self.grid_hw[1]
        self.scale = HEAD_DIM**-0.5  # NOTE: real head_dim (72), not padded

        # Precision-first: HiFi4 + fp32 dest accumulate on every matmul / SDPA.
        self.ck_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        sd = _load_vision_state_dict(model_path)
        self.state_dict = sd

        # --- patch_embed host consts ---
        proj_w, proj_b, pos_emb = build_patch_embed_const(sd, self.grid_hw)
        self.proj_w = self._to_dev(proj_w)  # [588,1152]
        self.proj_b = self._to_dev(proj_b.reshape(1, -1))  # [1,1152]
        self.pos_emb = self._to_dev(pos_emb.reshape(1, 1, self.L, HIDDEN))

        # --- rope cos/sin (always bf16: rotary_embedding_llama requires bf16) ---
        cos, sin = build_rope_cos_sin(self.grid_hw)
        self.rope_cos = self._to_dev(cos, dtype=ttnn.bfloat16)  # [1,1,L,pad_head_dim]
        self.rope_sin = self._to_dev(sin, dtype=ttnn.bfloat16)

        # --- attention mask (single full window over the real L tokens) ---
        # Plain non-causal SDPA + additive mask: real tokens attend to all real tokens
        # (full bidirectional), and never to padding rows. Padding-row outputs are sliced off.
        self.seq_pad = self._seq_pad(self.L)
        if self.seq_pad > self.L:
            mask = torch.zeros(1, 1, self.seq_pad, self.seq_pad, dtype=torch.float32)
            mask[:, :, :, self.L :] = float("-inf")  # no token may attend to padding cols
            mask[:, :, self.L :, :] = float("-inf")  # padding rows attend to nothing (avoid NaN: keep diag)
            # keep a valid row for padding queries so softmax doesn't produce NaN
            for i in range(self.L, self.seq_pad):
                mask[0, 0, i, i] = 0.0
            self.attn_mask = self._to_dev(mask, dtype=ttnn.bfloat16)
        else:
            self.attn_mask = None
        # transformation matrix for the interleaved rotary op (single tile)
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        self.rope_trans = ttnn.from_torch(
            get_rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # --- per-block weights ---
        self.blocks = [self._load_block(sd, i) for i in range(N_LAYERS)]

        # --- final layernorm ---
        self.final_ln_w = self._to_dev(sd["vision_model.encoder.final_layernorm.weight"].reshape(1, -1))
        self.final_ln_b = self._to_dev(sd["vision_model.encoder.final_layernorm.bias"].reshape(1, -1))

        # --- mlp1 projector ---
        self.mlp1_ln_w = self._to_dev(sd["mlp1.0.weight"].reshape(1, -1))  # LayerNorm(4608)
        self.mlp1_ln_b = self._to_dev(sd["mlp1.0.bias"].reshape(1, -1))
        self.mlp1_w1 = self._to_dev(sd["mlp1.1.weight"].t().contiguous())  # [4608,2048]
        self.mlp1_b1 = self._to_dev(sd["mlp1.1.bias"].reshape(1, -1))
        self.mlp1_w2 = self._to_dev(sd["mlp1.3.weight"].t().contiguous())  # [2048,2048]
        self.mlp1_b2 = self._to_dev(sd["mlp1.3.bias"].reshape(1, -1))

    def _to_dev(self, t, layout=ttnn.TILE_LAYOUT, dtype=None):
        return ttnn.from_torch(
            t,
            dtype=dtype or self.dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_block(self, sd, i):
        p = f"vision_model.encoder.blocks.{i}"
        # wqkv: fused [3456,1152] -> need q,k,v per-head padded then re-fused.
        wqkv = sd[f"{p}.wqkv.weight"].float()  # [3456,1152] (out, in)
        wqkv_b = sd[f"{p}.wqkv.bias"].float()  # [3456]
        wq, wk, wv = torch.chunk(wqkv, 3, dim=0)  # each [1152,1152] (out, in)
        bq, bk, bv = torch.chunk(wqkv_b, 3, dim=0)  # each [1152]
        # transpose to (in, out) for matmul, pad out per-head 72->96
        wq_t = _pad_per_head(wq.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)  # [1152, 1536]
        wk_t = _pad_per_head(wk.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wv_t = _pad_per_head(wv.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wqkv_fused = torch.cat([wq_t, wk_t, wv_t], dim=-1)  # [1152, 3*1536]
        bq_p = _pad_per_head(bq, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        bk_p = _pad_per_head(bk, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        bv_p = _pad_per_head(bv, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wqkv_b_fused = torch.cat([bq_p, bk_p, bv_p], dim=-1)  # [3*1536]

        # wo: [1152,1152] (out,in). nlp_concat_heads emits padded-head layout, so pad wo INPUT
        # (which corresponds to per-head dims) with zeros in the padded lanes.
        wo = sd[f"{p}.wo.weight"].float()  # [1152,1152] (out, in=n_heads*head_dim)
        wo_in = wo.reshape(HIDDEN, N_HEADS, HEAD_DIM)
        wo_in = F.pad(wo_in, (0, PAD_HEAD_DIM - HEAD_DIM))  # pad input head_dim
        wo_t = wo_in.reshape(HIDDEN, N_HEADS * PAD_HEAD_DIM).t().contiguous()  # [1536, 1152] (in, out)
        wo_b = sd[f"{p}.wo.bias"].float()

        return {
            "norm0_w": self._to_dev(sd[f"{p}.norm0.weight"].reshape(1, -1)),
            "norm0_b": self._to_dev(sd[f"{p}.norm0.bias"].reshape(1, -1)),
            "norm1_w": self._to_dev(sd[f"{p}.norm1.weight"].reshape(1, -1)),
            "norm1_b": self._to_dev(sd[f"{p}.norm1.bias"].reshape(1, -1)),
            "wqkv": self._to_dev(wqkv_fused),  # [1152, 4608]
            "wqkv_b": self._to_dev(wqkv_b_fused.reshape(1, -1)),
            "wo": self._to_dev(wo_t),  # [1536, 1152]
            "wo_b": self._to_dev(wo_b.reshape(1, -1)),
            "fc0_w": self._to_dev(sd[f"{p}.mlp.fc0.weight"].t().contiguous()),  # [1152,4304]
            "fc0_b": self._to_dev(sd[f"{p}.mlp.fc0.bias"].reshape(1, -1)),
            "fc1_w": self._to_dev(sd[f"{p}.mlp.fc1.weight"].t().contiguous()),  # [4304,1152]
            "fc1_b": self._to_dev(sd[f"{p}.mlp.fc1.bias"].reshape(1, -1)),
        }

    # ------------------------------------------------------------------ #
    def _layer_norm(self, x, w, b):
        return ttnn.layer_norm(x, epsilon=LN_EPS, weight=w, bias=b, compute_kernel_config=self.ck_hifi4)

    def _attention(self, x_norm, blk):
        """x_norm: [1,1,seq_pad,HIDDEN] -> attn output [1,1,seq_pad,HIDDEN]."""
        # fused qkv
        xqkv = ttnn.linear(
            x_norm,
            blk["wqkv"],
            bias=blk["wqkv_b"],
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,seq_pad, 3*N_HEADS*PAD_HEAD_DIM]

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=N_HEADS,
            num_kv_heads=N_HEADS,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # each [1, N_HEADS, seq_pad, PAD_HEAD_DIM]
        ttnn.deallocate(xqkv)

        # rotary embeddings (interleaved convention). rotary_embedding_llama requires bf16
        # inputs; cos/sin are bf16. SDPA below runs in the model's activation dtype.
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)
        q = ttnn.experimental.rotary_embedding_llama(
            q, self.rope_cos, self.rope_sin, self.rope_trans, is_decode_mode=False
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, self.rope_cos, self.rope_sin, self.rope_trans, is_decode_mode=False
        )

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.attn_mask,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.ck_sdpa,
        )  # [1, N_HEADS, seq_pad, PAD_HEAD_DIM]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # [1,1,seq_pad, N_HEADS*PAD_HEAD_DIM]
        out = ttnn.linear(
            attn,
            blk["wo"],
            bias=blk["wo_b"],
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn)
        return out

    def _mlp(self, x_norm, blk):
        h = ttnn.linear(
            x_norm,
            blk["fc0_w"],
            bias=blk["fc0_b"],
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = ttnn.gelu(h)  # tanh-approx GELU (matches PytorchGELUTanh)
        out = ttnn.linear(
            h,
            blk["fc1_w"],
            bias=blk["fc1_b"],
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return out

    def _block(self, x, blk):
        n0 = self._layer_norm(x, blk["norm0_w"], blk["norm0_b"])
        attn = self._attention(n0, blk)
        ttnn.deallocate(n0)
        x = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        n1 = self._layer_norm(x, blk["norm1_w"], blk["norm1_b"])
        mlp = self._mlp(n1, blk)
        ttnn.deallocate(n1)
        x = ttnn.add(x, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp)
        return x

    # ------------------------------------------------------------------ #
    def patch_embed(self, pixel_values):
        """pixel_values torch [L,3,14,14] -> ttnn [1,1,seq_pad,HIDDEN] (real rows then padding)."""
        L = pixel_values.shape[0]
        assert L == self.L, f"pixel rows {L} != grid L {self.L}"
        pix_flat = pixel_values.float().reshape(L, -1)  # [L,588] C-order (c,kh,kw)
        seq_pad = self._seq_pad(L)
        if seq_pad > L:
            pix_flat = F.pad(pix_flat, (0, 0, 0, seq_pad - L))
        x = self._to_dev(pix_flat.reshape(1, 1, seq_pad, -1))  # [1,1,seq_pad,588]
        x = ttnn.linear(
            x,
            self.proj_w,
            bias=self.proj_b,
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,seq_pad,HIDDEN]
        # add pos_emb (only over real L rows)
        if seq_pad > L:
            pe = ttnn.pad(self.pos_emb, [(0, 0), (0, 0), (0, seq_pad - L), (0, 0)], value=0.0)
        else:
            pe = self.pos_emb
        x = ttnn.add(x, pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    @staticmethod
    def _seq_pad(L):
        return int(math.ceil(L / 128) * 128)

    def encoder(self, x):
        for blk in self.blocks:
            x = self._block(x, blk)
        x = self._layer_norm(x, self.final_ln_w, self.final_ln_b)
        return x

    def patch_merger(self, x_torch):
        """Host-side 2x2 spatial merge (matches modeling_vit.patch_merger), returns [L/4, 4608].

        Done on host between encoder and mlp1 because the merge permute over the (h,w)
        grid is a pure layout reshuffle; doing it on host keeps the device path exact
        and avoids a tilized reshape hang. (Inference-time host work limited to a reshape.)
        """
        h, w = self.grid_hw
        kh, kw = MERGE
        nh, nw = h // kh, w // kw
        seq = x_torch[: self.L].reshape(nh, kh, nw, kw, HIDDEN)
        seq = seq.permute(0, 2, 1, 3, 4).contiguous().reshape(nh * nw, kh * kw * HIDDEN)
        return seq  # [L/4, 4608]

    def mlp1(self, x):
        """x ttnn [1,1,Nmerged,4608] -> [1,1,Nmerged,2048]."""
        x = ttnn.layer_norm(
            x, epsilon=LN_EPS, weight=self.mlp1_ln_w, bias=self.mlp1_ln_b, compute_kernel_config=self.ck_hifi4
        )
        x = ttnn.linear(
            x,
            self.mlp1_w1,
            bias=self.mlp1_b1,
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.gelu(x)
        x = ttnn.linear(
            x,
            self.mlp1_w2,
            bias=self.mlp1_b2,
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x

    # ------------------------------------------------------------------ #
    def forward(self, pixel_values, return_intermediates=False):
        """pixel_values torch [L,3,14,14] -> vit_proj ttnn [Nmerged, 2048].

        If return_intermediates, also return dict of host tensors for incremental PCC.
        """
        L = self.L
        seq_pad = self._seq_pad(L)

        x = self.patch_embed(pixel_values)  # [1,1,seq_pad,HIDDEN]
        inter = {}
        if return_intermediates:
            inter["patch_embed"] = ttnn.to_torch(x)[0, 0, :L].float()

        x = self.encoder(x)  # [1,1,seq_pad,HIDDEN]
        enc_torch = ttnn.to_torch(x)[0, 0].float()  # [seq_pad,HIDDEN]
        ttnn.deallocate(x)
        if return_intermediates:
            inter["encoder_out"] = enc_torch[:L]

        merged = self.patch_merger(enc_torch)  # [L/4, 4608]
        nmerged = merged.shape[0]
        merged_pad = self._seq_pad(nmerged)
        if merged_pad > nmerged:
            merged = F.pad(merged, (0, 0, 0, merged_pad - nmerged))
        xm = self._to_dev(merged.reshape(1, 1, merged_pad, MLP1_IN))

        proj = self.mlp1(xm)  # [1,1,merged_pad,2048]
        ttnn.deallocate(xm)
        proj_torch = ttnn.to_torch(proj)[0, 0, :nmerged].float()  # [Nmerged,2048]
        ttnn.deallocate(proj)

        if return_intermediates:
            return proj_torch, inter
        return proj_torch
