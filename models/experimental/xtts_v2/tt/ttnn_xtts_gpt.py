# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the XTTS-v2 GPT transformer core (Block 3).

Reference: models/experimental/xtts_v2/reference/xtts_gpt_ref.py
Architecture (HF GPT2, 30 blocks, causal, wpe nulled) + XTTS final_norm:

    inputs_embeds [1,S,1024]
      -> for each of 30 blocks:
           h = x + attn(ln_1(x))          # causal MHA, 16 heads, head_dim 64
           x = h + mlp(ln_2(h))           # c_fc(1024->4096) -> gelu_new -> c_proj(4096->1024)
      -> ln_f(x)                          # GPT2's final LayerNorm
      -> final_norm(x)                    # XTTS's extra LayerNorm
      = latents [1,S,1024]

GPT2 Conv1D weights are stored [in, out], which matches ttnn.linear's x[.,in]@W[in,out]
convention directly (no transpose).

Target: PCC > 0.9999 vs the CPU reference on the golden input.
"""

from dataclasses import dataclass

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gpt_core_state


@dataclass
class TTNNGPTConfig:
    n_embd: int = 1024
    n_layer: int = 30
    n_head: int = 16
    n_inner: int = 4096
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _compute_config(math_fidelity=ttnn.MathFidelity.HiFi4):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def preprocess_gpt_parameters(device, ckpt_path=None, dtype=ttnn.bfloat16):
    """Load the transformer-core weights from the XTTS checkpoint into TTNN tensors."""
    core = load_gpt_core_state(ckpt_path) if ckpt_path else load_gpt_core_state()
    cfg = TTNNGPTConfig()

    def lin(w, b):
        return {
            "weight": ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
            "bias": ttnn.from_torch(b.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        }

    def norm(w, b):
        return {
            "weight": ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
            "bias": ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        }

    params = {"blocks": []}
    for i in range(cfg.n_layer):
        p = f"h.{i}."
        params["blocks"].append(
            {
                "ln_1": norm(core[p + "ln_1.weight"], core[p + "ln_1.bias"]),
                "c_attn": lin(core[p + "attn.c_attn.weight"], core[p + "attn.c_attn.bias"]),
                "attn_proj": lin(core[p + "attn.c_proj.weight"], core[p + "attn.c_proj.bias"]),
                "ln_2": norm(core[p + "ln_2.weight"], core[p + "ln_2.bias"]),
                "c_fc": lin(core[p + "mlp.c_fc.weight"], core[p + "mlp.c_fc.bias"]),
                "mlp_proj": lin(core[p + "mlp.c_proj.weight"], core[p + "mlp.c_proj.bias"]),
            }
        )
    params["ln_f"] = norm(core["ln_f.weight"], core["ln_f.bias"])
    params["final_norm"] = norm(core["final_norm.weight"], core["final_norm.bias"])
    return params


class TTNNGPTCore:
    def __init__(
        self,
        device,
        parameters,
        config: TTNNGPTConfig = None,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activation_dtype=ttnn.bfloat16,
        attention="sdpa",
    ):
        self.device = device
        self.params = parameters
        self.config = config or TTNNGPTConfig()
        self.compute_kernel_config = _compute_config(math_fidelity)
        # SDPA requires bf16 q/k/v; the rest of the graph may run in a higher-precision
        # activation dtype (e.g. float32) for better PCC over 30 residual layers.
        self.activation_dtype = activation_dtype
        # "sdpa": flash-attention (bf16 q/k/v only). "manual": matmul+softmax, runs in
        # activation_dtype (fp32-capable) — needed to clear PCC>0.9999 over 30 layers.
        self.attention = attention
        self.scale = 1.0 / (self.config.head_dim**0.5)
        self._causal_mask = {}

    def _get_causal_mask(self, S):
        if S not in self._causal_mask:
            m = torch.zeros(1, 1, S, S)
            m.masked_fill_(torch.triu(torch.ones(S, S), diagonal=1).bool(), -1e9)
            self._causal_mask[S] = ttnn.from_torch(
                m, dtype=self.activation_dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._causal_mask[S]

    def _layer_norm(self, x, p):
        return ttnn.layer_norm(
            x,
            weight=p["weight"],
            bias=p["bias"],
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _linear(self, x, p):
        return ttnn.linear(
            x,
            p["weight"],
            bias=p["bias"],
            compute_kernel_config=self.compute_kernel_config,
        )

    def _attn(self, x, block):
        cfg = self.config
        B, S, _ = x.shape
        qkv = self._linear(x, block["c_attn"])  # [1,S,3072]

        q = qkv[:, :, 0 : cfg.n_embd]
        k = qkv[:, :, cfg.n_embd : 2 * cfg.n_embd]
        v = qkv[:, :, 2 * cfg.n_embd : 3 * cfg.n_embd]
        ttnn.deallocate(qkv)

        def to_heads(t):
            t = ttnn.reshape(t, (B, S, cfg.n_head, cfg.head_dim))
            return ttnn.permute(t, (0, 2, 1, 3))  # [B, nh, S, dh]

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        if self.attention == "manual":
            # Full-precision attention in activation_dtype (fp32-capable).
            kt = ttnn.permute(k, (0, 1, 3, 2))  # [B, nh, dh, S]
            scores = ttnn.matmul(q, kt, compute_kernel_config=self.compute_kernel_config)
            scores = ttnn.multiply(scores, self.scale)
            scores = ttnn.add(scores, self._get_causal_mask(S))
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
            attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(kt)
            ttnn.deallocate(scores)
            ttnn.deallocate(probs)
        else:
            # SDPA (flash-attention) only accepts bf16/bfloat8/bfloat4 q/k/v.
            if self.activation_dtype != ttnn.bfloat16:
                q = ttnn.typecast(q, ttnn.bfloat16)
                k = ttnn.typecast(k, ttnn.bfloat16)
                v = ttnn.typecast(v, ttnn.bfloat16)
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config,
            )
            if self.activation_dtype != ttnn.bfloat16:
                attn = ttnn.typecast(attn, self.activation_dtype)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.permute(attn, (0, 2, 1, 3))  # [B, S, nh, dh]
        attn = ttnn.reshape(attn, (B, S, cfg.n_embd))
        out = self._linear(attn, block["attn_proj"])
        ttnn.deallocate(attn)
        return out

    def _mlp(self, x, block):
        h = self._linear(x, block["c_fc"])
        h = ttnn.gelu(h, variant=ttnn.GeluVariant.Tanh)  # gelu_new
        h = self._linear(h, block["mlp_proj"])
        return h

    def __call__(self, inputs_embeds):
        x = inputs_embeds
        for block in self.params["blocks"]:
            x = ttnn.add(x, self._attn(self._layer_norm(x, block["ln_1"]), block))
            x = ttnn.add(x, self._mlp(self._layer_norm(x, block["ln_2"]), block))
        x = self._layer_norm(x, self.params["ln_f"])
        x = self._layer_norm(x, self.params["final_norm"])
        return x
