# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `decoder_layer` (Qwen3DecoderLayer) for FLUX.2-klein-9B's text encoder.

One pre-norm transformer block:

    h = x + self_attn(input_layernorm(x))
    y = h + mlp(post_attention_layernorm(h))

Tensor-parallel scheme (Megatron-standard, derived from
`models/tt_transformers/tt/decoder.py` + `.../mlp.py`):

  * The RESIDUAL STREAM stays REPLICATED -- every chip holds the whole 4096-wide
    hidden state. That is the invariant the whole block is built around: both
    sublayers take a full hidden vector in and put a full hidden vector back, so
    the two residual adds are plain local elementwise adds with no resharding.
  * `self_attn` reuses the TP attention that already graduated at TP=8 (q/k/v
    column-parallel over the head axis, o_proj row-parallel + all_reduce). It is
    imported rather than copied so the block cannot drift from the scheme
    gathered-PCC already validated.
  * The MLP is the textbook column-then-row pair: `gate_proj` and `up_proj` are
    COLUMN-parallel (their outputs feed SiLU and the elementwise gate, both
    per-element, so a chip can do its own 1536-wide slice with no collective),
    and `down_proj` is ROW-parallel -- it reduces the intermediate axis back to
    the model dim, so each chip owns the rows matching its own slice, emits a
    PARTIAL sum, and one `all_reduce` makes the result whole again.
    12288 / 8 = 1536 = 48 tiles, so the split is exact and tile-aligned.
  * Both RMSNorm gammas are per-element over the full model dim and stay
    REPLICATED, as the principles require -- and they can be, precisely because
    the residual stream is replicated, so every chip normalizes the same vector
    and gets the same answer.

Only one collective per sublayer, both of them all_reduces at the point where a
row-parallel projection returns to the model dim. The math is unchanged from the
torch reference -- only placement differs -- so the gathered output still matches
the single-device golden.
"""
from __future__ import annotations

import torch

import ttnn

from .attention import TtAttention
from .m_l_p import TtMLP


class TtDecoderLayer:
    def __init__(
        self,
        mesh_device,
        attention,
        mlp,
        input_norm_gamma,
        post_attn_norm_gamma,
        hidden_size,
        norm_eps,
        num_devices,
    ) -> None:
        self.mesh_device = mesh_device
        self.attention = attention
        # The feed-forward sublayer IS the graduated `m_l_p` / `mlp` component, so it
        # is that body rather than a second inline copy of the same column-then-row
        # scheme -- which is exactly the drift this file's docstring warns about.
        self.mlp = mlp
        self.input_norm_gamma = input_norm_gamma
        self.post_attn_norm_gamma = post_attn_norm_gamma
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.num_devices = num_devices
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_layer stub needs the torch module to source its weights")

        num_devices = _num_devices(device)
        hidden_size = torch_module.mlp.hidden_size
        intermediate_size = torch_module.mlp.intermediate_size
        if intermediate_size % (ttnn.TILE_SIZE * num_devices):
            raise RuntimeError(
                f"MLP TP needs intermediate_size divisible by TILE_SIZE*devices: "
                f"intermediate_size={intermediate_size}, devices={num_devices}"
            )

        # Both sublayers bring their own (already gathered-PCC-validated) TP scheme:
        # attention is column-parallel q/k/v + row-parallel o_proj, and the MLP is
        # column-parallel gate/up + row-parallel down. Both are the graduated bodies.
        attention = TtAttention.build(device, torch_module.self_attn)
        mlp = TtMLP.build(device, torch_module.mlp)

        return cls(
            mesh_device=device,
            attention=attention,
            mlp=mlp,
            input_norm_gamma=_norm_gamma(torch_module.input_layernorm.weight, hidden_size, device, num_devices),
            post_attn_norm_gamma=_norm_gamma(
                torch_module.post_attention_layernorm.weight, hidden_size, device, num_devices
            ),
            hidden_size=hidden_size,
            norm_eps=torch_module.input_layernorm.variance_epsilon,
            num_devices=num_devices,
        )

    # -------------------------------------------------------------- forward

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        is_causal=False,
        **kwargs,
    ):
        in_shape = list(hidden_states.shape)
        seq_len = int(in_shape[-2])
        # Product over every leading dim, so [B, S, H], [B, 1, S, H] and the decode
        # step's [1, 1, B, H] all fold without changing the element count.
        batch = 1
        for d in in_shape[:-2]:
            batch *= int(d)

        # Carry the block in [batch, 1, seq, hidden]: that is the layout TtAttention
        # reads its batch/seq from, and it keeps the residual adds plain elementwise.
        x = ttnn.reshape(hidden_states, (batch, 1, seq_len, self.hidden_size))

        # ---- attention sublayer (replicated in, replicated out).
        attn_in = ttnn.rms_norm(
            x,
            epsilon=self.norm_eps,
            weight=self.input_norm_gamma,
            compute_kernel_config=self.compute_kernel_config,
        )
        attn_out = self.attention(
            attn_in,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cur_pos=cur_pos,
            mode=mode,
            is_causal=is_causal,
        )
        attn_out = ttnn.reshape(attn_out, (batch, 1, seq_len, self.hidden_size))
        h = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)

        # ---- MLP sublayer: the graduated `m_l_p` body. It does the column-parallel
        # gate/up, the elementwise SiLU gate, the row-parallel down and the single
        # all_reduce that restores the replicated residual stream this block rests on.
        ff_in = ttnn.rms_norm(
            h,
            epsilon=self.norm_eps,
            weight=self.post_attn_norm_gamma,
            compute_kernel_config=self.compute_kernel_config,
        )
        ff_out = self.mlp(ff_in)
        ttnn.deallocate(ff_in)

        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)

        return ttnn.reshape(out, tuple(in_shape))


# ------------------------------------------------------------------ helpers


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _replicate_mapper(device, num_devices):
    if num_devices <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _norm_gamma(weight, dim, device, num_devices):
    """ttnn.rms_norm wants gamma as [1, 1, dim // TILE, TILE] in ROW_MAJOR
    (see models/common/rmsnorm.py). Per-element over the full model dim, so
    it is REPLICATED on every chip."""
    return ttnn.from_torch(
        weight.detach().to(torch.bfloat16).reshape(1, 1, dim // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device, num_devices),
    )


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def decoder_layer(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)
