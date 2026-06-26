# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# UNVALIDATED — authored without Tenstorrent hardware.
#
# This module was written in AUTHORING-ONLY mode on a machine with NO ttnn /
# TT_METAL hardware available, so verify.py / pytest were NOT run. The code has
# been hand-traced for tensor shapes and every tt_dit primitive signature was
# confirmed against library source, but it has NOT executed on device. Treat all
# numerics as unverified until the deferred verify.py command passes on a TT box.
#
# Highest-risk areas for the pcc-debugger to watch (see test file header):
#   1. rotate-half MRoPE convention (cos/sin = cat(freqs, freqs); _rotate_half),
#      precomputed on host from the reference Ideogram4MRoPE.
#   2. block-diagonal segment-id attention mask fed as an additive bias to SDPA.
#   3. tanh-gated 4-branch AdaLN (scale_msa, gate_msa, scale_mlp, gate_mlp),
#      no shift terms — differs from the 6-branch ports.
#   4. the double-RMSNorm sandwich residual structure (norm1 pre-scale on input,
#      norm2 post-norm on the attn/ff output before the gate).
#   5. head_dim=256 / 18 heads (not mesh-friendly) — TP path is UNVALIDATED;
#      the validated target config is single-device (TP=1).
# =============================================================================

from __future__ import annotations

import torch

import ttnn

from ...layers.feedforward import FeedForward
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate

# Per-token role indicators (mirrors reference src/ideogram4/constants.py).
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

MATH_FIDELITY_HIFI2 = ttnn.MathFidelity.HiFi2
MATH_FIDELITY_HIFI4 = ttnn.MathFidelity.HiFi4


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    """Half-split rotate, matching reference _rotate_half: [-x2, x1].

    Reference (modeling_ideogram4.py): x1=x[..., :half], x2=x[..., half:],
    returns cat((-x2, x1)). NOT the interleaved alt_complex_rotate90 convention.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """q_embed = q * cos + rotate_half(q) * sin (reference _apply_rotary_pos_emb)."""
    return x * cos + _rotate_half(x) * sin


class Ideogram4TransformerBlock(Module):
    """Single-stream Ideogram 4.0 transformer block.

    Faithful port of reference `Ideogram4TransformerBlock`. Text and image tokens
    are a SINGLE unified sequence sharing one set of projections (no separate text
    branch, no separate modulation). Forward mirrors the reference exactly:

        mod = adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4)
        gate_* = tanh(gate_*); scale_* = 1 + scale_*

        attn_out = attention(attention_norm1(x) * scale_msa, ...)
        x = x + gate_msa * attention_norm2(attn_out)
        x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))

    Validated target config: single device (TP=1). The tensor-parallel path
    (tp_factor>1) is UNVALIDATED — 18 heads are not mesh-friendly (flag for
    hw-mapper), so the first-win config keeps TP=1.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adaln_dim: int,
        attention_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.adaln_dim = adaln_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.padding_config = padding_config

        tp_factor = parallel_config.tensor_parallel.factor if parallel_config is not None else 1
        assert tp_factor == 1, (
            "Ideogram4TransformerBlock validated path is TP=1 only; "
            "TP>1 is unvalidated (18 heads are not mesh-friendly). Flag for hw-mapper."
        )

        # --- attention projections (fused QKV, no bias; matches reference qkv/o) ---
        self.qkv = Linear(hidden_size, 3 * hidden_size, bias=False, mesh_device=mesh_device)
        self.o = Linear(hidden_size, hidden_size, bias=False, mesh_device=mesh_device)

        # QK-RMSNorm over head_dim (reference norm_q / norm_k = Ideogram4RMSNorm, eps=1e-5).
        self.norm_q = RMSNorm(
            embedding_dim=self.head_dim, norm_eps=attention_eps, bias=False, mesh_device=mesh_device
        )
        self.norm_k = RMSNorm(
            embedding_dim=self.head_dim, norm_eps=attention_eps, bias=False, mesh_device=mesh_device
        )

        # --- the four block RMSNorms (no affine bias; weight only) ---
        self.attention_norm1 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.attention_norm2 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.ffn_norm1 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.ffn_norm2 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)

        # --- SwiGLU MLP: w2(silu(w1(x)) * w3(x)) ---
        # FeedForward fuses w1 (gate) and w3 (value) into ff1 via the "swiglu"
        # activation (ff1 out doubled, ttnn.chunk + silu inside the linear).
        self.feed_forward = FeedForward(
            dim=hidden_size,
            dim_out=hidden_size,
            inner_dim=intermediate_size,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
        )

        # --- AdaLN: project adaln_input -> 4 * hidden_size (with bias) ---
        self.adaln_modulation = Linear(adaln_dim, 4 * hidden_size, bias=True, mesh_device=mesh_device)

        # SDPA config (fidelity recipe §4 — HiFi2, fp32 acc off; flip on if attn PCC suffers).
        device_grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(device_grid.x, device_grid.y),
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=MATH_FIDELITY_HIFI2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # flip True if attention PCC is the culprit
        )
        # Higher fidelity for the small AdaLN projection (matmul-class but stability matters).
        self.adaln_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Reference submodule names:
        #   attention.qkv / attention.o / attention.norm_q / attention.norm_k
        #   feed_forward.w1 / feed_forward.w2 / feed_forward.w3
        #   attention_norm1/2, ffn_norm1/2, adaln_modulation
        attn = pop_substate(state, "attention")
        if "qkv.weight" in attn:
            state["qkv.weight"] = attn["qkv.weight"]
        if "o.weight" in attn:
            state["o.weight"] = attn["o.weight"]
        if "norm_q.weight" in attn:
            state["norm_q.weight"] = attn["norm_q.weight"]
        if "norm_k.weight" in attn:
            state["norm_k.weight"] = attn["norm_k.weight"]

        # SwiGLU: reference is w2(silu(w1(x)) * w3(x)).
        # FeedForward.ff1 with "swiglu" computes: chunk(ff1_out,2) -> a, gate; a * silu(gate).
        # So a must be w3 (value) and gate must be w1 (silu'd). ff1.weight = [w3; w1] stacked
        # on the output dim, then transposed by Linear._prepare_torch_state.
        ff = pop_substate(state, "feed_forward")
        if "w1.weight" in ff and "w3.weight" in ff:
            # nn.Linear weight is [out, in]; stack along out so first half=value, second half=gate.
            w_value = ff["w3.weight"]  # value branch -> "a"
            w_gate = ff["w1.weight"]  # gate branch -> silu(gate)
            state["feed_forward.ff1.weight"] = torch.cat([w_value, w_gate], dim=0)
        if "w2.weight" in ff:
            state["feed_forward.ff2.weight"] = ff["w2.weight"]

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adaln_input: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Single-stream block forward.

        Args:
            x: [B, L, hidden_size] unified text+image sequence (replicated).
            cos, sin: [B, 1, L, head_dim] rotary tables (rotate-half convention),
                precomputed on host from Ideogram4MRoPE. Broadcast over heads.
            adaln_input: [B, 1, adaln_dim] (or [B, L, adaln_dim]) SiLU'd time cond.
            attn_mask: [B, 1, L, L] additive mask (0 = attend, -inf = block), built
                from segment ids. None => full attention.
        """
        batch_size, seq_len, _ = x.shape

        # --- AdaLN: 4-branch, tanh gates, (1 + scale). ---
        mod = self.adaln_modulation(adaln_input, compute_kernel_config=self.adaln_compute_kernel_config)
        scale_msa, gate_msa, scale_mlp, gate_mlp = ttnn.chunk(mod, 4, -1)
        gate_msa = ttnn.tanh(gate_msa, fast_and_approximate_mode=False)
        gate_mlp = ttnn.tanh(gate_mlp, fast_and_approximate_mode=False)
        scale_msa = scale_msa + 1.0
        scale_mlp = scale_mlp + 1.0

        # ----------------- attention sub-block -----------------
        attn_in = self.attention_norm1(x) * scale_msa
        attn_out = self._attention(attn_in, cos=cos, sin=sin, attn_mask=attn_mask)
        x = x + gate_msa * self.attention_norm2(attn_out)

        # ----------------- feed-forward sub-block -----------------
        ff_in = self.ffn_norm1(x) * scale_mlp
        ff_out = self.feed_forward(ff_in)
        x = x + gate_mlp * self.ffn_norm2(ff_out)

        return x

    def _attention(
        self,
        x: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)  # [B, L, 3*hidden_size]
        # Split into heads: ttnn.transformer.split_query_key_value_and_split_heads
        # consumes [B, L, 3*H*hd] (interleaved as q|k|v on the last dim, matching the
        # reference view(B,L,3,H,hd).unbind(2)) and returns [B, H, L, hd] each.
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            transpose_key=False,
            memory_config=qkv.memory_config(),
        )

        # QK-RMSNorm over head_dim (applied per-head, before RoPE — matches reference,
        # which norms q/k of shape [..., hd] then transposes then applies rope).
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )  # [B, H, L, hd]

        out = ttnn.transformer.concatenate_heads(out)  # [B, L, hidden_size]
        return self.o(out)


class Ideogram4Transformer(Module):
    """Full Ideogram 4.0 single-stream denoiser (34 blocks).

    Provided for completeness/structure; the bringup target validated here is the
    single block. The end-to-end model wires the input/LLM/time embeddings, MRoPE,
    the block stack, and the final layer. Several host-side preprocessing pieces
    (sinusoidal time embed, MRoPE table construction, masked token fusion) are
    expected to be precomputed on host and fed to the block stack, mirroring how
    the Mochi/Qwen ports precompute RoPE and timestep features on host.

    NOTE: only the block (Ideogram4TransformerBlock) is covered by the bringup
    test. This class is a thin scaffold and is itself UNVALIDATED.
    """

    def __init__(
        self,
        *,
        emb_dim: int = 4608,
        num_layers: int = 34,
        num_heads: int = 18,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        in_channels: int = 128,
        norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.mesh_device = mesh_device

        self.layers = ModuleList(
            Ideogram4TransformerBlock(
                hidden_size=emb_dim,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                norm_eps=norm_eps,
                adaln_dim=adaln_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        h: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adaln_input: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        for layer in self.layers:
            h = layer(h, cos=cos, sin=sin, adaln_input=adaln_input, attn_mask=attn_mask)
        return h
