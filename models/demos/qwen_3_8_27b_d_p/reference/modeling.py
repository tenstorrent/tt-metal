# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained torch reference for the Qwen3.5 text tower (Qwen3.8-27B).

**Purity contract** (see ``deepseek_v3_d_p/reference/kda/README.md``): this module imports torch and
nothing else — no ttnn, no device code, no mesh fixtures, no profilers. It is the semantic oracle
every PCC test measures against.

**Provenance.** Trimmed from ``transformers`` 5.12.1
``src/transformers/models/qwen3_5/modeling_qwen3_5.py`` (the generated, flattened file, itself
composed from ``qwen3_next`` + ``qwen3_vl``). Upstream line numbers are recorded per class below.
Dropped: the vision tower, the MTP head, generation/cache plumbing, the FLA/causal-conv1d fast
paths (the ``torch_*`` fallbacks upstream selects when those libraries are absent ARE the
reference), and the multimodal 3-D position-id machinery (text-only prefill).

**Dtype convention** (recipe section 4): parameters and activations are ``torch.float16``.
Where upstream forces ``float32`` inside an op — RMSNorm's mean, the gate's ``exp``/``softplus``,
the chunked delta rule's whole body — that upcast is kept, because it is part of the semantics
rather than a shape-tuned choice. Every tensor that crosses a module boundary is fp16.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import LINEAR_ATTENTION, Qwen35TextConfig

REF_DTYPE = torch.float16

# The chunk width the FLA chunked delta rule factorizes over. Not a config field upstream either —
# it is the kernel's tiling constant (upstream default `chunk_size=64`), and the device op's too.
DELTA_CHUNK_SIZE = 64

# --------------------------------------------------------------------------------------
# fp16 storage, fp32 accumulation
# --------------------------------------------------------------------------------------
# Every tensor the reference holds and every golden it writes is fp16 (recipe section 4). What
# runs *inside* a matmul is a separate question, and on this host it is a decisive one: torch has
# no vectorised fp16 GEMM for x86, so a literal fp16 ``mm`` falls back to a scalar loop measured
# here at **0.001 TFLOP/s against 3.58 for bf16** — 3500x slower, which turns a ~30-minute
# full-depth golden trace into ~70 hours.
#
# So the projections accumulate in fp32 and round back to fp16 at every module boundary. That is
# not a compromise on the oracle's fidelity, it is closer to it: a real fp16 GEMM (GPU tensor
# cores, and the Tenstorrent device under HiFi4 with fp32_dest_acc_en) accumulates in fp32 too.
# The scalar fp16-accumulating fallback is the odd one out.
#
# ``fp16_accumulation()`` restores the literal behaviour; the D1 tests use it so the vendored
# reference is checked against upstream HF op-for-op, and ``test_accumulation_modes_agree``
# measures what the difference is worth.
_FP32_ACCUM = os.environ.get("QWEN35_REF_FP16_ACCUM") != "1"


@contextlib.contextmanager
def fp16_accumulation() -> Iterator[None]:
    """Run the reference with literal fp16 accumulation — upstream-identical, and very slow."""
    global _FP32_ACCUM
    previous = _FP32_ACCUM
    _FP32_ACCUM = False
    try:
        yield
    finally:
        _FP32_ACCUM = previous


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``F.linear`` with fp32 accumulation, returning ``x``'s dtype. See the note above."""
    if not _FP32_ACCUM:
        return F.linear(x, weight, bias)
    out = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
    return out.to(x.dtype)


def conv1d_fp32(x: torch.Tensor, weight: torch.Tensor, *, groups: int, padding: int) -> torch.Tensor:
    """Depthwise ``F.conv1d`` with fp32 accumulation, returning ``x``'s dtype."""
    if not _FP32_ACCUM:
        return F.conv1d(x, weight, None, padding=padding, groups=groups)
    out = F.conv1d(x.float(), weight.float(), None, padding=padding, groups=groups)
    return out.to(x.dtype)


# ======================================================================================
# Norms
# ======================================================================================


class Qwen35RMSNorm(nn.Module):
    """upstream modeling_qwen3_5.py:736 — Gemma-style ``(1 + weight)`` fold, mean in fp32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=REF_DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """upstream modeling_qwen3_5.py:187 — the Gated DeltaNet output norm.

    Two things separate it from :class:`Qwen35RMSNorm` and both matter: the gain is applied
    **plain** (no ``1 +`` fold), and the gate is ``silu(z)``, not ``sigmoid(z)``. Using the
    sigmoid form (what ``ttnn.experimental.kda.sigmoid_gated_rms_norm`` implements) is a silent
    ~0.9 PCC, not an error.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=REF_DTYPE))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


# ======================================================================================
# RoPE — partial rotary, interleaved mrope
# ======================================================================================


def compute_inv_freq(cfg: Qwen35TextConfig) -> torch.Tensor:
    """upstream modeling_qwen3_5.py:116. ``dim`` is the PARTIAL width (head_dim * factor = 64)."""
    dim = cfg.rotary_dim
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: tuple[int, int, int]) -> torch.Tensor:
    """upstream modeling_qwen3_5.py:169. ``freqs`` is [3, bs, seq, rotary_dim/2]; returns [bs, seq, ...].

    Reorganises T/H/W frequency blocks from chunked ``[TTT..HHH..WWW]`` to interleaved
    ``[THWTHW...TT]``. For a **text-only** prompt the three position-id rows are identical, so
    every lane it copies is already equal and this reduces to ``freqs[0]`` — but the reference runs
    the real thing so nothing has to be taken on trust (``test_rope_vs_ref`` pins the equivalence).
    """
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim][..., idx]
    return freqs_t


class Qwen35RotaryEmbedding(nn.Module):
    """upstream modeling_qwen3_5.py:95, text-only (``position_ids`` [3, bs, seq])."""

    def __init__(self, cfg: Qwen35TextConfig) -> None:
        super().__init__()
        assert cfg.rope_type == "default", f"only default rope is implemented, got {cfg.rope_type}"
        self.mrope_section = cfg.mrope_section
        self.register_buffer("inv_freq", compute_inv_freq(cfg), persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [seq] or [bs, seq] or [3, bs, seq]. Returns fp16 cos/sin [bs, seq, rotary_dim]."""
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq @ pos).transpose(2, 3)  # [3, bs, seq, rotary_dim/2]
        freqs = apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)  # [bs, seq, rotary_dim]
        return emb.cos().to(REF_DTYPE), emb.sin().to(REF_DTYPE)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """upstream modeling_qwen3_5.py:562 — half-split (NOT pairwise-interleaved) rotation."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """upstream modeling_qwen3_5.py:570. Rotates only ``cos.shape[-1]`` dims; the rest pass through."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    def _rot(x: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)

    return _rot(q), _rot(k)


# ======================================================================================
# Dense MLP
# ======================================================================================


class Qwen35MLP(nn.Module):
    """upstream modeling_qwen3_5.py:720 — plain SiLU SwiGLU, no bias, no clamp, no alpha."""

    def __init__(self, cfg: Qwen35TextConfig) -> None:
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False, dtype=REF_DTYPE)
        self.up_proj = nn.Linear(h, i, bias=False, dtype=REF_DTYPE)
        self.down_proj = nn.Linear(i, h, bias=False, dtype=REF_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = linear(x, self.gate_proj.weight)
        up = linear(x, self.up_proj.weight)
        return linear(F.silu(gate) * up, self.down_proj.weight)


# ======================================================================================
# Full attention (GQA + output gate + per-head QK-norm + partial RoPE)
# ======================================================================================


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """upstream modeling_qwen3_5.py:608."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


@dataclass
class AttentionCapture:
    """Post-RoPE K and raw V for one full-attention layer, ``[1, num_kv_heads, seq, head_dim]``."""

    key: torch.Tensor
    value: torch.Tensor


class Qwen35Attention(nn.Module):
    """upstream modeling_qwen3_5.py:645 — eager causal GQA.

    The output gate is the piece a GQA port is most likely to lose: ``q_proj`` is **twice** the
    usual width and the second half of each head is a sigmoid gate applied to the attention output
    *before* ``o_proj``. Reading ``q_proj.weight.shape[0]`` as ``num_heads * head_dim`` silently
    halves the head count instead of failing.
    """

    def __init__(self, cfg: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_key_value_groups = cfg.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        bias = cfg.attention_bias
        q_out = self.num_heads * self.head_dim * (2 if cfg.attn_output_gate else 1)
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=bias, dtype=REF_DTYPE)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, dtype=REF_DTYPE)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, dtype=REF_DTYPE)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=bias, dtype=REF_DTYPE)
        self.q_norm = Qwen35RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def project_raw(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (q, k, v, gate) **before** QK-norm, head-split as ``[b, n, s, d]``.

        Split out from :meth:`project` so the device's fused-projection test measures the fusion
        and the head permutation alone. QK-norm is close to a per-row rescale, so folding it into
        that comparison keeps PCC high even when the norm is missing entirely.
        """
        b, s, _ = hidden_states.shape
        if self.cfg.attn_output_gate:
            q, gate = torch.chunk(
                linear(hidden_states, self.q_proj.weight, self.q_proj.bias).view(b, s, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
            gate = gate.reshape(b, s, -1)
        else:
            q = linear(hidden_states, self.q_proj.weight, self.q_proj.bias).view(b, s, -1, self.head_dim)
            gate = None
        q = q.view(b, s, -1, self.head_dim).transpose(1, 2)
        k = linear(hidden_states, self.k_proj.weight, self.k_proj.bias).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = linear(hidden_states, self.v_proj.weight, self.v_proj.bias).view(b, s, -1, self.head_dim).transpose(1, 2)
        return q, k, v, gate

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (q [b,nh,s,d], k [b,nkv,s,d], v [b,nkv,s,d], gate [b,s,nh*d]), pre-RoPE, post-QK-norm."""
        q, k, v, gate = self.project_raw(hidden_states)
        return self.q_norm(q), self.k_norm(k), v, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_kv: Optional[AttentionCapture] = None,
    ) -> tuple[torch.Tensor, AttentionCapture]:
        """One-shot or chunked. ``past_kv`` is the accumulated prefix; the returned capture is the
        prefix **including** this chunk, which is what the golden trace stores."""
        b, s, _ = hidden_states.shape
        q, k, v, gate = self.project(hidden_states)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv.key, k], dim=2)
            v = torch.cat([past_kv.value, v], dim=2)
        capture = AttentionCapture(key=k, value=v)

        kv_len = k.shape[2]
        # Causal over the FULL prefix, bottom-right aligned: query token i sits at absolute
        # position ``kv_len - s + i``. ``is_causal=True`` would align top-left and silently let a
        # chunk attend the wrong window, so the mask is always explicit.
        causal = torch.ones(s, kv_len, dtype=torch.bool).tril(diagonal=kv_len - s)
        if _FP32_ACCUM:
            # Exact (flash softmax is not an approximation) and ~4x faster than materialising the
            # [b, nh, s, kv] score matrix — which at isl 5120 is 2.5 GB in fp32, and whose
            # attn @ v in fp16 hits the scalar-GEMM fallback described at the top of this file.
            out = F.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=causal,
                scale=self.scaling,
                enable_gqa=self.num_key_value_groups > 1,
            ).to(q.dtype)
        else:
            key_states = repeat_kv(k, self.num_key_value_groups)
            value_states = repeat_kv(v, self.num_key_value_groups)
            attn = torch.matmul(q.float(), key_states.float().transpose(2, 3)) * self.scaling
            attn = attn.masked_fill(~causal, float("-inf"))
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, value_states)
        out = out.transpose(1, 2).reshape(b, s, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return linear(out, self.o_proj.weight, self.o_proj.bias), capture


# ======================================================================================
# Gated DeltaNet (linear attention)
# ======================================================================================


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """upstream modeling_qwen3_5.py:240 — FLA's L2 norm. The epsilon is INSIDE the rsqrt, added to
    the sum of squares, not to the norm; the usual ``x / (||x|| + eps)`` spelling is a different
    function."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = DELTA_CHUNK_SIZE,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """upstream modeling_qwen3_5.py:246, verbatim maths.

    Shapes: q/k ``[b, t, h, dk]``, v ``[b, t, h, dv]``, g/beta ``[b, t, h]``, state ``[b, h, dk, dv]``.
    The whole body runs in fp32 (upstream does too); the output is cast back to ``query.dtype``.

    What the chunking actually is: the recurrence ``S_t = g_t S_{t-1} + k_t (v_t - g_t S_{t-1}^T k_t) beta_t``
    is factorized per 64-token chunk into a WY representation — ``attn`` is the inverse of a unit
    lower-triangular matrix built by forward substitution, and it turns the sequential in-chunk
    delta updates into two matmuls. Only the chunk-to-chunk carry stays sequential. The device op
    (``ttnn.transformer.chunk_gated_delta_rule``) implements exactly this factorization, so the two
    agree to fp32 rounding rather than approximately.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@dataclass
class GdnCapture:
    """A Gated DeltaNet layer's whole carried state — the GDN analogue of a KV cache entry.

    ``conv_state`` is the last ``kernel-1`` pre-conv tokens ``[b, conv_dim, kernel-1]``;
    ``recurrent_state`` is the delta-rule matrix state ``[b, num_v_heads, head_k_dim, head_v_dim]``.
    A chunked prefill reproduces a one-shot prefill only if BOTH are carried.
    """

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class Qwen35GatedDeltaNet(nn.Module):
    """upstream modeling_qwen3_5.py:371 (Qwen3.5's four-way split projection, not Qwen3-Next's
    fused ``in_proj_qkvz`` / ``in_proj_ba``).

    Per token: ``in_proj_qkv`` -> 4-tap causal depthwise conv -> SiLU -> split q/k/v ->
    per-head L2-normalised chunked gated delta rule -> silu-gated RMSNorm against ``in_proj_z`` ->
    ``out_proj``. ``in_proj_b``/``in_proj_a`` produce the per-head ``beta`` and log-decay ``g``.
    """

    def __init__(self, cfg: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = cfg.gdn_key_dim
        self.value_dim = cfg.gdn_value_dim
        self.conv_dim = cfg.gdn_conv_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.num_v_groups = cfg.gdn_num_value_groups

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=REF_DTYPE,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=REF_DTYPE))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=REF_DTYPE))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False, dtype=REF_DTYPE)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False, dtype=REF_DTYPE)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False, dtype=REF_DTYPE)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, dtype=REF_DTYPE)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, dtype=REF_DTYPE)

    # --- the pieces, exposed so each gets its own PCC test --------------------------------
    def causal_conv(self, mixed_qkv_bct: torch.Tensor, conv_state: Optional[torch.Tensor]) -> torch.Tensor:
        """``mixed_qkv_bct`` [b, conv_dim, t] -> SiLU(conv) [b, conv_dim, t], channel-wise causal.

        With ``conv_state`` (the previous chunk's last 3 tokens) the history is prepended and the
        result trimmed back to ``t``; without it, ``padding=kernel-1`` zero-pads on the left, which
        is the one-shot case. Upstream does both in the same branch.
        """
        t = mixed_qkv_bct.shape[-1]
        x = mixed_qkv_bct if conv_state is None else torch.cat([conv_state, mixed_qkv_bct], dim=-1)
        conv = conv1d_fp32(x, self.conv1d.weight, groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        out = F.silu(conv[:, :, : x.shape[-1]])
        return out[:, :, -t:]

    def next_conv_state(self, mixed_qkv_bct: torch.Tensor, conv_state: Optional[torch.Tensor]) -> torch.Tensor:
        """The conv history to carry into the next chunk: the last ``kernel-1`` PRE-conv tokens.

        Upstream spells this ``F.pad(x, (kernel - x.shape[-1], 0))`` on the history-prepended input,
        which is a left-pad for a short chunk and a left-TRIM otherwise — i.e. the last ``kernel``
        columns. It then keeps ``kernel-1`` of them (the cache's own width). Chunks here are always
        far longer than 4, so the trim is the only branch this bring-up takes.
        """
        x = mixed_qkv_bct if conv_state is None else torch.cat([conv_state, mixed_qkv_bct], dim=-1)
        keep = self.conv_kernel_size - 1
        if x.shape[-1] >= keep:
            return x[:, :, -keep:].clone()
        return F.pad(x, (keep - x.shape[-1], 0))

    def gates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (beta [b, t, hv] fp16, g [b, t, hv] fp32 log-decay).

        ``g`` is built in fp32 on purpose: ``A_log`` held in fp16 and exponentiated in fp16 can
        reach ``-inf`` (upstream's own comment), which poisons the whole scan.
        """
        beta = linear(hidden_states, self.in_proj_b.weight).sigmoid()
        a = linear(hidden_states, self.in_proj_a.weight)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return beta, g

    def split_heads(self, conv_out_bct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[b, conv_dim, t] -> q/k [b, t, num_v_heads, head_k_dim] (GQA-expanded), v [b, t, hv, dv]."""
        b, _, t = conv_out_bct.shape
        mixed = conv_out_bct.transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(b, t, -1, self.head_k_dim)
        key = key.reshape(b, t, -1, self.head_k_dim)
        value = value.reshape(b, t, -1, self.head_v_dim)
        if self.num_v_groups > 1:
            query = query.repeat_interleave(self.num_v_groups, dim=2)
            key = key.repeat_interleave(self.num_v_groups, dim=2)
        return query, key, value

    def forward(
        self, hidden_states: torch.Tensor, state: Optional[GdnCapture] = None
    ) -> tuple[torch.Tensor, GdnCapture]:
        b, t, _ = hidden_states.shape
        conv_state = state.conv_state if state is not None else None
        recurrent_state = state.recurrent_state if state is not None else None

        mixed_qkv = linear(hidden_states, self.in_proj_qkv.weight).transpose(1, 2)  # [b, conv_dim, t]
        z = linear(hidden_states, self.in_proj_z.weight).reshape(b, t, -1, self.head_v_dim)
        beta, g = self.gates(hidden_states)

        new_conv_state = self.next_conv_state(mixed_qkv, conv_state)
        conv_out = self.causal_conv(mixed_qkv, conv_state)
        query, key, value = self.split_heads(conv_out)

        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(b, t, -1)
        return linear(core_attn_out, self.out_proj.weight), GdnCapture(new_conv_state, last_recurrent_state)


# ======================================================================================
# Decoder layer + model
# ======================================================================================


class Qwen35DecoderLayer(nn.Module):
    """upstream modeling_qwen3_5.py:756. Identical residual shape for both layer types; only the
    token mixer differs, which is why the TT side picks a mixer strategy once at construction
    instead of branching on ``layer_type`` inside ``forward``."""

    def __init__(self, cfg: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        self.layer_idx = layer_idx
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = Qwen35GatedDeltaNet(cfg, layer_idx)
        else:
            self.self_attn = Qwen35Attention(cfg, layer_idx)
        self.mlp = Qwen35MLP(cfg)
        self.input_layernorm = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: Optional[AttentionCapture | GdnCapture] = None,
    ) -> tuple[torch.Tensor, AttentionCapture | GdnCapture]:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == LINEAR_ATTENTION:
            h, new_state = self.linear_attn(h, state)
        else:
            h, new_state = self.self_attn(h, position_embeddings, state)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        return residual + h, new_state


class Qwen35TextModel(nn.Module):
    """The text tower: embedding, 64 hybrid layers, final norm, LM head.

    ``forward`` takes and returns the per-layer carried state, so a single call is a one-shot
    prefill and a loop over chunks is a chunked prefill — the property P2 grades.
    """

    def __init__(self, cfg: Qwen35TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=REF_DTYPE)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=REF_DTYPE)
        self.rotary_emb = Qwen35RotaryEmbedding(cfg)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        states: Optional[list] = None,
        skip_lm_head: bool = False,
        on_layer: Optional[Callable[[int, object, torch.Tensor], None]] = None,
    ) -> tuple[torch.Tensor, list]:
        """``on_layer(layer_idx, new_state, hidden_states)`` fires after each layer — the seam the
        golden-trace generator uses to stream one layer's capture to disk and drop it."""
        assert (input_ids is None) != (inputs_embeds is None), "pass exactly one of input_ids/inputs_embeds"
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        seq = h.shape[1]
        position_ids = torch.arange(start_pos, start_pos + seq, dtype=torch.long)[None, :]
        position_embeddings = self.rotary_emb(position_ids)

        states = list(states) if states is not None else [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, states[i] = layer(h, position_embeddings, states[i])
            if on_layer is not None:
                on_layer(i, states[i], h)

        h = self.norm(h)
        if skip_lm_head:
            return h, states
        return linear(h, self.lm_head.weight), states


def init_random_weights(model: nn.Module, seed: int = 0, std: float = 0.02) -> None:
    """Fill a reference with the SAME random weights both sides of a PCC test use.

    Norm gains are drawn around their HF init (``Qwen35RMSNorm`` is 1-centred via the ``1 + w``
    fold, so its parameter is 0-centred; the gated norm's is 1-centred), and ``A_log``/``dt_bias``
    follow ``Qwen3_5PreTrainedModel._init_weights`` — a uniform A in ``(0, 16]`` and a ones
    ``dt_bias``. Drawing ``A_log`` from a plain normal instead puts the decay in the wrong regime
    and makes the delta-rule scan look better-conditioned than it is.
    """
    gen = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            with torch.no_grad():
                module.weight.normal_(0.0, std, generator=gen)
            if isinstance(module, nn.Linear) and module.bias is not None:
                with torch.no_grad():
                    module.bias.zero_()
        elif isinstance(module, nn.Conv1d):
            with torch.no_grad():
                module.weight.normal_(0.0, 0.5, generator=gen)
        elif isinstance(module, Qwen35RMSNorm):
            with torch.no_grad():
                module.weight.normal_(0.0, 0.1, generator=gen)
        elif isinstance(module, Qwen35RMSNormGated):
            with torch.no_grad():
                module.weight.normal_(1.0, 0.1, generator=gen)
    for module in model.modules():
        if isinstance(module, Qwen35GatedDeltaNet):
            with torch.no_grad():
                a = torch.empty(module.num_v_heads).uniform_(0, 16, generator=gen).clamp_min(1e-4)
                module.A_log.copy_(a.log().to(REF_DTYPE))
                module.dt_bias.fill_(1.0)
