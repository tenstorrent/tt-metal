# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the Mistral-Medium-3.5-128B (``ministral3``) decoder.

**torch only — no ttnn, no device code.** This is the thing every device module is measured
against, so it has to be runnable on a host with no Tenstorrent hardware attached.

Computation is in **bf16** (recipe §4): the reference, the goldens and the device all share one
convention, so a PCC gap is a real implementation difference and not a dtype artefact. The two
places HF itself promotes to fp32 — the RMSNorm variance and the attention softmax — are promoted
here as well, because the golden trace was produced by ``transformers`` 5.12.1 and matching its
accumulation is the point.

What the layer emits per call is fixed by the golden trace's ``metadata.json``:

* ``k_is_post_rope: True``  — K is cached **after** RoPE.
* ``v_is_raw: True``        — V is cached straight off ``v_proj``.
* ``k_layout: hf_half_split`` — K keeps HF's ``rotate_half`` pairing (``[x1 | x2]`` halves), *not*
  Meta's interleaved pairing. The device side caches Meta-interleaved, so every golden/device K
  comparison must permute one of them; :func:`hf_to_meta_head_perm` is that permutation and lives
  here so the reference owns the layout definition.

Architecture (from ``reference/config.json``, ``text_config``): 88 layers, hidden 12288, 96 Q heads
/ 8 KV heads (GQA group 12), head_dim 128, dense SwiGLU FFN with intermediate 28672, plain Mistral
RMSNorm (no ``1 + w`` fold), YaRN RoPE, no QK-norm, no biases, no sliding window, no MoE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig

REF_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------------------------
# Head layout
# ---------------------------------------------------------------------------------------------
def hf_to_meta_head_perm(head_dim: int, rotary_dim: int | None = None) -> list[int]:
    """Index vector converting one HF half-split head to Meta interleaved: ``meta = hf[..., perm]``.

    HF pairs element ``i`` with ``i + head_dim/2``; Meta pairs ``2j`` with ``2j+1``. So Meta slot
    ``m`` must read HF slot ``half*(m % 2) + (m // 2)``. Any tail beyond ``rotary_dim`` is
    unrotated and passes through in place.
    """
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    half = rotary_dim // 2
    perm = list(range(head_dim))
    for m in range(rotary_dim):
        perm[m] = half * (m % 2) + (m // 2)
    return perm


def hf_to_meta(x: torch.Tensor, rotary_dim: int | None = None) -> torch.Tensor:
    """Convert a ``[..., head_dim]`` tensor from HF half-split to Meta interleaved pairing."""
    return x[..., hf_to_meta_head_perm(x.shape[-1], rotary_dim)]


# ---------------------------------------------------------------------------------------------
# YaRN RoPE
# ---------------------------------------------------------------------------------------------
def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool = True
) -> tuple[float, float]:
    low = _yarn_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001  # guard against a zero-width ramp
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def yarn_inv_freq(cfg: MistralMediumConfig) -> torch.Tensor:
    """YaRN inverse frequencies, ``[head_dim/2]`` fp32 — transformers' ``_compute_yarn_parameters``.

    YaRN interpolates low-frequency dims (which would otherwise index positions the model never saw
    during training) and extrapolates high-frequency ones, ramping linearly between the two
    correction dims derived from ``beta_fast`` / ``beta_slow``.

    Note transformers **recomputes** ``factor`` as ``max_position_embeddings /
    original_max_position_embeddings`` whenever the latter is present, ignoring the config's own
    ``factor``. Here 262144/4096 = 64.0, which is what the config says anyway; ``from_json`` +
    :func:`yarn_effective_factor` keep the two reconciled rather than silently diverging.
    """
    dim = cfg.head_dim
    base = cfg.rope_theta
    factor = yarn_effective_factor(cfg)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _yarn_correction_range(
        cfg.rope_beta_fast, cfg.rope_beta_slow, dim, base, cfg.rope_original_max_position_embeddings
    )
    # 1 at the extrapolated (high-frequency) end, 0 at the interpolated end.
    extrapolation_factor = 1 - _yarn_linear_ramp(low, high, dim // 2)
    return inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor


def yarn_effective_factor(cfg: MistralMediumConfig) -> float:
    """The scaling factor transformers actually uses (see :func:`yarn_inv_freq`)."""
    if cfg.rope_original_max_position_embeddings:
        return cfg.max_position_embeddings / cfg.rope_original_max_position_embeddings
    return cfg.rope_factor


class MistralYarnRotaryEmbedding(nn.Module):
    """Builds ``(cos, sin)`` for a run of absolute positions, already scaled by ``attention_scaling``."""

    def __init__(self, cfg: MistralMediumConfig, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.register_buffer("inv_freq", yarn_inv_freq(cfg), persistent=False)
        self.attention_scaling = cfg.attention_scaling

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``position_ids`` ``[batch, seq]`` -> ``cos``/``sin`` ``[batch, seq, head_dim]``."""
        # fp32 for the angle: at position 262143 a bf16 angle has ~1024-unit granularity.
        freqs = position_ids[..., None].float() * self.inv_freq.to(torch.float32)
        emb = torch.cat((freqs, freqs), dim=-1)  # HF half-split duplication
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(self.dtype), sin.to(self.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF's half-split rotation: ``[x1 | x2] -> [-x2 | x1]``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``[batch, heads, seq, head_dim]`` Q and K."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ---------------------------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------------------------
class MistralRMSNorm(nn.Module):
    """Plain Mistral RMSNorm: ``w * x / sqrt(mean(x^2) + eps)``.

    No ``1 + w`` fold (that is Gemma) and no bias. The variance is accumulated in fp32 exactly as
    HF does — at hidden 12288 a bf16 sum of squares loses enough mantissa to move PCC.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x32.to(in_dtype)


class MistralMLP(nn.Module):
    """Dense SwiGLU FFN: ``down(silu(gate(x)) * up(x))``. 12288 -> 28672 -> 12288, no biases."""

    def __init__(self, cfg: MistralMediumConfig, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(h, i, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(i, h, bias=False, dtype=dtype)
        assert cfg.hidden_act == "silu", f"reference implements silu only, config says {cfg.hidden_act}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[b, n_kv, s, d] -> [b, n_kv*n_rep, s, d]`` by repeating each KV head ``n_rep`` times."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class MistralAttention(nn.Module):
    """GQA attention, 96 Q heads over 8 KV heads, head_dim 128, no QK-norm and no biases.

    ``forward`` returns ``(output, k_post_rope, v_raw)`` — the two cache tensors are returned in the
    golden trace's own convention (post-RoPE K, raw V, both HF half-split) so a layer's output can
    be compared against ``kv_cache/layer_N.safetensors`` with no further reinterpretation.
    """

    def __init__(self, cfg: MistralMediumConfig, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        self.cfg = cfg
        h, hd = cfg.hidden_size, cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = cfg.num_key_value_groups
        self.head_dim = hd
        self.scaling = cfg.softmax_scale

        self.q_proj = nn.Linear(h, self.num_heads * hd, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(h, self.num_kv_heads * hd, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(h, self.num_kv_heads * hd, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * hd, h, bias=False, dtype=dtype)

        # `Ministral3Attention` also scales Q by `1 + beta*log(1 + floor(pos/original_max_pos))`.
        # beta is 0 for this checkpoint, so that factor is identically 1 and is not implemented.
        assert cfg.rope_llama_4_scaling_beta == 0.0, (
            f"llama_4_scaling_beta={cfg.rope_llama_4_scaling_beta} needs a position-dependent Q "
            "scale that neither the reference nor the device implements"
        )
        assert cfg.sliding_window is None, "sliding_window would need a windowed mask and a bounded cache"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``hidden_states`` ``[b, s, hidden]``.

        ``past_k`` / ``past_v`` are the already-cached ``[b, n_kv, s_past, head_dim]`` tensors for a
        chunked call; the returned K/V are the **new** chunk's only, matching what the device writes
        into the cache for this chunk.
        """
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k_cache_out, v_cache_out = k, v  # what layer N of the golden trace holds for this chunk

        if past_k is not None:
            k = torch.cat([past_k.to(k.dtype), k], dim=2)
            v = torch.cat([past_v.to(v.dtype), v], dim=2)

        kf = repeat_kv(k, self.num_kv_groups)
        vf = repeat_kv(v, self.num_kv_groups)

        # fp32 scores + softmax, as HF's eager path does; a bf16 softmax over 10k keys drifts.
        scores = torch.matmul(q.float(), kf.float().transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[..., : kf.shape[-2]].float()
        probs = F.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(probs, vf)

        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out), k_cache_out, v_cache_out


class MistralDecoderLayer(nn.Module):
    """``x + attn(norm(x))`` then ``x + mlp(norm(x))`` — standard pre-norm, no extra residual scaling."""

    def __init__(self, cfg: MistralMediumConfig, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.self_attn = MistralAttention(cfg, dtype)
        self.post_attention_layernorm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.mlp = MistralMLP(cfg, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, k, v = self.self_attn(self.input_layernorm(hidden_states), cos, sin, attention_mask, past_k, past_v)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, k, v


# ---------------------------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------------------------
def causal_mask(q_len: int, kv_len: int, dtype: torch.dtype = REF_DTYPE, device="cpu") -> torch.Tensor:
    """Additive causal mask ``[1, 1, q_len, kv_len]`` for a chunk whose queries are the **last**
    ``q_len`` positions of a ``kv_len``-long context. ``sliding_window`` is null for this model, so
    there is no windowed variant."""
    offset = kv_len - q_len
    q_pos = torch.arange(q_len, device=device)[:, None] + offset
    k_pos = torch.arange(kv_len, device=device)[None, :]
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(k_pos > q_pos, torch.finfo(dtype).min)
    return mask[None, None]


# ---------------------------------------------------------------------------------------------
# Weight plumbing
# ---------------------------------------------------------------------------------------------
@dataclass
class LayerWeights:
    """One decoder layer's dequantized bf16 weights, in HF orientation ``[out_features, in]``.

    The same container feeds the reference and the device modules, so a PCC comparison can never be
    reading two different weight sets.
    """

    input_layernorm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    _SUFFIXES = (
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )

    @classmethod
    def random(cls, cfg: MistralMediumConfig, seed: int = 0, dtype: torch.dtype = REF_DTYPE) -> "LayerWeights":
        """Deterministic random weights for host-side and mock-device tests.

        Projections are scaled by ``1/sqrt(in_features)`` so activations keep unit-ish magnitude
        through 88 layers instead of overflowing bf16.
        """
        g = torch.Generator().manual_seed(seed)

        def lin(out_f, in_f):
            return (torch.randn(out_f, in_f, generator=g, dtype=torch.float32) / math.sqrt(in_f)).to(dtype)

        def norm(n):
            return (1.0 + 0.02 * torch.randn(n, generator=g, dtype=torch.float32)).to(dtype)

        h, i, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
        return cls(
            input_layernorm=norm(h),
            q_proj=lin(cfg.num_attention_heads * hd, h),
            k_proj=lin(cfg.num_key_value_heads * hd, h),
            v_proj=lin(cfg.num_key_value_heads * hd, h),
            o_proj=lin(h, cfg.num_attention_heads * hd),
            post_attention_layernorm=norm(h),
            gate_proj=lin(i, h),
            up_proj=lin(i, h),
            down_proj=lin(h, i),
        )

    @classmethod
    def from_state_dict(cls, sd: dict, prefix: str) -> "LayerWeights":
        """Pull one layer out of a dequantized state dict keyed ``{prefix}{suffix}.weight``."""
        return cls(**{s.split(".")[-1]: sd[f"{prefix}{s}.weight"] for s in cls._SUFFIXES})

    def to_module(self, layer: MistralDecoderLayer) -> MistralDecoderLayer:
        """Load into a :class:`MistralDecoderLayer` (weights are copied, not aliased)."""
        with torch.no_grad():
            layer.input_layernorm.weight.copy_(self.input_layernorm)
            layer.post_attention_layernorm.weight.copy_(self.post_attention_layernorm)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer.self_attn, name).weight.copy_(getattr(self, name))
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer.mlp, name).weight.copy_(getattr(self, name))
        return layer


def build_layer(cfg: MistralMediumConfig, weights: LayerWeights, dtype: torch.dtype = REF_DTYPE) -> MistralDecoderLayer:
    """A decoder layer loaded with ``weights`` and put in eval mode."""
    layer = MistralDecoderLayer(cfg, dtype)
    weights.to_module(layer)
    return layer.eval()


# ---------------------------------------------------------------------------------------------
# Whole model (M1)
# ---------------------------------------------------------------------------------------------
class MistralModel(nn.Module):
    """Embedding -> ``num_hidden_layers`` x decoder -> final norm -> lm head.

    There is no layer-type table: ``config.json`` has no ``layer_types`` and ``sliding_window`` is
    null, so all 88 layers are the same full-attention block and the stack is a plain loop. A hybrid
    model would need a dispatch here and the device stack would have to mirror it; asserting the
    uniformity in one place is cheaper than discovering it is not uniform at layer 44.

    ``lm_head`` is a separate tensor — ``tie_word_embeddings`` is False for this checkpoint.
    """

    def __init__(self, cfg: MistralMediumConfig, dtype: torch.dtype = REF_DTYPE):
        super().__init__()
        assert cfg.sliding_window is None, "a windowed layer type would need per-layer mask dispatch"
        self.cfg = cfg
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([MistralDecoderLayer(cfg, dtype) for _ in range(cfg.num_hidden_layers)])
        self.norm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_offset: int = 0,
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        want_logits: bool = True,
    ) -> tuple[torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One prefill chunk of ``input_ids`` ``[b, s]``.

        Args:
            position_offset: absolute position of this chunk's first token.
            past: per-layer ``(k, v)`` for the preceding chunks, in the same order as ``layers``.
                None for a one-shot or first-chunk call.
            want_logits: False skips the lm head. The head is a ``[12288, 131072]`` matmul over
                every token, which for a 10240-token prefill is the single most expensive op in the
                model and is pure waste when the test only compares K/V.

        Returns:
            ``(logits or None, new_kv)`` where ``new_kv[i]`` is **this chunk's** ``(k, v)`` for layer
            ``i`` — not the concatenated cache — matching what the device writes per chunk and what
            the golden trace stores per layer.
        """
        b, s = input_ids.shape
        past_len = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(position_offset, position_offset + s, dtype=torch.int64)[None].expand(b, -1)
        cos, sin = MistralYarnRotaryEmbedding(self.cfg, self.dtype)(positions)
        mask = causal_mask(s, past_len + s, dtype=self.dtype)

        hidden = self.embed_tokens(input_ids).to(self.dtype)
        new_kv = []
        for i, layer in enumerate(self.layers):
            pk, pv = (None, None) if past is None else past[i]
            hidden, k, v = layer(hidden, cos, sin, mask, pk, pv)
            new_kv.append((k, v))
        hidden = self.norm(hidden)
        return (self.lm_head(hidden) if want_logits else None), new_kv


@dataclass
class ModelWeights:
    """The whole model's bf16 weights: the tail tensors plus one :class:`LayerWeights` per layer."""

    embed_tokens: torch.Tensor
    layers: list[LayerWeights]
    norm: torch.Tensor
    lm_head: torch.Tensor

    @classmethod
    def random(cls, cfg: MistralMediumConfig, seed: int = 0, dtype: torch.dtype = REF_DTYPE) -> "ModelWeights":
        """Deterministic random weights. Only usable at reduced size — see the note in ``README.md``.

        Per-layer seeds are ``seed + 1 + i`` so a layer's weights do not depend on how many layers
        precede it; that makes a reduced-depth run a genuine prefix of a deeper one.
        """
        g = torch.Generator().manual_seed(seed)
        emb = (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g, dtype=torch.float32) * 0.02).to(dtype)
        head = (
            torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g, dtype=torch.float32) / math.sqrt(cfg.hidden_size)
        ).to(dtype)
        norm = (1.0 + 0.02 * torch.randn(cfg.hidden_size, generator=g, dtype=torch.float32)).to(dtype)
        return cls(
            embed_tokens=emb,
            layers=[LayerWeights.random(cfg, seed=seed + 1 + i, dtype=dtype) for i in range(cfg.num_hidden_layers)],
            norm=norm,
            lm_head=head,
        )

    def to_module(self, model: MistralModel) -> MistralModel:
        with torch.no_grad():
            model.embed_tokens.weight.copy_(self.embed_tokens)
            model.norm.weight.copy_(self.norm)
            model.lm_head.weight.copy_(self.lm_head)
        assert len(self.layers) == len(model.layers), f"{len(self.layers)} weight sets for {len(model.layers)} layers"
        for w, layer in zip(self.layers, model.layers):
            w.to_module(layer)
        return model


def build_model(cfg: MistralMediumConfig, weights: ModelWeights, dtype: torch.dtype = REF_DTYPE) -> MistralModel:
    """A whole model loaded with ``weights`` and put in eval mode.

    At full size this allocates ~128 B parameters; callers are expected to use a reduced config for
    anything that runs on a host. The full-depth ground truth is the prepared golden trace, not a
    host forward of this module.
    """
    model = MistralModel(cfg, dtype)
    weights.to_module(model)
    return model.eval()
