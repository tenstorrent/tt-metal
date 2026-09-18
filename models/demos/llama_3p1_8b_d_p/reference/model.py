# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for Llama-3.1-8B (tt-blaze#4147).

The correctness oracle the rest of the prefill bring-up is measured against: the MLP, KV cache,
attention, RoPE and the decoder layer all PCC against this.

A **fresh** reference, in the same spirit as ``gpt_oss_d_p/reference/model.py``, rather than an
import of ``models/tt_transformers`` — that package is a code-lineage source at most. Module
structure and parameter names mirror HuggingFace ``modeling_llama`` exactly
(``model.layers.N.self_attn.q_proj.weight`` and friends), so a checkpoint state dict loads with no
key translation and any divergence from HF is a real difference rather than a naming artefact.

**Frame.** This file computes in the **HF frame**: q/k weights as the checkpoint ships them
(permuted), and the half-split ``rotate_half`` rotation. That is what makes it comparable to HF
logits. It is *not* the frame blaze decode writes, which is Meta-interleaved — so anything
exporting K for a golden must convert, and :func:`to_meta_frame` is the only supported way to do
it. See ``tt/rope.py`` for why the two frames are the same numbers in a different order, and why
getting this wrong produces a golden that makes a broken prefill look correct.

**Dtype.** Goldens must be round-tripped through the device dtype (``bfloat8_b`` for the KV cache)
before PCC. A full-precision golden leaves a spurious ~0.94-0.96 gap that reads as a real bug.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


# =====================================================================================
# Frame conversion
# =====================================================================================
def to_meta_frame(x: torch.Tensor) -> torch.Tensor:
    """HF half-split ``[r0..r63, i0..i63]`` -> Meta interleaved ``[r0, i0, r1, i1, ...]``.

    Applied on the last (``head_dim``) axis. This is the conversion a golden K must go through
    before it can be compared against what the device wrote, because blaze decode — and therefore
    the prefill model — keeps K in the Meta-interleaved frame.

    Identical to ``tt_transformers.load_checkpoints.reverse_permute_1d``, restated here so the
    reference has no dependency on that package.
    """
    half = x.shape[-1] // 2
    return torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)


def from_meta_frame(x: torch.Tensor) -> torch.Tensor:
    """Meta interleaved -> HF half-split. Inverse of :func:`to_meta_frame`."""
    return torch.cat((x[..., ::2], x[..., 1::2]), dim=-1)


# =====================================================================================
# RoPE (HF frame)
# =====================================================================================
def llama3_inv_freq(
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    orig_max_pos: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
) -> torch.Tensor:
    """Llama-3.1 (``rope_type="llama3"``) inverse frequencies — the HF formula verbatim.

    Duplicated from ``tt/rope.py`` on purpose: a reference that imports the thing it grades cannot
    catch an error in it. ``tests/unit/test_rope_vs_ref.py`` pins both against HuggingFace, so the
    two copies are held together by an external oracle rather than by one importing the other.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    low_freq_wavelen = orig_max_pos / low_freq_factor
    high_freq_wavelen = orig_max_pos / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_llama)


def build_hf_cos_sin(positions: torch.Tensor, head_dim: int = Llama31_8BConfig.HEAD_DIM):
    """HF half-split cos/sin for the given absolute positions: ``[len(positions), head_dim]``."""
    freqs = torch.outer(positions.float(), llama3_inv_freq(head_dim=head_dim))
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF / GPT-NeoX split-half rotation."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply HF-frame RoPE to ``[batch, heads, seq, head_dim]`` with ``[seq, head_dim]`` tables."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


# =====================================================================================
# Modules
# =====================================================================================
class Llama31RMSNorm(nn.Module):
    """HF ``LlamaRMSNorm``: normalise in float32, scale by weight, cast back.

    The float32 upcast is not cosmetic — normalising in bf16 changes the result enough to move
    end-of-sequence logits, and HF does the upcast, so a reference that skips it would report a gap
    that the device does not actually have.
    """

    def __init__(self, dim: int = Llama31_8BConfig.EMB_SIZE, eps: float = Llama31_8BConfig.RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class Llama31MLP(nn.Module):
    """Dense SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    No biases, and no SwiGLU clamp — Llama-3.1's SwiGLU is unclamped, in contrast to gpt-oss's 7.0
    limit. Both are asserted rather than assumed, because a clamp only changes the result at large
    activation magnitudes and would otherwise pass a short-prompt comparison.
    """

    def __init__(
        self,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
    ):
        super().__init__()
        assert Llama31_8BConfig.SWIGLU_LIMIT is None, (
            "Llama-3.1-8B's SwiGLU is unclamped; a non-None SWIGLU_LIMIT means the dim SSOT changed "
            "and this reference no longer matches the model it grades."
        )
        assert not Llama31_8BConfig.ATTENTION_BIAS, "Llama-3.1-8B has no projection biases"
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def torch_weights(self) -> dict:
        """The three projection matrices, keyed by name, for the TTNN module to shard."""
        return {
            "gate_proj": self.gate_proj.weight.data,
            "up_proj": self.up_proj.weight.data,
            "down_proj": self.down_proj.weight.data,
        }


class Llama31Attention(nn.Module):
    """Full-causal GQA: 32 query heads over 8 KV heads (group size 4), ``head_dim`` 128.

    No attention sinks, no sliding window, no QK-norm, no biases — every one of the 32 layers is
    identical in shape, which is what lets the migration address table skip a per-layer branch.
    """

    def __init__(
        self,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        n_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS,
        n_kv_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        head_dim: int = Llama31_8BConfig.HEAD_DIM,
    ):
        super().__init__()
        assert Llama31_8BConfig.SLIDING_WINDOW is None, "Llama-3.1-8B is full attention on every layer"
        assert not Llama31_8BConfig.USE_QK_NORM, "Llama-3.1-8B has no QK norm"
        assert n_heads % n_kv_heads == 0, f"{n_heads} query heads do not group evenly over {n_kv_heads} KV heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ):
        """``x`` is ``[batch, seq, emb]``; ``cos``/``sin`` are ``[seq, head_dim]`` for this chunk.

        ``past_kv`` is the ``(k, v)`` already computed for earlier positions, in HF frame and in
        natural (position) order. Returns the layer output, and — when ``return_kv`` — the full
        ``(k, v)`` including this chunk, which is what the golden generator captures per layer.
        """
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        kv_len = k.shape[2]

        # GQA: repeat each KV head n_rep times so the einsum is plain MHA.
        k_rep = k.repeat_interleave(self.n_rep, dim=1)
        v_rep = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k_rep.transpose(2, 3)) * self.scale
        # Causal mask keyed on ABSOLUTE position: query i of this chunk sits at global position
        # kv_len - seq + i, so it may attend to keys 0 .. that position. Masking on the chunk-local
        # index instead would silently hide the whole prefix on every chunk after the first.
        query_pos = torch.arange(kv_len - seq, kv_len).view(seq, 1)
        key_pos = torch.arange(kv_len).view(1, kv_len)
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))

        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, v_rep).transpose(1, 2).reshape(batch, seq, -1)
        out = self.o_proj(out)
        return (out, (k, v)) if return_kv else (out, None)


class Llama31DecoderLayer(nn.Module):
    """``x + attn(norm(x))`` then ``x + mlp(norm(x))`` — pre-norm, as HF."""

    def __init__(self):
        super().__init__()
        self.self_attn = Llama31Attention()
        self.mlp = Llama31MLP()
        self.input_layernorm = Llama31RMSNorm()
        self.post_attention_layernorm = Llama31RMSNorm()

    def forward(self, x, cos, sin, past_kv=None, return_kv=False):
        attn_out, kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv, return_kv=return_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, kv


class Llama31Model(nn.Module):
    """The 32-layer decoder stack plus embeddings, final norm and LM head.

    ``lm_head`` is included because the acceptance criterion is a **logits** comparison against HF;
    the prefill model itself skips the LM head (only decode needs it).
    """

    def __init__(
        self,
        num_layers: int = Llama31_8BConfig.NUM_LAYERS,
        vocab_size: int = Llama31_8BConfig.VOCAB_SIZE,
    ):
        """``num_layers`` and ``vocab_size`` are overridable only to keep tests affordable.

        At the real values the embedding and LM head are 525M parameters each, which dominates both
        memory and runtime in a comparison that is really about the decoder stack. Every other
        dimension comes from the SSOT and is not a parameter, because changing one would mean the
        reference no longer describes the model it grades.
        """
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, Llama31_8BConfig.EMB_SIZE)
        self.layers = nn.ModuleList(Llama31DecoderLayer() for _ in range(num_layers))
        self.norm = Llama31RMSNorm()
        self.lm_head = nn.Linear(Llama31_8BConfig.EMB_SIZE, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        past_kvs: Optional[list] = None,
        return_kv: bool = False,
    ):
        """Returns ``(logits, per_layer_kv)``. ``per_layer_kv`` is ``None`` unless ``return_kv``.

        ``start_pos`` is the absolute position of ``input_ids[0]``, so a continuation gets the right
        RoPE positions; passing 0 for a later chunk would rotate it as if it were the start of the
        sequence.
        """
        batch, seq = input_ids.shape
        x = self.embed_tokens(input_ids)
        positions = torch.arange(start_pos, start_pos + seq)
        cos, sin = build_hf_cos_sin(positions)

        kvs = [] if return_kv else None
        for index, layer in enumerate(self.layers):
            past = past_kvs[index] if past_kvs is not None else None
            x, kv = layer(x, cos, sin, past_kv=past, return_kv=return_kv)
            if return_kv:
                kvs.append(kv)

        x = self.norm(x)
        return self.lm_head(x), kvs

    # -----------------------------------------------------------------------------
    # Checkpoint loading
    # -----------------------------------------------------------------------------
    def hf_key_map(self) -> dict:
        """This module's parameter names keyed by their HuggingFace name.

        The two differ only by HF's ``model.`` prefix on everything except ``lm_head``, because the
        submodule and parameter names were chosen to match. Kept explicit so a mismatch is a loud
        KeyError rather than a silently untrained tensor.
        """
        mapping = {"model.embed_tokens.weight": "embed_tokens.weight", "model.norm.weight": "norm.weight"}
        mapping["lm_head.weight"] = "lm_head.weight"
        for index in range(self.num_layers):
            prefix, own = f"model.layers.{index}.", f"layers.{index}."
            for suffix in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ):
                mapping[prefix + suffix] = own + suffix
        return mapping

    def load_hf_state_dict(self, hf_state_dict: dict, strict: bool = True, consume: bool = False):
        """Load a HuggingFace Llama-3.1-8B state dict.

        Weights are taken exactly as the checkpoint ships them — q/k stay permuted, so this model
        computes in the HF frame. Do not un-permute here: that would make the logits comparison
        against HF fail while making the KV golden accidentally right, which is the worst of both.

        Copies into each parameter in place rather than assembling a dtype-converted dict and
        handing it to ``load_state_dict``. That matters at this model's size: the dict would hold a
        second full fp32 copy of the weights (~32 GB) alongside the model's own ~32 GB and the
        checkpoint's ~16 GB, which is what OOM-killed golden-KV generation on a 62 GB host.

        Args:
            consume: drop each source tensor from ``hf_state_dict`` once copied, so the checkpoint's
                ~16 GB is released as the load proceeds instead of at the end. Off by default
                because it mutates the caller's dict; the golden generator sets it.
        """
        mapping = self.hf_key_map()
        own = dict(self.named_parameters())
        missing = []
        loaded_keys = set()
        for hf_key, own_key in mapping.items():
            if hf_key not in hf_state_dict:
                missing.append(hf_key)
                continue
            tensor = hf_state_dict[hf_key]
            param = own[own_key]
            expected = tuple(param.shape)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{hf_key}: checkpoint has {tuple(tensor.shape)}, model expects {expected}")
            with torch.no_grad():
                param.copy_(tensor)
            loaded_keys.add(own_key)
            if consume:
                del hf_state_dict[hf_key]

        if missing and strict:
            raise KeyError(f"checkpoint is missing {len(missing)} expected keys, first few: {missing[:5]}")
        unloaded = sorted(set(own) - loaded_keys)
        if strict and unloaded:
            raise KeyError(f"unloaded parameters after mapping: {unloaded[:5]}")
        return unloaded


DEFAULT_CHECKPOINT = Path("/mnt/models/meta-llama/Llama-3.1-8B-Instruct")


def load_hf_state_dict(checkpoint_dir: Path = DEFAULT_CHECKPOINT) -> dict:
    """Read a sharded safetensors checkpoint into a single state dict.

    Imported lazily inside the function so merely importing this module does not pull in
    ``safetensors`` — ``tt/`` modules are held to an import-light contract and the scaffold test
    checks it.
    """
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {checkpoint_dir}")
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(str(shard)))
    return state_dict


def load_reference_model(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT,
    num_layers: int = Llama31_8BConfig.NUM_LAYERS,
    dtype: torch.dtype = torch.float32,
) -> Llama31Model:
    """Build the reference and load real weights into it.

    ``num_layers`` below 32 loads a truncated stack, which is how the per-layer PCC tests stay
    affordable; the logits comparison needs all 32.

    Loads with ``consume=True`` and does not keep a reference to the checkpoint dict, so the
    checkpoint's ~16 GB is released tensor-by-tensor as it is copied in. At the full 32 layers in
    fp32 the model alone is ~32 GB, and holding the checkpoint alongside it to the end is enough to
    OOM a 62 GB host.
    """
    model = Llama31Model(num_layers=num_layers).to(dtype)
    model.load_hf_state_dict(
        load_hf_state_dict(checkpoint_dir),
        strict=num_layers == Llama31_8BConfig.NUM_LAYERS,
        consume=True,
    )
    model.eval()
    return model
