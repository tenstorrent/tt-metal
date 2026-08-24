"""Reference-model loader for Voxtral TTS (``model_type: voxtral_tts``).

Why this file exists
--------------------
This repo cannot be loaded through ``transformers``:

* it ships **no** ``config.json`` (only a Mistral-native ``params.json``), so
  ``AutoConfig.from_pretrained`` raises ``Unrecognized model ... should have a
  ``model_type`` key in its config.json``;
* it has no ``auto_map`` / ``trust_remote_code`` modelling module either;
* ``voxtral_tts`` is **not** a ``transformers`` architecture. ``transformers``
  ships ``voxtral`` / ``voxtral_realtime`` (audio *understanding*) and
  ``ministral`` / ``mistral*`` (text), but nothing that matches this
  checkpoint's audio-generation stack.

The architecture therefore lives outside ``transformers``: it is implemented by
the model's own runtime, ``vllm-omni`` (``README.md`` declares
``library_name: vllm``, ``pipeline_tag: text-to-speech``,
``tags: [mistral-common]``), under
``vllm_omni/model_executor/models/voxtral_tts/``.

Importing that package here is not viable as a *reference*: every vllm module
is built from a ``VllmConfig`` and is wired to vllm's parallel/quantised layer
stack, a CUDA platform, a paged KV-cache manager and a running engine. What
this file does instead is what the upstream code does minus the serving
machinery: it defines the **same native modules, in pure PyTorch, under the
exact key names the checkpoint ships**, and fills every one of them with the
**real weights** from ``consolidated.safetensors``.

The module layout is a 1:1 mirror of the checkpoint, so per-component PCC tests
can address submodules by their native names:

    layers.{0..25}.attention.{wq,wk,wv,wo}      text backbone (Ministral-3-3B
    layers.{0..25}.feed_forward.{w1,w2,w3}      shaped: dim 3072, 26 layers,
    layers.{0..25}.{attention_norm,ffn_norm}    32 q-heads / 8 kv-heads,
    norm                                        head_dim 128, rope_theta 1e6,
    mm_audio_embeddings.tok_embeddings          tied embeddings)
    mm_audio_embeddings.audio_codebook_embeddings.embeddings
    acoustic_transformer.*                      flow-matching acoustic head
    audio_tokenizer.decoder_blocks.{0..7}       neural codec decoder
    audio_tokenizer.output_proj
    audio_tokenizer.quantizer.semantic_codebook

Semantics (RMSNorm, SwiGLU, GQA, native interleaved RoPE, ALiBi + sliding-window
codec attention, layer-scale, weight-normed causal convs, FSQ acoustic codebook,
EMA semantic codebook, Euler flow-matching with CFG) follow
``vllm_omni.model_executor.models.voxtral_tts`` exactly.

Scope note (not a limitation of this loader): the published checkpoint ships the
codec **decoder** only -- there are no ``input_proj.*`` / ``encoder_blocks.*``
tensors in ``consolidated.safetensors``. Upstream handles the same absence (see
``VoxtralTTSAudioTokenizer._encoder_loaded``). Rather than fabricate randomly
initialised encoder weights, the encoder is simply not built, so **every
parameter and persistent buffer of the returned module comes from the real
checkpoint** -- verified as a strict bijection at load time.

How this was verified (worth knowing before you "fix" the RoPE)
--------------------------------------------------------------
Do **not** sanity-check this model with a text continuation. ``tied_embeddings``
makes ``tok_embeddings`` double as an LM head, but upstream never evaluates it --
``compute_logits`` fabricates logits via ``fake_logits_for_audio_tokens`` -- so
the text head is vestigial and scores *worse than uniform* no matter what the
attention does. All three RoPE conventions (interleaved / rotate_half / none)
score identically on text, so text proves nothing here.

The real check is the TTS path, and it is sharply discriminative. Driving this
module end to end (``encode_speech_request`` prompt -> voice embedding spliced in
at the ``audio`` token positions -> backbone -> acoustic transformer -> codes fed
back through ``mm_audio_embeddings`` -> codec) with the native interleaved RoPE
produces genuine speech: the model raises ``[END_AUDIO]`` by itself at a duration
that tracks the input text (1.5 s for "Hello.", 7.6 s for a long sentence), with
~98% of energy below 4 kHz and an f0 that follows the requested voice (~135 Hz
for ``casual_male``, ~282 Hz for ``neutral_female``). Swapping in the
``rotate_half`` convention instead yields a 387 Hz buzz that never terminates.

Import-safe: no I/O, no downloads and no global state at import time. All work
happens inside ``load_reference_model``.

Deterministic: flow matching draws its ``x_0`` noise from a per-module
``torch.Generator`` seeded with a fixed seed, so repeated runs of the same input
produce identical codes. Call ``model.reset_rng()`` to rewind it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["load_reference_model"]

# Special tokens predicted by the audio codebook heads, mirroring
# ``AudioSpecialTokens``: index 0 = [EMPTY_AUDIO], 1 = [END_AUDIO]. All audio
# codes emitted by the model are offset by this many entries.
_N_AUDIO_SPECIAL_TOKENS = 2
_EMPTY_AUDIO_TOKEN_ID = 0
_END_AUDIO_TOKEN_ID = 1

# ``params.json`` does not carry ``n_decoding_steps``; upstream's config parser
# (``_remap_voxtral_tts_audio_args``) warns and falls back to 7 Euler steps.
_DEFAULT_N_DECODING_STEPS = 7
# Default classifier-free-guidance weight (``VoxtralTTSForConditionalGeneration._DEFAULT_CFG_ALPHA``).
_DEFAULT_CFG_ALPHA = 1.2

_DEFAULT_SEED = 0

# Defaults from upstream's ``AudioTokenizerArgs`` dataclass, applied to any field
# ``params.json`` leaves out. This repo omits the three ``encoder_*`` fields; they
# still matter because the codec's sliding-window size is threaded through the
# encoder's downsampling strides before the decoder doubles it back up.
_CODEC_ARG_DEFAULTS = {
    "channels": 1,
    "sampling_rate": 24000,
    "pretransform_patch_size": 240,
    "patch_proj_kernel_size": 7,
    "semantic_codebook_size": 8192,
    "semantic_dim": 256,
    "acoustic_codebook_size": 21,
    "acoustic_dim": 36,
    "conv_weight_norm": True,
    "causal": True,
    "attn_sliding_window_size": 16,
    "half_attn_window_upon_downsampling": True,
    "dim": 1024,
    "hidden_dim": 4096,
    "head_dim": 128,
    "n_heads": 8,
    "n_kv_heads": 8,
    "qk_norm_eps": 1e-6,
    "qk_norm": True,
    "use_biases": False,
    "norm_eps": 1e-2,
    "layer_scale": True,
    "layer_scale_init": None,
    "encoder_transformer_lengths_str": "2,2,2,2",
    "encoder_convs_kernels_str": "4,4,4,3",
    "encoder_convs_strides_str": "2,2,2,1",
    "decoder_transformer_lengths_str": "2,2,2,2",
    "decoder_convs_kernels_str": "3,4,4,4",
    "decoder_convs_strides_str": "1,2,2,2",
}

# Defaults from upstream's ``AcousticTransformerArgs`` dataclass.
_ACOUSTIC_ARG_DEFAULTS = {
    "dim": 768,
    "n_layers": 3,
    "head_dim": 128,
    "hidden_dim": 2048,
    "n_heads": 6,
    "n_kv_heads": 2,
    "use_biases": False,
    "norm_eps": 1e-5,
    "n_decoding_steps": None,
}


def _round_up(n: int, multiple: int) -> int:
    return multiple * ((n + multiple - 1) // multiple)


# --------------------------------------------------------------------------- #
# shared primitives
# --------------------------------------------------------------------------- #


class _RMSNorm(nn.RMSNorm):
    """Upstream uses ``apex.normalization.FusedRMSNorm`` when available and
    falls back to ``torch.nn.RMSNorm``; in practice (and in this environment)
    it is the torch one. Subclassed only to give PCC dumps a readable repr."""

    def extra_repr(self) -> str:
        return f"{tuple(self.normalized_shape)}, eps={self.eps}"


class _FeedForward(nn.Module):
    """SwiGLU: ``w2(silu(w1(x)) * w3(x))``."""

    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """(B, H_kv, T, D) -> (B, H_kv * repeats, T, D)."""
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


# --------------------------------------------------------------------------- #
# text backbone (Ministral-shaped decoder with native interleaved RoPE)
# --------------------------------------------------------------------------- #


def _rope_freqs(head_dim: int, end: int, theta: float, device, dtype=torch.float32) -> torch.Tensor:
    """Angles for Mistral's *native* RoPE: (end, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(end, device=device, dtype=dtype)
    return torch.outer(t, freqs)


def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate *adjacent pairs* of channels.

    This is the native Mistral / mistral-inference convention
    (``view_as_complex`` over ``(..., head_dim // 2, 2)``), i.e. GPT-J-style
    interleaving -- **not** the ``rotate_half`` layout that ``transformers``
    uses. The consolidated checkpoint stores wq/wk in the native layout (HF
    conversion scripts permute them precisely to switch conventions), so the
    unpermuted weights must be paired with this rotation.

    x:      (B, H, T, D)
    angles: (T, D // 2)
    """
    orig_dtype = x.dtype
    x_ = x.float().unflatten(-1, (-1, 2))
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x0, x1 = x_[..., 0], x_[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)
    return out.flatten(-2).to(orig_dtype)


class _Attention(nn.Module):
    """Causal GQA. ``n_heads * head_dim`` (4096) != ``dim`` (3072), hence the
    non-square ``wo``."""

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.n_heads = args["n_heads"]
        self.n_kv_heads = args["n_kv_heads"]
        self.head_dim = args["head_dim"]
        self.repeats = self.n_heads // self.n_kv_heads
        dim = args["dim"]
        use_biases = args["use_biases"]

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=use_biases)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=use_biases)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=use_biases)

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        layer_cache: dict | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq = _apply_rope(xq, angles)
        xk = _apply_rope(xk, angles)

        if layer_cache is not None:
            if "k" in layer_cache:
                xk = torch.cat([layer_cache["k"], xk], dim=2)
                xv = torch.cat([layer_cache["v"], xv], dim=2)
            layer_cache["k"], layer_cache["v"] = xk, xv

        keys = _repeat_kv(xk, self.repeats)
        values = _repeat_kv(xv, self.repeats)

        # The queries are the last `seqlen` positions of `keys`. is_causal=True
        # aligns the diagonal to the *start* of the keys, so it is only correct
        # when nothing is cached; a single decode step attends to everything,
        # and a chunked prefill needs an explicitly offset mask.
        n_keys = keys.shape[2]
        past = n_keys - seqlen
        if seqlen == 1:
            out = F.scaled_dot_product_attention(xq, keys, values)
        elif past == 0:
            out = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
        else:
            q_pos = torch.arange(past, n_keys, device=x.device).unsqueeze(1)
            k_pos = torch.arange(n_keys, device=x.device).unsqueeze(0)
            out = F.scaled_dot_product_attention(xq, keys, values, attn_mask=q_pos >= k_pos)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


class _TransformerBlock(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.attention = _Attention(args)
        self.feed_forward = _FeedForward(args["dim"], args["hidden_dim"], args["use_biases"])
        self.attention_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])
        self.ffn_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])

    def forward(self, x, angles, layer_cache=None):
        h = x + self.attention(self.attention_norm(x), angles, layer_cache)
        return h + self.feed_forward(self.ffn_norm(h))


# --------------------------------------------------------------------------- #
# audio token embeddings
# --------------------------------------------------------------------------- #


class _MultiVocabEmbeddings(nn.Module):
    """One flat embedding table shared by all 37 codebooks, addressed through
    per-codebook offsets (mirrors ``MultiVocabEmbeddings``)."""

    def __init__(self, codebook_sizes: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.codebook_sizes = codebook_sizes
        self.total_vocab_size = sum(codebook_sizes)
        padded_size = _round_up(self.total_vocab_size, 128)
        self.embeddings = nn.Embedding(padded_size, embedding_dim)
        offsets = [0]
        for size in codebook_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        # Derived, not shipped in the checkpoint -> plain attribute, not a buffer.
        # device is pinned so that constructing under ``torch.device("meta")``
        # still yields a real tensor.
        self._offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, K, L) -> (B, K, L, D)."""
        offsets = self._offsets.to(codes.device)
        return self.embeddings(codes + offsets[None, :, None])


class _MMAudioEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, dim: int, codebook_sizes: list[int]) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.audio_codebook_embeddings = _MultiVocabEmbeddings(codebook_sizes, dim)


# --------------------------------------------------------------------------- #
# flow-matching acoustic transformer
# --------------------------------------------------------------------------- #


class _BidirectionalAttention(nn.Module):
    """Full (non-causal, no-RoPE) attention over the 3-token acoustic sequence."""

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.n_heads = args["n_heads"]
        self.n_kv_heads = args["n_kv_heads"]
        self.head_dim = args["head_dim"]
        self.repeats = self.n_heads // self.n_kv_heads
        dim, use_biases = args["dim"], args["use_biases"]

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=use_biases)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=use_biases)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=use_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = _repeat_kv(k, self.repeats)
        v = _repeat_kv(v, self.repeats)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


class _AcousticTransformerBlock(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.attention = _BidirectionalAttention(args)
        self.feed_forward = _FeedForward(args["dim"], args["hidden_dim"], args["use_biases"])
        self.attention_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])
        self.ffn_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class _FlowMatchingAudioTransformer(nn.Module):
    """Predicts one audio frame (1 semantic code + 36 acoustic codes) from one
    backbone hidden state.

    The semantic code is an argmax over ``semantic_codebook_output``; the 36
    acoustic values are produced by Euler-integrating a flow-matching velocity
    field with classifier-free guidance, then finite-scalar-quantised to 21
    levels.
    """

    def __init__(self, audio_model_args: dict, n_decoding_steps: int, generator: torch.Generator) -> None:
        super().__init__()
        args = {**_ACOUSTIC_ARG_DEFAULTS, **audio_model_args["acoustic_transformer_args"]}
        self.args = args

        self.semantic_codebook_size = audio_model_args["semantic_codebook_size"]
        self.n_acoustic_codebook = audio_model_args["n_acoustic_codebook"]
        self.acoustic_embeddings_levels = audio_model_args["acoustic_codebook_size"]

        dim = args["dim"]
        self.input_projection = nn.Linear(self.n_acoustic_codebook, dim, bias=False)
        self.time_projection = nn.Linear(dim, dim, bias=False)
        self.llm_projection = nn.Linear(args["input_dim"], dim, bias=False)

        self.semantic_codebook_output = nn.Linear(
            dim, _round_up(self.semantic_codebook_size + _N_AUDIO_SPECIAL_TOKENS, 128), bias=args["use_biases"]
        )
        self.acoustic_codebook_output = nn.Linear(dim, self.n_acoustic_codebook, bias=False)

        self.layers = nn.ModuleList([_AcousticTransformerBlock(args) for _ in range(args["n_layers"])])
        self.norm = _RMSNorm(dim, eps=args["norm_eps"])

        self.n_steps = n_decoding_steps
        # ``TimeEmbedding`` uses theta=10000 and is fully derived -> kept as a
        # plain attribute so that every registered buffer comes from the file.
        self._time_theta = 10000.0
        self._generator = generator
        self._schedule_cache: dict[Any, tuple] = {}

    # -- derived tensors ---------------------------------------------------- #

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        dim = self.args["dim"]
        device = self.input_projection.weight.device
        inv_freq = torch.exp(
            -math.log(self._time_theta) * torch.arange(dim // 2, device=device, dtype=torch.float32) / (dim // 2)
        )
        emb = torch.einsum("bi,j->bj", t.float(), inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)

    def _schedule(self, dtype: torch.dtype, device) -> tuple:
        key = (dtype, str(device))
        cached = self._schedule_cache.get(key)
        if cached is None:
            timesteps = torch.linspace(0, 1, self.n_steps + 1, device=device).to(dtype)
            t_proj_table = self.time_projection(self._time_embedding(timesteps.view(-1, 1)).to(dtype))
            dts = timesteps[1:] - timesteps[:-1]
            cached = (timesteps, t_proj_table, dts)
            self._schedule_cache[key] = cached
        return cached

    # -- forward ------------------------------------------------------------ #

    def _predict_velocity(self, x_t, llm_proj, t_proj) -> torch.Tensor:
        x_t = x_t.to(llm_proj.dtype)
        h = torch.cat(
            [self.input_projection(x_t.unsqueeze(1)), t_proj.unsqueeze(1), llm_proj.unsqueeze(1)],
            dim=1,
        )
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.acoustic_codebook_output(h[:, 0, :])

    def decode_one_frame(self, semantic_code: torch.Tensor, llm_hidden: torch.Tensor, cfg_alpha) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = semantic_code != _END_AUDIO_TOKEN_ID

        # Deterministic x_0: drawn from the module's seeded CPU generator.
        x_0 = torch.randn(
            B, self.n_acoustic_codebook, generator=self._generator, dtype=torch.float32, device="cpu"
        ).to(device=llm_hidden.device, dtype=llm_hidden.dtype)

        timesteps, t_proj_table, dts = self._schedule(llm_hidden.dtype, llm_hidden.device)

        # cond + uncond in one 2B batch (the uncond branch conditions on zeros).
        llm_proj = self.llm_projection(torch.cat([llm_hidden, torch.zeros_like(llm_hidden)], dim=0))

        cfg_alpha = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device).unsqueeze(1)

        sampled = x_0
        for i in range(len(timesteps) - 1):
            t_proj = t_proj_table[i].unsqueeze(0).expand(B, -1)
            v_all = self._predict_velocity(
                x_t=torch.cat([sampled, sampled], dim=0),
                llm_proj=llm_proj,
                t_proj=torch.cat([t_proj, t_proj], dim=0),
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t
            sampled = sampled + v_t * dts[i]

        sampled = torch.clamp(sampled, -1, 1)
        scaled = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        codes = scaled.round().long()
        codes[~should_decode] = _EMPTY_AUDIO_TOKEN_ID
        return codes + _N_AUDIO_SPECIAL_TOKENS

    def forward(self, llm_hidden: torch.Tensor, cfg_alpha: torch.Tensor | float = _DEFAULT_CFG_ALPHA) -> torch.Tensor:
        """(B, dim) backbone hidden state -> (B, 37) audio codes."""
        if not isinstance(cfg_alpha, torch.Tensor):
            cfg_alpha = torch.full((llm_hidden.shape[0],), float(cfg_alpha), dtype=llm_hidden.dtype)

        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, _EMPTY_AUDIO_TOKEN_ID] = -float("inf")
        semantic_logit[:, (_N_AUDIO_SPECIAL_TOKENS + self.semantic_codebook_size) :] = -float("inf")
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        acoustic_codes = self.decode_one_frame(semantic_code.squeeze(1), llm_hidden, cfg_alpha)
        return torch.cat([semantic_code, acoustic_codes], dim=1)


# --------------------------------------------------------------------------- #
# neural codec (decoder side)
# --------------------------------------------------------------------------- #


def _pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    """``F.pad`` wrapper that tolerates reflect-padding shorter-than-pad inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        return padded[..., : padded.shape[-1] - extra_pad]
    return F.pad(x, paddings, mode, value)


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias
        )
        # New-style parametrisation -> params are named
        # ``conv.parametrizations.weight.original{0,1}``, exactly as shipped.
        self.conv = nn.utils.parametrizations.weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = stride
        self._effective_kernel_size = (kernel_size - 1) * dilation + 1
        self._padding_total = self._effective_kernel_size - self._stride

    def extra_repr(self) -> str:
        c = self.conv
        return (
            f"{c.in_channels}, {c.out_channels}, kernel_size={c.kernel_size[0]}, "
            f"stride={c.stride[0]}, pad_mode={self.pad_mode}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (self._effective_kernel_size - self._padding_total)
        extra_padding = target_length - x.shape[-1]
        x = _pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class _CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=use_bias)
        self.conv = nn.utils.parametrizations.weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def extra_repr(self) -> str:
        c = self.conv
        return f"{c.in_channels}, {c.out_channels}, kernel_size={c.kernel_size[0]}, stride={c.stride[0]}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> torch.Tensor:
        r = 2.0 ** (-8.0 / n)
        return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    m = 2 ** math.floor(math.log2(n_heads))
    return torch.cat([slopes_power_of_2(m), slopes_power_of_2(2 * m)[::2][: n_heads - m]])


class _CodecAttention(nn.Module):
    """Causal, ALiBi-biased, sliding-window attention with QK-RMSNorm."""

    def __init__(self, args: dict, sliding_window: int) -> None:
        super().__init__()
        self.n_heads = args["n_heads"]
        self.n_kv_heads = args["n_kv_heads"]
        self.head_dim = args["head_dim"]
        self.repeats = self.n_heads // self.n_kv_heads
        self.causal = args["causal"]
        self.sliding_window = sliding_window
        dim = args["dim"]

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=args["use_biases"])

        self.qk_norm = args["qk_norm"]
        if self.qk_norm:
            self.q_norm = _RMSNorm(self.n_heads * self.head_dim, eps=args["qk_norm_eps"])
            self.k_norm = _RMSNorm(self.n_kv_heads * self.head_dim, eps=args["qk_norm_eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.qk_norm:
            xq, xk = self.q_norm(xq), self.k_norm(xk)

        q = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = _repeat_kv(xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2), self.repeats)
        v = _repeat_kv(xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2), self.repeats)

        positions = torch.arange(seqlen, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # rel_pos[i, j] = j - i

        slopes = _alibi_slopes(self.n_heads).to(device=x.device, dtype=x.dtype)
        attn_bias = slopes.view(-1, 1, 1) * rel_pos.unsqueeze(0).to(x.dtype)
        if self.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))
        window_right = 0 if self.causal else self.sliding_window
        outside = (rel_pos < -self.sliding_window) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside.unsqueeze(0), float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0))
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


class _CodecTransformerBlock(nn.Module):
    def __init__(self, args: dict, sliding_window: int) -> None:
        super().__init__()
        self.attention = _CodecAttention(args, sliding_window)
        self.feed_forward = _FeedForward(args["dim"], args["hidden_dim"], args["use_biases"])
        self.attention_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])
        self.ffn_norm = _RMSNorm(args["dim"], eps=args["norm_eps"])

        self.layer_scale = args["layer_scale"]
        if self.layer_scale:
            self.attention_scale = nn.Parameter(torch.empty(args["dim"]))
            self.ffn_scale = nn.Parameter(torch.empty(args["dim"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.layer_scale:
            r = self.ffn_scale * r
        return h + r


class _CodecTransformer(nn.Module):
    def __init__(self, args: dict, n_layers: int, sliding_window: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_CodecTransformerBlock(args, sliding_window) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _SemanticCodebook(nn.Module):
    """EMA (Euclidean) codebook -- the usable embedding is
    ``embedding_sum / cluster_usage``."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.empty(codebook_size))
        self.register_buffer("embedding_sum", torch.empty(codebook_size, codebook_dim))
        self._embedding_cache: torch.Tensor | None = None

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding_cache is None:
            self._embedding_cache = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return self._embedding_cache

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, 1, T) -> (B, semantic_dim, T)."""
        return F.embedding(codes.squeeze(1), self.embedding.to(codes.device)).transpose(1, 2)


class _AcousticCodebook(nn.Module):
    """Finite scalar quantisation: level i maps back to ``2i/(L-1) - 1``."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (codes * 2 / (self.n_levels - 1) - 1).to(dtype)


class _MistralAudioCodebook(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.semantic_dim = args["semantic_dim"]
        self.acoustic_dim = args["acoustic_dim"]
        self.semantic_codebook = _SemanticCodebook(args["semantic_codebook_size"], self.semantic_dim)
        self.acoustic_codebook = _AcousticCodebook(args["acoustic_codebook_size"], self.acoustic_dim)

    @property
    def num_codebooks(self) -> int:
        return 1 + self.acoustic_codebook.num_codebooks

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: (B, 37, T) -> (B, 292, T)."""
        semantic = self.semantic_codebook.decode(codes[:, :1, :]).to(dtype)
        acoustic = self.acoustic_codebook.decode(codes[:, 1:, :], dtype)
        return torch.cat([semantic, acoustic], dim=1)


class _AudioTokenizer(nn.Module):
    """Decoder half of the neural codec: 37 discrete codes/frame -> 24 kHz PCM.

    The published checkpoint ships no encoder tensors, so ``input_proj`` /
    ``encoder_blocks`` are deliberately not constructed (see module docstring).
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        args = {**_CODEC_ARG_DEFAULTS, **args}
        self.args = args
        self.patch_size = args["pretransform_patch_size"]
        self.latent_dim = args["semantic_dim"] + args["acoustic_dim"]

        kernels = [int(v) for v in args["decoder_convs_kernels_str"].split(",")]
        strides = [int(v) for v in args["decoder_convs_strides_str"].split(",")]
        lengths = [int(v) for v in args["decoder_transformer_lengths_str"].split(",")]

        # The sliding window is halved on every encoder 2x-downsample and doubled
        # on every decoder 2x-upsample; it is threaded through the *encoder*
        # first upstream, so replay that even though the encoder isn't built.
        window = args["attn_sliding_window_size"]
        if args["half_attn_window_upon_downsampling"]:
            for stride in [int(v) for v in args["encoder_convs_strides_str"].split(",")]:
                if stride > 1:
                    window //= 2

        blocks: list[nn.Module] = [
            _CausalConv1d(
                self.latent_dim,
                args["dim"],
                kernel_size=kernels[0],
                stride=strides[0],
                pad_mode="replicate",
                use_bias=False,
            )
        ]
        if args["half_attn_window_upon_downsampling"] and strides[0] > 1:
            window *= 2

        for idx, n_layers in enumerate(lengths):
            blocks.append(_CodecTransformer(args, n_layers, window))
            if (idx + 1 != len(lengths)) and (kernels[idx + 1] != 1 or strides[idx + 1] != 1):
                blocks.append(
                    _CausalConvTranspose1d(
                        args["dim"], args["dim"], kernel_size=kernels[idx + 1], stride=strides[idx + 1], use_bias=False
                    )
                )
                if args["half_attn_window_upon_downsampling"] and strides[idx + 1] > 1:
                    window *= 2

        self.decoder_blocks = nn.ModuleList(blocks)
        self.quantizer = _MistralAudioCodebook(args)
        self.output_proj = _CausalConv1d(
            args["dim"],
            args["pretransform_patch_size"],
            kernel_size=args["patch_proj_kernel_size"],
            use_weight_norm=args["conv_weight_norm"],
            use_bias=False,
        )

        scale_factor = math.prod(strides)
        self.sampling_rate = args["sampling_rate"]
        self.frame_rate = self.sampling_rate / (self.patch_size * scale_factor)
        self.downsample_factor = int(self.sampling_rate / self.frame_rate)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb.transpose(1, 2).contiguous()  # (B, D, T) -> (B, T, D)
        for block in self.decoder_blocks:
            if isinstance(block, (_CausalConv1d, _CausalConvTranspose1d)):
                x = block(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = block(x)
        x = self.output_proj(x.transpose(1, 2))  # (B, patch_size, T)
        # "b (c h) t -> b c (t h)" with c == channels == 1
        b, ch, t = x.shape
        return x.view(b, ch // self.patch_size, self.patch_size, t).permute(0, 1, 3, 2).reshape(b, -1, t * self.patch_size)

    @property
    def _weight_dtype(self) -> torch.dtype:
        return self.output_proj.conv.parametrizations.weight.original1.dtype

    def decode(self, codes: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """codes: (B, 37, T) of *raw quantiser* codes -> (B, 1, T * 1920) waveform.

        Compute always runs in the codec's own weight dtype (bf16 here), which
        is how upstream drives it (``decode(..., dtype=torch.bfloat16)``);
        ``dtype`` only selects the dtype of the returned waveform.
        """
        out = self._forward_decoder(self.quantizer.decode(codes, self._weight_dtype))
        return out if dtype is None else out.to(dtype)

    def decode_frames(self, codes: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Same as ``decode`` but takes model-emitted codes (offset by the 2
        audio special tokens) of shape (B, T, 37), cutting at [END_AUDIO]."""
        if codes.dim() != 3:
            raise ValueError(f"expected (B, T, K) codes, got {tuple(codes.shape)}")
        eoa = codes[0, :, 0] == _END_AUDIO_TOKEN_ID
        cut = int(eoa.long().argmax()) if bool(eoa.any()) else codes.shape[1]
        return self.decode((codes[:, :cut, :] - _N_AUDIO_SPECIAL_TOKENS).transpose(1, 2), dtype)


# --------------------------------------------------------------------------- #
# top-level reference model
# --------------------------------------------------------------------------- #


class VoxtralTTSReferenceModel(nn.Module):
    """Pure-PyTorch reference for ``model_type: voxtral_tts``.

    Module names match ``consolidated.safetensors`` one-for-one, so a
    per-component PCC harness can walk ``model.layers[i].attention.wq``,
    ``model.acoustic_transformer.layers[i]``,
    ``model.audio_tokenizer.decoder_blocks[i]``, ... directly.
    """

    model_type = "voxtral_tts"

    def __init__(self, params: dict, seed: int = _DEFAULT_SEED) -> None:
        super().__init__()
        self.params = params
        self.dim = params["dim"]
        self.n_layers = params["n_layers"]
        self.vocab_size = params["vocab_size"]
        self.rope_theta = params["rope_theta"]
        self.head_dim = params["head_dim"]
        self.tied_embeddings = params.get("tied_embeddings", True)
        self.max_seq_len = params.get("max_seq_len", 65536)

        mm = params["multimodal"]
        audio_model_args = mm["audio_model_args"]
        codec_args = mm["audio_tokenizer_args"]

        self._generator = torch.Generator(device="cpu")
        self._seed = seed
        self._generator.manual_seed(seed)

        backbone_args = {
            "dim": self.dim,
            "n_heads": params["n_heads"],
            "n_kv_heads": params["n_kv_heads"],
            "head_dim": self.head_dim,
            "hidden_dim": params["hidden_dim"],
            "norm_eps": params["norm_eps"],
            "use_biases": params.get("use_biases", False),
        }
        self.layers = nn.ModuleList([_TransformerBlock(backbone_args) for _ in range(self.n_layers)])
        self.norm = _RMSNorm(self.dim, eps=params["norm_eps"])

        codebook_sizes = [audio_model_args["semantic_codebook_size"] + _N_AUDIO_SPECIAL_TOKENS] + [
            audio_model_args["acoustic_codebook_size"] + _N_AUDIO_SPECIAL_TOKENS
        ] * audio_model_args["n_acoustic_codebook"]
        self.mm_audio_embeddings = _MMAudioEmbeddings(self.vocab_size, self.dim, codebook_sizes)

        n_steps = audio_model_args["acoustic_transformer_args"].get("n_decoding_steps") or _DEFAULT_N_DECODING_STEPS
        self.acoustic_transformer = _FlowMatchingAudioTransformer(audio_model_args, n_steps, self._generator)
        self.audio_tokenizer = _AudioTokenizer(codec_args)

        self._rope_cache: dict[Any, torch.Tensor] = {}

    # -- determinism -------------------------------------------------------- #

    def reset_rng(self, seed: int | None = None) -> None:
        """Rewind the flow-matching noise source (call before a repeat run)."""
        self._generator.manual_seed(self._seed if seed is None else seed)

    # -- embeddings --------------------------------------------------------- #

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.mm_audio_embeddings.tok_embeddings(input_ids)

    def embed_audio_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, 37, L) -> (B, L, dim), summed across codebooks."""
        return self.mm_audio_embeddings.audio_codebook_embeddings(codes).sum(dim=1)

    # -- backbone ----------------------------------------------------------- #

    def _angles(self, positions: torch.Tensor) -> torch.Tensor:
        key = (str(positions.device), self.max_seq_len)
        table = self._rope_cache.get(key)
        if table is None:
            table = _rope_freqs(self.head_dim, self.max_seq_len, self.rope_theta, positions.device)
            self._rope_cache[key] = table
        return table[positions]

    def forward_backbone(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: list[dict] | None = None,
    ) -> torch.Tensor:
        """(B, T, dim) -> (B, T, dim), the ported GPT stack."""
        if positions is None:
            start = 0 if cache is None or not cache[0] else cache[0]["k"].shape[2]
            positions = torch.arange(start, start + hidden_states.shape[1], device=hidden_states.device)
        angles = self._angles(positions)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, angles, None if cache is None else cache[i])
        return self.norm(hidden_states)

    def make_cache(self) -> list[dict]:
        return [{} for _ in range(self.n_layers)]

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ``tied_embeddings`` -> the output projection is the input table.
        return F.linear(hidden_states, self.mm_audio_embeddings.tok_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        cache: list[dict] | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Text-backbone forward: token ids (or embeddings) -> logits.

        This is the module the ttnn GPT port mirrors; ``return_hidden=True``
        yields the normed hidden states that feed ``acoustic_transformer``.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.forward_backbone(inputs_embeds, positions, cache)
        return hidden_states if return_hidden else self.lm_head(hidden_states)

    # -- audio front-to-back ------------------------------------------------ #

    def generate_audio_frame(
        self, hidden_states: torch.Tensor, cfg_alpha: float = _DEFAULT_CFG_ALPHA
    ) -> torch.Tensor:
        """(B, dim) backbone hidden -> (B, 37) audio codes."""
        return self.acoustic_transformer(hidden_states, cfg_alpha)

    def decode_audio(self, codes: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """(B, T, 37) model-emitted codes -> (B, 1, samples) 24 kHz waveform."""
        return self.audio_tokenizer.decode_frames(codes, dtype)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _read_params(model_id: str) -> dict:
    params_path = os.path.join(model_id, "params.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            f"{model_id} has no params.json; this loader targets Mistral-native voxtral_tts checkpoints"
        )
    with open(params_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)
    model_type = params.get("model_type")
    if model_type != "voxtral_tts":
        raise ValueError(f"expected params.json model_type='voxtral_tts', got {model_type!r}")
    return params


def _find_checkpoint(model_id: str) -> str:
    for name in ("consolidated.safetensors", "model.safetensors"):
        path = os.path.join(model_id, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no consolidated.safetensors found under {model_id}")


def load_reference_model(model_id: str):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model,
    loaded from whatever real format the repo actually ships."""
    from safetensors.torch import safe_open

    params = _read_params(model_id)
    checkpoint = _find_checkpoint(model_id)

    # Build on `meta` so nothing is randomly initialised, then fill every tensor
    # from the checkpoint. Anything the file does not cover would surface as an
    # uninitialised tensor, so the bijection check below is enforced, not assumed.
    with torch.device("meta"):
        model = VoxtralTTSReferenceModel(params)

    targets: dict[str, torch.Tensor] = {}
    targets.update(dict(model.named_parameters()))
    targets.update({name: buf for name, buf in model.named_buffers() if buf is not None})

    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as fh:
        ckpt_keys = set(fh.keys())
        missing = sorted(set(targets) - ckpt_keys)
        unexpected = sorted(ckpt_keys - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                "voxtral_tts checkpoint does not match the reference module tree "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]})"
            )
        for key in targets:
            tensor = fh.get_tensor(key)
            expected = tuple(targets[key].shape)
            if tuple(tensor.shape) != expected:
                raise RuntimeError(f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} != module {expected}")
            state[key] = tensor

    model.to_empty(device="cpu")
    # assign=True keeps the checkpoint's bf16 storage instead of copying into the
    # (uninitialised) meta-materialised tensors.
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # pragma: no cover - strict=True already raises
        raise RuntimeError(f"load_state_dict reported {incompatible}")

    model.eval()
    model.requires_grad_(False)
    return model
