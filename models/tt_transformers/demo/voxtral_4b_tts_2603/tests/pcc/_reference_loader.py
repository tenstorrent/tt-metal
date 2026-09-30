# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reference loader for ``mistralai/Voxtral-4B-TTS-2603``.

The repo is *config-less*: it ships a native Mistral ``consolidated.safetensors``
+ ``params.json`` + ``tekken.json`` and no ``config.json``, so ``AutoConfig``
raises "Unrecognized model" (``params.json`` declares ``model_type:
"voxtral_tts"``, for which transformers ships no class).

The checkpoint is 386 tensors / 4002.35 M params in FIVE groups, and a
text-backbone-only loader silently covers 85% of it::

    layers                234   3026.35 M  -> MistralForCausalLM.model.layers
    mm_audio_embeddings     2    430.57 M  -> embed_tokens + audio codebook table
    norm                    1      0.00 M  -> model.norm
    acoustic_transformer   33    393.85 M  -> FlowMatchingAudioTransformer
    audio_tokenizer       116    151.58 M  -> neural codec (DECODER ONLY)

So this loader is a hybrid of two strategies:

* the text backbone is a native->HF key conversion onto
  ``transformers``' ``MistralForCausalLM`` (strategy 2), and
* the two audio submodels have no code in the repo and no class in
  transformers.  Their architecture lives in the model's own serving stack,
  **vLLM-Omni** (Apache-2.0), in a pure-python wheel with no compiled deps:
  ``vllm_omni/model_executor/models/voxtral_tts/{voxtral_tts_audio_generation,
  voxtral_tts_audio_tokenizer}.py``.  The vllm imports in those files are
  serving glue; the model math is plain ``torch.nn``, and it is vendored below
  so this loader has no vllm dependency (strategy 5).

The returned module subclasses ``MistralForCausalLM`` and attaches the two
audio submodels, so every captured backbone path (``model``,
``model.embed_tokens``, ``model.layers.0[.self_attn|.mlp|.input_layernorm]``,
``model.rotary_emb``, ``lm_head``) stays resolvable.

Two things a name-shaped guess gets wrong, both silently:

1. ``acoustic_transformer`` is **bidirectional, non-causal and RoPE-free**.
   Its tensors are named exactly like the text backbone
   (``attention.wq/wk/wv/wo``, ``feed_forward.w1/w2/w3``) but
   ``BidirectionalAttention`` applies no positional encoding and no causal
   mask, so the RoPE permute must NOT be applied here.  The
   ``rope_theta: 10000.0`` in ``acoustic_transformer_args`` is dead config.
2. The codec decoder's attention sliding windows are ``[2, 4, 8, 16]``, not
   ``[16, 32, 64, 128]``: the window halves on every 2x encoder downsample and
   the decoder inherits the encoder's exit value (2).  The encoder's window
   arithmetic therefore has to be replayed even though the OSS checkpoint
   ships no encoder weights and this loader builds no encoder modules.

Verification is STRUCTURAL: every one of the 386 checkpoint tensors is
consumed, no tensor is left on ``meta``, the RoPE theta is asserted to have
landed, the weight-norm convs are asserted to reconstruct, and the codec frame
rate is asserted to be 12.5 Hz.  It is a TTS model whose backbone emits audio
codebook tokens through a tied (effectively untrained) text head, so generated
*text* is near-uniform garbage even when the load is bit-correct -- never
"verify" this checkpoint by reading a continuation.
"""

import inspect
import json
import math
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional, Tuple, Union, get_args, get_origin, get_type_hints

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import MistralConfig, MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

# This loader reaches every substantial group of the checkpoint.
REFERENCE_LOADER_CONTRACT = 2

_DEFAULT_MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"

# Number of Euler ODE steps for the flow-matching sampler.  Absent from
# params.json; vllm-omni's parser warns and defaults to 7.
_DEFAULT_N_DECODING_STEPS = 7


# --------------------------------------------------------------------------
# vendored from vllm-omni (Apache-2.0) -- shared helpers
# --------------------------------------------------------------------------


class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by the audio codebook heads."""

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens():
        return list(AudioSpecialTokens)

    @staticmethod
    def id(token):
        return AudioSpecialTokens.all_special_tokens().index(token)


def from_nested_dict(cls, d):
    """Recursively instantiate dataclasses from nested dicts.

    vllm-omni reads ``f.type`` off ``dataclasses.fields``.  That only works
    because its own module has no ``from __future__ import annotations``: with
    postponed evaluation ``f.type`` is a *string*, ``is_dataclass`` is False,
    and a nested dataclass field silently stays a raw dict.  Resolving through
    ``get_type_hints`` is correct either way.
    """
    if not is_dataclass(cls):
        return d

    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = hints.get(f.name, f.type)

        origin = get_origin(field_type)
        if origin is Union:
            non_none = [a for a in get_args(field_type) if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]

        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------
# vendored from vllm-omni -- flow-matching acoustic transformer
# --------------------------------------------------------------------------


@dataclass
class AcousticTransformerArgs:
    input_dim: int
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5
    n_decoding_steps: Optional[int] = None


@dataclass
class MultimodalAudioModelArgs:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self):
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(self, pad_to_multiple=128, include_special_tokens=True):
        def _round_up(n, multiple):
            return multiple * ((n + multiple - 1) // multiple)

        result = []
        for cb_size in self.codebook_sizes:
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up(cb_size, pad_to_multiple)
            result.append(cb_size)
        return result


class BidirectionalAttention(nn.Module):
    """Attention layer without ANY positional encoding and without a causal mask."""

    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.layer_id = layer_id
        self.head_dim = args.head_dim

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        self.repeats = self.n_local_heads // self.n_local_kv_heads

    def _native_attention(self, query, key, value):
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def _forward_attention(self, query, key, value):
        key, value = repeat_kv(key, value, repeats=self.repeats)
        bsz, seqlen, _, _ = query.shape
        output = self._native_attention(query, key, value)
        return output.view(bsz, seqlen, -1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        output = self._forward_attention(query=xq, key=xk, value=xv, **kwargs)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        return h + r


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for the flow-matching time step."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self._dim = dim
        self._theta = theta
        self.register_buffer("inv_freq", self._build_inv_freq(), persistent=False)

    def _build_inv_freq(self, device=None) -> torch.Tensor:
        half = self._dim // 2
        return torch.exp(
            -math.log(self._theta) * torch.arange(half, device=device).float() / half
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformer(nn.Module):
    """Velocity field of the flow-matching acoustic sampler.

    Its sequence is literally THREE tokens --
    ``[input_projection(x_t), time_projection(t_emb), llm_projection(llm_hidden)]``
    -- and ``semantic_codebook_output`` reads the RAW backbone hidden state,
    not the stack output; only ``acoustic_codebook_output`` reads position 0.
    """

    def __init__(self, audio_model_args: dict) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        assert isinstance(self.model_args, MultimodalAudioModelArgs)
        args = self.model_args.acoustic_transformer_args
        assert isinstance(args, AcousticTransformerArgs), (
            "acoustic_transformer_args stayed a raw dict -- from_nested_dict did not recurse"
        )
        self.acoustic_transformer_args = args

        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert len(set(acoustic_codebook_sizes)) == 1, "only 1 size for acoustic codebooks supported"
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        self._n_steps = args.n_decoding_steps
        self._noise_scale = 1.0
        self.register_buffer("_timesteps", self._build_timesteps(), persistent=False)

    def _build_timesteps(self, device=None) -> torch.Tensor:
        return torch.linspace(0, 1, self._n_steps + 1, device=device)

    def _init_audio_embeddings_layer(self) -> None:
        args = self.acoustic_transformer_args
        self.time_embedding = TimeEmbedding(args.dim)
        self.input_projection = nn.Linear(self.acoustic_embeddings_dim, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)

    def _init_output_layer(self) -> None:
        args = self.acoustic_transformer_args
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            args.dim, padded_codebook_sizes[0], args.use_biases
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        args = self.acoustic_transformer_args
        self.layers_ids = list(range(args.n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = AcousticTransformerBlock(layer_id=layer_id, args=args)
        self.norm = nn.RMSNorm(args.dim, args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h

    def _predict_velocity(self, x_t, llm_proj, t_proj) -> torch.Tensor:
        x_t = x_t.to(llm_proj.dtype)
        inputs = torch.concatenate(
            [
                self.input_projection(x_t.unsqueeze(1)),
                t_proj.unsqueeze(1),
                llm_proj.unsqueeze(1),
            ],
            dim=1,
        )
        attn_output = self.forward_attention_layers(inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(-1, inputs.shape[1], final_hidden.shape[-1])
        return self.acoustic_codebook_output(final_hidden[:, 0, :])

    def decode_one_frame(self, semantic_code, llm_hidden, cfg_alpha) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = semantic_code != self._end_audio_token_id

        x_0 = torch.randn(
            B,
            self.model_args.n_acoustic_codebook,
            dtype=llm_hidden.dtype,
            device=llm_hidden.device,
        )
        x_0 = self._noise_scale * x_0

        timesteps = self._timesteps.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        t_emb_table = self.time_embedding(timesteps.view(-1, 1)).to(llm_hidden.dtype)
        t_proj_table = self.time_projection(t_emb_table)
        dts = timesteps[1:] - timesteps[:-1]

        llm_batched = torch.cat([llm_hidden, torch.zeros_like(llm_hidden)], dim=0)
        llm_proj_batched = self.llm_projection(llm_batched)

        cfg_alpha = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device).unsqueeze(1)

        sampled = x_0
        for i in range(len(timesteps) - 1):
            dt = dts[i]
            t_proj = t_proj_table[i].unsqueeze(0).expand(B, -1)

            v_all = self._predict_velocity(
                x_t=torch.cat([sampled, sampled], dim=0),
                llm_proj=llm_proj_batched,
                t_proj=torch.cat([t_proj, t_proj], dim=0),
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t
            sampled = sampled + v_t * dt

        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens)

    def forward(self, llm_hidden: torch.Tensor, cfg_alpha: torch.Tensor) -> torch.Tensor:
        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, self._empty_audio_token_id] = -float("inf")
        semantic_logit[
            :, (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size) :
        ] = -float("inf")

        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)
        acoustic_codes = self.decode_one_frame(
            semantic_code.squeeze(1), llm_hidden, cfg_alpha=cfg_alpha
        )
        return torch.concatenate([semantic_code, acoustic_codes], dim=1)


# --------------------------------------------------------------------------
# vendored from vllm-omni -- neural audio codec (decoder side)
# --------------------------------------------------------------------------

weight_norm = torch.nn.utils.parametrizations.weight_norm


@dataclass
class AudioTokenizerArgs:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7

    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36

    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: Optional[float] = None

    encoder_transformer_lengths_str: str = "2,2,2,2"
    encoder_convs_kernels_str: str = "4,4,4,3"
    encoder_convs_strides_str: str = "2,2,2,1"

    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    def __post_init__(self) -> None:
        assert (
            len(self.encoder_transformer_lengths)
            == len(self.encoder_convs_kernels)
            == len(self.encoder_convs_strides)
        )
        assert (
            len(self.decoder_transformer_lengths)
            == len(self.decoder_convs_kernels)
            == len(self.decoder_convs_strides)
        )

    def __str2list__(self, input_str: str) -> Tuple[int, ...]:
        return tuple(int(i) for i in input_str.split(","))

    @property
    def encoder_transformer_lengths(self):
        return self.__str2list__(self.encoder_transformer_lengths_str)

    @property
    def encoder_convs_kernels(self):
        return self.__str2list__(self.encoder_convs_kernels_str)

    @property
    def encoder_convs_strides(self):
        return self.__str2list__(self.encoder_convs_strides_str)

    @property
    def decoder_transformer_lengths(self):
        return self.__str2list__(self.decoder_transformer_lengths_str)

    @property
    def decoder_convs_kernels(self):
        return self.__str2list__(self.decoder_convs_kernels_str)

    @property
    def decoder_convs_strides(self):
        return self.__str2list__(self.decoder_convs_strides_str)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / (
            self.pretransform_patch_size * math.prod(self.encoder_convs_strides)
        )


class SemanticCodebook(nn.Module):
    """Euclidean-distance codebook for semantic quantization.

    ``cluster_usage`` / ``embedding_sum`` are BUFFERS, not parameters -- they
    carry 2.105 M of the checkpoint's 4002.35 M and are easy to miss in a
    parameter-only census.
    """

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.codebook_size = codebook_size
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        B, D, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, D)
        embedding = self.embedding.to(x.device)
        distances = torch.cdist(x, embedding, p=2)
        return distances.argmin(dim=-1).view(B, 1, T)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        assert codes.shape[1] == self.num_codebooks == 1
        codes = codes.squeeze(1)
        quantized = F.embedding(codes, self.embedding.to(codes.device))
        return quantized.permute(0, 2, 1)

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_sizes(self):
        return [self.codebook_size]


class AcousticCodebook(nn.Module):
    """Finite scalar quantization for the acoustic codebooks (weight-free)."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.dim = codebook_dim
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def _quantize(self, x, levels, ste: bool = True):
        scaled_x = ((x + 1) / 2) * (levels - 1)
        if ste:
            return scaled_x + (scaled_x.round() - scaled_x).detach()
        return scaled_x.round()

    def _rescale(self, x, levels):
        return (x * 2 / (levels - 1)) - 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        x = torch.tanh(x)
        levels = torch.ones_like(x) * self.n_levels
        return self._quantize(x, levels, ste=False).long()

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        return self._rescale(codes, self.n_levels).to(dtype)

    @property
    def codebook_sizes(self):
        return [self.n_levels for _ in range(self.num_codebooks)]


class MistralAudioCodebook(nn.Module):
    def __init__(self, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self.semantic_codebook = SemanticCodebook(
            codebook_size=args.semantic_codebook_size, codebook_dim=args.semantic_dim
        )
        self.acoustic_codebook = AcousticCodebook(
            codebook_size=args.acoustic_codebook_size, codebook_dim=args.acoustic_dim
        )
        self.semantic_dim = args.semantic_dim
        self.acoustic_dim = args.acoustic_dim
        self.total_dim = self.semantic_dim + self.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return self.semantic_codebook.num_codebooks + self.acoustic_codebook.num_codebooks

    @property
    def codebook_sizes(self):
        return self.semantic_codebook.codebook_sizes + self.acoustic_codebook.codebook_sizes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        semantic_codes = self.semantic_codebook.encode(x[:, : self.semantic_dim, :])
        acoustic_codes = self.acoustic_codebook.encode(x[:, self.semantic_dim :, :])
        return torch.cat([semantic_codes, acoustic_codes], dim=1)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        n_sem = self.semantic_codebook.num_codebooks
        semantic_emb = self.semantic_codebook.decode(codes[:, :n_sem, :]).to(dtype)
        acoustic_emb = self.acoustic_codebook.decode(codes[:, n_sem:, :]).to(dtype)
        return torch.cat([semantic_emb, acoustic_emb], dim=1)


def pad1d(x: torch.Tensor, paddings, mode: str = "constant", value: float = 0.0):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
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
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (
            self._effective_kernel_size - self._padding_total
        )
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
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
        conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=use_bias
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


class MultiVocabEmbeddings(nn.Module):
    """Audio codebook lookup table (37 codebooks packed into one padded table)."""

    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = list(self.model_args.get_codebook_sizes(pad_to_multiple=None))
        offsets = [0]
        for c in self.codebook_sizes[:-1]:
            offsets.append(offsets[-1] + c)
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        self.total_vocab_size = sum(self.codebook_sizes)
        padded_size = 128 * ((self.total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(padded_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.offsets = self.offsets.to(input_ids.device)
        input_ids = input_ids + self.offsets[None, :, None]
        return self.embeddings(input_ids)


class CodecAttention(nn.Module):
    """Sliding-window causal attention with ALiBi and QK-norm."""

    def __init__(self, args: AudioTokenizerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.layer_id = layer_id
        self.sliding_window = args.attn_sliding_window_size

        self.register_buffer("alibi_slopes", self._build_alibi_slopes(), persistent=False)

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        if args.qk_norm:
            self.q_norm = nn.RMSNorm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = nn.RMSNorm(args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps)

    def _build_alibi_slopes(self, device=None) -> torch.Tensor:
        n_heads = self.n_local_heads

        def slopes_power_of_2(n: int) -> torch.Tensor:
            r = 2.0 ** (-8.0 / n)
            return torch.tensor([r**i for i in range(n)], dtype=torch.float32, device=device)

        if math.log2(n_heads).is_integer():
            slopes = slopes_power_of_2(n_heads)
        else:
            m = 2 ** math.floor(math.log2(n_heads))
            slopes = torch.cat(
                [slopes_power_of_2(m), slopes_power_of_2(2 * m)[::2][: n_heads - m]]
            )
        return slopes.to(torch.float32).contiguous()

    def _native_attention(self, xq, xk, xv) -> torch.Tensor:
        B, S, H, D = xq.shape
        Hkv = xk.shape[2]

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if H != Hkv:
            repeats = H // Hkv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        positions = torch.arange(S, device=xq.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)

        alibi_slopes = self.alibi_slopes.to(dtype=xq.dtype)
        attn_bias = alibi_slopes.view(H, 1, 1) * rel_pos.unsqueeze(0).to(xq.dtype)

        if self.args.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        window_left = self.sliding_window
        window_right = 0 if self.args.causal else self.sliding_window
        outside_window = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0))
        return output.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.args.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        output = self._native_attention(xq, xk, xv)
        output = output.view(bsz, seqlen, self.n_local_heads * self.args.head_dim)
        return self.wo(output).squeeze(0)


class CodecTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = CodecAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.hidden_dim, use_biases=args.use_biases
        )
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_norm = None
        self.post_ffn_norm = None
        self.args = args

        self.layer_scale = args.layer_scale
        if self.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = args.layer_scale_init
            self.attention_scale = nn.Parameter(torch.full((args.dim,), init_scale))
            self.ffn_scale = nn.Parameter(torch.full((args.dim,), init_scale))

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        if self.post_attention_norm is not None:
            r = self.post_attention_norm(r)
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        if self.post_ffn_norm is not None:
            r = self.post_ffn_norm(r)
        if self.layer_scale:
            r = self.ffn_scale * r
        return h + r


class CodecTransformer(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, n_layers: int) -> None:
        super().__init__()
        self.args = args
        self.n_layers = n_layers
        self.layers_ids = list(range(n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = CodecTransformerBlock(layer_id=layer_id, args=args)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = input_ids
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h


class VoxtralTTSAudioTokenizer(nn.Module):
    """Neural audio codec.

    The open-source checkpoint ships the DECODER, quantizer and audio-token
    table only -- there are no ``input_proj.*`` / ``encoder_blocks.*`` tensors.
    This builds no encoder modules (building them would inject random weights
    into a "real-weight" reference) but still replays the encoder's
    sliding-window arithmetic, because the decoder inherits the window value
    the encoder loop exits with.
    """

    def __init__(
        self,
        codec_args: dict,
        audio_model_args: dict,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        args = from_nested_dict(AudioTokenizerArgs, codec_args)
        self.args = args
        if not args.causal:
            raise NotImplementedError("the causal mask is hard-coded in forward")

        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        # No encoder weights in the OSS checkpoint.
        self.input_proj = None
        self.encoder_blocks = nn.ModuleList()
        self._encoder_available = False

        # Replay the encoder window arithmetic ONLY: 16 -> 8 -> 4 -> 2.
        cur_window_size = args.attn_sliding_window_size
        enc_kernels = args.encoder_convs_kernels
        enc_strides = args.encoder_convs_strides
        for idx in range(len(args.encoder_transformer_lengths)):
            is_last_layer = idx == len(args.encoder_transformer_lengths) - 1
            if (enc_kernels[idx] != 1) or (enc_strides[idx] != 1) or is_last_layer:
                if args.half_attn_window_upon_downsampling and (enc_strides[idx] > 1):
                    assert enc_strides[idx] == 2, "only supporting 2x downsampling"
                    cur_window_size = cur_window_size // 2
                    assert cur_window_size >= 2

        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_model_args, embedding_dim=text_hidden_size
        )

        decoder_blocks = []
        dec_kernels = args.decoder_convs_kernels
        dec_strides = args.decoder_convs_strides
        dec_lengths = args.decoder_transformer_lengths

        decoder_blocks.append(
            CausalConv1d(
                self.latent_dim,
                args.dim,
                kernel_size=dec_kernels[0],
                stride=dec_strides[0],
                pad_mode="replicate",
                use_bias=False,
            )
        )
        if args.half_attn_window_upon_downsampling and (dec_strides[0] > 1):
            assert dec_strides[0] == 2, "only supporting 2x upsampling"
            cur_window_size = cur_window_size * 2

        self.decoder_window_sizes = []
        for idx, n_layers in enumerate(dec_lengths):
            layer_args = from_nested_dict(AudioTokenizerArgs, codec_args)
            layer_args.attn_sliding_window_size = cur_window_size
            self.decoder_window_sizes.append(cur_window_size)
            decoder_blocks.append(CodecTransformer(args=layer_args, n_layers=n_layers))

            if (idx + 1 != len(dec_lengths)) and (
                (dec_kernels[idx + 1] != 1) or (dec_strides[idx + 1] != 1)
            ):
                decoder_blocks.append(
                    CausalConvTranspose1d(
                        args.dim,
                        args.dim,
                        kernel_size=dec_kernels[idx + 1],
                        stride=dec_strides[idx + 1],
                        use_bias=False,
                    )
                )
                if args.half_attn_window_upon_downsampling and (dec_strides[idx + 1] > 1):
                    assert dec_strides[idx + 1] == 2, "only supporting 2x upsampling"
                    cur_window_size = cur_window_size * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.quantizer = MistralAudioCodebook(args)
        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        scale_factor = math.prod(enc_strides)
        assert scale_factor == math.prod(dec_strides)
        self._frame_rate = args.sampling_rate / (self.patch_size * scale_factor)
        self._sampling_rate = args.sampling_rate
        self._channels = args.channels
        if self._channels != 1:
            raise NotImplementedError

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def downsample_factor(self) -> int:
        assert self._sampling_rate % self._frame_rate == 0
        return int(self._sampling_rate / self._frame_rate)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def codebook_sizes(self):
        return self.quantizer.codebook_sizes

    def encode_tokens(self, x):
        audio_embeddings = []
        for audio_code in x:
            emb = self.audio_token_embedding(audio_code)
            emb = emb.sum(dim=1)
            audio_embeddings.append(emb.squeeze(0))
        return audio_embeddings

    def encode_waveforms(self, x):
        raise RuntimeError(
            "encode_waveforms requires encoder weights, which the open-source "
            "checkpoint does not ship (no input_proj.* / encoder_blocks.* tensors)."
        )

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.permute(0, 2, 1).contiguous()
        for block in self.decoder_blocks:
            if type(block) in (CausalConvTranspose1d, CausalConv1d):
                emb = block(emb.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                emb = block(emb)
        emb = emb.permute(0, 2, 1)
        emb = self.output_proj(emb)
        b, ch, t = emb.shape
        # "b (c h) t -> b c (t h)" with h = patch_size
        c = ch // self.patch_size
        emb = emb.view(b, c, self.patch_size, t)
        return emb.permute(0, 1, 3, 2).reshape(b, c, t * self.patch_size)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        emb = self.quantizer.decode(codes, dtype)
        return self._forward_decoder(emb)

    def forward(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.decode(codes, dtype)


# --------------------------------------------------------------------------
# the reference model: Mistral text backbone + the two audio submodels
# --------------------------------------------------------------------------


class VoxtralTTSReferenceModel(MistralForCausalLM):
    """``MistralForCausalLM`` plus the TTS half of the checkpoint.

    Subclassing (rather than wrapping) keeps every captured backbone
    submodule path resolvable while still covering all 386 tensors.
    """

    def __init__(self, config: MistralConfig, audio_model_args=None, codec_args=None) -> None:
        super().__init__(config)
        self.acoustic_transformer = (
            FlowMatchingAudioTransformer(audio_model_args) if audio_model_args is not None else None
        )
        self.audio_tokenizer = (
            VoxtralTTSAudioTokenizer(codec_args, audio_model_args, config.hidden_size)
            if codec_args is not None
            else None
        )


_LAYER_MAP = {
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}

_AUDIO_CODEBOOK_TABLE = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
_TEXT_EMBED_TABLE = "mm_audio_embeddings.tok_embeddings.weight"


def _resolve_repo(model_id: str) -> str:
    """Local directory holding params.json + consolidated.safetensors."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["params.json", "consolidated.safetensors"])


def _build_text_config(params: dict) -> MistralConfig:
    """Native params.json -> MistralConfig.

    ``head_dim`` is 128 here, NOT dim // n_heads (= 96), so it has to be set
    explicitly or q_proj comes out the wrong width.  ``sliding_window`` stays
    None: the ``attn_sliding_window_size: 16`` in params.json belongs to the
    CODEC, not to this stack.
    """
    scaled = [k for k in ("yarn", "llama_4_scaling", "rope_scaling") if params.get(k)]
    if scaled:
        raise RuntimeError(
            f"params.json declares scaled RoPE {scaled}; plain MistralForCausalLM would use "
            "the wrong positional encoding."
        )

    kwargs = dict(
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=params["hidden_dim"],
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        head_dim=params["head_dim"],
        rms_norm_eps=params["norm_eps"],
        max_position_embeddings=params.get("max_position_embeddings") or params["max_seq_len"],
        sliding_window=params.get("sliding_window"),
        tie_word_embeddings=params.get("tied_embeddings", True),
        bos_token_id=params.get("multimodal", {}).get("bos_token_id", 1),
        eos_token_id=2,
        use_cache=True,
        attn_implementation="eager",
    )

    # transformers 4.x takes `rope_theta=`; 5.x moved it into
    # `rope_parameters={"rope_type": ..., "rope_theta": ...}`.  Passing only
    # the 5.x spelling to a 4.x build lands it in **kwargs and the model
    # silently runs at the DEFAULT theta 1e4 instead of 1e6.
    theta = params["rope_theta"]
    sig = inspect.signature(MistralConfig.__init__).parameters
    if "rope_parameters" in sig:
        kwargs["rope_parameters"] = {"rope_type": "default", "rope_theta": theta}
    elif "rope_theta" in sig:
        kwargs["rope_theta"] = theta
    else:
        raise RuntimeError("MistralConfig accepts neither rope_theta nor rope_parameters")

    return MistralConfig(**kwargs)


def _config_rope_theta(cfg: MistralConfig) -> float:
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        return float(params["rope_theta"])
    return float(getattr(cfg, "rope_theta"))


def _permute_rope(w: torch.Tensor, n_heads: int, head_dim: int, hidden: int) -> torch.Tensor:
    """Native Mistral interleaved-RoPE layout -> transformers rotate_half layout.

    Verified against a hand-written native interleaved-RoPE attention on
    layer 0: PCC 0.99999976 permuted vs 0.98317850 unpermuted.
    """
    return w.view(n_heads, head_dim // 2, 2, hidden).transpose(1, 2).reshape(n_heads * head_dim, hidden)


def _remap(reader, cfg: MistralConfig):
    """Return (state_dict, consumed_checkpoint_keys)."""
    hidden, head_dim = cfg.hidden_size, cfg.head_dim
    available = set(reader.keys())
    consumed = set()

    def take(key: str) -> torch.Tensor:
        if key not in available:
            raise RuntimeError(f"{key!r} missing from consolidated.safetensors")
        consumed.add(key)
        return reader.get_tensor(key)

    state = {
        "model.embed_tokens.weight": take(_TEXT_EMBED_TABLE),
        "model.norm.weight": take("norm.weight"),
    }
    if cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    else:
        state["lm_head.weight"] = take("output.weight")

    # --- text backbone: RoPE permute on q/k, w1/w2/w3 -> gate/down/up
    for i in range(cfg.num_hidden_layers):
        for src_suffix, dst_suffix in _LAYER_MAP.items():
            w = take(f"layers.{i}.{src_suffix}")
            if src_suffix == "attention.wq.weight":
                w = _permute_rope(w, cfg.num_attention_heads, head_dim, hidden)
            elif src_suffix == "attention.wk.weight":
                w = _permute_rope(w, cfg.num_key_value_heads, head_dim, hidden)
            state[f"model.layers.{i}.{dst_suffix}"] = w

    # --- acoustic transformer: identity map.  NO RoPE permute -- the stack is
    #     bidirectional and RoPE-free despite the backbone-shaped names.
    for key in sorted(k for k in available if k.startswith("acoustic_transformer.")):
        state[key] = take(key)

    # --- codec: identity map, plus the audio codebook table which the
    #     checkpoint parks next to the text embedding table.
    for key in sorted(k for k in available if k.startswith("audio_tokenizer.")):
        state[key] = take(key)
    state["audio_tokenizer.audio_token_embedding.embeddings.weight"] = take(_AUDIO_CODEBOOK_TABLE)

    state = {k: v.to(torch.float32) for k, v in state.items()}
    return state, consumed


class _skip_param_init:
    """No-op the expensive random init of Linear/Embedding during construction.

    Every such parameter is replaced by a checkpoint tensor below, and the
    strict key check guarantees none is left holding ``torch.empty`` memory.
    Conv init is left alone so ``weight_norm`` registers on sane values.
    """

    _targets = (nn.Linear, nn.Embedding)

    def __enter__(self):
        self._saved = [(cls, cls.reset_parameters) for cls in self._targets]
        for cls, _ in self._saved:
            cls.reset_parameters = lambda self: None
        return self

    def __exit__(self, *exc):
        for cls, fn in self._saved:
            cls.reset_parameters = fn
        return False


def _rebuild_nonpersistent_buffers(model: nn.Module, cfg: MistralConfig) -> None:
    """Re-materialise the buffers that are not in the checkpoint.

    An ``assign=True`` load leaves every non-persistent buffer on ``meta``
    (or, after a plain ``to_empty``, holding uninitialised memory), which is
    silently wrong rather than loud.
    """
    model.model.rotary_emb = MistralRotaryEmbedding(config=cfg, device="cpu")

    for module in model.modules():
        if isinstance(module, TimeEmbedding):
            module.register_buffer("inv_freq", module._build_inv_freq(), persistent=False)
        elif isinstance(module, CodecAttention):
            module.register_buffer("alibi_slopes", module._build_alibi_slopes(), persistent=False)
        elif isinstance(module, FlowMatchingAudioTransformer):
            module.register_buffer("_timesteps", module._build_timesteps(), persistent=False)
        elif isinstance(module, SemanticCodebook):
            # Drop any cached embedding so it is recomputed from the loaded buffers.
            module.register_buffer("_embedding", None, persistent=False)


def _verify(model: nn.Module, cfg: MistralConfig, params: dict, n_ckpt_tensors: int) -> None:
    """Structural verification.

    This is a TTS model: the backbone emits AUDIO codebook tokens and its tied
    text head is effectively untrained, so a generated continuation is
    near-uniform garbage even when the load is bit-correct.  Never verify this
    checkpoint by reading generated text.
    """
    on_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if on_meta:
        raise RuntimeError(f"{len(on_meta)} tensors left on meta, e.g. {on_meta[:5]}")

    # The RoPE theta actually took (see the 4.x/5.x spelling trap).
    want_theta = _config_rope_theta(cfg)
    inv_freq = model.model.rotary_emb.inv_freq
    got_theta = float(inv_freq[1] ** (-cfg.head_dim / 2))
    if abs(got_theta - want_theta) / want_theta > 1e-3:
        raise RuntimeError(
            f"rotary_emb runs at theta {got_theta:.6g}, expected {want_theta:.6g} "
            "-- the rope_theta/rope_parameters spelling did not take"
        )
    assert want_theta == params["rope_theta"], (want_theta, params["rope_theta"])

    # Weight-norm convs reconstruct.  `g` (original0) CAN be negative, so the
    # identity is per-out-channel ||w|| == |g|, not == g.
    n_wn = 0
    for name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None or "weight" not in parametrizations:
            continue
        g = parametrizations.weight.original0
        w = module.weight
        norms = w.reshape(w.shape[0], -1).norm(dim=1) if g.shape[0] == w.shape[0] else None
        if norms is None:
            continue
        rel = (norms - g.flatten().abs()).abs().max() / g.flatten().abs().max()
        if rel > 1e-4:
            raise RuntimeError(f"weight_norm did not reconstruct at {name}: rel err {rel:.3g}")
        n_wn += 1
    if n_wn != 5:
        raise RuntimeError(f"expected 5 weight-normed convs in the codec decoder, found {n_wn}")

    # Codec geometry: 12.5 Hz == 1920 samples/frame, matching audio_encoding_args.
    codec = model.audio_tokenizer
    enc_frame_rate = params["multimodal"]["audio_model_args"]["audio_encoding_args"]["frame_rate"]
    if codec.frame_rate != enc_frame_rate:
        raise RuntimeError(f"codec frame_rate {codec.frame_rate} != {enc_frame_rate} from params.json")
    if codec.downsample_factor != 1920:
        raise RuntimeError(f"codec downsample_factor {codec.downsample_factor} != 1920")
    if codec.decoder_window_sizes != [2, 4, 8, 16]:
        raise RuntimeError(
            f"codec decoder sliding windows {codec.decoder_window_sizes} != [2, 4, 8, 16]"
        )

    # Every substantial group of the checkpoint is reachable from this module.
    n_state = len(model.state_dict())
    expected = n_ckpt_tensors + (1 if cfg.tie_word_embeddings else 0)
    if n_state != expected:
        raise RuntimeError(f"model state_dict has {n_state} keys, expected {expected}")


def load_reference_model(model_id: str = _DEFAULT_MODEL_ID):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model.

    Built from the native Mistral ``consolidated.safetensors`` + ``params.json``
    this repo actually ships: there is no HF-format checkpoint, no config.json
    and no ``model_type``/``auto_map``.  The text backbone is converted onto
    ``MistralForCausalLM``; the acoustic transformer and the neural audio codec
    are the vendored vLLM-Omni reference implementations.  All 386 checkpoint
    tensors (4002.35 M params + buffers) are consumed.
    """
    torch.manual_seed(0)

    repo = _resolve_repo(model_id)
    with open(os.path.join(repo, "params.json")) as f:
        params = json.load(f)

    cfg = _build_text_config(params)

    multimodal = params["multimodal"]
    audio_model_args = dict(multimodal["audio_model_args"])
    codec_args = dict(multimodal["audio_tokenizer_args"])

    # n_decoding_steps is absent from params.json; vllm-omni defaults to 7.
    acoustic_args = dict(audio_model_args["acoustic_transformer_args"])
    acoustic_args.setdefault("n_decoding_steps", _DEFAULT_N_DECODING_STEPS)
    if acoustic_args["n_decoding_steps"] is None:
        acoustic_args["n_decoding_steps"] = _DEFAULT_N_DECODING_STEPS
    audio_model_args["acoustic_transformer_args"] = acoustic_args

    ckpt = os.path.join(repo, "consolidated.safetensors")
    with safe_open(ckpt, framework="pt") as reader:
        n_ckpt_tensors = len(reader.keys())
        all_keys = set(reader.keys())
        state, consumed = _remap(reader, cfg)

    unconsumed = sorted(all_keys - consumed)
    if unconsumed:
        raise RuntimeError(
            f"{len(unconsumed)} checkpoint tensors are not covered by this reference "
            f"(a partial reference is a silent failure): {unconsumed[:8]}"
        )

    # The backbone is built on meta so its 3.4 B params are never allocated
    # twice; the audio halves are built for real so their computed buffers
    # (alibi slopes, sinusoidal inv_freq, weight-norm parametrisations) are
    # materialised by the same arithmetic the reference uses.
    with torch.device("meta"):
        model = VoxtralTTSReferenceModel(cfg)
    with _skip_param_init():
        model.acoustic_transformer = FlowMatchingAudioTransformer(audio_model_args)
        model.audio_tokenizer = VoxtralTTSAudioTokenizer(
            codec_args, audio_model_args, cfg.hidden_size
        )

    expected_keys = set(model.state_dict().keys())
    got_keys = set(state.keys())
    missing = sorted(expected_keys - got_keys)
    unexpected = sorted(got_keys - expected_keys)
    # rotary_emb's inv_freq is non-persistent and rebuilt below.
    missing = [k for k in missing if not k.startswith("model.rotary_emb.")]
    if missing or unexpected:
        raise RuntimeError(
            f"consolidated -> reference remap incomplete: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )

    model.load_state_dict(state, strict=False, assign=True)

    _rebuild_nonpersistent_buffers(model, cfg)
    model.tie_weights()
    model.eval()
    model.requires_grad_(False)

    _verify(model, cfg, params, n_ckpt_tensors)
    return model


if __name__ == "__main__":
    m = load_reference_model()
    print(type(m).__name__, "params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    with torch.no_grad():
        out = m(input_ids=torch.tensor([[1, 35, 4380, 1395, 1261]]))
    print("logits", tuple(out.logits.shape))
