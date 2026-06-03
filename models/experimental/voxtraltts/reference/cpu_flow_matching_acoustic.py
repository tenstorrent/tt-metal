# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference: Voxtral flow-matching acoustic transformer (vLLM-Omni compatible, no vLLM import).

Use this as the **golden** for TT acoustic work: same checkpoint keys as ``acoustic_transformer.*``,
same math as ``FlowMatchingAudioTransformer`` (bidirectional GQA **without RoPE**). Pair with
``tt/acoustic_model.py`` ``predict_velocity`` for PCC tests.

``tt/`` stays TTNN-only; this module is PyTorch and lives under ``reference/`` with
``functional.py`` and other CPU references.

See:
https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts_audio_generation.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn import RMSNorm
except ImportError:
    RMSNorm = None


def _rms_norm_cls(dim: int, eps: float):
    if RMSNorm is not None:
        return RMSNorm(dim, eps=eps)

    class _ManualRMSNorm(nn.Module):
        def __init__(self, d: int, e: float):
            super().__init__()
            self.eps = e
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(v + self.eps)
            return x * self.weight

    return _ManualRMSNorm(dim, eps)


class AudioSpecialTokens(str, Enum):
    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [AudioSpecialTokens.empty_audio, AudioSpecialTokens.end_audio]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


@dataclass
class AcousticTransformerArgs:
    input_dim: int
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5
    n_decoding_steps: int | None = None


@dataclass
class MultimodalAudioModelArgs:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(self, pad_to_multiple: int | None = 128, include_special_tokens: bool = True) -> list[int]:
        def _round_up(n: int, m: int) -> int:
            return m * ((n + m - 1) // m)

        result: list[int] = []
        for i, cb_size in enumerate(self.codebook_sizes):
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up(cb_size, pad_to_multiple)
            result.append(cb_size)
        return result


def _from_nested_dict(cls: Any, d: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return d
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]
        if is_dataclass(field_type) and isinstance(value, dict):
            value = _from_nested_dict(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
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


class BidirectionalAttention(nn.Module):
    """GQA attention without RoPE (matches vLLM-Omni acoustic block)."""

    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.layer_id = layer_id

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

    def _native_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        key, value = repeat_kv(xk, xv, repeats=self.repeats)
        output = self._native_attention(query=xq, key=key, value=value)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = _rms_norm_cls(args.dim, eps=args.norm_eps)
        self.ffn_norm = _rms_norm_cls(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(-math.log(theta) * torch.arange(dim // 2).float() / (dim // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq.to(t.dtype))
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformerRef(nn.Module):
    """Same tensor names as checkpoint ``acoustic_transformer.*`` and vLLM ``FlowMatchingAudioTransformer``."""

    def __init__(self, audio_model_args: dict[str, Any]) -> None:
        super().__init__()
        ama = dict(audio_model_args)
        at = ama.get("acoustic_transformer_args")
        if isinstance(at, dict):
            ama["acoustic_transformer_args"] = AcousticTransformerArgs(**at)
        self.model_args: MultimodalAudioModelArgs = _from_nested_dict(MultimodalAudioModelArgs, ama)
        args = self.model_args.acoustic_transformer_args
        assert isinstance(args, AcousticTransformerArgs)
        self.acoustic_transformer_args = args

        if args.n_decoding_steps is None:
            args.n_decoding_steps = 8

        self.num_non_acoustic_embeddings = 1
        acoustic_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=None, include_special_tokens=False)[1:]
        assert len(set(acoustic_sizes)) == 1
        self.acoustic_embeddings_levels = acoustic_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        self.sigma = args.sigma
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, args.n_decoding_steps + 1),
            persistent=False,
        )

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        name, loaded_weight = weight
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        if name in params:
            params[name].data.copy_(loaded_weight)
        elif name in buffers:
            buffers[name].copy_(loaded_weight)
        return name

    def _init_audio_embeddings_layer(self) -> None:
        args = self.acoustic_transformer_args
        self.time_embedding = TimeEmbedding(args.dim)
        input_dim = self.acoustic_embeddings_dim
        self.input_projection = nn.Linear(input_dim, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)

    def _init_output_layer(self) -> None:
        args = self.acoustic_transformer_args
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            args.dim,
            padded_codebook_sizes[0],
            bias=args.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        args = self.acoustic_transformer_args
        self.layers_ids = list(range(args.n_layers))
        self.layers = nn.ModuleDict({str(i): AcousticTransformerBlock(i, args) for i in self.layers_ids})
        self.norm = _rms_norm_cls(args.dim, eps=args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h

    def decode_one_frame(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        cfg_alpha: torch.Tensor,
        *,
        collect_fm_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = semantic_code.shape[0]
        should_decode = semantic_code != self._end_audio_token_id

        x_0 = torch.randn(B, self.model_args.n_acoustic_codebook, device=llm_hidden.device, dtype=llm_hidden.dtype)
        x_0 = self._noise_scale * x_0

        timesteps = self._timesteps.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        llm_hidden_zero = torch.zeros_like(llm_hidden)
        ca = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        if ca.dim() == 0:
            cfg_alpha = ca.reshape(1, 1).expand(B, 1)
        elif ca.dim() == 1:
            cfg_alpha = ca.unsqueeze(1)
        else:
            cfg_alpha = ca

        sampled = x_0
        fm_debug: dict[str, torch.Tensor] | None = {} if collect_fm_debug else None
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)

            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            if collect_fm_debug and i == 0:
                v_all, step_debug = self._predict_velocity_debug(
                    x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched
                )
                fm_debug = step_debug
            else:
                v_all = self._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t

            sampled = sampled + v_t * dt

        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        codes = output_codes + len(AudioSpecialTokens.all_special_tokens())
        if collect_fm_debug:
            assert fm_debug is not None
            # Debug-only (numerics unchanged): expose the continuous pre-round FSQ value whose
            # ``round()`` yields the acoustic codes, for numerical-accuracy PCC / round-flip analysis.
            fm_debug["sampled"] = sampled.float()
            fm_debug["scaled_x"] = scaled_x.float()
            return codes, fm_debug
        return codes

    def _predict_velocity_debug(
        self,
        x_t: torch.Tensor,
        llm_output: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        debug: dict[str, torch.Tensor] = {}
        time_dtype = self.time_projection.weight.dtype
        llm_dtype = self.llm_projection.weight.dtype
        input_dtype = self.input_projection.weight.dtype

        t_emb_p = self.time_projection(t_emb.to(dtype=time_dtype)).to(dtype=input_dtype)
        llm_p = self.llm_projection(llm_output.to(dtype=llm_dtype)).to(dtype=input_dtype)
        x_t_p = x_t.to(dtype=input_dtype)
        p0 = self.input_projection(x_t_p.unsqueeze(1))
        debug["proj_input"] = p0.float()
        debug["proj_time"] = t_emb_p.unsqueeze(1).float()
        debug["proj_llm"] = llm_p.unsqueeze(1).float()

        h = torch.cat([p0, t_emb_p.unsqueeze(1), llm_p.unsqueeze(1)], dim=1)
        debug["concat_input"] = h.unsqueeze(1).float()

        for i in self.layers_ids:
            layer = self.layers[str(i)]
            n1 = layer.attention_norm(h)
            debug[f"layer{i}.attn_norm"] = n1.unsqueeze(1).float()
            a = layer.attention(n1)
            debug[f"layer{i}.attn_out"] = a.unsqueeze(1).float()
            h = h + a
            debug[f"layer{i}.post_attn"] = h.unsqueeze(1).float()
            n2 = layer.ffn_norm(h)
            debug[f"layer{i}.ffn_norm"] = n2.unsqueeze(1).float()
            f = layer.feed_forward(n2)
            debug[f"layer{i}.ffn_out"] = f.unsqueeze(1).float()
            h = h + f
            debug[f"layer{i}.post_ffn"] = h.unsqueeze(1).float()

        h = self.norm(h)
        debug["final_norm"] = h.unsqueeze(1).float()
        v = self.acoustic_codebook_output(h[:, 0, :])
        debug["velocity"] = v.unsqueeze(1).unsqueeze(1).float()
        return v.float(), debug

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        llm_output: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Keep input dtypes aligned with projection weights to avoid F.linear dtype mismatch.
        time_dtype = self.time_projection.weight.dtype
        llm_dtype = self.llm_projection.weight.dtype
        input_dtype = self.input_projection.weight.dtype

        t_emb = self.time_projection(t_emb.to(dtype=time_dtype)).to(dtype=input_dtype)
        llm_output = self.llm_projection(llm_output.to(dtype=llm_dtype)).to(dtype=input_dtype)
        x_t = x_t.to(dtype=input_dtype)

        acoustic_and_semantic_embeddings = [
            self.input_projection(x_t.unsqueeze(1)),
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ]
        acoustic_transformer_inputs = torch.cat(acoustic_and_semantic_embeddings, dim=1)

        attn_output = self.forward_attention_layers(acoustic_transformer_inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(-1, acoustic_transformer_inputs.shape[1], final_hidden.shape[-1])
        return self.acoustic_codebook_output(final_hidden[:, 0, :])

    def forward(
        self,
        llm_hidden: torch.Tensor,
        cfg_alpha: torch.Tensor,
        *,
        return_debug: bool = False,
        collect_semantic_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w_dtype = self.semantic_codebook_output.weight.dtype
        semantic_logit = self.semantic_codebook_output(llm_hidden.to(dtype=w_dtype)).float()
        semantic_logit[:, self._empty_audio_token_id] = float("-inf")
        semantic_logit[
            :, (len(AudioSpecialTokens.all_special_tokens()) + self.model_args.semantic_codebook_size) :
        ] = float("-inf")

        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)
        need_debug = return_debug or collect_semantic_logits
        debug: dict[str, torch.Tensor] | None = {} if need_debug else None
        if debug is not None:
            debug["semantic_logits"] = semantic_logit

        if return_debug:
            frame_out = self.decode_one_frame(
                semantic_code.squeeze(1),
                llm_hidden.to(dtype=self.llm_projection.weight.dtype),
                cfg_alpha=cfg_alpha,
                collect_fm_debug=True,
            )
            assert isinstance(frame_out, tuple)
            acoustic_codes, fm_debug = frame_out
            debug.update({f"fm.{k}": v for k, v in fm_debug.items()})
        else:
            acoustic_codes = self.decode_one_frame(
                semantic_code.squeeze(1),
                llm_hidden.to(dtype=self.llm_projection.weight.dtype),
                cfg_alpha=cfg_alpha,
            )

        codes = torch.cat([semantic_code, acoustic_codes], dim=1)
        if need_debug:
            return codes, debug
        return codes


def build_audio_model_args_from_voxtral_config(cfg) -> dict[str, Any]:
    """Build ``audio_model_args`` dict for :class:`FlowMatchingAudioTransformerRef` from ``VoxtralConfig``."""
    am = cfg.audio_model_args
    at = am.acoustic_transformer_args
    acoustic_transformer_args = {
        "input_dim": at.input_dim,
        "dim": at.dim,
        "n_layers": at.n_layers,
        "head_dim": at.head_dim,
        "hidden_dim": at.hidden_dim,
        "n_heads": at.n_heads,
        "n_kv_heads": at.n_kv_heads,
        "use_biases": at.use_biases,
        "norm_eps": at.sigma,
        "sigma": at.sigma,
        "n_decoding_steps": 8,
    }
    return {
        "semantic_codebook_size": am.semantic_codebook_size,
        "acoustic_codebook_size": am.acoustic_codebook_size,
        "n_acoustic_codebook": am.n_acoustic_codebook,
        "acoustic_transformer_args": acoustic_transformer_args,
    }
