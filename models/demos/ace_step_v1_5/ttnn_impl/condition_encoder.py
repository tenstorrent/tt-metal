# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN ACE condition encoder for text-to-music / instrumental demos.

This ports the quality-critical structure of ``AceStepConditionEncoder`` for the
fast-preprocess path:

* ``encoder.text_projector``
* lyric encoder (8 transformer layers)
* timbre encoder (4 transformer layers)
* HF-compatible packed sequence order: lyric, timbre, valid text, padded text

The official 5 Hz LM / handler can still produce the payload tensors; this module
replaces the final Torch ``prepare_condition`` encoder/context assembly.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

import ttnn

from .math_perf_env import (
    ace_step_cond_linear_perf_enabled,
    ace_step_cond_linear_program_config,
    ace_step_init_hifi2_linear_compute_kernel_config,
    ace_step_linear_l1_memory_config,
    ace_step_reshape_kwargs,
)
from .qwen3_embedding_encoder import Qwen3EmbeddingEncoderConfig, _TtQwen3EncoderLayer
from .text_projector import TtAceStepTextProjector, load_text_projector_weight_numpy


def load_condition_weights_np(safetensors_path: str) -> Dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    prefixes = (
        "encoder.text_projector.",
        "encoder.lyric_encoder.",
        "encoder.timbre_encoder.",
        "null_condition_emb",
    )
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k.startswith(prefixes):
                out[k] = sf.get_tensor(k).detach().to(torch.float32).cpu().numpy()
    return out


def _as_weight(weights_np: Dict[str, np.ndarray], key: str, *, device, dtype, mem, mapper, row_major: bool = False):
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    return ttnn.as_tensor(
        weights_np[key],
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=mem,
        mesh_mapper=mapper,
    )


def _rope_cos_sin_np(*, max_seq_len: int, head_dim: int, rope_theta: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(int(max_seq_len), dtype=np.float32)
    dim = np.arange(0, int(head_dim), 2, dtype=np.float32)
    inv_freq = 1.0 / (float(rope_theta) ** (dim / float(head_dim)))
    freqs = np.einsum("i,j->ij", pos, inv_freq, dtype=np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return (
        np.cos(emb).astype(np.float32).reshape(1, 1, int(max_seq_len), int(head_dim)),
        np.sin(emb).astype(np.float32).reshape(1, 1, int(max_seq_len), int(head_dim)),
    )


def _bidirectional_attn_bias_np(
    attention_mask_01: np.ndarray,
    *,
    sliding_window: int | None = None,
) -> np.ndarray:
    m = np.asarray(attention_mask_01, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    b, s = int(m.shape[0]), int(m.shape[1])
    i = np.arange(s, dtype=np.int32)[:, None]
    j = np.arange(s, dtype=np.int32)[None, :]
    keep = np.ones((s, s), dtype=bool)
    if sliding_window is not None:
        keep &= np.abs(i - j) <= int(sliding_window)
    keep = keep.reshape(1, 1, s, s) & (m.reshape(b, 1, 1, s) > 0.5)
    return np.where(keep, np.float32(0.0), np.float32(-1.0e9)).astype(np.float32)


def _to_numpy_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().to(dtype=x.dtype).float().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _to_numpy_mask(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class _TtAceStepTinyEncoder:
    """ACE lyric/timbre encoder, including the S=1 dummy-token fast path."""

    def __init__(
        self,
        *,
        weights_np: Dict[str, np.ndarray],
        prefix: str,
        input_dim: int,
        num_layers: int,
        max_seq_len: int,
        sliding_window: int | None,
        device,
        cfg: Qwen3EmbeddingEncoderConfig,
        dtype,
        mem,
        mapper,
        linear_compute_kernel_config=None,
        linear_perf: bool = False,
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.mem = mem
        self.input_dim = int(input_dim)
        self.hidden_size = int(cfg.hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self._linear_ck = linear_compute_kernel_config
        self._linear_perf = bool(linear_perf)
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._embed_pc_cache: dict = {}
        self.embed_w = _as_weight(
            weights_np, f"{prefix}.embed_tokens.weight", device=device, dtype=dtype, mem=mem, mapper=mapper
        )
        self.embed_b = ttnn.as_tensor(
            weights_np[f"{prefix}.embed_tokens.bias"].reshape(1, 1, 1, -1),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.norm_w = _as_weight(
            weights_np, f"{prefix}.norm.weight", device=device, dtype=dtype, mem=mem, mapper=mapper
        )
        _layer_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            linear_perf=linear_perf,
            activation_l1_memory_config=activation_l1_memory_config,
            linear_output_l1_memory_config=linear_output_l1_memory_config,
        )
        self.layers = [
            _TtQwen3EncoderLayer(
                device=device,
                weights_np=weights_np,
                prefix=f"{prefix}.layers.{i}",
                cfg=cfg,
                dtype=dtype,
                mem=mem,
                mapper=mapper,
                **_layer_kw,
            )
            for i in range(int(num_layers))
        ]
        self.layer_attention_types = [
            "sliding_attention" if (i % 2 == 0) else "full_attention" for i in range(int(num_layers))
        ]
        cos_np, sin_np = _rope_cos_sin_np(
            max_seq_len=self.max_seq_len,
            head_dim=int(cfg.head_dim),
            rope_theta=float(cfg.rope_theta),
        )
        self.cos_tt = ttnn.as_tensor(
            cos_np, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper
        )
        self.sin_tt = ttnn.as_tensor(
            sin_np, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper
        )

    def _l1_activation(self, t: ttnn.Tensor) -> ttnn.Tensor:
        if self._act_l1 is None:
            return t
        return ttnn.to_memory_config(t, self._act_l1)

    def _embed_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        if self._linear_perf:
            key = (int(batch_size), int(seq_len), int(self.input_dim))
            pc = self._embed_pc_cache.get(key)
            if pc is None:
                pc = ace_step_cond_linear_program_config(
                    self.device,
                    seq_len=int(seq_len),
                    in_dim=int(self.input_dim),
                    out_dim=self.hidden_size,
                    batch_size=int(batch_size),
                )
                if pc is not None:
                    self._embed_pc_cache[key] = pc
            if pc is not None:
                kw["program_config"] = pc
        if self._linear_out_l1 is not None:
            kw["memory_config"] = self._linear_out_l1
        return kw

    def __call__(
        self,
        inputs_f32: np.ndarray | None = None,
        attention_mask_01: np.ndarray | None = None,
        *,
        output_first_token: bool = False,
    ) -> ttnn.Tensor:
        if inputs_f32 is None:
            x_np = np.zeros((1, 1, self.input_dim), dtype=np.float32)
        else:
            x_np = np.asarray(inputs_f32, dtype=np.float32)
        if x_np.ndim != 3:
            raise ValueError(f"encoder input must be [B,S,D], got {x_np.shape}")
        b, s, d = int(x_np.shape[0]), int(x_np.shape[1]), int(x_np.shape[2])
        if d != self.input_dim:
            raise ValueError(f"expected input dim {self.input_dim}, got {d}")
        if s > self.max_seq_len:
            raise ValueError(f"sequence length {s} exceeds max_seq_len={self.max_seq_len}")
        mask_np = (
            np.ones((b, s), dtype=np.float32)
            if attention_mask_01 is None
            else np.asarray(attention_mask_01, dtype=np.float32).reshape(b, s)
        )
        _sr = ace_step_reshape_kwargs(ttnn)
        x = ttnn.as_tensor(
            x_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        x = ttnn.reshape(x, (b, 1, s, self.input_dim), **_sr)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self._l1_activation(x)
        lin_embed = self._embed_linear_kwargs(batch_size=b, seq_len=s)
        h = ttnn.linear(x, self.embed_w, bias=self.embed_b, transpose_b=True, **lin_embed)
        cos_tt = ttnn.slice(self.cos_tt, (0, 0, 0, 0), (1, 1, s, int(self.cos_tt.shape[-1])))
        sin_tt = ttnn.slice(self.sin_tt, (0, 0, 0, 0), (1, 1, s, int(self.sin_tt.shape[-1])))
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        bias_cache: dict[str, ttnn.Tensor] = {}
        for layer, attn_type in zip(self.layers, self.layer_attention_types):
            use_sliding = attn_type == "sliding_attention" and self.sliding_window is not None
            cache_key = "sliding" if use_sliding else "full"
            if cache_key not in bias_cache:
                bias_np = _bidirectional_attn_bias_np(
                    mask_np,
                    sliding_window=self.sliding_window if use_sliding else None,
                )
                bias_cache[cache_key] = ttnn.as_tensor(
                    bias_np,
                    device=self.device,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.mem,
                    mesh_mapper=mapper,
                )
            h = layer(h, cos_tt, sin_tt, bias_cache[cache_key])
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        h = ttnn.rms_norm(h, weight=self.norm_w, epsilon=float(1e-6), memory_config=self.mem)
        if output_first_token:
            h = ttnn.slice(h, (0, 0, 0, 0), (b, 1, 1, self.hidden_size))
            return ttnn.reshape(h, (b, 1, self.hidden_size), **_sr)
        return ttnn.reshape(h, (b, s, self.hidden_size), **_sr)


class TtAceStepInstrumentalConditionEncoder:
    """Build TTNN ``encoder_hidden_states`` for the fast instrumental path."""

    def __init__(self, *, device, checkpoint_safetensors_path: str, dtype=None) -> None:
        self.device = device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None)
        if self.dtype is None:
            raise RuntimeError("bfloat16 required")
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        self.weights_np = load_condition_weights_np(str(checkpoint_safetensors_path))
        init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
        cond_linear_perf = ace_step_cond_linear_perf_enabled()
        sdpa_compute_kernel_config = None
        linear_compute_kernel_config = None
        l1_mc = ace_step_linear_l1_memory_config(ttnn) if cond_linear_perf else None
        if cond_linear_perf:
            linear_compute_kernel_config = ace_step_init_hifi2_linear_compute_kernel_config(device)
            if callable(init_ck):
                sdpa_compute_kernel_config = init_ck(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
        elif callable(init_ck):
            sdpa_compute_kernel_config = init_ck(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            linear_compute_kernel_config = init_ck(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        sdpa_program_config = None
        if hasattr(device, "compute_with_storage_grid_size") and hasattr(ttnn, "SDPAProgramConfig"):
            sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
        cfg = Qwen3EmbeddingEncoderConfig(
            hidden_size=2048,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=6144,
            max_seq_len=1,
            rope_theta=1_000_000.0,
        )
        self.text_projector = TtAceStepTextProjector(
            device=device,
            weight_f32_numpy=load_text_projector_weight_numpy(str(checkpoint_safetensors_path)),
            weights_dtype=self.dtype,
            weight_memory_config=self.mem,
        )
        _enc_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            linear_perf=cond_linear_perf,
            activation_l1_memory_config=l1_mc,
            linear_output_l1_memory_config=l1_mc,
        )
        self.lyric_encoder = _TtAceStepTinyEncoder(
            weights_np=self.weights_np,
            prefix="encoder.lyric_encoder",
            input_dim=1024,
            num_layers=8,
            max_seq_len=256,
            sliding_window=128,
            device=device,
            cfg=cfg,
            dtype=self.dtype,
            mem=self.mem,
            mapper=mapper,
            **_enc_kw,
        )
        for layer in self.lyric_encoder.layers:
            layer._sdpa_compute_kernel_config = sdpa_compute_kernel_config
            layer._sdpa_program_config = sdpa_program_config
        self.timbre_encoder = _TtAceStepTinyEncoder(
            weights_np=self.weights_np,
            prefix="encoder.timbre_encoder",
            input_dim=64,
            num_layers=4,
            max_seq_len=750,
            sliding_window=128,
            device=device,
            cfg=cfg,
            dtype=self.dtype,
            mem=self.mem,
            mapper=mapper,
            **_enc_kw,
        )
        for layer in self.timbre_encoder.layers:
            layer._sdpa_compute_kernel_config = sdpa_compute_kernel_config
            layer._sdpa_program_config = sdpa_program_config
        self.null_condition_emb = ttnn.as_tensor(
            self.weights_np["null_condition_emb"],
            device=device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

    def forward(
        self, text_hidden_b1sd: ttnn.Tensor, attention_mask_01: np.ndarray
    ) -> tuple[ttnn.Tensor, np.ndarray, ttnn.Tensor]:
        text_proj = self.text_projector.forward_from_hidden(text_hidden_b1sd, activation_dtype=self.dtype)
        b, s, d = int(text_proj.shape[0]), int(text_proj.shape[1]), int(text_proj.shape[2])
        if b != 1:
            raise ValueError("TtAceStepInstrumentalConditionEncoder currently supports B=1 only.")
        attn = np.asarray(attention_mask_01, dtype=np.float32).reshape(1, -1)
        if int(attn.shape[1]) != s:
            raise ValueError(f"attention_mask length {attn.shape[1]} != text sequence length {s}")
        valid = int(attn.sum())

        lyric = self.lyric_encoder()
        timbre = self.timbre_encoder()
        text_valid = ttnn.slice(text_proj, (0, 0, 0), (1, valid, d)) if valid > 0 else None
        parts = [lyric, timbre]
        if text_valid is not None:
            parts.append(text_valid)
        if valid < s:
            text_pad = ttnn.slice(text_proj, (0, valid, 0), (1, s, d))
            parts.append(text_pad)
        else:
            text_pad = None
        enc = ttnn.concat([ttnn.to_layout(p, ttnn.TILE_LAYOUT) for p in parts], dim=1)
        enc_mask = np.concatenate(
            [
                np.ones((1, 2 + valid), dtype=np.float32),
                np.zeros((1, s - valid), dtype=np.float32),
            ],
            axis=1,
        )
        for t in (text_proj, text_valid, text_pad):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return enc, enc_mask, self.null_condition_emb

    def forward_payload(self, payload: dict) -> tuple[ttnn.Tensor, np.ndarray, ttnn.Tensor, ttnn.Tensor]:
        """TTNN equivalent of ACE ``prepare_condition`` for the official handler payload.

        Batch size 1 is the demo path used by ``run_prompt_to_wav.py``.
        """
        text_np = _to_numpy_f32(payload["text_hidden_states"])
        text_mask_np = _to_numpy_mask(payload["text_attention_mask"])
        lyric_np = _to_numpy_f32(payload["lyric_hidden_states"])
        lyric_mask_np = _to_numpy_mask(payload["lyric_attention_mask"])
        timbre_np = _to_numpy_f32(payload["refer_audio_acoustic_hidden_states_packed"])
        order_mask_np = _to_numpy_mask(payload["refer_audio_order_mask"]).astype(np.int64).reshape(-1)
        if text_np.shape[0] != 1 or lyric_np.shape[0] != 1:
            raise ValueError("TTNN official condition encoder currently supports demo batch size B=1 only.")
        if timbre_np.shape[0] != 1 or order_mask_np.shape != (1,) or int(order_mask_np[0]) != 0:
            raise ValueError("TTNN official condition encoder currently supports one packed timbre segment.")

        text_proj = self.text_projector.forward(text_np, activation_dtype=self.dtype)
        lyric = self.lyric_encoder(lyric_np, lyric_mask_np)
        timbre = self.timbre_encoder(timbre_np, None, output_first_token=True)

        lyric_valid = int(np.asarray(lyric_mask_np, dtype=np.float32).sum())
        text_valid = int(np.asarray(text_mask_np, dtype=np.float32).sum())
        parts = []
        mask_parts = []
        if lyric_valid > 0:
            parts.append(ttnn.slice(lyric, (0, 0, 0), (1, lyric_valid, self.lyric_encoder.hidden_size)))
            mask_parts.append(np.ones((1, lyric_valid), dtype=np.float32))
        parts.append(timbre)
        mask_parts.append(np.ones((1, 1), dtype=np.float32))
        if text_valid > 0:
            parts.append(ttnn.slice(text_proj, (0, 0, 0), (1, text_valid, self.lyric_encoder.hidden_size)))
            mask_parts.append(np.ones((1, text_valid), dtype=np.float32))
        if lyric_valid < int(lyric_np.shape[1]):
            parts.append(
                ttnn.slice(lyric, (0, lyric_valid, 0), (1, int(lyric_np.shape[1]), self.lyric_encoder.hidden_size))
            )
            mask_parts.append(np.zeros((1, int(lyric_np.shape[1]) - lyric_valid), dtype=np.float32))
        if text_valid < int(text_np.shape[1]):
            parts.append(
                ttnn.slice(text_proj, (0, text_valid, 0), (1, int(text_np.shape[1]), self.lyric_encoder.hidden_size))
            )
            mask_parts.append(np.zeros((1, int(text_np.shape[1]) - text_valid), dtype=np.float32))
        enc = ttnn.concat([ttnn.to_layout(p, ttnn.TILE_LAYOUT) for p in parts], dim=1)
        enc_mask = np.concatenate(mask_parts, axis=1).astype(np.float32)

        src_np = _to_numpy_f32(payload["src_latents"])
        chunk_np = _to_numpy_mask(payload["chunk_mask"]).astype(np.float32)
        is_covers_np = _to_numpy_mask(payload["is_covers"]).reshape(-1) > 0
        hints = payload.get("precomputed_lm_hints_25Hz")
        if hints is not None and bool(is_covers_np[0]):
            src_np = _to_numpy_f32(hints)[:, : src_np.shape[1], :]
        ctx_np = np.concatenate([src_np.astype(np.float32), chunk_np.astype(np.float32)], axis=-1)
        ctx = ttnn.as_tensor(
            ctx_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        return enc, enc_mask, ctx, self.null_condition_emb


__all__ = ["TtAceStepInstrumentalConditionEncoder", "load_condition_weights_np"]
