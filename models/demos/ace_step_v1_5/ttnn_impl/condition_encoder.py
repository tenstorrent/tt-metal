# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN ACE condition encoder for text-to-music / instrumental demos.

This ports the quality-critical structure of ``AceStepConditionEncoder`` for the
handler / instrumental condition path:

* ``encoder.text_projector``
* lyric encoder (8 transformer layers)
* timbre encoder (4 transformer layers)
* HF-compatible packed sequence order: lyric, timbre, valid text, padded text

The official 5 Hz LM / handler can still produce the payload tensors; this module
replaces the final Torch ``prepare_condition`` encoder/context assembly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

import ttnn

from .math_perf_env import (
    ace_step_concat_kwargs,
    ace_step_cond_linear_program_config,
    ace_step_cond_rms_norm_kwargs,
    ace_step_ensure_row_major_layout,
    ace_step_ensure_tile_layout,
    ace_step_init_cond_linear_compute_kernel_config,
    ace_step_init_cond_sdpa_compute_kernel_config,
    ace_step_linear_l1_memory_config,
    ace_step_linear_weight_dtype,
    ace_step_reshape_kwargs,
    ace_step_rms_norm_block_sharded,
    ace_step_sdpa_mask_memory_config,
    ace_step_to_layout_kwargs,
    ace_step_upload_f32_np_as_bf16_tile,
)
from .qwen3_embedding_encoder import Qwen3EmbeddingEncoderConfig, _TtQwen3EncoderLayer
from .text_projector import TtAceStepTextProjector, load_text_projector_weight_numpy


def load_condition_weights_np(safetensors_path: str) -> Dict[str, np.ndarray]:
    from models.demos.ace_step_v1_5.weight_cache import load_prefix_weights_np

    return load_prefix_weights_np(
        safetensors_path,
        (
            "encoder.text_projector.",
            "encoder.lyric_encoder.",
            "encoder.timbre_encoder.",
            "null_condition_emb",
        ),
        component="condition-encoder-weights",
        tag="condition_encoder_np",
    )


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


def _cond_seq_for_concat(t: ttnn.Tensor) -> ttnn.Tensor:
    """``[B,1,S,H]`` TILE → ``[B,S,H]`` for ``ttnn.concat(..., dim=1)`` (single reshape at boundary)."""
    sh = t.shape
    if len(sh) == 4:
        b, one, s, h = int(sh[0]), int(sh[1]), int(sh[2]), int(sh[3])
        if one != 1:
            raise ValueError(f"expected [B,1,S,H] for condition concat, got {sh}")
        return ttnn.reshape(t, (b, s, h), **ace_step_reshape_kwargs(ttnn))
    return t


def _text_seq_len_from_payload(payload: dict) -> int:
    """Sequence length S for ``text_hidden_states`` (supports ``[B,S,D]`` or ``[B,1,S,D]``)."""
    text_np = _to_numpy_f32(payload["text_hidden_states"])
    if text_np.ndim == 4:
        return int(text_np.shape[2])
    if text_np.ndim == 3:
        return int(text_np.shape[1])
    raise ValueError(f"text_hidden_states must be [B,S,D] or [B,1,S,D], got {text_np.shape}")


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
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
        linear_weight_dtype=None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.mem = mem
        self.input_dim = int(input_dim)
        self.hidden_size = int(cfg.hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._rms_norm_kw = ace_step_cond_rms_norm_kwargs(ttnn, linear_output_l1_memory_config, device=device)
        self._embed_pc_cache: dict = {}
        _w_dtype = linear_weight_dtype if linear_weight_dtype is not None else dtype
        self.embed_w = _as_weight(
            weights_np,
            f"{prefix}.embed_tokens.weight",
            device=device,
            dtype=_w_dtype,
            mem=mem,
            mapper=mapper,
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
        self._rope_cache: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._attn_bias_cache: dict[tuple[str, int], ttnn.Tensor] = {}

    def _rope_tables_for_seq(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Cached RoPE cos/sin views for sequence length *s* (avoid per-forward slice ops)."""
        s = int(seq_len)
        cached = self._rope_cache.get(s)
        if cached is not None:
            return cached
        head_dim = int(self.cos_tt.shape[-1])
        cached = (
            ttnn.slice(self.cos_tt, (0, 0, 0, 0), (1, 1, s, head_dim)),
            ttnn.slice(self.sin_tt, (0, 0, 0, 0), (1, 1, s, head_dim)),
        )
        self._rope_cache[s] = cached
        return cached

    def _l1_activation(self, t: ttnn.Tensor) -> ttnn.Tensor:
        if self._act_l1 is None:
            return t
        return ttnn.to_memory_config(t, self._act_l1)

    def _embed_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
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
        x = ace_step_ensure_tile_layout(ttnn, x, l1_mc=self._act_l1)
        x = self._l1_activation(x)
        lin_embed = self._embed_linear_kwargs(batch_size=b, seq_len=s)
        h = ttnn.linear(x, self.embed_w, bias=self.embed_b, transpose_b=True, **lin_embed)
        cos_tt, sin_tt = self._rope_tables_for_seq(s)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        bias_cache: dict[str, ttnn.Tensor] = {}
        try:
            for layer, attn_type in zip(self.layers, self.layer_attention_types):
                use_sliding = attn_type == "sliding_attention" and self.sliding_window is not None
                cache_key = ("sliding" if use_sliding else "full", s)
                if cache_key not in bias_cache:
                    bias_np = _bidirectional_attn_bias_np(
                        mask_np,
                        sliding_window=self.sliding_window if use_sliding else None,
                    )
                    if cache_key not in self._attn_bias_cache:
                        self._attn_bias_cache[cache_key] = ace_step_upload_f32_np_as_bf16_tile(
                            ttnn,
                            bias_np,
                            device=self.device,
                            dtype=self.dtype,
                            memory_config=ace_step_sdpa_mask_memory_config(ttnn) or self.mem,
                            mesh_mapper=mapper,
                        )
                    bias_cache[cache_key] = self._attn_bias_cache[cache_key]
                h = layer(h, cos_tt, sin_tt, bias_cache[cache_key])
        finally:
            # ``cos_tt`` / ``sin_tt`` are views into persistent ``self.cos_tt`` / ``self.sin_tt``;
            # ``bias_cache`` entries point at ``self._attn_bias_cache`` — do not deallocate.
            pass
        h = ace_step_ensure_tile_layout(ttnn, h, l1_mc=self._linear_out_l1)
        h = ace_step_rms_norm_block_sharded(
            ttnn,
            h,
            self.norm_w,
            float(1e-6),
            device=self.device,
            l1_mc=self._linear_out_l1,
            compute_kernel_config=self._rms_norm_kw.get("compute_kernel_config"),
        )
        if output_first_token:
            return ttnn.slice(h, (0, 0, 0, 0), (b, 1, 1, self.hidden_size))
        return h

    def forward_device(
        self,
        x_bsd: ttnn.Tensor,
        bias_full: ttnn.Tensor,
        bias_sliding: ttnn.Tensor,
        *,
        output_first_token: bool = False,
    ) -> ttnn.Tensor:
        """Trace-safe forward: ``x_bsd`` ``[B,S,D_in]`` on device; attn biases pre-uploaded."""
        b = int(x_bsd.shape[0])
        s = int(x_bsd.shape[1])
        d = int(x_bsd.shape[2])
        if d != self.input_dim:
            raise ValueError(f"expected input dim {self.input_dim}, got {d}")
        if s > self.max_seq_len:
            raise ValueError(f"sequence length {s} exceeds max_seq_len={self.max_seq_len}")
        _sr = ace_step_reshape_kwargs(ttnn)
        # Use ensure_row_major_layout to skip the Untilize when x_bsd is already ROW_MAJOR
        # (avoids a redundant UntilizeDeviceOperation on each timbre forward pass).
        x = ace_step_ensure_row_major_layout(ttnn, x_bsd)
        x = ttnn.reshape(x, (b, 1, s, self.input_dim), **_sr)
        x = ace_step_ensure_tile_layout(ttnn, x, l1_mc=self._act_l1)
        x = self._l1_activation(x)
        lin_embed = self._embed_linear_kwargs(batch_size=b, seq_len=s)
        h = ttnn.linear(x, self.embed_w, bias=self.embed_b, transpose_b=True, **lin_embed)
        cos_tt, sin_tt = self._rope_tables_for_seq(s)
        for layer, attn_type in zip(self.layers, self.layer_attention_types):
            use_sliding = attn_type == "sliding_attention" and self.sliding_window is not None
            bias_tt = bias_sliding if use_sliding else bias_full
            h = layer(h, cos_tt, sin_tt, bias_tt)
        h = ace_step_ensure_tile_layout(ttnn, h, l1_mc=self._linear_out_l1)
        h = ace_step_rms_norm_block_sharded(
            ttnn,
            h,
            self.norm_w,
            float(1e-6),
            device=self.device,
            l1_mc=self._linear_out_l1,
        )
        if output_first_token:
            return ttnn.slice(h, (0, 0, 0, 0), (b, 1, 1, self.hidden_size))
        return h

    def make_attn_bias_dev(self, attention_mask_01: np.ndarray, *, use_sliding: bool) -> ttnn.Tensor:
        """Host mask → device bias tile (call outside trace capture; refresh on CQ1 staging)."""
        mask_np = np.asarray(attention_mask_01, dtype=np.float32).reshape(int(attention_mask_01.shape[0]), -1)
        s = int(mask_np.shape[1])
        bias_np = _bidirectional_attn_bias_np(
            mask_np,
            sliding_window=self.sliding_window if use_sliding else None,
        )
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        s = int(mask_np.shape[1])
        cache_key = ("sliding" if use_sliding else "full", s)
        if cache_key not in self._attn_bias_cache:
            self._attn_bias_cache[cache_key] = ace_step_upload_f32_np_as_bf16_tile(
                ttnn,
                bias_np,
                device=self.device,
                dtype=self.dtype,
                memory_config=ace_step_sdpa_mask_memory_config(ttnn) or self.mem,
                mesh_mapper=mapper,
            )
        return self._attn_bias_cache[cache_key]


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
        linear_weight_dtype = ace_step_linear_weight_dtype(ttnn, self.dtype)
        linear_compute_kernel_config = ace_step_init_cond_linear_compute_kernel_config(device)
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        sdpa_compute_kernel_config = ace_step_init_cond_sdpa_compute_kernel_config(device)
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
            weights_dtype=linear_weight_dtype,
            weight_memory_config=self.mem,
            linear_compute_kernel_config=linear_compute_kernel_config,
            activation_l1_memory_config=l1_mc,
        )
        _enc_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            activation_l1_memory_config=l1_mc,
            linear_output_l1_memory_config=l1_mc,
            linear_weight_dtype=linear_weight_dtype,
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
        self.null_condition_emb = ttnn.as_tensor(
            self.weights_np["null_condition_emb"],
            device=device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

        # Instrumental-only path (e2e_model_tt + run_prompt_to_wav --ttnn-condition-embedding)
        # always invokes lyric_encoder() and timbre_encoder() with no args. Both encoders take
        # zero-filled dummy inputs (see _TtAceStepTinyEncoder.__call__) and produce a deterministic
        # constant output — 8 + 4 transformer layers' worth of compute that is identical on every
        # call. Pre-compute these once and reuse the device tensors in forward(), which:
        #   - eliminates ~12 layers of per-call work in the e2e pipeline (~5–10ms saved per prompt)
        #   - makes forward() trace-safe (no per-call ttnn.as_tensor of dummy x_np / bias_np)
        # `_lyric_const_tt` and `_timbre_const_tt` are persistent for the lifetime of this encoder
        # instance and intentionally never deallocated in forward().
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        self._lyric_const_tt = ace_step_ensure_tile_layout(
            ttnn, _cond_seq_for_concat(self.lyric_encoder()), l1_mc=l1_mc
        )
        self._timbre_const_tt = ace_step_ensure_tile_layout(
            ttnn, _cond_seq_for_concat(self.timbre_encoder()), l1_mc=l1_mc
        )

        # Lazy trace + 2CQ state for :meth:`forward_traced` (see perf test
        # ``test_condition_encoder_trace_2cq``). Freed by :meth:`release_trace`.
        self._trace_id: Optional[Any] = None
        self._persistent_text_hidden: Optional[ttnn.Tensor] = None
        self._text_hidden_host: Optional[ttnn.Tensor] = None
        self._persistent_enc_output: Optional[ttnn.Tensor] = None
        self._trace_op_event: Any = None
        self._trace_valid: Optional[int] = None

        # Official ``forward_payload`` trace + 2CQ (lyric + timbre + text + concat on CQ0).
        self._payload_trace_id: Optional[Any] = None
        self._payload_persistent_text: Optional[ttnn.Tensor] = None
        self._payload_text_host: Optional[ttnn.Tensor] = None
        self._payload_persistent_lyric: Optional[ttnn.Tensor] = None
        self._payload_persistent_timbre: Optional[ttnn.Tensor] = None
        self._payload_lyric_host: Optional[ttnn.Tensor] = None
        self._payload_timbre_host: Optional[ttnn.Tensor] = None
        self._payload_lyric_bias_full: Optional[ttnn.Tensor] = None
        self._payload_lyric_bias_sliding: Optional[ttnn.Tensor] = None
        self._payload_timbre_bias_full: Optional[ttnn.Tensor] = None
        self._payload_timbre_bias_sliding: Optional[ttnn.Tensor] = None
        self._payload_enc_mask_dev: Optional[ttnn.Tensor] = None
        self._payload_persistent_enc: Optional[ttnn.Tensor] = None
        self._payload_trace_op_event: Any = None
        self._payload_shape_key: Optional[tuple[int, ...]] = None
        self._payload_parts_persist: list[ttnn.Tensor] = []
        self._payload_part_hosts: list[ttnn.Tensor] = []
        self._payload_stage_write_event: Any = None
        self._cap_text_valid: int = 0
        self._cap_text_s: int = 0
        self._cap_lyric_valid: int = 0
        self._cap_lyric_s: int = 0
        self._cap_timbre_s: int = 0

        # Context latents ``concat([src_latents, chunk_mask], dim=-1)`` trace + 2CQ.
        self._ctx_trace_id: Optional[Any] = None
        self._ctx_persistent_src: Optional[ttnn.Tensor] = None
        self._ctx_persistent_chunk: Optional[ttnn.Tensor] = None
        self._ctx_persistent_out: Optional[ttnn.Tensor] = None
        self._ctx_src_host: Optional[ttnn.Tensor] = None
        self._ctx_chunk_host: Optional[ttnn.Tensor] = None
        self._ctx_trace_op_event: Any = None
        self._ctx_stage_write_event: Any = None
        self._ctx_frames_key: Optional[int] = None

    @staticmethod
    def _enc_mask_np(valid: int, seq_len: int) -> np.ndarray:
        v = int(valid)
        s = int(seq_len)
        return np.concatenate(
            [
                np.ones((1, 2 + v), dtype=np.float32),
                np.zeros((1, s - v), dtype=np.float32),
            ],
            axis=1,
        )

    def forward_device(self, text_hidden_b1sd: ttnn.Tensor, valid: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Trace-safe forward.

        ``text_hidden_b1sd``: ``[B, 1, S, D_text]`` device tensor (e.g. Qwen3 hidden states).
        ``valid``: number of valid (unpadded) text tokens; precomputed by the caller from the host
            attention mask via ``int(attn.sum())``.

        Returns ``(enc, null_condition_emb)`` — both device tensors. The host-side ``enc_mask``
        produced by the legacy :meth:`forward` is omitted here because it does not depend on any
        device computation (caller can build it from ``valid`` and the prompt seq length).

        All ops are device-only; safe to call inside ``begin_trace_capture``.
        """
        text_proj = self.text_projector.forward_from_hidden(text_hidden_b1sd, activation_dtype=self.dtype)
        b, s, d = int(text_proj.shape[0]), int(text_proj.shape[1]), int(text_proj.shape[2])
        if b != 1:
            raise ValueError("TtAceStepInstrumentalConditionEncoder currently supports B=1 only.")
        v = int(valid)
        if v < 0 or v > s:
            raise ValueError(f"valid must be in [0, {s}], got {v}")

        parts: list[ttnn.Tensor] = [self._lyric_const_tt, self._timbre_const_tt]
        text_valid = ttnn.slice(text_proj, (0, 0, 0), (1, v, d)) if v > 0 else None
        if text_valid is not None:
            parts.append(text_valid)
        text_pad = ttnn.slice(text_proj, (0, v, 0), (1, s, d)) if v < s else None
        if text_pad is not None:
            parts.append(text_pad)
        _tl = ace_step_to_layout_kwargs(ttnn)
        _ck = ace_step_concat_kwargs(ttnn)
        enc = ttnn.concat(
            [ace_step_ensure_tile_layout(ttnn, p) for p in parts],
            dim=1,
            **_ck,
        )
        for t in (text_proj, text_valid, text_pad):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except RuntimeError:
                    pass
        return enc, self.null_condition_emb

    def forward_traced(self, text_hidden_b1sd: ttnn.Tensor, valid: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Trace + 2CQ replay of :meth:`forward_device` for the fast instrumental path.

        Captures ``text_projector`` + slice + concat against persistent input/output buffers.
        Each replay refreshes the input buffer from *text_hidden_b1sd* via a host staging tensor
        on CQ 1, then ``execute_trace`` on CQ 0.

        Requires ``num_command_queues=2``. Output ``enc`` is a **persistent buffer** — do not
        deallocate; finish using it before the next :meth:`forward_traced` call.

        Re-captures automatically when *valid* changes (slice boundaries depend on it).
        """
        if not hasattr(ttnn, "begin_trace_capture") or not hasattr(ttnn, "execute_trace"):
            raise RuntimeError(
                "TtAceStepInstrumentalConditionEncoder.forward_traced requires trace support "
                "(begin_trace_capture / execute_trace)."
            )
        if int(text_hidden_b1sd.shape[0]) != 1:
            raise NotImplementedError(
                f"forward_traced supports batch_size=1 only (got {int(text_hidden_b1sd.shape[0])})."
            )
        v = int(valid)
        if self._trace_id is not None and self._trace_valid is not None and self._trace_valid != v:
            self._release_instrumental_trace()
        if self._trace_id is None:
            self._capture_trace(text_hidden_b1sd, v)
        return self._replay_trace(text_hidden_b1sd)

    def _capture_trace(self, text_hidden_b1sd: ttnn.Tensor, valid: int) -> None:
        """Warmup, allocate persistent buffers, capture trace on CQ 0."""
        enc_warm, _ = self.forward_device(text_hidden_b1sd, valid)
        ttnn.synchronize_device(self.device)
        try:
            ttnn.deallocate(enc_warm)
        except Exception:
            pass

        th = ttnn.to_torch(text_hidden_b1sd).contiguous()
        _l1_mc = ace_step_linear_l1_memory_config(ttnn) or self.mem
        self._text_hidden_host = ttnn.from_torch(
            th,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._persistent_text_hidden = ttnn.from_torch(
            th,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_l1_mc,
        )

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._persistent_enc_output, _ = self.forward_device(self._persistent_text_hidden, valid)
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._trace_valid = int(valid)
        self._trace_op_event = ttnn.record_event(self.device, 0)

    def _replay_trace(self, text_hidden_b1sd: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._trace_id is None or self._persistent_text_hidden is None or self._text_hidden_host is None:
            raise RuntimeError("TtAceStepInstrumentalConditionEncoder._replay_trace called before capture.")

        if hasattr(ttnn, "copy_device_to_host_tensor"):
            ttnn.wait_for_event(1, self._trace_op_event)
            ttnn.copy_device_to_host_tensor(text_hidden_b1sd, self._text_hidden_host, cq_id=1)
            ttnn.copy_host_to_device_tensor(self._text_hidden_host, self._persistent_text_hidden, cq_id=1)
        else:
            th = ttnn.to_torch(text_hidden_b1sd).contiguous()
            host_updated = ttnn.from_torch(
                th.to(dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.wait_for_event(1, self._trace_op_event)
            ttnn.copy_host_to_device_tensor(host_updated, self._persistent_text_hidden, cq_id=1)

        write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        self._trace_op_event = ttnn.record_event(self.device, 0)
        ttnn.synchronize_device(self.device)
        return self._persistent_enc_output, self.null_condition_emb

    def release_trace(self) -> None:
        """Release instrumental + official payload trace state. Safe to call repeatedly."""
        self._release_instrumental_trace()
        self._release_payload_trace()

    def _release_instrumental_trace(self) -> None:
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._trace_id)
            except Exception:
                pass
            self._trace_id = None
        for attr in ("_persistent_text_hidden", "_persistent_enc_output"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._text_hidden_host = None
        self._trace_op_event = None
        self._trace_valid = None

    def _release_payload_trace(self) -> None:
        if self._payload_trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._payload_trace_id)
            except Exception:
                pass
            self._payload_trace_id = None
        if self._payload_persistent_text is not None:
            try:
                ttnn.deallocate(self._payload_persistent_text)
            except Exception:
                pass
            self._payload_persistent_text = None
        if self._payload_persistent_enc is not None:
            try:
                ttnn.deallocate(self._payload_persistent_enc)
            except Exception:
                pass
            self._payload_persistent_enc = None
        for t in self._payload_parts_persist:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        self._payload_parts_persist = []
        self._payload_part_hosts = []
        for attr in (
            "_payload_persistent_lyric",
            "_payload_persistent_timbre",
            "_payload_lyric_bias_full",
            "_payload_lyric_bias_sliding",
            "_payload_timbre_bias_full",
            "_payload_timbre_bias_sliding",
            "_payload_enc_mask_dev",
        ):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._payload_lyric_host = None
        self._payload_timbre_host = None
        self._payload_text_host = None
        self._payload_trace_op_event = None
        self._payload_stage_write_event = None
        self._payload_shape_key = None
        self._cap_text_valid = 0
        self._cap_text_s = 0
        self._cap_lyric_valid = 0
        self._cap_lyric_s = 0
        self._cap_timbre_s = 0
        self._release_ctx_trace()

    def _release_ctx_trace(self) -> None:
        if self._ctx_trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._ctx_trace_id)
            except Exception:
                pass
            self._ctx_trace_id = None
        for attr in ("_ctx_persistent_src", "_ctx_persistent_chunk", "_ctx_persistent_out"):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._ctx_src_host = None
        self._ctx_chunk_host = None
        self._ctx_trace_op_event = None
        self._ctx_stage_write_event = None
        self._ctx_frames_key = None

    def forward(
        self, text_hidden_b1sd: ttnn.Tensor, attention_mask_01: np.ndarray
    ) -> tuple[ttnn.Tensor, np.ndarray, ttnn.Tensor]:
        """Legacy host-input wrapper around :meth:`forward_device`.

        Computes ``valid`` from the host attention mask, calls :meth:`forward_device`, and assembles
        the host ``enc_mask`` numpy array expected by ``e2e_model_tt`` / ``run_prompt_to_wav``.
        """
        s = int(text_hidden_b1sd.shape[2])
        attn = np.asarray(attention_mask_01, dtype=np.float32).reshape(1, -1)
        if int(attn.shape[1]) != s:
            raise ValueError(f"attention_mask length {attn.shape[1]} != text sequence length {s}")
        valid = int(attn.sum())

        enc, null_emb = self.forward_device(text_hidden_b1sd, valid)
        enc_mask = self._enc_mask_np(valid, s)
        return enc, enc_mask, null_emb

    @staticmethod
    def _official_enc_mask_from_payload(payload: dict) -> np.ndarray:
        lyric_mask_np = _to_numpy_mask(payload["lyric_attention_mask"])
        text_mask_np = _to_numpy_mask(payload["text_attention_mask"])
        lyric_valid = int(np.asarray(lyric_mask_np, dtype=np.float32).sum())
        text_valid = int(np.asarray(text_mask_np, dtype=np.float32).sum())
        lyric_s = int(np.asarray(_to_numpy_f32(payload["lyric_hidden_states"]), dtype=np.float32).shape[1])
        text_s = _text_seq_len_from_payload(payload)
        mask_parts: list[np.ndarray] = []
        if lyric_valid > 0:
            mask_parts.append(np.ones((1, lyric_valid), dtype=np.float32))
        mask_parts.append(np.ones((1, 1), dtype=np.float32))
        if text_valid > 0:
            mask_parts.append(np.ones((1, text_valid), dtype=np.float32))
        if lyric_valid < lyric_s:
            mask_parts.append(np.zeros((1, lyric_s - lyric_valid), dtype=np.float32))
        if text_valid < text_s:
            mask_parts.append(np.zeros((1, text_s - text_valid), dtype=np.float32))
        return np.concatenate(mask_parts, axis=1).astype(np.float32)

    @staticmethod
    def _ctx_arrays_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray]:
        src_np = _to_numpy_f32(payload["src_latents"])
        chunk_np = _to_numpy_mask(payload["chunk_mask"]).astype(np.float32)
        is_covers_np = _to_numpy_mask(payload["is_covers"]).reshape(-1) > 0
        hints = payload.get("precomputed_lm_hints_25Hz")
        if hints is not None and bool(is_covers_np[0]):
            src_np = _to_numpy_f32(hints)[:, : src_np.shape[1], :]
        return np.ascontiguousarray(src_np.astype(np.float32)), np.ascontiguousarray(chunk_np.astype(np.float32))

    def _official_ctx_from_payload(self, payload: dict) -> ttnn.Tensor:
        src_np, chunk_np = self._ctx_arrays_from_payload(payload)
        return self.ctx_concat_traced(src_np, chunk_np, use_trace=False)

    def _as_ctx_row_major_tensor(self, arr_np: np.ndarray) -> ttnn.Tensor:
        return ttnn.as_tensor(
            np.ascontiguousarray(arr_np, dtype=np.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )

    def _ctx_concat_device_body(self) -> ttnn.Tensor:
        if self._ctx_persistent_src is None or self._ctx_persistent_chunk is None:
            raise RuntimeError("context trace persistent src/chunk buffers are not allocated.")
        return ttnn.concat([self._ctx_persistent_src, self._ctx_persistent_chunk], dim=-1)

    def _install_ctx_persist_buffers(self, src_new: ttnn.Tensor, chunk_new: ttnn.Tensor) -> None:
        if not hasattr(ttnn, "clone"):
            raise RuntimeError("ttnn.clone is required for context latents trace.")
        self._ctx_persistent_src = ttnn.clone(src_new)
        self._ctx_persistent_chunk = ttnn.clone(chunk_new)
        self._ctx_src_host = self._host_staging_buffer_matching(self._ctx_persistent_src)
        self._ctx_chunk_host = self._host_staging_buffer_matching(self._ctx_persistent_chunk)

    def _capture_ctx_trace_program(self) -> None:
        ctx_warm = self._ctx_concat_device_body()
        ttnn.synchronize_device(self.device)
        try:
            ttnn.deallocate(ctx_warm)
        except Exception:
            pass
        old_ctx = self._ctx_persistent_out
        self._ctx_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._ctx_persistent_out = self._ctx_concat_device_body()
        ttnn.end_trace_capture(self.device, self._ctx_trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        if old_ctx is not None and old_ctx is not self._ctx_persistent_out:
            try:
                ttnn.deallocate(old_ctx)
            except Exception:
                pass
        self._ctx_trace_op_event = ttnn.record_event(self.device, 0)

    def _stage_ctx_inputs_cq1(
        self,
        src_new: ttnn.Tensor,
        chunk_new: ttnn.Tensor,
        *,
        skip_cq0_wait: bool = False,
    ) -> None:
        if self._ctx_persistent_src is None or self._ctx_persistent_chunk is None:
            raise RuntimeError("context trace staging called before persist buffers exist.")
        if not skip_cq0_wait:
            if self._ctx_trace_op_event is None:
                raise RuntimeError("context trace CQ1 staging called before a prior execute (op_event is None).")
            ttnn.wait_for_event(1, self._ctx_trace_op_event)
        self._stage_copy_tensor_cq1(src_new, self._ctx_persistent_src, self._ctx_src_host)
        self._stage_copy_tensor_cq1(chunk_new, self._ctx_persistent_chunk, self._ctx_chunk_host)
        self._deallocate_tensors([src_new, chunk_new])
        self._ctx_stage_write_event = ttnn.record_event(self.device, 1)

    def _execute_ctx_trace_only(self) -> ttnn.Tensor:
        if self._ctx_trace_id is None or self._ctx_persistent_out is None:
            raise RuntimeError("context trace execute called before capture.")
        if self._ctx_stage_write_event is None:
            raise RuntimeError("context trace execute called before CQ1 staging completed.")
        ttnn.wait_for_event(0, self._ctx_stage_write_event)
        self._ctx_stage_write_event = None
        ttnn.execute_trace(self.device, self._ctx_trace_id, cq_id=0, blocking=True)
        self._ctx_trace_op_event = ttnn.record_event(self.device, 0)
        ttnn.synchronize_device(self.device)
        return self._ctx_persistent_out

    def ctx_concat_traced(
        self,
        src_latents_np: np.ndarray,
        chunk_mask_np: np.ndarray,
        *,
        use_trace: bool = True,
    ) -> ttnn.Tensor:
        """Build ``[B, T, 128]`` context latents (64 src + 64 chunk) on device.

        When ``use_trace`` and trace APIs exist, captures ``ttnn.concat`` on CQ0 with CQ1 input
        refresh. Otherwise uploads a pre-concatenated host tensor (legacy eager path).
        """
        src_np = np.ascontiguousarray(src_latents_np.astype(np.float32))
        chunk_np = np.ascontiguousarray(chunk_mask_np.astype(np.float32))
        if int(src_np.shape[0]) != 1 or int(chunk_np.shape[0]) != 1:
            raise ValueError("ctx_concat_traced supports batch size B=1 only.")
        frames = int(src_np.shape[1])
        if int(chunk_np.shape[1]) != frames:
            raise ValueError(f"src_latents T={frames} != chunk_mask T={int(chunk_np.shape[1])}")
        if int(src_np.shape[2]) != 64 or int(chunk_np.shape[2]) != 64:
            raise ValueError("ctx_concat_traced expects src/chunk last dim 64.")

        if not use_trace or not hasattr(ttnn, "begin_trace_capture") or not hasattr(ttnn, "execute_trace"):
            ctx_np = np.concatenate([src_np, chunk_np], axis=-1)
            return self._as_ctx_row_major_tensor(ctx_np)

        if self._ctx_frames_key is not None and self._ctx_frames_key != frames:
            self._release_ctx_trace()

        src_new = self._as_ctx_row_major_tensor(src_np)
        chunk_new = self._as_ctx_row_major_tensor(chunk_np)

        if self._ctx_persistent_src is None or self._ctx_frames_key != frames:
            if self._ctx_frames_key is not None and self._ctx_frames_key != frames:
                self._release_ctx_trace()
            self._install_ctx_persist_buffers(src_new, chunk_new)
            self._ctx_frames_key = int(frames)

        skip_wait = self._ctx_trace_op_event is None
        self._stage_ctx_inputs_cq1(src_new, chunk_new, skip_cq0_wait=skip_wait)
        if self._ctx_trace_id is None:
            if self._ctx_stage_write_event is not None:
                ttnn.wait_for_event(0, self._ctx_stage_write_event)
            self._capture_ctx_trace_program()
        return self._execute_ctx_trace_only()

    def _official_ctx_traced_from_payload(self, payload: dict) -> ttnn.Tensor:
        src_np, chunk_np = self._ctx_arrays_from_payload(payload)
        return self.ctx_concat_traced(src_np, chunk_np, use_trace=True)

    def _payload_shape_key_from_payload(self, payload: dict) -> tuple[int, ...]:
        lyric_mask_np = _to_numpy_mask(payload["lyric_attention_mask"])
        text_mask_np = _to_numpy_mask(payload["text_attention_mask"])
        lyric_valid = int(np.asarray(lyric_mask_np, dtype=np.float32).sum())
        text_valid = int(np.asarray(text_mask_np, dtype=np.float32).sum())
        lyric_s = int(_to_numpy_f32(payload["lyric_hidden_states"]).shape[1])
        timbre_s = int(_to_numpy_f32(payload["refer_audio_acoustic_hidden_states_packed"]).shape[1])
        text_s = _text_seq_len_from_payload(payload)
        return (lyric_valid, lyric_s, timbre_s, text_valid, text_s)

    def _upload_payload_lyric_timbre(self, payload: dict) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        lyric_np = _to_numpy_f32(payload["lyric_hidden_states"])
        timbre_np = _to_numpy_f32(payload["refer_audio_acoustic_hidden_states_packed"])
        order_mask_np = _to_numpy_mask(payload["refer_audio_order_mask"]).astype(np.int64).reshape(-1)
        if lyric_np.shape[0] != 1:
            raise ValueError("TTNN official condition encoder currently supports demo batch size B=1 only.")
        if timbre_np.shape[0] != 1 or order_mask_np.shape != (1,) or int(order_mask_np[0]) != 0:
            raise ValueError("TTNN official condition encoder currently supports one packed timbre segment.")
        lyric_tt = ttnn.as_tensor(
            np.ascontiguousarray(lyric_np, dtype=np.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        timbre_tt = ttnn.as_tensor(
            np.ascontiguousarray(timbre_np, dtype=np.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        return lyric_tt, timbre_tt

    def _ensure_payload_encoder_biases(self, lyric_mask_np: np.ndarray, *, timbre_s: int) -> None:
        if self._payload_lyric_bias_full is not None:
            return
        self._payload_lyric_bias_full = self.lyric_encoder.make_attn_bias_dev(lyric_mask_np, use_sliding=False)
        self._payload_lyric_bias_sliding = self.lyric_encoder.make_attn_bias_dev(lyric_mask_np, use_sliding=True)
        ones = np.ones((1, int(timbre_s)), dtype=np.float32)
        self._payload_timbre_bias_full = self.timbre_encoder.make_attn_bias_dev(ones, use_sliding=False)
        self._payload_timbre_bias_sliding = self.timbre_encoder.make_attn_bias_dev(ones, use_sliding=True)

    def upload_enc_mask_dev(self, enc_mask_np: np.ndarray) -> ttnn.Tensor:
        """Upload the official payload ``enc_mask`` once per shape; safe outside trace capture."""
        arr = np.asarray(enc_mask_np, dtype=np.float32).reshape(1, -1)
        if self._payload_enc_mask_dev is not None and int(self._payload_enc_mask_dev.shape[1]) == int(arr.shape[1]):
            return self._payload_enc_mask_dev
        if self._payload_enc_mask_dev is not None:
            try:
                ttnn.deallocate(self._payload_enc_mask_dev)
            except Exception:
                pass
        self._payload_enc_mask_dev = ttnn.as_tensor(
            arr,
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        return self._payload_enc_mask_dev

    def _lyric_timbre_parts_eager(self, payload: dict) -> tuple[list[ttnn.Tensor], ttnn.Tensor]:
        """Lyric + timbre concat operands (no text).

        Returns ``(parts, lyric_root)``. ``parts`` may include views into ``lyric_root``; the caller
        must clone or copy before ``ttnn.deallocate(lyric_root)`` (same as eager ``forward_payload``).
        """
        lyric_np = _to_numpy_f32(payload["lyric_hidden_states"])
        lyric_mask_np = _to_numpy_mask(payload["lyric_attention_mask"])
        timbre_np = _to_numpy_f32(payload["refer_audio_acoustic_hidden_states_packed"])
        order_mask_np = _to_numpy_mask(payload["refer_audio_order_mask"]).astype(np.int64).reshape(-1)
        if lyric_np.shape[0] != 1:
            raise ValueError("TTNN official condition encoder currently supports demo batch size B=1 only.")
        if timbre_np.shape[0] != 1 or order_mask_np.shape != (1,) or int(order_mask_np[0]) != 0:
            raise ValueError("TTNN official condition encoder currently supports one packed timbre segment.")

        lyric = _cond_seq_for_concat(self.lyric_encoder(lyric_np, lyric_mask_np))
        timbre = _cond_seq_for_concat(self.timbre_encoder(timbre_np, None, output_first_token=True))
        lyric_valid = int(np.asarray(lyric_mask_np, dtype=np.float32).sum())
        lyric_s = int(lyric_np.shape[1])
        parts: list[ttnn.Tensor] = []
        if lyric_valid > 0:
            parts.append(ttnn.slice(lyric, (0, 0, 0), (1, lyric_valid, self.lyric_encoder.hidden_size)))
        parts.append(timbre)
        if lyric_valid < lyric_s:
            parts.append(ttnn.slice(lyric, (0, lyric_valid, 0), (1, lyric_s, self.lyric_encoder.hidden_size)))
        return parts, lyric

    @staticmethod
    def _deallocate_tensors(tensors: list[ttnn.Tensor]) -> None:
        for t in tensors:
            if t is None:
                continue
            try:
                ttnn.deallocate(t)
            except Exception:
                pass

    def _text_hidden_b1sd_upload(self, payload: dict) -> ttnn.Tensor:
        text_np = _to_numpy_f32(payload["text_hidden_states"])
        if text_np.ndim == 4:
            text_np = np.ascontiguousarray(text_np[:, 0, :, :], dtype=np.float32)
        elif text_np.ndim == 3:
            text_np = np.ascontiguousarray(text_np, dtype=np.float32)
        else:
            raise ValueError(f"text_hidden_states must be [B,S,D] or [B,1,S,D], got {text_np.shape}")
        b, s, d = int(text_np.shape[0]), int(text_np.shape[1]), int(text_np.shape[2])
        if b != 1:
            raise ValueError("TTNN official condition encoder currently supports demo batch size B=1 only.")
        text_b1sd = np.ascontiguousarray(text_np.reshape(1, 1, s, d), dtype=np.float32)
        return ttnn.as_tensor(
            text_b1sd,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )

    def _payload_trace_enc_body(self) -> ttnn.Tensor:
        """Device-only lyric + timbre encoders, text projector, and concat (trace-safe).

        Operand order matches :meth:`forward_payload`: lyric_valid, timbre, text_valid,
        lyric_pad, text_pad.
        """
        if (
            self._payload_persistent_text is None
            or self._payload_persistent_lyric is None
            or self._payload_persistent_timbre is None
            or self._payload_lyric_bias_full is None
            or self._payload_timbre_bias_full is None
        ):
            raise RuntimeError("payload trace persistent buffers are not allocated.")
        lyric = _cond_seq_for_concat(
            self.lyric_encoder.forward_device(
                self._payload_persistent_lyric,
                self._payload_lyric_bias_full,
                self._payload_lyric_bias_sliding,
            )
        )
        timbre = _cond_seq_for_concat(
            self.timbre_encoder.forward_device(
                self._payload_persistent_timbre,
                self._payload_timbre_bias_full,
                self._payload_timbre_bias_sliding,
                output_first_token=True,
            )
        )
        text_proj = self.text_projector.forward_from_hidden(self._payload_persistent_text, activation_dtype=self.dtype)
        d = int(self.lyric_encoder.hidden_size)
        s = int(text_proj.shape[1])
        v = int(self._cap_text_valid)
        lyric_valid = int(self._cap_lyric_valid)
        lyric_s = int(self._cap_lyric_s)

        parts: list[ttnn.Tensor] = []
        if lyric_valid > 0:
            parts.append(ttnn.slice(lyric, (0, 0, 0), (1, lyric_valid, d)))
        parts.append(timbre)
        text_valid = ttnn.slice(text_proj, (0, 0, 0), (1, v, d)) if v > 0 else None
        if text_valid is not None:
            parts.append(text_valid)
        if lyric_valid < lyric_s:
            parts.append(ttnn.slice(lyric, (0, lyric_valid, 0), (1, lyric_s, d)))
        text_pad = ttnn.slice(text_proj, (0, v, 0), (1, s, d)) if v < s else None
        if text_pad is not None:
            parts.append(text_pad)
        enc = ttnn.concat(
            [ace_step_ensure_tile_layout(ttnn, p) for p in parts],
            dim=1,
            **ace_step_concat_kwargs(ttnn),
        )
        for t in (lyric, timbre, text_proj, text_valid, text_pad):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except RuntimeError:
                    pass
        return enc

    @staticmethod
    def _host_staging_buffer_matching(dev_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Host buffer for ``copy_device_to_host_tensor`` with matching layout/page config."""
        th = ttnn.to_torch(dev_tensor).contiguous()
        if th.dtype != torch.bfloat16:
            th = th.to(dtype=torch.bfloat16)
        return ttnn.from_torch(th, dtype=ttnn.bfloat16, layout=dev_tensor.layout)

    def _stage_copy_tensor_cq1(self, src: ttnn.Tensor, dst: ttnn.Tensor, host_buf: ttnn.Tensor) -> None:
        if hasattr(ttnn, "copy_device_to_host_tensor"):
            ttnn.copy_device_to_host_tensor(src, host_buf, cq_id=1)
            ttnn.copy_host_to_device_tensor(host_buf, dst, cq_id=1)
        else:
            host_updated = self._host_staging_buffer_matching(src)
            ttnn.copy_host_to_device_tensor(host_updated, dst, cq_id=1)

    def _install_payload_persist_buffers(
        self,
        lyric_new: ttnn.Tensor,
        timbre_new: ttnn.Tensor,
        text_new: ttnn.Tensor,
    ) -> None:
        """Clone lyric/timbre inputs + text into persistent buffers (no trace active)."""
        if not hasattr(ttnn, "clone"):
            raise RuntimeError("ttnn.clone is required for official payload condition trace.")
        self._payload_parts_persist = []
        self._payload_part_hosts = []
        self._payload_persistent_lyric = ttnn.clone(lyric_new)
        self._payload_persistent_timbre = ttnn.clone(timbre_new)
        self._payload_lyric_host = self._host_staging_buffer_matching(self._payload_persistent_lyric)
        self._payload_timbre_host = self._host_staging_buffer_matching(self._payload_persistent_timbre)
        self._payload_persistent_text = ttnn.clone(text_new)
        self._payload_text_host = self._host_staging_buffer_matching(self._payload_persistent_text)

    def _capture_payload_trace_program(self) -> None:
        """Warm + capture text projector + concat (persistent lyric/timbre parts already installed)."""
        enc_warm = self._payload_trace_enc_body()
        ttnn.synchronize_device(self.device)
        try:
            ttnn.deallocate(enc_warm)
        except Exception:
            pass
        old_enc = self._payload_persistent_enc
        self._payload_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._payload_persistent_enc = self._payload_trace_enc_body()
        ttnn.end_trace_capture(self.device, self._payload_trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        if old_enc is not None and old_enc is not self._payload_persistent_enc:
            try:
                ttnn.deallocate(old_enc)
            except Exception:
                pass
        self._payload_trace_op_event = ttnn.record_event(self.device, 0)

    def _stage_payload_inputs_cq1(
        self,
        lyric_new: ttnn.Tensor,
        timbre_new: ttnn.Tensor,
        text_new: ttnn.Tensor,
        *,
        skip_cq0_wait: bool = False,
    ) -> None:
        if (
            self._payload_persistent_lyric is None
            or self._payload_persistent_timbre is None
            or self._payload_persistent_text is None
        ):
            raise RuntimeError("payload trace persistent buffers are not allocated.")
        if not skip_cq0_wait:
            if self._payload_trace_op_event is None:
                raise RuntimeError("payload trace CQ1 staging called before a prior execute (op_event is None).")
            ttnn.wait_for_event(1, self._payload_trace_op_event)
        self._stage_copy_tensor_cq1(lyric_new, self._payload_persistent_lyric, self._payload_lyric_host)
        self._stage_copy_tensor_cq1(timbre_new, self._payload_persistent_timbre, self._payload_timbre_host)
        self._stage_copy_tensor_cq1(text_new, self._payload_persistent_text, self._payload_text_host)
        self._deallocate_tensors([lyric_new, timbre_new, text_new])
        self._payload_stage_write_event = ttnn.record_event(self.device, 1)

    def _execute_payload_trace_only(self) -> ttnn.Tensor:
        if self._payload_trace_id is None or self._payload_persistent_enc is None:
            raise RuntimeError("payload trace execute called before capture.")
        if self._payload_stage_write_event is None:
            raise RuntimeError("payload trace execute called before CQ1 staging completed.")
        ttnn.wait_for_event(0, self._payload_stage_write_event)
        self._payload_stage_write_event = None
        ttnn.execute_trace(self.device, self._payload_trace_id, cq_id=0, blocking=True)
        self._payload_trace_op_event = ttnn.record_event(self.device, 0)
        ttnn.synchronize_device(self.device)
        return self._payload_persistent_enc

    def _release_payload_trace_id_only(self) -> None:
        """Drop the active trace handle without freeing persistent payload buffers."""
        if self._payload_trace_id is None:
            return
        try:
            ttnn.release_trace(self.device, self._payload_trace_id)
        except Exception:
            pass
        self._payload_trace_id = None

    def forward_payload_traced(self, payload: dict) -> tuple[ttnn.Tensor, np.ndarray, ttnn.Tensor, ttnn.Tensor]:
        """Trace + 2CQ ``forward_payload`` for the default handler demo path.

        Trace captures lyric (8L) + timbre (4L) + text projector + concat on CQ0. Inputs refresh on CQ1
        before each replay. Context latents use a separate trace for ``concat([src_latents, chunk_mask])``.
        :meth:`upload_enc_mask_dev` stages ``enc_mask`` on device for DiT cross-attention.

        Returns persistent ``enc`` and ``ctx`` — clone both before :meth:`release_trace``.
        """
        if not hasattr(ttnn, "begin_trace_capture") or not hasattr(ttnn, "execute_trace"):
            raise RuntimeError("forward_payload_traced requires trace support (begin_trace_capture / execute_trace).")

        shape_key = self._payload_shape_key_from_payload(payload)
        if self._payload_shape_key is not None and self._payload_shape_key != shape_key:
            self._release_payload_trace()

        lyric_mask_np = _to_numpy_mask(payload["lyric_attention_mask"])
        text_mask_np = _to_numpy_mask(payload["text_attention_mask"])
        self._cap_lyric_valid = int(np.asarray(lyric_mask_np, dtype=np.float32).sum())
        self._cap_lyric_s = int(_to_numpy_f32(payload["lyric_hidden_states"]).shape[1])
        self._cap_timbre_s = int(_to_numpy_f32(payload["refer_audio_acoustic_hidden_states_packed"]).shape[1])
        self._cap_text_valid = int(np.asarray(text_mask_np, dtype=np.float32).sum())
        self._cap_text_s = _text_seq_len_from_payload(payload)

        enc_mask = self._official_enc_mask_from_payload(payload)
        self.upload_enc_mask_dev(enc_mask)

        ttnn.synchronize_device(self.device)
        lyric_new, timbre_new = self._upload_payload_lyric_timbre(payload)
        text_new = self._text_hidden_b1sd_upload(payload)

        fresh_install = self._payload_persistent_text is None or self._payload_shape_key != shape_key
        if fresh_install:
            if self._payload_shape_key is not None and self._payload_shape_key != shape_key:
                self._release_payload_trace()
            self._ensure_payload_encoder_biases(lyric_mask_np, timbre_s=int(self._cap_timbre_s))
            self._install_payload_persist_buffers(lyric_new, timbre_new, text_new)
            self._payload_shape_key = shape_key
            self._deallocate_tensors([lyric_new, timbre_new, text_new])
        else:
            skip_wait = self._payload_trace_op_event is None
            self._stage_payload_inputs_cq1(lyric_new, timbre_new, text_new, skip_cq0_wait=skip_wait)

        if self._payload_trace_id is None:
            if self._payload_stage_write_event is not None:
                ttnn.wait_for_event(0, self._payload_stage_write_event)
            self._capture_payload_trace_program()
            # First capture reads persistent buffers filled by install (no CQ1 staging yet).
            enc = self._payload_persistent_enc
        else:
            enc = self._execute_payload_trace_only()
        ttnn.synchronize_device(self.device)
        ctx = self._official_ctx_traced_from_payload(payload)
        return enc, enc_mask, ctx, self.null_condition_emb

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
        lyric = _cond_seq_for_concat(self.lyric_encoder(lyric_np, lyric_mask_np))
        timbre = _cond_seq_for_concat(self.timbre_encoder(timbre_np, None, output_first_token=True))

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
        text_s = _text_seq_len_from_payload(payload)
        if text_valid < text_s:
            parts.append(ttnn.slice(text_proj, (0, text_valid, 0), (1, text_s, self.lyric_encoder.hidden_size)))
            mask_parts.append(np.zeros((1, text_s - text_valid), dtype=np.float32))
        _tl = ace_step_to_layout_kwargs(ttnn)
        _ck = ace_step_concat_kwargs(ttnn)
        enc = ttnn.concat(
            [ace_step_ensure_tile_layout(ttnn, p) for p in parts],
            dim=1,
            **_ck,
        )
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
