# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN audio-code detokenizer for ACE-Step LM 5 Hz code hints."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import numpy as np

import ttnn

from .condition_encoder import _bidirectional_attn_bias_np, _rope_cos_sin_np
from .math_perf_env import ace_step_concat_kwargs, ace_step_reshape_kwargs, ace_step_safe_deallocate
from .qwen3_embedding_encoder import Qwen3EmbeddingEncoderConfig, _TtQwen3EncoderLayer


def load_audio_detokenizer_weights_np(safetensors_path: str) -> Dict[str, np.ndarray]:
    from models.experimental.ace_step_v1_5.utils.weight_cache import load_prefix_weights_np

    return load_prefix_weights_np(
        safetensors_path,
        (
            "tokenizer.quantizer.project_out.",
            "detokenizer.",
        ),
        component="audio-code-detokenizer-weights",
        tag="audio_detokenizer_np",
    )


_FSQ_LEVELS: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
_FSQ_MAX_AUDIO_CODE: int = int(np.prod(_FSQ_LEVELS)) - 1  # 64000 - 1


def parse_audio_code_string(code_str: str) -> list[int]:
    """Parse ``<|audio_code_NNN|>`` tokens out of a serialized LM string.

    Stays on host because the upstream LM handler emits the audio-code stream as decoded
    text. Eliminating this step would require changing the LM-to-detokenizer handshake to
    pass raw token IDs plus a tokenizer-side audio-code-ID mapping (vendored
    ``acestep.core.generation.handler.audio_codes`` API). The regex parse itself is O(N)
    over a short string (~75 tokens for 15 s @ 5 Hz) and runs in microseconds.
    """
    return [max(0, min(int(x), _FSQ_MAX_AUDIO_CODE)) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str or "")]


def fsq_codes_from_indices_np(indices: np.ndarray) -> np.ndarray:
    """Match ``ResidualFSQ(...levels=[8,8,8,5,5,5], num_quantizers=1).get_codes_from_indices``.

    Retained for tests / A-B reference (``TtAceStepAudioCodeDetokenizer`` no longer calls this on
    the hot path; the FSQ unpack is done on device via ``ttnn.embedding`` against a precomputed
    codebook in :meth:`TtAceStepAudioCodeDetokenizer.forward`).
    """
    levels = np.asarray(_FSQ_LEVELS, dtype=np.int64)
    basis = np.cumprod(np.asarray([1, *levels[:-1]], dtype=np.int64), dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)[..., None]
    level_indices = (idx // basis) % levels
    codes = level_indices.astype(np.float32) * (2.0 / (levels.astype(np.float32) - 1.0)) - 1.0
    return codes.astype(np.float32)


def _build_fsq_codebook_np() -> np.ndarray:
    """Materialize the full ``[prod(levels), len(levels)] = [64000, 6]`` FSQ codebook once.

    Each row is ``fsq_codes_from_indices_np(np.array([i]))[0]`` for ``i in [0, 63999]``.
    Cheap (~64k * 6 fp32 ≈ 1.5 MB on host, uploaded once to TTNN as bf16 ≈ 768 KB).
    """
    n = int(np.prod(_FSQ_LEVELS))
    return fsq_codes_from_indices_np(np.arange(n, dtype=np.int64))


def _detok_chunk_n() -> int:
    """Max audio-code count per TTNN detokenizer forward (L1 budget; see ``ace_step_detok_chunk_n``)."""
    from .math_perf_env import ace_step_detok_chunk_n

    return ace_step_detok_chunk_n()


def _as_weight(weights_np: Dict[str, np.ndarray], key: str, *, device, dtype, mem, mapper):
    return ttnn.as_tensor(
        weights_np[key],
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
        mesh_mapper=mapper,
    )


class TtAceStepAudioCodeDetokenizer:
    """Decode ACE 5 Hz LM audio-code tokens into 25 Hz latent hints on TTNN."""

    def __init__(self, *, device, checkpoint_safetensors_path: str, dtype=None) -> None:
        self.device = device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None)
        if self.dtype is None:
            raise RuntimeError("bfloat16 required")
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        weights_np = load_audio_detokenizer_weights_np(str(checkpoint_safetensors_path))

        cfg = Qwen3EmbeddingEncoderConfig(
            hidden_size=2048,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=6144,
            max_seq_len=5,
            rope_theta=1_000_000.0,
        )
        init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
        sdpa_compute_kernel_config = None
        linear_compute_kernel_config = None
        if callable(init_ck):
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
                k_chunk_size=32,
                exp_approx_mode=False,
            )

        self.quantizer_project_out_w = _as_weight(
            weights_np,
            "tokenizer.quantizer.project_out.weight",
            device=device,
            dtype=self.dtype,
            mem=self.mem,
            mapper=mapper,
        )
        self.quantizer_project_out_b = ttnn.as_tensor(
            weights_np["tokenizer.quantizer.project_out.bias"].reshape(1, 1, 1, -1),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        self.embed_w = _as_weight(
            weights_np, "detokenizer.embed_tokens.weight", device=device, dtype=self.dtype, mem=self.mem, mapper=mapper
        )
        self.embed_b = ttnn.as_tensor(
            weights_np["detokenizer.embed_tokens.bias"].reshape(1, 1, 1, -1),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        self.special_tokens = ttnn.as_tensor(
            weights_np["detokenizer.special_tokens"].reshape(1, 1, 5, 2048),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        self.layers = [
            _TtQwen3EncoderLayer(
                device=device,
                weights_np=weights_np,
                prefix=f"detokenizer.layers.{i}",
                cfg=cfg,
                dtype=self.dtype,
                mem=self.mem,
                mapper=mapper,
                sdpa_compute_kernel_config=sdpa_compute_kernel_config,
                sdpa_program_config=sdpa_program_config,
                linear_compute_kernel_config=linear_compute_kernel_config,
                # Fused batch = number of audio-code rows; cond reuse-mcast-1D program_config
                # drives per_core_M = B * ceil(S/32) and can exceed WH/BH static CB budgets.
                use_cond_linear_program_config=False,
            )
            for i in range(2)
        ]
        self.norm_w = _as_weight(
            weights_np, "detokenizer.norm.weight", device=device, dtype=self.dtype, mem=self.mem, mapper=mapper
        )
        self.proj_out_w = _as_weight(
            weights_np, "detokenizer.proj_out.weight", device=device, dtype=self.dtype, mem=self.mem, mapper=mapper
        )
        self.proj_out_b = ttnn.as_tensor(
            weights_np["detokenizer.proj_out.bias"].reshape(1, 1, 1, -1),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        cos_np, sin_np = _rope_cos_sin_np(max_seq_len=5, head_dim=128, rope_theta=1_000_000.0)
        self.cos_tt = ttnn.as_tensor(
            cos_np, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=self.mem, mesh_mapper=mapper
        )
        self.sin_tt = ttnn.as_tensor(
            sin_np, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=self.mem, mesh_mapper=mapper
        )

        # Precomputed FSQ codebook for on-device unpack. Replaces ``fsq_codes_from_indices_np``
        # in the hot path: per call we only upload a [1, N] uint32 index tensor and run
        # ``ttnn.embedding(idx, codebook)`` to materialize [1, N, 6] codes on device.
        # Pad vocab rows for TP-4 tile alignment; keep replicated (not vocab-TP) under DiT TP.
        from models.experimental.ace_step_v1_5.utils.ace_step_tp import (
            ace_step_pad_embedding_rows,
            ace_step_vocab_mesh_mapper,
        )
        from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_device_num_chips

        self._fsq_mapper = ace_step_vocab_mesh_mapper(device) if ace_step_device_num_chips(device) > 1 else mapper
        fsq_np = ace_step_pad_embedding_rows(
            _build_fsq_codebook_np(),
            num_devices=max(1, ace_step_device_num_chips(device)),
        )
        self.fsq_codebook_tt = ttnn.as_tensor(
            fsq_np,
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=self._fsq_mapper,
        )
        self._detok_trace_id: Optional[Any] = None
        self._detok_n_codes: Optional[int] = None
        self._detok_ids_dev: Optional[ttnn.Tensor] = None
        self._detok_ids_host: Optional[ttnn.Tensor] = None
        self._detok_out_dev: Optional[ttnn.Tensor] = None
        self._detok_bias_dev: Optional[ttnn.Tensor] = None
        self._detok_op_event: Any = None

    def _create_attn_bias_dev(self, n_codes: int) -> ttnn.Tensor:
        """Attention bias for ``n_codes`` rows; must be allocated *before* trace capture."""
        n = int(n_codes)
        mapper = (
            self._fsq_mapper
            if self._fsq_mapper is not None
            else (ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None)
        )
        bias_np = _bidirectional_attn_bias_np(np.ones((n, 5), dtype=np.float32), sliding_window=None)
        return ttnn.as_tensor(
            bias_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

    def _forward_on_ids_dev(
        self, ids_dev: ttnn.Tensor, n_codes: int, *, bias_tt: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """Device decode from ``[1, N]`` uint32 indices already on device."""
        n = int(n_codes)
        _sr = ace_step_reshape_kwargs(ttnn)
        _lin_kw = {"memory_config": self.mem} if self.mem is not None else {}
        x = ttnn.embedding(
            ids_dev,
            self.fsq_codebook_tt,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=self.mem,
        )
        x = ttnn.reshape(x, (1, 1, n, 6), **_sr)
        q = ttnn.linear(x, self.quantizer_project_out_w, bias=self.quantizer_project_out_b, transpose_b=True, **_lin_kw)
        q = ttnn.linear(q, self.embed_w, bias=self.embed_b, transpose_b=True, **_lin_kw)
        q = ttnn.reshape(q, (1, n, 1, 2048), **_sr)
        q = ttnn.repeat(q, (1, 1, 5, 1))
        special = ttnn.repeat(self.special_tokens, (1, n, 1, 1))
        h = ttnn.add(q, special, memory_config=self.mem)
        h = ttnn.reshape(h, (n, 1, 5, 2048), **_sr)
        owns_bias = False
        if bias_tt is None:
            bias_tt = self._create_attn_bias_dev(n)
            owns_bias = True
        try:
            for layer in self.layers:
                h = layer(h, self.cos_tt, self.sin_tt, bias_tt)
        finally:
            if owns_bias:
                try:
                    ttnn.deallocate(bias_tt)
                except Exception:
                    pass
        h = ttnn.rms_norm(ttnn.to_layout(h, ttnn.TILE_LAYOUT), weight=self.norm_w, epsilon=1e-6, memory_config=self.mem)
        h = ttnn.linear(h, self.proj_out_w, bias=self.proj_out_b, transpose_b=True, **_lin_kw)
        return ttnn.reshape(h, (1, n * 5, 64), **_sr)

    def _forward_n_codes(self, code_ids: list[int]) -> ttnn.Tensor:
        n = int(len(code_ids))
        chunk_n = _detok_chunk_n()
        if n <= chunk_n:
            return self._forward_n_codes_once(code_ids)
        parts: list[ttnn.Tensor] = []
        for start in range(0, n, chunk_n):
            end = min(start + chunk_n, n)
            parts.append(self._forward_n_codes_once(code_ids[start:end]))
        if len(parts) == 1:
            return parts[0]
        out = (
            ttnn.concat(parts, dim=1, **ace_step_concat_kwargs(ttnn))
            if hasattr(ttnn, "concat")
            else ttnn.concatenate(parts, dim=1, **ace_step_concat_kwargs(ttnn))
        )
        for part in parts:
            ace_step_safe_deallocate(ttnn, part)
        return out

    def _forward_n_codes_once(self, code_ids: list[int]) -> ttnn.Tensor:
        n = int(len(code_ids))
        mapper = (
            self._fsq_mapper
            if self._fsq_mapper is not None
            else (ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None)
        )
        ids_np = np.asarray(code_ids, dtype=np.uint32).reshape(1, n)
        ids_tt = ttnn.as_tensor(
            ids_np,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        try:
            return self._forward_on_ids_dev(ids_tt, n)
        finally:
            try:
                ttnn.deallocate(ids_tt)
            except Exception:
                pass

    def forward(self, code_str: str) -> ttnn.Tensor | None:
        code_ids = parse_audio_code_string(code_str)
        if not code_ids:
            return None
        return self._forward_n_codes(code_ids)

    def forward_traced(self, code_str: str) -> ttnn.Tensor | None:
        """Trace + 2CQ detokenizer for a fixed ``N = len(code_ids)`` (recapture when N changes)."""
        if not hasattr(ttnn, "begin_trace_capture"):
            return self.forward(code_str)
        from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_device_num_command_queues

        if ace_step_device_num_command_queues(self.device) < 2:
            return self.forward(code_str)
        code_ids = parse_audio_code_string(code_str)
        if not code_ids:
            return None
        n = int(len(code_ids))
        chunk_n = _detok_chunk_n()
        max_n = int(os.environ.get("ACE_STEP_MAX_AUDIO_CODES", "200"))
        # Trace capture is fixed-shape; long code streams use chunked eager forward().
        if n > max_n or n > chunk_n:
            return self.forward(code_str)
        if self._detok_trace_id is not None and self._detok_n_codes != n:
            self.release_trace()
        if self._detok_trace_id is None:
            ids_np = np.asarray(code_ids, dtype=np.uint32).reshape(1, n)
            self._detok_ids_host = ttnn.from_torch(ids_np, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._detok_ids_dev = ttnn.as_tensor(
                ids_np,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.mem,
            )
            self._detok_bias_dev = self._create_attn_bias_dev(n)
            warm = self._forward_on_ids_dev(self._detok_ids_dev, n, bias_tt=self._detok_bias_dev)
            ttnn.synchronize_device(self.device)
            try:
                ttnn.deallocate(warm)
            except Exception:
                pass
            self._detok_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._detok_out_dev = self._forward_on_ids_dev(self._detok_ids_dev, n, bias_tt=self._detok_bias_dev)
            ttnn.end_trace_capture(self.device, self._detok_trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            self._detok_n_codes = n
            self._detok_op_event = ttnn.record_event(self.device, 0)

        ids_np = np.asarray(code_ids, dtype=np.uint32).reshape(1, n)
        host_ids = ttnn.from_torch(ids_np, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.wait_for_event(1, self._detok_op_event)
        ttnn.copy_host_to_device_tensor(host_ids, self._detok_ids_dev, cq_id=1)
        write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(self.device, self._detok_trace_id, cq_id=0, blocking=True)
        self._detok_op_event = ttnn.record_event(self.device, 0)
        ttnn.synchronize_device(self.device)
        if hasattr(ttnn, "clone"):
            return ttnn.clone(self._detok_out_dev)
        return self._detok_out_dev

    def release_trace(self) -> None:
        if self._detok_trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._detok_trace_id)
            except Exception:
                pass
            self._detok_trace_id = None
        for t in (self._detok_ids_dev, self._detok_out_dev, self._detok_bias_dev):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        self._detok_ids_dev = None
        self._detok_out_dev = None
        self._detok_bias_dev = None
        self._detok_ids_host = None
        self._detok_op_event = None
        self._detok_n_codes = None


__all__ = ["TtAceStepAudioCodeDetokenizer", "parse_audio_code_string", "fsq_codes_from_indices_np"]
