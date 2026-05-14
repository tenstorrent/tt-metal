# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN audio-code detokenizer for ACE-Step LM 5 Hz code hints."""

from __future__ import annotations

import re
from typing import Dict

import numpy as np

import ttnn

from .condition_encoder import _bidirectional_attn_bias_np, _rope_cos_sin_np
from .qwen3_embedding_encoder import Qwen3EmbeddingEncoderConfig, _TtQwen3EncoderLayer


def load_audio_detokenizer_weights_np(safetensors_path: str) -> Dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    prefixes = (
        "tokenizer.quantizer.project_out.",
        "detokenizer.",
    )
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k.startswith(prefixes):
                out[k] = sf.get_tensor(k).detach().to(torch.float32).cpu().numpy()
    return out


def parse_audio_code_string(code_str: str) -> list[int]:
    max_audio_code = 63999
    return [max(0, min(int(x), max_audio_code)) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str or "")]


def fsq_codes_from_indices_np(indices: np.ndarray) -> np.ndarray:
    """Match ``ResidualFSQ(...levels=[8,8,8,5,5,5], num_quantizers=1).get_codes_from_indices``."""
    levels = np.asarray([8, 8, 8, 5, 5, 5], dtype=np.int64)
    basis = np.cumprod(np.asarray([1, *levels[:-1]], dtype=np.int64), dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)[..., None]
    level_indices = (idx // basis) % levels
    codes = level_indices.astype(np.float32) * (2.0 / (levels.astype(np.float32) - 1.0)) - 1.0
    return codes.astype(np.float32)


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

    def forward(self, code_str: str) -> ttnn.Tensor | None:
        code_ids = parse_audio_code_string(code_str)
        if not code_ids:
            return None
        code_np = fsq_codes_from_indices_np(np.asarray(code_ids, dtype=np.int64)).reshape(1, len(code_ids), 6)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        x = ttnn.as_tensor(
            code_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        x = ttnn.reshape(x, (1, 1, len(code_ids), 6))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        q = ttnn.linear(x, self.quantizer_project_out_w, bias=self.quantizer_project_out_b, transpose_b=True)
        q = ttnn.linear(q, self.embed_w, bias=self.embed_b, transpose_b=True)
        q = ttnn.reshape(q, (1, len(code_ids), 1, 2048))
        q = ttnn.repeat(q, (1, 1, 5, 1))
        special = ttnn.repeat(self.special_tokens, (1, len(code_ids), 1, 1))
        h = ttnn.add(q, special)
        h = ttnn.reshape(h, (len(code_ids), 1, 5, 2048))
        bias_np = _bidirectional_attn_bias_np(np.ones((len(code_ids), 5), dtype=np.float32), sliding_window=None)
        bias_tt = ttnn.as_tensor(
            bias_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        for layer in self.layers:
            h = layer(h, self.cos_tt, self.sin_tt, bias_tt)
        h = ttnn.rms_norm(ttnn.to_layout(h, ttnn.TILE_LAYOUT), weight=self.norm_w, epsilon=1e-6, memory_config=self.mem)
        h = ttnn.linear(h, self.proj_out_w, bias=self.proj_out_b, transpose_b=True)
        h = ttnn.reshape(h, (1, len(code_ids) * 5, 64))
        return h


__all__ = ["TtAceStepAudioCodeDetokenizer", "parse_audio_code_string", "fsq_codes_from_indices_np"]
