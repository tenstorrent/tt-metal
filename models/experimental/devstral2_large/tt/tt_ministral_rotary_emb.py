# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN rotary tables for Devstral-2 (123B) text models.

The architecture matches Hugging Face ``Ministral3RotaryEmbedding``. Host-side frequencies use **NumPy
only** (no PyTorch in this module): ``inv_freq`` and ``attention_scaling`` match ``transformers``
defaults / YaRN logic, then tables are uploaded with ``ttnn.from_torch`` (which accepts NumPy arrays).
Device behavior inherits :class:`HfRotarySetup`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.prefetcher import Prefetcher
from models.tt_transformers.tt.rope import HfRotarySetup
from ttnn import replicate_tensor_to_mesh_mapper


def _compute_default_inv_freq_numpy(config: Any) -> Tuple[np.ndarray, float]:
    """Match ``Ministral3RotaryEmbedding.compute_default_rope_parameters`` (HF)."""
    base = float(config.rope_parameters["rope_theta"])
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(dim)
    idx = np.arange(0, dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (idx / dim))
    return inv_freq.astype(np.float32), 1.0


def _compute_yarn_inv_freq_numpy(config: Any) -> Tuple[np.ndarray, float]:
    """Match ``transformers.modeling_rope_utils._compute_yarn_parameters`` (HF)."""
    rope_parameters_dict = config.rope_parameters

    base = float(rope_parameters_dict["rope_theta"])
    partial_rotary_factor = float(rope_parameters_dict.get("partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = int(rope_parameters_dict["original_max_position_embeddings"])

    if factor is None:
        factor = float(config.max_position_embeddings / original_max_position_embeddings)
    else:
        factor = float(factor)

    def get_mscale(scale: float, ms: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * ms * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale is not None and mscale_all_dim is not None:
            attention_factor = float(get_mscale(factor, float(mscale)) / get_mscale(factor, float(mscale_all_dim)))
        else:
            attention_factor = float(get_mscale(factor))

    beta_fast = float(rope_parameters_dict.get("beta_fast") or 32)
    beta_slow = float(rope_parameters_dict.get("beta_slow") or 1)

    def find_correction_dim(num_rotations: float, dim_v: int, base_v: float, max_position_embeddings: int) -> float:
        return (dim_v * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base_v))

    def find_correction_range(
        low_rot: float, high_rot: float, dim_v: int, base_v: float, max_position_embeddings: int, truncate: bool
    ) -> Tuple[float, float]:
        low = find_correction_dim(low_rot, dim_v, base_v, max_position_embeddings)
        high = find_correction_dim(high_rot, dim_v, base_v, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim_v - 1)

    def linear_ramp_factor(min_v: float, max_v: float, dim_half: int) -> np.ndarray:
        if min_v == max_v:
            max_v += 0.001
        linear_func = (np.arange(dim_half, dtype=np.float64) - min_v) / (max_v - min_v)
        return np.clip(linear_func, 0, 1).astype(np.float64)

    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = bool(config.rope_parameters.get("truncate", True))
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    ramp = linear_ramp_factor(low, high, dim // 2)
    inv_freq_extrapolation_factor = 1.0 - ramp
    inv_freq = inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * (
        inv_freq_extrapolation_factor
    )
    return inv_freq.astype(np.float32), float(attention_factor)


def ministral3_inv_freq_and_attention_scaling(config: Any) -> Tuple[np.ndarray, float]:
    """
    ``inv_freq`` (length ``head_dim/2``) and scalar ``attention_scaling`` for Ministral3 / HF-compatible rope.
    """
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    if not isinstance(config, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(config)!r}")

    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()

    rope_type = config.rope_parameters["rope_type"]
    if rope_type == "default":
        return _compute_default_inv_freq_numpy(config)
    if rope_type == "yarn":
        return _compute_yarn_inv_freq_numpy(config)
    raise NotImplementedError(
        f"rope_type={rope_type!r} is not ported to NumPy in this module; extend tt_ministral_rotary_emb or use HF."
    )


def ministral3_hf_cos_sin_tables(
    head_dim: int,
    max_seq_len: int,
    config: Any,
    *,
    table_dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Host cos/sin tables ``[max_seq_len, head_dim]`` matching HF ``Ministral3RotaryEmbedding.forward``
    for positions ``0 .. max_seq_len - 1``. Computed with NumPy only.
    """
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    if not isinstance(config, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(config)!r}")

    cfg_hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if int(cfg_hd) != int(head_dim):
        raise ValueError(f"head_dim={head_dim} does not match config head dim ({cfg_hd}).")

    inv_freq, attention_scaling = ministral3_inv_freq_and_attention_scaling(config)
    t = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq.astype(np.float64))
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_hf = (np.cos(emb) * attention_scaling).astype(table_dtype)
    sin_hf = (np.sin(emb) * attention_scaling).astype(table_dtype)
    return cos_hf, sin_hf


def _upload_hf_cos_sin_4d(
    cos_hf: np.ndarray,
    sin_hf: np.ndarray,
    device: Any,
    datatype: ttnn.DataType,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    cos_4d = cos_hf[np.newaxis, np.newaxis, ...]
    sin_4d = sin_hf[np.newaxis, np.newaxis, ...]
    mapper = replicate_tensor_to_mesh_mapper(device)
    cos_tt = ttnn.from_torch(
        cos_4d,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=mapper,
    )
    sin_tt = ttnn.from_torch(
        sin_4d,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=mapper,
    )
    return cos_tt, sin_tt


class TtMinistral3RotaryEmbedding(HfRotarySetup):
    """
    HF-aligned Ministral3 RoPE caches on device via NumPy table build + :class:`HfRotarySetup` lookup.
    """

    def __init__(
        self,
        device: Any,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        config: Any,
        use_qk_fused: bool = False,
        datatype: ttnn.DataType = ttnn.bfloat16,
        shard_batch_to_mesh_dim: Optional[int] = 1,
        prefetcher: Optional[Prefetcher] = None,
    ) -> None:
        LightweightModule.__init__(self)
        if use_qk_fused:
            raise NotImplementedError("use_qk_fused")
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        table_np = np.float32 if datatype == ttnn.bfloat16 else np.float64
        cos_hf, sin_hf = ministral3_hf_cos_sin_tables(head_dim, max_seq_len, config, table_dtype=table_np)

        self.cos_matrix, self.sin_matrix = _upload_hf_cos_sin_4d(cos_hf, sin_hf, device, datatype)
        self.cos_matrix_prefill, self.sin_matrix_prefill = _upload_hf_cos_sin_4d(cos_hf, sin_hf, device, datatype)

        self.cos_matrix_2d = ttnn.reshape(self.cos_matrix, (max_seq_len, head_dim))
        self.sin_matrix_2d = ttnn.reshape(self.sin_matrix, (max_seq_len, head_dim))

        self.transformation_mat = None
        self.transformation_mat_prefill = None

        self._ministral3_config = config

    def slice_rot_mats_prefill(self, start_pos: int, seq_len: int) -> list:
        """
        Cos/sin slices ``[1, 1, seq_len, head_dim]`` on device for full prefill starting at ``start_pos``.

        Matches the slicing policy used in ``Transformer.prepare_inputs_prefill`` (pad dim 2 if needed).
        """
        mat_len = self.cos_matrix_prefill.shape[2]
        required_end = start_pos + seq_len
        if mat_len < required_end:
            raise RuntimeError(
                f"RoPE prefill needs positions through {required_end} but cos_matrix_prefill length is {mat_len}; "
                "build TtMinistral3RotaryEmbedding with a larger max_seq_len."
            )
        prefill_start_pos = start_pos
        slice_end = min(mat_len, required_end)
        cos_slice = self.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        pad_len = max(0, required_end - mat_len)
        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)
        return [cos_slice, sin_slice]

    @property
    def ministral3_config(self) -> Any:
        return self._ministral3_config


TtDevstral2LargeRotaryEmbedding = TtMinistral3RotaryEmbedding

__all__ = [
    "TtDevstral2LargeRotaryEmbedding",
    "TtMinistral3RotaryEmbedding",
    "ministral3_hf_cos_sin_tables",
    "ministral3_inv_freq_and_attention_scaling",
]
