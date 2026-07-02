# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# HF-aligned Ministral3 RoPE tables (NumPy build + device caches).

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.experimental.devstral2_small.devstral_utils.multimodal_demo_helpers import resolve_rope_parameters
from models.tt_transformers.tt.prefetcher import Prefetcher


def _compute_default_inv_freq_numpy(config: Any, rope_parameters: dict) -> Tuple[np.ndarray, float]:
    """Match ``Ministral3RotaryEmbedding.compute_default_rope_parameters`` (HF)."""
    base = float(rope_parameters["rope_theta"])
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(dim)
    idx = np.arange(0, dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (idx / dim))
    return inv_freq.astype(np.float32), 1.0


def _compute_yarn_inv_freq_numpy(config: Any, rope_parameters_dict: dict) -> Tuple[np.ndarray, float]:
    """Match ``transformers.modeling_rope_utils._compute_yarn_parameters`` (HF)."""
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

    truncate = bool(rope_parameters_dict.get("truncate", True))
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    ramp = linear_ramp_factor(low, high, dim // 2)
    inv_freq_extrapolation_factor = 1.0 - ramp
    inv_freq = inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * (
        inv_freq_extrapolation_factor
    )
    return inv_freq.astype(np.float32), float(attention_factor)


def ministral3_inv_freq_and_attention_scaling(config: Any) -> Tuple[np.ndarray, float]:
    """``inv_freq`` (length ``head_dim/2``) and scalar ``attention_scaling`` for Ministral3 / HF-compatible rope."""
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    if not isinstance(config, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(config)!r}")

    rope_parameters = resolve_rope_parameters(config)
    rope_type = rope_parameters["rope_type"]
    if rope_type == "default":
        return _compute_default_inv_freq_numpy(config, rope_parameters)
    if rope_type == "yarn":
        return _compute_yarn_inv_freq_numpy(config, rope_parameters)
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
    """Host cos/sin ``[max_seq_len, head_dim]`` matching HF Ministral3RotaryEmbedding (NumPy)."""
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


def _host_upload(
    host_array: np.ndarray,
    device: Any,
    datatype: ttnn.DataType,
    *,
    layout: ttnn.Layout,
) -> ttnn.Tensor:
    """NumPy host buffer -> tilize/typecast on host -> ``to_device`` (no device Tilize/Untilize)."""
    mapper = ttnn.replicate_tensor_to_mesh_mapper(device)
    host_tt = ttnn.from_torch(
        host_array,
        layout=layout,
        dtype=datatype,
        mesh_mapper=mapper,
    )
    return ttnn.to_device(host_tt, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _upload_decode_embedding_tables(
    cos_hf: np.ndarray,
    sin_hf: np.ndarray,
    device: Any,
    datatype: ttnn.DataType,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """ROW_MAJOR ``[max_seq_len, head_dim]`` for ``ttnn.embedding`` (weights must stay row-major)."""
    return (
        _host_upload(cos_hf, device, datatype, layout=ttnn.ROW_MAJOR_LAYOUT),
        _host_upload(sin_hf, device, datatype, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def _upload_prefill_cos_sin_4d(
    cos_hf: np.ndarray,
    sin_hf: np.ndarray,
    device: Any,
    datatype: ttnn.DataType,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """TILE ``[1, 1, max_seq_len, head_dim]`` for legacy prefill ``rotary_embedding``."""
    cos_4d = cos_hf[np.newaxis, np.newaxis, ...]
    sin_4d = sin_hf[np.newaxis, np.newaxis, ...]
    return (
        _host_upload(cos_4d, device, datatype, layout=ttnn.TILE_LAYOUT),
        _host_upload(sin_4d, device, datatype, layout=ttnn.TILE_LAYOUT),
    )


def _pad_rot_idx_to_nearest_32(
    rot_idx: ttnn.Tensor,
    *,
    pad_tail: ttnn.Tensor | None = None,
    pad_width: int = 0,
) -> ttnn.Tensor:
    """Pad index width to 32 so embedding can fuse TILE output (no post-embedding ``to_layout`` tilize).

    Use a preallocated ``pad_tail`` (from :class:`TtMinistral3RotaryEmbedding`) during trace capture;
    ``ttnn.zeros`` uploads are not allowed while a trace is being recorded.
    """
    if len(rot_idx.shape) == 1:
        rot_idx = ttnn.unsqueeze(rot_idx, 0)
    batch = int(rot_idx.shape[-1])
    pad_size = nearest_32(batch) - batch
    if pad_size == 0:
        return rot_idx
    if pad_tail is not None and pad_size == pad_width:
        return ttnn.concat([rot_idx, pad_tail], dim=-1)
    pad_shape = list(rot_idx.shape)
    pad_shape[-1] = pad_size
    pad = ttnn.zeros(pad_shape, device=rot_idx.device(), dtype=rot_idx.dtype, layout=rot_idx.get_layout())
    return ttnn.concat([rot_idx, pad], dim=-1)


class TtMinistral3RotaryEmbedding(LightweightModule):
    """HF-aligned Ministral3 RoPE caches on device via NumPy table build."""

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
        super().__init__()
        if use_qk_fused:
            raise NotImplementedError("use_qk_fused")
        rope_theta = float(resolve_rope_parameters(config)["rope_theta"])
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.prefetcher = prefetcher
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1

        table_np = np.float32 if datatype == ttnn.bfloat16 else np.float64
        cos_hf, sin_hf = ministral3_hf_cos_sin_tables(head_dim, max_seq_len, config, table_dtype=table_np)

        self.cos_matrix_2d, self.sin_matrix_2d = _upload_decode_embedding_tables(cos_hf, sin_hf, device, datatype)
        self.cos_matrix_prefill, self.sin_matrix_prefill = _upload_prefill_cos_sin_4d(cos_hf, sin_hf, device, datatype)
        # Legacy decode slice path / prefill share TILE 4D views.
        self.cos_matrix = self.cos_matrix_prefill
        self.sin_matrix = self.sin_matrix_prefill

        self.transformation_mat = None
        self.transformation_mat_prefill = None

        self._ministral3_config = config
        self._rot_idx_pad_width = nearest_32(batch_size) - batch_size
        if self._rot_idx_pad_width > 0:
            self._rot_idx_pad = ttnn.zeros(
                (1, self._rot_idx_pad_width),
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            self._rot_idx_pad = None

    def get_rot_idxs(self, position_idxs: Any, on_host: bool = False) -> ttnn.Tensor:
        import torch

        if not isinstance(position_idxs, torch.Tensor):
            raise TypeError(f"position_idxs must be a torch.Tensor, got {type(position_idxs)}")
        if len(position_idxs.shape) != 1:
            raise ValueError("position idxs must be a [batch] tensor")

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)
        if torch.min(position_idxs) < 0:
            raise ValueError("position idxs must be non-negative")

        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)
        common = {
            "dtype": ttnn.uint32,
            "layout": ttnn.ROW_MAJOR_LAYOUT,
            "mesh_mapper": ttnn.replicate_tensor_to_mesh_mapper(self.device),
        }
        if on_host:
            return ttnn.as_tensor(position_idxs, **common)
        return ttnn.as_tensor(
            position_idxs,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **common,
        )

    def get_rot_mats(
        self,
        position_idxs: Union[Any, ttnn.Tensor, np.ndarray],
        return_rot_idxs: bool = False,
    ) -> List[ttnn.Tensor]:
        """Decode cos/sin via embedding; ROW_MAJOR tables + padded indices -> fused TILE (no extra tilize)."""
        import torch

        if isinstance(position_idxs, np.ndarray):
            position_idxs = torch.from_numpy(position_idxs)
        if isinstance(position_idxs, ttnn.Tensor):
            rot_idx = position_idxs
            if rot_idx.dtype != ttnn.uint32:
                rot_idx = ttnn.typecast(rot_idx, dtype=ttnn.uint32)
            rot_idx = _pad_rot_idx_to_nearest_32(rot_idx, pad_tail=self._rot_idx_pad, pad_width=self._rot_idx_pad_width)
        elif isinstance(position_idxs, torch.Tensor):
            rot_idx = self.get_rot_idxs(position_idxs.reshape(-1))
        else:
            raise TypeError(f"position_idxs must be torch.Tensor or ttnn.Tensor, got {type(position_idxs)}")

        cos_emb = ttnn.embedding(rot_idx, self.cos_matrix_2d, layout=ttnn.TILE_LAYOUT)
        sin_emb = ttnn.embedding(rot_idx, self.sin_matrix_2d, layout=ttnn.TILE_LAYOUT)
        cos_sliced = ttnn.unsqueeze_to_4D(cos_emb)
        sin_sliced = ttnn.unsqueeze_to_4D(sin_emb)

        if return_rot_idxs:
            return [cos_sliced, sin_sliced], position_idxs
        return [cos_sliced, sin_sliced]

    def slice_rot_mats_prefill(self, start_pos: int, seq_len: int) -> list:
        """Cos/sin ``[1,1,seq_len,head_dim]`` on device from ``start_pos`` (pad if needed)."""
        mat_len = self.cos_matrix_prefill.shape[2]
        required_end = start_pos + seq_len
        if mat_len < required_end:
            raise RuntimeError(
                f"RoPE prefill needs positions through {required_end} but cos_matrix_prefill length is {mat_len}; "
                "build TtMinistral3RotaryEmbedding with a larger max_seq_len."
            )
        cos_slice = self.cos_matrix_prefill[:, :, start_pos:required_end, :]
        sin_slice = self.sin_matrix_prefill[:, :, start_pos:required_end, :]
        return [cos_slice, sin_slice]

    def get_both_trans_mats(self) -> dict:
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    @property
    def ministral3_config(self) -> Any:
        return self._ministral3_config


__all__ = [
    "TtMinistral3RotaryEmbedding",
    "ministral3_hf_cos_sin_tables",
    "ministral3_inv_freq_and_attention_scaling",
]
