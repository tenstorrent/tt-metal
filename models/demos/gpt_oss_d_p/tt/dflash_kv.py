# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS DFlash context-KV construction.

The target prefill already produces the drafter's pre-norm hidden state with
``TtDFlashFeatureAccumulator``.  This module runs only the eight drafter
layers' context path (hidden norm, K/V projections, K norm and RoPE) and writes
decode-compatible K/V caches.  Attention and MLP are intentionally omitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_to_halfsplit_perm

from .attention.kv_cache import GptOssKVCache
from .attention.operations import apply_rope
from .rope import build_indexed_rope, build_transformation_mat, build_yarn_cos_sin

DFLASH_NUM_LAYERS = 8
DFLASH_NUM_KV_HEADS = 8

_HIDDEN_NORM = "hidden_norm.weight"
_K_PROJ = "layers.{i}.self_attn.k_proj.weight"
_V_PROJ = "layers.{i}.self_attn.v_proj.weight"
_K_NORM = "layers.{i}.self_attn.k_norm.weight"


@dataclass(frozen=True)
class DFlashKVConfig:
    checkpoint_path: Path
    hidden_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    yarn_factor: float
    yarn_orig_max_pos: int
    yarn_beta_fast: float
    yarn_beta_slow: float

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        expected_hidden_size: int,
        expected_num_hidden_layers: int = DFLASH_NUM_LAYERS,
        expected_num_key_value_heads: int = DFLASH_NUM_KV_HEADS,
        expected_head_dim: int = 64,
    ) -> "DFlashKVConfig":
        path = Path(checkpoint_path)
        config_file = path / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"DFlash checkpoint is missing {config_file}")
        data = json.loads(config_file.read_text())
        rope = data.get("rope_scaling") or {}
        actual = {
            "hidden_size": int(data.get("hidden_size", -1)),
            "num_hidden_layers": int(data.get("num_hidden_layers", -1)),
            "num_key_value_heads": int(data.get("num_key_value_heads", -1)),
            "head_dim": int(data.get("head_dim", -1)),
        }
        expected = {
            "hidden_size": expected_hidden_size,
            "num_hidden_layers": expected_num_hidden_layers,
            "num_key_value_heads": expected_num_key_value_heads,
            "head_dim": expected_head_dim,
        }
        for name, value in expected.items():
            if actual[name] != value:
                raise ValueError(f"DFlash checkpoint {name}={actual[name]}, expected {value}")
        config = cls(
            checkpoint_path=path,
            hidden_size=actual["hidden_size"],
            num_hidden_layers=actual["num_hidden_layers"],
            num_key_value_heads=actual["num_key_value_heads"],
            head_dim=actual["head_dim"],
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 150000.0)),
            yarn_factor=float(rope.get("factor", 32.0)),
            yarn_orig_max_pos=int(rope.get("original_max_position_embeddings", 4096)),
            yarn_beta_fast=float(rope.get("beta_fast", 32.0)),
            yarn_beta_slow=float(rope.get("beta_slow", 1.0)),
        )
        # Read and validate only the tail subset, failing before a target prefill is launched.
        load_dflash_kv_weights(config)
        return config

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


def required_dflash_kv_keys(config: DFlashKVConfig) -> tuple[str, ...]:
    keys = [_HIDDEN_NORM]
    for layer_idx in range(config.num_hidden_layers):
        keys.extend(
            (
                _K_PROJ.format(i=layer_idx),
                _V_PROJ.format(i=layer_idx),
                _K_NORM.format(i=layer_idx),
            )
        )
    return tuple(keys)


def load_dflash_kv_weights(config: DFlashKVConfig) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    weights_file = config.checkpoint_path / "model.safetensors"
    if not weights_file.is_file():
        raise FileNotFoundError(f"DFlash checkpoint is missing {weights_file}")
    expected_shapes = {_HIDDEN_NORM: (config.hidden_size,)}
    for layer_idx in range(config.num_hidden_layers):
        expected_shapes[_K_PROJ.format(i=layer_idx)] = (config.kv_dim, config.hidden_size)
        expected_shapes[_V_PROJ.format(i=layer_idx)] = (config.kv_dim, config.hidden_size)
        expected_shapes[_K_NORM.format(i=layer_idx)] = (config.head_dim,)
    result = {}
    with safe_open(str(weights_file), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for key, shape in expected_shapes.items():
            if key not in available:
                raise KeyError(f"DFlash checkpoint {weights_file} is missing key {key!r}")
            value = handle.get_tensor(key)
            if tuple(value.shape) != shape:
                raise ValueError(f"DFlash {key} has shape {tuple(value.shape)}, expected {shape}")
            result[key] = value
    return result


def _meta_permutation(head_dim: int) -> torch.Tensor:
    """HF/Qwen half-split output order -> TT decode's interleaved RoPE order."""
    return torch.argsort(interleaved_to_halfsplit_perm(head_dim))


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(value.dtype)


def reference_dflash_kv(
    reduced_hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: DFlashKVConfig,
    *,
    start_pos: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Independent torch context-path reference in decode-compatible Meta K layout.

    ``reduced_hidden`` is ``[..., sequence, hidden]``. Returned layer lists use
    ``[..., kv_heads, sequence, head_dim]``.
    """
    if reduced_hidden.shape[-1] != config.hidden_size:
        raise ValueError(
            f"reduced_hidden width={reduced_hidden.shape[-1]}, expected DFlash hidden={config.hidden_size}"
        )
    hidden = _rms_norm(reduced_hidden, weights[_HIDDEN_NORM], config.rms_norm_eps)
    sequence = hidden.shape[-2]
    cos, sin = build_yarn_cos_sin(
        start_pos + sequence,
        config.head_dim,
        rope_theta=config.rope_theta,
        yarn_factor=config.yarn_factor,
        yarn_orig_max_pos=config.yarn_orig_max_pos,
        yarn_beta_fast=config.yarn_beta_fast,
        yarn_beta_slow=config.yarn_beta_slow,
    )
    cos = cos[0, 0, start_pos : start_pos + sequence].to(hidden.dtype)
    sin = sin[0, 0, start_pos : start_pos + sequence].to(hidden.dtype)
    src = _meta_permutation(config.head_dim)
    keys, values = [], []
    for layer_idx in range(config.num_hidden_layers):
        k_weight = weights[_K_PROJ.format(i=layer_idx)].reshape(
            config.num_key_value_heads, config.head_dim, config.hidden_size
        )[:, src, :]
        k = F.linear(hidden, k_weight.reshape(config.kv_dim, config.hidden_size).to(hidden.dtype))
        v = F.linear(hidden, weights[_V_PROJ.format(i=layer_idx)].to(hidden.dtype))
        prefix = k.shape[:-1]
        k = k.reshape(*prefix, config.num_key_value_heads, config.head_dim).movedim(-2, -3)
        v = v.reshape(*prefix, config.num_key_value_heads, config.head_dim).movedim(-2, -3)
        k_norm = weights[_K_NORM.format(i=layer_idx)][src]
        k = _rms_norm(k, k_norm, config.rms_norm_eps)
        # Meta/interleaved rotate-half: [-x1, x0, -x3, x2, ...].
        rotated = torch.stack((-k[..., 1::2], k[..., 0::2]), dim=-1).flatten(-2)
        k = k * cos + rotated * sin
        keys.append(k)
        values.append(v)
    return keys, values


class TtDFlashKVBuilder:
    """Eight-layer TTNN context path writing a caller-owned ``GptOssKVCache``."""

    def __init__(
        self,
        mesh_device,
        config: DFlashKVConfig,
        weights: dict[str, torch.Tensor],
        *,
        max_seq_len: int,
        chunk_sizes: tuple[int, ...],
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 1,
        weight_dtype=ttnn.bfloat8_b,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.topology = topology
        self.num_links = num_links
        self.tp = mesh_device.shape[tp_axis]
        if config.num_key_value_heads != self.tp:
            raise ValueError(
                f"DFlash KV heads ({config.num_key_value_heads}) must equal TP ({self.tp}); "
                "the decode-compatible cache holds one head per TP column"
            )
        dims = [None, None]
        dims[tp_axis] = 1
        column_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        if config.hidden_size % ttnn.TILE_SIZE:
            raise ValueError(f"DFlash hidden size {config.hidden_size} must be tile-aligned")
        self.hidden_norm = ttnn.as_tensor(
            weights[_HIDDEN_NORM].reshape(1, 1, -1, ttnn.TILE_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        src = _meta_permutation(config.head_dim)
        self.k_proj, self.v_proj, self.k_norm = [], [], []
        for layer_idx in range(config.num_hidden_layers):
            kw = weights[_K_PROJ.format(i=layer_idx)].reshape(
                config.num_key_value_heads, config.head_dim, config.hidden_size
            )[:, src, :]
            kw = kw.reshape(config.kv_dim, config.hidden_size)
            self.k_proj.append(self._linear_weight(kw, column_mapper, weight_dtype))
            self.v_proj.append(self._linear_weight(weights[_V_PROJ.format(i=layer_idx)], column_mapper, weight_dtype))
            norm = weights[_K_NORM.format(i=layer_idx)][src].reshape(1, 1, -1, ttnn.TILE_SIZE)
            self.k_norm.append(
                ttnn.as_tensor(
                    norm,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate,
                )
            )
        self.rope = {
            chunk_size: build_indexed_rope(
                mesh_device,
                head_dim=config.head_dim,
                max_seq_len=max_seq_len,
                chunk_size=chunk_size,
                sp_axis=sp_axis,
                rope_theta=config.rope_theta,
                yarn_factor=config.yarn_factor,
                yarn_orig_max_pos=config.yarn_orig_max_pos,
                yarn_beta_fast=config.yarn_beta_fast,
                yarn_beta_slow=config.yarn_beta_slow,
            )
            for chunk_size in chunk_sizes
        }
        self.transformation_mat = build_transformation_mat(mesh_device)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _linear_weight(self, weight: torch.Tensor, mapper, dtype):
        return ttnn.as_tensor(
            weight.transpose(-2, -1).contiguous(),
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def _split_heads(self, projected):
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            projected,
            num_heads=1,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return heads

    def forward(
        self,
        reduced_hidden,
        kv_cache: GptOssKVCache,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        chunk_size: int,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        layer_ack_base: int = 0,
    ) -> None:
        cfg = self.config
        if kv_cache.bounded_sliding:
            raise ValueError("DFlash KV cache must use the unbounded decode-compatible layout")
        if kv_cache.num_layers != cfg.num_hidden_layers:
            raise ValueError(f"DFlash cache has {kv_cache.num_layers} layers, expected {cfg.num_hidden_layers}")
        if not 0 <= slot_id < kv_cache.num_users:
            raise ValueError(f"slot_id {slot_id} out of range [0, {kv_cache.num_users})")
        if chunk_size not in self.rope:
            raise ValueError(f"chunk_size={chunk_size} has no DFlash RoPE table")
        if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
            raise ValueError(f"actual_start={actual_start} must be non-negative and {ttnn.TILE_SIZE}-token aligned")
        if not actual_start < actual_end <= actual_start + chunk_size:
            raise ValueError(
                f"real range [{actual_start}, {actual_end}) must lie within "
                f"chunk [{actual_start}, {actual_start + chunk_size})"
            )
        if actual_start + chunk_size > kv_cache.max_seq_len:
            raise ValueError(
                f"chunk [{actual_start}, {actual_start + chunk_size}) exceeds "
                f"DFlash cache capacity {kv_cache.max_seq_len}"
            )
        gathered_hidden = reduced_hidden
        if self.tp > 1:
            gathered_hidden = ttnn.all_gather(
                reduced_hidden,
                dim=3,
                cluster_axis=self.tp_axis,
                num_links=self.num_links,
                topology=self.topology,
            )
        target_hidden = ttnn.rms_norm(
            gathered_hidden,
            weight=self.hidden_norm,
            epsilon=cfg.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if gathered_hidden is not reduced_hidden:
            ttnn.deallocate(gathered_hidden)
        for layer_idx in range(cfg.num_hidden_layers):
            k = ttnn.linear(
                target_hidden,
                self.k_proj[layer_idx],
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v = ttnn.linear(
                target_hidden,
                self.v_proj[layer_idx],
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k, v = self._split_heads(k), self._split_heads(v)
            k = ttnn.rms_norm(
                k,
                weight=self.k_norm[layer_idx],
                epsilon=cfg.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            k = apply_rope(
                k,
                self.rope[chunk_size],
                self.transformation_mat,
                kv_actual_global=actual_start,
                cluster_axis=self.sp_axis,
            )
            if k.dtype != kv_cache.k.dtype:
                cast = ttnn.typecast(k, kv_cache.k.dtype)
                ttnn.deallocate(k)
                k = cast
            if v.dtype != kv_cache.v.dtype:
                cast = ttnn.typecast(v, kv_cache.v.dtype)
                ttnn.deallocate(v)
                v = cast
            for cache, value in ((kv_cache.k, k), (kv_cache.v, v)):
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    value,
                    slot_idx=slot_id,
                    layer_idx=layer_idx,
                    num_layers=cfg.num_hidden_layers,
                    kv_actual_global=actual_start,
                    cluster_axis=self.sp_axis,
                    valid_global=actual_end,
                )
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    cache,
                    slot_idx=slot_id,
                    layer_idx=layer_idx,
                    num_layers=cfg.num_hidden_layers,
                    valid_global=actual_end,
                    chunk_size_global=chunk_size,
                    cluster_axis=self.sp_axis,
                    pad_align=ttnn.TILE_SIZE,
                )
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if on_layer_complete is not None:
                ttnn.synchronize_device(self.mesh_device)
                on_layer_complete(layer_ack_base + layer_idx)
        ttnn.deallocate(target_hidden)
