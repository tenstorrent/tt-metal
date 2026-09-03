# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device weight loading for the Kimi K3 attention residuals.

The whole stack's queries are 5 MB of `[d]` vectors, so the `.tensorbin` cache saves no
measurable conversion time, and padding one row out to a tile costs it 83 MB on disk to
hold them. The reason to have one anyway is that `TtAttnRes` is then a peer in the block's
cache protocol instead of the one submodule a caller has to special-case.

`TtAttnRes` is a single instance over all 93 layers, so unlike the per-layer modules this
takes the whole checkpoint at once and the prefix names the op rather than a layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import NUM_LAYERS
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import CHECKPOINT_PREFIX, fold_queries


def _parallel_geometry(device: ttnn.MeshDevice, tensor_parallel_axis: int) -> tuple[tuple[int, int], int]:
    if tensor_parallel_axis not in (0, 1):
        raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {tensor_parallel_axis}")
    mesh_shape = tuple(device.shape)
    return mesh_shape, mesh_shape[tensor_parallel_axis]


def _cache_artifact_names(num_layers: int = NUM_LAYERS) -> tuple[str, ...]:
    per_layer = tuple(f"layers.{idx}.{part}" for idx in range(num_layers) for part in ("self_attention", "mlp"))
    return per_layer + ("output",)


def _cache_stem(cache_name_prefix: str, name: str) -> str:
    """The tensorbin identity, which is the caller's namespace and nothing else.

    Placement and dtype belong to the cache root and the serialized suffix; folding them
    in here would make two geometries collide on one stem or silently miss each other.
    """
    return f"{cache_name_prefix}.{name}"


def _serialized_path(cache_file: Path, dtype: ttnn.DataType) -> Path:
    return Path(f"{cache_file}_dtype_{dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin")


def walk_sites(pre, post, output=None):
    """The queries the walk issues, in issue order.

    Layer 0 has nothing sealed to read against, so `pre[0]` is held but never issued —
    which is why a 93-layer stack holds 187 queries and takes 186 reads. In the checkpoint
    that entry is a dead constant against a `1.7e-5` projection, which is the architecture
    agreeing with the driver.

    `output` is the single model-level read after the stack. A caller gating a block's own
    sites omits it.
    """
    sites = [post[0]]
    for pre_query, post_query in zip(pre[1:], post[1:]):
        sites += [pre_query, post_query]
    return sites if output is None else sites + [output]


@dataclass(frozen=True)
class AttnResWeights:
    """Every folded query the stack holds, placed on device and sharded on `d`."""

    pre: tuple[ttnn.Tensor, ...]
    post: tuple[ttnn.Tensor, ...]
    output: ttnn.Tensor
    tensor_parallel_size: int
    tensor_parallel_axis: int

    def walk_order(self) -> list[ttnn.Tensor]:
        """Every query this holds, in the order the walk issues them."""
        return walk_sites(self.pre, self.post, self.output)

    @classmethod
    def check_cache_complete(
        cls,
        cache_path: Path | None,
        cache_name_prefix: str,
        *,
        num_layers: int = NUM_LAYERS,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> bool:
        """Return whether every query's tensorbin exists at this dtype and layout."""
        if cache_path is None:
            return False
        cache_path = Path(cache_path)
        for name in _cache_artifact_names(num_layers):
            if not _serialized_path(cache_path / _cache_stem(cache_name_prefix, name), dtype).is_file():
                return False
        return True

    @classmethod
    def build_ttnn_cache(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        cache_path: Path,
        cache_name_prefix: str,
        mesh_device: ttnn.MeshDevice,
        *,
        num_layers: int = NUM_LAYERS,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_parallel_axis: int = 1,
        prefix: str = CHECKPOINT_PREFIX,
    ) -> None:
        """Write every tensorbin without spending device memory on the result."""
        result = load_attn_res_weights(
            mesh_device,
            state_dict,
            cache_path,
            cache_name_prefix=cache_name_prefix,
            num_layers=num_layers,
            dtype=dtype,
            tensor_parallel_axis=tensor_parallel_axis,
            prefix=prefix,
            _load_to_device=False,
        )
        assert result is None

    @classmethod
    def from_cache(
        cls,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        cache_name_prefix: str,
        *,
        num_layers: int = NUM_LAYERS,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_parallel_axis: int = 1,
    ) -> "AttnResWeights":
        """Construct device weights from a complete cache, touching no checkpoint."""
        result = load_attn_res_weights(
            mesh_device,
            None,
            cache_path,
            cache_name_prefix=cache_name_prefix,
            num_layers=num_layers,
            dtype=dtype,
            tensor_parallel_axis=tensor_parallel_axis,
        )
        assert result is not None
        return result


def load_attn_res_weights(
    device: ttnn.MeshDevice,
    state_dict: Mapping[str, torch.Tensor] | None,
    tensor_cache_path: Path | None = None,
    *,
    cache_name_prefix: str = "attn_res",
    num_layers: int = NUM_LAYERS,
    dtype: ttnn.DataType = ttnn.bfloat16,
    tensor_parallel_axis: int = 1,
    prefix: str = CHECKPOINT_PREFIX,
    _load_to_device: bool = True,
) -> AttnResWeights | None:
    """Fold the checkpoint's weight pairs into queries and shard them on `d`.

    A query is dotted against the residual stream, so it shards exactly like the stream.
    Passing `state_dict=None` loads from `tensor_cache_path` instead and reads no weights.
    """
    mesh_shape, tensor_parallel_size = _parallel_geometry(device, tensor_parallel_axis)
    if tensor_parallel_size <= 1:
        raise ValueError(
            f"AttnRes needs a tensor-parallel axis wider than one chip, got {tensor_parallel_size} "
            f"on mesh axis {tensor_parallel_axis} of a {mesh_shape} mesh"
        )
    if state_dict is None:
        if not _load_to_device:
            raise ValueError("building the AttnRes TTNN cache requires a state_dict")
        if not AttnResWeights.check_cache_complete(
            tensor_cache_path, cache_name_prefix, num_layers=num_layers, dtype=dtype
        ):
            raise FileNotFoundError(f"incomplete AttnRes TTNN cache for {cache_name_prefix!r} at {tensor_cache_path!r}")
    if tensor_cache_path is not None:
        tensor_cache_path = Path(tensor_cache_path)
        tensor_cache_path.mkdir(parents=True, exist_ok=True)

    mesh_dims = [None, None]
    mesh_dims[tensor_parallel_axis] = 3
    mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(mesh_dims), mesh_shape=mesh_shape)

    def place(query: torch.Tensor | None, name: str) -> ttnn.Tensor:
        cache_file = tensor_cache_path / _cache_stem(cache_name_prefix, name) if tensor_cache_path else None
        if query is None:
            return ttnn.load_tensor(_serialized_path(cache_file, dtype), device=device)
        return ttnn.as_tensor(
            query.reshape(1, 1, 1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device if _load_to_device else None,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _load_to_device else None,
            cache_file_name=cache_file,
        )

    if state_dict is None:
        pre, post, output = [None] * num_layers, [None] * num_layers, None
    else:
        pre, post, output = fold_queries(state_dict, num_layers, prefix)

    placed_pre = tuple(place(pre[idx], f"layers.{idx}.self_attention") for idx in range(num_layers))
    placed_post = tuple(place(post[idx], f"layers.{idx}.mlp") for idx in range(num_layers))
    placed_output = place(output, "output")

    if not _load_to_device:
        return None
    return AttnResWeights(
        pre=placed_pre,
        post=placed_post,
        output=placed_output,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
    )
