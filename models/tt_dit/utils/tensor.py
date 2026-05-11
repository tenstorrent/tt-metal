# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import EllipsisType


# Module-level lazy thread pool for parallel host-side shard reassembly.
# uint8/bf16 strided copies through PyTorch's TensorIterator release the GIL,
# so dispatching the per-shard scatters across a few threads gives near-linear
# scaling. A persistent pool avoids paying ~1 ms of thread-startup per call.
# Callers can override via the `pool` parameter on `_reassemble_2d` /
# `fast_device_to_host` if they manage their own pool lifetime.
_DEFAULT_REASSEMBLE_POOL: ThreadPoolExecutor | None = None
_DEFAULT_REASSEMBLE_WORKERS = min(8, os.cpu_count() or 8)


def _get_default_reassemble_pool() -> ThreadPoolExecutor:
    global _DEFAULT_REASSEMBLE_POOL
    if _DEFAULT_REASSEMBLE_POOL is None:
        _DEFAULT_REASSEMBLE_POOL = ThreadPoolExecutor(
            max_workers=_DEFAULT_REASSEMBLE_WORKERS,
            thread_name_prefix="tt_dit_reassemble",
        )
    return _DEFAULT_REASSEMBLE_POOL


# Try to load the C++/AVX2 planar concat extension (see
# models/tt_dit/utils/cpp/).  When available, _yuv_planar_d2h uses it as a
# drop-in replacement for the Python thread-pool scatter — same byte layout,
# ~2× faster end-to-end with a persistent output buffer.  The wrapper sets
# HAS_CPP_PLANAR_CONCAT=False on unbuilt / non-AVX2 hosts; the existing
# torch_threaded path remains the fallback.
from .planar_concat import HAS_CPP_PLANAR_CONCAT
from .planar_concat import planar_concat_cpp as _planar_concat_cpp_impl

# Persistent output buffer for the C++ planar concat fast path.  The Wan VAE
# encode loop allocates a 112 MB buffer per frame batch otherwise — fresh
# np.empty pays ~6–8 ms of first-touch page-fault overhead on hosts without
# THP=always, which dwarfs the kernel itself.  Reuse across calls eliminates
# that tax; shape changes (e.g. switching resolutions) reallocate lazily.
_PLANAR_OUT_BUF: np.ndarray | None = None
_PLANAR_OUT_SHAPE: tuple[int, int] | None = None


def _get_planar_out_buf(T: int, row_stride: int) -> np.ndarray:
    global _PLANAR_OUT_BUF, _PLANAR_OUT_SHAPE
    shape = (T, row_stride)
    if _PLANAR_OUT_SHAPE != shape:
        _PLANAR_OUT_BUF = np.empty(shape, dtype=np.uint8)
        _PLANAR_OUT_SHAPE = shape
    return _PLANAR_OUT_BUF


def typed_tensor(
    x: torch.Tensor,
    dtype: ttnn.DataType,
    device: ttnn.Device | None = None,
    mesh_axis=None,
    shard_dim=None,
    layout=ttnn.TILE_LAYOUT,
    on_host=False,
) -> ttnn.Tensor:
    """
    Replicates or shards a tensor based on the mesh_axis and shard_dim
    """
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=None if on_host else device,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor(
    x: torch.Tensor,
    device: ttnn.Device | None = None,
    mesh_axis=None,
    shard_dim=None,
    layout=ttnn.TILE_LAYOUT,
    on_host=False,
) -> ttnn.Tensor:
    return typed_tensor(
        x,
        ttnn.bfloat16,
        device,
        mesh_axis,
        shard_dim,
        layout,
        on_host=on_host,
    )


def float32_tensor(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    return typed_tensor(x, ttnn.float32, device, mesh_axis, shard_dim, layout)


def bf16_tensor_host(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )


def typed_tensor_2dshard(
    x: torch.Tensor,
    device: ttnn.Device,
    shard_mapping: dict[int, int],
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    on_host=False,
) -> ttnn.Tensor:
    assert len(shard_mapping) == 2
    assert all(0 <= k <= 1 and 0 <= v < len(x.shape) for k, v in shard_mapping.items())
    mapper_dims = [None, None]
    for k, v in shard_mapping.items():
        mapper_dims[k] = v
    mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)
    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=None if on_host else device,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor_2dshard(
    x: torch.Tensor,
    device: ttnn.Device,
    shard_mapping: dict[int, int],
    layout=ttnn.TILE_LAYOUT,
    on_host: bool = False,
) -> ttnn.Tensor:
    return typed_tensor_2dshard(x, device, shard_mapping, layout, dtype=ttnn.bfloat16, on_host=on_host)


def from_torch(
    x: torch.Tensor,
    /,
    *,
    device: ttnn.MeshDevice | None = None,
    layout: ttnn.Layout = ttnn.Layout.TILE,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    pad_value: float | None = None,
    mesh_axes: Sequence[int | None | EllipsisType] | None = None,
    on_host: bool = False,
) -> ttnn.Tensor:
    """Convert a torch.Tensor to a ttnn.Tensor with convenient mesh distribution."""
    if mesh_axes is not None:
        if device is None:
            msg = "device must be specified if mesh_axes is given"
            raise ValueError(msg)

        mesh_rank = len(list(device.shape))
        mesh_axes = canonicalize_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

        placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
        placements = [ttnn.PlacementShard(p) if p is not None else ttnn.PlacementReplicate() for p in placements]
        mesh_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))
    else:
        mesh_mapper = None

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=None if on_host else device,
        mesh_mapper=mesh_mapper,
        pad_value=pad_value,
    )


def to_torch(
    x: ttnn.Tensor,
    /,
    *,
    mesh_axes: Sequence[int | None | EllipsisType] | None = None,
    composer_device: ttnn.MeshDevice | None = None,
) -> torch.Tensor:
    """Converts a ttnn.Tensor to a torch.Tensor.

    Strips away redundant data returned by calling ttnn.to_torch on a replicated tensor. If the
    tensor is distributed and not on device, composer_device must be provided.
    """
    if x.tensor_topology().distribution_shape().mesh_size() == 1:
        return ttnn.to_torch(x)

    if mesh_axes is None:
        mesh_axes = (None,) * len(x.shape)

    composer_device = composer_device or x.device()
    if composer_device is None:
        msg = "composer_device must be specified for distributed host tensors"
        raise ValueError(msg)

    mesh_rank = len(list(composer_device.shape))
    mesh_axes = canonicalize_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

    replicated_mesh_axes = list(set(range(mesh_rank)) - {axis for axis in mesh_axes if axis is not None})
    mesh_axes = replicated_mesh_axes + list(mesh_axes)

    placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
    assert all(p is not None for p in placements)

    mesh_composer = ttnn.create_mesh_composer(composer_device, ttnn.MeshComposerConfig(placements))

    x = x.reshape([1] * len(replicated_mesh_axes) + list(x.shape))
    return ttnn.to_torch(x, mesh_composer=mesh_composer)[(0,) * len(replicated_mesh_axes)]


def verify_tensor_mesh_axes(mesh_axes: Sequence[int | None], /, *, tensor_rank: int, mesh_rank: int) -> None:
    """Validates tensor mesh axes specification."""
    if len(mesh_axes) != tensor_rank:
        msg = f"mesh axis list {tuple(mesh_axes)} should have length {tensor_rank}"
        raise ValueError(msg)

    for axis in mesh_axes:
        if axis is not None and (axis < 0 or axis >= mesh_rank):
            msg = f"all mesh axes in mesh axis list {tuple(mesh_axes)} should be positive and smaller than {mesh_rank}"
            raise ValueError(msg)

    non_none_values = [p for p in mesh_axes if p is not None]
    if len(non_none_values) != len(set(non_none_values)):
        msg = f"mesh axis list {tuple(mesh_axes)} contains duplicate mesh axis assignments"
        raise ValueError(msg)


def canonicalize_tensor_mesh_axes(
    mesh_axes: Sequence[int | None | EllipsisType], /, *, tensor_rank: int, mesh_rank: int
) -> tuple[int | None, ...]:
    """Canonicalizes mesh axes specification by expanding Ellipsis and validating."""
    mesh_axes = list(mesh_axes)

    if Ellipsis in mesh_axes:
        if mesh_axes.count(Ellipsis) > 1:
            msg = "mesh_axes can contain at most one Ellipsis"
            raise ValueError(msg)

        ellipsis_index = mesh_axes.index(Ellipsis)
        mesh_axes[ellipsis_index : ellipsis_index + 1] = [None] * (tensor_rank - len(mesh_axes) + 1)

    verify_tensor_mesh_axes(mesh_axes, tensor_rank=tensor_rank, mesh_rank=mesh_rank)

    return tuple(mesh_axes)


def _invert_placements(placements: Sequence[int | None], *, output_rank: int) -> tuple[int | None, ...]:
    out = [None] * output_rank

    for i, p in enumerate(placements):
        if p is not None:
            out[p] = i

    return tuple(out)


def local_device_to_torch(tt_tensor: ttnn.Tensor) -> torch.Tensor:
    """Convert a ttnn device tensor to a torch tensor by reading from the local device.

    In a distributed environment, iterates over the mesh coordinates to find the
    tensor shard that belongs to the local device before calling ``ttnn.to_torch``.
    """
    mesh_device = tt_tensor.device()
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    coords = list(tt_tensor.tensor_topology().mesh_coords())
    device_tensors = ttnn.get_device_tensors(tt_tensor)

    torch_tensor = None
    for coord, device_tensor in zip(coords, device_tensors):
        if view is None or view.is_local(coord):
            torch_tensor = ttnn.to_torch(device_tensor)
            break

    if torch_tensor is None:
        msg = "Failed to find local device tensor"
        raise RuntimeError(msg)
    return torch_tensor


_to_torch_zero_copy_warned = False


def _to_torch_zero_copy(t: ttnn.Tensor) -> torch.Tensor:
    """Convert a host ttnn tensor to a PyTorch tensor, preferring zero-copy.

    Uses ``to_torch_with_padded_shape`` when available — for ROW_MAJOR host
    tensors this wraps the existing buffer directly (zero-copy) instead of
    copying through ``decode_tensor_data`` as ``to_torch`` always does.

    Falls back to ``to_torch`` if the method is removed, with a one-time
    warning so the performance regression is visible.

    TODO: Once ``to_torch`` supports a ``padded_output`` parameter (or the
    zero-copy path becomes the default for ROW_MAJOR), switch to that and
    remove this helper.
    """
    global _to_torch_zero_copy_warned
    try:
        return t.to_torch_with_padded_shape()
    except AttributeError:
        if not _to_torch_zero_copy_warned:
            import logging

            logging.getLogger(__name__).warning(
                "to_torch_with_padded_shape unavailable, falling back to to_torch (slower d2h)"
            )
            _to_torch_zero_copy_warned = True
        return ttnn.to_torch(t)


_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint8: torch.uint8,
    ttnn.uint16: torch.int16,
    ttnn.int32: torch.int32,
}


def _host_buffer_to_torch(buf, padded_shape: list[int], tt_dtype: ttnn.DataType) -> torch.Tensor:
    """Zero-copy conversion of a HostBuffer to a torch tensor.

    Uses the DLPack protocol to get a uint8 view of the raw buffer memory,
    then reinterprets it as the correct dtype and reshapes.
    """
    torch_dtype = _TTNN_TO_TORCH_DTYPE[tt_dtype]
    raw = torch.from_dlpack(buf)
    return raw.view(torch_dtype).reshape(padded_shape)


def float_to_unit_range(t: ttnn.Tensor) -> ttnn.Tensor:
    """On-device denormalization: map from [-1.0, 1.0] to [0.0, 1.0]."""
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    t = ttnn.add(t, 1.0)
    t = ttnn.multiply(t, 0.5)
    t = ttnn.clamp(t, min=0.0, max=1.0)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    return t


def float_to_uint8(t: ttnn.Tensor) -> ttnn.Tensor:
    """On-device float-to-uint8: map from [-1.0, 1.0] to [0, 255]"""
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    t = ttnn.add(t, 1.0)  # shift to [0, 2.0]
    t = ttnn.multiply(t, 0.5 * 255.0)  # scale to [0, 1.0] then [0, 255]
    t = ttnn.clamp(t, min=0.0, max=255.0)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.typecast(t, ttnn.uint8)


def _get_inter_host_axis(mesh_device: ttnn.MeshDevice, view, mesh_shape: tuple[int, ...]) -> int:
    """Return the mesh axis that spans multiple hosts (0 or 1).

    In a 2D mesh, one axis typically spans hosts (inter-host) while the other
    is fully local to each host (intra-host).  Finds a local coordinate first,
    then checks whether varying each axis stays local.
    """
    from ttnn._ttnn.multi_device import MeshCoordinate

    # Find any coordinate that is local to this host.
    ref_r, ref_c = 0, 0
    for r in range(mesh_shape[0]):
        for c in range(mesh_shape[1]):
            if view.is_local(MeshCoordinate(r, c)):
                ref_r, ref_c = r, c
                break
        else:
            continue
        break

    # Check axis 0: vary row while keeping the local column fixed.
    if mesh_shape[0] > 1:
        if not all(view.is_local(MeshCoordinate(r, ref_c)) for r in range(mesh_shape[0])):
            return 0
    # Check axis 1: vary column while keeping the local row fixed.
    if mesh_shape[1] > 1:
        if not all(view.is_local(MeshCoordinate(ref_r, c)) for c in range(mesh_shape[1])):
            return 1
    # Both axes are fully local — shouldn't happen in a true distributed env.
    return 0


def _reassemble_2d(
    mesh_coords: list,
    shards: list[torch.Tensor],
    shard_shape: list[int],
    mesh_shape: tuple[int, ...],
    concat_dims: list[int | None],
    permute: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    *,
    pool: ThreadPoolExecutor | None = None,
) -> torch.Tensor:
    """Reassemble per-device shards into a single tensor for a 2D mesh.

    Per-shard scatters are dispatched across a ThreadPoolExecutor — uint8/bf16
    strided copies through PyTorch's TensorIterator release the GIL, so a few
    threads make near-linear progress.  When *permute* is given the source
    permute view is fed directly to ``copy_()`` so no contiguous intermediate
    is materialised; the permute and the scatter share a single strided pass.

    Pass *pool* to reuse a persistent pool; otherwise a module-level lazy
    default is used.
    """
    d0, d1 = concat_dims

    if d0 is not None and d1 is not None:
        s0, s1 = shard_shape[d0], shard_shape[d1]
        full_shape = list(shard_shape)
        full_shape[d0] *= mesh_shape[0]
        full_shape[d1] *= mesh_shape[1]
        ndim = len(full_shape)

        if pool is None:
            pool = _get_default_reassemble_pool()

        if permute is not None:
            out_shape = [full_shape[p] for p in permute]
            out_dtype = dtype if dtype is not None else shards[0].dtype
            perm_list = list(permute)
            d0_out = perm_list.index(d0)
            d1_out = perm_list.index(d1)

            out = torch.empty(out_shape, dtype=out_dtype)

            def _scatter_perm(coord, shard):
                r, c = int(coord[0]), int(coord[1])
                slices = [slice(None)] * ndim
                slices[d0_out] = slice(r * s0, (r + 1) * s0)
                slices[d1_out] = slice(c * s1, (c + 1) * s1)
                # Strided->strided copy in one pass: no .contiguous() materialisation.
                out[tuple(slices)].copy_(shard.permute(*permute))

            futures = [pool.submit(_scatter_perm, coord, shard) for coord, shard in zip(mesh_coords, shards)]
            for f in futures:
                f.result()
            return out

        out_dtype = dtype if dtype is not None else shards[0].dtype
        out = torch.empty(full_shape, dtype=out_dtype)

        def _scatter(coord, shard):
            r, c = int(coord[0]), int(coord[1])
            slices = [slice(None)] * ndim
            slices[d0] = slice(r * s0, (r + 1) * s0)
            slices[d1] = slice(c * s1, (c + 1) * s1)
            out[tuple(slices)].copy_(shard)

        futures = [pool.submit(_scatter, coord, shard) for coord, shard in zip(mesh_coords, shards)]
        for f in futures:
            f.result()
        return out

    if d0 is not None:
        by_pos = sorted(zip(mesh_coords, shards), key=lambda x: int(x[0][0]))
        return torch.cat([s for _, s in by_pos], dim=d0)
    if d1 is not None:
        by_pos = sorted(zip(mesh_coords, shards), key=lambda x: int(x[0][1]))
        return torch.cat([s for _, s in by_pos], dim=d1)
    return shards[0]


def fast_device_to_host(
    tt_tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    concat_dims: list[int | None],
    ccl_manager=None,
    root: int | None = None,
    *,
    pre_transfer_fn: Callable[[ttnn.Tensor], ttnn.Tensor] | None = None,
    permute: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> torch.Tensor | None:
    """Fast D2H transfer using async DMA and zero-copy to_torch.

    On a single-host system, this avoids the on-device all_gather by reading
    all per-device shards concurrently with async DMA, converting to PyTorch
    with zero-copy when possible, and concatenating on host.

    On a multi-host (distributed) system, this uses a hybrid approach: an
    on-device all_gather for the inter-host axis only, then fast async DMA +
    zero-copy for local shards with host-side concatenation for the intra-host
    axis.

    Args:
        tt_tensor: Multi-device ttnn tensor on ``mesh_device``.
        mesh_device: The mesh device.
        concat_dims: Per mesh axis, the tensor dimension to concatenate along,
            or ``None`` to skip that axis.  E.g. ``[3, 4]`` means concatenate
            along dim 3 for mesh axis 0 and dim 4 for mesh axis 1.
        ccl_manager: Optional :class:`CCLManager` instance.  Required for
            multi-host environments where only local devices are accessible.
        root: If set, only the host with this MPI rank performs the D2H
            transfer and returns the assembled tensor; all other ranks return
            ``None``.  If ``None`` (default), all ranks perform D2H.
        pre_transfer_fn: Optional on-device transformation applied just before
            the DMA read.  For multi-host, runs after ``mesh_partition``; for
            single-host, runs right before ``.cpu()``.  Typical use is
            ``float_to_uint8`` to shrink data before the PCIe transfer.
        permute: If set, each shard is permuted before being written into the
            output.  Fuses the permutation into the scatter write so that no
            intermediate tensor in the original layout is ever materialised.
        dtype: Output dtype.  When combined with ``permute``, the dtype
            conversion is fused into the scatter write (single-pass copy).
        pool: Optional ThreadPoolExecutor for parallel host-side reassembly.
            If ``None``, a module-level lazy default pool is used.  Pass a
            persistent pool to avoid per-call thread-startup overhead in tight
            loops, or to share workers with other concurrent host work.
    """
    mesh_shape = tuple(mesh_device.shape)

    if len(mesh_shape) != 2:
        raise ValueError(
            f"fast_device_to_host only supports 2D meshes, got mesh shape {mesh_shape} (ndim={len(mesh_shape)})"
        )
    if len(concat_dims) != 2:
        raise ValueError(
            f"concat_dims must have exactly 2 elements for a 2D mesh, got {len(concat_dims)}: {concat_dims}"
        )

    # --- Multi-host: hybrid on-device collective + fast local DMA -----------
    if ttnn.using_distributed_env():
        if ccl_manager is None:
            msg = "fast_device_to_host requires ccl_manager in a distributed (multi-host) environment"
            raise ValueError(msg)

        view = mesh_device.get_view()
        rank = int(ttnn.distributed_context_get_rank())

        inter_host_axis = _get_inter_host_axis(mesh_device, view, mesh_shape)
        intra_host_axis = 1 - inter_host_axis

        # Step 1: On-device all_gather + repeat + mesh_partition.
        # All_gather replicates the inter-host axis.  Repeat + mesh_partition
        # then re-shard it so every local device holds *unique* data,
        # maximising PCIe bandwidth during the DMA read.
        gathered_tensor = tt_tensor
        inter_dim = concat_dims[inter_host_axis]
        if inter_dim is not None and mesh_shape[inter_host_axis] > 1:
            gathered_tensor = ttnn.to_layout(gathered_tensor, ttnn.TILE_LAYOUT)
            gathered_tensor = ccl_manager.all_gather(
                gathered_tensor,
                dim=inter_dim,
                mesh_axis=inter_host_axis,
                use_hyperparams=True,
                use_persistent_buffer=True,
            )
            n_hosts = int(ttnn.distributed_context_get_size())
            if n_hosts > 1:
                repeat_dims = [1] * len(gathered_tensor.shape)
                repeat_dims[inter_dim] = n_hosts
                gathered_tensor = ttnn.repeat(gathered_tensor, repeat_dims)
                gathered_tensor = ttnn.mesh_partition(gathered_tensor, dim=inter_dim, cluster_axis=inter_host_axis)
            if pre_transfer_fn is not None:
                gathered_tensor = pre_transfer_fn(gathered_tensor)
            else:
                gathered_tensor = ttnn.to_layout(gathered_tensor, ttnn.ROW_MAJOR_LAYOUT)
        elif pre_transfer_fn is not None:
            gathered_tensor = pre_transfer_fn(gathered_tensor)

        # Step 2: Only root rank (if specified) does D2H.
        if root is not None and rank != root:
            return None

        # Step 3: DMA all local shards and reassemble on host.
        # Single .cpu() on the mesh tensor batches all local DMA reads into
        # one C++ dispatch — the reader thread pool processes all device
        # completion queues in parallel.
        host_tensor = gathered_tensor.cpu(blocking=False)
        ttnn.synchronize_device(mesh_device)

        # Extract local shard buffers via get_shard (zero-copy, no MPI).
        host_mesh_coords = list(host_tensor.tensor_topology().mesh_coords())
        distributed_buf = host_tensor.host_buffer()
        tt_dtype = host_tensor.dtype
        padded_shape = list(host_tensor.padded_shape)
        logical_shape = list(host_tensor.shape)
        trim = tuple(slice(0, d) for d in logical_shape)

        local_coords_and_bufs = []
        for c in host_mesh_coords:
            if not view.is_local(c):
                continue
            buf = distributed_buf.get_shard(c)
            if buf is not None:
                local_coords_and_bufs.append((c, buf))

        shards = [_host_buffer_to_torch(buf, padded_shape, tt_dtype)[trim] for _, buf in local_coords_and_bufs]

        # Build local mesh shape and 0-based coordinates for _reassemble_2d.
        local_inter_positions = sorted({int(c[inter_host_axis]) for c, _ in local_coords_and_bufs})
        local_intra_positions = sorted({int(c[intra_host_axis]) for c, _ in local_coords_and_bufs})
        local_mesh_shape = [0, 0]
        local_mesh_shape[inter_host_axis] = len(local_inter_positions)
        local_mesh_shape[intra_host_axis] = len(local_intra_positions)
        local_mesh_shape = tuple(local_mesh_shape)

        inter_remap = {pos: i for i, pos in enumerate(local_inter_positions)}
        intra_remap = {pos: i for i, pos in enumerate(local_intra_positions)}
        local_coords = []
        for c, _ in local_coords_and_bufs:
            coord = [0, 0]
            coord[inter_host_axis] = inter_remap[int(c[inter_host_axis])]
            coord[intra_host_axis] = intra_remap[int(c[intra_host_axis])]
            local_coords.append(tuple(coord))

        return _reassemble_2d(
            local_coords, shards, logical_shape, local_mesh_shape, concat_dims, permute, dtype, pool=pool
        )

    # --- Single-host: async DMA on all devices + host-side concat -----------

    # Grab mesh coordinates from the device tensor before DMA.
    mesh_coords = list(tt_tensor.tensor_topology().mesh_coords())

    if pre_transfer_fn is not None:
        tt_tensor = pre_transfer_fn(tt_tensor)

    # Single .cpu() on the mesh tensor batches all DMA reads into one C++
    # dispatch — host buffers are allocated in parallel and the reader thread
    # pool processes all completion-queue reads concurrently.
    host_tensor = tt_tensor.cpu(blocking=False)
    ttnn.synchronize_device(mesh_device)

    # Extract per-shard host tensors (single-host: just wraps each shard).
    host_shard_tensors = ttnn.get_device_tensors(host_tensor)

    # Zero-copy to_torch, trimmed to logical (un-padded) shape.
    logical_shape = list(host_shard_tensors[0].shape)
    trim = tuple(slice(0, d) for d in logical_shape)
    shards = [_to_torch_zero_copy(s)[trim] for s in host_shard_tensors]

    return _reassemble_2d(mesh_coords, shards, logical_shape, mesh_shape, concat_dims, permute, dtype, pool=pool)


# ---------------------------------------------------------------------------
# YUV 4:2:0 planar D2H — on-device color conversion + batched D2H + planar concat
# ---------------------------------------------------------------------------

# BT.601 coefficients for inputs in [-1, 1] -> uint8 [0, 255].
# Must match yuv_conversion.hpp::yuv_bt601_coefficients() in ttnn experimental.
_BT601_Y_COEFF = (32.74, 64.28, 12.48, 125.5)
_BT601_CB_COEFF = (-18.90, -37.10, 56.00, 128.0)
_BT601_CR_COEFF = (56.00, -46.89, -9.11, 128.0)


def _bt601_yuv_coefficients():
    """Default BT.601 YUV coefficients for ttnn.experimental.yuv_conversion."""
    return ttnn.experimental.YUVCoefficients(y=list(_BT601_Y_COEFF), cb=list(_BT601_CB_COEFF), cr=list(_BT601_CR_COEFF))


def _yuv_planar_d2h(
    tt_Y: ttnn.Tensor,
    tt_Cb: ttnn.Tensor,
    tt_Cr: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    H: int,
    W: int,
    T: int,
    *,
    view=None,
    pool: ThreadPoolExecutor | None = None,
) -> np.ndarray:
    """Batched D2H of three YUV ttnn tensors into ffmpeg yuv420p planar uint8.

    Per-shard input shapes (kernel-native BHWT with C=1):

      * tt_Y:        ``(1, h_per_y, w_per_y, T)``
      * tt_Cb/tt_Cr: ``(1, h_per_uv, w_per_uv, T)``

    Output: ``(T, H*W + 2*(H/2 * W/2))`` numpy uint8 — per-frame
    ``[Y plane | Cb plane | Cr plane]`` in row-major.

    Kicks off all three ``cpu(blocking=False)`` calls before a single
    ``synchronize_device`` so the reads overlap.  Per-shard scatters then
    take one of two paths:

      * **C++/AVX2 fast path** (when ``HAS_CPP_PLANAR_CONCAT`` is True and
        the local mesh is rectangular): one ``planar_concat_cpp`` call into
        a module-level persistent output buffer.  ~2× faster than the
        Python path on warm calls, but the returned buffer is **reused
        across calls** — copy out (or feed ffmpeg) before the next call.
      * **Python fallback**: per-shard ``_write`` tasks on the shared
        reassembly ThreadPoolExecutor (torch's strided-copy backend).
        Each scatter is a strided->strided copy; allocates a fresh output
        per call.

    Either way, the byte layout is identical.

    Args:
        view: Optional mesh device view for multi-host environments.
            When provided, uses ``host_buffer()`` / ``get_shard()`` with
            ``view.is_local()`` filtering instead of ``get_device_tensors()``.
    """
    Hu, Wu = H // 2, W // 2
    hw = H * W
    uv = Hu * Wu
    row_stride = hw + 2 * uv

    # Async D2H all 3 outputs, single sync — overlaps three D2H reads.
    host_Y = tt_Y.cpu(blocking=False)
    host_Cb = tt_Cb.cpu(blocking=False)
    host_Cr = tt_Cr.cpu(blocking=False)
    ttnn.synchronize_device(mesh_device)

    if view is not None:
        # --- Multi-host: extract local shards via host_buffer/get_shard ---
        def _extract_local(host_tensor):
            host_mesh_coords = list(host_tensor.tensor_topology().mesh_coords())
            distributed_buf = host_tensor.host_buffer()
            tt_dtype = host_tensor.dtype
            padded_shape = list(host_tensor.padded_shape)
            logical_shape = list(host_tensor.shape)
            trim = tuple(slice(0, d) for d in logical_shape)

            coords_and_shards = []
            for c in host_mesh_coords:
                if not view.is_local(c):
                    continue
                buf = distributed_buf.get_shard(c)
                if buf is not None:
                    coords_and_shards.append((c, _host_buffer_to_torch(buf, padded_shape, tt_dtype)[trim]))
            return coords_and_shards

        Y_coords_shards = _extract_local(host_Y)
        Cb_coords_shards = _extract_local(host_Cb)
        Cr_coords_shards = _extract_local(host_Cr)

        # Remap global mesh coordinates to 0-based local coordinates.
        all_local_coords = [c for c, _ in Y_coords_shards]
        local_row_positions = sorted({int(c[0]) for c in all_local_coords})
        local_col_positions = sorted({int(c[1]) for c in all_local_coords})
        row_remap = {pos: i for i, pos in enumerate(local_row_positions)}
        col_remap = {pos: i for i, pos in enumerate(local_col_positions)}
        TP_eff = len(local_row_positions)
        SP_eff = len(local_col_positions)

        h_per_y, w_per_y = H // TP_eff, W // SP_eff
        h_per_uv, w_per_uv = Hu // TP_eff, Wu // SP_eff

        mesh_coords = [(row_remap[int(c[0])], col_remap[int(c[1])]) for c in all_local_coords]
        Y_shards = [s for _, s in Y_coords_shards]
        Cb_shards = [s for _, s in Cb_coords_shards]
        Cr_shards = [s for _, s in Cr_coords_shards]
    else:
        # --- Single-host: extract all shards via get_device_tensors ---
        TP_eff, SP_eff = tuple(mesh_device.shape)
        h_per_y, w_per_y = H // TP_eff, W // SP_eff
        h_per_uv, w_per_uv = Hu // TP_eff, Wu // SP_eff

        mesh_coords = list(tt_Y.tensor_topology().mesh_coords())

        def _extract(host_tensor):
            host_shards = ttnn.get_device_tensors(host_tensor)
            logical_shape = list(host_shards[0].shape)
            trim = tuple(slice(0, d) for d in logical_shape)
            return [_to_torch_zero_copy(s)[trim] for s in host_shards]

        Y_shards = _extract(host_Y)  # each (1, h_per_y, w_per_y, T)
        Cb_shards = _extract(host_Cb)  # each (1, h_per_uv, w_per_uv, T)
        Cr_shards = _extract(host_Cr)

    # --- C++/AVX2 fast path ---------------------------------------------
    #
    # Drop-in replacement for the torch_threaded scatter below: same byte
    # layout, ~2× faster with the persistent output buffer.  The C++
    # binding assumes shards are passed in row-major (r, c) order, so we
    # sort by coord first.  The fast path requires a complete TP_eff ×
    # SP_eff rectangular submesh; if the local coords are sparse (could
    # happen on irregular multi-host topologies), we fall through to the
    # Python path which handles arbitrary coord sets.
    if HAS_CPP_PLANAR_CONCAT and len(mesh_coords) == TP_eff * SP_eff:
        triples = sorted(
            zip(mesh_coords, Y_shards, Cb_shards, Cr_shards),
            key=lambda t: (int(t[0][0]), int(t[0][1])),
        )
        out = _get_planar_out_buf(T, row_stride)
        return _planar_concat_cpp_impl(
            [t[1] for t in triples],
            [t[2] for t in triples],
            [t[3] for t in triples],
            "CHWT",
            (TP_eff, SP_eff),
            out=out,
        )

    # --- Python fallback (torch_threaded scatter) ------------------------
    # Allocate the planar output and view each plane region as a (T, h, w)
    # strided torch tensor (no copy, shares storage with `out`).
    out = np.empty((T, row_stride), dtype=np.uint8)
    out_t = torch.from_numpy(out)
    y_view = out_t.as_strided((T, H, W), (row_stride, W, 1), 0)
    u_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw)
    v_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw + uv)

    if pool is None:
        pool = _get_default_reassemble_pool()

    def _write(view, shard, r, c, h_per, w_per):
        # shard (1, h_per, w_per, T) -> squeeze(0).permute(2, 0, 1) -> strided (T, h_per, w_per).
        src = shard.squeeze(0).permute(2, 0, 1)
        view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per].copy_(src)

    futures = []
    for coord, shard in zip(mesh_coords, Y_shards):
        r, c = int(coord[0]), int(coord[1])
        futures.append(pool.submit(_write, y_view, shard, r, c, h_per_y, w_per_y))
    for coord, shard in zip(mesh_coords, Cb_shards):
        r, c = int(coord[0]), int(coord[1])
        futures.append(pool.submit(_write, u_view, shard, r, c, h_per_uv, w_per_uv))
    for coord, shard in zip(mesh_coords, Cr_shards):
        r, c = int(coord[0]), int(coord[1])
        futures.append(pool.submit(_write, v_view, shard, r, c, h_per_uv, w_per_uv))
    for f in futures:
        f.result()

    return out


def _trim_yuv420p_planar_height(planar: np.ndarray, full_H: int, full_W: int, new_H: int) -> np.ndarray:
    """Trim the H dimension of a flattened YUV 4:2:0 planar uint8 buffer.

    Input ``planar`` has shape ``(T, full_H*full_W + 2*(full_H/2)*(full_W/2))``
    uint8 and is laid out as ``[Y plane | Cb plane | Cr plane]`` per frame.
    Returns a new buffer of shape
    ``(T, new_H*full_W + 2*(new_H/2)*(full_W/2))`` keeping the top ``new_H``
    rows of Y and the top ``new_H/2`` rows of Cb / Cr.

    Used after ``fast_device_to_host_yuv`` when the VAE pads its output height
    (``new_logical_h < full_H``); the bottom rows of every plane contain
    garbage that ffmpeg would otherwise encode.

    No-op when ``new_H == full_H``.
    """
    if new_H == full_H:
        return planar
    if new_H > full_H:
        raise ValueError(f"new_H ({new_H}) must not exceed full_H ({full_H})")
    if new_H % 2 != 0 or full_W % 2 != 0:
        raise ValueError(f"YUV 4:2:0 trim requires even new_H and full_W (got new_H={new_H}, full_W={full_W})")

    full_Hu, full_Wu = full_H // 2, full_W // 2
    new_Hu = new_H // 2

    full_hw = full_H * full_W
    full_uv = full_Hu * full_Wu
    full_row_stride = full_hw + 2 * full_uv

    new_hw = new_H * full_W
    new_uv = new_Hu * full_Wu
    new_row_stride = new_hw + 2 * new_uv

    T = planar.shape[0]
    out = np.empty((T, new_row_stride), dtype=planar.dtype)

    # Strided 3D views into source / dest planes — no copy until the assignment.
    src_y = np.lib.stride_tricks.as_strided(planar, shape=(T, full_H, full_W), strides=(full_row_stride, full_W, 1))
    src_u = np.lib.stride_tricks.as_strided(
        planar[:, full_hw:], shape=(T, full_Hu, full_Wu), strides=(full_row_stride, full_Wu, 1)
    )
    src_v = np.lib.stride_tricks.as_strided(
        planar[:, full_hw + full_uv :],
        shape=(T, full_Hu, full_Wu),
        strides=(full_row_stride, full_Wu, 1),
    )

    dst_y = np.lib.stride_tricks.as_strided(
        out, shape=(T, new_H, full_W), strides=(new_row_stride, full_W, 1), writeable=True
    )
    dst_u = np.lib.stride_tricks.as_strided(
        out[:, new_hw:], shape=(T, new_Hu, full_Wu), strides=(new_row_stride, full_Wu, 1), writeable=True
    )
    dst_v = np.lib.stride_tricks.as_strided(
        out[:, new_hw + new_uv :],
        shape=(T, new_Hu, full_Wu),
        strides=(new_row_stride, full_Wu, 1),
        writeable=True,
    )

    # Inner W stride matches (1) on both sides, so numpy's strided iterator
    # collapses to a per-row memcpy of `full_W` (or `full_Wu`) bytes.
    dst_y[:] = src_y[:, :new_H, :]
    dst_u[:] = src_u[:, :new_Hu, :]
    dst_v[:] = src_v[:, :new_Hu, :]

    return out


def fast_device_to_host_yuv(
    tt_video_BCTHW: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    ccl_manager=None,
    root: int | None = None,
    coefficients=None,
    pool: ThreadPoolExecutor | None = None,
    debug: bool = False,
    logical_h: int | None = None,
) -> np.ndarray | None:
    """On-device YUV 4:2:0 conversion + batched D2H + planar uint8 concat.

    Takes a sharded BCTHW bf16 row-major tensor with values in ``[-1, 1]`` —
    typically the output of the Wan VAE — and returns a single numpy uint8
    array in ffmpeg ``AV_PIX_FMT_YUV420P`` layout.

    On a single-host system, reads all per-device shards concurrently with
    async DMA, converts each shard's YUV planes, and scatters into the
    planar output on host.

    On a multi-host (distributed) system, uses a hybrid approach: an
    on-device all_gather for the inter-host axis only, then re-shards with
    repeat + mesh_partition so each local device holds unique data, runs the
    YUV conversion on the gathered data, and performs fast async DMA + planar
    concat for local shards only.

    Pipeline:
      1. (Multi-host only) ``all_gather`` + ``repeat`` + ``mesh_partition``
         on the inter-host axis of the bf16 BCTHW input.
      2. Permute BCTHW -> BCHWT (T moves to the last position) and reshape to
         drop the B=1 dim, landing in CHWT — the layout the YUV kernel expects.
      3. ``ttnn.experimental.yuv_conversion`` runs on each device's local shard,
         producing 3 uint8 outputs (Y full-res, Cb/Cr 4:2:0 subsampled).
      4. Async ``cpu(blocking=False)`` on all three outputs followed by a
         single ``synchronize_device`` so the D2H reads overlap.
      5. Per-shard scatters are dispatched across the shared reassembly thread
         pool using torch's strided-copy backend.

    Output shape: ``(T, H*W + 2*(H/2 * W/2))`` numpy uint8 — one row per frame,
    ``[Y plane | Cb plane | Cr plane]`` in row-major.

    Args:
        tt_video_BCTHW: Sharded ttnn tensor with shape ``(1, 3, T, H, W)``,
            bfloat16, ROW_MAJOR_LAYOUT, sharded ``{axis 0: dim 3 (H), axis 1: dim 4 (W)}``.
            Values must lie in ``[-1, 1]`` — the YUV kernel's expected range.
        mesh_device: The mesh device.
        ccl_manager: Optional :class:`CCLManager` instance.  Required for
            multi-host environments where only local devices are accessible.
        root: If set, only the host with this MPI rank performs the D2H
            transfer and returns the assembled array; all other ranks return
            ``None``.  If ``None`` (default), all ranks perform D2H.
        coefficients: ``ttnn.experimental.YUVCoefficients`` to use for the
            per-channel weights and offsets.  Defaults to BT.601.
        pool: Optional ``ThreadPoolExecutor`` for the host-side reassembly.
            If ``None``, the module-level lazy default pool is used.
        debug: If ``True``, print diagnostic shape information.
        logical_h: Optional logical (un-padded) height of the output.  When
            the VAE pads ``H`` to a coarser size, pass the true logical height
            here and the function will trim the bottom rows of each plane in
            the planar buffer on host (Y -> top ``logical_h`` rows; Cb/Cr ->
            top ``logical_h/2`` rows).  Must be even and ``<= H``.  Defaults to
            ``None`` (no trim).

    Returns:
        ``np.ndarray`` of shape ``(T, H'*W + 2*(H'/2 * W/2))``, dtype uint8,
        where ``H' = logical_h if logical_h is not None else H``.
        Returns ``None`` for non-root ranks when ``root`` is set.

    Raises:
        AssertionError: if ``B != 1``, ``C != 3``, or H/W are not even.
        ValueError: if ``logical_h`` is set and is greater than ``H`` or odd.
    """
    if coefficients is None:
        coefficients = _bt601_yuv_coefficients()

    # NOTE: ttnn ``.shape`` on a multi-device sharded tensor returns the
    # per-shard (local) shape, not the global logical shape.  We derive the
    # global H, W from the mesh shape, assuming H is sharded on axis 0 and W
    # on axis 1 (the convention this function documents).  All on-device ops
    # (permute, reshape, yuv_conversion) operate on per-shard semantics, so
    # we use ``h_per, w_per`` for the reshape target; ``_yuv_planar_d2h``
    # then takes the global ``H, W`` to size the output buffer.
    mesh_shape = tuple(mesh_device.shape)
    B, C, T, h_per, w_per = tt_video_BCTHW.shape
    assert B == 1, f"fast_device_to_host_yuv requires B=1, got {B}"
    assert C == 3, f"fast_device_to_host_yuv requires C=3 (RGB), got {C}"

    TP, SP = mesh_shape
    H, W = h_per * TP, w_per * SP

    if debug:
        print(f"  [yuv-d2h] input per-shard: {list(tt_video_BCTHW.shape)}")
        print(f"  [yuv-d2h] global H={H}, W={W}, T={T}  (mesh TP={TP}, SP={SP})")

    # Sharding convention: axis 0 -> dim 3 (H), axis 1 -> dim 4 (W).
    concat_dims: list[int | None] = [3, 4]

    # --- Multi-host: hybrid on-device collective + fast local DMA -----------
    d2h_view = None
    if ttnn.using_distributed_env():
        if ccl_manager is None:
            msg = "fast_device_to_host_yuv requires ccl_manager in a distributed (multi-host) environment"
            raise ValueError(msg)

        d2h_view = mesh_device.get_view()
        rank = int(ttnn.distributed_context_get_rank())

        inter_host_axis = _get_inter_host_axis(mesh_device, d2h_view, mesh_shape)

        inter_dim = concat_dims[inter_host_axis]
        if inter_dim is not None and mesh_shape[inter_host_axis] > 1:
            # Move the gather dim out of the tile dims (last two) to position 2
            # (dim=-3). This avoids the composite_all_gather path's tile-padded
            # check and makes any concat fallback a cheap outer-dim memcpy.
            # BCTHW dims: B=0 C=1 T=2 H=3 W=4. inter_dim is 3 (H) or 4 (W).
            if inter_dim == 4:
                pre_dims = (0, 1, 4, 2, 3)  # BCTHW -> BCWTH
                post_dims = (0, 1, 3, 4, 2)  # BCWTH -> BCTHW
            else:  # inter_dim == 3
                pre_dims = (0, 1, 3, 2, 4)  # BCTHW -> BCHTW
                post_dims = (0, 1, 3, 2, 4)  # BCHTW -> BCTHW (same swap)
            ag_dim = 2

            tt_video_BCTHW = ttnn.permute(tt_video_BCTHW, pre_dims)
            tt_video_BCTHW = ttnn.to_layout(tt_video_BCTHW, ttnn.TILE_LAYOUT)
            tt_video_BCTHW = ccl_manager.all_gather(
                tt_video_BCTHW,
                dim=ag_dim,
                mesh_axis=inter_host_axis,
                use_hyperparams=True,
                use_persistent_buffer=True,
            )
            # Drop back to ROW_MAJOR before repeat/mesh_partition so ttnn.repeat
            # doesn't wrap itself in an Untilize → Repeat → Tilize roundtrip.
            tt_video_BCTHW = ttnn.to_layout(tt_video_BCTHW, ttnn.ROW_MAJOR_LAYOUT)
            n_hosts = int(ttnn.distributed_context_get_size())
            if n_hosts > 1:
                repeat_dims = [1] * len(tt_video_BCTHW.shape)
                repeat_dims[ag_dim] = n_hosts
                tt_video_BCTHW = ttnn.repeat(tt_video_BCTHW, repeat_dims)
                tt_video_BCTHW = ttnn.mesh_partition(tt_video_BCTHW, dim=ag_dim, cluster_axis=inter_host_axis)
            tt_video_BCTHW = ttnn.permute(tt_video_BCTHW, post_dims)

        # Recompute per-shard dims after gather — they may have grown.
        B, C, T, h_per, w_per = tt_video_BCTHW.shape

        if debug:
            print(f"  [yuv-d2h] (distributed) post-gather per-shard: {list(tt_video_BCTHW.shape)}")

        if root is not None and rank != root:
            return None

    assert (
        h_per % 2 == 0 and w_per % 2 == 0
    ), f"per-shard H and W must be even for 4:2:0 (got h_per={h_per}, w_per={w_per})"

    # 1. Reorder BCTHW -> CHWT for the YUV kernel.  Shapes here are per-shard.
    tt_BCHWT = ttnn.permute(tt_video_BCTHW, (0, 1, 3, 4, 2))
    if debug:
        print(f"  [yuv-d2h] after permute(0,1,3,4,2) per-shard: {list(tt_BCHWT.shape)}")

    tt_CHWT = ttnn.reshape(tt_BCHWT, (C, h_per, w_per, T))
    if debug:
        print(f"  [yuv-d2h] after reshape to (C,h_per,w_per,T) per-shard: {list(tt_CHWT.shape)}")

    # 2. On-device YUV 4:2:0 -> 3 uint8 tensors.
    tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_CHWT, coefficients)
    if debug:
        print(f"  [yuv-d2h] yuv outputs per-shard:")
        print(f"  [yuv-d2h]   Y : {list(tt_Y.shape)}")
        print(f"  [yuv-d2h]   Cb: {list(tt_Cb.shape)}")
        print(f"  [yuv-d2h]   Cr: {list(tt_Cr.shape)}")

    # 3+4. Batched D2H + planar concat — uses GLOBAL H, W to size the buffer.
    out = _yuv_planar_d2h(tt_Y, tt_Cb, tt_Cr, mesh_device, H, W, T, view=d2h_view, pool=pool)

    # 5. Optional host-side trim of the height in the planar buffer for the
    # case where the VAE pads H beyond its logical extent.
    if logical_h is not None and logical_h != H:
        if debug:
            print(f"  [yuv-d2h] trimming H {H} -> logical_h {logical_h}")
        out = _trim_yuv420p_planar_height(out, H, W, logical_h)

    return out


def upsample(
    x: ttnn.Tensor,
    /,
    *,
    scale_factor: int,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Wrapper around ttnn.upsample that allows for padded tensors."""
    change_layout = x.layout == ttnn.TILE_LAYOUT and x.shape != x.padded_shape

    if change_layout:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    x = ttnn.upsample(x, scale_factor=scale_factor, memory_config=memory_config)

    if change_layout:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    return x


def unflatten(x: ttnn.Tensor, dim: int, sizes: Sequence[int]) -> ttnn.Tensor:
    """Expands a dimension of the input tensor over multiple dimensions.

    Args:
        x: ttnn.Tensor
            The input tensor to unflatten
        dim: int
            Dimension to be unflattened, specified as an index into input.shape.
        sizes: Sequence[int]
            New shape of the unflattened dimension. One of its elements can be -1 in which case the corresponding output dimension is inferred. Otherwise, the product of sizes must equal
    Returns:
        ttnn.Tensor
            The unflattened tensor.
    """
    assert (
        x.shape[dim] % abs(math.prod(sizes)) == 0
    ), f"The total number of elements in the new shape {sizes} must be equal or a factor of the number of elements (when using inferred dimensions) in the original shape {x.shape[dim]}"
    new_shape = list(x.shape)
    if dim == -1:
        new_shape[-1:] = sizes
    else:
        new_shape[dim : dim + 1] = sizes
    return ttnn.reshape(x, new_shape)
