# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import EllipsisType


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


def bfloat16_bcthw_to_yuv420_planes(t: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """On-device RGB→YUV420p for a (B, 3, T, H, W) bf16 tensor in [-1, 1].

    Returns (Y, U, V) uint8 tensors with shapes:
        Y: (B, 1, T, H, W)
        U: (B, 1, T, H/2, W/2)
        V: (B, 1, T, H/2, W/2)

    Uses BT.601 coefficients matching the host-side reference
    :func:`models.tt_dit.utils.video.rgb_to_yuv420p`: the math is applied
    after mapping [-1, 1] to [0, 255] so the coefficients are identical to the
    uint8 host path. 2×2 chroma subsampling is done per-plane with a
    reshape-to-7D + two single-axis ``ttnn.mean`` reductions.

    H, W and the per-shard dimensions must all be even; the 2×2 windows never
    cross shard boundaries at any currently used mesh/resolution pair.

    Alternative downsample paths were evaluated and discarded:
      * ``ttnn.avg_pool2d`` on flattened RGB: 23 ms per invocation — the
        Untilize → reshape → pool → reshape → Tilize round-trip around a pool
        on sharded data eats the savings the pool itself provides.
      * Multi-axis ``ttnn.mean(dim=[4, 6])``: decomposes into
        reduce→permute→reduce→permute; the permutes alone cost ~50 ms.
      * Slice+add on the "2" axis: slices with nonzero begin on sharded
        tensors cost 2.5 ms each, and ttnn inserts extra Tilize/Untilize
        around every op — 23 ms total.
    """
    B, C, T, H, W = t.shape
    assert C == 3, f"Expected C=3 (RGB), got shape {tuple(t.shape)}"
    assert H % 2 == 0 and W % 2 == 0, f"YUV420p requires even H/W, got {H}x{W}"

    # Scale [-1, 1] → [0, 255] in bf16. Stay in TILE for the elementwise path.
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    t = ttnn.add(t, 1.0)
    t = ttnn.multiply(t, 127.5)

    # Split R, G, B along C (non-strided slices on a non-trailing dim).
    r = ttnn.slice(t, [0, 0, 0, 0, 0], [B, 1, T, H, W])
    g = ttnn.slice(t, [0, 1, 0, 0, 0], [B, 2, T, H, W])
    b = ttnn.slice(t, [0, 2, 0, 0, 0], [B, 3, T, H, W])

    def _combine(cr: float, cg: float, cb: float, offset: float) -> ttnn.Tensor:
        out = ttnn.multiply(r, cr)
        out = ttnn.add(out, ttnn.multiply(g, cg))
        out = ttnn.add(out, ttnn.multiply(b, cb))
        out = ttnn.add(out, offset)
        return ttnn.clamp(out, min=0.0, max=255.0)

    # BT.601 limited-range coefficients from [0, 255] RGB.
    y = _combine(0.257, 0.504, 0.098, 16.0)
    u = _combine(-0.148, -0.291, 0.439, 128.0)
    v = _combine(0.439, -0.368, -0.071, 128.0)

    def _downsample_2x2(x: ttnn.Tensor) -> ttnn.Tensor:
        # 7D reshape needs ROW_MAJOR (TILE reshape fails on the tiny "2" dims).
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [B, 1, T, H // 2, 2, W // 2, 2])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.mean(x, dim=6, keepdim=False)  # average within each column pair
        x = ttnn.mean(x, dim=4, keepdim=False)  # average within each row pair
        return x  # (B, 1, T, H/2, W/2)

    u = _downsample_2x2(u)
    v = _downsample_2x2(v)

    # Final layout + dtype: ROW_MAJOR uint8 for fast D2H zero-copy.
    y = ttnn.typecast(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint8)
    u = ttnn.typecast(ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint8)
    v = ttnn.typecast(ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint8)
    return y, u, v


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
) -> torch.Tensor:
    """Reassemble per-device shards into a single tensor for a 2D mesh.

    For the common case where both axes are concatenated, writes each shard
    directly into a pre-allocated output buffer.  When *permute* and/or *dtype*
    are given the permutation and type conversion are fused into the scatter
    write, halving total memory traffic.
    """
    d0, d1 = concat_dims

    if d0 is not None and d1 is not None:
        s0, s1 = shard_shape[d0], shard_shape[d1]
        full_shape = list(shard_shape)
        full_shape[d0] *= mesh_shape[0]
        full_shape[d1] *= mesh_shape[1]
        ndim = len(full_shape)

        if permute is not None:
            out_shape = [full_shape[p] for p in permute]
            out_dtype = dtype if dtype is not None else shards[0].dtype
            perm_list = list(permute)
            d0_out = perm_list.index(d0)
            d1_out = perm_list.index(d1)

            out = torch.empty(out_shape, dtype=out_dtype)
            for coord, shard in zip(mesh_coords, shards):
                r, c = int(coord[0]), int(coord[1])
                slices = [slice(None)] * ndim
                slices[d0_out] = slice(r * s0, (r + 1) * s0)
                slices[d1_out] = slice(c * s1, (c + 1) * s1)
                out[tuple(slices)] = shard.permute(*permute).contiguous()
            return out

        out_dtype = dtype if dtype is not None else shards[0].dtype
        out = torch.empty(full_shape, dtype=out_dtype)
        for coord, shard in zip(mesh_coords, shards):
            r, c = int(coord[0]), int(coord[1])
            slices = [slice(None)] * ndim
            slices[d0] = slice(r * s0, (r + 1) * s0)
            slices[d1] = slice(c * s1, (c + 1) * s1)
            out[tuple(slices)] = shard
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

        return _reassemble_2d(local_coords, shards, logical_shape, local_mesh_shape, concat_dims, permute, dtype)

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

    return _reassemble_2d(mesh_coords, shards, logical_shape, mesh_shape, concat_dims, permute, dtype)


def fast_device_to_host_yuv420(
    tt_tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    tp_axis: int = 0,
    sp_axis: int = 1,
) -> torch.Tensor:
    """On-device RGB→YUV420p conversion plus fast D2H.

    Input: (B, 3, T, H, W) bf16 in [-1, 1], sharded H on mesh ``tp_axis`` (dim
    3 in BCTHW) and W on mesh ``sp_axis`` (dim 4).

    Output: (B, T, H*W*3//2) uint8 planar YUV420p, per-frame bytes laid out as
    ``[Y plane][U plane][V plane]`` — the exact layout ffmpeg expects with
    ``-pix_fmt yuv420p``. Each batch element can be fed frame-by-frame as
    ``out[b, t]`` after reshaping to (H*3//2, W), or as a flat write.

    Single-host only.
    """
    if ttnn.using_distributed_env():
        raise NotImplementedError("fast_device_to_host_yuv420 is single-host only")

    # Note: on a mesh-sharded tensor, ``tt_tensor.shape`` reports the
    # **per-shard** logical dims (not the full logical shape). We only need it
    # here to sanity-check the channel count.
    assert tt_tensor.shape[1] == 3, f"Expected C=3 (RGB), got shape {tuple(tt_tensor.shape)}"

    y_dev, u_dev, v_dev = bfloat16_bcthw_to_yuv420_planes(tt_tensor)

    # Y is (B, 1, T, H, W); U, V are (B, 1, T, H/2, W/2). In all three the
    # spatial dims occupy positions 3 and 4, so concat_dims is identical.
    concat_dims: list[int | None] = [None, None]
    concat_dims[tp_axis] = 3
    concat_dims[sp_axis] = 4

    y_host = fast_device_to_host(y_dev, mesh_device, concat_dims)
    u_host = fast_device_to_host(u_dev, mesh_device, concat_dims)
    v_host = fast_device_to_host(v_dev, mesh_device, concat_dims)

    # Read true full dims from the reassembled host tensor.
    B, _, T, H_full, W_full = y_host.shape
    y_flat = y_host.squeeze(1).reshape(B, T, H_full * W_full)
    u_flat = u_host.squeeze(1).reshape(B, T, (H_full * W_full) // 4)
    v_flat = v_host.squeeze(1).reshape(B, T, (H_full * W_full) // 4)
    return torch.cat([y_flat, u_flat, v_flat], dim=2)


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
