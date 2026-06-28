# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device YUV 4:2:0 conversion + fast device-to-host into ffmpeg yuv420p planar bytes.

Isolated from ``tensor.py`` so the LTX wiring stays decoupled from that module's
d2h internals; reuses the shard-extraction primitives it already exposes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import ttnn

from .tensor import _to_torch_zero_copy, _host_buffer_to_torch, _get_inter_host_axis
from .planar_concat import HAS_CPP_PLANAR_CONCAT
from .planar_concat import planar_concat_cpp as _planar_concat_cpp_impl

# Persistent host-side reassembly pool — strided uint8 copies release the GIL,
# so a few threads scale near-linearly; persistence avoids per-call startup.
_DEFAULT_REASSEMBLE_POOL: ThreadPoolExecutor | None = None
_DEFAULT_REASSEMBLE_WORKERS = min(8, os.cpu_count() or 8)


def _get_default_reassemble_pool() -> ThreadPoolExecutor:
    global _DEFAULT_REASSEMBLE_POOL
    if _DEFAULT_REASSEMBLE_POOL is None:
        _DEFAULT_REASSEMBLE_POOL = ThreadPoolExecutor(
            max_workers=_DEFAULT_REASSEMBLE_WORKERS,
            thread_name_prefix="tt_dit_yuv_reassemble",
        )
    return _DEFAULT_REASSEMBLE_POOL


# Persistent output buffer for the C++ planar-concat fast path — fresh np.empty
# pays first-touch page-fault overhead per call that dwarfs the kernel; reuse
# eliminates it. The returned buffer is reused across calls (copy out / feed
# ffmpeg before the next call). Shape changes reallocate lazily.
_PLANAR_OUT_BUF: np.ndarray | None = None
_PLANAR_OUT_SHAPE: tuple[int, int] | None = None


def _get_planar_out_buf(T: int, row_stride: int) -> np.ndarray:
    global _PLANAR_OUT_BUF, _PLANAR_OUT_SHAPE
    shape = (T, row_stride)
    if _PLANAR_OUT_SHAPE != shape:
        _PLANAR_OUT_BUF = np.empty(shape, dtype=np.uint8)
        _PLANAR_OUT_SHAPE = shape
    return _PLANAR_OUT_BUF


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


def _trim_yuv420p_planar(planar: np.ndarray, full_H: int, full_W: int, new_H: int, new_W: int) -> np.ndarray:
    """Trim H and/or W of a flattened YUV 4:2:0 planar uint8 buffer.

    Input ``(T, full_H*full_W + 2*(full_H/2)*(full_W/2))`` laid out ``[Y | Cb | Cr]`` per frame;
    returns the top-left ``new_H x new_W`` crop in the same packed layout. Used when the VAE pads
    its output (LTX pads W 1920->2048 on 4x8); the padded rows/cols hold garbage ffmpeg would encode.
    """
    if new_H == full_H and new_W == full_W:
        return planar
    if new_H > full_H or new_W > full_W:
        raise ValueError(f"new dims ({new_H}x{new_W}) must not exceed full ({full_H}x{full_W})")
    if new_H % 2 or new_W % 2 or full_H % 2 or full_W % 2:
        raise ValueError(f"YUV 4:2:0 trim requires even dims (got new {new_H}x{new_W}, full {full_H}x{full_W})")

    full_Hu, full_Wu = full_H // 2, full_W // 2
    new_Hu, new_Wu = new_H // 2, new_W // 2
    full_hw, full_uv = full_H * full_W, full_Hu * full_Wu
    new_hw, new_uv = new_H * new_W, new_Hu * new_Wu
    full_row = full_hw + 2 * full_uv
    new_row = new_hw + 2 * new_uv

    T = planar.shape[0]
    out = np.empty((T, new_row), dtype=planar.dtype)
    ast = np.lib.stride_tricks.as_strided

    src_y = ast(planar, (T, full_H, full_W), (full_row, full_W, 1))
    src_u = ast(planar[:, full_hw:], (T, full_Hu, full_Wu), (full_row, full_Wu, 1))
    src_v = ast(planar[:, full_hw + full_uv :], (T, full_Hu, full_Wu), (full_row, full_Wu, 1))

    dst_y = ast(out, (T, new_H, new_W), (new_row, new_W, 1), writeable=True)
    dst_u = ast(out[:, new_hw:], (T, new_Hu, new_Wu), (new_row, new_Wu, 1), writeable=True)
    dst_v = ast(out[:, new_hw + new_uv :], (T, new_Hu, new_Wu), (new_row, new_Wu, 1), writeable=True)

    dst_y[:] = src_y[:, :new_H, :new_W]
    dst_u[:] = src_u[:, :new_Hu, :new_Wu]
    dst_v[:] = src_v[:, :new_Hu, :new_Wu]

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
    logical_w: int | None = None,
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

    # 5. Optional host-side trim of H and/or W in the planar buffer for the case where the
    # VAE pads its output beyond the logical extent (LTX pads W 1920->2048 on 4x8).
    new_H = logical_h if logical_h is not None else H
    new_W = logical_w if logical_w is not None else W
    if new_H != H or new_W != W:
        if debug:
            print(f"  [yuv-d2h] trimming {H}x{W} -> {new_H}x{new_W}")
        out = _trim_yuv420p_planar(out, H, W, new_H, new_W)

    return out

