# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Python wrapper around the C++ planar concat extension.

The C++ source lives in ``models/tt_dit/utils/cpp/`` and produces
``cpp/build/_planar_concat<abi>.so`` when built via ``cpp/build.sh``.  This
module loads it via ``importlib.util`` so callers don't need to put the
build directory on ``sys.path``.

If the ``.so`` isn't built (or fails to import), :data:`HAS_CPP_PLANAR_CONCAT`
is False and :func:`planar_concat_cpp` is None.  The benchmark harness in
``test_fast_device_to_host.py`` already has a ``HAS_*`` skip pattern that
this slots into.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    pass


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "cpp", "build")


def _try_load_extension():
    if not os.path.isdir(_BUILD_DIR):
        return None
    candidates = [f for f in os.listdir(_BUILD_DIR) if f.startswith("_planar_concat") and f.endswith(".so")]
    if not candidates:
        return None
    so_path = os.path.join(_BUILD_DIR, candidates[0])
    spec = importlib.util.spec_from_file_location("_planar_concat", so_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


_ext = _try_load_extension()
HAS_CPP_PLANAR_CONCAT: bool = _ext is not None


def _to_numpy(shard) -> np.ndarray:
    """Convert a torch tensor to a contiguous numpy uint8 array (zero-copy when possible)."""
    if isinstance(shard, np.ndarray):
        if not shard.flags.c_contiguous:
            shard = np.ascontiguousarray(shard)
        return shard
    # torch.Tensor or anything else with .numpy()
    if hasattr(shard, "contiguous") and hasattr(shard, "numpy"):
        t = shard.contiguous()
        return t.numpy()
    raise TypeError(f"unsupported shard type: {type(shard)!r}")


if HAS_CPP_PLANAR_CONCAT:

    def planar_concat_cpp(
        y_shards: Sequence,
        u_shards: Sequence,
        v_shards: Sequence,
        dim_order: str,
        mesh_shape: tuple[int, int] = (4, 8),
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized YUV 4:2:0 planar concat — C++/AVX2 implementation.

        Equivalent in output to the ``planar_concat_torch_threaded``
        reference in ``test_fast_device_to_host.py`` but runs in C++ with a
        persistent ``std::thread`` pool and AVX2 byte-tile transposes for
        the CHWT path.

        Args:
            y_shards, u_shards, v_shards: lists of ``TP*SP`` per-shard
                tensors/arrays.  Per-shard shape:
                  * CHWT: ``(1, h_per, w_per, T)`` uint8 contiguous.
                  * CTHW: ``(1, T, h_per, w_per)`` uint8 contiguous.
                UV shards must have ``h_per/2`` and ``w_per/2`` of the Y
                shards' dims (4:2:0 subsampling).
            dim_order: ``"CHWT"`` or ``"CTHW"``.
            mesh_shape: ``(TP, SP)``.  Defaults to ``(4, 8)``.
            out: Optional pre-allocated output buffer of shape
                ``(T, H*W + 2*(H/2 * W/2))`` uint8.  When ``None``, a fresh
                ``np.empty`` is allocated each call (matches the convention
                of the other variants in the test harness).  Reusing a
                buffer across calls eliminates ~50 ms of first-touch page
                faults per call on systems without THP=always.

        Returns:
            ``np.ndarray`` of shape ``(T, H*W + 2*(H/2 * W/2))``, dtype
            ``uint8``.  Per-frame layout: ``[Y plane | Cb plane | Cr plane]``.
        """
        y_np = [_to_numpy(s) for s in y_shards]
        u_np = [_to_numpy(s) for s in u_shards]
        v_np = [_to_numpy(s) for s in v_shards]

        TP, SP = int(mesh_shape[0]), int(mesh_shape[1])
        if dim_order == "CHWT":
            _, h_per, w_per, T = y_np[0].shape
        elif dim_order == "CTHW":
            _, T, h_per, w_per = y_np[0].shape
        else:
            raise ValueError(f"dim_order must be 'CHWT' or 'CTHW', got {dim_order!r}")
        H, W = h_per * TP, w_per * SP
        Hu, Wu = H // 2, W // 2
        row_stride = H * W + 2 * Hu * Wu

        if out is None:
            out = np.empty((T, row_stride), dtype=np.uint8)
        elif out.shape != (T, row_stride) or out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise ValueError(
                f"out must be C-contiguous uint8 with shape ({T}, {row_stride}); "
                f"got shape {out.shape} dtype {out.dtype} c_contig={out.flags.c_contiguous}"
            )

        _ext.planar_concat(y_np, u_np, v_np, dim_order, mesh_shape, out)
        return out

    def set_thread_pool_size(n_threads: int) -> None:
        """Set the C++ thread pool size.  Must be called BEFORE first scatter."""
        _ext.set_thread_pool_size(int(n_threads))

else:
    planar_concat_cpp = None  # type: ignore[assignment]

    def set_thread_pool_size(n_threads: int) -> None:  # noqa: D401
        """No-op fallback when the C++ extension isn't built."""
