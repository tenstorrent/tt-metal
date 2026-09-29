# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Python wrapper around the C++ planar concat extension.

The C++ source lives in ``models/tt_dit/utils/cpp/``. On first import this module builds it
with ``cpp/build.sh`` into a per-user cache directory keyed by the source hash and the Python
ABI (``$TT_DIT_PLANAR_CONCAT_CACHE`` or ``~/.cache/tt_dit/planar_concat/<key>/``), a two-second
g++ compile, then loads it via ``importlib.util``. A hand-built ``cpp/build/_planar_concat*.so``
still wins when present. Nobody ran ``build.sh`` in practice, so the YUV export path fell back
to the torch scatter everywhere (measured on the 4x8 galaxy: the same clip's mp4 export 1.6 s
with the fallback vs 0.7 s with the extension).

``TT_DIT_PLANAR_CONCAT_BUILD=0`` disables the build; without AVX2, a compiler, or a writable
cache dir the build is skipped. In every failure case :data:`HAS_CPP_PLANAR_CONCAT` is False
and :func:`planar_concat_cpp` is None, exactly as before: the caller falls back to torch.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    pass


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "cpp")
_BUILD_DIR = os.path.join(_SRC_DIR, "build")  # a hand-run build.sh lands here and takes precedence
_SOURCES = (
    "build.sh",
    "planar_concat.cpp",
    "planar_concat.hpp",
    "planar_concat_bindings.cpp",
    "transpose_avx2.cpp",
    "transpose_avx2.hpp",
)


def _find_so(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    candidates = sorted(f for f in os.listdir(directory) if f.startswith("_planar_concat") and f.endswith(".so"))
    return os.path.join(directory, candidates[0]) if candidates else None


def _cache_dir() -> str:
    """Per-user build cache, keyed so a source or interpreter change rebuilds and a checkout move does not."""
    import hashlib
    import sysconfig

    h = hashlib.sha256()
    for name in _SOURCES:
        with open(os.path.join(_SRC_DIR, name), "rb") as f:
            h.update(name.encode())
            h.update(f.read())
    h.update(str(sysconfig.get_config_var("EXT_SUFFIX")).encode())
    root = os.environ.get("TT_DIT_PLANAR_CONCAT_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "tt_dit", "planar_concat"
    )
    return os.path.join(root, h.hexdigest()[:16])


def _host_can_build() -> bool:
    if os.environ.get("TT_DIT_PLANAR_CONCAT_BUILD", "1") in ("0", "false", "False"):
        return False
    try:
        with open("/proc/cpuinfo") as f:
            if " avx2" not in f.read():
                return False  # build.sh refuses too: the streaming stores would fault at runtime
    except OSError:
        return False
    import shutil

    return shutil.which(os.environ.get("CXX", "g++")) is not None


def _build_into_cache() -> str | None:
    """Run cpp/build.sh into the cache dir once; concurrent importers wait on a lock. None on any failure."""
    import subprocess
    import sys

    target = _cache_dir()
    so = _find_so(target)
    if so:
        return so
    if not _host_can_build():
        return None
    try:
        os.makedirs(target, exist_ok=True)
        lock_path = os.path.join(target, ".build.lock")
        with open(lock_path, "w") as lock:
            try:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            so = _find_so(target)  # another process may have built while we waited
            if so:
                return so
            env = dict(os.environ, BUILD_DIR=target, PYTHON=sys.executable)
            r = subprocess.run(
                ["bash", os.path.join(_SRC_DIR, "build.sh")], env=env, capture_output=True, text=True, timeout=300
            )
        if r.returncode != 0:
            _log(f"planar_concat: C++ extension build failed (using the torch fallback): {r.stderr.strip()[-400:]}")
            return None
        return _find_so(target)
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"planar_concat: C++ extension build skipped (using the torch fallback): {e}")
        return None


def _log(msg: str) -> None:
    try:
        from loguru import logger

        logger.warning(msg)
    except ImportError:
        import sys

        print(msg, file=sys.stderr)


def _load_so(so_path: str):
    spec = importlib.util.spec_from_file_location("_planar_concat", so_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _try_load_extension():
    so = _find_so(_BUILD_DIR) or _build_into_cache()
    return _load_so(so) if so else None


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
        out_H: int | None = None,
        out_W: int | None = None,
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
        # Logical (cropped) output: when the VAE pads a global tail, write only the valid frame.
        oH = H if out_H is None else out_H
        oW = W if out_W is None else out_W
        oHu, oWu = oH // 2, oW // 2
        row_stride = oH * oW + 2 * oHu * oWu

        if out is None:
            out = np.empty((T, row_stride), dtype=np.uint8)
        elif out.shape != (T, row_stride) or out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise ValueError(
                f"out must be C-contiguous uint8 with shape ({T}, {row_stride}); "
                f"got shape {out.shape} dtype {out.dtype} c_contig={out.flags.c_contiguous}"
            )

        _ext.planar_concat(y_np, u_np, v_np, dim_order, mesh_shape, out, oH, oW)
        return out

    def set_thread_pool_size(n_threads: int) -> None:
        """Set the C++ thread pool size.  Must be called BEFORE first scatter."""
        _ext.set_thread_pool_size(int(n_threads))

else:
    planar_concat_cpp = None  # type: ignore[assignment]

    def set_thread_pool_size(n_threads: int) -> None:  # noqa: D401
        """No-op fallback when the C++ extension isn't built."""
