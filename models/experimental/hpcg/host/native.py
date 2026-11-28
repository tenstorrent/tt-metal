# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Loads host/native_setup.cpp, compiling it with the system C++ compiler on first use, and allocates the large
host tables.

The shared library is cached under $HPCG_NATIVE_CACHE (default ~/.cache/tt-metal-hpcg), keyed by a hash of the
source, so it is built once per source change. lib() returns None when no compiler is available; callers then
use their numpy versions.
"""

import ctypes
import hashlib
import mmap
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

SRC = pathlib.Path(__file__).with_name("native_setup.cpp")
_LIB = {}


def _array(dtype):
    """ctypes argument type of a C-contiguous numpy array of dtype: the checks of np.ctypeslib.ndpointer(dtype,
    flags="C_CONTIGUOUS") at 1.45 instead of 2.47 us per argument (ndpointer builds a ctypes view of each array);
    a setup passes about 260 arrays."""
    dt = np.dtype(dtype)

    class ArrayPointer(ctypes.c_void_p):
        @classmethod
        def from_param(cls, a):
            if not isinstance(a, np.ndarray) or a.dtype != dt or not a.flags.c_contiguous:
                raise TypeError(f"expected a C-contiguous {dt} numpy array, got {type(a).__name__}")
            return ctypes.c_void_p(a.__array_interface__["data"][0])

    return ArrayPointer


i32p, i64p, f32p, f64p = (_array(t) for t in (np.int32, np.int64, np.float32, np.float64))
I32, I64 = ctypes.c_int32, ctypes.c_int64


def _build():
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        return None
    tag = hashlib.sha1(SRC.read_bytes()).hexdigest()[:16]
    cache = pathlib.Path(os.environ.get("HPCG_NATIVE_CACHE", pathlib.Path.home() / ".cache" / "tt-metal-hpcg"))
    so = cache / f"native_setup_{tag}.so"
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            out = pathlib.Path(tmp) / so.name
            r = subprocess.run(
                [cxx, "-O3", "-fopenmp", "-shared", "-fPIC", "-std=c++17", str(SRC), "-o", str(out)],
                capture_output=True,
                text=True,
            )
            if r.returncode:
                raise RuntimeError(f"building {SRC.name} failed:\n{r.stderr}")
            os.replace(out, so)  # atomic: a concurrent build never sees a half-written file
    lib = ctypes.CDLL(str(so))
    lib.hpcg_set_threads.argtypes = [I32]
    lib.hpcg_ownership.restype = None
    lib.hpcg_ownership.argtypes = [I32, i64p, i64p, i64p, i64p, I32, I64, i64p]
    if "OMP_NUM_THREADS" not in os.environ:
        # a quarter of the logical CPUs: the planners' parallel regions are short, so starting threads costs more
        # than they save past a few, and one preempted thread (by the generation thread, or any other) delays the
        # whole region by a scheduler time slice. First setup on the P150's host (24 logical CPUs), median of 6:
        # 104^3 9.65 ms with 12 threads, 9.1 with 8, 8.9 with 6, 9.25 with 4; 192^3 and 288^3 the same for 4 to 12
        # (the card's matrix generation is the longer path there); all 24 were 0.9 ms slower at 192^3
        lib.hpcg_set_threads(max(1, (os.cpu_count() or 4) // 4))
    lib.hpcg_build_level.restype = I64
    lib.hpcg_build_level.argtypes = [I32, I32, I32, i32p, i32p, i32p, i64p, I64, I64, i64p, i64p, f32p, i32p, f32p]
    lib.hpcg_link.restype = I64
    geo = [I32, I32, I32, i32p, i32p, i32p, i64p]  # nx, nbands, nl, line_band, band_y0, band_ny, band_start
    lib.hpcg_link.argtypes = geo + [I64] + geo + [I64, i32p, f32p, i32p, f32p]
    lib.hpcg_plan_meta.restype = I64
    # raw pointers (SparseRowPlan.meta caches them): called for every launch, where ndpointer checks of 17 arrays
    # cost more than the call
    vp = ctypes.c_void_p
    lib.hpcg_plan_meta.argtypes = [I32, I32, I32, vp, I64] + [vp] * 11 + [I64, I64] + [vp] * 4
    lib.hpcg_plan_meta_into.restype = I64
    lib.hpcg_plan_meta_into.argtypes = [I32, I32, I32, vp, I64] + [vp] * 11 + [I64, I64, I64] + [vp] * 4
    lib.hpcg_plan_windows_transfer.restype = I64
    lib.hpcg_plan_windows_transfer.argtypes = [I32] + geo + geo + [I32, I32, i64p, I32, I32, I64, I64, I64] + [i64p] * 4
    lib.hpcg_transfer_tables.restype = I64
    lib.hpcg_transfer_tables.argtypes = [I32] + geo + geo + [I32, I32, i64p, I64] + [i64p] * 4 + [i32p, f32p]
    lib.hpcg_transfer_pad.restype = None
    lib.hpcg_transfer_pad.argtypes = [I32] + geo + geo + [I32, i64p]
    lib.hpcg_plan_windows_stencil.restype = I64
    lib.hpcg_plan_windows_stencil.argtypes = geo + [I32, I32, i64p, I32, I32, I64, I64, I64] + [i64p] * 5
    lib.hpcg_tile_permute.restype = None
    lib.hpcg_tile_permute.argtypes = [f32p, f32p, I64]
    lib.hpcg_plan_windows.restype = I64
    lib.hpcg_plan_windows.argtypes = [
        I32,
        I64,
        i32p,
        I32,
        I32,
        i64p,
        I32,
        I32,
        I32,
        I64,
        I64,
        I64,
        i64p,
        i64p,
        i64p,
        i64p,
    ]
    lib.hpcg_plan_offsets.restype = I64
    lib.hpcg_plan_offsets.argtypes = [
        I32,
        I64,
        i32p,
        I32,
        I32,
        i64p,
        I32,
        I32,
        I64,
        I64,
        i64p,
        i64p,
        i64p,
        i64p,
        I64,
        i32p,
        i64p,
    ]
    lib.hpcg_plan_exchange.restype = I64
    lib.hpcg_plan_exchange.argtypes = [I32, I32, i64p, I64, i64p, i64p, i64p, i64p, ctypes.c_void_p]
    lib.hpcg_plan_exchange_grouped.restype = I64
    lib.hpcg_plan_exchange_grouped.argtypes = [I32, I32, i64p, I64, i64p, i64p, i64p, i64p, i64p, I64] + [i64p] * 4
    return lib


def lib():
    """The compiled helper, or None if it cannot be built (HPCG_NATIVE=0 also disables it)."""
    if "lib" not in _LIB:
        _LIB["lib"] = _build() if os.environ.get("HPCG_NATIVE", "1") != "0" else None
    return _LIB["lib"]


_MADV_HUGEPAGE = 14
_HUGE_MIN = 2 << 20  # one huge page


def zeros(shape, dtype):
    """A zero-filled array like np.zeros, in fresh anonymous memory with transparent huge pages requested
    (madvise MADV_HUGEPAGE). The first write to a new page costs a page fault; with 2 MB pages instead of 4 KB ones,
    filling a new 765 MB table took 0.05 s instead of 0.25 s on the P150's host (np.zeros' calloc memory gets 4 KB
    pages). Small arrays, and hosts where madvise fails, use np.zeros."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes < _HUGE_MIN:
        return np.zeros(shape, dtype)
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)  # the kernel zero-fills new pages
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.c_char.from_buffer(m)
        libc.madvise(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(nbytes), _MADV_HUGEPAGE)
        del buf  # release the buffer export; the array below keeps the mapping alive
    except (OSError, AttributeError):
        pass  # no madvise: plain 4 KB pages, still zero-filled
    return np.frombuffer(m, dtype).reshape(shape)
