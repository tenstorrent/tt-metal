# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper of libx280s_host.so, the bit-exact host reference of the x280 sampling library.

    from x280s_ref import X280S
    lib = X280S()                      # loads libx280s_host.so next to this file (or $X280S_LIB)
    tok, st = lib.sample(row, temperature=0.7, top_k=50, top_p=0.9, seed=1, user=0, step=0)

`row` is a 1-D numpy array: float32 (DTYPE_F32) or uint16 holding bfloat16 bits (DTYPE_BF16).
A 2-D array with a column count > vocab is accepted row by row via `sample_rows`.
"""

import ctypes
import os

import numpy as np

DTYPE_F32 = 0
DTYPE_BF16 = 1
K_MAX = 1024
ABI_VERSION = 1


class Params(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("top_p", ctypes.c_float),
        ("_pad", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
    ]


class Work(ctypes.Structure):
    _fields_ = [("cand", ctypes.c_uint32 * (2 * K_MAX)), ("weight", ctypes.c_float * K_MAX)]


class Stats(ctypes.Structure):
    _fields_ = [
        ("nan_count", ctypes.c_uint32),
        ("cap_applied", ctypes.c_uint32),
        ("greedy", ctypes.c_uint32),
        ("k_eff", ctypes.c_uint32),
        ("n_kept", ctypes.c_uint32),
        ("pick", ctypes.c_uint32),
        ("fallback", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("x_max", ctypes.c_float),
        ("S", ctypes.c_float),
        ("threshold", ctypes.c_float),
        ("kept_sum", ctypes.c_float),
        ("u", ctypes.c_float),
        ("target", ctypes.c_float),
        ("r", ctypes.c_uint64),
    ]

    FLOAT_FIELDS = ("x_max", "S", "threshold", "kept_sum", "u", "target")

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_ if not name.startswith("_")}

    def float_bits(self):
        """The fp32 intermediates as uint32 bit patterns (for bit-exact comparisons)."""
        return {k: int(np.float32(getattr(self, k)).view(np.uint32)) for k in self.FLOAT_FIELDS}


def _default_lib_path():
    return os.environ.get("X280S_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libx280s_host.so"))


class X280S:
    def __init__(self, path=None):
        self.lib = ctypes.CDLL(path or _default_lib_path())
        L = self.lib
        vp, u32, u64, i32 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32
        L.x280s_sample_row.argtypes = [
            vp,
            u32,
            u32,
            u32,
            ctypes.POINTER(Params),
            u32,
            u64,
            ctypes.POINTER(Work),
            ctypes.POINTER(Stats),
        ]
        L.x280s_sample_row.restype = i32
        L.x280s_argmax_row.argtypes = [vp, u32, u32, u32, ctypes.POINTER(Stats)]
        L.x280s_argmax_row.restype = i32
        L.x280s_expf.argtypes = [ctypes.c_float]
        L.x280s_expf.restype = ctypes.c_float
        L.x280s_splitmix64.argtypes = [u64]
        L.x280s_splitmix64.restype = u64
        for n in (
            "x280s_abi_version",
            "x280s_sizeof_work",
            "x280s_sizeof_stats",
            "x280s_sizeof_params",
            "x280s_has_rvv",
        ):
            getattr(L, n).argtypes = []
            getattr(L, n).restype = u32
        assert L.x280s_abi_version() == ABI_VERSION, "ABI version mismatch"
        assert L.x280s_sizeof_work() == ctypes.sizeof(Work)
        assert L.x280s_sizeof_stats() == ctypes.sizeof(Stats)
        assert L.x280s_sizeof_params() == ctypes.sizeof(Params)
        self.work = Work()

    @staticmethod
    def _prep(row):
        row = np.asarray(row)
        if row.dtype == np.float32:
            dtype = DTYPE_F32
        elif row.dtype == np.uint16:
            dtype = DTYPE_BF16
        else:
            raise TypeError("row must be float32 or uint16 (bfloat16 bits), got %s" % row.dtype)
        if row.ndim != 1:
            raise ValueError("row must be 1-D")
        if row.strides[0] % row.itemsize:
            row = np.ascontiguousarray(row)
        stride = row.strides[0] // row.itemsize
        if stride <= 0:
            row = np.ascontiguousarray(row)
            stride = 1
        return row, dtype, stride

    def sample(self, row, temperature, top_k, top_p, seed, user=0, step=0, vocab=None):
        row, dtype, stride = self._prep(row)
        V = row.shape[0] if vocab is None else vocab
        p = Params(temperature, top_k, top_p, 0, seed & 0xFFFFFFFFFFFFFFFF)
        st = Stats()
        tok = self.lib.x280s_sample_row(
            row.ctypes.data,
            dtype,
            V,
            stride,
            ctypes.byref(p),
            user,
            step & 0xFFFFFFFFFFFFFFFF,
            ctypes.byref(self.work),
            ctypes.byref(st),
        )
        return tok, st

    def argmax(self, row, vocab=None):
        row, dtype, stride = self._prep(row)
        V = row.shape[0] if vocab is None else vocab
        st = Stats()
        tok = self.lib.x280s_argmax_row(row.ctypes.data, dtype, V, stride, ctypes.byref(st))
        return tok, st

    def expf(self, x):
        return np.float32(self.lib.x280s_expf(float(np.float32(x))))

    def splitmix64(self, x):
        return self.lib.x280s_splitmix64(x & 0xFFFFFFFFFFFFFFFF)


def bf16_bits(x):
    """float32 array -> bfloat16 bits (uint16), round to nearest even (NaN kept quiet)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    rounded[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return rounded


def bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
