# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Optional, locally compiled CPU backend; never loaded or built on import."""
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SOURCE = Path(__file__).with_name("csrc") / "quantize.cpp"


def library_path():
    cache = Path(os.environ.get("TT_BFP_QUANT_CACHE", Path.home() / ".cache/tt-bfp-quant"))
    key = hashlib.sha256(SOURCE.read_bytes() + (sys.platform + platform.machine()).encode()).hexdigest()[:20]
    return cache / key / ("quantize.dylib" if sys.platform == "darwin" else "quantize.so")


def build_native(openmp="auto"):
    """Compile once. Linux tries OpenMP; macOS defaults to serial C++.

    -ffp-contract=off preserves the experiment's separate FP32 multiply/subtract.
    Set CXX to select a compiler; TT_BFP_QUANT_CACHE chooses the output directory.
    """
    if os.name != "posix":
        raise RuntimeError("The native backend supports Linux/macOS; use backend='numpy' elsewhere")
    if openmp not in ("auto", "on", "off"):
        raise ValueError("openmp must be auto, on or off")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("Install a C++17 compiler (g++ or clang++), or use backend='numpy'")
    target = library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    attempts = [True, False] if openmp == "auto" and sys.platform != "darwin" else [openmp == "on"]
    failures = []
    for enabled in attempts:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            output = Path(tmp) / target.name
            command = compiler + ["-O3", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off"]
            if enabled:
                command += ["-fopenmp"]
            command += [str(SOURCE), "-o", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                failures.append(result.stderr)
                continue
            output.replace(target)
            record = {
                "library": str(target),
                "openmp": enabled,
                "command": command,
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            }
            target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            load_library.cache_clear()
            return record
    raise RuntimeError("Native backend build failed:\n" + "\n".join(failures))


@functools.lru_cache(maxsize=1)
def load_library():
    lib = ctypes.CDLL(str(library_path()))
    lib.search_bfp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.search_bfp.restype = None
    lib.gptq_block.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.gptq_block.restype = None
    return lib


def resolve_backend(backend):
    if backend not in ("auto", "numpy", "native"):
        raise ValueError("backend must be auto, numpy or native")
    available = library_path().is_file()
    if backend == "native" and not available:
        raise RuntimeError("Run 'tt-bfp-quant build' first, or use backend='numpy'")
    return "native" if available and backend != "numpy" else "numpy"


def search_native(x, bits, deltas, threads):
    out = np.empty_like(x)
    candidates = np.asarray(deltas, dtype=np.int32)
    counts = np.zeros(len(deltas), dtype=np.int64)
    load_library().search_bfp(
        x.ctypes.data,
        out.ctypes.data,
        x.size // 16,
        bits,
        candidates.ctypes.data,
        len(deltas),
        counts.ctypes.data,
        threads,
    )
    return out, {str(d): int(n) for d, n in zip(deltas, counts)}


def gptq_native_block(block, upper, deltas, threads):
    import torch

    block, upper = block.contiguous(), upper.contiguous()
    quantized, errors = torch.empty_like(block), torch.empty_like(block)
    candidates = np.asarray(deltas, dtype=np.int32)
    counts = np.zeros(len(deltas), dtype=np.int64)
    load_library().gptq_block(
        block.data_ptr(),
        upper.data_ptr(),
        quantized.data_ptr(),
        errors.data_ptr(),
        block.shape[0],
        block.shape[1],
        candidates.ctypes.data,
        len(deltas),
        counts.ctypes.data,
        threads,
    )
    return quantized, errors, {str(d): int(n) for d, n in zip(deltas, counts)}
