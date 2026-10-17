# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-local chunked prefill SDPA with FP32 accumulator recurrence.

Run ``python -m models.demos.k2_horizon_7b_qb2.tt.accurate_attention.build``
once during setup. Kernel compilation occurs through the normal TTNN JIT.
"""

import hashlib
import importlib.util
import json
import sysconfig
from functools import lru_cache
from pathlib import Path


def _sha256_handle(handle):
    """hashlib.file_digest equivalent that also runs on Python 3.10 serving images."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load():
    build = Path(__file__).resolve().parent / ".build"
    library = build / ("_k2_accurate_attention" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not library.exists():
        # Serving images have no separate model-setup step; build once on first use.
        # Host compilation only (clang++-20 + this checkout's build/build.ninja).
        from .build import build as _build

        _build()
    provenance = json.loads((build / "provenance.json").read_text())
    root = Path(__file__).resolve().parents[5]
    for relative, expected in provenance["files"].items():
        with (root / relative).open("rb") as handle:
            actual = _sha256_handle(handle)
        if actual != expected:
            raise RuntimeError(f"Accurate attention build is stale ({relative}); rerun accurate_attention.build")
    with library.open("rb") as handle:
        if _sha256_handle(handle) != provenance["library_sha256"]:
            raise RuntimeError("Accurate attention binding does not match its build provenance; rebuild it")
    spec = importlib.util.spec_from_file_location("_k2_accurate_attention", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = (build / "kernel_path.txt").read_text().strip()
    if kernel != provenance["kernel_path"]:
        raise RuntimeError("Accurate attention kernel path does not match its build provenance; rebuild it")
    return module, kernel


def accurate_attention(
    q,
    k,
    v,
    page_table,
    *,
    chunk_start_idx=None,
    chunk_start_idx_tensor=None,
    q_chunk_size=128,
    k_chunk_size=128,
    fp32_output_accumulator=True,
    grid=(8, 8),
    math_fidelity="HiFi4",
):
    """Same paged causal tensor contract as chunked_scaled_dot_product_attention, with FP32 recurrence."""
    module, kernel = _load()
    return module.attention(
        q,
        k,
        v,
        page_table,
        chunk_start_idx_tensor,
        chunk_start_idx,
        q_chunk_size,
        k_chunk_size,
        kernel,
        str(Path(kernel).parent),
        fp32_output_accumulator,
        grid[0],
        grid[1],
        {"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}[math_fidelity],
    )
