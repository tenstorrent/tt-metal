"""Model-local full-context paged SDPA with FP32 accumulator recurrence.

Run ``python -m models.autoports.ifm_k2_horizon_7b.tt.accurate_attention.build``
once during setup. Kernel compilation occurs through the normal TTNN JIT.
"""

import hashlib
import importlib.util
import json
import sysconfig
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load():
    build = Path(__file__).resolve().parent / ".build"
    library = build / ("_k2_accurate_attention" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not library.exists():
        raise RuntimeError("Build the model-local accurate_attention binding before decoder setup")
    provenance = json.loads((build / "provenance.json").read_text())
    root = Path(__file__).resolve().parents[5]
    for relative, expected in provenance["files"].items():
        with (root / relative).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Accurate attention build is stale ({relative}); rerun accurate_attention.build")
    with library.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != provenance["library_sha256"]:
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
    chunk_start_idx_tensors=None,
    q_chunk_size=128,
    k_chunk_size=128,
    fp32_output_accumulator=True,
    packed_gqa=False,
):
    """Same paged causal tensor contract as chunked_scaled_dot_product_attention.

    Scalar offsets retain the original reader/compute/writer contract. A list
    of B=2/4/8/16/32 scalar owners binds each request's offset to its independent
    reader cores while preserving the existing FP32 kernel arithmetic.
    """
    offsets = [] if chunk_start_idx_tensors is None else list(chunk_start_idx_tensors)
    if packed_gqa and len(offsets) < 2:
        raise ValueError("Packed GQA requires at least two per-request raw position owners")
    if chunk_start_idx_tensors is not None:
        if chunk_start_idx is not None or chunk_start_idx_tensor is not None:
            raise ValueError("Provide either one offset or a list of per-request offsets")
        if not offsets:
            raise ValueError("Per-request offsets cannot be empty")
        if len(offsets) not in (1, 2, 4, 8, 16, 32) or q_chunk_size != 32 or k_chunk_size != 128:
            raise ValueError("Per-request offsets require B=1/2/4/8/16/32 and Q32/K128")
        if not fp32_output_accumulator:
            raise ValueError("Per-request offsets preserve FP32 output recurrence")
        chunk_start_idx_tensor = offsets[0]
        if len(offsets) == 1:
            offsets = []  # Keep the original B1 descriptor and cache contract.
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
        offsets,
        packed_gqa,
        str(Path(kernel).with_name("writer_packed_gqa.cpp")) if packed_gqa else "",
    )
