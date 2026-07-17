# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Env-gated capture of per-(chunk, layer) MoE input activations from a real
chunked-prefill run, so a single MoE layer can be replayed in isolation.

Design
------
The chunked-prefill driver (``run_chunked_transformer_no_pcc``) loops over
chunks and iterations; inside each block's ``_moe_path`` the tensor
``moe_input`` (= squeezed post-attention-norm output) is what feeds
``TtMoe.forward``. This module lets that call site dump ``moe_input`` to a
per-(chunk, layer) safetensors file, without changing any behaviour when
capture is disabled (the guard is a single bool read).

The isolated replay test (``test_moe_layer_isolated.py``) rebuilds one layer's
``TtMoe`` with the real cached weights and re-shards the captured activation
with the inverse mesh mapper, so the MoE forward runs bit-identically to the
full chunked test.

Usage (recording side)
-----------------------
Set the env vars, then run the normal chunked no-PCC test with ``iters1``::

    TT_DS_CAPTURE_MOE_DIR=/path/to/capture     # enables capture; output dir
    TT_DS_CAPTURE_MOE_CHUNKS=0,3,5             # optional chunk allowlist (default: all)
    TT_DS_CAPTURE_MOE_LAYERS=1,27,50           # optional layer allowlist (default: all)

The driver calls ``enable_from_env()`` once and ``set_chunk(c)`` per chunk on
iteration 0; the block's ``_moe_path`` calls ``record(...)``.

Storage note: each activation is (CHUNK=5120, emb=7168) bf16 ≈ 73 MB, so ALWAYS
use the allowlists unless you really want every (chunk, layer) pair (~48 GB).
"""

import os
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Module-level capture state (set by the recording driver; read by the hook).
# ---------------------------------------------------------------------------
_enabled: bool = False
_out_dir: Path | None = None
_chunk_allowlist: set[int] | None = None  # None => all chunks
_layer_allowlist: set[int] | None = None  # None => all layers
_current_chunk: int | None = None
_written: set[tuple[int, int]] = set()  # (chunk, layer) already dumped this session


def _parse_int_set(env_val: str | None) -> set[int] | None:
    if not env_val:
        return None
    return {int(tok) for tok in env_val.replace(" ", "").split(",") if tok != ""}


def enable_from_env() -> bool:
    """Enable capture iff TT_DS_CAPTURE_MOE_DIR is set. Returns whether enabled."""
    global _enabled, _out_dir, _chunk_allowlist, _layer_allowlist, _written
    out = os.environ.get("TT_DS_CAPTURE_MOE_DIR")
    if not out:
        _enabled = False
        return False
    _out_dir = Path(out)
    _out_dir.mkdir(parents=True, exist_ok=True)
    _chunk_allowlist = _parse_int_set(os.environ.get("TT_DS_CAPTURE_MOE_CHUNKS"))
    _layer_allowlist = _parse_int_set(os.environ.get("TT_DS_CAPTURE_MOE_LAYERS"))
    _written = set()
    _enabled = True
    logger.info(
        f"[moe_input_capture] ENABLED dir={_out_dir} "
        f"chunks={sorted(_chunk_allowlist) if _chunk_allowlist else 'ALL'} "
        f"layers={sorted(_layer_allowlist) if _layer_allowlist else 'ALL'}"
    )
    return True


def disable() -> None:
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def set_chunk(chunk: int) -> None:
    """Record which chunk the driver is about to run (called per chunk on iter 0)."""
    global _current_chunk
    _current_chunk = chunk


def _wanted(chunk: int, layer: int) -> bool:
    if _chunk_allowlist is not None and chunk not in _chunk_allowlist:
        return False
    if _layer_allowlist is not None and layer not in _layer_allowlist:
        return False
    return (chunk, layer) not in _written


def capture_path(out_dir: Path | str, chunk: int, layer: int) -> Path:
    """Canonical per-(chunk, layer) file path. Shared with the replay loader."""
    return Path(out_dir) / f"moe_input_chunk_{chunk}_layer_{layer}.safetensors"


def record(mesh_device, layer_idx: int, moe_input, actual_isl) -> None:
    """Gather ``moe_input`` to a full logical torch tensor and save it.

    ``moe_input`` is (0, -1)-sharded (dim 0 across mesh rows = SP/seq, dim -1
    across mesh cols = TP/hidden). We compose it back with the inverse of that
    sharding so the replay side can re-shard with ShardTensor2dMesh(dims=(0, -1)).
    Only called when is_enabled() is True.
    """
    if _current_chunk is None:
        logger.warning("[moe_input_capture] record() called before set_chunk(); skipping")
        return
    chunk = _current_chunk
    if not _wanted(chunk, layer_idx):
        return

    import ttnn

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape))
    x_host = ttnn.to_torch(moe_input, mesh_composer=composer)

    path = capture_path(_out_dir, chunk, layer_idx)
    from safetensors.torch import save_file

    # Persist as-is; store dtype/layout/actual_isl as string metadata so the
    # replay side can reconstruct the device tensor identically.
    meta = {
        "chunk": str(chunk),
        "layer": str(layer_idx),
        "orig_dtype": str(moe_input.get_dtype()),
        "orig_layout": str(moe_input.get_layout()),
        "actual_isl": "" if actual_isl is None else str(actual_isl),
        "shard_dims": "0,-1",
        "logical_shape": ",".join(str(s) for s in x_host.shape),
    }
    save_file({"x": x_host.contiguous()}, str(path), metadata=meta)
    _written.add((chunk, layer_idx))
    logger.info(
        f"[moe_input_capture] wrote chunk={chunk} layer={layer_idx} "
        f"shape={tuple(x_host.shape)} dtype={moe_input.get_dtype()} -> {path.name}"
    )
