# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Mesh topology helpers for Dots OCR.

Selects the right ``ttnn.MeshShape`` based on the ``MESH_DEVICE`` env var,
mirroring the convention used throughout the tt-metal repo / ``tt_transformers``:

- ``N150``         → ``(1, 1)`` — single Wormhole chip
- ``N300``         → ``(1, 2)`` — two-chip Wormhole card
- ``T3K``          → ``(1, 8)`` — Wormhole LLMBox (8 chips, 1×8 ring)
- ``T3K_2x4``      → ``(2, 4)`` — Wormhole LLMBox (8 chips, 2×4)
- ``N150x4``       → ``(1, 4)`` — 4-chip LLMBox-lite (only valid where fabric supports 1×4; see
                     note in ``test_rmsnorm_1d.py`` about ring topology restrictions)
- ``TG``           → ``(8, 4)`` — Galaxy (32 chips)

Unset / unknown values default to ``(1, 1)`` (single chip) so existing N150
runs continue to work unchanged.

Terminology note: earlier iterations of this demo used the abbreviation "WH LB"
("Wormhole Low Batch") as shorthand for single-chip Wormhole. That clashes with
the wider tt-metal convention where **"WH LB" = Wormhole LLMBox** (multi-chip
hardware, typically T3K-class). The helpers here replace that shorthand.

Environment variables consumed here:

- ``MESH_DEVICE``             — topology selector (see table above). Same knob as
                                ``tt_transformers``, ``qwen25_vl``, Llama demos, etc.
- ``DOTS_T3K_OPEN_FULL_MESH`` — T3K / ``T3K_1X8`` / ``T3K_2X4``: open the **physical**
                                multi-device mesh (8 WH devices on T3K), then ``create_submesh``
                                to the logical 1×1 or 1×2 shape dots.mocr can use (default ``1``).
                                Set to ``0`` to open only the small logical mesh (legacy).
- ``DOTS_MAX_SEQ_LEN``        — canonical cap on prefill / KV length (optional).
- ``DOTS_MAX_SEQ_LEN_WH_LB``  — legacy alias; still honored for back-compat.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from loguru import logger

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn

# Canonical name -> (rows, cols) mesh shape.
# Keep in sync with the aliases in _MESH_ALIASES below.
_MESH_SHAPES: dict[str, Tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150X4": (1, 4),
    "T3K": (1, 8),
    "T3K_1X8": (1, 8),
    "T3K_2X4": (2, 4),
    "TG": (8, 4),
}

# --- Model-level TP limits ---------------------------------------------------
# dots.mocr GQA config: ``num_attention_heads=12``, ``num_key_value_heads=2``.
# Tensor parallelism shards along ``cluster_shape[1]`` and tt_transformers asserts
# ``n_kv_heads % cluster_shape[1] == 0`` — so TP degree must divide 2. The only
# supported TP factors are therefore 1 (N150) and 2 (N300).
#
# On hardware with more chips (T3K 1×8, Galaxy, …), we can still run dots.mocr
# by opening a 1×2 submesh and leaving the extra chips idle. That's strictly
# better than failing at startup, but users should know they're not getting the
# full TP speedup the hardware could theoretically provide with a different
# model.
DOTS_MAX_TP_COLS = 2
DOTS_MAX_DP_ROWS = 1  # no data-parallel support yet
_DOTS_TP_NOTE = (
    "dots.mocr has num_key_value_heads=2, so TP degree must divide 2. "
    "Supported TP factors are 1 (N150) and 2 (N300)."
)

# Accept common casings / synonyms users hit in CI and tt_transformers docs.
_MESH_ALIASES: dict[str, str] = {
    "n150": "N150",
    "n300": "N300",
    "t3k": "T3K",
    "t3000": "T3K",
    "t3k_1x8": "T3K_1X8",
    "t3k-1x8": "T3K_1X8",
    "t3k_2x4": "T3K_2X4",
    "t3k-2x4": "T3K_2X4",
    "n150x4": "N150X4",
    "n150-4": "N150X4",
    "tg": "TG",
    "galaxy": "TG",
}


def _canonicalize(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = name.strip()
    if not key:
        return None
    upper = key.upper()
    if upper in _MESH_SHAPES:
        return upper
    return _MESH_ALIASES.get(key.lower())


def resolve_mesh_shape(name: Optional[str] = None) -> Tuple[int, int]:
    """
    Resolve a topology name to a ``(rows, cols)`` tuple.

    If ``name`` is ``None``, falls back to the ``MESH_DEVICE`` env var.
    If unset or unrecognized, returns ``(1, 1)`` (single chip) and logs a warning.
    """
    selector = name if name is not None else os.environ.get("MESH_DEVICE")
    canonical = _canonicalize(selector)
    if canonical is None:
        if selector:
            logger.warning(
                f"dots_ocr.mesh: unknown MESH_DEVICE={selector!r}; falling back to single chip (1,1). "
                f"Known values: {sorted(_MESH_SHAPES)}"
            )
        return (1, 1)
    return _MESH_SHAPES[canonical]


def resolve_physical_mesh_shape(name: Optional[str] = None) -> Tuple[int, int]:
    """
    Physical device grid for ``MESH_DEVICE`` (e.g. **T3K → (1, 8)** for the 8-chip Wormhole LLMBox).

    Same values as :func:`resolve_mesh_shape`; provided as a clearer name when contrasting with
    :func:`resolve_supported_mesh_shape` (logical / model submesh).
    """
    return resolve_mesh_shape(name)


def resolve_supported_mesh_shape(name: Optional[str] = None) -> Tuple[int, int]:
    """
    Like ``resolve_mesh_shape`` but clamps the result to what dots.mocr can
    actually run with (see ``DOTS_MAX_TP_COLS`` / ``DOTS_MAX_DP_ROWS``).

    If the requested topology exceeds dots.mocr's TP ceiling, this opens a
    1x2 submesh on the same hardware and logs a warning. That means
    ``MESH_DEVICE=T3K`` will "just work" on T3K hardware — it'll use 2 of the
    8 chips with tensor parallelism degree 2.
    """
    rows, cols = resolve_mesh_shape(name)

    # For dots.mocr, TP>2 is not possible (n_kv_heads=2). On multi-chip systems like T3K,
    # TP=2 (1x2 submesh) is supported but can be undesirable for bringup/debug (long compiles,
    # fabric/dispatch complexity). Default T3K to TP=1 unless explicitly requested.
    selector = name if name is not None else os.environ.get("MESH_DEVICE")
    canonical = _canonicalize(selector) if selector is not None else None
    if canonical in ("T3K", "T3K_1X8", "T3K_2X4"):
        req = os.environ.get("DOTS_T3K_TP", "").strip()
        if req and req not in ("1", "2"):
            logger.warning(f"dots_ocr.mesh: invalid DOTS_T3K_TP={req!r}; expected '1' or '2'. Using default.")
            req = ""
        if req == "1" or (req == "" and os.environ.get("DOTS_T3K_DEFAULT_TP1", "1") != "0"):
            cols = 1

    clamped_cols = min(cols, DOTS_MAX_TP_COLS)
    clamped_rows = min(rows, DOTS_MAX_DP_ROWS)

    if (clamped_rows, clamped_cols) != (rows, cols):
        selector = name if name is not None else os.environ.get("MESH_DEVICE")
        # T3K is a very common target — suppress the log for it to keep test output clean.
        # The clamping is expected and correct behavior for dots.mocr.
        if selector and str(selector).upper() not in ("T3K", "T3K_1X8", "T3K_2X4"):
            logger.info(
                f"dots_ocr.mesh: MESH_DEVICE={selector!r} -> requested mesh ({rows},{cols}) clamped to "
                f"({clamped_rows},{clamped_cols}). (dots.mocr only supports TP<=2)"
            )
    return (clamped_rows, clamped_cols)


def default_mesh_shape():
    """
    Build the ``ttnn.MeshShape`` for the current ``MESH_DEVICE``, clamped to
    what dots.mocr supports (``DOTS_MAX_TP_COLS`` / ``DOTS_MAX_DP_ROWS``).

    Use this instead of constructing ``ttnn.MeshShape(1, 1)`` directly so T3K /
    N300 / TG runs pick up the right topology without source edits.
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    rows, cols = resolve_supported_mesh_shape()
    return ttnn.MeshShape(rows, cols)


def open_mesh_device(mesh_shape=None):
    """
    Open a ``ttnn`` mesh device for dots.mocr.

    **T3K (8-device Wormhole LLMBox):** by default (``DOTS_T3K_OPEN_FULL_MESH=1``), opens the
    **physical** mesh (e.g. ``1×8``), then ``create_submesh`` to the **logical** shape
    (``1×1`` or ``1×2`` from :func:`resolve_supported_mesh_shape`) so the full system is
    initialized while the model still runs at TP≤2. Use :func:`close_dots_mesh_device` to
    close both the submesh and parent mesh.

    If ``mesh_shape`` is passed explicitly, behavior is unchanged: that shape is opened directly.

    Legacy path (``DOTS_T3K_OPEN_FULL_MESH=0``): open only the logical small mesh (no parent).
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    if mesh_shape is not None:
        device = ttnn.open_mesh_device(mesh_shape)
    else:
        selector = os.environ.get("MESH_DEVICE")
        canonical = _canonicalize(selector)
        logical = resolve_supported_mesh_shape(selector)
        physical = resolve_physical_mesh_shape(selector)
        open_full = os.environ.get("DOTS_T3K_OPEN_FULL_MESH", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        t3k_family = canonical in ("T3K", "T3K_1X8", "T3K_2X4")
        device = None
        if open_full and t3k_family and physical != logical and hasattr(ttnn, "MeshShape"):
            try:
                full = ttnn.open_mesh_device(ttnn.MeshShape(physical[0], physical[1]))
                sub = full.create_submesh(ttnn.MeshShape(logical[0], logical[1]))
                setattr(sub, "_dots_parent_mesh_device", full)
                device = sub
                logger.info(
                    f"dots_ocr.mesh: T3K-class system — opened physical mesh {physical} "
                    f"({full.get_num_devices()} devices), running dots.mocr on submesh {logical} "
                    f"({sub.get_num_devices()} devices). Set DOTS_T3K_OPEN_FULL_MESH=0 to open {logical} only."
                )
            except Exception as exc:
                logger.warning(
                    f"dots_ocr.mesh: full mesh {physical} + submesh {logical} failed ({exc!r}); "
                    f"falling back to logical mesh only."
                )
                device = None
        if device is None:
            device = ttnn.open_mesh_device(ttnn.MeshShape(logical[0], logical[1]))
            try:
                if physical != logical and t3k_family:
                    logger.info(
                        f"dots_ocr.mesh: opened logical mesh {logical} only "
                        f"(physical system would be {physical}; enable DOTS_T3K_OPEN_FULL_MESH=1 for 8-device init)."
                    )
            except Exception:
                pass
    try:
        logger.info(f"dots_ocr.mesh: mesh device shape={tuple(device.shape)} num_devices={device.get_num_devices()}")
    except Exception:
        pass
    return device


def close_dots_mesh_device(device) -> None:
    """
    Close a mesh opened via :func:`open_mesh_device`.

    If the device was created as a **submesh** of a full T3K (etc.) mesh, closes the submesh
    first, then the parent. Otherwise calls ``ttnn.close_mesh_device`` once.
    """
    ttnn = get_ttnn()
    if ttnn is None:
        return
    parent = getattr(device, "_dots_parent_mesh_device", None)
    ttnn.close_mesh_device(device)
    if parent is not None:
        ttnn.close_mesh_device(parent)


def assert_supported_topology(mesh_device) -> None:
    """
    Sanity-check that the opened mesh shape is one dots.mocr can actually run
    on (TP <= ``DOTS_MAX_TP_COLS``, DP == 1 for now).

    Raises ``AssertionError`` with a clear message otherwise. Callers that open
    a mesh via :func:`open_mesh_device` already get a logical TP≤2 mesh; this is
    for code paths that open a mesh externally (e.g. tt_transformers fixtures).
    """
    shape = tuple(mesh_device.shape)
    rows, cols = shape if len(shape) == 2 else (1, shape[0])
    assert rows <= DOTS_MAX_DP_ROWS and cols <= DOTS_MAX_TP_COLS, (
        f"dots_ocr: mesh shape {shape} exceeds dots.mocr TP/DP ceiling "
        f"(rows<={DOTS_MAX_DP_ROWS}, cols<={DOTS_MAX_TP_COLS}). {_DOTS_TP_NOTE}"
    )


def get_max_seq_len_cap() -> Optional[int]:
    """
    Canonical cap knob for prefill / KV length.

    Reads ``DOTS_MAX_SEQ_LEN`` first (preferred) and falls back to the legacy
    ``DOTS_MAX_SEQ_LEN_WH_LB`` name if the new one isn't set. Returns ``None``
    when neither is set.
    """
    val = os.environ.get("DOTS_MAX_SEQ_LEN")
    if val is None:
        val = os.environ.get("DOTS_MAX_SEQ_LEN_WH_LB")
        if val is not None:
            logger.warning(
                "dots_ocr: DOTS_MAX_SEQ_LEN_WH_LB is deprecated; prefer DOTS_MAX_SEQ_LEN "
                "(WH LB in the rest of tt-metal refers to Wormhole LLMBox, not single-chip)."
            )
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning(f"dots_ocr: invalid DOTS_MAX_SEQ_LEN value {val!r}; ignoring.")
        return None
