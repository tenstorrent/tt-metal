# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optional tt-blaze acceleration for GLM-4.7-Flash decode clusters.

Import-guarded end to end: tt-blaze is only importable inside its own tt-metal checkout (it
needs the -ftt-nttp / -ftt-constinit / -ftt-consteval / -ftt-no-dyninit SFPI flags), so on our
tree `blaze_available()` is False and every entry point returns None. Callers fall back to the
ttnn path, and nothing about the shipping model changes.

Enable with GLM4_MOE_LITE_BLAZE_QKV_A=1 in a tree where blaze imports.

WHAT THIS BUYS, AND WHAT IT COSTS. blaze's DRAMStreamingMatmul runs on the 8 DRAM-bank workers
with a 1x32 decode tile; ttnn spreads over 64 cores and pads to 32 rows. Measured standalone at
GLM's shapes, the fused q_a+kv_a pair is 9.5 us against 45.1 us for the single concatenated ttnn
matmul the model actually runs -- 4.76x. But blaze wants the activation replicated per bank
worker and returns its outputs sharded on those 8 cores, so integrating it adds a reshard on each
side that the ttnn path does not pay. Those conversions are the whole question: at ~36 us of
headroom per layer they may or may not survive. That is why this exists as a measurable option
rather than a rewrite.
"""

from __future__ import annotations

import os
from typing import Any

import ttnn
from loguru import logger

_BLAZE: Any = None
_BLAZE_IMPORT_TRIED = False
_PROGRAM_CACHE: dict[int, Any] = {}


def _try_import_blaze() -> Any:
    global _BLAZE, _BLAZE_IMPORT_TRIED
    if _BLAZE_IMPORT_TRIED:
        return _BLAZE
    _BLAZE_IMPORT_TRIED = True
    try:
        from blaze.fused_program import FusedProgram
        from blaze.ops.glm_qkv_a_projection import GLMQKVAProjection
        from blaze.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment

        _BLAZE = {
            "FusedProgram": FusedProgram,
            "GLMQKVAProjection": GLMQKVAProjection,
            "bank_workers": get_pinned_optimal_dram_bank_to_logical_worker_assignment,
        }
        logger.info("tt-blaze available; GLM blaze ops can be enabled")
    except Exception as exc:  # pragma: no cover - the common case on our tree
        _BLAZE = None
        logger.debug("tt-blaze not available ({}); ttnn paths will be used", type(exc).__name__)
    return _BLAZE


def blaze_available() -> bool:
    return _try_import_blaze() is not None


def blaze_qkv_a_enabled() -> bool:
    """True when the caller should try the blaze q_a/kv_a path."""
    return os.environ.get("GLM4_MOE_LITE_BLAZE_QKV_A", "").strip() == "1" and blaze_available()


def _bank_worker_grid(device) -> tuple[ttnn.CoreRangeSet, int]:
    """The 8 DRAM-bank-pinned workers, in bank-id order.

    Order is load-bearing: it IS DRAMStreamingMatmul's bank-id assignment, so a sorted core list
    silently pairs workers with the wrong banks.
    """
    b = _try_import_blaze()
    cores = b["bank_workers"](device, ttnn.NOC.NOC_0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    return crs, len(cores)


def prepare_qkv_a_weights(device, w_q_a_torch, w_kv_a_torch, weight_dtype=ttnn.bfloat8_b) -> dict | None:
    """One-time weight prep for the blaze q_a/kv_a path.

    DRAMStreamingMatmul needs its weights DRAM-width-sharded across the banks AND column-major
    tile-shuffled. That is a load-time transform, so it costs nothing per token -- unlike the
    activation reshard, which does.
    """
    if not blaze_available():
        return None
    import torch  # local: only needed on the blaze path

    from tests.blaze.micro_ops.common.test_dram_streaming_matmul import (  # type: ignore
        _make_weights_tensor,
        _pad_to_dram_banks,
    )

    _, banks = _bank_worker_grid(device)
    out = {}
    for name, w in (("q_a", w_q_a_torch), ("kv_a", w_kv_a_torch)):
        k, n = int(w.shape[-2]), int(w.shape[-1])
        n_pad = _pad_to_dram_banks(n, 32, 32 * banks)
        if n_pad != n:
            w = torch.nn.functional.pad(w, (0, n_pad - n))
        out[name] = _make_weights_tensor(
            device, w, k=k, n_padded=n_pad, tile_w=32, num_banks=banks, weight_dtype=weight_dtype
        )
        out[f"{name}_n"] = n
        out[f"{name}_n_pad"] = n_pad
        out[f"{name}_k"] = k
    return out
