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

# ---------------------------------------------------------------------------
# THE BOUNDARY, and why the forward path is not wired yet.
#
# DRAMStreamingMatmul consumes its activation REPLICATED per DRAM-bank worker with a 1x32 decode
# tile. Building that from what the model holds at the q_kv_a call -- [1,1,32,2048] in standard
# 32x32 tiles -- measures at only **4.8 us/layer** on device:
#
#     slice(row 0) -> to_layout(ROW_MAJOR) -> repeat(x8) -> to_memory_config(height-sharded L1)
#
# against ~36 us of headroom (45.1 us ttnn fused q_kv_a vs 9.5 us blaze). So the reshard is cheap
# and the integration is worth having.
#
# What blocks it is the last hop: blaze reads `tensor.get_tile()` and raises
# "'NoneType' object has no attribute 'height'" on a ROW_MAJOR tensor, so it needs a real
# TILE-layout tensor whose tile is 1x32. No ttnn op produces one device-side -- all of
# `tilize(tile=)`, `tilize(output_tile=)` and `to_layout(TILE)` fail, and `ttnn.copy` into a
# correctly-specced `allocate_tensor_on_device` buffer is rejected because source and destination
# layouts differ (copy_device_operation.cpp:115). For 1x32 tiles the bytes are identical to
# row-major, so this is a type-system gap rather than a data problem.
#
# The right fix is not to fight ttnn at the tensor boundary but to do the retilize INSIDE the
# fused op, which is what blaze's `Retilize` micro-op exists for ("convert (1,32) row tiles to
# (N,32) standard tiles ... pattern from CreateQHeads"). Feeding GLMQKVAProjection the model's
# native activation and retilizing as its first phase keeps everything in one dispatch and
# removes the extra ttnn ops entirely.
#
# PROGRESS ON THE FIX. Passing a CBHandle instead of a tensor avoids the get_tile() call
# entirely, and `cb_from_tensor` does accept a tile override -- but only when `page_size` is
# given too, because row-major otherwise forces `eff_tile_desc = None`
# (fused_program.py:1380-1387). With `tile=Tile([1,32]), page_size=64` the tile error goes away
# and the next check is reached:
#
#     K mismatch: act gives 32, weights gives 2048
#
# `K_from_act = act_handle.num_pages * act_tile_w` (dram_streaming_matmul/common.py:278), so the
# CB needs 64 pages, not 1. `_resolve_tensor_geometry` derives that from a `total_size` kwarg,
# but `total_size` is NOT plumbed through to `BlazeProgram.cb_from_tensor`, which rejects it.
# So the remaining work is one of: plumb `total_size` through, or build the handle via the
# lower-level CB API that does expose page count.
#
# Verified so far: with blaze's own activation the fused op gates at PCC 0.9999 for both outputs.
# ---------------------------------------------------------------------------

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


def qkv_a(device, x, w, q_lora_rank: int, kvpe_dim: int, batch: int):
    """Blaze q_a/kv_a for the decode path, or None to fall back to ttnn.

    DEFAULT OFF, and deliberately so: at the model boundary this measures **0.52x** against the
    ttnn fused q_kv_a it replaces (94.0 us vs 47.5 us), correctness-gated at PCC 0.9999. It is
    wired up so the integration exists and can be flipped the moment the underlying limit is
    lifted -- blaze's DRAMStreamingMatmul runs on 8 DRAM-bank workers against ttnn's 80 cores,
    reaching 56 GB/s of a 512 GB/s device, and that is what makes it slow. Enabling this today
    costs ~2.2 ms/token; see blaze_eval/RESUME_HERE.md.

    Returns (q_a, kv) with logical widths, or None when disabled/unavailable so callers keep the
    ttnn path unchanged.
    """
    if not blaze_qkv_a_enabled():
        return None
    b = _try_import_blaze()
    state = _PROGRAM_CACHE.get(id(w))
    if state is None:
        prepared = prepare_qkv_a_weights(device, w.w_q_a_torch, w.w_kv_a_torch)
        if prepared is None:
            return None
        mk = lambda n: ttnn.from_torch(
            __import__("torch").zeros(1, 1, batch, n),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state = {"w": prepared, "q_out": mk(prepared["q_a_n_pad"]), "kv_out": mk(prepared["kv_a_n_pad"])}
        _PROGRAM_CACHE[id(w)] = state

    p = state["w"]
    f = b["FusedProgram"](
        kernel=None,
        device=device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="glm_qkv_a",
    )
    b["GLMQKVAProjection"].emit(
        f, x, p["q_a"], p["kv_a"], q_a_out=state["q_out"], kv_a_out=state["kv_out"], fp32_dest_acc_en=True
    )
    f.run()
    q_a, kv = state["q_out"], state["kv_out"]
    # The op pads N to a bank multiple; the model wants logical widths.
    if p["q_a_n_pad"] != q_lora_rank:
        q_a = ttnn.slice(q_a, [0, 0, 0, 0], [1, 1, batch, q_lora_rank])
    if p["kv_a_n_pad"] != kvpe_dim:
        kv = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, batch, kvpe_dim])
    return q_a, kv


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
