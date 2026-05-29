# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN helpers for 5 Hz LM narrow CFG logits (valid audio-token slice only)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from models.demos.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers import _device_key


@contextlib.contextmanager
def _strict_ttnn_no_fallback() -> Iterator[None]:
    import ttnn

    cfg = getattr(ttnn, "CONFIG", None)
    if cfg is None or not hasattr(cfg, "throw_exception_on_fallback"):
        yield
        return
    prev = bool(cfg.throw_exception_on_fallback)
    cfg.throw_exception_on_fallback = True
    try:
        yield
    finally:
        cfg.throw_exception_on_fallback = prev


@dataclass
class _CfgLogitsTraceState:
    trace_id: Optional[int] = None
    cond_buf: Optional[object] = None
    uncond_buf: Optional[object] = None
    out_buf: Optional[object] = None
    vocab_k: int = 0
    cfg_scale: float = 1.0
    op_event: Optional[object] = None
    stage_event: Optional[object] = None


_CFG_TRACE_CACHE: dict[tuple[int, int, int], _CfgLogitsTraceState] = {}


def _cfg_trace_key(device, k: int, cfg_scale: float) -> tuple[int, int, int]:
    return (_device_key(device), int(k), int(round(float(cfg_scale) * 1000)))


def cfg_linear_combination_bf16(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cfg_scale: float,
    *,
    device,
    memory_config=None,
    use_trace: bool = True,
) -> torch.Tensor:
    """``uncond + cfg_scale * (cond - uncond)`` on TTNN; optional 2CQ trace replay per ``[B,K]`` shape."""
    import ttnn

    if cond.shape != uncond.shape:
        raise ValueError(f"cond/uncond shape mismatch: {tuple(cond.shape)} vs {tuple(uncond.shape)}")
    if cond.dim() != 2:
        raise ValueError(f"expected rank-2 [B, K] logits, got {tuple(cond.shape)}")

    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    c = cond.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    u = uncond.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()

    if (
        not use_trace
        or not hasattr(ttnn, "begin_trace_capture")
        or not hasattr(ttnn, "execute_trace")
        or not hasattr(ttnn, "clone")
    ):
        with _strict_ttnn_no_fallback():
            tt_c = ttnn.from_torch(
                c, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            tt_u = ttnn.from_torch(
                u, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            diff = ttnn.subtract(tt_c, tt_u, memory_config=mem)
            ttnn.deallocate(tt_c)
            scaled = ttnn.multiply(diff, float(cfg_scale), memory_config=mem)
            ttnn.deallocate(diff)
            out_tt = ttnn.add(tt_u, scaled, memory_config=mem)
            ttnn.deallocate(tt_u)
            ttnn.deallocate(scaled)
            out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
            ttnn.deallocate(out_tt)
        return out

    key = _cfg_trace_key(device, int(c.shape[1]), float(cfg_scale))
    st = _CFG_TRACE_CACHE.get(key)
    if st is None:
        st = _CfgLogitsTraceState(vocab_k=int(c.shape[1]), cfg_scale=float(cfg_scale))
        _CFG_TRACE_CACHE[key] = st

    with _strict_ttnn_no_fallback():
        if st.trace_id is None:
            tt_c = ttnn.from_torch(
                c, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            tt_u = ttnn.from_torch(
                u, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            diff = ttnn.subtract(tt_c, tt_u, memory_config=mem)
            scaled = ttnn.multiply(diff, float(cfg_scale), memory_config=mem)
            out_warm = ttnn.add(tt_u, scaled, memory_config=mem)
            ttnn.synchronize_device(device)
            for t in (tt_c, tt_u, diff, scaled):
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
            st.cond_buf = ttnn.clone(
                ttnn.from_torch(c, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
            )
            st.uncond_buf = ttnn.clone(
                ttnn.from_torch(u, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
            )
            st.out_buf = ttnn.clone(out_warm)
            try:
                ttnn.deallocate(out_warm)
            except Exception:
                pass
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            diff = ttnn.subtract(st.cond_buf, st.uncond_buf, memory_config=mem)
            scaled = ttnn.multiply(diff, float(cfg_scale), memory_config=mem)
            st.out_buf = ttnn.add(st.uncond_buf, scaled, memory_config=mem)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            st.trace_id = tid
            ttnn.execute_trace(device, st.trace_id, cq_id=0, blocking=True)
            st.op_event = ttnn.record_event(device, 0)
            ttnn.synchronize_device(device)
            return ttnn.to_torch(st.out_buf, dtype=torch.float32).contiguous()

        if st.op_event is not None:
            ttnn.wait_for_event(1, st.op_event)
        host_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        host_u = ttnn.from_torch(u, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_c, st.cond_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(host_u, st.uncond_buf, cq_id=1)
        st.stage_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, st.stage_event)
        ttnn.execute_trace(device, st.trace_id, cq_id=0, blocking=True)
        st.op_event = ttnn.record_event(device, 0)
        ttnn.synchronize_device(device)
        return ttnn.to_torch(st.out_buf, dtype=torch.float32).contiguous()


__all__ = ["cfg_linear_combination_bf16"]
