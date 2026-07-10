# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# CQ0 ``execute_trace`` capture/replay for I2I cond encode (VAE encoder + ViT/aligner).
# Gaussian VAE sampling (``ttnn.randn``) stays outside trace — only the encoder forward is traced.
#
# Cached per device + geometry so recap + denoise cond encode (same image) can replay.

from __future__ import annotations

import time

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import COMPUTE_CQ
from models.experimental.hunyuan_image_3_0.tt.stage_trace import Trace2CQIO
from models.experimental.hunyuan_image_3_0.tt.trace_config import cond_encode_trace_enabled
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    Siglip2VisionInputs,
    build_siglip2_attention_mask,
    forward_vision_with_aligner,
)

_VAE_ENCODE_TRACERS: dict[tuple, "VaeEncodeTracer"] = {}
_VIT_ENCODE_TRACERS: dict[tuple, "VitEncodeTracer"] = {}


def _consumer_copy(trace_out: ttnn.Tensor) -> ttnn.Tensor:
    """Fresh device copy — returned tensors are deallocated by callers/inject."""
    work = ttnn.allocate_tensor_on_device(trace_out.spec, trace_out.device())
    ttnn.copy(trace_out, work)
    return work


class VaeEncodeTracer:
    """Capture VAE encoder forward on CQ0; ``execute_trace`` on replay."""

    def __init__(self, device, encoder, *, pixel_h: int, pixel_w: int, dtype):
        self.device = device
        self.encoder = encoder
        self.pixel_h = pixel_h
        self.pixel_w = pixel_w
        self.dtype = dtype
        self.io = Trace2CQIO(device)
        self.trace_id = None
        self._input_tt = None
        self._input_host = None
        self._output_tt = None
        self._capture_active = False

    def _abort_capture(self) -> None:
        if not self._capture_active or self.trace_id is None:
            return
        try:
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
        except Exception as exc:
            print(f"[cond_encode] VAE trace abort end_trace_capture: {exc}", flush=True)
        self._capture_active = False

    def _upload_input(self, host_bthwc: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_bthwc,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def _prepare_input(self, host_bthwc: torch.Tensor) -> ttnn.Tensor:
        if self._input_tt is None:
            self._input_tt = self._upload_input(host_bthwc)
        else:
            self._input_host = ttnn.from_torch(host_bthwc, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.io.stage_host_to_device(self._input_host, self._input_tt)
        return self._input_tt

    def capture_and_run(self, host_bthwc: torch.Tensor) -> ttnn.Tensor:
        total_t0 = time.perf_counter()
        x = self._prepare_input(host_bthwc)
        self.io.fence_compute_before_trace()
        for _ in range(2):
            out_w = self.encoder(x)
            ttnn.deallocate(out_w)
        ttnn.synchronize_device(self.device)

        print(
            f"[cond_encode] VAE trace encode ({self.pixel_h}x{self.pixel_w}): "
            "warmup done, begin_trace_capture on CQ0",
            flush=True,
        )
        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=COMPUTE_CQ)
        self._capture_active = True
        try:
            self._output_tt = self.encoder(x)
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
            self._capture_active = False
        except Exception:
            self._abort_capture()
            raise
        ttnn.synchronize_device(self.device)
        capture_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[cond_encode] VAE trace encode captured trace_id={self.trace_id} ({capture_ms:.1f} ms)",
            flush=True,
        )

        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        ttnn.synchronize_device(self.device)
        replay_ms = (time.perf_counter() - t0) * 1000
        print(f"[cond_encode] VAE trace encode execute_trace done ({replay_ms:.1f} ms)", flush=True)
        out = _consumer_copy(self._output_tt)
        total_ms = (time.perf_counter() - total_t0) * 1000
        print(
            f"[cond_encode] VAE trace encode ({self.pixel_h}x{self.pixel_w}) "
            f"capture+run total={total_ms:.1f} ms (capture={capture_ms:.1f} replay={replay_ms:.1f})",
            flush=True,
        )
        return out

    def replay(self, host_bthwc: torch.Tensor, *, restage_input: bool = True) -> ttnn.Tensor:
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.device)
        if restage_input:
            self._prepare_input(host_bthwc)
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        ttnn.synchronize_device(self.device)
        out = _consumer_copy(self._output_tt)
        print(
            f"[cond_encode] VAE trace encode ({self.pixel_h}x{self.pixel_w}) "
            f"replay total={(time.perf_counter() - t0) * 1000:.1f} ms",
            flush=True,
        )
        return out

    def release(self) -> None:
        self._abort_capture()
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None


class VitEncodeTracer:
    """Capture SigLIP2 vision + aligner on CQ0; ``execute_trace`` on replay."""

    def __init__(
        self,
        device,
        vision,
        aligner,
        *,
        spatial_shapes_hw: tuple[tuple[int, int], ...],
        seq_len: int,
        dtype,
    ):
        self.device = device
        self.vision = vision
        self.aligner = aligner
        self.spatial_shapes_hw = spatial_shapes_hw
        self.seq_len = seq_len
        self.dtype = dtype
        self.io = Trace2CQIO(device)
        self.trace_id = None
        self._pv_tt = None
        self._mask_tt = None
        self._mask_4d_tt = None
        self._pv_host = None
        self._mask_host = None
        self._output_tt = None
        self._capture_active = False

    def _abort_capture(self) -> None:
        if not self._capture_active or self.trace_id is None:
            return
        try:
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
        except Exception as exc:
            print(f"[cond_encode] ViT trace abort end_trace_capture: {exc}", flush=True)
        self._capture_active = False

    def _upload(self, host_pv: torch.Tensor, host_mask: torch.Tensor) -> None:
        kwargs = dict(
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        if self._pv_tt is None:
            self._pv_tt = ttnn.from_torch(host_pv.contiguous(), **kwargs)
            self._mask_tt = ttnn.from_torch(host_mask.contiguous(), **kwargs)
            self._rebuild_mask_4d()
        else:
            self._pv_host = ttnn.from_torch(host_pv.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
            self._mask_host = ttnn.from_torch(host_mask.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
            self.io.stage_host_to_device(self._pv_host, self._pv_tt)
            self.io.stage_host_to_device(self._mask_host, self._mask_tt)
            self._rebuild_mask_4d()

    def _rebuild_mask_4d(self) -> None:
        if self._mask_4d_tt is not None:
            ttnn.deallocate(self._mask_4d_tt)
        self._mask_4d_tt = build_siglip2_attention_mask(self._mask_tt)

    def _inputs(self) -> Siglip2VisionInputs:
        return Siglip2VisionInputs(
            pixel_values=self._pv_tt,
            spatial_shapes_hw=self.spatial_shapes_hw,
            pixel_attention_mask=self._mask_tt,
            attention_mask_4d=self._mask_4d_tt,
        )

    def _forward(self) -> ttnn.Tensor:
        return forward_vision_with_aligner(self.vision, self.aligner, self._inputs())

    def _prewarm_pos(self) -> None:
        geoms = [(hw[0], hw[1], self.seq_len) for hw in self.spatial_shapes_hw]
        self.vision.prewarm_pos_geometries(geoms)

    def capture_and_run(self, host_pv: torch.Tensor, host_mask: torch.Tensor) -> ttnn.Tensor:
        total_t0 = time.perf_counter()
        self._upload(host_pv, host_mask)
        self._prewarm_pos()
        self.io.fence_compute_before_trace()
        for _ in range(2):
            out_w = self._forward()
            ttnn.deallocate(out_w)
        ttnn.synchronize_device(self.device)

        print("[cond_encode] ViT trace encode: warmup done, begin_trace_capture on CQ0", flush=True)
        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=COMPUTE_CQ)
        self._capture_active = True
        try:
            self._output_tt = self._forward()
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
            self._capture_active = False
        except Exception:
            self._abort_capture()
            raise
        ttnn.synchronize_device(self.device)
        capture_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[cond_encode] ViT trace encode captured trace_id={self.trace_id} ({capture_ms:.1f} ms)",
            flush=True,
        )

        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        ttnn.synchronize_device(self.device)
        replay_ms = (time.perf_counter() - t0) * 1000
        print(f"[cond_encode] ViT trace encode execute_trace done ({replay_ms:.1f} ms)", flush=True)
        out = _consumer_copy(self._output_tt)
        total_ms = (time.perf_counter() - total_t0) * 1000
        print(
            f"[cond_encode] ViT trace encode capture+run total={total_ms:.1f} ms "
            f"(capture={capture_ms:.1f} replay={replay_ms:.1f})",
            flush=True,
        )
        return out

    def replay(self, host_pv: torch.Tensor, host_mask: torch.Tensor, *, restage_input: bool = True) -> ttnn.Tensor:
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.device)
        if restage_input:
            self._upload(host_pv, host_mask)
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        ttnn.synchronize_device(self.device)
        out = _consumer_copy(self._output_tt)
        print(
            f"[cond_encode] ViT trace encode replay total={(time.perf_counter() - t0) * 1000:.1f} ms",
            flush=True,
        )
        return out

    def release(self) -> None:
        self._abort_capture()
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None


def _vit_cache_key(device, spatial_shapes_hw: tuple[tuple[int, int], ...], seq_len: int) -> tuple:
    return (id(device), spatial_shapes_hw, seq_len)


def run_vae_encode_traced(
    mesh_device,
    vae_encoder,
    host_bthwc: torch.Tensor,
    *,
    pixel_h: int,
    pixel_w: int,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """VAE encoder forward via trace (capture on first call, replay after)."""
    if not cond_encode_trace_enabled():
        raise RuntimeError("cond encode trace disabled (set HY_COND_ENCODE_TRACE=1 and HY_TRACE=1)")
    key = (id(mesh_device), pixel_h, pixel_w, dtype)
    tracer = _VAE_ENCODE_TRACERS.get(key)
    if tracer is None:
        print(
            f"[cond_encode] VAE trace encode ({pixel_h}x{pixel_w}) first capture (HY_COND_ENCODE_TRACE=1)", flush=True
        )
        tracer = VaeEncodeTracer(mesh_device, vae_encoder, pixel_h=pixel_h, pixel_w=pixel_w, dtype=dtype)
        _VAE_ENCODE_TRACERS[key] = tracer
        return tracer.capture_and_run(host_bthwc)
    print(f"[cond_encode] VAE trace encode ({pixel_h}x{pixel_w}) execute_trace replay", flush=True)
    return tracer.replay(host_bthwc)


def run_vit_encode_traced(
    mesh_device,
    vision,
    aligner,
    host_pv: torch.Tensor,
    host_mask: torch.Tensor,
    *,
    spatial_shapes_hw: tuple[tuple[int, int], ...],
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """ViT + aligner forward via trace (capture on first call, replay after)."""
    if not cond_encode_trace_enabled():
        raise RuntimeError("cond encode trace disabled (set HY_COND_ENCODE_TRACE=1 and HY_TRACE=1)")
    seq_len = int(host_pv.shape[1])
    key = _vit_cache_key(mesh_device, spatial_shapes_hw, seq_len)
    tracer = _VIT_ENCODE_TRACERS.get(key)
    if tracer is None:
        print("[cond_encode] ViT trace encode first capture (HY_COND_ENCODE_TRACE=1)", flush=True)
        tracer = VitEncodeTracer(
            mesh_device,
            vision,
            aligner,
            spatial_shapes_hw=spatial_shapes_hw,
            seq_len=seq_len,
            dtype=dtype,
        )
        _VIT_ENCODE_TRACERS[key] = tracer
        return tracer.capture_and_run(host_pv, host_mask)
    print("[cond_encode] ViT trace encode execute_trace replay", flush=True)
    return tracer.replay(host_pv, host_mask)


def release_cond_encode_tracers() -> None:
    """Release cached cond-encode traces (call at pipeline teardown)."""
    for tracer in list(_VAE_ENCODE_TRACERS.values()):
        tracer.release()
    for tracer in list(_VIT_ENCODE_TRACERS.values()):
        tracer.release()
    _VAE_ENCODE_TRACERS.clear()
    _VIT_ENCODE_TRACERS.clear()


def invalidate_cond_encode_traces() -> None:
    """Drop cached traces before backbone load — replay after backbone invalidates trace DRAM."""
    if _VAE_ENCODE_TRACERS or _VIT_ENCODE_TRACERS:
        print("[cond_encode] invalidating cached cond-encode traces (stage boundary)", flush=True)
    release_cond_encode_tracers()
