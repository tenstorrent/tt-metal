# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# CQ0 ``execute_trace`` capture/replay for denoise CFG steps and VAE decode.
# CQ1 stages input H2D / output D2H (2CQ trace pattern, ViT/recaption style).
#
# Enabled when ``HY_TRACE=1``. Capture failures propagate (no eager fallback).

from __future__ import annotations

import os
import time

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import COMPUTE_CQ, IO_CQ
from models.experimental.hunyuan_image_3_0.tt.matmul_utils import spill_resident_emb_to_dram
from models.experimental.hunyuan_image_3_0.tt.pipeline import (
    HunyuanTtDenoiseStep,
    classifier_free_guidance_tt,
    latent_tt_to_torch,
)
from models.experimental.hunyuan_image_3_0.tt.trace_config import denoise_execute_trace_enabled


def _upload_trace_buffer(device, torch_data: torch.Tensor, *, dtype, layout) -> ttnn.Tensor:
    kwargs = dict(dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(torch_data.contiguous(), **kwargs)


def _copy_trace_buffer(torch_data: torch.Tensor, device_tt: ttnn.Tensor, *, dtype, layout, cq_id: int = IO_CQ) -> None:
    host_tt = ttnn.from_torch(torch_data.contiguous(), dtype=dtype, layout=layout)
    ttnn.copy_host_to_device_tensor(host_tt, device_tt, cq_id=cq_id)


class Trace2CQIO:
    """Fence CQ0 compute vs CQ1 host I/O around trace replay."""

    def __init__(self, device):
        self.device = device
        self._io_write_event = None
        self._compute_done_event = None

    def stage_host_to_device(self, host_tt: ttnn.Tensor, device_tt: ttnn.Tensor) -> None:
        if self._io_write_event is not None:
            ttnn.wait_for_event(IO_CQ, self._io_write_event)
            self._io_write_event = None
        if self._compute_done_event is not None:
            ttnn.wait_for_event(IO_CQ, self._compute_done_event)
            self._compute_done_event = None
        ttnn.copy_host_to_device_tensor(host_tt, device_tt, cq_id=IO_CQ)
        self._io_write_event = ttnn.record_event(self.device, IO_CQ)

    def fence_compute_before_trace(self) -> None:
        if self._io_write_event is not None:
            ttnn.wait_for_event(COMPUTE_CQ, self._io_write_event)
            self._io_write_event = None

    def record_compute_done(self) -> None:
        self._compute_done_event = ttnn.record_event(self.device, COMPUTE_CQ)


class DenoiseStepTracer:
    """Capture one CFG denoise forward on CQ0; ``execute_trace`` per scheduler step."""

    def __init__(
        self,
        device,
        step: HunyuanTtDenoiseStep,
        *,
        time_embed,
        time_embed_2,
        scheduler,
        cond: dict,
        uncond: dict,
        guidance_scale: float,
        mesh_device,
        batch: int,
        channels: int,
        h: int,
        w: int,
        timestep_emb=None,
        guidance_emb=None,
        timestep_r_emb=None,
        cfg_distilled: bool = False,
        use_meanflow: bool = False,
    ):
        self.device = device
        self.step = step
        self.time_embed = time_embed
        self.time_embed_2 = time_embed_2
        self.scheduler = scheduler
        self.cond = cond
        self.uncond = uncond
        self.guidance_scale = guidance_scale
        self.mesh_device = mesh_device
        self.do_cfg = uncond is not None and guidance_scale != 1.0 and not cfg_distilled
        self.batch = batch
        self.channels = channels
        self.h = h
        self.w = w
        self.timestep_emb = timestep_emb
        self.guidance_emb = guidance_emb
        self.timestep_r_emb = timestep_r_emb
        self.cfg_distilled = cfg_distilled
        self.use_meanflow = use_meanflow
        self.distill_guidance = 1000.0 * guidance_scale if cfg_distilled else None
        self.io = Trace2CQIO(device)

        self.trace_id = None
        self.replay_steps = 0
        self._pred_tt = None
        self._latent_tt = None
        self._sample_tt = None
        self._tvec_tt = None
        self._latent_host = None
        self._tvec_host = None
        # I2I: stable device buffers for base_embeds (addresses must not change across replay).
        self._base_cond_tt = None
        self._base_uncond_tt = None
        self._base_cond_host = None
        self._base_uncond_host = None
        self._timing_capture_ms = 0.0
        self._timing_replay_total_ms = 0.0
        self._cos_sin = None
        self._capture_active = False

    def _ensure_cos_sin(self) -> None:
        """Upload 2D RoPE tables once — ``prepare_cos_sin`` H2D is illegal during capture."""
        if self._cos_sin is not None:
            return
        rope = self.step.backbone.layers[0].self_attn.rope
        self._cos_sin = rope.prepare_cos_sin(self.step.seq_len, image_infos=self.cond["image_infos"])

    def _abort_capture(self) -> None:
        if not self._capture_active or self.trace_id is None:
            return
        try:
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
        except Exception as exc:
            print(f"[denoise] trace abort end_trace_capture: {exc}", flush=True)
        self._capture_active = False

    def _latent_nhwc_host(self, latent_bchw: torch.Tensor) -> torch.Tensor:
        B, C, hh, ww = latent_bchw.shape
        return latent_bchw.permute(0, 2, 3, 1).reshape(1, 1, B * hh * ww, C).contiguous()

    def _prepare_base_embeds_host(self, c: dict, t_scalar: float) -> torch.Tensor | None:
        """I2I host embeds for this timestep (optional distill / meanflow scatter).

        Matches eager ``denoise_loop._prepare_base_embeds`` before the device upload.
        Returns None for T2I (``text_pre`` path) or when embeds are already on device.
        """
        host = c.get("base_embeds_host")
        if host is None:
            return None
        emb = host
        idx = c.get("gen_timestep_scatter_index")
        needs_scatter = (
            (idx is not None and self.timestep_emb is not None)
            or (self.cfg_distilled and c.get("guidance_scatter_index") is not None and self.guidance_emb is not None)
            or (
                self.use_meanflow
                and c.get("gen_timestep_r_scatter_index") is not None
                and self.timestep_r_emb is not None
            )
        )
        if needs_scatter:
            from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
                scatter_distill_step_embeds,
            )

            t_r = float(self.scheduler.get_timestep_r(t_scalar)) if self.use_meanflow else None
            emb = scatter_distill_step_embeds(
                emb,
                t_scalar=float(t_scalar),
                gen_timestep_scatter_index=idx,
                timestep_emb=self.timestep_emb,
                guidance_scalar=self.distill_guidance,
                guidance_scatter_index=c.get("guidance_scatter_index"),
                guidance_emb=self.guidance_emb,
                t_r_scalar=t_r,
                gen_timestep_r_scatter_index=c.get("gen_timestep_r_scatter_index"),
                timestep_r_emb=self.timestep_r_emb,
            )
        if isinstance(emb, torch.Tensor):
            return emb.detach().to(torch.bfloat16).contiguous()
        return emb

    def _stage_one_base_embeds(
        self,
        host: torch.Tensor,
        *,
        device_attr: str,
        host_attr: str,
    ) -> ttnn.Tensor:
        """Allocate once, then H2D into the same buffer each step (trace-safe)."""
        buf = getattr(self, device_attr)
        if buf is None:
            buf = _upload_trace_buffer(self.device, host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            setattr(self, device_attr, buf)
            return buf
        host_tt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        setattr(self, host_attr, host_tt)
        self.io.stage_host_to_device(host_tt, buf)
        return buf

    def _stage_base_embeds(self, t_scalar: float) -> None:
        """Upload I2I ``base_embeds_host`` into fixed device buffers for this step."""
        host_c = self._prepare_base_embeds_host(self.cond, t_scalar)
        if host_c is not None:
            self._stage_one_base_embeds(host_c, device_attr="_base_cond_tt", host_attr="_base_cond_host")
        if self.do_cfg and self.uncond is not None:
            host_u = self._prepare_base_embeds_host(self.uncond, t_scalar)
            if host_u is not None:
                self._stage_one_base_embeds(host_u, device_attr="_base_uncond_tt", host_attr="_base_uncond_host")

    def _init_buffers(self, latent_bchw: torch.Tensor, t_scalar: float) -> None:
        nhwc = self._latent_nhwc_host(latent_bchw)
        t_host = torch.tensor([float(t_scalar)] * self.batch, dtype=torch.float32).reshape(1, 1, self.batch, 1)
        if self._latent_tt is None:
            self._latent_tt = _upload_trace_buffer(self.device, nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._sample_tt = _upload_trace_buffer(self.device, nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._tvec_tt = _upload_trace_buffer(self.device, t_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        self._latent_host = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._tvec_host = ttnn.from_torch(t_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        self.io.stage_host_to_device(self._latent_host, self._latent_tt)
        self.io.stage_host_to_device(self._tvec_host, self._tvec_tt)
        self.io.stage_host_to_device(self._latent_host, self._sample_tt)
        # I2I: re-scatter timestep tokens on host, then stage into stable DRAM buffers.
        self._stage_base_embeds(float(t_scalar))

    def _base_embeds_for(self, c: dict) -> ttnn.Tensor | None:
        if c is self.cond and self._base_cond_tt is not None:
            return self._base_cond_tt
        if c is self.uncond and self._base_uncond_tt is not None:
            return self._base_uncond_tt
        base = c.get("base_embeds")
        if base is not None and isinstance(base, ttnn.Tensor):
            return base
        return None

    def _one_forward(self, c: dict, te1, te2) -> ttnn.Tensor:
        kwargs = dict(
            t_emb1=te1,
            t_emb2=te2,
            image_infos=c["image_infos"],
            attention_mask=c["attention_mask"],
            batch=c.get("batch", self.batch),
        )
        base = self._base_embeds_for(c)
        if base is not None:
            kwargs["base_embeds"] = base
        elif c.get("base_embeds_host") is not None:
            raise KeyError(
                "I2I cond has base_embeds_host but trace base_embeds buffer was not staged; "
                "call _init_buffers/_stage_base_embeds before _one_forward"
            )
        else:
            kwargs["text_pre"] = c["text_pre"]
            kwargs["text_post"] = c.get("text_post")
        return self.step.forward_device(self._latent_tt, cos_sin=self._cos_sin, **kwargs)

    def _make_step_temb(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """M=32-padded t_emb for ResBlocks; DRAM-spilled so backbone MoE keeps L1 free."""
        te1 = self.time_embed.forward(
            self._tvec_tt,
            keep_resident=True,
            resident_next_n=2 * self.step.patch_embed.resblock.out_channels,
        )
        te2 = self.time_embed_2.forward(
            self._tvec_tt,
            keep_resident=True,
            resident_next_n=2 * self.step.final_layer.resblock.out_channels,
        )
        return spill_resident_emb_to_dram(te1), spill_resident_emb_to_dram(te2)

    def _cfg_forward(self) -> ttnn.Tensor:
        te1, te2 = self._make_step_temb()
        pred_c = self._one_forward(self.cond, te1, te2)
        pred_u = self._one_forward(self.uncond, te1, te2)
        pred = classifier_free_guidance_tt(pred_c, pred_u, self.guidance_scale)
        ttnn.deallocate(pred_c)
        ttnn.deallocate(pred_u)
        ttnn.deallocate(te1)
        ttnn.deallocate(te2)
        return pred

    def _step_forward(self) -> ttnn.Tensor:
        if self.do_cfg:
            return self._cfg_forward()
        te1, te2 = self._make_step_temb()
        pred = self._one_forward(self.cond, te1, te2)
        ttnn.deallocate(te1)
        ttnn.deallocate(te2)
        return pred

    def _capture(self, latent_bchw: torch.Tensor, t_scalar: float) -> None:
        t0 = time.perf_counter()
        self._ensure_cos_sin()
        self._init_buffers(latent_bchw, t_scalar)
        self.io.fence_compute_before_trace()
        for _ in range(2):
            pred_w = self._step_forward()
            ttnn.deallocate(pred_w)
        ttnn.synchronize_device(self.device)

        print("[denoise] trace CFG: warmup done, begin_trace_capture on CQ0", flush=True)
        self.io.fence_compute_before_trace()
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=COMPUTE_CQ)
        self._capture_active = True
        try:
            self._pred_tt = self._step_forward()
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
            self._capture_active = False
        except Exception:
            self._abort_capture()
            raise
        ttnn.synchronize_device(self.device)
        self._timing_capture_ms = (time.perf_counter() - t0) * 1000
        self.replay_steps = 1
        print(
            f"[denoise] trace CFG captured trace_id={self.trace_id} ({self._timing_capture_ms:.1f} ms)",
            flush=True,
        )

    def _replay(self) -> None:
        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        self._timing_replay_total_ms += (time.perf_counter() - t0) * 1000

    def _euler(self, dt_scalar: float) -> torch.Tensor:
        scaled = ttnn.multiply(self._pred_tt, float(dt_scalar))
        nxt = ttnn.add(self._sample_tt, scaled)
        ttnn.deallocate(scaled)
        latent = latent_tt_to_torch(nxt, self.mesh_device, batch=self.batch, channels=self.channels, h=self.h, w=self.w)
        ttnn.deallocate(nxt)
        return latent

    def run(self, init_latent: torch.Tensor, timesteps) -> torch.Tensor:
        timesteps = list(timesteps)
        total = len(timesteps)
        latent = init_latent
        verbose = os.environ.get("HY_VERBOSE", "1") != "0"
        loop_t0 = time.time()

        self.scheduler.set_begin_index(0)
        for step_i, t in enumerate(timesteps):
            step_t0 = time.time()
            if self.scheduler.step_index is None:
                self.scheduler._init_step_index(t)
            dt = float(
                self.scheduler.sigmas[self.scheduler.step_index + 1] - self.scheduler.sigmas[self.scheduler.step_index]
            )

            if verbose:
                print(
                    f"[denoise] trace step {step_i + 1}/{total} t={float(t):.0f} +CFG execute_trace ...",
                    flush=True,
                )

            if self.trace_id is None:
                self._capture(latent, float(t))
            else:
                self._init_buffers(latent, float(t))
                self._replay()
                self.replay_steps += 1

            latent = self._euler(dt)
            self.scheduler._step_index += 1

            if verbose:
                print(
                    f"[denoise] trace step {step_i + 1}/{total} done "
                    f"({time.time() - step_t0:.1f}s, total {time.time() - loop_t0:.0f}s)",
                    flush=True,
                )

        avg = self._timing_replay_total_ms / max(1, self.replay_steps - 1)
        print(
            "[denoise] execute_trace summary: "
            f"capture={self._timing_capture_ms:.1f} ms, "
            f"replay={self._timing_replay_total_ms:.1f} ms / {max(0, self.replay_steps - 1)} steps "
            f"(avg {avg:.1f} ms/step)",
            flush=True,
        )
        return latent

    def release(self) -> None:
        self._abort_capture()
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
        if self._cos_sin is not None:
            cos_tt, sin_tt = self._cos_sin
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
            self._cos_sin = None
        for name in ("_base_cond_tt", "_base_uncond_tt", "_base_cond_host", "_base_uncond_host"):
            t = getattr(self, name, None)
            if t is not None:
                try:
                    ttnn.deallocate(t, force=False)
                except Exception:
                    pass
                setattr(self, name, None)


class VaeDecodeTracer:
    """Capture VAE decode on CQ0; ``execute_trace`` for the production forward."""

    def __init__(
        self,
        device,
        decoder,
        *,
        dtype,
        mesh_shape=None,
        dims=None,
        ccl=None,
        h_mesh_axis=None,
        w_mesh_axis=None,
        spatial: bool = False,
    ):
        self.device = device
        self.decoder = decoder
        self.ccl = ccl
        self.h_mesh_axis = h_mesh_axis
        self.w_mesh_axis = w_mesh_axis
        self.mesh_shape = mesh_shape
        self.dims = dims
        self.dtype = dtype
        self.spatial = spatial
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
            print(f"[vae] trace abort end_trace_capture: {exc}", flush=True)
        self._capture_active = False

    def _scaled_bthwc_host(self, latent_bchw: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        z = (latent_bchw / scaling_factor).unsqueeze(2)
        return (z.bfloat16() if self.dtype == ttnn.bfloat16 else z.float()).permute(0, 2, 3, 4, 1).contiguous()

    def _upload_input(self, host) -> ttnn.Tensor:
        kwargs = dict(
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.spatial:
            kwargs["mesh_mapper"] = ttnn.ShardTensor2dMesh(self.device, mesh_shape=self.mesh_shape, dims=self.dims)
        else:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(host, **kwargs)

    def _prepare_input(self, latent_bchw: torch.Tensor, scaling_factor: float) -> ttnn.Tensor:
        host = self._scaled_bthwc_host(latent_bchw, scaling_factor)
        if self._input_tt is None:
            self._input_tt = self._upload_input(host)
        elif not self.spatial:
            self._input_host = ttnn.from_torch(host, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.io.stage_host_to_device(self._input_host, self._input_tt)
        return self._input_tt

    def capture_and_run(self, latent_bchw: torch.Tensor, *, scaling_factor: float) -> ttnn.Tensor:
        x = self._prepare_input(latent_bchw, scaling_factor)
        self.io.fence_compute_before_trace()
        for _ in range(2):
            out_w = self.decoder(x)
            ttnn.deallocate(out_w)
        ttnn.synchronize_device(self.device)

        print("[vae] trace decode: warmup done, begin_trace_capture on CQ0", flush=True)
        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=COMPUTE_CQ)
        self._capture_active = True
        try:
            self._output_tt = self.decoder(x)
            ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
            self._capture_active = False
        except Exception:
            self._abort_capture()
            raise
        ttnn.synchronize_device(self.device)
        capture_ms = (time.perf_counter() - t0) * 1000
        print(f"[vae] trace decode captured trace_id={self.trace_id} ({capture_ms:.1f} ms)", flush=True)

        t0 = time.perf_counter()
        self.io.fence_compute_before_trace()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        replay_ms = (time.perf_counter() - t0) * 1000
        print(f"[vae] trace decode execute_trace done ({replay_ms:.1f} ms)", flush=True)
        return self._output_tt

    def release(self) -> None:
        self._abort_capture()
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None


def try_denoise_execute_trace(**kwargs) -> torch.Tensor:
    """Run traced denoise loop (raises on capture failure)."""
    if not denoise_execute_trace_enabled():
        raise RuntimeError("denoise execute_trace disabled (HY_TRACE=0)")
    tracer = DenoiseStepTracer(**kwargs)
    try:
        return tracer.run(kwargs["init_latent"], kwargs["scheduler"].timesteps)
    finally:
        tracer.release()
