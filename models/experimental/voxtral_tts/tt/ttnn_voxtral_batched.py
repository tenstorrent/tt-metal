# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS, B users per step: the serving class for a batched server runner.

    pipe = TtVoxtralBatchedPipeline(max_batch=32)           # opens a one-chip mesh device
    pipe.warmup()
    wavs = pipe.synthesize_batch([("Hello.", "neutral_male", 0), ("Bonjour.", "fr_female", 3)])
    pipe.close()

One prefill per user into that user's KV-cache row (so prompts may differ in length), then one
decode step per frame for all users (TtVoxtralGPT.step_device at max_batch=B), the flow model at
B (its graph already takes a batch), per-row stop on [END_AUDIO] and per-row frame cap, and the
codec once per user. The per-frame device graph (backbone step, semantic head, flow solve) is
captured as one Metal trace after the prefills and released at the end of the batch, as
TtVoxtralPipeline does per request; capturing once at warmup is the Phase A follow-up.

Noise: row b draws its 36-wide frame noise from torch.Generator().manual_seed(seed_b) in the same
order TtVoxtralPipeline consumes the global RNG for one request, so a row with (text, voice, seed)
sees the same x0 sequence as the single-user pipeline does for that request.

Still on the host per frame (Phase B moves them on device): the next-frame embedding gather, the
semantic mask + argmax, FSQ quantisation, and the noise upload.
"""

import os
import time

import torch
import ttnn
from loguru import logger

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as backbone
from models.experimental.voxtral_tts.reference.voxtral_codec_ref import strip_offset_and_trim
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    CFG_ALPHA,
    DIM,
    EMPTY_AUDIO_ID,
    END_AUDIO_ID,
    N_ACOUSTIC_CODEBOOK,
    N_AUDIO_SPECIAL,
    N_DECODING_STEPS,
)
from models.experimental.voxtral_tts.reference.voxtral_paths import CKPT_NAME, resolve_model_dir
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flowmod
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TILE, TtVoxtralGPT
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (
    TRACE_REGION_SIZE,
    frame_budget,
    open_device,
)


class TtVoxtralBatchedPipeline:
    """B users per decode step on one chip. See the module docstring."""

    def __init__(self, mesh_device=None, ckpt_path=None, max_batch=32, max_seq_len=2048):
        t0 = time.perf_counter()
        if not 1 <= int(max_batch) <= TILE:
            raise ValueError(f"max_batch must be in 1..{TILE}, got {max_batch}")
        self.B = int(max_batch)
        self.model_dir = resolve_model_dir(ckpt_path)
        ckpt = os.path.join(self.model_dir, CKPT_NAME)
        self._owns_device = mesh_device is None
        if self._owns_device:
            mesh_device = open_device()
        self.mesh_device = self.device = mesh_device
        try:
            self.wb = backbone.load_backbone_state(ckpt)
            self.backbone = TtVoxtralGPT(mesh_device, state=self.wb, max_seq_len=max_seq_len, max_batch=self.B)
            self.flow = TtVoxtralFlow(mesh_device, ckpt_path=ckpt)
            self.codec = TtVoxtralCodecDecoder(mesh_device, ckpt_path=ckpt)
        except Exception:
            if self._owns_device:
                ttnn.close_device(mesh_device)
            raise
        self._tr = None
        self.last_timings = {}
        self.warmed = {}
        logger.info(
            f"[TtVoxtralBatchedPipeline] built (B={self.B}, max_seq_len={max_seq_len}, model={self.model_dir}) "
            f"in {time.perf_counter() - t0:.1f}s"
        )

    # ------------------------------------------------------------------
    # the per-frame device graph, traced
    # ------------------------------------------------------------------
    def _graph(self, xin, pos_u32, pos_i32, x0, cfg_alpha, n_steps):
        """ONE frame for all B rows, device tensors in and out: backbone step -> normed hidden
        [1,B,3072]; semantic logits [1,B,8320] fp32; flow solve x [B,1,36] fp32."""
        bb, fl, B = self.backbone, self.flow, self.B
        h = bb.step_device(ttnn.clone(xin), pos_u32, pos_i32)
        lg = ttnn.linear(
            ttnn.typecast(h, flowmod.SEMANTIC_DTYPE), fl.semantic_dev, compute_kernel_config=flowmod.COMPUTE_CONFIG
        )
        hh = ttnn.typecast(h, fl.dtype)
        # rows 0..B-1 conditioned on the hidden state, rows B..2B-1 unconditioned (zeros): the
        # reference's CFG pair, as one 2B forward.
        pair = ttnn.reshape(ttnn.concat([hh, ttnn.zeros_like(hh)], dim=1), [2 * B, 1, flowmod.FM_INPUT_DIM])
        return lg, fl._solve(x0, pair, B, n_steps, cfg_alpha)

    def _buffers(self):
        B, dev = self.B, self.device
        dv = lambda t, d, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(t.contiguous(), dtype=d, layout=layout, device=dev)
        Bpad = -(-B // TILE) * TILE
        return {
            "xin": dv(torch.zeros(1, B, DIM), self.backbone.dtype),
            "pos_u32": dv(torch.zeros(1, Bpad, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "pos_i32": ttnn.from_torch(torch.zeros(B, dtype=torch.int32), dtype=ttnn.int32, device=dev),
            "x0": dv(torch.zeros(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32),
        }

    def _fill(self, buf, x_host, positions, x0):
        """Copy one frame's host inputs into the trace's existing device buffers (no allocation)."""
        B = self.B
        Bpad = -(-B // TILE) * TILE
        padded = torch.zeros(1, Bpad, dtype=torch.int32)
        padded[0, :B] = positions
        host = lambda t, d, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(t.contiguous(), dtype=d, layout=layout)
        ttnn.copy_host_to_device_tensor(host(x_host.reshape(1, B, DIM), self.backbone.dtype), buf["xin"])
        ttnn.copy_host_to_device_tensor(host(padded, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), buf["pos_u32"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(positions.to(torch.int32), dtype=ttnn.int32), buf["pos_i32"])
        ttnn.copy_host_to_device_tensor(host(x0.reshape(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32), buf["x0"])

    def _trace_capture(self, positions, cfg_alpha, n_steps):
        """Capture the frame graph. The warm-up run and the capture both write K/V at `positions`
        (each row's first decode slot), which the first real traced frame then overwrites."""
        dev = self.device
        buf = self._buffers()
        self._fill(buf, torch.zeros(1, self.B, DIM), positions, torch.zeros(self.B, N_ACOUSTIC_CODEBOOK))
        self._graph(buf["xin"], buf["pos_u32"], buf["pos_i32"], buf["x0"], cfg_alpha, n_steps)  # program cache
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            lg, xr = self._graph(buf["xin"], buf["pos_u32"], buf["pos_i32"], buf["x0"], cfg_alpha, n_steps)
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._tr = (tid, buf, lg, xr)
        ttnn.synchronize_device(dev)

    def _trace_release(self):
        if self._tr is not None:
            ttnn.release_trace(self.device, self._tr[0])
            self._tr = None

    def _frame_codes(self, logits, xr, stopped):
        """host: masked argmax -> semantic [B]; FSQ -> acoustic [B,36]; END rows get EMPTY codes.
        -> codes [B,37] int64 with the special-token offset applied."""
        B = self.B
        sem = (logits.reshape(B, -1) + self.flow.semantic_mask_host).argmax(-1).long()
        ac = flowmod._fsq_quantize(xr.reshape(B, N_ACOUSTIC_CODEBOOK))
        ac[(sem == END_AUDIO_ID) | stopped] = EMPTY_AUDIO_ID
        return torch.cat([sem.reshape(B, 1), ac + N_AUDIO_SPECIAL], dim=1)

    def _step(self, codes, positions, x0, traced, cfg_alpha, n_steps, stopped):
        """Feed every row its last frame's codes at its own position -> next frame's codes [B,37]."""
        B = self.B
        x = torch.cat([backbone.embed_frame(self.wb, codes[b]) for b in range(B)], dim=1)  # [1,B,3072]
        if traced:
            tid, buf, lg, xr = self._tr
            self._fill(buf, x, positions, x0)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            return self._frame_codes(ttnn.to_torch(lg).float(), ttnn.to_torch(xr).float(), stopped)
        h = self.backbone.step_batched(x, positions)[0]  # [B,3072]
        sem = self.flow.semantic_code(h)
        ac = self.flow.decode_frame(sem, h, cfg_alpha=cfg_alpha, n_steps=n_steps, x_0=x0)
        ac[stopped] = EMPTY_AUDIO_ID + N_AUDIO_SPECIAL
        return torch.cat([sem, ac], dim=1)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def warmup(self, verbose=False):
        """Compile every prefill shape and codec bucket (as TtVoxtralPipeline does), the flow model
        at B, and capture one frame trace at B, then release it."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt
        from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref

        emit = logger.info if verbose else logger.debug
        t_all = time.perf_counter()
        bb, B = self.backbone, self.B
        step = gpt.PREFILL_MULTIPLE
        shapes = list(range(step, bb.max_seq_len + 1, step))
        t0 = time.perf_counter()
        for sp in shapes:
            bb.prefill(torch.zeros(1, sp, DIM), last_only=True, user=0)
        emit(f"[batched] warmup: prefill {len(shapes)} shapes in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self.flow(torch.zeros(B, DIM), x_0=torch.zeros(B, N_ACOUSTIC_CODEBOOK))  # eager flow at B (2B rows)
        bb.step_batched(torch.zeros(1, B, DIM), torch.full((B,), step, dtype=torch.int32))
        emit(f"[batched] warmup: flow + backbone step at B={B} in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        bucket = self.codec.bucket or 1
        buckets = list(range(bucket, bb.max_seq_len + 1, bucket))
        for n in buckets:
            self.codec(cref.make_synthetic_codes(n))
        emit(f"[batched] warmup: codec {len(buckets)} buckets in {time.perf_counter() - t0:.1f}s")
        traced = False
        if TRACE_REGION_SIZE > 0:
            t0 = time.perf_counter()
            try:
                self._trace_capture(torch.full((B,), step, dtype=torch.int32), CFG_ALPHA, N_DECODING_STEPS)
                traced = True
                emit(f"[batched] warmup: trace captured in {time.perf_counter() - t0:.1f}s")
            except Exception as exc:
                logger.warning(f"[batched] warmup: trace capture failed ({type(exc).__name__}: {exc})")
            finally:
                self._trace_release()
        self.warmed = {"prefill_shapes": shapes, "codec_buckets": buckets, "traced": traced, "batch": B}
        self.warmed["seconds"] = time.perf_counter() - t_all
        emit(f"[batched] warmup: total {self.warmed['seconds']:.1f}s")
        return self

    @staticmethod
    def noise_for(seed, n_frames):
        """[n_frames, 36]: the x0 sequence TtVoxtralPipeline's request with `seed` would draw."""
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(n_frames, N_ACOUSTIC_CODEBOOK, generator=g)

    @torch.no_grad()
    def generate_batch(self, requests, max_frames=None, cfg_alpha=CFG_ALPHA, n_steps=N_DECODING_STEPS, verbose=False):
        """requests: up to B of (text, voice, seed) -> list of frames [T_b, 37] int64 (offset applied,
        [END_AUDIO] excluded), one per request. Fewer than B requests are padded with request 0;
        padding rows are computed and dropped."""
        from models.experimental.voxtral_tts import frontend

        B, bb = self.B, self.backbone
        n = len(requests)
        if not 1 <= n <= B:
            raise ValueError(f"synthesize_batch takes 1..{B} requests, got {n}")
        reqs = list(requests) + [requests[0]] * (B - n)
        log = logger.info if verbose else logger.debug

        t0 = time.perf_counter()
        embeds = [frontend.build_prompt_embeds(t, v, self.wb, model_dir=self.model_dir) for t, v, _ in reqs]
        lens = torch.tensor([e.shape[1] for e in embeds], dtype=torch.int32)
        caps = []
        for (text, _, _), P in zip(reqs, lens.tolist()):
            room = bb.max_seq_len - P - 1
            if room < 1:
                raise ValueError(f"a {P}-token prompt leaves no room for audio in max_seq_len={bb.max_seq_len}")
            caps.append(min(frame_budget(text) if max_frames is None else int(max_frames), room))
        caps = torch.tensor(caps, dtype=torch.int32)
        h0 = torch.cat([bb.prefill(embeds[b], last_only=True, user=b) for b in range(B)], dim=1)[0]  # [B,3072]
        t_prefill = time.perf_counter() - t0

        F = int(caps.max())
        x0 = torch.stack([self.noise_for(s, F) for _, _, s in reqs], dim=1)  # [F, B, 36]
        stopped = torch.zeros(B, dtype=torch.bool)
        frames = [[] for _ in range(B)]
        frozen = lens.clone()  # a stopped row keeps rewriting its last slot instead of advancing

        t0 = time.perf_counter()
        # Frame 0 from the prefill hidden, eager (TtVoxtralPipeline does the same).
        codes = self.flow(h0, cfg_alpha=cfg_alpha, n_steps=n_steps, x_0=x0[0])
        traced = False
        if TRACE_REGION_SIZE > 0:
            try:
                self._trace_capture(lens, cfg_alpha, n_steps)
                traced = True
            except Exception as exc:
                self._trace_release()
                logger.warning(f"[batched] trace capture failed ({type(exc).__name__}), running eager")
        steps = 0
        try:
            for t in range(F):
                for b in range(B):
                    if stopped[b]:
                        continue
                    if int(codes[b, 0]) == END_AUDIO_ID or t >= int(caps[b]):
                        stopped[b] = True
                        frozen[b] = lens[b] + t
                    else:
                        frames[b].append(codes[b].clone())
                if bool(stopped.all()):
                    break
                positions = torch.where(stopped, frozen, lens + t)
                if t + 1 >= F:
                    break
                codes = self._step(codes, positions, x0[t + 1], traced, cfg_alpha, n_steps, stopped)
                steps += 1
                if verbose and (t + 1) % 25 == 0:
                    el = time.perf_counter() - t0
                    log(
                        f"[batched] {t + 1} frames, {int((~stopped).sum())} rows active, {el / (t + 1) * 1e3:.1f} ms/frame"
                    )
        finally:
            self._trace_release()
        t_decode = time.perf_counter() - t0
        n_frames = [len(f) for f in frames]
        self.last_timings = {
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "steps": steps,
            "decode_ms_per_frame": t_decode / max(steps, 1) * 1e3,
            "frames": n_frames[:n],
            "stopped_naturally": [bool(n_frames[b] < int(caps[b])) for b in range(n)],
            "traced": traced,
            "batch": B,
            "requests": n,
        }
        out = []
        for b in range(n):
            if not frames[b]:
                raise RuntimeError(f"row {b} emitted [END_AUDIO] on the first frame -- nothing to decode")
            out.append(torch.stack(frames[b], dim=0))
        return out

    @torch.no_grad()
    def decode(self, frames):
        """frames [T,37] -> waveform torch [1,1,T*1920] @ 24 kHz, via the codec (one user)."""
        return self.codec(strip_offset_and_trim(frames))

    @torch.no_grad()
    def synthesize_batch(self, requests, max_frames=None, cfg_alpha=CFG_ALPHA, verbose=False):
        """[(text, voice, seed), ...] (<= B) -> [waveform [1,1,N] @ 24 kHz, ...], same order."""
        frames = self.generate_batch(requests, max_frames=max_frames, cfg_alpha=cfg_alpha, verbose=verbose)
        t0 = time.perf_counter()
        wavs = [self.decode(f) for f in frames]
        self.last_timings["codec_s"] = time.perf_counter() - t0
        self.last_timings["audio_s"] = [float(w.shape[-1]) / 24000.0 for w in wavs]
        return wavs

    def synthesize(self, text, voice="neutral_male", seed=0, max_frames=None, cfg_alpha=CFG_ALPHA):
        """One request through the batched machinery (padding rows dropped)."""
        return self.synthesize_batch([(text, voice, seed)], max_frames=max_frames, cfg_alpha=cfg_alpha)[0]

    def close(self):
        self._trace_release()
        self.last_timings, self.warmed = {}, {}
        if self._owns_device and self.mesh_device is not None:
            ttnn.close_device(self.mesh_device)
            self.mesh_device = self.device = None
