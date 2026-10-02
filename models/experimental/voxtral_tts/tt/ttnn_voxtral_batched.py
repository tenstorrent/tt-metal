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
codec once per user. The per-frame device graph (backbone step, semantic head, flow solve, and with
the device loop the sampling glue) is captured as one Metal trace at warmup and replayed for every
batch (VOXTRAL_TRACE_PER_BATCH=1 captures per batch instead).

Noise: row b draws its 36-wide frame noise from torch.Generator().manual_seed(seed_b) in the same
order TtVoxtralPipeline consumes the global RNG for one request, so a row with (text, voice, seed)
sees the same x0 sequence as the single-user pipeline does for that request.

With the device loop (default, ttnn_voxtral_device_loop.py) nothing runs on the host between frames
except a stopped-mask readback every VOXTRAL_STOP_CHECK_EVERY frames; the noise table is uploaded once
per batch. VOXTRAL_DEVICE_LOOP=0 keeps the host loop (embedding gather, masked argmax, FSQ, noise
upload per frame).
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
from models.experimental.voxtral_tts.tt.ttnn_voxtral_device_loop import DeviceFrameLoop
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TILE, TtVoxtralGPT
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (
    TRACE_REGION_SIZE,
    frame_budget,
    open_device,
)


PER_BATCH_TRACE = os.environ.get("VOXTRAL_TRACE_PER_BATCH", "0") == "1"
# Phase B: sampling, stop logic, positions, noise and the next input embedding on device (one trace
# replay per frame, no host work between frames). VOXTRAL_DEVICE_LOOP=0 restores the host loop.
DEVICE_LOOP = os.environ.get("VOXTRAL_DEVICE_LOOP", "1") != "0"
CHECK_EVERY = int(os.environ.get("VOXTRAL_STOP_CHECK_EVERY", "8"))
# One-pass prefill for all users with 2D-multicast matmul configs (VOXTRAL_BATCHED_PREFILL=0: per user).
BATCHED_PREFILL = os.environ.get("VOXTRAL_BATCHED_PREFILL", "1") != "0"


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
            self.loop = None
            self.device_loop = DEVICE_LOOP
            self.batched_prefill = BATCHED_PREFILL
            if DEVICE_LOOP:
                self.loop = DeviceFrameLoop(
                    mesh_device,
                    self.B,
                    max_frames=max_seq_len,
                    audio_embeddings=self.wb["audio_embeddings"],
                    semantic_mask=self.flow.semantic_mask_host,
                    xin_dtype=self.backbone.dtype,
                    check_every=CHECK_EVERY,
                )
        except Exception:
            if self._owns_device:
                ttnn.close_device(mesh_device)
            raise
        self._tr = None
        self._tr_head = None
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
        return self._head(h, x0, cfg_alpha, n_steps)

    def _head(self, h, x0, cfg_alpha, n_steps):
        """normed hidden [1,B,3072] (device) -> (semantic logits [1,B,8320] fp32, flow solve x fp32)."""
        fl, B = self.flow, self.B
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
            "h0": dv(torch.zeros(1, B, DIM), self.backbone.dtype),  # frame-0 trace input (device loop)
            "pos_u32": dv(torch.zeros(1, Bpad, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "pos_i32": ttnn.from_torch(torch.zeros(B, dtype=torch.int32), dtype=ttnn.int32, device=dev),
            "x0": dv(
                torch.zeros(*((1, B, N_ACOUSTIC_CODEBOOK) if self.loop is not None else (B, 1, N_ACOUSTIC_CODEBOOK))),
                ttnn.float32,
            ),
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
        ttnn.copy_host_to_device_tensor(host(x0.reshape(tuple(buf["x0"].shape)), ttnn.float32), buf["x0"])

    def _trace_capture(self, positions, cfg_alpha, n_steps):
        """Capture the frame graph. The warm-up run and the capture both write K/V at `positions`
        (each row's first decode slot), which the first real traced frame then overwrites."""
        dev = self.device
        buf = self._buffers()
        self._fill(buf, torch.zeros(1, self.B, DIM), positions, torch.zeros(self.B, N_ACOUSTIC_CODEBOOK))
        lg, xr = self._graph(buf["xin"], buf["pos_u32"], buf["pos_i32"], buf["x0"], cfg_alpha, n_steps)  # program cache
        if self.loop is not None:
            self.loop.sample_and_advance(lg, xr, buf)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            lg, xr = self._graph(buf["xin"], buf["pos_u32"], buf["pos_i32"], buf["x0"], cfg_alpha, n_steps)
            if self.loop is not None:
                self.loop.sample_and_advance(lg, xr, buf)
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._tr = (tid, buf, lg, xr)
        self._tr_args = (float(cfg_alpha), int(n_steps))
        ttnn.synchronize_device(dev)
        if self.loop is not None:
            # Frame 0 as its own trace: semantic head + flow solve from the prefill hidden in buf["h0"],
            # then the sampler (eager, these ~60 small ops cost ~0.6 s of host dispatch per batch).
            lg0, xr0 = self._head(buf["h0"], buf["x0"], cfg_alpha, n_steps)
            self.loop.sample_and_advance(lg0, xr0, buf)
            ttnn.synchronize_device(dev)
            tid0 = ttnn.begin_trace_capture(dev, cq_id=0)
            try:
                lg0, xr0 = self._head(buf["h0"], buf["x0"], cfg_alpha, n_steps)
                self.loop.sample_and_advance(lg0, xr0, buf)
            finally:
                ttnn.end_trace_capture(dev, tid0, cq_id=0)
            self._tr_head = tid0
            ttnn.synchronize_device(dev)

    def _trace_release(self):
        if self._tr is not None:
            ttnn.release_trace(self.device, self._tr[0])
            self._tr = None
        if getattr(self, "_tr_head", None) is not None:
            ttnn.release_trace(self.device, self._tr_head)
            self._tr_head = None

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
        if self.batched_prefill and B > 1:
            t0 = time.perf_counter()
            gm = gpt.TtVoxtralGPT.PREFILL_GROUP_MULTIPLE
            sizes = sorted(
                {B, B // 2, B // 4} - {0}
            )  # common group sizes; other (users, length) shapes compile on first use
            for n_users in sizes:
                for sp in range(gm, 513, gm):  # length buckets up to 512 tokens
                    if bb.batched_prefill_fits(n_users, sp):
                        bb.prefill_batched([torch.zeros(1, sp, DIM)] * n_users, first_row=B - n_users)
            emit(f"[batched] warmup: one-pass prefill shapes in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self.flow(torch.zeros(B, DIM), x_0=torch.zeros(B, N_ACOUSTIC_CODEBOOK))  # eager flow at B (2B rows)
        bb.step_batched(torch.zeros(1, B, DIM), torch.full((B,), step, dtype=torch.int32))
        emit(f"[batched] warmup: flow + backbone step at B={B} in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        bucket = self.codec.bucket or 1
        # ceil, as TtVoxtralPipeline.warmup does: with a max_seq_len that is not a multiple of the bucket, a floor
        # left the top bucket cold, and its lazily cached codec tensors would be allocated while a trace is live.
        top = -(-bb.max_seq_len // bucket) * bucket
        buckets = list(range(bucket, top + 1, bucket))
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
                self._trace_release()
            # Trace once: keep it for every batch. Every tensor a replay touches (KV caches, the trace's
            # own input buffers, the flow constants) was allocated before this capture, prefill
            # temporaries are freed before the first replay, and per-frame inputs are copied into the
            # trace's buffers, so later allocations cannot be clobbered by a replay (the tt_transformers
            # rule). VOXTRAL_TRACE_PER_BATCH=1 restores capture-per-batch.
            if PER_BATCH_TRACE:
                self._trace_release()
        self.warmed = {"prefill_shapes": shapes, "codec_buckets": buckets, "traced": traced, "batch": B}
        self.warmed["seconds"] = time.perf_counter() - t_all
        emit(f"[batched] warmup: total {self.warmed['seconds']:.1f}s")
        return self

    def frame_graph_ms(self, replays=32, cfg_alpha=CFG_ALPHA, n_steps=N_DECODING_STEPS):
        """Device-only ms per frame: the traced frame graph (backbone step, semantic head, flow
        solve) replayed with no host work in between. The gap to generate_batch's ms/frame is
        the host work per frame (reads, copies, gathers, argmax, FSQ) that Phase B moves on device.
        Writes K/V at slot PREFILL_MULTIPLE in every row; the next prefill overwrites it."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt

        self._trace_capture(torch.full((self.B,), gpt.PREFILL_MULTIPLE, dtype=torch.int32), cfg_alpha, n_steps)
        try:
            tid = self._tr[0]
            for _ in range(3):
                ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            return (time.perf_counter() - t0) / replays * 1e3
        finally:
            self._trace_release()

    def backbone_graph_ms(self, replays=32):
        """Device-only ms per frame of the backbone step alone (traced), so the flow model's share
        of frame_graph_ms is measured, not inferred."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt

        dev, bb, B = self.device, self.backbone, self.B
        buf = self._buffers()
        self._fill(
            buf,
            torch.zeros(1, B, DIM),
            torch.full((B,), gpt.PREFILL_MULTIPLE, dtype=torch.int32),
            torch.zeros(B, N_ACOUSTIC_CODEBOOK),
        )
        for _ in range(3):
            bb.step_device(ttnn.clone(buf["xin"]), buf["pos_u32"], buf["pos_i32"])
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            out = bb.step_device(ttnn.clone(buf["xin"]), buf["pos_u32"], buf["pos_i32"])
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        try:
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / replays * 1e3
        finally:
            ttnn.release_trace(dev, tid)
            del out

    @staticmethod
    def noise_for(seed, n_frames):
        """[n_frames, 36]: the x0 sequence TtVoxtralPipeline's request with `seed` would draw.

        One randn of 36 values per frame, as the single-user pipeline draws them: torch's CPU
        normal sampler fills in blocks of 16 and recomputes the tail of a tensor whose size is
        not a multiple of 16, so randn(n_frames, 36)[t] is NOT the same as the t-th randn(1, 36)."""
        g = torch.Generator().manual_seed(int(seed))
        return torch.cat([torch.randn(1, N_ACOUSTIC_CODEBOOK, generator=g) for _ in range(n_frames)], dim=0)

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
        # Cache rows follow prompt length (shortest first) so the one-pass prefill can pad in groups;
        # `row_of[i]` is the row of request i, outputs are put back in request order at the end.
        order, bounds = gpt.TtVoxtralGPT.prefill_groups([e.shape[1] for e in embeds])
        row_of = [0] * B
        for row, i in enumerate(order):
            row_of[i] = row
        reqs = [reqs[i] for i in order]
        embeds = [embeds[i] for i in order]
        lens = torch.tensor([e.shape[1] for e in embeds], dtype=torch.int32)
        caps = []
        for (text, _, _), P in zip(reqs, lens.tolist()):
            room = bb.max_seq_len - P - 1
            if room < 1:
                raise ValueError(f"a {P}-token prompt leaves no room for audio in max_seq_len={bb.max_seq_len}")
            caps.append(min(frame_budget(text) if max_frames is None else int(max_frames), room))
        caps = torch.tensor(caps, dtype=torch.int32)
        groups = [(bounds[g], bounds[g + 1]) for g in range(len(bounds) - 1)]
        gm = gpt.TtVoxtralGPT.PREFILL_GROUP_MULTIPLE
        fits = all(bb.batched_prefill_fits(r1 - r0, -(-int(lens[r1 - 1]) // gm) * gm) for r0, r1 in groups)
        if self.batched_prefill and B > 1 and fits:
            h0 = torch.cat([bb.prefill_batched(embeds[r0:r1], first_row=r0) for r0, r1 in groups], dim=0)  # [B,3072]
        else:
            h0 = torch.cat([bb.prefill(embeds[b], last_only=True, user=b) for b in range(B)], dim=1)[0]  # [B,3072]
        t_prefill = time.perf_counter() - t0

        F = int(caps.max())
        x0 = torch.stack([self.noise_for(s, F) for _, _, s in reqs], dim=1)  # [F, B, 36]
        stopped = torch.zeros(B, dtype=torch.bool)
        frames = [[] for _ in range(B)]
        frozen = lens.clone()  # a stopped row keeps rewriting its last slot instead of advancing

        t0 = time.perf_counter()
        codes = None  # frame 0: host below for the host loop, device inside _device_loop
        traced = False
        kept = self._tr is not None and getattr(self, "_tr_args", None) == (float(cfg_alpha), int(n_steps))
        if kept:
            traced = True  # the warmup trace; positions, inputs and noise are copied in per frame
        elif TRACE_REGION_SIZE > 0:
            self._trace_release()
            try:
                self._trace_capture(lens, cfg_alpha, n_steps)
                traced = True
            except Exception as exc:
                self._trace_release()
                logger.warning(f"[batched] trace capture failed ({type(exc).__name__}), running eager")
        steps = 0
        if traced and self.device_loop and self.loop is not None:
            try:
                frames, steps, stopped = self._device_loop(h0, lens, caps, x0, F, cfg_alpha, n_steps, verbose, t0)
            finally:
                if PER_BATCH_TRACE or not kept:
                    self._trace_release()
            frames = [frames[row_of[i]] for i in range(B)]
            caps = caps[torch.tensor(row_of)]
            return self._finish(frames, steps, t_prefill, time.perf_counter() - t0, caps, n, traced, kept, B, True)
        # Frame 0 from the prefill hidden, eager (TtVoxtralPipeline does the same).
        codes = self.flow(h0, cfg_alpha=cfg_alpha, n_steps=n_steps, x_0=x0[0])
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
            if PER_BATCH_TRACE or not kept:
                self._trace_release()
        frames = [frames[row_of[i]] for i in range(B)]
        caps = caps[torch.tensor(row_of)]
        return self._finish(frames, steps, t_prefill, time.perf_counter() - t0, caps, n, traced, kept, B, False)

    def _finish(self, frames, steps, t_prefill, t_decode, caps, n, traced, kept, B, device_loop):
        n_frames = [len(f) for f in frames]
        self.last_timings = {
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "steps": steps,
            "decode_ms_per_frame": t_decode / max(steps, 1) * 1e3,
            "frames": n_frames[:n],
            "stopped_naturally": [bool(n_frames[b] < int(caps[b])) for b in range(n)],
            "traced": traced,
            "trace_reused": bool(kept),
            "device_loop": device_loop,
            "batch": B,
            "requests": n,
        }
        out = []
        for b in range(n):
            if not frames[b]:
                raise RuntimeError(f"row {b} emitted [END_AUDIO] on the first frame -- nothing to decode")
            out.append(torch.stack(frames[b], dim=0))
        return out

    def _device_loop(self, h0, lens, caps, x0, F, cfg_alpha, n_steps, verbose, t0):
        """Frame 0 from the prefill hidden through the device sampler (eager), then frames 1..F-1 as
        trace replays with no host work in between; the stopped mask is read every `check_every`
        frames, the codes once at the end. -> (frames per row, steps, stopped)."""
        B, loop = self.B, self.loop
        tid, buf, _, _ = self._tr
        # Frame 0: the sampler advances positions by one for live rows, so it is seeded one slot back
        # (lens - 1) and with frame index 0; afterwards pos = lens, the record holds frame 0 and buf
        # carries frame 1's input, exactly the state a replay expects.
        loop.seed0(buf, lens, caps, x0)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                h0.reshape(1, B, DIM).to(torch.float32).contiguous(), dtype=self.backbone.dtype, layout=ttnn.TILE_LAYOUT
            ),
            buf["h0"],
        )
        ttnn.execute_trace(self.device, self._tr_head, cq_id=0, blocking=False)
        steps = 0
        last = 0  # index of the last frame produced
        for t in range(1, F):
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            steps += 1
            last = t
            if t % loop.check_every == 0 and bool(loop.read_stopped().all()):
                break
            if verbose and t % 25 == 0:
                el = time.perf_counter() - t0
                logger.info(f"[batched/device-loop] {t} frames, {el / t * 1e3:.1f} ms/frame")
        cache = loop.read_codes()  # [B, F_pad, 37]; frame t at [:, t]
        frames = [[] for _ in range(B)]
        stopped = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            cap = int(caps[b])
            for t in range(0, last + 1):
                c = cache[b, t]
                if int(c[0]) == END_AUDIO_ID or t >= cap:
                    stopped[b] = True
                    break
                frames[b].append(c.clone())
        return frames, steps, stopped

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
