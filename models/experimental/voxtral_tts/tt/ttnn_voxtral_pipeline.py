# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Voxtral-TTS on device: text + voice preset -> 24 kHz waveform.

    tts = TtVoxtralPipeline()              # or (mesh_device=..., ckpt_path=...)
    tts.warmup()                           # once: compile every shape, capture the frame loop
    wav = tts.synthesize("Hello.", "neutral_male", seed=0)   # [1,1,N] @ 24 kHz; repeatable
    tts.close()

The backbone, flow model and codec on TTNN, with the per-frame loop traced; `generate` (prompt
embeds -> codes) and `decode` (codes -> waveform) are the two halves `synthesize` chains. The host
keeps the tokenizer, prompt assembly and three small per-frame steps (the embedding gather, the FSQ
quantise, the semantic mask + argmax), each cheaper there than a device dispatch.
"""

import math
import os
import time

import torch
import ttnn
from loguru import logger

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as backbone
from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID
from models.experimental.voxtral_tts.reference.voxtral_paths import CKPT_NAME, resolve_model_dir
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA, N_DECODING_STEPS, TtVoxtralFlow

FRAME_RATE = 12.5

# L1 scratch for the codec's convs. Every compiled codec bucket keeps its conv config tensors
# here, so it is sized for all the buckets warmup compiles.
L1_SMALL_SIZE = 131072
# DRAM reserved for the frame-loop trace. It must hold the captured trace, or capture fails and
# generate() runs eager with a warning; 0 skips tracing.
TRACE_REGION_SIZE = 250 * 1024 * 1024


def open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE):
    """-> a one-chip MeshDevice (ttnn.open_device opens a 1x1 mesh) with the L1 scratch and trace
    region the pipeline needs; the type TtVoxtralPipeline(mesh_device) takes."""
    return ttnn.open_device(device_id=device_id, l1_small_size=L1_SMALL_SIZE, trace_region_size=trace_region_size)


def frame_budget(text):
    """A frame cap for `text`: ~18 chars/s at 12.5 frames/s, x2.2 margin, floor 320. A cap, not a
    cost: generation stops on [END_AUDIO]."""
    return max(320, int(math.ceil(len(text) / 18.0 * FRAME_RATE * 2.2)))


class TtVoxtralPipeline:
    """All three stages on device, one persistent object for many requests (module docstring)."""

    def __init__(self, mesh_device=None, ckpt_path=None, max_seq_len=2048):
        """Opens (and owns) a one-chip mesh device if none is given; `ckpt_path` is the model
        directory; `max_seq_len` caps prompt + frames together."""
        t0 = time.perf_counter()
        self.model_dir = resolve_model_dir(ckpt_path)
        ckpt = os.path.join(self.model_dir, CKPT_NAME)
        self._owns_device = mesh_device is None
        if self._owns_device:
            mesh_device = open_device()
        self.mesh_device = self.device = mesh_device
        try:
            # Loaded once and shared: the backbone takes it instead of loading its own copy, and
            # the host embedding gather reads it every frame.
            self.wb = backbone.load_backbone_state(ckpt)
            self.backbone = TtVoxtralGPT(mesh_device, state=self.wb, max_seq_len=max_seq_len)
            self.flow = TtVoxtralFlow(mesh_device, ckpt_path=ckpt)
            self.codec = TtVoxtralCodecDecoder(mesh_device, ckpt_path=ckpt)
        except Exception:
            if self._owns_device:
                ttnn.close_device(mesh_device)
            raise
        self._tr = None  # (trace_id, input buffers, output tensors), built per generate()
        # Per-stage wall times from the last request, including the codec.
        self.last_timings = {}
        # What warmup() actually compiled, so callers (and tests) can check rather than assume.
        self.warmed = {}
        logger.info(f"[TtVoxtralPipeline] built (model={self.model_dir}) in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # TRACED FRAME LOOP
    # ------------------------------------------------------------------
    def _trace_capture(self, cfg_alpha, n_steps):
        """Capture the whole per-frame device graph: backbone step, semantic head, flow model. It must
        be ALL the per-frame device work: a device allocation while the trace exists can corrupt it."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt
        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, HEAD_DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow

        bb, fl, dev = self.backbone, self.flow, self.device
        B = 1
        dv = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
        buf = {
            "xin": dv(torch.zeros(1, 1, DIM), bb.dtype),
            "cos": dv(torch.zeros(1, 1, 1, HEAD_DIM), bb.dtype),
            "sin": dv(torch.zeros(1, 1, 1, HEAD_DIM), bb.dtype),
            "pos": ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=dev),
            "x0": dv(torch.zeros(B, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32),
        }

        def graph():
            # cos/sin are copied in interleaved and resharded here, inside the trace, for RoPE decode.
            cos = ttnn.to_memory_config(buf["cos"], gpt._ROPE_SHARD)
            sin = ttnn.to_memory_config(buf["sin"], gpt._ROPE_SHARD)
            x = ttnn.clone(buf["xin"])
            for i, w in enumerate(bb.layers):
                x = bb._layer_step(x, w, cos, sin, bb.caches[i], buf["pos"])
            h = bb._norm(x, bb.norm)
            lg = ttnn.linear(
                ttnn.typecast(h, flow.SEMANTIC_DTYPE), fl.semantic_dev, compute_kernel_config=flow.COMPUTE_CONFIG
            )
            hh = ttnn.typecast(h, fl.dtype)
            pair = ttnn.reshape(ttnn.concat([hh, ttnn.zeros_like(hh)], dim=1), [2 * B, 1, flow.FM_INPUT_DIM])
            return lg, fl._solve(buf["x0"], pair, B, n_steps, cfg_alpha)

        pos0 = bb.pos
        # Aim the capture's K/V writes at pos0; at 0 they corrupt the prompt.
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos0], dtype=torch.int32)), buf["pos"])
        graph()  # populate the program cache before capturing
        ttnn.synchronize_device(dev)
        bb.pos = pos0
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            lg, xr = graph()
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)  # an open capture wedges the device
        # registered immediately so a failure past this point still has something to release
        self._tr = (tid, buf, lg, xr)
        ttnn.synchronize_device(dev)
        bb.pos = pos0

    def _trace_release(self):
        if self._tr is not None:
            ttnn.release_trace(self.device, self._tr[0])
            self._tr = None

    def _traced_frame(self, codes):
        """One frame through the trace; nothing here may allocate on device. Host tensors are
        built without `device=` and copied into the trace's existing buffers."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt
        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, HEAD_DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow

        tid, buf, lg, xr = self._tr
        bb, fl, dev = self.backbone, self.flow, self.device
        pos = bb.pos
        cb, sb = gpt.rope_tables(1, offset=pos)
        host = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(
            host(backbone.embed_frame(self.wb, codes).reshape(1, 1, DIM), bb.dtype), buf["xin"]
        )
        ttnn.copy_host_to_device_tensor(host(cb.reshape(1, 1, 1, HEAD_DIM), bb.dtype), buf["cos"])
        ttnn.copy_host_to_device_tensor(host(sb.reshape(1, 1, 1, HEAD_DIM), bb.dtype), buf["sin"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32)), buf["pos"])
        ttnn.copy_host_to_device_tensor(host(torch.randn(1, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32), buf["x0"])
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        sem = (ttnn.to_torch(lg).float().reshape(1, -1) + fl.semantic_mask_host).argmax(-1).reshape(1, 1).long()
        bb.pos = pos + 1
        if int(sem[0, 0]) == END_AUDIO_ID:
            return torch.cat(
                [sem, torch.full((1, flow.N_ACOUSTIC_CODEBOOK), flow.EMPTY_AUDIO_ID, dtype=torch.long)], dim=1
            )
        ac = flow._fsq_quantize(ttnn.to_torch(xr).float().reshape(1, flow.N_ACOUSTIC_CODEBOOK))
        return torch.cat([sem, ac + flow.N_AUDIO_SPECIAL], dim=1)

    def warmup(self, max_frames=None, capture_trace=True, verbose=False, codec=True):
        """Compile every program the request path can reach, then capture (and release) the
        frame-loop trace. `max_frames` defaults to the cache length, the most any request can
        reach. `codec=False` skips the codec buckets (a caller that never synthesizes audio, e.g.
        a second pipeline built on a chip that already runs one). Sets `self.warmed`.
        """
        import time as _time

        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as _gpt

        t_all = _time.perf_counter()
        emit = logger.info if verbose else logger.debug
        log = lambda m: emit(f"[TtVoxtralPipeline] warmup: {m}")

        # 1) Prefill, at every padded shape this cache can hold. The expensive part.
        t0 = _time.perf_counter()
        step = _gpt.PREFILL_MULTIPLE
        shapes = list(range(step, self.backbone.max_seq_len + 1, step))
        for sp in shapes:
            self.backbone.reset()
            self.backbone.prefill(torch.zeros(1, sp, DIM), last_only=True)
        self.backbone.reset()
        log(f"prefill: {len(shapes)} shapes ({shapes[0]}..{shapes[-1]}) in {_time.perf_counter() - t0:.1f}s")

        # 2) the flow model once -- one shape, it is per-frame and length-independent.
        t0 = _time.perf_counter()
        h = self.backbone.prefill_last(torch.zeros(1, step, DIM))
        self.flow(h[:, 0])
        self.backbone.reset()
        log(f"Flow model: 1 shape in {_time.perf_counter() - t0:.1f}s")

        # 3) Codec, at every length bucket a request can reach.
        t0 = _time.perf_counter()
        from models.experimental.voxtral_tts.reference import voxtral_codec_ref as _cref

        bucket = self.codec.bucket or 1
        top = -(-(max_frames or self.backbone.max_seq_len) // bucket) * bucket
        buckets = list(range(bucket, top + 1, bucket)) if codec else []
        for n in buckets:
            self.codec(_cref.make_synthetic_codes(n))
        if buckets:
            log(f"codec: {len(buckets)} buckets ({buckets[0]}..{buckets[-1]}) in {_time.perf_counter() - t0:.1f}s")
        else:
            log("codec: skipped")

        # 4) The frame-loop trace, LAST, after every compile above.
        traced = False
        if capture_trace and TRACE_REGION_SIZE > 0:
            t0 = _time.perf_counter()
            try:
                self.backbone.reset()
                self.backbone.prefill(torch.zeros(1, step, DIM), last_only=True)
                self._trace_capture(CFG_ALPHA, N_DECODING_STEPS)
                traced = True
                log(f"trace captured in {_time.perf_counter() - t0:.1f}s")
            except Exception as exc:
                log(f"trace capture failed ({type(exc).__name__}), leaving it to generate()")
            finally:
                self._trace_release()
                self.backbone.reset()

        self.warmed = {
            "prefill_shapes": shapes,
            "codec_buckets": buckets,
            "traced": traced,
            "seconds": _time.perf_counter() - t_all,
        }
        log(f"total {self.warmed['seconds']:.1f}s")
        return self

    def close(self):
        """Release the trace, and the device if this instance opened it. Safe to call twice."""
        if self.mesh_device is not None:
            self._trace_release()
        self.last_timings = {}
        self.warmed = {}
        if self._owns_device and self.mesh_device is not None:
            ttnn.close_device(self.mesh_device)
            self.mesh_device = self.device = None

    @torch.no_grad()
    def synthesize(self, text, voice="neutral_male", seed=0, max_frames=None, cfg_alpha=CFG_ALPHA):
        """text + voice preset name -> waveform torch [1,1,N] float @ 24 kHz: the front end,
        `generate` and `decode` in one call. `max_frames` defaults to `frame_budget(text)`."""
        from models.experimental.voxtral_tts import frontend

        embeds = frontend.build_prompt_embeds(text, voice, self.wb, model_dir=self.model_dir)
        room = self.backbone.max_seq_len - embeds.shape[1]
        if room < 1:
            raise ValueError(
                f"a {embeds.shape[1]}-token prompt leaves no room for audio in max_seq_len={self.backbone.max_seq_len}"
            )
        cap = min(frame_budget(text) if max_frames is None else max_frames, room)
        self.backbone.reset()
        frames, _, _ = self.generate(embeds, max_frames=cap, cfg_alpha=cfg_alpha, seed=seed, verbose=False)
        return self.decode(frames)

    @torch.no_grad()
    def generate(self, embeds, max_frames=150, cfg_alpha=CFG_ALPHA, seed=0, verbose=True):
        """prompt embeds [1,P,3072] -> frames [T,37] int64 (offset applied, [END_AUDIO] excluded)."""
        if max_frames < 1:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        # Only the last position conditions the first frame.
        h = self.backbone.prefill_last(embeds)  # [1,1,3072]
        t_prefill = time.perf_counter() - t0
        log = logger.info if verbose else logger.debug
        log(f"[TtVoxtralPipeline] prefill P={embeds.shape[1]} in {t_prefill:.2f}s")

        frames, t0 = [], time.perf_counter()
        stopped = False
        # Frame 0 is eager and comes before the capture: its hidden state is prefill's, not a backbone step's.
        codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
        # Try to trace, fall back to eager: the caller may have opened the device with a different
        # trace region than TRACE_REGION_SIZE.
        traced = False
        if TRACE_REGION_SIZE > 0:
            try:
                self._trace_capture(cfg_alpha, N_DECODING_STEPS)
                traced = True
            except Exception as exc:
                self._trace_release()
                logger.warning(f"[TtVoxtralPipeline] trace capture failed ({type(exc).__name__}), running eager")
        try:
            for i in range(max_frames):
                if int(codes[0, 0]) == END_AUDIO_ID:
                    log(f"[TtVoxtralPipeline] [END_AUDIO] at frame {i} -- natural stop")
                    stopped = True
                    break
                frames.append(codes)
                if i + 1 == max_frames:
                    break  # the next frame could never be appended
                if traced:
                    codes = self._traced_frame(codes[0])
                else:
                    h = self.backbone.step(backbone.embed_frame(self.wb, codes[0])).reshape(1, 1, -1)
                    codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
                if (i + 1) % 10 == 0:
                    el = time.perf_counter() - t0
                    log(
                        f"[TtVoxtralPipeline]   {i+1} frames ({(i+1)/FRAME_RATE:.1f}s audio) "
                        f"| {el/(i+1):.2f}s/frame"
                    )
        finally:
            self._trace_release()  # the next generate() prefills, which allocates
        if not stopped:
            logger.warning(f"[TtVoxtralPipeline] hit max_frames={max_frames} without [END_AUDIO]")
        if not frames:
            raise RuntimeError("model emitted [END_AUDIO] on the first frame -- nothing to decode")
        t_decode = time.perf_counter() - t0
        out = torch.cat(frames, dim=0)
        # Same tuple as always -- callers unpack three values. The dict is additive.
        self.last_timings = {
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "frames": int(out.shape[0]),
            "decode_ms_per_frame": t_decode / max(out.shape[0], 1) * 1e3,
            "traced": traced,
        }
        return out, t_prefill, t_decode

    @torch.no_grad()
    def decode(self, frames):
        """frames [T,37] -> waveform torch [1,1,T*1920] @ 24 kHz, via the codec."""
        from models.experimental.voxtral_tts.reference.voxtral_codec_ref import strip_offset_and_trim

        t0 = time.perf_counter()
        wav = self.codec(strip_offset_and_trim(frames))
        self.last_timings["codec_s"] = time.perf_counter() - t0
        return wav
