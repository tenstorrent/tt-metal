# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Voxtral-TTS on device: text ids + voice preset -> 24 kHz waveform.

See NOTES.md [pipe-01].
"""

import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as backbone
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, END_AUDIO_ID
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import (
    CFG_ALPHA, N_DECODING_STEPS, TtVoxtralFlow)

FRAME_RATE = 12.5

# L1 scratch every caller needs: the codec's convs fail with "bank size is 0 B" without it.
L1_SMALL_SIZE = 65536
# NOTES.md [pipe-05] -- the frame loop is TRACED. 0 falls back to eager.
TRACE_REGION_SIZE = 250 * 1024 * 1024


def open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE):
    """Open a device configured the way every entry point here needs it. See NOTES.md [pipe-05]."""
    return ttnn.open_device(device_id=device_id, l1_small_size=L1_SMALL_SIZE,
                            trace_region_size=trace_region_size)


# NOTES.md [pipe-02] -- A HANG THAT SHAPED THE SHIPPED CONFIG, recorded because...


class TtVoxtralPipeline:
    """All three blocks on device. generate(embeds) -> frames; decode(frames) -> waveform."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, max_seq_len=2048):
        """`max_seq_len` holds prompt + generated frames TOGETHER and is the only cap on utterance
        length -- 2048 is ~136 s of audio. Raising it costs DRAM and nothing per frame (§6.69)."""
        self.device = device
        # NOTES.md [pipe-03] -- embed_frame is a host gather, so it needs the backbone's...
        self.wb = backbone.load_backbone_state(ckpt_path)
        self.backbone = TtVoxtralGPT(device, state=self.wb, max_seq_len=max_seq_len)
        self.flow = TtVoxtralFlow(device, ckpt_path=ckpt_path)
        self.codec = TtVoxtralCodecDecoder(device, ckpt_path=ckpt_path)
        self._tr = None            # (trace_id, input buffers, output tensors), built per generate()
        # Per-stage wall times from the last request, including the codec.
        self.last_timings = {}
        # What warmup() actually compiled, so callers (and tests) can check rather than assume.
        self.warmed = {}

    # ------------------------------------------------------------------
    # TRACED FRAME LOOP -- NOTES.md [pipe-05], STATUS.md 6.65
    # ------------------------------------------------------------------
    def _trace_capture(self, cfg_alpha, n_steps):
        """Capture the WHOLE per-frame device graph. See NOTES.md [pipe-05]."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt
        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, HEAD_DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow

        bb, fl, dev = self.backbone, self.flow, self.device
        B = 1
        dv = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                          device=dev)
        buf = {
            "xin": dv(torch.zeros(1, 1, DIM), bb.dtype),
            "cos": dv(torch.zeros(1, 1, 1, HEAD_DIM), bb.dtype),
            "sin": dv(torch.zeros(1, 1, 1, HEAD_DIM), bb.dtype),
            "pos": ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=dev),
            "x0": dv(torch.zeros(B, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32),
        }

        def graph():
            # resharded here, not on the host -- NOTES.md [pipe-05]
            cos = ttnn.to_memory_config(buf["cos"], gpt._ROPE_SHARD)
            sin = ttnn.to_memory_config(buf["sin"], gpt._ROPE_SHARD)
            x = ttnn.clone(buf["xin"])
            for i, w in enumerate(bb.layers):
                x = bb._layer_step(x, w, cos, sin, bb.caches[i], buf["pos"])
            h = bb._norm(x, bb.norm)
            lg = ttnn.linear(ttnn.typecast(h, flow.SEMANTIC_DTYPE), fl.semantic_dev,
                             compute_kernel_config=flow.COMPUTE_CONFIG)
            hh = ttnn.typecast(h, fl.dtype)
            pair = ttnn.reshape(ttnn.concat([hh, ttnn.zeros_like(hh)], dim=1),
                                [2 * B, 1, flow.FM_INPUT_DIM])
            return lg, fl._solve(buf["x0"], pair, B, n_steps, cfg_alpha)

        pos0 = bb.pos
        # AIM THE CAPTURE'S CACHE WRITES AT pos0 -- NOTES.md [pipe-05]. Left at 0 this corrupts the
        # prompt and the audio is garbage.
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([pos0], dtype=torch.int32)), buf["pos"])
        graph()                                   # populate the program cache before capturing
        ttnn.synchronize_device(dev)
        bb.pos = pos0
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            lg, xr = graph()
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)   # never leave a capture open -- [pipe-05]
        # registered immediately so a failure past this point still has something to release
        self._tr = (tid, buf, lg, xr)
        ttnn.synchronize_device(dev)
        bb.pos = pos0

    def _trace_release(self):
        if self._tr is not None:
            ttnn.release_trace(self.device, self._tr[0])
            self._tr = None

    def _traced_frame(self, codes):
        """One frame through the trace. NOTHING here may allocate -- NOTES.md [pipe-05]."""
        import models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt as gpt
        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, HEAD_DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow

        tid, buf, lg, xr = self._tr
        bb, fl, dev = self.backbone, self.flow, self.device
        pos = bb.pos
        cb, sb = gpt.rope_tables(1, offset=pos)
        host = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(
            host(backbone.embed_frame(self.wb, codes).reshape(1, 1, DIM), bb.dtype), buf["xin"])
        ttnn.copy_host_to_device_tensor(host(cb.reshape(1, 1, 1, HEAD_DIM), bb.dtype), buf["cos"])
        ttnn.copy_host_to_device_tensor(host(sb.reshape(1, 1, 1, HEAD_DIM), bb.dtype), buf["sin"])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32)), buf["pos"])
        ttnn.copy_host_to_device_tensor(
            host(torch.randn(1, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32), buf["x0"])
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        sem = ((ttnn.to_torch(lg).float().reshape(1, -1) + fl.semantic_mask_host)
               .argmax(-1).reshape(1, 1).long())
        bb.pos = pos + 1
        if int(sem[0, 0]) == END_AUDIO_ID:
            return torch.cat([sem, torch.full((1, flow.N_ACOUSTIC_CODEBOOK), flow.EMPTY_AUDIO_ID,
                                              dtype=torch.long)], dim=1)
        ac = flow._fsq_quantize(ttnn.to_torch(xr).float().reshape(1, flow.N_ACOUSTIC_CODEBOOK))
        return torch.cat([sem, ac + flow.N_AUDIO_SPECIAL], dim=1)

    def warmup(self, max_frames=640, capture_trace=True, verbose=False):
        """Compile every program the request path can reach, then capture the frame-loop trace.

        Prefill compiles per padded length and the codec per frame bucket, so without this the first
        request at each new length pays a compile. Everything that allocates is compiled BEFORE the
        trace capture, since a program compiled afterwards can land where the trace keeps its
        intermediates.

        Zero inputs are enough: this builds kernels and asserts nothing. The codec stage synthesises
        its own codes, because Block 2 on zeros emits [END_AUDIO] and the trim would leave nothing.

        Args:
            max_frames: how many frames of codec bucket to compile for.
            capture_trace: also capture and release the per-frame trace.
            verbose: per-stage timings.

        Sets `self.warmed` to what was compiled.
        """
        import time as _time

        from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM
        from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as _gpt

        t_all = _time.perf_counter()
        log = (lambda m: print(f"[warmup] {m}", flush=True)) if verbose else (lambda m: None)

        # 1) Prefill, at every padded shape this cache can hold. The expensive part.
        t0 = _time.perf_counter()
        step = _gpt.PREFILL_MULTIPLE
        shapes = list(range(step, self.backbone.max_seq_len + 1, step))
        for sp in shapes:
            self.backbone.reset()
            self.backbone.prefill(torch.zeros(1, sp, DIM), last_only=True)
        self.backbone.reset()
        log(f"prefill: {len(shapes)} shapes ({shapes[0]}..{shapes[-1]}) in "
            f"{_time.perf_counter() - t0:.1f}s")

        # 2) Block 2 once -- one shape, it is per-frame and length-independent.
        t0 = _time.perf_counter()
        h = self.backbone.prefill_last(torch.zeros(1, step, DIM))
        codes = self.flow(h[:, 0])
        self.backbone.reset()
        log(f"block 2: 1 shape in {_time.perf_counter() - t0:.1f}s")

        # 3) Codec, at every length bucket a request can reach.
        t0 = _time.perf_counter()
        from models.experimental.voxtral_tts.reference import voxtral_codec_ref as _cref

        bucket = self.codec.bucket or 1
        buckets = list(range(bucket, max(max_frames, bucket) + 1, bucket))
        for n in buckets:
            self.codec(_cref.make_synthetic_codes(n))
        log(f"codec: {len(buckets)} buckets ({buckets[0]}..{buckets[-1]}) in "
            f"{_time.perf_counter() - t0:.1f}s")

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
        """Release the trace and drop per-request state. Does not close the device: the caller owns it."""
        self._trace_release()
        self.last_timings = {}
        self.warmed = {}

    @torch.no_grad()
    def generate(self, embeds, max_frames=150, cfg_alpha=CFG_ALPHA, seed=0, verbose=True):
        """prompt embeds [1,P,3072] -> frames [T,37] int64 (offset applied, [END_AUDIO] excluded)."""
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        # NOTES.md [pipe-04] -- Only the last position conditions the first frame...
        h = self.backbone.prefill_last(embeds)   # [1,1,3072]
        t_prefill = time.perf_counter() - t0
        if verbose:
            print(f"[pipeline] prefill P={embeds.shape[1]} in {t_prefill:.2f}s")

        frames, t0 = [], time.perf_counter()
        stopped = False
        # NOTES.md [pipe-05] -- frame 0 is EAGER and must come BEFORE the capture
        codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
        # NOTES.md [pipe-05] -- try to trace, fall back to eager. The decision cannot come from
        # TRACE_REGION_SIZE: a caller may have opened the device with a different one.
        traced = False
        if TRACE_REGION_SIZE > 0:
            try:
                self._trace_capture(cfg_alpha, N_DECODING_STEPS)
                traced = True
            except Exception as exc:
                self._trace_release()
                if verbose:
                    print(f"[pipeline] trace capture failed ({type(exc).__name__}), running eager")
        try:
            for i in range(max_frames):
                if int(codes[0, 0]) == END_AUDIO_ID:
                    if verbose:
                        print(f"[pipeline] [END_AUDIO] at frame {i} -- natural stop")
                    stopped = True
                    break
                frames.append(codes)
                if i + 1 == max_frames:
                    break                       # the next frame could never be appended
                if traced:
                    codes = self._traced_frame(codes[0])
                else:
                    h = self.backbone.step(
                        backbone.embed_frame(self.wb, codes[0])).reshape(1, 1, -1)
                    codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
                if verbose and (i + 1) % 10 == 0:
                    el = time.perf_counter() - t0
                    print(f"[pipeline]   {i+1} frames ({(i+1)/FRAME_RATE:.1f}s audio) "
                          f"| {el/(i+1):.2f}s/frame")
        finally:
            self._trace_release()      # the next generate() prefills, which allocates -- [pipe-05]
        if verbose and not stopped:
            print(f"[pipeline] hit max_frames={max_frames} without [END_AUDIO]")
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
        """frames [T,37] -> waveform torch [1,1,T*1920] @ 24 kHz, via Block 3."""
        from models.experimental.voxtral_tts.reference.voxtral_codec_ref import strip_offset_and_trim

        t0 = time.perf_counter()
        wav = self.codec(strip_offset_and_trim(frames))
        self.last_timings["codec_s"] = time.perf_counter() - t0
        return wav
