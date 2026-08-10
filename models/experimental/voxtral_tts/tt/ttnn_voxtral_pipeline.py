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
# NOTES.md [pipe-05] -- the frame loop is TRACED, and the region has to exist at open_device time.
# 250 MB holds the whole per-frame graph (Block 1's 26 layers + the semantic projection + Block 2's
# 7 Euler steps). Set to 0 to run everything eagerly. STATUS.md 6.65.
TRACE_REGION_SIZE = 250 * 1024 * 1024


def open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE):
    """Open a device configured the way every entry point here needs it.

    `trace_region_size` must be non-zero for generate()'s traced frame loop. NOTE that merely
    allocating it shifts the allocator enough to move a free-running trajectory (95dc26363f), so a
    run opened with it is not frame-count comparable to one opened without.
    """
    return ttnn.open_device(device_id=device_id, l1_small_size=L1_SMALL_SIZE,
                            trace_region_size=trace_region_size)


# NOTES.md [pipe-02] -- A HANG THAT SHAPED THE SHIPPED CONFIG, recorded because...


class TtVoxtralPipeline:
    """All three blocks on device. generate(embeds) -> frames; decode(frames) -> waveform."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, max_seq_len=1024):
        self.device = device
        # NOTES.md [pipe-03] -- embed_frame is a host gather, so it needs the backbone's...
        self.wb = backbone.load_backbone_state(ckpt_path)
        self.backbone = TtVoxtralGPT(device, state=self.wb, max_seq_len=max_seq_len)
        self.flow = TtVoxtralFlow(device, ckpt_path=ckpt_path)
        self.codec = TtVoxtralCodecDecoder(device, ckpt_path=ckpt_path)
        self._tr = None            # (trace_id, input buffers, output tensors), built per generate()

    # ------------------------------------------------------------------
    # TRACED FRAME LOOP -- NOTES.md [pipe-05], STATUS.md 6.65
    # ------------------------------------------------------------------
    def _trace_capture(self, cfg_alpha, n_steps):
        """Capture the WHOLE per-frame device graph: Block 1, the semantic projection, Block 2.

        It has to be all of it. Once a trace exists, any later device ALLOCATION may be corrupted
        when the trace runs -- ttnn warns, then it hangs, then the board needs `tt-smi -r` (6.64).
        Leaving Block 1 or semantic_code eager would allocate every frame, so nothing is left out.

        Both halves are computed unconditionally and the [END_AUDIO] masking stays on host, which
        is where 6.31/6.50 put it: the acoustic decode does not depend on the semantic argmax.
        """
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
            # cos/sin arrive INTERLEAVED and are resharded here: copy_host_to_device_tensor into a
            # sharded destination is a layout constraint not worth fighting, and the reshard is
            # inside the trace so it costs no dispatch.
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
        # POINT THE CAPTURE AT THE SLOT THE FIRST TRACED FRAME WILL OVERWRITE ANYWAY.
        # graph() runs twice here (warm-up + capture) and each run WRITES K/V through
        # paged_update_cache at whatever `pos` holds. Left at 0 that destroys the prefilled
        # prompt's position 0 and every later attention reads the wreckage -- which is exactly the
        # garbage the first gate produced (WER 1 -> 1320). Aimed at pos0 the writes land where the
        # first real frame writes moments later, so they are harmless.
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([pos0], dtype=torch.int32)), buf["pos"])
        graph()                                   # populate the program cache before capturing
        ttnn.synchronize_device(dev)
        bb.pos = pos0
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            lg, xr = graph()
        finally:
            # An exception escaping between begin and end capture wedges the card for every later
            # run, so the close is unconditional.
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        bb.pos = pos0
        self._tr = (tid, buf, lg, xr)

    def _trace_release(self):
        if self._tr is not None:
            ttnn.release_trace(self.device, self._tr[0])
            self._tr = None

    def _traced_frame(self, codes, cfg_alpha):
        """One frame through the trace. Only copies, execute and D2H -- nothing allocates."""
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
        # NOTES.md [pipe-05] -- FRAME 0 IS EAGER AND MUST COME BEFORE THE CAPTURE. Its hidden state
        # comes from prefill rather than from a Block 1 step, so it does not fit the traced graph;
        # and running it after the capture would allocate, which is the thing that wedges the card.
        codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
        traced = TRACE_REGION_SIZE > 0
        if traced:
            self._trace_capture(cfg_alpha, N_DECODING_STEPS)
        try:
            for i in range(max_frames):
                if int(codes[0, 0]) == END_AUDIO_ID:
                    if verbose:
                        print(f"[pipeline] [END_AUDIO] at frame {i} -- natural stop")
                    stopped = True
                    break
                frames.append(codes)
                if traced:
                    codes = self._traced_frame(codes[0], cfg_alpha)
                else:
                    h = self.backbone.step(
                        backbone.embed_frame(self.wb, codes[0])).reshape(1, 1, -1)
                    codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)
                if verbose and (i + 1) % 10 == 0:
                    el = time.perf_counter() - t0
                    print(f"[pipeline]   {i+1} frames ({(i+1)/FRAME_RATE:.1f}s audio) "
                          f"| {el/(i+1):.2f}s/frame")
        finally:
            # Released unconditionally: the next generate() starts with a prefill, which allocates,
            # and a live trace makes that unsafe.
            self._trace_release()
        if verbose and not stopped:
            print(f"[pipeline] hit max_frames={max_frames} without [END_AUDIO]")
        if not frames:
            raise RuntimeError("model emitted [END_AUDIO] on the first frame -- nothing to decode")
        return torch.cat(frames, dim=0), t_prefill, time.perf_counter() - t0

    @torch.no_grad()
    def decode(self, frames):
        """frames [T,37] -> waveform torch [1,1,T*1920] @ 24 kHz, via Block 3."""
        from models.experimental.voxtral_tts.reference.voxtral_codec_ref import strip_offset_and_trim

        return self.codec(strip_offset_and_trim(frames))
