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
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA, TtVoxtralFlow

FRAME_RATE = 12.5

# L1 scratch every caller needs: the codec's convs fail with "bank size is 0 B" without it.
L1_SMALL_SIZE = 65536


def open_device(device_id=0):
    """Open a device configured the way every entry point here needs it."""
    return ttnn.open_device(device_id=device_id, l1_small_size=L1_SMALL_SIZE)


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
        for i in range(max_frames):
            codes = self.flow(h[:, 0], cfg_alpha=cfg_alpha)      # [1,37]
            if int(codes[0, 0]) == END_AUDIO_ID:
                if verbose:
                    print(f"[pipeline] [END_AUDIO] at frame {i} -- natural stop")
                stopped = True
                break
            frames.append(codes)
            h = self.backbone.step(backbone.embed_frame(self.wb, codes[0])).reshape(1, 1, -1)
            if verbose and (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"[pipeline]   {i+1} frames ({(i+1)/FRAME_RATE:.1f}s audio) "
                      f"| {el/(i+1):.2f}s/frame")
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
