# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Voxtral-TTS on device: text ids + voice preset -> 24 kHz waveform.

Mirrors reference/voxtral_pipeline_ref.py's `generate`, with all three blocks on TTNN:

    ids + voice --embed--> Block 1 prefill --> h ─┐
                                                 ├─> Block 2 -> 37 codes ─┬─> embed_frame ─┐
                                                 │                        │                │
                                        Block 1 step <────────────────────┴────────────────┘
                                                 │
                              accumulated frames ┴─> Block 3 -> waveform @ 24 kHz

Stops on [END_AUDIO] as the semantic code, or at max_frames. Frame rate is 12.5 Hz, so 150 frames
is 12 s of audio.

WHAT RUNS WHERE. Blocks 1-3 all run on device. Three host steps remain, each deliberate:
  * the tekken tokenizer and prompt assembly (upstream of everything; see voxtral_tokenizer_ref)
  * `embed_frame` -- a 37-way embedding gather + sum, per frame. ttnn.embedding needs a bf16 table
    and these tables are large-valued, the same reasoning as the codec's semantic gather.
  * Block 2's semantic argmax and FSQ quantise (see ttnn_voxtral_flow's docstring)

FIDELITY, measured per block against the fp32 CPU reference, on real prompts -- never random ones,
which are a pessimistic proxy and cost a lot of time once (STATUS.md trap #12):
    Block 1 prefill  PCC 0.999881  (last position -- the only one Block 2 consumes)
    Block 1 decode   PCC 0.99991
    Block 2 velocity PCC 0.9999989, semantic codes EXACT, 73/74 frame codes exact on synthetic h
    Block 3          real speech PCC 0.999984, worst sample 1.16% of peak
The end-to-end question is not any of those PCCs -- it is the WER of the decoded audio, because
that is what a listener gets. Full 15-case fixture, free-running: natural-text WER 0.88% over 341
words, 15/15 natural [END_AUDIO]. `compare_codes()` below is the finer-grained probe.

PERFORMANCE, steady state on one N150, case 2 (448 frames, first 30 discarded):

    Block 1 decode      34.9 ms/frame   45.0%
    Block 2 flow        42.5 ms/frame   54.8%
    host embed_frame     0.2 ms/frame    0.2%
    TOTAL               77.6 ms/frame   = 0.0776 s/frame
    prefill ~1.1 s once; Block 3 codec ~2.5 s once (~7% of the audio duration)
    RTF 0.969 steady state, 1.115 end to end   (RTF = generation / audio, lower is better)

A frame is 80 ms of audio at 12.5 Hz, so RTF = ms_per_frame / 80.

BLOCK 2 IS THE LARGER HALF and is where the next work belongs: 35 of its 42.5 ms is 7 SEQUENTIAL
Euler steps, each a 3-layer transformer over 3 tokens, so every matmul does 32 tile rows of work
for 6 useful ones. Block 1 has had its structural wins (the GQA row fold, mixed-precision weights,
the decode-native head layout) and its linears run at 194 GB/s -- the measured ceiling for a plain
interleaved matmul on this part, confirmed by hand-tuned program configs coming out SLOWER.

The one structural idea left is throughput, not latency: a 3-token sequence wastes 26 of 32 tile
rows and nothing in ONE utterance can fill them (Euler steps are sequential, frames are
autoregressive), but CONCURRENT REQUESTS fit exactly.

No comparison against ign/voxtral_p150_qb2 is available: their published figures are Blackhole
P150 on a larger 4B/32-layer variant, their 0.10 s/frame is a test threshold rather than an
achievement, and their RTF 0.7 is a four-card mesh. Running their code here was tried and their
tt-metal is ~1100 commits behind ours, so it does not load.
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


# A HANG THAT SHAPED THE SHIPPED CONFIG, recorded because the workaround is gone and the trigger
# is still live in ttnn. Multi-utterance runs used to hang inside Block 3's decode and take the
# card down with it (recovery needs a tt-smi board reset). It required FIVE things at once:
#
#     all-BFP8 weights in Block 1    <- the one we control; the mixed default avoids it
#     Block 2 in the loop            raw Block 1 steps alone: completes
#     Block 3 on device              codec on CPU: completes
#     >= 2 distinct codec buckets    one bucket everywhere: completes
#     a generation BETWEEN two same-bucket decodes
#
# Minimal repro under all-BFP8, ~90 s: short gen + 128-bucket decode, long gen, 512-bucket decode,
# long gen, 512-bucket decode -- the last is a pure cache HIT and hangs. tt_transformers never saw
# it because it uses the same mixed precision we now do.
#
# Measured and ELIMINATED, so none of these get retried: memory (flat, 8 GB free at the hang);
# program-cache COUNT (576 entries over 4 buckets completes, while we died at 310-341 and
# tt_transformers lived at 329); Block 3 length/content; a Block 1 leak (1400 steps clean); and
# every distinctive Block 1 op. The underlying ttnn failure -- a silent hang rather than an
# error -- is unreported upstream and still unexplained.


class TtVoxtralPipeline:
    """All three blocks on device. generate(embeds) -> frames; decode(frames) -> waveform."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, max_seq_len=1024):
        self.device = device
        # embed_frame is a host gather, so it needs the backbone's audio embedding table. Load it
        # BEFORE the backbone and hand the same dict over: our Block 1 would otherwise load its own
        # ~13 GB fp32 copy of the same file.
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
        # Only the last position conditions the first frame, matching the reference's
        # IncrementalBackbone.prefill which returns x[:, -1:]. Both backbones expose this as
        # prefill_last so the loop below does not care which one is running.
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
