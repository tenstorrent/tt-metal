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
  * Block 2's FSQ quantise -- clamp/scale/round on [B,36]; 36 values is not worth a dispatch.
    (Block 2's semantic argmax USED to be here too. It is on device now, in fp32 -- worth 1.49
    ms/frame; see ttnn_voxtral_flow.semantic_dev for why fp32 and not bf16.)

FIDELITY, measured per block against the fp32 CPU reference, on real prompts -- never random ones,
which are a pessimistic proxy and cost a lot of time once (STATUS.md trap #12):
    Block 1 prefill  PCC 0.999924 / 0.999883  (last position -- all Block 2 consumes)
    Block 1 decode   PCC 0.99985+, mean worst-sample 0.86% over 44 teacher-forced frames
    Block 2 velocity PCC 0.99998522, semantic codes EXACT, 71/74 frame codes exact on synthetic h
    Block 3          real speech PCC 0.999984, worst sample 1.16% of peak

END TO END, the number to quote is **long-form WER: 1 wrong word in 298**, plus 15/15 natural
[END_AUDIO] and the voice-identity check. NOT the 340-word natural-text headline: that bucket
includes 42 words of 3-to-6-word clips where one Whisper disagreement is worth 17-50%, and the
SAME CODE at seeds 0/1/2 spans 0.88-2.06% on it. See STATUS.md 6.7 before quoting any WER, and use
the teacher-forced gates in tests/tt_gates.py to judge a numerical change.

PERFORMANCE, steady state on one N150, long-form cases:

    Block 1 decode      ~25.7 ms/frame   ~51%
    Block 2 flow        ~23.0 ms/frame   ~46%
    host embed_frame      0.2 ms/frame    0.4%
    TOTAL               49.0-52.5 ms/frame, mean 50.4 over the 15-case fixture
    prefill 0.1-1.5 s once; Block 3 codec 97 ms warm, i.e. 0.4% -- but SECONDS the first
    time a bucket length is seen, which is COMPILE cost, not compute (STATUS.md 6.10)
    RTF 0.62-0.71 on 14 of 15 cases   (RTF = generation / audio, lower is better)

The 15th is case 0 at RTF 1.89, and that is COLD-START, not a slow case: it pays the first codec
bucket's kernel compiles and the first prefill shape. Every later case with the same shapes runs
at 0.74-0.80. Quote the steady-state number, and re-run a case twice if it looks anomalous.

A frame is 80 ms of audio at 12.5 Hz, so RTF = ms_per_frame / 80.

WHERE THE REMAINING TIME IS. Both blocks now stream every weight matmul at the 194 GB/s DRAM
ceiling, so neither has a byte or a layout trick left; each module's docstring carries the per-line
map. What is left is different in each:
  * Block 1 is at its floor except for w2, the last bf16 weight and the pinned trigger of the hang
    documented below. ~21 of its 25.7 ms is pure weight streaming at the ceiling, and everything
    that is not a matmul now totals under 5 ms.
  * Block 2 sits ~1.6x above its 13.4 ms weight-read floor, and that gap is DEVICE-side per-kernel
    cost -- proven by tracing, which removes host dispatch and changes nothing (6.6). Fewer ops
    does not help; bigger kernels and L1-resident operands do. Its worst single line is
    nlp_create_qkv_heads, whose ~97 us is a FIXED cost (same at 10.7x the data).
  * Block 3 is NOT a target: 97 ms warm against a ~26 s generation, i.e. 0.4%. Its
    seconds-scale appearance in a fresh run is first-call kernel compilation per bucket,
    not compute. An earlier version of this file called it ~9% of wall; that was derived
    by subtraction and conflated the two. STATUS.md 6.10.

The one structural idea left is throughput, not latency: a 3-token sequence wastes 26 of 32 tile
rows and nothing in ONE utterance can fill them (Euler steps are sequential, frames are
autoregressive), but CONCURRENT REQUESTS fit exactly.

Against ign/voxtral_p150_qb2: their code cannot run on our tree, so their tt-metal was built
separately and measured here at 598 ms/frame. That is their Blackhole-targeted code on our
Wormhole card, which answers "can we adopt theirs" (no) and NOT "is their P150 slow" (unmeasured).
STATUS.md 6.5 has the setup and the two findings it did corroborate.
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
