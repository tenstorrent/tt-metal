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

FIDELITY, measured per block against the fp32 CPU reference:
    Block 1 prefill  PCC 0.969   (bf16 weight floor; see ttnn_voxtral_backbone's precision notes)
    Block 1 decode   PCC 0.981
    Block 2 velocity PCC 0.9999989, semantic codes EXACT, 73/74 frame codes exact on synthetic h
    Block 3          real speech PCC 0.999984, worst sample 1.16% of peak
The end-to-end question is not any of those PCCs -- it is how many of the 37 INTEGER codes per
frame differ from the reference, because that is what changes the audio. `compare_codes()` below
measures exactly that, and attributes any excess over Block 2's own 1-in-74 to Block 1.
"""

import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as backbone
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, END_AUDIO_ID
from models.experimental.voxtral_tts.tt.ttnn_voxtral_backbone import TtVoxtralBackbone
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA, TtVoxtralFlow

FRAME_RATE = 12.5


class TtVoxtralPipeline:
    """All three blocks on device. generate(embeds) -> frames; decode(frames) -> waveform."""

    def __init__(self, device, hf_dir=None, ckpt_path=DEFAULT_CKPT, max_seq_len=1024):
        self.device = device
        self.backbone = TtVoxtralBackbone(device, hf_dir=hf_dir, max_seq_len=max_seq_len)
        self.flow = TtVoxtralFlow(device, ckpt_path=ckpt_path)
        self.codec = TtVoxtralCodecDecoder(device, ckpt_path=ckpt_path)
        # embed_frame is a host gather, so it needs the backbone's audio embedding table.
        self.wb = backbone.load_backbone_state(ckpt_path)

    @torch.no_grad()
    def generate(self, embeds, max_frames=150, cfg_alpha=CFG_ALPHA, seed=0, verbose=True):
        """prompt embeds [1,P,3072] -> frames [T,37] int64 (offset applied, [END_AUDIO] excluded)."""
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        # prefill returns ALL positions; only the last one conditions the first frame, matching the
        # reference's IncrementalBackbone.prefill which returns x[:, -1:].
        h_all = self.backbone.prefill(embeds)
        h = h_all[:, -1:]                     # [1,1,3072]
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


@torch.no_grad()
def compare_codes(pipe, embeds, n_frames=8, cfg_alpha=CFG_ALPHA, seed=0):
    """THE test that matters: do device and reference emit the same INTEGER codes?

    TEACHER-FORCED: both loops are fed the REFERENCE's codes each step. That makes every frame an
    INDEPENDENT measurement of "given identical input, do they agree?". Feeding each loop its own
    codes instead (which is what real generation does) is useless for attribution: after the first
    semantic mismatch the two are generating different sequences, so later frames compare unrelated
    trajectories rather than measuring error. Measured that way, frame 0 agreed exactly and every
    later frame looked catastrophic -- an artefact, not a result.

    Reports the semantic code (a wrong one changes the audio outright) separately from the 36
    acoustic codes (each one of 21 FSQ levels, so off-by-one is a small perturbation).
    """
    from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref

    wf = fref.load_flow_state()
    ref_dec = backbone.IncrementalBackbone(pipe.wb)

    torch.manual_seed(seed)
    h_ref = ref_dec.prefill(embeds)
    h_dev = pipe.backbone.prefill(embeds)[:, -1:]

    sem_bad = ac_bad = total_ac = 0
    print(f"  {'frame':>6} {'sem ref/dev':>14} {'acoustic diffs':>15} {'max |delta|':>12}")
    for i in range(n_frames):
        torch.manual_seed(1000 + i)          # same noise draw for both, so only the model differs
        c_ref = fref.reference_frame(h_ref[:, 0], wf, cfg_alpha=cfg_alpha)
        torch.manual_seed(1000 + i)
        c_dev = pipe.flow(h_dev[:, 0], cfg_alpha=cfg_alpha)
        s_ref, s_dev = int(c_ref[0, 0]), int(c_dev[0, 0])
        d = (c_ref[0, 1:] != c_dev[0, 1:])
        n_d = int(d.sum())
        mx = int((c_ref[0, 1:] - c_dev[0, 1:]).abs().max())
        sem_bad += s_ref != s_dev
        ac_bad += n_d
        total_ac += 36
        flag = "" if s_ref == s_dev else "  <- SEMANTIC MISMATCH"
        print(f"  {i:>6} {s_ref:>6}/{s_dev:<7} {n_d:>10}/36 {mx:>12}{flag}")
        if s_ref == END_AUDIO_ID or s_dev == END_AUDIO_ID:
            print("      [END_AUDIO] reached")
            break
        # teacher forcing: BOTH advance on the reference's codes
        emb = backbone.embed_frame(pipe.wb, c_ref[0])
        h_ref = ref_dec.step(emb)
        h_dev = pipe.backbone.step(emb).reshape(1, 1, -1)
    print(f"  => semantic mismatches {sem_bad}, acoustic {ac_bad}/{total_ac} "
          f"({ac_bad/max(total_ac,1)*100:.1f}%)")
    return sem_bad, ac_bad, total_ac


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        pipe = TtVoxtralPipeline(dev)
        # Synthetic prompt embeddings: this checks the WIRING and the code agreement without
        # needing the tokenizer or a voice preset. The real-text path is voxtral_pipeline_ref's job
        # on host; swapping in build_inputs_embeds() is a one-liner once this is trusted.
        torch.manual_seed(0)
        embeds = torch.randn(1, 128, 3072) * 0.02
        print("=== device vs reference, INTEGER codes (the test that predicts audio) ===")
        compare_codes(pipe, embeds, n_frames=8)
        print()
        print("=== end-to-end: generate + decode to waveform ===")
        frames, t_pre, t_gen = pipe.generate(embeds, max_frames=12, verbose=True)
        wav = pipe.decode(frames)
        audio_s = frames.shape[0] / FRAME_RATE
        print(f"  frames {tuple(frames.shape)} -> waveform {tuple(wav.shape)} "
              f"({audio_s:.1f}s audio)")
        print(f"  prefill {t_pre:.2f}s | generate {t_gen:.2f}s "
              f"({t_gen/max(frames.shape[0],1):.2f}s/frame) | RTF {(t_pre+t_gen)/audio_s:.2f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
