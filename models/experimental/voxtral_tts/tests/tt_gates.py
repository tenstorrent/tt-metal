# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every on-device gate for the TTNN blocks, in one place.

These are validation harnesses, not model code, and they used to live as `main()` functions inside
the four `tt/` modules -- which made those files harder to read for someone trying to understand
or optimize the model. Nothing here is imported by the model; the dependency only runs this way.

Each gate compares a block against the fp32 CPU reference and prints, rather than asserting: they
are for a human deciding whether a change is acceptable, not for CI. The pass/fail suite is
`tests/test_*.py`; this is the thing you run by hand after touching a block.

    python models/experimental/voxtral_tts/tests/tt_gates.py --gate wiring
    python models/experimental/voxtral_tts/tests/tt_gates.py --gate prefill26 --cases 0,2
    python models/experimental/voxtral_tts/tests/tt_gates.py --gate decode --cases 0 --steps 8
    python models/experimental/voxtral_tts/tests/tt_gates.py --gate flow
    python models/experimental/voxtral_tts/tests/tt_gates.py --gate codec
    python models/experimental/voxtral_tts/tests/tt_gates.py --gate codes      # blocks 1+2 e2e

ALWAYS GATE ON REAL PROMPTS, never random activations. Random embeddings are off-manifold and
reported PCC 0.892 where real prompts gave 0.9994 on the same weights -- STATUS.md trap #12, and
the most expensive measurement mistake in this port. `fixture_embeds` builds the real thing.
"""

import argparse
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM,
    HEAD_DIM,
    N_LAYERS,
    ROPE_THETA,
)
from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA, TtVoxtralFlow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (
    FRAME_RATE, TtVoxtralPipeline, open_device)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fixture_embeds(case_idx, w):
    """Fixture case -> real prompt embeds [1,P,3072], exactly as the pipeline builds them.

    REAL PROMPTS ARE NOT OPTIONAL for an accuracy number (STATUS.md trap #12): random embeddings
    are off-manifold and reported PCC 0.892 where these give 0.9994 on the same weights.
    """
    import os

    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref

    here = HERE
    case = json.load(open(os.path.join(here, "tests", "prompt_fixture.json")))["cases"][case_idx]
    ids = torch.tensor(case["ids"], dtype=torch.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(case["voice"]), w), case


def gate_wiring(dev, ref):
    """Increment 2: ONE layer against the reference. A RoPE convention error shows up here."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        causal_bias, pcc, rope_cis)

    S = 128
    gen = TtVoxtralGPT(dev, n_layers=1)
    w = ref.load_backbone_state()
    torch.manual_seed(0)
    x = torch.randn(1, S, DIM) * 0.02
    exp = ref._layer(x, w, "layers.0.", rope_cis(S, HEAD_DIM, ROPE_THETA),
                     causal_bias(S, torch.float32))
    got = gen.prefill(x, apply_final_norm=False)
    print(f"  [1 layer prefill] PCC {pcc(got, exp):.8f}  "
          f"maxabs {(got - exp).abs().max():.3e}")
    print("  NOTE: random inputs are a pessimistic proxy (trap #12) -- this gate is for WIRING")
    print("  and the RoPE convention. Judge accuracy on real prompts at 26 layers.")


def gate_prefill26(dev, ref, cases, n_layers=N_LAYERS):
    """Increment 3: the full stack, prefill, on REAL prompts vs `reference_forward`.

    Reports the LAST position separately because that is the only one Block 2 consumes; the
    all-positions number is there to catch a bug that only touches part of the sequence.
    """
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    print(f"  loading fp32 reference weights (~13 GB) -- shared with the device upload")
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w)
    print(f"  {n_layers} layers on device\n")
    print(f"  {'case':>4} {'voice':>16} {'P':>5} {'PCC all':>12} {'PCC last':>12} "
          f"{'worst last':>11} {'worst pos (which)':>20} {'device':>9}")
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        exp = ref.reference_forward(embeds, w, n_layers=n_layers)
        t0 = time.perf_counter()
        got = gen.prefill(embeds)
        dt = time.perf_counter() - t0
        el, xl = got[:, -1:], exp[:, -1:]
        worst = (el - xl).abs().max().item() / xl.abs().max().item() * 100
        # Per position, because a pooled PCC over the whole prompt is dominated by whichever
        # positions have the largest magnitude and hides which ones are actually weak.
        per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
        wi = min(range(P), key=lambda i: per[i])
        print(f"  {ci:>4} {case['voice']:>16} {P:>5} {pcc(got, exp):>12.6f} "
              f"{pcc(el, xl):>12.6f} {worst:>10.2f}% {per[wi]:>13.6f} (@{wi:>4}) {dt:>8.2f}s")
    print("\n  reference for comparison, same metric on the LAST position (STATUS.md, Block 1):")
    print("    tt_transformers, FF1_FF3 BFP8: 0.999564 at P=200, 0.999579 at P=312")


def gate_decode(dev, ref, cases, n_steps=8, n_layers=N_LAYERS):
    """Increment 4: on-device KV cache + decode steps vs `IncrementalBackbone.step()`.

    TEACHER-FORCED on REAL frames (`tests/real_frames_fixture.pt`, genuine Block 1+2 output): both
    sides advance on the SAME embedding every step, so each step is an independent measurement.
    Feeding each its own codes instead compares two diverging trajectories and tells you nothing
    (the same trap `ttnn_voxtral_pipeline.compare_codes` documents).
    """
    import os

    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    here = HERE
    frames = torch.load(os.path.join(here, "tests", "real_frames_fixture.pt")).long()
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w, max_seq_len=1024)
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        print(f"\n  case {ci} ({case['voice']}, P={P}), {n_steps} real frames teacher-forced")
        print(f"  {'step':>6} {'pos':>5} {'PCC':>11} {'worst':>8} {'ms':>8}")
        inc = ref.IncrementalBackbone(w, n_layers=n_layers)
        h_ref = inc.prefill(embeds)
        gen.reset()
        h_dev = gen.prefill(embeds, last_only=True)
        assert gen.pos == inc.pos == P, f"position mismatch after prefill: {gen.pos} vs {inc.pos}"
        worst = (h_dev - h_ref).abs().max().item() / h_ref.abs().max().item() * 100
        print(f"  {'prefill':>6} {P:>5} {pcc(h_dev, h_ref):>11.6f} {worst:>7.2f}%")
        for t in range(min(n_steps, frames.shape[0])):
            emb = ref.embed_frame(w, frames[t])
            h_ref = inc.step(emb)
            t0 = time.perf_counter()
            h_dev = gen.step(emb)
            dt = (time.perf_counter() - t0) * 1e3
            worst = (h_dev - h_ref).abs().max().item() / h_ref.abs().max().item() * 100
            print(f"  {t:>6} {gen.pos - 1:>5} {pcc(h_dev, h_ref):>11.6f} {worst:>7.2f}% {dt:>7.1f}")
    print("\n  reference for comparison (STATUS.md, Block 1): tt_transformers decode PCC 0.981,")
    print("  48 ms/step. The 0.981 is unexplained there; this path should not reproduce it.")


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
    ref_dec = bref.IncrementalBackbone(pipe.wb)

    torch.manual_seed(seed)
    h_ref = ref_dec.prefill(embeds)
    h_dev = pipe.backbone.prefill_last(embeds)

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
        emb = bref.embed_frame(pipe.wb, c_ref[0])
        h_ref = ref_dec.step(emb)
        h_dev = pipe.backbone.step(emb).reshape(1, 1, -1)
    print(f"  => semantic mismatches {sem_bad}, acoustic {ac_bad}/{total_ac} "
          f"({ac_bad/max(total_ac,1)*100:.1f}%)")
    return sem_bad, ac_bad, total_ac

def gate_codes():
    dev = open_device()
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


def gate_flow():
    """Compare against the CPU reference. The output is INTEGER codes, so equality is exact."""
    from models.experimental.voxtral_tts.reference import voxtral_flow_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    dev = open_device()
    try:
        gen = TtVoxtralFlow(dev)
        w = ref.load_flow_state()
        h, x_0 = ref.make_synthetic_inputs(batch=2, seed=0)

        # 1) one velocity evaluation -- the unit a trace would capture
        t_emb = ref.time_embedding(torch.tensor(0.375).view(1, 1).repeat(2, 1),
                                   w["time_embedding.inv_freq"])
        exp_v = ref.predict_velocity(x_0, h, t_emb, w)
        got_v = gen._predict_velocity(x_0, h, t_emb)
        print(f"  [velocity      ] PCC {pcc(got_v, exp_v):.8f}  maxabs {(got_v-exp_v).abs().max():.3e}")

        # 2) semantic code -- must match EXACTLY, it is an index
        exp_s, got_s = ref.semantic_code(h, w), gen.semantic_code(h)
        print(f"  [semantic code ] exact match: {bool((exp_s==got_s).all())}  {exp_s.flatten().tolist()}")

        # 3) full frame, deterministic x_0 -- 37 INTEGER codes, so exact or not
        exp_f = ref.reference_frame(h, w, x_0=x_0)
        got_f = gen(h, x_0=x_0)
        n_diff = int((exp_f != got_f).sum())
        print(f"  [full frame    ] {'IDENTICAL' if n_diff==0 else f'{n_diff} of {exp_f.numel()} codes differ'}")
        if n_diff:
            print(f"      ref  {exp_f[0, :10].tolist()}")
            print(f"      got  {got_f[0, :10].tolist()}")
    finally:
        ttnn.close_device(dev)


def gate_codec():

    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    device = open_device()
    try:
        gen = TtVoxtralCodecDecoder(device)
        w = ref.load_codec_state()
        for n_frames in (8, 24):
            codes = ref.make_synthetic_codes(n_frames)
            exp_lat = ref.quantizer_decode(codes, w)
            exp_wav = ref.reference_decode(codes, w)
            got_wav, stages = gen(codes, return_stages=True)

            tag = f"[codec T={n_frames}]"
            got_lat = TtVoxtralCodecDecoder._chw(gen.quantizer_decode(codes))
            print(f"\n{tag} {'quantizer':22s} PCC {pcc(got_lat, exp_lat):.6f}")
            x = ref.causal_conv1d(exp_lat, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
            print(f"{tag} {'after_input_conv':22s} PCC {pcc(stages['after_input_conv'], x):.6f}")
            for stage, tf_i in enumerate(ref.DEC_TF_BLOCKS):
                x = ref.codec_transformer(x.permute(0, 2, 1), w, tf_i, 2,
                                          ref.decoder_window_sizes()[stage]).permute(0, 2, 1)
                name = f"after_tf{tf_i} (win {ref.decoder_window_sizes()[stage]})"
                print(f"{tag} {name:22s} PCC {pcc(stages[f'after_tf{tf_i}'], x):.6f}")
                if stage < 3:
                    ci = ref.DEC_CONV_BLOCKS[stage + 1]
                    x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
                    print(f"{tag} {f'after_up{ci}':22s} PCC {pcc(stages[f'after_up{ci}'], x):.6f}")
            print(f"{tag} {'WAVEFORM':22s} PCC {pcc(got_wav, exp_wav):.6f}  "
                  f"shapes {tuple(got_wav.shape)} vs {tuple(exp_wav.shape)}")
            # the staged run above uses return_stages=True, which BYPASSES bucketing -- so also
            # exercise the DEFAULT path (bucketed) that real callers get.
            plain = gen(codes)
            print(f"{tag} {'bucketed (default path)':22s} PCC {pcc(plain, exp_wav):.6f}  "
                  f"shape {tuple(plain.shape)}")
            t0 = time.perf_counter()
            gen(codes)
            print(f"{tag} warm {(time.perf_counter() - t0) * 1000:.1f} ms")
    finally:
        ttnn.close_device(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True,
                    choices=("wiring", "prefill26", "decode", "flow", "codec", "codes"),
                    help="wiring = one Block 1 layer (fast, catches a RoPE convention error); "
                         "prefill26 = 26 layers on real prompts (~13 GB host RAM); "
                         "decode = KV cache + steps vs IncrementalBackbone; "
                         "flow = Block 2 vs its reference; codec = Block 3 vs its reference; "
                         "codes = blocks 1+2 end to end, integer-code agreement")
    ap.add_argument("--cases", default="0,2", help="prompt_fixture.json indices")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--steps", type=int, default=8, help="decode steps for --gate decode")
    args = ap.parse_args()
    cases = [int(c) for c in args.cases.split(",")]

    if args.gate == "flow":
        return gate_flow()
    if args.gate == "codec":
        return gate_codec()
    if args.gate == "codes":
        return gate_codes()

    dev = open_device()
    try:
        if args.gate == "wiring":
            gate_wiring(dev, bref)
        elif args.gate == "prefill26":
            gate_prefill26(dev, bref, cases, args.layers)
        else:
            gate_decode(dev, bref, cases, args.steps, args.layers)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
