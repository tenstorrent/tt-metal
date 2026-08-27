# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every on-device gate for the TTNN blocks, in one place.

These are validation harnesses, not model code, and they used to live as `main()` functions inside
the four `tt/` modules -- which made those files harder to read for someone trying to understand
or optimize the model. Nothing here is imported by the model; the dependency only runs this way.

ONE implementation of every on-device comparison, with three consumers:

  - `tests/test_*_pcc.py` import the `compare_*` primitives and ASSERT on them (that is the
    pass/fail suite, and it is what CI runs);
  - `scripts/quality_report.py` runs `--gate X --json` in a subprocess and reads the returned
    metrics as JSON -- no prose scraping, and each gate keeps its own device and timeout;
  - a human runs `--gate X` for the printed tables when triaging by hand.

The metric arithmetic lives in `compare_hidden` / `compare_codes_frame` so those three cannot
drift to three different notions of the same number. Renamed from `tt_gates.py`, whose docstring
claimed it was "not for CI" long after quality_report had made it exactly that.

    python models/experimental/voxtral_tts/tests/gates.py --gate wiring
    python models/experimental/voxtral_tts/tests/gates.py --gate prefill26   # all 15 prompts
    python models/experimental/voxtral_tts/tests/gates.py --gate decode      # all 15 prompts
    python models/experimental/voxtral_tts/tests/gates.py --gate decode --cases 0,2 --verbose
    python models/experimental/voxtral_tts/tests/gates.py --gate flow
    python models/experimental/voxtral_tts/tests/gates.py --gate codec
    python models/experimental/voxtral_tts/tests/gates.py --gate codes      # blocks 1+2 e2e

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


def compare_hidden(got, exp):
    """Device vs reference hidden state -> {"pcc", "worst_pct"}.

    ALWAYS both. PCC is a correlation and hides outliers -- it sits at 0.9998 while individual
    samples are badly wrong, and for audio the outliers are what you hear (STATUS 5.9). The
    worst-sample bound is the gate that matters."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc as _pcc

    return {
        "pcc": _pcc(got, exp),
        "worst_pct": (got - exp).abs().max().item() / max(exp.abs().max().item(), 1e-9) * 100,
    }


def compare_codes_frame(c_ref, c_dev):
    """One frame of integer codes -> {"sem_ok", "n_diff", "max_delta", "deltas"}.

    Semantic code (index 0) is reported separately from the 36 acoustic codes: a wrong semantic
    code changes the audio outright, while an acoustic code is one of 21 FSQ levels, so off-by-one
    is the smallest representable difference (STATUS 6.54 -- the bare count gets misread)."""
    d = (c_ref[0, 1:].long() - c_dev[0, 1:].long()).abs()
    nz = d[d != 0].tolist()
    return {
        "sem_ok": int(c_ref[0, 0]) == int(c_dev[0, 0]),
        "n_diff": int((d != 0).sum()),
        "max_delta": int(d.max()) if d.numel() else 0,
        "deltas": [int(v) for v in nz],
    }


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
    m = compare_hidden(got, exp)
    print(f"  [1 layer prefill] PCC {m['pcc']:.8f}  "
          f"maxabs {(got - exp).abs().max():.3e}")
    print("  NOTE: random inputs are a pessimistic proxy (trap #12) -- this gate is for WIRING")
    print("  and the RoPE convention. Judge accuracy on real prompts at 26 layers.")
    return {"wiring_pcc": m["pcc"]}


def gate_prefill26(dev, ref, cases, n_layers=N_LAYERS):
    """Increment 3: the full stack, prefill, on REAL prompts vs `reference_forward`.

    Reports the LAST position separately because that is the only one Block 2 consumes; the
    all-positions number is there to catch a bug that only touches part of the sequence.
    """
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    print("  loading fp32 reference weights (~13 GB) -- shared with the device upload")
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w)
    print(f"  {n_layers} layers on device\n")
    print(f"  {'case':>4} {'voice':>16} {'P':>5} {'PCC all':>12} {'PCC last':>12} "
          f"{'worst last':>11} {'worst pos (which)':>20} {'device':>9}")
    last_pccs = []
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        exp = ref.reference_forward(embeds, w, n_layers=n_layers)
        t0 = time.perf_counter()
        got = gen.prefill(embeds)
        dt = time.perf_counter() - t0
        el, xl = got[:, -1:], exp[:, -1:]
        m_last = compare_hidden(el, xl)
        last_pccs.append(m_last["pcc"])
        worst = m_last["worst_pct"]
        # Per position, because a pooled PCC over the whole prompt is dominated by whichever
        # positions have the largest magnitude and hides which ones are actually weak.
        per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
        wi = min(range(P), key=lambda i: per[i])
        print(f"  {ci:>4} {case['voice']:>16} {P:>5} {pcc(got, exp):>12.6f} "
              f"{pcc(el, xl):>12.6f} {worst:>10.2f}% {per[wi]:>13.6f} (@{wi:>4}) {dt:>8.2f}s")
    print("\n  reference for comparison, same metric on the LAST position (STATUS.md, Block 1):")
    print("    tt_transformers, FF1_FF3 BFP8: 0.999564 at P=200, 0.999579 at P=312")
    # the MIN across cases, matching what the quality report has always recorded
    return {"prefill_pcc_last": min(last_pccs) if last_pccs else None,
            "prefill_n_cases": len(last_pccs) or None}


def _p90(v):
    """The 90th percentile by nearest-rank. Deliberately not numpy's interpolating default, so a
    recorded p90 is always an observed sample and two runs of the same config give the same value."""
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]


def gate_decode(dev, ref, cases, n_steps=8, n_layers=N_LAYERS, verbose=False):
    """Increment 4: on-device KV cache + decode steps vs `IncrementalBackbone.step()`.

    TEACHER-FORCED on REAL frames (`tests/real_frames_fixture.pt`, genuine Block 1+2 output): both
    sides advance on the SAME embedding every step, so each step is an independent measurement.
    Feeding each its own codes instead compares two diverging trajectories and tells you nothing
    (the same trap `ttnn_voxtral_pipeline.compare_codes` documents).

    HOW TO READ THE OUTPUT, because this gate was misread twice (STATUS.md 6.15):

    Its PROMPT-TO-PROMPT SPREAD is 0.45 pp on mean worst-sample and 0.96 pp on p90 -- LARGER than
    any change ever gated with it, w2's BFP8 drop (0.10 pp) included. So:
      - It resolves PAIRED A/B: same prompts, same session, one thing changed. It is deterministic,
        and a repeat run reproduces bit-identically, so a difference there is real at 0.01 pp.
      - It does NOT support absolute levels, comparison against a number recorded in another
        session, or generalising an effect measured on one prompt pair to other prompts.
    That is why the summary prints the case list and the per-case spread next to every aggregate:
    an aggregate here is meaningless without knowing which prompts produced it. STATUS.md 6.8
    reported a 2-prompt pair to 0.01 pp WITHOUT recording which two, and its levels can no longer
    be reproduced by any pair of the 15.

    Per-step rows are off by default (--verbose) -- they are for debugging a specific step, and
    printing 330 of them buries the summary that should actually be read.
    """
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    here = HERE
    frames = torch.load(os.path.join(here, "tests", "real_frames_fixture.pt")).long()
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w, max_seq_len=1024)
    per_case, pooled_ws, pooled_pcc, pooled_ms = {}, [], [], []
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        print(f"\n  case {ci} ({case['voice']}, P={P}), {n_steps} real frames teacher-forced",
              flush=True)
        if verbose:
            print(f"  {'step':>6} {'pos':>5} {'PCC':>11} {'worst':>8} {'ms':>8}")
        inc = ref.IncrementalBackbone(w, n_layers=n_layers)
        h_ref = inc.prefill(embeds)
        gen.reset()
        h_dev = gen.prefill(embeds, last_only=True)
        assert gen.pos == inc.pos == P, f"position mismatch after prefill: {gen.pos} vs {inc.pos}"
        pre_ws = (h_dev - h_ref).abs().max().item() / h_ref.abs().max().item() * 100
        pre_pcc = pcc(h_dev, h_ref)
        if verbose:
            print(f"  {'prefill':>6} {P:>5} {pre_pcc:>11.6f} {pre_ws:>7.2f}%")
        ws, pc, ms = [], [], []
        for t in range(min(n_steps, frames.shape[0])):
            emb = ref.embed_frame(w, frames[t])
            h_ref = inc.step(emb)
            t0 = time.perf_counter()
            h_dev = gen.step(emb)
            dt = (time.perf_counter() - t0) * 1e3
            _m = compare_hidden(h_dev, h_ref)
            worst = _m["worst_pct"]
            ws.append(worst); pc.append(_m["pcc"]); ms.append(dt)
            if verbose:
                print(f"  {t:>6} {gen.pos - 1:>5} {pc[-1]:>11.6f} {worst:>7.2f}% {dt:>7.1f}")
        per_case[ci] = dict(voice=case["voice"], P=P, ws=ws, pcc=pc,
                            pre_ws=pre_ws, pre_pcc=pre_pcc)
        pooled_ws += ws; pooled_pcc += pc; pooled_ms += ms
        print(f"    mean {sum(ws)/len(ws):.2f}%  p90 {_p90(ws):.2f}%  max {max(ws):.2f}%  "
              f"min PCC {min(pc):.6f}   (prefill {pre_ws:.2f}%, PCC {pre_pcc:.6f})", flush=True)

    n = len(pooled_ws)
    means = [sum(d["ws"]) / len(d["ws"]) for d in per_case.values()]
    p90s = [_p90(d["ws"]) for d in per_case.values()]
    print(f"\n  {'=' * 78}")
    print(f"  DECODE SUMMARY -- cases {','.join(str(c) for c in cases)}  "
          f"({len(cases)} prompts x {n // max(len(cases), 1)} frames = {n} frames)")
    print(f"  {'=' * 78}")
    print(f"  {'pooled over all frames':<34} mean {sum(pooled_ws)/n:5.2f}%   "
          f"p90 {_p90(pooled_ws):5.2f}%   max {max(pooled_ws):5.2f}%   min PCC {min(pooled_pcc):.6f}")
    print(f"  {'per-case mean, min..max':<34} {min(means):5.2f}% .. {max(means):5.2f}%   "
          f"(spread {max(means)-min(means):.2f} pp)")
    print(f"  {'per-case p90,  min..max':<34} {min(p90s):5.2f}% .. {max(p90s):5.2f}%   "
          f"(spread {max(p90s)-min(p90s):.2f} pp)")
    print(f"  {'prefill worst-sample, min..max':<34} "
          f"{min(d['pre_ws'] for d in per_case.values()):5.2f}% .. "
          f"{max(d['pre_ws'] for d in per_case.values()):5.2f}%")
    print(f"\n  QUOTE THIS ONLY WITH THE CASE LIST ABOVE. The spread lines are why: on the full 15")
    print(f"  the per-case mean ranges ~0.45 pp, so an aggregate over a DIFFERENT prompt set is not")
    print(f"  comparable. Valid use is a paired A/B -- same cases, same session, one change.")
    print(f"  Deterministic: a repeat of the same config reproduces these bit-identically.")
    print(f"  IGNORE THE ms COLUMN. It ran {sum(pooled_ms)/n:.1f} ms/step here against ~23 ms in the")
    print(f"  real pipeline, because a 3.4B fp32 CPU reference step runs between device steps and")
    print(f"  starves host dispatch. It has read BFP8 as SLOWER than bf16. Use the pipeline for perf.")
    print(f"  tt_transformers, for comparison: decode PCC 0.981.")
    return {"decode_mean_pp": sum(pooled_ws) / n, "decode_p90_pp": _p90(pooled_ws),
            "decode_min_pcc": min(pooled_pcc)}


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

    from collections import Counter
    sem_bad = ac_bad = total_ac = 0
    deltas = Counter()
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
        for _v in (c_ref[0, 1:] - c_dev[0, 1:]).abs().tolist():
            if _v:
                deltas[int(_v)] += 1
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
    # STATUS.md 6.54 -- the count alone gets misread. |delta|=1 on a 21-level FSQ axis is the
    # smallest difference representable, so print the distribution rather than just the count.
    if deltas:
        off1 = deltas.get(1, 0)
        print(f"     |delta| histogram { {k: deltas[k] for k in sorted(deltas)} }   "
              f"off-by-one {off1}/{ac_bad} ({off1/max(ac_bad,1)*100:.0f}%)")
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
        print("=== device vs reference, INTEGER codes -- SYNTHETIC embeddings ===")
        print("  NOTE (STATUS.md 6.54): random embeddings are a PESSIMISTIC proxy, the same trap")
        print("  gate_wiring warns about -- this reads ~6x worse than real text. The cause is")
        print("  BLOCK 1, not Block 2 or FSQ: off-manifold, PCC(h_dev,h_ref) is 0.9865 against")
        print("  0.9999 on real prompts (15.6% vs 0.7% relative), and that error reaches the")
        print("  codes through Block 2. The fp32 reference flips NOTHING on either input.")
        print("  It is also NON-MONOTONIC in precision (6.55): bf16 FF weights make this number")
        print("  WORSE. Never rank a config on it. Judge accuracy on the real-prompt block below.")
        _synth_sem, _synth_bad, _synth_tot = compare_codes(pipe, embeds, n_frames=8)

        print()
        print("=== the same comparison on REAL prompts -- THIS is the accuracy number ===")
        tot_b = tot_n = 0
        for _ci in (0, 2, 3):
            _e, _c = fixture_embeds(_ci, pipe.wb)
            print(f"  -- case {_ci} ({_c['voice']}, P={_e.shape[1]})")
            _s, _b, _n = compare_codes(pipe, _e, n_frames=8)
            tot_b += _b
            tot_n += _n
        print(f"  ==> REAL-PROMPT TOTAL {tot_b}/{tot_n} ({tot_b/max(tot_n,1)*100:.1f}%)")
        print()
        print("=== end-to-end: generate + decode to waveform ===")
        frames, t_pre, t_gen = pipe.generate(embeds, max_frames=12, verbose=True)
        wav = pipe.decode(frames)
        audio_s = frames.shape[0] / FRAME_RATE
        print(f"  frames {tuple(frames.shape)} -> waveform {tuple(wav.shape)} "
              f"({audio_s:.1f}s audio)")
        print(f"  prefill {t_pre:.2f}s | generate {t_gen:.2f}s "
              f"({t_gen/max(frames.shape[0],1):.2f}s/frame) | RTF {(t_pre+t_gen)/audio_s:.2f}")
        return {"codes_real_n": tot_b, "codes_real_total": tot_n,
                "codes_real_pct": tot_b / max(tot_n, 1) * 100,
                "codes_synth_n": _synth_bad}
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
        _mv = compare_hidden(got_v, exp_v)
        print(f"  [velocity      ] PCC {_mv['pcc']:.8f}  maxabs {(got_v-exp_v).abs().max():.3e}")

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
        return {"flow_velocity_pcc": _mv["pcc"],
                "flow_semantic_exact": bool((exp_s == got_s).all()),
                "flow_codes_74": n_diff}
    finally:
        ttnn.close_device(dev)


def gate_codec():

    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    out = {}
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
            _wav_pcc = pcc(got_wav, exp_wav)
            if n_frames == 24:
                out["codec_pcc_t24"] = _wav_pcc
            print(f"{tag} {'WAVEFORM':22s} PCC {_wav_pcc:.6f}  "
                  f"shapes {tuple(got_wav.shape)} vs {tuple(exp_wav.shape)}")
            # the staged run above uses return_stages=True, which BYPASSES bucketing -- so also
            # exercise the DEFAULT path (bucketed) that real callers get.
            plain = gen(codes)
            print(f"{tag} {'bucketed (default path)':22s} PCC {pcc(plain, exp_wav):.6f}  "
                  f"shape {tuple(plain.shape)}")
            t0 = time.perf_counter()
            gen(codes)
            print(f"{tag} warm {(time.perf_counter() - t0) * 1000:.1f} ms")
        return out
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
    # ALL 15 by default, deliberately -- NOTES.md [gate-01]. Narrow with --cases only when
    # debugging one prompt, never when recording a number.
    ap.add_argument("--cases", default="all",
                    help='"all" (default) or prompt_fixture.json indices, e.g. "0,2"')
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--steps", type=int, default=22, help="decode steps per case for --gate decode")
    ap.add_argument("--verbose", action="store_true",
                    help="per-step rows as well as the per-case and pooled summary")
    # quality_report.py reads this instead of scraping the printed tables. The prose above stays
    # for humans; the JSON line is the machine contract, so a reworded table cannot silently
    # change a recorded metric.
    ap.add_argument("--json", action="store_true",
                    help="also emit the gate's metrics as one JSON line prefixed GATE_JSON:")
    args = ap.parse_args()
    if args.cases == "all":
        with open(os.path.join(HERE, "tests", "prompt_fixture.json")) as f:
            cases = list(range(len(json.load(f)["cases"])))
    else:
        cases = [int(c) for c in args.cases.split(",")]

    def _emit(metrics):
        if args.json:
            print("GATE_JSON:" + json.dumps(metrics if metrics is not None else {}))
        return 0

    if args.gate == "flow":
        return _emit(gate_flow())
    if args.gate == "codec":
        return _emit(gate_codec())
    if args.gate == "codes":
        return _emit(gate_codes())

    dev = open_device()
    try:
        if args.gate == "wiring":
            m = gate_wiring(dev, bref)
        elif args.gate == "prefill26":
            m = gate_prefill26(dev, bref, cases, args.layers)
        else:
            m = gate_decode(dev, bref, cases, args.steps, args.layers, args.verbose)
    finally:
        ttnn.close_device(dev)
    return _emit(m)


if __name__ == "__main__":
    raise SystemExit(main())
