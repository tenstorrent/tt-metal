# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming, stage A (tt/streaming.py): upstream's chunk schedule over fixed tokens, gated against upstream's own
streaming run on the same tokens (`COSYVOICE2_STREAM_REF`, scripts/streaming_reference.py's --out-dir; with
`COSYVOICE2_INPUTS`; skipped without them). Four checks per case:
- **the schedule**: the chunk offsets and hops equal upstream's;
- **the flow, per chunk**: our mel of the chunk's new frames against upstream's streaming mel. Upstream's own
  non-streaming mel of the same frames is the control: it must be further away than ours, at every middle chunk (the
  final chunk is non-streaming on both sides);
- **HiFT, mechanism**: `HiFTStream` fed upstream's mel pieces with upstream's F0 and noise for each call. Every chunk's
  emitted audio (the padded first and final ones included) and every seam against upstream's. The final call is
  padded to a bucket at its end and masked (tt/hifigan/valid_length.py), so it must end as upstream's does. The
  utterance's last 20 ms are gated on level, within 3 dB of upstream's, and its last 0.4 s on the difference, 15 dB
  below the signal, neither with an absolute floor (notes: D41); the final chunk's PCC is gated before those 0.4 s,
  where a quiet ending (-71 dBFS) would make PCC measure the port's own noise floor;
- **HiFT, own F0**: the same with our F0 predictor, judged on log-mel L1 (F0 differences drift the sine phase).
`COSYVOICE2_STREAM_OUT`, if set, also gets the fully offline-streamed audio (our flow and our HiFT) as wavs and a
results.json, for scripts/eval_wer_sim.py.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest
import torch

from models.experimental.cosyvoice2.tt.streaming import (
    MAX_TOKEN_HOP,
    PRE_LOOKAHEAD,
    TOKEN_HOP,
    prompt_pad,
    stream_schedule,
)

REF_DIR = os.environ.get("COSYVOICE2_STREAM_REF", "")
INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")
OUT_DIR = os.environ.get("COSYVOICE2_STREAM_OUT", "")


def test_stream_schedule_is_upstreams():
    # prompt 175 tokens: pad 0; 213 generated -> chunks of 25, 50, 100, then the final 38
    sched = stream_schedule(213, 175)
    assert [(c.offset, c.hop, c.final) for c in sched] == [(0, 25, False), (25, 50, False), (75, 100, False),
                                                           (175, 38, True)]  # fmt: skip
    # prompt 168: pad 7, so the first chunk is 32 tokens and prompt + first chunk is 200, a multiple of 25
    assert prompt_pad(168) == 7 and stream_schedule(100, 168)[0].hop == 32
    # a chunk needs hop + 3 look-ahead tokens: 27 tokens after a 175-token prompt make one final chunk only
    assert [(c.offset, c.hop, c.final) for c in stream_schedule(27, 175)] == [(0, 27, True)]
    assert [(c.offset, c.hop) for c in stream_schedule(28, 175)] == [(0, 25), (25, 3)]
    # the hop doubles to 100 and stays there
    hops = [c.hop for c in stream_schedule(1000, 175) if not c.final]
    assert hops[:4] == [TOKEN_HOP, 2 * TOKEN_HOP, MAX_TOKEN_HOP, MAX_TOKEN_HOP] and set(hops[2:]) == {MAX_TOKEN_HOP}
    # every token is emitted exactly once
    for n, p in ((213, 175), (95, 168), (347, 168), (5, 175), (1000, 3)):
        sched = stream_schedule(n, p)
        assert sched[-1].final and sum(c.hop for c in sched) == n
        assert all(a.offset + a.hop == b.offset for a, b in zip(sched, sched[1:]))
        assert all(n - c.offset >= c.hop + PRE_LOOKAHEAD for c in sched[:-1])


# The gate, set from the first measurement (2026-09-29; docs/VALIDATION.md, "Streaming"):
# - flow: our chunk mel vs upstream's streaming mel, relative L2 <= 0.03 (measured 0.0085-0.0182);
# - HiFT mechanism: each chunk's emitted audio PCC >= 0.999 (measured 0.99931-0.99985), the final chunk's before
#   its last TAIL_S; each crossfade +-40 ms PCC >= 0.998 (measured 0.99900-0.99989);
# - the utterance's end (2026-09-30, notes D41), neither with an absolute floor: the last 20 ms within 3 dB of
#   upstream's (RMS level; measured 0.2-0.5 dB), and over the last TAIL_S the difference at least 15 dB below the
#   signal. That one measured 19.5-27.5 dB over 36 final chunks (these six under six noise draws, 2026-09-30). The
#   five lowest are all 121-127105-0015. Its last 0.4 s is near-silent (-71 dBFS) except for one 20 ms burst at
#   -58 dBFS, and the error on that burst sets the margin. Unpadded, at its exact length, the call scores 19.9-24.0,
#   so that floor is the port's own error, not the padding. 15 dB leaves 4.5 dB below the lowest.
#   The criterion before (D38) had a -50 dBFS floor, and it let the end-padded final call silence
#   the last ~25 ms of every utterance: it passed 260-123440-0010's ending with the difference 2.5 dB below the
#   signal (B28).
#   The whole final chunk's PCC is not gated: 121-127105-0015's, 13 tokens ending near -71 dBFS, is 0.9975 even with
#   the call at its exact length (0.9970 masked), the port's own error at that level;
# - own F0: whole-utterance log-mel L1 <= 0.13 (measured 0.069-0.088).
FLOW_REL = 0.03
HIFT_CHUNK_PCC, HIFT_SEAM_PCC = 0.999, 0.998
END_S, END_DB = 0.02, 3.0
TAIL_S, TAIL_DB_BELOW = 0.4, 15.0
YOU_END_DB = END_DB  # the "you" clip's 11 draws (their own noise) against upstream's ending: measured 0.2-0.6 dB
OWN_F0_LOGMEL_L1 = 0.13
SEAM_PAD = 960


def _pcc(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def _dbfs(x) -> float:
    x = np.asarray(x, np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12)))


def _rel(a, b) -> float:
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).norm() / b.norm())


@pytest.mark.skipif(not (REF_DIR and INPUTS_DIR), reason="set COSYVOICE2_STREAM_REF and COSYVOICE2_INPUTS")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_offline_streaming_matches_upstream_streaming(device):
    import soundfile

    from models.experimental.cosyvoice2.tests.pcc.test_hift_chunked import _logmel
    from models.experimental.cosyvoice2.tt.hifigan.chunking import HOP, OVERLAP_FRAMES
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext
    from models.experimental.cosyvoice2.tt.streaming import HiFTStream, flow_chunk, stream_fixed_tokens

    import ttnn

    cases = sorted(glob.glob(os.path.join(REF_DIR, "*.npz")))
    assert cases, REF_DIR
    pipeline = CosyVoice2TTNN(device)
    dtype = getattr(ttnn, pipeline.config.hift_source_dtype)
    failures, results = [], []
    print("\n| case | chunk | offset | hop | flow rel. err | control rel. err | HiFT chunk PCC | seam PCC |\n"
          "|---|---|---|---|---|---|---|---|")  # fmt: skip
    for path in cases:
        case_id = os.path.basename(path)[: -len(".npz")]
        ref = np.load(path)
        ctx = PromptContext.from_npz(os.path.join(INPUTS_DIR, f"{case_id}.npz"))
        tokens = ref["tokens"].tolist()
        sched = stream_schedule(len(tokens), ctx.n_prompt_tokens)
        assert [c.offset for c in sched] == ref["offsets"].tolist(), case_id
        assert [c.hop for c in sched] == ref["hops"].tolist(), case_id

        mech = HiFTStream(pipeline.hift, pipeline.harmonics, dtype=dtype)
        own = HiFTStream(pipeline.hift, pipeline.harmonics, dtype=dtype)
        mech_audio, own_audio, seams, emitted = [], [], [], 0
        for k, chunk in enumerate(sched):
            want_mel = torch.from_numpy(ref[f"mel_{k}"])
            mel = flow_chunk(pipeline, ctx, tokens, chunk)
            flow_rel = _rel(mel, want_mel)
            ctrl = ref["nonstreaming_mel"][:, 2 * chunk.offset : 2 * (chunk.offset + chunk.hop)]
            ctrl_rel = _rel(ctrl, want_mel)
            if flow_rel > FLOW_REL:
                failures.append(f"{case_id} chunk {k}: flow rel. err {flow_rel:.4f}")
            if not chunk.final and ctrl_rel <= flow_rel:
                failures.append(f"{case_id} chunk {k}: the non-streaming control ({ctrl_rel:.4f}) is no further away")
            noise, f0 = torch.from_numpy(ref[f"hift_noise_{k}"]), torch.from_numpy(ref[f"hift_f0_{k}"])
            got = mech.step(want_mel, chunk.final, noise, f0=f0)
            own_audio.append(own.step(want_mel, chunk.final, noise))
            want = ref[f"speech_{k}"]
            tail_note = ""
            got_body, want_body = got, want
            if chunk.final:  # the utterance's end, against upstream's, with no floor (D41)
                n, tail = int(END_S * 24000), int(TAIL_S * 24000)
                got_db, want_db = _dbfs(got[-n:]), _dbfs(want[-n:])
                sig_db, diff_db = _dbfs(want[-tail:]), _dbfs(got[-tail:] - want[-tail:])
                tail_note = (f"; last {END_S * 1000:.0f} ms: {got_db:.1f} dBFS, upstream {want_db:.1f}; last {TAIL_S} s: "
                             f"signal {sig_db:.1f}, difference {diff_db:.1f} dBFS")  # fmt: skip
                if abs(got_db - want_db) > END_DB:
                    failures.append(
                        f"{case_id} final chunk: last {END_S * 1000:.0f} ms at {got_db:.1f} dBFS, upstream's {want_db:.1f}"
                    )
                if sig_db - diff_db < TAIL_DB_BELOW:
                    failures.append(
                        f"{case_id} final chunk: last {TAIL_S} s difference {diff_db:.1f} dBFS, signal {sig_db:.1f}"
                    )
                got_body, want_body = got[:-tail], want[:-tail]
            chunk_pcc = _pcc(got_body, want_body) if len(want_body) > 2 * SEAM_PAD else float("nan")
            seam_pcc = float("nan")
            if k > 0:  # this chunk's first 3,840 samples are the crossfade with the previous chunk
                seams.append(emitted)
                n = OVERLAP_FRAMES * HOP
                lo = emitted - SEAM_PAD
                seam_got = np.concatenate([mech_audio[-1][lo - emitted :], got[: n + SEAM_PAD]])
                seam_want = np.concatenate([ref[f"speech_{k - 1}"][lo - emitted :], want[: n + SEAM_PAD]])
                seam_pcc = _pcc(seam_got, seam_want)
                if seam_pcc < HIFT_SEAM_PCC:
                    failures.append(f"{case_id} seam {k}: PCC {seam_pcc:.5f}")
            if chunk_pcc < HIFT_CHUNK_PCC:  # nan (a final chunk no longer than its tail) is never below
                failures.append(f"{case_id} chunk {k}: HiFT PCC {chunk_pcc:.5f}")
            mech_audio.append(got)
            emitted += len(got)
            print(f"| {case_id} | {k}{' (final)' if chunk.final else ''} | {chunk.offset} | {chunk.hop} | "
                  f"{flow_rel:.4f} | {ctrl_rel:.4f} | {chunk_pcc:.5f} | {seam_pcc:.5f} |{tail_note}", flush=True)  # fmt: skip
        whole_want = torch.from_numpy(ref["audio"])
        own_l1 = float((_logmel(torch.from_numpy(np.concatenate(own_audio))) - _logmel(whole_want)).abs().mean())
        print(f"  {case_id}: own F0, log-mel L1 against upstream's streamed audio {own_l1:.4f}")
        if own_l1 > OWN_F0_LOGMEL_L1:
            failures.append(f"{case_id} own-F0 log-mel L1 {own_l1:.4f}")

        if OUT_DIR:  # the whole of stage A: our flow and our HiFT, own F0, upstream's noise draws
            os.makedirs(OUT_DIR, exist_ok=True)
            streamed = stream_fixed_tokens(
                pipeline, ctx, tokens, lambda k, samples: torch.from_numpy(ref[f"hift_noise_{k}"][:, :samples])
            )
            audio = np.concatenate([c.audio for c in streamed]).astype(np.float32)
            soundfile.write(os.path.join(OUT_DIR, f"{case_id}.wav"), audio, 24000)
            results.append({**json.loads(str(np.load(os.path.join(INPUTS_DIR, f"{case_id}.npz"))["case_json"])),
                            "wav": f"{case_id}.wav", "audio_s": round(len(audio) / 24000, 3),
                            "segment_tokens": [tokens], "streaming_chunks": len(streamed)})  # fmt: skip
    if OUT_DIR:
        with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
            json.dump({"backend": "ttnn-streaming-offline", "results": results}, fh, indent=2, ensure_ascii=False)
    assert not failures, failures


# Stage B: streaming interleaved with the LLM (tt/pipeline.py `synthesize_stream`). One short case, greedy sampling.
STREAM_CASE = "zero_shot_260-123286-0014"


@pytest.mark.skipif(not INPUTS_DIR, reason="set COSYVOICE2_INPUTS")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_streaming_interleaved_with_llm(device, expect_error):
    """With the decode trace on, a chunk's flow and HiFT run between decode steps while the trace is alive, after
    `warmup_streaming()` compiled and verified every streaming geometry. Checked:
    - before `warmup_streaming()`, the call is refused, before any device work (docs/VALIDATION.md: without it, the
      first chunk allocates under the live trace);
    - at least one chunk's audio is ready before the LLM has finished;
    - the streamed tokens (greedy) equal the batch tokens: the chunk work between decode steps doesn't disturb the
      decode;
    - no trace is left alive after the call;
    - the streamed audio equals stage A's offline streaming of the same tokens, with the same noise, bit for bit."""
    from dataclasses import replace

    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext
    from models.experimental.cosyvoice2.tt.streaming import stream_fixed_tokens

    pipeline = CosyVoice2TTNN(device, replace(CosyVoice2Config.reported(), sampler="greedy"))
    ctx = PromptContext.from_npz(os.path.join(INPUTS_DIR, f"{STREAM_CASE}.npz"))
    text = ctx.meta["case"]["text"]
    pipeline.warmup_buckets()
    with expect_error(RuntimeError, r"needs warmup_streaming\(\) first"):
        pipeline.synthesize_stream(ctx, text)
    pipeline.warmup_streaming()

    def noise_for(k, samples):
        return torch.randn(1, samples, pipeline.harmonics, generator=torch.Generator().manual_seed(1000 + k))

    streamed = pipeline.synthesize_stream(ctx, text, noise_for=noise_for)
    assert pipeline.live_traces() == [], pipeline.live_traces()
    tokens = streamed.tokens
    batch = [t for ids in (pipeline.text.encode(s) for s in pipeline.text.normalize(text, split=True))
             for t in pipeline.text_to_tokens(ctx, ids)]  # fmt: skip
    assert pipeline.live_traces() == [], pipeline.live_traces()
    during = [c for c in streamed.chunks if c["during_generation"]]
    print(
        f"\n  {STREAM_CASE}: {len(tokens)} tokens, {len(streamed.chunks)} chunks, {len(during)} ready during generation; "
        f"first audio {streamed.first_audio_s:.3f} s, RTF {streamed.rtf:.3f}"
    )
    for c in streamed.chunks:
        print(f"    chunk offset {c['offset']} hop {c['hop']}{' (final)' if c['final'] else ''}: start {c['start_s']:.3f} s, "
              f"flow {c['flow']:.3f} s (CFM {c['cfm']:.3f}), HiFT {c['hift']:.3f} s, ready {c['ready_s']:.3f} s")  # fmt: skip
    assert during, "no chunk was emitted while the LLM was generating"
    assert tokens == batch, "streaming changed the greedy tokens"
    offline = np.concatenate([c.audio for c in stream_fixed_tokens(pipeline, ctx, tokens, noise_for)])
    assert np.array_equal(streamed.audio, offline), float(np.abs(streamed.audio - offline).max())


# The "you" clip (notes: B28): 260-123440-0010 streamed over 11 noise draws. With the final HiFT call padded with
# silence, the last ~25 ms went silent, and Whisper appended "you" in 5 of the 11 draws. The device half renders the
# draws (stage A, TT's Stage 1 tokens) and checks each ending's level; with COSYVOICE2_YOU_OUT set it writes them,
# with a results.json, for tests/reference/test_you_clip.py to transcribe in the reference venv.
YOU_CASE = "zero_shot_260-123440-0010"
YOU_SEEDS = (1987, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # R5's default draw, then nine more; the eleventh is upstream's noise
YOU_OUT = os.environ.get("COSYVOICE2_YOU_OUT", "")


@pytest.mark.skipif(not (REF_DIR and INPUTS_DIR), reason="set COSYVOICE2_STREAM_REF and COSYVOICE2_INPUTS")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_you_clip_noise_draws(device):
    """Every draw's last 20 ms against upstream's streamed audio of the same tokens: within YOU_END_DB (its own noise
    moves the level a little; the silenced ending was 79 dB below)."""
    import soundfile

    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext
    from models.experimental.cosyvoice2.tt.streaming import stream_fixed_tokens

    ref = np.load(os.path.join(REF_DIR, f"{YOU_CASE}.npz"))
    ctx = PromptContext.from_npz(os.path.join(INPUTS_DIR, f"{YOU_CASE}.npz"))
    tokens = ref["tokens"].tolist()
    want_db = _dbfs(ref["audio"][-int(END_S * 24000) :])
    pipeline = CosyVoice2TTNN(device)

    def seeded(seed):
        gen = torch.Generator().manual_seed(seed)
        return lambda k, samples: torch.randn(1, samples, pipeline.harmonics, generator=gen)

    def upstream(k, samples):
        return torch.from_numpy(ref[f"hift_noise_{k}"][:, :samples])

    draws = [(f"seed{s}", seeded(s)) for s in YOU_SEEDS] + [("upstream_noise", upstream)]
    failures, results = [], []
    case = json.loads(str(np.load(os.path.join(INPUTS_DIR, f"{YOU_CASE}.npz"))["case_json"]))
    for name, noise_for in draws:
        audio = np.concatenate([c.audio for c in stream_fixed_tokens(pipeline, ctx, tokens, noise_for)]).astype(
            np.float32
        )
        got_db = _dbfs(audio[-int(END_S * 24000) :])
        print(f"  {YOU_CASE} {name}: last {END_S * 1000:.0f} ms {got_db:.1f} dBFS, upstream's {want_db:.1f}")
        if abs(got_db - want_db) > YOU_END_DB:
            failures.append(f"{name}: last 20 ms at {got_db:.1f} dBFS, upstream's {want_db:.1f}")
        if YOU_OUT:
            os.makedirs(YOU_OUT, exist_ok=True)
            soundfile.write(os.path.join(YOU_OUT, f"{YOU_CASE}_{name}.wav"), audio, 24000)
            results.append({**case, "case_id": f"{YOU_CASE}_{name}", "wav": f"{YOU_CASE}_{name}.wav",
                            "audio_s": round(len(audio) / 24000, 3), "segment_tokens": [tokens]})  # fmt: skip
    if YOU_OUT:
        with open(os.path.join(YOU_OUT, "results.json"), "w") as fh:
            json.dump({"backend": "ttnn-streaming-offline", "results": results}, fh, indent=2, ensure_ascii=False)
    assert not failures, failures
