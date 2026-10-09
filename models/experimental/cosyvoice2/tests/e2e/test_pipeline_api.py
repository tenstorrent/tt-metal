# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`CosyVoice2TTNN` through its public API: the wiring that the per-stage tests cannot see.

Host (no device): the configuration's context budget, and the refusal to build while an environment switch is set.

Device (real checkpoint, the corpus's prompts from `scripts/prepare_inputs.py`; skipped without
`COSYVOICE2_INPUTS`): consecutive utterances of different lengths on one device. The test checks that:
- each segment's audio is exactly 960 samples per speech token (2 mel frames x 480);
- no trace is alive after any call;
- repeating the first three calls with the same seeds gives the same tokens and audio;
- device memory does not grow. CosyVoice1's L1_SMALL grew with every new vocoder geometry until a 32 KB bank was
  exhausted. Here conv config tensors live in DRAM, so L1_SMALL must stay flat. The DRAM that new geometries add
  (prepared conv weights, cached per geometry) must stop growing once those geometries repeat;
- nothing is evicted from the conv caches: the pipeline is bucketed, and bucketed caches never evict.
The per-call memory table is printed, for docs/VALIDATION.md.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from models.experimental.cosyvoice2.tt.pipeline import ENV_SWITCHES, CosyVoice2Config, _refuse_env_switches

INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")
SEED = 1986


def test_context_budget_covers_the_prompt_limit():
    cfg = CosyVoice2Config.reported()
    # sos + (128 prompt-text + 80 segment tokens) + task + 750 prompt speech tokens, plus upstream's 20 x 80 = 1,600
    assert cfg.max_seq_len() == 2560
    assert cfg.max_tokens_for(80) == cfg.llm_steps_for(80) == 1600  # upstream's own limit; no HiFT cap below it
    assert (cfg.min_tokens_for(52), cfg.max_tokens_for(52), cfg.llm_steps_for(52)) == (104, 1040, 1040)
    assert CosyVoice2Config.eager().llm_decode_trace is False


def test_bucket_sets_cover_the_budget(expect_error):
    from models.experimental.cosyvoice2.tt.pipeline import bucket_at_least, bucket_for

    cfg = CosyVoice2Config.reported()
    flow, hift = cfg.flow_token_buckets(), cfg.hift_frame_buckets()
    # the longest flow input: 750 prompt + 1,600 generated tokens. HiFT: single passes only below one chunk (512
    # frames); a longer mel runs in 512-frame chunks, the 512 bucket's geometry
    assert flow[-1] > 750 + 1600 and hift == [256, 512]
    assert flow == sorted(set(flow))
    assert (len(flow), len(hift), len(cfg.llm_prefill_lengths())) == (17, 2, 8)
    assert bucket_at_least(256, hift) == 256 and bucket_at_least(257, hift) == 512
    # strictly above: an exact fit still gets a padded position, so it takes the warmed (masked) path
    assert bucket_for(255, flow) == 256 and bucket_for(256, flow) == 320
    with expect_error(ValueError, "exceeds the largest bucket"):
        bucket_for(flow[-1], flow)


def test_segment_past_the_cap_raises(expect_error):
    """Past `max_segment_speech_tokens`, the LLM (once the speech has not ended by the cap) and the flow (given more
    than the cap) raise `SegmentTooLong` with the segment's length, before any device work. The reported cap is
    upstream's own 1,600, which no segment of at most 80 text tokens can pass, so this sets a smaller one."""
    from dataclasses import replace
    from types import SimpleNamespace

    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN, SegmentTooLong

    class NeverEnds:
        """The LLM's `generate`, with the speech running to whatever step limit it is given."""

        def __init__(self):
            self.max_tokens = []

        def generate(self, ids, speech, *, max_tokens, **_):
            self.max_tokens.append(max_tokens)
            return [0] * max_tokens

    cfg = replace(CosyVoice2Config.reported(), max_segment_speech_tokens=1024)
    pipe = object.__new__(CosyVoice2TTNN)  # the stages' refusals only; no device
    pipe.config, pipe.llm = cfg, NeverEnds()
    no_prompt = SimpleNamespace(prompt_text_ids=None, llm_prompt_speech_tokens=None)

    # 52 text tokens allow 1,040 speech tokens upstream: the cap binds, the LLM runs one step past it, and raises
    with expect_error(SegmentTooLong, "52 text tokens had not ended after 1025 speech tokens: past .*=1024 .41.0"):
        pipe.text_to_tokens(no_prompt, [0] * 52)
    # 51 allow 1,020: upstream's own limit binds first and ends the segment there, as upstream does
    assert len(pipe.text_to_tokens(no_prompt, [0] * 51)) == 1020
    assert pipe.llm.max_tokens == [1025, 1020]

    with expect_error(SegmentTooLong, "a segment of 1025 speech tokens: past max_segment_speech_tokens=1024 .41.0 s"):
        pipe.tokens_to_mel([0] * 1025, no_prompt)


def test_environment_switches_are_refused(monkeypatch, expect_error):
    for name in ENV_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    _refuse_env_switches()
    monkeypatch.setenv("COSYVOICE2_FLOW_SDPA", "0")
    with expect_error(RuntimeError, "takes its configuration from CosyVoice2Config only"):
        _refuse_env_switches()


def _librispeech_cases():
    from models.experimental.cosyvoice2.tt.prompt import PromptContext

    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz")))]
    return [c for c in ctxs if c.meta["case"]["set"] == "librispeech"]


@pytest.mark.skipif(not INPUTS_DIR, reason="set COSYVOICE2_INPUTS to scripts/prepare_inputs.py's --out-dir")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
# No timeout: every new sequence length compiles kernels and verifies conv geometries, minutes per utterance on a
# cold kernel cache, and a device job is never killed mid-op (pytest.ini sets 300 s).
@pytest.mark.timeout(0)
def test_device_consecutive_utterances_of_different_lengths(device):
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN, device_memory
    from models.experimental.cosyvoice2.tt.prompt import RandomSources

    ctxs = _librispeech_cases()
    assert len(ctxs) == 6, "the corpus has two speakers x three targets"
    pipe = CosyVoice2TTNN(device)
    print(
        "\n| call | case | tokens | audio s | LLM s | flow encoder s | CFM s | HiFT s | wall s | RTF "
        "| DRAM MiB/bank | L1 KiB/bank | L1_SMALL B/bank |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    try:
        # the first three again, same seeds: same geometries, so determinism and steady-state memory
        order = ctxs + ctxs[:3]
        mems, first_pass = [], {}
        for i, ctx in enumerate(order):
            case = ctx.meta["case"]
            syn = pipe.synthesize(ctx, case["text"], rng=RandomSources(llm_seed=SEED))
            mems.append(device_memory(device))
            print(_row(i, case["case_id"], syn, mems[-1]), flush=True)
            assert pipe.live_traces() == [], f"call {i}: {pipe.live_traces()} still alive"
            assert [s.text for s in syn.segments] == ctx.meta["segments"]  # upstream's normalization and split
            for s in syn.segments:
                assert s.tokens and s.audio_samples == 960 * len(s.tokens), (case["case_id"], s.audio_samples)
            # HiFT clamps to +-0.99 in float32
            assert np.isfinite(syn.audio).all() and 0.01 < np.abs(syn.audio).max() <= np.float32(0.99)
            if i < 3:
                first_pass[case["case_id"]] = syn
            elif i >= len(ctxs):
                again = first_pass[case["case_id"]]
                assert syn.tokens == again.tokens, f"{case['case_id']}: tokens differ between identical calls"
                diff = float(np.abs(syn.audio - again.audio).max())
                print(f"  repeat of {case['case_id']}: audio max|diff| {diff:.3g}", flush=True)
                assert diff == 0.0, f"{case['case_id']}: audio differs between identical calls ({diff})"
    finally:
        pipe.release()

    assert pipe.conv_cache_evictions() == 0
    l1_small = [m["l1_small"] for m in mems]
    assert len(set(l1_small)) == 1, f"L1_SMALL changed across calls: {l1_small}"
    dram_second_pass = [m["dram"] for m in mems[len(ctxs) - 1 :]]
    assert len(set(dram_second_pass)) == 1, f"DRAM grew while repeating seen geometries: {dram_second_pass}"


def _row(i, case_id, syn, m) -> str:
    t = syn.stage_totals()
    return (
        f"| {i} | {case_id} | {len(syn.tokens)} | {syn.audio_s:.2f} | {t['llm_prefill'] + t['llm_decode']:.2f} "
        f"| {t['flow_encoder']:.2f} | {t['flow_cfm']:.2f} | {t['hift']:.2f} | {syn.wall_s:.2f} | {syn.rtf:.3f} "
        f"| {m['dram'] / 2**20:.2f} | {m['l1'] / 2**10:.1f} | {m['l1_small']} |"
    )
