# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Time to first audio -- what makes the pipeline streamable rather than chunked.

`tests/e2e/test_streaming.py` answers a different question: does chunked vocoding
reconstruct the same speech? It does -- and that is a content check on an utterance
that has **already been fully generated**. It says nothing about *when* the first
sample can be handed to a caller, and on the batch path the answer is "after the last
token", because `CosyVoiceTTNN.synthesize` runs the three stages strictly in order.

`synthesize_streaming` interleaves them: the AR decode loop calls back with each token
(`generate(on_token=...)`), and the moment a chunk's worth has accumulated the flow
decoder and the vocoder run on it and emit audio. Nothing overlaps in compute -- one
device, one command queue -- so the *total* gets slightly worse. What changes is that
time to first audio stops scaling with the length of the utterance.

Measured the only way it can be believed: **both schedules, in one process, on one
device, over the same tokens, with all three stages real.** The AR decoder is prefilled
from the captured prefix and stepped for every token, so the decode cadence is this
board's own; the token *identities* are replayed from the golden rather than sampled,
so both schedules cut chunks at exactly the same boundaries and the comparison is of
schedules and not of two different utterances.

**What this does not measure is how first audio behaves as the utterance grows**, and
that is a device limit rather than a choice -- see the note beside `tokens` below.

Both numbers are printed, in both directions. Reporting only the first-audio win would
be a sales pitch; reporting only the total would hide what streaming is for.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden
from models.demos.cosyvoice.tt.weights import default_weights_path

HIFT_WEIGHTS = default_weights_path()
FLOW_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "flow_")
LLM_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "llm_")
SAMPLE_RATE = 22050

# All three stages plus **one** decode trace. The 384 MB region the rest of the perf
# suite asks for is sized for `TracedDecodeStepInPlace`, which captures 65 traces; this
# test captures a single moving-cache trace and 64 MB covers it comfortably.
#
# The difference is not cosmetic on a 12 GB part. This test runs the flow decoder and
# the vocoder alongside a live trace, so every megabyte the trace region reserves is a
# megabyte they cannot have — and asking for 384 MB hung n300 outright while both 32 GB
# Blackhole boards were fine with it. Reserve what is used.
needs_l1_small = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 67108864}], indirect=True
)
needs_weights = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (HIFT_WEIGHTS, FLOW_WEIGHTS, LLM_WEIGHTS)),
    reason="export hift, flow and llm weights first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "llm.ar_forward_chunk.npz")), reason="generate goldens first"
)


# The utterance is vocoded twice on each schedule -- once to warm, once to measure --
# and the flow decoder JIT-compiles per mel length, so the one-time compile bill here
# is minutes rather than seconds. `pytest.ini`'s 300 s default would fire during
# warm-up and report a timeout for work that has not started being measured.
#
# Most of that bill is paid once per *machine* rather than once per run: mounting
# `~/.cache/tt-metal-cache` into the container carries the compiled kernels across
# runs (PERF.md, *Operational notes*). Without it, every configuration of the perf
# suite recompiles everything, and this test is where that shows up first.
@pytest.mark.timeout(3600)
@needs_weights
@needs_golden
@needs_l1_small
def test_device_streaming_first_audio_latency(device):
    """First-audio latency and total, batch schedule against streaming schedule."""
    import ttnn

    # **Skipped on Wormhole for a TTNN defect, isolated below the port.**
    # Re-seeding a trace's persistent buffers after that trace has *executed* hangs
    # Wormhole. This test captures once and re-seeds per pass (see below), so passes
    # 2-4 hit it. The minimal reproduction is `capture -> seed -> step() xN -> seed`
    # with no flow decoder, no vocoder and no allocation under a live trace, and it
    # hangs the same way; Blackhole runs that sequence in under a second.
    # `synthesize_streaming` escapes it by capturing and releasing per call, which is
    # why the shipped path runs on n300 and this test does not. See docs/VALIDATION.md.
    if "WORMHOLE" in str(device.arch()).upper():
        pytest.skip(
            "hangs Wormhole n300: re-seeding a trace's persistent buffers after the "
            "trace has executed; see docs/VALIDATION.md and PERF.md, Known limitations"
        )

    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.streaming import StreamConfig, TtStreamingSynthesizer
    from models.demos.cosyvoice.tt.weights import WeightBag

    # ------------------------------------------------------------------ inputs
    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")

    all_tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    mel_len1 = int(lr_g["call0.in_mel_len1"])
    prompt_tokens = all_tokens[:, :token_len1]
    generated = all_tokens[0, token_len1:].tolist()
    prompt_feat = as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
    embedding = as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)
    prefix = as_torch(load_golden("llm.ar_forward_chunk")["call0.in_xs"])

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    # ------------------------------------------------------------------ stages
    llm_bag = WeightBag.load(LLM_WEIGHTS)
    ar_meta = llm_bag.meta["ar_decoder"]
    dec = TtARDecoder(device, llm_bag.sub("llm"), ar_meta)
    speech_embedding = llm_bag.tensor("speech_embedding.weight")
    flow_bag = WeightBag.load(FLOW_WEIGHTS)
    flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))

    # **One length, and the reason is a device limit rather than a preference.**
    # The interleaved schedule runs the flow decoder and the vocoder *while the AR
    # decode trace is live*, and TTNN says so out loud -- "Allocating device buffers is
    # unsafe due to the existence of an active trace". At the golden utterance's length
    # that combination is stable and is what this test measures. Pushing it to a longer
    # utterance (a wider trace region plus more and larger live buffers) reproducibly
    # wedged the board: 45 minutes at 100 % CPU with the JIT cache flat, twice, on two
    # different p150a boards. So the longer arm is not measured and is not claimed;
    # `PERF.md` records it as an open limitation next to the L1_SMALL growth across
    # geometries it most likely shares a cause with.
    #
    # What is lost is the *scaling* demonstration, and what is not lost is the claim
    # the review actually turns on: first audio arrives before generation finishes.
    # That is measured here, head to head, at a length both schedules certify at.
    tokens = generated

    prefix_len = prefix.shape[1]
    max_len = ((prefix_len + len(tokens) + 1 + 127) // 128) * 128

    # **One capture for the whole test, re-seeded per pass.** Each pass needs a fresh
    # KV cache -- stepping consumes it -- but it does not need a fresh *trace*, and
    # capturing per pass is what broke this test: four 384 MB captures in one process
    # (two warm-up passes plus two measured) hung the board reproducibly, with the log
    # and the JIT cache both frozen. `seed()` exists precisely so a prefill can be
    # loaded into buffers a trace already points at, so the trace is captured once and
    # each pass re-prefills into it.
    def prefill():
        caches = dec.empty_cache(max_len, prefix_len)
        ys, caches = dec.forward_chunk_fixed(
            dev(prefix),
            caches,
            max_len,
            valid=prefix_len,
            mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
        )
        ttnn.deallocate(ys)
        return caches

    def reset_decoder():
        """Load a fresh prefill into the captured trace's buffers."""
        caches = prefill()
        step.seed(caches)
        TtARDecoder.free_caches(caches)
        return step

    _g = torch.Generator().manual_seed(1986)
    _phase = torch.empty(1, 1, 9).uniform_(-3.141592653589793, 3.141592653589793, generator=_g)
    _phase[0, 0, 0] = 0.0

    def rng(mel_frames, seed=1986):
        g = torch.Generator().manual_seed(seed + mel_frames)
        return _phase, torch.randn(1, mel_frames * 256, 9, generator=g)

    def flow_chunk(tokens):
        toks = torch.cat([prompt_tokens, torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)], dim=1)
        mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens))
        g = torch.Generator().manual_seed(1986 + len(tokens))
        z = torch.randn(1, mel_len1 + mel_len2, 80, generator=g)
        mel = flow.inference(
            ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            token_len1,
            mel_len1,
            mel_len2,
            dev(prompt_feat),
            dev(embedding),
            dev(z),
        )
        return mel, mel_len2

    from types import SimpleNamespace

    ctx = SimpleNamespace(flow_chunk=flow_chunk)
    cfg = StreamConfig()

    # ------------------------------------------------------------------ measure
    def decode_all(step, toks, on_token=None):
        """Every decode step, for real, in order. `on_token` is the streaming hook."""
        for i, token in enumerate(toks):
            step.step(speech_embedding[token].reshape(1, 1, -1), prefix_len + 1 + i)
            ttnn.synchronize_device(device)
            if on_token is not None:
                on_token(token)

    def run_batch(toks):
        """Every token, then all the mel, then all the audio -- `synthesize`'s order."""
        reset_decoder()
        t0 = time.perf_counter()
        decode_all(step, toks)
        llm_s = time.perf_counter() - t0
        mel, frames = flow_chunk(toks)
        phase, noise = rng(frames)
        whole, n_whole, src = hift.inference(
            mel, frames, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise, ttnn.float32)
        )
        ttnn.synchronize_device(device)
        total = time.perf_counter() - t0
        for t in (mel, whole, src):
            ttnn.deallocate(t)
        # First audio *is* the total here: nothing can be handed out earlier.
        return {"llm_s": llm_s, "first_s": total, "total_s": total, "audio_s": n_whole / SAMPLE_RATE}

    def run_streaming(toks):
        """The same three stages, interleaved -- `synthesize_streaming`'s order."""
        synth = TtStreamingSynthesizer(device, flow, hift, cfg)
        reset_decoder()
        first_s, chunks, n_samples = None, [], 0
        t0 = time.perf_counter()
        with synth.session(ctx, rng) as session:

            def push(token):
                nonlocal first_s
                for wav, n in session.push(token):
                    ttnn.synchronize_device(device)
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    chunks.append((wav, n))

            decode_all(step, toks, on_token=push)
            wav, n = session.finish()
            ttnn.synchronize_device(device)
            if first_s is None:  # an utterance shorter than one chunk never streams
                first_s = time.perf_counter() - t0
            chunks.append((wav, n))
        total = time.perf_counter() - t0
        n_chunks = len(chunks)
        for c, n in chunks:
            n_samples += n
            ttnn.deallocate(c)
        return {"first_s": first_s, "total_s": total, "n_chunks": n_chunks, "audio_s": n_samples / SAMPLE_RATE}

    # **Warm the flow decoder and the vocoder BEFORE the decode trace is captured, and
    # do it without the decoder.** Both orderings were tried on the same board. Warming
    # first and capturing after is stable; capturing first and warming through the
    # traced path hangs the device outright -- log frozen, JIT cache flat, 100 % CPU,
    # reproducible with a cleared cache on a freshly reset board. TTNN says why in the
    # same log: "Allocating device buffers is unsafe due to the existence of an active
    # trace." A capture reserves its region, and the flow decoder and vocoder then have
    # to find room around it for geometries the allocator has never seen.
    #
    # So every geometry either schedule will ask for is allocated, used and freed here,
    # with no trace live -- the batch path's whole-utterance shapes and, via one
    # throwaway streaming run, the chunk shapes that carry the mel cache and the
    # overlap trim. Only then is the trace captured.
    warm_mel, warm_frames = flow_chunk(tokens)
    wp, wn = rng(warm_frames)
    w_wav, _, w_src = hift.inference(
        warm_mel, warm_frames, phase_vec=dev(wp, ttnn.float32), sine_noise_unit=dev(wn, ttnn.float32)
    )
    for t in (warm_mel, w_wav, w_src):
        ttnn.deallocate(t)
    with TtStreamingSynthesizer(device, flow, hift, cfg).session(ctx, rng) as warm_session:
        for token in tokens:
            for wav, _n in warm_session.push(token):
                ttnn.deallocate(wav)
        ttnn.deallocate(warm_session.finish()[0])
    ttnn.synchronize_device(device)

    # One capture for the whole test; `reset_decoder` re-prefills into it per pass.
    step = TracedDecodeStep(dec, max_len).capture()

    batch = run_batch(tokens)
    stream = run_streaming(tokens)
    step.release()

    # ------------------------------------------------------------------ report
    chunk_seconds = cfg.chunk_size() / 50.0
    print(f"\n  first audio: batch schedule against streaming schedule, {len(tokens)} tokens")
    print(f"    chunk size {cfg.chunk_size()} tokens = {chunk_seconds:.2f} s of speech")
    print(
        f"    LLM alone                  {batch['llm_s']:6.3f} s   ({1e3 * batch['llm_s'] / len(tokens):.2f} ms/token)"
    )
    print(
        f"    batch      {batch['audio_s']:5.2f} s audio    first audio {batch['first_s']:6.3f} s = total {batch['total_s']:6.3f} s"
    )
    print(
        f"    streaming  {stream['audio_s']:5.2f} s audio, {stream['n_chunks']} chunks"
        f"   first audio {stream['first_s']:6.3f} s   total {stream['total_s']:6.3f} s"
    )
    print(f"    first-audio gain            {batch['first_s'] / stream['first_s']:6.2f}x")
    print(f"    cost of interleaving        {stream['total_s'] / batch['total_s']:6.2f}x on the total")

    # The claim, asserted: audio exists before generation has finished. This is what
    # "streaming begins after token generation completes" said was missing.
    assert stream["first_s"] < batch["total_s"], (
        f"streaming first audio {stream['first_s']:.3f} s is no earlier than the batch "
        f"path's {batch['total_s']:.3f} s -- the interleaving is not happening"
    )
    # A chunk must actually have been emitted mid-generation. Without this the test
    # would pass on an utterance too short to chunk, having proved nothing.
    assert stream["n_chunks"] >= 2, (
        f"only {stream['n_chunks']} chunk emitted for {len(tokens)} tokens against a "
        f"{cfg.chunk_size()}-token chunk size -- nothing streamed during generation"
    )
    # It must beat playback, or a player stalls waiting for samples...
    assert stream["first_s"] < batch["audio_s"], (
        f"first audio at {stream['first_s']:.3f} s for a {batch['audio_s']:.2f} s utterance "
        "-- the stream cannot stay ahead of playback"
    )
    # ...and sustain it, which is the bound that matters once playback has started.
    assert stream["total_s"] < stream["audio_s"], (
        f"streamed total {stream['total_s']:.3f} s exceeds the {stream['audio_s']:.2f} s it "
        "produces -- the stream cannot sustain real time"
    )
