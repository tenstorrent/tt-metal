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

# All three stages plus a decode trace, so this is the same device geometry
# `test_device_end_to_end_rtf` asks for.
needs_l1_small = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 402653184}], indirect=True
)
needs_weights = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (HIFT_WEIGHTS, FLOW_WEIGHTS, LLM_WEIGHTS)),
    reason="export hift, flow and llm weights first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "llm.ar_forward_chunk.npz")), reason="generate goldens first"
)


# Two utterance lengths, each vocoded twice (warm-up then measured) on both
# schedules, and the flow decoder JIT-compiles per mel length -- so the one-time
# compile bill here is minutes, not seconds. `pytest.ini`'s 300 s default would fire
# during warm-up and report a timeout for work that has not started being measured.
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

    # **Two lengths, because the claim is about scaling.** Time to first audio is
    # bounded below by one chunk's worth of tokens plus that chunk's flow and vocoder
    # work -- a *constant*. The batch path's first audio is the whole utterance. On a
    # short utterance the two are close, and quoting only the short one would
    # understate streaming exactly as badly as quoting only the long one overstates
    # it. The longer sequence is the golden's own tokens repeated, so it is a real
    # token stream of a realistic length; what it says as speech is not meaningful and
    # nothing here reads its content.
    #
    # **The long length is `short + 2 * hop`, and that is not arbitrary.** The chunk
    # scheduler cuts at multiples of `token_hop_len`, so shifting the total by an exact
    # multiple of the hop shifts the final chunk's start by the same amount and leaves
    # its *length* unchanged. Both lengths therefore stream through exactly the same
    # set of geometries. That matters because **every distinct mel length is a fresh
    # JIT compile of the flow decoder and the vocoder** -- an arbitrary second length
    # adds several minutes of compilation and, worse, several more live geometries on
    # one open device, which is the pattern the known L1_SMALL growth across geometries
    # is worst at. A first draft using an arbitrary 3x length wedged mid-test.
    #
    # For the same reason the **batch** schedule is measured at the short length only.
    # Its first audio *is* its total by construction -- nothing can be handed out
    # before the last token is vocoded -- so the head-to-head belongs where the batch
    # path is most favourable, and the long arm exists to show what streaming's first
    # audio does when the utterance grows. Adding a batch arm there would cost the
    # single most expensive geometry in the test to measure a quantity that only grows.
    short = len(generated)
    long = short + 2 * StreamConfig().token_hop_len
    lengths = [short, long]
    token_streams = {n: (generated * ((n // len(generated)) + 1))[:n] for n in lengths}

    prefix_len = prefix.shape[1]
    max_len = ((prefix_len + max(lengths) + 1 + 127) // 128) * 128

    def new_decoder():
        """A prefilled, captured decode step. Rebuilt per schedule so neither run
        inherits the other's cache state -- the KV buffers are consumed by stepping."""
        caches = dec.empty_cache(max_len, prefix_len)
        ys, caches = dec.forward_chunk_fixed(
            dev(prefix),
            caches,
            max_len,
            valid=prefix_len,
            mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
        )
        ttnn.deallocate(ys)
        step = TracedDecodeStep(dec, max_len).capture()
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
    def decode_all(step, tokens, on_token=None):
        """Every decode step, for real, in order. `on_token` is the streaming hook."""
        for i, token in enumerate(tokens):
            step.step(speech_embedding[token].reshape(1, 1, -1), prefix_len + 1 + i)
            ttnn.synchronize_device(device)
            if on_token is not None:
                on_token(token)

    def run_batch(tokens):
        """Every token, then all the mel, then all the audio -- `synthesize`'s order."""
        step = new_decoder()
        t0 = time.perf_counter()
        decode_all(step, tokens)
        llm_s = time.perf_counter() - t0
        mel, frames = flow_chunk(tokens)
        phase, noise = rng(frames)
        whole, n_whole, src = hift.inference(
            mel, frames, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise, ttnn.float32)
        )
        ttnn.synchronize_device(device)
        total = time.perf_counter() - t0
        step.release()
        for t in (mel, whole, src):
            ttnn.deallocate(t)
        # First audio *is* the total here: nothing can be handed out earlier.
        return {"llm_s": llm_s, "first_s": total, "total_s": total, "audio_s": n_whole / SAMPLE_RATE}

    def run_streaming(tokens):
        """The same three stages, interleaved -- `synthesize_streaming`'s order."""
        synth = TtStreamingSynthesizer(device, flow, hift, cfg)
        step = new_decoder()
        first_s, chunks, n_samples = None, [], 0
        t0 = time.perf_counter()
        with synth.session(ctx, rng) as session:

            def push(token):
                nonlocal first_s
                for wav, _n in session.push(token):
                    ttnn.synchronize_device(device)
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    chunks.append((wav, _n))

            decode_all(step, tokens, on_token=push)
            wav, n = session.finish()
            ttnn.synchronize_device(device)
            if first_s is None:  # an utterance shorter than one chunk never streams
                first_s = time.perf_counter() - t0
            chunks.append((wav, n))
        total = time.perf_counter() - t0
        step.release()
        n_chunks = len(chunks)
        for c, n in chunks:
            n_samples += n
            ttnn.deallocate(c)
        return {"first_s": first_s, "total_s": total, "n_chunks": n_chunks, "audio_s": n_samples / SAMPLE_RATE}

    # **Warm-up is the measured sequence itself, run once and thrown away.** Warming a
    # hand-listed set of geometries is how the first draft ended up compiling the
    # batch path's full-utterance flow decoder *inside* the timed region -- first audio
    # read 65 s for a 3.27 s utterance. Replaying the exact sequence cannot miss a
    # geometry, because it is the same sequence.
    run_batch(token_streams[short])
    for n in lengths:
        run_streaming(token_streams[n])
    ttnn.synchronize_device(device)

    batch = run_batch(token_streams[short])
    stream = {n: run_streaming(token_streams[n]) for n in lengths}

    # ------------------------------------------------------------------ report
    chunk_seconds = cfg.chunk_size() / 50.0
    s_short, s_long = stream[short], stream[long]
    print("\n  first audio: batch schedule against streaming schedule")
    print(f"    chunk size {cfg.chunk_size()} tokens = {chunk_seconds:.2f} s of speech")
    print(f"    batch     {short:4d} tokens, {batch['audio_s']:5.2f} s audio   LLM {batch['llm_s']:6.3f} s")
    print(f"              first audio {batch['first_s']:6.3f} s = total {batch['total_s']:6.3f} s")
    for n in lengths:
        r = stream[n]
        print(
            f"    streaming {n:4d} tokens, {r['audio_s']:5.2f} s audio, {r['n_chunks']} chunks"
            f"   first audio {r['first_s']:6.3f} s   total {r['total_s']:6.3f} s"
        )
    print(f"    first-audio gain at {short} tokens: {batch['first_s'] / s_short['first_s']:.2f}x")
    print(f"    cost of interleaving on the total:  {s_short['total_s'] / batch['total_s']:.2f}x")
    print(
        f"    {long / short:.2f}x the utterance moves streaming first audio "
        f"{s_long['first_s'] / s_short['first_s']:.2f}x and its total "
        f"{s_long['total_s'] / s_short['total_s']:.2f}x"
    )

    # The claim, asserted: audio exists before generation has finished. This is what
    # "streaming begins after token generation completes" said was missing.
    assert s_short["first_s"] < batch["total_s"], (
        f"streaming first audio {s_short['first_s']:.3f} s is no earlier than the batch "
        f"path's {batch['total_s']:.3f} s -- the interleaving is not happening"
    )
    # It must also beat playback, or a player stalls waiting for samples...
    assert s_short["first_s"] < batch["audio_s"], (
        f"first audio at {s_short['first_s']:.3f} s for a {batch['audio_s']:.2f} s utterance "
        "-- the stream cannot stay ahead of playback"
    )
    # ...and sustain it, which is the bound that matters once playback has started.
    for n in lengths:
        r = stream[n]
        assert r["total_s"] < r["audio_s"], (
            f"{n} tokens: streamed total {r['total_s']:.3f} s exceeds the {r['audio_s']:.2f} s it "
            "produces -- the stream cannot sustain real time"
        )

    # And the scaling claim, which is the reason streaming exists: first audio is
    # bounded by one chunk, so it must stay roughly put while the utterance -- and
    # therefore the total -- grows. A schedule where first audio grew with the total
    # would be the batch path wearing a callback.
    grew = s_long["total_s"] / s_short["total_s"]
    moved = s_long["first_s"] / s_short["first_s"]
    assert grew > 1.4, (
        f"the long arm is not actually longer: total {s_short['total_s']:.3f} -> "
        f"{s_long['total_s']:.3f} s for {short} -> {long} tokens"
    )
    assert moved < grew * 0.6, (
        f"first audio scales with the utterance almost as fast as the whole run does "
        f"({moved:.2f}x against {grew:.2f}x) -- the chunk schedule is not bounding it"
    )
