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
    # **Two lengths, not more, and 2x rather than 3x.** Every distinct utterance length
    # is a fresh JIT compile of the flow decoder and the vocoder, and each length here
    # is compiled for both schedules plus a throwaway warm-up pass. A third point, or a
    # 3x arm, costs minutes of compilation to sharpen a trend that two points already
    # establish -- and it timed out this test's first draft.
    lengths = [len(generated), 2 * len(generated)]
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
    synth_warm = TtStreamingSynthesizer(device, flow, hift, StreamConfig())

    # ------------------------------------------------------------------ warm-up
    # **Every geometry both schedules will ask for, not one.** The flow decoder and
    # the vocoder are JIT-compiled per mel length, and the two schedules run
    # *different* lengths: the batch path decodes a whole utterance in one call, the
    # streaming path decodes chunks. A warm-up that touches only the chunk geometry
    # leaves the batch path paying a ~60 s compile inside the timed region, which is
    # not a latency measurement of anything -- it made first audio read 65 s against
    # an utterance of 3.27 s the first time this ran.
    cfg = StreamConfig()
    for n in lengths:
        warm_mel, warm_frames = flow_chunk(token_streams[n])
        wp, wn = rng(warm_frames)
        w_wav, _, w_src = hift.inference(
            warm_mel, warm_frames, phase_vec=dev(wp, ttnn.float32), sine_noise_unit=dev(wn, ttnn.float32)
        )
        for t in (warm_mel, w_wav, w_src):
            ttnn.deallocate(t)
        # The streaming path vocodes chunks carrying the mel cache and the overlap
        # trim, so their frame counts are not any of the plain lengths above. One
        # throwaway streaming run per length compiles exactly those.
        with synth_warm.session(ctx, rng) as warm_session:
            for token in token_streams[n]:
                for wav, _n in warm_session.push(token):
                    ttnn.deallocate(wav)
            wav, _n = warm_session.finish()
            ttnn.deallocate(wav)
    ttnn.synchronize_device(device)

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
        first_s, chunks = None, []
        t0 = time.perf_counter()
        with synth.session(ctx, rng) as session:

            def push(token):
                nonlocal first_s
                for wav, _n in session.push(token):
                    ttnn.synchronize_device(device)
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    chunks.append(wav)

            decode_all(step, tokens, on_token=push)
            wav, _n = session.finish()
            ttnn.synchronize_device(device)
            if first_s is None:  # an utterance shorter than one chunk never streams
                first_s = time.perf_counter() - t0
            chunks.append(wav)
        total = time.perf_counter() - t0
        step.release()
        n_chunks = len(chunks)
        for c in chunks:
            ttnn.deallocate(c)
        return {"first_s": first_s, "total_s": total, "n_chunks": n_chunks}

    results = {}
    for n in lengths:
        tokens = token_streams[n]
        results[n] = (run_batch(tokens), run_streaming(tokens))

    # ------------------------------------------------------------------ report
    chunk_seconds = cfg.chunk_size() / 50.0
    print(f"\n  first-audio latency, batch schedule against streaming schedule")
    print(f"    chunk size {cfg.chunk_size()} tokens = {chunk_seconds:.2f} s of speech")
    print("    tokens  audio s  chunks   LLM s   batch first/total   stream first/total   TTFA gain")
    for n in lengths:
        b, st = results[n]
        print(
            f"    {n:6d}  {b['audio_s']:7.2f}  {st['n_chunks']:6d}  {b['llm_s']:6.3f}"
            f"   {b['first_s']:6.3f} / {b['total_s']:6.3f}"
            f"   {st['first_s']:6.3f} / {st['total_s']:6.3f}"
            f"   {b['first_s'] / st['first_s']:8.2f}x"
        )
    short, long = lengths[0], lengths[-1]
    b_s, s_s = results[short]
    b_l, s_l = results[long]
    print(
        f"    tripling the utterance moves batch first audio {b_l['first_s'] / b_s['first_s']:.2f}x "
        f"but streaming first audio only {s_l['first_s'] / s_s['first_s']:.2f}x"
    )

    for n in lengths:
        b, st = results[n]
        # The claim, asserted: audio exists before generation has finished. This is
        # what "streaming begins after token generation completes" said was missing.
        assert st["first_s"] < b["total_s"], (
            f"{n} tokens: streaming first audio {st['first_s']:.3f} s is no earlier than "
            f"the batch path's {b['total_s']:.3f} s -- the interleaving is not happening"
        )
        # It must also beat playback, or a player stalls waiting for samples...
        assert st["first_s"] < b["audio_s"], (
            f"{n} tokens: first audio at {st['first_s']:.3f} s for a {b['audio_s']:.2f} s "
            "utterance -- the stream cannot stay ahead of playback"
        )
        # ...and sustain it, which is the bound that matters once playback has started.
        assert st["total_s"] < b["audio_s"], (
            f"{n} tokens: streamed total {st['total_s']:.3f} s exceeds the {b['audio_s']:.2f} s "
            "it produces -- the stream cannot sustain real time"
        )

    # And the scaling claim, which is the reason streaming exists: first audio must
    # grow far more slowly than the utterance does. A schedule where it grew in step
    # would be the batch path wearing a callback.
    growth_batch = b_l["first_s"] / b_s["first_s"]
    growth_stream = s_l["first_s"] / s_s["first_s"]
    assert growth_stream < growth_batch * 0.6, (
        f"first audio scales with utterance length almost as fast as the batch path does "
        f"({growth_stream:.2f}x against {growth_batch:.2f}x over a {long / short:.0f}x longer "
        "utterance) -- the chunk schedule is not bounding it"
    )
