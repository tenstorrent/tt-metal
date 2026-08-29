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

    prefix_len, n_tokens = prefix.shape[1], len(generated)
    max_len = ((prefix_len + n_tokens + 1 + 127) // 128) * 128

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

    # Warm every kernel both schedules use, so neither pays a JIT compile for being
    # first. One short flow+vocoder call reaches all of them.
    warm_mel, warm_frames = flow_chunk(generated[: StreamConfig().chunk_size()])
    wp, wn = rng(warm_frames)
    w_wav, _, w_src = hift.inference(
        warm_mel, warm_frames, phase_vec=dev(wp, ttnn.float32), sine_noise_unit=dev(wn, ttnn.float32)
    )
    for t in (warm_mel, w_wav, w_src):
        ttnn.deallocate(t)
    ttnn.synchronize_device(device)

    def decode_all(step, on_token=None):
        """Every decode step, for real, in order. `on_token` is the streaming hook."""
        for i, token in enumerate(generated):
            step.step(speech_embedding[token].reshape(1, 1, -1), prefix_len + 1 + i)
            ttnn.synchronize_device(device)
            if on_token is not None:
                on_token(token)

    # ------------------------------------------------- batch schedule
    step = new_decoder()
    t0 = time.perf_counter()
    decode_all(step)
    llm_only_s = time.perf_counter() - t0
    mel, frames = flow_chunk(generated)
    phase, noise = rng(frames)
    whole, n_whole, src = hift.inference(
        mel, frames, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise, ttnn.float32)
    )
    ttnn.synchronize_device(device)
    batch_first_s = batch_total_s = time.perf_counter() - t0
    audio_seconds = n_whole / SAMPLE_RATE
    step.release()
    for t in (mel, whole, src):
        ttnn.deallocate(t)

    # ------------------------------------------------- streaming schedule
    synth = TtStreamingSynthesizer(device, flow, hift, StreamConfig())
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

        decode_all(step, on_token=push)
        wav, _n = session.finish()
        ttnn.synchronize_device(device)
        if first_s is None:  # an utterance shorter than one chunk never streams
            first_s = time.perf_counter() - t0
        chunks.append(wav)
    stream_total_s = time.perf_counter() - t0
    step.release()
    n_chunks = len(chunks)
    for c in chunks:
        ttnn.deallocate(c)

    # ------------------------------------------------- report
    chunk_seconds = StreamConfig().chunk_size() / 50.0
    print(f"\n  {n_tokens} tokens -> {audio_seconds:.2f} s of audio, {n_chunks} chunks")
    print(f"    LLM alone (all {n_tokens} steps)   {llm_only_s:6.3f} s   ({1e3*llm_only_s/n_tokens:.2f} ms/token)")
    print(f"    batch      first audio     {batch_first_s:6.3f} s   total {batch_total_s:6.3f} s")
    print(f"    streaming  first audio     {first_s:6.3f} s   total {stream_total_s:6.3f} s")
    print(f"    first-audio speedup        {batch_first_s / first_s:6.2f}x")
    print(f"    cost of interleaving       {stream_total_s / batch_total_s:6.2f}x on the total")
    print(f"    first chunk covers {chunk_seconds:.2f} s of speech; it arrives at {first_s:.3f} s")

    # The claim, asserted: audio exists before generation has finished. This is what
    # "streaming begins after token generation completes" said could not be shown.
    assert first_s < batch_total_s, (
        f"streaming first audio {first_s:.3f} s is no earlier than the batch path's "
        f"{batch_total_s:.3f} s -- the interleaving is not happening"
    )
    # And it must beat playback, or a player still stalls waiting for samples.
    assert first_s < audio_seconds, (
        f"first audio at {first_s:.3f} s for a {audio_seconds:.2f} s utterance -- "
        "the stream cannot stay ahead of playback"
    )
    # Interleaving costs total time; it must not cost so much that the stream falls
    # behind real time, which is the only bound that matters for a player.
    assert stream_total_s < audio_seconds, (
        f"streamed total {stream_total_s:.3f} s exceeds the {audio_seconds:.2f} s it "
        "produces -- the stream cannot sustain real time"
    )
