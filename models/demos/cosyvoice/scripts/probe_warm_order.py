# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run the flow decoder and the vocoder under a live AR decode trace.

`tests/perf/test_streaming_perf.py` warms the flow decoder and the vocoder, then
captures the AR decode trace (`--order shipping`). `--order reversed` captures first and
warms through the traced path, so both compile and allocate under the live trace: the
condition the open Wormhole hang in that test is narrowed to (`docs/VALIDATION.md`).
Both orders end with one interleaved pass and print `ORDER_<order>_SURVIVED` on
completion. Only survival is checked, not the audio.

Clear the JIT cache first; with a warm cache nothing compiles under the trace. Run under
a timeout, since the failure mode is a hang:

    timeout 900 python3 probe_warm_order.py --order reversed
    timeout 900 python3 probe_warm_order.py --order shipping   # control
"""
from __future__ import annotations

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("shipping", "reversed"), default="reversed")
    args = ap.parse_args()

    import torch

    import ttnn
    from models.demos.cosyvoice.tt.common import as_torch, load_golden
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.streaming import StreamConfig, TtStreamingSynthesizer
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    HIFT = default_weights_path()
    FLOW, LLM = HIFT.replace("hift_", "flow_"), HIFT.replace("hift_", "llm_")

    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=67108864)

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")

    all_tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    mel_len1 = int(lr_g["call0.in_mel_len1"])
    prompt_tokens = all_tokens[:, :token_len1]
    tokens = all_tokens[0, token_len1:].tolist()
    prompt_feat = as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
    embedding = as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)
    prefix = as_torch(load_golden("llm.ar_forward_chunk")["call0.in_xs"])

    llm_bag = WeightBag.load(LLM)
    dec = TtARDecoder(device, llm_bag.sub("llm"), llm_bag.meta["ar_decoder"])
    flow_bag = WeightBag.load(FLOW)
    flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
    hift = TtHiFTGenerator(device, WeightBag.load(HIFT))

    prefix_len = prefix.shape[1]
    max_len = ((prefix_len + len(tokens) + 1 + 127) // 128) * 128

    _g = torch.Generator().manual_seed(1986)
    _phase = torch.empty(1, 1, 9).uniform_(-3.141592653589793, 3.141592653589793, generator=_g)
    _phase[0, 0, 0] = 0.0

    def rng(mel_frames, seed=1986):
        g = torch.Generator().manual_seed(seed + mel_frames)
        return _phase, torch.randn(1, mel_frames * 256, 9, generator=g)

    def flow_chunk(toks):
        t = torch.cat([prompt_tokens, torch.tensor(toks, dtype=torch.int32).reshape(1, -1)], dim=1)
        mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(toks))
        g = torch.Generator().manual_seed(1986 + len(toks))
        z = torch.randn(1, mel_len1 + mel_len2, 80, generator=g)
        mel = flow.inference(
            ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
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

    def warm_flow_and_vocoder(synth):
        """Every geometry either schedule will ask for, allocated, used and freed."""
        wm, wf = flow_chunk(tokens)
        wp, wn = rng(wf)
        w_wav, _, w_src = hift.inference(wm, wf, phase_vec=dev(wp, ttnn.float32), sine_noise_unit=dev(wn, ttnn.float32))
        for t in (wm, w_wav, w_src):
            ttnn.deallocate(t)
        with synth.session(ctx, rng) as s:
            for token in tokens:
                for wav, _n in s.push(token):
                    ttnn.deallocate(wav)
            ttnn.deallocate(s.finish()[0])
        ttnn.synchronize_device(device)

    # Built before capture in both orders. Its carry buffers come from the first chunk it
    # synthesises: before capture in the shipping order, under the live trace in the
    # reversed one.
    synth = TtStreamingSynthesizer(device, flow, hift, cfg)
    t0 = time.perf_counter()

    if args.order == "shipping":
        print("  warming flow + vocoder BEFORE capture (the shipping order)", flush=True)
        warm_flow_and_vocoder(synth)
        print(f"  warm done at {time.perf_counter() - t0:.1f} s; capturing trace...", flush=True)
        step = TracedDecodeStep(dec, max_len).capture()
    else:
        print("  capturing trace FIRST, then warming through the traced path", flush=True)
        step = TracedDecodeStep(dec, max_len).capture()
        print(f"  capture done at {time.perf_counter() - t0:.1f} s; warming...", flush=True)
        warm_flow_and_vocoder(synth)
        print(f"  warm survived at {time.perf_counter() - t0:.1f} s", flush=True)

    # One real interleaved pass, so completing means more than a surviving warm-up.
    caches = dec.empty_cache(max_len, prefix_len)
    ys, caches = dec.forward_chunk_fixed(
        dev(prefix),
        caches,
        max_len,
        valid=prefix_len,
        mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
    )
    ttnn.deallocate(ys)
    step.seed(caches)
    TtARDecoder.free_caches(caches)

    speech_embedding = llm_bag.tensor("speech_embedding.weight")
    n_chunks = 0
    with synth.session(ctx, rng) as session:
        for i, token in enumerate(tokens):
            step.step(speech_embedding[token].reshape(1, 1, -1), prefix_len + 1 + i)
            ttnn.synchronize_device(device)
            for wav, _n in session.push(token):
                ttnn.deallocate(wav)
                n_chunks += 1
        ttnn.deallocate(session.finish()[0])
        n_chunks += 1
    ttnn.synchronize_device(device)

    step.release()
    synth.release_carry()
    print(f"  interleaved pass: {n_chunks} chunks in {time.perf_counter() - t0:.1f} s", flush=True)
    ttnn.close_device(device)
    print(f"ORDER_{args.order.upper()}_SURVIVED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
