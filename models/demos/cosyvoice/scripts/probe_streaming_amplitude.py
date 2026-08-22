# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where does the streamed waveform's amplitude go wrong on Wormhole?

`test_device_streamed_matches_non_streamed` fails on n300 with mel-space PCC `0.218`
against a `0.85` gate, and passes on both Blackhole boards at `0.9019`. An earlier pass
recorded it as arch-specific and left it there.

The diagnostic the test already prints says it is not a subtle numerical difference:

    RMS  streamed 0.63250   non-streamed 0.04970

**The streamed audio is 12.7x louder.** `0.63` RMS is clipping territory; `0.0497` is
ordinary speech. Something is contributing a large wrong signal rather than a slightly
wrong one, which is a much easier class of bug to find -- and the shape of it (fine on one
architecture, wrong on another) points at a buffer that is read before it is written, or
one whose contents survive from somewhere they should not.

So: print the amplitude of **every chunk** and of **every cache carried across a seam**,
and run the identical script on both architectures. Three outcomes, each pointing
somewhere different:

  - chunk 0 already loud            -> not the seam logic at all; the first flow/vocoder
                                       call differs, and the caches are innocent.
  - chunk 0 fine, chunk 1+ loud     -> the carry is the problem; the cache RMS says which
                                       of the three.
  - all chunks fine, total loud     -> the concatenation or the fade, not the synthesis.

    python3 models/demos/cosyvoice/scripts/probe_streaming_amplitude.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def main() -> int:
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.streaming import TtStreamingSynthesizer
    from models.demos.cosyvoice.tt.weights import WeightBag

    def golden(name):
        return np.load(os.path.join(GOLDEN, f"{name}.npz"))

    def as_torch(a):
        return torch.from_numpy(np.asarray(a)).float()

    device = ttnn.open_device(device_id=0, l1_small_size=262144, trace_region_size=402653184)
    try:
        emb_g, lr_g, cfm_g, spk_g = (
            golden(n) for n in ("flow.input_embedding", "flow.length_regulator", "flow.cfm", "flow.spk_embed_affine")
        )
        all_tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
        token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
        mel_len1 = int(lr_g["call0.in_mel_len1"])
        prompt_tokens = all_tokens[:, :token_len1]
        generated = all_tokens[0, token_len1:].tolist()
        prompt_feat = as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
        embedding = as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)

        flow_bag = WeightBag.load(os.path.join(GOLDEN, "flow_weights.npz"))
        flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))

        def dev(v, dtype=ttnn.bfloat16):
            return ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        _g = torch.Generator().manual_seed(1986)
        _phase = torch.empty(1, 1, 9).uniform_(-3.14159265, 3.14159265, generator=_g)
        _phase[0, 0, 0] = 0.0

        def rng(mel_frames, seed=1986):
            g = torch.Generator().manual_seed(seed + mel_frames)
            return _phase, torch.randn(1, mel_frames * 256, 9, generator=g)

        mels = {}

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
            # Record the flow's own output amplitude: if the mel is already wrong, the
            # vocoder and the seam machinery are both innocent.
            mels[len(tokens)] = rms(ttnn.to_torch(mel))
            return mel, mel_len2

        ctx = SimpleNamespace(flow_chunk=flow_chunk)

        mel, frames = flow_chunk(generated)
        phase, noise = rng(frames)
        whole, _, src = hift.inference(
            mel, frames, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise, ttnn.float32)
        )
        offline = ttnn.to_torch(whole).float().reshape(1, -1)
        for t in (mel, whole, src):
            ttnn.deallocate(t)

        synth = TtStreamingSynthesizer(device, flow, hift)
        chunks = synth.synthesize(generated, ctx, rng)
        pieces = [ttnn.to_torch(c).float().reshape(1, -1) for c in chunks]
        for c in chunks:
            ttnn.deallocate(c)
        streamed = torch.cat(pieces, dim=1)

        print(f"\n  arch: {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"\n  flow mel RMS by chunk-token-count (non-streamed is the largest):")
        for k in sorted(mels):
            print(f"    {k:>5} tokens -> mel RMS {mels[k]:.5f}")
        print(f"\n  waveform RMS")
        print(f"    non-streamed        {rms(offline):.5f}")
        for i, p in enumerate(pieces):
            print(f"    streamed chunk {i}    {rms(p):.5f}   ({p.shape[1]} samples)")
        print(f"    streamed, all       {rms(streamed):.5f}")
        print(f"\n  ratio streamed/non-streamed = {rms(streamed)/rms(offline):.2f}x")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
