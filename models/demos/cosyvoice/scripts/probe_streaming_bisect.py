# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which of the three streaming caches blows up chunk 1 on Wormhole?

`probe_streaming_amplitude.py` localised the failure precisely:

    mel RMS, every chunk    6.67 - 7.39     the flow is fine
    waveform, non-streamed  0.04970
    waveform, chunk 0       0.06015         fine
    waveform, chunk 1       0.92929         15x too loud

Chunk 0 and chunk 1 run identical code. The *only* difference is that chunk 1 receives
the three carried caches instead of `None`:

    hift_mel     [1, 20, 80]     prepended mel context
    hift_source  [1, 5120, 1]    NSF excitation tail, passed as `cache_source`
    hift_speech  [1, 5120, 1]    previous waveform tail, crossfaded into this one

So null them one at a time and see which one takes the amplitude back to normal.
Disabling any of them makes the *seam* worse -- that is what they are for -- but none of
them should change the overall level by 15x. Whichever disable restores the level is the
one carrying the bad data.

    python3 models/demos/cosyvoice/scripts/probe_streaming_bisect.py
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


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def main() -> int:
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.streaming import TtStreamingSynthesizer
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = lambda n: np.load(os.path.join(GOLDEN, f"{n}.npz"))  # noqa: E731
    at = lambda a: torch.from_numpy(np.asarray(a)).float()  # noqa: E731

    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        emb_g, lr_g, cfm_g, spk_g = (
            g(n) for n in ("flow.input_embedding", "flow.length_regulator", "flow.cfm", "flow.spk_embed_affine")
        )
        all_tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
        token_len1 = at(lr_g["call0.in_x1"]).shape[1]
        mel_len1 = int(lr_g["call0.in_mel_len1"])
        prompt_tokens = all_tokens[:, :token_len1]
        generated = all_tokens[0, token_len1:].tolist()
        prompt_feat = at(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
        embedding = at(spk_g["call0.in_x"]).reshape(1, 1, -1)

        flow_bag = WeightBag.load(os.path.join(GOLDEN, "flow_weights.npz"))
        flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))

        def dev(v, dtype=ttnn.bfloat16):
            return ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        _gen = torch.Generator().manual_seed(1986)
        _phase = torch.empty(1, 1, 9).uniform_(-3.14159265, 3.14159265, generator=_gen)
        _phase[0, 0, 0] = 0.0

        def rng(mel_frames, seed=1986):
            gg = torch.Generator().manual_seed(seed + mel_frames)
            return _phase, torch.randn(1, mel_frames * 256, 9, generator=gg)

        def flow_chunk(tokens):
            toks = torch.cat([prompt_tokens, torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)], dim=1)
            mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens))
            gg = torch.Generator().manual_seed(1986 + len(tokens))
            z = torch.randn(1, mel_len1 + mel_len2, 80, generator=gg)
            return (
                flow.inference(
                    ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
                    token_len1,
                    mel_len1,
                    mel_len2,
                    dev(prompt_feat),
                    dev(embedding),
                    dev(z),
                ),
                mel_len2,
            )

        ctx = SimpleNamespace(flow_chunk=flow_chunk)
        original = TtStreamingSynthesizer.token2wav

        def run(disable: str | None):
            def patched(self, mel, mel_frames, state, rng_, finalize):
                for field in ("hift_mel", "hift_source", "hift_speech", "mel_overlap"):
                    if disable in (field, "all") and getattr(state, field) is not None:
                        ttnn.deallocate(getattr(state, field))
                        setattr(state, field, None)
                return original(self, mel, mel_frames, state, rng_, finalize)

            TtStreamingSynthesizer.token2wav = patched
            try:
                synth = TtStreamingSynthesizer(device, flow, hift)
                chunks = synth.synthesize(generated, ctx, rng)
                out = [ttnn.to_torch(c).float().reshape(1, -1) for c in chunks]
                for c in chunks:
                    ttnn.deallocate(c)
                return [rms(p) for p in out]
            finally:
                TtStreamingSynthesizer.token2wav = original

        print(f"\n  arch {device.arch()}   -- non-streamed reference RMS is ~0.0497")
        print(f"  {'disabled cache':<18}{'chunk 0':>10}{'chunk 1':>10}   verdict")
        print("  " + "-" * 56)
        for d in (None, "mel_overlap", "hift_mel", "hift_source", "hift_speech", "all"):
            try:
                r = run(d)
            except Exception as exc:  # noqa: BLE001
                print(f"  {str(d):<18}{'':>20}   FAILED {str(exc)[:40]}")
                continue
            worst = max(r[1:]) if len(r) > 1 else r[0]
            verdict = "amplitude RESTORED" if worst < 0.2 else ""
            cols = "".join(f"{v:>10.5f}" for v in r[:2])
            print(f"  {str(d):<18}{cols}   {verdict}")
        print("\n  The row that restores the amplitude names the cache carrying bad data.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
