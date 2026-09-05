# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where does the fused decode path lose accuracy in the model?

Standalone (`probe_sdpa_decode.py`) it is PCC 0.99998. Wired in, traced-vs-untraced
drops to 0.88. `probe_sdpa_mask_shape.py` ruled out the two differences in the mask --
neither its value (-1e9 vs -1e4) nor a fully-suppressed leading chunk moves the number;
only `k_chunk_size` does, and the model computes that correctly.

So the remaining suspects are in the wiring, and they split three ways. This walks them
one at a time on the real decoder:

  A. untraced fused vs untraced explicit -- is the fused path right at all?
  B. traced fused vs untraced fused     -- does trace capture change it?
  C. traced explicit vs untraced explicit -- the known-good control (should be 1.0)

A failing and B clean means the arithmetic is wrong. B failing and A clean means
`sdpa_decode` does not survive `begin_trace_capture`, which would be worth knowing
before any of this ships.

    python3 models/demos/cosyvoice/scripts/probe_sdpa_wiring.py
"""
from __future__ import annotations

import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

PREFIX, MAX_LEN, STEPS = 209, 384, 6


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    path = os.path.join(os.path.dirname(__file__), "..", "tests", "golden", "llm_weights.npz")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    try:
        bag = WeightBag.load(path)
        meta = bag.meta["ar_decoder"]
        d_in = meta["input_size"]
        torch.manual_seed(0)
        prefix = torch.randn(1, PREFIX, d_in) * 0.1
        steps = [torch.randn(1, 1, d_in) * 0.1 for _ in range(STEPS)]

        def build(sdpa: bool):
            dec = TtARDecoder(device, bag.sub("llm"), meta)
            for layer in dec.layers:
                layer.attn.sdpa_decode = sdpa
            caches = dec.empty_cache(MAX_LEN, PREFIX)
            out, caches = dec.forward_chunk_fixed(
                dev(prefix),
                caches,
                MAX_LEN,
                PREFIX,
                mask=dev(right_aligned_bias(MAX_LEN, PREFIX, PREFIX, causal=True)),
            )
            ttnn.deallocate(out)
            return dec, caches

        def run_untraced(dec, caches):
            got = []
            for i, x in enumerate(steps):
                ys, caches = dec.forward_chunk_fixed(
                    dev(x),
                    caches,
                    MAX_LEN,
                    valid=PREFIX + 1 + i,
                    mask=dev(right_aligned_bias(MAX_LEN, PREFIX + 1 + i, 1)),
                )
                got.append(ttnn.to_torch(ys).float())
                ttnn.deallocate(ys)
            TtARDecoder.free_caches(caches)
            return got

        def run_traced(dec, caches):
            traced = TracedDecodeStep(dec, MAX_LEN).capture()
            traced.seed(caches)
            TtARDecoder.free_caches(caches)
            got = []
            for i, x in enumerate(steps):
                ys = traced.step(x, PREFIX + 1 + i)
                ttnn.synchronize_device(device)
                got.append(ttnn.to_torch(ys).float())
            traced.release()
            return got

        def run_body_untraced(dec, caches):
            """`TracedDecodeStep._body()` called directly, never captured.

            The traced arm differs from `forward_chunk_fixed` in **two** ways: it is
            traced, and it keeps a time-major cache (`cache_free=True`), whose k/v reach
            attention as permutes rather than concats. Comparing traced-vs-untraced
            moves both at once. This moves only the cache layout.
            """
            traced = TracedDecodeStep(dec, MAX_LEN)
            traced.seed(caches)
            TtARDecoder.free_caches(caches)
            got = []
            for i, x in enumerate(steps):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(x, dtype=dec.dtype, layout=ttnn.TILE_LAYOUT), traced.x_buf
                )
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        right_aligned_bias(MAX_LEN, min(PREFIX + 1 + i, MAX_LEN), 1),
                        dtype=dec.dtype,
                        layout=ttnn.TILE_LAYOUT,
                    ),
                    traced.mask_buf,
                )
                traced._body()
                ttnn.synchronize_device(device)
                got.append(ttnn.to_torch(traced.ys).float())
            traced.release()
            return got

        print("\n  building four decoders (prefill is explicit in all of them)")
        u_fused = run_untraced(*build(True))
        u_expl = run_untraced(*build(False))
        b_fused = run_body_untraced(*build(True))
        b_expl = run_body_untraced(*build(False))
        t_fused = run_traced(*build(True))
        t_expl = run_traced(*build(False))

        rows = (
            ("A  untraced fused vs untraced explicit", u_fused, u_expl),
            ("B  traced fused   vs untraced fused   ", t_fused, u_fused),
            ("C  traced explicit vs untraced explicit", t_expl, u_expl),
            ("D  traced fused   vs untraced explicit", t_fused, u_expl),
            ("E  free-cache fused vs concat-cache fused (untraced)", b_fused, u_fused),
            ("F  free-cache explicit vs concat-cache explicit    ", b_expl, u_expl),
            ("G  traced fused   vs free-cache fused (untraced)   ", t_fused, b_fused),
        )
        print(f"\n  {'comparison':<52}{'worst PCC':>14}{'worst step':>12}")
        print("  " + "-" * 78)
        for label, a, b in rows:
            worst, at = 1.0, -1
            for i, (x, y) in enumerate(zip(a, b)):
                p = pcc(x, y)
                if p < worst:
                    worst, at = p, i
            print(f"  {label:<52}{worst:>14.10f}{at:>12}")
        print("\n  E low, F clean -> the time-major cache is what the fused op mishandles.")
        print("  G low          -> tracing is the remaining variable.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
