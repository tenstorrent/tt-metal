# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""R.5 diagnosis (CPU): why testing/official_floor's tap check rejects compressor and attention (DS41-38, DS41-39).

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.reference.official_taps_check

(a) compressor (the durable check: reference/official_checks.py): the fp32 step with model.py's bf16 points emulated (bf16 pooled latent before the norm and after it,
    bf16 wk GEMM and k_norm, RoPE copied back into bf16, fp4 on the bf16 values) vs the official output: bit-equal
    means the map is right and the excess over 2 x the input nudge is the official arithmetic itself.
(b) attention / compressor with the context built from the official prefix state (official_taps.capture(states=True):
    the layer's own state at the chunk start, its sources' at the end) instead of the golden's
    (testing/harness.reference_ctx), same tapped inputs, same limit.
"""

from __future__ import annotations

import os
import time

import torch

LAYERS = [2, 3, 14, 20, 21, 24]  # the compressed-attention layers of the spec (KV / index sources and readers)


def main():
    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing import noise_floor as NF
    from models.demos.common.bringup.testing.harness import component_golden, reference_ctx
    from models.demos.deepseek_v41_flash_d_p.reference.official_checks import compressor_bf16
    from models.demos.deepseek_v41_flash_d_p.reference.official_taps import capture

    s = Spec.load(os.environ["BRINGUP_SPEC"])
    g, c = component_golden(s)
    t0 = time.time()
    taps, st = capture(hf_path(s), LAYERS, g.tokens(), c, g.chunk, states=True)
    print(f"captured in {time.time() - t0:.0f}s", flush=True)
    ref = s.hooks().reference(s, layers=LAYERS, dtype=torch.float32)
    k = CC.limits(s)["component_calib"]
    start, end = c * g.chunk, (c + 1) * g.chunk

    def official_ctx(L):
        state = ref.new_state(g.seq)
        own = dict(st[L]["start"])
        if "engram_tail" in own:  # the token ids' (the golden's is the same)
            own["engram_tail"] = g.state(L, at=start)["engram_tail"]
        ref.load_state(state, L, own, start)
        for j in ref.cfg.sources(L):
            ref.load_state(state, j, st[j]["end"], end)
        ctx = ref.chunk_context(L, start, g.chunk, state)
        ctx.extra["tokens"] = g.tokens()[start:end]
        return ctx

    def check(L, name, mk):
        t = taps[L][name]
        xs = [x.float() if x.is_floating_point() else x for x in t["inputs"]]
        cpu = ref.component(L, name)

        def run(*inp):
            return cpu(mk(L), *[x.clone() for x in inp])

        y = run(*xs)
        off = t["output"].float().reshape(y.shape)
        dt = NF.input_dtypes(s, s.block_type_of(L), name, len(xs))
        nud = max(CC.rel(o, y) for o in NF.nudged(run, xs, dt))
        return CC.rel(off, y), k * nud

    for L in LAYERS:
        for name in ("attention", "compressor"):
            if name not in taps[L]:
                continue
            rg, lg = check(L, name, lambda L: reference_ctx(ref, L, g, c, s))
            ro, lo = check(L, name, official_ctx)
            print(
                f"L{L:02d} {name:<10} golden ctx: rel {rg:.4f} limit {lg:.4f} {'ok' if rg <= lg else 'FAIL'}   "
                f"official ctx: rel {ro:.4f} limit {lo:.4f} {'ok' if ro <= lo else 'FAIL'}",
                flush=True,
            )

    d = ref.cfg.head_dim
    for L in LAYERS:
        if "compressor" not in taps[L]:
            continue
        t = taps[L]["compressor"]
        off = t["output"].float()
        em = compressor_bf16(ref, L, t["inputs"][0].float(), start)
        y = ref.component(L, "compressor")(reference_ctx(ref, L, g, c, s), t["inputs"][0].float())
        print(
            f"L{L:02d} compressor: official vs the bf16-emulated step rel {CC.rel(off, em):.2e} "
            f"(equal {(off == em).float().mean():.4f}); vs the fp32 step: ckv rel {CC.rel(off[:, :d], y[:, :d]):.4f}, "
            f"index_k rel {CC.rel(off[:, d:], y[:, d:]):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
