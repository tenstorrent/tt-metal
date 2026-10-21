# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference's numerics switch (FIX.6; F112 / F128; ds41_ref.py "Numerics").

1. native calls no quantizer: under native every kernels.py quantizer raises, and the forward still runs; the same
   forward under official calls them (negative control: the patch is live).
2. native and official agree to within the fp8 / fp4 rounding: rel = ||native - official|| / ||official|| per block
   output, one-shot prefill of the canonical prompt's first DS41_NUMERICS_SEQ tokens (default 4096) over the spec's
   layers, printed (the margins F128 uses). Layer 0's first steps see identical inputs, so their rel is one rounding's
   size and is bounded tightly (L0_STEP); deeper layers compound it through routing and top-k flips (bounded loosely).

Measured at 4096 tokens, layers 0 1 2 3 14 20 21 24 + final_norm (FIX.6, BREADCRUMBS):
    real checkpoint (BRINGUP_SCALED=0)   .028 .027 .031 .036 .091 .141 .031 .020 | .289   L0 q_resid .0115, kv .0289
    scaled quietbox_2x2                  .028 .035 .053 .074 .121 .185 .172 .198 | .214   L0 q_resid .0115, kv .0289

    BRINGUP_SCALED=0 scripts/run_safe_pytest.sh --no-precompile --run-all \
        models/demos/deepseek_v41_flash_d_p/reference/test_numerics.py
(--no-precompile: the wrapper's up-front collect pass stubs parts of the run, its numbers are not the model's.)
"""

from __future__ import annotations

import os

import pytest
import torch

from models.demos.deepseek_v41_flash_d_p.reference import kernels as K

SEQ = int(os.environ.get("DS41_NUMERICS_SEQ", "4096"))
SHORT = 512  # tokens of the quantizer-free check (covers a ratio-2 compressor group and the window ring)
QUANTIZERS = ("act_quant", "act_quant_dequant", "fp4_quant_dequant", "fp4_act_quant")
# Bounds (measured values in the docstring). A logic slip in the switch (a step skipped, a scale lost) moves a step
# by O(1); a quantizer left on under native, or one dropped under official, moves layer 0's first steps off one
# rounding's size.
L0_STEP = {"L0.q_resid": (0.008, 0.016), "L0.kv": (0.020, 0.040)}  # one fp8 q-dq (q_a input); two (wkv input, kv)
REL_MAX = 0.3  # every block output (max measured 0.198)
FINAL_MAX = 0.4  # final_norm (max measured 0.289: per-token rms_norm of the stream collapse)


def _spec():
    from models.demos.common.bringup.reference.golden import load_spec

    if not os.environ.get("BRINGUP_SPEC"):
        pytest.skip("BRINGUP_SPEC not set")
    return load_spec(None)


@pytest.fixture(scope="module")
def setup():
    from models.demos.common.bringup.core.metrics import cpu_threads
    from models.demos.common.bringup.reference.golden import text_tokens

    torch.set_num_threads(cpu_threads())
    spec = _spec()
    official = spec.hooks().reference(spec, layers=spec.layers(), dtype=torch.float32, numerics="official")
    assert official.numerics == "official"
    native = official.with_numerics("native")
    return spec, official, native, text_tokens(spec, SEQ)


def _forward(ref, tokens):
    outs = {}

    def rec(name, t):
        if name.endswith(".out") or name == "final_norm" or name in L0_STEP:
            outs[name] = t.detach().float().clone()

    state = ref.new_state(tokens.shape[0])
    ref.forward_chunk(tokens, 0, state, rec, logits_last_n=0)
    return outs


def _patched(monkeypatch, fn):
    for n in QUANTIZERS:
        monkeypatch.setattr(K, n, fn(n, getattr(K, n)))


@pytest.mark.timeout(3600)
def test_native_calls_no_quantizer(setup, monkeypatch):
    _, official, native, tokens = setup
    calls = {n: 0 for n in QUANTIZERS}

    def counting(n, f):
        def g(*a, **k):
            calls[n] += 1
            return f(*a, **k)

        return g

    _patched(monkeypatch, counting)
    _forward(official, tokens[:SHORT])
    print(f"official quantizer calls at {SHORT} tokens: {calls}")
    assert calls["act_quant_dequant"] > 0 and calls["fp4_quant_dequant"] > 0, calls

    def raising(n, f):
        def g(*a, **k):
            raise AssertionError(f"native called kernels.{n}")

        return g

    _patched(monkeypatch, raising)
    outs = _forward(native, tokens[:SHORT])
    assert outs and all(torch.isfinite(t).all() for t in outs.values())


@pytest.mark.timeout(7200)
def test_native_vs_official(setup):
    spec, official, native, tokens = setup
    a = _forward(official, tokens)
    b = _forward(native, tokens)
    rels = {}
    for n, x in a.items():
        rels[n] = float((b[n] - x).norm() / x.norm())
    print(f"native vs official, {tokens.shape[0]} tokens, one-shot (rel = ||native - official|| / ||official||):")
    for n, r in rels.items():
        print(f"  {n:12s} rel {r:.6f}")
    for n, (lo, hi) in L0_STEP.items():
        assert lo < rels[n] < hi, (n, rels[n], lo, hi)
    for n, r in rels.items():
        assert 0 < r < (FINAL_MAX if n == "final_norm" else REL_MAX if n.endswith(".out") else 1.0), (n, r)
