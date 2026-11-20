# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference's official numerics at the GPU's dtype (FIX.7; F128 / F134 / F136; ds41_ref.py
"Numerics": official at dtype bf16 keeps every activation in bf16 where DeepSeek's GPU code does).

1. Per step: DS41Reference(numerics="official", dtype=bf16) run on the oracle's own inputs of every step equals the
   oracle's output. The oracle is DeepSeek's inference/model.py at bf16 (the GPU configuration of generate.py, what
   hooks.hf_model(spec, None) builds; here with the first max(LAYERS) + 1 blocks, which is the same code and weights
   for those blocks), tapped per step by reference/official_taps.capture over a one-shot prefill of SHORT tokens.
   Bound: rel <= STEP_ROUNDINGS bf16 roundings (measured 0: bit for bit at layers 0-3); the top-k of the indexer
   overlaps the oracle's >= TOPK_OVERLAP over the rows that see a key. Control: the fp32 official reference on the
   same inputs misses its bf16 points (L0 attn_norm one rounding, rel 1.7e-3): the bound has teeth.
2. End to end: the block outputs (the 4 streams) of a one-shot forward agree with the oracle's taps within
   LAYER_ROUNDINGS bf16 roundings per layer (measured 0: bit for bit).
3. native is unchanged by the dtype: the native reference derived from the bf16 one (with_numerics) is fp32 and gives
   the same bytes as a native reference built at fp32; native at bf16 is refused.
4. The new margins (report, printed): rel = ||official - native|| / ||native|| per block output and final_norm,
   one-shot over the canonical prompt's first DS41_NUMERICS_SEQ tokens (default 4096) of the spec's layers, for
   official at bf16 and (for comparison) at fp32. Bounded loosely (every layer < REL_MAX).

    BRINGUP_SCALED=0|1 scripts/run_safe_pytest.sh --no-precompile --run-all \\
        models/demos/deepseek_v41_flash_d_p/reference/test_official_bf16.py
(--no-precompile: reference/test_numerics.py's gotcha; the wrapper's collect pass stubs parts of the run.)
"""

from __future__ import annotations

import os

import pytest
import torch

SHORT = int(os.environ.get("DS41_OFFICIAL_SHORT", "512"))  # covers the 128 window and ratio-2 compressor groups
SEQ = int(os.environ.get("DS41_NUMERICS_SEQ", "4096"))
LAYERS = [0, 1, 2, 3]  # window-only, engram, KV / index source (ratio 2), its reader
BF16_ROUNDING = 2.0**-9  # rel L2 of one bf16 rounding (round to nearest, 8 significant bits: ~2^-9 of the norm)
STEP_ROUNDINGS = 0.5  # per step on the oracle's inputs (measured 0)
LAYER_ROUNDINGS = 3.0  # per block output, end to end ("a few bf16 roundings"; measured 0)
TOPK_OVERLAP = 0.99
REL_MAX = 0.3  # report bound for official vs native (FIX.6 measured <= 0.198 at fp32)
FINAL_MAX = 0.4


def _spec():
    from models.demos.common.bringup.reference.golden import load_spec

    if not os.environ.get("BRINGUP_SPEC"):
        pytest.skip("BRINGUP_SPEC not set")
    return load_spec(None)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(b.shape), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over the rows where b selects anything of |a & b| / |b| (-1 = none)."""
    ov = []
    for ra, rb in zip(a.long(), b.long()):
        sb = set(rb[rb >= 0].tolist())
        if sb:
            ov.append(len(set(ra[ra >= 0].tolist()) & sb) / len(sb))
    return sum(ov) / max(len(ov), 1)


@pytest.fixture(scope="module")
def setup():
    from models.demos.common.bringup.core.metrics import cpu_threads
    from models.demos.common.bringup.reference.golden import hf_path, text_tokens
    from models.demos.deepseek_v41_flash_d_p.reference.official_taps import capture

    torch.set_num_threads(cpu_threads())
    spec = _spec()
    path = hf_path(spec)
    tokens = text_tokens(spec, max(SEQ, SHORT))
    taps = capture(path, LAYERS, tokens[:SHORT], 0, SHORT)
    ref = spec.hooks().reference(spec, layers=spec.layers(), dtype=torch.bfloat16, numerics="official")
    assert ref.bf16 and ref.dtype == torch.bfloat16
    return spec, ref, tokens, taps


def _steps(ref, taps, tokens):
    """Every step of LAYERS on the oracle's inputs (one fresh state, layers in order: readers see their source's)."""
    from models.demos.common.bringup.reference.interface import Ctx

    st, out = ref.new_state(SHORT), {}
    for L in LAYERS:
        for s in ref.block_graph(L):
            tp = taps[L][s.name]
            ctx = Ctx(L, 0, SHORT, st)
            ctx.extra["tokens"] = tokens[:SHORT]
            out[(L, s.name)] = (ref.component(L, s.name)(ctx, *[x.clone() for x in tp["inputs"]]), tp["output"])
    return out


@pytest.mark.timeout(3600)
def test_steps_match_oracle(setup):
    _, ref, tokens, taps = setup
    print(f"official bf16 vs the oracle (model.py at bf16) per step on the oracle's inputs, {SHORT} tokens:")
    failed = []
    for (L, name), (got, want) in _steps(ref, taps, tokens).items():
        if want.is_floating_point():
            r = rel(got.float(), want.float())
            eq = float((got.float().reshape(want.shape) == want.float()).float().mean())
            ok = r <= STEP_ROUNDINGS * BF16_ROUNDING
            print(f"  L{L} {name:14s} rel {r:.3e} ({r / BF16_ROUNDING:.3f} roundings) equal {eq:.6f}")
        else:
            o = overlap(got, want)
            ok = o >= TOPK_OVERLAP
            print(f"  L{L} {name:14s} top-k overlap {o:.6f}")
        if not ok:
            failed.append((L, name))
    assert not failed, failed
    # control: the fp32 official reference misses the bf16 points model.py rounds at (the test sees a dtype slip)
    ctrl = _steps(ref.with_numerics("official", torch.float32), taps, tokens)
    got, want = ctrl[(0, "attn_norm")]
    r = rel(got, want)
    print(f"  control: official fp32, L0 attn_norm rel {r:.3e} ({r / BF16_ROUNDING:.3f} roundings)")
    assert r > STEP_ROUNDINGS * BF16_ROUNDING, r


@pytest.mark.timeout(3600)
def test_layers_match_oracle(setup):
    _, ref, tokens, taps = setup
    outs = {}

    def rec(name, t):
        if name.endswith(".out"):
            outs[name] = t.detach().clone()

    sub = ref.with_numerics("official")
    sub.layer_ids = LAYERS
    sub.forward_chunk(tokens[:SHORT], 0, sub.new_state(SHORT), rec)
    print(f"official bf16 vs the oracle, block outputs (4 streams), one-shot {SHORT} tokens:")
    worst = 0.0
    for L in LAYERS:
        got, want = outs[f"L{L}.out"][:, :-4], taps[L]["ffn_residual"]["output"][:, :-4]
        r = rel(got, want)
        worst = max(worst, r / BF16_ROUNDING)
        print(f"  L{L} rel {r:.3e} ({r / BF16_ROUNDING:.3f} roundings) equal {float((got == want).float().mean()):.6f}")
    assert worst <= LAYER_ROUNDINGS, worst


@pytest.mark.timeout(3600)
def test_native_unchanged(setup):
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference

    spec, ref, tokens, _ = setup
    native = ref.with_numerics("native")
    assert native.dtype == torch.float32 and not native.bf16
    built = spec.hooks().reference(spec, layers=LAYERS, dtype=torch.float32, numerics="native")
    with pytest.raises(AssertionError):
        DS41Reference(built.ck.path, layers=[0], dtype=torch.bfloat16, numerics="native")

    def run(r):
        outs = {}
        r.forward_chunk(tokens[:SHORT], 0, r.new_state(SHORT), lambda n, t: outs.__setitem__(n, t.detach().clone()))
        return outs

    native.layer_ids = LAYERS
    a, b = run(native), run(built)
    assert a.keys() == b.keys()
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, bad[:5]
    assert all(v.dtype != torch.bfloat16 for v in a.values())


@pytest.mark.timeout(7200)
def test_margins_report(setup):
    spec, ref, tokens, _ = setup
    toks = tokens[:SEQ]

    def run(r):
        outs = {}

        def rec(name, t):
            if name.endswith(".out") or name == "final_norm":
                outs[name] = t.detach().float().clone()

        r.forward_chunk(toks, 0, r.new_state(SEQ), rec)
        return outs

    nat = run(ref.with_numerics("native"))
    b16 = run(ref)
    f32 = run(ref.with_numerics("official", torch.float32))
    print(f"official vs native, {SEQ} tokens, one-shot, rel = ||official - native|| / ||native||:")
    print(f"  {'boundary':12s} {'bf16':>9s} {'fp32':>9s}")
    for n in nat:
        print(f"  {n:12s} {rel(b16[n], nat[n]):9.5f} {rel(f32[n], nat[n]):9.5f}")
    for n in nat:
        assert 0 < rel(b16[n], nat[n]) < (FINAL_MAX if n == "final_norm" else REL_MAX), n
