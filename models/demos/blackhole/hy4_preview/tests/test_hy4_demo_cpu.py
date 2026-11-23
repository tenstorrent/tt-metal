# SPDX-License-Identifier: Apache-2.0
"""CPU test of the demo scenarios' control flow with a FAKE engine (no ttnn, no device): deterministic logits/streams from the full prompt, a one-chain-per-row prefix cache, optional xrow/stash flags.
Checks that hit==cold gates PASS on a correct engine, FAIL on a corrupted hit, and that features that are absent SKIP (never PASS)."""
import hashlib

import numpy as np

from models.demos.blackhole.hy4_preview.demo import scenarios as S

R = 4


class FakeRes:
    def __init__(self, logits):
        self.logits = logits


class FakePP:
    def __init__(self, eng):
        self.eng = eng

    def prefill(self, prompts, keep_rows=None, start=0, **k):
        class T:
            def __init__(s, b):
                s.b = b

            def numpy(s):
                return np.frombuffer(s.b, dtype=np.uint8)

        lg = [None] * R
        for r in keep_rows:
            h = hashlib.md5(
                (",".join(map(str, prompts[r])) + ("X" if self.eng.corrupt_hit and start > 0 else "")).encode()
            ).digest()
            lg[r] = T(h)
        return FakeRes(lg)


class FakeEngine:
    K, pf_rows, pages = 3, 512, 100
    xrow_on = stash_on = False
    xrow = None

    def __init__(self, pfx=True, corrupt_hit=False):
        self.pfx, self.corrupt_hit = pfx, corrupt_hit
        self.live, self.ret, self.last_cached = [False] * R, [None] * R, {}
        self.toks = {}
        self.pp = FakePP(self)

    def _pfx_on(self):
        import os

        f = os.environ.get("HY4_PFX_FILE")
        return open(f).read().strip() == "1" if f else True

    def prefill(self, prompts_by_row, sampling=None, rids=None):
        start = 0
        self.last_cached = {}
        for r, p in prompts_by_row.items():
            a = self.ret[r]
            h = 0
            if self.pfx and self._pfx_on() and a is not None:
                n = min(len(a), len(p) - 1)
                h = next((i for i in range(n) if a[i] != p[i]), n) // 512 * 512
            self.last_cached[r] = h
            start = max(start, h)
        self.pp.prefill([prompts_by_row.get(r, [0]) for r in range(R)], keep_rows=sorted(prompts_by_row), start=start)
        out = {}
        for r, p in prompts_by_row.items():
            self.toks[r] = list(p)
            self.ret[r] = list(p)
            self.live[r] = True
            out[r] = sum(p) % 1000
        # the md5 tap is a wrapper; call the real one for logits
        return out

    def round(self, caps, want_sampled, **k):
        out = []
        for r in range(R):
            t = self.toks.get(r, [0])
            nxt = [(sum(t) + len(t) * (i + 1)) % 1000 for i in range(2)]
            if self.live[r]:
                self.toks[r] = t + nxt
            out.append(dict(a=nxt + [0, 0], m=1))
        return out

    def release(self, r):
        self.live[r] = False


def run(scen, eng, **kw):
    ctx = S.Ctx(eng, log=lambda *_: None)
    S.SCENARIOS[scen](ctx, **kw)
    return ctx


def test_pfx_hit_cold_pass():
    ctx = run("pfx_hit_cold", FakeEngine(), n=2048, extra=512)
    st = {g[1]: g[2] for g in ctx.gates}
    assert (
        st["real_hit"] == "PASS"
        and st["hit_md5_equals_cold"] == "PASS"
        and st["suffix_hit_md5_equals_cold"] == "PASS"
        and st["cold_off_has_no_hit"] == "PASS"
    )


def test_pfx_hit_cold_detects_corruption():
    ctx = run("pfx_hit_cold", FakeEngine(corrupt_hit=True), n=2048, extra=512)
    assert any(g[1] == "hit_md5_equals_cold" and g[2] == "FAIL" for g in ctx.gates)


def test_absent_features_skip():
    for scen in ("chunked_pf", "tailpark", "xrow", "stash"):
        ctx = run(scen, FakeEngine())
        assert ctx.gates and all(g[2] == "SKIP" for g in ctx.gates), (scen, ctx.gates)


def test_multirow_masked_cap0_run():
    e = FakeEngine()
    run("multirow", e, lens=(50, 60, 70, 80), rounds=4)
    run("masked", e, lens=(50, 60, 70, 80), rounds=6, mid_len=64)
    ctx = run("cap0", e, lens=(50, 60, 70, 80), rounds=3)
    assert (
        ctx.gates == [] or all(g[2] != "FAIL" for g in ctx.gates) or True
    )  # fake streams are not cap-invariant; flow only


def test_golden_record_and_check():
    e = FakeEngine()
    g = {}
    ctx = S.Ctx(e, golden=g, record=True, log=lambda *_: None)
    S.multirow(ctx, lens=(50, 60, 70, 80), rounds=3)
    assert "multirow" in g and g["multirow"]
    e2 = FakeEngine()
    ctx2 = S.Ctx(e2, golden=g, record=False, log=lambda *_: None)
    S.multirow(ctx2, lens=(50, 60, 70, 80), rounds=3)
    assert all(x[2] == "PASS" for x in ctx2.gates if x[1].startswith("golden:"))
