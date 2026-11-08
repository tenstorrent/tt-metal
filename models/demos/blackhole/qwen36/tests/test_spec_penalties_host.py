# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the device penalties inside the fused MTP step (phase 3 part 2, PEN-1; design note 80 section 4 and 84).
Tests H14, H15, H16 of the plan, model side:

* the pen / update bodies of tt/spec_penalties.py run on a small torch emulation of the ttnn ops they use (``_Emu``), so the real op
  sequence (draft history, padding rows, row buckets, committed ids, overflow users) is exercised, not a copy of it;
* H14  the history of column j and the arithmetic equal the plugin reference (apply_speculative_penalties, when the plugin is on the
       path) and the vLLM op; the device model differs from the fp32 reference by at most one bf16 ulp on penalized tokens and by
       nothing elsewhere; a neutral row is bit for bit unchanged (also +-inf and -1e30);
* H15  the update trace keeps the counts equal to host counts over 1000 simulated steps with rejections, overflow users and host-walked
       steps (add_committed_host);
* H16  a rebuild writes exactly the named rows from host token lists (join, resume, slot move), a row bucket change does not touch the
       state, parse_penalties maps runner slots to engine rows;
* the engine on the fake device of test_spec_fused_engine_host.py: trace order (pen after the main trace, before topk; update after the
  accept), results equal to the host reference, fallback and overflow redo apply the penalties on the host and repair the counts, a
  greedy penalty row is refused, a step without penalties is untouched.

No device. Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_penalties_host.py
"""

import types

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FS,
    S_V,
    TOPK,
    VFY,
    K,
    T,
    _block,
    _plateau_logits,
    _pt,
    _SampledRig,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    PEN_ROWS,
    PenaltyDevice,
    apply_penalties_device_model,
    apply_penalties_exact,
    column_history_counts,
    counts_from_ids,
    mask_from_ids,
    parse_penalties,
    penalties_neutral,
    update_reference,
)

PEN, UPD = 21, 22  # trace ids of the fake pen trace and update trace


# --------------------------------------------------------------------------- #
# A torch emulation of the ttnn ops the bodies use
# --------------------------------------------------------------------------- #
class _Emu:
    float32, bfloat16, uint32, int32 = torch.float32, torch.bfloat16, torch.int64, torch.int32
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "tile", "rm", None

    class _Mapper:
        def __init__(self, *a, **k):
            pass

    ReplicateTensorToMesh = ShardTensorToMesh = ConcatMeshToTensor = _Mapper

    @staticmethod
    def from_torch(t, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        t = torch.as_tensor(t)
        return t.to(dtype).clone() if dtype is not None else t.clone()

    @staticmethod
    def copy_host_to_device_tensor(h, d):
        d.copy_(h.reshape(d.shape))

    @staticmethod
    def copy(src, dst):
        assert tuple(src.shape) == tuple(dst.shape), (tuple(src.shape), tuple(dst.shape))
        dst.copy_(src)

    @staticmethod
    def to_torch(t, mesh_composer=None):
        return t.clone()

    @staticmethod
    def synchronize_device(mesh):
        pass

    @staticmethod
    def deallocate(t):
        pass

    @staticmethod
    def typecast(x, dt):
        return x.to(dt)

    @staticmethod
    def to_layout(x, layout):
        return x

    @staticmethod
    def reshape(x, shape):
        return x.reshape(shape)

    @staticmethod
    def slice(x, starts, ends):
        return x[tuple(slice(a, b) for a, b in zip(starts, ends))]

    @staticmethod
    def pad(x, padding, value=0.0):
        flat = [v for lo_hi in reversed(padding) for v in lo_hi]
        return F.pad(x, flat, value=value)

    @staticmethod
    def transpose(x, d0, d1):
        return x.transpose(d0, d1)

    @staticmethod
    def _like(a, r):
        return r.to(a.dtype) if isinstance(a, torch.Tensor) else r.to(torch.float32)

    @classmethod
    def eq(cls, a, b):
        return cls._like(a, a == b)

    @classmethod
    def gt(cls, a, b):
        return cls._like(a, a > b)

    @classmethod
    def le(cls, a, b):
        return cls._like(a, a <= b)

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

    @staticmethod
    def multiply(a, b):
        return a * b

    @staticmethod
    def add_(a, b):
        a.add_(b)
        return a

    @staticmethod
    def where(c, x, y):
        return torch.where(c.bool(), x, y if isinstance(y, torch.Tensor) else torch.tensor(float(y), dtype=x.dtype))


EMU = _Emu()


def _device(B=8, V=96, seed=0, rows_state=True):
    tok = torch.zeros(1, B * T, dtype=torch.int32)
    pen = PenaltyDevice.allocate("mesh", B, K, V, V, tok, ops=EMU)
    return pen


def _bf16_ulp(v):
    v = v.double().abs().clamp_min(1e-30)
    return torch.pow(2.0, torch.floor(torch.log2(v)) - 7)


def _random_state(pen, g, density=0.15):
    """Random counts / prompt mask for the 32 rows (rows of one user equal, as the engine keeps them)."""
    V, B = pen.vocab, pen.B
    cnt = torch.zeros(B, V)
    msk = torch.zeros(B, V)
    for u in range(B):
        cnt[u] = torch.where(
            torch.rand(V, generator=g) < density,
            torch.randint(1, 5, (V,), generator=g),
            torch.zeros(V, dtype=torch.int64),
        ).float()
        msk[u] = (torch.rand(V, generator=g) < density).float()
    pen.counts.zero_()
    pen.mask.zero_()
    pen.counts.reshape(PEN_ROWS, V)[: B * T] = cnt.repeat_interleave(T, dim=0)
    pen.mask.reshape(PEN_ROWS, V)[: B * T] = msk.repeat_interleave(T, dim=0).to(torch.bfloat16)
    return cnt, msk


def _set_params(pen, rep, freq, pres):
    ps = types.SimpleNamespace(
        rep=torch.as_tensor(rep, dtype=torch.float32),
        freq=torch.as_tensor(freq, dtype=torch.float32),
        pres=torch.as_tensor(pres, dtype=torch.float32),
    )
    pen.stage_params(ps)


# --------------------------------------------------------------------------- #
# H14: the history, the arithmetic, the neutral rows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b", [1, 2, 4, 8])
def test_apply_body_equals_the_device_model_per_column_and_leaves_neutral_rows_alone(b):
    B, V = 8, 96
    pen = _device(B, V)
    g = torch.Generator().manual_seed(b)
    cnt, msk = _random_state(pen, g)
    rep = torch.tensor([1.3, 1.0, 1.0, 0.8, 1.0, 1.1, 1.0, 1.2])
    freq = torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.0])
    pres = torch.tensor([0.2, 0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0])
    _set_params(pen, rep, freq, pres)
    R = b * T
    toks = torch.randint(0, V, (1, B * T), generator=g, dtype=torch.int32)
    pen.tok_in = toks  # the live verify token buffer is B * T wide; a bucket step slices its first R words
    x = (torch.randn(1, 1, R, V, generator=g) * 3).to(torch.bfloat16)
    x0 = x.clone()
    pen.apply_body(x)
    tk = toks.reshape(B, T)
    for u in range(b):
        for j in range(T):
            r = u * T + j
            hist = column_history_counts(cnt[u], tk[u, 1:], K, T, device_rule=True)[j]
            want = apply_penalties_device_model(
                x0[0, 0, r : r + 1], hist[None], msk[u][None], rep[u : u + 1], freq[u : u + 1], pres[u : u + 1]
            )
            assert torch.equal(x[0, 0, r : r + 1], want), (b, u, j)
        if bool(penalties_neutral(rep[u], freq[u], pres[u])):
            assert torch.equal(x[0, 0, u * T : (u + 1) * T], x0[0, 0, u * T : (u + 1) * T]), f"neutral row {u} changed"
    assert int((x != x0).sum()) > 0, "no penalized row changed: the test proves nothing"


def test_neutral_parameters_are_a_bit_exact_no_op_also_for_infinities_and_the_mask_floor():
    pen = _device(8, 64)
    g = torch.Generator().manual_seed(3)
    _random_state(pen, g, density=0.5)
    _set_params(pen, [1.0] * 8, [0.0] * 8, [0.0] * 8)
    pen.tok_in = torch.randint(0, 64, (1, 32), generator=g, dtype=torch.int32)
    x = (torch.randn(1, 1, 32, 64, generator=g) * 3).to(torch.bfloat16)
    x[0, 0, 0, :4] = torch.tensor([float("-inf"), -1e30, -6.338e29, float("inf")], dtype=torch.bfloat16)
    x[0, 0, 3, 5] = 0.0
    x0 = x.clone()
    pen.apply_body(x)
    assert torch.equal(x, x0)


def test_device_model_is_within_one_bf16_ulp_of_the_fp32_reference_and_exact_elsewhere():
    g = torch.Generator().manual_seed(11)
    R, V = 24, 4000
    x = (torch.randn(R, V, generator=g) * 4).to(torch.bfloat16)
    cnt = torch.where(
        torch.rand(R, V, generator=g) < 0.2,
        torch.randint(1, 6, (R, V), generator=g),
        torch.zeros(R, V, dtype=torch.int64),
    ).float()
    msk = (torch.rand(R, V, generator=g) < 0.2).float()
    rep = 1.0 + 0.5 * torch.rand(R, generator=g)
    freq = 0.8 * torch.rand(R, generator=g)
    pres = torch.rand(R, generator=g)
    got = apply_penalties_device_model(x, cnt, msk, rep, freq, pres)
    ref = apply_penalties_exact(x, cnt, msk, rep, freq, pres)
    diff = (got.double() - ref.double()).abs()
    penalized = (cnt > 0) | (msk > 0)
    assert bool(
        (diff[penalized] <= 1.0 * _bf16_ulp(ref)[penalized]).all()
    ), "more than one bf16 ulp off the fp32 reference"
    assert torch.equal(got[~penalized], x[~penalized]), "an unpenalized token changed"
    # the half-ulp claim: bf16(reference) is the nearest bf16; the device is that or one step away
    near = ref.to(torch.bfloat16)
    assert float((got != near)[penalized].float().mean()) < 0.01


def test_reference_equals_the_plugin_reference_per_column():
    """H14: column j scores committed output + drafts[:min(j, valid)] (plugin apply_speculative_penalties, the vLLM op)."""
    sa = pytest.importorskip("vllm_tt_plugin.spec_accept")
    g = torch.Generator().manual_seed(5)
    rows, V = 3, 80
    logits = torch.randn(rows, T, V, generator=g) * 2
    drafts = torch.randint(0, V, (rows, K), generator=g)
    nvd = torch.tensor([3, 1, 0])
    outs = [torch.randint(0, V, (n,), generator=g).tolist() for n in (5, 0, 9)]
    prompts = [torch.randint(0, V, (n,), generator=g).tolist() for n in (4, 6, 2)]
    pad = max(len(p) for p in prompts)
    prompt_t = torch.tensor([p + [V] * (pad - len(p)) for p in prompts], dtype=torch.int64)
    rep, freq, pres = torch.tensor([1.2, 1.0, 0.9]), torch.tensor([0.3, 0.0, 0.1]), torch.tensor([0.0, 0.6, 0.2])
    pen = sa.SpecPenalties(
        prompt_token_ids=prompt_t, output_token_ids=outs, presence=pres, frequency=freq, repetition=rep
    )
    try:
        want = sa.apply_speculative_penalties(logits, drafts, nvd, pen)
    except (NotImplementedError, RuntimeError) as e:  # the vLLM custom op may be missing on a CPU build
        pytest.skip(f"vLLM penalty op unavailable: {e}")
    for u in range(rows):
        hist = column_history_counts(counts_from_ids(outs[u], V), drafts[u], int(nvd[u]), T)
        got = apply_penalties_exact(
            logits[u],
            hist,
            mask_from_ids(prompts[u], V).reshape(1, -1).expand(T, -1),
            rep[u].expand(T),
            freq[u].expand(T),
            pres[u].expand(T),
        )
        assert torch.allclose(got, want[u], atol=1e-5, rtol=1e-6), (u, (got - want[u]).abs().max())


def test_device_history_rule_agrees_with_the_plugin_rule_on_every_column_the_walk_reads():
    cnt = counts_from_ids([1, 2, 2], 10)
    drafts = [7, 8, 9]
    for nv in range(K + 1):
        dev = column_history_counts(cnt, drafts, nv, T, device_rule=True)
        plug = column_history_counts(cnt, drafts, nv, T)
        assert torch.equal(
            dev[: nv + 1], plug[: nv + 1]
        )  # columns 0..nv: the only ones accept / correction / bonus read


# --------------------------------------------------------------------------- #
# H15: the update trace over 1000 steps
# --------------------------------------------------------------------------- #
def test_update_trace_counts_equal_host_counts_over_1000_steps_with_rejections_overflow_and_host_walks():
    B, V = 8, 50
    pen = _device(B, V)
    g = torch.Generator().manual_seed(15)
    host = [torch.zeros(V) for _ in range(B)]
    W = 32
    for step in range(1000):
        committed = torch.full((B, T), -1, dtype=torch.int32)
        counts = torch.randint(1, T + 1, (B,), generator=g)  # 1 = every draft rejected
        ovf = (torch.rand(B, generator=g) < 0.05).to(torch.int32) * torch.randint(
            1, 4, (B,), generator=g, dtype=torch.int32
        )
        for u in range(B):
            committed[u, : int(counts[u])] = torch.randint(0, V, (int(counts[u]),), generator=g, dtype=torch.int32)
        res = torch.zeros(3, W, dtype=torch.int32)
        res[0, : B * T] = committed.reshape(-1)
        res[1, :B] = counts.to(torch.int32)
        res[2, :B] = ovf
        pen.bind_result(res)
        pen.update_body()
        redo = [u for u in range(B) if int(ovf[u])]
        if redo:  # the host redo walk commits something else: the engine adds it with add_committed_host
            true = torch.full((B, T), -1, dtype=torch.int32)
            for u in redo:
                n = int(torch.randint(1, T + 1, (1,), generator=g))
                true[u, :n] = torch.randint(0, V, (n,), generator=g, dtype=torch.int32)
            pen.add_committed_host(true, redo)
            for u in redo:
                committed[u] = true[u]
        for u in range(B):
            for t in committed[u].tolist():
                if t >= 0:
                    host[u][t] += 1.0
    dev = pen.counts.reshape(PEN_ROWS, V)
    for u in range(B):
        for j in range(T):
            assert torch.equal(dev[u * T + j], host[u]), (u, j)
    assert pen.stats["repairs"] > 0


def test_update_reference_matches_the_trace_on_a_single_step():
    B, V = 8, 30
    pen = _device(B, V)
    g = torch.Generator().manual_seed(2)
    committed = torch.randint(-1, V, (B, T), generator=g, dtype=torch.int32)
    ovf = torch.tensor([0, 1, 0, 0, 2, 0, 0, 0], dtype=torch.int32)
    res = torch.zeros(3, 32, dtype=torch.int32)
    res[0, : B * T], res[2, :B] = committed.reshape(-1), ovf
    pen.bind_result(res)
    pen.update_body()
    assert torch.equal(pen.counts.reshape(PEN_ROWS, V), update_reference(torch.zeros(PEN_ROWS, V), committed, ovf))


# --------------------------------------------------------------------------- #
# H16: rebuild, remap, bucket change
# --------------------------------------------------------------------------- #
def test_rebuild_writes_exactly_the_named_rows_and_nothing_else(expect_error):
    B, V = 8, 40
    pen = _device(B, V)
    g = torch.Generator().manual_seed(8)
    cnt, msk = _random_state(pen, g)
    before_c, before_m = pen.counts.clone(), pen.mask.clone()
    prompt, out = [3, 4, 4, 9], [5, 5, 5, 6, 39]
    pen.rebuild({2: (torch.tensor(prompt), torch.tensor(out))})
    c, m = pen.counts.reshape(PEN_ROWS, V), pen.mask.reshape(PEN_ROWS, V)
    for r in range(PEN_ROWS):
        if 2 * T <= r < 3 * T:
            assert torch.equal(c[r], counts_from_ids(out, V)) and torch.equal(m[r].float(), mask_from_ids(prompt, V))
        else:
            assert torch.equal(c[r], before_c.reshape(PEN_ROWS, V)[r]) and torch.equal(
                m[r], before_m.reshape(PEN_ROWS, V)[r]
            )
    # a second rebuild (resume after preemption: prompt + the output so far) replaces the first one
    pen.rebuild({2: (torch.tensor(prompt), torch.tensor(out + [7, 7]))})
    assert torch.equal(pen.counts.reshape(PEN_ROWS, V)[2 * T], counts_from_ids(out + [7, 7], V))
    assert pen.stats["rebuilds"] == 2 and pen.stats["rebuild_rows"] == 2
    with expect_error(ValueError, "outside range"):
        pen.rebuild({B: (torch.tensor([1]), torch.tensor([1]))})


def test_a_row_bucket_change_and_a_slot_move_do_not_touch_the_state():
    """The state is per engine row: a bucket b reads rows 0 .. b*T-1 and writes none; a runner slot move changes only the slot -> row map."""
    B, V = 8, 64
    pen = _device(B, V)
    g = torch.Generator().manual_seed(4)
    _random_state(pen, g)
    _set_params(pen, [1.2] * 8, [0.1] * 8, [0.2] * 8)
    toks = torch.randint(0, V, (1, B * T), generator=g, dtype=torch.int32)
    pen.tok_in = toks
    snap = (pen.counts.clone(), pen.mask.clone())
    xs = {}
    x_full = (torch.randn(1, 1, B * T, V, generator=g) * 3).to(torch.bfloat16)
    for b in (8, 4, 2, 1, 8):
        x = x_full[:, :, : b * T].clone()
        pen.apply_body(x)
        xs.setdefault(b, x)
        assert torch.equal(pen.counts, snap[0]) and torch.equal(pen.mask, snap[1]), f"bucket {b} edited the state"
    # a bucket's penalized logits equal the first rows of the full-width result
    full = x_full.clone()
    pen.apply_body(full)
    for b in (4, 2, 1):
        assert torch.equal(xs[b], full[:, :, : b * T]), b


def test_parse_penalties_maps_slots_to_engine_rows_and_ignores_padding_rows(expect_error):
    B, V = 4, 32
    phys = [2, 0, 3, 1]  # runner slot s -> engine row phys[s]
    inv = [phys.index(r) for r in range(B)]
    to_rows = lambda *xs: tuple(torch.as_tensor(x)[torch.tensor(inv)] for x in xs)  # noqa: E731
    pen = {
        "repetition": torch.tensor([1.3, 1.0, 1.0, 1.0]),
        "frequency": torch.tensor([0.0, 0.0, 0.5, 0.0]),
        "presence": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "rebuild": {0: (torch.tensor([1, 2]), torch.tensor([3])), 2: (torch.tensor([4]), torch.tensor([5, 5]))},
    }
    ps = parse_penalties(pen, B, to_rows, active=torch.ones(B, dtype=torch.bool))
    assert ps.rep[phys[0]] == pytest.approx(1.3) and ps.freq[phys[2]] == pytest.approx(0.5)
    assert ps.live.tolist() == [r in (phys[0], phys[2]) for r in range(B)]
    assert sorted(ps.rebuild) == sorted([phys[0], phys[2]])
    assert ps.rebuild[phys[2]][1].tolist() == [5, 5]
    ps2 = parse_penalties(pen, B, to_rows, active=torch.tensor([False, True, True, True]))  # engine row 0 is padding
    assert not bool(ps2.live[0]) and float(ps2.rep[0]) == 1.0
    with expect_error(ValueError, "repetition > 0"):
        parse_penalties({**pen, "repetition": torch.tensor([0.0, 1.0, 1.0, 1.0])}, B, to_rows)
    with expect_error(ValueError, "outside range"):
        parse_penalties({**pen, "rebuild": {9: (torch.tensor([1]), torch.tensor([1]))}}, B, to_rows)


def test_allocate_refuses_shapes_the_traces_cannot_serve(expect_error):
    with expect_error(ValueError, "B \\* T <= 32"):
        PenaltyDevice.allocate("mesh", 9, K, 16, 16, torch.zeros(1, 36, dtype=torch.int32), ops=EMU)
    with expect_error(ValueError, "exact in fp32"):
        PenaltyDevice.allocate("mesh", 2, K, 1 << 24, 1 << 24, torch.zeros(1, 8, dtype=torch.int32), ops=EMU)


# --------------------------------------------------------------------------- #
# The engine on the fake device
# --------------------------------------------------------------------------- #
class _PenRig(_SampledRig):
    """_SampledRig + the penalty state (the real PenaltyDevice over the torch emulation), a fake pen trace that edits the logits the
    verify 'computed' before the topk trace, and a fake update trace. ``set_logits`` keeps the raw logits."""

    def __init__(self, monkeypatch, B):
        super().__init__(monkeypatch, B)
        self.m._vfy_pen_trace_id = PEN
        self.e._pen_upd_trace = UPD
        self.pen = PenaltyDevice.allocate("mesh", B, K, S_V, S_V, torch.zeros(1, B * T, dtype=torch.int32), ops=EMU)
        self.e._pen = self.pen
        self.raw = None

    def set_logits(self, logits):
        self.raw = logits.to(torch.bfloat16)
        return super().set_logits(logits)

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == PEN:
            self.pen.tok_in = self.m._vfy_token_buf.data.reshape(1, -1)
            x = self.raw.reshape(1, 1, self.R, S_V).clone()
            self.pen.apply_body(x)
            _SampledRig.set_logits(
                self, x.reshape(self.R, S_V)
            )  # the topk trace and the accept read the penalized logits
        if tid == UPD:
            self.pen.bind_result(self.fs.result_out.data)
            self.pen.update_body()


def _pen_sampling(B, nvd, temps, pen_rows, *, rep=1.3, freq=0.0, pres=0.0, rebuild=None, topk=3):
    """The plugin's spec_sampling for a step with penalty rows: a greedy penalty row is a sampled row with temperature 1, top_k 1, top_p 0."""
    t = torch.tensor(temps, dtype=torch.float32)
    ss = {
        "temperature": torch.where(t > 0, t, torch.zeros(B)),
        "top_k": torch.where(t > 0, torch.full((B,), topk, dtype=torch.int32), torch.ones(B, dtype=torch.int32)),
        "top_p": torch.ones(B),
        "greedy": t <= 0,
        "uniforms": torch.zeros(B, K + 2),
    }
    r, f, p = torch.ones(B), torch.zeros(B), torch.zeros(B)
    for u in pen_rows:
        r[u], f[u], p[u] = rep, freq, pres
        if float(t[u]) <= 0:
            ss["temperature"][u], ss["top_k"][u], ss["top_p"][u], ss["greedy"][u] = 1.0, 1, 0.0, False
    ss["penalties"] = {"repetition": r, "frequency": f, "presence": p}
    if rebuild:
        ss["penalties"]["rebuild"] = rebuild
    return ss


def _expected_penalized_walk(raw, tok, nvd, outs, prompts, u, rep, freq, pres, exact):
    """Greedy walk of user u over the penalized logits (device model or fp32 exact): committed ids."""
    V = raw.shape[1]
    cnt = counts_from_ids(outs[u], V)
    hist = column_history_counts(cnt, tok[u, 1:], K, T, device_rule=True)
    msk = mask_from_ids(prompts[u], V).reshape(1, -1).expand(T, -1)
    rows = raw[u * T : (u + 1) * T].float()
    args = (rows, hist, msk, torch.full((T,), rep), torch.full((T,), freq), torch.full((T,), pres))
    pl = apply_penalties_exact(*args) if exact else apply_penalties_device_model(*args).float()
    am = [int((r == r.max()).nonzero()[0]) for r in pl]
    out, n = [], 0
    while n < int(nvd[u]) and am[n] == int(tok[u, n + 1]):
        n += 1
    return am[: n + 1]


def test_engine_pen_step_trace_order_result_and_counts(monkeypatch):
    B = 3
    r = _PenRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 31)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [3, 2, 1])
    # drafts follow the penalized argmax of row 1 (greedy penalty row) so some are accepted; rows 0 and 2 are plain greedy-like sampled rows
    outs = [[1, 2, 3], [5, 6, 7, 7], [0]]
    prompts = [[4, 5], [9, 10, 11], [3]]
    rebuild = {1: (torch.tensor(prompts[1]), torch.tensor(outs[1]))}
    ss = _pen_sampling(B, nvd, [0.0, 0.0, 0.0], pen_rows=[1], rep=1.4, freq=0.2, pres=0.1, rebuild=rebuild)
    # first pass: find what the penalized argmax of row 1 is, so the drafts agree with it
    want0 = _expected_penalized_walk(logits.to(torch.bfloat16), tok, nvd, outs, prompts, 1, 1.4, 0.2, 0.1, exact=False)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    trace = [x[1] for x in r.log if x[0] == "trace"]
    assert trace[:6] == [VFY, PEN, TOPK, FS, UPD, DRAFT], trace
    assert r.e.pen_steps == 1
    # the committed ids of the penalty row equal the host walk over the penalized logits (device model, bitwise)
    got = [int(t) for t in res.accepted_token_ids[1] if int(t) >= 0]
    assert got == want0[: len(got)] and len(got) == int(res.accepted_counts[1])
    # the device counts of every row: output + committed of this step (row 1 was rebuilt from outs[1]; other rows started at zero)
    c = r.pen.counts.reshape(PEN_ROWS, S_V)
    for u in range(B):
        base = counts_from_ids(outs[u], S_V) if u == 1 else torch.zeros(S_V)
        add = counts_from_ids([int(t) for t in res.accepted_token_ids[u] if int(t) >= 0], S_V)
        for j in range(T):
            assert torch.equal(c[u * T + j], base + add), (u, j)
    assert r.e.pen_steps == 1 and r.pen.stats["rebuilds"] == 1


def test_engine_step_without_penalties_replays_no_pen_trace(monkeypatch):
    B = 2
    r = _PenRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 7)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [2, 1])
    ss = _pen_sampling(B, nvd, [0.8, 0.0], pen_rows=[])
    r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    trace = [x[1] for x in r.log if x[0] == "trace"]
    assert PEN not in trace and UPD not in trace and r.e.pen_steps == 0
    del ss["penalties"]
    r.step(tok, pos + 1, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert PEN not in [x[1] for x in r.log if x[0] == "trace"]


def test_engine_second_step_without_rebuild_keeps_counting(monkeypatch):
    B = 2
    r = _PenRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 9)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [2, 2])
    rebuild = {0: (torch.tensor([1]), torch.tensor([2, 3]))}
    committed_all = []
    for step in range(3):
        ss = _pen_sampling(B, nvd, [0.0, 0.0], pen_rows=[0], rep=1.2, freq=0.1, rebuild=rebuild if step == 0 else None)
        res = r.step(tok, pos + 10 * step, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
        committed_all += [int(t) for t in res.accepted_token_ids[0] if int(t) >= 0]
        r.set_logits(logits)  # the verify 'computes' the same raw logits again
    want = counts_from_ids([2, 3] + committed_all, S_V)
    assert torch.equal(r.pen.counts.reshape(PEN_ROWS, S_V)[0], want)
    assert r.pen.stats["rebuilds"] == 1 and r.pen.stats["param_stages"] == 1, "unchanged parameters are staged once"


def test_engine_fallback_walks_the_penalized_logits_on_the_host_and_repairs_the_counts(monkeypatch):
    B = 2
    r = _PenRig(monkeypatch, B)
    r.e._fs_trace = None  # "no_sampled_trace": the whole step is walked on the host over the full logits
    logits = _smooth_logits(B * T, 13)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [3, 2])
    outs, prompts = [[8, 8, 9], [1]], [[2, 3], [4]]
    rebuild = {0: (torch.tensor(prompts[0]), torch.tensor(outs[0]))}
    ss = _pen_sampling(B, nvd, [0.0, 0.0], pen_rows=[0], rep=1.5, freq=0.3, pres=0.2, rebuild=rebuild)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert r.e.fused_fallbacks == {"no_sampled_trace": 1}
    assert PEN not in [x[1] for x in r.log if x[0] == "trace"] and UPD not in [x[1] for x in r.log if x[0] == "trace"]
    want = _expected_penalized_walk(logits.to(torch.bfloat16), tok, nvd, outs, prompts, 0, 1.5, 0.3, 0.2, exact=True)
    got = [int(t) for t in res.accepted_token_ids[0] if int(t) >= 0]
    assert got == want[: len(got)] and len(got) == int(res.accepted_counts[0])
    base = counts_from_ids(outs[0], S_V)
    assert torch.equal(
        r.pen.counts.reshape(PEN_ROWS, S_V)[0], base + counts_from_ids(got, S_V)
    ), "the host-walked commit is in the counts"
    assert r.e.pen_host_walks == 1
    # the caller's logits were not edited by the penalty
    assert torch.equal(r.m._vfy_logits_rm_out.data[0, 0, : r.R], logits.to(torch.bfloat16))


def test_engine_overflow_redo_applies_the_penalties_and_counts_the_redone_commit_once(monkeypatch):
    B = 2
    r = _PenRig(monkeypatch, B)
    logits = _plateau_logits(
        B * T, 5
    )  # two high logits then a plateau: top_k 3 ties reach the last candidate = overflow
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [2, 2])
    rebuild = {0: (torch.tensor([1]), torch.tensor([2]))}
    ss = _pen_sampling(B, nvd, [0.9, 0.9], pen_rows=[0], rep=1.05, rebuild=rebuild, topk=3)
    ss["uniforms"] = torch.full((B, K + 2), 0.5)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    assert r.e.fused_fallbacks.get("overflow_redo") == 1
    got = [int(t) for t in res.accepted_token_ids[0] if int(t) >= 0]
    want = counts_from_ids([2] + got, S_V)
    assert torch.equal(
        r.pen.counts.reshape(PEN_ROWS, S_V)[0], want
    ), "garbage of the overflowed device result was counted, or the redo was not"


def test_engine_refuses_a_greedy_penalty_row_and_penalties_without_traces(monkeypatch, expect_error):
    B = 2
    r = _PenRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 17)
    r.set_logits(logits)
    tok, pos, nvd = _block(B, [2, 2])
    ss = _pen_sampling(B, nvd, [0.0, 0.0], pen_rows=[0])
    ss["greedy"][0] = True
    with expect_error(ValueError, "marked greedy"):
        r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)
    ss = _pen_sampling(B, nvd, [0.0, 0.0], pen_rows=[0])
    r.e._pen = None
    with expect_error(ValueError, "no pen traces"):
        r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B), ss)


# --------------------------------------------------------------------------- #
# The adapter and the capability
# --------------------------------------------------------------------------- #
def _adapter(caps):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    cls = qv.Qwen36MTPContractForCausalLM
    fake = types.SimpleNamespace(_SPEC_K=K, model_capabilities=caps)
    fake._check_spec_penalties = lambda ss, B: cls._check_spec_penalties(fake, ss, B)
    return lambda ss, B=3: cls._check_spec_sampling(fake, ss, "fused_sample", B)


def test_adapter_gates_penalties_on_the_capability_and_checks_the_payload(expect_error):
    B = 3
    caps = {}
    chk = _adapter(caps)
    nvd = torch.full((B,), K)
    ss = _pen_sampling(
        B, nvd, [0.0, 0.8, 0.0], pen_rows=[0], rep=1.3, rebuild={0: (torch.tensor([1, 2]), torch.tensor([3]))}
    )
    with expect_error(ValueError, "supports_spec_device_penalties"):
        chk(ss)
    caps["supports_spec_device_penalties"] = True
    assert chk(ss) is ss
    # neutral values and no rebuild need no capability (an older plugin never sends the key; a newer one sends none for a neutral step)
    neutral = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[])
    caps.clear()
    assert chk(neutral) is neutral
    caps["supports_spec_device_penalties"] = True
    bad = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[0])
    bad["greedy"][0] = True
    with expect_error(ValueError, "marked greedy"):
        chk(bad)
    lp = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[1])
    lp["num_logprobs"] = torch.tensor([-2, 3, -2], dtype=torch.int32)
    caps["supports_spec_device_logprobs"] = True
    with expect_error(ValueError, "ask for logprobs and carry penalties"):
        chk(lp)
    lp["num_logprobs"] = torch.tensor(
        [2, -2, -2], dtype=torch.int32
    )  # a logprobs row without a penalty next to a penalty row: fine
    assert chk(lp) is lp
    for key, value in (
        ("repetition", torch.zeros(B)),
        ("frequency", torch.zeros(B, dtype=torch.float64)),
        ("presence", torch.zeros(2)),
    ):
        broken = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[0])
        broken["penalties"][key] = value
        with expect_error(ValueError, "penalties"):
            chk(broken)
    badrb = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[0], rebuild={7: (torch.tensor([1]), torch.tensor([1]))})
    with expect_error(ValueError, "rebuild"):
        chk(badrb)
    extra = _pen_sampling(B, nvd, [0.0, 0.8, 0.0], pen_rows=[0])
    extra["penalties"]["bias"] = torch.zeros(B)
    with expect_error(ValueError, "must be a dict"):
        chk(extra)


def test_the_capability_is_declared_from_the_kill_switches(monkeypatch):
    """Human-approved (2026-10-28): supports_spec_device_penalties is a plain bool decided at import by the fused switches and
    QWEN36_SPEC_DEVICE_PENALTIES (default on, model-side only)."""
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv
    from models.demos.blackhole.qwen36.tt import tp_common as tc

    assert isinstance(qv.Qwen36MTPContractForCausalLM.model_capabilities["supports_spec_device_penalties"], bool)

    def expected():
        return bool(tc.spec_fused_enabled() and tc.spec_fused_sampled_enabled() and tc.spec_device_penalties_enabled())

    for env, want in (
        ({"QWEN36_SPEC_FUSED": "1"}, True),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_DEVICE_PENALTIES": "0"}, False),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_FUSED_SAMPLED": "0"}, False),
        ({"QWEN36_SPEC_FUSED": "0"}, False),
    ):
        for key in ("QWEN36_SPEC_FUSED", "QWEN36_SPEC_FUSED_SAMPLED", "QWEN36_SPEC_DEVICE_PENALTIES"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert expected() is want, env
    monkeypatch.delenv("QWEN36_SPEC_FUSED", raising=False)
    monkeypatch.delenv("QWEN36_SPEC_DEVICE_PENALTIES", raising=False)
    assert tc.spec_device_penalties_enabled() is True, "the kill switch defaults to on"
