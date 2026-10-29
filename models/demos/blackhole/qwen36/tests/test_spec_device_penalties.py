# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the penalties inside the fused MTP step (phase 3 part 2, PEN-1; work/research_1026/80 section 4 and 84).
One TP mesh (MESH_DEVICE=P150x8), ONE pytest node per process (run_p8_device.sh). They need the pen traces and the update trace
(QWEN36_SPEC_FUSED=1, QWEN36_SPEC_DEVICE_PENALTIES default 1). They drive MTPSpecEngine.decode_forward directly (no plugin): a penalty
row is a SAMPLED row in ``spec_sampling`` (a greedy request as temperature 1, top_k 1, top_p 0) plus ``spec_sampling["penalties"]``.
THESE TESTS WERE WRITTEN ON A HOST WITHOUT DEVICE ACCESS AND HAVE NOT BEEN RUN. See the report 84 for the gates.

D11  test_d11_pen_trace_vs_reference: for the B-row trace and every row bucket, random bf16 shard logits are written into the persistent
     shard-logits buffer, the state is rebuilt from host lists, the pen trace is replayed and the shard logits are read back. Gate: a
     token that carries a penalty is within one bf16 ulp of the fp32 reference (apply_penalties_exact with the device history rule); every
     other token, and every token of a neutral row, is bit-identical to what was written.
D12  test_d12_counts_sync_1000_steps: 1000 sync fused steps with rejections, joins, finishes, runner slot remaps and row-bucket changes;
     greedy and sampled penalty rows next to neutral rows. Gate: after EVERY step the device counts and prompt mask of every live penalty row
     equal the host counts of the request (a rebuild is sent at a join and after a remap, as the plugin does).
     test_d12_counts_chained_deferred: the deferred depth-1 pipeline (chained steps, QWEN36_SPEC_CHAIN_SAMPLED=1), 300 steps; the counts
     equal the host counts once the pipeline is drained, and no step raised.
D13  test_d13_first_token_distribution: 8 identical requests per round (same prompt, a penalty row sampled at T 1, top_k 20), one verify step
     with no draft each: the committed token is the first sample of the penalized distribution. P8_D13_SAMPLES (default 20000) samples.
     Gate: the empirical frequency of each of the 5 most probable tokens is within 3 sigma of the expected probability, computed in float64
     on the raw logits of that step with the DEVICE penalty model (fp32 chain rounded to bf16: apply_penalties_device_model); no sample outside
     the top-k set. The distance to the exact fp32 penalties is logged, not gated (the user accepted the bf16 rounding, 2026-10-28).
D14  test_d14_no_compile_and_trace_memory: a run with penalties, joins, a finish and a host-walked fallback step under
     set_program_cache_misses_allowed(False): the program cache does not grow; the pen update trace grows the trace region by less than 4 MiB
     and the whole trace region stays below 90 percent of its size; every bucket has a pen trace.

Run (one node per process): work/scratch/phase3/run_p8_device.sh <name> [VAR=val ...]
"""

import math
import os
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env as _chain_env
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _setup as _chain_setup
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_engine import SampledDriver
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    apply_penalties_device_model,
    apply_penalties_exact,
    column_history_counts,
    counts_from_ids,
    mask_from_ids,
)

B, K = tc.B, tc.K
T = K + 1
D13_SAMPLES = int(os.environ.get("P8_D13_SAMPLES", "20000"))


def _setup(mesh_device, monkeypatch, env=None):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    _need_pen(eng)
    return model, eng, prompts


def _need_pen(eng):
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    assert eng._pen_ok(), "penalty traces missing (QWEN36_SPEC_DEVICE_PENALTIES=0, or the build was skipped)"
    model = eng.model
    assert model._vfy_pen_trace_id is not None
    for b, bk in (model._vfy_bk or {}).items():
        assert bk.pen_trace_id is not None, f"bucket {b} has no pen trace"


# --------------------------------------------------------------------------- #
# Drivers: a penalty row is a sampled row plus spec_sampling["penalties"]
# --------------------------------------------------------------------------- #
class PenMixin:
    """Adds ``penalties`` to the driver's spec_sampling. A request carries ``q["pen"]`` = (repetition, frequency, presence, greedy) or
    None. A rebuild entry (prompt ids, output ids) is sent the first time a request is seen in a row and after a remap, which is what the
    plugin does (``self._known[row]`` = the request id the device state was last written for)."""

    def _pen_init(self):
        self._known = {}

    def _spec_sampling(self):
        s = super()._spec_sampling()
        rep, freq, pres = torch.ones(B), torch.zeros(B), torch.zeros(B)
        rebuild = {}
        for r in range(B):
            q = self.reqs[r]
            if q is None or q.get("pen") is None:
                continue
            rp, fq, ps, greedy = q["pen"]
            rep[r], freq[r], pres[r] = rp, fq, ps
            if (
                greedy
            ):  # the argmax of the main trace is taken before the penalties: a greedy request is a sampled row with one candidate
                s["temperature"][r], s["top_k"][r], s["top_p"][r] = 1.0, 1, 0.0
            if self._known.get(r) != q["rid"]:
                rebuild[r] = (torch.tensor(q["prompt"], dtype=torch.int32), torch.tensor(q["out"], dtype=torch.int32))
                self._known[r] = q["rid"]
        s["penalties"] = {
            "repetition": rep,
            "frequency": freq,
            "presence": pres,
            **({"rebuild": rebuild} if rebuild else {}),
        }
        return s

    def admit_pen(self, row, rid, prompt, max_new, pen):
        self.admit(row, rid, prompt, max_new=max_new)
        if self.reqs[row] is not None:
            self.reqs[row]["pen"] = pen

    def remap(self, eng, perm):
        """Runner row i takes the request that sat at row perm[i] (a slot_remap); moved rows get a rebuild, as the plugin sends one."""
        eng.remap_slots(perm)
        self.reqs = [self.reqs[int(perm[i])] for i in range(B)]
        for i in range(B):
            if int(perm[i]) != i:
                self._known.pop(i, None)
        self.remapped = True


class PenDriver(PenMixin, SampledDriver):
    def __init__(self, eng, seed, **kw):
        super().__init__(eng, seed, **kw)
        self._pen_init()


class PenAsyncDriver(PenMixin, AsyncDriver):
    def __init__(self, eng, seed, **kw):
        super().__init__(eng, seed, **kw)
        self._pen_init()


def _host_state(eng, drv, rows):
    """(device counts [n, V], device mask [n, V], host counts, host mask) of the live penalty rows ``rows``."""
    V = int(eng.vocab)
    live = [r for r in rows if drv.reqs[r] is not None and drv.reqs[r].get("pen") is not None]
    if not live:
        return live, None
    dc, dm = eng._pen.read_rows([eng._phys[r] for r in live])
    hc = torch.stack([counts_from_ids(drv.reqs[r]["out"], V) for r in live])
    hm = torch.stack([mask_from_ids(drv.reqs[r]["prompt"], V) > 0 for r in live])
    return live, (dc, dm, hc, hm)


def _assert_counts(eng, drv, tag):
    live, st = _host_state(eng, drv, range(B))
    if st is None:
        return 0
    dc, dm, hc, hm = st
    for i, r in enumerate(live):
        assert torch.equal(dc[i], hc[i]), (
            f"[{tag}] row {r} (rid {drv.reqs[r]['rid']}): device counts differ from the host counts at "
            f"{(dc[i] != hc[i]).nonzero().reshape(-1)[:8].tolist()} (device {dc[i][dc[i] != hc[i]][:8].tolist()} host {hc[i][dc[i] != hc[i]][:8].tolist()})"
        )
        assert torch.equal(dm[i], hm[i]), f"[{tag}] row {r}: device prompt mask differs from the host mask"
    return len(live)


# --------------------------------------------------------------------------- #
# D11
# --------------------------------------------------------------------------- #
def _ulp(v):
    v = v.double().abs().clamp_min(1e-30)
    return torch.pow(2.0, torch.floor(torch.log2(v)) - 7)


@_marks
def test_d11_pen_trace_vs_reference(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    pen, V = eng._pen, int(eng.vocab)
    rep_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    cat = ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    targets = [(B, model._vfy_shard_logits_out, model._vfy_pen_trace_id)]
    targets += [(b, bk.shard_logits, bk.pen_trace_id) for b, bk in sorted((model._vfy_bk or {}).items(), reverse=True)]
    g = torch.Generator().manual_seed(11)
    # users: 0 rep, 1 freq, 2 pres, 3 all three, 4.. neutral
    rep = torch.tensor([1.3, 1.0, 1.0, 1.15, 1.0, 1.0, 1.0, 1.0])
    freq = torch.tensor([0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    pres = torch.tensor([0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    outs = [torch.randint(0, V, (n,), generator=g).tolist() for n in (40, 7, 150, 25, 30, 30, 30, 30)]
    prompts_ = [torch.randint(0, V, (n,), generator=g).tolist() for n in (200, 5, 60, 1000, 9, 9, 9, 9)]
    pen.rebuild({u: (torch.tensor(prompts_[u]), torch.tensor(outs[u])) for u in range(B)})
    pen.stage_params(types.SimpleNamespace(rep=rep, freq=freq, pres=pres))
    toks = torch.randint(0, V, (B, T), generator=g, dtype=torch.int32)
    h = ttnn.from_torch(
        toks.reshape(1, B * T), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep_mapper
    )
    ttnn.copy_host_to_device_tensor(h, model._vfy_token_buf)
    worst = {}
    for b, buf, tid in targets:
        assert tid is not None, f"bucket {b} has no pen trace"
        R = b * T
        x = (torch.randn(R, V, generator=g) * 3).to(torch.bfloat16)
        h = ttnn.from_torch(
            x.reshape(1, 1, R, V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=shard
        )
        ttnn.copy_host_to_device_tensor(h, buf)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(buf, mesh_composer=cat).reshape(R, V)
        n_pen = n_one = 0
        for u in range(b):
            cnt = counts_from_ids(outs[u], V)
            hist = column_history_counts(cnt, toks[u, 1:], K, T, device_rule=True)
            msk = mask_from_ids(prompts_[u], V).reshape(1, -1).expand(T, -1)
            sl = slice(u * T, (u + 1) * T)
            if bool((rep[u] == 1.0) & (freq[u] == 0.0) & (pres[u] == 0.0)):
                assert torch.equal(got[sl], x[sl]), f"[b{b}] neutral user {u} changed"
                continue
            ref = apply_penalties_exact(
                x[sl].float(),
                hist,
                msk,
                torch.full((T,), float(rep[u])),
                torch.full((T,), float(freq[u])),
                torch.full((T,), float(pres[u])),
            )
            penalized = (hist > 0) | (msk > 0)
            assert torch.equal(got[sl][~penalized], x[sl][~penalized]), f"[b{b}] user {u}: an unpenalized token changed"
            err = (got[sl].double() - ref.double()).abs() / _ulp(ref)
            assert (
                float(err[penalized].max()) <= 1.0
            ), f"[b{b}] user {u}: {float(err[penalized].max()):.3f} bf16 ulp off the fp32 reference"
            worst[b] = max(worst.get(b, 0.0), float(err[penalized].max()))
            n_pen += int(penalized.sum())
            model_out = apply_penalties_device_model(
                x[sl],
                hist,
                msk,
                torch.full((T,), float(rep[u])),
                torch.full((T,), float(freq[u])),
                torch.full((T,), float(pres[u])),
            )
            n_one += int((model_out != got[sl])[penalized].sum())
        logger.info(
            f"[d11] bucket {b}: {n_pen} penalized tokens, max error {worst.get(b, 0.0):.3f} ulp, differing from the host device-model {n_one}"
        )


# --------------------------------------------------------------------------- #
# D12
# --------------------------------------------------------------------------- #
@_marks
def test_d12_counts_sync_1000_steps(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = PenDriver(eng, seed=12, sampled=lambda s: True, drafts=True)
        g = torch.Generator().manual_seed(12)
        pens = [
            (1.3, 0.0, 0.0, True),
            (1.0, 0.3, 0.2, True),
            (1.1, 0.1, 0.4, False),
            (1.2, 0.2, 0.0, False),
            None,
            None,
            None,
            None,
        ]
        rid = [0]

        def admit(row, max_new):
            rid[0] += 1
            drv.admit_pen(row, rid[0], prompts[rid[0] % len(prompts)], max_new, pens[row] if rid[0] % 5 else None)

        for r in range(B):
            admit(r, int(torch.randint(15, 60, (1,), generator=g)))
        n_checked = n_remaps = 0
        for step in range(1000):
            if not drv.live():
                break
            drv.step()
            n_checked += _assert_counts(eng, drv, f"step {step}")
            for r in range(B):  # join: a finished row takes a new request
                if (
                    drv.reqs[r] is None and step < 950 and not (300 <= step < 420 and r >= 2)
                ):  # steps 300-420: rows 2.. stay empty (row bucket shrinks)
                    admit(r, int(torch.randint(15, 70, (1,), generator=g)))
            if step % 97 == 96:  # a runner slot remap: two rows swap their requests
                perm = list(range(B))
                a, b2 = (int(x) for x in torch.randperm(B, generator=g)[:2])
                perm[a], perm[b2] = perm[b2], perm[a]
                drv.remap(eng, perm)
                n_remaps += 1
        fb = dict(getattr(eng, "fused_fallbacks", {}))
        logger.info(
            f"[d12:sync] steps={drv.steps} checked_row_steps={n_checked} remaps={n_remaps} pen_steps={eng.pen_steps} host_walks={eng.pen_host_walks} "
            f"pen stats={eng._pen.stats} fallbacks={fb} buckets={dict(getattr(eng, 'bucket_steps', {}))}"
        )
        assert eng.pen_steps > 0 and n_checked > 500 and eng._pen.stats["rebuilds"] > 0 and n_remaps >= 5
        assert len(getattr(eng, "bucket_steps", {})) >= 2, "the run never changed the row bucket"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_d12_counts_chained_deferred(mesh_device, monkeypatch):
    _chain_env(monkeypatch)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    model, eng, tokenizer, prompts = _chain_setup(mesh_device)
    _need_pen(eng)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = PenAsyncDriver(eng, seed=13, sampled=lambda s: True)
        pens = [
            (1.3, 0.0, 0.0, True),
            (1.0, 0.3, 0.2, True),
            (1.1, 0.1, 0.4, False),
            (1.2, 0.2, 0.0, False),
            None,
            None,
            None,
            None,
        ]
        for u in range(B):
            drv.admit_pen(u, u, prompts[u], 10**6, pens[u])
        drv.pipeline(300)  # drains at the end
        n = _assert_counts(eng, drv, "chained")
        logger.info(
            f"[d12:chained] enqueued={drv.steps} chain_steps={eng.chain_steps} breaks={dict(eng.chain_breaks)} pen_steps={eng.pen_steps} "
            f"pen stats={eng._pen.stats} checked_rows={n}"
        )
        assert n == 4 and eng.chain_steps > 100 and eng.pen_steps >= 300
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


# --------------------------------------------------------------------------- #
# D13
# --------------------------------------------------------------------------- #
@_marks
def test_d13_first_token_distribution(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    V = int(eng.vocab)
    rep, freq, pres, temp, topk = 1.3, 0.2, 0.4, 1.0, 20
    prompt = prompts[0][:48]
    drv = PenDriver(eng, seed=13, sampled=lambda s: True, drafts=False)
    orig = drv._spec_sampling

    def sampling():  # T 1.0, top_k 20 (the driver's default is 0.6 / 20 with top_p 0.95)
        s = orig()
        s["temperature"][:] = temp
        s["top_p"][:] = 1.0
        return s

    drv._spec_sampling = sampling
    rounds = math.ceil(D13_SAMPLES / B)
    hist = torch.zeros(V, dtype=torch.float64)
    raw = out0 = None
    for rnd in range(rounds):
        for r in range(B):
            drv.admit_pen(r, rnd * B + r, prompt, 2, (rep, freq, pres, False))
        out0 = drv.reqs[0]["out"][0]
        drv.step()
        if raw is None:  # the raw logits of column 0 of row 0 (all rows are identical)
            lg = torch.as_tensor(model.fetch_verify_logits()).reshape(B * T, -1)[:, :V]
            raw = lg[0].to(torch.bfloat16)
        for r in range(B):
            hist[drv.outs[rnd * B + r][1]] += 1
        drv._known.clear()
    n = float(hist.sum())
    cnt = counts_from_ids([out0], V)
    msk = mask_from_ids(prompt, V)
    dev = apply_penalties_device_model(
        raw[None], cnt[None], msk[None], torch.tensor([rep]), torch.tensor([freq]), torch.tensor([pres])
    )[0]
    exact = apply_penalties_exact(
        raw[None].float(), cnt[None], msk[None], torch.tensor([rep]), torch.tensor([freq]), torch.tensor([pres])
    )[0]

    def probs(logits):
        v = logits.double() / temp
        keep = torch.topk(v, topk).indices
        p = torch.zeros(V, dtype=torch.float64)
        p[keep] = torch.softmax(v[keep], 0)
        return p

    p_dev, p_exact = probs(dev), probs(exact)
    outside = float(hist[p_dev == 0].sum())
    top5 = torch.topk(p_dev, 5).indices.tolist()
    rows = []
    for t in top5:
        p, f = float(p_dev[t]), float(hist[t]) / n
        sigma = math.sqrt(p * (1 - p) / n)
        rows.append((t, p, f, abs(f - p) / sigma, float(p_exact[t])))
    logger.info(
        f"[d13] samples={int(n)} outside_topk={outside} top5 (token, p_device_model, freq, sigmas, p_exact_fp32)={rows} "
        f"max |p_device_model - p_exact| over the top-k set={float((p_dev - p_exact).abs().max()):.4f}"
    )
    assert outside == 0, "a sample outside the top-k set of the penalized logits"
    for t, p, f, z, pe in rows:
        assert z <= 3.0, f"token {t}: frequency {f:.4f} vs expected {p:.4f} ({z:.2f} sigma)"


# --------------------------------------------------------------------------- #
# D14
# --------------------------------------------------------------------------- #
@_marks
def test_d14_no_compile_and_trace_memory(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    log = dict(getattr(model, "_vfy_mem_log", {}))
    logger.info(f"[d14] trace memory log (label: trace bytes, dram bytes): { {k: v for k, v in log.items()} }")
    labels = list(log)
    assert "pen_update" in log, f"no memory line after the pen update trace: {labels}"
    i = labels.index("pen_update")
    delta = log["pen_update"][0] - log[labels[i - 1]][0]
    cap = None
    try:
        tv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
        cap = int(tv.total_bytes_per_bank) * int(tv.num_banks)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[d14] trace region size unavailable: {e}")
    logger.info(
        f"[d14] pen update trace: +{delta / 2**20:.2f} MiB; last trace bytes {log[labels[-1]][0] / 2**20:.1f} MiB of {cap and cap / 2**20}"
    )
    assert delta < 4 * 2**20
    if cap:
        assert max(v[0] for v in log.values()) < 0.9 * cap
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = PenDriver(eng, seed=14, sampled=lambda s: True)
        pens = [(1.3, 0.0, 0.0, True), (1.0, 0.3, 0.2, False), None, (1.2, 0.2, 0.1, True)]
        for r in range(4):
            drv.admit_pen(r, r, prompts[r], 24 if r != 3 else 60, pens[r])
        for step in range(30):
            if not drv.live():  # every request finished (no join after step 24)
                break
            drv.step()
            if step == 10:  # a host-walked step: the penalties go through the host path and the counts are repaired
                saved, eng._fs_trace = eng._fs_trace, None
                try:
                    drv.step()
                finally:
                    eng._fs_trace = saved
                _assert_counts(eng, drv, "fallback")
            if (
                drv.reqs[0] is None and step < 24
            ):  # a join into a freed row while others run (the rebuild goes out with the next step)
                drv.admit_pen(0, 100, prompts[5], 40, pens[0])
        _assert_counts(eng, drv, "end")
        assert mesh_device.num_program_cache_entries() == n_prog, "a program was compiled after capture"
        assert eng.pen_host_walks >= 1
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
