"""Megakernel engine-ON row-independence gate (spec owner 11-10; the precondition for padding 1..7 live users onto the
8-user engine trace, QWEN36_R16_MK_PAD / QWEN36_R16_MK_PAD_MIN).

Two greedy runs through the PRODUCTION spec loop (step-harness driver, prod env, QWEN36_R16_MEGAKERNEL=1):
  F (full)   : 8 users in slots 0..7 (the k gate prompts in slots 0..k-1 + 8-k filler prompts), NEW tokens each;
  P (padded) : the same k gate prompts in slots 0..k-1 only; slots k..7 idle (the step replays the 8-user engine trace
               with idle rows: kvpos -1 / GDN ctrl HOLD).
Asserts:
  1. the k real users' engine outputs are BIT-IDENTICAL between F and P on every verify step (the bucket-8 x_out rows
     u*T .. u*T+T-1, read after each verify replay), and so are their token streams;
  2. every idle row of x_out in P is finite on every step;
  3. in P the idle slots' GDN state (rec_state[u], ring blocks (t*B+u)*Nv+h, both window halves [u]) and every KV block
     no real user's page table references are byte-for-byte unchanged across the decode.

  ROWIND_K=2 ROWIND_SEED=1 ROWIND_TOKENS=48 ROWIND_LAYERS=4 (GDN / attention layers checked for idle state; 'all')
  QWEN36_R16_MEGAKERNEL=1 pytest models/demos/blackhole/qwen36/tests/test_mk_row_independence.py -s -x
(prod env + full-mesh conftest, as test_r7k2s_ident). Run k = 1, 2, 5, 7 (k = 1 needs QWEN36_R16_MK_PAD_MIN=1).
Only meaningful once the engine writes x_out from real GDN (HOLD) + KV append; with stand-ins check 1 is vacuous.
"""
import hashlib
import json
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env, _marks, _setup, _streams

K = int(os.environ.get("ROWIND_K", "2"))
SEED = int(os.environ.get("ROWIND_SEED", "1"))
NEW = int(os.environ.get("ROWIND_TOKENS", "48"))
LAYERS = os.environ.get("ROWIND_LAYERS", "4")
OUT = os.environ.get("ROWIND_OUT", "")
BUCKET = 8


def _dev_shards(t):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _h(x):
    return hashlib.md5(x.contiguous().view(torch.uint8).numpy().tobytes() if x.dtype != torch.bfloat16
                       else x.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


class _XoutTap:
    """Wraps model.verify_after_external_sync (every verify read, traced or staged/enqueued, goes through it, after the
    device sync): reads the engine bucket's x_out and records per-slot row hashes + idle-row finiteness. The test runs
    with the step chain OFF so no later verify is enqueued before the read."""

    def __init__(self, model, T):
        self.model, self.T, self.steps = model, T, []
        self._orig = model.verify_after_external_sync
        mk1 = model._vfy_mk1
        assert mk1 is not None and mk1.ready and BUCKET in mk1.x_out, "engine bucket 8 not built (QWEN36_R16_MEGAKERNEL=1)"
        self.x_out = mk1.x_out[BUCKET]

        def wrapped(handle, *a, **k):
            r = self._orig(handle, *a, **k)
            if int(getattr(handle, "bucket", None) or BUCKET) == BUCKET and not getattr(handle, "kv_split", False):
                shards = _dev_shards(self.x_out)
                rows = shards[0].reshape(-1, shards[0].shape[-1])
                self.steps.append(dict(
                    slot={u: _h(rows[u * self.T:(u + 1) * self.T]) for u in range(BUCKET)},
                    finite={u: bool(torch.isfinite(rows[u * self.T:(u + 1) * self.T].float()).all()) for u in range(BUCKET)},
                    replicas_equal=all(torch.equal(shards[0], s) for s in shards[1:])))
            return r

        model.verify_after_external_sync = wrapped

    def close(self):
        self.model.verify_after_external_sync = self._orig


def _pick(seq, n):
    return list(seq) if LAYERS == "all" else list(seq)[:int(n)]


def _idle_state(model, idle):
    """Host snapshot of the idle slots' GDN state + the KV caches (subset of layers per ROWIND_LAYERS)."""
    snap = {}
    gdn = [(i, l.attention) for i, l in enumerate(model.layers) if not l.is_full_attention]
    att = [(i, l.attention) for i, l in enumerate(model.layers) if l.is_full_attention]
    for i, a in _pick(gdn, LAYERS):
        B, Nv = a.B, a.Nv
        for name, t in (("rec", getattr(a, "rec_state", None)), ("ring", getattr(a, "_spec_ring", None))):
            if t is None:
                continue
            for d, s in enumerate(_dev_shards(t)):
                if name == "rec":
                    snap[(i, name, d)] = s.reshape(B, -1)[idle].clone()
                else:
                    blocks = s.reshape(-1, s.shape[-2] * s.shape[-1])
                    T = blocks.shape[0] // (B * Nv)
                    ix = [(t_ * B + u) * Nv + h for t_ in range(T) for u in idle for h in range(Nv)]
                    snap[(i, name, d)] = blocks[ix].clone()
        for w, t in enumerate(getattr(a, "_verify_win_pair", None) or ()):
            for d, s in enumerate(_dev_shards(t)):
                snap[(i, f"win{w}", d)] = s.reshape(B, -1)[idle].clone()
    for i, a in _pick(att, LAYERS):
        if getattr(a, "use_paged", False):
            for name, t in (("k", a.paged_k), ("v", a.paged_v)):
                for d, s in enumerate(_dev_shards(t)):
                    snap[(i, name, d)] = s.reshape(s.shape[0], -1).clone()
    return snap


def _run(model, eng, prompts, slots, tap):
    drv = AsyncDriver(eng, seed=SEED, sampled=lambda s: False)
    for u in slots:
        drv.admit(u, u, prompts[u], max_new=NEW)
    i0 = len(tap.steps)
    drv.pipeline(len(slots) * NEW)
    return drv, _streams(drv), tap.steps[i0:]


@_marks
def test_mk_row_independence(mesh_device, monkeypatch):
    assert 1 <= K <= 7, K
    _env(monkeypatch)
    for v in ("QWEN36_SPEC_CHAIN", "VLLM_TT_SPEC_CHAIN", "QWEN36_SPEC_CHAIN_SAMPLED", "VLLM_TT_SPEC_CHAIN_SAMPLED"):
        monkeypatch.setenv(v, "0")  # sequential steps: x_out is read before the next verify is enqueued
    model, eng, tokenizer, prompts = _setup(mesh_device)
    T = eng.T
    tap = _XoutTap(model, T)
    real, idle = list(range(K)), list(range(K, BUCKET))
    try:
        _, sF, stF = _run(model, eng, prompts, list(range(BUCKET)), tap)
        pre = _idle_state(model, idle)
        drvP, sP, stP = _run(model, eng, prompts, real, tap)
        post = _idle_state(model, idle)
    finally:
        tap.close()
    # padded steps must have replayed the engine bucket (not an op-by-op small bucket)
    assert stP, "no engine-bucket verify replay in the padded run (QWEN36_R16_MK_PAD / PAD_MIN?)"
    # 1. real rows bit-identical per step + token streams identical
    n = min(len(stF), len(stP))
    diff = [(i, u) for i in range(n) for u in real if stF[i]["slot"][u] != stP[i]["slot"][u]]
    tok = {u: list(sF[u]) == list(sP[u]) for u in real}
    # 2. idle rows finite in P; x_out identical on every device
    nonfinite = [(i, u) for i, s in enumerate(stP) for u in idle if not s["finite"][u]]
    rep = all(s["replicas_equal"] for s in stF + stP)
    # 3. idle GDN state + KV blocks outside the real users' page tables unchanged
    pt = getattr(eng, "_last_pt", None)
    owned = set()
    if pt is not None:
        owned = {int(b) for b in torch.as_tensor(pt)[:K].reshape(-1).tolist()}
    changed = []
    for key, a in pre.items():
        b = post[key]
        if key[1] in ("k", "v"):
            free = [r for r in range(a.shape[0]) if r not in owned]
            if not torch.equal(a[free], b[free]):
                changed.append(key)
        elif not torch.equal(a, b):
            changed.append(key)
    rec = dict(k=K, seed=SEED, tokens=NEW, steps_full=len(stF), steps_padded=len(stP), row_diffs=diff[:20],
               n_row_diffs=len(diff), tokens_equal=tok, idle_nonfinite=nonfinite[:20], replicas_equal=rep,
               idle_state_changed=[list(map(str, c)) for c in changed[:20]], n_idle_state_changed=len(changed),
               owned_blocks=len(owned))
    logger.info(f"[rowind] {json.dumps(rec)}")
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    assert not diff, f"real rows differ between full and padded runs at (step, slot) {diff[:8]}"
    assert all(tok.values()), f"token streams differ: {tok}"
    assert not nonfinite, f"idle rows non-finite at (step, slot) {nonfinite[:8]}"
    assert rep, "x_out differs across the 8 devices"
    assert not changed, f"idle GDN state / free KV blocks changed: {changed[:8]}"
