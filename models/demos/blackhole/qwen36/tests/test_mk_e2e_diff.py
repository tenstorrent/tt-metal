"""Megakernel e2e-diff (M3 acceptance, RC gate 9): the engine (QWEN36_R16_MEGAKERNEL=1) vs the prod op-by-op verify layer
loop on the SAME real users, inputs, positions, page tables and state, per user x per layer residual. Does NOT use the
bench or its golds (spec owner / main, 11-10: two bugs hid behind shared bench assumptions).

Per case (k8: 8 live users at ragged contexts; k5: 5 live users, rows 5-7 idle = the padded engine path):
  1. drive the production spec loop (AsyncDriver, greedy) for E2E_WARM steps so every state is a real mid-decode state;
  2. snapshot every buffer a verify mutates (GDN rec_state / ring / window pair, every KV cache);
  3. REFERENCE: run the verify body once eagerly with the engine detached (op-by-op), recording x after each layer;
  4. restore the snapshot; ENGINE: run the same body with the engine attached (MK_AR_DBG=1: the residual after every
     all-reduce, ar_dbgt [32, AR_NAR * 2 * 5120]); the residual after layer L = the output of layer L's MLP-AR;
  5. restore again (the driver's state is left as it was).
Report: one line "[e2ediff] {json}" with pcc / maxabs {case: {user: {layer: x}}}, first_bad, nan_zero_rows,
final_err_spread, pass. Bar: PCC >= E2E_BAR (0.999) for every live user and layer, no NaN / all-zero live rows, worst-user
final-layer max-abs error <= 2x the best user's.
Plus case "long": one user past the engine's max context must route the step to the op-by-op split fallback.

  E2E_NL=16 E2E_CTX=1024,2048,4096,6144,8192,12288,16384,32000 E2E_WARM=3 E2E_K5=1 E2E_LONG=65536 E2E_BAR=0.999 E2E_OUT=<jsonl>
  QWEN36_R16_MEGAKERNEL=1 pytest models/demos/blackhole/qwen36/tests/test_mk_e2e_diff.py -s -x    (prod env, full mesh)
"""
import json
import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks

NL = int(os.environ.get("E2E_NL", "16"))
CTX = [int(c) for c in os.environ.get("E2E_CTX", "1024,2048,4096,6144,8192,12288,16384,32000").split(",")]
WARM = int(os.environ.get("E2E_WARM", "3"))
K5 = os.environ.get("E2E_K5", "1") == "1"
LONG = int(os.environ.get("E2E_LONG", "65536"))
BAR = float(os.environ.get("E2E_BAR", "0.999"))
OUT = os.environ.get("E2E_OUT", "")
TF = os.environ.get("E2E_TF", "1") == "1"  # teacher-forced per-layer mode (spec owner 11-11)
TF_NL = int(os.environ.get("E2E_TF_NL", str(NL)))
DBG = os.environ.get("E2E_DBG", "1") == "1"  # MK_AR_DBG per-AR residual dumps (more engine text); 0 = final x_out only
B, T = 8, tc.K + 1
DIM = 5120


def _pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def _rows(t):
    """device-0 [rows, DIM] float of a replicated residual tensor."""
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return x.reshape(-1, x.shape[-1])[:, :DIM]


def _mutable(model):
    out = []
    for l in model.layers:
        a = l.attention
        if l.is_full_attention:
            out += [a.paged_k, a.paged_v] if getattr(a, "use_paged", False) else []
        else:
            out += [t for t in (getattr(a, "rec_state", None), getattr(a, "_spec_ring", None)) if t is not None]
            out += list(getattr(a, "_verify_win_pair", None) or ())
    return [t for t in out if t is not None]


def _snapshot(ts):
    return [ttnn.from_device(t) for t in ts]


def _restore(ts, snap):
    for t, h in zip(ts, snap):
        ttnn.copy_host_to_device_tensor(h, t)
    ttnn.synchronize_device(ts[0].device())


def _layer_hooks(model, rec, rec_in=None):
    saved = []
    for i, l in enumerate(model.layers[:max(NL, TF_NL)]):
        f = l.forward

        def wrapped(*a, _f=f, _i=i, **k):
            if rec_in is not None:
                rec_in[_i] = _rows(a[0] if a else k["x"])
            y = _f(*a, **k)
            rec[_i] = _rows(y)
            return y

        saved.append((l, f))
        l.forward = wrapped
    return saved


def _mlp_ar_index(layer_types, aar=True, gar=False):
    """AR program order (AR contract): per layer [attention AR if attention-type and MK_REAL_AAR], then the MLP-AR. Under
    MK_REAL_GAR every layer has a mixer AR, so layer L's MLP-AR is 2L + 1."""
    if gar:
        return [2 * L + 1 for L in range(len(layer_types))]
    idx, k = [], 0
    for t in layer_types:
        k += 1 if (t == "attention" and aar) else 0
        idx.append(k)
        k += 1
    return idx


def _diff_case(model, live, tag):
    mk1 = model._vfy_mk1
    prog = mk1.programs[B]
    S = prog.program.setup
    has_dbg = S.get("ar_dbgt") is not None
    nl = int(prog.info["nl"])
    lt = mk1.layer_types[:nl]
    _defs = prog.info["env"].get("MK_DEFINES", "").split(",")
    aar, gar = "MK_REAL_AAR=1" in _defs, "MK_REAL_GAR=1" in _defs
    ts = _mutable(model)
    snap = _snapshot(ts)
    # reference: op-by-op
    ref, ref_in = {}, {}
    model._vfy_mk1 = None
    saved = _layer_hooks(model, ref, ref_in)
    L_attn = [L for L in range(min(nl, NL)) if lt[L] == "attention"]
    try:
        model._forward_verify_bucket_tp(B, None, None)
        ttnn.synchronize_device(model.mesh_device)
    finally:
        for l, f in saved:
            l.forward = f
        model._vfy_mk1 = mk1
    kv_ref = _kv_rows(model, L_attn, live)
    _restore(ts, snap)
    # engine
    model._forward_verify_bucket_tp(B, None, None)
    ttnn.synchronize_device(model.mesh_device)
    kv_eng = _kv_rows(model, L_attn, live)
    kva = {}
    for L in L_attn:
        if kv_ref.get(L) is None or kv_eng.get(L) is None:
            continue
        kva[str(L)] = dict(k_pcc=round(_pcc(kv_eng[L][0], kv_ref[L][0]), 6), v_pcc=round(_pcc(kv_eng[L][1], kv_ref[L][1]), 6),
                           k_maxabs=float((kv_eng[L][0] - kv_ref[L][0]).abs().max()),
                           v_maxabs=float((kv_eng[L][1] - kv_ref[L][1]).abs().max()), rows=int(kv_ref[L][0].shape[0]))
    logger.info(f"[e2ediff] KVA {tag}: {json.dumps(kva)}")
    if not has_dbg:   # final layer only: the engine's x_out (drained residual after the pass's last AR)
        xo = _rows(mk1.x_out[B])
        _restore(ts, snap)
        tf = _teacher_forced(model, live, ref, ref_in, ts, snap) if TF else {}
        Lf = min(nl, NL) - 1
        pcc, mx, bad = {}, {}, []
        for u in live:
            e, r = xo[u * T:(u + 1) * T], ref[Lf][u * T:(u + 1) * T]
            if not torch.isfinite(e).all() or not e.abs().sum():
                bad.append(dict(case=tag, user=u, layer=Lf, what="nan" if not torch.isfinite(e).all() else "zero"))
            pcc[u], mx[u] = {Lf: round(_pcc(e, r), 6)}, {Lf: float((e - r).abs().max())}
        return pcc, mx, bad, None, dict(tf, kva=kva) if tf else dict(kva=kva)
    dbg = ttnn.to_torch(ttnn.get_device_tensors(S["ar_dbgt"])[0]).float().reshape(32, -1)
    # the post-AR residual columns (2k+1) must be identical on every device (AR contract: mesh identity 0); the input
    # partial columns (2k) are per-device by design
    nar = dbg.shape[1] // (2 * DIM)
    res_cols = torch.cat([torch.arange((2 * k + 1) * DIM, (2 * k + 2) * DIM) for k in range(nar)])
    dbg_all_equal = all(torch.equal(ttnn.to_torch(d).float().reshape(32, -1)[:, res_cols], dbg[:, res_cols])
                        for d in ttnn.get_device_tensors(S["ar_dbgt"])[1:])
    _restore(ts, snap)
    tf = _teacher_forced(model, live, ref, ref_in, ts, snap) if TF else {}
    ari = _mlp_ar_index(lt, aar, gar)
    pcc, mx, bad = {}, {}, []
    for u in live:
        pcc[u], mx[u] = {}, {}
        for L in range(min(nl, NL)):
            k = ari[L]
            e = dbg[u * T:(u + 1) * T, (2 * k + 1) * DIM:(2 * k + 2) * DIM]
            r = ref[L][u * T:(u + 1) * T]
            if not torch.isfinite(e).all() or not e.abs().sum():
                bad.append(dict(case=tag, user=u, layer=L, what="nan" if not torch.isfinite(e).all() else "zero"))
            pcc[u][L] = round(_pcc(e, r), 6)
            mx[u][L] = float((e - r).abs().max())
    return pcc, mx, bad, dbg_all_equal, dict(tf, kva=kva) if tf else dict(kva=kva)


def _kv_rows(model, L_attn, live):
    """The K / V rows the step appends for the live users of attention layers L_attn: {L: (k [n, 256], v [n, 256])} read
    from the paged caches (device 0) at kvpos rows r = u*T + j >= 0 through the host page table."""
    aux = getattr(model, "_vfy_aux_staged", None)
    pt = model._vfy_pt_host
    if aux is None:
        return {}
    kp = [int(x) for x in torch.as_tensor(aux)[0, : B * T].tolist()]
    out = {}
    for L in L_attn:
        a = model.layers[L].attention
        kk = ttnn.to_torch(ttnn.get_device_tensors(a.paged_k)[0]).float()
        vv = ttnn.to_torch(ttnn.get_device_tensors(a.paged_v)[0]).float()
        ks, vs = [], []
        for u in live:
            for j in range(T):
                p_ = kp[u * T + j]
                if p_ < 0 or p_ > 2 ** 30:
                    continue
                blk = int(pt[u, p_ // kk.shape[2]])
                ks.append(kk[blk, 0, p_ % kk.shape[2]])
                vs.append(vv[blk, 0, p_ % kk.shape[2]])
        out[L] = (torch.stack(ks), torch.stack(vs)) if ks else None
    return out


def _layer_tensors(model, L):
    a = model.layers[L].attention
    if model.layers[L].is_full_attention:
        return [a.paged_k, a.paged_v] if getattr(a, "use_paged", False) else []
    return [t for t in (getattr(a, "rec_state", None), getattr(a, "_spec_ring", None)) if t is not None] + \
        list(getattr(a, "_verify_win_pair", None) or ())


def _teacher_forced(model, live, ref, ref_in, ts, snap):
    """Per layer L < TF_NL: a 1-layer engine built for model layer L (its type, its repacked weights, its KV / GDN state)
    is seeded with the op-by-op residual ENTERING L and compared with the op-by-op output of L, inside an engine-on verify
    body (the per-replay inputs are the body's own). Only layer L's state is restored before its run. -> {user: {L: pcc}}"""
    from models.demos.blackhole.qwen36.tt.megakernel.engine import build as EB
    from models.demos.blackhole.qwen36.tt.megakernel.engine import program as EP
    from models.demos.blackhole.qwen36.tt.megakernel.integ import stage1 as S1
    from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk as WM

    mk1 = model._vfy_mk1
    mesh = model.mesh_device
    idx = {id(t): i for i, t in enumerate(ts)}
    kv_all, gdn_all = S1.kv_descriptors(model), S1.gdn_descriptors(model)
    out = {u: {} for u in live}
    err = {}
    orig = mk1.forward

    def tf_forward(x, n_users, T_, in_body):
        pr = S1.per_replay_inputs(model, in_body)
        for L in range(min(TF_NL, len(model.layers))):
            lt = ["attention" if model.layers[L].is_full_attention else "gdn"]
            try:
                w = WM.repack(model, layouts=EP.weight_layouts(mesh), only_layers=[L])
                eng = EB.MkEngine(mesh, B, T_, "mk", lt, w, [d for d in kv_all if d["layer"] == L],
                                  [d for d in gdn_all if d["layer"] == L])
                for t in _layer_tensors(model, L):
                    if id(t) in idx:
                        ttnn.copy_host_to_device_tensor(snap[idx[id(t)]], t)
                xi = ttnn.from_torch(ref_in[L].reshape(1, 1, -1, DIM).bfloat16(), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                xo = eng.alloc_x_out()
                eng.launch(xi, xo, pr)
                ttnn.synchronize_device(mesh)
                e_all = _rows(xo)
                for u in live:
                    out[u][L] = round(_pcc(e_all[u * T:(u + 1) * T], ref[L][u * T:(u + 1) * T]), 6)
                ttnn.deallocate(xi)
                ttnn.deallocate(xo)
                del eng, w
            except Exception as e:  # noqa: BLE001 -- report per layer (e.g. a build overflow), keep going
                err[L] = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning(f"[e2ediff] teacher-forced layer {L} ({lt[0]}): {err[L]}")
        return x

    mk1.forward = tf_forward
    try:
        model._forward_verify_bucket_tp(B, None, None)
        ttnn.synchronize_device(mesh)
    finally:
        mk1.forward = orig
    _restore(ts, snap)
    return dict(pcc=out, err=err, types={L: ("attention" if model.layers[L].is_full_attention else "gdn")
                                         for L in range(min(TF_NL, len(model.layers)))})


def _long_prompt(tokenizer, n, salt):
    base = tc._prompt_pool(tokenizer, 4096)
    rep = (base * (n // len(base) + 1))
    return rep[salt * 31 % len(base):][:n]


@_marks
def test_mk_e2e_diff(mesh_device, monkeypatch):
    assert os.environ.get("QWEN36_R16_MEGAKERNEL", "").strip() == "1", "run with QWEN36_R16_MEGAKERNEL=1"
    monkeypatch.setenv("QWEN36_R16_MK_NL", str(NL))
    monkeypatch.setenv("QWEN36_R16_MK_KV_INJECT", os.environ.get("QWEN36_R16_MK_KV_INJECT", "1"))
    monkeypatch.setenv("QWEN36_R16_MK_ALLOW_NO_FALLBACK", os.environ.get("QWEN36_R16_MK_ALLOW_NO_FALLBACK", "0"))
    env = os.environ.get("QWEN36_R16_MK_ENV", "")
    if DBG and "MK_AR_DBG=" not in env:
        monkeypatch.setenv("QWEN36_R16_MK_ENV", (env + " MK_AR_DBG=1").strip())
    for v in ("QWEN36_SPEC_CHAIN", "VLLM_TT_SPEC_CHAIN"):
        monkeypatch.setenv(v, "0")
    nb = max(math.ceil((max(CTX) + 64 * T) / 64), 8)
    monkeypatch.setattr(tc, "B", B)
    monkeypatch.setattr(tc, "NB", nb)
    monkeypatch.setattr(tc, "NUM_BLOCKS", B * nb + 40)  # the warm table spans B x nb blocks (+ the MTP scratch block)
    model, eng, tokenizer, _ = tc._setup(mesh_device)
    mk1 = model._vfy_mk1
    assert mk1 is not None and mk1.ready and B in mk1.programs, f"engine bucket {B} not built: {getattr(mk1, 'reason', None)}"
    prompts = [_long_prompt(tokenizer, c, i) for i, c in enumerate(CTX)]
    cases = [("k8", list(range(B)))] + ([("k5", list(range(5)))] if K5 else [])
    res = dict(pcc={}, maxabs={}, nan_zero_rows=[], first_bad=None, bar=BAR, ctx=CTX, nl=NL, dbg_mesh_equal={})
    for tag, live in cases:
        drv = AsyncDriver(eng, seed=1, sampled=lambda s: False)
        for u in live:
            drv.admit(u, 100 * len(res["pcc"]) + u, prompts[u], max_new=WARM * T + 8)
        for _ in range(WARM):
            drv.step()
        pcc, mx, bad, deq, tf = _diff_case(model, live, tag)
        res.setdefault("kva", {})[tag] = tf.pop("kva", {})
        if tf.get("pcc") is not None:
            res.setdefault("tf", {})[tag] = {str(u): {str(L): v for L, v in d.items()} for u, d in tf["pcc"].items()}
            res.setdefault("tf_err", {})[tag] = {str(L): v for L, v in tf["err"].items()}
            res["tf_types"] = {str(L): v for L, v in tf["types"].items()}
            for u, d in tf["pcc"].items():
                logger.info(f"[e2ediff] TF {tag} user {u}: " + " ".join(f"L{L}{tf['types'][L][0]}:{v:.5f}" for L, v in d.items()))
        res["pcc"][tag] = {str(u): {str(L): v for L, v in d.items()} for u, d in pcc.items()}
        res["maxabs"][tag] = {str(u): {str(L): v for L, v in d.items()} for u, d in mx.items()}
        res["nan_zero_rows"] += bad
        res["dbg_mesh_equal"][tag] = deq
        for L in range(NL):
            for u in live:
                v = pcc[u].get(L)
                if v is not None and v < BAR and res["first_bad"] is None:
                    res["first_bad"] = dict(case=tag, user=u, layer=L, pcc=v)
        for r in drv.live():
            drv._finish(r)
    fin = [list(res["maxabs"]["k8"][str(u)].values())[-1] for u in range(B)]
    res["final_err_spread"] = round(max(fin) / max(min(fin), 1e-12), 3)
    # long context: a step with a user past the engine's max context must take the op-by-op split fallback
    mx_ctx = getattr(model, "_vfy_mk_max_ctx", None)
    gate = getattr(model, "_kv_split_gate", None)
    res["long"] = dict(ctx=LONG, engine_max_ctx=mx_ctx,
                       fallback=bool(gate is not None and mx_ctx and gate.want("verify", B, LONG)),
                       short_on_engine=bool(gate is not None and mx_ctx and not gate.want("verify", B, mx_ctx)))
    allp = [v for c in res["pcc"].values() for d in c.values() for v in d.values()]
    res["min_pcc"] = min(allp) if allp else None
    tfp = [v for c in res.get("tf", {}).values() for d in c.values() for v in d.values()]
    res["tf_min_pcc"] = min(tfp) if tfp else None
    res["tf_first_bad"] = next(({"case": c, "user": u, "layer": L, "pcc": v} for c, cd in res.get("tf", {}).items()
                                for u, d in cd.items() for L, v in d.items() if v < BAR), None)
    res["pass"] = bool((not TF or (tfp and min(tfp) >= BAR and not any(res.get("tf_err", {}).values()))) and allp and min(allp) >= BAR and not res["nan_zero_rows"] and res["final_err_spread"] <= 2.0
                       and res["long"]["fallback"] and res["long"]["short_on_engine"])
    logger.info(f"[e2ediff] {json.dumps(res)}")
    for tag in res["pcc"]:
        for u, d in res["pcc"][tag].items():
            logger.info(f"[e2ediff] {tag} user {u} ctx {CTX[int(u)]}: " + " ".join(f"L{L}:{v:.5f}" for L, v in d.items()))
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
    assert res["pass"], f"e2e-diff FAIL: first_bad {res['first_bad']} min_pcc {res['min_pcc']} nan/zero {res['nan_zero_rows'][:4]} " \
                        f"spread {res['final_err_spread']} long {res['long']}"
