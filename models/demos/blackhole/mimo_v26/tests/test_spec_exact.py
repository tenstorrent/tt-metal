# SPDX-License-Identifier: Apache-2.0
"""Spec-decode exactness + speed vs plain greedy decode (same prompts, same prefill path), on the 4x8 mesh.
Env: SPEC_LAYERS (48; <48 = skeleton run: capture layers are remapped, acceptance meaningless), SPEC_U (users per mesh row, <=4), SPEC_NEW (max new tokens), SPEC_START (first GSM8K row),
SPEC_OUT (jsonl), SPEC_TRACE (1). Prompts: GSM8K test rows, chat template with enable_thinking=False. n = 4*U prompts (one per (row,slot)).
Reports: token identity per user vs the plain stream (first divergence index), accepted tokens per round (mean over live users), plain decode ms/token, spec ms/round and tokens/s/user."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer
from models.demos.blackhole.mimo_v26.tests.decode_common import gsm_prompts
from models.demos.blackhole.mimo_v26.tt.dflash import DFlashDrafter
from models.demos.blackhole.mimo_v26.tt.generator import load_eos
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedGenerator, ChunkedPrefill
from models.demos.blackhole.mimo_v26.tt.spec import CAP, SpecEngine
from models.demos.blackhole.mimo_v26.tt.spec_dev import SpecDevice

E = os.environ.get
NL = int(E("SPEC_LAYERS", "48"))
U = int(E("SPEC_U", "4"))
NEW = int(E("SPEC_NEW", "256"))
START = int(E("SPEC_START", "0"))
TRACE = E("SPEC_TRACE", "1") == "1"
MODE = E(
    "SPEC_MODE", "both"
)  # plain | spec | both. NOTE finished users of a plain/spec run get position 0 of their slot overwritten (dead-slot rule), so a spec run after a plain run in the same process starts from a clobbered cache: use plain in one process, spec in another (SPEC_REF = tokens json of the plain run)
REF = E("SPEC_REF", "")
CHUNK = int(E("SPEC_CHUNK", "4"))
OUT = E("SPEC_OUT", "/tmp/spec.jsonl")
CH = int(E("SPEC_CH", "256"))
ISL = int(
    E("SPEC_ISL", "0")
)  # >0: Frankenstein book windows of this many tokens (long-context runs) instead of GSM8K prompts


def eng_place(n, rows):
    return [(i % rows, i // rows) for i in range(n)]


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("SPEC_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return dict(
        alloc_mb_per_bank=round(v.total_bytes_allocated_per_bank / 2**20, 1),
        free_mb_per_bank=round(v.total_bytes_free_per_bank / 2**20, 1),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 4 * 1024 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_spec_exact(mesh_device):
    md = mesh_device
    tok = tokenizer()
    if ISL:
        from models.demos.blackhole.mimo_v26.reference.prompts import BOOK_QUESTION
        from models.demos.blackhole.mimo_v26.tests.test_ctx64k import book_body, chat_frame

        body = book_body(tok)
        pre, post = chat_frame(tok, BOOK_QUESTION)
        n_ = ISL - len(pre) - len(post)
        stride = int(E("SPEC_STRIDE", "2400"))
        prompts = [pre + body[b * stride : b * stride + n_] + post for b in range(4 * U)]
    else:
        items = gsm_prompts(4 * U, tok, START)
        prompts = [it["prompt_ids"] for it in items]
    maxlen = max(len(p) for p in prompts)
    max_pos = -(-(maxlen + NEW + 64) // CH) * CH
    lay = KVLayout(users=U, max_pos=max_pos, chunk=CH)
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: None)
    cap = CAP if NL == 48 else tuple(min(i, NL - 1) for i in (0, 1, 2, 3, 3))
    pf = ChunkedPrefill(model, trace=True, cap_layers=cap)
    dr = DFlashDrafter(md, model.embed, model.lm_head, U)
    rec("env", layers=NL, U=U, new=NEW, cap=cap, max_pos=max_pos, dram=dram(md), prompt_lens=[len(p) for p in prompts])
    eos = load_eos()
    # ---- plain greedy
    gen = ChunkedGenerator(model, max_pos, pf, trace=TRACE, keep_logits=False, U=U, eos=eos)
    # ONE traced prefill shared by the plain and the spec run (a second traced-prefill replay after decode ops hangs the device, see LOG): the KV it wrote below the
    # prompt length is never modified by plain decode (writes start at position n), spec rounds overwrite positions >= n before reading them.
    rows = 4
    per_row = [[] for _ in range(rows)]
    for i, p in enumerate(prompts):
        per_row[i % rows].append(list(p))
    pre = pf.prefill(per_row, capture_feats=True)
    ft = pf.first_tokens(pre["last"])
    ids = torch.zeros(rows, U, dtype=torch.long)
    for (r_, u_), (t_, _) in ft.items():
        ids[r_, u_] = t_
    gen.prefill = lambda prompts_, S=None: (ids, None)
    if MODE == "numerics":
        # logits of the 8-row verify vs the 1-row plain decode on the SAME token stream (teacher forced): how big is the ulp-level noise, how often does it flip the argmax
        from models.demos.blackhole.mimo_v26.tt.decode import Decoder as _Dec

        ref = json.load(open(REF))
        pl = ref["plain"]
        g2 = ChunkedGenerator(model, max_pos, pf, trace=False, keep_logits=True, U=U, eos=eos)
        g2.prefill = lambda prompts_, S=None: (ids, None)
        teach = [t_ + [0] * (NEW - len(t_)) for t_ in pl]
        rr = g2.generate(prompts, NEW, S=maxlen, teacher=teach)
        plog = rr["step_logits"]  # [user][step] -> logits predicting token index step+1
        dv = _Dec(model, trace=False, U=U, block=8, keep_logits=True)
        rows_ = 4
        stats = []
        nflip = ntot = 0
        for j0 in range(0, NEW - 9, 8):
            blk = torch.zeros(rows_, U, 8, dtype=torch.long)
            posn = torch.zeros(rows_, U, dtype=torch.long)
            for i_, (r_, u_) in enumerate(eng_place(len(prompts), 4)):
                blk[r_, u_] = torch.tensor(pl[i_][j0 : j0 + 8])
                posn[r_, u_] = len(prompts[i_]) + j0
            dv.set_inputs(blk, posn, torch.zeros(rows_, U, dtype=torch.bool))
            res_ = dv.run()
            vl = res_[2]  # [rows, B, V] fp32 logits of the verify rows (B = U*8, row u*8+t)
            for i_, (r_, u_) in enumerate(eng_place(len(prompts), 4)):
                for t_ in range(8):
                    pv = plog[i_][j0 + t_].float()
                    vv = vl[r_, u_ * 8 + t_].float()
                    d = (pv - vv).abs()
                    top2 = torch.topk(pv, 2)
                    ntot += 1
                    flip = int(pv.argmax()) != int(vv.argmax())
                    nflip += flip
                    stats.append(
                        dict(
                            user=i_,
                            step=j0 + t_,
                            maxabs=float(d.max()),
                            top1_abs=float(abs(pv.max() - vv.max())),
                            plain_gap=float(top2.values[0] - top2.values[1]),
                            flip=bool(flip),
                        )
                    )
        ma = torch.tensor([s_["maxabs"] for s_ in stats])
        t1 = torch.tensor([s_["top1_abs"] for s_ in stats])
        rec(
            "numerics",
            rows=ntot,
            argmax_flips=nflip,
            flip_rate=round(nflip / ntot, 5),
            maxabs_logit_diff_mean=round(float(ma.mean()), 4),
            maxabs_logit_diff_max=round(float(ma.max()), 4),
            top1_logit_diff_mean=round(float(t1.mean()), 4),
            top1_logit_diff_frac_zero=round(float((t1 == 0).float().mean()), 4),
            flips=[s_ for s_ in stats if s_["flip"]][:20],
            plain_gap_lt_0p25_frac=round(
                float(torch.tensor([s_["plain_gap"] < 0.25 for s_ in stats]).float().mean()), 4
            ),
        )
        return
    if MODE == "gap":
        # target logit gap at the first divergence between the spec stream and the plain stream (REF json has plain + spec): teacher-force the PLAIN stream, keep logits
        ref = json.load(open(REF))
        pl, sp = ref["plain"], ref["spec"]
        g2 = ChunkedGenerator(model, max_pos, pf, trace=False, keep_logits=True, U=U, eos=eos)
        g2.prefill = lambda prompts_, S=None: (ids, None)
        teach = [t_ + [0] * (NEW - len(t_)) for t_ in pl]
        rr = g2.generate(prompts, NEW, S=maxlen, teacher=teach)
        res_gap = []
        for i, (a, b) in enumerate(zip(pl, sp)):
            n_ = min(len(a), len(b))
            d_ = next((k for k in range(n_) if a[k] != b[k]), None)
            if d_ is None:
                continue
            lg = rr["step_logits"][i][d_ - 1].float()  # logits that produced token index d_ (decode step d_-1)
            top = torch.topk(lg, 3)
            res_gap.append(
                dict(
                    user=i,
                    idx=d_,
                    plain_tok=a[d_],
                    spec_tok=b[d_],
                    top3=top.indices.tolist(),
                    top3_logits=[round(float(x), 3) for x in top.values],
                    gap_top1_top2=round(float(top.values[0] - top.values[1]), 4),
                    spec_tok_rank=int((lg > lg[b[d_]]).sum()),
                    gap_to_spec_tok=round(float(top.values[0] - lg[b[d_]]), 4),
                )
            )
        rec("divergence_gaps", gaps=res_gap)
        return
    plain, plain2 = None, None
    if MODE in ("plain", "both"):
        r = gen.generate(prompts, NEW, S=maxlen)
        plain = r["tokens"]
        steady = sorted(r["times"][5:])
        rec(
            "plain",
            ms_per_token_median=round(steady[len(steady) // 2] * 1e3, 2) if steady else None,
            steps=len(r["times"]),
            lens=[len(t) for t in plain],
            dram=dram(md),
        )
        gen.dec.release()
    if MODE == "plain":
        json.dump(dict(plain=plain), open(OUT.replace(".jsonl", "_tokens.json"), "w"))
        return
    if REF:
        plain = json.load(open(REF))["plain"]
    if MODE in ("dev", "devdebug"):
        eng = SpecEngine(
            model, pf, dr, U, max_pos, eos, cap=cap, trace=False
        )  # start(): prefill features -> drafter ring (host, before the loop)
        eng.start(prompts, NEW, res=pre)
        first = torch.zeros(4, U, dtype=torch.long)
        for i_, (r_, u_) in enumerate(eng.place_):
            first[r_, u_] = eng.out[i_][0]
        init = (eng.pos.clone(), eng.anchor.clone(), eng.alive.clone(), first, dr.ring_pos.clone())
        sd = SpecDevice(model, eng.dec, dr, U, max_pos, eos, NEW)
        if MODE == "devdebug":
            eng.round()  # host-driven round 1 from the initial state
            last = eng.last
            sd.init_state(*init)
            os.environ["MIMO_CHECK_CAP"] = "0"
            sd._round()
            ttnn.synchronize_device(md)
            g = sd.host_dbg
            blk = g(sd.dbg["block"]).reshape(4, U, 8).long()
            arow = g(sd.dbg["arow"]).reshape(4, U, 8).long()
            kk = g(sd.dbg["k"]).reshape(4, U).long()
            rec(
                "devdebug",
                block_equal=bool(torch.equal(blk, last["block"])),
                arow_equal=bool(torch.equal(arow, last["post"])),
                k_equal=bool(torch.equal(kk, last["k"])),
                host_block=last["block"].tolist(),
                dev_block=blk.tolist(),
                host_post=last["post"].tolist(),
                dev_post=arow.tolist(),
                host_k=last["k"].tolist(),
                dev_k=kk.tolist(),
            )
            st = sd.read_state()
            rec(
                "devdebug_state",
                pos=st["pos"].tolist(),
                host_pos=eng.pos.tolist(),
                anchor=st["anchor"].tolist(),
                host_anchor=eng.anchor.tolist(),
                cnt=st["cnt"].tolist(),
            )
            return
        sd.init_state(*init)
        sd.prepare()
        n_r, chunk_t, enq_t = sd.run(chunk=CHUNK)
        st = sd.read_state()
        out = []
        for i_, (r_, u_) in enumerate(eng.place_):
            toks_ = [int(x) for x in st["out"][r_, u_, : int(st["cnt"][r_, u_])]]
            cut = next((j_ for j_, x in enumerate(toks_) if x in set(eos)), None)
            toks_ = toks_[: cut + 1] if cut is not None else toks_
            out.append(toks_[:NEW])
        ident, first_div = [], []
        for a, b in zip(plain, out):
            n_ = min(len(a), len(b))
            d_ = next((k for k in range(n_) if a[k] != b[k]), None)
            first_div.append(d_)
            ident.append(d_ is None and len(a) == len(b))
        rc = st["rcount"]
        acc_mean = float(((st["cnt"] - 1).sum() / rc.sum()))
        v = sorted(chunk_t[1:])
        ve = sorted(enq_t[1:])
        rec(
            "spec_dev",
            n=len(prompts),
            identical=sum(ident),
            first_div=first_div,
            out_lens=[len(t_) for t_ in out],
            plain_lens=[len(t_) for t_ in plain],
            rounds_enqueued=n_r,
            chunk=CHUNK,
            alive_rounds_total=int(rc.sum()),
            accepted_mean=round(acc_mean, 3),
            chunk_wall_ms_median=round(1e3 * v[len(v) // 2], 2) if v else None,
            round_ms_median=round(1e3 * v[len(v) // 2] / CHUNK, 2) if v else None,
            enqueue_only_ms_per_chunk_median=round(1e3 * ve[len(ve) // 2], 3) if ve else None,
            chunk_walls_ms=[round(1e3 * x, 1) for x in chunk_t],
            dram=dram(md),
        )
        json.dump(dict(plain=plain, spec=out), open(OUT.replace(".jsonl", "_tokens.json"), "w"))
        sd.release()
        return
    # ---- spec
    eng = SpecEngine(model, pf, dr, U, max_pos, eos, cap=cap, trace=TRACE)
    eng.start(prompts, NEW, res=pre)
    t0 = time.perf_counter()
    out = eng.run()
    wall = time.perf_counter() - t0
    ident, first_div = [], []
    for i, (a, b) in enumerate(zip(plain, out)):
        n = min(len(a), len(b))
        d = next((k for k in range(n) if a[k] != b[k]), None)
        first_div.append(d)
        ident.append(d is None and len(a) == len(b))
    acc = [x for rr in eng.stats["rounds"] for x in rr]
    nr = len(eng.stats["rounds"])
    toks = sum(len(t) for t in out)

    def steady(key, skip=2):
        v = sorted(eng.stats[key][skip:])
        return dict(median_ms=round(1e3 * v[len(v) // 2], 2), mean_ms=round(1e3 * sum(v) / len(v), 2)) if v else None

    rec(
        "spec",
        n=len(prompts),
        identical=sum(ident),
        first_div=first_div,
        out_lens=[len(t) for t in out],
        plain_lens=[len(t) for t in plain],
        rounds=nr,
        accepted_mean=round(sum(acc) / max(1, len(acc)), 3),
        accepted_hist=[acc.count(k) for k in range(1, 9)],
        first_rounds_ms=dict(
            draft=[round(1e3 * x, 1) for x in eng.stats["t_draft"][:3]],
            verify=[round(1e3 * x, 1) for x in eng.stats["t_verify"][:3]],
        ),
        steady_set_round=steady("t_set"),
        steady_draft_incl_set_round=steady("t_draft"),
        steady_verify_incl_set_inputs=steady("t_verify"),
        steady_verify_set_inputs=steady("t_vin"),
        steady_append=steady("t_append"),
        wall_s=round(wall, 2),
        tok_s_user_including_capture=round(toks / len(prompts) / wall, 2),
        dram=dram(md),
    )
    json.dump(
        dict(plain=plain, spec=out, stats={k: v for k, v in eng.stats.items() if k != "rounds"}),
        open(OUT.replace(".jsonl", "_tokens.json"), "w"),
    )
    rec("texts", sample=[tok.decode(out[i], skip_special_tokens=True)[:300] for i in range(min(4, len(out)))])
    eng.dec.release()
    dr.release()
    pf.release()
