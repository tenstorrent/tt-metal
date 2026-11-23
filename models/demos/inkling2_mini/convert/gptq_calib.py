# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Our own GPTQ calibration of the routed experts to BFP4, in the cache's baked domain (opt-in, host / CPU only).

Three steps; the result is ``codes:DIR`` (``L{nn}.pt``) for ``convert.variant --experts-from`` (convert/gptq_codes.py):

1. corpus   python -m models.demos.inkling2_mini.convert.gptq_calib corpus --out calib/ids.pt --chat chat.jsonl \\
                --code-root CODE_DIR --prose-root DOCS_DIR
   Token sequences (chat-template rendered): ultrachat conversations, repo source files and docs as user
   messages. None of the golden / gate prompts are used.
2. capture  (card) prefill each sequence on the tt model with ``Inkling2Prefill.moe_tap`` set: writes per
   application and sequence the MoE input x (bf16 [n, H], after mlp_norm) and the routed expert ids [n, 8].
3. solve    python -m models.demos.inkling2_mini.convert.gptq_calib solve --ckpt CKPT --capture calib/ --out codes/
   Per layer (all its applications: the loop layers 6-17 see three), per expert: H = X^T X of the routed inputs,
   GPTQ of W_gate_up [2I, H] (act order, lazy blocks of 128, damp 0.01), then h = softcapped SwiGLU of the
   quantized gate_up, H_down = h^T h and GPTQ of W_down [H, I] (damp 1.0). Experts with < 16 routed tokens are
   rounded (RTN). Recipe as in Lottolabs' published notes; the grid is ours: one shared exponent per 16
   consecutive output rows at one input column (the cache's W^T tile face row), mantissa -7..7, so the solve
   picks each block's exponent when its column is quantized. W is the bf16 dequant our converter stores.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

H, I, E, TOP_K = 2304, 1152, 128, 8
# application -> layer (tt/model.py APPS order: 0-5, loop 6-17 x3, 18-23)
APP_LAYER = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
SEQ = 2048


# ---------------------------------------------------------------- 1. corpus
def corpus(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    chat = [json.loads(l) for l in open(args.chat)]
    seqs, kinds = [], []

    def pack(msgs_iter, n, kind):
        cur = []
        for msgs in msgs_iter:
            cur += tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
            while len(cur) >= SEQ:
                seqs.append(cur[:SEQ])
                kinds.append(kind)
                cur = []  # next sequence starts at a conversation start
                if kinds.count(kind) == n:
                    return

    def files(root, exts, skip):
        for p in sorted(Path(root).rglob("*")):
            if p.suffix in exts and p.is_file() and not any(s in str(p) for s in skip) and p.stat().st_size > 4000:
                yield p

    def as_user(paths, ask):
        for p in paths:
            text = p.read_text(errors="ignore")[:12000]
            yield [{"role": "user", "content": f"{ask}\n\n{text}"}]

    pack(iter(chat), args.n_chat, "chat")
    skip = ("inkling", "third_party", "/.git/", "build_", "python_env", "generated")
    pack(
        as_user(files(args.code_root, {".py", ".cpp", ".hpp"}, skip), "Explain what this code does."),
        args.n_code,
        "code",
    )
    pack(as_user(files(args.prose_root, {".md"}, skip), "Summarize this document."), args.n_prose, "prose")
    ids = torch.tensor(seqs, dtype=torch.int32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "kinds": kinds}, args.out)
    print(
        f"{args.out}: {ids.shape[0]} x {SEQ} tokens ({', '.join(f'{k} {kinds.count(k)}' for k in dict.fromkeys(kinds))})"
    )


# ---------------------------------------------------------------- 2. capture (card)
def capture(args):
    import ttnn
    from models.demos.inkling2_mini.tt.model import Inkling2Model
    from models.demos.inkling2_mini.tt.prefill_model import Inkling2Prefill

    d = torch.load(args.ids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=32768)
    try:
        model = Inkling2Model(dev, max_ctx=d["ids"].shape[1], log=lambda s: None)
        pf = Inkling2Prefill(model)
        seq = {"i": 0}

        def tap(app, s, x, ids):
            torch.save(
                {"x": x.to(torch.bfloat16).clone(), "ids": ids.to(torch.int16).clone(), "s": s},
                out / f"a{app:02d}_s{seq['i']:02d}_p{s:05d}.pt",
            )

        pf.moe_tap = tap
        for i in range(args.first, min(args.first + args.n, d["ids"].shape[0])):
            seq["i"] = d.get("first", 0) + i
            t0 = time.time()
            pf.prefill(d["ids"][i].tolist(), chunk=args.chunk)
            print(f"CAPTURE seq {i} {d['kinds'][i]} {time.time() - t0:.1f}s", flush=True)
    finally:
        ttnn.close_device(dev)


# ---------------------------------------------------------------- 3. solve (CPU)
def bfp4(w: torch.Tensor, mbits: int = 3):
    """w [..., R] fp32 -> (values, q int8, e uint8 [..., R / 16]): one exponent per 16 along the last dim, step
    2^(floor(log2 max) - (mbits - 1)), mantissa round-to-nearest-even clamped to 2^mbits - 1 (mbits 3: ttnn
    bfloat4_b, 7: bfloat8_b; moe_pcc.bfp_roundtrip)."""
    b = w.unflatten(-1, (-1, 16))
    exp = torch.frexp(b.abs().amax(-1, keepdim=True))[1] - 1
    step = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp - (mbits - 1))
    qmax = 2**mbits - 1
    q = torch.round(b / step).clamp(-qmax, qmax)
    return (q * step).flatten(-2), q.to(torch.int8).flatten(-2), (exp + 127).clamp(0, 255).to(torch.uint8).squeeze(-1)


def gptq(W: torch.Tensor, Hs: torch.Tensor, damp: float, block: int = 128, mbits: int = 3):
    """Batched GPTQ, act order. W [B, out, in] fp32, Hs [B, in, in] fp32 (sum of x x^T). The BFP blocks run along
    out, so a whole block is quantized with its column. Returns (q int8 [B, out, in], e uint8 [B, out/16, in])."""
    B, R, C = W.shape
    W = W.clone()
    Hs = Hs.clone()
    dead = torch.diagonal(Hs, dim1=1, dim2=2) == 0
    Hs.diagonal(dim1=1, dim2=2)[dead] = 1
    perm = torch.argsort(torch.diagonal(Hs, dim1=1, dim2=2), dim=1, descending=True)
    W = torch.gather(W, 2, perm[:, None, :].expand(B, R, C))
    Hs = torch.gather(torch.gather(Hs, 1, perm[:, :, None].expand(B, C, C)), 2, perm[:, None, :].expand(B, C, C))
    mean_diag = torch.diagonal(Hs, dim1=1, dim2=2).mean(1)
    Hs.diagonal(dim1=1, dim2=2).add_(damp * mean_diag[:, None])
    L = torch.linalg.cholesky(Hs)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)  # upper Cholesky of H^-1
    del L, Hinv, Hs
    Q = torch.empty(B, R, C, dtype=torch.int8)
    Ex = torch.empty(B, R // 16, C, dtype=torch.uint8)
    for c1 in range(0, C, block):
        c2 = min(c1 + block, C)
        W1 = W[:, :, c1:c2].clone()
        Err = torch.empty_like(W1)
        U1 = U[:, c1:c2, c1:c2]
        for j in range(c2 - c1):
            w = W1[:, :, j]
            v, q, e = bfp4(w, mbits)
            Q[:, :, c1 + j], Ex[:, :, c1 + j] = q, e
            d = U1[:, j, j][:, None]
            err = (w - v) / d
            W1[:, :, j:] -= err[:, :, None] * U1[:, j, None, j:]
            Err[:, :, j] = err
        W[:, :, c2:] -= torch.bmm(Err, U[:, c1:c2, c2:])
    inv = torch.argsort(perm, dim=1)
    Q = torch.gather(Q, 2, inv[:, None, :].expand(B, R, C))
    Ex = torch.gather(Ex, 2, inv[:, None, :].expand(B, R // 16, C))
    return Q, Ex


def rtn(W: torch.Tensor, mbits: int = 3):
    _, q, e = bfp4(W.transpose(-1, -2), mbits)  # blocks along out
    return q.transpose(-1, -2).contiguous(), e.transpose(-1, -2).contiguous()


def pack4(q: torch.Tensor) -> torch.Tensor:
    """int8 [..., C] in -7..7 -> uint8 [..., C / 2], low nibble = even index (two's complement)."""
    u = (q.to(torch.int16) & 0xF).to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def decode(q, e, mbits: int = 3):
    return torch.ldexp(q.float(), e.to(torch.int32).repeat_interleave(16, dim=-2) - 126 - mbits)


def swiglu(g, u):
    return 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g) * 25.0 * torch.tanh(u / 25.0)


def layer_inputs(cap: Path, layer: int, weights: dict | None = None):
    """All captured MoE inputs of ``layer``: (x fp32 [N, H], ids int64 [N, 8], row weight fp32 [N]). ``weights``
    {seq: w} picks the sequences and weights their rows in the Hessians (default: all, weight 1)."""
    xs, ids, ws = [], [], []
    for a in (a for a, l in enumerate(APP_LAYER) if l == layer):
        for f in sorted(cap.glob(f"a{a:02d}_s*.pt")):
            w = 1.0 if weights is None else weights.get(int(f.name.split("_")[1][1:]))
            if w is None:
                continue
            z = torch.load(f, weights_only=True)
            xs.append(z["x"])
            ids.append(z["ids"].long())
            ws.append(torch.full((len(z["x"]),), float(w)))
    return torch.cat(xs).float(), torch.cat(ids), torch.cat(ws)


def expert_weights(ck, layer: int):
    """bf16 dequant as the converter stores it, HF orientation: gate_up [E, 2I, H] (gate rows first), down [E, H, I]."""
    from models.demos.inkling2_mini.convert.source import deinterleave_gate_up, dequant_experts

    p = f"model.llm.layers.{layer}.mlp.experts."
    out = []
    for src, w13 in (("w13", True), ("w2", False)):
        base = p + src + "_weight"
        nv = ck.has(base + "_amax")
        t = dequant_experts(
            ck.experts(base, 0, E),
            ck.experts(base + "_scale", 0, E),
            ck.get(base + "_amax") if nv else None,
            ck.get(base + "_outer_scale_denominator") if nv else None,
        )
        out.append(deinterleave_gate_up(t) if w13 else t)
    return out


def solve_layer(ck, cap: Path, layer: int, args) -> dict:
    t0 = time.time()
    X, ids, rw = layer_inputs(cap, layer, args.weights)
    gb, db = (7 if layer in args.gu8 else 3), (7 if layer in args.dn8 else 3)  # mantissa bits: BFP8 / BFP4
    wgu, wdn = expert_weights(ck, layer)
    rows = [torch.nonzero((ids == e).any(1)).flatten() for e in range(E)]
    n_tok = [len(r) for r in rows]
    gu_q = torch.empty(E, 2 * I, H, dtype=torch.int8)
    gu_e = torch.empty(E, 2 * I // 16, H, dtype=torch.uint8)
    dn_q = torch.empty(E, H, I, dtype=torch.int8)
    dn_e = torch.empty(E, H // 16, I, dtype=torch.uint8)
    stats = {
        "tokens": len(X),
        "n_tok_min": min(n_tok),
        "n_tok_median": sorted(n_tok)[E // 2],
        "rtn_experts": [],
        "out_err": {},
    }
    err = {"gu_gptq": 0.0, "gu_rtn": 0.0, "dn_gptq": 0.0, "dn_rtn": 0.0, "gu_ref": 0.0, "dn_ref": 0.0}
    bs = args.batch
    for e0 in range(0, E, bs):
        es = list(range(e0, min(e0 + bs, E)))
        small = [e for e in es if n_tok[e] < args.min_tokens]
        stats["rtn_experts"] += small
        Hgu = torch.stack([(X[rows[e]] * rw[rows[e], None]).T @ X[rows[e]] for e in es])
        W = wgu[es].float()
        q, ex = gptq(W, Hgu, args.damp_gu, mbits=gb)
        for k, e in enumerate(es):
            if e in small:
                q[k], ex[k] = rtn(W[k], gb)
        gu_q[es], gu_e[es] = q, ex
        Wq = decode(q, ex, gb)
        hs = []
        for k, e in enumerate(es):
            x = X[rows[e]]
            y = x @ Wq[k].T
            hs.append(swiglu(y[:, :I], y[:, I:]))
            # output error on (up to) 1024 calibration rows (sums of squares), GPTQ vs RTN vs reference
            x, y = x[:1024], y[:1024]
            yr = x @ W[k].T
            yt = x @ decode(*rtn(W[k], gb), gb).T
            err["gu_ref"] += float((yr**2).sum())
            err["gu_gptq"] += float(((y - yr) ** 2).sum())
            err["gu_rtn"] += float(((yt - yr) ** 2).sum())
        Hdn = torch.stack([(h * rw[rows[e], None]).T @ h for e, h in zip(es, hs)])
        Wd = wdn[es].float()
        q, ex = gptq(Wd, Hdn, args.damp_dn, mbits=db)
        for k, e in enumerate(es):
            if e in small:
                q[k], ex[k] = rtn(Wd[k], db)
        dn_q[es], dn_e[es] = q, ex
        Wdq = decode(q, ex, db)
        for k, e in enumerate(es):
            h = hs[k][:1024]
            zr = h @ Wd[k].T
            err["dn_ref"] += float((zr**2).sum())
            err["dn_gptq"] += float(((h @ Wdq[k].T - zr) ** 2).sum())
            err["dn_rtn"] += float(((h @ decode(*rtn(Wd[k], db), db).T - zr) ** 2).sum())
        del Hgu, Hdn, hs, W, Wd, Wq, Wdq
        print(f"  L{layer:02d} experts {es[0]}-{es[-1]} {time.time() - t0:.0f}s", flush=True)
    # relative output MSE on the calibration rows (lower is better; RTN for comparison)
    stats["out_err"] = {
        "gate_up_gptq": err["gu_gptq"] / err["gu_ref"],
        "gate_up_rtn": err["gu_rtn"] / err["gu_ref"],
        "down_gptq": err["dn_gptq"] / err["dn_ref"],
        "down_rtn": err["dn_rtn"] / err["dn_ref"],
    }
    stats["seconds"] = round(time.time() - t0, 1)
    stats["mbits"] = {"gate_up": gb, "down": db}
    code = lambda q, mb: {"q8": q} if mb == 7 else {"q4": pack4(q)}  # noqa: E731
    return {"gate_up": code(gu_q, gb) | {"e": gu_e}, "down": code(dn_q, db) | {"e": dn_e}, "stats": stats}


def layer_list(spec: str) -> list[int]:
    return [x for part in spec.split(",") if part for x in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]


def solve(args):
    from models.demos.inkling2_mini.convert.source import Checkpoint

    torch.set_num_threads(args.threads)
    args.weights = {int(k): v for k, v in json.load(open(args.weights)).items()} if args.weights else None
    args.gu8, args.dn8 = set(layer_list(args.gu8)), set(layer_list(args.dn8))
    ck = Checkpoint(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = layer_list(args.layers)
    for layer in layers:
        f = out / f"L{layer:02d}.pt"
        if f.exists():
            continue
        z = solve_layer(ck, Path(args.capture), layer, args)
        tmp = f.with_suffix(".tmp")
        torch.save(z, tmp)
        os.replace(tmp, f)
        print(f"SOLVED L{layer:02d} {json.dumps(z['stats'])}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--out", required=True)
    c.add_argument("--tokenizer", default=os.environ.get("INKLING_CKPT"))
    c.add_argument("--chat", required=True, help="jsonl, one message list per line")
    c.add_argument("--code-root", required=True)
    c.add_argument("--prose-root", required=True)
    c.add_argument("--n-chat", type=int, default=12)
    c.add_argument("--n-code", type=int, default=6)
    c.add_argument("--n-prose", type=int, default=6)
    p = sub.add_parser("capture")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=None)
    s = sub.add_parser("solve")
    s.add_argument("--ckpt", default=os.environ.get("INKLING_CKPT"))
    s.add_argument("--capture", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--layers", default="0-23")
    s.add_argument("--damp-gu", type=float, default=0.01)
    s.add_argument("--damp-dn", type=float, default=1.0)
    s.add_argument("--min-tokens", type=int, default=16)
    s.add_argument("--batch", type=int, default=16, help="experts per batched solve (memory)")
    s.add_argument("--weights", help="json {seq: weight}: only these sequences, rows weighted in the Hessians")
    s.add_argument("--gu8", default="", help="layers whose gate_up is solved on the BFP8 grid (q8 codes)")
    s.add_argument("--dn8", default="", help="layers whose down is solved on the BFP8 grid (q8 codes)")
    s.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "40")))
    args = ap.parse_args(argv)
    {"corpus": corpus, "capture": capture, "solve": solve}[args.cmd](args)


if __name__ == "__main__":
    main()
