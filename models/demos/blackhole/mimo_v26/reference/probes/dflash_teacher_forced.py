"""Teacher-forced DFlash draft quality on the DEVICE captures (no target-model run): for anchors p in [P0, 120] of every sequence, feed the drafter the device hidden states of
layers [0,11,23,35,47] at positions < p and the gold token at p; compare its 7 draft tokens (target lm_head, bf16) with the gold tokens p+1..p+7.
Reports per-position top-1 accuracy and the expected accepted length per round (leading matches + 1) UNDER THE ASSUMPTION that the target's greedy token equals the gold token
(an upper-bound proxy: the gold text is the GSM8K question + human solution, not the model's own generation).
    python -m ...probes.dflash_teacher_forced <device_dir> <out.json> [--users 16] [--grid]
"""
import argparse
import json
import os
import time

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

from models.demos.blackhole.mimo_v26.reference import dflash_ref as D
from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import prompts as P
from models.demos.blackhole.mimo_v26.reference import ref_layer as R

torch.set_num_threads(32)
LAYERS = D.TARGET_LAYERS


def features(dev, conv, Wn, E, toks):
    """returns [B,S,5*4096] under a convention: 'out' (output of layer L, L47 pre-norm), 'out_postnorm' (L47 after model.norm = HF hidden_states[-1]), 'in' (input of layer L)."""
    fs = []
    for L in LAYERS:
        d = torch.load(f"{dev}/layer_{L}.pt")["prefill"]
        if conv == "in":
            fs.append(d["h_in"])
        elif conv == "out_postnorm" and L == 47:
            fs.append(R.rms_norm(d["h_out"], Wn))
        else:
            fs.append(d["h_out"])
    return torch.cat(fs, -1)


@torch.no_grad()
def run(feats, gold, sd, E, head, anchors, users, variant):
    B, S, _ = feats.shape
    hn = D.ctx_features(feats, sd)
    hits = []  # per anchor [7] bool
    meta = []
    for b in users:
        pos = torch.tensor(anchors)
        h = D.draft_blocks(hn[b], gold[b, pos], pos, sd, E, variant)  # [A,7,4096]
        pred = torch.cat([(h[i : i + 8].reshape(-1, 4096) @ head.T).argmax(-1) for i in range(0, h.shape[0], 8)]).view(
            -1, 7
        )
        tgt = torch.stack([gold[b, p + 1 : p + 8] for p in anchors])
        hits.append(pred == tgt)
        meta += [(b, p) for p in anchors]
    hits = torch.cat(hits)
    acc = hits.float().mean(0)
    acc_len = hits.long().cumprod(1).sum(1).float() + 1
    return acc.tolist(), acc_len.mean().item(), hits, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--p0", type=int, default=16)
    a = ap.parse_args()
    ck = Ld.Ckpt()
    E = ck.get("model.embed_tokens.weight")
    head = ck.get("lm_head.weight")
    Wn = ck.get("model.norm.weight")
    tok = P.tokenizer()
    toks, texts = P.gsm8k_tokens(16, 128, tok=tok)
    gold = torch.tensor([tok(t, add_special_tokens=False)["input_ids"][:128] for t in texts])
    assert torch.equal(gold[:, :128], toks)
    d0 = torch.load(f"{a.dev}/layer_0.pt")["prefill"]["h_in"]
    print(
        "device layer_0 h_in == embedding(tokens):",
        torch.equal(d0.to(torch.bfloat16), E[toks]),
        " max abs",
        (d0.float() - E[toks].float()).abs().max().item(),
        flush=True,
    )
    sd = D.load_dflash()
    print(
        "mask embedding == target embed row 151675:",
        torch.equal(sd["mask_embedding"], E[sd["mask_token_id"]]),
        flush=True,
    )
    prompt_len = [
        len(
            tok(
                tok.apply_chat_template(
                    [{"role": "user", "content": json.loads(l)["question"]}], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        for l, _ in zip(open(P.GSM8K), range(16))
    ]
    res = {"prompt_len": prompt_len}
    users = list(range(a.users))
    anchors_full = list(range(a.p0, 121))
    if a.grid:
        sub, anc = users, list(range(a.p0, 121, 2))
        for conv in ("out", "out_postnorm", "in"):
            feats = features(a.dev, conv, Wn, E, toks)
            for name, var in (
                ("base", {}),
                ("nosink", dict(sink=False)),
                ("vscale1", dict(vscale=1.0)),
                ("ropefull", dict(rope="full")),
            ):
                t0 = time.time()
                acc, L, _, _ = run(feats, gold, sd, E, head, anc, sub, var)
                res[f"grid/{conv}/{name}"] = {"acc": acc, "acc_len": L}
                print(
                    conv,
                    name,
                    "acc",
                    [round(x, 3) for x in acc],
                    "acc_len %.2f" % L,
                    "%.0fs" % (time.time() - t0),
                    flush=True,
                )
        json.dump(res, open(a.out, "w"))
        return
    best = json.loads(os.environ.get("DFLASH_CHOICE", '{"conv":"out","variant":{}}'))
    feats = features(a.dev, best["conv"], Wn, E, toks)
    acc, L, hits, meta = run(feats, gold, sd, E, head, anchors_full, users, best["variant"])
    ans = torch.tensor(
        [p >= prompt_len[b] - 1 for b, p in meta]
    )  # anchors whose block lies in the solution region (anchor token is the first answer-side token)
    res.update(
        {
            "choice": best,
            "n_anchors": len(meta),
            "per_pos_top1": acc,
            "expected_accept_len": L,
            "n_answer_region_anchors": int(ans.sum()),
        }
    )
    if ans.any():
        h2 = hits[ans]
        res["answer_region"] = {
            "per_pos_top1": h2.float().mean(0).tolist(),
            "expected_accept_len": (h2.long().cumprod(1).sum(1).float() + 1).mean().item(),
        }
    q = ~ans
    h3 = hits[q]
    res["question_region"] = {
        "per_pos_top1": h3.float().mean(0).tolist(),
        "expected_accept_len": (h3.long().cumprod(1).sum(1).float() + 1).mean().item(),
        "n": int(q.sum()),
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
