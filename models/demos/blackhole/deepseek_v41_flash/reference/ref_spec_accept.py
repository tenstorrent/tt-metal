# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU measurement of the DSpark (mtp.*) draft acceptance on REAL text, with the whole 40-layer reference resident in RAM.

    1. B chat prompts (real text, padded with filler words to one token length S) are prefilled through the backbone.
    2. G tokens are generated GREEDILY with the plain backbone (this is the stream speculative decoding must reproduce).
    3. After the prefill and after every decode step the checkpoint's own ``forward_spec`` logic (embed [token, noise x4] ->
       3 DSpark stages -> markov-biased greedy drafts) drafts 5 tokens from (sampled token, mean-over-streams hidden of
       the inputs of layers 37/38/39).

Because the draft only depends on the stream prefix (not on earlier drafts), the exact greedy-acceptance statistics of any k
can be computed offline from the saved stream + drafts (``analyse``). Saved to --out/results.pt after every step.

Speed patches (math unchanged): fp4 dequant via a byte LUT; fp8 dequant memoised (bf16-exact values, kept in fp32).
Run:  nice -n 10 python -m models.demos.blackhole.deepseek_v41_flash.reference.ref_spec_accept --B 8 --G 64
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels as K
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R

CKPT = R.CKPT_DIR
OUT = "/mnt/tt-data/ssinghal/dsv4-spec-accept"

PROMPTS = [
    "Explain in simple terms how a transformer neural network processes a sentence, and why attention is useful.",
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case, and show two example calls.",
    "What are the main causes of the French Revolution? Give a short, well structured answer.",
    "Translate the following sentence into German and French: The weather is lovely today, so we are going for a walk by the river.",
    "Give me a recipe for a simple vegetarian pasta dinner for two people, with ingredients and numbered steps.",
    "Summarize the plot of Romeo and Juliet in about five sentences.",
    "How does a binary search work, and what is its time complexity? Include a short example with a sorted list of numbers.",
    "Write a short, friendly email to a colleague asking to move our weekly meeting from Tuesday to Thursday afternoon.",
    "What is the difference between a stack and a queue in computer science? Give a real-world analogy for each.",
    "Describe three practical tips for staying focused while working from home.",
    "Explain what photosynthesis is and why it matters for life on Earth.",
    "Write a short poem about the ocean at sunrise.",
]

_LUT = torch.stack([K.FP4_TABLE[torch.arange(256) & 15], K.FP4_TABLE[torch.arange(256) >> 4]], -1).to(torch.bfloat16)


def _fast_dequant_fp4(w, s, block=32):
    b = w.view(torch.uint8)
    n = b.size(0)
    pair = torch.nn.functional.embedding(b.long(), _LUT).reshape(n, -1)
    return (pair.view(n, -1, block).float() * s.float().unsqueeze(-1)).reshape(n, -1)


_fp8_memo = {}
_orig_fp8 = K.dequant_fp8_weight


def _memo_dequant_fp8(w, s, block=32):
    key = (w.data_ptr(), tuple(w.shape))
    r = _fp8_memo.get(key)
    if r is None:
        r = _orig_fp8(w, s, block)
        _fp8_memo[key] = r
    return r


_fp4_memo, _fp4_bytes = {}, [0]
_FP4_CAP = float(os.environ.get("DSV41_FP4_CACHE_GB", "120")) * 2**30


def _memo_dequant_fp4(w, s, block=32):
    key = (w.data_ptr(), tuple(w.shape))
    r = _fp4_memo.get(key)
    if r is not None:
        return r.float()
    r = _fast_dequant_fp4(w, s, block)
    if _fp4_bytes[0] < _FP4_CAP:  # bf16 is exact for fp4 x e8m0 values; hot experts are re-used every step
        _fp4_memo[key] = r.to(torch.bfloat16)
        _fp4_bytes[0] += r.numel() * 2
    return r


K.dequant_fp4_weight = _memo_dequant_fp4
K.dequant_fp8_weight = _memo_dequant_fp8


def load_into(block, index, prefix, handles=None):
    """Same logic as ref_layer.build_layer, for any parameter prefix (``layers.N.`` / ``mtp.N.``)."""
    handles = {} if handles is None else handles
    with torch.no_grad():
        for name, p in block.named_parameters():
            key = prefix + name
            if key not in index:
                assert name.endswith("bias_vl"), key
                continue
            path = os.path.join(CKPT, index[key])
            if path not in handles:
                handles[path] = safe_open(path, "pt")
            t = handles[path].get_tensor(key)
            if p.dtype == torch.float4_e2m1fn_x2:
                t = t.view(torch.float4_e2m1fn_x2)
            elif t.dtype == torch.float8_e4m3fn and p.dtype != torch.float8_e4m3fn:
                skey = key.replace(".weight", ".scale")
                spath = os.path.join(CKPT, index[skey])
                if spath not in handles:
                    handles[spath] = safe_open(spath, "pt")
                s = handles[spath].get_tensor(skey)
                t = _orig_fp8(t, s, t.size(0) // s.size(0)).to(p.dtype)
            elif t.dtype != p.dtype:
                t = t.to(p.dtype)
            assert t.shape == p.shape, f"{key}: {tuple(t.shape)} vs {tuple(p.shape)}"
            p.data = t
    block.eval()
    return block


def build_backbone_layer(L, B, S_max):
    blk = R.build_layer(L, max_batch_size=B, max_seq_len=S_max)
    return blk


def build_mtp(mod, args, index, embed_w, head_w, B):
    """The 3 DSpark stages with tied embedding / head, like ``Transformer.__init__``."""
    embed = mod.ParallelEmbedding(args.vocab_size, args.dim)
    embed.weight.data = embed_w
    head = mod.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    head.weight.data = head_w.float()
    stages = []
    for i in range(args.n_mtp_layers):
        b = mod.DSparkBlock(args.n_layers + i, args)
        load_into(b, index, f"mtp.{i}.")
        b.embed, b.head = embed, head
        stages.append(b)
    return stages


@torch.no_grad()
def forward_spec(stages, mod, input_ids, main_hidden, start_pos, hc=4):
    h, main_x = stages[0].forward_embed(main_hidden, input_ids)
    pre_mix = mod.make_identity_pre_mix(h, hc)
    for layer in stages:
        h, pre_mix = layer(h, start_pos, pre_mix, main_x)
    if start_pos == 0:
        return None
    return stages[-1].forward_head(h, pre_mix, input_ids)


def make_prompts(tokenizer, B):
    sys.path.insert(0, os.path.join(CKPT, "encoding"))
    from encoding import encode_messages

    texts = PROMPTS[:B]
    gsm = os.environ.get("DSV41_GSM8K")
    if gsm:  # public GSM8K test questions (jsonl on NFS); DSV41_GSM8K_OFFSET picks the first question
        off = int(os.environ.get("DSV41_GSM8K_OFFSET", "0"))
        qs = [json.loads(l)["question"] for l in open(gsm)]
        from collections import defaultdict

        by_len = defaultdict(list)
        for q in qs:
            by_len[
                len(tokenizer.encode(encode_messages([{"role": "user", "content": q}], thinking_mode="chat")))
            ].append(q)
        L = max((l for l in by_len if 60 <= l <= 110), key=lambda l: (len(by_len[l]) >= B, -abs(l - 80)))
        texts = by_len[L][off : off + B]  # B questions of exactly the same templated length: no padding needed
        assert len(texts) == B
    toks = [tokenizer.encode(encode_messages([{"role": "user", "content": t}], thinking_mode="chat")) for t in texts]
    S = max(len(t) for t in toks)
    out = []
    for t, tx in zip(toks, texts):
        extra = ""
        while len(t) < S:
            extra += " please"
            t = tokenizer.encode(encode_messages([{"role": "user", "content": tx + extra}], thinking_mode="chat"))
            if len(t) > S:  # overshoot (multi-token filler): fall back to single filler tokens
                raise RuntimeError("filler overshoot")
        out.append(t)
    assert all(len(t) == S for t in out), [len(t) for t in out]
    return texts, torch.tensor(out), S


def analyse(res, ks=(1, 2, 3, 4, 5)):
    """Exact greedy-spec statistics from the saved stream: per-position match rates and the simulated tokens/step for each k."""
    stream = res["stream"]  # [B, S + n + 1]: prompt, then generated tokens (stream[:, S] is the first generated token)
    S = res["S"]
    drafts = torch.stack(
        res["drafts"], 0
    )  # [steps, B, 6]: [t_{i+1}, d1..d5] for the step whose input was position S + step
    steps, B, _ = drafts.shape
    n = stream.shape[1]
    out = {}
    # per-draft-position match rate over all (step, user) where the stream is long enough
    for j in range(1, 6):
        ok = tot = 0
        pre_ok = 0
        for t in range(steps):
            i = S + t  # position of the step's input token; drafts[:, j] is for position i + 1 + j
            if i + 1 + j < n:
                eq = drafts[t, :, j] == stream[:, i + 1 + j]
                tot += B
                ok += int(eq.sum())
        out[f"match_d{j}"] = ok / max(tot, 1)
    # accepted prefix length m(t, u) in 0..5 (drafts matching consecutively)
    acc = []
    for t in range(steps):
        i = S + t
        m = torch.zeros(B, dtype=torch.long)
        alive = torch.ones(B, dtype=torch.bool)
        valid = torch.ones(B, dtype=torch.bool)
        for j in range(1, 6):
            if i + 1 + j >= n:
                valid &= False
                break
            alive &= drafts[t, :, j] == stream[:, i + 1 + j]
            m += alive.long()
        acc.append((m, i + 1 + 5 < n))
    for k in ks:
        # mean accepted drafts per round for k draft tokens, all start positions with full lookahead
        v = [torch.clamp(m, max=k).float().mean().item() for m, full in acc if full]
        out[f"mean_accepted_k{k}"] = sum(v) / max(len(v), 1)
        # P(all of the first j accepted) for j<=k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--G", type=int, default=64)
    ap.add_argument("--layers", type=int, default=40, help="smoke test: use only the first N backbone layers")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--load-threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    K.FAKE_QUANT = False
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    mod = R.load_model_module()
    torch.set_default_dtype(torch.bfloat16)
    args = R.model_args(a.B, 256)
    args.temperature = 0
    nl = a.layers
    targets = tuple(args.dspark_target_layer_ids) if nl == 40 else tuple(range(nl - 3, nl))
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    texts, prompt, S = make_prompts(tok, a.B)
    log(f"B={a.B} prompt tokens S={S} G={a.G} targets={targets}")

    from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEmbedding, HostEngram, HostHead

    emb, head_host = HostEmbedding(), HostHead()
    engram = HostEngram(max_batch_size=a.B, max_seq_len=256)

    t0 = time.time()
    with ThreadPoolExecutor(a.load_threads) as ex:
        futs = [ex.submit(build_backbone_layer, L, a.B, 256) for L in range(nl)]
        blocks = []
        for L, f in enumerate(futs):
            blocks.append(f.result())
            if L % 5 == 4:
                log(f"built layers 0..{L} ({time.time() - t0:.0f}s)")
    stages = build_mtp(mod, args, index, emb.weight, head_host.head_w, a.B)
    log(f"all blocks + {len(stages)} DSpark stages built in {time.time() - t0:.0f}s")

    @torch.no_grad()
    def backbone(tokens, start_pos):
        """tokens [B, s] -> (next token [B], main_hidden [B, s, 15360])"""
        h, pm = emb(tokens)
        hashes = engram.hashes(tokens, start_pos)
        hid = []
        for L, blk in enumerate(blocks):
            if L in engram.mods:
                h = engram.apply(L, h, hashes)
            if L in targets:
                hid.append(h.mean(dim=2))
            h, pm = blk(h, start_pos, pm, None)
        logits = head_host(h, pm)
        return logits.argmax(-1), torch.cat(hid, -1)

    res = {
        "texts": texts,
        "prompt": prompt,
        "S": S,
        "stream": None,
        "drafts": [],
        "conf": [],
        "main_hidden_dec": [],
        "step_s": [],
        "args": vars(a),
    }
    t = time.time()
    nxt, hid_pre = backbone(prompt, 0)
    log(f"prefill done in {time.time() - t:.0f}s")
    res["main_hidden_pre"] = hid_pre.to(torch.bfloat16)
    forward_spec(stages, mod, nxt, hid_pre, 0)  # seeds the window caches of the 3 stages
    stream = torch.cat([prompt, nxt[:, None]], 1)
    for step in range(a.G):
        t = time.time()
        pos = S + step
        nxt, hid = backbone(stream[:, -1:], pos)
        t_bb = time.time() - t
        out_ids, _logits, conf = forward_spec(stages, mod, nxt, hid, pos)
        stream = torch.cat([stream, nxt[:, None]], 1)
        res["drafts"].append(out_ids.clone())  # [B, 6] = [t_{pos+1}, d1..d5]
        res["conf"].append(conf.float().clone())
        res["main_hidden_dec"].append(hid[:, 0].to(torch.bfloat16))
        res["step_s"].append(time.time() - t)
        res["stream"] = stream
        if step % 4 == 3 or step == a.G - 1:
            torch.save(res, os.path.join(a.out, "results.pt"))
            log(f"step {step}: {res['step_s'][-1]:.0f}s  stats {analyse(res)}")
        else:
            log(
                f"step {step}: {res['step_s'][-1]:.0f}s (backbone {t_bb:.0f}s) d1 {out_ids[:, 1].tolist()} stream {nxt.tolist()}"
            )
    for b in range(a.B):
        log(f"user {b}: {tok.decode(stream[b, S:].tolist())!r}")
    log("DONE " + json.dumps(analyse(res)))


if __name__ == "__main__":
    main()
