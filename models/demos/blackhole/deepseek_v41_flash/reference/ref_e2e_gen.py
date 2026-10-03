# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference GREEDY generation on real GSM8K prompts (chat template, non-thinking), cached for the end-to-end accuracy harness.

All B questions are padded (trailing newline tokens) / selected to ONE templated token length S so that the device decode (shared step
position) can run them together. Saved under --out:
  meta.pt        {idx, questions, gold, prompt [B,S], S, G}
  results.pt     {stream [B, S+n] (prompt + greedy tokens), done_at [B], topk_val/topk_idx [n,B,50], lse [n,B]}  (rewritten every few steps)
  logits_tf.pt   bf16 full logits of the first --full-logits steps [n,B,vocab] (teacher-forced KL on the device side)
  state/layer_L.pt  decode state after the PREFILL (window / comp / kv_state / score_state, gate_cutoff): bootstraps the device decode from CPU state
  kv_final/layer_L.pt  window + comp caches after the whole generation (per-layer KV error of the device-written state)
Row r of step i's logits predicts stream[:, S+i]  (step 0 = prefill logits of the last prompt token).
Run: nice python -m models.demos.blackhole.deepseek_v41_flash.reference.ref_e2e_gen --B 64 --S 55 --G 256
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels as K
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference import ref_spec_accept as SA  # fast dequant patches + helpers

OUT = "/mnt/tt-data/ssinghal/dsv4-e2e-ref"
GSM = "/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl"


def select_prompts(tok, B, S, offset=0):
    sys.path.insert(0, os.path.join(R.CKPT_DIR, "encoding"))
    from encoding import encode_messages

    enc = lambda q: tok.encode(encode_messages([{"role": "user", "content": q}], thinking_mode="chat"))
    rows = [json.loads(l) for l in open(GSM)]
    exact, padded = [], []
    for i, r in enumerate(rows):
        n = len(enc(r["question"]))
        if n == S:
            exact.append((i, r["question"], enc(r["question"])))
        elif S - 3 <= n < S:
            q = r["question"]
            for fill in ("\n", " ", "\n\n"):
                qq, t = q, enc(q)
                for _ in range(8):
                    if len(t) >= S:
                        break
                    qq += fill
                    t = enc(qq)
                if len(t) == S:
                    padded.append((i, qq, t))
                    break
    pool = (exact + padded)[offset : offset + B]
    assert len(pool) == B, (len(exact), len(padded))
    return (
        [p[0] for p in pool],
        [p[1] for p in pool],
        torch.tensor([p[2] for p in pool]),
        [rows[p[0]]["answer"] for p in pool],
        len(exact),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--S", type=int, default=55)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--full-logits", type=int, default=64)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--load-threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out + "/state", exist_ok=True)
    os.makedirs(a.out + "/kv_final", exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    K.FAKE_QUANT = False
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(R.CKPT_DIR)
    eos = tok.eos_token_id
    idx, questions, prompt, gold, n_exact = select_prompts(tok, a.B, a.S, a.offset)
    S = prompt.shape[1]
    log(f"B={a.B} S={S} (exact-length questions available: {n_exact}) eos={eos}")
    torch.save(
        {"idx": idx, "questions": questions, "gold": gold, "prompt": prompt, "S": S, "G": a.G}, a.out + "/meta.pt"
    )
    torch.set_default_dtype(torch.bfloat16)
    from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEmbedding, HostEngram, HostHead

    emb, head = HostEmbedding(), HostHead()
    max_seq = S + a.G + 8
    engram = HostEngram(max_batch_size=a.B, max_seq_len=max_seq)
    t0 = time.time()
    with ThreadPoolExecutor(a.load_threads) as ex:
        futs = [ex.submit(SA.build_backbone_layer, L, a.B, max_seq) for L in range(40)]
        blocks = []
        for L, f in enumerate(futs):
            blocks.append(f.result())
            if L % 5 == 4:
                log(f"built layers 0..{L} ({time.time() - t0:.0f}s)")

    def backbone(tokens, start_pos, dump_state=False):
        h, pm = emb(tokens)
        hashes = engram.hashes(tokens, start_pos)
        for L, blk in enumerate(blocks):
            if L in engram.mods:
                h = engram.apply(L, h, hashes)
            h, pm = blk(h, start_pos, pm, None)
            if dump_state:
                ratio = blk.attn.compress_ratio
                st = {"window": blk.attn.window_kv_cache.clone().float()}
                if ratio and blk.attn.is_kv_source:
                    st["comp"] = blk.attn.compress_kv_cache[:, : S // ratio].clone().float()
                    if ratio > 1:
                        st["kv_state"] = blk.attn.compressor.kv_state.clone()
                        st["score_state"] = blk.attn.compressor.score_state.clone()
                torch.save(
                    {
                        "state": st,
                        "S": S,
                        "ratio": ratio,
                        "is_kv_source": bool(ratio and blk.attn.is_kv_source),
                        "gate_cutoff": 0.0,
                    },
                    f"{a.out}/state/layer_{L}.pt",
                )
        return head(h, pm)

    t = time.time()
    logits = backbone(prompt, 0, dump_state=True)
    log(f"prefill done in {time.time() - t:.0f}s")
    stream, done_at = prompt, torch.full((a.B,), -1, dtype=torch.long)
    tv, ti, lse, full = [], [], [], []
    for step in range(a.G):
        t = time.time()
        lg = logits.float()
        v, i = lg.topk(50, -1)
        tv.append(v), ti.append(i), lse.append(torch.logsumexp(lg, -1))
        if step < a.full_logits:
            full.append(lg.to(torch.bfloat16))
        nxt = lg.argmax(-1)
        nxt = torch.where(done_at >= 0, torch.full_like(nxt, eos), nxt)  # finished users keep feeding EOS
        newly = (nxt == eos) & (done_at < 0)
        done_at[newly] = step
        stream = torch.cat([stream, nxt[:, None]], 1)
        res = {
            "stream": stream,
            "done_at": done_at,
            "topk_val": torch.stack(tv),
            "topk_idx": torch.stack(ti),
            "lse": torch.stack(lse),
            "S": S,
        }
        if step % 8 == 7 or bool((done_at >= 0).all()) or step == a.G - 1:
            torch.save(res, a.out + "/results.pt")
        if step == a.full_logits - 1:
            torch.save(torch.stack(full), a.out + "/logits_tf.pt")
        log(f"step {step}: {time.time() - t:.0f}s  done {int((done_at >= 0).sum())}/{a.B}")
        if bool((done_at >= 0).all()) or step == a.G - 1:
            break
        logits = backbone(stream[:, -1:], S + step)
    if len(full) and not os.path.exists(a.out + "/logits_tf.pt"):
        torch.save(torch.stack(full), a.out + "/logits_tf.pt")
    n = stream.shape[1]
    for L, blk in enumerate(blocks):  # final KV of every layer (positions < n-1 have been written)
        ratio = blk.attn.compress_ratio
        st = {"window": blk.attn.window_kv_cache.clone().float(), "n_written": n - 1}
        if ratio and blk.attn.is_kv_source:
            st["comp"] = blk.attn.compress_kv_cache[:, : (n - 1) // ratio].clone().float()
        torch.save(st, f"{a.out}/kv_final/layer_{L}.pt")
    for b in range(a.B):
        log(f"user {b}: {tok.decode(stream[b, S:].tolist())[:300]!r}")
    log("DONE")


if __name__ == "__main__":
    main()
