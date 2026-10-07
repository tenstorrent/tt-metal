# SPDX-License-Identifier: Apache-2.0
"""Long-context (64k) generation on the full 48-layer model: chunked traced prefill (tt/prefill.py) into the paged pool, then traced decode (tt/generator.py).
CTX_MODE=decode: B users (B/4 per mesh row), each a window of Frankenstein (Project Gutenberg, public domain; 102k tokens in the context_cache file, user b starts at
  body token b*CTX_STRIDE so windows overlap but differ) of CTX_ISL tokens + the 'quote and summarise' question, chat template with enable_thinking=False; CTX_NEW tokens decoded.
CTX_MODE=needle: passkey sentence inserted at depth CTX_DEPTHS (percent) of the window, question 'What is the pass key?', every depth CTX_REPEAT times (same prompt, different
  mesh rows), for each ISL in CTX_ISLS (comma list, one process).
Env: CTX_OUT (jsonl), CTX_TEXTS (json of generated texts), MIMO_CKPT, MIMO_EXPERT_OP."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference.prompts import BOOK, BOOK_QUESTION, tokenizer
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedGenerator, ChunkedPrefill

E = os.environ.get
OUT = E("CTX_OUT", "/tmp/ctx.jsonl")
CH = 2048


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("CTX_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    mb = lambda b: round(b / 2**20, 1)
    return dict(
        alloc_mb_per_bank=mb(v.total_bytes_allocated_per_bank), free_mb_per_bank=mb(v.total_bytes_free_per_bank)
    )


def chat_frame(tok, question):
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "XQXQ"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    pre_txt, post_txt = probe.split("XQXQ")
    return (
        tok(pre_txt, add_special_tokens=False)["input_ids"],
        tok(question + post_txt, add_special_tokens=False)["input_ids"],
    )


def book_body(tok):
    text = open(BOOK).read()
    text = text[text.index("*** START OF THE PROJECT GUTENBERG EBOOK") :]
    return tok(text, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 3 * 1024 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_ctx64k(mesh_device):
    md = mesh_device
    tok = tokenizer()
    body = book_body(tok)
    mode = E("CTX_MODE", "decode")
    new = int(E("CTX_NEW", "1100" if mode == "decode" else "24"))
    if mode == "decode":
        isls = [int(E("CTX_ISL", "65000"))]
        B = int(E("CTX_B", "4"))
    else:
        isls = [int(v) for v in E("CTX_ISLS", "65000").split(",")]
        depths = [int(v) for v in E("CTX_DEPTHS", "10,50,90").split(",")]
        rep = int(E("CTX_REPEAT", "2"))
        B = len(depths) * rep
    upr = -(-B // 4)
    max_pos = -(-(max(isls) + new + 16) // CH) * CH
    lay = KVLayout(users=upr, max_pos=max_pos, chunk=CH)
    rec("env", mode=mode, isls=isls, B=B, new=new, max_pos=max_pos, dram_before=dram(md), book_tokens=len(body))
    t0 = time.time()
    model = MiMoModel(md, n_layers=48, layout=lay, log=lambda s: None)
    rec("build", s=round(time.time() - t0, 1), dram=dram(md), cache=model.wc.stats)
    pf = ChunkedPrefill(model, trace=True)
    gen = ChunkedGenerator(model, max_pos, pf, trace=True, keep_logits=False)
    texts = {}
    for isl in isls:
        if mode == "decode":
            pre, post = chat_frame(tok, BOOK_QUESTION)
            stride = int(E("CTX_STRIDE", "2400"))
            n = isl - len(pre) - len(post)
            prompts = [pre + body[b * stride : b * stride + n] + post for b in range(B)]
            meta = [dict(user=b, start=b * stride) for b in range(B)]
        else:
            question = "\n\nWhat is the pass key mentioned in the text above? Answer with the number only."
            pre, post = chat_frame(tok, question)
            n = isl - len(pre) - len(post)
            prompts, meta = [], []
            keys = {d: str(10000 + (d * 7919 + isl) % 89999) for d in depths}
            for d in depths:
                needle = tok(
                    f" The pass key is {keys[d]}. Remember it. {keys[d]} is the pass key. ", add_special_tokens=False
                )["input_ids"]
                nb = n - len(needle)
                at = int(nb * d / 100)
                p = pre + body[:at] + needle + body[at:nb] + post
                assert len(p) == isl, (len(p), isl)
                for r_ in range(rep):
                    prompts.append(p)
                    meta.append(dict(depth=d, key=keys[d], repeat=r_))
            # order so that the repeats of a depth land on different mesh rows: i%4
            order = [j * rep + r_ for r_ in range(rep) for j in range(len(depths))]
            prompts, meta = [prompts[i] for i in order], [meta[i] for i in order]
        res = gen.generate(prompts, new, S=isl)
        times = res["times"]
        steady = sorted(times[5:])
        outs = [tok.decode(t, skip_special_tokens=True) for t in res["tokens"]]
        for m, o, t in zip(meta, outs, res["tokens"]):
            m["text"], m["ntok"] = o, len(t)
        out = dict(
            isl=isl,
            B=B,
            prefill_s=round(res["prefill_s"], 1),
            prefill_steps=gen.last_prefill_steps,
            decode_steps=len(times),
            step0_ms=round(times[0] * 1e3, 1) if times else None,
            step_ms_median_steady=round(steady[len(steady) // 2] * 1e3, 2) if steady else None,
            step_ms_p10_p90=[round(steady[len(steady) // 10] * 1e3, 2), round(steady[9 * len(steady) // 10] * 1e3, 2)]
            if steady
            else None,
            dram_after=dram(md),
            first_tokens=[t[0] for t in res["tokens"]],
        )
        if mode == "needle":
            for m in meta:
                m["hit"] = m["key"] in m["text"]
            out["hits"] = sum(m["hit"] for m in meta)
            out["of"] = len(meta)
            out["by"] = [(m["depth"], m["repeat"], m["hit"], m["text"][:60]) for m in meta]
        rec("result", **{k: v for k, v in out.items()})
        texts[str(isl)] = meta
        json.dump(texts, open(E("CTX_TEXTS", OUT.replace(".jsonl", "_texts.json")), "w"), ensure_ascii=False)
    gen.dec.release()
    pf.release()
