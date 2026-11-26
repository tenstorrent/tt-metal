# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Writes the CPU golden ($INKLING_GOLDEN) that the full-model test, the golden check and the demo compare with.

Host only, no ttnn: run it in a python with transformers >= 5.24 (``INKLING_REF_PYTHON``) on a machine with about
100 GB of RAM (the whole model is loaded in bf16; the full set of prompts took 666 s on 24 threads).

Loads the whole Inkling2LoopForConditionalGeneration in bf16 (experts dequantized by transformers on load) and runs
stock transformers code (eager attention). Hooks record, per layer application k = 0..47 (call order, see meta.json):
  in_{k}.pt        bf16 [1,T,H]       hidden_states input
  hidden_{k}.pt    bf16 [1,T,H]       layer output
  bs_{j}.pt        bf16 [1,T,nb,H]    block_states input (deduped; meta.json apps[k].bs -> j)
  msg_{k}.pt       bf16 [1,T,4,H]     messages (q,k,v,mlp), passes 2-3 only
  router_{k}.pt    {logits fp32 [T,E], weights fp32 [T,8] (scaled), indices int16 [T,8]}
Per prompt also: input_ids.pt, embed.pt, final_hidden.pt (after final attn-res + norm), kv/c{c:02d}.pt (every tensor
of cache layer c after prefill; c = L on pass 1, 24+(p-2)*12+(L-6) on passes 2-3, as the HF cache), logits:
  logits.pt fp32 [1,T,V] if T <= 64; logits_bf16.pt if T <= 2048; always logits_top32.pt {values, indices} +
  logits_lse.pt fp32 [T] + logits_last.pt fp32 [V]; top5.json at the last position.
greedy64_<name>.json for the greedy prompts; greedy.json = first 16 tokens of prompt0.

    $INKLING_REF_PYTHON models/demos/inkling2_mini/tests/make_golden.py $INKLING_CKPT $INKLING_GOLDEN [names...]

Prompt names: prompt0, prompt1 (the short prompts of tests/test_model.py and the demo), chat, math (test_mtp.py),
code1500, prose8k (the long prefill prompts of demo/perf.py). GOLDEN_THREADS sets the torch threads.
"""
import inspect
import json
import os
import resource
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer, DynamicCache, Inkling2LoopForConditionalGeneration
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

ckpt, out = sys.argv[1], sys.argv[2]
only = set(sys.argv[3:])
torch.set_num_threads(int(os.environ.get("GOLDEN_THREADS", "72")))
tok = AutoTokenizer.from_pretrained(ckpt)

# shortcut: a CPU with no bf16 matmul (e.g. Neoverse) runs bf16 gemm at ~1 GFLOP/s vs ~170 fp32, so bf16 matmuls
# are computed as fp32 and rounded back to bf16. Same as a bf16 gemm with fp32 accumulation, up to summation order.
# Set GOLDEN_FP32_COMPUTE=0 on a CPU with native bf16 matmul.
if os.environ.get("GOLDEN_FP32_COMPUTE", "1") == "1":
    _linear, _matmul = torch.nn.functional.linear, torch.matmul

    def _up(t):
        return t.float() if t is not None and t.dtype == torch.bfloat16 else t

    def linear(x, w, b=None):
        if torch.bfloat16 not in (x.dtype, w.dtype):
            return _linear(x, w, b)
        dt = torch.promote_types(x.dtype, w.dtype)
        return _linear(_up(x), _up(w), _up(b)).to(dt)

    def matmul(a, b, **kw):
        if torch.bfloat16 not in (a.dtype, b.dtype):
            return _matmul(a, b, **kw)
        return _matmul(_up(a), _up(b), **kw).to(torch.promote_types(a.dtype, b.dtype))

    torch.nn.functional.linear, torch.matmul = linear, matmul
    torch.Tensor.__matmul__ = lambda a, b: matmul(a, b)  # rel-pos bias uses `@`


def first_tokens(text, n):
    return tok.decode(tok.encode(text, add_special_tokens=False)[:n])


readme = open(os.path.join(ckpt, "README.md")).read()
code = open(M.__file__.replace("modeling_", "modular_")).read()  # transformers 5.24.0 source: fixed text
U = lambda c: {"role": "user", "content": c}
A = lambda c: {"role": "assistant", "content": c}
PROMPTS = {  # name -> messages
    "prompt0": [U("What is the capital of France? Answer in one sentence.")],
    "prompt1": [U("Write a haiku about a mountain lake at dawn.")],
    "code1500": [U("Explain what this Python code does, briefly.\n\n```python\n" + first_tokens(code, 1450) + "\n```")],
    "prose8k": [U("Summarize the following model card in three sentences.\n\n" + first_tokens(readme, 8400))],
    "chat": [
        U("Hi! I'm planning a trip to Kyoto in April."),
        A("That sounds lovely. April is cherry blossom season." " How many days will you stay?"),
        U("Five days. What should I not miss?"),
    ],
    "math": [U("Solve for x: 3x + 7 = 25. Show your steps.")],
}
GREEDY = {"prompt0": 64, "math": 64}

t0 = time.time()
config = AutoConfig.from_pretrained(ckpt)
del config.quantization_config  # modelopt is not a registered quantizer; the inkling2_loop mapping dequantizes experts
model = Inkling2LoopForConditionalGeneration.from_pretrained(
    ckpt, config=config, dtype=torch.bfloat16, experts_implementation="eager"
).eval()  # stock loop; grouped_mm runs bf16 torch.mm
load_s = time.time() - t0
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"full model loaded in {load_s:.0f}s, RSS {rss():.1f} GB", flush=True)

tm = model.model.language_model
tcfg = tm.config
ls, le, npass = tcfg.loop_layer_index_start, tcfg.loop_layer_index_end, tcfg.loop_num_passes
app_to_layer = list(range(ls)) + list(range(ls, le)) * npass + list(range(le, len(tm.layers)))
app_pass = [1] * le + [p for p in range(2, npass + 1) for _ in range(ls, le)] + [1] * (len(tm.layers) - le)
app_cache = [L if p == 1 else 24 + (p - 2) * 12 + (L - ls) for L, p in zip(app_to_layer, app_pass)]

rec = {"on": False}
S = {}


def save(name, t):
    torch.save(t.detach().contiguous().clone(), os.path.join(S["dir"], name))


def pre_hook(mod, args, kwargs):
    if not rec["on"]:
        return
    k = S["k"]
    hs, bs = args[0], args[1]
    msgs = args[2] if len(args) > 2 else kwargs.get("messages")
    save(f"in_{k:02d}.pt", hs.to(torch.bfloat16))
    prev = S["bs_prev"]
    if prev is None or prev.shape != bs.shape or not torch.equal(prev, bs):
        S["bs_n"] += 1
        save(f"bs_{S['bs_n']:02d}.pt", bs.to(torch.bfloat16))
        S["bs_prev"] = bs
    S["apps"].append(
        {
            "app": k,
            "layer": app_to_layer[k],
            "pass": app_pass[k],
            "cache": app_cache[k],
            "bs": S["bs_n"],
            "nb": bs.shape[-2],
            "msg": msgs is not None,
        }
    )
    if msgs is not None:
        save(f"msg_{k:02d}.pt", msgs.to(torch.bfloat16))


def post_hook(mod, args, kwargs, output):
    if rec["on"]:
        save(f"hidden_{S['k']:02d}.pt", output.to(torch.bfloat16))
        S["k"] += 1


def router_hook(mod, args, output):
    if rec["on"]:
        logits, weights, idx = output
        torch.save(
            {
                "logits": logits.float().clone(),
                "weights": weights.float().clone(),
                "indices": idx.to(torch.int16).clone(),
            },
            os.path.join(S["dir"], f"router_{S['k']:02d}.pt"),
        )


for layer in tm.layers:
    layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    layer.register_forward_hook(post_hook, with_kwargs=True)
    layer.mlp.gate.register_forward_hook(router_hook)
tm.embed_tokens.register_forward_hook(lambda m, a, o: save("embed.pt", o.to(torch.bfloat16)) if rec["on"] else None)
tm.norm.register_forward_hook(lambda m, a, o: save("final_hidden.pt", o.to(torch.bfloat16)) if rec["on"] else None)

cache_kw = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
meta = {
    "host": os.uname().nodename,
    "prompts": {},
    "app_to_layer": app_to_layer,
    "app_pass": app_pass,
    "app_cache": app_cache,
    "load_s": round(load_s, 1),
    "threads": torch.get_num_threads(),
    "experts_implementation": "eager",
    "fp32_compute": os.environ.get("GOLDEN_FP32_COMPUTE", "1") == "1",
}
os.makedirs(out, exist_ok=True)
with torch.inference_mode():
    for name, msgs in PROMPTS.items():
        if only and name not in only:
            continue
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, reasoning_effort="none", return_tensors="pt", return_dict=True
        )["input_ids"]
        d = S["dir"] = os.path.join(out, name)
        os.makedirs(os.path.join(d, "kv"), exist_ok=True)
        S.update(k=0, bs_prev=None, bs_n=-1, apps=[])
        torch.save(ids, os.path.join(d, "input_ids.pt"))
        cache = DynamicCache(config=tcfg)
        rec["on"] = True
        t = time.time()
        logits = model(input_ids=ids, use_cache=True, **{cache_kw: cache}).logits.float()
        dt = time.time() - t
        rec["on"] = False
        assert S["k"] == len(app_to_layer), S["k"]
        T = ids.shape[1]
        for c, layer in enumerate(cache.layers):
            v_ = vars(layer)
            torch.save(
                {a: v.clone() for a, v in v_.items() if isinstance(v, torch.Tensor)} |
                # conv_states: {state_idx: tensor}; 0/1 = k/v sconv, 2/3 = post-attn/post-mlp conv
                {
                    a: {i: x.clone() for i, x in v.items() if isinstance(x, torch.Tensor)}
                    for a, v in v_.items()
                    if a in ("conv_states", "recurrent_states")
                }
                | {a: v for a, v in v_.items() if type(v) in (int, bool)},
                os.path.join(d, "kv", f"c{c:02d}.pt"),
            )
        if T <= 64:
            torch.save(logits, os.path.join(d, "logits.pt"))
        if T <= 2048:
            torch.save(logits.to(torch.bfloat16), os.path.join(d, "logits_bf16.pt"))
        top = torch.topk(logits[0], 32, dim=-1)
        torch.save({"values": top.values, "indices": top.indices.int()}, os.path.join(d, "logits_top32.pt"))
        torch.save(torch.logsumexp(logits[0], -1), os.path.join(d, "logits_lse.pt"))
        torch.save(logits[0, -1].clone(), os.path.join(d, "logits_last.pt"))
        p = logits[0, -1].softmax(-1)
        t5 = torch.topk(p, 5)
        top5 = [{"id": int(i), "token": tok.decode([int(i)]), "prob": float(v)} for v, i in zip(t5.values, t5.indices)]
        json.dump(top5, open(os.path.join(d, "top5.json"), "w"), indent=1)
        json.dump(S["apps"], open(os.path.join(d, "apps.json"), "w"))
        meta["prompts"][name] = {
            "tokens": T,
            "prefill_s": round(dt, 1),
            "cache_layers": len(cache.layers),
            "messages": msgs,
        }
        del logits, cache
        print(f"{name}: T={T} prefill {dt:.1f}s top5={[x['token'] for x in top5]} RSS {rss():.1f} GB", flush=True)

        if name in GREEDY:
            t = time.time()
            gen = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=GREEDY[name], do_sample=False
            )
            new = gen[0, T:].tolist()
            g = {"prompt": name, "new_ids": new, "text": tok.decode(new), "seconds": round(time.time() - t, 1)}
            json.dump(g, open(os.path.join(out, f"greedy64_{name}.json"), "w"), indent=1)
            if name == "prompt0":
                json.dump(
                    {**g, "new_ids": new[:16], "text": tok.decode(new[:16])},
                    open(os.path.join(out, "greedy.json"), "w"),
                    indent=1,
                )
            print(f"greedy {name} ({g['seconds']}s): {g['text']!r}", flush=True)

meta["peak_rss_gb"] = round(rss(), 1)
meta["total_s"] = round(time.time() - t0, 1)
old = os.path.join(out, "meta.json")
if only and os.path.exists(old):  # partial rerun: keep the other prompts' entries
    prev = json.load(open(old))
    meta["prompts"] = prev.get("prompts", {}) | meta["prompts"]
json.dump(meta, open(old, "w"), indent=1)
print(f"done, peak RSS {meta['peak_rss_gb']} GB, total {meta['total_s']}s", flush=True)
