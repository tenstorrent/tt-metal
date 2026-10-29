# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming chat client for a running `vllm serve` of the resident model (stdlib HTTP; transformers for decoding).

parity: the prompts of prompts.py at temperature 0, token ids returned; with --ref (greedy_ref.py output, or the
    output of another parity run) each served token stream is compared with the reference: prompt ids equal,
    first differing token index, and the reference's top-1 - top-2 logit gap there when it has one.
speed:  per request TTFT (first token) and decode rate (tokens after the first / time after the first), with the
    given sampling parameters (omitted ones take the server's generation-config defaults). --doc FILE --doc-tokens N
    sends one prompt of about N tokens: an instruction and the start of FILE.

compare: two saved runs offline (--served parity output, --ref greedy_ref.py or parity output).

usage: python -m qwen36_resident.serve.chat_client parity --port 8000 --out served.json [--ref ref.json]
       python -m qwen36_resident.serve.chat_client compare --served served.json --ref ref.json --out cmp.json
       python -m qwen36_resident.serve.chat_client speed --port 8000 --out s.json [--temperature 0] [--top-k 20]
           [--top-p 0.95] [--max-tokens 256] [--thinking on|off|both] [--repeat 2] [--doc F --doc-tokens 2048]
"""
import argparse
import json
import os
import time
import urllib.request

from qwen36_resident.serve.prompts import PROMPTS, messages


def model_name(port):
    with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def chat(port, model, msgs, max_tokens, thinking, sampling, ignore_eos=False):
    """one streamed chat request: prompt ids, generated ids, text, TTFT and per-chunk arrival times"""
    body = dict(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        stream=True,
        stream_options=dict(include_usage=True),
        return_token_ids=True,
        chat_template_kwargs=dict(enable_thinking=thinking),
        **sampling,
    )
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    prompt_ids, ids, text, times, finish, usage = None, [], [], [], None, None
    load0 = os.getloadavg()[0]
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("prompt_token_ids") is not None:
                prompt_ids = d["prompt_token_ids"]
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("token_ids"):
                    ids += c["token_ids"]
                    times.append((time.perf_counter() - t0, len(c["token_ids"])))
                delta = c.get("delta") or {}
                text.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
                finish = c.get("finish_reason") or finish
    ttft = times[0][0] if times else None
    n_after = len(ids) - times[0][1] if times else 0
    decode_s = times[-1][0] - times[0][0] if times else 0.0
    return dict(
        prompt_ids=prompt_ids,
        token_ids=ids,
        text="".join(text),
        ttft_s=ttft,
        decode_tokens=n_after,
        decode_s=decode_s,
        decode_tok_s=n_after / decode_s if decode_s > 0 else None,
        chunks=len(times),
        # (arrival s, tokens) per chunk, and the host's 1-minute load average before and after the request
        chunk_times=[(round(t, 5), n) for t, n in times],
        loadavg=(load0, os.getloadavg()[0]),
        finish=finish,
        usage=usage,
    )


def ref_tokens(r):
    return r.get("host_tokens", r.get("token_ids"))


def compare(served, ref, tok):
    """served row vs reference row: first differing generated token over the served length"""
    want = ref_tokens(ref)
    got = served["token_ids"]
    n = len(got)
    first = next((k for k in range(min(n, len(want))) if got[k] != want[k]), None)
    if first is None and n > len(want):
        first = len(want)
    out = dict(
        prompt_ids_equal=served["prompt_ids"] == ref["prompt_ids"],
        compared=n,
        first_diff=first,
        text_equal=tok.decode(got, skip_special_tokens=True) == tok.decode(want[:n], skip_special_tokens=True),
    )
    if first is not None:
        out["served_tok"], out["ref_tok"] = got[first], want[first] if first < len(want) else None
        if "host_gaps" in ref and first < len(ref["host_gaps"]):
            out["ref_gap"] = ref["host_gaps"][first]
    return out


def check(r, ref, tok):
    """compare r with its reference row (when there is one): (equal, message)"""
    if ref is None:
        return True, ""
    c = r["vs_ref"] = compare(r, ref, tok)
    ok = c["prompt_ids_equal"] and c["first_diff"] is None and c["text_equal"]
    msg = f" | vs ref: prompt ids {'==' if c['prompt_ids_equal'] else '!='}, first diff {c['first_diff']}"
    if c["first_diff"] is not None:
        msg += f" (served {c.get('served_tok')} ref {c.get('ref_tok')}, ref gap {c.get('ref_gap')})"
    return ok, msg + f", text {'==' if c['text_equal'] else '!='}"


def load_rows(path):
    return {r["name"]: r for r in json.load(open(path))["prompts"]}


def run_parity(a, tok):
    model = model_name(a.port)
    ref = load_rows(a.ref) if a.ref else {}
    rows, ok = [], True
    for p in PROMPTS:
        r = chat(a.port, model, messages(p), a.max_tokens, p["thinking"], dict(temperature=0.0))
        r.update(name=p["name"], thinking=p["thinking"])
        msg = f"{p['name']}: {len(r['token_ids'])} tok, finish {r['finish']}, TTFT {r['ttft_s']:.3f} s"
        if r["decode_tok_s"]:
            msg += f", decode {r['decode_tok_s']:.1f} tok/s"
        eq, m = check(r, ref.get(p["name"]), tok)
        ok &= eq
        print(msg + m, flush=True)
        rows.append(r)
    json.dump(
        dict(model=model, max_tokens=a.max_tokens, prompts=rows, all_equal=ok if ref else None),
        open(a.out, "w"),
        indent=1,
    )
    if ref:
        print("PARITY " + ("PASS" if ok else "FAIL"), flush=True)


def run_compare(a, tok):
    served, ref = load_rows(a.served), load_rows(a.ref)
    ok = True
    for name, r in served.items():
        eq, m = check(r, ref.get(name), tok)
        ok &= eq
        print(f"{name}: {len(r['token_ids'])} tok" + m, flush=True)
    json.dump(dict(served=a.served, ref=a.ref, prompts=list(served.values()), all_equal=ok), open(a.out, "w"), indent=1)
    print("PARITY " + ("PASS" if ok else "FAIL"), flush=True)


def doc_prompt(tok, path, n_tokens):
    """an instruction and the first tokens of a text file, about n_tokens with the chat template"""
    head = "Summarize the following document in a few sentences, then list its main sections.\n\n"
    ids = tok(open(path).read(), add_special_tokens=False)["input_ids"]
    body = tok.decode(ids[: max(1, n_tokens - 40)])
    return [{"role": "user", "content": head + body}]


def run_speed(a, tok):
    model = model_name(a.port)
    sampling = {k: v for k, v in dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p).items() if v is not None}
    if a.doc:
        cases = [(f"doc{a.doc_tokens}", doc_prompt(tok, a.doc, a.doc_tokens), a.thinking != "off")]
    else:
        modes = dict(on=[True], off=[False], both=[True, False])[a.thinking]
        cases = [(f"{p['name']}/think={t}", messages(p), t) for t in modes for p in PROMPTS]
    rows = []
    for rep in range(a.repeat):
        for name, msgs, thinking in cases:
            r = chat(a.port, model, msgs, a.max_tokens, thinking, sampling, a.ignore_eos)
            r.update(name=name, thinking=thinking, rep=rep, prompt_tokens=len(r["prompt_ids"] or []))
            rows.append(r)
            print(
                f"[{rep}] {name}: prompt {r['prompt_tokens']} tok, {len(r['token_ids'])} out ({r['chunks']} chunks, "
                f"finish {r['finish']}), TTFT {r['ttft_s']:.3f} s, decode {r['decode_tok_s'] or 0:.1f} tok/s, "
                f"host load {r['loadavg'][1]:.1f}",
                flush=True,
            )
    last = [r for r in rows if r["rep"] == a.repeat - 1 and r["decode_tok_s"]]
    summary = dict(
        sampling=sampling,
        requests=len(last),
        mean_decode_tok_s=sum(r["decode_tok_s"] for r in last) / max(1, len(last)),
        pooled_decode_tok_s=sum(r["decode_tokens"] for r in last) / max(1e-9, sum(r["decode_s"] for r in last)),
        mean_ttft_s=sum(r["ttft_s"] for r in last) / max(1, len(last)),
    )
    print(json.dumps(summary), flush=True)
    for r in rows:
        r.pop("prompt_ids", None)
    json.dump(dict(model=model, summary=summary, rows=rows), open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "speed", "compare"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--served")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--thinking", choices=["on", "off", "both"], default="both")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--doc")
    ap.add_argument("--doc-tokens", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    dict(parity=run_parity, speed=run_speed, compare=run_compare)[a.mode](a, tok)


if __name__ == "__main__":
    main()
