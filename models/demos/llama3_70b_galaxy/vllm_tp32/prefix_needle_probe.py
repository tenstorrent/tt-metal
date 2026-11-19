#!/usr/bin/env python3
"""Prefix-hit correctness probe: pairs of prompts that share a 2K (or 4K) token prefix, greedy answers.

needle_probe.py sends each prompt once, so it never runs a prefill over cached KV (the sp1 path). Here each
case sends A = prefix + unique_A, then B = prefix + unique_B, one at a time, so B is a vLLM prefix-cache hit
on the prefix A just wrote. Each case is run with the needle in three places:
  prefix   the same code sits in the shared prefix; B can only answer from the cached KV that A wrote
  uniq10   A and B carry different codes at 10% of their own unique part
  uniq90   same at 90%; a B that answers A's code means KV from A leaked into B's new-token positions
Every case gets a fresh prefix (different filler offset), so A is always a cold miss.

Cases (prefix tokens, total tokens) and the shape B should take with LLAMA_GALAXY_PREFILL_PAD=1024:
  2048+1500 -> new 2048 over 2048 cached (2048_1_sp1 @2048, used by v1 too: control)
  2048+2600 -> new 3072 (3072_1_sp1 @2048): v1 recomputes at 5120, v2 uses the cache
  2048+3600 -> new 4096 (4096_1_sp1 @2048): v1 recomputes at 6144, v2 uses the cache
  2048+4500 -> new 5120 (5120_1_sp1 @2048): v1 recomputes at 8192, v2 uses the cache
  4096+1500 -> new 2048 over 4096 cached (2048_1_sp1 @4096, start >= 4096, v1 too: control)
Confirm the shape from the server log ('Prefilling User ... prompt_lens: N, prefill_seq_len: P,
num_cached_tokens: C'); the probe prints the expected line for each B. cached_tokens from
usage.prompt_tokens_details is recorded when the server runs with enable_prompt_tokens_details.

Runs in the server container against the server on localhost:8000, like needle_probe.py.
usage: candidate-05-prefix-needle.py <out.json>
env: NEEDLE_DRY=1 builds the prompts only (no server); NEEDLE_PLANT=1 expects code+1 (proves the gate can fail).
Exits 0 only if every A and B answer is right and no answer contains a backslash run.
"""
import glob, json, os, random, re, sys, urllib.request
from transformers import AutoTokenizer

out = sys.argv[1]
CASES = [(2048, 3548), (2048, 4648), (2048, 5648), (2048, 6548), (4096, 5596)]
POSITIONS = ("prefix", "uniq10", "uniq90")
BLOCK = 64  # vLLM KV block; prefixes are multiples of it so the whole prefix is cacheable

tok = AutoTokenizer.from_pretrained(os.environ.get("HF_MODEL", "/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct"))
REPO = os.environ.get("TT_METAL_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 4)
)  # tt-metal root: filler text
DRY = os.environ.get("NEEDLE_DRY") == "1"
PLANT = os.environ.get("NEEDLE_PLANT") == "1"
model = "dry" if DRY else json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["id"]

paras = []
for f in sorted(glob.glob(os.path.join(REPO, "**", "*.md"), recursive=True))[:400]:
    for line in open(f, errors="ignore"):
        line = line.strip()
        if len(line) > 80 and not line.startswith(("|", "#", "```", "<", "!", "-", "*")):
            paras.append(line)
filler = tok(" ".join(paras), add_special_tokens=False)["input_ids"]

# Chat template split around the user content, so the token ids of the prefix are identical in A and B.
SPLIT = "XXSPLITXX"
tmpl = tok.apply_chat_template([{"role": "user", "content": SPLIT}], add_generation_prompt=True, tokenize=False)
head_txt, tail_txt = tmpl.split(SPLIT)
head = tok(head_txt, add_special_tokens=False)["input_ids"]
tail = tok(tail_txt, add_special_tokens=False)["input_ids"]
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
question = enc("\n\nWhat is the secret access code mentioned in the text above? Answer with the 6-digit number only.")
needle = lambda code: enc(f" The secret access code is {code}. Remember it. ")

rng = random.Random(5005)


def take(n):
    start = rng.randint(0, len(filler) - n - 1)
    return filler[start : start + n]


def with_needle(body, code, depth):
    cut = int(len(body) * depth)
    nd = needle(code)
    return body[:cut] + nd + body[cut : len(body) - len(nd)]  # same length as body


def send(ids):
    body = {"model": model, "prompt": ids, "max_tokens": 12, "temperature": 0, "top_k": 1}
    if DRY:
        json.dumps(body)
        return None, None
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:  # a failed request is a failed probe
        return f"ERROR {e}", None
    details = (resp.get("usage") or {}).get("prompt_tokens_details") or {}
    return resp["choices"][0]["text"], details.get("cached_tokens")


def pad_len(n):  # LLAMA_GALAXY_PREFILL_PAD=1024 buckets, for the expected log line only
    if n <= 128:
        return 128
    if n <= 1024:
        return 1024
    p = 2 ** (n - 1).bit_length()
    return min([p] + [e for e in (3072, 5120, 6144) if n <= e < p])


results, ok_all = [], True
for prefix_len, total in CASES:
    assert prefix_len % BLOCK == 0
    for pos in POSITIONS:
        codes = {r: str(rng.randint(100000, 999999)) for r in ("A", "B")}
        if pos == "prefix":
            codes["B"] = codes["A"]
        prefix_body = take(prefix_len - len(head))
        if pos == "prefix":
            prefix_body = with_needle(prefix_body, codes["A"], 0.5)
        prefix = head + prefix_body
        assert len(prefix) == prefix_len
        for r in ("A", "B"):
            n_unique = total - prefix_len - len(question) - len(tail)
            unique = take(n_unique)
            if pos != "prefix":
                unique = with_needle(unique, codes[r], 0.1 if pos == "uniq10" else 0.9)
            ids = [int(i) for i in prefix + unique + question + tail]
            assert ids[:prefix_len] == prefix and len(ids) == total
            ans, cached = send(ids)
            if DRY:
                ans = codes[r]
            expect = str(int(codes[r]) + 1) if PLANT else codes[r]
            garbage = bool(re.search(r"\\{2,}", ans))
            other = codes["A"] if (r == "B" and pos != "prefix") else None
            leaked = other is not None and other in ans
            ok = expect in ans and not garbage and not leaked
            ok_all &= ok
            new = total - prefix_len
            shape = f"prompt_lens: {total}, prefill_seq_len: {pad_len(new)}, num_cached_tokens: {prefix_len}"
            if r == "A":
                shape = f"prompt_lens: {total}, prefill_seq_len: {pad_len(total)}, num_cached_tokens: 0"
            row = dict(
                prefix=prefix_len,
                len=total,
                pos=pos,
                req=r,
                code=codes[r],
                answer=ans,
                ok=ok,
                garbage=garbage,
                leaked_A_code=leaked,
                cached_tokens=cached,
                expect_log=shape,
            )
            results.append(row)
            print(
                f"prefix={prefix_len} len={total} pos={pos:6s} {r} ok={ok} cached={cached} "
                f"answer={ans.strip()[:30]!r} expect_log='{shape}'",
                flush=True,
            )

json.dump(results, open(out, "w"), indent=1)
b_rows = [r for r in results if r["req"] == "B"]
print(
    f"PREFIX-NEEDLE {'PASS' if ok_all else 'FAIL'} {sum(r['ok'] for r in results)}/{len(results)} "
    f"(B, cache-hit side: {sum(r['ok'] for r in b_rows)}/{len(b_rows)})"
)
sys.exit(0 if ok_all else 1)
