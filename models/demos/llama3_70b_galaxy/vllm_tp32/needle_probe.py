#!/usr/bin/env python3
"""Long-prompt correctness probe: hide a 6-digit code in real English text, ask for it back (greedy).

Covers the prefill buckets the MMLU subset never reaches (2K-8K tokens) and chunked-prefill resumes.
Runs in the server container against the server on localhost:8000.
usage: needle_probe.py <out.json> [lengths...]   default lengths: 1500 2112 3000 4160 5000 6000 7000
Each length is tried at 3 needle depths (10%, 50%, 90%). Exits 0 only if every answer is right and no
answer contains a backslash run.
"""
import glob, json, os, random, re, sys, urllib.request
from transformers import AutoTokenizer

out = sys.argv[1]
lengths = [int(x) for x in sys.argv[2:]] or [1500, 2112, 3000, 4160, 5000, 6000, 7000]
tok = AutoTokenizer.from_pretrained(os.environ.get("HF_MODEL", "/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct"))
REPO = os.environ.get("TT_METAL_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 4)
)  # tt-metal root: filler text
DRY = os.environ.get("NEEDLE_DRY") == "1"  # build prompts only, no server
model = "dry" if DRY else json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["id"]

# Real English text: tt-metal markdown docs, prose lines only.
paras = []
HERE = os.path.dirname(os.path.abspath(__file__))  # its README is not filler
docs = [
    f
    for f in glob.glob(os.path.join(REPO, "**", "*.md"), recursive=True)
    if os.path.dirname(os.path.abspath(f)) != HERE
]
for f in sorted(docs)[:400]:
    for line in open(f, errors="ignore"):
        line = line.strip()
        if len(line) > 80 and not line.startswith(("|", "#", "```", "<", "!", "-", "*")):
            paras.append(line)
filler_ids = tok(" ".join(paras), add_special_tokens=False)["input_ids"]

rng = random.Random(1234)
results, ok_all = [], True
for L in lengths:
    for depth in (0.1, 0.5, 0.9):
        code = str(rng.randint(100000, 999999))
        needle = f" The secret access code is {code}. Remember it. "
        question = (
            "\n\nWhat is the secret access code mentioned in the text above? Answer with the 6-digit number only."
        )
        n_needle = len(tok(needle, add_special_tokens=False)["input_ids"])
        n_q = len(tok(question, add_special_tokens=False)["input_ids"])
        body = L - n_needle - n_q - 40  # leave room for the chat template
        start = rng.randint(0, len(filler_ids) - body - 1)
        chunk = filler_ids[start : start + body]
        cut = int(len(chunk) * depth)
        text = tok.decode(chunk[:cut]) + needle + tok.decode(chunk[cut:]) + question
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
            ids = ids["input_ids"]
        ids = [int(i) for i in ids]
        body = {"model": model, "prompt": ids, "max_tokens": 12, "temperature": 0, "top_k": 1}
        req = urllib.request.Request(
            "http://localhost:8000/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            if DRY:
                ans = code
                json.dumps(body)
            else:
                ans = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]
        except Exception as e:  # a failed request is a failed probe
            ans = f"ERROR {e}"
        garbage = bool(re.search(r"\\\\{2,}", ans))
        expect = str(int(code) + 1) if os.environ.get("NEEDLE_PLANT") == "1" else code  # planted-failure check
        ok = expect in ans and not garbage
        ok_all &= ok
        results.append({"len": len(ids), "target_len": L, "depth": depth, "code": code, "answer": ans, "ok": ok})
        print(f"len={len(ids):5d} depth={depth:.1f} ok={ok} answer={ans.strip()[:30]!r}", flush=True)

json.dump(results, open(out, "w"), indent=1)
print(f"NEEDLE {'PASS' if ok_all else 'FAIL'} {sum(r['ok'] for r in results)}/{len(results)}")
sys.exit(0 if ok_all else 1)
