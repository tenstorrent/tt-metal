import glob
import json
import os
import sys

base = sys.argv[1]
files = glob.glob(os.path.join(base, "**", "samples_*.jsonl"), recursive=True)
if not files:
    print("no samples file found under", base)
    raise SystemExit

path = sorted(files)[-1]
print("samples file:", os.path.basename(path))

rows = []
raw = open(path).read()
dec = json.JSONDecoder()
i, n = 0, len(raw)
while i < n:
    while i < n and raw[i] in " \r\n\t":
        i += 1
    if i >= n:
        break
    try:
        obj, end = dec.raw_decode(raw, i)
    except ValueError:
        break
    rows.append(obj)
    i = end

print("documents:", len(rows))
print()
print("  %-4s %-6s %-10s %-9s %-8s %-8s %s" % ("doc", "score", "resp_words", "has_think", "boxed", "term", "tail"))

capped = 0
think_present = 0
for k, r in enumerate(rows):
    v = r.get("resps") or r.get("filtered_resps") or []
    while isinstance(v, list) and v:
        v = v[0]
    text = v if isinstance(v, str) else ""
    score = r.get("exact_match")
    words = len(text.split())
    has_think = "</think>" in text
    if has_think:
        think_present += 1
    boxed = "boxed" in text.lower()
    term = text.strip()[-1:] in ".!?\"')}]" if text.strip() else False
    # a response at/near the cap with no terminal punctuation smells like a cap hit
    if words > 12000:
        capped += 1
    print("  %-4d %-6s %-10d %-9s %-8s %-8s %r" % (
        k, score, words, has_think, boxed, term, text[-60:].replace("\n", " ")))

print()
print("  documents whose response contains </think>: %d/%d" % (think_present, len(rows)))
print("  documents with >12000 words (cap-suspect):  %d/%d" % (capped, len(rows)))
scores = [r.get("exact_match") for r in rows if r.get("exact_match") is not None]
if scores:
    print("  scores: %s  -> mean %.2f" % (scores, sum(scores) / len(scores)))
