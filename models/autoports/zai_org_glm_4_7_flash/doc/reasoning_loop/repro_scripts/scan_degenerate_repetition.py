"""Scan lm-eval sample jsonl files for degenerate repetition inside otherwise
non-empty content. Finds a repeated 120-char window (stride 7) and flags a doc
when the window recurs >= 20 times, or recurs >= 12 times in a response over
30,000 characters. This threshold was tightened by hand after a looser one
(>= 8 repeats) produced two false positives: a forum-thread prompt that
legitimately repeats speaker tags, and a bullet-list answer with genuinely
distinct bullets. Manually re-verify anything near the threshold before
trusting the count; do not treat this as ground truth without a spot check."""
import json, sys

def loopiness(text, win=120, stride=7):
    if len(text) < win * 6:
        return 0, ""
    c = {}
    for i in range(0, len(text) - win, stride):
        w = text[i:i+win]
        c[w] = c.get(w, 0) + 1
    if not c:
        return 0, ""
    w = max(c, key=c.get)
    return c[w], w

def scan(path, task):
    rows = [json.loads(l) for l in open(path)]
    seen = {}
    for r in rows:
        seen.setdefault(r["doc_id"], r)
    flagged = []
    for r in seen.values():
        text = r["resps"][0][0]
        reps, window = loopiness(text)
        if reps >= 20 or (len(text) > 30000 and reps >= 12):
            flagged.append({"task": task, "doc_id": r["doc_id"], "len": len(text),
                             "reps": reps, "window": window})
    return flagged

if __name__ == "__main__":
    all_flagged = []
    for path in sys.argv[1:]:
        task = "gpqa" if "gpqa" in path else "ifeval"
        all_flagged.extend(scan(path, task))
    print(f"{len(all_flagged)} flagged across {len(sys.argv)-1} file(s)")
    for x in sorted(all_flagged, key=lambda x: -x["reps"]):
        print(f"  {x['task']:8s} doc_id={x['doc_id']:4d} len={x['len']:7d} reps={x['reps']:5d}  {x['window'][:60]!r}")
