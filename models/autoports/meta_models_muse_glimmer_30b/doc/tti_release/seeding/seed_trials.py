"""Run the 32-way seed=0 repro N times and, for each, report which BATCH SLOT diverged
and whether the device seed pushed to the zero-seed slots was uniform at that step."""
import collections
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
TRACE = "/tmp/seedtrace.jsonl"


def ask(a):
    i, s = a
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Generate a list of 10 random colors."}],
        "max_tokens": 50,
        "temperature": 0.9,
        "seed": s,
    }
    r = json.load(
        urllib.request.urlopen(
            urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
            timeout=900,
        )
    )
    return i, s, (r["choices"][0]["message"]["content"] or "")


def one_trial(n):
    open(TRACE, "w").close()
    seeds = []
    for i in range(1, 17):
        seeds += [i, 0]
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        out = list(ex.map(ask, list(enumerate(seeds))))
    zero = [t for i, s, t in out if s == 0]
    nz = [t for i, s, t in out if s != 0]
    cz = len(set(zero))
    recs = [json.loads(l) for l in open(TRACE)]
    dec = [r for r in recs if "step" in r]
    tok = [r for r in recs if "tok_step" in r]
    if not dec:
        print(f"trial {n}: distinct_zero={cz} distinct_nonzero={len(set(nz))} (no trace)")
        return
    mgr = dec[0]["mgr_seed"]
    zslots = [i for i, s in enumerate(mgr) if s == 0]
    streams = {i: tuple(t["tokens"][i] for t in tok) for i in zslots}
    groups = collections.defaultdict(list)
    for i in zslots:
        groups[streams[i]].append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    div_info = ""
    if len(ordered) > 1:
        maj = streams[ordered[0][0]]
        for j, x in enumerate(maj):
            others = [g[0] for g in ordered[1:]]
            if any(streams[o][j] != x for o in others):
                d = dec[j] if j < len(dec) else None
                uniq = sorted({d["dev_seed"][i] for i in zslots}) if d else None
                div_info = (
                    f" first_div_tok_index={j} dev_seed_at_that_decode_step={uniq} pos={d['pos'][0] if d else None}"
                )
                break
    print(
        f"trial {n}: distinct_zero_text={cz} distinct_nonzero={len(set(nz))} "
        f"zero_slots={zslots} slot_groups={[sorted(g) for g in ordered]}{div_info}"
    )
    sys.stdout.flush()
    os.rename(TRACE, f"/tmp/seedtrace_trial{n}.jsonl")


if __name__ == "__main__":
    for n in range(1, int(sys.argv[1]) + 1):
        one_trial(n)
