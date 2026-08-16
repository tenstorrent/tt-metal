import collections
import concurrent.futures as cf
import json
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"


def ask(idx_seed):
    idx, seed = idx_seed
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Generate a list of 10 random colors."}],
        "max_tokens": 50,
        "temperature": 0.9,
        "seed": seed,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return idx, seed, (r["choices"][0]["message"]["content"] or "").strip()


def run(total, label):
    seeds = []
    for i in range(1, total // 2 + 1):
        seeds += [i, 0]
    with cf.ThreadPoolExecutor(max_workers=total) as ex:
        out = list(ex.map(ask, list(enumerate(seeds))))
    zero = [(i, t) for i, s, t in out if s == 0]
    nonzero = [(i, s, t) for i, s, t in out if s != 0]
    uz = collections.Counter(t for _, t in zero)
    print(f"--- {label}: total={total}")
    print(f"    seed=0 requests: {len(zero)}, distinct outputs: {len(uz)}")
    for text, n in uz.most_common():
        idxs = [i for i, t in zero if t == text]
        print(f"      n={n} batch_positions={idxs} :: {text[:90]!r}")
    un = set(t for _, _, t in nonzero)
    print(f"    seed!=0 requests: {len(nonzero)}, distinct: {len(un)}")
    return len(uz)


if __name__ == "__main__":
    res = {}
    for total in (2, 4, 8, 32):
        res[total] = run(total, f"batch{total}")
    print("SUMMARY distinct-seed0-outputs by batch size:", res)
