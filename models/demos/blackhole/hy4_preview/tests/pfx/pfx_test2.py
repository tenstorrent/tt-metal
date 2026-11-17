#!/usr/bin/env python3
"""pfx_test.py <host> <mode> [cases]  hit == cold gate for the row-local prefix cache.
mode 'strict': hybrid cap file = 0 (cold reference and hit both use the direct-to-pool path): per case N (base prompt tokens), E = base + 700 extra tokens:
   cold1 = E with pfx=0, cold2 = E with pfx=0 again (determinism of the reference), warm = B (pfx=1, 1 token), hit = E with pfx=1.
   PASS = hit tokens == cold1 tokens (and == cold2 tokens) and first-token logits md5 hit == cold, hit log shows pfx_start > 0.
mode 'prod': hybrid cap 12288 as in production (cold of a short prompt = chain path): informational token comparison.
Prompts are token-id lists (exact lengths) built from natural text; greedy, 48 new tokens, thinking off irrelevant (raw completions)."""
import json
import os
import re
import sys
import time
import urllib.request

H, MODE = sys.argv[1], sys.argv[2]
CASES = (
    [int(x) for x in sys.argv[3].split(",")]
    if len(sys.argv) > 3
    else [1024, 2560, 5000, 8000, 12000, 12288, 12800, 16000, 20000]
)
D = "/mnt/tt-data/ssinghal/hy4-logs/program/pfx"
LOG = f"{D}/logs/server_{os.environ.get('TAG', 'a1')}.log"
URL = f"http://{H}:8000/v1/completions"
CJK = re.compile(r"[　-鿿]")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/mnt/tt-data/Hy4-Preview")

import glob


def corpus():
    fs = sorted(glob.glob("/mnt/tt-data/ssinghal/tests/tt-metal/tech_reports/**/*.md", recursive=True))
    t = ""
    for f in fs:
        try:
            t += open(f, errors="ignore").read() + "\n\n"
        except OSError:
            pass
        if len(t) > 400000:
            break
    return t


_cache = {}


def ids(seed, n):
    # seed 1 = base text; seed 2 = the continuation (extra 700 tokens) taken right after the longest base (position 30000)
    if "all" not in _cache:
        _cache["all"] = tok(corpus(), add_special_tokens=False)["input_ids"]
    a = _cache["all"]
    assert len(a) > 31000, len(a)
    return a[:n] if seed == 1 else a[30000 : 30000 + n]


def setf(name, v):
    open(f"{D}/{name}", "w").write(str(v))


def loglines():
    with open(LOG, errors="replace") as f:
        return f.read().splitlines()


def req(p, mt=48):
    n0 = os.path.getsize(LOG)
    b = dict(model="tencent/Hy4-preview", prompt=p, max_tokens=mt, temperature=0, stream=False)
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
    dt = time.time() - t0
    txt = d["choices"][0]["text"]
    info = {}
    for _ in range(120):
        with open(LOG, "rb") as f:
            f.seek(n0)
            pl = [l for l in f.read().decode(errors="replace").splitlines() if "HY4 prefill call" in l]
        if pl:
            break
        time.sleep(0.5)
    if pl:
        m = re.search(
            r"lg_md5=\{\d+: '(\w+)'\} pfx_start=(\d+) hits=\{(\d+): (\d+)\}.*?mode=(\w+).*?rows=\[(\d+)\].*?prefill=([\d.]+)s",
            pl[-1],
        )
        if m:
            info = dict(
                md5=m.group(1),
                start=int(m.group(2)),
                hit=int(m.group(4)),
                mode=m.group(5),
                row=int(m.group(6)),
                pf=float(m.group(7)),
            )
    return txt, dt, info, d["usage"]["prompt_tokens"]


ok_all = True
rows = []
for N in CASES:
    B = ids(1, N)
    E = B + ids(2, 700)
    if MODE == "strict":
        setf("hybrid_cap", 0)
    else:
        setf("hybrid_cap", 12288)
    setf("pfx_file", 0)
    c1, t_c1, i1, pt = req(E)
    c2, t_c2, i2, _ = req(E)
    c3, t_c3, i3, _ = req(E)
    setf("pfx_file", 1)
    w, t_w, iw, _ = req(B, 1)
    h, t_h, ih, _ = req(E)
    h2, t_h2, ih2, _ = req(E)
    same = h == c1 == c2 == c3
    same2 = h2 == h
    same_md5 = ih.get("md5") is not None and ih.get("md5") == i1.get("md5") == i2.get("md5") == i3.get(
        "md5"
    ) == ih2.get("md5")
    det = c1 == c2 == c3
    hitok = ih.get("start", 0) > 0
    cjk = len(CJK.findall(h)) + len(CJK.findall(c1))
    ok = same_md5 and hitok and cjk == 0 and same
    ok_all &= ok
    rows.append((N, pt))
    json.dump(dict(N=N, cold=[c1, c2, c3], hit=[h, h2]), open(f"{D}/streams_{MODE}_{N}.json", "w"))
    fd = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if a == b else min(len(a), len(b)))
    print(
        f"   first divergence (char idx): cold1/cold2={fd(c1,c2)} cold1/cold3={fd(c1,c3)} cold1/hit={fd(c1,h)} cold2/hit={fd(c2,h)} hit/hit2={fd(h,h2)}",
        flush=True,
    )
    print(
        f"{MODE} N={N} E={pt}: hit==all3cold={same} cold3_det={det} hit2==hit={same2} md5_all_equal={same_md5} rows cold={i1.get('row')}/{i2.get('row')}/{i3.get('row')} warm={iw.get('row')} hit={ih.get('row')}/{ih2.get('row')} "
        f"hit_start={ih.get('start')}/{ih2.get('start')} cold_prefill={i1.get('pf')}s hit_prefill={ih.get('pf')}s hit_wall={t_h:.1f}s cold_wall={t_c1:.1f}s cjk={cjk} {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not same:
        print("  cold:", repr(c1[:120]), "\n  hit :", repr(h[:120]), flush=True)
print("PFX_TEST", MODE, "PASS" if ok_all else "FAIL", flush=True)
setf("hybrid_cap", 12288)
setf("pfx_file", 1)
sys.exit(0 if ok_all else 1)
