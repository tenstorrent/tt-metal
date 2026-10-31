#!/usr/bin/env python3
"""A/B of the structured modes in ONE server (plugin VLLM_TT_STRUCTURED_MODE_FILE): usage: mode_ab.py <modefile> <outdir> <modes: m0,m1,m2> [structured_workers=4] [plain_users=8] [osl=256] [reps=2]
Per mode: write the mode word, start S structured workers (json_schema / required tool call, streaming, greedy, looping), then run the plain decode client (dec.py: C plain users, real-text
prompts, per-user tok/s and TTFT) next to them; also 'none' = the plain users alone. Every structured output is validated. Prints one summary line per mode."""
import json
import subprocess
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import matrix as MX  # noqa: E402

mf, out, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
S = int(sys.argv[4]) if len(sys.argv) > 4 else 4
C = int(sys.argv[5]) if len(sys.argv) > 5 else 8
OSL = int(sys.argv[6]) if len(sys.argv) > 6 else 256
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 2
PY = "/mnt/tt-data/ssinghal/wt/v031/venv/bin/python"
DEC = "/mnt/tt-data/ssinghal/dsv4-logs/stg_ab/dec.py"
BIG = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "bio": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 6},
    },
    "required": ["name", "age", "bio", "tags"],
    "additionalProperties": False,
}
RF = {"type": "json_schema", "json_schema": {"name": "p", "schema": BIG, "strict": True}}


def ctl_worker(k, stop, rec, max_tokens):
    """control: the same prompts as the structured workers, NO grammar / tools (a plain short request with the same turnover)"""
    while not stop.is_set():
        msgs = (
            [{"role": "user", "content": "Write a long profile of a fictional person, with a detailed bio."}]
            if k % 2 == 0
            else MX.WQ
        )
        t0 = time.time()
        r = MX.post(
            "127.0.0.1",
            8000,
            dict(
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
                stream_options={"include_usage": True},
            ),
            True,
        )
        dt = time.time() - t0
        if r.get("http") != 200 or "sse_error" in r:
            rec.append((dt, 0, "error " + json.dumps(r)[:120]))
            continue
        rec.append((dt, (r.get("usage") or {}).get("completion_tokens") or max_tokens, "ok", r.get("ttft")))


def worker(k, stop, rec):
    while not stop.is_set():
        if k % 2 == 0:
            body = dict(
                messages=[
                    {"role": "user", "content": "Write a long profile of a fictional person, with a detailed bio."}
                ],
                response_format=RF,
                max_tokens=300,
            )
            check = MX._json_check(BIG)
        else:
            body = dict(messages=MX.WQ, tools=MX.tools(), tool_choice="required", max_tokens=300)
            check = MX._tool_check()
        t0 = time.time()
        r = MX.post("127.0.0.1", 8000, dict(body, temperature=0.0, stream_options={"include_usage": True}), True)
        dt = time.time() - t0
        if r.get("http") != 200 or "sse_error" in r:
            rec.append((dt, 0, "error " + json.dumps(r)[:120]))
            continue
        why = check(r["msg"])
        n = (r.get("usage") or {}).get("completion_tokens") or (len((r["msg"].get("content") or "")) // 4 or 1)
        rec.append((dt, n, why or "ok", r.get("ttft")))


for mode in modes:
    stop, rec = threading.Event(), []
    if mode.startswith("ctl"):
        open(mf, "w").write("m1\n")
        time.sleep(0.5)
        mt = 250 if mode == "ctl_dur" else None
        ths = [
            threading.Thread(target=ctl_worker, args=(k, stop, rec, mt if mt else (100 if k % 2 == 0 else 60)))
            for k in range(S)
        ]
    else:
        if mode != "none":
            open(mf, "w").write(mode + "\n")
            time.sleep(0.5)
        ths = [threading.Thread(target=worker, args=(k, stop, rec)) for k in range(S if mode != "none" else 0)]
    t_begin = time.strftime("%H:%M:%S", time.gmtime())
    [t.start() for t in ths]
    time.sleep(6 if ths else 0)
    p = subprocess.run(
        [PY, DEC, "127.0.0.1", "512", str(OSL), str(C), str(REPS), "ab_" + mode, out],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    stop.set()
    [t.join() for t in ths]
    summ = [l for l in p.stdout.splitlines() if l.startswith("SUMMARY")]
    ok = [r for r in rec if r[2] == "ok"]
    bad = [r for r in rec if r[2] != "ok"]
    lat = [r[0] for r in ok]
    ttft = [r[3] for r in ok if len(r) > 3 and r[3]]
    t_end = time.strftime("%H:%M:%S", time.gmtime())
    print(
        "[%s-%s UTC] mode %-8s plain: %s | structured: %d ok %d bad, mean latency %.1fs, mean tok/s %.1f, first-byte %.2fs %s"
        % (
            t_begin,
            t_end,
            mode,
            summ[0][8:] if summ else p.stdout[-200:],
            len(ok),
            len(bad),
            sum(lat) / max(1, len(lat)),
            sum(r[1] / r[0] for r in ok) / max(1, len(ok)),
            sum(ttft) / max(1, len(ttft)),
            [r[2] for r in bad][:2],
        ),
        flush=True,
    )
