"""Multi-benchmark MCQ eval over the OpenAI-compatible Molmo2 server (greedy). Usage: run_eval.py BENCH [--limit N] [--conc 4]
BENCH in: tempcompass, longvideobench, videomme, nextqa. Videos are read by the server via file:// (host server)."""
import argparse
import collections
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

D = "/home/ubuntu/ssinghal/molmo2/evals/data"
L = "ABCDEFGH"
SUFFIX = "Please respond with only the letter of the correct answer."


def tempcompass():
    d = pd.read_parquet(glob.glob(f"{D}/tempcompass/multi-choice/*.parquet")[0])
    vd = glob.glob(f"{D}/tempcompass/tempcompass_videos_x/videos")[0]
    for i, r in d.iterrows():
        yield dict(id=str(i), video=f"{vd}/{r.video_id}.mp4", q=r.question, gt=r.answer.strip()[0], cat=r.dim)


def longvideobench():
    d = json.load(open(f"{D}/longvideobench/lvb_val.json"))
    for r in d:
        opts = "\n".join(f"{L[k]}. {c}" for k, c in enumerate(r["candidates"]))
        yield dict(
            id=r["id"],
            video=f"{D}/longvideobench/videos/{r['video_path']}",
            q=f"{r['question']}\n{opts}",
            gt=L[r["correct_choice"]],
            cat=r["duration_group"],
        )


def videomme():
    f = glob.glob(f"{D}/videomme/videomme/*.parquet")[0]
    d = pd.read_parquet(f)
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{D}/videomme/**/*.mp4", recursive=True)}
    for _, r in d.iterrows():
        opts = "\n".join(r.options) if not isinstance(r.options, str) else r.options
        yield dict(id=r.question_id, video=vids.get(r.videoID), q=f"{r.question}\n{opts}", gt=r.answer, cat=r.duration)


def nextqa():
    d = pd.read_parquet(glob.glob(f"{D}/nextqa/MC/*.parquet")[0])
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{D}/nextqa/**/*.mp4", recursive=True)}
    for i, r in d.iterrows():
        opts = "\n".join(f"{L[k]}. {r[f'a{k}']}" for k in range(5))
        yield dict(id=str(i), video=vids.get(str(r.video)), q=f"{r.question}\n{opts}", gt=L[int(r.answer)], cat=r.type)


def lvbench():
    d = pd.read_parquet(glob.glob(f"{D}/lvbench/data/*.parquet")[0])
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{D}/lvbench/**/*.mp4", recursive=True)}
    for _, r in d.iterrows():
        yield dict(id=f"{r.key}_{r.uid}", video=vids.get(r.key), q=r.question, gt=r.answer, cat=r.type)


def motionbench():
    root = f"{D}/motionbench/MotionBench"
    vids = {os.path.basename(p): p for p in glob.glob(f"{root}/**/*.mp4", recursive=True)}
    for l in open(f"{root}/video_info.meta.jsonl"):
        j = json.loads(l)
        if "qa" not in j:
            continue
        for qa in j["qa"]:
            if qa["answer"] not in list("ABCD"):
                continue  # test split has hidden answers ("NA")
            yield dict(
                id=qa["uid"], video=vids.get(j["video_path"]), q=qa["question"], gt=qa["answer"], cat=j["question_type"]
            )


def egoschema():  # public 500-question subset (the full 5031 have hidden answers)
    d = pd.read_parquet(glob.glob(f"{D}/egoschema/Subset/*.parquet")[0])
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{D}/egoschema/**/*.mp4", recursive=True)}
    for _, r in d.iterrows():
        yield dict(
            id=r.question_idx,
            video=vids.get(r.video_idx),
            q=f"{r.question}\n" + "\n".join(r.option),
            gt=L[int(r.answer)],
            cat="subset",
        )


def perceptiontest():  # val split, 3-option MCQ
    d = pd.read_parquet(glob.glob(f"{D}/perceptiontest/mc_question_val/*.parquet")[0])
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{D}/perceptiontest/**/*.mp4", recursive=True)}
    for _, r in d.iterrows():
        opts = "\n".join(f"{L[k]}. {o}" for k, o in enumerate(r.options))
        yield dict(
            id=f"{r.video_name}_{r.question_id}",
            video=vids.get(r.video_name),
            q=f"{r.question}\n{opts}",
            gt=L[int(r.answer_id)],
            cat=r.area,
        )


BENCH = dict(
    egoschema=egoschema,
    perceptiontest=perceptiontest,
    lvbench=lvbench,
    motionbench=motionbench,
    tempcompass=tempcompass,
    longvideobench=longvideobench,
    videomme=videomme,
    nextqa=nextqa,
)
ap = argparse.ArgumentParser()
ap.add_argument("bench")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth question (deterministic subsample)")
ap.add_argument("--max-tokens", type=int, default=16)
a = ap.parse_args()
items = list(BENCH[a.bench]())
# videos whose decode segfaults the server (found by screen_videos.py); excluded and reported as skipped
_bp = f"/home/ubuntu/ssinghal/molmo2/evals/{a.bench}_bad_videos.json"
_bad = {os.path.abspath(b[0]) for b in json.load(open(_bp))} if os.path.exists(_bp) else set()
for it in items:
    if it["video"] and os.path.abspath(it["video"]) in _bad:
        it["video"] = None
items = items[:: a.stride]
if a.limit:
    items = items[: a.limit]
out_path = f"/home/ubuntu/ssinghal/molmo2/evals/results_{a.bench}.jsonl"
done = {}
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l)
        if not r.get("err") and not r.get("skipped"):
            done[r["id"]] = r


def run(it):
    if it["id"] in done:
        return done[it["id"]]
    if not it["video"] or not os.path.isfile(it["video"]):
        return dict(id=it["id"], skipped=True, cat=it["cat"])
    body = {
        "model": "allenai/Molmo2-8B",
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{it['q']}\n{SUFFIX}"},
                    {"type": "video_url", "video_url": {"url": "file://" + it["video"]}},
                ],
            }
        ],
    }
    t0 = time.time()
    try:
        r = requests.post(a.url, json=body, timeout=1800)
        out = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"HTTP{r.status_code}"
        err = r.status_code != 200
    except Exception as e:
        out, err = repr(e), True
    m = re.match(r"\s*\(?([A-H])\b", out or "")
    return dict(
        id=it["id"], gt=it["gt"], out=out, pred=m.group(1) if m else None, err=err, cat=it["cat"], sec=time.time() - t0
    )


res = []
with ThreadPoolExecutor(a.conc) as ex, open(out_path, "w") as fo:
    for n, r in enumerate(ex.map(run, items)):
        res.append(r)
        fo.write(json.dumps(r) + "\n")
        fo.flush()
        if n % 50 == 0:
            print(n, len(items), flush=True)
by = collections.defaultdict(lambda: [0, 0])
tot = [0, 0, 0, 0]
for r in res:
    if r.get("skipped"):
        tot[2] += 1
        continue
    if r["err"]:
        tot[3] += 1
    for k in (str(r["cat"]), "ALL"):
        by[k][1] += 1
        by[k][0] += r["pred"] == r["gt"]
for k, (c, n) in sorted(by.items()):
    print(f"{a.bench} {k:24s} {c}/{n} = {100*c/n:.2f}%")
print(f"skipped={tot[2]} errors={tot[3]}")
