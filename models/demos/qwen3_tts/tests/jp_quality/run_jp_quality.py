"""Qwen3-TTS Japanese regression check: generate -> transcribe (Whisper) -> report, then compare two runs.

Produces reports for a human to read; it is not a pass/fail gate (generation is
sampled, so single-sample differences are noise). See README.md.

    # run the test on the commit before the change and on the commit after it
    python run_jp_quality.py run --tree /path/to/before-tree --out /tmp/jpq_before
    python run_jp_quality.py run --tree /path/to/after-tree  --out /tmp/jpq_after
    python run_jp_quality.py compare /tmp/jpq_before /tmp/jpq_after     # -> /tmp/jpq_after/compare.md

    # steps of `run` individually (each is resumable; finished work is skipped)
    python run_jp_quality.py gen    --tree ... --out DIR
    python run_jp_quality.py score  --out DIR
    python run_jp_quality.py report --out DIR
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENTENCES = HERE / "jp_sentences.json"
WHISPER_ID = "openai/whisper-large-v3-turbo"
# Whisper's stock hallucinations on silence / very short audio.
HALLUCINATIONS = ["ご視聴ありがとうございました", "チャンネル登録", "お疲れ様でした"]


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_sentences(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sentences_hash(data):
    """Hash only what affects generation, so editing focus/listen/accept keeps runs resumable."""
    key = [data["seeds"]] + [[x["id"], x["text"], x.get("max_new_tokens")] for x in data["sentences"]]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:12]


def git(tree, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(tree), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tree_meta(tree):
    tree = Path(tree).resolve()
    return dict(
        path=str(tree),
        commit=git(tree, "rev-parse", "HEAD"),
        subject=git(tree, "log", "-1", "--format=%s"),
    )


# Settings that must match for two runs to be comparable (and for a run to be resumed).
RUN_SETTINGS = ("hf_id", "sentences_sha1")


def run_one(sid, seed, a):
    """Generate one utterance in its own process.

    Qwen3-TTS hangs in Talker prefill after several generations in one process, so each
    generation gets a fresh process, killed on timeout so a hang cannot stall the run.
    """
    out = Path(a.out)
    tag = f"{sid}_seed{seed}"
    logf = out / "logs" / f"{tag}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(HERE / "gen_one.py"),
        "--tree",
        str(Path(a.tree).resolve()),
        "--sentences",
        str(Path(a.sentences).resolve()),
        "--id",
        sid,
        "--seed",
        str(seed),
        "--out",
        str(out.resolve() / "wav"),
        "--hf-id",
        a.hf_id,
    ]
    t0 = time.time()
    with open(logf, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
    ok = rc == 0 and (out / "wav" / f"res_{tag}.json").exists()
    log(f"{'OK  ' if ok else 'FAIL'} {tag} rc={rc} {time.time() - t0:.0f}s")
    if not ok:
        log(f"see {logf}")
    return ok


def cmd_gen(a):
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    data = load_sentences(a.sentences)
    seeds = a.seeds if a.seeds is not None else data["seeds"]
    ids = [s["id"] for s in data["sentences"] if not a.only or s["id"] in a.only]

    meta_f = out / "meta.json"
    meta = dict(
        tree=tree_meta(a.tree),
        hf_id=a.hf_id,
        sentences_sha1=sentences_hash(data),
        started=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if meta_f.exists():
        old = json.loads(meta_f.read_text())
        for k in RUN_SETTINGS:
            if old.get(k) != meta[k]:
                sys.exit(f"{out} was generated with {k}={old.get(k)!r}, now {meta[k]!r}; use a new --out")
        if old["tree"]["commit"] != meta["tree"]["commit"]:
            sys.exit(f"{out} was generated from {old['tree']['commit']}, tree is now {meta['tree']['commit']}")
    else:
        meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    jobs = [(i, s) for i in ids for s in seeds if not (out / "wav" / f"res_{i}_seed{s}.json").exists()]
    log(f"{len(jobs)} generations to run ({len(ids) * len(seeds) - len(jobs)} already done)")
    for n, (sid, seed) in enumerate(jobs, 1):
        if not run_one(sid, seed, a):
            log(f"stopped after {n - 1}/{len(jobs)}; re-run the same command to resume")
            return False
    log("generation complete")
    return True


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    # Drop punctuation (P*, incl. 、。・？) and whitespace (Z*). ー is Lm, so it is kept.
    return "".join(c for c in s if unicodedata.category(c)[0] not in "PZ")


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return d[len(h)] / len(r)


class Whisper:
    def __init__(self, model_id):
        import torch
        import transformers
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        transformers.logging.set_verbosity_error()
        self.torch = torch
        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).eval()

    def __call__(self, path):
        import soundfile as sf
        import soxr

        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            x = soxr.resample(x, sr, 16000)
        feats = self.proc(x, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.no_grad():
            ids = self.model.generate(
                feats, language="ja", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=200
            )
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def score_sample(res, sent, transcript):
    accept = sent.get("accept") or [sent["text"]]
    cers = [cer(ref, transcript) for ref in accept]
    best = min(range(len(cers)), key=cers.__getitem__)
    return dict(
        res,
        transcript=transcript,
        cer=round(cers[best], 4),
        matched_ref=best,
        hallucination=any(h in transcript for h in HALLUCINATIONS),
    )


def cmd_score(a):
    out = Path(a.out)
    data = load_sentences(a.sentences)
    by_id = {s["id"]: s for s in data["sentences"]}
    scores_f = out / "scores.json"
    scores = json.loads(scores_f.read_text()) if scores_f.exists() else {"whisper": WHISPER_ID, "samples": {}}
    res_files = sorted(glob.glob(str(out / "wav" / "res_*.json")))
    todo = []
    for f in res_files:
        res = json.loads(Path(f).read_text(encoding="utf-8"))
        if res["id"] not in by_id:
            log(f"skipping {res['tag']}: sentence {res['id']!r} is no longer in {a.sentences}")
            scores["samples"].pop(res["tag"], None)
            continue
        wav = out / "wav" / f"{res['tag']}.wav"
        prev = scores["samples"].get(res["tag"])
        if prev and prev.get("wav_mtime") == wav.stat().st_mtime:
            # Transcripts are reused; CER is recomputed so edits to `accept` apply without re-transcribing.
            scores["samples"][res["tag"]] = dict(
                score_sample(res, by_id[res["id"]], prev["transcript"]), wav_mtime=prev["wav_mtime"]
            )
        else:
            todo.append((res, wav))
    log(f"transcribing {len(todo)} of {len(res_files)} samples with {WHISPER_ID} (CPU)")
    if todo or "reference" not in scores:
        w = Whisper(WHISPER_ID)
        if "reference" not in scores:
            ref_wav = HERE / data["reference"]["wav"]
            ref_text = (HERE / data["reference"]["text_file"]).read_text(encoding="utf-8").strip()
            tr = w(ref_wav)
            scores["reference"] = dict(text=ref_text, transcript=tr, cer=round(cer(ref_text, tr), 4))
            log(f"reference sanity: CER {scores['reference']['cer']:.3f}  '{tr}'")
        for res, wav in todo:
            tr = w(wav)
            s = dict(score_sample(res, by_id[res["id"]], tr), wav_mtime=wav.stat().st_mtime)
            scores["samples"][res["tag"]] = s
            log(f"{res['tag']:28s} CER {s['cer']:.3f}  '{tr}'")
            scores_f.write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    scores_f.write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    return True


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def header(meta, scores):
    return [
        f"- Model: `{meta['hf_id']}`",
        f"- ASR: `{scores['whisper']}` (CPU, greedy). Reference-voice transcript CER {scores['reference']['cer']:.3f}"
        f" (`{scores['reference']['transcript']}`)",
    ]


def tree_line(label, meta):
    t = meta["tree"]
    return f"- {label}: `{(t['commit'] or '?')[:11]}` {t['subject'] or ''} (`{t['path']}`)"


def flagged(rs):
    return any(r["hit_max"] or r["hallucination"] or r["cer"] >= 0.3 for r in rs)


def notes(r):
    return (" ✂truncated" if r["hit_max"] else "") + (" 👻hallucination" if r["hallucination"] else "")


FLAG_DOC = (
    "⚠ means a sample was truncated (hit max_new_tokens, ✂), contains a hallucination phrase (👻), or has CER ≥ 0.3"
)
CER_DOC = (
    "CER is the character error rate against the closest `accept` spelling (punctuation and whitespace removed). "
    "For number sentences Whisper's spelling choices can make CER misleading, so read the transcripts themselves."
)


def cmd_report(a):
    out = Path(a.out)
    data = load_sentences(a.sentences)
    meta = json.loads((out / "meta.json").read_text())
    scores = json.loads((out / "scores.json").read_text())
    samples = scores["samples"]

    L = ["# Qwen3-TTS Japanese quality report", "", tree_line("Tree", meta), *header(meta, scores), "", CER_DOC, ""]
    L += [
        "## Summary",
        "",
        "| Sentence | Focus | n | mean CER | max CER | truncated | hallucinated | sec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    flags = []
    for s in data["sentences"]:
        rs = [v for v in samples.values() if v["id"] == s["id"]]
        if not rs:
            continue
        c = [r["cer"] for r in rs]
        mark = "⚠ " if flagged(rs) else ""
        if mark:
            flags.append(s["id"])
        L.append(
            f"| {mark}{s['id']} | {s['focus']} | {len(rs)} | {mean(c):.3f} | {max(c):.3f} |"
            f" {sum(r['hit_max'] for r in rs) or ''} | {sum(r['hallucination'] for r in rs) or ''} |"
            f" {mean([r['audio_sec'] for r in rs]):.1f} |"
        )
    L += [
        "",
        f"Overall mean CER {mean([v['cer'] for v in samples.values()]):.3f} ({len(samples)} samples)",
        "",
        FLAG_DOC + ".",
        "",
    ]

    L += ["## Listening checklist", "", "Aspects CER cannot judge. Listen to the seed0 wav.", ""]
    ran = {v["id"] for v in samples.values()}
    for s in data["sentences"]:
        if s.get("listen") and s["id"] in ran:
            L.append(f"- [ ] **{s['id']}** — {s['listen']} (`wav/{s['id']}_seed0.wav`)")
    L += ["", "## Transcripts", ""]
    for s in data["sentences"]:
        rs = sorted((v for v in samples.values() if v["id"] == s["id"]), key=lambda r: r["seed"])
        if not rs:
            continue
        L += [f"### {'⚠ ' if s['id'] in flags else ''}{s['id']} — {s['focus']}", "", f"Input: {s['text']}", ""]
        L += ["| seed | CER | sec | frames | transcript |", "|---|---|---|---|---|"]
        for r in rs:
            L.append(
                f"| {r['seed']} | {r['cer']:.3f} | {r['audio_sec']:.1f} | {r['frames']}{notes(r)} | {r['transcript']} |"
            )
        L.append("")

    rep = out / "report.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {rep}")
    if flags:
        log(f"flagged: {', '.join(flags)}")
    return True


def load_run(d):
    d = Path(d)
    for f in ("meta.json", "scores.json"):
        if not (d / f).exists():
            sys.exit(f"{d / f} not found; run `run` (or `gen` + `score`) on {d} first")
    return json.loads((d / "meta.json").read_text()), json.loads((d / "scores.json").read_text())


def cmd_compare(a):
    data = load_sentences(a.sentences)
    mb, sb = load_run(a.before)
    ma, sa = load_run(a.after)
    diff = [f"{k}: before={mb.get(k)!r} after={ma.get(k)!r}" for k in RUN_SETTINGS if mb.get(k) != ma.get(k)]
    if diff:
        sys.exit(
            "runs were made with different settings, so differences would not be caused by the change:\n  "
            + "\n  ".join(diff)
        )
    B, A = sb["samples"], sa["samples"]
    tags = sorted(set(B) & set(A))
    if not tags:
        sys.exit("the two runs have no sentence/seed in common")
    same = {t for t in tags if B[t]["codes_sha1"] == A[t]["codes_sha1"]}

    L = [
        "# Qwen3-TTS Japanese regression comparison",
        "",
        tree_line("Before", mb),
        tree_line("After", ma),
        *header(ma, sa),
        "",
    ]
    L += [
        f"**Generated codes bit-identical: {len(same)}/{len(tags)}.** "
        "The same commit and seed always reproduce the same codes, so an identical sample is unaffected by the change. "
        + (
            "All samples are identical: the change does not alter Japanese generation "
            "(this check decodes codes itself, so changes limited to the tree's decoding/post-processing are not covered)."
            if len(same) == len(tags)
            else "Only the samples that differ are listed below; judge them by their transcripts and by ear."
        ),
        "",
        CER_DOC
        + " Generation is sampled, so **a single-sample difference is noise**: look at the means and at ⚠ rows.",
        "",
        "## Summary",
        "",
        "| Sentence | Focus | n | identical | before mean CER | after mean CER | Δ | after worse | after better |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    flags = []
    for s in data["sentences"]:
        ts = [t for t in tags if A[t]["id"] == s["id"]]
        if not ts:
            continue
        cb, ca = mean([B[t]["cer"] for t in ts]), mean([A[t]["cer"] for t in ts])
        worse_by = max(A[t]["cer"] - B[t]["cer"] for t in ts)
        mark = "⚠ " if (ca - cb > 0.05 or worse_by >= 0.1 or flagged([A[t] for t in ts])) else ""
        if mark:
            flags.append(s["id"])
        L.append(
            f"| {mark}{s['id']} | {s['focus']} | {len(ts)} | {sum(t in same for t in ts)}/{len(ts)} |"
            f" {cb:.3f} | {ca:.3f} | {ca - cb:+.3f} |"
            f" {sum(A[t]['cer'] > B[t]['cer'] for t in ts) or ''} | {sum(A[t]['cer'] < B[t]['cer'] for t in ts) or ''} |"
        )
    ob, oa = mean([B[t]["cer"] for t in tags]), mean([A[t]["cer"] for t in tags])
    L += ["", f"Overall mean CER {ob:.3f} → {oa:.3f} ({oa - ob:+.3f}, {len(tags)} samples)", ""]
    L += [
        FLAG_DOC
        + " in the after run, or any sample's CER rose by 0.1 or more, or the mean CER rose by more than 0.05.",
        "",
    ]

    changed = [t for t in tags if t not in same]
    if changed:
        L += ["## Samples that changed", ""]
        for s in data["sentences"]:
            ts = sorted((t for t in changed if A[t]["id"] == s["id"]), key=lambda t: A[t]["seed"])
            if not ts:
                continue
            L += [f"### {'⚠ ' if s['id'] in flags else ''}{s['id']} — {s['focus']}", "", f"Input: {s['text']}", ""]
            if s.get("listen"):
                L += [f"Listen for: {s['listen']}", ""]
            L += ["| seed | before CER | before transcript | after CER | after transcript |", "|---|---|---|---|---|"]
            for t in ts:
                b, r = B[t], A[t]
                L.append(
                    f"| {r['seed']} | {b['cer']:.3f}{notes(b)} | {b['transcript']} | {r['cer']:.3f}{notes(r)} | {r['transcript']} |"
                )
            L += ["", "Audio, before → after:", ""]
            for t in ts:
                L.append(
                    f"- `{Path(a.before).resolve() / 'wav' / t}.wav` → `{Path(a.after).resolve() / 'wav' / t}.wav`"
                )
            L.append("")

    dst = Path(a.to) if a.to else Path(a.after) / "compare.md"
    dst.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {dst}  ({len(same)}/{len(tags)} bit-identical, mean CER {ob:.3f} -> {oa:.3f})")
    return True


def cmd_run(a):
    ok = cmd_gen(a)
    cmd_score(a)
    cmd_report(a)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True, help="output directory (resumable)")
        p.add_argument("--sentences", default=str(SENTENCES))

    def gen_args(p):
        p.add_argument(
            "--tree", required=True, help="tt-metal checkout/worktree whose models/demos/qwen3_tts is evaluated"
        )
        p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=None)
        p.add_argument("--only", type=lambda s: s.split(","), default=None, help="comma-separated sentence ids")
        p.add_argument("--hf-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        p.add_argument("--timeout", type=int, default=300, help="seconds per generation process")

    p = sub.add_parser("gen", help="generate audio")
    common(p)
    gen_args(p)
    p = sub.add_parser("score", help="transcribe with Whisper and compute CER")
    common(p)
    p = sub.add_parser("report", help="write report.md for one run")
    common(p)
    p = sub.add_parser("run", help="gen + score + report")
    common(p)
    gen_args(p)
    p = sub.add_parser("compare", help="compare a run before the change with a run after it")
    p.add_argument("before", help="--out directory of the run before the change")
    p.add_argument("after", help="--out directory of the run after the change")
    p.add_argument("--to", default=None, help="output file (default: AFTER/compare.md)")
    p.add_argument("--sentences", default=str(SENTENCES))

    a = ap.parse_args()
    ok = {"gen": cmd_gen, "score": cmd_score, "report": cmd_report, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
