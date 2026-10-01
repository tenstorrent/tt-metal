#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Post-process the t86 eval pack (run on g15blx02 after copying the pack from blx03).

  post_eval.py parse <run.log>                  # timings/DRAM/load of one run as JSON
  post_eval.py pack <pack-dir> [--no-eval]      # every config: parse, link clips, ltx_eval, table

For each config the pack mode writes <cfg>/timing.json, links the seed clips as <cfg>/clips/seedN.mp4,
then runs ltx_eval twice: against ref_dv145 with partial VBench (eval_ref/, stills included) and against
the pack's own baseline clips without VBench (eval_ab/, PCC/PSNR only: the flag's own effect, without
the DiffVAE-vs-conv decoder difference the reference carries). The table lands in pack_summary.{json,md}.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \|")
STAGES = {
    "encode_s": re.compile(r"gemma text-encode: done in ([\d.]+)s"),
    "s1_s": re.compile(r"Stage 1 denoise: ([\d.]+)s"),
    "upsample_s": re.compile(r"Latent upsample: ([\d.]+)s"),
    "s2_s": re.compile(r"Stage 2 denoise: ([\d.]+)s"),
    "vae_s": re.compile(r"VAE decode \(forward\): ([\d.]+)s"),
    "audio_s": re.compile(r"Audio decode: ([\d.]+)s"),
    "export_s": re.compile(r"Video export: ([\d.]+)s"),
    "compute_s": re.compile(r"Total \(compute\): ([\d.]+)s"),
}
VAE_SPLIT = re.compile(
    r"VAE_DECODE_SPLIT traced=(\d) upload=([\d.]+)ms decode=([\d.]+)ms output=([\d.]+)ms total=([\d.]+)ms"
)
AUDIO_SPLIT = re.compile(r"STAGE_SPLIT mel_vae=([\d.]+)ms vocoder\+bwe=([\d.]+)ms")
DRAM = re.compile(r"\[dram\] (.*?): allocated +([\d.]+) GiB of ([\d.]+) GiB, largest contiguous free +([\d.]+) GiB")
GEN_START = "Running LTX AV Fast"
GEN_END = re.compile(r"E2E_WALL_S gen#(\S+): ([\d.]+)")
REF_DIR = Path("/home/smarton/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s/ref_dv145")
VBENCH_DIMS = "subject_consistency,background_consistency,motion_smoothness,imaging_quality"
VBENCH_TOL = 0.01


def _ts(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def parse_log(text):
    """Per-gen stage times, DRAM peak and load time from one test_pipeline_distilled log."""
    gens, cur, dram = {}, None, []
    first_ts = first_gen_ts = None
    for line in text.splitlines():
        ts = _ts(line)
        if ts and first_ts is None:
            first_ts = ts
        if GEN_START in line:
            cur = {}
            if first_gen_ts is None:
                first_gen_ts = ts
            continue
        if m := DRAM.search(line):
            dram.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), cur is not None))
        if cur is None:
            continue
        for key, pat in STAGES.items():
            if m := pat.search(line):
                cur[key] = float(m.group(1))
        if m := VAE_SPLIT.search(line):
            cur["vae_traced"] = int(m.group(1))
            cur["vae_upload_ms"], cur["vae_decode_ms"], cur["vae_output_ms"], cur["vae_total_ms"] = map(
                float, m.groups()[1:]
            )
        if m := AUDIO_SPLIT.search(line):
            cur["mel_vae_ms"], cur["vocoder_ms"] = float(m.group(1)), float(m.group(2))
        if m := GEN_END.search(line):
            cur["e2e_s"] = float(m.group(2))
            gens[m.group(1)] = cur
            cur = None
    out = {"gens": gens}
    if first_ts and first_gen_ts:
        out["load_s"] = round((first_gen_ts - first_ts).total_seconds(), 1)
    if dram:
        peak = max(dram, key=lambda d: d[1])
        out["dram_total_gib"] = peak[2]
        out["dram_peak_gib"] = peak[1]
        out["dram_peak_at"] = peak[0]
        out["dram_min_free_contig_gib"] = min(d[3] for d in dram)
        in_gens = [d for d in dram if d[4]]
        if in_gens:
            out["dram_peak_in_gens_gib"] = max(d[1] for d in in_gens)
    out["exit"] = m.group(1) if (m := re.search(r"RUN_EXIT\[[^\]]+\]=(\d+)", text)) else None
    return out


def _median(gens, labels, key):
    vals = [gens[g][key] for g in labels if g in gens and key in gens[g]]
    return round(statistics.median(vals), 3) if vals else None


STAGE_KEYS = ["encode_s", "s1_s", "upsample_s", "s2_s", "vae_s", "vae_decode_ms", "vae_output_ms", "audio_s"]
STAGE_KEYS += ["export_s", "compute_s", "e2e_s"]


def condense(parsed):
    """Medians over the fresh-prompt replays (encoder on the path) and over the 5 seed replays."""
    gens = parsed["gens"]
    fresh = [g for g in gens if g.isdigit() and int(g) >= 1]
    seeds = sorted(g for g in gens if g.startswith("seed"))
    row = {k: parsed.get(k) for k in ("load_s", "dram_peak_gib", "dram_min_free_contig_gib", "exit")}
    row["gen0_e2e_s"] = gens.get("0", {}).get("e2e_s")
    row["vae_traced"] = next((gens[g]["vae_traced"] for g in seeds if "vae_traced" in gens[g]), None)
    row["fresh"] = {k: _median(gens, fresh, k) for k in STAGE_KEYS}
    row["seeds"] = {k: _median(gens, seeds, k) for k in STAGE_KEYS}
    row["n_fresh"], row["n_seeds"] = len(fresh), len(seeds)
    return row


def link_clips(cfg_dir):
    clips = cfg_dir / "clips"
    clips.mkdir(exist_ok=True)
    found = []
    for mp4 in sorted(cfg_dir.glob("*_seed*.mp4")):
        seed = re.search(r"_(seed\d+)\.mp4$", mp4.name).group(1)
        link = clips / f"{seed}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / mp4.name)
        found.append(seed)
    return found


def run_eval(cand, ref, out, vbench, log):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "models/tt_dit/tests/models/ltx/tools/ltx_eval.py"),
    ]
    cmd += ["batch", "--cand-dir", str(cand), "--ref-dir", str(ref), "--out", str(out), "--vbench", vbench]
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    summary = out / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else {"error": f"rc={rc}, see {log}"}


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def table(rows, ref_vb):
    head = "| config | exit | load s | gen0 s | e2e fresh | e2e seeds | enc | S1 | ups | S2 | VAE | VAE dec ms | VAE out ms"
    head += " | audio | export | DRAM peak GiB | min free GiB | VBench Δ vs ref (worst dim) | PCC/PSNRmin vs ref | PCC/PSNRmin vs baseline |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, r in rows.items():
        f, s = r["fresh"], r["seeds"]
        vb = r.get("eval_ref", {}).get("vbench_mean") or {}
        deltas = {d: vb[d] - ref_vb[d] for d in vb if d in ref_vb}
        worst = min(deltas.items(), key=lambda kv: kv[1]) if deltas else None
        vb_txt = f"{worst[0]} {worst[1]:+.4f}" if worst else "-"
        if worst and worst[1] < -VBENCH_TOL:
            vb_txt += " FAIL"

        def par(ev):
            if not ev or "pcc_mean" not in ev:
                return "-"
            return f"{ev['pcc_mean']:.4f}/{ev['psnr_worst']:.1f}"

        lines.append(
            f"| {name}{' (VAE traced)' if r.get('vae_traced') else ''} | {r['exit']} | {_fmt(r['load_s'], 0)}"
            f" | {_fmt(r['gen0_e2e_s'], 1)} | {_fmt(f['e2e_s'])} | {_fmt(s['e2e_s'])} | {_fmt(f['encode_s'], 1)}"
            f" | {_fmt(s['s1_s'], 1)} | {_fmt(s['upsample_s'], 1)} | {_fmt(s['s2_s'], 1)} | {_fmt(s['vae_s'], 1)}"
            f" | {_fmt(s['vae_decode_ms'], 0)} | {_fmt(s['vae_output_ms'], 0)} | {_fmt(s['audio_s'], 1)}"
            f" | {_fmt(s['export_s'], 1)} | {_fmt(r['dram_peak_gib'])} | {_fmt(r['dram_min_free_contig_gib'])}"
            f" | {vb_txt} | {par(r.get('eval_ref'))} | {par(r.get('eval_ab'))} |"
        )
    lines.append("")
    lines.append(
        "Stage columns are medians over the 5 seed replays (default prompt; the encode is a cache hit there),"
        " except `e2e fresh` and `enc`, which are medians over gen#1-3 (fresh prompts, encoder on the path)."
        " Stage logs round to 0.1 s; VAE dec/out come from VAE_DECODE_SPLIT (ms, synced)."
    )
    return "\n".join(lines)


def pack(root, do_eval, ref_dir, vbench):
    global REF_DIR
    REF_DIR = ref_dir
    ref_vb = json.loads((REF_DIR / "eval/summary.json").read_text())["vbench_mean"]
    rows = {}
    cfgs = [p for p in sorted(root.iterdir()) if (p / "run.log").exists()]
    cfgs.sort(key=lambda p: p.name != "baseline")  # baseline first: the A/B needs its clips
    for cfg in cfgs:
        parsed = parse_log((cfg / "run.log").read_text(errors="replace"))
        (cfg / "timing.json").write_text(json.dumps(parsed, indent=1))
        row = condense(parsed)
        row["clips"] = link_clips(cfg)
        if do_eval and row["clips"]:
            row["eval_ref"] = run_eval(cfg / "clips", REF_DIR, cfg / "eval_ref", vbench, cfg / "eval_ref.log")
            base = root / "baseline/clips"
            if cfg.name != "baseline" and base.exists():
                row["eval_ab"] = run_eval(cfg / "clips", base, cfg / "eval_ab", "none", cfg / "eval_ab.log")
        rows[cfg.name] = row
        print(f"[post] {cfg.name}: exit={row['exit']} seeds={row['clips']} e2e_fresh={row['fresh']['e2e_s']}")
    (root / "pack_summary.json").write_text(json.dumps(rows, indent=1))
    md = table(rows, ref_vb)
    (root / "pack_summary.md").write_text(md + "\n")
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("parse")
    p.add_argument("log")
    k = sub.add_parser("pack")
    k.add_argument("root")
    k.add_argument("--no-eval", action="store_true", help="timings only, skip ltx_eval")
    k.add_argument("--ref-dir", type=Path, default=REF_DIR)
    k.add_argument("--vbench", default=VBENCH_DIMS, help="dims for the run against the reference, or 'none'")
    args = ap.parse_args(argv)
    if args.mode == "parse":
        parsed = parse_log(Path(args.log).read_text(errors="replace"))
        print(json.dumps({"condensed": condense(parsed), **parsed}, indent=1))
    else:
        pack(Path(args.root), not args.no_eval, args.ref_dir, args.vbench)


if __name__ == "__main__":
    main()
