"""Summarize the t140 eval pack: per-config warm e2e and stage times, and PCC/PSNR against baseline.

    python tmp/t140/post.py <root>          # root = rsynced /var/tmp/fasth3/t140 (one dir per config)
    python tmp/t140/post.py <root> --no-eval

Writes <root>/summary.json and <root>/summary.md. Quality uses ltx_eval video --vbench none per gen,
against the same gen (same prompt and seed) of the baseline config.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

STAGES = ["Encoder", "Stage 1 denoise", "Latent upsample", "Stage 2 denoise", "VAE decode", "Audio decode", "Total"]


def parse(log: Path) -> dict:
    text = log.read_text(errors="replace")
    gens = {}
    for m in re.finditer(r"E2E_WALL_S gen#(\d+): ([\d.]+)", text):
        gens.setdefault(int(m.group(1)), {})["e2e"] = float(m.group(2))
    exports = [float(x) for x in re.findall(r"Video export: ([\d.]+)s", text)]
    for gen, t in zip(sorted(gens), exports[-len(gens) :] if gens else []):
        gens[gen]["export"] = t
    # Each performance table names its output file ..._<gen>.mp4, then lists the stages.
    for block in re.split(r"LTX DISTILLED — PERFORMANCE", text)[1:]:
        m = re.search(r"_(\d+)\.mp4", block)
        if not m:
            continue
        g = gens.setdefault(int(m.group(1)), {})
        for st in STAGES:
            sm = re.search(r"│ " + re.escape(st) + r"\s+│\s+([\d.]+) s", block)
            if sm:
                g[st] = float(sm.group(1))
    flags = re.search(r"\[t140\] label=\S+ flags=(.*?) host=", text)
    return {
        "flags": flags.group(1) if flags else "",
        "exit": (re.findall(r"T140_EXIT\[\S+\]=(\d+)", text) or ["?"])[-1],
        "gens": gens,
    }


def evaluate(root: Path, cfg: str, gen: int) -> dict:
    cand = root / cfg / f"ltx_av_fast_1920x1088_{gen}.mp4"
    ref = root / "baseline" / cand.name
    if cfg == "baseline" or not cand.exists() or not ref.exists():
        return {}
    out = root / cfg / f"eval_gen{gen}"
    rep = out / f"{cand.stem}_report.json"
    if not rep.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "models.tt_dit.tests.models.ltx.tools.ltx_eval",
                "video",
                "--cand",
                str(cand),
                "--ref",
                str(ref),
                "--out",
                str(out),
                "--vbench",
                "none",
            ],
            check=False,
        )
    if not rep.exists():
        return {}
    p = json.loads(rep.read_text())["parity"]
    return {"pcc": p["pcc"], "pcc_min": p["pcc_min"], "psnr": p["psnr"], "psnr_min": p["psnr_min"]}


def main():
    root = Path(sys.argv[1])
    do_eval = "--no-eval" not in sys.argv
    order = [
        l.split()[0]
        for l in (Path(__file__).parent / "configs.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    cfgs = [c for c in order if (root / c / "run.log").exists()]
    res = {}
    for c in cfgs:
        r = parse(root / c / "run.log")
        if do_eval:
            for g in r["gens"]:
                r["gens"][g]["quality"] = evaluate(root, c, g)
        res[c] = r
    (root / "summary.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    base = res.get("baseline", {}).get("gens", {}).get(1, {}).get("e2e")
    cols = [
        "e2e",
        "Encoder",
        "Stage 1 denoise",
        "Latent upsample",
        "Stage 2 denoise",
        "VAE decode",
        "Audio decode",
        "export",
    ]
    lines = [
        "| config | exit | gen1 e2e (s) | d vs base | gen2 e2e | encode | S1 | upsample | S2 | VAE | audio | export | gen1 PCC (min) | gen1 PSNR (min) dB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for c in cfgs:
        g1 = res[c]["gens"].get(1, {})
        g2 = res[c]["gens"].get(2, {})
        q = g1.get("quality", {})
        d = f"{g1['e2e'] - base:+.3f}" if base and "e2e" in g1 else ""
        vals = [f"{g1[k]:.3f}" if k == "e2e" and k in g1 else (f"{g1[k]:.2f}" if k in g1 else "") for k in cols]
        pcc = f"{q['pcc']:.5f} ({q['pcc_min']:.5f})" if q else ("ref" if c == "baseline" else "")
        psnr = f"{q['psnr']:.2f} ({q['psnr_min']:.2f})" if q else ("ref" if c == "baseline" else "")
        lines.append(
            f"| {c} | {res[c]['exit']} | {vals[0]} | {d} | {g2.get('e2e', '')} | "
            + " | ".join(vals[1:])
            + f" | {pcc} | {psnr} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
