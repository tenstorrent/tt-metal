"""Summarize t164 eval-pack runs: warm e2e and stage times per config, PCC/PSNR against the baseline.

    python post.py <root> <configs.txt> [--no-eval]     # root = data/g15/t164 (one dir per label)

Each clip is scored only against the baseline clip with the same gen index, hence the same prompt and
seed (gen#0 uses the default prompt; with LTX_FRESH_PROMPTS=1, gen#N>=1 uses FRESH_LTX_PROMPTS[N-1]).
A config whose label ends in "5" is scored against "baseline5". Writes <root>/summary_<tag>.{json,md}.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

STAGES = ["Encoder", "Stage 1 denoise", "Latent upsample", "Stage 2 denoise", "VAE decode", "Audio decode", "Total"]
FRESH = ["paper boat", "fisherman", "barista"]


def env_of(text: str) -> dict:
    return dict(re.findall(r"^(LTX_[A-Z0-9_]+|SEED)=(\S*)$", text, re.M))


def prompt_of(env: dict, gen: int) -> str:
    if gen == 0 or env.get("LTX_FRESH_PROMPTS") != "1":
        return "default (rapper)"
    return FRESH[(gen - 1) % len(FRESH)]


def seed_of(env: dict, gen: int) -> int:
    seeds = [int(s) for s in env.get("LTX_E2E_SEEDS", env.get("SEED", "0")).split(",")]
    return int(env.get("SEED", "0")) if gen == 0 or gen > len(seeds) else seeds[gen - 1]


def parse(log: Path) -> dict:
    text = log.read_text(errors="replace")
    env = env_of(text)
    gens = {}
    for m in re.finditer(r"E2E_WALL_S gen#(\d+): ([\d.]+)", text):
        g = int(m.group(1))
        gens[g] = {"e2e": float(m.group(2)), "prompt": prompt_of(env, g), "seed": seed_of(env, g)}
    exports = [float(x) for x in re.findall(r"Video export: ([\d.]+)s", text)]
    for gen, t in zip(sorted(gens), exports[-len(gens) :] if gens else []):
        gens[gen]["export"] = t
    for block in re.split(r"LTX DISTILLED — PERFORMANCE", text)[1:]:
        m = re.search(r"_(\d+)\.mp4", block)
        if not m:
            continue
        g = gens.setdefault(int(m.group(1)), {})
        for st in STAGES:
            sm = re.search(r"│ " + re.escape(st) + r"\s+│\s+([\d.]+) s", block)
            if sm:
                g[st] = float(sm.group(1))
    flags = re.search(r"\[t164\] label=\S+ flags=(.*?) host=", text)
    wall = re.search(r"\[t164\] process wall (\d+) s", text)
    return {
        "flags": flags.group(1) if flags else "",
        "exit": (re.findall(r"T164_EXIT\[\S+\]=(\d+)", text) or ["?"])[-1],
        "process_wall_s": int(wall.group(1)) if wall else None,
        "gens": gens,
    }


def evaluate(root: Path, cfg: str, base: str, gen: int) -> dict:
    cand = root / cfg / f"ltx_av_fast_1920x1088_{gen}.mp4"
    ref = root / base / cand.name
    if cfg == base or not cand.exists() or not ref.exists():
        return {}
    out = root / cfg / f"eval_gen{gen}"
    rep = out / f"{cand.stem}_report.json"
    if not rep.exists():
        cmd = [
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
            "--stills",
            "72",
        ]
        subprocess.run(cmd, check=False)
    if not rep.exists():
        return {}
    p = json.loads(rep.read_text())["parity"]
    return {"pcc": p["pcc"], "pcc_min": p["pcc_min"], "psnr": p["psnr"], "psnr_min": p["psnr_min"]}


def main():
    root, cfg_file = Path(sys.argv[1]), Path(sys.argv[2])
    do_eval = "--no-eval" not in sys.argv
    order = [l.split()[0] for l in cfg_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    cfgs = [c for c in order if (root / c / "run.log").exists()]
    res = {}
    for c in cfgs:
        r = parse(root / c / "run.log")
        base = "baseline5" if c.endswith("5") else "baseline"
        if do_eval:
            for g in r["gens"]:
                r["gens"][g]["quality"] = evaluate(root, c, base, g)
        res[c] = r
    tag = cfg_file.stem
    (root / f"summary_{tag}.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    cols = ["Encoder", "Stage 1 denoise", "Latent upsample", "Stage 2 denoise", "VAE decode", "Audio decode", "export"]
    lines = [
        "| config | exit | gen1 e2e (s) | d vs base | gen2 e2e | encode | S1 | upsample | S2 | VAE | audio | export "
        "| gen1 PCC (min) | gen1 PSNR (min) dB | gen2 PCC | gen2 PSNR | job wall (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for c in cfgs:
        base = "baseline5" if c.endswith("5") else "baseline"
        b1 = res.get(base, {}).get("gens", {}).get(1, {}).get("e2e")
        g1, g2 = res[c]["gens"].get(1, {}), res[c]["gens"].get(2, {})
        q1, q2 = g1.get("quality", {}), g2.get("quality", {})
        d = f"{g1['e2e'] - b1:+.3f}" if b1 and "e2e" in g1 else ""
        vals = [f"{g1[k]:.2f}" if k in g1 else "" for k in cols]
        ref = "ref" if c == base else ""
        f = lambda q, k, n: f"{q[k]:.{n}f}" if q else ref
        lines.append(
            f"| {c} | {res[c]['exit']} | {g1.get('e2e', '')} | {d} | {g2.get('e2e', '')} | "
            + " | ".join(vals)
            + f" | {f(q1, 'pcc', 5)} ({f(q1, 'pcc_min', 5)}) | {f(q1, 'psnr', 2)} ({f(q1, 'psnr_min', 2)})"
            + f" | {f(q2, 'pcc', 5)} | {f(q2, 'psnr', 2)} | {res[c]['process_wall_s'] or ''} |"
        )
    (root / f"summary_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
