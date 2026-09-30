"""t16 post: pull the 5-seed clips from blx03, write meta.json with per-seed timings. Usage: post.py <job_id>"""
import json, re, subprocess, sys
from pathlib import Path

job = sys.argv[1]
B = Path("/home/smarton/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s/ref_dv145")
R = "g14blx03:/home/smarton/fasth3/out/ltx25_1080p_6s/seeds5"
B.mkdir(parents=True, exist_ok=True)
(B / "raw").mkdir(exist_ok=True)
subprocess.run(["rsync", "-a", f"{R}/run.log", f"{R}/ltx_av_fast_1920x1088_1.mp4", str(B / "raw") + "/"], check=True)
for s in range(5):
    subprocess.run(["rsync", "-a", f"{R}/ltx_av_fast_1920x1088_seed{s}.mp4", str(B / f"seed{s}.mp4")], check=True)
log = (B / "raw" / "run.log").read_text(errors="replace")

# Per-gen stage times, in log order; each gen ends with its E2E_WALL_S line.
pats = {
    "text_encode_s": r"gemma text-encode: done in ([\d.]+)s",
    "stage1_s": r"Stage 1 denoise: ([\d.]+)s",
    "stage2_s": r"Stage 2 denoise: ([\d.]+)s",
    "vae_decode_s": r"VAE decode \(forward\): ([\d.]+)s",
    "audio_decode_s": r"Audio decode: ([\d.]+)s",
    "export_s": r"Video export: ([\d.]+)s",
    "compute_total_s": r"Total \(compute\): ([\d.]+)s",
}
gens, cur = {}, {}
for line in log.splitlines():
    for k, p in pats.items():
        m = re.search(p, line)
        if m:
            cur[k] = float(m.group(1))
    m = re.search(r"E2E_WALL_S gen#(\S+): ([\d.]+)", line)
    if m:
        cur["e2e_wall_s"] = float(m.group(2))
        gens[m.group(1)] = cur
        cur = {}
# Two-decimal tables, printed once per gen in the same order.
tables = re.findall(
    r"Stage 1 denoise\s+│\s+([\d.]+) s.*?Stage 2 denoise\s+│\s+([\d.]+) s.*?VAE decode\s+│\s+([\d.]+) s.*?Audio decode\s+│\s+([\d.]+) s.*?Total\s+│\s+([\d.]+) s",
    log,
    re.S,
)
for g, t in zip(gens.values(), tables):
    g["table"] = dict(zip(["stage1_s", "stage2_s", "vae_decode_s", "audio_decode_s", "total_s"], map(float, t)))

head = re.search(r"\[run25\] (.*)", log)
env_lines = subprocess.run(
    [
        "ssh",
        "g14blx03",
        "grep -E '^export|^PYTEST' ~/fasth3/t16/tmp/blx03/run25.sh; cat ~/fasth3/t16/tmp/blx03/env16.yaml; git -C ~/fasth3/t16 rev-parse HEAD",
    ],
    capture_output=True,
    text=True,
).stdout
meta = {
    "what": "LTX-2.5 distilled, 1088x1920, 145f @ 24fps (6.04s), DiffVAE decode, default steps (S1 8, S2 3), traced replay",
    "host": "g14blx03 (bh 4x8, sp1tp0 ring)",
    "broker_job": job,
    "run25": head.group(1) if head else None,
    "extra_env": {"LTX_SEEDS": "0,1,2,3,4", "LTX_FRESH_PROMPTS": "0", "W": "/home/smarton/fasth3/t16"},
    "commit_and_env": env_lines,
    "prompt": "DEFAULT_LTX_PROMPT (models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py)",
    "notes": "gen#0 = trace-capture warmup (seed 0), gen#1 = first pure replay (seed 0); seedN = later replays, same prompt. "
    "Text-encode is on the path each gen (no embedding cache hit expected for identical prompt? see text_encode_s).",
    "gens": gens,
    "seeds": {f"seed{s}": {"file": f"seed{s}.mp4", "seed": s, **gens.get(f"seed{s}", {})} for s in range(5)},
}
(B / "meta.json").write_text(json.dumps(meta, indent=1))
print(
    json.dumps(
        {
            k: {kk: v.get(kk) for kk in ("stage1_s", "stage2_s", "vae_decode_s", "compute_total_s", "e2e_wall_s")}
            for k, v in gens.items()
        },
        indent=1,
    )
)
