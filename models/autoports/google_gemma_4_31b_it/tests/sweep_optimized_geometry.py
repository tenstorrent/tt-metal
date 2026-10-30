"""Precision-locked whole-layer geometry comparisons, serialized on one ASIC."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attention-precision", default="bfp8")
    p.add_argument("--inputs")
    p.add_argument("--skip", nargs="*", default=[])
    a = p.parse_args()
    out = ROOT / "doc/optimized_decoder" / f"geometry_{a.layer}_{a.attention_precision}"
    out.mkdir(parents=True, exist_ok=True)
    base = {"default": {"block": 1}, "carry": True}
    candidates = [("control", base)]
    attention = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if a.layer == 0 else ["self_attn.qk_proj"]
    for block in [3, 7, 21]:
        g = copy.deepcopy(base)
        for role in attention:
            g[role] = {"cores": 8, "block": block}
        candidates.append((f"qkv_b{block}", g))
    for block in [4, 8, 16, 32]:
        g = copy.deepcopy(base)
        g["self_attn.o_proj"] = {"cores": 8, "block": block}
        candidates.append((f"o_b{block}", g))
    for cores, block in [(8, 3), (8, 7), (7, 8), (7, 12), (7, 24), (14, 6), (14, 12), (28, 6), (56, 3)]:
        g = copy.deepcopy(base)
        for role in ["mlp.gate_proj", "mlp.up_proj"]:
            g[role] = {"cores": cores, "block": block}
        candidates.append((f"gate_c{cores}_b{block}", g))
    for cores, block in [(8, 7), (8, 14), (8, 28), (7, 16), (7, 24), (14, 24), (28, 12), (56, 12)]:
        g = copy.deepcopy(base)
        g["mlp.down_proj"] = {"cores": cores, "block": block}
        candidates.append((f"down_c{cores}_b{block}", g))
    for readers in [2, 3]:
        for group, roles in [
            ("qkv", attention),
            ("o", ["self_attn.o_proj"]),
            ("gate", ["mlp.gate_proj", "mlp.up_proj"]),
            ("down", ["mlp.down_proj"]),
        ]:
            g = copy.deepcopy(base)
            for role in roles:
                g[role] = {"cores": 8, "block": 7 if group != "o" else 8, "readers": readers}
            candidates.append((f"{group}_r{readers}", g))
    records = json.loads((out / "manifest.json").read_text()) if a.resume and (out / "manifest.json").exists() else []
    completed = {r["candidate"] for r in records if r["phase"] == "timing" or r["exit_code"] != 0}
    for name, g in candidates:
        if name in completed or name in a.skip:
            continue
        for phase, module in [("correctness", "run_optimized_decoder"), ("timing", "time_optimized_candidate")]:
            cmd = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests." + module,
                "--layer",
                str(a.layer),
                "--precision",
                "bfp4",
                "--attention-precision",
                a.attention_precision,
                "--layout",
                "--geometry",
                json.dumps(g),
                "--result",
                str(out / f"{name}_{phase}.json"),
            ]
            if phase == "correctness":
                cmd += ["--seq", "33"]
                if a.inputs:
                    cmd += ["--inputs", a.inputs]
            print("START", name, phase, flush=True)
            with (out / f"{name}_{phase}.log").open("w") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"
                    ),
                )
                try:
                    code = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    # Preserve the live failure before stopping this queue. Recovery belongs to its owner.
                    triage = [
                        sys.executable,
                        "tools/tt-triage.py",
                        "--llm-output",
                        "--llm-output-path",
                        str(out / f"{name}_triage.txt"),
                        "--triage-summary-path",
                        str(out / f"{name}_triage_summary.txt"),
                    ]
                    try:
                        subprocess.run(triage, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                    proc.kill()
                    proc.wait()
                    (out / f"{name}_timeout.json").write_text(
                        json.dumps(dict(command=cmd, status="timeout_after_triage", requires_recovery=True))
                    )
                    raise RuntimeError("Candidate stalled; queue stopped for device recovery")
            records.append(
                dict(
                    candidate=name,
                    phase=phase,
                    geometry=g,
                    precision="MLP BFP4 LoFi / attention " + a.attention_precision,
                    command=cmd,
                    exit_code=code,
                    source_hashes={
                        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in [ROOT / "tt/optimized_decoder.py", ROOT / "tests/optimized_layout_candidate.py"]
                    },
                )
            )
            (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
            print("END", name, phase, code, flush=True)
            if code:
                break


if __name__ == "__main__":
    main()
