"""Cumulative candidates preserve the current precision and residual contracts."""

import argparse
import copy
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
    a = p.parse_args()
    out = ROOT / "doc/optimized_decoder" / f"combined_{a.layer}"
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "default": {"block": 1},
        "carry": True,
        "lifetime": True,
        "self_attn.o_proj": {"cores": 8, "block": 16},
        "mlp.gate_proj": {"cores": 56, "block": 3},
        "mlp.up_proj": {"cores": 56, "block": 3},
        "mlp.down_proj": {"cores": 8, "block": 14},
    }
    qroles = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if a.layer == 0 else ["self_attn.qk_proj"]
    for role in qroles:
        base[role] = {"cores": 8, "block": 7, "readers": 2}
    candidates = [("combined", base)]
    for block in [12, 21]:
        g = copy.deepcopy(base)
        for role in qroles:
            g[role] = {"cores": 8 if block == 21 else 14, "block": block, "readers": 2}
        candidates.append((f"q_b{block}", g))
    for readers in [1, 2, 3]:
        for block in [6, 12, 24]:
            g = copy.deepcopy(base)
            for role in ["mlp.gate_proj", "mlp.up_proj"]:
                g[role] = {"cores": 56, "block": block, "readers": readers}
            candidates.append((f"gate56_b{block}_r{readers}", g))
    for block in [16, 32]:
        g = copy.deepcopy(base)
        g["self_attn.o_proj"] = {"cores": 8, "block": block, "readers": 3}
        candidates.append((f"o_b{block}_r3", g))
    for block in [14, 28, 56]:
        g = copy.deepcopy(base)
        g["mlp.down_proj"] = {"cores": 8, "block": block, "readers": 3}
        candidates.append((f"down_b{block}_r3", g))
    records = json.loads((out / "manifest.json").read_text()) if a.resume and (out / "manifest.json").exists() else []
    done = {r["candidate"] for r in records if r["phase"] == "timing" or r["exit_code"]}
    for name, g in candidates:
        if name in done:
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
                "bfp4",
                "--layout",
                "--geometry",
                json.dumps(g),
                "--result",
                str(out / f"{name}_{phase}.json"),
            ]
            if phase == "correctness":
                cmd += ["--seq", "33", "--inputs", f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs.pt"]
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
            records.append(dict(candidate=name, phase=phase, geometry=g, command=cmd, exit_code=code))
            (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
            print("END", name, phase, code, flush=True)
            if code:
                break


if __name__ == "__main__":
    main()
