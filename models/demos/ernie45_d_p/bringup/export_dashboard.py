# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Render the bring-up dashboard from the ledger (tasks.yaml + state.json + results/ + components.yaml).

    python models/demos/ernie45_d_p/bringup/export_dashboard.py
    -> models/demos/ernie45_d_p/bringup/dashboard/index.html  (self-contained; data inlined)
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

BRINGUP = Path(__file__).resolve().parent
REPO = BRINGUP.parents[3]
sys.path.insert(0, str(REPO))
from models.demos.ernie45_d_p.bringup import metrics as M  # noqa: E402
from models.demos.ernie45_d_p.bringup.plan import plan  # noqa: E402
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig  # noqa: E402

# Which task's per-layer metrics feed which PCC trail. Metric names must end with _Lxx.
TRAILS = [
    ("P1.2", "pcc_hidden_L", "CPU ref vs HF (512 tok)"),
    ("P2.11", "pcc_layer_L", "TT 2k->2k"),
    ("P2.12", "pcc_layer_L", "TT 8k->8k"),
    ("P2.13", "pcc_layer_L", "TT 50k->55k (b)"),
    ("P2.14", "pcc_layer_L", "TT 55k@5k (a)"),
]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    spec = yaml.safe_load((BRINGUP / "tasks.yaml").read_text())
    state = json.loads((BRINGUP / "state.json").read_text()) if (BRINGUP / "state.json").exists() else {}
    comp_doc = yaml.safe_load((BRINGUP / "components.yaml").read_text())
    comps, findings = comp_doc["components"], comp_doc.get("findings", [])

    commits = {}
    for line in git("log", "--format=%h|%cI|%s", "--grep=[ernie45_d_p]", "--fixed-strings").splitlines():
        sha, when, subj = line.split("|", 2)
        m = re.match(r"\[ernie45_d_p\]\[([^\]]+)\]", subj)
        if m and m.group(1) not in commits:
            commits[m.group(1)] = {"sha": sha, "when": when, "subject": subj}

    tasks = []
    for t in spec["tasks"]:
        s = state.get(t["id"], {})
        tasks.append(
            {
                "id": t["id"],
                "title": t["title"],
                "phase": t["phase"],
                "deps": t.get("deps", []),
                "cmd": t["gate"]["cmd"],
                "thresholds": t["gate"].get("metrics", {}),
                "status": s.get("status", "TODO"),
                "last_run": s.get("last_run"),
                "duration_s": s.get("duration_s"),
                "metrics": s.get("metrics", {}),
                "commit": commits.get(t["id"], {}).get("sha"),
                "notes": t.get("notes"),
            }
        )
    status_of = {t["id"]: t["status"] for t in tasks}
    for c in comps:
        c["state"] = "device" if c["task"] and status_of.get(c["task"]) == "PASS" else "cpu"
        if c["tag"] == "CPU":
            c["state"] = "cpu"

    trails = []
    for tid, prefix, label in TRAILS:
        got = M.load(tid)
        pts = sorted((int(k[len(prefix) :]), v["value"]) for k, v in got.items() if k.startswith(prefix))
        if pts:
            trails.append({"task": tid, "label": label, "points": pts})

    cfg = ErnieConfig()
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "--short", "HEAD"),
        "model": spec["model"],
        "target": spec["target"],
        "config": {
            "hidden": cfg.hidden_size,
            "layers": cfg.num_hidden_layers,
            "q_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "experts": cfg.moe_num_experts,
            "top_k": cfg.moe_k,
            "shared": cfg.moe_num_shared_experts,
            "moe_ffn": cfg.moe_intermediate_size,
            "dense_ffn": cfg.intermediate_size,
            "vocab": cfg.vocab_size,
            "rope_theta": cfg.rope_theta,
            "layer_types": ["moe" if cfg.is_moe_layer(i) else "dense" for i in range(cfg.num_hidden_layers)],
        },
        "tasks": tasks,
        "components": comps,
        "plan": plan(cfg),
        "trails": trails,
        "findings": findings,
        "timing": [
            {
                "task": tid,
                "label": label,
                "chunks": sorted(
                    (int(k.rsplit("_c", 1)[1]), v["value"])
                    for k, v in M.load(tid).items()
                    if k.startswith("chunk_seconds_c")
                ),
            }
            for tid, _, label in TRAILS
            if tid.startswith("P2") and M.load(tid)
        ],
        "box": {"name": "Blackhole QuietBox", "chips": "4x p150b", "mesh": "1x4 (FABRIC_1D_RING)"},
    }
    p21 = state.get("P2.1", {}).get("metrics", {})
    if p21:
        data["box"].update(worker_grid=p21.get("worker_grid"), dram_banks=p21.get("dram_banks"))

    tpl = (BRINGUP / "dashboard" / "template.html").read_text()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data))
    (BRINGUP / "dashboard" / "index.html").write_text(out)
    print(f"wrote {BRINGUP / 'dashboard' / 'index.html'} ({len(out) // 1024} KB)")


if __name__ == "__main__":
    main()
