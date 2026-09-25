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
    ("P3.4", "pcc_layer_L", "TT 55k@5k, SDPA cfg A"),
]


# One 50k->55k chunk, phase by phase (sections from tests/perf/test_profile_chunk.py signposts).
# pat: local | ring (all_reduce over the 4 chips). cell(c) -> what chip c holds. ms are summed over the 28 layers.
PROFILE_STEPS = [
    (
        "other.attn_norm",
        "Input RMSNorm",
        "attn",
        "other",
        "local",
        "Every chip normalizes the same replicated chunk.",
        "[5120, 2560]",
        "[5120, 2560]",
        lambda c: ("tokens 51200-56319", "full width"),
    ),
    (
        "attn.qkv",
        "Q/K/V projection (fused)",
        "attn",
        "compute",
        "local",
        "One matmul per chip into its own heads: 5 query heads plus its single KV head, 896 outputs per token.",
        "[5120, 2560] x [2560, 896]",
        "[5120, 896]",
        lambda c: (f"Q heads {5 * c}-{5 * c + 4}", f"KV head {c}"),
    ),
    (
        "attn.rope",
        "Interleaved RoPE at offset 51200",
        "attn",
        "compute",
        "local",
        "Rotary embedding for absolute positions 51200-56319, applied to this chip's 5 Q heads and 1 K head.",
        "[1, 5, 5120, 128] + [1, 1, 5120, 128]",
        "same",
        lambda c: (f"Q heads {5 * c}-{5 * c + 4}", f"K head {c}"),
    ),
    (
        "attn.kv_write",
        "Write new K/V into the cache",
        "attn",
        "memory",
        "local",
        "The chunk's K and V (head c) land at positions 51200-56319 of this chip's cache (bf16 attention cache).",
        "[1, 1, 5120, 128] x 2",
        "cache[51200:56320]",
        lambda c: (f"KV head {c}", "positions 51200-56319"),
    ),
    (
        "attn.sdpa",
        "Chunked causal attention (SDPA)",
        "attn",
        "compute",
        "local",
        "Each chip scores its 5 query heads for 5,120 new tokens against all 56,320 keys of its KV head "
        "(51,200 cached + the causal part of the chunk), then takes the weighted sum of values.",
        "Q [5, 5120, 128] vs K/V [1, 56320, 128]",
        "[5, 5120, 128]",
        lambda c: (f"Q heads {5 * c}-{5 * c + 4}", f"vs KV head {c}, 56k keys"),
    ),
    (
        "attn.o_proj",
        "Output projection (partial sum)",
        "attn",
        "compute",
        "local",
        "Heads are concatenated and projected with this chip's 640 rows of Wo, giving a partial sum over heads.",
        "[5120, 640] x [640, 2560]",
        "[5120, 2560] partial",
        lambda c: ("Wo rows", f"{640 * c}-{640 * c + 639}"),
    ),
    (
        "attn.all_reduce",
        "All-reduce attention output",
        "attn",
        "comm",
        "ring",
        "The four partial sums are added across the ring so every chip holds the full attention output.",
        "[5120, 2560] partial",
        "[5120, 2560] replicated",
        lambda c: ("partial sum", "-> full"),
    ),
    (
        "other.residual",
        "Residual adds + post-attention RMSNorm",
        "other",
        "other",
        "local",
        "Two residual adds per layer and the post-attention norm, on replicated activations.",
        "[5120, 2560]",
        "[5120, 2560]",
        lambda c: ("tokens 51200-56319", "full width"),
    ),
    (
        "moe.router",
        "Router (softmax, bias-corrected top-6)",
        "moe",
        "other",
        "local",
        "Every chip computes the same routing in fp32: 64 expert scores, top-6 chosen with the bias, weights renormalized.",
        "[5120, 2560] x [2560, 64]",
        "top-6 ids + weights [5120, 6]",
        lambda c: ("all 64 experts", "same routing"),
    ),
    (
        "moe.dispatch",
        "Dispatch (local regroup)",
        "moe",
        "memory",
        "local",
        "Count tokens per expert, then copy each (token, expert) pair routed to this chip's 16 experts into that "
        "expert's contiguous block. No token leaves its chip.",
        "[5120, 2560] + top-6 ids",
        "expert blocks",
        lambda c: (f"experts {16 * c}-{16 * c + 15}", "{rows} rows"),
    ),
    (
        "moe.experts",
        "Routed experts (fused FFN)",
        "moe",
        "compute",
        "local",
        "unified_routed_expert_moe runs all 16 local experts in one program, each over just its own tokens (SwiGLU, FFN 1536).",
        "per expert [n, 2560] x [2560, 1536] x2, x [1536, 2560]",
        "expert blocks (same shape)",
        lambda c: (f"experts {16 * c}-{16 * c + 15}", "{rows} rows"),
    ),
    (
        "moe.combine_reduce",
        "Combine + weighted top-6 sum",
        "moe",
        "memory",
        "local",
        "Results go back under their token and choice; each token's local expert outputs are weighted and summed. "
        "Choices owned by other chips stay zero.",
        "expert blocks",
        "[5120, 2560] partial (local experts)",
        lambda c: (f"experts {16 * c}-{16 * c + 15}", "-> per token"),
    ),
    (
        "moe.shared",
        "Shared experts (tensor-parallel)",
        "moe",
        "compute",
        "local",
        "The two shared experts (FFN 3072) are split 4 ways: gate/up by output columns, down by input rows.",
        "[5120, 2560] x [2560, 768] x2, x [768, 2560]",
        "[5120, 2560] partial",
        lambda c: ("shared FFN cols", f"{768 * c}-{768 * c + 767}"),
    ),
    (
        "moe.all_reduce",
        "All-reduce MoE output",
        "moe",
        "comm",
        "ring",
        "Routed partial + shared partial are added locally, then one all-reduce sums the four chips.",
        "[5120, 2560] partial",
        "[5120, 2560] replicated",
        lambda c: ("partial sum", "-> full"),
    ),
    (
        "dense_mlp",
        "Dense SwiGLU (layer 0 only)",
        "moe",
        "compute",
        "local",
        "Layer 0 has a dense MLP (FFN 12288) split 4 ways, followed by an all-reduce.",
        "[5120, 2560] x [2560, 3072] x2, x [3072, 2560]",
        "[5120, 2560] partial",
        lambda c: ("MLP cols", f"{3072 * c}-{3072 * c + 3071}"),
    ),
    (
        "model.other.embed",
        "Embedding lookup",
        "other",
        "other",
        "local",
        "Replicated embedding table lookup for the chunk's 5,120 tokens.",
        "5120 ids",
        "[5120, 2560]",
        lambda c: ("full table", "103k x 2560"),
    ),
    (
        "model.other.final_norm",
        "Final RMSNorm",
        "other",
        "other",
        "local",
        "Final norm of the last layer's output.",
        "[5120, 2560]",
        "[5120, 2560]",
        lambda c: ("tokens 51200-56319", "full width"),
    ),
]


def load_profile() -> dict | None:
    # Latest profiled configuration wins (P3.4 = SDPA config A); the P3.3 baseline is kept for comparison.
    rout_p = M.RESULTS_DIR / "P3.3_routing.json"
    prof_p = next((c for c in (M.RESULTS_DIR / f"{t}_profile.json" for t in ("P3.4", "P3.3")) if c.exists()), None)
    if prof_p is None:
        return None
    prof = json.loads(prof_p.read_text())
    base_p = M.RESULTS_DIR / "P3.3_profile.json"
    base = json.loads(base_p.read_text()) if (base_p.exists() and base_p != prof_p) else None
    rout = json.loads(rout_p.read_text()) if rout_p.exists() else {}
    rows = {int(k): v for k, v in rout.get("routed_rows_per_chip_total", {}).items()}
    steps = []
    for key, name, part, bound, pat, what, inp, out, cell in PROFILE_STEPS:
        if key not in prof["sections_ms"]:
            continue
        pc = prof["sections_ms_per_chip"].get(key, {})
        cells = []
        for c in range(4):
            a, b = cell(c)
            r = f"{round(rows[c] / 27):,}" if c in rows else "?"
            cells.append([a, b.replace("{rows}", r + " avg")])
        steps.append(
            dict(
                key=key,
                name=name,
                part=part,
                bound=bound,
                pat=pat,
                what=what,
                inp=inp,
                out=out,
                cells=cells,
                ms=round(prof["sections_ms"][key], 2),
                per_chip=[round(pc.get(str(c), 0.0), 2) for c in range(4)],
                programs=prof["programs"].get(key, 0),
            )
        )
    sd = prof.get("sdpa") or {}
    cfg = (
        f"SDPA {sd.get('fidelity')}, fp32 acc {'on' if sd.get('fp32') else 'off'}, "
        f"{'approx' if sd.get('exp_approx') else 'exact'} exp, q{sd.get('q')}/k{sd.get('k')}"
        if sd
        else "SDPA base config (HiFi4, fp32 acc on, exact exp, q256/k256)"
    )
    cmp = ""
    if base:
        cmp = (
            f" Before (P3.3, HiFi4 + fp32 acc): chunk {base['wall_ms']:.0f} ms, SDPA {base['sections_ms'].get('attn.sdpa', 0):.0f} ms;"
            f" now chunk {prof['wall_ms']:.0f} ms, SDPA {prof['sections_ms'].get('attn.sdpa', 0):.0f} ms."
        )
        for st in steps:
            if st["key"] in base["sections_ms"]:
                st["before_ms"] = round(base["sections_ms"][st["key"]], 2)
    return {
        "wall_ms": round(prof["wall_ms"], 1),
        "steps": steps,
        "routing": rout,
        "config": cfg,
        "source": prof_p.name,
        "note": f"{cfg} ({prof_p.stem}).{cmp} Device kernel time per phase, summed over the 28 layers of one 5,120-token "
        "chunk at positions 51,200-56,319 (golden 50k KV prefix loaded). Per chip = that chip's own programs; the bar "
        "uses the slowest chip per phase. Measured with the device profiler.",
    }


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
        "profile": load_profile(),
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
            if tid.startswith(("P2", "P3")) and M.load(tid)
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
