# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU analysis of paired native-router inputs/outputs from stack diagnostics."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots/e9ce717642b2074082875be233c02c729ba47960"
)


def main():
    base = torch.load(DOC / "local/paired_routes_True_snapshots.pt", weights_only=True)
    idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    for label in ("False", "teacher"):
        data = torch.load(DOC / f"local/paired_routes_{label}_snapshots.pt", weights_only=True)
        for decode, count in ((False, 33), (True, 2)):
            for name, experts, k in [("self_attn.v_router", 64, 4), ("mlp.gate", 100, 8)]:
                key = f"{decode}_route_{name}"
                weight = f"model.layers.3.{name}.bias"
                with safe_open(SNAP / idx[weight], framework="pt") as f:
                    bias = f.get_tensor(weight).float()
                bs = base[key].reshape(-1, experts)[:count]
                ms = data[key].reshape(-1, experts)[:count]
                bl = base[key + ".logits"].reshape(-1, base[key + ".logits"].shape[-1])[:count, :experts].float()
                ml = data[key + ".logits"].reshape(-1, data[key + ".logits"].shape[-1])[:count, :experts].float()
                bc = bl.sigmoid() + bias
                mc = ml.sigmoid() + bias
                changes = []
                for token in range(count):
                    bi = torch.nonzero(bs[token]).flatten().tolist()
                    mi = torch.nonzero(ms[token]).flatten().tolist()
                    if bi == mi:
                        continue
                    sorted_scores = bc[token].sort(descending=True).values
                    changes.append(
                        dict(
                            token=token,
                            baseline_ids=bi,
                            multichip_ids=mi,
                            baseline_cut_margin=(sorted_scores[k - 1] - sorted_scores[k]).item(),
                            max_score_perturbation=(bc[token] - mc[token]).abs().max().item(),
                            changed_expert_scores=[
                                dict(expert=e, baseline=bc[token, e].item(), multichip=mc[token, e].item())
                                for e in sorted(set(bi) ^ set(mi))
                            ],
                        )
                    )
                rows.append(
                    dict(
                        control=label,
                        phase="decode" if decode else "prefill",
                        router=name,
                        logical_rows=count,
                        scope="both users" if decode else "final user only (user 1 of batch 2)",
                        changed_rows=len(changes),
                        changes=changes,
                    )
                )
    result = dict(
        status="controlled",
        method="IDs are actual native outputs; margins are approximate float-sigmoid reconstructions of actual BF16 logits plus pinned checkpoint bias, not exact native decision margins; prefill captures final user 1 only because each per-user route call overwrites the snapshot",
        rows=rows,
    )
    (DOC / "routing_sensitivity_control.json").write_text(json.dumps(result, indent=2) + "\n")
    print([(r["control"], r["phase"], r["router"], r["changed_rows"]) for r in rows])


if __name__ == "__main__":
    main()
