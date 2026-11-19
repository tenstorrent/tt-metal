"""Offline TT-to-HF top-ID evidence; no model/device run or gate replacement."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/full_model"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_mapping(label):
    """Return request, step, HF reference families for expected/actual TT IDs."""
    if match := re.fullmatch(r"private_migration_vs_uninterrupted/token(42|43|44)/slot([01])", label):
        token, slot = map(int, match.groups())
        return (slot if token == 42 else 1 - slot), token - 41, "mixed", "migrated"
    if match := re.fullmatch(
        r"(repeat_solo_prefill|repeat_solo_decode|mixed_prefill|mixed_decode|pooled_vs_private_b2_prefill|pooled_vs_private_b2_decode)/row([01])",
        label,
    ):
        kind, row = match.group(1), int(match.group(2))
        step = 0 if "prefill" in kind else 1
        if kind.startswith("repeat_solo"):
            return row, step, f"solo{row}", f"solo{row}"
        return row, step, "mixed" if kind.startswith("pooled") else f"solo{row}", "mixed"
    if match := re.fullmatch(
        r"(migrated_decode_vs_solo|migrated_decode_vs_private_b2|pooled_migration_vs_uninterrupted_b2)/slot([01])",
        label,
    ):
        kind, row = match.group(1), 1 - int(match.group(2))
        reference = f"solo{row}" if kind.endswith("solo") else ("mixed" if kind.startswith("pooled") else "migrated")
        return row, 2, reference, "migrated"
    if label == "continued_survivor_vs_solo":
        return 0, 3, "solo0", "survivor"
    if label == "continued_survivor_vs_private":
        return 0, 3, "survivor", "survivor"
    if label == "reused_prefill_vs_private":
        return 1, 0, "solo1", "solo1"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hf_json = DOC / "shared_pool_hf_control_v2.json"
    tt_json = DOC / "shared_pool_generator_v5.json"
    capture_json = DOC / "moe_decode_capture_v1.json"
    hf_metadata = json.loads(hf_json.read_text())
    tt = json.loads(tt_json.read_text())
    capture_metadata = json.loads(capture_json.read_text())
    hf_pt = Path(hf_metadata["tensor_artifact"]["path"])
    assert digest(hf_pt) == hf_metadata["tensor_artifact"]["sha256"]
    hf = torch.load(hf_pt, map_location="cpu", weights_only=False)["runs"]
    tt_pt = Path(capture_metadata["capture_file"])
    captured = torch.load(tt_pt, map_location="cpu", weights_only=False)
    entries, excluded = [], []

    def score(token, request, step, family):
        logits = hf[f"{family}/repeat0"][step][request]["logits"].float()
        top = logits.topk(100).indices.tolist()
        maximum = logits.max()
        return {
            "hf_run": f"{family}/repeat0",
            "request": request,
            "phase": "prefill" if step == 0 else f"forced_{step+41}",
            "logical_position": len(hf_metadata["prompts"][request]) - 1 + step,
            "hf_argmax": int(logits.argmax()),
            "hf_top100_ids": top,
            "hf_global_max_tie_ids": (logits == maximum).nonzero().flatten().tolist(),
            "tt_token_id": token,
            "same_argmax_id": token == int(logits.argmax()),
            "tt_token_is_hf_global_max": bool(logits[token] == maximum),
            **{f"tt_token_in_hf_top{k}": token in top[:k] for k in (1, 5, 100)},
            "hf_strict_rank_of_tt_token": int((logits > logits[token]).sum()) + 1,
        }

    # Every saved per-row ranked ID with a matching HF history. Duplicated
    # controls remain visible, rather than presenting them as independent accuracy samples.
    for comparison in tt["comparisons"]:
        if "actual_top1" not in comparison:
            continue
        label = comparison["label"]
        mapping = row_mapping(label)
        if mapping is None:
            excluded.append(
                {
                    "label": label,
                    "reason": (
                        "Flattened multi-row global top-k is not a per-row top-ID; each row is covered by explicit row comparisons"
                        if label.startswith("private_baseline_rebuild/")
                        else "No matching HF continuation was saved for this later resident-input decode"
                    ),
                }
            )
            continue
        request, step, expected_family, actual_family = mapping
        for side, family in (("expected", expected_family), ("actual", actual_family)):
            token = comparison[f"{side}_top1"]
            assert 0 <= token < 248320
            entries.append(
                {
                    "source": str(tt_json),
                    "label": label,
                    "side": side,
                    "tt_id_kind": "saved double-precision flattened topk(5)[0] ranked ID; TT tie set and true argmax unavailable",
                    "tt_true_argmax_known": False,
                    "device_sampler_token_claim": False,
                    **score(token, request, step, family),
                }
            )

    # Independent historical eager capture has full TT logits at prefill/42/43.
    # Its scope/source differs from the tracked pool probe; never silently
    # substitute it for current production or for the pool probe's unseen tensors.
    for phase in ("prefill", "decode"):
        for name, tensor in captured[phase].items():
            step = 0 if phase == "prefill" else int(name.rsplit("token", 1)[1]) - 41
            rows = tensor.reshape(-1, tensor.shape[-1])
            requests = (0, 1) if name.startswith("batch") else (int(name[4]),)
            assert len(requests) == rows.shape[0]
            for row, request in zip(rows, requests, strict=True):
                family = "mixed" if name.startswith("batch") else f"solo{request}"
                values = row.float()
                assert torch.isfinite(values).all()
                token = int(values.argmax())
                ties = (values == values.max()).nonzero().flatten().tolist()
                entries.append(
                    {
                        "source": str(tt_pt),
                        "label": f"{phase}/{name}/request{request}",
                        "tt_id_kind": "exact argmax computed from saved full TT logits, historical eager reduced capture",
                        "tt_true_argmax_known": True,
                        "device_sampler_token_claim": False,
                        "tt_global_max": float(values.max()),
                        "tt_global_max_tie_ids": ties,
                        "tt_topk5_first": int(values.topk(5).indices[0]),
                        "tt_topk100_first": int(values.topk(100).indices[0]),
                        **score(token, request, step, family),
                    }
                )
    report = {
        "scope": "Historical reduced real-layer offline top-ID comparison, not current full-model acceptance",
        "command": [sys.executable, *sys.argv],
        "source_sha256": digest(__file__),
        "input_hashes": {str(path): digest(path) for path in (hf_json, hf_pt, tt_json, capture_json, tt_pt)},
        "hf_backend": hf_metadata["backend"],
        "checkpoint_revision": hf_metadata["revision"],
        "tt_capture_source_sha256": capture_metadata["source_sha256"],
        "tt_capture_scope": capture_metadata["scope"],
        "layers": [0, 1, 3],
        "prompts": hf_metadata["prompts"],
        "forced_tokens": [42, 43, 44],
        "metric": "TT exact-argmax or explicitly labeled saved ranked ID belongs to matching-history HF top-k",
        "frozen_TT_cross_geometry_status": "FAILED; not replaced or waived by this artifact",
        "no_aggregate_accuracy_claim": "Entries include duplicate lifecycle comparisons and two source experiments",
        "entry_count": len(entries),
        "full_TT_logit_rows": sum(e["tt_true_argmax_known"] for e in entries),
        "saved_ranked_ID_rows": sum(not e["tt_true_argmax_known"] for e in entries),
        "top5_miss_entries": [
            e["label"] + "/" + e.get("side", "full_logits") for e in entries if not e["tt_token_in_hf_top5"]
        ],
        "top100_miss_entries": [
            e["label"] + "/" + e.get("side", "full_logits") for e in entries if not e["tt_token_in_hf_top100"]
        ],
        "excluded_labels": excluded,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "entry_count",
                    "full_TT_logit_rows",
                    "saved_ranked_ID_rows",
                    "top5_miss_entries",
                    "top100_miss_entries",
                    "excluded_labels",
                    "frozen_TT_cross_geometry_status",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
